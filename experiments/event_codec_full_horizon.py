"""Fixed single-seed, 1800-round codec audit with held-out prediction metrics.

python experiments/event_codec_full_horizon.py verify
python experiments/event_codec_full_horizon.py preflight
python experiments/event_codec_full_horizon.py run --workers 2
python experiments/event_codec_full_horizon.py analyze

Commit results afterwards:
git add experiments/results/event_codec_full_horizon_v1
git commit -m "Add full-horizon single-seed codec results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

import event_codec_pilot as pilot
import stc_cifar10_confirmation as confirmation
import stc_cifar10_development as development
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.cifar10_benchmark import LAYOUT, download_cifar10, make_cifar10_federation
from neuromorphicfl.confirmation_metrics import target_crossings
from neuromorphicfl.event_codec_pilot import run_codec_pilot, traffic_summary
from neuromorphicfl.final_baseline_campaign import _architecture_ops

OUT = REPO / "experiments/results/event_codec_full_horizon_v1"
DATA = pilot.DATA
ROUNDS, EVAL_STRIDE = 1800, 60
PARTITION_SEED, TRAIN_SEED = 3500, 83500
TARGETS = (0.45, 0.50, 0.55)
POINTS = ("event", "stc_primary", "stc_sparse")
CONFIRMATION_IDS = {"stc_primary": "stc_primary", "stc_sparse": "stc_sparse_secondary"}
MAIN_VARIANT = {"event": "both_codecs", "stc_primary": "lossless_stc", "stc_sparse": "lossless_stc"}
atomic_json, atomic_csv = pilot.atomic_json, pilot.atomic_csv
digest, text_hash = pilot.digest, pilot.text_hash


def sampled_rounds(config):
    return tuple(sorted({1, config.rounds, *range(config.eval_stride, config.rounds + 1, config.eval_stride)}))


def config_for(point):
    if point == "event":
        return confirmation.reference_config("event")
    if point in CONFIRMATION_IDS:
        return confirmation.config_for(confirmation.POINT_BY_ID[CONFIRMATION_IDS[point]])
    raise ValueError(point)


def protocol_spec():
    return {"campaign": "event_codec_full_horizon_v1", "rounds": ROUNDS,
            "architecture": "cifar_cnn", "dimension": LAYOUT.dimension, "n_clients": 10,
            "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED, "regime": "strong",
            "points": {name: asdict(config_for(name)) for name in POINTS},
            "evaluation_rounds": list(sampled_rounds(config_for("event"))),
            "test_set": "fixed_CIFAR10_test_set_no_tuning", "accuracy_targets": list(TARGETS),
            "endpoint": "final_round_1800_not_best_test_checkpoint",
            "target_crossing": "first_saved_sample_no_interpolation_unreached_censored",
            "training_runs": 3, "seed_pairs": 1, "confidence_intervals": False,
            "selection_or_early_stopping": False, "checkpoint_stride": EVAL_STRIDE,
            "source_policy": "frozen_120_round_host_and_codecs_reused_without_edits",
            "evaluation_checkpoint": "test_metrics_and_model_saved_atomically_in_same_NPZ",
            "timing_scope": "codec_audit_overhead_not_deployment_runtime",
            "hash_format": "sha256_utf8_lf_v1"}


def source_hashes():
    return {**pilot.sources(), "experiments/event_codec_full_horizon.py": text_hash(Path(__file__)),
            "experiments/stc_cifar10_confirmation.py": text_hash(REPO / "experiments/stc_cifar10_confirmation.py"),
            "experiments/p19_cifar10_cnn_final_horizons.py": text_hash(REPO / "experiments/p19_cifar10_cnn_final_horizons.py")}


def event_reference():
    return confirmation.load_reference("event", 0)


def build_contract():
    previous = pilot.load_contract()
    runtime = development.runtime_manifest()
    if runtime != previous["runtime"]:
        raise RuntimeError("Use the same recorded environment as the 120-round codec pilot")
    extracted = download_cifar10(DATA)
    fingerprints = {name: development.file_hash(extracted / name) for name in
                    [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]}
    if fingerprints != previous["data_sha256"]:
        raise RuntimeError("CIFAR batch fingerprints differ from the codec pilot")
    _, _, reference = event_reference()
    return {"protocol": protocol_spec(), "source_hashes": source_hashes(), "runtime": runtime,
            "data_sha256": fingerprints, "archived_event_reference": reference,
            "pilot_protocol_sha256": text_hash(pilot.OUT / "protocol.json")}


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science["protocol"] != protocol_spec() or science["source_hashes"] != source_hashes():
        raise RuntimeError("Full-horizon protocol or sources changed")
    if science["archived_event_reference"] != event_reference()[2]:
        raise RuntimeError("Archived Event reference changed")
    if science["pilot_protocol_sha256"] != text_hash(pilot.OUT / "protocol.json"):
        raise RuntimeError("Pilot provenance changed")
    return science


class PredictiveCheckpoint:
    """Add held-out evaluation atomically to the unchanged host's checkpoints."""

    def __init__(self, path, provenance, *, federation, architecture, config):
        self.base = BaselineCheckpoint(path, provenance)
        self.federation, self.config = federation, config
        self.layout, _, _, self.metrics = _architecture_ops(architecture)
        self.evaluations = []

    def load(self, identity):
        saved = self.base.load(identity)
        if saved is None:
            self.evaluations = []
            return None
        progress = saved["progress"]
        self.evaluations = progress.get("predictive_history", [])
        completed = int(progress["completed_round"])
        expected = [r for r in sampled_rounds(self.config) if r <= completed]
        if [row["round"] for row in self.evaluations] != expected:
            raise ValueError("Checkpoint held-out evaluation coverage differs")
        records = {row["round"]: row for row in progress["records"]}
        for row in self.evaluations:
            if row["model_sha256"] != records[row["round"]]["model_sha256"]:
                raise ValueError("Prediction/model fingerprint mismatch")
            if not all(np.isfinite(row[field]) for field in ("test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy")):
                raise ValueError("Nonfinite saved prediction metrics")
        return saved

    def save(self, identity, arrays, progress):
        rnd = int(progress["completed_round"])
        if rnd in sampled_rounds(self.config):
            if self.evaluations and self.evaluations[-1]["round"] >= rnd:
                raise ValueError("Duplicate held-out checkpoint evaluation")
            model_hash = hashlib.sha256(arrays["model"].astype("<f4").tobytes()).hexdigest()
            if model_hash != progress["records"][-1]["model_sha256"]:
                raise AssertionError("Evaluation uses a different model from the codec record")
            _, ce, accuracy, macro, worst, _ = self.metrics(
                arrays["model"], self.federation.X_test, self.federation.y_test,
                layout=self.layout, regularization=self.config.regularization)
            if not all(np.isfinite(value) for value in (ce, accuracy, macro, worst)):
                raise FloatingPointError("Nonfinite held-out metric")
            self.evaluations.append({"round": rnd, "test_ce": float(ce), "test_accuracy": float(accuracy),
                                     "macro_accuracy": float(macro), "worst_class_accuracy": float(worst),
                                     "model_sha256": model_hash})
        self.base.save(identity, arrays, {**progress, "predictive_history": self.evaluations})


def evaluated_run(*, federation, architecture, method, config, seed, checkpoint_path,
                  provenance, progress_callback=None, stop_after=None):
    checkpoint = PredictiveCheckpoint(checkpoint_path, provenance, federation=federation,
                                      architecture=architecture, config=config)

    def progress(row):
        if progress_callback is not None:
            latest = checkpoint.evaluations[-1] if checkpoint.evaluations else {}
            progress_callback(row, latest)

    kwargs = dict(federation=federation, architecture=architecture, method=method, config=config,
                  seed=seed, checkpoint=checkpoint, checkpoint_stride=config.eval_stride,
                  progress_callback=progress)
    # Force a round-1 checkpoint once; the following call resumes its exact RNG,
    # model, residuals and counters. Existing host sources remain untouched.
    if not checkpoint_path.exists():
        result = run_codec_pilot(**kwargs, stop_after=1)
        if config.rounds == 1 or stop_after == 1:
            return attach_evaluations(result, checkpoint.evaluations)
    result = run_codec_pilot(**kwargs, stop_after=stop_after)
    return attach_evaluations(result, checkpoint.evaluations)


def attach_evaluations(result, evaluations):
    result["evaluation"] = result["history"].merge(pd.DataFrame(evaluations), on="round", validate="one_to_one")
    if len(result["evaluation"]) != len(result["history"]):
        raise AssertionError("Training and held-out history coverage differ")
    return result


def trajectory_tables(result, *, n_clients, dimension):
    curves = []
    for _, evaluation in result["evaluation"].iterrows():
        prefix = result["rounds"][result["rounds"]["round"] <= int(evaluation["round"])]
        table = traffic_summary(prefix, n_clients=n_clients, dimension=dimension)
        for field, value in evaluation.items():
            table[field] = value
        curves.append(table)
    history = pd.concat(curves, ignore_index=True)
    final = result["evaluation"].iloc[-1]
    summary = result["summary"].copy()
    for field in ("train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy"):
        summary["final_" + field] = float(final[field])
    summary["model_sha256"] = final.model_sha256
    return summary, history


def metadata(point, science):
    return {"point": point, "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
            "evaluation_scope": "single_seed_full_horizon_fixed_protocol",
            "contract_sha256": digest(science)}


def paths(point, out=OUT):
    return {kind: out / f"{point}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("summary", "rounds", "evaluation", "history", "receipt")}


def read_receipt(point, science, out=OUT):
    files = paths(point, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("point") != point or receipt.get("contract_sha256") != digest(science):
        raise RuntimeError("Full-horizon result identity differs")
    if receipt["state"] == "complete":
        for kind in ("summary", "rounds", "evaluation", "history"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing or modified result: {point}/{kind}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown full-horizon completion state")
    return receipt


def save_result(point, science, result, out=OUT):
    summary, history = trajectory_tables(result, n_clients=science["protocol"]["n_clients"],
                                          dimension=science["protocol"]["dimension"])
    files = paths(point, out)
    tables = {"summary": summary, "history": history, "evaluation": result["evaluation"], "rounds": result["rounds"]}
    for kind, table in tables.items():
        atomic_csv(table.assign(**metadata(point, science)), files[kind])
    atomic_json({"point": point, "contract_sha256": digest(science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def run_point(point, science, out=OUT):
    with development.point_lock(out / "locks" / f"{point}.lock"):
        old = read_receipt(point, science, out)
        if old is not None:
            return f"reuse {point}: {old['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/data/runtime differs from frozen full-horizon contract")
        # Keep the real held-out CIFAR test slot. The reused host never touches
        # it; the checkpoint evaluator reads it only at the declared samples.
        federation = make_cifar10_federation(root=DATA, regime="strong", seed=PARTITION_SEED,
                                            train_eval_size=5000, n_clients=10)

        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % EVAL_STRIDE == 0:
                print(f"{point}: {row['round']}/{ROUNDS}, test_acc={evaluation['test_accuracy']:.4f}, "
                      f"test_ce={evaluation['test_ce']:.6f}, server={row['exact_downlink_bits']} bits", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = evaluated_run(federation=federation, architecture="cifar_cnn",
                                       method="event" if point == "event" else "stc", config=config_for(point),
                                       seed=TRAIN_SEED, checkpoint_path=out / "checkpoints" / f"{point}.npz",
                                       provenance={"contract_sha256": digest(science), "point": point},
                                       progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"point": point, "contract_sha256": digest(science), "state": "numerical_failure",
                         "error": str(exc)}, paths(point, out)["receipt"])
            return f"numerical failure {point}: {exc}"
        save_result(point, science, result, out)
        return f"done {point}: final test_acc={result['evaluation'].iloc[-1].test_accuracy:.4f}"


def run(workers, only=None):
    science = build_contract()
    manifest = OUT / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != science:
        raise RuntimeError("Existing full-horizon protocol/data/runtime differs; preserve its evidence")
    if not manifest.exists():
        atomic_json(science, manifest)
    points = (only,) if only else POINTS
    if workers == 1:
        for point in points:
            print(run_point(point, science), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_point, point, science) for point in points]
            for future in as_completed(jobs):
                print(future.result(), flush=True)


def verified_point(point, science, out=OUT):
    receipt = read_receipt(point, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError(f"Full-horizon point incomplete: {point}; no point is silently omitted")
    tables = {kind: pd.read_csv(path, float_precision="round_trip") for kind, path in paths(point, out).items() if kind != "receipt"}
    expected_rounds = list(science["protocol"]["evaluation_rounds"])
    if tables["rounds"]["round"].tolist() != list(range(1, ROUNDS + 1)) or tables["evaluation"]["round"].tolist() != expected_rounds:
        raise RuntimeError("Full-horizon round/evaluation coverage differs")
    for kind, table in tables.items():
        for field, value in metadata(point, science).items():
            if not table[field].eq(value).all():
                raise RuntimeError(f"Result metadata differs: {point}/{kind}/{field}")
    evaluation = tables["evaluation"]
    metric_fields = ["train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy"]
    if not np.isfinite(evaluation[metric_fields]).all().all() or not evaluation.test_accuracy.between(0, 1).all():
        raise RuntimeError("Invalid predictive metrics")
    hashes = tables["rounds"].set_index("round").model_sha256
    if not np.array_equal(evaluation.model_sha256, hashes.loc[expected_rounds].to_numpy()):
        raise RuntimeError("Evaluation and codec model fingerprints differ")
    rebuilt_summary, rebuilt_history = trajectory_tables(
        {"rounds": tables["rounds"], "evaluation": evaluation[["round", *metric_fields, "model_sha256"]],
         "summary": traffic_summary(tables["rounds"], n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])},
        n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for name, rebuilt in (("summary", rebuilt_summary), ("history", rebuilt_history)):
        for field in rebuilt:
            actual = tables[name][field]
            expected = rebuilt[field]
            if field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified"):
                match = np.array_equal(actual.to_numpy(), expected.to_numpy())
            else:
                match = np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
            if not match:
                raise RuntimeError(f"Reconstructed {name} differs: {point}/{field}")
    return tables


def reference_check(point, tables):
    if point == "event":
        _, reference, _ = event_reference()
        label = "archived_event_seed_3500"
    else:
        candidate = confirmation.POINT_BY_ID[CONFIRMATION_IDS[point]]
        if not confirmation.paths(candidate, 0)[2].exists():
            return {"point": point, "status": "matching_confirmation_receipt_not_available"}
        science = confirmation.load_execution()
        receipt = confirmation.read_receipt(candidate, 0, science)
        if receipt["state"] != "complete":
            return {"point": point, "status": receipt["state"]}
        reference = pd.read_csv(confirmation.paths(candidate, 0)[1], float_precision="round_trip")
        label = "matching_native_stc_confirmation_seed_3500"
    evaluation = tables["evaluation"]
    if evaluation["round"].tolist() != reference["round"].tolist():
        raise RuntimeError("Regression reference evaluation schedule differs")
    differences = {}
    for field in ("train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy"):
        # Old P19 CSVs contain about ten significant digits. Their printed
        # metric values are compared at the existing archive tolerance.
        differences[field] = float(np.max(np.abs(evaluation[field] - reference[field])))
        if not np.allclose(evaluation[field], reference[field], rtol=1e-8, atol=1e-9):
            raise RuntimeError(f"Prediction regression differs: {point}/{field}")
    rounds = tables["rounds"]
    prefix = rounds.legacy_uplink_bits.cumsum().iloc[evaluation["round"].to_numpy(dtype=int) - 1].to_numpy()
    if not np.array_equal(prefix, reference.uplink_packetized_bits):
        raise RuntimeError("Regression native upload counters differ")
    if point != "event":
        server = rounds.legacy_server_bits.cumsum().iloc[evaluation["round"].to_numpy(dtype=int) - 1].to_numpy()
        if not np.array_equal(server, reference.stc_downlink_packetized_bits):
            raise RuntimeError("Regression native server counters differ")
    return {"point": point, "status": "matched", "reference": label,
            "max_absolute_metric_differences": differences, "native_traffic_exact": True,
            "metric_tolerance": {"rtol": 1e-8, "atol": 1e-9}}


def targets_table(history):
    output = []
    for (point, variant), rows in history.groupby(["point", "variant"], sort=False):
        rows = rows.sort_values("round").copy()
        rows["uplink_packetized_bits"] = rows.uplink_bits
        for crossing in target_crossings(rows, targets=TARGETS, horizon=ROUNDS,
                                         comparison_id=point + "__" + variant, partition_seed=PARTITION_SEED):
            crossing.update(point=point, variant=variant)
            crossing["unicast_downlink_bits"] = (int(rows.loc[rows["round"] == crossing["first_sampled_round"], "unicast_downlink_bits"].iloc[0])
                                                  if crossing["reached"] else float("nan"))
            output.append(crossing)
    return pd.DataFrame(output)


def plot(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    for point, label, color in (("event", "Event", "#333333"), ("stc_primary", "STC p=1", "#D55E00"), ("stc_sparse", "STC p=0.25", "#0072B2")):
        rows = history[(history.point == point) & (history.variant == MAIN_VARIANT[point])]
        axes[0].plot(rows["round"], 100 * rows.test_accuracy, color=color, label=label)
        axes[1].plot(rows["round"], rows.test_ce, color=color, label=label)
        axes[2].plot(rows.unicast_total_bits / 1e6, 100 * rows.test_accuracy, color=color, label=label + " recoded")
        native = history[(history.point == point) & (history.variant == ("legacy" if point == "event" else "native_stc"))]
        axes[2].plot(native.unicast_total_bits / 1e6, 100 * native.test_accuracy,
                     color=color, ls="--", lw=1, label=label + " original")
    axes[0].set(xlabel="Communication round", ylabel="Test accuracy (%)", xlim=(0, ROUNDS))
    axes[1].set(xlabel="Communication round", ylabel="Test cross-entropy", xlim=(0, ROUNDS))
    axes[2].set(xlabel="Upload + unicast download (Mbit)", ylabel="Test accuracy (%)")
    for axis in axes:
        axis.grid(alpha=.22)
        axis.legend(fontsize=8)
    fig.suptitle("Fixed full-horizon codec diagnostic: one partition seed (3500)", fontsize=11, y=.99)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out / "accuracy_traffic.png", dpi=180)
    fig.savefig(out / "accuracy_traffic.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    summaries, histories, regressions = [], [], []
    for point in POINTS:
        tables = verified_point(point, science, out)
        regressions.append(reference_check(point, tables))
        summaries.append(tables["summary"])
        histories.append(tables["history"])
    summary, history = pd.concat(summaries, ignore_index=True), pd.concat(histories, ignore_index=True)
    atomic_csv(summary, out / "comparison.csv")
    atomic_csv(history, out / "comparison_history.csv")
    atomic_csv(targets_table(history), out / "traffic_to_target.csv")
    atomic_json({"seed_pairs": 1, "confidence_intervals": False, "model_reconstruction": "bitwise_each_round",
                 "source_and_receipts_verified": True, "regression_references": regressions,
                 "endpoint": "final_round_1800", "target_rule": protocol_spec()["target_crossing"],
                 "test_driven_configuration_selection": False}, out / "analysis_audit.json")
    plot(history, out)
    print(summary[["point", "variant", "final_test_accuracy", "final_test_ce", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits"]].to_string(index=False))
    print("One fixed seed: descriptive diagnostic, no uncertainty or SOTA claim.")


def preflight():
    previous = pilot.load_contract()
    _, _, reference = event_reference()
    print("Frozen codec pilot source hashes: unchanged")
    print("Full horizon: 1800 rounds, seed 3500 / SGD 83500, Event + STC p=1 and p=.25")
    print("Final-round test metrics; samples at round 1 and every 60; no selection")
    print("Runtime matches codec pilot:", development.runtime_manifest() == previous["runtime"])
    print("Archived Event reference validated:", reference["partition_seed"])
    print("Existing CIFAR data:", (DATA / "cifar-10-batches-py/data_batch_1").exists())


def status():
    if not (OUT / "protocol.json").exists():
        print("Full-horizon experiment not started")
        return
    science = load_contract()
    for point in POINTS:
        receipt = read_receipt(point, science)
        if receipt is not None:
            print(point, receipt["state"])
        else:
            path = OUT / "checkpoints" / f"{point}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                print(point, "checkpoint round", metadata["progress"]["completed_round"])
            else:
                print(point, "pending")


def smoke_worker(point):
    from dataclasses import replace
    from stc_stage1 import smoke_federation
    config = replace(config_for(point), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
    federation = smoke_federation()
    with tempfile.TemporaryDirectory() as folder:
        kwargs = dict(federation=federation, architecture="cifar_cnn", method="event" if point == "event" else "stc",
                      config=config, seed=TRAIN_SEED, provenance={"synthetic": True})
        full = evaluated_run(**kwargs, checkpoint_path=Path(folder) / "full.npz")
        resumed_path = Path(folder) / "resumed.npz"
        evaluated_run(**kwargs, checkpoint_path=resumed_path, stop_after=1)
        resumed = evaluated_run(**kwargs, checkpoint_path=resumed_path)
        np.testing.assert_array_equal(full["model"].view(np.uint32), resumed["model"].view(np.uint32))
        pd.testing.assert_frame_equal(full["evaluation"], resumed["evaluation"])
        pd.testing.assert_frame_equal(full["summary"], resumed["summary"])
        return full["evaluation"].iloc[-1].model_sha256


def smoke():
    expected = {point: smoke_worker(point) for point in POINTS}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = dict(zip(POINTS, pool.map(smoke_worker, POINTS)))
    if actual != expected:
        raise AssertionError("Spawned full-horizon evaluator differs")
    print("All three synthetic points: exact model/prediction resume and spawned workers")
    print("Synthetic smoke only; no CIFAR research training was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--point", choices=POINTS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("workers must be 1, 2 or 3")
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_event_codec_full_horizon.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke()
    elif args.command == "smoke":
        smoke()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        run(args.workers, args.point)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
