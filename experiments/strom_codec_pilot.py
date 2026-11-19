"""Matched 600-round Ström-style pulse/codecs pilot; six fixed operating points.

python experiments/strom_codec_pilot.py verify
python experiments/strom_codec_pilot.py preflight
python experiments/strom_codec_pilot.py run --workers 2
python experiments/strom_codec_pilot.py analyze
git add experiments/results/strom_codec_pilot_v1
git commit -m "Add comparable Strom-style codec pilot results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
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

import event_codec_full_horizon as full
import p19_cifar10_cnn_final_horizons as p19
import strom_codec_host as host

OUT = REPO / "experiments/results/strom_codec_pilot_v1"
ROUNDS, PARTITION_SEED, TRAIN_SEED = 600, 3500, 83500
TARGETS = full.TARGETS
atomic_csv, atomic_json, digest, text_hash = full.atomic_csv, full.atomic_json, full.digest, full.text_hash
EVAL_FIELDS = ("round", "train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy", "model_sha256")


@dataclass(frozen=True)
class Point:
    point_id: str
    threshold: float
    native_gain: float


POINTS = tuple(Point(f"strom_t{tau_id}_g{gain_id}", tau, gain)
               for tau_id, tau in (("0025", .0025), ("0100", .01))
               for gain_id, gain in (("005", .05), ("010", .1), ("100", 1.)))
POINT_BY_ID = {point.point_id: point for point in POINTS}
REFERENCE_POINTS = full.POINTS


def config_for(point):
    if isinstance(point, str):
        point = POINT_BY_ID[point]
    return replace(full.config_for("event"), rounds=ROUNDS, strom_threshold=point.threshold,
                   server_gain=point.native_gain, server_gain_exponent=0.)


def protocol_spec():
    return {"campaign": "strom_codec_pilot_v1", "scope": "single_seed_scaling_and_codec_pilot_not_confirmation",
            "architecture": "cifar_cnn", "dimension": 20570, "n_clients": 10, "regime": "strong",
            "rounds": ROUNDS, "partition_seed": PARTITION_SEED, "training_seed": TRAIN_SEED,
            "points": {p.point_id: asdict(config_for(p)) for p in POINTS},
            "evaluation_rounds": list(full.sampled_rounds(config_for(POINTS[0]))),
            "new_training_runs": 6, "reused_prefixes": list(REFERENCE_POINTS),
            "native_rule": "accumulate_weighted_delta_over_local_lr_one_pulse_per_coordinate_subtractive_residual",
            "gain_units": "native_gain_over_local_lr_is_FedAvg_equivalent_gain",
            "effective_FedAvg_gains": [1., 2., 20.], "legacy_controls_native_gain": 1.,
            "position_and_downlink_codecs": "identical_frozen_Event_codec_portfolio_with_own_fixed_threshold_amplitude",
            "primary_accounting": "conservative_unicast", "broadcast_accounting": "secondary_logical_model",
            "control_accounting": "initialization_in_both_32_bits_per_client_round_only_in_unicast_as_frozen_references",
            "accuracy_targets": list(TARGETS), "target_rule": "first_saved_sample_no_interpolation_unreached_censored",
            "intervals": False, "test_based_selection": False, "all_six_settings_reported": True,
            "test_set": "fixed_CIFAR10_test_diagnostics_not_for_selection",
            "reference_prefix_note": "first_600_rounds_of_unchanged_1800_round_trajectories_no_new_selection",
            "source_policy": "no_changes_or_new_modules_in_frozen_src_package", "hash_format": "sha256_utf8_lf_v1"}


def archived_reference(threshold):
    point = p19.POINT_BY_ID["cifar_strom_quality" if threshold == .0025 else "cifar_strom_near"]
    files = p19.output_paths(point, 0)
    summary = pd.read_csv(files[0], float_precision="round_trip")
    history = pd.read_csv(files[1], float_precision="round_trip")
    expected = {"method": "strom", "architecture": "cifar_cnn", "dimension": 20570,
                "point_id": point.point_id, "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
                "rounds": 1800, "local_steps": 5, "local_lr": .05, "strom_threshold": threshold,
                "server_gain": 1., "init_scale": 1.}
    if len(summary) != 1 or any(summary.iloc[0][field] != value for field, value in expected.items()):
        raise RuntimeError("Archived Ström metadata differs")
    if history["round"].tolist() != list(full.sampled_rounds(full.config_for("event"))):
        raise RuntimeError("Archived Ström evaluation coverage differs")
    for field in ("point_id", "partition_seed", "train_seed", "rounds"):
        if not history[field].eq(expected[field]).all():
            raise RuntimeError("Archived Ström history metadata differs")
    provenance = {"point_id": point.point_id, "files": {path.relative_to(REPO).as_posix(): text_hash(path) for path in files},
                  "scope": "legacy_printed_curve_and_native_uploads_not_archived_full_models",
                  "missing_per_run_fields": [k for k in asdict(p19.config_for(point)) if k not in summary]}
    return history[history["round"] <= ROUNDS].copy(), provenance


def source_hashes():
    return {**full.source_hashes(), **{f"experiments/{name}": text_hash(REPO / "experiments" / name)
                                    for name in ("strom_codec_host.py", "strom_codec_pilot.py")}}


def build_science():
    original = full.load_contract()
    reused = {}
    for point in REFERENCE_POINTS:
        full.verified_point(point, original)
        reused[point] = {path.relative_to(REPO).as_posix(): text_hash(path) for path in full.paths(point).values()}
    return {"protocol": protocol_spec(), "source_hashes": source_hashes(),
            "runtime": original["runtime"], "data_sha256": original["data_sha256"],
            "source_contract_sha256": digest(original), "reused_full_horizon_evidence": reused,
            "archived_strom_references": [archived_reference(tau)[1] for tau in (.0025, .01)]}


def build_contract():
    science = build_science()
    if digest(full.build_contract()) != science["source_contract_sha256"]:
        raise RuntimeError("Current runtime/data differs from frozen full-horizon evidence")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Ström pilot protocol, sources or reference evidence changed")
    return science


def metadata(point, science, *, reused=False):
    return {"point": point, "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
            "contract_sha256": digest(science), "evaluation_scope": "single_seed_600_round_codec_scaling_pilot",
            "evidence_origin": "reused_1800_round_prefix" if reused else "new_strom_codec_run"}


def identity(point, science):
    return {**metadata(point, science), "config_sha256": digest(asdict(config_for(point)))}


def paths(point, out=OUT):
    return {kind: out / f"{point}_p{PARTITION_SEED}_r{ROUNDS}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("summary", "history", "rounds", "evaluation", "receipt")}


def read_receipt(point, science, out=OUT):
    files = paths(point, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != identity(point, science):
        raise RuntimeError("Ström pilot result identity differs")
    if receipt["state"] == "complete":
        for kind in ("summary", "history", "rounds", "evaluation"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Changed Ström evidence: {point}/{kind}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown Ström completion state")
    return receipt


def save_result(point, science, result, out=OUT):
    summary, history = host.trajectory_tables(result, n_clients=science["protocol"]["n_clients"],
                                             dimension=science["protocol"]["dimension"])
    files = paths(point, out)
    tables = {"summary": summary, "history": history, "rounds": result["rounds"], "evaluation": result["evaluation"]}
    for kind, table in tables.items():
        atomic_csv(table.assign(**metadata(point, science)), files[kind])
    atomic_json({"identity": identity(point, science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def run_point(point, science, out=OUT):
    with full.development.point_lock(out / "locks" / f"{point}.lock"):
        receipt = read_receipt(point, science, out)
        if receipt is not None:
            return f"reuse {point}: {receipt['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/runtime/data differs")
        federation = full.make_cifar10_federation(root=full.DATA, regime="strong", seed=PARTITION_SEED,
                                                 train_eval_size=5000, n_clients=10)

        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point}: {row['round']}/{ROUNDS}, test_acc={evaluation['test_accuracy']:.4f}, "
                      f"residual_norm={row['residual_l2_norm']:.4f}", flush=True)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = host.evaluated_run(federation=federation, architecture="cifar_cnn", config=config_for(point),
                                            seed=TRAIN_SEED, checkpoint_path=out / "checkpoints" / f"{point}.npz",
                                            provenance=identity(point, science), progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": identity(point, science), "state": "numerical_failure", "error": str(exc)}, paths(point, out)["receipt"])
            return f"numerical failure {point}: {exc}"
        save_result(point, science, result, out)
        return f"done {point}"


def run(workers, point=None):
    science = build_contract()
    manifest = OUT / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != science:
        raise RuntimeError("Existing Ström campaign differs; preserve its evidence")
    if not manifest.exists():
        atomic_json(science, manifest)
    points = tuple(POINT_BY_ID) if point is None else (point,)
    jobs = [p for p in points if read_receipt(p, science) is None]
    print(f"{len(jobs)} new Ström runs pending; Event and both STC prefixes reused", flush=True)
    if workers == 1:
        for name in jobs:
            print(run_point(name, science), flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for job in as_completed([pool.submit(run_point, name, science) for name in jobs]):
                print(job.result(), flush=True)


def verified_tables(point, science, out=OUT):
    receipt = read_receipt(point, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError(f"Ström point incomplete: {point}")
    tables = {kind: pd.read_csv(path, float_precision="round_trip", dtype={"model_sha256": str})
              for kind, path in paths(point, out).items() if kind != "receipt"}
    for table in tables.values():
        for field, value in metadata(point, science).items():
            if not table[field].eq(value).all():
                raise RuntimeError("Ström result metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    if (rounds["round"].tolist() != list(range(1, science["protocol"]["rounds"] + 1))
            or evaluation["round"].tolist() != science["protocol"]["evaluation_rounds"]):
        raise RuntimeError("Ström pilot coverage differs")
    if not rounds.method.eq("strom").all() or not np.isfinite(evaluation[list(EVAL_FIELDS[1:-1])]).all().all():
        raise RuntimeError("Invalid Ström predictive evidence")
    if not evaluation.test_accuracy.between(0, 1).all():
        raise RuntimeError("Invalid Ström accuracy")
    if not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"]):
        raise RuntimeError("Ström prediction/model fingerprints differ")
    reconstructed = {"rounds": rounds, "evaluation": evaluation[list(EVAL_FIELDS)],
                     "summary": host.traffic_summary(rounds, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])}
    summary, history = host.trajectory_tables(reconstructed, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind, expected in (("summary", summary), ("history", history)):
        for field in expected:
            a, b = tables[kind][field], expected[field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1e-12, atol=1e-12)):
                raise RuntimeError(f"Rebuilt Ström {kind} differs: {field}")
    return tables


def reused_tables(point, science):
    original = full.verified_point(point, full.load_contract())
    rounds = original["rounds"][original["rounds"]["round"] <= ROUNDS].copy()
    evaluation = original["evaluation"][original["evaluation"]["round"] <= ROUNDS][list(EVAL_FIELDS)].copy()
    result = {"rounds": rounds, "evaluation": evaluation,
              "summary": full.traffic_summary(rounds, n_clients=10, dimension=20570)}
    summary, history = full.trajectory_tables(result, n_clients=10, dimension=20570)
    return {kind: table.assign(**metadata(point, science, reused=True)) for kind, table in
            (("rounds", rounds), ("evaluation", evaluation), ("summary", summary), ("history", history))}


def regression_check(point, tables):
    candidate = POINT_BY_ID[point]
    if candidate.native_gain != 1.:
        return {"point": point, "status": "normalized_gain_no_archived_curve_native_equivalence_checked_synthetically"}
    reference, _ = archived_reference(candidate.threshold)
    evaluation, rounds = tables["evaluation"], tables["rounds"]
    if evaluation["round"].tolist() != reference["round"].tolist():
        raise RuntimeError("Legacy Ström regression coverage differs")
    differences = {}
    for field in EVAL_FIELDS[1:-1]:
        differences[field] = float(np.max(np.abs(evaluation[field].to_numpy() - reference[field].to_numpy())))
        if not np.allclose(evaluation[field], reference[field], rtol=1e-8, atol=1e-9):
            raise RuntimeError(f"Legacy Ström regression differs: {point}/{field}")
    prefix = rounds.legacy_uplink_bits.cumsum().iloc[evaluation["round"].to_numpy(dtype=int) - 1]
    if not np.array_equal(prefix, reference.uplink_packetized_bits):
        raise RuntimeError("Legacy Ström native upload prefixes differ")
    return {"point": point, "status": "matched", "maximum_absolute_metric_differences": differences}


def optimized(frame):
    return frame[[variant == ("both_codecs" if point in POINT_BY_ID else full.MAIN_VARIANT[point])
                  for point, variant in zip(frame.point, frame.variant)]].copy()


def target_table(history):
    rows = []
    for (point, variant), group in history.groupby(["point", "variant"], sort=False):
        group = group.sort_values("round").assign(uplink_packetized_bits=lambda x: x.uplink_bits)
        for row in full.target_crossings(group, targets=TARGETS, horizon=ROUNDS,
                                        comparison_id=point + "__" + variant, partition_seed=PARTITION_SEED):
            row.update(point=point, variant=variant)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#999999"]
    entries = [("event", "Event", "black", "-"), ("stc_primary", "STC p=1", "#D55E00", "-"),
               ("stc_sparse", "STC p=0.25", "#882255", "-")]
    entries += [(p.point_id, f"Ström τ={p.threshold:g}, gain={p.native_gain:g}", color,
                 "--" if p.native_gain == 1 else "-") for p, color in zip(POINTS, colors)]
    for point, label, color, style in entries:
        rows = history[history.point == point].sort_values("round")
        axes[0].plot(rows["round"], 100 * rows.test_accuracy, label=label, color=color, ls=style)
        axes[1].plot(rows.unicast_total_bits / 1e6, 100 * rows.test_accuracy, label=label, color=color, ls=style)
    axes[0].set(xlabel="Communication round", ylabel="Test accuracy (%)")
    axes[1].set(xlabel="Total unicast traffic (Mbit, log scale)", ylabel="Test accuracy (%)", xscale="log")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Ström-style residual pulses: same lossless codecs, six fixed settings, one seed")
    fig.tight_layout(rect=(0, .15, 1, .95))
    fig.savefig(out / "pilot_comparison.png", dpi=180)
    fig.savefig(out / "pilot_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    completion = []
    for point in POINT_BY_ID:
        receipt = read_receipt(point, science, out)
        completion.append({"point": point, "state": receipt["state"] if receipt else "pending"})
    atomic_json({"expected": 6, "results": completion}, out / "completion_audit.json")
    if any(row["state"] != "complete" for row in completion):
        raise RuntimeError("Ström grid incomplete or failed; no point is dropped. See completion_audit.json")
    tables = {point: verified_tables(point, science, out) for point in POINT_BY_ID}
    regression = [regression_check(point, table) for point, table in tables.items()]
    tables.update({point: reused_tables(point, science) for point in REFERENCE_POINTS})
    summary = pd.concat([t["summary"] for t in tables.values()], ignore_index=True)
    history = pd.concat([t["history"] for t in tables.values()], ignore_index=True)
    main = optimized(summary)
    main["final_train_l2_penalty"] = main.final_train_objective - main.final_train_ce
    main["native_strom_gain"] = [POINT_BY_ID[p].native_gain if p in POINT_BY_ID else np.nan for p in main.point]
    main["fedavg_equivalent_strom_gain"] = main.native_strom_gain / full.config_for("event").local_lr
    main["strom_threshold_gradient_units"] = [POINT_BY_ID[p].threshold if p in POINT_BY_ID else np.nan for p in main.point]
    event = main[main.point == "event"].iloc[0]
    contrasts = [{"reference": row.point, "event_accuracy_difference_pp": 100 * (event.final_test_accuracy - row.final_test_accuracy),
                  "event_test_ce_difference": event.final_test_ce - row.final_test_ce,
                  "event_over_reference_unicast_ratio": event.unicast_total_bits / row.unicast_total_bits,
                  "event_over_reference_uplink_ratio": event.uplink_bits / row.uplink_bits}
                 for _, row in main[main.point != "event"].iterrows()]
    targets = target_table(history)
    for name, table in (("comparison.csv", summary), ("comparison_history.csv", history), ("main_runs.csv", main),
                        ("event_contrasts.csv", pd.DataFrame(contrasts)), ("traffic_to_target.csv", targets)):
        atomic_csv(table, out / name)
    atomic_json({"scope": protocol_spec()["scope"], "new_trajectories": 6, "reused_prefixes": 3,
                 "roundtrip_and_client_model_checks_at_execution": True, "legacy_curve_checks": regression,
                 "intervals": False, "test_based_selection": False,
                 "future_tuning_requires_separate_development_validation": True}, out / "analysis_audit.json")
    plot(optimized(history), out)
    print(main[["point", "final_test_accuracy", "final_test_ce", "unicast_total_bits", "fedavg_equivalent_strom_gain"]].to_string(index=False))
    print("Pilot only: all six settings retained, no selected winner or convergence/SOTA claim.")


def smoke_worker(point):
    from stc_stage1 import smoke_federation
    config = replace(config_for(point), rounds=3, eval_stride=2, local_steps=1, batch_size=2)
    kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", config=config, seed=TRAIN_SEED, provenance={"synthetic": True})
    with tempfile.TemporaryDirectory() as folder:
        result = host.evaluated_run(**kwargs, checkpoint_path=Path(folder) / "complete.npz")
        path = Path(folder) / "resume.npz"
        host.evaluated_run(**kwargs, checkpoint_path=path, stop_after=2)
        resumed = host.evaluated_run(**kwargs, checkpoint_path=path)
        for field in ("model", "client_model", "strom_state"):
            np.testing.assert_array_equal(result[field].view(np.uint32), resumed[field].view(np.uint32))
        pd.testing.assert_frame_equal(result["evaluation"], resumed["evaluation"])
        return result["evaluation"].iloc[-1].model_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--point", choices=tuple(POINT_BY_ID))
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_strom_codec.py", "-v"],
                       cwd=REPO, env=env, check=True)
    if args.command in ("verify", "smoke"):
        points = (POINTS[0].point_id, POINTS[-1].point_id)
        expected = [smoke_worker(point) for point in points]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            if list(pool.map(smoke_worker, points)) != expected:
                raise AssertionError("Spawned Ström workers differ")
        print("Synthetic inputs: exact residual/model/prediction resume and spawned workers. No CIFAR research training.")
    elif args.command == "preflight":
        science = build_science()
        print("Frozen package, codecs and earlier sources: unchanged")
        print("6 new runs x 600 rounds; partition 3500, SGD 83500; 3 existing prefixes reused")
        print("Native gains .05/.1/1 correspond to FedAvg-equivalent gains 1/2/20")
        print("Runtime matches full-horizon evidence:", full.development.runtime_manifest() == science["runtime"])
    elif args.command == "run":
        run(args.workers, args.point)
    elif args.command == "status":
        if not (OUT / "protocol.json").exists():
            print("Ström codec pilot not started")
        else:
            science = load_contract()
            for point in POINT_BY_ID:
                receipt = read_receipt(point, science)
                state = receipt["state"] if receipt else "pending"
                path = OUT / "checkpoints" / f"{point}.npz"
                if receipt is None and path.exists():
                    with np.load(path, allow_pickle=False) as data:
                        state = "checkpoint " + str(json.loads(str(data["metadata"].item()))["progress"]["completed_round"])
                print(point, state)
    elif args.command == "analyze":
        analyze()


if __name__ == "__main__":
    main()
