"""Common-seed core: 30 new dense/STC/EF-TopK runs and 20 verified pulse controls.

python experiments/core_fresh_comparison.py verify
python experiments/core_fresh_comparison.py preflight
python experiments/core_fresh_comparison.py run --workers 2
python experiments/core_fresh_comparison.py analyze
git add experiments/results/core_fresh_comparison_v1
git commit -m "Add common-seed five-method core comparison results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
import event_strom_fresh_confirmation as previous
import core_float_host as host
from neuromorphicfl.confirmation_metrics import interval

full = previous.full
OUT = REPO / "experiments/results/core_fresh_comparison_v1"
SEEDS, ROUNDS, TARGETS = previous.SEEDS, previous.ROUNDS, previous.TARGETS
REUSED, NEW = ("event", "strom"), ("dense", "stc", "ef_topk")
POINTS = (*REUSED, *NEW)
MAIN = {"event": "both_codecs", "strom": "both_codecs", "dense": "dense_float32",
        "stc": "lossless_stc", "ef_topk": "lossless_ef_topk"}
LABELS = {"event": "Tuned Event", "strom": "Calibrated Ström", "dense": "Scheduled Dense FedAvg",
          "stc": "STC p=1", "ef_topk": "EF-TopK 5%"}
atomic_json, atomic_csv, digest, text_hash = previous.atomic_json, previous.atomic_csv, previous.digest, previous.text_hash
train_seed = previous.train_seed


def config_for(point):
    if point in REUSED:
        return previous.config_for(point)
    if point == "stc":
        return full.config_for("stc_primary")
    if point in ("dense", "ef_topk"):
        return full.confirmation.reference_config("dense_scheduled" if point == "dense" else "ef_topk")
    raise ValueError("Unknown core method")


def protocol_spec():
    return {**previous.protocol_spec(), "campaign": "core_fresh_comparison_v1",
            "points": {p: asdict(config_for(p)) for p in POINTS}, "main_variants": MAIN,
            "new_training_runs": 30, "reused_training_results": 20, "new_methods": list(NEW), "reused_methods": list(REUSED),
            "primary_predictive_endpoint": "Event_minus_each_reference_round_1800_test_accuracy_pp_descriptive",
            "primary_communication_accounting": "actual_lossless_frames_conservative_unicast",
            "comparisons": [["event", p] for p in POINTS if p != "event"],
            "equal_tuning_budget_claim": False,
            "selection_policies": {"event_strom": "completed_frozen_validation_CE_selections",
                                   "dense": "frozen_M3_training_objective_schedule_g4_alpha0p3_scale100",
                                   "stc": "original_training_objective_primary_p1_gain2",
                                   "ef_topk": "archived_P3_training_CE_quality_choice_global_fraction0p05_gain1"},
            "STC_scope": "primary_p1_dense_ternary_both_residuals_not_p0p25_secondary",
            "EF_TopK_scope": "native_global_topk_weighted_client_delta_error_feedback_float32_values_lossless_wire_extension",
            "dense_scope": "scheduled_server_gain_native_dense_float32_updates",
            "development_checkpoint_reuse": False, "reused_control_source": "event_strom_fresh_confirmation_v1"}


def selection_provenance():
    _, selected, audit = full.confirmation.validated_evidence()
    if selected["choices"]["quality"]["point_id"] != "stc_u1_d1_g2":
        raise RuntimeError("Frozen primary STC selection differs")
    dense_path = REPO / "experiments/results/m3_dense_schedule/frozen_selection.json"
    dense = json.loads(dense_path.read_text())
    if (dense["gain0"], dense["gain_exponent"], dense["gain_scale"], dense["rounds"], dense["selection_metric"]) != (
            4, .3, 100, 1800, "final_train_objective"):
        raise RuntimeError("Frozen dense selection differs")
    ef_path = REPO / "experiments/results/p3_cifar10/selection.json"
    ef = json.loads(ef_path.read_text())
    if ef["quality"]["ef_topk"] != "ef_k05" or ef["selection_metric"] != "minimum final_train_ce":
        raise RuntimeError("Archived EF-TopK quality selection differs")
    base = config_for("event")
    for point in NEW:
        config = config_for(point)
        for field in ("rounds", "eval_stride", "local_steps", "local_lr", "batch_size", "regularization", "init_scale"):
            if getattr(config, field) != getattr(base, field):
                raise RuntimeError("Core local-learning protocol differs")
    return {"STC_development_selection_sha256": digest(selected), "STC_development_campaign_sha256": audit["campaign_sha256"],
            "frozen_dense_selection_sha256": text_hash(dense_path), "archived_EF_selection_sha256": text_hash(ef_path),
            "configs": {p: asdict(config_for(p)) for p in NEW}}


def reference_key(point, index):
    return f"{point}_s{index}"


def build_science():
    old = previous.load_contract()
    references = {}
    for i in range(len(SEEDS)):
        for point in REUSED:
            previous.verified_tables(point, i, old)
            references[reference_key(point, i)] = {kind: {"path": path.relative_to(REPO).as_posix(), "sha256": text_hash(path)}
                                                  for kind, path in previous.paths(point, i).items()}
    files = ("experiments/core_fresh_comparison.py", "experiments/core_float_host.py",
             "experiments/stc_cifar10_development_audit.py", "experiments/stc_cifar10_development.py",
             "experiments/m3_dense_schedule.py", "experiments/p12_headline_ten_seed.py")
    return {"protocol": protocol_spec(), "runtime": old["runtime"], "data_sha256": old["data_sha256"],
            "source_hashes": {**old["source_hashes"], **{p: text_hash(REPO / p) for p in files}},
            "selection_provenance": selection_provenance(), "reference_contract_sha256": digest(old),
            "references": references, "partition_seed_provenance": old["seed_freshness"]}


def build_contract():
    science = build_science()
    if full.development.runtime_manifest() != science["runtime"]:
        raise RuntimeError("Use the recorded Windows training environment")
    root = full.download_cifar10(full.DATA)
    if {n: full.development.file_hash(root / n) for n in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR batch fingerprints differ")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text())
    if science != build_science():
        raise RuntimeError("Core protocol, sources, selections or reference evidence changed")
    return science


def metadata(point, index, science):
    config = config_for(point)
    training_seed = train_seed(index)
    return {"point": point, "seed_index": index, "partition_seed": SEEDS[index], "train_seed": training_seed,
            "config_sha256": digest(asdict(config)), "contract_sha256": digest(science),
            "evaluation_scope": "common_seed_five_method_core", "evidence_origin": "verified_reuse" if point in REUSED else "new_run"}


def paths(point, index, out=OUT):
    train_seed(index)
    stem = f"{point}_s{index}_p{SEEDS[index]}_r{ROUNDS}"
    return {kind: out / f"{stem}_{kind}.{'json' if kind == 'receipt' else 'csv'}"
            for kind in ("summary", "history", "rounds", "evaluation", "receipt")}


def read_receipt(point, index, science, out=OUT):
    files = paths(point, index, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text())
    if receipt.get("identity") != metadata(point, index, science):
        raise RuntimeError("Core receipt identity differs")
    state = receipt.get("state")
    if point in REUSED:
        expected = science["references"][reference_key(point, index)]
        if state != "reused" or receipt.get("references") != expected:
            raise RuntimeError("Pulse controls require verified original references")
        for entry in expected.values():
            if text_hash(REPO / entry["path"]) != entry["sha256"]:
                raise RuntimeError("Reused pulse evidence changed")
    elif state == "complete":
        for kind in ("summary", "history", "rounds", "evaluation"):
            if text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError("New core evidence changed")
    elif state != "numerical_failure":
        raise RuntimeError("New core baselines cannot reuse old trajectories")
    return receipt


def register_reuse(science, out=OUT):
    for i in range(len(SEEDS)):
        for point in REUSED:
            value = {"identity": metadata(point, i, science), "state": "reused",
                     "references": science["references"][reference_key(point, i)]}
            if read_receipt(point, i, science, out) is None:
                atomic_json(value, paths(point, i, out)["receipt"])


def trajectory_tables(point, result, science):
    account = full.traffic_summary if point == "stc" else host.traffic_summary
    kwargs = {"n_clients": science["protocol"]["n_clients"], "dimension": science["protocol"]["dimension"]}
    rounds, evaluation = result["rounds"], result["evaluation"]
    summary = account(rounds, **kwargs)
    curves = [account(rounds[rounds["round"] <= int(row["round"])], **kwargs).assign(**row.to_dict()) for _, row in evaluation.iterrows()]
    for field in previous.METRIC_FIELDS:
        summary["final_" + field] = float(evaluation.iloc[-1][field])
    summary["model_sha256"] = evaluation.iloc[-1].model_sha256
    return {"summary": summary, "history": pd.concat(curves, ignore_index=True), "rounds": rounds, "evaluation": evaluation}


def save_result(point, index, science, result, out=OUT):
    tables = trajectory_tables(point, result, science)
    files = paths(point, index, out)
    for kind, frame in tables.items():
        atomic_csv(frame.assign(**metadata(point, index, science)), files[kind])
    atomic_json({"identity": metadata(point, index, science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def verified_tables(point, index, science, out=OUT):
    receipt = read_receipt(point, index, science, out)
    if receipt is None:
        raise RuntimeError("Missing core result")
    if point in REUSED:
        old = json.loads((previous.OUT / "protocol.json").read_text())
        if digest(old) != science["reference_contract_sha256"]:
            raise RuntimeError("Reused pulse contract differs")
        return {k: v.assign(**metadata(point, index, science)) for k, v in previous.verified_tables(point, index, old).items()}
    if receipt["state"] != "complete":
        raise RuntimeError("Incomplete core baseline")
    tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
              for kind, file in paths(point, index, out).items() if kind != "receipt"}
    for frame in tables.values():
        for field, value in metadata(point, index, science).items():
            if not frame[field].eq(value).all():
                raise RuntimeError("Core result metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    if (rounds["round"].tolist() != list(range(1, ROUNDS + 1)) or not rounds.method.eq(point).all()
            or evaluation["round"].tolist() != science["protocol"]["evaluation_rounds"]
            or not np.isfinite(evaluation[list(previous.METRIC_FIELDS)]).all().all()
            or not evaluation[["test_accuracy", "macro_accuracy", "worst_class_accuracy"]].apply(lambda c: c.between(0, 1)).all().all()
            or not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"])):
        raise RuntimeError("Core horizon/predictions/model fingerprints differ")
    rebuilt = trajectory_tables(point, {"rounds": rounds, "evaluation": evaluation[["round", *previous.METRIC_FIELDS, "model_sha256"]]}, science)
    for kind in ("summary", "history"):
        for field in rebuilt[kind]:
            actual, expected = tables[kind][field], rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=1e-12, atol=1e-12)):
                raise RuntimeError("Reconstructed core metrics/traffic differ")
    return tables


def run_point(point, index, science, out=OUT):
    if point not in NEW:
        raise ValueError("Only the three missing baselines may be trained")
    train_seed(index)
    lock = out / "locks" / (paths(point, index, out)["receipt"].stem + ".lock")
    with full.development.point_lock(lock):
        receipt = read_receipt(point, index, science, out)
        if receipt is not None:
            return f"skip {point} p{SEEDS[index]}: {receipt['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker differs from frozen core protocol")
        federation = full.make_cifar10_federation(root=full.DATA, regime="strong", seed=SEEDS[index], train_eval_size=5000, n_clients=10)
        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point} p{SEEDS[index]}: {row['round']}/{ROUNDS}, test_acc={evaluation['test_accuracy']:.4f}, "
                      f"test_ce={evaluation['test_ce']:.6f}", flush=True)
        fn = full.evaluated_run if point == "stc" else host.evaluated_run
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = fn(federation=federation, architecture="cifar_cnn", method=point, config=config_for(point), seed=train_seed(index),
                            checkpoint_path=out / "checkpoints" / (lock.stem + ".npz"), provenance=metadata(point, index, science),
                            progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": metadata(point, index, science), "state": "numerical_failure", "error": str(exc)},
                        paths(point, index, out)["receipt"])
            return f"numerical failure {point} p{SEEDS[index]}: {exc}"
        save_result(point, index, science, result, out)
        return f"done {point} p{SEEDS[index]}"


def run(workers, point=None, seed_index=None, out=OUT):
    if not 1 <= workers <= 30 or (point is not None and point not in NEW):
        raise ValueError("Invalid worker count or method")
    if seed_index is not None:
        train_seed(seed_index)
    with full.development.point_lock(out / "locks/controller.lock"):
        science = build_contract()
        previous.development.immutable_json(science, out / "protocol.json")
        register_reuse(science, out)
        jobs = [(p, i) for i in (range(len(SEEDS)) if seed_index is None else (seed_index,))
                for p in (NEW if point is None else (point,)) if read_receipt(p, i, science, out) is None]
        print(f"20 verified pulse controls; {len(jobs)} new baseline runs pending in this selection", flush=True)
        if workers == 1:
            for p, i in jobs:
                print(run_point(p, i, science, out), flush=True)
        elif jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                for future in as_completed([pool.submit(run_point, p, i, science, out) for p, i in jobs]):
                    print(future.result(), flush=True)


def optimized(frame):
    return frame[frame.variant == frame.point.map(MAIN)].copy()


def paired_endpoints(runs):
    rows, summaries = [], []
    event = previous.indexed(runs, "event")
    for reference in POINTS[1:]:
        baseline = previous.indexed(runs, reference)
        for name, field, ratio, scale in previous.METRICS:
            values = event[field] / baseline[field] if ratio else scale * (event[field] - baseline[field])
            rows.extend({"partition_seed": int(seed), "candidate": "event", "reference": reference, "metric": name, "value": float(value)}
                        for seed, value in values.items())
            summaries.append({"candidate": "event", "reference": reference, "metric": name, **interval(values, ratio=ratio),
                              "direction": "candidate_over_reference" if ratio else "candidate_minus_reference",
                              "primary_predictive_endpoint": name == "test_accuracy_difference_pp"})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def paired_targets(targets):
    rows = []
    for reference in POINTS[1:]:
        for target in TARGETS:
            frame = targets[targets.target_accuracy == target]
            event, baseline = (previous.indexed(frame, p) for p in ("event", reference))
            common = event.reached & baseline.reached
            ratios = event.loc[common, "unicast_total_bits"] / baseline.loc[common, "unicast_total_bits"]
            rows.append({"candidate": "event", "reference": reference, "target_accuracy": target, "n_total": len(SEEDS),
                         "n_candidate_reached": int(event.reached.sum()), "n_reference_reached": int(baseline.reached.sum()),
                         "n_common_hits": int(common.sum()), **interval(ratios, ratio=True), "direction": "candidate_over_reference",
                         "estimand": "unicast_ratio_conditional_on_both_reaching_target",
                         "candidate_fewer_rounds": int((event.loc[common, "first_sampled_round"] < baseline.loc[common, "first_sampled_round"]).sum()),
                         "same_sampled_round": int((event.loc[common, "first_sampled_round"] == baseline.loc[common, "first_sampled_round"]).sum()),
                         "candidate_fewer_bits": int((ratios < 1).sum())})
    return pd.DataFrame(rows)


def plot(history, targets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8))
    colors = dict(zip(POINTS, ("#D55E00", "#0072B2", "#009E73", "#CC79A7", "#777777")))
    for point in POINTS:
        group = history[history.point == point].groupby("round")
        accuracy, ce, bits = group.test_accuracy.mean(), group.test_ce.mean(), group.unicast_total_bits.mean()
        for axis, x, y in ((axes[0, 0], accuracy.index, 100 * accuracy), (axes[0, 1], bits / 1e6, 100 * accuracy),
                           (axes[1, 0], ce.index, ce)):
            axis.plot(x, y, color=colors[point], label=LABELS[point])
    for j, reference in enumerate(POINTS[1:]):
        axes[1, 1].plot([], [], "o", color=colors[reference], label=LABELS[reference])
        rows = targets[targets.reference == reference].sort_values("target_accuracy")
        for x, (_, row) in enumerate(rows.iterrows()):
            position = x + (j - 1.5) * .16
            if np.isfinite(row.estimate):
                error = [[max(0, row.estimate - row.ci95_low)], [max(0, row.ci95_high - row.estimate)]] if np.isfinite(row.ci95_low) else None
                axes[1, 1].errorbar(position, row.estimate, yerr=error, fmt="o", capsize=2,
                                    color=colors[reference])
            anchor = row.ci95_high if np.isfinite(row.ci95_high) else row.estimate if np.isfinite(row.estimate) else 1
            axes[1, 1].annotate(f"{int(row.n_common_hits)}/{int(row.n_total)}", (position, anchor),
                                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6, color=colors[reference])
    axes[0, 0].set(xlabel="Communication rounds", ylabel="Mean test accuracy (%)", title="Ten matched seeds")
    axes[0, 1].set(xlabel="Mean total unicast traffic (Mbit; log scale)", ylabel="Mean test accuracy (%)", xscale="log",
                   title="Descriptive mean trajectories")
    axes[1, 0].set(xlabel="Communication rounds", ylabel="Mean test cross-entropy", title="Predictive loss")
    axes[1, 1].axhline(1, ls="--", lw=.8, color="#777777")
    axes[1, 1].set_xticks(range(len(TARGETS)), [f"{100*t:g}%" for t in TARGETS])
    axes[1, 1].set(xlabel="Target test accuracy (labels: common hits)", ylabel="Event / reference unicast traffic", yscale="log",
                   title="First-crossing ratios and descriptive 95% intervals")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Common-seed five-method core comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    completion = []
    for i, seed in enumerate(SEEDS):
        for point in POINTS:
            try:
                receipt = read_receipt(point, i, science, out)
                completion.append({"point": point, "partition_seed": seed, "state": receipt["state"] if receipt else "pending",
                                   "error": receipt.get("error", "") if receipt else ""})
            except (RuntimeError, OSError, KeyError, ValueError) as exc:
                completion.append({"point": point, "partition_seed": seed, "state": "invalid", "error": str(exc)})
    atomic_json({"expected": 50, "results": completion}, out / "completion_audit.json")
    if any(r["state"] != ("reused" if r["point"] in REUSED else "complete") for r in completion):
        raise RuntimeError("Core grid incomplete or failed; no seed is dropped. See completion_audit.json")
    tables = [verified_tables(p, i, science, out) for i in range(len(SEEDS)) for p in POINTS]
    summary, history = (pd.concat([t[k] for t in tables], ignore_index=True) for k in ("summary", "history"))
    runs = previous.add_regularization_columns(optimized(summary), config_for("event").regularization)
    main_history = optimized(history)
    pairs, paired = paired_endpoints(runs)
    targets = previous.target_outcomes(main_history)
    target_summary = paired_targets(targets)
    for name, frame in (("comparison.csv", summary), ("comparison_history.csv", history), ("main_runs.csv", runs),
                        ("endpoint_summary.csv", previous.descriptive_summary(runs)), ("paired_per_seed.csv", pairs),
                        ("paired_summary.csv", paired), ("traffic_to_target.csv", targets), ("paired_target_summary.csv", target_summary)):
        atomic_csv(frame, out / name)
    atomic_json({"new_training_results": 30, "verified_reused_controls": 20, "seed_pairs": 10,
                 "verified_round_records": sum(len(t["rounds"]) for t in tables),
                 "verified_evaluated_model_fingerprints": sum(len(t["evaluation"]) for t in tables),
                 "all_expected_results_retained": True, "original_pulse_results_modified": False,
                 "test_tuning_or_early_stopping": False, "equivalence_or_noninferiority_claim": False,
                 "intervals": science["protocol"]["intervals"], "source_and_config_sha256": digest(science)}, out / "analysis_audit.json")
    plot(main_history, target_summary, out)
    print(paired.to_string(index=False))
    print(target_summary.to_string(index=False))


def preflight():
    science = build_science()
    print("Common-seed core: 30 new baseline runs; 20 verified Event/Ström controls; 1800 rounds")
    print("Dense: frozen g(r)=4*(1+r/100)^(-0.3); STC: p_up=p_down=1, gain=2; EF-TopK: global 5%, gain=1")
    print("Seeds:", SEEDS, "SGD:", [train_seed(i) for i in range(len(SEEDS))])
    print("Compact CIFAR CNN; full 50000 training / 10000 test images; conservative unicast primary")
    print("EF-TopK: exact float32 payload/support coding and aggregate/replay portfolio; native residual rule preserved")
    print("Runtime matches recorded Windows environment:", full.development.runtime_manifest() == science["runtime"])
    print("Frozen settings; no test-based tuning, early stopping, or favourable-seed exclusion")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("Core campaign not started: 30 new baselines, 20 existing pulse controls")
        return
    science = load_contract(out)
    for i, seed in enumerate(SEEDS):
        labels = []
        for point in POINTS:
            receipt = read_receipt(point, i, science, out)
            label = receipt["state"] if receipt else "pending"
            checkpoint = out / "checkpoints" / (paths(point, i, out)["receipt"].stem + ".npz")
            if receipt is None and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"].item()))
                if meta["provenance"] != metadata(point, i, science):
                    raise RuntimeError("Core checkpoint provenance differs")
                label = f"checkpoint {meta['progress']['completed_round']}"
            labels.append(f"{point}: {label}")
        print(seed, "; ".join(labels))


def smoke_worker(job):
    from dataclasses import replace
    from stc_stage1 import smoke_federation
    point, index = job
    config = replace(config_for(point), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
    fn = full.evaluated_run if point == "stc" else host.evaluated_run
    kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method=point, config=config,
                  seed=train_seed(index), provenance={"synthetic_point": point, "index": index})
    with tempfile.TemporaryDirectory() as folder:
        a, b = Path(folder) / "full.npz", Path(folder) / "resumed.npz"
        before = fn(**kwargs, checkpoint_path=a)
        fn(**kwargs, checkpoint_path=b, stop_after=2)
        after = fn(**kwargs, checkpoint_path=b)
        with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise AssertionError("Checkpoint array names differ")
            for field in left.files:
                if field != "metadata":
                    np.testing.assert_array_equal(left[field].view(np.uint32), right[field].view(np.uint32))
        fields = [c for c in before["rounds"] if c != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(before["rounds"][fields], after["rounds"][fields], check_exact=True)
        pd.testing.assert_frame_equal(before["evaluation"], after["evaluation"], check_exact=True)
        if before["rng_state"] != after["rng_state"]:
            raise AssertionError("Resumed RNG differs")
        return {"point": point, "seed": train_seed(index), "model": after["evaluation"].iloc[-1].model_sha256,
                "traffic": digest(after["summary"].to_dict("records"))}


def smoke():
    jobs = [(p, i) for i in (0, 9) for p in NEW]
    expected = [smoke_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = list(pool.map(smoke_worker, jobs))
    if actual != expected:
        raise AssertionError("Spawned baseline workers differ")
    print("Synthetic first/last seeds: all three methods resume exactly; spawned workers match. No CIFAR research training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, choices=range(1, 31), default=1)
    parser.add_argument("--point", choices=NEW)
    parser.add_argument("--seed-index", type=int, choices=range(10))
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_core_fresh_comparison.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke()
    elif args.command == "run":
        run(args.workers, args.point, args.seed_index)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
