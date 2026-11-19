"""Fresh matched confirmation of frozen tuned Event and calibrated Ström.

python experiments/event_strom_fresh_confirmation.py verify
python experiments/event_strom_fresh_confirmation.py preflight
python experiments/event_strom_fresh_confirmation.py run --workers 2
python experiments/event_strom_fresh_confirmation.py analyze
git add experiments/results/event_strom_fresh_confirmation_v1
git commit -m "Add fresh tuned Event and Strom confirmation results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
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
import event_codec_development as development
import event_codec_development_audit as development_audit
import strom_codec_host as strom
from neuromorphicfl.confirmation_metrics import add_regularization_columns, interval, target_crossings

full = development.full
OUT = REPO / "experiments/results/event_strom_fresh_confirmation_v1"
POINTS, SEEDS, ROUNDS, TARGETS = ("event", "strom"), tuple(range(6000, 7000, 100)), 1800, (.45, .50, .55)
EVENT_ID = "event_t0p05_q0p01"
METRIC_FIELDS = ("train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy")
METRICS = (("test_accuracy_difference_pp", "final_test_accuracy", False, 100),
           ("test_ce_difference", "final_test_ce", False, 1),
           ("train_objective_difference", "final_train_objective", False, 1),
           ("train_l2_penalty_difference", "final_train_l2_penalty", False, 1),
           ("unicast_traffic_ratio", "unicast_total_bits", True, 1),
           ("broadcast_traffic_ratio", "broadcast_total_bits", True, 1),
           ("upload_traffic_ratio", "uplink_bits", True, 1),
           ("downlink_traffic_ratio", "unicast_downlink_bits", True, 1))
atomic_json, atomic_csv, digest, text_hash = development.atomic_json, development.atomic_csv, development.digest, development.text_hash


def train_seed(index):
    if not 0 <= index < len(SEEDS):
        raise ValueError("Seed index outside the fixed fresh grid")
    return 80000 + SEEDS[index]


def config_for(point):
    if point not in POINTS:
        raise ValueError("Unknown fresh confirmation method")
    return replace(development.config_for_id(EVENT_ID if point == "event" else development.STROM_ID), rounds=ROUNDS)


def seed_freshness():
    # Snapshot of recorded JSON partition_seed/partition_seeds metadata, audited
    # before this campaign. Pin the snapshot so later new results cannot change
    # checkpoint identity. ResNet seeds 4600--5100 are deliberately avoided.
    prior = list(range(2500, 5200, 100))
    if set(SEEDS).intersection(prior):
        raise RuntimeError("Fresh seed grid overlaps recorded prior partitions")
    return {"audited_evidence_commit": "838167711b474764b5056c7a7270bbe486b06904",
            "scope": "recorded_JSON_partition_seed_and_partition_seeds_metadata_under_experiments_results",
            "recorded_prior_partition_seeds": prior, "selected_partition_seeds": list(SEEDS),
            "overlap": [], "freshness_means": "new_partition_and_SGD_seeds_not_a_new_test_dataset"}


def protocol_spec():
    return {"campaign": "event_strom_fresh_confirmation_v1", "architecture": "cifar_cnn", "dimension": 20570,
            "rounds": ROUNDS, "n_clients": 10, "regime": "strong", "dominant_class_fraction": .55,
            "training_examples": 50000, "test_examples": 10000, "training_diagnostic_examples": 5000,
            "partition_seeds": list(SEEDS), "training_seeds": [train_seed(i) for i in range(len(SEEDS))],
            "points": {p: asdict(config_for(p)) for p in POINTS}, "initialization_seed": 7777,
            "evaluation_rounds": list(full.sampled_rounds(config_for("event"))),
            "new_training_runs": 20, "reused_training_results": 0, "development_checkpoint_reuse": False,
            "previously_inspected_seed_exclusion": "all_seeds_fresh_no_pilot_excluded_from_summary",
            "primary_predictive_endpoint": "Event_minus_Strom_round_1800_test_accuracy_in_percentage_points",
            "primary_communication_accounting": "both_codecs_actual_frames_conservative_unicast",
            "secondary_endpoints": ["test_CE", "upload", "download", "final_unicast_ratio", "broadcast_supplementary",
                                    "regularized_training_diagnostics", "accuracy_target_traffic_rounds_and_hit_coverage"],
            "accuracy_targets": list(TARGETS), "target_rule": "first_saved_crossing_no_interpolation",
            "unreached_targets": "censored_no_cost_ratio_unless_both_reach_report_full_coverage",
            "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_on_log_scale",
            "uncertainty_scope": "fresh_partition_SGD_variation_on_fixed_previously_inspected_official_test_set",
            "equivalence_or_noninferiority_claim": False, "tuning_or_early_stopping": False,
            "endpoint": "round_1800_not_best_test_checkpoint", "checkpoint_stride": 60,
            "hash_format": "sha256_utf8_lf_v1"}


def selection_provenance():
    science, _, frozen, _ = development_audit.validate()
    if (frozen["selection"]["quality"]["point_id"] != EVENT_ID
            or frozen["selected_configs"]["quality"] != asdict(config_for("event"))
            or frozen["selected_configs"]["within_strom_budget"] != asdict(config_for("event"))
            or science["protocol"]["fixed_Strom_config"] != asdict(config_for("strom"))):
        raise RuntimeError("Frozen Event/Ström selection differs; preserve the confirmation settings")
    files = [development.OUT / name for name in ("protocol.json", "frozen_selection.json", "screening_selection.json",
                                                "split_manifest.json", "split_indices.csv")]
    files.append(development.paths(EVENT_ID, 1800)["receipt"])
    files.append(development.previous.OUT / "frozen_selection.json")
    return science, {"development_contract_sha256": digest(science),
                     "files": {p.relative_to(REPO).as_posix(): text_hash(p) for p in files},
                     "configs": {p: asdict(config_for(p)) for p in POINTS}}


def build_science():
    source, selected = selection_provenance()
    original = development.previous.frozen_base()  # Metadata only; no old test curves.
    return {"protocol": protocol_spec(), "selection_provenance": selected, "seed_freshness": seed_freshness(),
            "source_hashes": {**source["source_hashes"],
                              "experiments/event_codec_development_audit.py": text_hash(Path(development_audit.__file__)),
                              "experiments/strom_codec_development_audit.py": text_hash(REPO / "experiments/strom_codec_development_audit.py"),
                              "experiments/event_strom_fresh_confirmation.py": text_hash(Path(__file__))},
            "runtime": source["runtime"], "data_sha256": original["data_sha256"],
            "full_horizon_protocol_sha256": text_hash(full.OUT / "protocol.json")}


def build_contract():
    science = build_science()
    if full.development.runtime_manifest() != science["runtime"]:
        raise RuntimeError("Use the recorded Windows environment before fresh confirmation training")
    root = full.download_cifar10(full.DATA)
    if {name: full.development.file_hash(root / name) for name in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR training/test batch fingerprints differ")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Fresh confirmation protocol, sources or selection provenance changed")
    return science


def metadata(point, index, science):
    if point not in POINTS:
        raise ValueError("Unknown fresh confirmation method")
    training_seed = train_seed(index)
    return {"point": point, "seed_index": index, "partition_seed": SEEDS[index],
            "train_seed": training_seed, "config_sha256": digest(asdict(config_for(point))), "contract_sha256": digest(science),
            "evaluation_scope": "fresh_frozen_Event_Strom_full_training_confirmation", "evidence_origin": "new_fresh_full_training_run"}


def paths(point, index, out=OUT):
    train_seed(index)
    stem = f"{point}_s{index}_p{SEEDS[index]}_r{ROUNDS}"
    return {kind: out / f"{stem}_{kind}.{'json' if kind == 'receipt' else 'csv'}"
            for kind in ("summary", "rounds", "evaluation", "history", "receipt")}


def read_receipt(point, index, science, out=OUT):
    files = paths(point, index, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != metadata(point, index, science):
        raise RuntimeError("Fresh confirmation receipt identity differs")
    if receipt.get("state") == "complete":
        for kind in ("summary", "rounds", "evaluation", "history"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing or changed fresh evidence: {point}/{index}/{kind}")
    elif receipt.get("state") != "numerical_failure":
        raise RuntimeError("Every point requires a new fresh run; training-result reuse is prohibited")
    return receipt


def trajectory_tables(point, result, science):
    account = strom.traffic_summary if point == "strom" else full.traffic_summary
    result = {**result, "summary": account(result["rounds"], n_clients=science["protocol"]["n_clients"],
                                          dimension=science["protocol"]["dimension"])}
    assemble = strom.trajectory_tables if point == "strom" else full.trajectory_tables
    summary, history = assemble(result, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    return {"summary": summary, "history": history, "rounds": result["rounds"], "evaluation": result["evaluation"]}


def save_result(point, index, science, result, out=OUT):
    tables = trajectory_tables(point, result, science)
    files = paths(point, index, out)
    for kind, frame in tables.items():
        atomic_csv(frame.assign(**metadata(point, index, science)), files[kind])
    atomic_json({"identity": metadata(point, index, science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def verified_tables(point, index, science, out=OUT):
    receipt = read_receipt(point, index, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError("Incomplete fresh confirmation point")
    tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
              for kind, file in paths(point, index, out).items() if kind != "receipt"}
    for frame in tables.values():
        for field, value in metadata(point, index, science).items():
            if not frame[field].eq(value).all():
                raise RuntimeError("Fresh confirmation metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    if (rounds["round"].tolist() != list(range(1, ROUNDS + 1))
            or evaluation["round"].tolist() != science["protocol"]["evaluation_rounds"] or not rounds.method.eq(point).all()):
        raise RuntimeError("Fresh confirmation horizon/method/evaluation coverage differs")
    if (not np.isfinite(evaluation[list(METRIC_FIELDS)]).all().all()
            or not evaluation[["test_accuracy", "macro_accuracy", "worst_class_accuracy"]].apply(lambda c: c.between(0, 1)).all().all()
            or not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"])):
        raise RuntimeError("Invalid fresh predictions or model fingerprints")
    rebuilt = trajectory_tables(point, {"rounds": rounds, "evaluation": evaluation[["round", *METRIC_FIELDS, "model_sha256"]]}, science)
    for kind in ("summary", "history"):
        for field in rebuilt[kind]:
            actual, expected = tables[kind][field], rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=1e-12, atol=1e-12)):
                raise RuntimeError("Reconstructed fresh traffic/metrics differ")
    return tables


def run_point(point, index, science, out=OUT):
    if point not in POINTS:
        raise ValueError("Unknown fresh confirmation method")
    train_seed(index)
    lock = out / "locks" / (paths(point, index, out)["receipt"].stem + ".lock")
    with full.development.point_lock(lock):
        receipt = read_receipt(point, index, science, out)
        if receipt is not None:
            return f"skip {point} p{SEEDS[index]}: {receipt['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/runtime/data differs from the frozen fresh campaign")
        federation = full.make_cifar10_federation(root=full.DATA, regime="strong", seed=SEEDS[index], train_eval_size=5000, n_clients=10)

        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point} p{SEEDS[index]}: {row['round']}/{ROUNDS}, test_acc={evaluation['test_accuracy']:.4f}, "
                      f"test_ce={evaluation['test_ce']:.6f}", flush=True)

        fn = strom.evaluated_run if point == "strom" else full.evaluated_run
        kwargs = {"method": "event"} if point == "event" else {}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = fn(federation=federation, architecture="cifar_cnn", config=config_for(point), seed=train_seed(index),
                            checkpoint_path=out / "checkpoints" / (lock.stem + ".npz"), provenance=metadata(point, index, science),
                            progress_callback=progress, **kwargs)
        except FloatingPointError as exc:
            atomic_json({"identity": metadata(point, index, science), "state": "numerical_failure", "error": str(exc)},
                        paths(point, index, out)["receipt"])
            return f"numerical failure {point} p{SEEDS[index]}: {exc}"
        save_result(point, index, science, result, out)
        return f"done {point} p{SEEDS[index]}"


def run(workers, point=None, seed_index=None, out=OUT):
    if not 1 <= workers <= 20 or (point is not None and point not in POINTS):
        raise ValueError("Invalid worker count or method")
    if seed_index is not None:
        train_seed(seed_index)
    with full.development.point_lock(out / "locks" / "controller.lock"):
        science = build_contract()
        development.immutable_json(science, out / "protocol.json")
        jobs = [(p, i) for i in (range(len(SEEDS)) if seed_index is None else (seed_index,))
                for p in (POINTS if point is None else (point,)) if read_receipt(p, i, science, out) is None]
        print(f"Fresh paired campaign: {len(jobs)} new runs pending in this selection; zero reused trajectories", flush=True)
        if workers == 1:
            for p, i in jobs:
                print(run_point(p, i, science, out), flush=True)
        elif jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(run_point, p, i, science, out) for p, i in jobs]
                for future in as_completed(futures):
                    print(future.result(), flush=True)


def optimized(frame):
    return frame[frame.variant == "both_codecs"].copy()


def indexed(frame, point):
    rows = frame[frame.point == point].set_index("partition_seed")
    if rows.index.has_duplicates or set(rows.index) != set(SEEDS):
        raise ValueError("Missing or duplicate fresh paired seeds")
    return rows.sort_index()


def paired_endpoints(runs):
    event, baseline = (indexed(runs, point) for point in POINTS)
    rows, summaries = [], []
    for seed in SEEDS:
        for name, field, ratio, scale in METRICS:
            value = float(event.loc[seed, field] / baseline.loc[seed, field]) if ratio else float(scale * (event.loc[seed, field] - baseline.loc[seed, field]))
            rows.append({"partition_seed": seed, "candidate": "event", "reference": "strom", "metric": name, "value": value})
    pairs = pd.DataFrame(rows)
    for name, group in pairs.groupby("metric", sort=False):
        summaries.append({"candidate": "event", "reference": "strom", "metric": name,
                          **interval(group.value, ratio=name.endswith("_ratio")),
                          "direction": "Event_over_Strom" if name.endswith("_ratio") else "Event_minus_Strom",
                          "primary_predictive_endpoint": name == "test_accuracy_difference_pp"})
    return pairs, pd.DataFrame(summaries)


def target_outcomes(history):
    rows = []
    for (point, seed), frame in history.groupby(["point", "partition_seed"], sort=False):
        frame = frame.sort_values("round").copy()
        frame["uplink_packetized_bits"] = frame.uplink_bits
        for row in target_crossings(frame, targets=TARGETS, horizon=ROUNDS, comparison_id=point, partition_seed=int(seed)):
            row["point"] = point
            row["accuracy_at_crossing"] = (float(frame.loc[frame["round"] == row["first_sampled_round"], "test_accuracy"].iloc[0])
                                           if row["reached"] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_targets(crossings):
    rows = []
    for target in TARGETS:
        frame = crossings[crossings.target_accuracy == target]
        event, baseline = (indexed(frame, point) for point in POINTS)
        common = event.reached & baseline.reached
        ratios = event.loc[common, "unicast_total_bits"] / baseline.loc[common, "unicast_total_bits"]
        rows.append({"target_accuracy": target, "n_total": len(SEEDS), "n_event_reached": int(event.reached.sum()),
                     "n_strom_reached": int(baseline.reached.sum()), "n_common_hits": int(common.sum()),
                     **interval(ratios, ratio=True), "direction": "Event_over_Strom",
                     "estimand": "unicast_ratio_conditional_on_both_reaching_target",
                     "event_fewer_rounds": int((event.loc[common, "first_sampled_round"] < baseline.loc[common, "first_sampled_round"]).sum()),
                     "same_sampled_round": int((event.loc[common, "first_sampled_round"] == baseline.loc[common, "first_sampled_round"]).sum()),
                     "event_fewer_bits": int((ratios < 1).sum())})
    return pd.DataFrame(rows)


def descriptive_summary(runs):
    fields = ["final_test_accuracy", "final_test_ce", "final_train_objective", "final_train_l2_penalty", "final_parameter_l2_norm",
              "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"]
    table = runs.groupby("point")[fields].agg(["mean", "std", "count"])
    table.columns = ["_".join(c) for c in table.columns]
    return table.reset_index()


def plot(history, targets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    for point, label, color in (("event", "Tuned Event", "#D55E00"), ("strom", "Calibrated Ström", "#0072B2")):
        frame = history[history.point == point].groupby("round")
        acc, ce, bits = frame.test_accuracy.mean(), frame.test_ce.mean(), frame.unicast_total_bits.mean()
        axes[0, 0].plot(acc.index, 100 * acc, color=color, label=label)
        axes[0, 1].plot(bits / 1e6, 100 * acc, color=color, label=label)
        axes[1, 0].plot(ce.index, ce, color=color, label=label)
    for x, (_, row) in enumerate(targets.sort_values("target_accuracy").iterrows()):
        if np.isfinite(row.estimate):
            error = ([[max(0, row.estimate - row.ci95_low)], [max(0, row.ci95_high - row.estimate)]]
                     if np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high) else None)
            axes[1, 1].errorbar(x, row.estimate, yerr=error, fmt="o", color="#D55E00", capsize=3)
        anchor = row.ci95_high if np.isfinite(row.ci95_high) else row.estimate if np.isfinite(row.estimate) else 1
        axes[1, 1].annotate(f"{int(row.n_common_hits)}/{int(row.n_total)}", (x, anchor),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].set(xlabel="Communication rounds", ylabel="Mean test accuracy (%)", title="Ten fresh matched seeds")
    axes[0, 1].set(xlabel="Mean total unicast traffic (Mbit)", ylabel="Mean test accuracy (%)", title="Descriptive mean trajectories")
    axes[1, 0].set(xlabel="Communication rounds", ylabel="Mean test cross-entropy", title="Predictive loss remains visible")
    axes[1, 1].axhline(1, color="#777777", ls="--", lw=.8)
    axes[1, 1].set_xticks(range(len(TARGETS)), [f"{100 * t:g}%" for t in TARGETS])
    axes[1, 1].set_xlim(-.25, len(TARGETS) - .75)
    axes[1, 1].set(xlabel="Target test accuracy (labels: common hits)", ylabel="Event / Ström unicast traffic",
                   title="Conditional ratios and descriptive 95% intervals")
    axes[1, 1].margins(y=.18)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend(fontsize=8)
    fig.suptitle("Frozen tuned Event and calibrated Ström: fresh-seed confirmation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "confirmation.png", dpi=180)
    fig.savefig(out / "confirmation.pdf", metadata={"CreationDate": None, "ModDate": None})
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
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                completion.append({"point": point, "partition_seed": seed, "state": "invalid", "error": str(exc)})
    atomic_json({"expected": len(SEEDS) * len(POINTS), "results": completion}, out / "completion_audit.json")
    if any(row["state"] != "complete" for row in completion):
        raise RuntimeError("Fresh grid incomplete or failed; no seed is dropped. See completion_audit.json")
    tables = [verified_tables(point, i, science, out) for i in range(len(SEEDS)) for point in POINTS]
    summary = pd.concat([t["summary"] for t in tables], ignore_index=True)
    history = pd.concat([t["history"] for t in tables], ignore_index=True)
    runs = add_regularization_columns(optimized(summary), config_for("event").regularization)
    main_history = optimized(history)
    pairs, paired = paired_endpoints(runs)
    targets = target_outcomes(main_history)
    target_summary = paired_targets(targets)
    for name, frame in (("comparison.csv", summary), ("comparison_history.csv", history), ("main_runs.csv", runs),
                        ("endpoint_summary.csv", descriptive_summary(runs)), ("paired_per_seed.csv", pairs),
                        ("paired_summary.csv", paired), ("traffic_to_target.csv", targets), ("paired_target_summary.csv", target_summary)):
        atomic_csv(frame, out / name)
    atomic_json({"fresh_seed_pairs": len(SEEDS), "new_training_runs": len(tables), "reused_training_results": 0,
                 "verified_round_records": sum(len(t["rounds"]) for t in tables),
                 "verified_evaluated_model_fingerprints": sum(len(t["evaluation"]) for t in tables),
                 "frozen_selection_sha256": digest(science["selection_provenance"]),
                 "source_receipts_model_fingerprints_and_prefix_costs_verified": True,
                 "seed_freshness": science["seed_freshness"], "intervals": protocol_spec()["intervals"],
                 "equivalence_or_noninferiority_claim": False, "tuning_or_early_stopping": False,
                 "target_coverage_required_for_interpretation": True}, out / "analysis_audit.json")
    plot(main_history, target_summary, out)
    print(paired.to_string(index=False))
    print(target_summary.to_string(index=False))


def preflight():
    science = build_science()
    print("Validation-selected Event (threshold 0.05, quantum 0.01) and Ström (threshold 0.05, native gain 0.10) verified")
    print("Ten fresh matched pairs; 20 new runs; no reused full-training or validation trajectories")
    print("Partition seeds:", ", ".join(map(str, SEEDS)))
    print("SGD seeds:", ", ".join(str(train_seed(i)) for i in range(len(SEEDS))))
    print("Full 50000-image training set / 10000-image official test; compact CNN; 1800 rounds; frozen codecs")
    print("Prior recorded seed snapshot overlap:", science["seed_freshness"]["overlap"])
    print("Runtime matches recorded Windows environment:", full.development.runtime_manifest() == science["runtime"])
    print("All ten pairs retained; no test-based tuning, early stopping or favourable-seed exclusion")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("Fresh Event/Ström confirmation not started; 20 new runs required")
        return
    science = load_contract(out)
    for index, seed in enumerate(SEEDS):
        labels = []
        for point in POINTS:
            receipt = read_receipt(point, index, science, out)
            label = receipt["state"] if receipt else "pending"
            checkpoint = out / "checkpoints" / (paths(point, index, out)["receipt"].stem + ".npz")
            if receipt is None and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"].item()))
                if meta.get("provenance") != metadata(point, index, science):
                    raise RuntimeError("Fresh checkpoint provenance differs")
                label = f"checkpoint {meta['progress']['completed_round']}"
            labels.append(f"{point}: {label}")
        print(seed, "; ".join(labels))


def smoke_worker(job):
    from stc_stage1 import smoke_federation
    point, index = job
    config = replace(config_for(point), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
    fn = strom.evaluated_run if point == "strom" else full.evaluated_run
    kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", config=config, seed=train_seed(index),
                  provenance={"synthetic_point": point, "synthetic_seed_index": index, "config": asdict(config)})
    if point == "event":
        kwargs["method"] = "event"
    with tempfile.TemporaryDirectory() as folder:
        baseline_path, resumed_path = Path(folder) / "full.npz", Path(folder) / "resumed.npz"
        baseline = fn(**kwargs, checkpoint_path=baseline_path)
        fn(**kwargs, checkpoint_path=resumed_path, stop_after=2)
        resumed = fn(**kwargs, checkpoint_path=resumed_path)
        with np.load(baseline_path, allow_pickle=False) as before, np.load(resumed_path, allow_pickle=False) as after:
            if set(before.files) != set(after.files):
                raise AssertionError("Resumed checkpoint arrays differ")
            for field in before.files:
                if field != "metadata":
                    np.testing.assert_array_equal(before[field].view(np.uint32), after[field].view(np.uint32))
        pd.testing.assert_frame_equal(baseline["evaluation"], resumed["evaluation"], check_exact=True)
        fields = [c for c in baseline["rounds"] if c != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(baseline["rounds"][fields], resumed["rounds"][fields], check_exact=True)
        if baseline["rng_state"] != resumed["rng_state"]:
            raise AssertionError("Resumed RNG state differs")
        return {"point": point, "train_seed": train_seed(index), "model_sha256": resumed["evaluation"].iloc[-1].model_sha256,
                "summary_sha256": digest(resumed["summary"].to_dict("records"))}


def smoke():
    jobs = [(p, i) for i in (0, 9) for p in POINTS]
    expected = [smoke_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = list(pool.map(smoke_worker, jobs))
    if actual != expected:
        raise AssertionError("Spawned fresh pair seed/model/traffic mapping differs")
    print("Synthetic first/last pairs: exact model, client replica, Event/residual state, RNG, metrics and traffic resume; spawned workers match")
    print("No CIFAR research training was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--point", choices=POINTS)
    parser.add_argument("--seed-index", type=int, choices=range(len(SEEDS)))
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_event_strom_fresh_confirmation.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke()
    elif args.command == "smoke":
        smoke()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        run(args.workers, args.point, args.seed_index)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
