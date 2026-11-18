"""Ten matched seeds, frozen codecs, 1800 rounds; reuse the observed first seed.

python experiments/event_codec_ten_seed.py verify
python experiments/event_codec_ten_seed.py preflight
python experiments/event_codec_ten_seed.py run --workers 2
python experiments/event_codec_ten_seed.py analyze
git add experiments/results/event_codec_ten_seed_v1
git commit -m "Add ten-seed Event and STC codec confirmation results"
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

import event_codec_full_horizon as full
from neuromorphicfl.confirmation_metrics import add_regularization_columns, interval, target_crossings

OUT = REPO / "experiments/results/event_codec_ten_seed_v1"
SEEDS = tuple(range(3500, 4500, 100))
POINTS, ROUNDS, TARGETS = full.POINTS, 1800, full.TARGETS
atomic_json, atomic_csv = full.atomic_json, full.atomic_csv
digest, text_hash = full.digest, full.text_hash
config_for = full.config_for
METRICS = (
    ("test_accuracy_difference_pp", "final_test_accuracy", False, 100),
    ("test_ce_difference", "final_test_ce", False, 1),
    ("train_objective_difference", "final_train_objective", False, 1),
    ("train_l2_penalty_difference", "final_train_l2_penalty", False, 1),
    ("unicast_traffic_ratio", "unicast_total_bits", True, 1),
    ("broadcast_traffic_ratio", "broadcast_total_bits", True, 1),
    ("upload_traffic_ratio", "uplink_bits", True, 1),
    ("downlink_traffic_ratio", "unicast_downlink_bits", True, 1),
)


def train_seed(index):
    if not 0 <= index < len(SEEDS):
        raise ValueError("Seed index is outside the fixed ten-seed grid")
    return 80000 + SEEDS[index]


def protocol_spec():
    return {"campaign": "event_codec_ten_seed_v1", "rounds": ROUNDS, "architecture": "cifar_cnn",
            "dimension": full.LAYOUT.dimension, "n_clients": 10, "regime": "strong",
            "partition_seeds": list(SEEDS), "training_seeds": [train_seed(i) for i in range(len(SEEDS))],
            "points": {point: asdict(config_for(point)) for point in POINTS},
            "evaluation_rounds": list(full.sampled_rounds(config_for("event"))),
            "expected_point_seed_results": 30, "reused_results": 3, "new_training_runs": 27,
            "previously_inspected_seed": SEEDS[0], "sensitivity_excludes_seed": SEEDS[0],
            "scopes": {"all_10": "all seeds including previously inspected pilot",
                       "additional_9": "exclude the inspected seed 3500"},
            "primary_endpoint": "Event minus STC p=1 final-round test accuracy in percentage points",
            "secondary_endpoints": ["test_CE", "upload", "download", "unicast_and_broadcast_total",
                                    "first_saved_accuracy_target_cost_and_hit_coverage"],
            "accuracy_targets": list(TARGETS), "target_rule": "first_saved_sample_no_interpolation",
            "unreached_targets": "censored_no_achievement_cost",
            "target_ratio": "Event_over_reference_conditional_on_both_reaching_target_report_coverage",
            "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_on_log_scale",
            "uncertainty_scope": "partition_and_SGD_variation_on_fixed_dataset_not_new_test_datasets",
            "equivalence_or_noninferiority_claim": False,
            "endpoint": "round_1800_not_best_test_checkpoint", "tuning_or_early_stopping": False,
            "checkpoint_stride": 60, "hash_format": "sha256_utf8_lf_v1",
            "source_policy": "reuse_frozen_full_horizon_host_codecs_and_atomic_prediction_checkpoints"}


def source_hashes():
    return {**full.source_hashes(), "experiments/event_codec_ten_seed.py": text_hash(Path(__file__))}


def seed_zero_provenance():
    original = full.load_contract()
    points = {}
    for point in POINTS:
        full.verified_point(point, original)
        files = full.paths(point)
        points[point] = {"files": {kind: {"path": path.relative_to(REPO).as_posix(), "sha256": text_hash(path)}
                                  for kind, path in files.items()}, "source_contract_sha256": digest(original)}
    return {"protocol_sha256": text_hash(full.OUT / "protocol.json"),
            "contract_sha256": digest(original), "points": points}


def event_references():
    return [full.confirmation.load_reference("event", i)[2] for i in range(len(SEEDS))]


def build_science():
    original = full.load_contract()
    return {"protocol": protocol_spec(), "source_hashes": source_hashes(),
            "runtime": original["runtime"], "data_sha256": original["data_sha256"],
            "seed_zero": seed_zero_provenance(), "event_references": event_references()}


def build_contract():
    science = build_science()
    actual = full.build_contract()  # Checks actual runtime and raw batches.
    if digest(actual) != science["seed_zero"]["contract_sha256"]:
        raise RuntimeError("Current execution differs from the frozen full-horizon experiment")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Ten-seed protocol, sources, references or reused evidence changed")
    return science


def metadata(point, index, science):
    return {"point": point, "seed_index": index, "partition_seed": SEEDS[index], "train_seed": train_seed(index),
            "evaluation_scope": "ten_seed_fixed_codec_confirmation", "contract_sha256": digest(science),
            "evidence_origin": "reused_full_horizon" if index == 0 else "new_codec_run",
            "source_contract_sha256": science["seed_zero"]["contract_sha256"] if index == 0 else digest(science)}


def identity(point, index, science):
    return {**metadata(point, index, science), "config_sha256": digest(asdict(config_for(point)))}


def paths(point, index, out=OUT):
    stem = f"{point}_s{index}_p{SEEDS[index]}_r{ROUNDS}"
    return {kind: out / f"{stem}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("summary", "rounds", "evaluation", "history", "receipt")}


def read_receipt(point, index, science, out=OUT):
    files = paths(point, index, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != identity(point, index, science):
        raise RuntimeError("Ten-seed receipt identity differs")
    if receipt["state"] == "reused":
        if index != 0 or receipt.get("source") != science["seed_zero"]["points"][point]:
            raise RuntimeError("Invalid reused evidence identity")
        for info in receipt["source"]["files"].values():
            if text_hash(REPO / info["path"]) != info["sha256"]:
                raise RuntimeError("Reused full-horizon evidence changed")
    elif receipt["state"] == "complete":
        if index == 0:
            raise RuntimeError("The inspected seed must be reused, not retrained")
        for kind in ("summary", "rounds", "evaluation", "history"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing or changed evidence: {point}, seed {index}, {kind}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown ten-seed completion state")
    return receipt


def register_reuse(science, out=OUT):
    for point in POINTS:
        file = paths(point, 0, out)["receipt"]
        with full.development.point_lock(out / "locks" / (file.stem + ".lock")):
            if read_receipt(point, 0, science, out) is None:
                atomic_json({"identity": identity(point, 0, science), "state": "reused",
                             "source": science["seed_zero"]["points"][point]}, file)


def save_result(point, index, science, result, out=OUT):
    if index == 0:
        raise ValueError("Seed zero is referenced from the completed single-seed experiment")
    summary, history = full.trajectory_tables(result, n_clients=science["protocol"]["n_clients"],
                                             dimension=science["protocol"]["dimension"])
    tables = {"summary": summary, "history": history, "evaluation": result["evaluation"], "rounds": result["rounds"]}
    files = paths(point, index, out)
    for kind, table in tables.items():
        atomic_csv(table.assign(**metadata(point, index, science)), files[kind])
    atomic_json({"identity": identity(point, index, science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def run_point(point, index, science, out=OUT):
    if index == 0:
        raise ValueError("Seed zero is reused and must never enter the training queue")
    lock = out / "locks" / (paths(point, index, out)["receipt"].stem + ".lock")
    with full.development.point_lock(lock):
        old = read_receipt(point, index, science, out)
        if old is not None:
            return f"reuse {point}, seed {SEEDS[index]}: {old['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/data/runtime differs from the ten-seed contract")
        fed = full.make_cifar10_federation(root=full.DATA, regime="strong", seed=SEEDS[index],
                                          train_eval_size=5000, n_clients=10)

        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point} p{SEEDS[index]}: {row['round']}/{ROUNDS}, "
                      f"test_acc={evaluation['test_accuracy']:.4f}, test_ce={evaluation['test_ce']:.6f}", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = full.evaluated_run(federation=fed, architecture="cifar_cnn",
                                            method="event" if point == "event" else "stc", config=config_for(point),
                                            seed=train_seed(index), checkpoint_path=out / "checkpoints" / (lock.stem + ".npz"),
                                            provenance=identity(point, index, science), progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": identity(point, index, science), "state": "numerical_failure",
                         "error": str(exc)}, paths(point, index, out)["receipt"])
            return f"numerical failure {point}, seed {SEEDS[index]}: {exc}"
        save_result(point, index, science, result, out)
        return f"done {point}, seed {SEEDS[index]}"


def run(workers, point=None, seed_index=None):
    science = build_contract()
    file = OUT / "protocol.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) != science:
        raise RuntimeError("Existing ten-seed contract differs; preserve its evidence")
    if not file.exists():
        atomic_json(science, file)
    register_reuse(science)
    indices = range(1, len(SEEDS)) if seed_index is None else (() if seed_index == 0 else (seed_index,))
    points = POINTS if point is None else (point,)
    jobs = [(name, i) for i in indices for name in points if read_receipt(name, i, science) is None]
    print(f"3 inspected-seed results reused; {len(jobs)} new runs pending in this selection", flush=True)
    if workers == 1:
        for name, i in jobs:
            print(run_point(name, i, science), flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_point, name, i, science) for name, i in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)


def verified_tables(point, index, science, out=OUT):
    receipt = read_receipt(point, index, science, out)
    if receipt is None or receipt["state"] not in ("complete", "reused"):
        raise RuntimeError("Incomplete ten-seed point")
    if index == 0:
        tables = full.verified_point(point, full.load_contract())
        tables = {kind: table.assign(**metadata(point, index, science)) for kind, table in tables.items()}
    else:
        tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
                  for kind, file in paths(point, index, out).items() if kind != "receipt"}
    for kind, table in tables.items():
        for field, value in metadata(point, index, science).items():
            if not table[field].eq(value).all():
                raise RuntimeError(f"Evidence metadata mismatch: {point}/{index}/{kind}/{field}")
    evaluation, rounds = tables["evaluation"], tables["rounds"]
    if rounds["round"].tolist() != list(range(1, ROUNDS + 1)) or evaluation["round"].tolist() != science["protocol"]["evaluation_rounds"]:
        raise RuntimeError("Ten-seed horizon/evaluation coverage differs")
    fields = ["train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy"]
    if not np.isfinite(evaluation[fields]).all().all() or not evaluation.test_accuracy.between(0, 1).all():
        raise RuntimeError("Invalid predictive metrics")
    if not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"]):
        raise RuntimeError("Predictive/codec model fingerprints differ")
    reconstructed = {"rounds": rounds, "evaluation": evaluation[["round", *fields, "model_sha256"]],
                     "summary": full.traffic_summary(rounds, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])}
    summary, history = full.trajectory_tables(reconstructed, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind, expected in (("summary", summary), ("history", history)):
        for field in expected:
            actual = tables[kind][field]
            match = (np.array_equal(actual, expected[field]) if field.endswith("bits") or field in
                     ("round", "rounds", "variant", "model_sha256", "roundtrip_verified") else
                     np.allclose(actual, expected[field], rtol=1e-12, atol=1e-12))
            if not match:
                raise RuntimeError(f"Reconstructed evidence differs: {point}/{index}/{kind}/{field}")
    return tables


def regression_check(point, index, tables):
    if point == "event":
        reference = full.confirmation.load_reference("event", index)[1]
        origin = "archived_event"
    else:
        native = full.confirmation.POINT_BY_ID[full.CONFIRMATION_IDS[point]]
        files = full.confirmation.paths(native, index)
        if not files[2].exists():
            return {"point": point, "partition_seed": SEEDS[index], "status": "native_confirmation_not_available"}
        receipt = full.confirmation.read_receipt(native, index, full.confirmation.load_execution())
        if receipt["state"] != "complete":
            return {"point": point, "partition_seed": SEEDS[index], "status": receipt["state"]}
        reference = pd.read_csv(files[1], float_precision="round_trip")
        origin = "native_STC_confirmation"
    evaluation, rounds = tables["evaluation"], tables["rounds"]
    if evaluation["round"].tolist() != reference["round"].tolist():
        raise RuntimeError("Regression sample schedules differ")
    for field in ("train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy"):
        if not np.allclose(evaluation[field], reference[field], rtol=1e-8, atol=1e-9):
            raise RuntimeError(f"Regression metric differs: {point}, {SEEDS[index]}, {field}")
    prefix = rounds.legacy_uplink_bits.cumsum().iloc[evaluation["round"].to_numpy(dtype=int) - 1]
    if not np.array_equal(prefix, reference.uplink_packetized_bits):
        raise RuntimeError("Native upload prefix regression differs")
    if point != "event":
        server = rounds.legacy_server_bits.cumsum().iloc[evaluation["round"].to_numpy(dtype=int) - 1]
        if not np.array_equal(server, reference.stc_downlink_packetized_bits):
            raise RuntimeError("Native server prefix regression differs")
    return {"point": point, "partition_seed": SEEDS[index], "status": "matched", "reference": origin}


def optimized(frame):
    return frame[[variant == full.MAIN_VARIANT[point] for point, variant in zip(frame.point, frame.variant)]].copy()


def paired_endpoints(runs):
    rows = []
    event = runs[runs.point == "event"].set_index("partition_seed")
    if event.index.has_duplicates:
        raise ValueError("Duplicate Event seed")
    for reference in POINTS[1:]:
        baseline = runs[runs.point == reference].set_index("partition_seed")
        if baseline.index.has_duplicates or set(baseline.index) != set(event.index):
            raise ValueError("Paired seed sets differ")
        for seed in sorted(event.index):
            a, b = event.loc[seed], baseline.loc[seed]
            for name, field, ratio, scale in METRICS:
                value = float(a[field] / b[field]) if ratio else float(scale * (a[field] - b[field]))
                rows.append({"reference": reference, "partition_seed": seed, "metric": name, "value": value})
    pairs = pd.DataFrame(rows)
    summaries = []
    for scope in ("all_10", "additional_9"):
        selected = pairs if scope == "all_10" else pairs[pairs.partition_seed != SEEDS[0]]
        for (reference, metric), group in selected.groupby(["reference", "metric"], sort=False):
            summaries.append({"scope": scope, "reference": reference, "metric": metric,
                              **interval(group.value, ratio=metric.endswith("_ratio")),
                              "primary_endpoint": reference == "stc_primary" and metric == "test_accuracy_difference_pp",
                              "direction": "Event_over_reference" if metric.endswith("_ratio") else "Event_minus_reference"})
    return pairs, pd.DataFrame(summaries)


def target_outcomes(history):
    rows = []
    for (point, variant, seed), group in history.groupby(["point", "variant", "partition_seed"], sort=False):
        group = group.sort_values("round").copy()
        group["uplink_packetized_bits"] = group.uplink_bits
        for row in target_crossings(group, targets=TARGETS, horizon=ROUNDS,
                                    comparison_id=point + "__" + variant, partition_seed=int(seed)):
            row.update(point=point, variant=variant)
            row["unicast_downlink_bits"] = (int(group.loc[group["round"] == row["first_sampled_round"], "unicast_downlink_bits"].iloc[0])
                                             if row["reached"] else float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_target_summary(crossings):
    rows = []
    for scope in ("all_10", "additional_9"):
        selected = crossings if scope == "all_10" else crossings[crossings.partition_seed != SEEDS[0]]
        for reference in POINTS[1:]:
            for target in TARGETS:
                a = selected[(selected.point == "event") & (selected.target_accuracy == target)].set_index("partition_seed")
                b = selected[(selected.point == reference) & (selected.target_accuracy == target)].set_index("partition_seed")
                if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index):
                    raise ValueError("Paired target seed sets differ")
                joined = a.join(b, lsuffix="_event", rsuffix="_reference")
                common = joined[joined.reached_event & joined.reached_reference]
                ratios = common.unicast_total_bits_event / common.unicast_total_bits_reference
                rows.append({"scope": scope, "reference": reference, "target_accuracy": target,
                             "n_total": len(joined), "n_event_reached": int(a.reached.sum()),
                             "n_reference_reached": int(b.reached.sum()), "n_common_hits": len(common),
                             **interval(ratios, ratio=True), "direction": "Event_over_reference",
                             "estimand": "conditional_on_both_reaching_target"})
    return pd.DataFrame(rows)


def descriptive_summary(runs):
    output = []
    fields = ["final_test_accuracy", "final_test_ce", "final_train_objective", "final_train_l2_penalty",
              "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"]
    for scope in ("all_10", "additional_9"):
        selected = runs if scope == "all_10" else runs[runs.partition_seed != SEEDS[0]]
        result = selected.groupby("point")[fields].agg(["mean", "std", "count"])
        result.columns = ["_".join(column) for column in result.columns]
        output.append(result.reset_index().assign(scope=scope))
    return pd.concat(output, ignore_index=True)


def plot(history, pairs, targets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    colors = {"event": "#333333", "stc_primary": "#D55E00", "stc_sparse": "#0072B2"}
    labels = {"event": "Event", "stc_primary": "STC p=1", "stc_sparse": "STC p=0.25"}
    for point in POINTS:
        group = history[history.point == point].groupby("round")
        accuracy = group.test_accuracy.agg(["mean", "std"])
        traffic = group.unicast_total_bits.mean()
        rounds = accuracy.index.to_numpy()
        axes[0, 0].plot(rounds, 100 * accuracy["mean"], color=colors[point], label=labels[point])
        axes[0, 0].fill_between(rounds, 100 * (accuracy["mean"] - accuracy["std"]),
                              100 * (accuracy["mean"] + accuracy["std"]), color=colors[point], alpha=.12)
        axes[0, 1].plot(traffic / 1e6, 100 * accuracy["mean"], color=colors[point], label=labels[point])
    axes[0, 0].set(xlabel="Communication round", ylabel="Test accuracy (%)", title="Mean and seed standard deviation")
    axes[0, 1].set(xlabel="Mean total unicast traffic (Mbit)", ylabel="Mean test accuracy (%)", title="Descriptive mean trajectories")
    accuracy_pairs = pairs[pairs.metric == "test_accuracy_difference_pp"]
    for i, reference in enumerate(POINTS[1:]):
        for scope, offset, marker in (("all_10", -.10, "o"), ("additional_9", .10, "s")):
            row = accuracy_pairs[(accuracy_pairs.reference == reference) & (accuracy_pairs.scope == scope)].iloc[0]
            axes[1, 0].errorbar(row.estimate, i + offset, xerr=[[max(0, row.estimate - row.ci95_low)], [max(0, row.ci95_high - row.estimate)]],
                                fmt=marker, color=colors[reference], mfc=colors[reference] if scope == "all_10" else "white",
                                capsize=3, label=("All 10 seeds" if scope == "all_10" else "Additional 9") if i == 0 else None)
    axes[1, 0].axvline(0, color="#777777", ls="--", lw=.8)
    axes[1, 0].set_yticks([0, 1], ["vs STC p=1", "vs STC p=0.25"])
    axes[1, 0].set(xlabel="Event minus STC final accuracy (percentage points)", title="Paired means and descriptive 95% intervals")
    for j, reference in enumerate(POINTS[1:]):
        selected = targets[(targets.scope == "all_10") & (targets.reference == reference)].sort_values("target_accuracy")
        for k, (_, row) in enumerate(selected.iterrows()):
            x = k + (j - .5) * .30
            label = labels[reference] if k == 0 else None
            if np.isfinite(row.estimate):
                if np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high):
                    axes[1, 1].errorbar(x, row.estimate, yerr=[[max(0, row.estimate - row.ci95_low)], [max(0, row.ci95_high - row.estimate)]],
                                        fmt="o", color=colors[reference], capsize=3, label=label)
                    anchor = row.ci95_high
                else:
                    axes[1, 1].plot(x, row.estimate, "o", color=colors[reference], label=label)
                    anchor = row.estimate
            else:
                anchor = 1.0
            axes[1, 1].annotate(f"{int(row.n_common_hits)}/{int(row.n_total)}", (x, anchor),
                                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=colors[reference])
    axes[1, 1].axhline(1, color="#777777", ls="--", lw=.8)
    axes[1, 1].set_xticks(range(len(TARGETS)), [f"{100 * value:g}%" for value in TARGETS])
    axes[1, 1].set_xlim(-.4, len(TARGETS) - .6)
    axes[1, 1].margins(y=.15)
    axes[1, 1].set(xlabel="Target test accuracy (labels: common-hit pairs)", ylabel="Event / STC unicast traffic",
                   title="Conditional geometric ratios and 95% intervals")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Frozen codec comparison: ten matched seeds, including inspected pilot", fontsize=12, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(out / "confirmation.png", dpi=180)
    fig.savefig(out / "confirmation.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    completion = []
    for i in range(len(SEEDS)):
        for point in POINTS:
            try:
                receipt = read_receipt(point, i, science, out)
                completion.append({"point": point, "partition_seed": SEEDS[i], "state": receipt["state"] if receipt else "pending",
                                   "error": receipt.get("error", "") if receipt else ""})
            except RuntimeError as exc:
                completion.append({"point": point, "partition_seed": SEEDS[i], "state": "invalid", "error": str(exc)})
    atomic_json({"expected": len(SEEDS) * len(POINTS), "results": completion}, out / "completion_audit.json")
    if any(row["state"] not in ("complete", "reused") for row in completion):
        raise RuntimeError("Ten-seed grid incomplete or failed; see completion_audit.json. No seed is dropped.")
    summaries, histories, references = [], [], []
    for i in range(len(SEEDS)):
        for point in POINTS:
            tables = verified_tables(point, i, science, out)
            references.append(regression_check(point, i, tables))
            summaries.append(tables["summary"])
            histories.append(tables["history"])
    summary, history = pd.concat(summaries, ignore_index=True), pd.concat(histories, ignore_index=True)
    main_runs = add_regularization_columns(optimized(summary), config_for("event").regularization)
    main_history = optimized(history)
    if len(main_runs) != len(SEEDS) * len(POINTS) or main_runs.duplicated(["point", "partition_seed"]).any():
        raise RuntimeError("Paired endpoint grid differs")
    pairs, paired_summary = paired_endpoints(main_runs)
    crossings = target_outcomes(history)
    target_summary = paired_target_summary(optimized(crossings))
    for name, table in (("comparison.csv", summary), ("comparison_history.csv", history), ("main_runs.csv", main_runs),
                        ("endpoint_summary.csv", descriptive_summary(main_runs)), ("paired_per_seed.csv", pairs),
                        ("paired_summary.csv", paired_summary), ("traffic_to_target.csv", crossings),
                        ("paired_target_summary.csv", target_summary)):
        atomic_csv(table, out / name)
    atomic_json({"seed_pairs": len(SEEDS), "reused_seed": SEEDS[0], "new_training_runs": 27,
                 "source_receipt_and_prefix_cost_checks": True, "regression_references": references,
                 "intervals": protocol_spec()["intervals"], "scopes": protocol_spec()["scopes"],
                 "target_coverage_required_for_interpretation": True,
                 "equivalence_or_noninferiority_claim": False}, out / "analysis_audit.json")
    plot(main_history, paired_summary, target_summary, out)
    print(paired_summary[paired_summary.scope == "all_10"].to_string(index=False))
    print(target_summary[target_summary.scope == "all_10"].to_string(index=False))


def preflight():
    science = build_science()
    print("Frozen pilot, codecs and full-horizon sources: unchanged")
    print("3 seed-3500 trajectories verified for reuse; 27 new runs at 3600,...,4400")
    print("1800 rounds, same test metrics and 45/50/55% targets; no tuning")
    print("Ten archived Event references validated:", len(science["event_references"]))
    print("Runtime matches full-horizon run:", full.development.runtime_manifest() == science["runtime"])
    print("Both all-ten and additional-nine summaries will be reported")


def status():
    if not (OUT / "protocol.json").exists():
        print("Ten-seed codec confirmation not started")
        return
    science = load_contract()
    for i in range(len(SEEDS)):
        values = []
        for point in POINTS:
            receipt = read_receipt(point, i, science)
            value = receipt["state"] if receipt else "pending"
            checkpoint = OUT / "checkpoints" / (paths(point, i)["receipt"].stem + ".npz")
            if receipt is None and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as data:
                    saved = json.loads(str(data["metadata"].item()))
                value = f"checkpoint {saved['progress']['completed_round']}"
            values.append(f"{point}: {value}")
        print(SEEDS[i], "; ".join(values))


def smoke_worker(index):
    from dataclasses import replace
    from stc_stage1 import smoke_federation
    output = []
    with tempfile.TemporaryDirectory() as folder:
        for point in POINTS:
            config = replace(config_for(point), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
            kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="event" if point == "event" else "stc",
                          config=config, seed=train_seed(index), provenance={"synthetic_seed_index": index})
            result = full.evaluated_run(**kwargs, checkpoint_path=Path(folder) / f"{point}_full.npz")
            resumed_path = Path(folder) / f"{point}_resume.npz"
            full.evaluated_run(**kwargs, checkpoint_path=resumed_path, stop_after=1)
            resumed = full.evaluated_run(**kwargs, checkpoint_path=resumed_path)
            np.testing.assert_array_equal(result["model"].view(np.uint32), resumed["model"].view(np.uint32))
            pd.testing.assert_frame_equal(result["evaluation"], resumed["evaluation"])
            output.append(result["evaluation"].iloc[-1].model_sha256)
    return output


def smoke():
    indices = (1, 2)
    expected = [smoke_worker(i) for i in indices]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = list(pool.map(smoke_worker, indices))
    if actual != expected:
        raise AssertionError("Spawned seed mapping differs")
    print("Two synthetic seeds, all three points: exact prediction/model resume and spawned workers")
    print("No CIFAR research training was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--point", choices=POINTS)
    parser.add_argument("--seed-index", type=int, choices=range(len(SEEDS)))
    args = parser.parse_args()
    if not 1 <= args.workers <= 27:
        parser.error("workers must be between 1 and 27")
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_event_codec_ten_seed.py", "-v"],
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
