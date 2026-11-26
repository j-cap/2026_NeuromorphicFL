"""Audit the completed selected ResNet core without training or rewriting it.

python experiments/resnet14_core_confirmation_audit.py
git add experiments/results/resnet14_core_confirmation_v1
git commit -m "Refresh selected ResNet14 core confirmation audit"
git push origin main
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_core_confirmation as study
from strom_codec_ten_seed_audit import independent_interval
import numpy as np
import pandas as pd

parent = study.parent
ORIGINAL = {"protocol.json", "confirmation_completion.json", "analysis_audit.json", "confirmation_runs.csv",
            "confirmation_history.csv", "endpoint_summary.csv", "paired_per_seed.csv", "paired_summary.csv",
            "traffic_to_target.csv", "paired_target_summary.csv", "confirmation.png", "confirmation.pdf"}
SUPPLEMENTARY = {"analysis_endpoints.csv", "analysis_endpoint_effects.csv", "analysis_target_effects.csv",
                 "analysis_target_descriptive.csv", "analysis_codec_diagnostics.csv", "analysis_compute_timings.csv",
                 "analysis_compute_summary.csv", "analysis_independent_audit.json"}
DEFINITIONS = (
    ("accuracy_difference_pp", "final_test_accuracy", False, 100),
    ("test_ce_difference", "final_test_ce", False, 1),
    ("unicast_ratio", "unicast_total_bits", True, 1),
    ("upload_ratio", "uplink_bits", True, 1),
    ("download_ratio", "unicast_downlink_bits", True, 1),
    ("broadcast_ratio", "broadcast_total_bits", True, 1),
)


def raw_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(frame, method):
    result = frame[frame.comparison_id == method].set_index("partition_seed").sort_index()
    if result.index.has_duplicates or set(result.index) != set(study.SEEDS):
        raise RuntimeError("Missing or duplicated seed in " + method)
    return result


def compare_saved(name, frame):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    tolerance = 1e-10 if name in ("paired_summary.csv", "paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance)


def interval(values, ratio):
    estimate, low, high = independent_interval(values, ratio)
    return {"n": len(values), "estimate": float(estimate), "ci95_low": float(low), "ci95_high": float(high)}


def add_savings(row):
    return {"Event_saving_percent": 100 * (1 - row["estimate"]),
            "Event_saving_ci95_low": 100 * (1 - row["ci95_high"]),
            "Event_saving_ci95_high": 100 * (1 - row["ci95_low"])}


def main():
    originals = {p.name: raw_hash(p) for p in study.OUT.iterdir()
                 if p.is_file() and p.suffix in (".json", ".csv", ".png", ".pdf") and p.name not in SUPPLEMENTARY}
    if not ORIGINAL.issubset(originals):
        raise RuntimeError("Expected all original completed-core tables and figures")
    recorded = json.loads((study.OUT / "protocol.json").read_text())
    source_files = {entry["path"] for ref in recorded["references"].values() for entry in ref.values()}
    source_bytes = {p: raw_hash(REPO / p) for p in sorted(source_files)}
    science = study.load_contract()
    tables, completion, summaries, histories = {}, [], [], []
    for seed in study.SEEDS:
        for method in study.METHODS:
            receipt = study.read_outcome(method, seed, science)
            expected = "reused" if method in study.REUSED else "complete"
            if receipt is None or receipt["state"] != expected:
                raise RuntimeError("All thirty new and twenty reused outcomes are required")
            table = study.verified_tables(method, seed, science)
            if method in study.NEW:
                study.baseline_traffic(table, study.point_for(method), clients=10, dimension=175258)
            else:
                study.pulse_traffic(table, clients=10, dimension=175258)
                # The source contract/identities and all original columns remain exact.
                prior = json.loads((study.pulse.OUT / "protocol.json").read_text())
                original = study.pulse.verified_tables(method, seed, prior)
                for kind, frame in original.items():
                    pd.testing.assert_frame_equal(frame, table[kind][frame.columns], check_exact=True)
            tables[method, seed] = table
            completion.append({"method": method, "partition_seed": seed, "state": expected,
                               "origin": "verified_reuse" if method in study.REUSED else "new_run"})
            summaries.append(table["summary"][table["summary"].variant == study.MAIN[method]])
            histories.append(table["history"][table["history"].variant == study.MAIN[method]])
    if json.loads((study.OUT / "confirmation_completion.json").read_text()) != {
            "expected": 50, "expected_new": 30, "expected_reused": 20, "results": completion}:
        raise RuntimeError("Original completion ledger differs")
    rounds = sum(len(t["rounds"]) for t in tables.values())
    models = sum(len(t["evaluation"]) for t in tables.values())
    if (len(tables), rounds, models) != (50, 90000, 1550):
        raise RuntimeError("Expected complete ten-seed five-method coverage")
    runs = parent.add_regularization_columns(pd.concat(summaries, ignore_index=True), study.config_for("event").regularization)
    history = pd.concat(histories, ignore_index=True)
    if len(runs) != 50 or len(history) != 1550 or runs.model_sha256.nunique() != 50:
        raise RuntimeError("Primary variant/model identity coverage differs")
    compare_saved("confirmation_runs.csv", runs)
    compare_saved("confirmation_history.csv", history)
    fields = ["final_test_accuracy", "final_test_ce", "final_test_worst_class_accuracy", "final_train_objective", "final_train_ce",
              "final_train_l2_penalty", "final_parameter_l2_norm", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"]
    descriptive = runs.groupby("comparison_id", sort=False)[fields].agg(["mean", "std", "count"])
    descriptive.columns = ["_".join(c) for c in descriptive.columns]
    compare_saved("endpoint_summary.csv", descriptive.reset_index())

    # First crossings come from raw predictive samples, with exact prefix costs.
    crossings = []
    for (method, seed), table in sorted(tables.items()):
        curve = table["history"][table["history"].variant == study.MAIN[method]].set_index("round")
        for target in parent.TARGETS:
            hits = table["evaluation"][table["evaluation"].test_accuracy >= target]
            row = {"comparison_id": method, "partition_seed": seed, "target_accuracy": target, "reached": not hits.empty,
                   "first_sampled_round": np.nan, "censor_round": np.nan if not hits.empty else study.ROUNDS,
                   "upload_bits": np.nan, "unicast_total_bits": np.nan, "broadcast_total_bits": np.nan}
            if not hits.empty:
                first = int(hits.iloc[0]["round"])
                cost = curve.loc[first]
                row.update(first_sampled_round=first, upload_bits=int(cost.uplink_bits),
                           unicast_total_bits=int(cost.unicast_total_bits), broadcast_total_bits=int(cost.broadcast_total_bits))
            crossings.append(row)
    targets = pd.DataFrame(crossings)
    compare_saved("traffic_to_target.csv", targets)
    event = indexed(runs, "event")
    pairs, effects, target_effects = [], [], []
    for method in study.METHODS[1:]:
        ref = indexed(runs, method)
        for metric, field, ratio, scale in DEFINITIONS:
            values = event[field] / ref[field] if ratio else scale * (event[field] - ref[field])
            pairs.extend({"reference": method, "partition_seed": int(s), "metric": metric, "value": float(v)} for s, v in values.items())
            row = {"reference": method, "metric": metric, **interval(values, ratio),
                   "direction": "Event_over_reference" if ratio else "Event_minus_reference"}
            if ratio:
                row.update(add_savings(row))
            effects.append(row)
        for target in parent.TARGETS:
            frame = targets[targets.target_accuracy == target]
            a, b = indexed(frame, "event"), indexed(frame, method)
            common = a.reached & b.reached
            estimate = interval(a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"], True)
            row = {"reference": method, "target_accuracy": target, "n_total": len(study.SEEDS),
                   "n_event_reached": int(a.reached.sum()), "n_reference_reached": int(b.reached.sum()),
                   "n_common_hits": int(common.sum()), **estimate}
            row.update(add_savings(row)); target_effects.append(row)
    effects, target_effects = pd.DataFrame(effects), pd.DataFrame(target_effects)
    compare_saved("paired_per_seed.csv", pd.DataFrame(pairs))
    compare_saved("paired_summary.csv", effects)
    compare_saved("paired_target_summary.csv", target_effects)
    old = pd.read_csv(study.pulse.OUT / "paired_summary.csv", float_precision="round_trip").query('scope == "all_10"')
    f = ["metric", "n", "estimate", "ci95_low", "ci95_high"]
    pd.testing.assert_frame_equal(effects[effects.reference == "strom"][f].reset_index(drop=True), old[f].reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    old = pd.read_csv(study.pulse.OUT / "paired_target_summary.csv", float_precision="round_trip").query('scope == "all_10"')
    f = ["target_accuracy", "n_total", "n_event_reached", "n_common_hits", "n", "estimate", "ci95_low", "ci95_high"]
    pd.testing.assert_frame_equal(target_effects[target_effects.reference == "strom"][f].reset_index(drop=True), old[f].reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)
    expected_audit = {"new_runs": 30, "reused_runs": 20, "seed_pairs": 10, "rounds_verified": rounds,
                      "sampled_model_fingerprints": models, "all_outcomes_retained": True,
                      "selection_sha256": parent.digest(science["selection"]), "test_based_tuning_or_stopping": False,
                      "development_checkpoint_reuse": False, "extra_sparse_STC_runs": 0, "unreached_targets_censored": True,
                      "scope": science["protocol"]["scope"]}
    if expected_audit != json.loads((study.OUT / "analysis_audit.json").read_text()):
        raise RuntimeError("Original audit scope differs")

    endpoint = runs.copy()
    endpoint["test_accuracy_percent"] = 100 * endpoint.final_test_accuracy
    endpoint["worst_class_accuracy_percent"] = 100 * endpoint.final_test_worst_class_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoint[field + "_Gbit"] = endpoint[field + "_bits"] / 1e9
    for i, row in effects.iterrows():
        field = next(field for metric, field, _, _ in DEFINITIONS if metric == row.metric)
        a, b = event[field], indexed(runs, row.reference)[field]
        effects.loc[i, "Event_lower_count"] = int((a < b).sum())
        effects.loc[i, "Event_equal_count"] = int((a == b).sum())
        effects.loc[i, "Event_higher_count"] = int((a > b).sum())
    for i, row in target_effects.iterrows():
        frame = targets[targets.target_accuracy == row.target_accuracy]
        a, b = indexed(frame, "event"), indexed(frame, row.reference)
        common = a.reached & b.reached
        for label, field, operator in (("Event_fewer_bits", "unicast_total_bits", np.less),
                                       ("Event_fewer_rounds", "first_sampled_round", np.less),
                                       ("same_sampled_round", "first_sampled_round", np.equal),
                                       ("Event_later_rounds", "first_sampled_round", np.greater)):
            target_effects.loc[i, label] = int(operator(a.loc[common, field], b.loc[common, field]).sum())
    target_descriptive = targets.groupby(["comparison_id", "target_accuracy"]).agg(
        reached=("reached", "sum"), total=("reached", "size"), mean_first_sampled_round=("first_sampled_round", "mean"),
        mean_unicast_bits=("unicast_total_bits", "mean")).reset_index()
    diagnostics, timing_rows, pooled_timing = [], [], []
    for (method, seed), table in tables.items():
        r = table["rounds"]
        main = table["summary"][table["summary"].variant == study.MAIN[method]].iloc[0]
        uplink_modes = Counter()
        for value in r.uplink_modes:
            uplink_modes.update(json.loads(value))
        diagnostics.append({"comparison_id": method, "partition_seed": seed, "coordinate_events": int(r.coordinate_events.sum()),
                            "late_120_nonzero_client_fraction": r.tail(120).coordinate_events.mean() / (10 * 175258),
                            "unicast_downlink_percent_of_total": 100 * main.unicast_downlink_bits / main.unicast_total_bits,
                            "uplink_modes": json.dumps(dict(sorted(uplink_modes.items())), sort_keys=True),
                            "downlink_modes": json.dumps(r.downlink_mode.value_counts().to_dict(), sort_keys=True)})
        if method in study.NEW:
            pooled_timing.append(r[["training_seconds", "codec_seconds"]].assign(comparison_id=method, campaign="confirmation_fast"))
            timing_rows.append({"comparison_id": method, "partition_seed": seed, "campaign": "confirmation_fast", "rounds": len(r),
                                "training_median_seconds": r.training_seconds.median(), "codec_median_seconds": r.codec_seconds.median(),
                                "training_sum_hours": r.training_seconds.sum() / 3600, "codec_sum_hours": r.codec_seconds.sum() / 3600})
    for method in study.NEW:
        p = study.point_for(method)
        old = pd.read_csv(study.development.paths(p, 1800)["rounds"], usecols=["training_seconds", "codec_seconds"])
        pooled_timing.append(old.assign(comparison_id=method, campaign="prior_validation_original_codec"))
        timing_rows.append({"comparison_id": method, "partition_seed": 7000, "campaign": "prior_validation_original_codec", "rounds": len(old),
                            "training_median_seconds": old.training_seconds.median(), "codec_median_seconds": old.codec_seconds.median(),
                            "training_sum_hours": old.training_seconds.sum() / 3600, "codec_sum_hours": old.codec_seconds.sum() / 3600})
    compute = pd.concat(pooled_timing, ignore_index=True).groupby(["comparison_id", "campaign"]).agg(
        recorded_rounds=("codec_seconds", "size"), training_median_seconds=("training_seconds", "median"),
        codec_median_seconds=("codec_seconds", "median"), training_sum_seconds=("training_seconds", "sum"),
        codec_sum_seconds=("codec_seconds", "sum")).reset_index()
    compute["recorded_component_sum_hours"] = (compute.training_sum_seconds + compute.codec_sum_seconds) / 3600
    for name, frame in (("analysis_endpoints.csv", endpoint), ("analysis_endpoint_effects.csv", effects),
                        ("analysis_target_effects.csv", target_effects), ("analysis_target_descriptive.csv", target_descriptive),
                        ("analysis_codec_diagnostics.csv", pd.DataFrame(diagnostics)), ("analysis_compute_timings.csv", pd.DataFrame(timing_rows)),
                        ("analysis_compute_summary.csv", compute)):
        parent.atomic_csv(frame, study.OUT / name)
    if (originals != {p: raw_hash(study.OUT / p) for p in originals}
            or source_bytes != {p: raw_hash(REPO / p) for p in source_bytes}):
        raise RuntimeError("Original campaign or reused pulse evidence changed")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")], cwd=REPO, text=True).strip()
    parent.atomic_json({
        "results_commit": result_commit, "contract_sha256": parent.digest(science), "selection_sha256": parent.digest(science["selection"]),
        "audit_source_sha256": parent.text_hash(Path(__file__)),
        "helper_sha256": {p: parent.text_hash(REPO / p) for p in ("experiments/strom_codec_ten_seed_audit.py",
                          "experiments/resnet14_transfer_audit.py", "experiments/resnet14_baseline_development_audit.py")},
        "new_complete_receipts": 30, "verified_reuse_receipts": 20, "seed_pairs": 10,
        "verified_round_records": rounds, "verified_evaluated_model_fingerprints": models, "distinct_final_models": 50,
        "reproduced_original_analysis_csvs": 7, "original_campaign_file_count": len(originals),
        "original_campaign_sha256_bytes": originals, "read_only_pulse_source_file_count": len(source_bytes),
        "read_only_pulse_source_sha256_bytes": source_bytes, "original_campaign_and_pulse_sources_preserved": True,
        "independent_frame_and_traffic_prefix_checks": True, "independent_paired_intervals": True,
        "raw_evaluation_first_crossings_and_full_coverage_verified": True, "previous_Event_Strom_contrast_exactly_reproduced": True,
        "new_training_runs": 0, "new_tuning": False, "extra_sparse_STC_confirmation": False,
        "limits": ["selected five operating points, not a complete STC/EF frontier or global optimum",
                   "one fixed previously inspected CIFAR10 test set and shared initialization",
                   "descriptive unadjusted paired intervals; no equivalence/noninferiority or universal faster-convergence claim",
                   "compute sections exclude evaluation/checkpoint/setup; timing contrast uses different development/confirmation campaigns",
                   "saved identities, hashes, metrics and counters only; no image/packet/model arrays or full training replay"],
    }, study.OUT / "analysis_independent_audit.json")
    print("Verified 30 new fits + 20 exact pulse reuses, 90000 rounds, 1550 sampled models and seven original CSVs.")
    print(descriptive[["final_test_accuracy_mean", "final_test_accuracy_std", "unicast_total_bits_mean"]].to_string())
    print(target_effects[["reference", "target_accuracy", "Event_fewer_bits", "Event_fewer_rounds", "same_sampled_round", "Event_later_rounds"]].to_string(index=False))


if __name__ == "__main__":
    main()
