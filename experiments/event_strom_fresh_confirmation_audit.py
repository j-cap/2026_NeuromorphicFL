"""Audit fresh Event/Ström evidence without training or rewriting original analysis.

python experiments/event_strom_fresh_confirmation_audit.py
git add experiments/results/event_strom_fresh_confirmation_v1
git commit -m "Refresh fresh Event and Strom confirmation evidence audit"
git push origin main
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import event_strom_fresh_confirmation as study
from strom_codec_development_audit import independent_counts
from strom_codec_ten_seed_audit import independent_interval
import numpy as np
import pandas as pd

ORIGINAL = ("comparison.csv", "comparison_history.csv", "main_runs.csv", "endpoint_summary.csv",
            "paired_per_seed.csv", "paired_summary.csv", "traffic_to_target.csv", "paired_target_summary.csv",
            "completion_audit.json", "analysis_audit.json", "confirmation.png", "confirmation.pdf")


def evidence_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix in (".png", ".pdf") else study.text_hash(path)


def compare_saved(name, rebuilt):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    tolerance = 1e-10 if name in ("paired_summary.csv", "paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance)


def check_statistics(runs, targets, paired, target_intervals):
    definitions = {name: (field, ratio, scale) for name, field, ratio, scale in study.METRICS}
    event, strom = (study.indexed(runs, point) for point in study.POINTS)
    for _, row in paired.iterrows():
        field, ratio, scale = definitions[row.metric]
        values = event[field] / strom[field] if ratio else scale * (event[field] - strom[field])
        if int(row.n) != 10 or row.candidate != "event" or row.reference != "strom":
            raise RuntimeError("Paired endpoint scope or denominator differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(values, ratio),
                                   rtol=1e-10, atol=1e-10)
    for _, row in target_intervals.iterrows():
        selected = targets[targets.target_accuracy == row.target_accuracy]
        a, b = (study.indexed(selected, point) for point in study.POINTS)
        common = a.reached & b.reached
        values = a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"]
        if (int(row.n_total), int(row.n_event_reached), int(row.n_strom_reached), int(row.n_common_hits), int(row.n)) != (
                10, int(a.reached.sum()), int(b.reached.sum()), int(common.sum()), len(values)):
            raise RuntimeError("Target coverage differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(values, True),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)


def validate():
    science = study.load_contract()
    tables, completion = {}, []
    for index, seed in enumerate(study.SEEDS):
        for point in study.POINTS:
            receipt = study.read_receipt(point, index, science)
            if receipt is None or receipt["state"] != "complete":
                raise RuntimeError(f"Missing fresh completed result: {point}/{seed}")
            tables[point, seed] = study.verified_tables(point, index, science)
            independent_counts(tables[point, seed], clients=10, dimension=20570)
            completion.append({"point": point, "partition_seed": seed, "state": "complete", "error": ""})
    if json.loads((study.OUT / "completion_audit.json").read_text()) != {"expected": 20, "results": completion}:
        raise RuntimeError("Saved completion coverage differs")
    summary = pd.concat([table["summary"] for table in tables.values()], ignore_index=True)
    history = pd.concat([table["history"] for table in tables.values()], ignore_index=True)
    rounds = pd.concat([table["rounds"] for table in tables.values()], ignore_index=True)
    runs = study.add_regularization_columns(study.optimized(summary), study.config_for("event").regularization)
    pairs, intervals = study.paired_endpoints(runs)
    targets = study.target_outcomes(study.optimized(history))
    target_intervals = study.paired_targets(targets)
    rebuilt = {"comparison.csv": summary, "comparison_history.csv": history, "main_runs.csv": runs,
               "endpoint_summary.csv": study.descriptive_summary(runs), "paired_per_seed.csv": pairs,
               "paired_summary.csv": intervals, "traffic_to_target.csv": targets, "paired_target_summary.csv": target_intervals}
    for name, frame in rebuilt.items():
        compare_saved(name, frame)
    check_statistics(runs, targets, intervals, target_intervals)
    counts = (len(tables), len(rounds), sum(len(t["evaluation"]) for t in tables.values()),
              len(summary), len(history), len(runs), len(pairs), len(targets))
    if counts != (20, 36000, 620, 80, 2480, 20, 80, 60):
        raise RuntimeError(f"Complete-grid coverage differs: {counts}")
    expected_audit = {"fresh_seed_pairs": 10, "new_training_runs": 20, "reused_training_results": 0,
                      "verified_round_records": 36000, "verified_evaluated_model_fingerprints": 620,
                      "frozen_selection_sha256": study.digest(science["selection_provenance"]),
                      "source_receipts_model_fingerprints_and_prefix_costs_verified": True,
                      "seed_freshness": science["seed_freshness"], "intervals": science["protocol"]["intervals"],
                      "equivalence_or_noninferiority_claim": False, "tuning_or_early_stopping": False,
                      "target_coverage_required_for_interpretation": True}
    if json.loads((study.OUT / "analysis_audit.json").read_text()) != expected_audit:
        raise RuntimeError("Saved analysis verification scope differs")
    return science, runs, summary, rounds, targets, intervals, target_intervals


def savings(row):
    return {"Event_saving_percent": 100 * (1 - row.estimate),
            "Event_saving_ci95_low": 100 * (1 - row.ci95_high),
            "Event_saving_ci95_high": 100 * (1 - row.ci95_low)}


def target_pairs(targets):
    rows = []
    for target in study.TARGETS:
        selected = targets[targets.target_accuracy == target]
        a, b = (study.indexed(selected, point) for point in study.POINTS)
        for seed in study.SEEDS:
            hit = bool(a.loc[seed, "reached"] and b.loc[seed, "reached"])
            row = {"partition_seed": seed, "target_accuracy": target, "Event_reached": bool(a.loc[seed, "reached"]),
                   "Strom_reached": bool(b.loc[seed, "reached"]), "common_hit": hit}
            for label, frame in (("Event", a), ("Strom", b)):
                for field in ("first_sampled_round", "accuracy_at_crossing", "unicast_total_bits", "upload_bits", "broadcast_total_bits"):
                    row[label + "_" + field] = frame.loc[seed, field]
            row["Event_over_Strom_unicast_ratio"] = a.loc[seed, "unicast_total_bits"] / b.loc[seed, "unicast_total_bits"] if hit else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def codec_diagnostics(rounds, summary):
    rows = []
    for (point, seed), frame in rounds.groupby(["point", "partition_seed"], sort=False):
        endpoint = summary[(summary.point == point) & (summary.partition_seed == seed)].set_index("variant")
        total = int(endpoint.loc["both_codecs", "unicast_total_bits"])
        correction = int(10 * (frame.exact_downlink_bits - frame.integer_only_downlink_bits).sum())
        rows.append({"point": point, "partition_seed": seed,
                     "coordinate_events": int(frame.coordinate_events.sum()),
                     "late_120_firing_fraction": frame.tail(120).coordinate_events.mean() / (10 * 20570),
                     "counts_patch_rounds": int(frame.downlink_mode.eq("counts_patch").sum()),
                     "ordered_fallback_rounds": int(frame.downlink_mode.eq("ordered").sum()),
                     "correction_bits": correction, "correction_percent_of_total": 100 * correction / total,
                     "maximum_canonical_error": frame.canonical_max_abs_error.max(),
                     "legacy_unicast_bits": int(endpoint.loc["legacy", "unicast_total_bits"]),
                     "improved_unicast_bits": total,
                     "codec_unicast_reduction_percent": 100 * (1 - total / endpoint.loc["legacy", "unicast_total_bits"])})
    return pd.DataFrame(rows)


def main():
    originals = {name: evidence_hash(study.OUT / name) for name in ORIGINAL}
    science, runs, summary, rounds, targets, intervals, target_intervals = validate()
    endpoints = runs.copy()
    endpoints["test_accuracy_percent"] = 100 * endpoints.final_test_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoints[field + "_Mbit"] = endpoints[field + "_bits"] / 1e6
    effects = intervals.copy()
    event, strom = (study.indexed(runs, point) for point in study.POINTS)
    definitions = {name: field for name, field, _, _ in study.METRICS}
    for index, row in effects.iterrows():
        if row.metric.endswith("_ratio"):
            for field, value in savings(row).items():
                effects.loc[index, field] = value
        field = definitions[row.metric]
        for label, mask in (("lower", event[field] < strom[field]), ("equal", event[field] == strom[field]), ("higher", event[field] > strom[field])):
            effects.loc[index, "Event_" + label + "_count"] = int(mask.sum())
    target_effects = target_intervals.copy()
    for index, row in target_effects.iterrows():
        for field, value in savings(row).items():
            target_effects.loc[index, field] = value
    outputs = {"analysis_endpoint_comparison.csv": endpoints, "analysis_endpoint_effects.csv": effects,
               "analysis_target_pairs.csv": target_pairs(targets), "analysis_target_effects.csv": target_effects,
               "analysis_codec_diagnostics.csv": codec_diagnostics(rounds, summary)}
    for name, table in outputs.items():
        study.atomic_csv(table, study.OUT / name)
    if originals != {name: evidence_hash(study.OUT / name) for name in ORIGINAL}:
        raise RuntimeError("Original Windows analysis was unexpectedly modified")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")],
                                           cwd=REPO, text=True).strip()
    study.atomic_json({"results_commit": result_commit, "contract_sha256": study.digest(science),
                       "audit_source_sha256": study.text_hash(Path(__file__)),
                       "independent_statistics_helper_sha256": study.text_hash(REPO / "experiments/strom_codec_ten_seed_audit.py"),
                       "fresh_complete_receipts_verified": 20, "verified_round_records": 36000,
                       "verified_evaluated_model_fingerprints": 620, "reproduced_original_analysis_csvs": 8,
                       "original_analysis_sha256": originals, "original_analysis_preserved": True,
                       "original_hash_format": "sha256_UTF8_LF_for_text_raw_bytes_for_PNG_PDF",
                       "independent_prefix_and_endpoint_accounting": True, "independent_paired_intervals_and_coverage": True,
                       "all_ten_fresh_pairs_retained": True, "development_checkpoint_reused": False,
                       "test_tuning_or_early_stopping": False, "equivalence_or_noninferiority_claim": False,
                       "limits": ["partition/SGD variation on one fixed previously inspected test set",
                                  "compact CNN, full participation and strong-skew regime only",
                                  "secondary intervals are descriptive and unadjusted",
                                  "raw images, packets and NPZ weights are not reloaded",
                                  "no causal leak/reset/quantum attribution or physical-network timing claim"]},
                      study.OUT / "analysis_independent_audit.json")
    print("Verified 20 fresh runs, 36000 rounds, 620 sampled model fingerprints and eight original analysis tables.")
    print(target_effects[["target_accuracy", "Event_saving_percent", "Event_saving_ci95_low", "Event_saving_ci95_high",
                          "event_fewer_rounds", "event_fewer_bits"]].to_string(index=False))


if __name__ == "__main__":
    main()
