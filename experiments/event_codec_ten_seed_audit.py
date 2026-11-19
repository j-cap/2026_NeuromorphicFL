"""Audit committed ten-seed evidence without rewriting the original analysis.

python experiments/event_codec_ten_seed_audit.py
git add experiments/results/event_codec_ten_seed_v1
git commit -m "Refresh ten-seed codec evidence audit"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import event_codec_ten_seed as ten
import numpy as np
import pandas as pd


def compare_saved(name, rebuilt, *, statistics=False):
    saved = pd.read_csv(ten.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    # SciPy's t-quantile implementation differs slightly across the recorded
    # Windows runtime and the analysis runtime. Endpoint differences are below
    # 2e-11 percentage points here; retain original tables and allow 1e-10.
    tolerance = 1e-10 if statistics else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance)


def validate():
    science = ten.load_contract()
    summaries, histories, rounds, references, completion = [], [], [], [], []
    for i, seed in enumerate(ten.SEEDS):
        for point in ten.POINTS:
            receipt = ten.read_receipt(point, i, science)
            expected = "reused" if i == 0 else "complete"
            if receipt is None or receipt["state"] != expected:
                raise RuntimeError(f"Missing required {expected} result: {point}/{seed}")
            result = ten.verified_tables(point, i, science)
            summaries.append(result["summary"])
            histories.append(result["history"])
            rounds.append(result["rounds"])
            references.append(ten.regression_check(point, i, result))
            completion.append({"point": point, "partition_seed": seed, "state": expected})
    summary, history = pd.concat(summaries, ignore_index=True), pd.concat(histories, ignore_index=True)
    raw_rounds = pd.concat(rounds, ignore_index=True)
    main = ten.add_regularization_columns(ten.optimized(summary), ten.config_for("event").regularization)
    pairs, intervals = ten.paired_endpoints(main)
    targets = ten.target_outcomes(history)
    target_intervals = ten.paired_target_summary(ten.optimized(targets))
    for name, rebuilt in (("comparison.csv", summary), ("comparison_history.csv", history),
                          ("main_runs.csv", main), ("endpoint_summary.csv", ten.descriptive_summary(main)),
                          ("paired_per_seed.csv", pairs), ("paired_summary.csv", intervals),
                          ("traffic_to_target.csv", targets), ("paired_target_summary.csv", target_intervals)):
        compare_saved(name, rebuilt, statistics=name in ("paired_summary.csv", "paired_target_summary.csv"))
    if len(summary) != 80 or len(history) != 2480 or len(raw_rounds) != 54000 or len(targets) != 240:
        raise RuntimeError("Unexpected complete-grid coverage")
    pilot = ten.full.pilot.load_contract()
    if science["runtime"] != pilot["runtime"] or science["data_sha256"] != pilot["data_sha256"]:
        raise RuntimeError("Recorded data/runtime differs from the pilot")
    return science, main, summary, history, raw_rounds, targets, pairs, intervals, target_intervals, references, completion


def target_pairs(targets, history):
    crossings = ten.optimized(targets).set_index(["partition_seed", "target_accuracy"])
    curves = ten.optimized(history).set_index(["point", "partition_seed", "round"])
    event = crossings[crossings.point == "event"]
    output = []
    for reference in ten.POINTS[1:]:
        baseline = crossings[crossings.point == reference]
        joined = event.join(baseline, lsuffix="_event", rsuffix="_reference", validate="one_to_one")
        for (seed, target), row in joined.iterrows():
            hit = bool(row.reached_event and row.reached_reference)
            record = {"reference": reference, "partition_seed": int(seed), "target_accuracy": target,
                      "event_reached": bool(row.reached_event), "reference_reached": bool(row.reached_reference),
                      "common_hit": hit}
            for suffix, point in (("event", "event"), ("reference", reference)):
                reached = bool(row[f"reached_{suffix}"])
                rnd = int(row[f"first_sampled_round_{suffix}"]) if reached else None
                actual = float(curves.loc[(point, seed, rnd), "test_accuracy"]) if reached else np.nan
                record.update({f"{suffix}_first_saved_round": rnd,
                               f"{suffix}_accuracy_at_crossing": actual,
                               f"{suffix}_unicast_bits": row[f"unicast_total_bits_{suffix}"]})
            ratio = row.unicast_total_bits_event / row.unicast_total_bits_reference if hit else np.nan
            record.update(traffic_ratio=ratio, reduction_percent=100 * (1 - ratio),
                          event_uses_fewer_bits=bool(ratio < 1) if hit else None)
            output.append(record)
    return pd.DataFrame(output)


def codec_diagnostics(raw_rounds, summary):
    diagnostic, windows = [], []
    for seed in ten.SEEDS:
        event = raw_rounds[(raw_rounds.point == "event") & (raw_rounds.partition_seed == seed)]
        if not event.active_clients.eq(10).all():
            raise RuntimeError("Unexpected participation")
        if not np.array_equal(event.legacy_uplink_bits, 16 * event.coordinate_events + 64 * event.active_clients):
            raise RuntimeError("Legacy pulse-length accounting differs")
        if not np.array_equal(event.legacy_server_bits, event.legacy_uplink_bits):
            raise RuntimeError("Legacy replay stream differs")
        ordered = 128 + 8 * np.ceil((20570 + 2 * event.coordinate_events) / 8).astype(np.int64)
        if not np.array_equal(event.ordered_downlink_bits, ordered):
            raise RuntimeError("Ordered frame lengths differ")
        if not np.array_equal(event.exact_downlink_bits,
                              np.minimum(event.aggregate_exact_downlink_bits, event.ordered_downlink_bits)):
            raise RuntimeError("Shortest exact downlink selection differs")
        endpoint = summary[(summary.point == "event") & (summary.partition_seed == seed)].set_index("variant")
        correction = 10 * (event.exact_downlink_bits - event.integer_only_downlink_bits).sum()
        diagnostic.append({"partition_seed": seed, "coordinate_events": int(event.coordinate_events.sum()),
                           "weighted_cancellation_fraction": 1 - event.aggregate_l1.sum() / event.coordinate_events.sum(),
                           "counts_patch_rounds": int(event.downlink_mode.eq("counts_patch").sum()),
                           "corrected_coordinate_rounds": int(event.correction_nonzeros.sum()),
                           "maximum_canonical_error": event.canonical_max_abs_error.max(),
                           "correction_percent_of_total_unicast": 100 * correction / endpoint.loc["both_codecs", "unicast_total_bits"],
                           "codec_unicast_reduction_percent": 100 * (1 - endpoint.loc["both_codecs", "unicast_total_bits"] / endpoint.loc["legacy", "unicast_total_bits"])})
        for first, last in ((1, 120), (121, 600), (601, 1200), (1201, 1800)):
            rows = event[event["round"].between(first, last)]
            windows.append({"partition_seed": seed, "first_round": first, "last_round": last,
                            "upload_reduction_percent": 100 * (1 - rows.codec_uplink_bits.sum() / rows.legacy_uplink_bits.sum()),
                            "server_stream_reduction_percent": 100 * (1 - rows.exact_downlink_bits.sum() / rows.legacy_server_bits.sum())})
    return pd.DataFrame(diagnostic), pd.DataFrame(windows)


def main():
    science, runs, summary, history, rounds, targets, pairs, intervals, target_ci, refs, completion = validate()
    per_target = target_pairs(targets, history)
    target_report = target_ci.copy()
    target_report["reduction_percent"] = 100 * (1 - target_report.estimate)
    target_report["reduction_ci95_low"] = 100 * (1 - target_report.ci95_high)
    target_report["reduction_ci95_high"] = 100 * (1 - target_report.ci95_low)
    for i, row in target_report.iterrows():
        group = per_target[(per_target.reference == row.reference) & (per_target.target_accuracy == row.target_accuracy)]
        if row.scope == "additional_9":
            group = group[group.partition_seed != ten.SEEDS[0]]
        target_report.loc[i, "event_cost_wins"] = int(group.event_uses_fewer_bits.eq(True).sum())
    codec, windows = codec_diagnostics(rounds, summary)
    endpoint = runs.copy()
    endpoint["test_accuracy_percent"] = 100 * endpoint.final_test_accuracy
    for name in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoint[name + "_Mbit"] = endpoint[name + "_bits"] / 1e6
    direction = intervals.copy()
    ratio = direction.metric.str.endswith("_ratio")
    direction.loc[ratio, "reduction_percent"] = 100 * (1 - direction.loc[ratio, "estimate"])
    direction.loc[ratio, "reduction_ci95_low"] = 100 * (1 - direction.loc[ratio, "ci95_high"])
    direction.loc[ratio, "reduction_ci95_high"] = 100 * (1 - direction.loc[ratio, "ci95_low"])
    for name, table in (("analysis_endpoint_comparison.csv", endpoint), ("analysis_effects.csv", direction),
                        ("analysis_target_pairs.csv", per_target), ("analysis_target_summary.csv", target_report),
                        ("analysis_codec_diagnostics.csv", codec), ("analysis_codec_windows.csv", windows)):
        ten.atomic_csv(table, ten.OUT / name)
    try:
        source_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(ten.OUT / "protocol.json")],
                                               cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = "unavailable"
    recorded = json.loads((ten.OUT / "analysis_audit.json").read_text(encoding="utf-8"))
    audit = {"results_commit": source_commit, "contract_sha256": ten.digest(science),
             "audit_source_sha256": ten.text_hash(Path(__file__)), "required_results": 30,
             "verified_reused_results": 3, "verified_new_results": 27,
             "round_records": len(rounds), "evaluated_models": 30 * 31,
             "all_eight_saved_csv_tables_reproduced": True, "statistics_comparison_tolerance": 1e-10,
             "recorded_data_and_runtime_match_pilot": True, "completion": completion,
             "independently_available_regression_references": refs,
             "user_local_recorded_regression_references": recorded["regression_references"],
             "all_methods_reach_all_prespecified_targets": bool(ten.optimized(targets).reached.all()),
             "endpoint_accuracy_win_counts": {point: int(((pairs.reference == point) &
                  (pairs.metric == "test_accuracy_difference_pp") & (pairs.value > 0)).sum()) for point in ten.POINTS[1:]},
             "event_counts_patch_rounds": int(codec.counts_patch_rounds.sum()),
             "maximum_canonical_error": float(codec.maximum_canonical_error.max()),
             "limitations": ["compact CNN and one CIFAR-10 federation regime; ten partition/SGD seeds on a fixed dataset",
                             "secondary intervals descriptive and unadjusted; no equivalence/noninferiority claim",
                             "first sampled crossings; no interpolation, deployment early stopping or checkpoint selection",
                             "raw images, full packet streams, NPZ models and per-example predictions not reloaded",
                             "20 native STC regression references reported locally but not committed; new STC receipts verify"]}
    ten.atomic_json(audit, ten.OUT / "analysis_independent_audit.json")
    print("Verified 30 trajectories, 54000 round records, 930 model evaluations and all eight saved CSV tables.")
    print(target_report[["scope", "reference", "target_accuracy", "reduction_percent", "reduction_ci95_low", "reduction_ci95_high", "event_cost_wins"]].to_string(index=False))
    print("Original analysis and frozen experiment sources were not overwritten.")


if __name__ == "__main__":
    main()
