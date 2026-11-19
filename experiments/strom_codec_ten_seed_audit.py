"""Audit calibrated Ström confirmation without rewriting its original analysis.

python experiments/strom_codec_ten_seed_audit.py
git add experiments/results/strom_codec_ten_seed_v1
git commit -m "Refresh calibrated Strom ten-seed evidence audit"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import strom_codec_ten_seed as ten
import numpy as np
import pandas as pd
from scipy.stats import t

ANALYSIS_NAMES = ("comparison.csv", "comparison_history.csv", "main_runs.csv", "endpoint_summary.csv",
                  "paired_per_seed.csv", "paired_summary.csv", "traffic_to_target.csv", "paired_target_summary.csv")


def compare_saved(name, rebuilt):
    saved = pd.read_csv(ten.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    # Preserve the recorded Windows analysis. SciPy quantiles can differ at
    # tiny floating-point scale between the recorded and analysis environments.
    tolerance = 1e-10 if name in ("paired_summary.csv", "paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance)


def check_traffic(tables, *, n_clients, dimension, pulse):
    rounds = tables["rounds"]
    checkpoint = 32 * dimension + 64
    if pulse:
        variants = (("legacy", "legacy_uplink_bits", "legacy_server_bits"),
                    ("uplink_codec_replay", "codec_uplink_bits", "codec_replay_bits"),
                    ("aggregate_downlink", "legacy_uplink_bits", "exact_downlink_bits"),
                    ("both_codecs", "codec_uplink_bits", "exact_downlink_bits"))
        if (not np.array_equal(rounds.legacy_uplink_bits, 16 * rounds.coordinate_events + 64 * rounds.active_clients)
                or not np.array_equal(rounds.legacy_server_bits, rounds.legacy_uplink_bits)
                or not np.array_equal(rounds.ordered_downlink_bits, 128 + 8 * ((dimension + 2 * rounds.coordinate_events + 7) // 8))
                or not np.array_equal(rounds.exact_downlink_bits, np.minimum(rounds.aggregate_exact_downlink_bits, rounds.ordered_downlink_bits))):
            raise RuntimeError("Pulse frame lengths or exact-downlink selection differ")
        # This field counts nonempty uploads, not clients that performed local SGD.
        if not rounds.active_clients.between(0, n_clients).all():
            raise RuntimeError("Invalid nonempty-upload count")
    else:
        variants = (("native_stc", "legacy_uplink_bits", "legacy_server_bits"),
                    ("lossless_stc", "codec_uplink_bits", "exact_downlink_bits"))
    for variant, up, down in variants:
        values = rounds[[up, down]].to_numpy()
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise RuntimeError("Invalid frame bit counters")
        upload = rounds[up].cumsum().to_numpy()
        stream = rounds[down].to_numpy()
        if pulse and variant in ("legacy", "uplink_codec_replay"):
            stream = np.minimum(stream, checkpoint)
        stream = stream.cumsum()
        unicast_down = n_clients * (checkpoint + 32 * rounds["round"].to_numpy() + stream)
        totals = {"uplink_bits": upload, "server_stream_bits": stream,
                  "unicast_downlink_bits": unicast_down, "unicast_total_bits": upload + unicast_down,
                  "broadcast_downlink_bits": checkpoint + stream, "broadcast_total_bits": upload + checkpoint + stream}
        history = tables["history"][tables["history"].variant == variant]
        indices = history["round"].to_numpy(dtype=int) - 1
        summary = tables["summary"][tables["summary"].variant == variant].iloc[0]
        for field, array in totals.items():
            if not np.array_equal(history[field], array[indices]) or int(summary[field]) != int(array[-1]):
                raise RuntimeError(f"Independent {variant} prefix/endpoint accounting differs: {field}")


def independent_interval(values, ratio):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.array([np.nan] * 3)
    work = np.log(values) if ratio else values
    mean = work.mean()
    half = t.ppf(.975, len(work) - 1) * np.std(work, ddof=1) / np.sqrt(len(work)) if len(work) > 1 else np.nan
    result = np.array([mean, mean - half, mean + half])
    return np.exp(result) if ratio else result


def check_statistics(runs, targets, paired, target_intervals):
    definitions = {name: (field, ratio, scale) for name, field, ratio, scale in ten.previous.METRICS}
    for _, row in paired.iterrows():
        selected = runs if row.scope == "all_10" else runs[runs.partition_seed != ten.SEEDS[0]]
        a = selected[selected.point == row.candidate].set_index("partition_seed").sort_index()
        b = selected[selected.point == row.reference].set_index("partition_seed").sort_index()
        field, ratio, scale = definitions[row.metric]
        values = a[field] / b[field] if ratio else scale * (a[field] - b[field])
        if int(row.n) != len(values):
            raise RuntimeError("Paired endpoint denominator differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(values, ratio), rtol=1e-10, atol=1e-10)
    for _, row in target_intervals.iterrows():
        selected = targets[targets.target_accuracy == row.target_accuracy]
        if row.scope == "additional_9":
            selected = selected[selected.partition_seed != ten.SEEDS[0]]
        a = selected[selected.point == row.candidate].set_index("partition_seed").sort_index()
        b = selected[selected.point == row.reference].set_index("partition_seed").sort_index()
        common = a.reached & b.reached
        ratios = a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"]
        if (int(row.n_total), int(row.n_candidate_reached), int(row.n_reference_reached), int(row.n_common_hits), int(row.n)) != (
                len(a), int(a.reached.sum()), int(b.reached.sum()), int(common.sum()), len(ratios)):
            raise RuntimeError("Paired target coverage differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(ratios, True), rtol=1e-10, atol=1e-10, equal_nan=True)


def validate():
    science = ten.load_contract()
    tables, completion = {}, []
    for index, seed in enumerate(ten.SEEDS):
        for point in ten.POINTS:
            receipt = ten.read_receipt(point, index, science)
            expected = "complete" if point == "strom" else "reused"
            if receipt is None or receipt["state"] != expected:
                raise RuntimeError(f"Missing {expected} result: {point}/{seed}")
            tables[point, seed] = ten.verified_tables(point, index, science)
            check_traffic(tables[point, seed], n_clients=10, dimension=20570, pulse=point in ("event", "strom"))
            completion.append({"point": point, "partition_seed": seed, "state": expected, "error": ""})
    saved_completion = json.loads((ten.OUT / "completion_audit.json").read_text())
    if saved_completion != {"expected": 40, "results": completion}:
        raise RuntimeError("Saved completion coverage differs")
    summary = pd.concat([table["summary"] for table in tables.values()], ignore_index=True)
    history = pd.concat([table["history"] for table in tables.values()], ignore_index=True)
    rounds = pd.concat([table["rounds"] for table in tables.values()], ignore_index=True)
    runs = ten.add_regularization_columns(ten.optimized(summary), ten.config_for().regularization)
    pairs, intervals = ten.paired_endpoints(runs)
    targets = ten.target_outcomes(ten.optimized(history))
    target_intervals = ten.paired_target_summary(targets)
    rebuilt = {"comparison.csv": summary, "comparison_history.csv": history, "main_runs.csv": runs,
               "endpoint_summary.csv": ten.previous.descriptive_summary(runs), "paired_per_seed.csv": pairs,
               "paired_summary.csv": intervals, "traffic_to_target.csv": targets, "paired_target_summary.csv": target_intervals}
    for name, table in rebuilt.items():
        compare_saved(name, table)
    check_statistics(runs, targets, intervals, target_intervals)
    counts = (len(tables), len(rounds), sum(len(t["evaluation"]) for t in tables.values()), len(summary), len(history), len(runs), len(pairs), len(targets))
    if counts != (40, 72000, 1240, 120, 3720, 40, 400, 120):
        raise RuntimeError(f"Complete-grid coverage differs: {counts}")
    return science, runs, summary, rounds, targets, intervals, target_intervals


def reductions(row):
    # Both directions are explicit: E/S > 1 translates to a saving for S of
    # 1 - 1/(E/S), rather than the larger E-over-S percentage increase.
    return {"candidate_reduction_percent": 100 * (1 - row.estimate),
            "candidate_reduction_ci95_low": 100 * (1 - row.ci95_high),
            "candidate_reduction_ci95_high": 100 * (1 - row.ci95_low),
            "reference_reduction_percent": 100 * (1 - 1 / row.estimate),
            "reference_reduction_ci95_low": 100 * (1 - 1 / row.ci95_low),
            "reference_reduction_ci95_high": 100 * (1 - 1 / row.ci95_high)}


def target_pairs(targets):
    rows = []
    for candidate, reference in ten.COMPARISONS:
        for target in ten.TARGETS:
            selected = targets[targets.target_accuracy == target]
            a = selected[selected.point == candidate].set_index("partition_seed")
            b = selected[selected.point == reference].set_index("partition_seed")
            for seed in ten.SEEDS:
                hit = bool(a.loc[seed, "reached"] and b.loc[seed, "reached"])
                record = {"candidate": candidate, "reference": reference, "partition_seed": seed,
                          "target_accuracy": target, "candidate_reached": bool(a.loc[seed, "reached"]),
                          "reference_reached": bool(b.loc[seed, "reached"]), "common_hit": hit}
                for label, frame in (("candidate", a), ("reference", b)):
                    for field in ("first_sampled_round", "accuracy_at_crossing", "unicast_total_bits"):
                        record[label + "_" + field] = frame.loc[seed, field]
                ratio = a.loc[seed, "unicast_total_bits"] / b.loc[seed, "unicast_total_bits"] if hit else np.nan
                record.update(traffic_ratio=ratio, candidate_uses_fewer_bits=bool(ratio < 1) if hit else None,
                              candidate_uses_fewer_rounds=bool(a.loc[seed, "first_sampled_round"] < b.loc[seed, "first_sampled_round"]) if hit else None,
                              same_sampled_round=bool(a.loc[seed, "first_sampled_round"] == b.loc[seed, "first_sampled_round"]) if hit else None)
                rows.append(record)
    return pd.DataFrame(rows)


def codec_diagnostics(rounds, summary):
    rows = []
    for (point, seed), frame in rounds.groupby(["point", "partition_seed"], sort=False):
        endpoint = summary[(summary.point == point) & (summary.partition_seed == seed)].set_index("variant")
        legacy = "legacy" if point in ("event", "strom") else "native_stc"
        cost = endpoint.loc[ten.MAIN_VARIANT[point], "unicast_total_bits"]
        record = {"point": point, "partition_seed": seed, "legacy_unicast_bits": int(endpoint.loc[legacy, "unicast_total_bits"]),
                  "improved_unicast_bits": int(cost), "codec_unicast_reduction_percent": 100 * (1 - cost / endpoint.loc[legacy, "unicast_total_bits"])}
        if point in ("event", "strom"):
            correction = 10 * (frame.exact_downlink_bits - frame.integer_only_downlink_bits).sum()
            record.update(coordinate_events=int(frame.coordinate_events.sum()),
                          late_120_firing_fraction=frame.tail(120).coordinate_events.mean() / (10 * 20570),
                          counts_patch_rounds=int(frame.downlink_mode.eq("counts_patch").sum()),
                          ordered_fallback_rounds=int(frame.downlink_mode.eq("ordered").sum()),
                          correction_bits=int(correction), correction_percent_of_total=100 * correction / cost,
                          maximum_canonical_error=frame.canonical_max_abs_error.max(),
                          final_residual_norm=frame.iloc[-1].get("residual_l2_norm", np.nan),
                          final_residual_max_over_threshold=frame.iloc[-1].get("residual_max_abs", np.nan) / .05 if point == "strom" else np.nan)
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    original_hashes = {name: ten.text_hash(ten.OUT / name) for name in ANALYSIS_NAMES}
    science, runs, summary, rounds, targets, intervals, target_intervals = validate()
    endpoints = runs.copy()
    endpoints["test_accuracy_percent"] = 100 * endpoints.final_test_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoints[field + "_Mbit"] = endpoints[field + "_bits"] / 1e6
    effects = intervals.copy()
    definition = {name: (field, ratio, scale) for name, field, ratio, scale in ten.previous.METRICS}
    for index, row in effects.iterrows():
        if row.metric.endswith("_ratio"):
            for field, value in reductions(row).items():
                effects.loc[index, field] = value
        scope = runs if row.scope == "all_10" else runs[runs.partition_seed != ten.SEEDS[0]]
        a = scope[scope.point == row.candidate].set_index("partition_seed")
        b = scope[scope.point == row.reference].set_index("partition_seed")
        field = definition[row.metric][0]
        for label, mask in (("lower", a[field] < b[field]), ("equal", a[field] == b[field]), ("higher", a[field] > b[field])):
            effects.loc[index, "candidate_" + label + "_count"] = int(mask.sum())
    per_target = target_pairs(targets)
    target_effects = target_intervals.copy()
    for index, row in target_effects.iterrows():
        for field, value in reductions(row).items():
            target_effects.loc[index, field] = value
        group = per_target[(per_target.candidate == row.candidate) & (per_target.reference == row.reference)
                           & (per_target.target_accuracy == row.target_accuracy)]
        if row.scope == "additional_9":
            group = group[group.partition_seed != ten.SEEDS[0]]
        for field in ("candidate_uses_fewer_bits", "candidate_uses_fewer_rounds", "same_sampled_round"):
            target_effects.loc[index, field + "_count"] = int(group[field].eq(True).sum())
    outputs = {"analysis_endpoint_comparison.csv": endpoints, "analysis_endpoint_effects.csv": effects,
               "analysis_target_pairs.csv": per_target, "analysis_target_effects.csv": target_effects,
               "analysis_codec_diagnostics.csv": codec_diagnostics(rounds, summary)}
    for name, table in outputs.items():
        ten.atomic_csv(table, ten.OUT / name)
    if original_hashes != {name: ten.text_hash(ten.OUT / name) for name in ANALYSIS_NAMES}:
        raise RuntimeError("Original analysis was unexpectedly modified")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(ten.OUT / "protocol.json")],
                                           cwd=REPO, text=True).strip()
    ten.atomic_json({"results_commit": result_commit, "contract_sha256": ten.digest(science),
                     "audit_source_sha256": ten.text_hash(Path(__file__)), "new_Strom_receipts_verified": 10,
                     "reused_Event_STC_receipts_verified": 30, "verified_round_records": 72000,
                     "verified_evaluated_model_fingerprints": 1240, "reproduced_original_analysis_csvs": 8,
                     "original_analysis_sha256": original_hashes, "original_analysis_preserved": True,
                     "independent_prefix_and_endpoint_accounting": True, "independent_paired_intervals_and_coverage": True,
                     "development_checkpoint_reused": False, "test_tuning_or_early_stopping": False,
                     "frozen_development_selection_sha256": ten.digest(science["development_selection"]),
                     "equivalence_or_noninferiority_claim": False,
                     "limits": ["fixed compact CNN and full-participation strong-skew regime",
                                "uncertainty across partition/SGD seeds on a fixed previously inspected test set",
                                "secondary intervals are descriptive, unadjusted for multiple comparisons",
                                "raw images, packet bytes, model weights and residual arrays are not reloaded",
                                "first saved crossings are retrospective and can later fall below target",
                                "no causal leak/reset/quantum attribution from these algorithm comparisons"]},
                    ten.OUT / "analysis_independent_audit.json")
    print("Verified 10 new Strom and 30 reused controls, 72000 rounds, 1240 sampled model fingerprints and eight original analysis tables.")
    print(target_effects[(target_effects.scope == "all_10") & (target_effects.candidate == "event") & (target_effects.reference == "strom")]
          [["target_accuracy", "estimate", "ci95_low", "ci95_high", "candidate_uses_fewer_bits_count", "candidate_uses_fewer_rounds_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
