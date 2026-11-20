"""Audit the complete common-seed core without training or changing old outputs.

python experiments/core_fresh_comparison_audit.py
git add experiments/results/core_fresh_comparison_v1
git commit -m "Refresh common-seed core comparison evidence audit"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import core_fresh_comparison as study
from event_strom_fresh_confirmation_audit import evidence_hash, savings
from strom_codec_ten_seed_audit import check_traffic, independent_interval
import numpy as np
import pandas as pd

ORIGINAL = ("comparison.csv", "comparison_history.csv", "main_runs.csv", "endpoint_summary.csv",
            "paired_per_seed.csv", "paired_summary.csv", "traffic_to_target.csv", "paired_target_summary.csv",
            "completion_audit.json", "analysis_audit.json", "comparison.png", "comparison.pdf")


def compare_saved(name, rebuilt):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    tolerance = 1e-10 if name in ("paired_summary.csv", "paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=tolerance, atol=tolerance)


def check_float_traffic(tables, *, point, config, clients=10, dimension=20570):
    rounds = tables["rounds"]
    checkpoint = 32 * dimension + 64
    topk = max(1, int(round(config.topk_fraction * dimension)))
    expected_up = clients * (checkpoint if point == "dense" else topk * (32 + (dimension-1).bit_length()) + 64)
    if not rounds.legacy_uplink_bits.eq(expected_up).all() or not rounds.active_clients.eq(clients).all():
        raise RuntimeError("Dense/EF native frame counts differ")
    if not np.array_equal(rounds.active_clients, [sum(json.loads(v).values()) for v in rounds.uplink_modes]):
        raise RuntimeError("Dense/EF encoded frame counts differ")
    if point == "dense":
        if not rounds.codec_uplink_bits.eq(expected_up).all() or not rounds.exact_downlink_bits.eq(checkpoint).all():
            raise RuntimeError("Dense float32 frame sizes differ")
        variants = (("dense_float32", "codec_uplink_bits", "exact_downlink_bits", False),)
    else:
        if (not rounds.coordinate_events.eq(clients * topk).all()
                or not np.array_equal(rounds.exact_downlink_bits, np.minimum(rounds.aggregate_downlink_bits, rounds.ordered_downlink_bits))):
            raise RuntimeError("EF support count or exact downlink selection differs")
        variants = (("native_ef_topk", "legacy_uplink_bits", "legacy_server_bits", True),
                    ("lossless_ef_topk", "codec_uplink_bits", "exact_downlink_bits", False))
    for variant, up, down, fallback in variants:
        counters = rounds[[up, down]].to_numpy()
        if not np.isfinite(counters).all() or (counters < 0).any() or (counters != np.floor(counters)).any():
            raise RuntimeError("Invalid float32 frame counters")
        upload = rounds[up].cumsum().to_numpy()
        stream = (np.minimum(rounds[down], checkpoint) if fallback else rounds[down]).cumsum().to_numpy()
        unicast = clients * (checkpoint + 32 * rounds["round"].to_numpy() + stream)
        totals = {"uplink_bits": upload, "server_stream_bits": stream, "unicast_downlink_bits": unicast,
                  "unicast_total_bits": upload + unicast, "broadcast_downlink_bits": checkpoint + stream,
                  "broadcast_total_bits": upload + checkpoint + stream}
        history = tables["history"][tables["history"].variant == variant]
        indices = history["round"].to_numpy(dtype=int) - 1
        summary = tables["summary"][tables["summary"].variant == variant].iloc[0]
        for field, values in totals.items():
            if not np.array_equal(history[field], values[indices]) or int(summary[field]) != int(values[-1]):
                raise RuntimeError(f"Independent {point}/{variant}/{field} accounting differs")


def check_statistics(runs, targets, paired, target_intervals):
    definitions = {name: (field, ratio, scale) for name, field, ratio, scale in study.previous.METRICS}
    for _, row in paired.iterrows():
        a, b = (study.previous.indexed(runs, p) for p in ("event", row.reference))
        field, ratio, scale = definitions[row.metric]
        values = a[field] / b[field] if ratio else scale * (a[field] - b[field])
        if int(row.n) != 10:
            raise RuntimeError("Paired endpoint denominator differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(values, ratio), rtol=1e-10, atol=1e-10)
    for _, row in target_intervals.iterrows():
        selected = targets[targets.target_accuracy == row.target_accuracy]
        a, b = (study.previous.indexed(selected, p) for p in ("event", row.reference))
        common = a.reached & b.reached
        values = a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"]
        expected = (10, int(a.reached.sum()), int(b.reached.sum()), int(common.sum()), len(values))
        if (int(row.n_total), int(row.n_candidate_reached), int(row.n_reference_reached), int(row.n_common_hits), int(row.n)) != expected:
            raise RuntimeError("Target coverage differs")
        np.testing.assert_allclose([row.estimate, row.ci95_low, row.ci95_high], independent_interval(values, True),
                                   rtol=1e-10, atol=1e-10, equal_nan=True)


def validate():
    science = study.load_contract()
    tables, completion = {}, []
    for i, seed in enumerate(study.SEEDS):
        for point in study.POINTS:
            receipt = study.read_receipt(point, i, science)
            expected = "reused" if point in study.REUSED else "complete"
            if receipt is None or receipt["state"] != expected:
                raise RuntimeError(f"Missing {expected} core result: {point}/{seed}")
            table = study.verified_tables(point, i, science)
            if point in ("dense", "ef_topk"):
                check_float_traffic(table, point=point, config=study.config_for(point))
            else:
                check_traffic(table, n_clients=10, dimension=20570, pulse=point in study.REUSED)
            tables[point, seed] = table
            completion.append({"point": point, "partition_seed": seed, "state": expected, "error": ""})
    if json.loads((study.OUT / "completion_audit.json").read_text()) != {"expected": 50, "results": completion}:
        raise RuntimeError("Saved completion coverage differs")
    summary, history = (pd.concat([v[k] for v in tables.values()], ignore_index=True) for k in ("summary", "history"))
    runs = study.previous.add_regularization_columns(study.optimized(summary), study.config_for("event").regularization)
    pairs, intervals = study.paired_endpoints(runs)
    targets = study.previous.target_outcomes(study.optimized(history))
    target_intervals = study.paired_targets(targets)
    for (point, seed), table in tables.items():
        curve = table["history"][table["history"].variant == study.MAIN[point]]
        for target in study.TARGETS:
            hits = table["evaluation"][table["evaluation"].test_accuracy >= target]
            saved = targets[(targets.point == point) & (targets.partition_seed == seed) & (targets.target_accuracy == target)].iloc[0]
            if bool(saved.reached) != (not hits.empty):
                raise RuntimeError("First-crossing hit status differs")
            if not hits.empty:
                first = hits.iloc[0]
                costs = curve[curve["round"] == int(first["round"])].iloc[0]
                if int(saved.first_sampled_round) != int(first["round"]) or saved.accuracy_at_crossing != first.test_accuracy:
                    raise RuntimeError("First-crossing round/accuracy differs")
                for field in ("unicast_total_bits", "broadcast_total_bits"):
                    if int(saved[field]) != int(costs[field]):
                        raise RuntimeError("First-crossing traffic differs")
            elif int(saved.censor_round) != 1800 or not np.isnan(saved.unicast_total_bits):
                raise RuntimeError("Unreached target censoring differs")
    rebuilt = {"comparison.csv": summary, "comparison_history.csv": history, "main_runs.csv": runs,
               "endpoint_summary.csv": study.previous.descriptive_summary(runs), "paired_per_seed.csv": pairs,
               "paired_summary.csv": intervals, "traffic_to_target.csv": targets, "paired_target_summary.csv": target_intervals}
    for name, frame in rebuilt.items():
        compare_saved(name, frame)
    check_statistics(runs, targets, intervals, target_intervals)
    counts = (len(tables), sum(len(t["rounds"]) for t in tables.values()), sum(len(t["evaluation"]) for t in tables.values()),
              len(summary), len(history), len(runs), len(pairs), len(targets))
    if counts != (50, 90000, 1550, 130, 4030, 50, 320, 150):
        raise RuntimeError(f"Complete core coverage differs: {counts}")
    old = pd.read_csv(study.previous.OUT / "paired_summary.csv", float_precision="round_trip")
    fields = ["metric", "n", "estimate", "ci95_low", "ci95_high"]
    pd.testing.assert_frame_equal(intervals[intervals.reference == "strom"][fields].reset_index(drop=True), old[fields],
                                  check_dtype=False, rtol=1e-10, atol=1e-10)
    audit = json.loads((study.OUT / "analysis_audit.json").read_text())
    if audit != {"new_training_results": 30, "verified_reused_controls": 20, "seed_pairs": 10, "verified_round_records": 90000,
                 "verified_evaluated_model_fingerprints": 1550, "all_expected_results_retained": True,
                 "original_pulse_results_modified": False, "test_tuning_or_early_stopping": False,
                 "equivalence_or_noninferiority_claim": False, "intervals": science["protocol"]["intervals"],
                 "source_and_config_sha256": study.digest(science)}:
        raise RuntimeError("Saved core audit scope differs")
    return science, runs, summary, tables, targets, intervals, target_intervals


def main():
    originals = {name: evidence_hash(study.OUT / name) for name in ORIGINAL}
    science, runs, summary, tables, targets, intervals, target_intervals = validate()
    effects = intervals.copy()
    definitions = {name: field for name, field, _, _ in study.previous.METRICS}
    for index, row in effects.iterrows():
        if row.metric.endswith("_ratio"):
            for field, value in savings(row).items():
                effects.loc[index, field] = value
        a, b = (study.previous.indexed(runs, p) for p in ("event", row.reference))
        field = definitions[row.metric]
        for label, mask in (("lower", a[field] < b[field]), ("equal", a[field] == b[field]), ("higher", a[field] > b[field])):
            effects.loc[index, "Event_" + label + "_count"] = int(mask.sum())
    target_effects = target_intervals.copy()
    for index, row in target_effects.iterrows():
        for field, value in savings(row).items():
            target_effects.loc[index, field] = value
    codec = []
    for (point, seed), table in tables.items():
        endpoint = table["summary"].set_index("variant")
        main = endpoint.loc[study.MAIN[point]]
        legacy = endpoint.iloc[0]
        rounds = table["rounds"]
        codec.append({"point": point, "partition_seed": seed, "legacy_unicast_bits": int(legacy.unicast_total_bits),
                      "main_unicast_bits": int(main.unicast_total_bits),
                      "lossless_codec_reduction_percent": 100*(1-main.unicast_total_bits/legacy.unicast_total_bits),
                      "aggregate_or_native_downlink_rounds": int((~rounds.downlink_mode.eq("ordered_replay") & ~rounds.downlink_mode.eq("ordered")).sum()),
                      "ordered_replay_rounds": int(rounds.downlink_mode.isin(["ordered_replay", "ordered"]).sum()),
                      "unicast_download_fraction": main.unicast_downlink_bits/main.unicast_total_bits})
    target_descriptive = targets.groupby(["point", "target_accuracy"]).agg(
        reached=("reached", "sum"), total=("reached", "size"), mean_rounds=("first_sampled_round", "mean"),
        mean_unicast_bits=("unicast_total_bits", "mean")).reset_index()
    for name, frame in (("analysis_endpoint_effects.csv", effects), ("analysis_target_effects.csv", target_effects),
                        ("analysis_codec_diagnostics.csv", pd.DataFrame(codec)), ("analysis_target_descriptive.csv", target_descriptive)):
        study.atomic_csv(frame, study.OUT / name)
    if originals != {name: evidence_hash(study.OUT / name) for name in ORIGINAL}:
        raise RuntimeError("Original core analysis unexpectedly modified")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")], cwd=REPO, text=True).strip()
    study.atomic_json({"results_commit": commit, "contract_sha256": study.digest(science), "audit_source_sha256": study.text_hash(Path(__file__)),
                       "helper_source_sha256": {p: study.text_hash(REPO/p) for p in (
                           "experiments/strom_codec_ten_seed_audit.py", "experiments/event_strom_fresh_confirmation_audit.py")},
                       "new_complete_receipts_verified": 30, "verified_reuse_receipts": 20, "verified_round_records": 90000,
                       "verified_evaluated_model_fingerprints": 1550, "reproduced_original_analysis_csvs": 8,
                       "all_ten_pairs_retained": True, "original_analysis_sha256": originals, "original_analysis_preserved": True,
                       "hash_format": "sha256_UTF8_LF_for_text_raw_bytes_for_PNG_PDF", "independent_prefix_and_endpoint_accounting": True,
                       "independent_paired_intervals_and_coverage": True, "first_crossings_checked_against_raw_evaluations": True,
                       "previous_Event_Strom_contrast_reproduced": True, "test_tuning_or_early_stopping": False,
                       "equivalence_or_noninferiority_claim": False, "equal_tuning_budget_claim": False,
                       "limits": ["compact CNN/full participation/strong-skew regime and fixed previously inspected test set",
                                  "descriptive unadjusted intervals; historical baseline selection criteria differ",
                                  "raw images, packets and NPZ model weights are not reloaded",
                                  "target ratios condition on common hits; one Dense 55pct target is censored",
                                  "no causal mechanism, wall-clock or field-wide SOTA claim"]},
                      study.OUT/"analysis_independent_audit.json")
    print("Verified 30 new and 20 reused results, 90000 rounds, 1550 sampled model fingerprints and eight original analysis tables.")
    print(effects[effects.metric == "unicast_traffic_ratio"][["reference", "Event_saving_percent", "Event_saving_ci95_low", "Event_saving_ci95_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
