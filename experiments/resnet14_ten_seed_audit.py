"""Audit twenty saved ResNet confirmation outcomes without training or Torch.

python experiments/resnet14_ten_seed_audit.py
git add experiments/results/resnet14_ten_seed_v1
git commit -m "Refresh ResNet14 ten-seed evidence audit"
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
import resnet14_ten_seed as study
from resnet14_transfer_audit import independent_traffic
from strom_codec_ten_seed_audit import independent_interval
import numpy as np
import pandas as pd

parent = study.parent
ORIGINAL = ("protocol.json", "confirmation_completion.json", "confirmation_runs.csv", "confirmation_history.csv",
            "paired_per_seed.csv", "paired_summary.csv", "traffic_to_target.csv", "paired_target_summary.csv",
            "analysis_audit.json", "confirmation.png", "confirmation.pdf")
SUPPLEMENTARY = {"analysis_endpoints.csv", "analysis_scope_summary.csv", "analysis_endpoint_effects.csv",
                 "analysis_target_effects.csv", "analysis_codec_diagnostics.csv", "analysis_independent_audit.json"}


def raw_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_saved(name, frame):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    tolerance = 1e-10 if name in ("paired_summary.csv", "paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=tolerance, atol=tolerance)


def add_savings(frame):
    result = frame.copy()
    for i, row in result.iterrows():
        if "metric" not in result or row.metric.endswith("_ratio"):
            result.loc[i, "Event_saving_percent"] = 100 * (1 - row.estimate)
            result.loc[i, "Event_saving_ci95_low"] = 100 * (1 - row.ci95_high)
            result.loc[i, "Event_saving_ci95_high"] = 100 * (1 - row.ci95_low)
    return result


def main():
    originals = {p.name: raw_hash(p) for p in study.OUT.iterdir()
                 if p.is_file() and p.suffix in (".json", ".csv", ".png", ".pdf") and p.name not in SUPPLEMENTARY}
    if not set(ORIGINAL).issubset(originals):
        raise RuntimeError("Expected the complete original ten-seed analysis and figures")
    science = study.load_contract()
    completed, tables = [], {}
    reused, new = 0, 0
    for seed in study.SEEDS:
        for method in study.METHODS:
            receipt = study.read_outcome(method, seed, science)
            expected_state = "reused" if seed in study.PILOT_SEEDS else "complete"
            if receipt is None or receipt["state"] != expected_state:
                raise RuntimeError("All six reused and fourteen complete new outcomes are required")
            origin = "reused_pilot" if seed in study.PILOT_SEEDS else "new_additional"
            completed.append({"method": method, "partition_seed": seed, "state": expected_state, "origin": origin})
            table = study.verified_tables(method, seed, science)
            independent_traffic(table)
            tables[method, seed] = table
            if seed in study.PILOT_SEEDS:
                prior = parent.load_contract()
                original = parent.verified_tables(study.point_for(method), "confirmation", seed, study.ROUNDS, prior, parent.OUT)
                for kind in original:
                    pd.testing.assert_frame_equal(original[kind], table[kind][original[kind].columns], check_exact=True)
                reused += 1
            else:
                new += 1
    expected_completion = {"expected": 20, "expected_new": 14, "expected_reused": 6, "results": completed}
    if json.loads((study.OUT / "confirmation_completion.json").read_text()) != expected_completion:
        raise RuntimeError("Original completion and provenance coverage differ")
    rounds = sum(len(t["rounds"]) for t in tables.values())
    models = sum(len(t["evaluation"]) for t in tables.values())
    if (reused, new, rounds, models) != (6, 14, 36000, 620):
        raise RuntimeError("Expected complete twenty-outcome ResNet coverage")
    summary = pd.concat([t["summary"] for t in tables.values()], ignore_index=True)
    history = pd.concat([t["history"] for t in tables.values()], ignore_index=True)
    runs = parent.add_regularization_columns(summary[summary.variant == "both_codecs"].copy(), parent.BASE.regularization)
    if runs.model_sha256.nunique() != 20:
        raise RuntimeError("Expected twenty distinct final confirmation models")
    compare_saved("confirmation_runs.csv", runs)
    compare_saved("confirmation_history.csv", history)
    a, b = (runs[runs.method == m].set_index("partition_seed").sort_index() for m in study.METHODS)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(study.SEEDS) or set(b.index) != set(study.SEEDS):
        raise RuntimeError("Endpoint seed pairing differs")
    definitions = (("accuracy_difference_pp", "final_test_accuracy", False, 100), ("test_ce_difference", "final_test_ce", False, 1),
                   ("unicast_ratio", "unicast_total_bits", True, 1), ("upload_ratio", "uplink_bits", True, 1),
                   ("download_ratio", "unicast_downlink_bits", True, 1), ("broadcast_ratio", "broadcast_total_bits", True, 1))
    pairs, effects = [], []
    for metric, field, ratio, scale in definitions:
        values = a[field] / b[field] if ratio else scale * (a[field] - b[field])
        pairs.extend({"partition_seed": int(seed), "metric": metric, "value": float(value),
                      "origin": "reused_pilot" if seed in study.PILOT_SEEDS else "new_additional"} for seed, value in values.items())
        for scope, seeds in study.SCOPES.items():
            estimate, low, high = independent_interval(values.loc[list(seeds)], ratio)
            effects.append({"scope": scope, "metric": metric, "n": len(seeds), "estimate": estimate,
                            "ci95_low": low, "ci95_high": high, "direction": "Event_over_Strom" if ratio else "Event_minus_Strom"})
    effects = pd.DataFrame(effects)
    compare_saved("paired_per_seed.csv", pd.DataFrame(pairs))
    compare_saved("paired_summary.csv", effects)
    crossings = []
    for (method, seed), table in sorted(tables.items()):
        curve = table["history"][table["history"].variant == "both_codecs"]
        for target in parent.TARGETS:
            hits = table["evaluation"][table["evaluation"].test_accuracy >= target]
            row = {"comparison_id": method, "partition_seed": seed, "target_accuracy": target, "reached": not hits.empty,
                   "first_sampled_round": np.nan, "censor_round": np.nan if not hits.empty else study.ROUNDS,
                   "upload_bits": np.nan, "unicast_total_bits": np.nan, "broadcast_total_bits": np.nan}
            if not hits.empty:
                first = hits.iloc[0]
                cost = curve[curve["round"] == int(first["round"])].iloc[0]
                row.update(first_sampled_round=int(first["round"]), upload_bits=int(cost.uplink_bits),
                           unicast_total_bits=int(cost.unicast_total_bits), broadcast_total_bits=int(cost.broadcast_total_bits))
            crossings.append(row)
    targets = pd.DataFrame(crossings)
    compare_saved("traffic_to_target.csv", targets)
    target_effects = []
    for scope, seeds in study.SCOPES.items():
        for target, frame in targets[targets.partition_seed.isin(seeds)].groupby("target_accuracy"):
            x, y = (frame[frame.comparison_id == m].set_index("partition_seed").sort_index() for m in study.METHODS)
            common = x.reached & y.reached
            estimate, low, high = independent_interval(x.loc[common, "unicast_total_bits"] / y.loc[common, "unicast_total_bits"], True)
            target_effects.append({"scope": scope, "target_accuracy": target, "n_total": len(seeds),
                                   "n_event_reached": int(x.reached.sum()), "n_strom_reached": int(y.reached.sum()),
                                   "n_common_hits": int(common.sum()), "n": int(common.sum()),
                                   "estimate": estimate, "ci95_low": low, "ci95_high": high})
    target_effects = pd.DataFrame(target_effects)
    compare_saved("paired_target_summary.csv", target_effects)
    expected_audit = {"new_runs": 14, "reused_runs": 6, "seed_pairs": 10, "additional_pairs": 7,
                      "rounds_verified": rounds, "sampled_model_fingerprints": models, "all_pairs_retained": True,
                      "test_based_selection_or_stopping": False, "selection_sha256": parent.digest(science["selection"]),
                      "scopes": list(study.SCOPES), "unreached_targets_censored": True,
                      "scope": "fixed_configurations_ten_matched_ResNet_pairs_including_three_inspected_pilot_pairs"}
    if expected_audit != json.loads((study.OUT / "analysis_audit.json").read_text()):
        raise RuntimeError("Original saved coverage audit differs")
    endpoint = runs.copy()
    endpoint["test_accuracy_percent"] = 100 * endpoint.final_test_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoint[field + "_Gbit"] = endpoint[field + "_bits"] / 1e9
    descriptive = []
    for scope, seeds in study.SCOPES.items():
        for method in study.METHODS:
            frame = endpoint[(endpoint.method == method) & endpoint.partition_seed.isin(seeds)]
            row = {"scope": scope, "method": method, "n": len(frame)}
            for field in ("test_accuracy_percent", "final_test_ce", "uplink_Gbit", "unicast_downlink_Gbit",
                          "unicast_total_Gbit", "broadcast_total_Gbit", "final_train_objective", "final_train_ce",
                          "final_train_l2_penalty", "final_parameter_l2_norm", "final_test_worst_class_accuracy"):
                row[field + "_mean"], row[field + "_sd"] = frame[field].mean(), frame[field].std(ddof=1)
            descriptive.append(row)
    effects, target_effects = add_savings(effects), add_savings(target_effects)
    for i, row in target_effects.iterrows():
        frame = targets[(targets.target_accuracy == row.target_accuracy) & targets.partition_seed.isin(study.SCOPES[row.scope])]
        x, y = (frame[frame.comparison_id == m].set_index("partition_seed").sort_index() for m in study.METHODS)
        common = x.reached & y.reached
        target_effects.loc[i, "Event_fewer_bits"] = int((x.loc[common, "unicast_total_bits"] < y.loc[common, "unicast_total_bits"]).sum())
        target_effects.loc[i, "Event_fewer_rounds"] = int((x.loc[common, "first_sampled_round"] < y.loc[common, "first_sampled_round"]).sum())
        target_effects.loc[i, "same_sampled_round"] = int((x.loc[common, "first_sampled_round"] == y.loc[common, "first_sampled_round"]).sum())
        target_effects.loc[i, "Event_later_rounds"] = int((x.loc[common, "first_sampled_round"] > y.loc[common, "first_sampled_round"]).sum())
    diagnostics = []
    for (method, seed), table in tables.items():
        r = table["rounds"]
        main = table["summary"].set_index("variant").loc["both_codecs"]
        correction = 10 * int((r.exact_downlink_bits - r.integer_only_downlink_bits).sum())
        diagnostics.append({"method": method, "partition_seed": seed, "coordinate_events": int(r.coordinate_events.sum()),
                            "late_120_firing_fraction": r.tail(120).coordinate_events.mean() / (10 * 175258),
                            "counts_patch_rounds": int(r.downlink_mode.eq("counts_patch").sum()),
                            "ordered_rounds": int(r.downlink_mode.eq("ordered").sum()), "exact_fidelity_overhead_bits": correction,
                            "exact_fidelity_overhead_percent": 100 * correction / main.unicast_total_bits})
    for name, frame in (("analysis_endpoints.csv", endpoint), ("analysis_scope_summary.csv", pd.DataFrame(descriptive)),
                        ("analysis_endpoint_effects.csv", effects), ("analysis_target_effects.csv", target_effects),
                        ("analysis_codec_diagnostics.csv", pd.DataFrame(diagnostics))):
        parent.atomic_csv(frame, study.OUT / name)
    if originals != {name: raw_hash(study.OUT / name) for name in originals}:
        raise RuntimeError("Original campaign evidence or analysis changed")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")], cwd=REPO, text=True).strip()
    parent.atomic_json({
        "results_commit": result_commit, "contract_sha256": parent.digest(science), "audit_source_sha256": parent.text_hash(Path(__file__)),
        "helper_sha256": {p: parent.text_hash(REPO / p) for p in ("experiments/resnet14_transfer_audit.py", "experiments/strom_codec_ten_seed_audit.py")},
        "new_complete_receipts_verified": 14, "reused_pilot_receipts_verified": 6, "seed_pairs": 10, "additional_pairs": 7,
        "round_records_verified": rounds, "sampled_model_fingerprints_verified": models, "distinct_final_models": 20,
        "additional_training_rounds": 25200, "additional_evaluation_points": 434, "original_analysis_csvs_reproduced": 6,
        "original_campaign_file_count": len(originals), "original_campaign_sha256_bytes": originals,
        "original_campaign_preserved": True, "pilot_metrics_counters_and_identities_exactly_preserved": True,
        "independent_frame_and_traffic_prefix_checks": True, "independent_paired_intervals": True,
        "targets_rebuilt_from_raw_evaluations": True, "all_pairs_and_target_coverage_retained": True,
        "scopes": list(study.SCOPES), "test_based_tuning_or_stopping": False,
        "limits": ["one fixed previously inspected CIFAR10 test set and shared initialization",
                   "two fixed calibrated Torch pulse configurations; other ResNet baselines not evaluated here",
                   "descriptive unadjusted paired intervals; no field-wide SOTA or causal architecture mechanism claim",
                   "saved metrics/counters/hashes only; no image/packet/model arrays or training replay"],
    }, study.OUT / "analysis_independent_audit.json")
    print("Verified fourteen new and six reused outcomes, 36000 rounds, 620 sampled fingerprints, exact pilot reuse and six original CSVs.")
    print(effects.to_string(index=False))
    print(target_effects.to_string(index=False))


if __name__ == "__main__":
    main()
