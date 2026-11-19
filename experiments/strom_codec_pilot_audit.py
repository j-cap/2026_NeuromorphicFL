"""Verify and summarize the committed Ström pilot without research training.

python experiments/strom_codec_pilot_audit.py
git add experiments/results/strom_codec_pilot_v1
git commit -m "Refresh Strom-style codec pilot analysis"
git push origin main
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import strom_codec_pilot as pilot
import numpy as np
import pandas as pd


def validate():
    science = pilot.load_contract()
    tables, regressions = {}, []
    for point in pilot.POINT_BY_ID:
        tables[point] = pilot.verified_tables(point, science)
        regressions.append(pilot.regression_check(point, tables[point]))
    tables.update({point: pilot.reused_tables(point, science) for point in pilot.REFERENCE_POINTS})
    summary = pd.concat([table["summary"] for table in tables.values()], ignore_index=True)
    history = pd.concat([table["history"] for table in tables.values()], ignore_index=True)
    main = pilot.optimized(summary)
    main["final_train_l2_penalty"] = main.final_train_objective - main.final_train_ce
    main["native_strom_gain"] = [pilot.POINT_BY_ID[p].native_gain if p in pilot.POINT_BY_ID else np.nan for p in main.point]
    main["fedavg_equivalent_strom_gain"] = main.native_strom_gain / pilot.full.config_for("event").local_lr
    main["strom_threshold_gradient_units"] = [pilot.POINT_BY_ID[p].threshold if p in pilot.POINT_BY_ID else np.nan for p in main.point]
    event = main[main.point == "event"].iloc[0]
    contrasts = pd.DataFrame([{
        "reference": row.point,
        "event_accuracy_difference_pp": 100 * (event.final_test_accuracy - row.final_test_accuracy),
        "event_test_ce_difference": event.final_test_ce - row.final_test_ce,
        "event_over_reference_unicast_ratio": event.unicast_total_bits / row.unicast_total_bits,
        "event_over_reference_uplink_ratio": event.uplink_bits / row.uplink_bits}
        for _, row in main[main.point != "event"].iterrows()])
    targets = pilot.target_table(history)
    for name, rebuilt in (("comparison.csv", summary), ("comparison_history.csv", history),
                          ("main_runs.csv", main), ("event_contrasts.csv", contrasts), ("traffic_to_target.csv", targets)):
        saved = pd.read_csv(pilot.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
        pd.testing.assert_frame_equal(saved.reset_index(drop=True), rebuilt.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    if len(summary) != 32 or len(history) != 352 or len(main) != 9 or len(targets) != 96:
        raise RuntimeError("Pilot grid coverage differs")
    original = pilot.full.load_contract()
    if science["runtime"] != original["runtime"] or science["data_sha256"] != original["data_sha256"]:
        raise RuntimeError("Recorded data/runtime differs from the reused full-horizon experiment")
    return science, tables, main, history, pilot.optimized(targets), regressions


def main():
    science, tables, endpoints, history, targets, regressions = validate()
    endpoints = endpoints.copy()
    endpoints["test_accuracy_percent"] = 100 * endpoints.final_test_accuracy
    endpoints["inferred_parameter_l2_norm"] = np.sqrt(2 * endpoints.final_train_l2_penalty / .0005)
    for name in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoints[name + "_Mbit"] = endpoints[name + "_bits"] / 1e6
    target_rows = []
    event = targets[targets.point == "event"].set_index("target_accuracy")
    curves = pilot.optimized(history).set_index(["point", "round"])
    for _, row in targets[targets.point != "event"].iterrows():
        reference = event.loc[row.target_accuracy]
        common = bool(row.reached and reference.reached)
        record = {"reference": row.point, "target_accuracy": row.target_accuracy,
                  "event_reached": bool(reference.reached), "reference_reached": bool(row.reached),
                  "event_first_saved_round": reference.first_sampled_round,
                  "reference_first_saved_round": row.first_sampled_round,
                  "event_unicast_Mbit": reference.unicast_total_bits / 1e6,
                  "reference_unicast_Mbit": row.unicast_total_bits / 1e6,
                  "event_reduction_percent": 100 * (1 - reference.unicast_total_bits / row.unicast_total_bits) if common else np.nan}
        for prefix, point, reached, rnd in (("event", "event", reference.reached, reference.first_sampled_round),
                                             ("reference", row.point, row.reached, row.first_sampled_round)):
            record[prefix + "_accuracy_at_crossing"] = float(curves.loc[(point, int(rnd)), "test_accuracy"]) if reached else np.nan
        target_rows.append(record)
    diagnostics = []
    for point, table in tables.items():
        if point not in pilot.POINT_BY_ID:
            continue
        rounds = table["rounds"]
        config = pilot.config_for(point)
        if not np.array_equal(rounds.legacy_uplink_bits, 16 * rounds.coordinate_events + 64 * rounds.active_clients):
            raise RuntimeError("Pulse packet accounting differs")
        if not np.array_equal(rounds.legacy_server_bits, rounds.legacy_uplink_bits):
            raise RuntimeError("Replay accounting differs")
        if not np.array_equal(rounds.exact_downlink_bits,
                              np.minimum(rounds.aggregate_exact_downlink_bits, rounds.ordered_downlink_bits)):
            raise RuntimeError("Exact downlink format selection differs")
        ordered = 128 + 8 * np.ceil((20570 + 2 * rounds.coordinate_events) / 8).astype(np.int64)
        if not np.array_equal(ordered, rounds.ordered_downlink_bits):
            raise RuntimeError("Ordered-pulse frame lengths differ")
        last = rounds.iloc[-1]
        variants = table["summary"].set_index("variant")
        total = variants.loc["both_codecs", "unicast_total_bits"]
        diagnostics.append({"point": point, "threshold": config.strom_threshold, "native_gain": config.server_gain,
                            "fedavg_equivalent_gain": config.server_gain / config.local_lr,
                            "coordinate_events": int(rounds.coordinate_events.sum()),
                            "late_120_round_firing_fraction": rounds.tail(120).coordinate_events.mean() / (10 * 20570),
                            "final_residual_l2_gradient_units": last.residual_l2_norm,
                            "final_residual_l2_model_units": config.server_gain * last.residual_l2_norm,
                            "final_max_residual_gradient_units": last.residual_max_abs,
                            "final_max_residual_over_threshold": last.residual_max_abs / config.strom_threshold,
                            "counts_patch_rounds": int(rounds.downlink_mode.eq("counts_patch").sum()),
                            "ordered_fallback_rounds": int(rounds.downlink_mode.eq("ordered").sum()),
                            "exact_fidelity_overhead_percent_of_total": 100 * 10 * (rounds.exact_downlink_bits - rounds.integer_only_downlink_bits).sum() / total,
                            "legacy_unicast_Mbit": variants.loc["legacy", "unicast_total_bits"] / 1e6,
                            "codec_unicast_reduction_percent": 100 * (1 - total / variants.loc["legacy", "unicast_total_bits"])})
    for name, table in (("analysis_endpoint_comparison.csv", endpoints),
                        ("analysis_target_comparison.csv", pd.DataFrame(target_rows)),
                        ("analysis_codec_diagnostics.csv", pd.DataFrame(diagnostics))):
        pilot.atomic_csv(table, pilot.OUT / name)
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(pilot.OUT / "protocol.json")],
                                           cwd=REPO, text=True).strip()
    pilot.atomic_json({"results_commit": result_commit, "contract_sha256": pilot.digest(science),
                       "audit_source_sha256": pilot.text_hash(Path(__file__)), "scope": pilot.protocol_spec()["scope"],
                       "verified_new_receipts": 6, "verified_reused_full_horizon_prefixes": 3,
                       "new_round_records": 3600, "new_model_evaluations": 66,
                       "reproduced_analysis_csv_tables": 5, "legacy_regressions": regressions,
                       "recorded_data_and_runtime_match_references": True,
                       "test_based_selection": False, "configuration_frozen_for_confirmation": False,
                       "residual_backlog_interpretation": "configuration-range_and_one-pulse-cap_concern_not_causal_proof",
                       "limits": ["one previously inspected seed and a 600-round horizon; no confidence intervals",
                                  "highest observed endpoint accuracy does not select a final baseline",
                                  "four normalized-gain trajectories lack independent archived real-data curves",
                                  "raw images, full packets, final residual arrays and model checkpoints not reloaded",
                                  "broader threshold/gain development tuning required before final comparative claims"]},
                      pilot.OUT / "analysis_independent_audit.json")
    print("Verified six new trajectories, three reused prefixes, all five analysis CSVs and two legacy regressions.")
    print(endpoints[["point", "test_accuracy_percent", "final_test_ce", "unicast_total_Mbit"]].to_string(index=False))
    print("Single-seed pilot; no test-based setting selected for confirmation.")


if __name__ == "__main__":
    main()
