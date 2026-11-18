"""Independently audit and summarize committed full-horizon codec evidence.

python experiments/event_codec_full_horizon_audit.py
git add experiments/results/event_codec_full_horizon_v1
git commit -m "Refresh full-horizon codec analysis"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import event_codec_full_horizon as full
import numpy as np
import pandas as pd


def validate():
    science = full.load_contract()
    tables, histories, references = [], [], []
    for point in full.POINTS:
        result = full.verified_point(point, science)
        references.append(full.reference_check(point, result))
        tables.append(result["summary"])
        histories.append(result["history"])
    summary, history = pd.concat(tables, ignore_index=True), pd.concat(histories, ignore_index=True)
    targets = full.targets_table(history)
    for name, rebuilt in (("comparison.csv", summary), ("comparison_history.csv", history), ("traffic_to_target.csv", targets)):
        original = pd.read_csv(full.OUT / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), original.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    previous = full.pilot.load_contract()
    if science["runtime"] != previous["runtime"] or science["data_sha256"] != previous["data_sha256"]:
        raise RuntimeError("Full horizon and pilot recorded runtime/data identities differ")
    return science, summary, history, targets, references


def optimized_rows(frame):
    return frame[[variant == full.MAIN_VARIANT[point] for point, variant in zip(frame.point, frame.variant)]].copy()


def plot(endpoint, targets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    rows = endpoint.set_index("point").loc[list(full.POINTS)]
    up = rows.uplink_bits.to_numpy() / 1e6
    fixed = (10 * (32 * 20570 + 64) + 32 * 10 * 1800) / 1e6
    stream = rows.unicast_downlink_bits.to_numpy() / 1e6 - fixed
    x = np.arange(3)
    axes[0].bar(x, up, color="#0072B2", label="Uplink")
    axes[0].bar(x, stream, bottom=up, color="#E69F00", label="Server stream, 10 copies")
    axes[0].bar(x, np.full(3, fixed), bottom=up + stream, color="#BBBBBB", label="Initialization + requests")
    for i, total in enumerate(up + stream + fixed):
        axes[0].text(i, total + 12, f"{total:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(x, ["Event", "STC p=1", "STC p=0.25"])
    axes[0].set(ylabel="Final unicast traffic (Mbit)", title="Similar totals, different traffic direction", ylim=(0, 1000))
    axes[0].legend(loc="upper left", fontsize=8)
    width = .25
    for i, (point, label, color) in enumerate((("event", "Event", "#333333"),
                                               ("stc_primary", "STC p=1", "#D55E00"),
                                               ("stc_sparse", "STC p=0.25", "#0072B2"))):
        selected = targets[targets.point == point].sort_values("target_accuracy")
        costs = selected.unicast_total_bits.to_numpy() / 1e6
        positions = x + (i - 1) * width
        axes[1].bar(positions, costs, width=width, color=color, label=label)
        for position, value in zip(positions, costs):
            axes[1].text(position, value + 4, f"{value:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(x, ["45%", "50%", "55%"])
    axes[1].set(xlabel="Target test accuracy", ylabel="Unicast traffic to target (Mbit)",
                title="First saved target crossing", ylim=(0, 365))
    axes[1].legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.22)
        axis.set_axisbelow(True)
    fig.suptitle("Improved encodings: 1,800 rounds, one matched partition (3500)", fontsize=11, y=.98)
    fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.18, wspace=.29)
    fig.savefig(full.OUT / "analysis_endpoint_targets.png", dpi=180)
    fig.savefig(full.OUT / "analysis_endpoint_targets.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    science, summary, history, targets, references = validate()
    endpoint, target = optimized_rows(summary), optimized_rows(targets)
    event_rounds = pd.read_csv(full.OUT / "event_rounds.csv", float_precision="round_trip")
    for point in full.POINTS:
        point_eval = pd.read_csv(full.OUT / f"{point}_evaluation.csv", float_precision="round_trip").set_index("round")
        selected = target.point == point
        target.loc[selected, "accuracy_at_first_sample"] = [
            float(point_eval.loc[int(row.first_sampled_round), "test_accuracy"]) if row.reached else float("nan")
            for _, row in target[selected].iterrows()]
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoint[field + "_Mbit"] = endpoint[field + "_bits"] / 1e6
    endpoint["test_accuracy_percent"] = 100 * endpoint.final_test_accuracy
    endpoint["train_l2_penalty"] = endpoint.final_train_objective - endpoint.final_train_ce
    endpoint["inferred_parameter_l2_norm"] = np.sqrt(2 * endpoint.train_l2_penalty / .0005)
    full.atomic_csv(endpoint, full.OUT / "analysis_endpoint_comparison.csv")
    full.atomic_csv(target, full.OUT / "analysis_target_comparison.csv")
    event = endpoint.set_index("point").loc["event"]
    comparisons = []
    for point in ("stc_primary", "stc_sparse"):
        reference = endpoint.set_index("point").loc[point]
        comparisons.append({"reference": point,
                            "event_accuracy_difference_pp": 100 * (event.final_test_accuracy - reference.final_test_accuracy),
                            "event_test_ce_difference": event.final_test_ce - reference.final_test_ce,
                            "event_unicast_reduction_percent": 100 * (1 - event.unicast_total_bits / reference.unicast_total_bits),
                            "event_uplink_reduction_percent": 100 * (1 - event.uplink_bits / reference.uplink_bits),
                            "event_downlink_increase_percent": 100 * (event.unicast_downlink_bits / reference.unicast_downlink_bits - 1),
                            "event_broadcast_reduction_percent": 100 * (1 - event.broadcast_total_bits / reference.broadcast_total_bits)})
    target_ratios = []
    for accuracy in full.TARGETS:
        selected = target[target.target_accuracy == accuracy].set_index("point")
        for point in ("stc_primary", "stc_sparse"):
            a, b = selected.loc["event"], selected.loc[point]
            target_ratios.append({"target_accuracy": accuracy, "reference": point,
                                  "event_first_saved_round": a.first_sampled_round,
                                  "reference_first_saved_round": b.first_sampled_round,
                                  "event_accuracy_at_first_sample": a.accuracy_at_first_sample,
                                  "reference_accuracy_at_first_sample": b.accuracy_at_first_sample,
                                  "event_unicast_bits": a.unicast_total_bits, "reference_unicast_bits": b.unicast_total_bits,
                                  "event_reduction_percent": 100 * (1 - a.unicast_total_bits / b.unicast_total_bits)})
    full.atomic_csv(pd.DataFrame(target_ratios), full.OUT / "analysis_target_ratios.csv")
    if not (event_rounds.active_clients == 10).all():
        raise RuntimeError("Unexpected Event client participation")
    if not np.array_equal(event_rounds.legacy_uplink_bits, 16 * event_rounds.coordinate_events + 64 * event_rounds.active_clients):
        raise RuntimeError("Legacy Event pulse accounting differs")
    ordered = 128 + 8 * np.ceil((20570 + 2 * event_rounds.coordinate_events) / 8).astype(np.int64)
    if not np.array_equal(event_rounds.ordered_downlink_bits, ordered):
        raise RuntimeError("Ordered Event frame lengths differ")
    if not np.array_equal(event_rounds.exact_downlink_bits,
                          np.minimum(event_rounds.aggregate_exact_downlink_bits, event_rounds.ordered_downlink_bits)):
        raise RuntimeError("Selected exact Event frame costs differ")
    events, mass = int(event_rounds.coordinate_events.sum()), int(event_rounds.aggregate_l1.sum())
    windows = []
    for first, last in ((1, 120), (121, 600), (601, 1200), (1201, 1800)):
        rows = event_rounds[event_rounds["round"].between(first, last)]
        windows.append({"first_round": first, "last_round": last,
                        "upload_reduction_percent": 100 * (1 - rows.codec_uplink_bits.sum() / rows.legacy_uplink_bits.sum()),
                        "server_stream_reduction_percent": 100 * (1 - rows.exact_downlink_bits.sum() / rows.legacy_uplink_bits.sum()),
                        "weighted_cancellation_fraction": 1 - rows.aggregate_l1.sum() / rows.coordinate_events.sum()})
    full.atomic_csv(pd.DataFrame(windows), full.OUT / "analysis_codec_windows.csv")
    reported = json.loads((full.OUT / "analysis_audit.json").read_text(encoding="utf-8"))
    audit = {"scope": "one_fixed_seed_final_round_and_prespecified_targets",
             "all_three_receipts_and_source_hashes_verified": True, "all_saved_derived_tables_reproduced": True,
             "recorded_data_and_runtime_match_pilot": True,
             "independently_available_regression_references": references,
             "user_local_recorded_regression_references": reported["regression_references"],
             "endpoint_event_vs_references": comparisons,
             "event_coordinate_events": events, "event_aggregate_l1": mass,
             "weighted_cancellation_fraction": 1 - mass / events,
             "event_aggregate_exact_frames_selected": int((event_rounds.downlink_mode == "counts_patch").sum()),
             "event_corrected_coordinate_rounds": int(event_rounds.correction_nonzeros.sum()),
             "maximum_count_times_jump_error": float(event_rounds.canonical_max_abs_error.max()),
             "roundoff_correction_fraction_of_final_unicast": float(10 * (event_rounds.exact_downlink_bits - event_rounds.integer_only_downlink_bits).sum() / event.unicast_total_bits),
             "limitations": ["one partition/training seed; no confidence intervals", "first saved crossing at 60-round intervals",
                             "full packet streams and per-example predictions are not committed",
                             "recorded dataset fingerprints checked; raw images not redownloaded",
                             "matching native STC confirmation data may exist locally but is not yet committed"]}
    full.atomic_json(audit, full.OUT / "analysis_independent_audit.json")
    plot(endpoint, target)
    print(endpoint[["point", "test_accuracy_percent", "final_test_ce", "uplink_Mbit", "unicast_downlink_Mbit", "unicast_total_Mbit"]].to_string(index=False))
    print(pd.DataFrame(target_ratios).to_string(index=False))


if __name__ == "__main__":
    main()
