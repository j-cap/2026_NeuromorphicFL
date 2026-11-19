"""Audit saved Ström development evidence and summarize it without training.

python experiments/strom_codec_development_audit.py
git add experiments/results/strom_codec_development_v1
git commit -m "Refresh validation-only Strom development analysis"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import strom_codec_development as study
import numpy as np
import pandas as pd

VARIANTS = (
    ("legacy", "legacy_uplink_bits", "legacy_server_bits"),
    ("uplink_codec_replay", "codec_uplink_bits", "codec_replay_bits"),
    ("aggregate_downlink", "legacy_uplink_bits", "exact_downlink_bits"),
    ("both_codecs", "codec_uplink_bits", "exact_downlink_bits"),
)
TARGETS = (.45, .50, .55)  # Same thresholds as the earlier comparison; validation diagnostics here.


def equal(a, b, message):
    if not np.array_equal(a, b):
        raise RuntimeError(message)


def independent_counts(tables, clients, dimension):
    """Reconstruct every history/summary total directly from saved round counters."""
    rounds = tables["rounds"]
    # The archived field counts clients with nonempty uploads, not local SGD participants.
    if not rounds.active_clients.between(0, clients).all():
        raise RuntimeError("Invalid nonempty-upload count")
    equal(rounds.active_clients, [sum(json.loads(modes).values()) for modes in rounds.uplink_modes],
          "Nonempty-upload codec counts differ")
    equal(rounds.legacy_uplink_bits, 16 * rounds.coordinate_events + 64 * rounds.active_clients,
          "Legacy pulse lengths differ")
    equal(rounds.legacy_server_bits, rounds.legacy_uplink_bits, "Legacy replay lengths differ")
    equal(rounds.exact_downlink_bits, np.minimum(rounds.aggregate_exact_downlink_bits, rounds.ordered_downlink_bits),
          "Exact downlink selection differs")
    equal(rounds.ordered_downlink_bits, 128 + 8 * ((dimension + 2 * rounds.coordinate_events + 7) // 8),
          "Ordered pulse lengths differ")
    checkpoint = 32 * dimension + 64
    for variant, up, down in VARIANTS:
        upload = rounds[up].cumsum().to_numpy()
        stream = rounds[down].to_numpy()
        if variant in ("legacy", "uplink_codec_replay"):
            stream = np.minimum(stream, checkpoint)
        stream = stream.cumsum()
        unicast_down = clients * (checkpoint + 32 * rounds["round"].to_numpy() + stream)
        totals = {"uplink_bits": upload, "server_stream_bits": stream,
                  "unicast_downlink_bits": unicast_down, "unicast_total_bits": upload + unicast_down,
                  "broadcast_downlink_bits": checkpoint + stream,
                  "broadcast_total_bits": upload + checkpoint + stream}
        history = tables["history"].query("variant == @variant")
        indices = history["round"].to_numpy(dtype=int) - 1
        summary = tables["summary"].query("variant == @variant").iloc[0]
        for field, values in totals.items():
            equal(history[field], values[indices], f"{variant} history {field} differs")
            if int(summary[field]) != int(values[-1]):
                raise RuntimeError(f"{variant} summary {field} differs")


def independently_selected(frame):
    rows = frame.query("method == 'strom' and state == 'complete'")
    rows = rows.sort_values(["final_validation_ce", "unicast_total_bits", "point_id"])
    budget = int(frame.query("method == 'event'").iloc[0].unicast_total_bits)
    eligible = rows[rows.unicast_total_bits <= budget]
    return {"quality": None if rows.empty else rows.iloc[0].point_id,
            "within_event_budget": None if eligible.empty else eligible.iloc[0].point_id}


def selection_ids(selection):
    return {role: selection[role]["point_id"] if selection[role] else None
            for role in ("quality", "within_event_budget")}


def validate():
    science = study.load_contract()
    split = study.load_split(science)
    points, decision = study.verify_screening(science, split)
    frozen = json.loads((study.OUT / "frozen_selection.json").read_text())
    by_id = {p.point_id: p for p in points}
    promoted = (study.EVENT, *[by_id[key] for key in decision["promoted_strom_ids"]])
    tables = {}
    for horizon, candidates in ((study.SCREEN, points), (study.FINAL, promoted)):
        for point in candidates:
            tables[point.point_id, horizon] = study.verified_tables(point, horizon, science, split)
            independent_counts(tables[point.point_id, horizon], science["protocol"]["n_clients"],
                               science["protocol"]["dimension"])
    for point in promoted:
        for kind in ("rounds", "evaluation", "history"):
            before, after = (tables[point.point_id, h][kind] for h in (study.SCREEN, study.FINAL))
            fields = [c for c in before if c not in ("codec_audit_seconds", "stage_horizon")]
            after = after[after["round"] <= study.SCREEN]
            pd.testing.assert_frame_equal(before[fields].reset_index(drop=True), after[fields].reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
    for horizon, decision_data in ((study.SCREEN, decision), (study.FINAL, frozen)):
        saved = pd.read_csv(study.OUT / ("screening.csv" if horizon == study.SCREEN else "finalists.csv"),
                            float_precision="round_trip")
        saved["error"] = saved["error"].fillna("")
        candidates = points if horizon == study.SCREEN else promoted
        rebuilt = study.ledger(candidates, horizon, science, split)
        pd.testing.assert_frame_equal(saved, rebuilt, check_dtype=False, check_exact=True)
        if independently_selected(saved) != selection_ids(decision_data["selection"]):
            raise RuntimeError("Independent selection differs")
    if (frozen["selection"] != study.select_points(study.ledger(promoted, study.FINAL, science, split))
            or frozen["final_receipts"] != study.receipt_fingerprints(promoted, study.FINAL, study.OUT)
            or frozen["screening_selection_sha256"] != study.text_hash(study.OUT / "screening_selection.json")
            or frozen["contract_sha256"] != study.digest(science) or frozen["split_sha256"] != study.digest(split)
            or frozen["official_test_evaluations"] != 0 or frozen["confirmation_included"]):
        raise RuntimeError("Frozen final provenance or scope differs")
    for role, choice in frozen["selection"].items():
        if role not in ("quality", "within_event_budget"):
            continue
        expected = study.asdict(study.config_for(by_id[choice["point_id"]])) if choice else None
        if frozen["selected_configs"][role] != expected:
            raise RuntimeError("Frozen final configuration differs")
    return science, tables, frozen


def summarize(science, tables):
    endpoints, targets, diagnostics, curves = [], [], [], []
    for (point_id, horizon), table in tables.items():
        point = study.EVENT if point_id == study.EVENT.point_id else next(p for p in study.INITIAL if p.point_id == point_id)
        config = study.config_for(point)
        summary = table["summary"].set_index("variant")
        endpoint = summary.loc["both_codecs"].to_dict()
        endpoint.update(point_id=point_id, horizon=horizon, threshold=point.threshold,
                        equivalent_gain=point.equivalent_gain if point.method == "strom" else np.nan,
                        native_gain=config.server_gain)
        endpoint["final_train_l2_penalty"] = endpoint["final_train_objective"] - endpoint["final_train_ce"]
        endpoint["inferred_parameter_l2_norm"] = np.sqrt(2 * endpoint["final_train_l2_penalty"] / config.regularization)
        endpoint["validation_train_ce_gap"] = endpoint["final_validation_ce"] - endpoint["final_train_ce"]
        endpoints.append(endpoint)
        history = table["history"].query("variant == 'both_codecs'").copy()
        history["horizon"] = horizon
        if horizon == study.FINAL:
            curves.append(history)
        for target in TARGETS:
            hits = history[history.validation_accuracy >= target]
            record = {"point_id": point_id, "horizon": horizon, "target_validation_accuracy": target,
                      "reached": not hits.empty, "first_saved_round": np.nan,
                      "accuracy_at_crossing": np.nan, "unicast_total_bits": np.nan}
            if not hits.empty:
                first = hits.iloc[0]
                record.update(first_saved_round=int(first["round"]), accuracy_at_crossing=first.validation_accuracy,
                              unicast_total_bits=int(first.unicast_total_bits))
            targets.append(record)
        rounds, last = table["rounds"], table["rounds"].iloc[-1]
        diagnostics.append({"point_id": point_id, "horizon": horizon,
                            "coordinate_events": int(rounds.coordinate_events.sum()),
                            "late_120_round_firing_fraction": rounds.tail(120).coordinate_events.mean() / (10 * 20570),
                            "counts_patch_rounds": int(rounds.downlink_mode.eq("counts_patch").sum()),
                            "ordered_fallback_rounds": int(rounds.downlink_mode.eq("ordered").sum()),
                            "canonical_roundoff_max": rounds.canonical_max_abs_error.max(),
                            "exact_fidelity_overhead_bits": int(10 * (rounds.exact_downlink_bits - rounds.integer_only_downlink_bits).sum()),
                            "legacy_unicast_bits": int(summary.loc["legacy", "unicast_total_bits"]),
                            "codec_unicast_reduction_percent": 100 * (1 - endpoint["unicast_total_bits"] / summary.loc["legacy", "unicast_total_bits"]),
                            "final_residual_l2_gradient_units": last.get("residual_l2_norm", np.nan),
                            "final_residual_max_over_threshold": last.get("residual_max_abs", np.nan) / point.threshold if point.threshold else np.nan})
    endpoints, targets = pd.DataFrame(endpoints), pd.DataFrame(targets)
    contrasts, target_pairs = [], []
    for _, row in endpoints.query("method == 'strom'").iterrows():
        event = endpoints[(endpoints.method == "event") & (endpoints.horizon == row.horizon)].iloc[0]
        contrast = {"point_id": row.point_id, "horizon": row.horizon,
                    "strom_minus_event_validation_accuracy_pp": 100 * (row.final_validation_accuracy - event.final_validation_accuracy),
                    "strom_minus_event_validation_ce": row.final_validation_ce - event.final_validation_ce}
        for field in ("uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"):
            contrast["strom_reduction_percent_" + field] = 100 * (1 - row[field] / event[field])
        contrasts.append(contrast)
        for target in TARGETS:
            hit = targets[(targets.point_id == row.point_id) & (targets.horizon == row.horizon)
                          & (targets.target_validation_accuracy == target)].iloc[0]
            ref = targets[(targets.point_id == "event_reference") & (targets.horizon == row.horizon)
                          & (targets.target_validation_accuracy == target)].iloc[0]
            both = hit.reached and ref.reached
            target_pairs.append({"point_id": row.point_id, "horizon": row.horizon,
                                 "target_validation_accuracy": target, "event_reached": ref.reached, "strom_reached": hit.reached,
                                 "event_first_saved_round": ref.first_saved_round, "strom_first_saved_round": hit.first_saved_round,
                                 "event_accuracy_at_crossing": ref.accuracy_at_crossing, "strom_accuracy_at_crossing": hit.accuracy_at_crossing,
                                 "event_unicast_bits": ref.unicast_total_bits, "strom_unicast_bits": hit.unicast_total_bits,
                                 "strom_reduction_percent": 100 * (1 - hit.unicast_total_bits / ref.unicast_total_bits) if both else np.nan})
    return {"analysis_endpoint_comparison.csv": endpoints, "analysis_endpoint_contrasts.csv": pd.DataFrame(contrasts),
            "analysis_traffic_to_targets.csv": targets, "analysis_target_comparison.csv": pd.DataFrame(target_pairs),
            "analysis_codec_diagnostics.csv": pd.DataFrame(diagnostics), "analysis_finalist_curves.csv": pd.concat(curves, ignore_index=True)}


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6))
    for key, name, color in (("event_reference", "Event", "#D55E00"),
                             ("strom_t0p05_g2", r"Ström ($\tau=0.05$, $g=2$)", "#0072B2")):
        rows = curves[(curves.point_id == key) & (curves["round"] >= 60)]
        for axis, xfield, metric, ylabel, scale in (
            (axes[0, 0], "round", "validation_ce", "Validation cross-entropy", 1),
            (axes[1, 0], "round", "validation_accuracy", "Validation accuracy (%)", 100),
            (axes[0, 1], "unicast_total_bits", "validation_ce", "Validation cross-entropy", 1),
            (axes[1, 1], "unicast_total_bits", "validation_accuracy", "Validation accuracy (%)", 100)):
            x = rows[xfield] / (1e6 if xfield.endswith("bits") else 1)
            axis.plot(x, scale * rows[metric], color=color, marker="o", markersize=3, linewidth=1.4, label=name)
            axis.set(xlabel="Total unicast traffic (Mbit)" if xfield.endswith("bits") else "Communication rounds", ylabel=ylabel)
            axis.grid(alpha=.22)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    fig.suptitle("Fixed Event and development-selected Ström\nValidation samples from round 60; one development seed", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 180} if suffix == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(study.OUT / ("analysis_trajectories." + suffix), **kwargs)
    plt.close(fig)


def main():
    science, tables, frozen = validate()
    outputs = summarize(science, tables)
    for name, frame in outputs.items():
        study.atomic_csv(frame, study.OUT / name)
    plot(outputs["analysis_finalist_curves.csv"])
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")],
                                           cwd=REPO, text=True).strip()
    study.atomic_json({"results_commit": result_commit, "contract_sha256": study.digest(science),
                       "audit_source_sha256": study.text_hash(Path(__file__)), "completion_receipts_verified": len(tables),
                       "archived_round_records": sum(len(t["rounds"]) for t in tables.values()),
                       "archived_model_evaluations": sum(len(t["evaluation"]) for t in tables.values()),
                       "verified_finalist_screening_prefixes": 2, "independent_traffic_reconstruction": True,
                       "independent_fixed_horizon_selection": True, "split_disjointness_verified": True,
                       "official_test_evaluations": 0, "confirmation_included": False,
                       "selection": frozen["selection"],
                       "derived_csv_sha256": {name: study.text_hash(study.OUT / name) for name in outputs},
                       "limits": ["one development partition/SGD seed; no confidence intervals or significance claim",
                                  "validation used for configuration selection; no independent predictive confirmation",
                                  "only the promoted candidate is evaluated to 1800; no full-grid optimum claim",
                                  "first saved target crossings are retrospective validation diagnostics, without interpolation",
                                  "raw images, packets, residual arrays and model checkpoints are not reloaded",
                                  "old STC test results use a different training split and are not combined with these scores"]},
                      study.OUT / "analysis_independent_audit.json")
    print("Verified 18 completion receipts, 13200 archived rounds, 238 archived model evaluations and two exact screening prefixes.")
    print(outputs["analysis_endpoint_comparison.csv"].query("horizon == 1800")[["point_id", "final_validation_ce", "final_validation_accuracy", "unicast_total_bits"]].to_string(index=False))
    print("Development evidence only; frozen settings are ready for independent confirmation.")


if __name__ == "__main__":
    main()
