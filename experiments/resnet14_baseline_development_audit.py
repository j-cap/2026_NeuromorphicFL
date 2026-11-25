"""Audit saved ResNet baseline calibration without training or reading images.

python experiments/resnet14_baseline_development_audit.py
git add experiments/results/resnet14_baseline_development_v1
git commit -m "Refresh ResNet14 baseline calibration evidence audit"
git push origin main
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_baseline_development as study
from resnet14_transfer_audit import independent_traffic as pulse_traffic
import numpy as np
import pandas as pd

parent = study.parent
SUPPLEMENTARY = {
    "analysis_endpoints.csv", "analysis_event_comparisons.csv", "analysis_validation_targets.csv",
    "analysis_codec_diagnostics.csv", "analysis_screening_ranks.csv", "analysis_validation_curves.csv",
    "analysis_independent_audit.json", "validation_tradeoff.png", "validation_tradeoff.pdf",
}
REQUIRED = {"protocol.json", "promotion.json", "frozen_selection.json", "screening.csv", "finalists.csv",
            "selected.csv", "validation_comparison.csv", "analysis_audit.json"}
EXPLORATORY_TARGETS = (.65, .70, .75)


def raw_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_saved(name, frame):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    if "error" in saved:
        saved["error"] = saved.error.fillna("")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def independent_choice(frame, count):
    chosen = {}
    for method in study.METHODS:
        rows = [r for r in frame.to_dict("records") if r["method"] == method and r["state"] == "complete"]
        if not rows or not all(np.isfinite(r["final_validation_ce"]) for r in rows):
            raise RuntimeError("Missing finite calibration family: " + method)
        rows.sort(key=lambda r: (r["final_validation_ce"], r["unicast_total_bits"], r["point_id"]))
        chosen[method] = [r["point_id"] for r in rows[:count]]
    return chosen


def independent_traffic(table, point, *, clients, dimension):
    r = table["rounds"]
    fields = ["codec_uplink_bits", "exact_downlink_bits", "aggregate_downlink_bits", "ordered_downlink_bits"]
    bits = r[fields].to_numpy()
    if not np.isfinite(bits).all() or (bits < 0).any() or (bits % 8 != 0).any():
        raise RuntimeError("Invalid complete-byte frame counters")
    if (not r.active_clients.eq(clients).all()
            or not np.array_equal(r.active_clients, [sum(json.loads(v).values()) for v in r.uplink_modes])
            or not r.coordinate_events.between(0, clients * dimension).all()
            or not np.array_equal(r.coordinate_events, np.floor(r.coordinate_events))
            or not r.roundtrip_verified.eq(True).all()
            or not r.wire_protocol.eq(study.host.WIRE_PROTOCOL).all()
            or not r.model_sha256.str.fullmatch("[0-9a-f]{64}").all()):
        raise RuntimeError("Invalid participation, support, codec or model identity")
    for value in r.uplink_modes:
        modes = json.loads(value)
        if not all(isinstance(n, int) and n > 0 for n in modes.values()):
            raise RuntimeError("Invalid uplink mode multiplicities")
    expected_gain = point.gain * (1 + r["round"].to_numpy() / 100.) ** (-point.exponent)
    if not np.allclose(r.server_gain, expected_gain, rtol=1e-14, atol=1e-14):
        raise RuntimeError("Recorded gain differs from the frozen schedule")
    initial = 32 * dimension + 64
    if point.method == "ef_topk":
        if (not np.array_equal(r.ordered_downlink_bits, 96 + 32 * clients + r.codec_uplink_bits)
                or not np.array_equal(r.exact_downlink_bits, np.minimum(r.aggregate_downlink_bits, r.ordered_downlink_bits))
                or not np.array_equal(r.downlink_mode.eq("ordered_replay"), r.ordered_downlink_bits < r.aggregate_downlink_bits)):
            raise RuntimeError("EF aggregate/replay frame selection differs")
        allowed = lambda mode: mode == "dense_float32" or mode.startswith("support_")
        if any(not all(allowed(m) for m in json.loads(value)) for value in r.uplink_modes):
            raise RuntimeError("Unknown EF uplink representation")
        if not r.downlink_mode.map(lambda m: m == "ordered_replay" or m.startswith("aggregate_")).all():
            raise RuntimeError("Unknown EF downlink representation")
    else:
        mode = study.host.MAIN[point.method]
        if (not r.ordered_downlink_bits.eq(0).all() or not np.array_equal(r.exact_downlink_bits, r.aggregate_downlink_bits)
                or not r.downlink_mode.eq(mode).all()
                or any(json.loads(v) != {mode: clients} for v in r.uplink_modes)):
            raise RuntimeError("Dense/STC frame representation differs")
        if point.method == "dense" and (not r.codec_uplink_bits.eq(clients * initial).all()
                                         or not r.exact_downlink_bits.eq(initial).all()):
            raise RuntimeError("Dense full-float32 frame length differs")
    upload = r.codec_uplink_bits.to_numpy(dtype=np.int64).cumsum()
    stream = r.exact_downlink_bits.to_numpy(dtype=np.int64).cumsum()
    down = clients * (initial + 32 * r["round"].to_numpy(dtype=np.int64) + stream)
    totals = {"uplink_bits": upload, "server_stream_bits": stream, "unicast_downlink_bits": down,
              "unicast_total_bits": upload + down, "broadcast_downlink_bits": initial + stream,
              "broadcast_total_bits": upload + initial + stream}
    h, summary = table["history"], table["summary"].iloc[0]
    if not h.variant.eq(study.host.MAIN[point.method]).all() or len(table["summary"]) != 1:
        raise RuntimeError("Unexpected primary accounting variant")
    indices = h["round"].to_numpy(dtype=int) - 1
    if not np.array_equal(h["rounds"], indices + 1) or summary["rounds"] != len(r):
        raise RuntimeError("Accounting prefix lengths differ")
    for field, values in totals.items():
        if not np.array_equal(h[field], values[indices]) or int(summary[field]) != int(values[-1]):
            raise RuntimeError("Independent baseline traffic reconstruction differs: " + field)


def dominated(frame, row, quality, higher):
    q, cost = frame[quality], frame.unicast_total_bits
    better = q >= row[quality] if higher else q <= row[quality]
    strict = q > row[quality] if higher else q < row[quality]
    return bool((better & (cost <= row.unicast_total_bits) & (strict | (cost < row.unicast_total_bits))).any())


def plot(curves, endpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = [
        ("event_t0p01_q0p0025", "Event", "#D55E00", "-", "o"),
        ("strom_t0p005_g2", "Strom", "#0072B2", "-", "s"),
        ("dense_g1p5_a0", "Dense", "#666666", "-", "D"),
        ("stc_u1_d1_g4", "STC p=1", "#009E73", "-", "^"),
        ("ef_topk_p0p01_g4", "EF-TopK p=0.01", "#CC79A7", "-", "v"),
        ("stc_u0p05_d0p05_g4", "STC p=0.05 (secondary)", "#56B4E9", "--", "^"),
        ("stc_u0p01_d0p01_g4", "STC p=0.01 (secondary)", "#E69F00", ":", "^"),
    ]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        handles = []
        for key, label, color, line, marker in styles:
            h = curves[curves.point_id == key]
            row = endpoints[endpoints.point_id == key].iloc[0]
            handle, = axes[0].plot(h.unicast_total_Gbit, 100 * h.validation_accuracy, label=label,
                                   color=color, linestyle=line, linewidth=1.6)
            handles.append(handle)
            axes[0].scatter(row.unicast_total_Gbit, row.validation_accuracy_percent, color=color, marker=marker, s=27, zorder=4)
            axes[1].scatter(row.unicast_total_Gbit, row.final_validation_ce, color=color, marker=marker, s=55, zorder=4)
        others = endpoints[~endpoints.point_id.isin([s[0] for s in styles])]
        axes[1].scatter(others.unicast_total_Gbit, others.final_validation_ce, facecolors="none", edgecolors="#777777", s=40)
        offsets = {"Event": (-30, -19), "Strom": (7, 5), "Dense": (-42, 8), "STC p=1": (8, 9),
                   "EF-TopK p=0.01": (6, -15), "STC p=0.05 (secondary)": (-14, 10),
                   "STC p=0.01 (secondary)": (5, -17)}
        for key, label, color, _, _ in styles:
            row = endpoints[endpoints.point_id == key].iloc[0]
            text = label.replace(" (secondary)", "").replace("EF-TopK p=0.01", "EF-TopK")
            axes[1].annotate(text, (row.unicast_total_Gbit, row.final_validation_ce), xytext=offsets[label],
                             textcoords="offset points", fontsize=8.5, color=color)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("Cumulative conservative unicast (Gbit)")
            ax.grid(True, alpha=.22)
        axes[0].set(title="(a) Validation accuracy along each trajectory", ylabel="Validation accuracy (%)", ylim=(8, 81))
        axes[1].set(title="(b) Round-1800 validation cross-entropy", ylabel="Validation cross-entropy", ylim=(.65, 1.13))
        axes[1].set_xlim(.48, 350)
        fig.legend(handles=handles, labels=[s[1] for s in styles], loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(.5, .005), fontsize=9)
        fig.subplots_adjust(left=.065, right=.98, top=.91, bottom=.28, wspace=.25)
        for extension in ("png", "pdf"):
            fig.savefig(study.OUT / ("validation_tradeoff." + extension), dpi=180)
        plt.close(fig)


def main():
    originals = {p.name: raw_hash(p) for p in study.OUT.iterdir()
                 if p.is_file() and p.suffix in (".json", ".csv", ".png", ".pdf") and p.name not in SUPPLEMENTARY}
    if not REQUIRED.issubset(originals):
        raise RuntimeError("Original calibration evidence is incomplete")
    science = study.load_contract()
    screen = study.ledger(study.POINTS, study.SCREEN, science)
    promotion = study.decision(screen, 3, study.SCREEN, science)
    if (promotion != json.loads((study.OUT / "promotion.json").read_text())
            or promotion["selected"] != independent_choice(screen, 3)):
        raise RuntimeError("Frozen promotion differs from independent validation-CE ranking")
    points = tuple(study.BY_ID[k] for m in study.METHODS for k in promotion["selected"][m])
    final = study.ledger(points, study.FINAL, science)
    chosen = independent_choice(final, 1)
    frozen = {**study.decision(final, 1, study.FINAL, science),
              "configs": {m: asdict(study.config_for(study.BY_ID[ids[0]])) for m, ids in chosen.items()},
              "promotion_sha256": parent.text_hash(study.OUT / "promotion.json"), "confirmation_launched": False,
              "development_checkpoint_reuse_for_confirmation": False, "automatic_boundary_extension": False}
    if frozen != json.loads((study.OUT / "frozen_selection.json").read_text()) or frozen["selected"] != chosen:
        raise RuntimeError("Frozen final selection differs from the predeclared rule")
    study.validate_prefixes(points, science)
    tables = {}
    for horizon, candidates in ((study.SCREEN, study.POINTS), (study.FINAL, points)):
        for p in candidates:
            receipt = study.read_receipt(p, horizon, science)
            if receipt["state"] == "complete":
                table = study.verified_tables(p, horizon, science)
                independent_traffic(table, p, clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
                tables[p.point_id, horizon] = table
    # This audit describes the complete received campaign; it must not silently drop failed outcomes.
    if (len(tables), int(screen.state.eq("complete").sum()), int(final.state.eq("complete").sum())) != (63, 54, 9):
        raise RuntimeError("This completed-campaign audit expects all 54 screens and nine finalists")
    archived_rounds = sum(len(t["rounds"]) for t in tables.values())
    archived_evals = sum(len(t["evaluation"]) for t in tables.values())
    unique_rounds = archived_rounds - len(points) * study.SCREEN
    unique_evals = archived_evals - sum(len(tables[p.point_id, study.SCREEN]["evaluation"]) for p in points)
    if (archived_rounds, archived_evals, unique_rounds, unique_evals) != (48600, 873, 43200, 774):
        raise RuntimeError("Calibration work accounting differs")
    selected = final[final.point_id.isin(k[0] for k in chosen.values())].copy()
    selected["origin"] = "new_baseline_choice"
    prior = parent.load_contract()
    control_points = tuple(study.pulse.point_for(m) for m in study.pulse.METHODS)
    control_files = {str(path.relative_to(REPO)): raw_hash(path) for p in control_points
                     for path in parent.paths(p, "development", 7000, study.FINAL, parent.OUT).values()}
    controls = parent.ledger(control_points, study.FINAL, prior, parent.OUT)
    controls["origin"] = "read_only_pulse_incumbent"
    for p in control_points:
        table = parent.verified_tables(p, "development", 7000, study.FINAL, prior, parent.OUT)
        if any(c.startswith("test_") for c in table["evaluation"]):
            raise RuntimeError("Expected validation-only pulse control evidence")
        pulse_traffic(table)
        tables[p.point_id, study.FINAL] = table
    for name, frame in (("screening.csv", screen), ("finalists.csv", final), ("selected.csv", selected),
                        ("validation_comparison.csv", pd.concat([controls, selected], ignore_index=True))):
        compare_saved(name, frame)
    expected_audit = {"screen_candidates": 54, "screen_complete": 54, "screen_failed": 0, "promoted": 9,
                      "final_complete": 9, "final_failed": 0, "selection_sha256": parent.digest(frozen),
                      "max_unique_training_rounds": 43200, "official_test_images_or_metrics_read": False,
                      "new_confirmation_runs": 0,
                      "scope": "single_split_validation_only_baseline_calibration_not_predictive_confirmation"}
    if expected_audit != json.loads((study.OUT / "analysis_audit.json").read_text()):
        raise RuntimeError("Original coverage audit differs")
    endpoints = pd.concat([controls, final.assign(origin="new_baseline_finalist")], ignore_index=True)
    endpoints["role"] = np.where(endpoints.point_id.isin([p.point_id for p in control_points] + [k[0] for k in chosen.values()]),
                                  "frozen_primary", "secondary_finalist")
    endpoints = parent.add_regularization_columns(endpoints, parent.BASE.regularization)
    endpoints["validation_accuracy_percent"] = 100 * endpoints.final_validation_accuracy
    endpoints["validation_worst_class_accuracy_percent"] = 100 * endpoints.final_validation_worst_class_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoints[field + "_Gbit"] = endpoints[field + "_bits"] / 1e9
    endpoints["CE_cost_frontier"] = [not dominated(endpoints, row, "final_validation_ce", False) for _, row in endpoints.iterrows()]
    endpoints["accuracy_cost_frontier"] = [not dominated(endpoints, row, "final_validation_accuracy", True) for _, row in endpoints.iterrows()]
    event = endpoints[endpoints.method == "event"].iloc[0]
    comparisons, crossings, curves, diagnostics = [], [], [], []
    for _, row in endpoints.iterrows():
        if row.method != "event":
            effect = {"comparator": row.point_id, "role": row.role,
                      "Event_minus_comparator_accuracy_pp": 100 * (event.final_validation_accuracy - row.final_validation_accuracy),
                      "Event_minus_comparator_validation_ce": event.final_validation_ce - row.final_validation_ce}
            for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
                effect["Event_" + field + "_saving_percent"] = 100 * (1 - event[field + "_bits"] / row[field + "_bits"])
            comparisons.append(effect)
        table = tables[row.point_id, study.FINAL]
        h = table["history"]
        h = h[h.variant == ("both_codecs" if row.method in study.pulse.METHODS else study.host.MAIN[row.method])].copy()
        h["unicast_total_Gbit"] = h.unicast_total_bits / 1e9
        h["role"] = row.role
        curves.append(h)
        for target in (*parent.TARGETS, *EXPLORATORY_TARGETS):
            hits = h[h.validation_accuracy >= target]
            crossing = {"point_id": row.point_id, "method": row.method, "role": row.role, "target_accuracy": target,
                        "target_role": "previously_declared_diagnostic" if target in parent.TARGETS else "exploratory_after_inspection",
                        "reached": not hits.empty, "first_sampled_round": np.nan,
                        "unicast_total_bits": np.nan, "censor_round": np.nan if not hits.empty else study.FINAL}
            if not hits.empty:
                crossing.update(first_sampled_round=int(hits.iloc[0]["round"]), unicast_total_bits=int(hits.iloc[0].unicast_total_bits))
            crossings.append(crossing)
        r = table["rounds"]
        modes = Counter()
        for value in r.uplink_modes:
            modes.update(json.loads(value))
        diagnostics.append({"point_id": row.point_id, "method": row.method, "role": row.role,
                            "coordinate_events": int(r.coordinate_events.sum()),
                            "late_120_nonzero_client_fraction": float(r.tail(120).coordinate_events.mean() / (10 * 175258)),
                            "unicast_downlink_percent_of_total": 100 * row.unicast_downlink_bits / row.unicast_total_bits,
                            "uplink_modes": json.dumps(dict(sorted(modes.items())), sort_keys=True),
                            "downlink_modes": json.dumps(r.downlink_mode.value_counts().to_dict(), sort_keys=True),
                            "gain_at_grid_upper_boundary": bool(row.method in study.METHODS and row.gain == 4.),
                            "lossless_roundtrips_recorded": bool(row.roundtrip_verified)})
    ranked = screen.sort_values(["method", "final_validation_ce", "unicast_total_bits", "point_id"]).copy()
    ranked["within_method_CE_rank"] = ranked.groupby("method").cumcount() + 1
    ranked["promoted"] = ranked.point_id.isin(p.point_id for p in points)
    curves = pd.concat(curves, ignore_index=True)
    for name, frame in (("analysis_endpoints.csv", endpoints), ("analysis_event_comparisons.csv", pd.DataFrame(comparisons)),
                        ("analysis_validation_targets.csv", pd.DataFrame(crossings)), ("analysis_validation_curves.csv", curves),
                        ("analysis_codec_diagnostics.csv", pd.DataFrame(diagnostics)), ("analysis_screening_ranks.csv", ranked)):
        parent.atomic_csv(frame, study.OUT / name)
    plot(curves, endpoints)
    if (originals != {name: raw_hash(study.OUT / name) for name in originals}
            or control_files != {name: raw_hash(REPO / name) for name in control_files}):
        raise RuntimeError("Original calibration or pulse control evidence changed")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")], cwd=REPO, text=True).strip()
    parent.atomic_json({
        "results_commit": result_commit, "contract_sha256": parent.digest(science), "selection_sha256": parent.digest(frozen),
        "audit_source_sha256": parent.text_hash(Path(__file__)),
        "helper_sha256": {p: parent.text_hash(REPO / p) for p in ("experiments/resnet14_transfer_audit.py",)},
        "complete_screen_receipts": 54, "complete_final_receipts": 9, "failed_receipts": 0,
        "exact_continuation_prefixes": 9, "archived_baseline_round_rows": archived_rounds,
        "archived_baseline_evaluation_rows": archived_evals, "unique_baseline_training_rounds": unique_rounds,
        "unique_baseline_evaluation_points": unique_evals, "pulse_validation_controls": 2,
        "pulse_control_round_rows_verified": 3600, "pulse_control_sampled_fingerprints_verified": 62,
        "original_analysis_csvs_reproduced": 4, "original_campaign_file_count": len(originals),
        "original_campaign_sha256_bytes": originals, "pulse_control_sha256_bytes": control_files,
        "original_campaign_and_controls_preserved": True, "independent_selection_and_traffic_prefix_checks": True,
        "all_nine_finalists_reported": True, "test_images_or_saved_test_metrics_read": False,
        "new_training_runs": 0, "new_confirmation_runs": 0, "automatic_retuning": False,
        "limits": ["one validation split and one shared initialization/SGD seed; no predictive test confirmation",
                   "primary choices minimize CE among three promoted finalists per family; accuracy can rank differently",
                   "STC and EF chosen gain four is the grid maximum; no global optimum claim",
                   "Pareto flags compare the eleven observed final endpoints only, separately for CE and accuracy",
                   "65/70/75 percent targets and sparse-STC secondary emphasis are exploratory after inspection",
                   "saved hashes, metrics and counters verified; no packet/model/image arrays or training replay"],
    }, study.OUT / "analysis_independent_audit.json")
    print("Verified 54 screens, nine exact continuations, four original CSVs, and both validation controls.")
    print(endpoints[["point_id", "role", "final_validation_ce", "validation_accuracy_percent", "unicast_total_Gbit",
                     "CE_cost_frontier", "accuracy_cost_frontier"]].to_string(index=False))


if __name__ == "__main__":
    main()
