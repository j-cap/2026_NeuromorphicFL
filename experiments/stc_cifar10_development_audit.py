"""Audit frozen stage-2 evidence after Git text-line-ending normalization.

The original campaign source, receipts and selection remain unchanged. Restore
only a byte representation whose SHA-256 exactly matches its recorded hash in
a temporary copy, then run the original campaign's semantic checks there.

    python experiments/stc_cifar10_development_audit.py
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import stc_cifar10_development as campaign
import numpy as np
import pandas as pd


def recover_recorded_bytes(data: bytes, expected: str) -> tuple[bytes, str]:
    """Accept only an exact recorded hash, optionally restoring LF/CRLF."""
    data.decode("utf-8")
    lf = data.replace(b"\r\n", b"\n")
    for kind, candidate in (("raw", data), ("LF", lf), ("CRLF", lf.replace(b"\n", b"\r\n"))):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return candidate, kind
    raise RuntimeError("Content does not match its recorded hash under raw/LF/CRLF encodings")


def validated_evidence(root: Path = campaign.OUT) -> tuple[pd.DataFrame, dict, dict]:
    contract = campaign.load_contract(root)
    frozen = json.loads((root / "frozen_selection.json").read_text(encoding="utf-8"))
    if set(frozen["receipt_sha256"]) != set(campaign.POINT_BY_ID):
        raise RuntimeError("Frozen receipt set differs from the declared grid")
    counts = {"raw": 0, "LF": 0, "CRLF": 0}
    with tempfile.TemporaryDirectory(prefix="stc-original-wire-bytes-") as directory:
        restored = Path(directory)
        for name in ("protocol.json", "frozen_selection.json"):
            (restored / name).write_bytes((root / name).read_bytes())
        for point in campaign.POINTS:
            summary, history, receipt_path = campaign.paths(point, root)
            receipt_bytes, kind = recover_recorded_bytes(
                receipt_path.read_bytes(), frozen["receipt_sha256"][point.point_id],
            )
            counts[kind] += 1
            (restored / receipt_path.name).write_bytes(receipt_bytes)
            receipt = json.loads(receipt_bytes)
            if receipt["state"] == "complete":
                for name, path in (("summary", summary), ("history", history)):
                    data, kind = recover_recorded_bytes(path.read_bytes(), receipt[name + "_sha256"])
                    counts[kind] += 1
                    (restored / path.name).write_bytes(data)
        frame = campaign.verified_rows(campaign.load_contract(restored), restored)
        with redirect_stdout(io.StringIO()):
            selected = campaign.select(restored)
            campaign.analyze(restored)
        if selected != frozen:
            raise RuntimeError("The original selection rule does not reproduce the frozen selection")
        for name in ("development_summary.csv", "development_ranking.csv", "development_frontier.csv"):
            pd.testing.assert_frame_equal(
                pd.read_csv(root / name, float_precision="round_trip"),
                pd.read_csv(restored / name, float_precision="round_trip"),
                check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
            )
    return frame, frozen, {
        "campaign_sha256": campaign.digest(contract),
        "source_hashes_match": True, "selection_reproduced": True,
        "aggregate_tables_reproduced": True, "file_hash_matches": counts,
        "runtime": contract["runtime"], "development_runs": len(frame),
        "complete": int((frame.state == "complete").sum()),
        "numerical_failures": int((frame.state == "numerical_failure").sum()),
    }


def key_points(frame: pd.DataFrame, root: Path = campaign.OUT) -> pd.DataFrame:
    reference = frame[frame.point_id == campaign.REFERENCE.point_id].iloc[0]
    ids = ["event_reference", "stc_u1_d1_g2", "stc_u0p25_d0p25_g2",
           "stc_u0p1_d0p1_g2", "stc_u0p025_d0p025_g2", "stc_u0p01_d0p01_g1",
           "stc_u0p5_d0p5_g1"]
    records = []
    for point_id in ids:
        row = frame[frame.point_id == point_id].iloc[0]
        history = pd.read_csv(campaign.paths(campaign.POINT_BY_ID[point_id], root)[1], float_precision="round_trip")
        penalty = row.final_train_objective - row.final_train_ce
        records.append({
            "point_id": point_id, "final_train_objective": row.final_train_objective,
            "final_train_ce": row.final_train_ce,
            "regularization_penalty": penalty,
            "parameter_l2_norm": np.sqrt(2 * penalty / campaign.CIFAR_BASE.regularization),
            "upload_Mbit": row.uplink_packetized_bits / 1e6,
            "downlink_unicast_Mbit": row.unicast_hybrid_downlink_bits / 1e6,
            "unicast_total_Mbit": row.unicast_hybrid_total_bits / 1e6,
            "broadcast_total_Mbit": row.broadcast_total_bits / 1e6,
            "unicast_traffic_ratio_to_event": row.unicast_hybrid_total_bits / reference.unicast_hybrid_total_bits,
            "upload_ratio_to_event": row.uplink_packetized_bits / reference.uplink_packetized_bits,
            "downlink_ratio_to_event": row.unicast_hybrid_downlink_bits / reference.unicast_hybrid_downlink_bits,
            "final_objective_reduction_percent": 100 * (1 - row.final_train_objective / reference.final_train_objective),
            "minimum_train_objective": history.train_objective.min(),
            "minimum_objective_round": int(history.loc[history.train_objective.idxmin(), "round"]),
            "last_six_objective_mean": history.train_objective.tail(6).mean(),
            "last_six_objective_std": history.train_objective.tail(6).std(),
        })
    return pd.DataFrame(records)


def plot(frame: pd.DataFrame, root: Path = campaign.OUT) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.4), layout="constrained")
    left, ce_axis = axes[0]
    trajectory, penalty_axis = axes[1]
    valid = frame[frame.state == "complete"]
    stc = valid[valid.method == "stc"]
    event = valid[valid.method == "event"].iloc[0]
    for gain, color in ((1, "#0072B2"), (2, "#D55E00")):
        group = stc[stc.server_gain == gain]
        for axis, field in ((left, "final_train_objective"), (ce_axis, "final_train_ce")):
            axis.scatter(group.unicast_hybrid_total_bits / 1e6, group[field],
                         c=color, marker="o" if gain == 1 else "s", s=42,
                         label=f"STC, gain {gain}", zorder=3)
    for axis, field in ((left, "final_train_objective"), (ce_axis, "final_train_ce")):
        axis.scatter(event.unicast_hybrid_total_bits / 1e6, event[field],
                     c="#333333", marker="*", s=145, label="Event-FedAvg", zorder=4)
        axis.axhline(event[field], color="#777777", lw=0.8, ls="--")
        axis.set_xscale("log")
        axis.set_xlabel("Upload + unicast download (Mbit)")
        axis.grid(alpha=0.25, which="both")
    for point_id, text, offset in (
        ("stc_u1_d1_g2", "p = 1 (frozen primary)", (-8, 48)),
        ("stc_u0p25_d0p25_g2", "p = 0.25", (-32, 16)),
        ("stc_u0p025_d0p025_g2", "p = 0.025", (10, -24)),
    ):
        row = stc[stc.point_id == point_id].iloc[0]
        left.annotate(text, (row.unicast_hybrid_total_bits / 1e6, row.final_train_objective),
                      xytext=offset, textcoords="offset points", fontsize=9,
                      ha="right" if offset[0] < 0 else "left",
                      arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.7})
    left.set(xscale="log", xlabel="Upload + unicast download (Mbit)",
             ylabel="Final regularized training objective", title="Development objective versus traffic")
    left.set_ylim(0.89, max(2.15, float(stc.final_train_objective.max()) + 0.08))
    left.legend(loc="upper right", frameon=True, fontsize=9)
    ce_axis.set(ylabel="Final training cross-entropy", title="Predictive loss versus traffic")

    curves = (
        ("event_reference", "Event-FedAvg", "#333333", "-"),
        ("stc_u1_d1_g2", "STC p=1, gain 2", "#D55E00", "-"),
        ("stc_u0p25_d0p25_g2", "STC p=0.25, gain 2", "#0072B2", "--"),
        ("stc_u0p025_d0p025_g2", "STC p=0.025, gain 2", "#009E73", "-."),
    )
    for point_id, label, color, style in curves:
        history = pd.read_csv(campaign.paths(campaign.POINT_BY_ID[point_id], root)[1])
        trajectory.plot(history["round"], history.train_objective, label=label, color=color, ls=style, lw=1.6)
        penalty_axis.plot(history["round"], history.train_objective - history.train_ce,
                          label=label, color=color, ls=style, lw=1.6)
    trajectory.set(xlabel="Communication round", ylabel="Regularized training objective",
              title="Training trajectories on one development partition", xlim=(0, 1800))
    penalty_axis.set(xlabel="Communication round", ylabel="L2 penalty (objective minus cross-entropy)",
                     title="Regularization contribution", xlim=(0, 1800))
    for axis in (trajectory, penalty_axis):
        axis.grid(alpha=0.25)
    trajectory.legend(loc="upper right", fontsize=9)
    fig.suptitle("STC compact-CIFAR development — one training partition, 1,800 rounds", fontsize=12)
    fig.savefig(root / "analysis_frontier.png", dpi=180)
    fig.savefig(root / "analysis_frontier.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main() -> None:
    frame, frozen, audit = validated_evidence()
    points = key_points(frame)
    reference = frame[frame.point_id == campaign.REFERENCE.point_id].iloc[0]
    better = frame[(frame.method == "stc") & (frame.state == "complete")
                   & (frame.final_train_objective < reference.final_train_objective)
                   & (frame.unicast_hybrid_total_bits < reference.unicast_hybrid_total_bits)]
    audit["stc_points_with_lower_objective_and_unicast_traffic"] = better.point_id.tolist()
    better_ce = frame[(frame.method == "stc") & (frame.state == "complete")
                      & (frame.final_train_ce < reference.final_train_ce)
                      & (frame.unicast_hybrid_total_bits < reference.unicast_hybrid_total_bits)]
    audit["stc_points_with_lower_cross_entropy_and_unicast_traffic"] = better_ce.point_id.tolist()
    selected = frame[frame.point_id == frozen["choices"]["quality"]["point_id"]].iloc[0]
    objective_gap = reference.final_train_objective - selected.final_train_objective
    ce_gap = reference.final_train_ce - selected.final_train_ce
    audit["selected_objective_gap_regularization_fraction"] = float((objective_gap - ce_gap) / objective_gap)
    audit["selected_cross_entropy_reduction_percent"] = float(100 * ce_gap / reference.final_train_ce)
    audit["frozen_choices"] = frozen["choices"]
    campaign.atomic_json(audit, campaign.OUT / "analysis_audit.json")
    campaign.atomic_csv(points, campaign.OUT / "analysis_key_points.csv")
    plot(frame)
    print(json.dumps(audit, indent=2))
    print(points.to_string(index=False))


if __name__ == "__main__":
    main()
