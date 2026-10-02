"""Single-seed long-horizon audit for the compact CIFAR-10 CNN.

The seven operating points are the frozen points shown in the manuscript's
communication frontier.  This audit selects a common final horizon before any
new ten-seed campaign is launched.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from p12_headline_ten_seed import CIFAR_BASE


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "p18_cifar10_cnn_horizon_audit"
PARTITION_SEED = 3500
TRAIN_SEED = 80000 + PARTITION_SEED
ROUNDS = 1800
EVAL_STRIDE = 60


@dataclass(frozen=True)
class Point:
    point_id: str
    comparison: str
    method: str
    configuration: str


POINTS = (
    Point("cifar_event", "quality-selected", "event", "event_t025_q005"),
    Point("cifar_strom_quality", "quality-selected", "strom", "strom_t0025"),
    Point("cifar_ef_quality", "quality-selected", "ef_topk", "ef_k05"),
    Point("cifar_sign_quality", "quality-selected", "sign_ef", "fixed"),
    Point("cifar_dense_gain2", "quality-selected", "dense", "dense_gain_2p0"),
    Point("cifar_strom_near", "nearest-traffic", "strom", "strom_t01"),
    Point("cifar_ef_traffic", "nearest-traffic", "ef_topk", "ef_k01"),
)
POINT_BY_ID = {point.point_id: point for point in POINTS}
METHOD_LABELS = {
    "event": "Event-FedAvg", "strom": "Strom", "ef_topk": "EF-TopK",
    "sign_ef": "Sign-EF", "dense": "Dense FedAvg",
}
METHOD_COLORS = {
    "event": "#D55E00", "strom": "#0072B2", "ef_topk": "#009E73",
    "sign_ef": "#CC79A7", "dense": "#595959",
}


def config_for(point: Point):
    config = replace(CIFAR_BASE, rounds=ROUNDS, eval_stride=EVAL_STRIDE)
    if point.configuration == "dense_gain_2p0":
        return replace(config, server_gain=2.0)
    if point.configuration == "strom_t0025":
        return replace(config, strom_threshold=0.0025)
    if point.configuration == "strom_t01":
        return replace(config, strom_threshold=0.01)
    if point.configuration == "ef_k05":
        return replace(config, topk_fraction=0.05)
    if point.configuration == "ef_k01":
        return replace(config, topk_fraction=0.01)
    if point.configuration in {"event_t025_q005", "fixed"}:
        return config
    raise ValueError(point.configuration)


def stem(point: Point) -> str:
    return f"{point.point_id}_p{PARTITION_SEED}_r{ROUNDS}"


def summary_path(point: Point) -> Path:
    return OUT / f"{stem(point)}.csv"


def history_path(point: Point) -> Path:
    return OUT / f"{stem(point)}_history.csv"


def run_point(point_id: str, force: bool = False) -> str:
    point = POINT_BY_ID[point_id]
    if summary_path(point).exists() and history_path(point).exists() and not force:
        return f"reuse {point_id}"
    federation = make_cifar10_federation(
        root=DATA, regime="strong", seed=PARTITION_SEED
    )
    result = run_final_baseline(
        federation=federation,
        architecture="cifar_cnn",
        method=point.method,
        config=config_for(point),
        seed=TRAIN_SEED,
        record_history=True,
    )
    history = result.pop("history")
    assert isinstance(history, pd.DataFrame)
    metadata = {
        "point_id": point.point_id,
        "benchmark": "cifar_cnn",
        "comparison": point.comparison,
        "method": point.method,
        "configuration": point.configuration,
        "partition_seed": PARTITION_SEED,
        "train_seed": TRAIN_SEED,
        "rounds": ROUNDS,
    }
    result.update(metadata)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        summary_path(point), index=False, float_format="%.10g"
    )
    history.assign(**metadata).to_csv(
        history_path(point), index=False, float_format="%.10g"
    )
    return (
        f"done {point_id}: test_ce={float(result['final_test_ce']):.4f}, "
        f"test_acc={100 * float(result['final_test_accuracy']):.2f}%"
    )


def campaign(workers: int, force: bool) -> None:
    pending = [
        point for point in POINTS
        if force or not (summary_path(point).exists() and history_path(point).exists())
    ]
    print(f"P18: {len(POINTS) - len(pending)}/{len(POINTS)} complete, "
          f"{len(pending)} pending, workers={workers}", flush=True)
    if workers == 1:
        for point in pending:
            print(run_point(point.point_id, force=force), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_point, point.point_id, force): point.point_id
            for point in pending
        }
        for future in as_completed(futures):
            point_id = futures[future]
            try:
                print(future.result(), flush=True)
            except BaseException as error:
                raise RuntimeError(f"P18 point failed: {point_id}") from error


def load_histories() -> pd.DataFrame:
    missing = [path for point in POINTS if not (path := history_path(point)).exists()]
    if missing:
        raise RuntimeError(f"P18 incomplete: {len(missing)} histories missing")
    return pd.concat(
        [pd.read_csv(history_path(point)) for point in POINTS], ignore_index=True
    )


def smooth(values: pd.Series) -> pd.Series:
    return values.rolling(window=5, center=True, min_periods=1).mean()


def analyze() -> None:
    histories = load_histories()
    analysis_rows = []
    milestone_rows = []
    for point in POINTS:
        curve = histories[histories.point_id == point.point_id].sort_values("round").copy()
        curve["smoothed_test_ce"] = smooth(curve.test_ce)
        late = curve[curve["round"] >= 0.8 * ROUNDS]
        slope = float(np.polyfit(late["round"], late["smoothed_test_ce"], 1)[0])
        minimum_index = curve["smoothed_test_ce"].idxmin()
        minimum_round = int(curve.loc[minimum_index, "round"])
        minimum_ce = float(curve.loc[minimum_index, "smoothed_test_ce"])
        final_ce = float(curve["smoothed_test_ce"].iloc[-1])
        late_improvement = float(
            late["smoothed_test_ce"].iloc[0] - late["smoothed_test_ce"].iloc[-1]
        )
        plateau = bool(
            abs(slope) <= 1e-4
            and late_improvement <= 0.01
            and final_ce - minimum_ce <= 0.01
        )
        if plateau:
            status = "plateau"
        elif slope >= 0 or (
            minimum_round < 0.8 * ROUNDS and final_ce - minimum_ce > 0.01
        ):
            status = "plateau_or_overfit"
        else:
            status = "still_improving"
        analysis_rows.append({
            "point_id": point.point_id,
            "comparison": point.comparison,
            "method": point.method,
            "final_round": ROUNDS,
            "final_test_ce": float(curve.test_ce.iloc[-1]),
            "final_test_accuracy": float(curve.test_accuracy.iloc[-1]),
            "minimum_smoothed_test_ce": minimum_ce,
            "minimum_smoothed_ce_round": minimum_round,
            "final_minus_minimum_smoothed_ce": final_ce - minimum_ce,
            "late_window_ce_improvement": late_improvement,
            "late_window_ce_slope_per_round": slope,
            "plateau_at_final_round": plateau,
            "trajectory_status": status,
        })
        for milestone in (120, 300, 600, 900, 1200, 1500, 1800):
            selected = curve.iloc[(curve["round"] - milestone).abs().argsort()[:1]].iloc[0]
            milestone_rows.append({
                "point_id": point.point_id,
                "method": point.method,
                "requested_round": milestone,
                "observed_round": int(selected["round"]),
                "test_ce": float(selected["test_ce"]),
                "test_accuracy": float(selected["test_accuracy"]),
                "unicast_hybrid_total_bits": float(selected["unicast_hybrid_total_bits"]),
            })
    analysis = pd.DataFrame(analysis_rows)
    milestones = pd.DataFrame(milestone_rows)
    analysis.to_csv(OUT / "horizon_analysis.csv", index=False, float_format="%.10g")
    milestones.to_csv(OUT / "milestones.csv", index=False, float_format="%.10g")
    print(analysis.to_string(index=False))


def figure() -> None:
    histories = load_histories()
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.8))
    for point in POINTS:
        curve = histories[histories.point_id == point.point_id].sort_values("round")
        linestyle = "-" if point.comparison == "quality-selected" else "--"
        label = METHOD_LABELS[point.method]
        if point.comparison == "nearest-traffic":
            label += " (nearest traffic)"
        axes[0].plot(curve["round"], smooth(curve["test_ce"]),
                     color=METHOD_COLORS[point.method], linestyle=linestyle,
                     linewidth=1.35, label=label)
        axes[1].plot(curve["round"], 100 * curve["test_accuracy"],
                     color=METHOD_COLORS[point.method], linestyle=linestyle,
                     linewidth=1.15)
    for axis in axes:
        axis.axvspan(0.8 * ROUNDS, ROUNDS, color="#bdbdbd", alpha=0.16, linewidth=0)
        axis.set_xlim(0, ROUNDS)
        axis.set_xlabel("Communication round")
        axis.grid(True, color="#dddddd", linewidth=0.45)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Test cross-entropy")
    axes[0].set_title("Held-out loss")
    axes[1].set_ylabel("Test accuracy [%]")
    axes[1].set_title("Held-out accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18, wspace=0.25)
    metadata = {"Creator": "p18_cifar10_cnn_horizon_audit.py", "CreationDate": None}
    fig.savefig(OUT / "horizon_audit.pdf", metadata=metadata)
    fig.savefig(OUT / "horizon_audit.png", dpi=220)
    plt.close(fig)


def protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "purpose": "select a common compact-CNN horizon before a ten-seed rerun",
        "partition_seed": PARTITION_SEED,
        "train_seed": TRAIN_SEED,
        "rounds": ROUNDS,
        "eval_stride": EVAL_STRIDE,
        "points": [point.__dict__ for point in POINTS],
        "selection": "all operating points frozen before the horizon audit",
        "diagnostic_window": "final 20% of rounds",
    }
    (OUT / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n")


def status() -> None:
    complete = [
        point.point_id for point in POINTS
        if summary_path(point).exists() and history_path(point).exists()
    ]
    print(f"P18 compact-CNN horizon audit: {len(complete)}/{len(POINTS)} complete")
    for point in POINTS:
        state = "complete" if point.point_id in complete else "pending"
        print(f"  {point.point_id}: {state}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    campaign_parser = commands.add_parser("campaign")
    campaign_parser.add_argument("--workers", type=int, choices=range(1, 8), default=1)
    campaign_parser.add_argument("--force", action="store_true")
    point_parser = commands.add_parser("point")
    point_parser.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point_parser.add_argument("--force", action="store_true")
    commands.add_parser("analyze")
    commands.add_parser("figure")
    commands.add_parser("protocol")
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "campaign":
        protocol()
        campaign(args.workers, args.force)
    elif args.command == "point":
        protocol()
        print(run_point(args.point_id, args.force))
    elif args.command == "analyze":
        analyze()
    elif args.command == "figure":
        figure()
    elif args.command == "protocol":
        protocol()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
