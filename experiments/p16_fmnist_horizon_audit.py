"""Single-seed horizon audit for the Fashion-MNIST MLP and CNN tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_baseline_point import build_config
from neuromorphicfl.fmnist_event_benchmark import ensure_fashion_mnist
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "fashion-mnist"
OUT = REPO / "experiments" / "results" / "p16_fmnist_horizon_audit"
PARTITION_SEED = 2500
TRAIN_SEED = 70000 + PARTITION_SEED


@dataclass(frozen=True)
class Point:
    architecture: str
    method: str
    value: float | None
    rounds: int
    eval_stride: int

    @property
    def point_id(self) -> str:
        return f"{self.architecture}_{self.method}"


POINTS = (
    Point("mlp", "event", None, 1500, 15),
    Point("mlp", "strom", 0.00125, 1500, 15),
    Point("mlp", "ef_topk", 0.05, 1500, 15),
    Point("mlp", "sign_ef", None, 1500, 15),
    Point("mlp", "dense", None, 1500, 15),
    Point("cnn", "event", None, 1800, 10),
    Point("cnn", "strom", 0.005, 1800, 10),
    Point("cnn", "ef_topk", 0.05, 1800, 10),
    Point("cnn", "sign_ef", None, 1800, 10),
    Point("cnn", "dense", None, 1800, 10),
)
POINT_BY_ID = {point.point_id: point for point in POINTS}
METHOD_LABELS = {
    "event": "Event-FedAvg", "strom": "STrom", "ef_topk": "EF-TopK",
    "sign_ef": "Sign-EF", "dense": "Dense FedAvg",
}
METHOD_COLORS = {
    "event": "#D55E00", "strom": "#0072B2", "ef_topk": "#009E73",
    "sign_ef": "#CC79A7", "dense": "#595959",
}


def summary_path(point: Point) -> Path:
    return OUT / f"{point.point_id}_p{PARTITION_SEED}_r{point.rounds}_summary.csv"


def history_path(point: Point) -> Path:
    return OUT / f"{point.point_id}_p{PARTITION_SEED}_r{point.rounds}_history.csv"


def prepare_data() -> None:
    ensure_fashion_mnist(DATA)
    print("Fashion-MNIST is ready")


@contextmanager
def dataset_lock(timeout_seconds: float = 120.0):
    """Portable inter-process lock for checksum validation and dataset reads."""
    lock_directory = DATA / ".p16_loader_lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            os.mkdir(lock_directory)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {lock_directory}")
            time.sleep(0.1)
    try:
        yield
    finally:
        lock_directory.rmdir()


def run_point(point_id: str, force: bool = False) -> None:
    point = POINT_BY_ID[point_id]
    if summary_path(point).exists() and history_path(point).exists() and not force:
        print(f"skip existing {point_id}")
        return
    # The upstream loader verifies and, if needed, replaces dataset archives.
    # Serialize this short phase so concurrent workers cannot replace a file
    # while another worker hashes or reads it.
    DATA.mkdir(parents=True, exist_ok=True)
    with dataset_lock():
        federation = make_multiclass_federation(
            root=DATA, regime="strong", seed=PARTITION_SEED
        )
    config = replace(
        build_config(point.architecture, point.method, point.value),
        rounds=point.rounds,
        eval_stride=point.eval_stride,
    )
    result = run_final_baseline(
        federation=federation, architecture=point.architecture,
        method=point.method, config=config, seed=TRAIN_SEED,
        record_history=True,
    )
    history = result.pop("history")
    assert isinstance(history, pd.DataFrame)
    metadata = {
        "point_id": point.point_id, "partition_seed": PARTITION_SEED,
        "train_seed": TRAIN_SEED, "configuration_value": point.value,
    }
    result.update(metadata)
    history = history.assign(
        architecture=point.architecture, method=point.method, **metadata
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(summary_path(point), index=False, float_format="%.10g")
    history.to_csv(history_path(point), index=False, float_format="%.10g")
    print(
        f"{point_id}: round={point.rounds}, "
        f"test_ce={float(result['final_test_ce']):.4f}, "
        f"test_acc={100 * float(result['final_test_accuracy']):.2f}%"
    )


def campaign(workers: int, force: bool = False) -> None:
    if workers == 1:
        for point in POINTS:
            run_point(point.point_id, force=force)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_point, point.point_id, force) for point in POINTS]
        for future in futures:
            future.result()


def load_histories() -> pd.DataFrame:
    missing = [history_path(point) for point in POINTS if not history_path(point).exists()]
    if missing:
        raise RuntimeError(f"audit incomplete: {len(missing)} histories missing")
    return pd.concat(
        [pd.read_csv(history_path(point)) for point in POINTS], ignore_index=True
    )


def smooth(values: pd.Series, window: int = 5) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def analyze() -> pd.DataFrame:
    histories = load_histories()
    rows: list[dict[str, float | int | str | bool]] = []
    for point in POINTS:
        curve = histories[histories.point_id == point.point_id].sort_values("round").copy()
        curve["smoothed_test_ce"] = smooth(curve.test_ce)
        final_round = int(curve["round"].iloc[-1])
        late = curve[curve["round"] >= 0.8 * final_round]
        slope = float(np.polyfit(late["round"], late["smoothed_test_ce"], 1)[0])
        minimum_index = curve["smoothed_test_ce"].idxmin()
        minimum_round = int(curve.loc[minimum_index, "round"])
        final_smoothed_ce = float(curve["smoothed_test_ce"].iloc[-1])
        minimum_smoothed_ce = float(curve.loc[minimum_index, "smoothed_test_ce"])
        late_improvement = float(
            late["smoothed_test_ce"].iloc[0] - late["smoothed_test_ce"].iloc[-1]
        )
        plateau = bool(
            abs(slope) <= 1e-4 and late_improvement <= 0.01
            and final_smoothed_ce - minimum_smoothed_ce <= 0.01
        )
        if plateau:
            trajectory_status = "plateau"
        elif (
            slope >= 0.0
            or (
                minimum_round < 0.8 * final_round
                and final_smoothed_ce - minimum_smoothed_ce > 0.01
            )
        ):
            trajectory_status = "plateau_or_overfit"
        else:
            trajectory_status = "still_improving"
        rows.append({
            "point_id": point.point_id, "architecture": point.architecture,
            "method": point.method, "final_round": final_round,
            "final_test_ce": float(curve["test_ce"].iloc[-1]),
            "final_test_accuracy": float(curve["test_accuracy"].iloc[-1]),
            "minimum_smoothed_test_ce": minimum_smoothed_ce,
            "minimum_smoothed_ce_round": minimum_round,
            "final_minus_minimum_smoothed_ce": final_smoothed_ce - minimum_smoothed_ce,
            "late_window_ce_improvement": late_improvement,
            "late_window_ce_slope_per_round": slope,
            "plateau_at_final_round": plateau,
            "trajectory_status": trajectory_status,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "horizon_analysis.csv", index=False, float_format="%.10g")
    print(frame.to_string(index=False))
    return frame


def progression() -> pd.DataFrame:
    horizons = {"mlp": (600, 900, 1200, 1500), "cnn": (400, 600, 900, 1200, 1800)}
    rows: list[dict[str, float | int | str]] = []
    for architecture, architecture_horizons in horizons.items():
        for method in METHOD_LABELS:
            previous_ce: float | None = None
            previous_accuracy: float | None = None
            for rounds in architecture_horizons:
                path = OUT / (
                    f"{architecture}_{method}_p{PARTITION_SEED}_r{rounds}_summary.csv"
                )
                if not path.exists():
                    raise RuntimeError(f"missing saved horizon endpoint: {path.name}")
                endpoint = pd.read_csv(path).iloc[0]
                test_ce = float(endpoint["final_test_ce"])
                accuracy = float(endpoint["final_test_accuracy"])
                rows.append({
                    "architecture": architecture,
                    "method": method,
                    "rounds": rounds,
                    "final_test_ce": test_ce,
                    "final_test_accuracy": accuracy,
                    "ce_change_from_previous_horizon": (
                        np.nan if previous_ce is None else test_ce - previous_ce
                    ),
                    "accuracy_change_from_previous_horizon": (
                        np.nan if previous_accuracy is None else accuracy - previous_accuracy
                    ),
                })
                previous_ce = test_ce
                previous_accuracy = accuracy
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "horizon_progression.csv", index=False, float_format="%.10g")
    print(frame.to_string(index=False))
    return frame


def figure() -> None:
    histories = load_histories()
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.8))
    for axis, architecture, title in zip(
        axes, ("mlp", "cnn"), ("Fashion-MNIST MLP", "Fashion-MNIST CNN")
    ):
        for method in METHOD_LABELS:
            curve = histories[
                (histories.architecture == architecture) & (histories.method == method)
            ].sort_values("round")
            axis.plot(
                curve["round"], smooth(curve["test_ce"]),
                color=METHOD_COLORS[method], linewidth=1.45,
                label=METHOD_LABELS[method],
            )
            axis.scatter(
                curve["round"], curve["test_ce"], color=METHOD_COLORS[method],
                s=6, alpha=0.22, linewidths=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Communication round")
        axis.set_ylabel("Test cross-entropy")
        axis.grid(True, color="#dddddd", linewidth=0.45)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.18, wspace=0.25)
    metadata = {"Creator": "p16_fmnist_horizon_audit.py", "CreationDate": None}
    fig.savefig(OUT / "horizon_audit.pdf", metadata=metadata)
    fig.savefig(OUT / "horizon_audit.png", dpi=220)
    plt.close(fig)


def protocol() -> None:
    payload = {
        "schema_version": 1,
        "purpose": "select common final horizons before the ten-seed rerun",
        "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
        "points": [point.__dict__ | {"point_id": point.point_id} for point in POINTS],
        "plateau_rule": {
            "smoothing": "centered five-evaluation moving average",
            "late_window": "last 20% of communication rounds",
            "absolute_slope_per_round_at_most": 1e-4,
            "late_window_improvement_at_most": 0.01,
            "final_minus_minimum_smoothed_ce_at_most": 0.01,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n")


def status() -> None:
    done = sum(
        summary_path(point).exists() and history_path(point).exists()
        for point in POINTS
    )
    print(f"P16 horizon audit: {done}/{len(POINTS)} points complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-data")
    run = commands.add_parser("campaign")
    run.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    run.add_argument("--force", action="store_true")
    point = commands.add_parser("point")
    point.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point.add_argument("--force", action="store_true")
    commands.add_parser("analyze")
    commands.add_parser("progression")
    commands.add_parser("figure")
    commands.add_parser("protocol")
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_data()
    elif args.command == "campaign":
        campaign(args.workers, force=args.force)
    elif args.command == "point":
        run_point(args.point_id, force=args.force)
    elif args.command == "analyze":
        analyze()
    elif args.command == "progression":
        progression()
    elif args.command == "figure":
        figure()
    elif args.command == "protocol":
        protocol()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
