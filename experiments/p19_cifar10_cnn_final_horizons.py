"""Ten-seed compact CIFAR-10 CNN campaign at the P18-audited horizon.

The runner is resumable and Windows-safe. A run is complete only when both its
summary and history CSV exist. Dataset preparation happens in the parent process
before workers start, and each CSV is written atomically.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

# Avoid severe BLAS oversubscription when several campaign workers are used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import download_cifar10, make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from p12_headline_ten_seed import CIFAR_BASE


DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "p19_cifar10_cnn_final_horizons"
P18 = REPO / "experiments" / "results" / "p18_cifar10_cnn_horizon_audit"
PARTITION_SEEDS = tuple(range(3500, 4500, 100))
ROUNDS = 1800
EVAL_STRIDE = 60
REQUIRED_METRICS = (
    "final_train_objective",
    "final_test_ce",
    "final_test_accuracy",
    "final_worst_class_accuracy",
    "uplink_packetized_bits",
    "broadcast_total_bits",
    "unicast_hybrid_total_bits",
    "coordinate_events",
)


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
    "event": "Event-FedAvg",
    "strom": "Strom",
    "ef_topk": "EF-TopK",
    "sign_ef": "Sign-EF",
    "dense": "Dense FedAvg",
}
METHOD_COLORS = {
    "event": "#D55E00",
    "strom": "#0072B2",
    "ef_topk": "#009E73",
    "sign_ef": "#CC79A7",
    "dense": "#595959",
}


def train_seed(partition_seed: int) -> int:
    return 80000 + partition_seed


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


def output_paths(point: Point, seed_index: int) -> tuple[Path, Path]:
    partition_seed = PARTITION_SEEDS[seed_index]
    stem = f"{point.point_id}_s{seed_index}_p{partition_seed}_r{ROUNDS}"
    return OUT / f"{stem}_summary.csv", OUT / f"{stem}_history.csv"


def complete(point: Point, seed_index: int) -> bool:
    return all(path.exists() for path in output_paths(point, seed_index))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def metadata(point: Point, seed_index: int) -> dict[str, object]:
    partition_seed = PARTITION_SEEDS[seed_index]
    return {
        "point_id": point.point_id,
        "benchmark": "cifar_cnn",
        "comparison": point.comparison,
        "method": point.method,
        "configuration": point.configuration,
        "seed_index": seed_index,
        "partition_seed": partition_seed,
        "train_seed": train_seed(partition_seed),
        "rounds": ROUNDS,
    }


def prepare_data() -> None:
    download_cifar10(DATA)
    print(f"CIFAR-10 prepared at {DATA}")


def p18_paths(point: Point) -> tuple[Path, Path]:
    stem = f"{point.point_id}_p{PARTITION_SEEDS[0]}_r{ROUNDS}"
    return P18 / f"{stem}.csv", P18 / f"{stem}_history.csv"


def import_p18_seed_zero() -> None:
    imported = 0
    for point in POINTS:
        destination_summary, destination_history = output_paths(point, 0)
        if destination_summary.exists() and destination_history.exists():
            continue
        source_summary, source_history = p18_paths(point)
        if not source_summary.exists() or not source_history.exists():
            continue
        summary = pd.read_csv(source_summary)
        history = pd.read_csv(source_history)
        if len(summary) != 1:
            raise RuntimeError(f"invalid P18 summary: {source_summary}")
        checks = {
            "point_id": point.point_id,
            "partition_seed": PARTITION_SEEDS[0],
            "train_seed": train_seed(PARTITION_SEEDS[0]),
            "rounds": ROUNDS,
        }
        for column, expected in checks.items():
            if column not in summary or summary.iloc[0][column] != expected:
                raise RuntimeError(f"P18 seed-zero mismatch for {point.point_id}: {column}")
        extra = metadata(point, 0)
        for column, value in extra.items():
            summary[column] = value
            history[column] = value
        atomic_csv(summary, destination_summary)
        atomic_csv(history, destination_history)
        imported += 1
        print(f"imported P18 seed zero: {point.point_id}", flush=True)
    print(f"P18 seed-zero import complete: {imported} newly imported", flush=True)


def run_point(point_id: str, seed_index: int, force: bool = False) -> str:
    point = POINT_BY_ID[point_id]
    summary_path, history_path = output_paths(point, seed_index)
    if summary_path.exists() and history_path.exists() and not force:
        return f"reuse {point_id} seed={seed_index}"
    info = metadata(point, seed_index)
    federation = make_cifar10_federation(
        root=DATA,
        regime="strong",
        seed=int(info["partition_seed"]),
    )
    result = run_final_baseline(
        federation=federation,
        architecture="cifar_cnn",
        method=point.method,
        config=config_for(point),
        seed=int(info["train_seed"]),
        record_history=True,
    )
    history = result.pop("history")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("run_final_baseline did not return a history DataFrame")
    result.update(info)
    atomic_csv(pd.DataFrame([result]), summary_path)
    atomic_csv(history.assign(**info), history_path)
    return (
        f"done {point_id} seed={seed_index}: "
        f"test_ce={float(result['final_test_ce']):.4f}, "
        f"test_acc={100 * float(result['final_test_accuracy']):.2f}%"
    )


def campaign(workers: int, start_seed: int, end_seed: int, force: bool) -> None:
    if start_seed > end_seed:
        raise ValueError("start seed must not exceed end seed")
    prepare_data()
    if not force and start_seed == 0:
        import_p18_seed_zero()
    jobs = [
        (point.point_id, seed_index)
        for seed_index in range(start_seed, end_seed + 1)
        for point in POINTS
        if force or not complete(point, seed_index)
    ]
    total = len(POINTS) * (end_seed - start_seed + 1)
    print(
        f"P19 seeds {start_seed}-{end_seed}: {total-len(jobs)}/{total} complete, "
        f"{len(jobs)} pending, workers={workers}",
        flush=True,
    )
    if workers == 1:
        for point_id, seed_index in jobs:
            print(run_point(point_id, seed_index, force), flush=True)
        return
    executor = ProcessPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(run_point, point_id, seed_index, force): (point_id, seed_index)
        for point_id, seed_index in jobs
    }
    try:
        for future in as_completed(futures):
            point_id, seed_index = futures[future]
            try:
                print(future.result(), flush=True)
            except BaseException as error:
                for other in futures:
                    other.cancel()
                raise RuntimeError(
                    f"P19 run failed: {point_id} seed={seed_index}"
                ) from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def load_complete() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [
        (point.point_id, seed_index)
        for point in POINTS
        for seed_index in range(10)
        if not complete(point, seed_index)
    ]
    if missing:
        raise RuntimeError(f"P19 incomplete: {len(missing)} point/seed pairs missing")
    summaries = []
    histories = []
    for point in POINTS:
        for seed_index in range(10):
            summary_path, history_path = output_paths(point, seed_index)
            summaries.append(pd.read_csv(summary_path))
            histories.append(pd.read_csv(history_path))
    runs = pd.concat(summaries, ignore_index=True, sort=False)
    history = pd.concat(histories, ignore_index=True, sort=False)
    if len(runs) != 70 or runs.duplicated(["point_id", "seed_index"]).any():
        raise RuntimeError("invalid P19 run grid")
    return runs, history


def paired_interval(values: np.ndarray) -> dict[str, float | int]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    critical = 2.262157  # two-sided 95% t critical value for nine degrees of freedom
    half_width = critical * std / math.sqrt(n)
    return {
        "n": n,
        "mean_difference_points": 100 * mean,
        "std_difference_points": 100 * std,
        "ci95_low_points": 100 * (mean - half_width),
        "ci95_high_points": 100 * (mean + half_width),
        "n_positive": int(np.sum(values > 0)),
    }


def finalize() -> None:
    runs, histories = load_complete()
    summary = (
        runs.groupby(
            ["point_id", "benchmark", "comparison", "method", "configuration"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed_index", "size"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in REQUIRED_METRICS
                for stat in ("mean", "std")
            },
        )
        .sort_values(["comparison", "method", "point_id"])
    )
    indexed = runs.set_index(["point_id", "seed_index"])
    paired_rows = []
    for point in POINTS:
        if point.point_id == "cifar_event":
            continue
        differences = np.array([
            float(indexed.loc[("cifar_event", seed), "final_test_accuracy"])
            - float(indexed.loc[(point.point_id, seed), "final_test_accuracy"])
            for seed in range(10)
        ])
        paired_rows.append({
            "comparison_id": f"cifar_event_minus_{point.point_id}",
            "event_point": "cifar_event",
            "comparator_point": point.point_id,
            **paired_interval(differences),
        })
    atomic_csv(runs.sort_values(["point_id", "seed_index"]), OUT / "runs.csv")
    atomic_csv(histories.sort_values(["point_id", "seed_index", "round"]), OUT / "histories.csv")
    atomic_csv(summary, OUT / "aggregate.csv")
    atomic_csv(pd.DataFrame(paired_rows), OUT / "paired_differences.csv")
    protocol(write=True)
    print(summary[[
        "point_id", "n_seeds", "final_test_accuracy_mean",
        "final_test_accuracy_std", "unicast_hybrid_total_bits_mean",
    ]].to_string(index=False))


def verify() -> None:
    runs, histories = load_complete()
    problems = []
    expected_pairs = {(point.point_id, seed) for point in POINTS for seed in range(10)}
    observed_pairs = set(zip(runs.point_id.astype(str), runs.seed_index.astype(int)))
    if observed_pairs != expected_pairs:
        problems.append("summary grid does not match the frozen 7 x 10 protocol")
    for row in runs.itertuples(index=False):
        expected_partition = PARTITION_SEEDS[int(row.seed_index)]
        if int(row.partition_seed) != expected_partition:
            problems.append(f"partition seed drift: {row.point_id}/s{row.seed_index}")
        if int(row.train_seed) != train_seed(expected_partition):
            problems.append(f"training seed drift: {row.point_id}/s{row.seed_index}")
        if int(row.rounds) != ROUNDS:
            problems.append(f"horizon drift: {row.point_id}/s{row.seed_index}")
        if not (
            float(row.uplink_packetized_bits)
            < float(row.broadcast_total_bits)
            < float(row.unicast_hybrid_total_bits)
        ):
            problems.append(f"traffic ordering failed: {row.point_id}/s{row.seed_index}")
    maxima = histories.groupby(["point_id", "seed_index"])["round"].max()
    if set(maxima.astype(int)) != {ROUNDS} or len(maxima) != 70:
        problems.append("one or more histories do not reach round 1800")
    for metric in REQUIRED_METRICS:
        if metric not in runs or not np.isfinite(pd.to_numeric(runs[metric], errors="coerce")).all():
            problems.append(f"invalid required metric: {metric}")
    if problems:
        raise RuntimeError("\n".join(problems))
    print("P19 verification passed: 70 matched runs and complete histories")


def smooth(values: pd.Series) -> pd.Series:
    return values.rolling(window=5, center=True, min_periods=1).mean()


def figure() -> None:
    _, histories = load_complete()
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.8))
    for point in POINTS:
        point_history = histories[histories.point_id == point.point_id]
        pivot_ce = point_history.pivot(index="round", columns="seed_index", values="test_ce")
        pivot_acc = point_history.pivot(index="round", columns="seed_index", values="test_accuracy")
        rounds = pivot_ce.index.to_numpy()
        ce_mean = smooth(pivot_ce.mean(axis=1)).to_numpy()
        ce_std = pivot_ce.std(axis=1, ddof=1).to_numpy()
        acc_mean = (100 * pivot_acc.mean(axis=1)).to_numpy()
        acc_std = (100 * pivot_acc.std(axis=1, ddof=1)).to_numpy()
        linestyle = "-" if point.comparison == "quality-selected" else "--"
        label = METHOD_LABELS[point.method]
        if point.comparison == "nearest-traffic":
            label += " (nearest traffic)"
        color = METHOD_COLORS[point.method]
        axes[0].plot(rounds, ce_mean, color=color, linestyle=linestyle, linewidth=1.3, label=label)
        axes[0].fill_between(rounds, ce_mean - ce_std, ce_mean + ce_std, color=color, alpha=0.10)
        axes[1].plot(rounds, acc_mean, color=color, linestyle=linestyle, linewidth=1.15)
        axes[1].fill_between(rounds, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.10)
    for axis in axes:
        axis.axvspan(0.8 * ROUNDS, ROUNDS, color="#bdbdbd", alpha=0.12, linewidth=0)
        axis.set_xlim(0, ROUNDS)
        axis.set_xlabel("Communication round")
        axis.grid(True, color="#dddddd", linewidth=0.45)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Test cross-entropy")
    axes[0].set_title("Held-out loss")
    axes[1].set_ylabel("Test accuracy [%]")
    axes[1].set_title("Held-out accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.18, wspace=0.25)
    metadata_pdf = {"Creator": "p19_cifar10_cnn_final_horizons.py", "CreationDate": None}
    fig.savefig(OUT / "convergence.pdf", metadata=metadata_pdf)
    fig.savefig(OUT / "convergence.png", dpi=220)
    plt.close(fig)
    print(f"saved {OUT / 'convergence.pdf'}")


def protocol(write: bool = False) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "purpose": "ten matched CIFAR-10 CNN seeds at the P18-audited horizon",
        "horizon": ROUNDS,
        "eval_stride": EVAL_STRIDE,
        "seed_indices": list(range(10)),
        "partition_seeds": list(PARTITION_SEEDS),
        "training_seed_rule": "80000 + partition_seed",
        "selection": "all operating points and the horizon frozen before P19",
        "seed_zero_reuse": "P18 seed zero is imported after strict metadata checks",
        "points": [asdict(point) for point in POINTS],
        "reporting": {
            "table": "mean plus sample standard deviation over ten matched seeds",
            "paired_uncertainty": "two-sided 95% paired t interval",
            "checkpoint": "final round; no test-selected early stopping",
        },
    }
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "protocol.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    else:
        print(json.dumps(payload, indent=2))
    return payload


def status() -> None:
    complete_count = sum(complete(point, seed) for point in POINTS for seed in range(10))
    print(f"P19 compact CIFAR-10 CNN: {complete_count}/70 runs complete")
    for point in POINTS:
        seeds = [seed for seed in range(10) if complete(point, seed)]
        print(f"  {point.point_id}: {len(seeds)}/10 {seeds}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    sub.add_parser("import-p18")
    sub.add_parser("protocol")
    sub.add_parser("status")
    sub.add_parser("verify")
    sub.add_parser("finalize")
    sub.add_parser("figure")
    point_parser = sub.add_parser("point")
    point_parser.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point_parser.add_argument("--seed-index", type=int, choices=range(10), required=True)
    point_parser.add_argument("--force", action="store_true")
    campaign_parser = sub.add_parser("campaign")
    campaign_parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    campaign_parser.add_argument("--start-seed", type=int, choices=range(10), default=0)
    campaign_parser.add_argument("--end-seed", type=int, choices=range(10), default=9)
    campaign_parser.add_argument("--force", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-data":
        prepare_data()
    elif args.command == "import-p18":
        import_p18_seed_zero()
    elif args.command == "protocol":
        protocol()
    elif args.command == "status":
        status()
    elif args.command == "verify":
        verify()
    elif args.command == "finalize":
        finalize()
    elif args.command == "figure":
        figure()
    elif args.command == "point":
        prepare_data()
        print(run_point(args.point_id, args.seed_index, args.force))
    elif args.command == "campaign":
        protocol(write=True)
        campaign(args.workers, args.start_seed, args.end_seed, args.force)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
