"""Ten-seed Fashion-MNIST campaigns at the P16 audited final horizons.

This runner is intentionally resumable: a run is complete only when both its
summary and history CSV exist. It uses matched data/training seeds across all
methods and writes each CSV atomically so an interruption cannot masquerade as
a finished run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np
import pandas as pd

from final_baseline_point import build_config
from neuromorphicfl.fmnist_event_benchmark import ensure_fashion_mnist
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline


DATA = REPO / "data" / "fashion-mnist"
OUT = REPO / "experiments" / "results" / "p17_fmnist_final_horizons"
P16 = REPO / "experiments" / "results" / "p16_fmnist_horizon_audit"
PARTITION_SEEDS = tuple(range(2500, 3500, 100))
REQUIRED_METRICS = (
    "final_train_objective", "final_test_ce", "final_test_accuracy",
    "final_worst_class_accuracy", "uplink_packetized_bits",
    "broadcast_total_bits", "unicast_hybrid_total_bits", "coordinate_events",
)


@dataclass(frozen=True)
class Point:
    architecture: str
    method: str
    value: float | None
    comparison: str
    configuration: str
    rounds: int
    eval_stride: int

    @property
    def point_id(self) -> str:
        return f"fmnist_{self.architecture}_{self.method}_{self.comparison}"


POINTS = (
    Point("mlp", "event", None, "quality", "fixed", 1500, 15),
    Point("mlp", "strom", 0.00125, "quality", "0.00125", 1500, 15),
    Point("mlp", "ef_topk", 0.05, "quality", "0.05", 1500, 15),
    Point("mlp", "sign_ef", None, "quality", "fixed", 1500, 15),
    Point("mlp", "dense", None, "quality", "fixed", 1500, 15),
    Point("mlp", "strom", 0.02, "traffic", "0.02", 1500, 15),
    Point("mlp", "ef_topk", 0.01, "traffic", "0.01", 1500, 15),
    Point("cnn", "event", None, "quality", "fixed", 1800, 10),
    Point("cnn", "strom", 0.005, "quality", "0.005", 1800, 10),
    Point("cnn", "ef_topk", 0.05, "quality", "0.05", 1800, 10),
    Point("cnn", "sign_ef", None, "quality", "fixed", 1800, 10),
    Point("cnn", "dense", None, "quality", "fixed", 1800, 10),
    Point("cnn", "strom", 0.02, "traffic", "0.02", 1800, 10),
    Point("cnn", "ef_topk", 0.01, "traffic", "0.01", 1800, 10),
)
POINT_BY_ID = {point.point_id: point for point in POINTS}


def train_seed(partition_seed: int) -> int:
    return 70000 + partition_seed


def architecture_points(architecture: str) -> tuple[Point, ...]:
    return tuple(point for point in POINTS if point.architecture == architecture)


def output_paths(point: Point, seed_index: int) -> tuple[Path, Path]:
    partition_seed = PARTITION_SEEDS[seed_index]
    stem = f"{point.point_id}_s{seed_index}_p{partition_seed}_r{point.rounds}"
    directory = OUT / point.architecture
    return directory / f"{stem}_summary.csv", directory / f"{stem}_history.csv"


def complete(point: Point, seed_index: int) -> bool:
    return all(path.exists() for path in output_paths(point, seed_index))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def dataset_lock(timeout_seconds: float = 300.0):
    """Serialize dataset validation using an atomic cross-platform lock file."""
    lock_file = DATA / ".p17_loader.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            break
        except FileExistsError:
            # A newly created lock can briefly be empty before its owner PID is
            # written. Never classify an empty/unreadable lock as stale.
            try:
                owner_text = lock_file.read_text(encoding="ascii").strip()
                owner = int(owner_text) if owner_text else None
            except (OSError, ValueError):
                owner = None
            if owner is not None and not process_alive(owner):
                try:
                    lock_file.unlink()
                    continue
                except (FileNotFoundError, PermissionError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {lock_file}; remove it only when no P17 process is running"
                )
            time.sleep(0.2)
    try:
        yield
    finally:
        # Antivirus/indexing software can briefly hold a just-closed file on
        # Windows. Retry release instead of failing a completed dataset load.
        for attempt in range(20):
            try:
                lock_file.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)


def remove_legacy_lock() -> None:
    """Remove the directory lock used by the first P17 revision."""
    legacy = DATA / ".p17_loader_lock"
    if legacy.exists():
        for attempt in range(20):
            try:
                shutil.rmtree(legacy)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)


def worker_initializer() -> None:
    """Keep Windows console Ctrl+C events out of worker processes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def guarded_parent_interrupts(double_press_seconds: float = 2.0):
    """Ignore one stray Ctrl+C; require a quick second press to stop."""
    previous_handler = signal.getsignal(signal.SIGINT)
    last_interrupt = 0.0

    def handler(signum, frame):  # noqa: ARG001
        nonlocal last_interrupt
        now = time.monotonic()
        if now - last_interrupt <= double_press_seconds:
            signal.signal(signal.SIGINT, previous_handler)
            raise KeyboardInterrupt
        last_interrupt = now
        print(
            "\nP17 received a console interrupt. It was ignored to protect "
            "the running jobs. Press Ctrl+C again within two seconds to stop.",
            flush=True,
        )

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def prepare_data() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    remove_legacy_lock()
    with dataset_lock():
        ensure_fashion_mnist(DATA)
    print("Fashion-MNIST is ready")


def metadata(point: Point, seed_index: int, source: str) -> dict[str, object]:
    partition_seed = PARTITION_SEEDS[seed_index]
    return {
        "point_id": point.point_id,
        "benchmark": f"fmnist_{point.architecture}",
        "architecture": point.architecture,
        "comparison": "quality-selected" if point.comparison == "quality" else "traffic-matched",
        "method": point.method,
        "configuration": point.configuration,
        "configuration_value": point.value,
        "seed_index": seed_index,
        "partition_seed": partition_seed,
        "train_seed": train_seed(partition_seed),
        "rounds": point.rounds,
        "source_campaign": source,
    }


def import_p16_seed_zero(point: Point) -> bool:
    """Reuse P16 seed zero only for its five quality-selected operating points."""
    if point.comparison != "quality" or complete(point, 0):
        return complete(point, 0)
    source_stem = f"{point.architecture}_{point.method}_p2500_r{point.rounds}"
    source_summary = P16 / f"{source_stem}_summary.csv"
    source_history = P16 / f"{source_stem}_history.csv"
    if not source_summary.exists() or not source_history.exists():
        return False
    summary = pd.read_csv(source_summary)
    history = pd.read_csv(source_history)
    info = metadata(point, 0, "p16-import")
    for key, value in info.items():
        summary[key] = value
        history[key] = value
    target_summary, target_history = output_paths(point, 0)
    atomic_csv(summary, target_summary)
    atomic_csv(history, target_history)
    print(f"imported P16 seed zero: {point.point_id}")
    return True


def run_point(point_id: str, seed_index: int, force: bool = False) -> str:
    point = POINT_BY_ID[point_id]
    if complete(point, seed_index) and not force:
        return f"skip {point_id} seed={seed_index}"
    if seed_index == 0 and not force and import_p16_seed_zero(point):
        return f"reuse {point_id} seed=0"
    print(f"launch {point_id} seed={seed_index}", flush=True)
    partition_seed = PARTITION_SEEDS[seed_index]
    DATA.mkdir(parents=True, exist_ok=True)
    with dataset_lock():
        federation = make_multiclass_federation(
            root=DATA, regime="strong", seed=partition_seed
        )
    config = replace(
        build_config(point.architecture, point.method, point.value),
        rounds=point.rounds,
        eval_stride=point.eval_stride,
    )
    started = time.monotonic()
    result = run_final_baseline(
        federation=federation,
        architecture=point.architecture,
        method=point.method,
        config=config,
        seed=train_seed(partition_seed),
        record_history=True,
    )
    history = result.pop("history")
    if not isinstance(history, pd.DataFrame):
        raise TypeError("run_final_baseline did not return a history DataFrame")
    info = metadata(point, seed_index, "p17")
    result.update(info)
    result["elapsed_seconds"] = time.monotonic() - started
    for key, value in info.items():
        history[key] = value
    summary_path, history_path = output_paths(point, seed_index)
    atomic_csv(pd.DataFrame([result]), summary_path)
    atomic_csv(history, history_path)
    return (
        f"done {point_id} seed={seed_index} round={point.rounds} "
        f"test_acc={100 * float(result['final_test_accuracy']):.2f}%"
    )


def jobs(architecture: str, start_seed: int, end_seed: int, force: bool):
    return [
        (point.point_id, seed_index, force)
        for seed_index in range(start_seed, end_seed + 1)
        for point in architecture_points(architecture)
        if force or not complete(point, seed_index)
    ]


def campaign(architecture: str, workers: int, start_seed: int, end_seed: int,
             force: bool = False) -> None:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if start_seed > end_seed:
        raise ValueError("start seed must not exceed end seed")
    # This runs once in the parent process before any Windows workers spawn.
    remove_legacy_lock()
    pending = jobs(architecture, start_seed, end_seed, force)
    total = len(architecture_points(architecture)) * (end_seed - start_seed + 1)
    print(f"P17 {architecture}: {total-len(pending)}/{total} complete, "
          f"{len(pending)} pending, workers={workers}")
    if workers == 1:
        for arguments in pending:
            print(run_point(*arguments), flush=True)
        return
    with guarded_parent_interrupts():
        with ProcessPoolExecutor(
            max_workers=workers, initializer=worker_initializer
        ) as executor:
            future_to_job = {
                executor.submit(run_point, *arguments): arguments for arguments in pending
            }
            for future in as_completed(future_to_job):
                print(future.result(), flush=True)


def load_architecture(architecture: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for point in architecture_points(architecture)
        for seed_index in range(10)
        for path in output_paths(point, seed_index)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"{architecture} incomplete: {len(missing)} files missing")
    summaries = []
    histories = []
    for point in architecture_points(architecture):
        for seed_index in range(10):
            summary_path, history_path = output_paths(point, seed_index)
            summaries.append(pd.read_csv(summary_path))
            histories.append(pd.read_csv(history_path))
    runs = pd.concat(summaries, ignore_index=True, sort=False)
    history = pd.concat(histories, ignore_index=True, sort=False)
    expected_rows = len(architecture_points(architecture)) * 10
    if len(runs) != expected_rows or runs.duplicated(["point_id", "seed_index"]).any():
        raise RuntimeError(f"invalid {architecture} run grid")
    return runs, history


def aggregate(architecture: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, history = load_architecture(architecture)
    metrics = [metric for metric in REQUIRED_METRICS if metric in runs.columns]
    summary = (
        runs.groupby(
            ["point_id", "benchmark", "comparison", "method", "configuration"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed_index", "size"),
            partition_seeds=("partition_seed", lambda x: ";".join(str(int(v)) for v in sorted(x))),
            training_seeds=("train_seed", lambda x: ";".join(str(int(v)) for v in sorted(x))),
            **{f"{metric}_{stat}": (metric, stat) for metric in metrics for stat in ("mean", "std")},
        )
        .sort_values(["comparison", "method", "point_id"])
    )
    directory = OUT / architecture
    atomic_csv(runs, directory / "runs.csv")
    atomic_csv(history, directory / "histories.csv")
    atomic_csv(summary, directory / "aggregate.csv")
    print(summary[["point_id", "n_seeds", "final_test_accuracy_mean",
                   "unicast_hybrid_total_bits_mean"]].to_string(index=False))
    return runs, history


def finalize() -> None:
    collections = [aggregate(architecture) for architecture in ("mlp", "cnn")]
    runs = pd.concat([item[0] for item in collections], ignore_index=True, sort=False)
    histories = pd.concat([item[1] for item in collections], ignore_index=True, sort=False)
    aggregates = pd.concat(
        [pd.read_csv(OUT / architecture / "aggregate.csv") for architecture in ("mlp", "cnn")],
        ignore_index=True, sort=False,
    )
    atomic_csv(runs, OUT / "runs.csv")
    atomic_csv(histories, OUT / "histories.csv")
    atomic_csv(aggregates, OUT / "aggregate.csv")
    protocol(write=True)
    print(f"finalized {len(runs)} runs and {len(histories)} history rows")


def verify() -> None:
    problems: list[str] = []
    for architecture in ("mlp", "cnn"):
        try:
            runs, histories = load_architecture(architecture)
        except RuntimeError as error:
            problems.append(str(error))
            continue
        expected_round = 1500 if architecture == "mlp" else 1800
        if set(runs["rounds"].astype(int)) != {expected_round}:
            problems.append(f"{architecture}: unexpected summary horizon")
        maxima = histories.groupby(["point_id", "seed_index"])["round"].max()
        if set(maxima.astype(int)) != {expected_round}:
            problems.append(f"{architecture}: one or more histories stop before round {expected_round}")
        for metric in REQUIRED_METRICS:
            if metric not in runs or not np.isfinite(pd.to_numeric(runs[metric], errors="coerce")).all():
                problems.append(f"{architecture}: invalid required metric {metric}")
    if problems:
        raise RuntimeError("\n".join(problems))
    print("P17 verification passed: 140 matched runs, complete histories, finite metrics")


def status() -> None:
    for architecture in ("mlp", "cnn"):
        points = architecture_points(architecture)
        complete_count = sum(complete(point, seed) for point in points for seed in range(10))
        print(f"{architecture}: {complete_count}/{len(points) * 10} runs complete")
        for point in points:
            seeds = [seed for seed in range(10) if complete(point, seed)]
            print(f"  {point.point_id}: {len(seeds)}/10 {seeds}")


def protocol(write: bool = False) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "purpose": "ten matched seeds at P16-audited final horizons",
        "horizons": {"mlp": 1500, "cnn": 1800},
        "seed_indices": list(range(10)),
        "partition_seeds": list(PARTITION_SEEDS),
        "training_seed_rule": "70000 + partition_seed",
        "selection": "all operating points frozen before P17; no P17 tuning",
        "seed_zero_reuse": "P16 quality-selected seed zero is imported when available",
        "points": [asdict(point) | {"point_id": point.point_id} for point in POINTS],
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    sub.add_parser("protocol")
    sub.add_parser("status")
    sub.add_parser("verify")
    sub.add_parser("finalize")
    point = sub.add_parser("point")
    point.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point.add_argument("--seed-index", type=int, choices=range(10), required=True)
    point.add_argument("--force", action="store_true")
    campaign_parser = sub.add_parser("campaign")
    campaign_parser.add_argument("--architecture", choices=("mlp", "cnn"), required=True)
    campaign_parser.add_argument("--workers", type=int, default=1)
    campaign_parser.add_argument("--start-seed", type=int, choices=range(10), default=0)
    campaign_parser.add_argument("--end-seed", type=int, choices=range(10), default=9)
    campaign_parser.add_argument("--force", action="store_true")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--architecture", choices=("mlp", "cnn"), required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-data":
        prepare_data()
    elif args.command == "protocol":
        protocol()
    elif args.command == "status":
        status()
    elif args.command == "verify":
        verify()
    elif args.command == "finalize":
        finalize()
    elif args.command == "point":
        print(run_point(args.point_id, args.seed_index, args.force))
    elif args.command == "campaign":
        campaign(args.architecture, args.workers, args.start_seed, args.end_seed, args.force)
    elif args.command == "aggregate":
        aggregate(args.architecture)


if __name__ == "__main__":
    main()
