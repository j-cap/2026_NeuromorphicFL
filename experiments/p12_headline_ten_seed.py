from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from final_baseline_point import build_config
from neuromorphicfl.cifar10_benchmark import (
    download_cifar10,
    make_cifar10_federation,
)
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from neuromorphicfl.fmnist_event_benchmark import ensure_fashion_mnist
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from p3_cifar10_campaign import CONFIGS as CIFAR_CONFIGS
from p8_targeted_revision import CONFIGS as P8_CONFIGS


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "experiments" / "results" / "p12_headline_ten_seed"

FMNIST_PARTITION_SEEDS = tuple(range(2500, 3500, 100))
CIFAR_PARTITION_SEEDS = tuple(range(3500, 4500, 100))
ORIGINAL_SEED_INDICES = (0, 1, 2)
EXTENSION_SEED_INDICES = tuple(range(3, 10))

Metric = Literal["accuracy", "worst_class_accuracy"]


@dataclass(frozen=True)
class Point:
    point_id: str
    benchmark: str
    role: str
    method: str
    configuration: str


POINTS = (
    Point("fmnist_mlp_event", "fmnist_mlp", "event", "event", "fixed"),
    Point(
        "fmnist_mlp_ef_quality",
        "fmnist_mlp",
        "quality",
        "ef_topk",
        "0.05",
    ),
    Point(
        "fmnist_mlp_strom_near",
        "fmnist_mlp",
        "traffic",
        "strom",
        "0.02",
    ),
    Point("fmnist_cnn_event", "fmnist_cnn", "event", "event", "fixed"),
    Point(
        "fmnist_cnn_strom_quality",
        "fmnist_cnn",
        "quality",
        "strom",
        "0.005",
    ),
    Point(
        "fmnist_cnn_strom_near",
        "fmnist_cnn",
        "traffic",
        "strom",
        "0.02",
    ),
    Point("cifar_event", "cifar_cnn", "event", "event", "event_t025_q005"),
    Point(
        "cifar_strom_quality",
        "cifar_cnn",
        "quality",
        "strom",
        "strom_t0025",
    ),
    Point(
        "cifar_strom_near",
        "cifar_cnn",
        "traffic",
        "strom",
        "strom_t01",
    ),
    Point(
        "cifar_ef_quality",
        "cifar_cnn",
        "qualification",
        "ef_topk",
        "ef_k05",
    ),
    Point(
        "cifar_dense_gain2",
        "cifar_cnn",
        "dense_reference",
        "dense",
        "dense_gain_2p0",
    ),
)
POINT_BY_ID = {point.point_id: point for point in POINTS}

COMPARISONS: tuple[tuple[str, str, str, Metric], ...] = (
    (
        "fmnist_mlp_event_minus_quality_accuracy",
        "fmnist_mlp_event",
        "fmnist_mlp_ef_quality",
        "accuracy",
    ),
    (
        "fmnist_mlp_event_minus_near_accuracy",
        "fmnist_mlp_event",
        "fmnist_mlp_strom_near",
        "accuracy",
    ),
    (
        "fmnist_cnn_event_minus_quality_accuracy",
        "fmnist_cnn_event",
        "fmnist_cnn_strom_quality",
        "accuracy",
    ),
    (
        "fmnist_cnn_event_minus_near_accuracy",
        "fmnist_cnn_event",
        "fmnist_cnn_strom_near",
        "accuracy",
    ),
    (
        "cifar_event_minus_quality_accuracy",
        "cifar_event",
        "cifar_strom_quality",
        "accuracy",
    ),
    (
        "cifar_event_minus_near_accuracy",
        "cifar_event",
        "cifar_strom_near",
        "accuracy",
    ),
    (
        "cifar_event_minus_dense_accuracy",
        "cifar_event",
        "cifar_dense_gain2",
        "accuracy",
    ),
    (
        "cifar_event_minus_ef_worst_class",
        "cifar_event",
        "cifar_ef_quality",
        "worst_class_accuracy",
    ),
)

METRICS = (
    "final_train_objective",
    "final_test_ce",
    "final_test_accuracy",
    "final_worst_class_accuracy",
    "uplink_packetized_bits",
    "broadcast_total_bits",
    "unicast_hybrid_total_bits",
    "coordinate_events",
)


def seed_pair(benchmark: str, seed_index: int) -> tuple[int, int]:
    if not 0 <= seed_index < 10:
        raise ValueError("seed index must be in [0, 9]")
    if benchmark.startswith("fmnist_"):
        partition_seed = FMNIST_PARTITION_SEEDS[seed_index]
        return partition_seed, 70000 + partition_seed
    if benchmark == "cifar_cnn":
        partition_seed = CIFAR_PARTITION_SEEDS[seed_index]
        return partition_seed, 80000 + partition_seed
    raise ValueError(f"unknown benchmark: {benchmark}")


def prepare_data(dataset: str) -> None:
    if dataset == "fmnist":
        ensure_fashion_mnist(DATA / "fashion-mnist")
    elif dataset == "cifar10":
        download_cifar10(DATA / "cifar-10")
    else:
        raise ValueError(dataset)
    print(f"prepared {dataset}")


def run_point(point_id: str, seed_index: int) -> Path:
    point = POINT_BY_ID[point_id]
    partition_seed, train_seed = seed_pair(point.benchmark, seed_index)

    if point.benchmark.startswith("fmnist_"):
        architecture = point.benchmark.removeprefix("fmnist_")
        value = None if point.configuration == "fixed" else float(point.configuration)
        config = build_config(architecture, point.method, value)
        federation = make_multiclass_federation(
            root=DATA / "fashion-mnist", regime="strong", seed=partition_seed
        )
        result = run_final_baseline(
            federation=federation,
            architecture=architecture,
            method=point.method,
            config=config,
            seed=train_seed,
        )
    else:
        if point.configuration in CIFAR_CONFIGS:
            method, config = CIFAR_CONFIGS[point.configuration]
        else:
            method, config = P8_CONFIGS[point.configuration]
        if method != point.method:
            raise AssertionError(f"method mismatch for {point_id}")
        federation = make_cifar10_federation(
            root=DATA / "cifar-10", regime="strong", seed=partition_seed
        )
        result = run_final_baseline(
            federation=federation,
            architecture="cifar_cnn",
            method=method,
            config=config,
            seed=train_seed,
        )

    result.update(
        {
            "point_id": point.point_id,
            "benchmark": point.benchmark,
            "role": point.role,
            "configuration": point.configuration,
            "seed_index": seed_index,
            "seed_subset": "original" if seed_index in ORIGINAL_SEED_INDICES else "extension",
            "partition_seed": partition_seed,
            "train_seed": train_seed,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{point_id}_s{seed_index}_p{partition_seed}.csv"
    pd.DataFrame([result]).to_csv(output, index=False, float_format="%.10g")
    print(
        pd.DataFrame([result])[
            [
                "point_id",
                "seed_index",
                "partition_seed",
                "train_seed",
                "final_test_accuracy",
                "final_worst_class_accuracy",
                "unicast_hybrid_total_bits",
            ]
        ].to_string(index=False)
    )
    return output


def result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def load_and_validate(root: Path) -> pd.DataFrame:
    files = result_files(root)
    if not files:
        raise RuntimeError(f"no point results found below {root}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    expected = {(point.point_id, seed_index) for point in POINTS for seed_index in range(10)}
    observed = {
        (str(row.point_id), int(row.seed_index)) for row in data.itertuples(index=False)
    }
    if observed != expected or len(data) != len(expected):
        raise RuntimeError(
            f"P12 grid mismatch: rows={len(data)}, missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if data.duplicated(["point_id", "seed_index"]).any():
        raise RuntimeError("duplicate P12 point/seed rows")

    for row in data.itertuples(index=False):
        point = POINT_BY_ID[str(row.point_id)]
        expected_partition, expected_train = seed_pair(point.benchmark, int(row.seed_index))
        if int(row.partition_seed) != expected_partition or int(row.train_seed) != expected_train:
            raise RuntimeError(f"seed mapping drift for {row.point_id}/s{row.seed_index}")
        if str(row.method) != point.method or str(row.configuration) != point.configuration:
            raise RuntimeError(f"configuration drift for {row.point_id}/s{row.seed_index}")
        if not (
            float(row.uplink_packetized_bits)
            < float(row.broadcast_total_bits)
            < float(row.unicast_hybrid_total_bits)
        ):
            raise RuntimeError(f"traffic ordering failed for {row.point_id}/s{row.seed_index}")
    return data.sort_values(["benchmark", "point_id", "seed_index"])


def aggregate_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(
            ["point_id", "benchmark", "role", "method", "configuration"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed_index", "size"),
            partition_seeds=(
                "partition_seed",
                lambda values: ";".join(str(int(value)) for value in values),
            ),
            training_seeds=(
                "train_seed",
                lambda values: ";".join(str(int(value)) for value in values),
            ),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in METRICS
                for stat in ("mean", "std")
            },
        )
        .sort_values(["benchmark", "role", "point_id"])
    )
    if set(summary.n_seeds) != {10}:
        raise RuntimeError("every P12 summary row must contain ten seeds")
    return summary


def paired_interval(differences: np.ndarray) -> dict[str, float | int]:
    n = len(differences)
    if n not in (7, 10):
        raise ValueError(f"unsupported paired sample size: {n}")
    mean = float(np.mean(differences))
    std = float(np.std(differences, ddof=1))
    critical = {7: 2.446912, 10: 2.262157}[n]
    half_width = critical * std / math.sqrt(n)
    return {
        "n": n,
        "mean_difference_points": 100.0 * mean,
        "std_difference_points": 100.0 * std,
        "ci95_low_points": 100.0 * (mean - half_width),
        "ci95_high_points": 100.0 * (mean + half_width),
        "n_positive": int(np.sum(differences > 0)),
        "n_negative": int(np.sum(differences < 0)),
        "n_zero": int(np.sum(differences == 0)),
    }


def paired_analysis(data: pd.DataFrame) -> pd.DataFrame:
    metric_columns = {
        "accuracy": "final_test_accuracy",
        "worst_class_accuracy": "final_worst_class_accuracy",
    }
    rows: list[dict[str, object]] = []
    indexed = data.set_index(["point_id", "seed_index"])
    for comparison_id, event_id, comparator_id, metric in COMPARISONS:
        column = metric_columns[metric]
        all_differences = np.array(
            [
                float(indexed.loc[(event_id, seed_index), column])
                - float(indexed.loc[(comparator_id, seed_index), column])
                for seed_index in range(10)
            ]
        )
        extension_differences = all_differences[list(EXTENSION_SEED_INDICES)]
        for subset, differences in (
            ("all_ten", all_differences),
            ("new_seven", extension_differences),
        ):
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "event_point": event_id,
                    "comparator_point": comparator_id,
                    "metric": metric,
                    "subset": subset,
                    **paired_interval(differences),
                }
            )
    return pd.DataFrame(rows)


def protocol_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "P12 ten-seed extension of frozen headline operating points",
        "original_seed_indices": list(ORIGINAL_SEED_INDICES),
        "extension_seed_indices": list(EXTENSION_SEED_INDICES),
        "fmnist_partition_seeds": list(FMNIST_PARTITION_SEEDS),
        "fmnist_training_seeds": [seed_pair("fmnist_mlp", i)[1] for i in range(10)],
        "cifar_partition_seeds": list(CIFAR_PARTITION_SEEDS),
        "cifar_training_seeds": [seed_pair("cifar_cnn", i)[1] for i in range(10)],
        "points": [asdict(point) for point in POINTS],
        "comparisons": [
            {
                "comparison_id": comparison_id,
                "event_point": event_id,
                "comparator_point": comparator_id,
                "metric": metric,
            }
            for comparison_id, event_id, comparator_id, metric in COMPARISONS
        ],
        "reporting": {
            "table": "mean plus sample standard deviation over all ten paired seeds",
            "paired_uncertainty": "two-sided 95% paired t interval",
            "confirmatory_subset": "new_seven",
            "selection": "all methods and hyperparameters frozen before extension seeds",
        },
    }


def aggregate(root: Path) -> None:
    data = load_and_validate(root)
    summary = aggregate_summary(data)
    paired = paired_analysis(data)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "heldout_runs.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.10g")
    paired.to_csv(OUT / "paired_differences.csv", index=False, float_format="%.10g")
    (OUT / "protocol.json").write_text(
        json.dumps(protocol_manifest(), indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nPaired differences:\n", paired.to_string(index=False))


def validate_protocol() -> None:
    if len(POINTS) != 11 or len(POINT_BY_ID) != len(POINTS):
        raise AssertionError("P12 requires eleven unique frozen points")
    for point in POINTS:
        if point.benchmark.startswith("fmnist_"):
            if point.configuration != "fixed":
                float(point.configuration)
        elif point.configuration not in CIFAR_CONFIGS and point.configuration not in P8_CONFIGS:
            raise AssertionError(f"unknown CIFAR configuration: {point.configuration}")
    expected_fmnist = tuple(range(2500, 3500, 100))
    expected_cifar = tuple(range(3500, 4500, 100))
    if FMNIST_PARTITION_SEEDS != expected_fmnist or CIFAR_PARTITION_SEEDS != expected_cifar:
        raise AssertionError("P12 seed protocol drift")
    print(json.dumps(protocol_manifest(), indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--dataset", choices=("fmnist", "cifar10"), required=True)
    point = sub.add_parser("point")
    point.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point.add_argument("--seed-index", type=int, choices=range(10), required=True)
    aggregate_command = sub.add_parser("aggregate")
    aggregate_command.add_argument("--input-dir", type=Path, required=True)
    sub.add_parser("validate-protocol")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-data":
        prepare_data(args.dataset)
    elif args.command == "point":
        run_point(args.point_id, args.seed_index)
    elif args.command == "aggregate":
        aggregate(args.input_dir)
    elif args.command == "validate-protocol":
        validate_protocol()
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
