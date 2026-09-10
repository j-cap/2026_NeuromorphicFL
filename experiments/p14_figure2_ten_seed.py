"""Complete Figure 2 with ten matched seeds per frozen operating point.

P12 already contains ten-seed results for eleven of the twenty-one unique
Figure 2 operating points.  This campaign runs the remaining ten points in the
same pinned implementation and with the same partition/training seed mapping.
All ten seeds are rerun for those points because the historical three-seed
Fashion-MNIST campaign retained aggregates rather than seed-level artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import pandas as pd

from final_baseline_point import build_config
from neuromorphicfl.cifar10_benchmark import make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from p12_headline_ten_seed import CIFAR_BASE, METRICS, POINTS as P12_POINTS, seed_pair


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "experiments" / "results" / "p14_figure2_ten_seed"
P12_RUNS = REPO / "experiments" / "results" / "p12_headline_ten_seed" / "heldout_runs.csv"


@dataclass(frozen=True)
class Point:
    point_id: str
    benchmark: str
    comparison: str
    method: str
    configuration: str


# These are exactly the Figure 2 points not already included in P12.  The
# quality/traffic labels and numerical settings are frozen by the original
# three-seed campaign; this script performs no new model selection.
MISSING_POINTS = (
    Point("fmnist_mlp_strom_quality", "fmnist_mlp", "quality-selected", "strom", "0.00125"),
    Point("fmnist_mlp_sign_quality", "fmnist_mlp", "quality-selected", "sign_ef", "fixed"),
    Point("fmnist_mlp_dense_quality", "fmnist_mlp", "quality-selected", "dense", "fixed"),
    Point("fmnist_mlp_ef_traffic", "fmnist_mlp", "traffic-matched", "ef_topk", "0.01"),
    Point("fmnist_cnn_ef_quality", "fmnist_cnn", "quality-selected", "ef_topk", "0.05"),
    Point("fmnist_cnn_sign_quality", "fmnist_cnn", "quality-selected", "sign_ef", "fixed"),
    Point("fmnist_cnn_dense_quality", "fmnist_cnn", "quality-selected", "dense", "fixed"),
    Point("fmnist_cnn_ef_traffic", "fmnist_cnn", "traffic-matched", "ef_topk", "0.01"),
    Point("cifar_sign_quality", "cifar_cnn", "quality-selected", "sign_ef", "fixed"),
    Point("cifar_ef_traffic", "cifar_cnn", "traffic-matched", "ef_topk", "0.01"),
)
POINT_BY_ID = {point.point_id: point for point in MISSING_POINTS}


def output_path(point: Point, seed_index: int) -> Path:
    partition_seed, _ = seed_pair(point.benchmark, seed_index)
    return OUT / "points" / f"{point.point_id}_s{seed_index}_p{partition_seed}.csv"


def run_point(point_id: str, seed_index: int, *, force: bool = False) -> Path:
    point = POINT_BY_ID[point_id]
    path = output_path(point, seed_index)
    if path.exists() and not force:
        print(f"skip existing {path}")
        return path

    partition_seed, train_seed = seed_pair(point.benchmark, seed_index)
    if point.benchmark.startswith("fmnist_"):
        architecture = point.benchmark.removeprefix("fmnist_")
        value = None if point.configuration == "fixed" else float(point.configuration)
        config = build_config(architecture, point.method, value)
        federation = make_multiclass_federation(
            root=DATA / "fashion-mnist", regime="strong", seed=partition_seed
        )
    else:
        architecture = "cifar_cnn"
        if point.method == "sign_ef":
            config = CIFAR_BASE
        elif point.method == "ef_topk":
            config = replace(CIFAR_BASE, topk_fraction=float(point.configuration))
        else:
            raise AssertionError(f"unsupported CIFAR point: {point}")
        federation = make_cifar10_federation(
            root=DATA / "cifar-10", regime="strong", seed=partition_seed
        )

    result = run_final_baseline(
        federation=federation,
        architecture=architecture,
        method=point.method,
        config=config,
        seed=train_seed,
    )
    result.update(
        {
            "point_id": point.point_id,
            "benchmark": point.benchmark,
            "comparison": point.comparison,
            "configuration": point.configuration,
            "seed_index": seed_index,
            "partition_seed": partition_seed,
            "train_seed": train_seed,
            "source_campaign": "p14",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False, float_format="%.10g")
    print(
        pd.DataFrame([result])[
            ["point_id", "seed_index", "final_test_accuracy", "unicast_hybrid_total_bits"]
        ].to_string(index=False)
    )
    return path


def run_campaign(start_seed: int, end_seed: int, *, force: bool = False) -> None:
    for seed_index in range(start_seed, end_seed + 1):
        for point in MISSING_POINTS:
            run_point(point.point_id, seed_index, force=force)


def load_new_results() -> pd.DataFrame:
    files = sorted((OUT / "points").glob("*.csv"))
    if not files:
        raise RuntimeError(f"no P14 point files below {OUT / 'points'}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    expected = {(point.point_id, seed) for point in MISSING_POINTS for seed in range(10)}
    observed = {(str(row.point_id), int(row.seed_index)) for row in data.itertuples()}
    if observed != expected or len(data) != len(expected):
        raise RuntimeError(
            f"P14 grid mismatch: rows={len(data)}, missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if data.duplicated(["point_id", "seed_index"]).any():
        raise RuntimeError("duplicate P14 point/seed rows")
    return data


def aggregate() -> None:
    if not P12_RUNS.exists():
        raise RuntimeError(f"missing authoritative P12 runs: {P12_RUNS}")
    p12 = pd.read_csv(P12_RUNS).copy()
    p12["comparison"] = p12["role"].map(
        {"event": "quality-selected", "quality": "quality-selected",
         "traffic": "traffic-matched", "qualification": "quality-selected",
         "dense_reference": "quality-selected"}
    )
    p12["source_campaign"] = "p12"
    new = load_new_results()
    combined = pd.concat([p12, new], ignore_index=True, sort=False)

    expected_rows = (len(P12_POINTS) + len(MISSING_POINTS)) * 10
    if len(combined) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} combined rows, found {len(combined)}")
    if combined.duplicated(["point_id", "seed_index"]).any():
        raise RuntimeError("duplicate combined point/seed rows")
    if set(combined.groupby("point_id").size()) != {10}:
        raise RuntimeError("every Figure 2 point must contain exactly ten seeds")

    summary = (
        combined.groupby(
            ["point_id", "benchmark", "comparison", "method", "configuration"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed_index", "size"),
            partition_seeds=("partition_seed", lambda x: ";".join(str(int(v)) for v in x)),
            training_seeds=("train_seed", lambda x: ";".join(str(int(v)) for v in x)),
            **{f"{metric}_{stat}": (metric, stat) for metric in METRICS for stat in ("mean", "std")},
        )
        .sort_values(["benchmark", "comparison", "method", "point_id"])
    )
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "heldout_runs.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.10g")
    print(summary[["benchmark", "comparison", "method", "configuration", "n_seeds",
                   "final_test_accuracy_mean", "unicast_hybrid_total_bits_mean"]].to_string(index=False))


def validate_protocol() -> None:
    if len(MISSING_POINTS) != 10 or len(POINT_BY_ID) != 10:
        raise AssertionError("P14 requires ten unique missing Figure 2 points")
    overlap = set(POINT_BY_ID).intersection(point.point_id for point in P12_POINTS)
    if overlap:
        raise AssertionError(f"P14 must not duplicate P12 point IDs: {sorted(overlap)}")
    manifest = {
        "schema_version": 1,
        "purpose": "complete every frozen Figure 2 operating point with ten matched seeds",
        "reused_p12_points": [point.point_id for point in P12_POINTS],
        "rerun_points": [asdict(point) for point in MISSING_POINTS],
        "new_runs": len(MISSING_POINTS) * 10,
        "combined_runs": (len(P12_POINTS) + len(MISSING_POINTS)) * 10,
        "seed_mapping": "identical to P12",
        "selection": "all points and hyperparameters frozen by the original Figure 2 campaign",
    }
    print(json.dumps(manifest, indent=2))


def status() -> None:
    expected = {
        output_path(point, seed_index)
        for point in MISSING_POINTS
        for seed_index in range(10)
    }
    completed = {path for path in expected if path.exists()}
    print(f"P14 complete: {len(completed)}/{len(expected)} runs")
    for point in MISSING_POINTS:
        seeds = [seed for seed in range(10) if output_path(point, seed).exists()]
        print(f"  {point.point_id}: {len(seeds)}/10 seeds {seeds}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    point = sub.add_parser("point")
    point.add_argument("--point-id", choices=sorted(POINT_BY_ID), required=True)
    point.add_argument("--seed-index", type=int, choices=range(10), required=True)
    point.add_argument("--force", action="store_true")
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--start-seed", type=int, choices=range(10), default=0)
    campaign.add_argument("--end-seed", type=int, choices=range(10), default=9)
    campaign.add_argument("--force", action="store_true")
    sub.add_parser("aggregate")
    sub.add_parser("validate-protocol")
    sub.add_parser("status")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "point":
        run_point(args.point_id, args.seed_index, force=args.force)
    elif args.command == "campaign":
        if args.start_seed > args.end_seed:
            raise ValueError("start seed must not exceed end seed")
        run_campaign(args.start_seed, args.end_seed, force=args.force)
    elif args.command == "aggregate":
        aggregate()
    elif args.command == "validate-protocol":
        validate_protocol()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
