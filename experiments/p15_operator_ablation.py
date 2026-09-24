"""Matched-traffic Event-FedAvg operator ablations on CIFAR-10 CNN.

This campaign is intentionally separate from the frozen P8 audit. It isolates
persistent versus memoryless state and full versus subtractive reset. Candidate
thresholds are selected on partition 2400 by proximity to the frozen
Event-FedAvg traffic, then evaluated on the ten P12/P14 held-out seed pairs.
The authoritative frozen Event-FedAvg held-out runs are reused rather than
retrained. Existing outputs are skipped unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig, run_final_baseline
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "p15_operator_ablation"
P12_RUNS = (
    REPO
    / "experiments"
    / "results"
    / "p12_headline_ten_seed"
    / "heldout_runs.csv"
)
DEVELOPMENT_SEED = 2400
HELDOUT_SEEDS = tuple(range(3500, 4500, 100))

BASE = FinalBaselineConfig(
    local_steps=5,
    local_lr=0.05,
    batch_size=32,
    regularization=5e-4,
    rounds=120,
    eval_stride=30,
    rho=0.999,
    threshold=0.025,
    jump0=0.005,
    jump_scale=100.0,
    jump_exponent=0.2,
    init_scale=1.0,
)

MEMORYLESS_THRESHOLDS = (0.0025, 0.005, 0.01, 0.025)
SUBTRACTIVE_THRESHOLDS = (0.0125, 0.025, 0.05, 0.1)
CONFIGS: dict[str, tuple[str, FinalBaselineConfig]] = {
    "frozen_full_persistent": ("frozen", BASE),
}
for threshold in MEMORYLESS_THRESHOLDS:
    suffix = str(threshold).replace(".", "p")
    CONFIGS[f"memoryless_t{suffix}"] = (
        "memoryless",
        replace(BASE, threshold=threshold, event_memory="memoryless"),
    )
for threshold in SUBTRACTIVE_THRESHOLDS:
    suffix = str(threshold).replace(".", "p")
    CONFIGS[f"subtractive_t{suffix}"] = (
        "subtractive",
        replace(BASE, threshold=threshold, event_reset="subtractive"),
    )


def seed_pair(partition_seed: int) -> tuple[int, int]:
    return partition_seed, 80000 + partition_seed


def output_path(config_name: str, partition_seed: int, tag: str) -> Path:
    return OUT / "points" / f"{config_name}_p{partition_seed}_{tag}.csv"


def run_point(
    config_name: str, partition_seed: int, tag: str, *, force: bool = False
) -> Path:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown configuration: {config_name}")
    path = output_path(config_name, partition_seed, tag)
    if path.exists() and not force:
        print(f"skip existing {path}")
        return path
    family, config = CONFIGS[config_name]
    _, train_seed = seed_pair(partition_seed)
    federation = make_cifar10_federation(
        root=DATA, regime="strong", seed=partition_seed
    )
    result = run_final_baseline(
        federation=federation,
        architecture="cifar_cnn",
        method="event",
        config=config,
        seed=train_seed,
    )
    result.update(
        config_name=config_name,
        family=family,
        partition_seed=partition_seed,
        train_seed=train_seed,
        tag=tag,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(path, index=False, float_format="%.10g")
    print(
        pd.DataFrame([result])[
            [
                "config_name",
                "final_train_ce",
                "final_test_accuracy",
                "unicast_hybrid_total_bits",
            ]
        ].to_string(index=False)
    )
    return path


def smoke() -> None:
    rng = np.random.default_rng(1500)
    client_x = tuple(
        rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8)
        for _ in range(10)
    )
    client_y = tuple(np.tile(np.arange(10, dtype=np.int64), 2) for _ in range(10))
    federation = MulticlassFederation(
        client_X=client_x,
        client_y=client_y,
        X_train_eval=np.concatenate(client_x)[:100],
        y_train_eval=np.concatenate(client_y)[:100],
        X_test=rng.integers(0, 256, size=(100, 3, 32, 32), dtype=np.uint8),
        y_test=np.tile(np.arange(10, dtype=np.int64), 10),
        regime="synthetic",
        client_class_counts=np.full((10, 10), 2, dtype=int),
        periods=np.ones(10, dtype=int),
        weights=np.full(10, 0.1),
    )
    probe = replace(
        BASE,
        rounds=2,
        local_steps=1,
        batch_size=8,
        eval_stride=1,
        local_lr=0.01,
        regularization=0.0,
        init_scale=0.3,
        threshold=1e-4,
    )
    outputs = {}
    for name, config in {
        "frozen": probe,
        "memoryless": replace(probe, event_memory="memoryless"),
        "subtractive": replace(probe, event_reset="subtractive"),
    }.items():
        outputs[name] = run_final_baseline(
            federation=federation,
            architecture="cifar_cnn",
            method="event",
            config=config,
            seed=1515,
        )
        if not np.isfinite(float(outputs[name]["final_test_ce"])):
            raise AssertionError(f"non-finite smoke result for {name}")
    if outputs["frozen"]["event_memory"] != "persistent":
        raise AssertionError("default event memory changed")
    if outputs["frozen"]["event_reset"] != "full":
        raise AssertionError("default event reset changed")
    if len(CONFIGS) != 9:
        raise AssertionError("P15 development grid must contain 1+4+4 points")
    if HELDOUT_SEEDS != tuple(range(3500, 4500, 100)):
        raise AssertionError("P15 CIFAR held-out partitions drifted from P12")
    if [seed_pair(seed)[1] for seed in HELDOUT_SEEDS] != list(
        range(83500, 84500, 100)
    ):
        raise AssertionError("P15 CIFAR training seeds drifted from P12")
    print("P15 operator-ablation smoke checks passed")


def develop(*, force: bool = False) -> None:
    for config_name in CONFIGS:
        run_point(config_name, DEVELOPMENT_SEED, "dev", force=force)


def select() -> Path:
    files = sorted((OUT / "points").glob(f"*_p{DEVELOPMENT_SEED}_dev.csv"))
    if len(files) != len(CONFIGS):
        raise RuntimeError(f"development grid incomplete: {len(files)}/{len(CONFIGS)}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    reference = data[data.config_name == "frozen_full_persistent"]
    if len(reference) != 1:
        raise RuntimeError("missing unique frozen development reference")
    reference_bits = float(reference.iloc[0].unicast_hybrid_total_bits)
    if not math.isfinite(reference_bits) or reference_bits <= 0:
        raise RuntimeError("invalid frozen development traffic")

    data["traffic_log_distance"] = (
        data.unicast_hybrid_total_bits.astype(float).map(math.log)
        - math.log(reference_bits)
    ).abs()
    selected = {"frozen": "frozen_full_persistent"}
    for family in ("memoryless", "subtractive"):
        candidates = data[(data.family == family) & np.isfinite(data.final_train_ce)]
        if candidates.empty:
            raise RuntimeError(f"no finite development candidate for {family}")
        selected[family] = str(
            candidates.sort_values(
                ["traffic_log_distance", "final_train_ce", "config_name"]
            ).iloc[0].config_name
        )
    protocol = {
        "development_seed": DEVELOPMENT_SEED,
        "development_train_seed": seed_pair(DEVELOPMENT_SEED)[1],
        "selection_metric": (
            "minimum absolute log-traffic distance to the frozen Event-FedAvg "
            "development run within each ablation family; final_train_ce breaks ties"
        ),
        "reference_unicast_hybrid_total_bits": reference_bits,
        "heldout_seeds": list(HELDOUT_SEEDS),
        "heldout_train_seeds": [seed_pair(seed)[1] for seed in HELDOUT_SEEDS],
        "selected": selected,
        "base_config": asdict(BASE),
        "configuration_frozen_before_heldout_evaluation": True,
        "frozen_heldout_source": str(P12_RUNS.relative_to(REPO)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selection.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    data.sort_values(["family", "final_train_ce", "config_name"]).to_csv(
        OUT / "development_summary.csv", index=False, float_format="%.10g"
    )
    print(json.dumps(protocol, indent=2))
    return path


def heldout(*, force: bool = False) -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    for partition_seed in HELDOUT_SEEDS:
        for family in ("memoryless", "subtractive"):
            config_name = selection["selected"][family]
            run_point(config_name, partition_seed, "heldout", force=force)


def frozen_heldout_runs() -> pd.DataFrame:
    if not P12_RUNS.exists():
        raise RuntimeError(f"missing authoritative P12 runs: {P12_RUNS}")
    data = pd.read_csv(P12_RUNS)
    data = data[data.point_id == "cifar_event"].copy()
    if len(data) != len(HELDOUT_SEEDS):
        raise RuntimeError(
            f"expected {len(HELDOUT_SEEDS)} frozen CIFAR event runs, found {len(data)}"
        )
    expected = {seed_pair(seed) for seed in HELDOUT_SEEDS}
    observed = {
        (int(row.partition_seed), int(row.train_seed)) for row in data.itertuples()
    }
    if observed != expected:
        raise RuntimeError("P12 frozen Event-FedAvg seed mapping drift")
    checks = {
        "rho": BASE.rho,
        "threshold": BASE.threshold,
        "jump0": BASE.jump0,
        "jump_exponent": BASE.jump_exponent,
        "local_steps": BASE.local_steps,
        "local_lr": BASE.local_lr,
        "rounds": BASE.rounds,
    }
    for field, expected_value in checks.items():
        values = data[field].astype(float).to_numpy()
        if not np.allclose(values, float(expected_value), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"P12 frozen configuration drift for {field}")
    data["config_name"] = "frozen_full_persistent"
    data["family"] = "frozen"
    data["tag"] = "heldout"
    return data


def aggregate() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    selected = set(selection["selected"].values())
    files = sorted((OUT / "points").glob("*_heldout.csv"))
    if not files:
        raise RuntimeError("no P15 held-out variant files found")
    variants = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    variants = variants[variants.config_name.isin(selected - {"frozen_full_persistent"})]
    data = pd.concat([frozen_heldout_runs(), variants], ignore_index=True, sort=False)
    expected = {(name, seed) for name in selected for seed in HELDOUT_SEEDS}
    observed = {(str(row.config_name), int(row.partition_seed)) for row in data.itertuples()}
    if observed != expected or len(data) != len(expected):
        raise RuntimeError(
            f"held-out grid mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    metrics = (
        "final_train_ce",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "coordinate_events",
        "broadcast_total_bits",
        "unicast_hybrid_total_bits",
    )
    summary = (
        data.groupby(["family", "config_name"], as_index=False)
        .agg(
            n_seeds=("partition_seed", "size"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in metrics
                for stat in ("mean", "std")
            },
        )
        .sort_values(["family", "config_name"])
    )
    data.to_csv(OUT / "heldout_runs.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.10g")
    paired_rows = []
    reference = data[data.family == "frozen"].set_index("partition_seed").sort_index()
    for family in ("memoryless", "subtractive"):
        comparison = (
            data[data.family == family].set_index("partition_seed").sort_index()
        )
        if list(reference.index) != list(comparison.index):
            raise RuntimeError(f"paired seed mismatch for {family}")
        accuracy_difference = (
            comparison.final_test_accuracy.astype(float)
            - reference.final_test_accuracy.astype(float)
        ).to_numpy()
        traffic_ratio = (
            comparison.unicast_hybrid_total_bits.astype(float)
            / reference.unicast_hybrid_total_bits.astype(float)
        ).to_numpy()
        mean_difference = float(np.mean(accuracy_difference))
        std_difference = float(np.std(accuracy_difference, ddof=1))
        half_width = 2.262157 * std_difference / math.sqrt(len(accuracy_difference))
        paired_rows.append(
            {
                "comparison": f"{family}_minus_frozen",
                "n_pairs": len(accuracy_difference),
                "mean_accuracy_difference_points": 100.0 * mean_difference,
                "std_accuracy_difference_points": 100.0 * std_difference,
                "ci95_low_points": 100.0 * (mean_difference - half_width),
                "ci95_high_points": 100.0 * (mean_difference + half_width),
                "variant_wins": int(np.sum(accuracy_difference > 0)),
                "mean_paired_traffic_ratio": float(np.mean(traffic_ratio)),
            }
        )
    pd.DataFrame(paired_rows).to_csv(
        OUT / "paired.csv", index=False, float_format="%.10g"
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(paired_rows).to_string(index=False))


def status() -> None:
    for tag, seeds in (("dev", (DEVELOPMENT_SEED,)), ("heldout", HELDOUT_SEEDS)):
        count = len(list((OUT / "points").glob(f"*_{tag}.csv")))
        expected = len(CONFIGS) if tag == "dev" else 2 * len(seeds)
        print(f"{tag}: {count}/{expected} point files")


def campaign(*, force: bool = False) -> None:
    develop(force=force)
    select()
    heldout(force=force)
    aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "develop",
            "select",
            "heldout",
            "aggregate",
            "campaign",
            "status",
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "develop":
        develop(force=args.force)
    elif args.command == "select":
        select()
    elif args.command == "heldout":
        heldout(force=args.force)
    elif args.command == "aggregate":
        aggregate()
    elif args.command == "campaign":
        campaign(force=args.force)
    else:
        status()


if __name__ == "__main__":
    main()
