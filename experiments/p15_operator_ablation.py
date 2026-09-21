"""Development-selected Event-FedAvg operator ablations on CIFAR-10 CNN.

This campaign is intentionally separate from the frozen P8 audit. It isolates
persistent versus memoryless state, full versus subtractive reset, and an
independently scheduled versus coupled server quantum. Candidate variants are
selected on partition 2400, then evaluated on the ten P12/P14 held-out seed
pairs. Existing outputs are skipped unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig, run_final_baseline
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "p15_operator_ablation"
DEVELOPMENT_SEED = 2400
HELDOUT_SEEDS = tuple(range(2500, 3500, 100))

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

THRESHOLDS = (0.0025, 0.005, 0.01, 0.025)
CONFIGS: dict[str, tuple[str, FinalBaselineConfig]] = {
    "frozen_full_persistent": ("frozen", BASE),
}
for threshold in THRESHOLDS:
    suffix = str(threshold).replace(".", "p")
    CONFIGS[f"memoryless_t{suffix}"] = (
        "memoryless",
        replace(BASE, threshold=threshold, event_memory="memoryless"),
    )
    CONFIGS[f"subtractive_t{suffix}"] = (
        "subtractive",
        replace(BASE, threshold=threshold, event_reset="subtractive"),
    )
    CONFIGS[f"coupled_t{suffix}"] = (
        "coupled",
        replace(BASE, threshold=threshold, jump0=threshold, jump_exponent=0.0),
    )


def seed_pair(partition_seed: int) -> tuple[int, int]:
    return partition_seed, 70000 + partition_seed


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
    print("P15 operator-ablation smoke checks passed")


def develop(*, force: bool = False) -> None:
    for config_name in CONFIGS:
        run_point(config_name, DEVELOPMENT_SEED, "dev", force=force)


def select() -> Path:
    files = sorted((OUT / "points").glob(f"*_p{DEVELOPMENT_SEED}_dev.csv"))
    if len(files) != len(CONFIGS):
        raise RuntimeError(f"development grid incomplete: {len(files)}/{len(CONFIGS)}")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    selected = {"frozen": "frozen_full_persistent"}
    for family in ("memoryless", "subtractive", "coupled"):
        candidates = data[(data.family == family) & np.isfinite(data.final_train_ce)]
        if candidates.empty:
            raise RuntimeError(f"no finite development candidate for {family}")
        selected[family] = str(
            candidates.sort_values(["final_train_ce", "config_name"]).iloc[0].config_name
        )
    protocol = {
        "development_seed": DEVELOPMENT_SEED,
        "selection_metric": "minimum final_train_ce within each ablation family",
        "heldout_seeds": list(HELDOUT_SEEDS),
        "selected": selected,
        "base_config": asdict(BASE),
        "configuration_frozen_before_heldout_evaluation": True,
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
        for config_name in selection["selected"].values():
            run_point(config_name, partition_seed, "heldout", force=force)


def aggregate() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    selected = set(selection["selected"].values())
    files = sorted((OUT / "points").glob("*_heldout.csv"))
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    data = data[data.config_name.isin(selected)].copy()
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
    print(summary.to_string(index=False))


def status() -> None:
    for tag, seeds in (("dev", (DEVELOPMENT_SEED,)), ("heldout", HELDOUT_SEEDS)):
        count = len(list((OUT / "points").glob(f"*_{tag}.csv")))
        expected = len(CONFIGS) if tag == "dev" else 4 * len(seeds)
        print(f"{tag}: {count}/{expected} point files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("smoke", "develop", "select", "heldout", "aggregate", "status")
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
    else:
        status()


if __name__ == "__main__":
    main()
