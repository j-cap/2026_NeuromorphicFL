from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import LAYOUT, make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig,
    run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation


DATA = Path("data/cifar-10")
OUT = Path("experiments/results/p8_targeted_revision")
DEVELOPMENT_SEED = 3400
HELDOUT_SEEDS = (3500, 3600, 3700)

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
    topk_fraction=0.01,
    strom_threshold=0.01,
    init_scale=1.0,
    server_gain=1.0,
)

DENSE_GAINS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
DENSE_CONFIGS = {
    f"dense_gain_{str(gain).replace('.', 'p')}":
    ("dense", replace(BASE, server_gain=gain))
    for gain in DENSE_GAINS
}
MECHANISM_CONFIGS = {
    "event_frozen": ("event", BASE),
    "event_no_leak": ("event", replace(BASE, rho=1.0)),
    # One-factor quantum-coupling sibling: q_r is fixed to the trigger
    # threshold while the full-reset event encoder is otherwise unchanged.
    "event_coupled_quantum": (
        "event", replace(BASE, jump0=BASE.threshold, jump_exponent=0.0)
    ),
}
CONFIGS = {**DENSE_CONFIGS, **MECHANISM_CONFIGS}


def _synthetic_federation(seed: int = 19) -> MulticlassFederation:
    rng = np.random.default_rng(seed)
    client_x = tuple(
        rng.integers(0, 256, size=(40, 3, 32, 32), dtype=np.uint8)
        for _ in range(10)
    )
    client_y = tuple(np.tile(np.arange(10, dtype=np.int64), 4) for _ in range(10))
    test_x = rng.integers(0, 256, size=(100, 3, 32, 32), dtype=np.uint8)
    test_y = np.tile(np.arange(10, dtype=np.int64), 10)
    return MulticlassFederation(
        client_X=client_x,
        client_y=client_y,
        X_train_eval=np.concatenate(client_x)[:100],
        y_train_eval=np.concatenate(client_y)[:100],
        X_test=test_x,
        y_test=test_y,
        regime="synthetic",
        client_class_counts=np.full((10, 10), 4, dtype=int),
        periods=np.ones(10, dtype=int),
        weights=np.full(10, 0.1),
    )


def smoke() -> None:
    assert LAYOUT.dimension == 20570
    federation = _synthetic_federation()
    cfg = replace(
        BASE, rounds=2, local_steps=1, batch_size=8, eval_stride=1,
        local_lr=0.01, regularization=0.0, init_scale=0.3,
    )
    dense_1 = run_final_baseline(
        federation=federation, architecture="cifar_cnn", method="dense",
        config=cfg, seed=123,
    )
    dense_half = run_final_baseline(
        federation=federation, architecture="cifar_cnn", method="dense",
        config=replace(cfg, server_gain=0.5), seed=123,
    )
    assert dense_1["final_test_ce"] != dense_half["final_test_ce"]
    assert dense_1["uplink_packetized_bits"] == dense_half["uplink_packetized_bits"]
    assert dense_1["unicast_hybrid_total_bits"] == dense_half["unicast_hybrid_total_bits"]

    for name in MECHANISM_CONFIGS:
        method, config = MECHANISM_CONFIGS[name]
        result = run_final_baseline(
            federation=federation, architecture="cifar_cnn", method=method,
            config=replace(config, rounds=2, local_steps=1, batch_size=8,
                           eval_stride=1, local_lr=0.01, regularization=0.0,
                           init_scale=0.3),
            seed=456,
        )
        assert np.isfinite(float(result["final_test_ce"]))
    print("P8 smoke checks passed")


def run_point(config_name: str, partition_seed: int, tag: str) -> Path:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown configuration: {config_name}")
    method, config = CONFIGS[config_name]
    train_seed = 80000 + partition_seed
    federation = make_cifar10_federation(
        root=DATA, regime="strong", seed=partition_seed
    )
    result = run_final_baseline(
        federation=federation,
        architecture="cifar_cnn",
        method=method,
        config=config,
        seed=train_seed,
    )
    result.update({
        "config_name": config_name,
        "family": "dense_gain" if config_name in DENSE_CONFIGS else "mechanism",
        "partition_seed": partition_seed,
        "train_seed": train_seed,
        "tag": tag,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{config_name}_p{partition_seed}_{tag}.csv"
    pd.DataFrame([result]).to_csv(path, index=False, float_format="%.10g")
    print(pd.DataFrame([result])[[
        "config_name", "family", "partition_seed", "server_gain", "rho",
        "threshold", "jump0", "jump_exponent", "final_train_ce",
        "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
        "unicast_hybrid_total_bits",
    ]].to_string(index=False))
    return path


def _point_files(root: Path, tag: str) -> list[Path]:
    return sorted(root.glob(f"*_p*_{tag}.csv"))


def select_development(root: Path) -> None:
    files = _point_files(root, "dev")
    if not files:
        raise RuntimeError("no development files")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    observed = set(data.config_name)
    if observed != set(CONFIGS):
        raise RuntimeError(
            f"development grid mismatch: missing={sorted(set(CONFIGS)-observed)}, "
            f"unexpected={sorted(observed-set(CONFIGS))}"
        )
    if not np.all(data.partition_seed == DEVELOPMENT_SEED):
        raise RuntimeError("selection may use only the development partition")
    finite_dense = data[
        (data.family == "dense_gain") & np.isfinite(data.final_train_ce)
    ].sort_values(["final_train_ce", "config_name"])
    if finite_dense.empty:
        raise RuntimeError("no finite dense server-gain candidate")
    dense_selected = str(finite_dense.iloc[0].config_name)
    selection = {
        "development_seed": DEVELOPMENT_SEED,
        "dense_selection_metric": "minimum final_train_ce",
        "dense_grid": list(DENSE_CONFIGS),
        "dense_selected": dense_selected,
        "mechanism_variants": list(MECHANISM_CONFIGS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "base_config": asdict(BASE),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data.sort_values(["family", "final_train_ce", "config_name"]).to_csv(
        OUT / "development_summary.csv", index=False, float_format="%.10g"
    )
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


def run_heldout(selection_path: Path, config_name: str, partition_seed: int) -> Path:
    if partition_seed not in HELDOUT_SEEDS:
        raise ValueError("held-out run requested outside the frozen seed set")
    selection = json.loads(selection_path.read_text())
    permitted = {selection["dense_selected"], *selection["mechanism_variants"]}
    if config_name not in permitted:
        raise ValueError(f"configuration was not frozen for held-out use: {config_name}")
    return run_point(config_name, partition_seed, "heldout")


def _aggregate(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_train_ce", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "uplink_packetized_bits",
        "broadcast_total_bits", "unicast_hybrid_total_bits",
        "coordinate_events",
    ]
    return (
        data.groupby(["family", "config_name"], as_index=False)
        .agg(**{
            f"{metric}_{stat}": (metric, stat)
            for metric in metrics for stat in ("mean", "std")
        })
        .sort_values(["family", "config_name"])
    )


def aggregate(root: Path, selection_path: Path) -> None:
    files = _point_files(root, "heldout")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    selection = json.loads(selection_path.read_text())
    expected_configs = {selection["dense_selected"], *selection["mechanism_variants"]}
    counts = data.groupby("config_name").size().to_dict()
    expected_counts = {name: len(HELDOUT_SEEDS) for name in expected_configs}
    if counts != expected_counts:
        raise RuntimeError(f"held-out grid mismatch: {counts} != {expected_counts}")
    for name, group in data.groupby("config_name"):
        if set(group.partition_seed) != set(HELDOUT_SEEDS):
            raise RuntimeError(f"wrong seeds for {name}")

    summary = _aggregate(data)
    event = summary[summary.config_name == "event_frozen"].iloc[0]
    dense = summary[summary.config_name == selection["dense_selected"]].iloc[0]
    no_leak = summary[summary.config_name == "event_no_leak"].iloc[0]
    coupled = summary[summary.config_name == "event_coupled_quantum"].iloc[0]
    decision = {
        "dense_selected": selection["dense_selected"],
        "dense_server_gain": float(
            data[data.config_name == selection["dense_selected"]].server_gain.iloc[0]
        ),
        "event_minus_dense_accuracy_points": 100.0 * float(
            event.final_test_accuracy_mean - dense.final_test_accuracy_mean
        ),
        "event_over_dense_unicast_ratio": float(
            event.unicast_hybrid_total_bits_mean
            / dense.unicast_hybrid_total_bits_mean
        ),
        "no_leak_minus_frozen_accuracy_points": 100.0 * float(
            no_leak.final_test_accuracy_mean - event.final_test_accuracy_mean
        ),
        "no_leak_over_frozen_unicast_ratio": float(
            no_leak.unicast_hybrid_total_bits_mean
            / event.unicast_hybrid_total_bits_mean
        ),
        "coupled_minus_frozen_accuracy_points": 100.0 * float(
            coupled.final_test_accuracy_mean - event.final_test_accuracy_mean
        ),
        "coupled_over_frozen_unicast_ratio": float(
            coupled.unicast_hybrid_total_bits_mean
            / event.unicast_hybrid_total_bits_mean
        ),
        "interpretation_rule": (
            "Report means and sample standard deviations over the three frozen "
            "held-out seeds; do not claim statistical significance."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data.sort_values(["family", "config_name", "partition_seed"]).to_csv(
        OUT / "heldout_runs.csv", index=False, float_format="%.10g"
    )
    summary.to_csv(OUT / "heldout_summary.csv", index=False, float_format="%.10g")
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\caption{P8 held-out server-scale and mechanism audit on CIFAR-10.}",
        r"\label{tab:p8-audit}", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Configuration & Acc. [\%] & Worst [\%] & Total [Mbit]\\", r"\midrule",
    ]
    order = [selection["dense_selected"], "event_frozen", "event_no_leak",
             "event_coupled_quantum"]
    labels = {
        selection["dense_selected"]: "Dense, tuned gain",
        "event_frozen": "Event-FedAvg",
        "event_no_leak": r"Event, $\rho=1$",
        "event_coupled_quantum": r"Event, $q=\vartheta$",
    }
    for name in order:
        row = summary[summary.config_name == name].iloc[0]
        lines.append(
            f"{labels[name]} & "
            f"{100*row.final_test_accuracy_mean:.2f}$\\pm${100*row.final_test_accuracy_std:.2f} & "
            f"{100*row.final_worst_class_accuracy_mean:.1f}$\\pm${100*row.final_worst_class_accuracy_std:.1f} & "
            f"{row.unicast_hybrid_total_bits_mean/1e6:.1f}$\\pm${row.unicast_hybrid_total_bits_std/1e6:.1f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "p8_summary.tex").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    point = sub.add_parser("point")
    point.add_argument("--config", required=True, choices=sorted(CONFIGS))
    point.add_argument("--partition-seed", type=int, default=DEVELOPMENT_SEED)
    point.add_argument("--tag", default="dev")
    select = sub.add_parser("select")
    select.add_argument("--input-dir", type=Path, required=True)
    heldout = sub.add_parser("heldout")
    heldout.add_argument("--selection", type=Path, required=True)
    heldout.add_argument("--config", required=True, choices=sorted(CONFIGS))
    heldout.add_argument("--partition-seed", type=int, required=True)
    aggregate_cmd = sub.add_parser("aggregate")
    aggregate_cmd.add_argument("--input-dir", type=Path, required=True)
    aggregate_cmd.add_argument("--selection", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "point":
        run_point(args.config, args.partition_seed, args.tag)
    elif args.command == "select":
        select_development(args.input_dir)
    elif args.command == "heldout":
        run_heldout(args.selection, args.config, args.partition_seed)
    elif args.command == "aggregate":
        aggregate(args.input_dir, args.selection)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
