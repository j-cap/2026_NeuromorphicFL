from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import make_cifar10_federation
from neuromorphicfl.cifar10_resnet import LAYOUT, initialize_resnet14, loss_and_gradient
from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig,
    Method,
    run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation


DATA = Path("data/cifar-10")
OUT = Path("experiments/results/p13_cifar10_resnet14")
DEVELOPMENT_SEED = 4100
PILOT_SEEDS = (4200, 4300, 4400)
EXTENSION_SEEDS = (4500, 4600, 4700, 4800, 4900, 5000, 5100)
DEVELOPMENT_TAG = "dev-r1800"
HELDOUT_TAG = "heldout-r1800"

# The horizon was amended from 180 to 1,800 rounds after the dense-only audit.
# Dense gain 1.5 was frozen before inspecting any compressed-method result.
BASE = FinalBaselineConfig(
    local_steps=5,
    local_lr=0.05,
    batch_size=32,
    regularization=5e-4,
    rounds=1800,
    eval_stride=15,
    rho=0.999,
    threshold=0.025,
    jump0=0.005,
    jump_scale=100.0,
    jump_exponent=0.3,
    topk_fraction=0.01,
    strom_threshold=0.01,
    init_scale=1.0,
)


CONFIGS: dict[str, tuple[Method, FinalBaselineConfig]] = {
    "dense_g05": ("dense", replace(BASE, server_gain=0.5)),
    "dense_g10": ("dense", replace(BASE, server_gain=1.0)),
    "dense_g15": ("dense", replace(BASE, server_gain=1.5)),
    "dense_g20": ("dense", replace(BASE, server_gain=2.0)),
    "event_t0125_q0025": ("event", replace(BASE, threshold=0.0125, jump0=0.0025)),
    "event_t0125_q005": ("event", replace(BASE, threshold=0.0125, jump0=0.005)),
    "event_t025_q0025": ("event", replace(BASE, threshold=0.025, jump0=0.0025)),
    "event_t025_q005": ("event", replace(BASE, threshold=0.025, jump0=0.005)),
    "event_t025_q01": ("event", replace(BASE, threshold=0.025, jump0=0.01)),
    "event_t05_q005": ("event", replace(BASE, threshold=0.05, jump0=0.005)),
    "event_t05_q01": ("event", replace(BASE, threshold=0.05, jump0=0.01)),
    "sign_ef": ("sign_ef", BASE),
    "ef_k005": ("ef_topk", replace(BASE, topk_fraction=0.005)),
    "ef_k01": ("ef_topk", replace(BASE, topk_fraction=0.01)),
    "ef_k025": ("ef_topk", replace(BASE, topk_fraction=0.025)),
    "strom_t005": ("strom", replace(BASE, strom_threshold=0.005)),
    "strom_t01": ("strom", replace(BASE, strom_threshold=0.01)),
    "strom_t02": ("strom", replace(BASE, strom_threshold=0.02)),
}
METHODS: tuple[Method, ...] = ("event", "dense", "sign_ef", "ef_topk", "strom")
DEVELOPMENT_CONFIGS = tuple(
    name for name, (method, _) in CONFIGS.items()
    if method != "dense" or name == "dense_g15"
)


def _synthetic_federation(seed: int = 13) -> MulticlassFederation:
    rng = np.random.default_rng(seed)
    clients_X = tuple(
        rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8)
        for _ in range(2)
    )
    clients_y = tuple(
        rng.integers(0, 10, size=20, dtype=np.int64) for _ in range(2)
    )
    test_X = rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8)
    test_y = rng.integers(0, 10, size=20, dtype=np.int64)
    return MulticlassFederation(
        client_X=clients_X,
        client_y=clients_y,
        X_train_eval=np.concatenate(clients_X),
        y_train_eval=np.concatenate(clients_y),
        X_test=test_X,
        y_test=test_y,
        regime="synthetic",
        client_class_counts=np.zeros((2, 10), dtype=int),
        periods=np.ones(2, dtype=int),
        weights=np.full(2, 0.5),
    )


def smoke() -> None:
    assert LAYOUT.dimension == 175258
    assert sum(size for _, _, size in LAYOUT.activity_groups()) == LAYOUT.dimension
    rng = np.random.default_rng(9)
    X = rng.integers(0, 256, size=(2, 3, 32, 32), dtype=np.uint8)
    y = np.array([1, 8], dtype=np.int64)
    w = initialize_resnet14(seed=17, scale=0.3)
    _, _, gradient = loss_and_gradient(w, X, y, regularization=0.0)
    direction = rng.normal(size=w.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-3
    plus, _ = loss_and_gradient(
        w + epsilon * direction, X, y, regularization=0.0, need_gradient=False
    )
    minus, _ = loss_and_gradient(
        w - epsilon * direction, X, y, regularization=0.0, need_gradient=False
    )
    finite_difference = (plus - minus) / (2 * epsilon)
    analytic = float(gradient @ direction)
    if not math.isclose(finite_difference, analytic, rel_tol=3e-2, abs_tol=2e-3):
        raise AssertionError((finite_difference, analytic))

    config = replace(
        BASE, rounds=1, local_steps=1, batch_size=2, eval_stride=1,
        local_lr=0.005, regularization=0.0, threshold=0.001, jump0=0.0005,
    )
    federation = _synthetic_federation()
    for method in METHODS:
        result = run_final_baseline(
            federation=federation,
            architecture="cifar_resnet14",
            method=method,
            config=config,
            seed=99,
            record_history=True,
            record_group_activity=(method == "event"),
        )
        assert np.isfinite(float(result["final_test_ce"]))
        assert int(result["replay_rounds"]) + int(result["checkpoint_rounds"]) == 1
        if method == "event":
            activity = result["group_activity"]
            assert isinstance(activity, pd.DataFrame)
            assert len(activity) == len(LAYOUT.activity_groups())
    print(f"P13 smoke passed; ResNet-14 dimension={LAYOUT.dimension}")


def _result_path(config_name: str, partition_seed: int, tag: str) -> Path:
    return OUT / f"{config_name}_p{partition_seed}_{tag}.csv"


def run_point(config_name: str, partition_seed: int, tag: str) -> Path:
    if config_name not in CONFIGS:
        raise ValueError(config_name)
    method, config = CONFIGS[config_name]
    train_seed = 90000 + partition_seed
    federation = make_cifar10_federation(root=DATA, regime="strong", seed=partition_seed)
    result = run_final_baseline(
        federation=federation,
        architecture="cifar_resnet14",
        method=method,
        config=config,
        seed=train_seed,
        record_history=True,
        record_group_activity=(method == "event"),
    )
    history = result.pop("history")
    activity = result.pop("group_activity", None)
    result.update({
        "config_name": config_name,
        "partition_seed": partition_seed,
        "train_seed": train_seed,
        "tag": tag,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    path = _result_path(config_name, partition_seed, tag)
    pd.DataFrame([result]).to_csv(path, index=False, float_format="%.10g")
    history.assign(
        config_name=config_name, method=method, partition_seed=partition_seed,
        train_seed=train_seed, tag=tag,
    ).to_csv(path.with_name(path.stem + "_history.csv"), index=False, float_format="%.10g")
    if activity is not None:
        activity.assign(
            config_name=config_name, partition_seed=partition_seed, train_seed=train_seed,
        ).to_csv(path.with_name(path.stem + "_activity.csv"), index=False, float_format="%.10g")
    print(pd.DataFrame([result])[[
        "config_name", "method", "partition_seed", "final_train_ce",
        "final_test_accuracy", "final_worst_class_accuracy",
        "coordinate_events", "unicast_hybrid_total_bits",
    ]].to_string(index=False))
    return path


def _files(tag: str) -> list[Path]:
    return sorted(
        path for path in OUT.glob(f"*_p*_{tag}.csv")
        if not path.name.endswith(("_history.csv", "_activity.csv"))
    )


def select() -> None:
    frames = [pd.read_csv(path) for path in _files(DEVELOPMENT_TAG)]
    if not frames:
        raise RuntimeError("no development results")
    data = pd.concat(frames, ignore_index=True)
    if set(data.config_name) != set(DEVELOPMENT_CONFIGS):
        raise RuntimeError("development grid is incomplete")
    if set(data.partition_seed) != {DEVELOPMENT_SEED}:
        raise RuntimeError("selection may use only the development partition")
    quality = {}
    for method in METHODS:
        rows = data[data.method == method].sort_values(
            ["final_train_ce", "unicast_hybrid_total_bits", "config_name"]
        )
        quality[method] = str(rows.iloc[0].config_name)
    selection = {
        "development_seed": DEVELOPMENT_SEED,
        "selection_metric": "minimum final training cross-entropy",
        "quality": quality,
        "configuration_frozen_before_heldout_evaluation": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data.sort_values(["method", "final_train_ce"]).to_csv(
        OUT / "development_summary.csv", index=False, float_format="%.10g"
    )
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


def heldout(selection_path: Path, method: Method, partition_seed: int) -> Path:
    allowed = set(PILOT_SEEDS) | set(EXTENSION_SEEDS)
    if partition_seed not in allowed:
        raise ValueError(f"held-out seed must be one of {sorted(allowed)}")
    selection = json.loads(selection_path.read_text())
    return run_point(selection["quality"][method], partition_seed, HELDOUT_TAG)


def assess(selection_path: Path) -> None:
    selection = json.loads(selection_path.read_text())
    frames = [pd.read_csv(path) for path in _files(HELDOUT_TAG)]
    if not frames:
        raise RuntimeError("no held-out results")
    data = pd.concat(frames, ignore_index=True)
    expected_configs = set(selection["quality"].values())
    if not set(data.config_name).issubset(expected_configs):
        raise RuntimeError("held-out results contain a non-frozen configuration")
    metrics = [
        "final_train_ce", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "coordinate_events",
        "unicast_hybrid_total_bits",
    ]
    summary = data.groupby(["method", "config_name"], as_index=False).agg(**{
        f"{metric}_{stat}": (metric, stat)
        for metric in metrics for stat in ("mean", "std")
    })
    summary.to_csv(OUT / "heldout_summary.csv", index=False, float_format="%.10g")

    pilot_complete = all(
        len(data[(data.method == method) & data.partition_seed.isin(PILOT_SEEDS)]) == 3
        for method in METHODS
    )
    histories = []
    for path in OUT.glob(f"*_{HELDOUT_TAG}_history.csv"):
        histories.append(pd.read_csv(path))
    stable = False
    relative_tail_improvement: dict[str, float] = {}
    if histories:
        history = pd.concat(histories, ignore_index=True)
        for method in METHODS:
            selected_name = selection["quality"][method]
            selected_history = history[
                (history.config_name == selected_name)
                & history.partition_seed.isin(PILOT_SEEDS)
            ]
            improvements = []
            for _, trajectory in selected_history.groupby("partition_seed"):
                trajectory = trajectory.sort_values("round")
                cutoff = 0.8 * BASE.rounds
                tail = trajectory[trajectory["round"] >= cutoff]
                if len(tail) >= 2:
                    improvements.append(
                        (float(tail.iloc[0].train_ce) - float(tail.iloc[-1].train_ce))
                        / max(abs(float(tail.iloc[0].train_ce)), 1e-12)
                    )
            if improvements:
                relative_tail_improvement[method] = float(np.mean(improvements))
        stable = bool(
            len(relative_tail_improvement) == len(METHODS)
            and all(abs(value) < 0.01 for value in relative_tail_improvement.values())
        )

    activity_frames = [
        pd.read_csv(path) for path in OUT.glob(f"*_{HELDOUT_TAG}_activity.csv")
        if any(f"_p{seed}_" in path.name for seed in PILOT_SEEDS)
    ]
    no_starved_group = False
    min_group_events = None
    if activity_frames:
        activity = pd.concat(activity_frames, ignore_index=True)
        totals = activity.groupby(["partition_seed", "group"]).coordinate_events.sum()
        min_group_events = int(totals.min())
        no_starved_group = bool((totals > 0).all())

    chance_exceeded = False
    finite = bool(np.isfinite(data[metrics]).all().all())
    if pilot_complete:
        dense = data[
            (data.method == "dense") & data.partition_seed.isin(PILOT_SEEDS)
        ]
        chance_exceeded = bool(dense.final_test_accuracy.mean() >= 0.20)
    decision = {
        "pilot_complete": pilot_complete,
        "all_metrics_finite": finite,
        "dense_mean_accuracy_at_least_20_percent": chance_exceeded,
        "event_no_zero-event_semantic_group": no_starved_group,
        "minimum_group_event_count": min_group_events,
        "mean_relative_train_ce_improvement_over_final_20_percent_by_method": relative_tail_improvement,
        "all_methods_practical_stability_below_1_percent_diagnostic": stable,
        "matched_1800_round_budget_complete": bool(
            pilot_complete and (data[data.partition_seed.isin(PILOT_SEEDS)].rounds == BASE.rounds).all()
        ),
        "proceed_to_extension": bool(
            pilot_complete and finite and chance_exceeded and no_starved_group
            and (data[data.partition_seed.isin(PILOT_SEEDS)].rounds == BASE.rounds).all()
        ),
        "interpretation": (
            "The gate concerns experimental validity, not whether Event-FedAvg wins. "
            "A valid unfavorable comparison remains reportable."
        ),
    }
    (OUT / "gate.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


def write_protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "amended_after_dense_only_audit_before_compressed_method_results",
        "amendment": (
            "Dense-only audits motivated a matched 1800-round budget and fixed "
            "dense_g15; the rejected dense gains are not rerun in development."
        ),
        "dataset": "CIFAR-10 python version",
        "partition": {
            "clients": 10,
            "dominant_class_fraction": 0.55,
            "development_seed": DEVELOPMENT_SEED,
            "pilot_seeds": PILOT_SEEDS,
            "extension_seeds": EXTENSION_SEEDS,
        },
        "architecture": {
            "family": "CIFAR ResNet-14",
            "normalization": "GroupNorm with 4 groups",
            "depth_rule": "6n+2 with n=2",
            "dimension": LAYOUT.dimension,
            "tensor_layout": [
                {"name": name, "shape": shape, "size": int(np.prod(shape))}
                for name, shape in LAYOUT.tensor_specs()
            ],
        },
        "base_training": asdict(BASE),
        "candidate_configs": {
            name: {"method": method, "config": asdict(config)}
            for name, (method, config) in CONFIGS.items()
        },
        "amended_development_configs": DEVELOPMENT_CONFIGS,
        "selection": "minimum final training CE on development partition only",
        "pilot_gate": {
            "all_metrics_finite": True,
            "dense_mean_test_accuracy_minimum": 0.20,
            "all_semantic_parameter_groups_emit_at_least_one event": True,
            "matched_round_budget": BASE.rounds,
            "tail_training_ce_change": "reported as a diagnostic, not an extension gate",
            "outcome_independence": "Event-FedAvg need not outperform a baseline",
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(OUT / "protocol.json")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke")
    commands.add_parser("protocol")
    point = commands.add_parser("point")
    point.add_argument("--config", required=True, choices=sorted(CONFIGS))
    point.add_argument("--partition-seed", type=int, default=DEVELOPMENT_SEED)
    point.add_argument("--tag", default="dev")
    commands.add_parser("select")
    held = commands.add_parser("heldout")
    held.add_argument("--selection", type=Path, required=True)
    held.add_argument("--method", choices=METHODS, required=True)
    held.add_argument("--partition-seed", type=int, required=True)
    assess_cmd = commands.add_parser("assess")
    assess_cmd.add_argument("--selection", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "protocol":
        write_protocol()
    elif args.command == "point":
        run_point(args.config, args.partition_seed, args.tag)
    elif args.command == "select":
        select()
    elif args.command == "heldout":
        heldout(args.selection, args.method, args.partition_seed)
    elif args.command == "assess":
        assess(args.selection)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
