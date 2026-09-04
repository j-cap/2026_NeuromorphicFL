from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import (
    LAYOUT,
    class_count_matrix,
    federation_audit,
    initialize_cnn,
    loss_and_gradient,
    make_cifar10_federation,
)
from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig,
    Method,
    run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation


DATA = Path("data/cifar-10")
OUT = Path("experiments/results/p3_cifar10")
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
)


CONFIGS: dict[str, tuple[Method, FinalBaselineConfig]] = {
    "dense": ("dense", BASE),
    "sign_ef": ("sign_ef", BASE),
    "ef_k005": ("ef_topk", replace(BASE, topk_fraction=0.005)),
    "ef_k01": ("ef_topk", replace(BASE, topk_fraction=0.01)),
    "ef_k025": ("ef_topk", replace(BASE, topk_fraction=0.025)),
    "ef_k05": ("ef_topk", replace(BASE, topk_fraction=0.05)),
    "strom_t0025": ("strom", replace(BASE, strom_threshold=0.0025)),
    "strom_t005": ("strom", replace(BASE, strom_threshold=0.005)),
    "strom_t01": ("strom", replace(BASE, strom_threshold=0.01)),
    "strom_t02": ("strom", replace(BASE, strom_threshold=0.02)),
    "strom_t04": ("strom", replace(BASE, strom_threshold=0.04)),
    "event_t0125_q0025": (
        "event", replace(BASE, threshold=0.0125, jump0=0.0025)
    ),
    "event_t025_q0025": (
        "event", replace(BASE, threshold=0.025, jump0=0.0025)
    ),
    "event_t025_q005": (
        "event", replace(BASE, threshold=0.025, jump0=0.005)
    ),
    "event_t05_q005": (
        "event", replace(BASE, threshold=0.05, jump0=0.005)
    ),
    "event_t05_q01": (
        "event", replace(BASE, threshold=0.05, jump0=0.01)
    ),
}

TUNING_CONFIGS = tuple(name for name in CONFIGS if name not in {"dense"})
QUALITY_METHODS: tuple[Method, ...] = (
    "event", "strom", "ef_topk", "sign_ef", "dense"
)
TRAFFIC_METHODS: tuple[Method, ...] = ("strom", "ef_topk")


def _synthetic_federation(seed: int = 11) -> MulticlassFederation:
    rng = np.random.default_rng(seed)
    clients_X, clients_y = [], []
    counts = np.full((10, 10), 4, dtype=int)
    for client in range(10):
        clients_X.append(rng.integers(0, 256, size=(40, 3, 32, 32), dtype=np.uint8))
        clients_y.append(np.tile(np.arange(10, dtype=np.int64), 4))
    test_X = rng.integers(0, 256, size=(100, 3, 32, 32), dtype=np.uint8)
    test_y = np.tile(np.arange(10, dtype=np.int64), 10)
    return MulticlassFederation(
        client_X=tuple(clients_X), client_y=tuple(clients_y),
        X_train_eval=np.concatenate(clients_X)[:100],
        y_train_eval=np.concatenate(clients_y)[:100],
        X_test=test_X, y_test=test_y, regime="synthetic",
        client_class_counts=counts, periods=np.ones(10, dtype=int),
        weights=np.full(10, 0.1),
    )


def run_smoke() -> None:
    assert LAYOUT.dimension == 20570
    counts = class_count_matrix("strong")
    assert np.all(counts.sum(axis=0) == 5000)
    assert np.all(counts.sum(axis=1) == 5000)
    assert np.all(np.diag(counts) == 2750)

    fed = _synthetic_federation()
    w = initialize_cnn(seed=9, scale=0.3)
    _, _, gradient = loss_and_gradient(
        w, fed.X_test[:3], fed.y_test[:3], regularization=0.0
    )
    direction = np.random.default_rng(12).normal(size=w.shape).astype(np.float32)
    direction /= np.linalg.norm(direction)
    epsilon = 2e-3
    plus, _ = loss_and_gradient(
        w + epsilon * direction, fed.X_test[:3], fed.y_test[:3],
        regularization=0.0, need_gradient=False,
    )
    minus, _ = loss_and_gradient(
        w - epsilon * direction, fed.X_test[:3], fed.y_test[:3],
        regularization=0.0, need_gradient=False,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = float(gradient @ direction)
    assert math.isclose(finite_difference, analytic, rel_tol=2e-2, abs_tol=2e-3), (
        finite_difference, analytic
    )
    cfg = replace(
        BASE, rounds=2, local_steps=1, batch_size=8, eval_stride=1,
        local_lr=0.01, regularization=0.0, threshold=0.005,
        jump0=0.001, strom_threshold=0.001, topk_fraction=0.01,
    )
    summaries = []
    for method in QUALITY_METHODS:
        result = run_final_baseline(
            federation=fed, architecture="cifar_cnn", method=method,
            config=cfg, seed=1234,
        )
        assert np.isfinite(float(result["final_test_ce"]))
        assert int(result["replay_rounds"]) + int(result["checkpoint_rounds"]) == 2
        assert int(result["broadcast_total_bits"]) == (
            int(result["uplink_packetized_bits"])
            + int(result["broadcast_downlink_bits"])
        )
        summaries.append((method, float(result["final_test_ce"])))

    first = run_final_baseline(
        federation=fed, architecture="cifar_cnn", method="dense",
        config=cfg, seed=4321,
    )
    second = run_final_baseline(
        federation=fed, architecture="cifar_cnn", method="dense",
        config=cfg, seed=4321,
    )
    assert first["final_test_ce"] == second["final_test_ce"]
    print("P3 smoke passed:", summaries)


def _run_point(config_name: str, partition_seed: int, tag: str) -> Path:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown config {config_name}")
    method, config = CONFIGS[config_name]
    train_seed = 80000 + partition_seed
    federation = make_cifar10_federation(
        root=DATA, regime="strong", seed=partition_seed
    )
    result = run_final_baseline(
        federation=federation, architecture="cifar_cnn", method=method,
        config=config, seed=train_seed, record_history=True,
    )
    history = result.pop("history")
    assert isinstance(history, pd.DataFrame)
    result.update({
        "config_name": config_name,
        "partition_seed": partition_seed,
        "train_seed": train_seed,
        "tag": tag,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{config_name}_p{partition_seed}_{tag}"
    path = OUT / f"{stem}.csv"
    pd.DataFrame([result]).to_csv(path, index=False, float_format="%.10g")
    history.assign(
        config_name=config_name, method=method,
        partition_seed=partition_seed, train_seed=train_seed, tag=tag,
    ).to_csv(OUT / f"{stem}_history.csv", index=False, float_format="%.10g")
    if partition_seed == DEVELOPMENT_SEED and config_name == "dense":
        federation_audit(federation).to_csv(OUT / "partition_audit.csv", index=False)
        (OUT / "protocol.json").write_text(json.dumps({
            "dataset": "CIFAR-10 python version",
            "dataset_md5": "c58f30108f718f92721af3b95e74349a",
            "architecture": asdict(LAYOUT),
            "dimension": LAYOUT.dimension,
            "partition": {
                "clients": 10, "samples_per_client": 5000,
                "dominant_class_fraction": 0.55,
                "development_seed": DEVELOPMENT_SEED,
                "heldout_seeds": HELDOUT_SEEDS,
            },
            "base_training": asdict(BASE),
            "selection_metric": "final_train_ce on development partition",
        }, indent=2) + "\n")
    print(pd.DataFrame([result])[[
        "config_name", "method", "partition_seed", "final_train_ce",
        "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
        "uplink_packetized_bits", "unicast_hybrid_total_bits",
    ]].to_string(index=False))
    return path


def _result_files(root: Path, tag: str) -> list[Path]:
    return sorted(
        path for path in root.glob(f"*_p*_{tag}.csv")
        if not path.name.endswith("_history.csv")
    )


def select_development(root: Path) -> None:
    frames = [pd.read_csv(path) for path in _result_files(root, "dev")]
    if not frames:
        raise RuntimeError("no development result files")
    data = pd.concat(frames, ignore_index=True)
    observed = set(data.config_name)
    expected = set(CONFIGS)
    if observed != expected:
        raise RuntimeError(
            f"development grid mismatch; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    if not np.all(data.partition_seed == DEVELOPMENT_SEED):
        raise RuntimeError("selection may use only the frozen development partition")

    quality: dict[str, str] = {}
    for method in QUALITY_METHODS:
        candidates = data[data.method == method].sort_values(
            ["final_train_ce", "unicast_hybrid_total_bits", "config_name"]
        )
        quality[method] = str(candidates.iloc[0].config_name)

    event_row = data[data.config_name == quality["event"]].iloc[0]
    target = float(event_row.unicast_hybrid_total_bits)
    traffic: dict[str, str] = {}
    for method in TRAFFIC_METHODS:
        candidates = data[data.method == method].copy()
        candidates["traffic_log_distance"] = np.abs(
            np.log(candidates.unicast_hybrid_total_bits / target)
        )
        candidates = candidates.sort_values(
            ["traffic_log_distance", "final_train_ce", "config_name"]
        )
        traffic[method] = str(candidates.iloc[0].config_name)

    selection = {
        "development_seed": DEVELOPMENT_SEED,
        "selection_metric": "minimum final_train_ce",
        "quality": quality,
        "traffic_target_config": quality["event"],
        "traffic_target_total_bits": int(target),
        "traffic": traffic,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    data.sort_values(["method", "final_train_ce"]).to_csv(
        OUT / "development_summary.csv", index=False, float_format="%.10g"
    )
    print(json.dumps(selection, indent=2))


def run_heldout(
    selection_path: Path, method: Method, comparison: str, partition_seed: int,
) -> Path:
    if partition_seed not in HELDOUT_SEEDS:
        raise ValueError("held-out runs are restricted to the frozen held-out seeds")
    selection = json.loads(selection_path.read_text())
    if comparison == "quality":
        config_name = selection["quality"][method]
    elif comparison == "traffic" and method in TRAFFIC_METHODS:
        config_name = selection["traffic"][method]
    else:
        raise ValueError(f"invalid comparison {comparison} for {method}")
    return _run_point(config_name, partition_seed, f"heldout_{comparison}")


def _aggregate_group(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
        "uplink_packetized_bits", "broadcast_total_bits",
        "unicast_hybrid_total_bits", "coordinate_events",
    ]
    return (
        data.groupby(["comparison", "method", "config_name"], as_index=False)
        .agg(**{
            f"{metric}_{stat}": (metric, stat)
            for metric in metrics for stat in ("mean", "std")
        })
        .sort_values(["comparison", "method"])
    )


def aggregate_heldout(root: Path, selection_path: Path) -> None:
    quality_files = _result_files(root, "heldout_quality")
    traffic_files = _result_files(root, "heldout_traffic")
    data = pd.concat([pd.read_csv(path) for path in quality_files + traffic_files],
                     ignore_index=True)
    data["comparison"] = data.tag.str.replace("heldout_", "", regex=False)
    counts = data.groupby(["comparison", "method"]).size()
    expected = {
        **{("quality", method): 3 for method in QUALITY_METHODS},
        **{("traffic", method): 3 for method in TRAFFIC_METHODS},
    }
    if counts.to_dict() != expected:
        raise RuntimeError(f"held-out grid mismatch: {counts.to_dict()}")

    summary = _aggregate_group(data)
    selection = json.loads(selection_path.read_text())
    event_rows = data[
        (data.comparison == "quality") & (data.method == "event")
    ].copy()
    event_rows["comparison"] = "traffic"
    traffic_data = pd.concat([
        event_rows, data[data.comparison == "traffic"]
    ], ignore_index=True)
    traffic_summary = _aggregate_group(traffic_data)

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "heldout_runs.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "heldout_summary.csv", index=False, float_format="%.10g")
    traffic_summary.to_csv(
        OUT / "traffic_matched_summary.csv", index=False, float_format="%.10g"
    )
    pd.concat([
        summary[summary.comparison == "quality"], traffic_summary
    ], ignore_index=True).to_csv(
        OUT / "master_results.csv", index=False, float_format="%.10g"
    )

    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    plot = pd.concat([
        summary[summary.comparison == "quality"].assign(series="quality-selected"),
        traffic_summary.assign(series="traffic-matched"),
    ], ignore_index=True)
    markers = {"event": "*", "strom": "s", "ef_topk": "o", "sign_ef": "^", "dense": "D"}
    colors = {"quality-selected": "tab:blue", "traffic-matched": "tab:orange"}
    for row in plot.itertuples():
        axis.errorbar(
            row.unicast_hybrid_total_bits_mean / 1e6,
            100.0 * row.final_test_accuracy_mean,
            xerr=row.unicast_hybrid_total_bits_std / 1e6,
            yerr=100.0 * row.final_test_accuracy_std,
            marker=markers[row.method], color=colors[row.series],
            linestyle="none", capsize=3, markersize=9 if row.method == "event" else 6,
            alpha=0.9,
        )
        axis.annotate(
            row.method.replace("_", "-"),
            (row.unicast_hybrid_total_bits_mean / 1e6,
             100.0 * row.final_test_accuracy_mean),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("total bidirectional unicast traffic [Mbit, log scale]")
    axis.set_ylabel("CIFAR-10 test accuracy [%]")
    axis.grid(True, which="both", alpha=0.25)
    axis.set_title("P3 held-out communication--performance operating points")
    fig.savefig(OUT / "cifar10_frontier.png", dpi=200)
    fig.savefig(OUT / "cifar10_frontier.pdf")
    plt.close(fig)

    quality_event = summary[
        (summary.comparison == "quality") & (summary.method == "event")
    ].iloc[0]
    traffic_rows = traffic_summary.sort_values("method")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{CIFAR-10 held-out results over three partitions. Traffic is conservative bidirectional unicast communication.}",
        r"\label{tab:cifar10-p3}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Method & Acc. [\%] & Worst [\%] & Total [Mbit]\\", r"\midrule",
    ]
    for row in traffic_rows.itertuples():
        label = "Event-FedAvg" if row.method == "event" else row.method.replace("_", "-")
        lines.append(
            f"{label} & {100*row.final_test_accuracy_mean:.2f} $\\pm$ {100*row.final_test_accuracy_std:.2f} "
            f"& {100*row.final_worst_class_accuracy_mean:.1f} $\\pm$ {100*row.final_worst_class_accuracy_std:.1f} "
            f"& {row.unicast_hybrid_total_bits_mean/1e6:.1f} $\\pm$ {row.unicast_hybrid_total_bits_std/1e6:.1f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "p3_cifar10_summary.tex").write_text("\n".join(lines) + "\n")

    event_acc = float(quality_event.final_test_accuracy_mean)
    competitors = traffic_summary[traffic_summary.method != "event"]
    dominated = bool(np.any(
        (competitors.final_test_accuracy_mean >= event_acc)
        & (competitors.unicast_hybrid_total_bits_mean
           <= float(quality_event.unicast_hybrid_total_bits_mean))
    ))
    decision = "fail" if dominated else "pass_or_qualified_pass"
    (OUT / "decision.json").write_text(json.dumps({
        "classification": decision,
        "note": "Final wording requires inspection of uncertainty and quality-selected points.",
        "selection": selection,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nTraffic matched:\n", traffic_summary.to_string(index=False))
    print("\nPreliminary classification:", decision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("smoke")
    point = sub.add_parser("point")
    point.add_argument("--config", required=True, choices=sorted(CONFIGS))
    point.add_argument("--partition-seed", type=int, default=DEVELOPMENT_SEED)
    point.add_argument("--tag", default="dev")
    select = sub.add_parser("select")
    select.add_argument("--input-dir", type=Path, required=True)
    heldout = sub.add_parser("heldout")
    heldout.add_argument("--selection", type=Path, required=True)
    heldout.add_argument("--method", required=True, choices=QUALITY_METHODS)
    heldout.add_argument("--comparison", required=True, choices=("quality", "traffic"))
    heldout.add_argument("--partition-seed", type=int, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--input-dir", type=Path, required=True)
    aggregate.add_argument("--selection", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        run_smoke()
    elif args.command == "point":
        _run_point(args.config, args.partition_seed, args.tag)
    elif args.command == "select":
        select_development(args.input_dir)
    elif args.command == "heldout":
        run_heldout(args.selection, args.method, args.comparison, args.partition_seed)
    elif args.command == "aggregate":
        aggregate_heldout(args.input_dir, args.selection)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
