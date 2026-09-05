"""P11: factorial audit of Event-FedAvg aggregate alignment.

The experiment changes only two factors from the frozen Fashion-MNIST MLP
setting: the partition regime (IID or strong non-IID) and the number of local
SGD steps (one or five).  Every audited round is reconstructed by an
independent replay so reference-gradient evaluation cannot perturb later event
decisions in the hybrid trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig,
    run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import (
    MulticlassFederation,
    make_multiclass_federation,
)


DATA = Path("data/fashion-mnist")
OUT = Path("experiments/results/p11_alignment_factorial")
REGIMES = ("iid", "strong")
LOCAL_STEPS = (1, 5)
PARTITION_SEEDS = (2500, 2600, 2700)
KAPPA = 8.0


def experiment_config(local_steps: int) -> FinalBaselineConfig:
    if local_steps not in LOCAL_STEPS:
        raise ValueError(f"unsupported local-step count: {local_steps}")
    return FinalBaselineConfig(
        local_steps=local_steps,
        local_lr=0.1,
        rounds=150,
        eval_stride=15,
        rho=0.999,
        threshold=0.025,
        jump0=0.005,
        jump_scale=100.0,
        jump_exponent=0.1,
    )


def audited_rounds(rounds: int, stride: int) -> tuple[int, ...]:
    if stride <= 0:
        raise ValueError("audit stride must be positive")
    return tuple(sorted({1, rounds, *range(stride, rounds + 1, stride)}))


def add_diagnostics(audit: pd.DataFrame) -> pd.DataFrame:
    frame = audit.copy()
    target = KAPPA * frame["gradient_sq_norm"]
    frame["kappa_gradient_sq"] = target
    frame["defect"] = np.maximum(target - frame["net_alignment"], 0.0)
    frame["coverage_deficit"] = np.maximum(
        target - frame["ideal_local_mass"], 0.0
    )
    frame["memory_penalty"] = frame["memory_opposition_penalty"]
    frame["drift_penalty"] = np.maximum(-frame["local_drift_term"], 0.0)
    frame["heterogeneity_penalty"] = np.maximum(
        -frame["heterogeneity_term"], 0.0
    )
    frame["defect_upper_bound"] = frame[
        [
            "coverage_deficit",
            "memory_penalty",
            "drift_penalty",
            "heterogeneity_penalty",
        ]
    ].sum(axis=1)
    frame["defect_bound_slack"] = frame["defect_upper_bound"] - frame["defect"]
    if float(frame["defect_bound_slack"].min()) < -1e-5:
        raise AssertionError("P11 defect upper bound failed")
    return frame


def weighted_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    weights = frame["jump"].to_numpy(float)
    top = float(np.sum(weights * frame[numerator].to_numpy(float)))
    bottom = float(np.sum(weights * frame[denominator].to_numpy(float)))
    return top / bottom if bottom > 0.0 else float("nan")


def summarize(audit: pd.DataFrame) -> dict[str, float | int]:
    late = audit[audit["round"] >= 101]
    denominator = "gradient_sq_norm"
    return {
        "audited_rounds": int(len(audit)),
        "weighted_alignment_ratio": weighted_ratio(
            audit, "net_alignment", denominator
        ),
        "positive_alignment_fraction": float(np.mean(audit["net_alignment"] > 0.0)),
        "objective_decrease_fraction": float(
            np.mean(audit["objective_change"] < 0.0)
        ),
        "positive_without_descent_fraction": float(
            np.mean(
                (audit["net_alignment"] > 0.0)
                & (audit["objective_change"] >= 0.0)
            )
        ),
        "late_alignment_ratio": weighted_ratio(late, "net_alignment", denominator),
        "late_objective_decrease_fraction": float(
            np.mean(late["objective_change"] < 0.0)
        ),
        "harmful_mass_share": float(
            audit["harmful_mass"].sum()
            / (audit["descent_mass"].sum() + audit["harmful_mass"].sum())
        ),
        "event_local_mass_fraction": float(
            audit["ideal_local_mass"].sum()
            / (audit["ideal_local_mass"].sum() + audit["silent_local_mass"].sum())
        ),
        "ideal_local_ratio": weighted_ratio(audit, "ideal_local_mass", denominator),
        "memory_penalty_ratio": weighted_ratio(
            audit, "memory_opposition_penalty", denominator
        ),
        "local_drift_ratio": weighted_ratio(audit, "local_drift_term", denominator),
        "heterogeneity_ratio": weighted_ratio(
            audit, "heterogeneity_term", denominator
        ),
        "mean_curvature_remainder": float(audit["curvature_remainder"].mean()),
        "mean_coordinate_events_per_audit": float(
            audit["coordinate_events"].mean()
        ),
        "mean_aggregate_energy": float(audit["aggregate_energy"].mean()),
        "mean_cancellation_fraction": float(
            audit["cancellation_fraction"].mean()
        ),
        "max_memory_opposition_penalty": float(
            audit["memory_opposition_penalty"].max()
        ),
        "defect_to_target_fraction": weighted_ratio(
            audit, "defect", "kappa_gradient_sq"
        ),
        "max_alignment_identity_error": float(
            audit["alignment_identity_error"].max()
        ),
        "max_decomposition_identity_error": float(
            audit["decomposition_identity_error"].max()
        ),
        "max_update_reconstruction_error": float(
            audit["update_reconstruction_error"].max()
        ),
        "min_defect_bound_slack": float(audit["defect_bound_slack"].min()),
    }


def run_point(
    *,
    regime: str,
    local_steps: int,
    partition_seed: int,
    train_seed: int,
    audit_stride: int,
    reference_per_client: int,
) -> None:
    if regime not in REGIMES:
        raise ValueError(f"unsupported regime: {regime}")
    if partition_seed not in PARTITION_SEEDS:
        raise ValueError(f"unsupported held-out partition seed: {partition_seed}")
    OUT.mkdir(parents=True, exist_ok=True)
    config = experiment_config(local_steps)
    federation = make_multiclass_federation(
        root=DATA,
        regime=regime,
        seed=partition_seed,
    )

    result = run_final_baseline(
        federation=federation,
        architecture="mlp",
        method="event",
        config=config,
        seed=train_seed,
    )

    audit_rows: list[pd.Series] = []
    targets = audited_rounds(config.rounds, audit_stride)
    for target_round in targets:
        snapshot = run_final_baseline(
            federation=federation,
            architecture="mlp",
            method="event",
            config=replace(config, rounds=target_round),
            seed=train_seed,
            alignment_audit_rounds=(target_round,),
            alignment_client_reference_size=reference_per_client,
        )
        trace = snapshot["alignment_audit"]
        if not isinstance(trace, pd.DataFrame) or len(trace) != 1:
            raise AssertionError("independent single-round audit returned invalid trace")
        audit_rows.append(trace.iloc[0])

    audit = add_diagnostics(pd.DataFrame(audit_rows).reset_index(drop=True))
    if len(audit) != len(targets):
        raise AssertionError("audit trace is incomplete")
    if float(audit["decomposition_identity_error"].max()) > 1e-4:
        raise AssertionError("decomposition identity failed")

    summary = summarize(audit)
    summary.update(
        {
            "regime": regime,
            "local_steps": local_steps,
            "partition_seed": partition_seed,
            "train_seed": train_seed,
            "audit_stride": audit_stride,
            "audit_protocol": "independent_single_round_replay",
            "reference_per_client": reference_per_client,
            "final_train_objective": float(result["final_train_objective"]),
            "final_test_ce": float(result["final_test_ce"]),
            "final_test_accuracy": float(result["final_test_accuracy"]),
            "final_worst_class_accuracy": float(
                result["final_worst_class_accuracy"]
            ),
            "coordinate_events_total": int(result["coordinate_events"]),
            "uplink_packetized_bits": int(result["uplink_packetized_bits"]),
            "unicast_hybrid_total_bits": int(
                result["unicast_hybrid_total_bits"]
            ),
        }
    )
    stem = f"{regime}_e{local_steps}_p{partition_seed}_t{train_seed}"
    audit.assign(
        regime=regime,
        local_steps=local_steps,
        partition_seed=partition_seed,
        train_seed=train_seed,
    ).to_csv(OUT / f"rounds_{stem}.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / f"summary_{stem}.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


SUMMARY_METRICS = (
    "weighted_alignment_ratio",
    "positive_alignment_fraction",
    "objective_decrease_fraction",
    "positive_without_descent_fraction",
    "late_alignment_ratio",
    "late_objective_decrease_fraction",
    "harmful_mass_share",
    "event_local_mass_fraction",
    "ideal_local_ratio",
    "memory_penalty_ratio",
    "local_drift_ratio",
    "heterogeneity_ratio",
    "mean_curvature_remainder",
    "mean_coordinate_events_per_audit",
    "mean_aggregate_energy",
    "mean_cancellation_fraction",
    "max_memory_opposition_penalty",
    "defect_to_target_fraction",
    "final_test_ce",
    "final_test_accuracy",
    "final_worst_class_accuracy",
    "coordinate_events_total",
    "uplink_packetized_bits",
    "unicast_hybrid_total_bits",
)


def _validate_complete_design(summaries: pd.DataFrame, rounds: pd.DataFrame) -> None:
    expected = {
        (regime, steps, seed)
        for regime in REGIMES
        for steps in LOCAL_STEPS
        for seed in PARTITION_SEEDS
    }
    observed = set(
        zip(
            summaries["regime"],
            summaries["local_steps"].astype(int),
            summaries["partition_seed"].astype(int),
        )
    )
    if observed != expected or len(summaries) != len(expected):
        raise AssertionError("P11 summaries do not contain the complete 2x2 design")
    counts = rounds.groupby(["regime", "local_steps", "partition_seed"]).size()
    if len(counts) != len(expected) or not np.all(counts.to_numpy() == 31):
        raise AssertionError("every P11 point must contain exactly 31 audit snapshots")
    if float(rounds["alignment_identity_error"].max()) >= 1e-4:
        raise AssertionError("aggregate alignment identity failed")
    if float(rounds["decomposition_identity_error"].max()) >= 1e-4:
        raise AssertionError("alignment decomposition identity failed")
    if float(rounds["update_reconstruction_error"].max()) >= 1e-6:
        raise AssertionError("server update reconstruction failed")


def _aggregate_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (regime, steps), group in summaries.groupby(
        ["regime", "local_steps"], sort=False
    ):
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "regime": regime,
                    "local_steps": int(steps),
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _paired_effects(summaries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    indexed = summaries.set_index(["regime", "local_steps", "partition_seed"])
    comparisons = [
        ("heterogeneity", "iid", 1, "strong", 1),
        ("heterogeneity", "iid", 5, "strong", 5),
        ("local_steps", "iid", 1, "iid", 5),
        ("local_steps", "strong", 1, "strong", 5),
    ]
    for factor, regime_a, steps_a, regime_b, steps_b in comparisons:
        for metric in SUMMARY_METRICS:
            deltas = np.array(
                [
                    float(indexed.loc[(regime_b, steps_b, seed), metric])
                    - float(indexed.loc[(regime_a, steps_a, seed), metric])
                    for seed in PARTITION_SEEDS
                ]
            )
            rows.append(
                {
                    "factor": factor,
                    "from": f"{regime_a}_e{steps_a}",
                    "to": f"{regime_b}_e{steps_b}",
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "mean_paired_difference": float(np.mean(deltas)),
                    "std_paired_difference": float(np.std(deltas, ddof=1)),
                    "min_paired_difference": float(np.min(deltas)),
                    "max_paired_difference": float(np.max(deltas)),
                }
            )
    return pd.DataFrame(rows)


def _plot(aggregates: pd.DataFrame, destination: Path) -> None:
    cells = [(regime, steps) for regime in REGIMES for steps in LOCAL_STEPS]
    labels = ["IID\n$E=1$", "IID\n$E=5$", "non-IID\n$E=1$", "non-IID\n$E=5$"]

    def values(metric: str) -> tuple[np.ndarray, np.ndarray]:
        subset = aggregates[aggregates["metric"] == metric].set_index(
            ["regime", "local_steps"]
        )
        means = np.array([subset.loc[cell, "mean"] for cell in cells], dtype=float)
        stds = np.array([subset.loc[cell, "std"] for cell in cells], dtype=float)
        return means, stds

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    x = np.arange(len(cells))
    alignment, alignment_std = values("weighted_alignment_ratio")
    axes[0].errorbar(
        x,
        alignment,
        yerr=alignment_std,
        fmt="o",
        color="black",
        capsize=3,
        linewidth=1,
    )
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_ylabel(r"trajectory alignment $\widehat{\kappa}_{\mathcal{A}}$")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.18
    components = (
        ("ideal_local_ratio", r"$P$", "0.20"),
        ("local_drift_ratio", r"$L$", "0.45"),
        ("heterogeneity_ratio", r"$B$", "0.72"),
    )
    for offset, (metric, label, color) in enumerate(components):
        means, _ = values(metric)
        axes[1].bar(
            x + (offset - 1) * width,
            means,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"$q_r$-weighted term / $\|\nabla F\|^2$")
    axes[1].set_xticks(x, labels)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=200)
    plt.close(fig)


def aggregate() -> None:
    summary_paths = sorted(OUT.glob("summary_*.csv"))
    round_paths = sorted(OUT.glob("rounds_*.csv"))
    if not summary_paths or not round_paths:
        raise RuntimeError("no P11 point results found")
    summaries = pd.concat(
        [pd.read_csv(path) for path in summary_paths], ignore_index=True
    )
    rounds = pd.concat([pd.read_csv(path) for path in round_paths], ignore_index=True)
    _validate_complete_design(summaries, rounds)
    summaries = summaries.sort_values(
        ["regime", "local_steps", "partition_seed"]
    ).reset_index(drop=True)
    rounds = rounds.sort_values(
        ["regime", "local_steps", "partition_seed", "round"]
    ).reset_index(drop=True)
    aggregates = _aggregate_metrics(summaries)
    effects = _paired_effects(summaries)
    summaries.to_csv(OUT / "combined_summary.csv", index=False)
    rounds.to_csv(OUT / "combined_rounds.csv", index=False)
    aggregates.to_csv(OUT / "aggregate_metrics.csv", index=False)
    effects.to_csv(OUT / "paired_effects.csv", index=False)
    _plot(aggregates, OUT / "p11_alignment_factorial")
    print(aggregates.to_string(index=False))
    print(effects.to_string(index=False))


def smoke_test() -> None:
    rng = np.random.default_rng(20260905)
    client_X = tuple(rng.normal(size=(32, 784)).astype(np.float32) for _ in range(10))
    client_y = tuple(rng.integers(0, 10, size=32, dtype=np.int64) for _ in range(10))
    federation = MulticlassFederation(
        client_X=client_X,
        client_y=client_y,
        X_train_eval=rng.normal(size=(64, 784)).astype(np.float32),
        y_train_eval=rng.integers(0, 10, size=64, dtype=np.int64),
        X_test=rng.normal(size=(64, 784)).astype(np.float32),
        y_test=rng.integers(0, 10, size=64, dtype=np.int64),
        regime="smoke",
        client_class_counts=np.zeros((10, 10), dtype=int),
        periods=np.ones(10, dtype=int),
        weights=np.full(10, 0.1),
    )
    for local_steps in LOCAL_STEPS:
        config = replace(
            experiment_config(local_steps),
            batch_size=8,
            rounds=4,
            eval_stride=2,
        )
        plain = run_final_baseline(
            federation=federation,
            architecture="mlp",
            method="event",
            config=config,
            seed=1235,
        )
        audited = run_final_baseline(
            federation=federation,
            architecture="mlp",
            method="event",
            config=config,
            seed=1235,
            alignment_audit_rounds=(config.rounds,),
            alignment_client_reference_size=16,
        )
        audit = audited.pop("alignment_audit")
        audited.pop("alignment_audit_rounds")
        audited.pop("alignment_reference_size")
        audited.pop("alignment_client_reference_size")
        if plain.keys() != audited.keys():
            raise AssertionError("audit instrumentation changed the result schema")
        for key in plain:
            left, right = plain[key], audited[key]
            if isinstance(left, float):
                if not np.isclose(left, right, equal_nan=True):
                    raise AssertionError(f"audit changed trajectory metric {key}")
            elif left != right:
                raise AssertionError(f"audit changed trajectory metric {key}")
        audit = add_diagnostics(audit)
        if float(audit["alignment_identity_error"].max()) >= 1e-4:
            raise AssertionError("alignment identity failed")
        if float(audit["decomposition_identity_error"].max()) >= 1e-4:
            raise AssertionError("decomposition identity failed")
        if float(audit["update_reconstruction_error"].max()) >= 1e-6:
            raise AssertionError("update reconstruction failed")
    if audited_rounds(150, 5) != (1, *range(5, 151, 5)):
        raise AssertionError("audit-round contract drift")
    print("P11 smoke test passed for E=1 and E=5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=REGIMES, default="strong")
    parser.add_argument("--local-steps", choices=LOCAL_STEPS, type=int, default=5)
    parser.add_argument("--partition-seed", choices=PARTITION_SEEDS, type=int, default=2500)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--audit-stride", type=int, default=5)
    parser.add_argument("--reference-per-client", type=int, default=6000)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    elif args.aggregate:
        aggregate()
    else:
        run_point(
            regime=args.regime,
            local_steps=args.local_steps,
            partition_seed=args.partition_seed,
            train_seed=args.train_seed or 70000 + args.partition_seed,
            audit_stride=args.audit_stride,
            reference_per_client=args.reference_per_client,
        )


if __name__ == "__main__":
    main()
