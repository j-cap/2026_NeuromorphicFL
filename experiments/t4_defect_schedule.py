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
OUT = Path("experiments/results/t4_defect_schedule")
KAPPA = 8.0
VARIANTS = ("baseline", "admissible", "admissible_matched")


def schedule_sum(rounds: int, scale: float, exponent: float) -> float:
    indices = np.arange(1, rounds + 1, dtype=float)
    return float(np.sum((1.0 + indices / scale) ** (-exponent)))


def variant_config(variant: str) -> FinalBaselineConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    rounds = 150
    scale = 100.0
    baseline_exponent = 0.1
    admissible_exponent = 0.6
    jump0 = 0.005
    if variant == "admissible_matched":
        jump0 *= schedule_sum(rounds, scale, baseline_exponent) / schedule_sum(
            rounds, scale, admissible_exponent
        )
    exponent = baseline_exponent if variant == "baseline" else admissible_exponent
    return FinalBaselineConfig(
        local_steps=5,
        local_lr=0.1,
        rounds=rounds,
        eval_stride=15,
        rho=0.999,
        threshold=0.025,
        jump0=jump0,
        jump_scale=scale,
        jump_exponent=exponent,
    )


def add_defect_components(audit: pd.DataFrame) -> pd.DataFrame:
    frame = audit.copy()
    target = KAPPA * frame["gradient_sq_norm"]
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
        raise AssertionError("T4 defect upper bound failed")
    return frame


def weighted_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    weights = frame["jump"].to_numpy(float)
    top = float(np.sum(weights * frame[numerator].to_numpy(float)))
    bottom = float(np.sum(weights * frame[denominator].to_numpy(float)))
    return top / bottom if bottom > 0.0 else float("nan")


def summarize(audit: pd.DataFrame) -> dict[str, float | int]:
    late = audit[audit["round"] >= 101]
    target_column = "kappa_gradient_sq"
    audit = audit.assign(**{target_column: KAPPA * audit["gradient_sq_norm"]})
    late = late.assign(**{target_column: KAPPA * late["gradient_sq_norm"]})
    return {
        "audited_rounds": int(len(audit)),
        "weighted_alignment_ratio": weighted_ratio(
            audit, "net_alignment", "gradient_sq_norm"
        ),
        "net_positive_fraction": float(np.mean(audit["net_alignment"] > 0.0)),
        "objective_decrease_fraction": float(
            np.mean(audit["objective_change"] < 0.0)
        ),
        "late_alignment_ratio": weighted_ratio(
            late, "net_alignment", "gradient_sq_norm"
        ),
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
        "defect_to_target_fraction": weighted_ratio(
            audit, "defect", target_column
        ),
        "coverage_to_target_fraction": weighted_ratio(
            audit, "coverage_deficit", target_column
        ),
        "memory_to_target_fraction": weighted_ratio(
            audit, "memory_penalty", target_column
        ),
        "drift_to_target_fraction": weighted_ratio(
            audit, "drift_penalty", target_column
        ),
        "heterogeneity_to_target_fraction": weighted_ratio(
            audit, "heterogeneity_penalty", target_column
        ),
        "mean_curvature_remainder": float(audit["curvature_remainder"].mean()),
        "late_mean_curvature_remainder": float(late["curvature_remainder"].mean()),
        "max_decomposition_identity_error": float(
            audit["decomposition_identity_error"].max()
        ),
        "min_defect_bound_slack": float(audit["defect_bound_slack"].min()),
    }


def run_point(
    variant: str,
    partition_seed: int,
    train_seed: int,
    audit_stride: int,
    reference_per_client: int,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = variant_config(variant)
    federation = make_multiclass_federation(
        root=DATA,
        regime="strong",
        seed=partition_seed,
    )
    result = run_final_baseline(
        federation=federation,
        architecture="mlp",
        method="event",
        config=config,
        seed=train_seed,
    )
    audit_rounds = sorted(
        {
            1,
            config.rounds,
            *range(audit_stride, config.rounds + 1, audit_stride),
        }
    )
    audit_rows: list[pd.Series] = []
    for target_round in audit_rounds:
        snapshot = run_final_baseline(
            federation=federation,
            architecture="mlp",
            method="event",
            config=replace(config, rounds=target_round),
            seed=train_seed,
            alignment_audit_rounds=(target_round,),
            alignment_client_reference_size=reference_per_client,
        )
        snapshot_audit = snapshot["alignment_audit"]
        assert isinstance(snapshot_audit, pd.DataFrame)
        if len(snapshot_audit) != 1:
            raise AssertionError("single-round audit produced an invalid trace")
        audit_rows.append(snapshot_audit.iloc[0])
    audit = pd.DataFrame(audit_rows).reset_index(drop=True)
    audit = add_defect_components(audit)
    summary = summarize(audit)
    summary.update(
        {
            "variant": variant,
            "partition_seed": partition_seed,
            "train_seed": train_seed,
            "audit_stride": audit_stride,
            "audit_protocol": "independent_single_round_replay",
            "reference_per_client": reference_per_client,
            "jump0": config.jump0,
            "jump_exponent": config.jump_exponent,
            "finite_horizon_quantum_sum": config.jump0
            * schedule_sum(config.rounds, config.jump_scale, config.jump_exponent),
            "asymptotically_admissible": int(0.5 < config.jump_exponent <= 1.0),
            "final_train_objective": float(result["final_train_objective"]),
            "final_test_accuracy": float(result["final_test_accuracy"]),
            "final_worst_class_accuracy": float(
                result["final_worst_class_accuracy"]
            ),
            "coordinate_events_total": int(result["coordinate_events"]),
            "uplink_packetized_bits": int(result["uplink_packetized_bits"]),
        }
    )
    stem = f"{variant}_p{partition_seed}_t{train_seed}"
    audit.assign(
        variant=variant,
        partition_seed=partition_seed,
        train_seed=train_seed,
    ).to_csv(OUT / f"rounds_{stem}.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / f"summary_{stem}.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


def aggregate() -> None:
    summary_paths = sorted(OUT.glob("summary_*_p*.csv"))
    round_paths = sorted(OUT.glob("rounds_*_p*.csv"))
    if not summary_paths or not round_paths:
        raise RuntimeError("no T4 point results found")
    summaries = pd.concat(
        [pd.read_csv(path) for path in summary_paths], ignore_index=True
    )
    rounds = pd.concat([pd.read_csv(path) for path in round_paths], ignore_index=True)
    summaries.to_csv(OUT / "combined_summary.csv", index=False)
    rounds.to_csv(OUT / "combined_rounds.csv", index=False)

    metrics = [
        "weighted_alignment_ratio",
        "late_alignment_ratio",
        "objective_decrease_fraction",
        "late_objective_decrease_fraction",
        "defect_to_target_fraction",
        "coverage_to_target_fraction",
        "memory_to_target_fraction",
        "drift_to_target_fraction",
        "heterogeneity_to_target_fraction",
        "event_local_mass_fraction",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "coordinate_events_total",
    ]
    rows: list[dict[str, float | str]] = []
    for variant, group in summaries.groupby("variant", sort=False):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    aggregate_metrics = pd.DataFrame(rows)
    aggregate_metrics.to_csv(OUT / "aggregate_metrics.csv", index=False)

    round_means = (
        rounds.groupby(["variant", "round"], as_index=False)
        .agg(
            alignment_ratio=("alignment_ratio", "mean"),
            objective_change=("objective_change", "mean"),
        )
    )
    labels = {
        "baseline": r"baseline $\alpha=0.1$",
        "admissible": r"admissible $\alpha=0.6$",
        "admissible_matched": r"matched $\alpha=0.6$",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for variant in VARIANTS:
        group = round_means[round_means["variant"] == variant]
        axes[0].plot(
            group["round"], group["alignment_ratio"], marker="o", markersize=2,
            label=labels[variant],
        )
        axes[1].plot(
            group["round"], group["objective_change"], marker="o", markersize=2,
            label=labels[variant],
        )
    axes[0].axhline(0.0, color="0.2", linewidth=0.8)
    axes[0].set_title("Mean alignment ratio")
    axes[0].set_ylabel(r"$A_r/\|\nabla F_r\|_2^2$")
    axes[1].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1].set_title("Mean objective change")
    axes[1].set_ylabel(r"$F(w^{r+1})-F(w^r)$")

    component_columns = [
        "coverage_to_target_fraction",
        "memory_to_target_fraction",
        "drift_to_target_fraction",
        "heterogeneity_to_target_fraction",
    ]
    component_labels = ["coverage", "memory", "drift/noise", "heterogeneity"]
    x = np.arange(len(VARIANTS), dtype=float)
    bottom = np.zeros(len(VARIANTS), dtype=float)
    for column, label in zip(component_columns, component_labels):
        values = np.array(
            [
                summaries.loc[summaries["variant"] == variant, column].mean()
                for variant in VARIANTS
            ]
        )
        axes[2].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[2].set_xticks(x, ["base", "admiss.", "matched"])
    axes[2].set_title(r"Defect-bound components ($\kappa=8$)")
    axes[2].set_ylabel("weighted fraction of target")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes[:2]:
        axis.set_xlabel("round")
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "t4_defect_schedule_summary.png", dpi=180)
    fig.savefig(OUT / "t4_defect_schedule_summary.pdf")
    plt.close(fig)
    print(aggregate_metrics.to_string(index=False))


def smoke_test() -> None:
    rng = np.random.default_rng(904)
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
    config = replace(
        variant_config("baseline"),
        local_steps=2,
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
    assert plain.keys() == audited.keys()
    for key in plain:
        left, right = plain[key], audited[key]
        if isinstance(left, float):
            assert np.isclose(left, right, equal_nan=True), key
        else:
            assert left == right, key
    audit = add_defect_components(audit)
    assert float(audit["decomposition_identity_error"].max()) < 1e-4
    assert float(audit["alignment_identity_error"].max()) < 1e-4
    assert float(audit["update_reconstruction_error"].max()) < 1e-6
    assert float(audit["defect_bound_slack"].min()) >= -1e-5
    baseline = variant_config("baseline")
    matched = variant_config("admissible_matched")
    baseline_sum = baseline.jump0 * schedule_sum(
        baseline.rounds, baseline.jump_scale, baseline.jump_exponent
    )
    matched_sum = matched.jump0 * schedule_sum(
        matched.rounds, matched.jump_scale, matched.jump_exponent
    )
    assert np.isclose(baseline_sum, matched_sum)
    print("T4 smoke test passed; decomposition, bound, and trajectory are valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="baseline")
    parser.add_argument("--partition-seed", type=int, default=2500)
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
        train_seed = args.train_seed or 70000 + args.partition_seed
        run_point(
            args.variant,
            args.partition_seed,
            train_seed,
            args.audit_stride,
            args.reference_per_client,
        )


if __name__ == "__main__":
    main()
