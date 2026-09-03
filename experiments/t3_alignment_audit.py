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
OUT = Path("experiments/results/t3_alignment_audit")


def frozen_mlp_config() -> FinalBaselineConfig:
    return FinalBaselineConfig(
        local_steps=5,
        local_lr=0.1,
        rounds=150,
        eval_stride=15,
        rho=0.999,
        threshold=0.025,
        jump0=0.005,
        jump_scale=100.0,
        jump_exponent=0.1,
    )


def summarize(audit: pd.DataFrame) -> dict[str, float | int]:
    q_weight = audit["jump"].to_numpy(float)
    grad_sq = audit["gradient_sq_norm"].to_numpy(float)
    net = audit["net_alignment"].to_numpy(float)
    ratios = audit["alignment_ratio"].to_numpy(float)
    alignment_denominator = float(np.sum(q_weight * grad_sq))
    weighted_alignment_ratio = (
        float(np.sum(q_weight * net) / alignment_denominator)
        if alignment_denominator > 0.0
        else float("nan")
    )

    # This is an audit calibration, not a theorem constant: test whether half
    # the observed q-weighted mean alignment can be supported with a small
    # realized defect burden.
    kappa_candidate = 0.5 * max(weighted_alignment_ratio, 0.0)
    beta = np.maximum(kappa_candidate * grad_sq - net, 0.0)
    beta_budget = float(np.sum(q_weight * beta))
    target_budget = float(kappa_candidate * alignment_denominator)

    mass = audit["descent_mass"].to_numpy(float) + audit["harmful_mass"].to_numpy(float)
    harmful_mass = audit["harmful_mass"].to_numpy(float)
    return {
        "audited_rounds": int(len(audit)),
        "reference_size": int(audit["reference_size"].iloc[0]),
        "weighted_alignment_ratio": weighted_alignment_ratio,
        "alignment_ratio_min": float(np.nanmin(ratios)),
        "alignment_ratio_q10": float(np.nanquantile(ratios, 0.10)),
        "alignment_ratio_median": float(np.nanmedian(ratios)),
        "alignment_ratio_q90": float(np.nanquantile(ratios, 0.90)),
        "net_positive_fraction": float(np.mean(net > 0.0)),
        "objective_decrease_fraction": float(np.mean(audit["objective_change"] < 0.0)),
        "harmful_mass_share": (
            float(np.sum(harmful_mass) / np.sum(mass))
            if float(np.sum(mass)) > 0.0
            else float("nan")
        ),
        "mean_events": float(np.mean(audit["coordinate_events"])),
        "mean_cancellation_fraction": float(np.mean(audit["cancellation_fraction"])),
        "mean_event_energy_ratio": float(np.mean(audit["event_energy_ratio"])),
        "kappa_candidate": float(kappa_candidate),
        "weighted_beta_budget": beta_budget,
        "beta_to_target_fraction": (
            beta_budget / target_budget if target_budget > 0.0 else float("nan")
        ),
        "max_alignment_identity_error": float(
            np.max(audit["alignment_identity_error"])
        ),
        "max_update_reconstruction_error": float(
            np.max(audit["update_reconstruction_error"])
        ),
    }


def plot_single(audit: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    rounds = audit["round"]

    axes[0, 0].axhline(0.0, color="0.3", linewidth=0.8)
    axes[0, 0].plot(rounds, audit["alignment_ratio"], marker="o", markersize=3)
    axes[0, 0].set_ylabel(r"$(D_r-H_r)/\|\nabla F_r\|_2^2$")
    axes[0, 0].set_title("Aggregate alignment")

    axes[0, 1].plot(rounds, audit["harmful_share"], marker="o", markersize=3)
    axes[0, 1].axhline(0.5, color="0.3", linewidth=0.8, linestyle="--")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylabel("harmful gradient-mass share")
    axes[0, 1].set_title("Directional composition")

    axes[1, 0].plot(rounds, audit["coordinate_events"], label="client events")
    axes[1, 0].plot(rounds, audit["aggregate_nonzeros"], label="server nonzeros")
    axes[1, 0].set_ylabel("coordinates")
    axes[1, 0].set_title("Events and cancellation")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1, 1].plot(rounds, audit["objective_change"], label=r"$F(w^{r+1})-F(w^r)$")
    axes[1, 1].plot(rounds, audit["first_order_change"], label="first-order term")
    axes[1, 1].set_ylabel("reference-objective change")
    axes[1, 1].set_title("Observed versus first-order change")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xlabel("round")
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_point(partition_seed: int, train_seed: int, audit_stride: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    federation = make_multiclass_federation(
        root=DATA,
        regime="strong",
        seed=partition_seed,
    )
    result = run_final_baseline(
        federation=federation,
        architecture="mlp",
        method="event",
        config=frozen_mlp_config(),
        seed=train_seed,
        alignment_audit_stride=audit_stride,
    )
    audit = result.pop("alignment_audit")
    assert isinstance(audit, pd.DataFrame)
    summary = summarize(audit)
    summary.update(
        {
            "partition_seed": partition_seed,
            "train_seed": train_seed,
            "audit_stride": audit_stride,
            "final_train_objective": float(result["final_train_objective"]),
            "final_test_accuracy": float(result["final_test_accuracy"]),
            "final_worst_class_accuracy": float(result["final_worst_class_accuracy"]),
            "coordinate_events_total": int(result["coordinate_events"]),
        }
    )

    stem = f"mlp_p{partition_seed}_t{train_seed}"
    audit.assign(partition_seed=partition_seed, train_seed=train_seed).to_csv(
        OUT / f"rounds_{stem}.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(OUT / f"summary_{stem}.csv", index=False)
    plot_single(audit, OUT / f"alignment_{stem}", f"T3 alignment audit: seed {partition_seed}")
    print(pd.DataFrame([summary]).to_string(index=False))


def aggregate() -> None:
    summaries = pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("summary_mlp_p*.csv"))],
        ignore_index=True,
    )
    rounds = pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("rounds_mlp_p*.csv"))],
        ignore_index=True,
    )
    if summaries.empty or rounds.empty:
        raise RuntimeError("no point results found")
    summaries.to_csv(OUT / "combined_summary.csv", index=False)
    rounds.to_csv(OUT / "combined_rounds.csv", index=False)

    numeric = [
        "weighted_alignment_ratio",
        "alignment_ratio_q10",
        "alignment_ratio_median",
        "net_positive_fraction",
        "objective_decrease_fraction",
        "harmful_mass_share",
        "mean_cancellation_fraction",
        "beta_to_target_fraction",
        "final_test_accuracy",
    ]
    aggregate_rows = []
    for column in numeric:
        values = summaries[column].to_numpy(float)
        aggregate_rows.append(
            {
                "metric": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    aggregate_summary = pd.DataFrame(aggregate_rows)
    aggregate_summary.to_csv(OUT / "aggregate_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for seed, group in rounds.groupby("partition_seed"):
        axes[0].plot(group["round"], group["alignment_ratio"], marker="o", markersize=2, label=str(seed))
        axes[1].plot(group["round"], group["harmful_share"], marker="o", markersize=2, label=str(seed))
    axes[0].axhline(0.0, color="0.2", linewidth=0.8)
    axes[0].set_title("Aggregate alignment ratio")
    axes[0].set_ylabel(r"$(D_r-H_r)/\|\nabla F_r\|_2^2$")
    axes[1].axhline(0.5, color="0.2", linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Harmful gradient-mass share")
    axes[1].set_ylabel(r"$H_r/(D_r+H_r)$")
    for axis in axes:
        axis.set_xlabel("round")
        axis.grid(alpha=0.2)
    axes[1].legend(title="partition seed", frameon=False)
    fig.savefig(OUT / "t3_alignment_audit_summary.png", dpi=180)
    fig.savefig(OUT / "t3_alignment_audit_summary.pdf")
    plt.close(fig)
    print(aggregate_summary.to_string(index=False))


def smoke_test() -> None:
    rng = np.random.default_rng(903)
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
        frozen_mlp_config(),
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
        seed=1234,
    )
    audited = run_final_baseline(
        federation=federation,
        architecture="mlp",
        method="event",
        config=config,
        seed=1234,
        alignment_audit_stride=1,
    )
    audit = audited.pop("alignment_audit")
    assert isinstance(audit, pd.DataFrame)
    audited.pop("alignment_audit_stride")
    audited.pop("alignment_reference_size")
    assert plain.keys() == audited.keys()
    for key in plain:
        left, right = plain[key], audited[key]
        if isinstance(left, float):
            assert np.isclose(left, right, equal_nan=True), key
        else:
            assert left == right, key
    assert float(audit["alignment_identity_error"].max()) < 1e-4
    assert float(audit["update_reconstruction_error"].max()) < 1e-6
    assert np.all(audit["aggregate_energy"] <= 10 * audit["coordinate_events"] + 1e-12)
    print("T3 alignment audit smoke test passed; instrumentation is trajectory-invariant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-seed", type=int, default=2500)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--audit-stride", type=int, default=5)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    elif args.aggregate:
        aggregate()
    else:
        train_seed = args.train_seed or 70000 + args.partition_seed
        run_point(args.partition_seed, train_seed, args.audit_stride)


if __name__ == "__main__":
    main()
