from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.mechanism_audit import (
    AuditConfig,
    AuditProblem,
    check_fedlif_error_feedback_equivalence,
    check_fullreset_ema_equivalence,
    pareto_mask,
    run_audit_batch,
)

OUT = ROOT / "experiments" / "results" / "09_mechanism_audit"

R = 40
D = 0.05
PROBLEM = AuditProblem(
    thetas=np.array([D, -D]),
    weights=np.array([0.5, 0.5]),
    periods=np.array([R, 1]),
    gamma=0.05,
    w0_low=0.93,
    w0_high=1.13,
)

SIGMAS = [0.0, 0.25]
Q_VALUES = [0.05, 0.10, 0.20]
RHO_VALUES = [1.0, 0.999, 0.995, 0.99]
ACC_THRESHOLDS = [0.35, 0.50, 0.70, 0.90]
EMA_BETAS = [0.90, 0.98, 0.995]
MEMORYLESS_THRESHOLDS = [0.03, 0.05, 0.08, 0.12]
FULL_PRECISION_STEPS = [0.0025, 0.005, 0.01, 0.02]
BASE_SEED = 9020260829


def evaluate(config: AuditConfig, *, sigma: float, n_ticks: int, n_seeds: int, tail: int):
    return run_audit_batch(
        problem=PROBLEM,
        config=config,
        noise_std=sigma,
        n_ticks=n_ticks,
        n_seeds=n_seeds,
        tail=tail,
        seed=BASE_SEED + int(1000 * sigma),
    )


def run_grid(*, n_ticks: int, n_seeds: int, quick: bool) -> pd.DataFrame:
    rows: list[dict] = []

    q_values = [0.05, 0.10] if quick else Q_VALUES
    rho_values = [1.0, 0.995, 0.99] if quick else RHO_VALUES
    acc_thresholds = [0.50, 0.90] if quick else ACC_THRESHOLDS
    ema_betas = [0.90, 0.995] if quick else EMA_BETAS
    memoryless_thresholds = [0.05, 0.12] if quick else MEMORYLESS_THRESHOLDS
    fp_steps = [0.0025, 0.01] if quick else FULL_PRECISION_STEPS
    sigma_values = [0.25] if quick else SIGMAS
    tail = min(1200, max(200, n_ticks // 3))

    for sigma in sigma_values:
        for jump in q_values:
            for rho in rho_values:
                for threshold in acc_thresholds:
                    for family, method in [
                        ("FedLIF subtractive", "fedlif"),
                        ("LIF full reset", "lif_full_reset"),
                    ]:
                        cfg = AuditConfig(
                            method=method,
                            rho=rho,
                            threshold=threshold,
                            jump=jump,
                        )
                        rows.append(
                            {
                                "sigma": sigma,
                                "family": family,
                                "jump": jump,
                                "rho": rho,
                                "threshold": threshold,
                                "ema_beta": np.nan,
                                "full_precision_step": np.nan,
                                **evaluate(
                                    cfg,
                                    sigma=sigma,
                                    n_ticks=n_ticks,
                                    n_seeds=n_seeds,
                                    tail=tail,
                                ),
                            }
                        )

            for threshold in acc_thresholds:
                cfg = AuditConfig(
                    method="if_remote_reset",
                    rho=1.0,
                    threshold=threshold,
                    jump=jump,
                )
                rows.append(
                    {
                        "sigma": sigma,
                        "family": "IF remote reset",
                        "jump": jump,
                        "rho": 1.0,
                        "threshold": threshold,
                        "ema_beta": np.nan,
                        "full_precision_step": np.nan,
                        **evaluate(
                            cfg,
                            sigma=sigma,
                            n_ticks=n_ticks,
                            n_seeds=n_seeds,
                            tail=tail,
                        ),
                    }
                )

            for threshold in memoryless_thresholds:
                cfg = AuditConfig(
                    method="memoryless_pulse",
                    threshold=threshold,
                    jump=jump,
                )
                rows.append(
                    {
                        "sigma": sigma,
                        "family": "Memoryless pulse",
                        "jump": jump,
                        "rho": np.nan,
                        "threshold": threshold,
                        "ema_beta": np.nan,
                        "full_precision_step": np.nan,
                        **evaluate(
                            cfg,
                            sigma=sigma,
                            n_ticks=n_ticks,
                            n_seeds=n_seeds,
                            tail=tail,
                        ),
                    }
                )

            for beta in ema_betas:
                for threshold in memoryless_thresholds:
                    cfg = AuditConfig(
                        method="ema_pulse",
                        threshold=threshold,
                        jump=jump,
                        ema_beta=beta,
                    )
                    rows.append(
                        {
                            "sigma": sigma,
                            "family": "EMA + memoryless pulse",
                            "jump": jump,
                            "rho": np.nan,
                            "threshold": threshold,
                            "ema_beta": beta,
                            "full_precision_step": np.nan,
                            **evaluate(
                                cfg,
                                sigma=sigma,
                                n_ticks=n_ticks,
                                n_seeds=n_seeds,
                                tail=tail,
                            ),
                        }
                    )

            cfg = AuditConfig(method="periodic_sign", jump=jump)
            rows.append(
                {
                    "sigma": sigma,
                    "family": "Periodic sign",
                    "jump": jump,
                    "rho": np.nan,
                    "threshold": np.nan,
                    "ema_beta": np.nan,
                    "full_precision_step": np.nan,
                    **evaluate(
                        cfg,
                        sigma=sigma,
                        n_ticks=n_ticks,
                        n_seeds=n_seeds,
                        tail=tail,
                    ),
                }
            )

        for step in fp_steps:
            cfg = AuditConfig(method="full_precision", full_precision_step=step)
            rows.append(
                {
                    "sigma": sigma,
                    "family": "Full precision",
                    "jump": np.nan,
                    "rho": np.nan,
                    "threshold": np.nan,
                    "ema_beta": np.nan,
                    "full_precision_step": step,
                    **evaluate(
                        cfg,
                        sigma=sigma,
                        n_ticks=n_ticks,
                        n_seeds=n_seeds,
                        tail=tail,
                    ),
                }
            )

    return pd.DataFrame(rows)


def add_pareto_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_rows = []
    global_rows = []
    for sigma in sorted(df["sigma"].unique()):
        sigma_df = df[df["sigma"] == sigma]
        global_front = sigma_df[
            pareto_mask(sigma_df["mean_events"], sigma_df["tail_rmse"])
        ].copy()
        global_rows.append(global_front)
        for _, family_df in sigma_df.groupby("family"):
            family_front = family_df[
                pareto_mask(family_df["mean_events"], family_df["tail_rmse"])
            ].copy()
            family_rows.append(family_front)
    return pd.concat(family_rows, ignore_index=True), pd.concat(global_rows, ignore_index=True)


def summarize_best(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sigma in sorted(df["sigma"].unique()):
        for family, family_df in df[df["sigma"] == sigma].groupby("family"):
            row = family_df.loc[family_df["tail_rmse"].idxmin()]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["sigma", "tail_rmse"])


def save_pareto_plot(df: pd.DataFrame, family_pareto: pd.DataFrame, *, sigma: float):
    plt.figure(figsize=(9, 6))
    sigma_df = df[df["sigma"] == sigma]
    for family, family_df in sigma_df.groupby("family"):
        if family == "Full precision":
            continue
        plt.scatter(family_df["mean_events"], family_df["tail_rmse"], alpha=0.35, s=20, label=family)
        front = family_pareto[
            (family_pareto["sigma"] == sigma) & (family_pareto["family"] == family)
        ].sort_values("mean_events")
        if len(front) > 1:
            plt.plot(front["mean_events"], front["tail_rmse"], linewidth=1.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean transmitted signed events")
    plt.ylabel("Tail RMSE")
    plt.title(f"Experiment 09: error--communication trade-off (sigma={sigma})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"pareto_sigma_{str(sigma).replace('.', 'p')}.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=4500)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n_ticks = min(args.ticks, 1000) if args.quick else args.ticks
    n_seeds = min(args.seeds, 24) if args.quick else args.seeds

    OUT.mkdir(parents=True, exist_ok=True)
    grid = run_grid(n_ticks=n_ticks, n_seeds=n_seeds, quick=args.quick)
    family_pareto, global_pareto = add_pareto_labels(grid)
    best = summarize_best(grid)

    equivalence = pd.DataFrame(
        [
            {
                "mapping": "FedLIF subtractive <-> decayed error feedback",
                **check_fedlif_error_feedback_equivalence(),
            },
            {
                "mapping": "LIF full reset <-> reset EMA trigger",
                **check_fullreset_ema_equivalence(),
            },
        ]
    )

    grid.to_csv(OUT / "mechanism_grid.csv", index=False)
    family_pareto.to_csv(OUT / "family_pareto.csv", index=False)
    global_pareto.to_csv(OUT / "global_pareto.csv", index=False)
    best.to_csv(OUT / "best_by_family.csv", index=False)
    equivalence.to_csv(OUT / "equivalence_checks.csv", index=False)

    for sigma in sorted(grid["sigma"].unique()):
        save_pareto_plot(grid, family_pareto, sigma=sigma)

    print("Best configuration per family")
    print(
        best[
            [
                "sigma",
                "family",
                "jump",
                "rho",
                "threshold",
                "ema_beta",
                "full_precision_step",
                "mean_events",
                "tail_rmse",
                "harmful_fraction",
            ]
        ].to_string(index=False)
    )
    print("\nExact-equivalence checks")
    print(equivalence.to_string(index=False))
    print(f"\nResults saved under {OUT}")


if __name__ == "__main__":
    main()
