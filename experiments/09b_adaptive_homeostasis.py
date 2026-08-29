from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.homeostatic import (
    AdaptiveThresholdConfig,
    run_homeostatic_batch,
)
from neuromorphicfl.mechanism_audit import AuditProblem, pareto_mask

OUT = ROOT / "experiments" / "results" / "09b_adaptive_homeostasis"

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

RHO = 0.999
JUMP = 0.05
SIGMA = 0.25
BASE_SEED = 9520260829
TAIL = 1200

FIXED_THRESHOLDS = [0.25, 0.35, 0.50, 0.70, 0.90, 1.20, 1.50, 2.00, 2.50, 3.00, 4.00]
BASE_THRESHOLDS = [0.25, 0.35, 0.50, 0.70, 0.90]
ADAPT_INCREMENTS = [0.10, 0.25, 0.50, 1.00]
ADAPT_TAUS = [50.0, 200.0, 500.0]


def evaluate(config: AdaptiveThresholdConfig, *, n_ticks: int, n_seeds: int,
             noise_std: float = SIGMA, noise_std_after: float | None = None):
    return run_homeostatic_batch(
        problem=PROBLEM,
        config=config,
        noise_std=noise_std,
        noise_std_after=noise_std_after,
        noise_switch_tick=n_ticks // 2,
        n_ticks=n_ticks,
        n_seeds=n_seeds,
        tail=min(TAIL, max(200, n_ticks // 3)),
        seed=BASE_SEED,
    )


def row_from_result(*, family: str, base_threshold: float,
                    adapt_increment: float, adapt_tau: float | None, result):
    return {
        "family": family,
        "base_threshold": base_threshold,
        "adapt_increment": adapt_increment,
        "adapt_tau": adapt_tau,
        "mean_events": result.mean_events,
        "mean_slow_events": result.mean_slow_events,
        "tail_rmse": result.tail_rmse,
        "whole_mse": result.whole_mse,
        "harmful_fraction": result.harmful_fraction,
        "tail_threshold_slow": result.mean_tail_threshold_slow,
        "tail_threshold_fast": result.mean_tail_threshold_fast,
    }


def run_stationary_grid(*, n_ticks: int, n_seeds: int, quick: bool):
    rows = []
    fixed_thresholds = [0.5, 1.5, 2.5] if quick else FIXED_THRESHOLDS
    base_thresholds = [0.5, 0.9] if quick else BASE_THRESHOLDS
    increments = [0.25, 1.0] if quick else ADAPT_INCREMENTS
    taus = [200.0] if quick else ADAPT_TAUS

    for threshold in fixed_thresholds:
        cfg = AdaptiveThresholdConfig(
            rho=RHO,
            base_threshold=threshold,
            adapt_increment=0.0,
            adapt_tau=200.0,
            jump=JUMP,
        )
        result = evaluate(cfg, n_ticks=n_ticks, n_seeds=n_seeds)
        rows.append(
            row_from_result(
                family="Fixed full-reset LIF",
                base_threshold=threshold,
                adapt_increment=0.0,
                adapt_tau=np.nan,
                result=result,
            )
        )

    for base_threshold in base_thresholds:
        for increment in increments:
            for tau in taus:
                cfg = AdaptiveThresholdConfig(
                    rho=RHO,
                    base_threshold=base_threshold,
                    adapt_increment=increment,
                    adapt_tau=tau,
                    jump=JUMP,
                )
                result = evaluate(cfg, n_ticks=n_ticks, n_seeds=n_seeds)
                rows.append(
                    row_from_result(
                        family="Adaptive-threshold LIF",
                        base_threshold=base_threshold,
                        adapt_increment=increment,
                        adapt_tau=tau,
                        result=result,
                    )
                )

    return pd.DataFrame(rows)


def select_under_budget(df: pd.DataFrame, *, family: str, budget: float):
    feasible = df[(df["family"] == family) & (df["mean_events"] <= budget)]
    if feasible.empty:
        return None
    return feasible.loc[feasible["tail_rmse"].idxmin()]


def run_noise_shift(*, grid: pd.DataFrame, n_ticks: int, n_seeds: int):
    # Compare methods selected under the same nominal scalar communication budget.
    budget = 30.0
    fixed = select_under_budget(grid, family="Fixed full-reset LIF", budget=budget)
    adaptive = select_under_budget(grid, family="Adaptive-threshold LIF", budget=budget)
    if fixed is None or adaptive is None:
        return pd.DataFrame(), {}

    configs = {
        "Fixed nominal": AdaptiveThresholdConfig(
            rho=RHO,
            base_threshold=float(fixed["base_threshold"]),
            adapt_increment=0.0,
            adapt_tau=200.0,
            jump=JUMP,
        ),
        "Adaptive": AdaptiveThresholdConfig(
            rho=RHO,
            base_threshold=float(adaptive["base_threshold"]),
            adapt_increment=float(adaptive["adapt_increment"]),
            adapt_tau=float(adaptive["adapt_tau"]),
            jump=JUMP,
        ),
    }

    rows = []
    traces = {}
    for label, config in configs.items():
        result = evaluate(
            config,
            n_ticks=n_ticks,
            n_seeds=n_seeds,
            noise_std=0.10,
            noise_std_after=0.60,
        )
        rows.append(
            {
                "family": label,
                "base_threshold": config.base_threshold,
                "adapt_increment": config.adapt_increment,
                "adapt_tau": config.adapt_tau,
                "mean_events": result.mean_events,
                "mean_slow_events": result.mean_slow_events,
                "tail_rmse": result.tail_rmse,
                "whole_mse": result.whole_mse,
                "harmful_fraction": result.harmful_fraction,
                "pre_switch_event_rate": result.pre_switch_event_rate,
                "post_switch_event_rate": result.post_switch_event_rate,
            }
        )
        traces[label] = result.threshold_trace
    return pd.DataFrame(rows), traces


def save_pareto(grid: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    for family, sub in grid.groupby("family"):
        front = sub[pareto_mask(sub["mean_events"], sub["tail_rmse"])].sort_values("mean_events")
        plt.scatter(sub["mean_events"], sub["tail_rmse"], alpha=0.25, s=18)
        plt.plot(front["mean_events"], front["tail_rmse"], marker="o", label=family)
    plt.xlabel("Mean transmitted events")
    plt.ylabel("Tail RMSE")
    plt.title("Experiment 09b: adaptive threshold versus fixed threshold")
    plt.tight_layout()
    plt.legend()
    plt.savefig(OUT / "adaptive_threshold_pareto.png", dpi=180)
    plt.close()


def save_noise_shift(traces: dict[str, np.ndarray], *, n_ticks: int):
    if not traces:
        return
    plt.figure(figsize=(9, 5))
    x = np.arange(n_ticks + 1)
    for label, trace in traces.items():
        plt.plot(x, trace[0], label=f"{label}, slow")
        plt.plot(x, trace[1], label=f"{label}, fast")
    plt.axvline(n_ticks // 2, linestyle="--")
    plt.xlabel("Wall-clock tick")
    plt.ylabel("Mean firing threshold")
    plt.title("Experiment 09b: threshold response to noise increase")
    plt.tight_layout()
    plt.legend()
    plt.savefig(OUT / "noise_shift_thresholds.png", dpi=180)
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

    grid = run_stationary_grid(n_ticks=n_ticks, n_seeds=n_seeds, quick=args.quick)
    grid.to_csv(OUT / "stationary_grid.csv", index=False)
    save_pareto(grid)

    noise_shift, traces = run_noise_shift(grid=grid, n_ticks=n_ticks, n_seeds=n_seeds)
    noise_shift.to_csv(OUT / "noise_shift.csv", index=False)
    save_noise_shift(traces, n_ticks=n_ticks)

    global_front = grid[pareto_mask(grid["mean_events"], grid["tail_rmse"])].sort_values("mean_events")
    global_front.to_csv(OUT / "global_pareto.csv", index=False)

    print("Global stationary Pareto front")
    print(global_front.to_string(index=False))
    print("\nNoise-shift comparison")
    print(noise_shift.to_string(index=False))
    print(f"\nResults saved under {OUT}")


if __name__ == "__main__":
    main()
