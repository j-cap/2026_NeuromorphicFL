from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.timing_aware import (
    TimingProblem,
    run_timing_batch,
    timing_quality_tables,
)

OUT = ROOT / "experiments" / "results" / "09c_event_time_aggregation"
BASE_SEED = 9720260829


def problem_for_ratio(ratio: int) -> TimingProblem:
    return TimingProblem(
        thetas=np.array([0.05, -0.05]),
        weights=np.array([0.5, 0.5]),
        periods=np.array([ratio, 1]),
        gamma=0.05,
        rho=0.999,
        threshold=1.5,
        jump=0.05,
        w0_low=0.93,
        w0_high=1.13,
    )


def add_pareto_mask(df: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(df), dtype=bool)
    x = df["mean_events"].to_numpy()
    y = df["tail_rmse"].to_numpy()
    for i in range(len(df)):
        dominated = (
            (x <= x[i])
            & (y <= y[i])
            & ((x < x[i]) | (y < y[i]))
        )
        if np.any(dominated):
            keep[i] = False
    return df[keep].copy().sort_values("mean_events")


def run_experiment(*, ticks: int, seeds: int, quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tail = min(1500, max(300, ticks // 4))

    # ------------------------------------------------------------------
    # 9c-A: Does inter-event timing itself predict event quality?
    # ------------------------------------------------------------------
    calibration_seeds = min(seeds, 300)
    baseline = run_timing_batch(
        problem=problem_for_ratio(40),
        noise_std=0.25,
        n_ticks=ticks,
        n_seeds=calibration_seeds,
        tail=tail,
        seed=BASE_SEED,
        weighting="fixed",
        log_events=True,
    )
    event_log = baseline["event_log"]
    quality, per_client = timing_quality_tables(event_log)

    # Conditioning check: raw wall time may merely proxy distance to the optimum.
    conditioned_rows = []
    w_bins = [0, 0.025, 0.05, 0.10, 0.20, 0.50, np.inf]
    w_labels = ["<=.025", ".025-.05", ".05-.10", ".10-.20", ".20-.50", ">.50"]
    event_log = event_log.copy()
    event_log["absw_bin"] = pd.cut(
        np.abs(event_log["w_before"]),
        bins=w_bins,
        labels=w_labels,
        include_lowest=True,
    )
    for (client, absw_bin), sub in event_log.groupby(
        ["client", "absw_bin"], observed=False
    ):
        if len(sub) < 50:
            continue
        conditioned_rows.append(
            {
                "client": int(client),
                "absw_bin": str(absw_bin),
                "events": len(sub),
                "harm_fraction": sub["global_harmful"].mean(),
                "tau_local_harm_spearman": sub["tau_local"].corr(
                    sub["global_harmful"].astype(float), method="spearman"
                ),
                "tau_wall_harm_spearman": sub["tau_wall"].corr(
                    sub["global_harmful"].astype(float), method="spearman"
                ),
            }
        )
    conditioned = pd.DataFrame(conditioned_rows)

    # ------------------------------------------------------------------
    # 9c-B: use timing as a jump-size signal, then run controls.
    # ------------------------------------------------------------------
    eval_seeds = min(seeds, 500)
    local_pool = event_log["tau_local"].to_numpy()
    wall_pools = {
        int(client): sub["tau_wall"].to_numpy()
        for client, sub in event_log.groupby("client")
    }

    tau_values = [50, 100, 250, 500, 1000] if not quick else [100, 250, 500]
    alpha_values = [0.5, 1.0] if not quick else [0.5]
    rows = []

    def evaluate(weighting: str, **kwargs):
        return run_timing_batch(
            problem=problem_for_ratio(40),
            noise_std=0.25,
            n_ticks=ticks,
            n_seeds=eval_seeds,
            tail=tail,
            seed=BASE_SEED + 100,
            weighting=weighting,
            shuffled_local_pool=local_pool,
            shuffled_wall_pools=wall_pools,
            **kwargs,
        )

    fixed = evaluate("fixed")
    rows.append({"family": "Fixed q", "tau_c": np.nan, "alpha": np.nan, **fixed})

    for tau_c in tau_values:
        for alpha in alpha_values:
            for weighting, family in [
                ("inverse_local", "Local-completion timing"),
                ("inverse_wall", "Raw wall timing"),
                ("shuffled_local", "Shuffled local timing"),
            ]:
                out = evaluate(weighting, tau_c=tau_c, alpha=alpha)
                rows.append(
                    {
                        "family": family,
                        "tau_c": tau_c,
                        "alpha": alpha,
                        **out,
                    }
                )

    tradeoff = pd.DataFrame(rows).drop(columns=["event_log"])

    # Best raw-wall rule and controls that preserve either client identity or the
    # marginal timing distribution but destroy dynamic interval information.
    wall_candidates = tradeoff[tradeoff["family"] == "Raw wall timing"]
    best_wall = wall_candidates.loc[wall_candidates["tail_rmse"].idxmin()]
    best_tau = float(best_wall["tau_c"])
    best_alpha = float(best_wall["alpha"])

    logged_wall = run_timing_batch(
        problem=problem_for_ratio(40),
        noise_std=0.25,
        n_ticks=ticks,
        n_seeds=eval_seeds,
        tail=tail,
        seed=BASE_SEED + 100,
        weighting="inverse_wall",
        tau_c=best_tau,
        alpha=best_alpha,
        log_events=True,
    )
    mean_client_weights = (
        logged_wall["event_log"].groupby("client")["jump_weight"].mean().to_dict()
    )

    control_rows = []
    for weighting, label, extra in [
        ("inverse_wall", "True dynamic wall timing", {}),
        (
            "shuffled_wall_within_client",
            "Wall timing shuffled within client",
            {},
        ),
        (
            "static_client",
            "Static client multipliers",
            {"static_client_weights": mean_client_weights},
        ),
    ]:
        out = run_timing_batch(
            problem=problem_for_ratio(40),
            noise_std=0.25,
            n_ticks=ticks,
            n_seeds=eval_seeds,
            tail=tail,
            seed=BASE_SEED + 100,
            weighting=weighting,
            tau_c=best_tau,
            alpha=best_alpha,
            shuffled_wall_pools=wall_pools,
            **extra,
        )
        control_rows.append({"control": label, **out})
    controls = pd.DataFrame(control_rows).drop(columns=["event_log"])

    # Conventional global step-size schedule control.
    schedule_rows = []
    schedule_scales = [500, 1000, 2000, 5000] if not quick else [1000, 5000]
    schedule_exponents = [0.5, 1.0]
    for scale in schedule_scales:
        for exponent in schedule_exponents:
            out = run_timing_batch(
                problem=problem_for_ratio(40),
                noise_std=0.25,
                n_ticks=ticks,
                n_seeds=eval_seeds,
                tail=tail,
                seed=BASE_SEED + 100,
                weighting="global_wall_schedule",
                schedule_scale=scale,
                schedule_exponent=exponent,
            )
            schedule_rows.append(
                {
                    "scale": scale,
                    "exponent": exponent,
                    **out,
                }
            )
    schedules = pd.DataFrame(schedule_rows).drop(columns=["event_log"])

    # ------------------------------------------------------------------
    # 9c-C: compact robustness grid. Keep the selected timing and schedule
    # rules fixed; do not retune them per operating point.
    # ------------------------------------------------------------------
    best_schedule = schedules.loc[schedules["tail_rmse"].idxmin()]
    ratios = [5, 20, 40, 80] if not quick else [5, 40]
    sigmas = [0.0, 0.25, 0.5] if not quick else [0.25]
    robustness_rows = []
    for sigma in sigmas:
        for ratio in ratios:
            prob = problem_for_ratio(ratio)
            for weighting, method, kwargs in [
                ("fixed", "Fixed q", {}),
                (
                    "inverse_wall",
                    "Event-time-aware q",
                    {"tau_c": best_tau, "alpha": best_alpha},
                ),
                (
                    "global_wall_schedule",
                    "Global q schedule",
                    {
                        "schedule_scale": float(best_schedule["scale"]),
                        "schedule_exponent": float(best_schedule["exponent"]),
                    },
                ),
            ]:
                out = run_timing_batch(
                    problem=prob,
                    noise_std=sigma,
                    n_ticks=min(ticks, 5000),
                    n_seeds=min(eval_seeds, 300),
                    tail=min(tail, 1200),
                    seed=112233 + ratio * 100 + int(1000 * sigma),
                    weighting=weighting,
                    **kwargs,
                )
                robustness_rows.append(
                    {
                        "sigma": sigma,
                        "ratio": ratio,
                        "method": method,
                        **out,
                    }
                )
    robustness = pd.DataFrame(robustness_rows).drop(columns=["event_log"])

    # ------------------------------------------------------------------
    # Save results and plots.
    # ------------------------------------------------------------------
    event_log.to_csv(OUT / "baseline_event_log.csv", index=False)
    quality.to_csv(OUT / "timing_quality_bins.csv", index=False)
    per_client.to_csv(OUT / "timing_quality_by_client.csv", index=False)
    conditioned.to_csv(OUT / "timing_quality_conditioned_on_state.csv", index=False)
    tradeoff.to_csv(OUT / "timing_aggregation_grid.csv", index=False)
    controls.to_csv(OUT / "timing_controls.csv", index=False)
    schedules.to_csv(OUT / "global_schedule_controls.csv", index=False)
    robustness.to_csv(OUT / "robustness_grid.csv", index=False)
    add_pareto_mask(tradeoff).to_csv(OUT / "tradeoff_pareto.csv", index=False)

    plt.figure(figsize=(8, 5))
    qplot = quality[quality["events"] > 0]
    plt.plot(qplot["p_global_harm"].to_numpy(), marker="o", label="global harmful")
    plt.plot(qplot["p_local_wrong"].to_numpy(), marker="o", label="local wrong")
    plt.xticks(
        np.arange(len(qplot)),
        qplot["local_bin"].astype(str).to_numpy(),
        rotation=35,
    )
    plt.xlabel("Local gradient completions since previous event")
    plt.ylabel("Probability")
    plt.title("Experiment 09c-A: event quality versus inter-event interval")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "timing_quality.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for family, sub in tradeoff.groupby("family"):
        plt.scatter(sub["mean_events"], sub["tail_rmse"], label=family, alpha=0.7)
    plt.xlabel("Mean transmitted events")
    plt.ylabel("Tail RMSE")
    plt.title("Experiment 09c-B: timing-aware aggregation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "timing_aggregation.png", dpi=180)
    plt.close()

    print("Baseline timing-quality summary")
    print(per_client.to_string(index=False))
    print("\nTiming trade-off")
    print(tradeoff.sort_values("tail_rmse").head(15).to_string(index=False))
    print("\nDynamic timing controls")
    print(controls.to_string(index=False))
    print("\nBest global jump-size schedule")
    print(best_schedule.to_string())
    print("\nRobustness grid")
    print(robustness.to_string(index=False))
    print(f"\nResults saved under {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ticks = min(args.ticks, 1200) if args.quick else args.ticks
    seeds = min(args.seeds, 40) if args.quick else args.seeds
    run_experiment(ticks=ticks, seeds=seeds, quick=args.quick)


if __name__ == "__main__":
    main()
