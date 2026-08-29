from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.delayed_async import (
    DelayedAsyncClient,
    run_delayed_rate_normalized_scalar_federation,
)

OUT = ROOT / "experiments" / "results" / "08_heterogeneous_delayed_async"

# Stress design: local optima are deliberately close to the global operating
# region, so sufficiently delayed gradients can become direction-stale.
D = 0.05
THETAS = np.array([D, -D])  # client 0 slow, client 1 fast
WEIGHTS = np.array([0.5, 0.5])
W0 = 1.03
GAMMA = 0.05
DELTA = 0.5
Q = 0.10
TAIL = 1500
R_VALUES = [5, 10, 20, 40, 80]
RHO_VALUES = [1.0, 0.999, 0.9975, 0.995, 0.99, 0.98]
SIGMA_VALUES = [0.0, 0.25]
BASE_SEED = 20260829


def _global_objective(w: np.ndarray) -> np.ndarray:
    """Equal-weight aggregate objective; optimum is w*=0."""

    return 0.5 * np.sum(
        WEIGHTS[:, None] * (w[None, :] - THETAS[:, None]) ** 2,
        axis=0,
    )


def simulate_batch(
    *,
    slow_period: int,
    rho: float,
    sigma: float,
    n_ticks: int,
    n_seeds: int,
    use_fresh_gradient_oracle: bool = False,
    randomized_initial_conditions: bool = False,
):
    """Vectorized delayed-gradient asynchronous federation.

    A client starts a gradient computation from a server snapshot and returns it
    only after its compute period. The slow client has period R, the fast client
    period one. To prevent the fast client from gaining larger optimization
    weight merely because it completes more jobs, the local evidence gains are

        gamma_i = gamma * p_i * T_i.

    Hence, for a slowly varying model, the mean evidence injection per wall-clock
    tick is approximately -gamma p_i g_i for every client.

    ``local_stale`` labels an event whose sign disagrees with the emitting
    client's exact *current* local descent direction. ``global_harmful`` labels
    an event that increases the intended aggregate objective exactly.
    """

    periods = np.array([slow_period, 1], dtype=int)
    evidence_gains = GAMMA * WEIGHTS * periods

    rng = np.random.default_rng(
        BASE_SEED
        + 1009 * slow_period
        + int(1e6 * rho)
        + int(1000 * sigma)
        + 31 * int(use_fresh_gradient_oracle)
        + 47 * int(randomized_initial_conditions)
    )

    if randomized_initial_conditions:
        w = rng.uniform(0.93, 1.13, size=n_seeds)
    else:
        w = np.full(n_seeds, W0, dtype=float)

    z = np.zeros((2, n_seeds), dtype=float)
    snapshots = np.tile(w, (2, 1))
    next_completion = periods.copy()

    events = np.zeros(n_seeds, dtype=int)
    events_per_client = np.zeros((2, n_seeds), dtype=int)
    local_stale = np.zeros(n_seeds, dtype=int)
    local_stale_per_client = np.zeros((2, n_seeds), dtype=int)
    global_harm = np.zeros(n_seeds, dtype=int)
    global_harm_per_client = np.zeros((2, n_seeds), dtype=int)
    stale_and_harm = np.zeros(n_seeds, dtype=int)
    stale_gradient_results = np.zeros(n_seeds, dtype=int)
    stale_gradient_per_client = np.zeros((2, n_seeds), dtype=int)
    gradient_results = np.zeros(n_seeds, dtype=int)
    gradient_results_per_client = np.zeros((2, n_seeds), dtype=int)

    whole_excess = np.zeros(n_seeds)
    tail_w2 = np.zeros(n_seeds)
    tail = min(TAIL, n_ticks // 2)

    for tick in range(1, n_ticks + 1):
        z *= rho

        active = [i for i in range(2) if next_completion[i] == tick]
        if len(active) == 2 and tick % 2 == 1:
            active.reverse()

        for client in active:
            snapshot = snapshots[client].copy()
            current = w.copy()

            snapshot_descent = -np.sign(snapshot - THETAS[client])
            current_descent = -np.sign(current - THETAS[client])
            gradient_stale = (
                (snapshot_descent != 0)
                & (current_descent != 0)
                & (snapshot_descent != current_descent)
            )
            stale_gradient_results += gradient_stale.astype(int)
            stale_gradient_per_client[client] += gradient_stale.astype(int)
            gradient_results += 1
            gradient_results_per_client[client] += 1

            gradient_point = current if use_fresh_gradient_oracle else snapshot
            gradient = (
                gradient_point
                - THETAS[client]
                + rng.normal(0.0, sigma, size=n_seeds)
            )
            z[client] -= evidence_gains[client] * gradient

            while True:
                mask = np.abs(z[client]) >= DELTA
                if not np.any(mask):
                    break

                sign = np.zeros(n_seeds, dtype=int)
                sign[mask] = np.sign(z[client, mask]).astype(int)
                z[client, mask] -= DELTA * sign[mask]

                idx = np.flatnonzero(mask)
                w_before = w[idx].copy()
                event_sign = sign[idx]

                current_descent = -np.sign(w_before - THETAS[client])
                event_stale = (
                    (current_descent != 0)
                    & (event_sign != current_descent)
                )

                f_before = _global_objective(w_before)
                w_after = w_before + Q * event_sign
                f_after = _global_objective(w_after)
                event_harmful = f_after > f_before + 1e-15

                w[idx] = w_after
                events[idx] += 1
                events_per_client[client, idx] += 1
                local_stale[idx] += event_stale.astype(int)
                local_stale_per_client[client, idx] += event_stale.astype(int)
                global_harm[idx] += event_harmful.astype(int)
                global_harm_per_client[client, idx] += event_harmful.astype(int)
                stale_and_harm[idx] += (event_stale & event_harmful).astype(int)

            # Start the next job from the latest server model.
            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        # F(w)-F(w*) = 0.5*w^2 for the symmetric equal-weight setup.
        whole_excess += 0.5 * w**2
        if tick > n_ticks - tail:
            tail_w2 += w**2

    event_count = int(events.sum())
    stale_count = int(local_stale.sum())
    harm_count = int(global_harm.sum())

    return {
        "tail_rmse": float(np.sqrt(np.mean(tail_w2 / tail))),
        "whole_excess_objective": float(np.mean(whole_excess / n_ticks)),
        "mean_events": float(np.mean(events)),
        "mean_slow_events": float(np.mean(events_per_client[0])),
        "mean_fast_events": float(np.mean(events_per_client[1])),
        "stale_gradient_fraction": float(stale_gradient_results.sum() / gradient_results.sum()),
        "slow_stale_gradient_fraction": float(
            stale_gradient_per_client[0].sum() / gradient_results_per_client[0].sum()
        ),
        "local_stale_event_fraction": float(stale_count / event_count) if event_count else np.nan,
        "slow_local_stale_fraction": float(
            local_stale_per_client[0].sum() / events_per_client[0].sum()
        )
        if events_per_client[0].sum()
        else np.nan,
        "global_harm_fraction": float(harm_count / event_count) if event_count else np.nan,
        "slow_global_harm_fraction": float(
            global_harm_per_client[0].sum() / events_per_client[0].sum()
        )
        if events_per_client[0].sum()
        else np.nan,
        "global_harm_given_local_stale": float(stale_and_harm.sum() / stale_count)
        if stale_count
        else np.nan,
        "local_stale_given_global_harm": float(stale_and_harm.sum() / harm_count)
        if harm_count
        else np.nan,
    }


def run_grid(*, n_ticks: int, n_seeds: int, r_values, rho_values):
    rows = []
    for sigma in SIGMA_VALUES:
        for slow_period in r_values:
            for rho in rho_values:
                rows.append(
                    {
                        "sigma": sigma,
                        "R": slow_period,
                        "method": "delayed",
                        "rho": rho,
                        **simulate_batch(
                            slow_period=slow_period,
                            rho=rho,
                            sigma=sigma,
                            n_ticks=n_ticks,
                            n_seeds=n_seeds,
                        ),
                    }
                )

            # Causal baseline: same completion schedule and rate normalization,
            # but remove computation-delay staleness by using current gradients.
            rows.append(
                {
                    "sigma": sigma,
                    "R": slow_period,
                    "method": "fresh-gradient oracle",
                    "rho": 1.0,
                    **simulate_batch(
                        slow_period=slow_period,
                        rho=1.0,
                        sigma=sigma,
                        n_ticks=n_ticks,
                        n_seeds=n_seeds,
                        use_fresh_gradient_oracle=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, r_values):
    rows = []
    for sigma in SIGMA_VALUES:
        for slow_period in r_values:
            delayed = df[
                (df["sigma"] == sigma)
                & (df["R"] == slow_period)
                & (df["method"] == "delayed")
            ]
            if_row = delayed[delayed["rho"] == 1.0].iloc[0]
            best = delayed.loc[delayed["tail_rmse"].idxmin()]
            oracle = df[
                (df["sigma"] == sigma)
                & (df["R"] == slow_period)
                & (df["method"] == "fresh-gradient oracle")
            ].iloc[0]
            rows.append(
                {
                    "sigma": sigma,
                    "R": slow_period,
                    "IF_tail_rmse": if_row["tail_rmse"],
                    "best_rho": best["rho"],
                    "best_tail_rmse": best["tail_rmse"],
                    "fresh_oracle_tail_rmse": oracle["tail_rmse"],
                    "IF_local_stale": if_row["local_stale_event_fraction"],
                    "best_local_stale": best["local_stale_event_fraction"],
                    "fresh_oracle_local_stale": oracle["local_stale_event_fraction"],
                    "IF_global_harm": if_row["global_harm_fraction"],
                    "best_global_harm": best["global_harm_fraction"],
                    "fresh_oracle_global_harm": oracle["global_harm_fraction"],
                    "IF_events": if_row["mean_events"],
                    "best_events": best["mean_events"],
                    "fresh_oracle_events": oracle["mean_events"],
                    "IF_slow_events": if_row["mean_slow_events"],
                    "best_slow_events": best["mean_slow_events"],
                    "IF_harm_given_stale": if_row["global_harm_given_local_stale"],
                    "IF_slow_stale_gradient": if_row["slow_stale_gradient_fraction"],
                }
            )
    return pd.DataFrame(rows)


def save_heatmap(df, *, sigma, metric, title, filename, r_values, rho_values, fmt):
    sub = df[(df["sigma"] == sigma) & (df["method"] == "delayed")]
    pivot = sub.pivot(index="R", columns="rho", values=metric).reindex(
        index=r_values, columns=rho_values
    )
    values = pivot.to_numpy()

    plt.figure(figsize=(9, 5))
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(image, label=metric.replace("_", " "))
    plt.xticks(range(len(rho_values)), [str(v) for v in rho_values])
    plt.yticks(range(len(r_values)), [str(v) for v in r_values])
    plt.xlabel("Retention rho")
    plt.ylabel("Slow/fast compute delay R")
    plt.title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "--" if not np.isfinite(value) else format(value, fmt)
            plt.text(col, row, label, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def run_random_initial_robustness(*, n_ticks: int, n_seeds: int):
    rows = []
    slow_period = 40
    for sigma in SIGMA_VALUES:
        for rho in [1.0, 0.999, 0.9975, 0.995, 0.99]:
            rows.append(
                {
                    "sigma": sigma,
                    "method": "delayed",
                    "rho": rho,
                    **simulate_batch(
                        slow_period=slow_period,
                        rho=rho,
                        sigma=sigma,
                        n_ticks=n_ticks,
                        n_seeds=n_seeds,
                        randomized_initial_conditions=True,
                    ),
                }
            )
        rows.append(
            {
                "sigma": sigma,
                "method": "fresh-gradient oracle",
                "rho": 1.0,
                **simulate_batch(
                    slow_period=slow_period,
                    rho=1.0,
                    sigma=sigma,
                    n_ticks=n_ticks,
                    n_seeds=n_seeds,
                    use_fresh_gradient_oracle=True,
                    randomized_initial_conditions=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_smoke_check():
    """Check the reusable delayed simulator on a deterministic configuration."""

    clients = [
        DelayedAsyncClient(theta=D, period=5, weight=0.5),
        DelayedAsyncClient(theta=-D, period=1, weight=0.5),
    ]
    run = run_delayed_rate_normalized_scalar_federation(
        clients=clients,
        n_ticks=200,
        w0=W0,
        gamma=GAMMA,
        threshold=DELTA,
        jump=Q,
        rho=1.0,
        noise_std=0.0,
        seed=0,
    )
    if not np.all(np.isfinite(run.w)):
        raise RuntimeError("Reference delayed simulator produced non-finite state values.")
    if np.any(np.abs(run.membrane) >= DELTA + 1e-12):
        raise RuntimeError("Subtractive reset invariant violated in reference simulator.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=400)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    reference_smoke_check()

    if args.quick:
        n_ticks = min(args.ticks, 1200)
        n_seeds = min(args.seeds, 24)
        r_values = [5, 40]
        rho_values = [1.0, 0.995, 0.99]
    else:
        n_ticks = args.ticks
        n_seeds = args.seeds
        r_values = R_VALUES
        rho_values = RHO_VALUES

    OUT.mkdir(parents=True, exist_ok=True)

    grid = run_grid(
        n_ticks=n_ticks,
        n_seeds=n_seeds,
        r_values=r_values,
        rho_values=rho_values,
    )
    summary = summarize(grid, r_values)
    robustness = run_random_initial_robustness(
        n_ticks=n_ticks,
        n_seeds=n_seeds if not args.quick else min(n_seeds, 24),
    )

    grid.to_csv(OUT / "regime_map.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    robustness.to_csv(OUT / "random_initial_robustness_R40.csv", index=False)

    save_heatmap(
        grid,
        sigma=0.0,
        metric="local_stale_event_fraction",
        title="Zero noise: locally stale event fraction",
        filename="zero_noise_local_stale.png",
        r_values=r_values,
        rho_values=rho_values,
        fmt=".2f",
    )
    save_heatmap(
        grid,
        sigma=0.0,
        metric="global_harm_fraction",
        title="Zero noise: globally harmful event fraction",
        filename="zero_noise_global_harm.png",
        r_values=r_values,
        rho_values=rho_values,
        fmt=".2f",
    )
    save_heatmap(
        grid,
        sigma=0.0,
        metric="tail_rmse",
        title="Zero noise: tail RMSE to the aggregate optimum",
        filename="zero_noise_tail_rmse.png",
        r_values=r_values,
        rho_values=rho_values,
        fmt=".3f",
    )
    save_heatmap(
        grid,
        sigma=0.25,
        metric="local_stale_event_fraction",
        title="Noisy gradients: locally stale/wrong-direction events",
        filename="noisy_local_stale.png",
        r_values=r_values,
        rho_values=rho_values,
        fmt=".2f",
    )

    print(summary.to_string(index=False))
    print("\nRandom-initial-condition robustness at R=40:")
    print(robustness.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
