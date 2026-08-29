from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "07_asynchrony_memory_map"

W0 = -1.0
SIGMA = 0.5
GAMMA = 0.05
DELTA = 0.5
Q = 0.05
TAIL = 1000
R_VALUES = [1, 2, 5, 10, 20]
RHO_VALUES = [1.0, 0.999, 0.995, 0.99, 0.98, 0.95]
BASE_SEED = 20260829


def _emit_events(w, z, client, threshold, jump, hard_reset, stats):
    """Apply all threshold crossings for one vectorized client state."""
    while True:
        mask = np.abs(z[client]) >= threshold
        if not np.any(mask):
            return w, z

        s = np.zeros_like(w, dtype=int)
        s[mask] = np.sign(z[client, mask]).astype(int)
        z[client, mask] -= threshold * s[mask]

        idx = np.flatnonzero(mask)
        w_before = w[idx].copy()
        w_after = w_before + jump * s[idx]
        harmful = 0.5 * w_after**2 > 0.5 * w_before**2 + 1e-15

        w[idx] = w_after
        stats["events"][idx] += 1
        stats["harmful"][idx] += harmful.astype(int)
        if client == 0:
            stats["slow_events"][idx] += 1
            stats["slow_harmful"][idx] += harmful.astype(int)

        if hard_reset:
            z[1 - client, idx] = 0.0


def simulate_batch(
    *,
    rho: float,
    slow_period: int,
    fast_noise: np.ndarray,
    slow_noise: np.ndarray,
    n_ticks: int,
    tail: int,
    leak_mode: str = "wall_clock",
    hard_reset: bool = False,
    instantaneous_sign: bool = False,
):
    """Vectorized two-client homogeneous asynchronous simulation.

    Client 1 evaluates every tick; client 0 evaluates every ``slow_period``
    ticks. Both optimize F_i(w)=0.5*w^2. The fast client is processed first on
    ticks where both clients are active.

    ``leak_mode='wall_clock'`` applies rho to all stored evidence every tick.
    ``leak_mode='local_step'`` applies rho only when that client evaluates a
    new gradient. The latter is a diagnostic used to check whether slow-client
    silencing is merely a wall-clock-decay artifact.
    """
    if leak_mode not in {"wall_clock", "local_step"}:
        raise ValueError("leak_mode must be 'wall_clock' or 'local_step'.")

    n_seeds = fast_noise.shape[1]
    w = np.full(n_seeds, W0, dtype=float)
    z = np.zeros((2, n_seeds), dtype=float)
    stats = {
        "events": np.zeros(n_seeds, dtype=int),
        "slow_events": np.zeros(n_seeds, dtype=int),
        "harmful": np.zeros(n_seeds, dtype=int),
        "slow_harmful": np.zeros(n_seeds, dtype=int),
    }
    whole_w2 = np.zeros(n_seeds)
    tail_w2 = np.zeros(n_seeds)
    tail_abs = np.zeros(n_seeds)
    slow_index = 0

    for tick in range(n_ticks):
        if not instantaneous_sign and leak_mode == "wall_clock":
            z *= rho

        # Fast client first.
        for client, active, noise in (
            (1, True, fast_noise[tick]),
            (0, tick % slow_period == 0, slow_noise[slow_index] if tick % slow_period == 0 else None),
        ):
            if not active:
                continue

            if client == 0:
                slow_index += 1

            gradient = w + noise

            if instantaneous_sign:
                s = -np.sign(gradient).astype(int)
                mask = s != 0
                idx = np.flatnonzero(mask)
                w_before = w[idx].copy()
                w_after = w_before + Q * s[idx]
                harmful = 0.5 * w_after**2 > 0.5 * w_before**2 + 1e-15
                w[idx] = w_after
                stats["events"][idx] += 1
                stats["harmful"][idx] += harmful.astype(int)
                if client == 0:
                    stats["slow_events"][idx] += 1
                    stats["slow_harmful"][idx] += harmful.astype(int)
                continue

            if leak_mode == "local_step":
                z[client] *= rho
            z[client] -= GAMMA * gradient
            w, z = _emit_events(w, z, client, DELTA, Q, hard_reset, stats)

        whole_w2 += w**2
        if tick >= n_ticks - tail:
            tail_w2 += w**2
            tail_abs += np.abs(w)

    return {
        "tail_rmse": float(np.sqrt(np.mean(tail_w2 / tail))),
        "tail_mae": float(np.mean(tail_abs / tail)),
        "whole_run_mse": float(np.mean(whole_w2 / n_ticks)),
        "mean_events": float(np.mean(stats["events"])),
        "mean_slow_events": float(np.mean(stats["slow_events"])),
        "harmful_fraction": float(np.sum(stats["harmful"]) / np.sum(stats["events"]))
        if np.sum(stats["events"])
        else np.nan,
        "slow_harmful_fraction": float(np.sum(stats["slow_harmful"]) / np.sum(stats["slow_events"]))
        if np.sum(stats["slow_events"])
        else np.nan,
    }


def make_noise(slow_period: int, sigma: float, n_ticks: int, n_seeds: int):
    """Common random numbers used for every rho at one asynchrony ratio."""
    rng = np.random.default_rng(BASE_SEED + 1009 * slow_period + int(1000 * sigma))
    fast = rng.normal(0.0, sigma, size=(n_ticks, n_seeds))
    n_slow = (n_ticks - 1) // slow_period + 1
    slow = rng.normal(0.0, sigma, size=(n_slow, n_seeds))
    return fast, slow


def run_map(*, sigma: float, n_ticks: int, n_seeds: int, leak_mode: str):
    rows = []
    tail = min(TAIL, n_ticks // 2)
    for slow_period in R_VALUES:
        fast_noise, slow_noise = make_noise(slow_period, sigma, n_ticks, n_seeds)
        for rho in RHO_VALUES:
            out = simulate_batch(
                rho=rho,
                slow_period=slow_period,
                fast_noise=fast_noise,
                slow_noise=slow_noise,
                n_ticks=n_ticks,
                tail=tail,
                leak_mode=leak_mode,
            )
            rows.append(
                {
                    "R": slow_period,
                    "rho": rho,
                    "memory_horizon": np.inf if rho == 1.0 else 1.0 / (1.0 - rho),
                    **out,
                }
            )
    return pd.DataFrame(rows)


def run_baselines(*, sigma: float, n_ticks: int, n_seeds: int):
    rows = []
    tail = min(TAIL, n_ticks // 2)
    for slow_period in R_VALUES:
        fast_noise, slow_noise = make_noise(slow_period, sigma, n_ticks, n_seeds)
        for name, kwargs in (
            ("IF", dict(rho=1.0)),
            ("Hard reset IF", dict(rho=1.0, hard_reset=True)),
            ("Instantaneous sign", dict(rho=1.0, instantaneous_sign=True)),
        ):
            out = simulate_batch(
                slow_period=slow_period,
                fast_noise=fast_noise,
                slow_noise=slow_noise,
                n_ticks=n_ticks,
                tail=tail,
                leak_mode="wall_clock",
                **kwargs,
            )
            rows.append({"R": slow_period, "baseline": name, **out})
    return pd.DataFrame(rows)


def best_by_rho(df: pd.DataFrame):
    rows = []
    for slow_period in R_VALUES:
        sub = df[df["R"] == slow_period]
        best = sub.loc[sub["tail_rmse"].idxmin()]
        if_row = sub[sub["rho"] == 1.0].iloc[0]
        rows.append(
            {
                "R": slow_period,
                "best_rho": best["rho"],
                "best_memory_horizon": best["memory_horizon"],
                "best_tail_rmse": best["tail_rmse"],
                "IF_tail_rmse": if_row["tail_rmse"],
                "rmse_improvement_vs_IF_pct": 100.0
                * (if_row["tail_rmse"] - best["tail_rmse"])
                / if_row["tail_rmse"]
                if if_row["tail_rmse"] > 0.0
                else 0.0,
                "best_events": best["mean_events"],
                "IF_events": if_row["mean_events"],
                "best_slow_events": best["mean_slow_events"],
                "best_harmful_fraction": best["harmful_fraction"],
                "IF_harmful_fraction": if_row["harmful_fraction"],
            }
        )
    return pd.DataFrame(rows)


def save_heatmap(df: pd.DataFrame, metric: str, title: str, filename: str, fmt: str):
    pivot = df.pivot(index="R", columns="rho", values=metric).reindex(index=R_VALUES, columns=RHO_VALUES)
    values = pivot.to_numpy()

    plt.figure(figsize=(9, 5))
    image = plt.imshow(values, aspect="auto")
    plt.colorbar(image, label=metric.replace("_", " "))
    plt.xticks(range(len(RHO_VALUES)), [str(v) for v in RHO_VALUES])
    plt.yticks(range(len(R_VALUES)), [str(v) for v in R_VALUES])
    plt.xlabel("Retention rho")
    plt.ylabel("Slow/fast compute-period ratio R")
    plt.title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "--" if not np.isfinite(value) else format(value, fmt)
            plt.text(col, row, label, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=300)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small smoke-test configuration (1500 ticks, 50 seeds).",
    )
    args = parser.parse_args()
    if args.quick:
        args.ticks = 1500
        args.seeds = 50

    if args.ticks < 200 or args.seeds < 1:
        raise ValueError("Use at least 200 ticks and one Monte Carlo seed.")

    OUT.mkdir(parents=True, exist_ok=True)

    # Main stochastic map.
    stochastic = run_map(sigma=SIGMA, n_ticks=args.ticks, n_seeds=args.seeds, leak_mode="wall_clock")
    stochastic.to_csv(OUT / "stochastic_map.csv", index=False)
    best = best_by_rho(stochastic)
    best.to_csv(OUT / "best_by_asynchrony.csv", index=False)
    baselines = run_baselines(sigma=SIGMA, n_ticks=args.ticks, n_seeds=args.seeds)
    baselines.to_csv(OUT / "baselines.csv", index=False)

    save_heatmap(stochastic, "tail_rmse", "Tail RMSE: asynchrony x LIF memory", "tail_rmse.png", ".3f")
    save_heatmap(
        stochastic,
        "harmful_fraction",
        "Objective-increasing event fraction",
        "harmful_fraction.png",
        ".2f",
    )
    save_heatmap(stochastic, "mean_events", "Communication events", "communication.png", ".1f")
    save_heatmap(stochastic, "mean_slow_events", "Slow-client participation", "slow_events.png", ".1f")

    # Check that conclusions are not just a wall-clock-decay artifact.
    local_step = run_map(sigma=SIGMA, n_ticks=args.ticks, n_seeds=args.seeds, leak_mode="local_step")
    local_step.to_csv(OUT / "local_step_leak_map.csv", index=False)
    best_local = best_by_rho(local_step)
    best_local.to_csv(OUT / "best_local_step.csv", index=False)

    # Zero-noise falsification control: removes stochastic-gradient filtering.
    zero_noise = run_map(sigma=0.0, n_ticks=args.ticks, n_seeds=max(1, min(args.seeds, 20)), leak_mode="wall_clock")
    zero_noise.to_csv(OUT / "zero_noise_map.csv", index=False)
    best_zero = best_by_rho(zero_noise)
    best_zero.to_csv(OUT / "best_zero_noise.csv", index=False)
    save_heatmap(
        zero_noise,
        "slow_harmful_fraction",
        "Zero-noise slow-client harmful events",
        "zero_noise_slow_harmful.png",
        ".2f",
    )

    comparison = best[["R", "best_rho", "best_tail_rmse", "best_slow_events"]].rename(
        columns={
            "best_rho": "wall_clock_best_rho",
            "best_tail_rmse": "wall_clock_tail_rmse",
            "best_slow_events": "wall_clock_slow_events",
        }
    )
    comparison = comparison.merge(
        best_local[["R", "best_rho", "best_tail_rmse", "best_slow_events"]].rename(
            columns={
                "best_rho": "local_step_best_rho",
                "best_tail_rmse": "local_step_tail_rmse",
                "best_slow_events": "local_step_slow_events",
            }
        ),
        on="R",
    )
    comparison = comparison.merge(
        best_zero[["R", "best_rho", "best_tail_rmse"]].rename(
            columns={"best_rho": "zero_noise_best_rho", "best_tail_rmse": "zero_noise_tail_rmse"}
        ),
        on="R",
    )
    comparison.to_csv(OUT / "mechanism_comparison.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(comparison["R"], comparison["wall_clock_best_rho"], marker="o", label="wall-clock leak")
    plt.plot(comparison["R"], comparison["local_step_best_rho"], marker="o", label="local-step leak")
    plt.plot(comparison["R"], comparison["zero_noise_best_rho"], marker="o", label="zero-noise control")
    plt.xlabel("Asynchrony ratio R")
    plt.ylabel("Best rho by tail RMSE")
    plt.title("Best memory retention versus asynchrony")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "best_rho_vs_asynchrony.png", dpi=180)
    plt.close()

    print("Experiment 07: asynchrony-memory regime map")
    print(f"ticks={args.ticks}, seeds={args.seeds}, sigma={SIGMA}")
    print("\nBest stochastic settings:")
    print(best.to_string(index=False))
    print("\nMechanism comparison:")
    print(comparison.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
