from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.async_simulation import AsyncClient, run_async_scalar_federation

OUT = ROOT / "experiments" / "results" / "06_two_client_async"
OUT.mkdir(parents=True, exist_ok=True)

# Both clients optimize the same quadratic F_i(w)=0.5*w^2. This deliberately
# removes statistical heterogeneity. A harmful event is classified exactly by
# whether the fixed q-sized server jump increases the current quadratic.
CLIENTS = [
    AsyncClient(theta=0.0, period=5),  # client 0: slow
    AsyncClient(theta=0.0, period=1),  # client 1: fast
]

W0 = -1.0
N_TICKS = 3000
N_SEEDS = 400
SIGMA = 0.5
GAMMA = 0.05
DELTA = 0.5
Q = 0.05

CONFIGS = [
    ("IF", 1.0, False),
    ("IF oracle reset", 1.0, True),
    ("LIF rho=0.995", 0.995, False),
    ("LIF rho=0.98", 0.98, False),
    ("LIF rho=0.95", 0.95, False),
]


def run_ensemble(label: str, rho: float, oracle_reset: bool):
    return [
        run_async_scalar_federation(
            clients=CLIENTS,
            n_ticks=N_TICKS,
            w0=W0,
            gamma=GAMMA,
            threshold=DELTA,
            jump=Q,
            rho=rho,
            noise_std=SIGMA,
            seed=seed,
            oracle_reset_others=oracle_reset,
            # If both are active on the same tick, the fast client updates the
            # server first; the slow client then sees the newly broadcast model.
            active_order=[1, 0],
        )
        for seed in range(N_SEEDS)
    ]


def main():
    results = {
        label: run_ensemble(label, rho, oracle)
        for label, rho, oracle in CONFIGS
    }

    rows = []
    for label, runs in results.items():
        W = np.vstack([run.w for run in runs])
        total_events = np.asarray([run.communications[-1] for run in runs])
        harmful = np.asarray([run.harmful_events for run in runs])
        slow_events = np.asarray([run.events_per_client[0] for run in runs])
        slow_harmful = np.asarray([run.harmful_per_client[0] for run in runs])

        rows.append(
            {
                "method": label,
                "whole_run_mean_w2": np.mean(W**2),
                "tail_rmse": np.sqrt(np.mean(W[:, -1000:] ** 2)),
                "tail_mae": np.mean(np.abs(W[:, -1000:])),
                "mean_total_events": np.mean(total_events),
                "objective_increasing_event_fraction": np.sum(harmful) / np.sum(total_events),
                "mean_slow_client_events": np.mean(slow_events),
                "slow_client_objective_increasing_fraction": np.sum(slow_harmful) / np.sum(slow_events)
                if np.sum(slow_events) > 0
                else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    x = np.arange(N_TICKS + 1)
    plt.figure(figsize=(9, 5))
    for label, runs in results.items():
        W = np.vstack([run.w for run in runs])
        plt.plot(x, np.mean(np.abs(W), axis=0), label=label)
    plt.xlabel("Wall-clock ticks")
    plt.ylabel("Mean |w|")
    plt.title("Asynchronous two-client event-driven optimization")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "mae_vs_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(summary["mean_total_events"], summary["tail_rmse"])
    for _, row in summary.iterrows():
        plt.annotate(row["method"], (row["mean_total_events"], row["tail_rmse"]))
    plt.xlabel("Mean total communication events")
    plt.ylabel("Tail RMSE")
    plt.title("Communication--accuracy trade-off")
    plt.tight_layout()
    plt.savefig(OUT / "communication_accuracy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(summary["objective_increasing_event_fraction"], summary["tail_rmse"])
    for _, row in summary.iterrows():
        plt.annotate(
            row["method"],
            (row["objective_increasing_event_fraction"], row["tail_rmse"]),
        )
    plt.xlabel("Fraction of objective-increasing events")
    plt.ylabel("Tail RMSE")
    plt.title("Harmful events versus optimization accuracy")
    plt.tight_layout()
    plt.savefig(OUT / "harmful_events_vs_accuracy.png", dpi=180)
    plt.close()

    print("Two-client asynchronous federation")
    print(
        f"slow period={CLIENTS[0].period}, fast period={CLIENTS[1].period}, "
        f"sigma={SIGMA}, Delta={DELTA}, q={Q}"
    )
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
