from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "experiments" / "results" / "05_controlled_async_drift"
OUT.mkdir(parents=True, exist_ok=True)

# Fixed local objective F_A(w)=0.5*(w-theta_A)^2.
THETA_A = 0.25
W_OLD = 0.0
W_NEW = 0.5

# The client has accumulated positive evidence at the old model. Other clients
# move the server model to W_NEW before this client has fired. At W_NEW the
# client's gradient changes sign, making Z_INIT genuinely stale.
Z_INIT = 0.40
GAMMA = 0.05
DELTA = 0.50
SIGMA = 0.25
MAX_STEPS = 500
N_MC = 10_000
RHOS = [1.0, 0.995, 0.98, 0.95]

G_NEW = W_NEW - THETA_A  # positive: correct post-drift update sign is negative


def deterministic_passage(rho: float) -> tuple[float, float]:
    """Return stale-sign erasure and first correct threshold passage times."""
    z = Z_INIT
    zero_crossing = np.nan
    first_correct_event = np.nan

    for k in range(1, MAX_STEPS + 1):
        z = rho * z - GAMMA * G_NEW
        if np.isnan(zero_crossing) and z <= 0.0:
            zero_crossing = float(k)
        if z <= -DELTA:
            first_correct_event = float(k)
            break

    return zero_crossing, first_correct_event


def stochastic_trial(rho: float, rng: np.random.Generator):
    z = Z_INIT
    zero_crossing = np.nan

    for k in range(1, MAX_STEPS + 1):
        gradient = G_NEW + rng.normal(0.0, SIGMA)
        z = rho * z - GAMMA * gradient

        if np.isnan(zero_crossing) and z <= 0.0:
            zero_crossing = float(k)

        if abs(z) >= DELTA:
            sign = 1 if z > 0.0 else -1
            return float(k), sign, zero_crossing

    return np.nan, 0, zero_crossing


def main():
    deterministic_rows = []
    for rho in RHOS:
        erase, fire = deterministic_passage(rho)
        deterministic_rows.append(
            {
                "rho": rho,
                "stale_sign_erasure_steps": erase,
                "first_correct_event_steps": fire,
            }
        )
    deterministic = pd.DataFrame(deterministic_rows)
    deterministic.to_csv(OUT / "deterministic_passage.csv", index=False)

    rng = np.random.default_rng(20260829)
    stochastic_rows = []
    for rho in RHOS:
        delays, signs, erase_times = [], [], []
        for _ in range(N_MC):
            delay, sign, erase = stochastic_trial(rho, rng)
            delays.append(delay)
            signs.append(sign)
            erase_times.append(erase)

        delays = np.asarray(delays, dtype=float)
        signs = np.asarray(signs, dtype=int)
        erase_times = np.asarray(erase_times, dtype=float)
        stochastic_rows.append(
            {
                "rho": rho,
                "mean_stale_sign_erasure": np.nanmean(erase_times),
                "mean_first_event_delay": np.nanmean(delays)
                if np.any(np.isfinite(delays))
                else np.nan,
                "p_first_event_correct": np.mean(signs == -1),
                "p_first_event_stale_wrong": np.mean(signs == 1),
                "p_no_event_by_horizon": np.mean(signs == 0),
            }
        )

    stochastic = pd.DataFrame(stochastic_rows)
    stochastic.to_csv(OUT / "stochastic_passage.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(
        deterministic["rho"],
        deterministic["stale_sign_erasure_steps"],
        marker="o",
        label="stale sign erased",
    )
    plt.plot(
        deterministic["rho"],
        deterministic["first_correct_event_steps"],
        marker="o",
        label="first correct event",
    )
    plt.xlabel("Retention rho")
    plt.ylabel("Gradient evaluations after server-model drift")
    plt.title("Controlled asynchronous drift: forgetting vs useful firing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "passage_times.png", dpi=180)
    plt.close()

    print("Controlled asynchronous-drift diagnostic")
    print(f"Local optimum theta_A={THETA_A}, server model {W_OLD} -> {W_NEW}")
    print(f"Initial stale membrane z0={Z_INIT}, Delta={DELTA}, gamma={GAMMA}")
    print("\nDeterministic first passage:")
    print(deterministic.to_string(index=False))
    print("\nStochastic first passage:")
    print(stochastic.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
