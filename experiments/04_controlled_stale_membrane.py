from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "04_controlled_stale_membrane"
OUT.mkdir(parents=True, exist_ok=True)

# After the switch, use a constant negative gradient g=-G, which requires a
# positive model-update event. The membrane therefore receives a constant
# positive deterministic input U = gamma*G.
G = 0.4
GAMMA = 0.05
U = GAMMA * G
DELTA = 0.5
NOISE_STD = 0.4
N_MONTE_CARLO = 5000
MAX_STEPS = 1000

Z0_VALUES = [-0.45, -0.30, -0.10]
RHO_VALUES = [1.0, 0.995, 0.98, 0.95]


def deterministic_zero_crossing_steps(z0: float, rho: float) -> float:
    """First k for which the deterministic membrane is nonnegative."""
    if rho == 1.0:
        return float(np.ceil(-z0 / U))

    equilibrium = U / (1.0 - rho)
    # z_k = equilibrium + rho^k (z0-equilibrium)
    k = np.log(equilibrium / (equilibrium - z0)) / np.log(rho)
    return float(np.ceil(k))


def deterministic_correct_threshold_steps(z0: float, rho: float) -> float:
    """First k for which the deterministic membrane reaches +DELTA.

    For rho<1, firing is impossible when the forced membrane equilibrium is
    not above +DELTA. This is the discrete LIF deadzone condition.
    """
    if rho == 1.0:
        return float(np.ceil((DELTA - z0) / U))

    equilibrium = U / (1.0 - rho)
    if equilibrium <= DELTA:
        return np.inf

    k = np.log((equilibrium - DELTA) / (equilibrium - z0)) / np.log(rho)
    return float(np.ceil(k))


def run_noisy_trial(z0: float, rho: float, rng: np.random.Generator):
    z = float(z0)
    zero_crossing = np.nan

    for k in range(1, MAX_STEPS + 1):
        # g_k = -G + eps_k and z_{k+1}=rho*z_k-gamma*g_k.
        eps = rng.normal(0.0, NOISE_STD)
        z = rho * z + U - GAMMA * eps

        if np.isnan(zero_crossing) and z >= 0.0:
            zero_crossing = float(k)

        if abs(z) >= DELTA:
            event_sign = 1 if z > 0.0 else -1
            return float(k), event_sign, zero_crossing

    return np.nan, 0, zero_crossing


def main():
    deterministic_rows = []
    for z0 in Z0_VALUES:
        for rho in RHO_VALUES:
            deterministic_rows.append(
                {
                    "z0": z0,
                    "rho": rho,
                    "zero_crossing_steps": deterministic_zero_crossing_steps(z0, rho),
                    "first_correct_threshold_steps": deterministic_correct_threshold_steps(z0, rho),
                }
            )

    deterministic = pd.DataFrame(deterministic_rows)
    deterministic.to_csv(OUT / "deterministic_first_passage.csv", index=False)

    rng = np.random.default_rng(20260829)
    noisy_rows = []
    for z0 in Z0_VALUES:
        for rho in RHO_VALUES:
            event_delays, event_signs, zero_delays = [], [], []
            for _ in range(N_MONTE_CARLO):
                event_delay, event_sign, zero_delay = run_noisy_trial(z0, rho, rng)
                event_delays.append(event_delay)
                event_signs.append(event_sign)
                zero_delays.append(zero_delay)

            event_delays = np.asarray(event_delays, dtype=float)
            event_signs = np.asarray(event_signs, dtype=int)
            zero_delays = np.asarray(zero_delays, dtype=float)

            noisy_rows.append(
                {
                    "z0": z0,
                    "rho": rho,
                    "mean_zero_crossing_steps": np.nanmean(zero_delays),
                    "mean_first_event_delay": np.nanmean(event_delays),
                    "p_first_event_correct": np.mean(event_signs == 1),
                    "p_wrong_event": np.mean(event_signs == -1),
                    "p_no_event_by_horizon": np.mean(event_signs == 0),
                }
            )

    noisy = pd.DataFrame(noisy_rows)
    noisy.to_csv(OUT / "noisy_first_passage.csv", index=False)

    plt.figure(figsize=(8, 5))
    for z0 in Z0_VALUES:
        subset = deterministic[deterministic["z0"] == z0]
        plt.plot(
            subset["rho"],
            subset["zero_crossing_steps"],
            marker="o",
            label=f"zero crossing, z0={z0}",
        )
        plt.plot(
            subset["rho"],
            subset["first_correct_threshold_steps"],
            marker="x",
            linestyle="--",
            label=f"+Delta hit, z0={z0}",
        )
    plt.xlabel("Leak retention rho")
    plt.ylabel("Deterministic gradient evaluations")
    plt.title("Leak can erase stale sign sooner without firing sooner")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "deterministic_first_passage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for z0 in Z0_VALUES:
        subset = noisy[noisy["z0"] == z0]
        plt.plot(subset["rho"], subset["mean_zero_crossing_steps"], marker="o", label=f"zero crossing, z0={z0}")
        plt.plot(subset["rho"], subset["mean_first_event_delay"], marker="x", linestyle="--", label=f"first event, z0={z0}")
    plt.xlabel("Leak retention rho")
    plt.ylabel("Mean gradient evaluations")
    plt.title("Noisy stale-state cancellation versus useful firing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "noisy_first_passage.png", dpi=180)
    plt.close()

    print("Deterministic analysis")
    print(deterministic.to_string(index=False))
    print("\nNoisy Monte Carlo analysis")
    print(noisy.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
