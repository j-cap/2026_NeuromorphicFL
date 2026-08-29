from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.objectives import PiecewiseQuadraticObjective
from neuromorphicfl.optimizers import IFSGD, LIFSGD, SGD, SignSGD
from neuromorphicfl.simulation import run_scalar_optimizer

OUT = ROOT / "experiments" / "results" / "03_moderate_optimum_shift"
OUT.mkdir(parents=True, exist_ok=True)

# Let all IF/LIF methods reach the stationary regime before moving the optimum.
A = 1.0
SIGMA = 2.0
W0 = 3.0
THETA_OLD = 0.0
THETA_NEW = 0.5
SWITCH_STEP = 2000
N_STEPS = 3500
N_SEEDS = 400

GAMMA = 0.05
DELTA = 0.50
Q = 0.05
ETA_SGD = Q * GAMMA / DELTA

objective = PiecewiseQuadraticObjective(
    a=A,
    theta_before=THETA_OLD,
    theta_after=THETA_NEW,
    switch_step=SWITCH_STEP,
)


def make_optimizer(label: str):
    if label == "SGD":
        return SGD(learning_rate=ETA_SGD)
    if label == "signSGD":
        return SignSGD(step_size=Q)
    if label in ("IF-SGD", "IF-SGD oracle reset"):
        return IFSGD(gamma=GAMMA, threshold=DELTA, jump=Q)
    if label.startswith("LIF-SGD"):
        rho = float(label.split("=")[-1])
        return LIFSGD(gamma=GAMMA, threshold=DELTA, jump=Q, rho=rho)
    raise ValueError(label)


def run_ensemble(label: str):
    W, Z, C, E = [], [], [], []
    for seed in range(N_SEEDS):
        run = run_scalar_optimizer(
            optimizer=make_optimizer(label),
            n_steps=N_STEPS,
            w0=W0,
            gradient_fn=lambda w, k: objective.gradient(w, k),
            target_fn=lambda k: objective.theta(k),
            noise_std=SIGMA,
            seed=seed,
            membrane_reset_step=SWITCH_STEP if label == "IF-SGD oracle reset" else None,
        )
        W.append(run.w)
        Z.append(run.membrane)
        C.append(run.communications)
        E.append(run.event_signs)
    return np.vstack(W), np.vstack(Z), np.vstack(C), E


def recovery_steps(W: np.ndarray, target: np.ndarray, tol: float = 0.10, hold: int = 100):
    mae = np.mean(np.abs(W - target[None, :]), axis=0)
    for k in range(SWITCH_STEP, N_STEPS + 1 - hold):
        if np.all(mae[k : k + hold] < tol):
            return k - SWITCH_STEP
    return np.nan


def main():
    labels = [
        "SGD",
        "signSGD",
        "IF-SGD",
        "IF-SGD oracle reset",
        "LIF-SGD rho=0.995",
        "LIF-SGD rho=0.98",
        "LIF-SGD rho=0.95",
    ]
    results = {label: run_ensemble(label) for label in labels}
    target = np.array([objective.theta(k) for k in range(N_STEPS + 1)])

    rows = []
    for label, (W, Z, C, events) in results.items():
        err = W - target[None, :]
        first_delays, first_correct = [], []
        for run_idx, evs in enumerate(events):
            post = [(k, s) for k, s in evs if k >= SWITCH_STEP]
            first_delays.append(post[0][0] - SWITCH_STEP if post else np.nan)
            if post:
                desired = np.sign(THETA_NEW - W[run_idx, SWITCH_STEP])
                first_correct.append(post[0][1] == desired if desired != 0 else True)
            else:
                first_correct.append(np.nan)

        rows.append(
            {
                "method": label,
                "pre_switch_mae_last_200": np.mean(np.abs(err[:, SWITCH_STEP - 200 : SWITCH_STEP])),
                "post_switch_mae_0_50": np.mean(np.abs(err[:, SWITCH_STEP : SWITCH_STEP + 50])),
                "post_switch_mae_0_200": np.mean(np.abs(err[:, SWITCH_STEP : SWITCH_STEP + 200])),
                "post_switch_mae_0_500": np.mean(np.abs(err[:, SWITCH_STEP : SWITCH_STEP + 500])),
                "tail_rmse_last_500": np.sqrt(np.mean(err[:, -500:] ** 2)),
                "mean_first_event_delay": np.nanmean(first_delays) if np.any(np.isfinite(first_delays)) else np.nan,
                "p_first_event_correct": np.nanmean(first_correct) if np.any(pd.notna(first_correct)) else np.nan,
                "recovery_steps_mae_lt_0p1": recovery_steps(W, target),
                "post_switch_communication": np.mean(C[:, -1] - C[:, SWITCH_STEP]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    membrane_rows = []
    for label, (W, Z, _, _) in results.items():
        if np.isnan(Z).all():
            continue
        z_switch = Z[:, SWITCH_STEP]
        desired = np.sign(THETA_NEW - W[:, SWITCH_STEP])
        membrane_rows.append(
            {
                "method": label,
                "mean_z_at_switch": np.mean(z_switch),
                "mean_abs_z_at_switch": np.mean(np.abs(z_switch)),
                "fraction_membrane_opposes_new_direction": np.mean(np.sign(z_switch) == -desired),
            }
        )
    pd.DataFrame(membrane_rows).to_csv(OUT / "membrane_at_switch.csv", index=False)

    x = np.arange(N_STEPS + 1)
    plt.figure(figsize=(9, 5))
    for label in labels:
        W = results[label][0]
        mae = np.mean(np.abs(W - target[None, :]), axis=0)
        sl = slice(SWITCH_STEP - 150, SWITCH_STEP + 800)
        plt.plot(x[sl], mae[sl], label=label)
    plt.axvline(SWITCH_STEP, linestyle="--")
    plt.xlabel("Gradient evaluations")
    plt.ylabel("Mean absolute tracking error")
    plt.title("Moderate optimum shift after stationary convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "mae_post_switch.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    event_methods = summary[summary["method"].str.contains("IF|LIF", regex=True)]
    plt.plot(event_methods["post_switch_communication"], event_methods["post_switch_mae_0_500"], marker="o")
    for _, row in event_methods.iterrows():
        plt.annotate(row["method"], (row["post_switch_communication"], row["post_switch_mae_0_500"]))
    plt.xlabel("Post-switch communication events")
    plt.ylabel("MAE over first 500 post-switch steps")
    plt.title("Communication--adaptation trade-off after a moderate shift")
    plt.tight_layout()
    plt.savefig(OUT / "communication_tradeoff.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
