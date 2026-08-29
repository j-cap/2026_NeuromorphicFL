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


OUT = ROOT / "experiments" / "results" / "02_moving_optimum"
OUT.mkdir(parents=True, exist_ok=True)

A = 1.0
SIGMA = 2.0
W0 = 3.0
THETA_OLD = 0.0
THETA_NEW = 2.0
SWITCH_STEP = 500
N_STEPS = 3000
N_SEEDS = 300

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
        opt = make_optimizer(label)
        run = run_scalar_optimizer(
            optimizer=opt,
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


def recovery_steps(W: np.ndarray, target: np.ndarray, tol: float = 0.2, hold: int = 100):
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
        first_delays, events100, wrong100 = [], [], []
        for evs in events:
            post100 = [(k, s) for k, s in evs if SWITCH_STEP <= k < SWITCH_STEP + 100]
            events100.append(len(post100))
            wrong100.append(sum(s < 0 for _, s in post100))
            post = [(k, s) for k, s in evs if k >= SWITCH_STEP]
            first_delays.append(post[0][0] - SWITCH_STEP if post else np.nan)

        rows.append(
            {
                "method": label,
                "pre_switch_mse_last_200": np.mean(err[:, SWITCH_STEP - 200 : SWITCH_STEP] ** 2),
                "post_switch_mse_0_100": np.mean(err[:, SWITCH_STEP : SWITCH_STEP + 100] ** 2),
                "post_switch_mse_0_300": np.mean(err[:, SWITCH_STEP : SWITCH_STEP + 300] ** 2),
                "post_switch_mse_0_1000": np.mean(err[:, SWITCH_STEP : SWITCH_STEP + 1000] ** 2),
                "tail_mse_last_500": np.mean(err[:, -500:] ** 2),
                "tail_mae_last_500": np.mean(np.abs(err[:, -500:])),
                "mean_signed_error_last_500": np.mean(err[:, -500:]),
                "recovery_steps": recovery_steps(W, target),
                "mean_first_event_delay": np.nanmean(first_delays) if np.any(~np.isnan(first_delays)) else np.nan,
                "events_first_100": np.mean(events100),
                "wrong_direction_events_first_100": np.mean(wrong100),
                "post_switch_communication": np.mean(C[:, -1] - C[:, SWITCH_STEP]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    mem_rows = []
    for label, (_, Z, _, _) in results.items():
        if np.isnan(Z).all():
            continue
        zsw = Z[:, SWITCH_STEP]
        mem_rows.append(
            {
                "method": label,
                "mean_z_at_switch": np.mean(zsw),
                "mean_abs_z_at_switch": np.mean(np.abs(zsw)),
                "p_z_negative_at_switch": np.mean(zsw < 0),
                "p_abs_z_gt_0p2": np.mean(np.abs(zsw) > 0.2),
            }
        )
    pd.DataFrame(mem_rows).to_csv(OUT / "membrane_at_switch.csv", index=False)

    x = np.arange(N_STEPS + 1)

    # Primary convergence plot: MAE, not signed ensemble mean.
    plt.figure(figsize=(9, 5))
    for label in ["SGD", "signSGD", "IF-SGD", "LIF-SGD rho=0.98"]:
        W = results[label][0]
        mae = np.mean(np.abs(W - target[None, :]), axis=0)
        plt.plot(x, mae, label=label)
    plt.axvline(SWITCH_STEP, linestyle="--")
    plt.xlabel("Gradient evaluations")
    plt.ylabel("Mean absolute tracking error")
    plt.title("Changing optimum: mean absolute error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "mae.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for label in ["SGD", "signSGD", "IF-SGD", "LIF-SGD rho=0.98"]:
        W = results[label][0]
        rmse = np.sqrt(np.mean((W - target[None, :]) ** 2, axis=0))
        plt.plot(x, rmse, label=label)
    plt.axvline(SWITCH_STEP, linestyle="--")
    plt.xlabel("Gradient evaluations")
    plt.ylabel("RMSE to current optimum")
    plt.title("Changing optimum: RMSE")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "rmse.png", dpi=180)
    plt.close()

    # Secondary diagnostic: signed mean + dispersion exposes signSGD cancellation.
    plt.figure(figsize=(9, 5))
    for label in ["SGD", "signSGD"]:
        W = results[label][0]
        mean_w = np.mean(W, axis=0)
        std_w = np.std(W, axis=0)
        plt.plot(x, mean_w, label=f"{label} mean")
        plt.fill_between(x, mean_w - std_w, mean_w + std_w, alpha=0.18)
    plt.plot(x, target, linestyle=":", label="optimum")
    plt.axvline(SWITCH_STEP, linestyle="--")
    plt.xlabel("Gradient evaluations")
    plt.ylabel("Parameter w")
    plt.title("Signed ensemble mean can hide signSGD oscillations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "signed_mean_with_std.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for label in [
        "IF-SGD",
        "IF-SGD oracle reset",
        "LIF-SGD rho=0.995",
        "LIF-SGD rho=0.98",
        "LIF-SGD rho=0.95",
    ]:
        W = results[label][0]
        mae = np.mean(np.abs(W - target[None, :]), axis=0)
        sl = slice(SWITCH_STEP - 100, SWITCH_STEP + 700)
        plt.plot(x[sl], mae[sl], label=label)
    plt.axvline(SWITCH_STEP, linestyle="--")
    plt.xlabel("Gradient evaluations")
    plt.ylabel("Mean |w - optimum|")
    plt.title("Post-switch adaptation and stale membrane evidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "post_switch_adaptation.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
