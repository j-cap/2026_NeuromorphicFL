from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuromorphicfl.metrics import communication_reduction
from neuromorphicfl.objectives import QuadraticObjective
from neuromorphicfl.optimizers import IFSGD, LIFSGD, SGD, SignSGD
from neuromorphicfl.simulation import run_scalar_optimizer


OUT = ROOT / "experiments" / "results" / "01_stochastic_quadratic"
OUT.mkdir(parents=True, exist_ok=True)

A = 1.0
SIGMA = 2.0
W0 = 3.0
N_STEPS = 5000
N_SEEDS = 200
GAMMA = 0.05
DELTA = 0.50
Q = 0.05
RHO = 0.98
ETA_SGD = Q * GAMMA / DELTA

objective = QuadraticObjective(a=A, theta=0.0)


def make_optimizer(name: str, rho: float = RHO):
    if name == "SGD":
        return SGD(learning_rate=ETA_SGD)
    if name == "signSGD":
        return SignSGD(step_size=Q)
    if name == "IF-SGD":
        return IFSGD(gamma=GAMMA, threshold=DELTA, jump=Q)
    if name == "LIF-SGD":
        return LIFSGD(gamma=GAMMA, threshold=DELTA, jump=Q, rho=rho)
    raise ValueError(name)


def run_ensemble(name: str, rho: float = RHO):
    trajectories = []
    communications = []
    for seed in range(N_SEEDS):
        opt = make_optimizer(name, rho=rho)
        run = run_scalar_optimizer(
            optimizer=opt,
            n_steps=N_STEPS,
            w0=W0,
            gradient_fn=lambda w, k: objective.gradient(w),
            target_fn=lambda k: objective.theta,
            noise_std=SIGMA,
            seed=seed,
        )
        trajectories.append(run.w)
        communications.append(run.communications)
    return np.vstack(trajectories), np.vstack(communications)


def fixed_w_encoder_stats(w: float, rho: float, n: int = 150_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = 0.0
    events = 0
    wrong = 0
    for _ in range(n):
        g = A * w + rng.normal(0.0, SIGMA)
        z = rho * z - GAMMA * g
        while abs(z) >= DELTA:
            s = 1 if z > 0.0 else -1
            events += 1
            if w != 0.0 and s != -np.sign(w):
                wrong += 1
            z -= s * DELTA
    rate = events / n
    p_wrong = np.nan if events == 0 or w == 0.0 else wrong / events
    return rate, p_wrong


def gaussian_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def main():
    methods = ["SGD", "signSGD", "IF-SGD", "LIF-SGD"]
    ensembles = {m: run_ensemble(m) for m in methods}

    rows = []
    for method, (W, C) in ensembles.items():
        rows.append(
            {
                "method": method,
                "mean_final_abs_w": np.mean(np.abs(W[:, -1])),
                "tail_mse_last_1000": np.mean(W[:, -1000:] ** 2),
                "whole_run_mean_w2": np.mean(W ** 2),
                "mean_communication_events": np.mean(C[:, -1]),
                "communication_reduction_vs_periodic": communication_reduction(
                    np.mean(C[:, -1]), N_STEPS
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "optimization_summary.csv", index=False)

    w_grid = np.array([0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 3.0])
    enc_rows = []
    for w in w_grid:
        if_rate, if_wrong = fixed_w_encoder_stats(w, rho=1.0, seed=100 + int(100 * w))
        lif_rate, lif_wrong = fixed_w_encoder_stats(w, rho=RHO, seed=200 + int(100 * w))
        sign_wrong = 0.5 if w == 0.0 else gaussian_cdf(-A * abs(w) / SIGMA)
        enc_rows.append(
            {
                "abs_w": w,
                "abs_true_gradient": A * w,
                "signsgd_wrong_probability": sign_wrong,
                "if_event_rate": if_rate,
                "if_wrong_event_probability": if_wrong,
                "lif_event_rate": lif_rate,
                "lif_wrong_event_probability": lif_wrong,
            }
        )
    encoder = pd.DataFrame(enc_rows)
    encoder.to_csv(OUT / "encoder_characterization.csv", index=False)

    rho_values = [1.0, 0.995, 0.99, 0.98, 0.95, 0.90]
    sweep_rows = []
    for rho in rho_values:
        method = "IF-SGD" if rho == 1.0 else "LIF-SGD"
        W, C = run_ensemble(method, rho=rho)
        sweep_rows.append(
            {
                "rho": rho,
                "memory_horizon_proxy": np.inf if rho == 1.0 else 1.0 / (1.0 - rho),
                "mean_communication_events": np.mean(C[:, -1]),
                "tail_mse": np.mean(W[:, -1000:] ** 2),
                "whole_run_mean_w2": np.mean(W ** 2),
                "mean_final_abs_w": np.mean(np.abs(W[:, -1])),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "leak_sweep.csv", index=False)

    x = np.arange(N_STEPS + 1)
    plt.figure(figsize=(8, 5))
    for method, (W, _) in ensembles.items():
        plt.plot(x, np.mean(np.abs(W), axis=0), label=method)
    plt.xlabel("Minibatch / gradient evaluations")
    plt.ylabel("Mean |w|")
    plt.title("Stochastic quadratic optimization")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "optimization_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(encoder.abs_w, encoder.if_event_rate, marker="o", label="IF-SGD")
    plt.plot(encoder.abs_w, encoder.lif_event_rate, marker="o", label=f"LIF-SGD rho={RHO}")
    plt.plot(w_grid, GAMMA * A * w_grid / DELTA, linestyle="--", label="Noiseless high-signal prediction")
    plt.xlabel("|w|")
    plt.ylabel("Communication events per minibatch")
    plt.title("Gradient magnitude encoded in event rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "event_rate.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(encoder.abs_w, encoder.signsgd_wrong_probability, marker="o", label="signSGD")
    plt.plot(encoder.abs_w, encoder.if_wrong_event_probability, marker="o", label="IF-SGD")
    plt.plot(encoder.abs_w, encoder.lif_wrong_event_probability, marker="o", label=f"LIF-SGD rho={RHO}")
    plt.xlabel("|w|")
    plt.ylabel("Wrong-sign probability")
    plt.title("Temporal accumulation improves sign reliability")
    plt.ylim(-0.02, 0.52)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "wrong_sign_probability.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sweep.mean_communication_events, sweep.whole_run_mean_w2, marker="o")
    for _, row in sweep.iterrows():
        label = "IF" if row.rho == 1.0 else f"rho={row.rho:.3g}"
        plt.annotate(label, (row.mean_communication_events, row.whole_run_mean_w2))
    plt.xlabel("Mean communication events")
    plt.ylabel("Whole-run mean w^2")
    plt.title("Leak communication–responsiveness trade-off")
    plt.tight_layout()
    plt.savefig(OUT / "leak_tradeoff.png", dpi=180)
    plt.close()

    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
