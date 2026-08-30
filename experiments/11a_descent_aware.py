from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.descent_aware import (
    DescentAwareConfig,
    estimator_summary,
    run_descent_aware_batch,
)
from neuromorphicfl.vector_quadratic import (
    VectorQuadraticEnsemble,
    make_diagonal_quadratic_ensemble,
)


RESULT_DIR = Path("experiments/results/11a_descent_aware")


def make_heterogeneity_ensemble(
    *, n_runs: int, theta_heterogeneity: float, seed: int
) -> VectorQuadraticEnsemble:
    n_clients = 10
    dimension = 20
    periods = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
    weights = np.full(n_clients, 1.0 / n_clients)
    base_h = np.geomspace(0.2, 5.0, dimension)
    rng = np.random.default_rng(seed)
    h = base_h[None, None, :] * np.exp(
        rng.normal(0.0, 0.15, size=(n_runs, n_clients, dimension))
    )
    theta0 = rng.normal(0.0, 0.10, size=(n_runs, 1, dimension))
    theta = theta0 + rng.normal(
        0.0, theta_heterogeneity, size=(n_runs, n_clients, dimension)
    )
    hbar = np.sum(weights[None, :, None] * h, axis=1)
    wstar = np.sum(weights[None, :, None] * h * theta, axis=1) / hbar
    w0 = wstar + 0.8
    threshold_scale = base_h / np.median(base_h)
    return VectorQuadraticEnsemble(
        h=h,
        theta=theta,
        hbar=hbar,
        wstar=wstar,
        w0=w0,
        threshold_scale=threshold_scale,
        periods=periods,
        weights=weights,
    )


def run_one(
    *,
    ensemble: VectorQuadraticEnsemble,
    method: str,
    eta: float,
    n_ticks: int,
    tail: int,
    noise_std: float = 0.25,
    seed: int = 40404,
    record_estimator: bool = False,
) -> dict[str, object]:
    result = run_descent_aware_batch(
        ensemble=ensemble,
        config=DescentAwareConfig(
            method=method,
            eta=eta,
            noise_std=noise_std,
        ),
        n_ticks=n_ticks,
        tail=tail,
        seed=seed,
        record_estimator=record_estimator,
    )
    return {
        "method": method,
        "eta": eta,
        "noise_std": noise_std,
        **{k: v for k, v in result.items() if k != "estimator_log"},
        "estimator_log": result["estimator_log"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_runs, n_ticks, tail = 10, 400, 100
        eta_oracle = [0.5, 1.0]
        eta_event = [0.25, 0.5]
        heterogeneity_grid = [0.0, 0.25]
        noise_grid = [0.0, 0.25]
        diagnostic_runs, diagnostic_ticks, diagnostic_tail = 8, 300, 75
    else:
        n_runs, n_ticks, tail = 80, 2000, 500
        eta_oracle = [0.5, 1.0, 1.5]
        eta_event = [0.25, 0.5, 1.0]
        heterogeneity_grid = [0.0, 0.05, 0.10, 0.25, 0.50]
        noise_grid = [0.0, 0.10, 0.25, 0.50]
        diagnostic_runs, diagnostic_ticks, diagnostic_tail = 30, 1500, 400

    ensemble = make_diagonal_quadratic_ensemble(n_runs=n_runs, seed=222)
    rows: list[dict[str, object]] = []
    estimator_log = None

    for method in ["fixed", "global_schedule"]:
        out = run_one(
            ensemble=ensemble,
            method=method,
            eta=1.0,
            n_ticks=n_ticks,
            tail=tail,
        )
        rows.append({k: v for k, v in out.items() if k != "estimator_log"})

    for method in ["global_oracle", "local_oracle", "cert_oracle"]:
        for eta in eta_oracle:
            out = run_one(
                ensemble=ensemble,
                method=method,
                eta=eta,
                n_ticks=n_ticks,
                tail=tail,
            )
            rows.append({k: v for k, v in out.items() if k != "estimator_log"})

    for method in ["event_local", "event_cert_oracle"]:
        for eta in eta_event:
            record = method == "event_local" and np.isclose(eta, 0.5)
            out = run_one(
                ensemble=ensemble,
                method=method,
                eta=eta,
                n_ticks=n_ticks,
                tail=tail,
                record_estimator=record,
            )
            rows.append({k: v for k, v in out.items() if k != "estimator_log"})
            if record:
                estimator_log = out["estimator_log"]

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "main_results.csv", index=False)

    best_rows = []
    for method, subset in results.groupby("method"):
        best = subset.loc[subset["tail_gap"].idxmin()]
        best_rows.append(best.to_dict())
    pd.DataFrame(best_rows).to_csv(RESULT_DIR / "family_best.csv", index=False)

    if estimator_log is not None and len(estimator_log):
        overall, by_period = estimator_summary(estimator_log)
        overall.to_csv(RESULT_DIR / "estimator_summary.csv", index=False)
        by_period.to_csv(RESULT_DIR / "estimator_by_period.csv", index=False)

    # Causal client-heterogeneity sweep.
    heterogeneity_rows = []
    for sigma_theta in heterogeneity_grid:
        het_ensemble = make_heterogeneity_ensemble(
            n_runs=diagnostic_runs,
            theta_heterogeneity=sigma_theta,
            seed=700 + int(1000 * sigma_theta),
        )
        for method, eta in [
            ("global_oracle", 1.0),
            ("local_oracle", 1.0),
            ("cert_oracle", 1.0),
            ("event_local", 0.25),
            ("event_cert_oracle", 1.0),
        ]:
            out = run_one(
                ensemble=het_ensemble,
                method=method,
                eta=eta,
                n_ticks=diagnostic_ticks,
                tail=diagnostic_tail,
                seed=60606,
            )
            heterogeneity_rows.append(
                {
                    "sigma_theta": sigma_theta,
                    **{k: v for k, v in out.items() if k != "estimator_log"},
                }
            )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    heterogeneity.to_csv(RESULT_DIR / "heterogeneity_sweep.csv", index=False)

    # Gradient-noise robustness on the original heterogeneous ensemble.
    diagnostic_ensemble = make_diagonal_quadratic_ensemble(
        n_runs=diagnostic_runs, seed=222
    )
    noise_rows = []
    for sigma_g in noise_grid:
        for method, eta in [
            ("global_schedule", 1.0),
            ("global_oracle", 1.0),
            ("local_oracle", 0.5),
            ("cert_oracle", 1.0),
            ("event_local", 0.25),
            ("event_cert_oracle", 1.0),
        ]:
            out = run_one(
                ensemble=diagnostic_ensemble,
                method=method,
                eta=eta,
                n_ticks=diagnostic_ticks,
                tail=diagnostic_tail,
                noise_std=sigma_g,
                seed=70707,
            )
            noise_rows.append(
                {
                    "sigma_g": sigma_g,
                    **{k: v for k, v in out.items() if k != "estimator_log"},
                }
            )
    noise_results = pd.DataFrame(noise_rows)
    noise_results.to_csv(RESULT_DIR / "noise_sweep.csv", index=False)

    # Figures.
    plt.figure(figsize=(9, 6))
    for method, subset in results.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["tail_gap"], label=method, s=55)
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11A: descent-aware event sizing")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "tail_gap_vs_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [
        "global_oracle",
        "local_oracle",
        "cert_oracle",
        "event_local",
        "event_cert_oracle",
    ]:
        subset = heterogeneity[heterogeneity["method"] == method]
        plt.plot(
            subset["sigma_theta"], subset["tail_gap"], marker="o", label=method
        )
    plt.yscale("log")
    plt.xlabel("Client-optimum heterogeneity")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11A: price of local information under heterogeneity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "heterogeneity_tail_gap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [
        "global_oracle",
        "local_oracle",
        "cert_oracle",
        "event_local",
        "event_cert_oracle",
    ]:
        subset = heterogeneity[heterogeneity["method"] == method]
        plt.plot(
            subset["sigma_theta"],
            subset["harmful_applied_fraction"],
            marker="o",
            label=method,
        )
    plt.xlabel("Client-optimum heterogeneity")
    plt.ylabel("Globally harmful fraction of accepted events")
    plt.title("Experiment 11A: local/global descent inconsistency")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "heterogeneity_harmful_events.png", dpi=180)
    plt.close()

    print("Experiment 11A complete")
    print(pd.DataFrame(best_rows).to_string(index=False))


if __name__ == "__main__":
    main()
