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


RESULT_DIR = Path("experiments/results/11a_descent_aware_sizing")


def make_heterogeneity_ensemble(
    *,
    n_runs: int,
    theta_std: float,
    seed: int,
) -> VectorQuadraticEnsemble:
    """Controlled client-optimum heterogeneity sweep for the 10A geometry."""

    n_clients = 10
    dimension = 20
    periods = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
    weights = np.full(n_clients, 1.0 / n_clients)
    base_h = np.geomspace(0.2, 5.0, dimension)
    rng = np.random.default_rng(seed)
    h = base_h[None, None, :] * np.exp(
        rng.normal(0.0, 0.15, size=(n_runs, n_clients, dimension))
    )
    shared_theta = rng.normal(0.0, 0.10, size=(n_runs, 1, dimension))
    theta = shared_theta + rng.normal(
        0.0, theta_std, size=(n_runs, n_clients, dimension)
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


def run_primary(*, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if quick:
        n_runs, n_ticks, tail = 12, 400, 100
        oracle_etas = [1.0]
        event_etas = [0.25, 1.0]
    else:
        n_runs, n_ticks, tail = 80, 2000, 500
        oracle_etas = [0.5, 1.0, 1.5]
        event_etas = [0.25, 0.5, 1.0]

    ensemble = make_diagonal_quadratic_ensemble(n_runs=n_runs, seed=222)
    rows: list[dict[str, float | str]] = []
    estimator_log = None

    def record(method: str, eta: float = 1.0, *, record_estimator: bool = False) -> None:
        nonlocal estimator_log
        result = run_descent_aware_batch(
            ensemble=ensemble,
            config=DescentAwareConfig(method=method, eta=eta),
            n_ticks=n_ticks,
            tail=tail,
            seed=40404,
            record_estimator=record_estimator,
        )
        rows.append(
            {
                "method": method,
                "eta": eta,
                **{k: v for k, v in result.items() if k != "estimator_log"},
            }
        )
        if record_estimator:
            estimator_log = result["estimator_log"]

    record("fixed")
    record("global_schedule")
    for method in ("global_oracle", "local_oracle", "cert_oracle"):
        for eta in oracle_etas:
            record(method, eta)
    for method in ("event_local", "event_cert_oracle"):
        for eta in event_etas:
            record(
                method,
                eta,
                record_estimator=(method == "event_local" and eta == 0.5),
            )

    results = pd.DataFrame(rows)
    if estimator_log is None or estimator_log.empty:
        return results, pd.DataFrame(), pd.DataFrame()
    overall, by_period = estimator_summary(estimator_log)
    return results, overall, by_period


def run_heterogeneity(*, quick: bool) -> pd.DataFrame:
    theta_values = [0.0, 0.10, 0.25] if quick else [0.0, 0.05, 0.10, 0.25, 0.50]
    n_runs, n_ticks, tail = (12, 400, 100) if quick else (30, 1500, 400)
    methods = [
        ("global_oracle", 1.0),
        ("local_oracle", 1.0),
        ("cert_oracle", 1.0),
        ("event_local", 0.25),
        ("event_cert_oracle", 1.0),
    ]
    rows = []
    for theta_std in theta_values:
        ensemble = make_heterogeneity_ensemble(
            n_runs=n_runs,
            theta_std=theta_std,
            seed=700 + int(theta_std * 1000),
        )
        for method, eta in methods:
            result = run_descent_aware_batch(
                ensemble=ensemble,
                config=DescentAwareConfig(method=method, eta=eta),
                n_ticks=n_ticks,
                tail=tail,
                seed=60606,
            )
            rows.append(
                {
                    "theta_std": theta_std,
                    "method": method,
                    "eta": eta,
                    **{k: v for k, v in result.items() if k != "estimator_log"},
                }
            )
    return pd.DataFrame(rows)


def run_noise(*, quick: bool) -> pd.DataFrame:
    noise_values = [0.0, 0.25, 0.50] if quick else [0.0, 0.10, 0.25, 0.50]
    n_runs, n_ticks, tail = (12, 400, 100) if quick else (80, 1500, 400)
    ensemble = make_diagonal_quadratic_ensemble(n_runs=n_runs, seed=222)
    methods = [
        ("global_schedule", 1.0),
        ("global_oracle", 1.0),
        ("local_oracle", 0.5),
        ("cert_oracle", 1.0),
        ("event_local", 0.25),
        ("event_cert_oracle", 1.0),
    ]
    rows = []
    for noise_std in noise_values:
        for method, eta in methods:
            result = run_descent_aware_batch(
                ensemble=ensemble,
                config=DescentAwareConfig(
                    method=method,
                    eta=eta,
                    noise_std=noise_std,
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=70707,
            )
            rows.append(
                {
                    "noise_std": noise_std,
                    "method": method,
                    "eta": eta,
                    **{k: v for k, v in result.items() if k != "estimator_log"},
                }
            )
    return pd.DataFrame(rows)


def save_plots(primary: pd.DataFrame, heterogeneity: pd.DataFrame, noise: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    for method, subset in primary.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["tail_gap"], label=method, s=55)
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11A: descent-aware event sizing")
    plt.legend()
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
        plt.plot(subset["theta_std"], subset["tail_gap"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("Client-optimum heterogeneity sigma_theta")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11A: local descent fails as client objectives separate")
    plt.legend()
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
            subset["theta_std"],
            subset["harmful_applied_fraction"],
            marker="o",
            label=method,
        )
    plt.xlabel("Client-optimum heterogeneity sigma_theta")
    plt.ylabel("Globally harmful fraction of accepted events")
    plt.title("Experiment 11A: local/global descent inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "heterogeneity_harm.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method in [
        "global_schedule",
        "global_oracle",
        "cert_oracle",
        "event_local",
        "event_cert_oracle",
    ]:
        subset = noise[noise["method"] == method]
        plt.plot(subset["noise_std"], subset["tail_gap"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("Gradient-noise standard deviation")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11A: noise robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "noise_tail_gap.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    primary, estimator, estimator_by_period = run_primary(quick=args.quick)
    heterogeneity = run_heterogeneity(quick=args.quick)
    noise = run_noise(quick=args.quick)

    primary.to_csv(RESULT_DIR / "primary_results.csv", index=False)
    heterogeneity.to_csv(RESULT_DIR / "heterogeneity_sweep.csv", index=False)
    noise.to_csv(RESULT_DIR / "noise_sweep.csv", index=False)
    if not estimator.empty:
        estimator.to_csv(RESULT_DIR / "estimator_summary.csv", index=False)
        estimator_by_period.to_csv(
            RESULT_DIR / "estimator_by_period.csv", index=False
        )

    save_plots(primary, heterogeneity, noise)

    print("Experiment 11A complete")
    columns = [
        "method",
        "eta",
        "tail_gap",
        "whole_gap",
        "payload_bits",
        "candidate_events",
        "accepted_events",
        "acceptance_fraction",
        "harmful_applied_fraction",
        "local_good_global_bad_fraction",
    ]
    print(primary.sort_values("tail_gap")[columns].to_string(index=False))
    if not estimator.empty:
        print("\nFirst-passage estimator summary")
        print(estimator.to_string(index=False))


if __name__ == "__main__":
    main()
