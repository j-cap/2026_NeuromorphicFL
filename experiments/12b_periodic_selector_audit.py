from __future__ import annotations

"""Audit whether the Experiment-11C periodic Top-K uncertainty selector transfers to 12B.

The canonical Experiment 12B uses candidate-driven on-demand coordinate/block
refresh.  This script adds a deliberately separate negative-control family:
periodically refresh K scalar gradient components chosen by the current absolute
or relative certificate radius.  It is used only as an audit; the winning 12B
algorithm remains the on-demand selective certificate implemented in
``selective_block_certificate.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.logistic_certificate import (
    aggregate_componentwise_bound,
    componentwise_curvature_bound,
    global_gradient,
    make_synthetic_logistic_ensemble,
    minibatch_gradient,
    test_metrics,
    train_loss,
)
from neuromorphicfl.selective_block_certificate import (
    SelectiveCertificateConfig,
    _global_gradient_block_run,
    _train_loss_run,
    run_selective_certificate,
)

RESULT_DIR = Path("experiments/results/12b_selective_block_certificate")


def _singleton_radius(
    *,
    w: np.ndarray,
    calibration_model: np.ndarray,
    aggregate_bound: np.ndarray,
) -> np.ndarray:
    """Exact 12B one-coordinate block+residual radius for all runs/coordinates."""
    n_runs, dimension = w.shape
    diagonal = np.diagonal(aggregate_bound, axis1=1, axis2=2)
    off_sum = np.sum(aggregate_bound, axis=2) - diagonal
    delta = np.abs(w[:, None, :] - calibration_model)
    diagonal_delta = np.diagonal(delta, axis1=1, axis2=2)
    work = delta.copy()
    ids = np.arange(dimension)
    work[:, ids, ids] = -np.inf
    max_outside = np.max(work, axis=2)
    return diagonal * diagonal_delta + off_sum * max_outside


def run_periodic_selector(
    *,
    ensemble,
    selector: str,
    topk: int,
    refresh_period: int,
    n_ticks: int,
    seed: int,
    eval_stride: int = 30,
    verify_harm: bool = False,
) -> dict[str, float | str | int]:
    """Periodic scalar refresh using the same valid 12B singleton certificate.

    ``selector`` is one of ``absolute``, ``relative``, ``random`` or
    ``round_robin``.  The refresh itself is exact for the selected gradient
    coordinates; communication accounts only for the selected scalar components.
    """
    n_runs = ensemble.n_runs
    n_clients = ensemble.n_clients
    dimension = ensemble.dimension
    weights = ensemble.weights
    periods = ensemble.periods
    rng = np.random.default_rng(seed)

    client_bound = componentwise_curvature_bound(ensemble)
    aggregate_bound = aggregate_componentwise_bound(ensemble, client_bound)
    coordinate_L = np.diagonal(aggregate_bound, axis1=1, axis2=2)

    # Scalar block+residual summary: two floats/client/coordinate plus mapping.
    mapping_bits = dimension * int(np.ceil(np.log2(dimension)))
    curvature_bits = 32 * n_clients * 2 * dimension + mapping_bits

    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    membrane = np.zeros((n_clients, n_runs, dimension))

    payload = np.full(n_runs, curvature_bits, dtype=np.int64)
    refresh_bits = np.zeros(n_runs, dtype=np.int64)
    candidates = np.zeros(n_runs, dtype=np.int64)
    accepted = np.zeros(n_runs, dtype=np.int64)
    harmful = np.zeros(n_runs, dtype=np.int64)
    client_accepted = np.zeros((n_runs, n_clients), dtype=np.int64)
    refresh_count = np.ones((n_runs, dimension), dtype=np.int64)

    calibrated_gradient = global_gradient(w, ensemble)
    calibration_model = np.repeat(w[:, None, :], dimension, axis=1)
    initial_bits = n_clients * dimension * 32
    payload += initial_bits
    refresh_bits += initial_bits

    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    round_robin_cursor = 0
    history = []

    def refresh_selected(chosen: np.ndarray) -> None:
        for run in range(n_runs):
            for coordinate in chosen[run]:
                coordinate = int(coordinate)
                calibrated_gradient[run, coordinate] = _global_gradient_block_run(
                    w=w[run],
                    ensemble=ensemble,
                    run=run,
                    block=[coordinate],
                )[0]
                calibration_model[run, coordinate] = w[run]
                refresh_count[run, coordinate] += 1
            bits = n_clients * len(chosen[run]) * 32
            payload[run] += bits
            refresh_bits[run] += bits

    for tick in range(1, n_ticks + 1):
        if refresh_period > 0 and tick % refresh_period == 0:
            radius = _singleton_radius(
                w=w,
                calibration_model=calibration_model,
                aggregate_bound=aggregate_bound,
            )
            k = min(topk, dimension)
            if selector == "absolute":
                score = radius
                chosen = np.argpartition(score, -k, axis=1)[:, -k:]
            elif selector == "relative":
                score = radius / (np.abs(calibrated_gradient) + radius + 1e-12)
                chosen = np.argpartition(score, -k, axis=1)[:, -k:]
            elif selector == "random":
                chosen = np.stack(
                    [rng.choice(dimension, k, replace=False) for _ in range(n_runs)]
                )
            elif selector == "round_robin":
                coords = np.array(
                    [(round_robin_cursor + offset) % dimension for offset in range(k)]
                )
                chosen = np.tile(coords, (n_runs, 1))
                round_robin_cursor = (round_robin_cursor + k) % dimension
            else:
                raise ValueError(f"unknown selector {selector}")
            refresh_selected(chosen)

        membrane *= 0.999
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            gradient = minibatch_gradient(
                snapshots[client], ensemble, client, rng, 64
            )
            membrane[client] -= 0.1 * weights[client] * periods[client] * gradient
            mask = np.abs(membrane[client]) >= 0.05
            if np.any(mask):
                count = mask.sum(axis=1)
                candidates += count
                payload += count * event_bits

                for run in range(n_runs):
                    for coordinate in np.flatnonzero(mask[run]):
                        sign = float(np.sign(membrane[client, run, coordinate]))
                        radius = _singleton_radius(
                            w=w[run : run + 1],
                            calibration_model=calibration_model[run : run + 1],
                            aggregate_bound=aggregate_bound[run : run + 1],
                        )[0, coordinate]
                        lower_alignment = (
                            -sign * calibrated_gradient[run, coordinate] - radius
                        )
                        if lower_alignment > 0.0:
                            jump = lower_alignment / max(
                                float(coordinate_L[run, coordinate]), 1e-12
                            )
                            if verify_harm:
                                before = _train_loss_run(
                                    w=w[run], ensemble=ensemble, run=run
                                )
                            w[run, coordinate] += sign * jump
                            accepted[run] += 1
                            client_accepted[run, client] += 1
                            if verify_harm:
                                after = _train_loss_run(
                                    w=w[run], ensemble=ensemble, run=run
                                )
                                if after > before + 1e-10:
                                    harmful[run] += 1
                membrane[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            test_loss, test_accuracy = test_metrics(w, ensemble)
            history.append(
                {
                    "tick": tick,
                    "train_loss": float(np.mean(train_loss(w, ensemble))),
                    "test_loss": float(np.mean(test_loss)),
                    "test_accuracy": float(np.mean(test_accuracy)),
                    "payload_bits": float(np.mean(payload)),
                }
            )

    history = pd.DataFrame(history)
    total_candidates = int(candidates.sum())
    total_accepted = int(accepted.sum())
    return {
        "selector": selector,
        "topk": topk,
        "refresh_period": refresh_period,
        "final_test_loss": float(history.iloc[-1]["test_loss"]),
        "final_test_accuracy": float(history.iloc[-1]["test_accuracy"]),
        "whole_train_loss": float(history["train_loss"].mean()),
        "payload_bits": float(np.mean(payload)),
        "curvature_bits": float(curvature_bits),
        "refresh_bits": float(np.mean(refresh_bits)),
        "event_bits": float(np.mean(payload) - curvature_bits - np.mean(refresh_bits)),
        "accepted_events": float(np.mean(accepted)),
        "acceptance": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_fraction": (
            float(harmful.sum() / total_accepted)
            if verify_harm and total_accepted
            else np.nan
        ),
        "slow_accepted_share": (
            float(client_accepted[:, -2:].sum() / client_accepted.sum())
            if client_accepted.sum()
            else np.nan
        ),
        "refresh_coverage": float(np.mean((refresh_count > 1).mean(axis=1))),
    }


def main(quick: bool) -> None:
    n_runs, n_ticks = (3, 400) if quick else (8, 1200)
    strong = make_synthetic_logistic_ensemble(
        n_runs=n_runs, heterogeneity="strong", seed=1300
    )

    rows = []
    # Canonical 12B winner.
    canonical = run_selective_certificate(
        ensemble=strong,
        config=SelectiveCertificateConfig(
            block_size=1,
            summary_kind="block_residual",
            min_refresh_gap=50,
        ),
        n_ticks=n_ticks,
        seed=60606,
    )
    canonical.pop("history", None)
    rows.append({"configuration": "on_demand_gap50", **canonical})

    for selector, topk, period in [
        ("absolute", 1, 10),
        ("absolute", 2, 25),
        ("absolute", 4, 25),
        ("absolute", 8, 25),
        ("absolute", 8, 50),
        ("relative", 4, 25),
        ("random", 4, 25),
        ("round_robin", 4, 25),
    ]:
        result = run_periodic_selector(
            ensemble=strong,
            selector=selector,
            topk=topk,
            refresh_period=period,
            n_ticks=n_ticks,
            seed=60606,
        )
        rows.append(
            {
                "configuration": f"periodic_{selector}_K{topk}_C{period}",
                **result,
            }
        )

    output = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULT_DIR / "audit_periodic_selector.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(args.quick)
