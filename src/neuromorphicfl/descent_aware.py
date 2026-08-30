from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .vector_quadratic import VectorQuadraticEnsemble, excess_objective


DescentMethod = Literal[
    "fixed",
    "global_schedule",
    "global_oracle",
    "local_oracle",
    "cert_oracle",
    "event_local",
    "event_cert_oracle",
]


@dataclass(frozen=True)
class DescentAwareConfig:
    method: DescentMethod
    eta: float = 1.0
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    fixed_jump: float = 0.015
    schedule_jump0: float = 0.02
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.25
    noise_std: float = 0.25
    jump_cap: float = 1.0
    header_bits: int = 32


def run_descent_aware_batch(
    *,
    ensemble: VectorQuadraticEnsemble,
    config: DescentAwareConfig,
    n_ticks: int,
    tail: int,
    seed: int,
    record_estimator: bool = False,
) -> dict[str, object]:
    """Run Experiment 11A on the normalized full-reset coordinate-LIF trigger.

    A threshold crossing is always a transmitted candidate event and resets the
    corresponding membrane. The selected method changes only server-side event
    acceptance and jump amplitude. Rejected candidate events therefore still
    count toward communication. This deliberately isolates descent-aware sizing
    from any later attempt to modify the trigger itself.

    The oracle variants use exact synthetic quadratic information and are not
    deployable federated algorithms. They are upper-bound / information-gap
    diagnostics.
    """

    h = ensemble.h
    theta = ensemble.theta
    hbar = ensemble.hbar
    wstar = ensemble.wstar
    periods = ensemble.periods
    weights = ensemble.weights
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    n_clients = ensemble.n_clients
    tail = min(tail, n_ticks)

    thresholds = config.delta0 * ensemble.threshold_scale
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    local_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    accepted_events = np.zeros(n_runs, dtype=np.int64)
    harmful_applied = np.zeros(n_runs, dtype=np.int64)
    local_good_global_bad = np.zeros(n_runs, dtype=np.int64)
    client_candidates = np.zeros((n_runs, n_clients), dtype=np.int64)
    client_accepted = np.zeros((n_runs, n_clients), dtype=np.int64)
    coordinate_candidates = np.zeros((n_runs, dimension), dtype=np.int64)
    coordinate_accepted = np.zeros((n_runs, dimension), dtype=np.int64)

    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)
    jump_values: list[float] = []
    estimator_rows: list[dict[str, object]] = []

    for tick in range(1, n_ticks + 1):
        z *= config.rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            noisy_gradient = h[:, client, :] * (
                snapshots[client] - theta[:, client, :]
            ) + rng.normal(0.0, config.noise_std, size=(n_runs, dimension))
            rate_weight = weights[client] * periods[client]
            gamma_i = config.gamma * rate_weight
            z[client] -= gamma_i * noisy_gradient

            mask = np.abs(z[client]) >= thresholds[None, :]
            if np.any(mask):
                rows, cols = np.where(mask)
                signs = np.sign(z[client, rows, cols])

                n_events = mask.sum(axis=1)
                candidate_events += n_events
                client_candidates[:, client] += n_events
                coordinate_candidates += mask.astype(np.int64)
                payload_bits += n_events * event_bits
                has_packet = n_events > 0
                packetized_bits += (
                    n_events * event_bits
                    + has_packet.astype(np.int64) * config.header_bits
                )

                w_before = w[rows, cols]
                global_curvature = hbar[rows, cols]
                global_gradient = global_curvature * (w_before - wstar[rows, cols])
                local_curvature = h[rows, client, cols]
                local_gradient = local_curvature * (
                    w_before - theta[rows, client, cols]
                )
                global_alignment = -signs * global_gradient
                local_alignment = -signs * local_gradient

                if config.method == "fixed":
                    jump = np.full(len(rows), config.fixed_jump)
                elif config.method == "global_schedule":
                    jump = np.full(
                        len(rows),
                        config.schedule_jump0
                        * (1.0 + tick / config.schedule_scale)
                        ** (-config.schedule_exponent),
                    )
                elif config.method == "global_oracle":
                    jump = (
                        config.eta
                        * np.maximum(global_alignment, 0.0)
                        / global_curvature
                    )
                elif config.method == "local_oracle":
                    jump = (
                        config.eta
                        * np.maximum(local_alignment, 0.0)
                        / local_curvature
                    )
                elif config.method == "cert_oracle":
                    disagreement = np.abs(local_gradient - global_gradient)
                    lower_alignment = np.maximum(
                        local_alignment - disagreement, 0.0
                    )
                    jump = config.eta * lower_alignment / global_curvature
                elif config.method in ("event_local", "event_cert_oracle"):
                    tau = local_age[client, rows, cols].astype(float)
                    retention = config.rho ** periods[client]
                    if abs(1.0 - retention) < 1e-10:
                        magnitude_hat = thresholds[cols] / (
                            gamma_i * np.maximum(tau, 1.0)
                        )
                    else:
                        magnitude_hat = (
                            thresholds[cols]
                            * (1.0 - retention)
                            / (
                                gamma_i
                                * np.maximum(
                                    1.0 - retention ** np.maximum(tau, 1.0),
                                    1e-15,
                                )
                            )
                        )

                    gradient_hat = -signs * magnitude_hat
                    if config.method == "event_local":
                        jump = config.eta * magnitude_hat / local_curvature
                    else:
                        # Oracle diagnostic: exact signed estimation error and exact
                        # local/global disagreement form a valid conservative margin.
                        estimation_error = np.abs(gradient_hat - local_gradient)
                        disagreement = np.abs(local_gradient - global_gradient)
                        lower_alignment = np.maximum(
                            magnitude_hat - estimation_error - disagreement,
                            0.0,
                        )
                        jump = config.eta * lower_alignment / global_curvature

                    if record_estimator:
                        for coord, event_tau, mag_hat, grad_hat, grad_local, grad_global in zip(
                            cols,
                            tau,
                            magnitude_hat,
                            gradient_hat,
                            local_gradient,
                            global_gradient,
                        ):
                            estimator_rows.append(
                                {
                                    "client": client,
                                    "coordinate": int(coord),
                                    "period": int(periods[client]),
                                    "tau_local": float(event_tau),
                                    "magnitude_hat": float(mag_hat),
                                    "gradient_hat": float(grad_hat),
                                    "local_gradient": float(grad_local),
                                    "global_gradient": float(grad_global),
                                }
                            )
                else:
                    raise ValueError(f"unknown method {config.method}")

                jump = np.clip(jump, 0.0, config.jump_cap)
                accepted = jump > 1e-14
                delta_w = jump * signs

                # Exact one-coordinate objective changes for the diagonal quadratics.
                delta_global = (
                    global_gradient * delta_w
                    + 0.5 * global_curvature * delta_w**2
                )
                delta_local = (
                    local_gradient * delta_w
                    + 0.5 * local_curvature * delta_w**2
                )
                np.add.at(w, (rows, cols), delta_w)

                np.add.at(accepted_events, rows, accepted.astype(np.int64))
                np.add.at(
                    harmful_applied,
                    rows,
                    (accepted & (delta_global > 1e-15)).astype(np.int64),
                )
                np.add.at(
                    local_good_global_bad,
                    rows,
                    (
                        accepted
                        & (delta_local < -1e-15)
                        & (delta_global > 1e-15)
                    ).astype(np.int64),
                )
                for idx, run in enumerate(rows):
                    if accepted[idx]:
                        client_accepted[run, client] += 1
                        coordinate_accepted[run, cols[idx]] += 1
                        jump_values.append(float(jump[idx]))

                # A candidate spike fires/resets even if the server rejects q=0.
                z[client][mask] = 0.0
                local_age[client][mask] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        current_gap = excess_objective(w, wstar, hbar)
        whole_gap += current_gap
        if tick > n_ticks - tail:
            tail_gap += current_gap
            tail_dist2 += np.sum((w - wstar) ** 2, axis=1)

    total_candidates = int(candidate_events.sum())
    total_accepted = int(accepted_events.sum())
    slow_start = max(0, n_clients - 2)

    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(
            np.sqrt(np.mean(tail_dist2 / (tail * dimension)))
        ),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "candidate_events": float(np.mean(candidate_events)),
        "accepted_events": float(np.mean(accepted_events)),
        "acceptance_fraction": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_applied_fraction": (
            harmful_applied.sum() / total_accepted if total_accepted else np.nan
        ),
        "local_good_global_bad_fraction": (
            local_good_global_bad.sum() / total_accepted
            if total_accepted
            else np.nan
        ),
        "slow_candidate_share": (
            client_candidates[:, slow_start:].sum() / client_candidates.sum()
            if client_candidates.sum()
            else np.nan
        ),
        "slow_accepted_share": (
            client_accepted[:, slow_start:].sum() / client_accepted.sum()
            if client_accepted.sum()
            else np.nan
        ),
        "candidate_coordinate_coverage": float(
            np.mean((coordinate_candidates > 0).mean(axis=1))
        ),
        "accepted_coordinate_coverage": float(
            np.mean((coordinate_accepted > 0).mean(axis=1))
        ),
        "jump_mean": float(np.mean(jump_values)) if jump_values else 0.0,
        "jump_p50": float(np.median(jump_values)) if jump_values else 0.0,
        "jump_p95": (
            float(np.quantile(jump_values, 0.95)) if jump_values else 0.0
        ),
        "jump_max": float(np.max(jump_values)) if jump_values else 0.0,
        "estimator_log": pd.DataFrame(estimator_rows)
        if record_estimator
        else None,
    }


def estimator_summary(event_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize first-passage gradient-magnitude/sign calibration."""

    log = event_log.copy()
    log["abs_local"] = np.abs(log["local_gradient"])
    log["abs_global"] = np.abs(log["global_gradient"])
    log["relative_magnitude_error"] = (
        np.abs(log["magnitude_hat"] - log["abs_local"])
        / np.maximum(log["abs_local"], 1e-9)
    )
    log["local_sign_correct"] = (
        np.sign(log["gradient_hat"]) == np.sign(log["local_gradient"])
    )
    overall = pd.DataFrame(
        [
            {
                "events": len(log),
                "median_magnitude_hat": log["magnitude_hat"].median(),
                "median_abs_local_gradient": log["abs_local"].median(),
                "median_relative_magnitude_error": log[
                    "relative_magnitude_error"
                ].median(),
                "p90_relative_magnitude_error": log[
                    "relative_magnitude_error"
                ].quantile(0.9),
                "local_sign_accuracy": log["local_sign_correct"].mean(),
                "spearman_hat_vs_abs_local": log["magnitude_hat"].corr(
                    log["abs_local"], method="spearman"
                ),
                "spearman_hat_vs_abs_global": log["magnitude_hat"].corr(
                    log["abs_global"], method="spearman"
                ),
            }
        ]
    )
    by_period = (
        log.groupby("period")
        .agg(
            events=("magnitude_hat", "size"),
            median_relative_error=("relative_magnitude_error", "median"),
            p90_relative_error=(
                "relative_magnitude_error",
                lambda x: x.quantile(0.9),
            ),
            sign_accuracy=("local_sign_correct", "mean"),
        )
        .reset_index()
    )
    correlations = []
    for period, subset in log.groupby("period"):
        correlations.append(
            {
                "period": period,
                "spearman_magnitude": subset["magnitude_hat"].corr(
                    subset["abs_local"], method="spearman"
                ),
            }
        )
    corr = pd.DataFrame(correlations)
    by_period = by_period.merge(corr, on="period", how="left")
    return overall, by_period
