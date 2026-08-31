from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vector_quadratic import VectorQuadraticEnsemble, excess_objective


@dataclass(frozen=True)
class SparseIntervalConfig:
    confidence: float = 0.8
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    eta: float = 1.0
    noise_std: float = 0.25
    jump_cap: float = 1.0
    header_bits: int = 32


@dataclass(frozen=True)
class SparseIntervalCalibration:
    confidence: float
    error_table: dict[tuple[int, int, int], float]
    prior_table: dict[tuple[int, int], float]
    global_error: float
    global_prior: float
    coordinate_class: np.ndarray
    local_smoothness_bound: np.ndarray
    global_curvature_bound: np.ndarray


def _tau_bin(values: np.ndarray) -> np.ndarray:
    return np.select(
        [values <= 2, values <= 5, values <= 10, values <= 25, values <= 75],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype(int)


def collect_first_passage_calibration(
    *,
    ensemble: VectorQuadraticEnsemble,
    n_ticks: int,
    seed: int,
    rho: float = 0.999,
    gamma: float = 0.05,
    delta0: float = 0.25,
    jump0: float = 0.02,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.25,
    noise_std: float = 0.25,
) -> pd.DataFrame:
    """Collect event-time gradient estimates on a separate calibration fleet."""

    h, theta = ensemble.h, ensemble.theta
    periods, weights = ensemble.periods, ensemble.weights
    n_runs, dimension, n_clients = (
        ensemble.n_runs,
        ensemble.dimension,
        ensemble.n_clients,
    )
    thresholds = delta0 * ensemble.threshold_scale

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    local_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)
    rows_out: list[dict[str, float | int]] = []

    for tick in range(1, n_ticks + 1):
        z *= rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            gradient = h[:, client, :] * (
                snapshots[client] - theta[:, client, :]
            ) + rng.normal(0.0, noise_std, size=(n_runs, dimension))
            rate_weight = weights[client] * periods[client]
            gamma_i = gamma * rate_weight
            z[client] -= gamma_i * gradient
            mask = np.abs(z[client]) >= thresholds[None, :]

            if np.any(mask):
                run_idx, coord_idx = np.where(mask)
                signs = np.sign(z[client, run_idx, coord_idx])
                tau = local_age[client, run_idx, coord_idx].astype(float)
                retention = rho ** periods[client]
                if abs(1.0 - retention) < 1e-12:
                    magnitude_hat = thresholds[coord_idx] / (
                        gamma_i * np.maximum(tau, 1.0)
                    )
                else:
                    magnitude_hat = (
                        thresholds[coord_idx]
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
                local_gradient = h[run_idx, client, coord_idx] * (
                    w[run_idx, coord_idx] - theta[run_idx, client, coord_idx]
                )
                for r, j, t, gh, gl in zip(
                    run_idx, coord_idx, tau, gradient_hat, local_gradient
                ):
                    rows_out.append(
                        {
                            "client": client,
                            "coordinate": int(j),
                            "period": int(periods[client]),
                            "tau_local": float(t),
                            "gradient_hat": float(gh),
                            "local_gradient": float(gl),
                        }
                    )

                jump = jump0 * (1.0 + tick / schedule_scale) ** (-schedule_exponent)
                np.add.at(w, (run_idx, coord_idx), jump * signs)
                z[client][mask] = 0.0
                local_age[client][mask] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

    return pd.DataFrame(rows_out)


def build_sparse_interval_calibration(
    *,
    ensemble: VectorQuadraticEnsemble,
    event_log: pd.DataFrame,
    confidence: float,
) -> SparseIntervalCalibration:
    """Build held-out empirical event-error/prior radii and smoothness bounds."""

    log = event_log.copy()
    log["tau_bin"] = _tau_bin(log["tau_local"].to_numpy())
    log["error"] = np.abs(log["gradient_hat"] - log["local_gradient"])
    cuts = np.quantile(ensemble.threshold_scale, [1.0 / 3.0, 2.0 / 3.0])
    coordinate_class = np.digitize(ensemble.threshold_scale, cuts)
    log["coord_class"] = coordinate_class[log["coordinate"].to_numpy(dtype=int)]

    global_error = float(log["error"].quantile(confidence))
    error_table: dict[tuple[int, int, int], float] = {}
    for key, subset in log.groupby(["period", "tau_bin", "coord_class"]):
        error_table[tuple(map(int, key))] = (
            float(subset["error"].quantile(confidence))
            if len(subset) >= 40
            else global_error
        )

    log["abs_local"] = np.abs(log["local_gradient"])
    global_prior = float(log["abs_local"].quantile(confidence))
    prior_table: dict[tuple[int, int], float] = {}
    for key, subset in log.groupby(["period", "coord_class"]):
        prior_table[tuple(map(int, key))] = (
            float(subset["abs_local"].quantile(confidence))
            if len(subset) >= 100
            else global_prior
        )

    local_smoothness_bound = np.quantile(
        ensemble.h.reshape(-1, ensemble.dimension), confidence, axis=0
    )
    global_curvature_bound = np.quantile(
        ensemble.hbar, confidence, axis=0
    )
    return SparseIntervalCalibration(
        confidence=confidence,
        error_table=error_table,
        prior_table=prior_table,
        global_error=global_error,
        global_prior=global_prior,
        coordinate_class=coordinate_class,
        local_smoothness_bound=local_smoothness_bound,
        global_curvature_bound=global_curvature_bound,
    )


def run_sparse_interval_certificate(
    *,
    ensemble: VectorQuadraticEnsemble,
    calibration: SparseIntervalCalibration,
    config: SparseIntervalConfig,
    n_ticks: int,
    tail: int,
    seed: int,
    record_coverage: bool = True,
) -> dict[str, float]:
    """Run timing-only staleness-aware global-gradient interval certification.

    The server starts from zero-centered calibrated priors. A firing client-coordinate
    refreshes only that contribution using event sign and first-passage time. Between
    events, uncertainty grows with configured coordinate smoothness and server-model
    displacement. No dense gradient refresh is used.
    """

    h, theta, hbar, wstar = (
        ensemble.h,
        ensemble.theta,
        ensemble.hbar,
        ensemble.wstar,
    )
    periods, weights = ensemble.periods, ensemble.weights
    n_runs, dimension, n_clients = (
        ensemble.n_runs,
        ensemble.dimension,
        ensemble.n_clients,
    )
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

    belief_mean = np.zeros((n_clients, n_runs, dimension))
    belief_radius = np.zeros((n_clients, n_runs, dimension))
    belief_reference = np.repeat(w[None, :, :], n_clients, axis=0)
    for client in range(n_clients):
        radii = np.array(
            [
                calibration.prior_table.get(
                    (
                        int(periods[client]),
                        int(calibration.coordinate_class[j]),
                    ),
                    calibration.global_prior,
                )
                for j in range(dimension)
            ]
        )
        belief_radius[client] = radii[None, :]

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    accepted_events = np.zeros(n_runs, dtype=np.int64)
    harmful_events = np.zeros(n_runs, dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)
    local_cover = global_cover = local_total = global_total = 0

    for tick in range(1, n_ticks + 1):
        z *= config.rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            gradient = h[:, client, :] * (
                snapshots[client] - theta[:, client, :]
            ) + rng.normal(
                0.0, config.noise_std, size=(n_runs, dimension)
            )
            rate_weight = weights[client] * periods[client]
            gamma_i = config.gamma * rate_weight
            z[client] -= gamma_i * gradient
            mask = np.abs(z[client]) >= thresholds[None, :]

            if np.any(mask):
                run_idx, coord_idx = np.where(mask)
                signs = np.sign(z[client, run_idx, coord_idx])
                tau = local_age[client, run_idx, coord_idx].astype(float)
                n_events = mask.sum(axis=1)
                candidate_events += n_events
                payload_bits += n_events * event_bits
                has_packet = n_events > 0
                packetized_bits += (
                    n_events * event_bits
                    + has_packet.astype(np.int64) * config.header_bits
                )

                retention = config.rho ** periods[client]
                if abs(1.0 - retention) < 1e-12:
                    magnitude_hat = thresholds[coord_idx] / (
                        gamma_i * np.maximum(tau, 1.0)
                    )
                else:
                    magnitude_hat = (
                        thresholds[coord_idx]
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
                tau_bins = _tau_bin(tau)
                coord_classes = calibration.coordinate_class[coord_idx]
                base_error = np.array(
                    [
                        calibration.error_table.get(
                            (
                                int(periods[client]),
                                int(tb),
                                int(cc),
                            ),
                            calibration.global_error,
                        )
                        for tb, cc in zip(tau_bins, coord_classes)
                    ]
                )
                belief_mean[client, run_idx, coord_idx] = gradient_hat
                belief_radius[client, run_idx, coord_idx] = base_error
                belief_reference[client, run_idx, coord_idx] = w[run_idx, coord_idx]

                means = belief_mean[:, run_idx, coord_idx]
                radii = belief_radius[:, run_idx, coord_idx]
                refs = belief_reference[:, run_idx, coord_idx]
                current = w[run_idx, coord_idx][None, :]
                smoothness = calibration.local_smoothness_bound[coord_idx][None, :]
                effective_radius = radii + smoothness * np.abs(current - refs)
                aggregate_mean = np.sum(weights[:, None] * means, axis=0)
                aggregate_radius = np.sum(
                    weights[:, None] * effective_radius, axis=0
                )
                lower_alignment = np.maximum(
                    -signs * aggregate_mean - aggregate_radius, 0.0
                )
                jump = (
                    config.eta
                    * lower_alignment
                    / np.maximum(
                        calibration.global_curvature_bound[coord_idx], 1e-12
                    )
                )
                jump = np.clip(jump, 0.0, config.jump_cap)
                accepted = jump > 1e-14
                delta_w = jump * signs
                global_gradient = hbar[run_idx, coord_idx] * (
                    w[run_idx, coord_idx] - wstar[run_idx, coord_idx]
                )
                delta_f = (
                    global_gradient * delta_w
                    + 0.5 * hbar[run_idx, coord_idx] * delta_w**2
                )
                np.add.at(w, (run_idx, coord_idx), delta_w)
                np.add.at(accepted_events, run_idx, accepted.astype(np.int64))
                np.add.at(
                    harmful_events,
                    run_idx,
                    (accepted & (delta_f > 1e-15)).astype(np.int64),
                )

                if record_coverage:
                    true_local = np.stack(
                        [
                            h[run_idx, k, coord_idx]
                            * (
                                w[run_idx, coord_idx]
                                - theta[run_idx, k, coord_idx]
                            )
                            for k in range(n_clients)
                        ],
                        axis=0,
                    )
                    local_cover += int(
                        np.sum(
                            np.abs(true_local - means)
                            <= effective_radius + 1e-12
                        )
                    )
                    local_total += true_local.size
                    true_global = hbar[run_idx, coord_idx] * (
                        w[run_idx, coord_idx] - wstar[run_idx, coord_idx]
                    )
                    global_cover += int(
                        np.sum(
                            np.abs(true_global - aggregate_mean)
                            <= aggregate_radius + 1e-12
                        )
                    )
                    global_total += len(run_idx)

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
    return {
        "confidence": calibration.confidence,
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
            harmful_events.sum() / total_accepted if total_accepted else np.nan
        ),
        "local_belief_coverage": (
            local_cover / local_total if local_total else np.nan
        ),
        "global_belief_coverage": (
            global_cover / global_total if global_total else np.nan
        ),
    }
