from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .rotated_geometry import RotatedQuadraticEnsemble, rotated_excess_objective


CertificateMethod = Literal[
    "schedule",
    "global_oracle",
    "periodic_naive",
    "periodic_cert",
    "periodic_event_cert",
    "adaptive_cert",
]
BoundMode = Literal["componentwise", "row_l2"]


@dataclass(frozen=True)
class CalibratedCertificateConfig:
    method: CertificateMethod
    calibration_period: int = 200
    min_refresh_gap: int = 50
    bound_mode: BoundMode = "componentwise"
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    jump0: float = 0.02
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.25
    noise_std: float = 0.25
    header_bits: int = 32


def _local_gradients(
    w: np.ndarray,
    ensemble: RotatedQuadraticEnsemble,
) -> np.ndarray:
    error = w[:, None, :] - ensemble.theta
    return np.einsum("rcjk,rck->rcj", ensemble.H, error)


def calibration_payload_bits(ensemble: RotatedQuadraticEnsemble) -> int:
    return int(ensemble.n_clients * ensemble.dimension * 32)


def calibration_packetized_bits(
    ensemble: RotatedQuadraticEnsemble,
    header_bits: int = 32,
) -> int:
    return int(
        ensemble.n_clients * (ensemble.dimension * 32 + header_bits)
    )


def estimate_first_passage_error_margins(
    *,
    ensemble: RotatedQuadraticEnsemble,
    quantile: float = 0.99,
    n_ticks: int = 1200,
    seed: int = 8181,
    rho: float = 0.999,
    gamma: float = 0.05,
    delta0: float = 0.25,
    noise_std: float = 0.25,
    jump0: float = 0.02,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.25,
) -> dict[int, float]:
    """Offline empirical margins for the event-time local-gradient estimate.

    This is a diagnostic/statistical helper, not a deterministic certificate.
    Errors are grouped by client compute period so slow-client degradation is
    retained rather than hidden by one pooled margin.
    """

    H = ensemble.H
    theta = ensemble.theta
    U = ensemble.U
    periods = ensemble.periods
    weights = ensemble.weights
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    n_clients = ensemble.n_clients

    thresholds = delta0 * ensemble.diag_scale
    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    local_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)
    errors: dict[int, list[float]] = {
        int(period): [] for period in np.unique(periods)
    }

    for tick in range(1, n_ticks + 1):
        z *= rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            deterministic = np.einsum(
                "rij,rj->ri",
                H[:, client],
                snapshots[client] - theta[:, client],
            )
            noise_latent = rng.normal(
                0.0, noise_std, size=(n_runs, dimension)
            )
            noise_native = np.einsum("rnl,rl->rn", U, noise_latent)
            gradient = deterministic + noise_native
            rate_weight = weights[client] * periods[client]
            gamma_i = gamma * rate_weight
            z[client] -= gamma_i * gradient

            mask = np.abs(z[client]) >= thresholds
            if np.any(mask):
                rows, cols = np.where(mask)
                signs = np.sign(z[client, rows, cols])
                tau = local_age[client, rows, cols].astype(float)
                retention = rho ** periods[client]
                magnitude_hat = (
                    thresholds[rows, cols]
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
                current_local = np.einsum(
                    "rij,rj->ri",
                    H[:, client],
                    w - theta[:, client],
                )[rows, cols]
                errors[int(periods[client])].extend(
                    np.abs(gradient_hat - current_local).tolist()
                )

                jump = jump0 * (
                    1.0 + tick / schedule_scale
                ) ** (-schedule_exponent)
                w += jump * np.sign(z[client]) * mask
                z[client][mask] = 0.0
                local_age[client][mask] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

    return {
        period: float(np.quantile(values, quantile))
        for period, values in errors.items()
        if values
    }


def run_calibrated_certificate_batch(
    *,
    ensemble: RotatedQuadraticEnsemble,
    config: CalibratedCertificateConfig,
    n_ticks: int,
    tail: int,
    seed: int,
    event_error_margins: dict[int, float] | None = None,
) -> dict[str, float]:
    """Run native-coordinate Experiment 11B.

    Calibration messages contain one full-precision local-gradient vector from
    every client. Between calibrations, the server keeps a stale aggregate
    gradient and a conservative drift envelope. The primary componentwise bound
    is

        |Delta g_i,j| <= sum_k |H_i,jk| |Delta w_k|.

    It is exact/tight on the diagonal quadratic and conservative after rotation.
    Static curvature bounds are treated as preconfigured benchmark metadata and
    are not counted as recurring communication.
    """

    H = ensemble.H
    theta = ensemble.theta
    Hbar = ensemble.Hbar
    wstar = ensemble.wstar
    U = ensemble.U
    periods = ensemble.periods
    weights = ensemble.weights
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    n_clients = ensemble.n_clients
    tail = min(tail, n_ticks)

    thresholds = config.delta0 * ensemble.diag_scale
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    cal_payload = calibration_payload_bits(ensemble)
    cal_packetized = calibration_packetized_bits(
        ensemble, config.header_bits
    )

    row_norm = np.linalg.norm(H, axis=3)
    abs_H = np.abs(H)
    aggregate_diag = np.diagonal(Hbar, axis1=1, axis2=2)

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    local_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    event_payload_bits = np.zeros(n_runs, dtype=np.int64)
    calibration_bits = np.zeros(n_runs, dtype=np.int64)
    calibration_count = np.zeros(n_runs, dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    accepted_events = np.zeros(n_runs, dtype=np.int64)
    harmful_events = np.zeros(n_runs, dtype=np.int64)
    client_candidates = np.zeros((n_runs, n_clients), dtype=np.int64)
    client_accepted = np.zeros((n_runs, n_clients), dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)

    calibrated_local_gradient = np.zeros((n_runs, n_clients, dimension))
    calibration_model = w.copy()
    last_calibration = np.zeros(n_runs, dtype=np.int64)

    def calibrate(run_mask: np.ndarray, tick: int) -> None:
        run_ids = np.flatnonzero(run_mask)
        if len(run_ids) == 0:
            return
        error = w[run_ids, None, :] - theta[run_ids]
        calibrated_local_gradient[run_ids] = np.einsum(
            "rcjk,rck->rcj", H[run_ids], error
        )
        calibration_model[run_ids] = w[run_ids]
        last_calibration[run_ids] = tick
        payload_bits[run_ids] += cal_payload
        packetized_bits[run_ids] += cal_packetized
        calibration_bits[run_ids] += cal_payload
        calibration_count[run_ids] += 1

    def global_radius() -> np.ndarray:
        delta = w - calibration_model
        if config.bound_mode == "componentwise":
            client_radius = np.einsum(
                "rcjk,rk->rcj", abs_H, np.abs(delta)
            )
            return np.sum(
                weights[None, :, None] * client_radius, axis=1
            )
        if config.bound_mode == "row_l2":
            norm_delta = np.linalg.norm(delta, axis=1)
            return (
                np.sum(weights[None, :, None] * row_norm, axis=1)
                * norm_delta[:, None]
            )
        raise ValueError(f"unknown bound mode {config.bound_mode}")

    calibrated_methods = {
        "periodic_naive",
        "periodic_cert",
        "periodic_event_cert",
        "adaptive_cert",
    }
    if config.method in calibrated_methods:
        calibrate(np.ones(n_runs, dtype=bool), 0)

    for tick in range(1, n_ticks + 1):
        if (
            config.method in {
                "periodic_naive",
                "periodic_cert",
                "periodic_event_cert",
            }
            and config.calibration_period > 0
            and tick % config.calibration_period == 0
        ):
            calibrate(np.ones(n_runs, dtype=bool), tick)

        z *= config.rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            deterministic = np.einsum(
                "rij,rj->ri",
                H[:, client],
                snapshots[client] - theta[:, client],
            )
            noise_latent = rng.normal(
                0.0, config.noise_std, size=(n_runs, dimension)
            )
            noise_native = np.einsum("rnl,rl->rn", U, noise_latent)
            gradient = deterministic + noise_native
            rate_weight = weights[client] * periods[client]
            gamma_i = config.gamma * rate_weight
            z[client] -= gamma_i * gradient

            mask = np.abs(z[client]) >= thresholds
            if np.any(mask):
                rows, cols = np.where(mask)
                signs = np.sign(z[client, rows, cols])
                n_events = mask.sum(axis=1)
                candidate_events += n_events
                client_candidates[:, client] += n_events
                event_cost = n_events * event_bits
                event_payload_bits += event_cost
                payload_bits += event_cost
                has_packet = n_events > 0
                packetized_bits += (
                    event_cost
                    + has_packet.astype(np.int64) * config.header_bits
                )
                curvature = aggregate_diag[rows, cols]

                if config.method == "schedule":
                    jump = config.jump0 * (
                        1.0 + tick / config.schedule_scale
                    ) ** (-config.schedule_exponent)
                    jumps = np.full(len(rows), jump)
                    accepted = np.ones(len(rows), dtype=bool)

                elif config.method == "global_oracle":
                    global_gradient = np.einsum(
                        "rij,rj->ri", Hbar, w - wstar
                    )[rows, cols]
                    alignment = -signs * global_gradient
                    jumps = np.maximum(alignment, 0.0) / curvature
                    accepted = jumps > 1e-14

                else:
                    center_all = np.sum(
                        weights[None, :, None] * calibrated_local_gradient,
                        axis=1,
                    )
                    radius_all = global_radius()
                    center = center_all[rows, cols].copy()

                    if config.method == "periodic_naive":
                        radius = np.zeros(len(rows))
                    elif config.method in {"periodic_cert", "adaptive_cert"}:
                        radius = radius_all[rows, cols].copy()
                    elif config.method == "periodic_event_cert":
                        tau = local_age[client, rows, cols].astype(float)
                        retention = config.rho ** periods[client]
                        magnitude_hat = (
                            thresholds[rows, cols]
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
                        center += weights[client] * (
                            gradient_hat
                            - calibrated_local_gradient[rows, client, cols]
                        )
                        delta = w - calibration_model
                        if config.bound_mode == "componentwise":
                            client_radius = np.einsum(
                                "rjk,rk->rj",
                                abs_H[:, client],
                                np.abs(delta),
                            )[rows, cols]
                        else:
                            client_radius = (
                                row_norm[rows, client, cols]
                                * np.linalg.norm(delta, axis=1)[rows]
                            )
                        margin = (event_error_margins or {}).get(
                            int(periods[client]), np.inf
                        )
                        radius = (
                            radius_all[rows, cols]
                            - weights[client] * client_radius
                            + weights[client] * margin
                        )
                    else:
                        raise ValueError(f"unknown method {config.method}")

                    lower_alignment = -signs * center - radius
                    jumps = np.maximum(lower_alignment, 0.0) / curvature
                    accepted = jumps > 1e-14

                    if config.method == "adaptive_cert":
                        needs_refresh = np.zeros(n_runs, dtype=bool)
                        for idx, run in enumerate(rows):
                            if (
                                not accepted[idx]
                                and tick - last_calibration[run]
                                >= config.min_refresh_gap
                            ):
                                needs_refresh[run] = True
                        if np.any(needs_refresh):
                            calibrate(needs_refresh, tick)
                            center_all = np.sum(
                                weights[None, :, None]
                                * calibrated_local_gradient,
                                axis=1,
                            )
                            radius_all = global_radius()
                            center = center_all[rows, cols]
                            radius = radius_all[rows, cols]
                            lower_alignment = -signs * center - radius
                            jumps = (
                                np.maximum(lower_alignment, 0.0) / curvature
                            )
                            accepted = jumps > 1e-14

                delta_w = jumps * signs
                global_gradient = np.einsum(
                    "rij,rj->ri", Hbar, w - wstar
                )[rows, cols]
                objective_change = (
                    global_gradient * delta_w
                    + 0.5 * curvature * delta_w**2
                )
                np.add.at(w, (rows, cols), delta_w)
                np.add.at(
                    accepted_events, rows, accepted.astype(np.int64)
                )
                np.add.at(
                    harmful_events,
                    rows,
                    (accepted & (objective_change > 1e-12)).astype(np.int64),
                )
                for idx, run in enumerate(rows):
                    if accepted[idx]:
                        client_accepted[run, client] += 1

                z[client][mask] = 0.0
                local_age[client][mask] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        current_gap = rotated_excess_objective(w, ensemble)
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
        "event_bits": float(np.mean(event_payload_bits)),
        "calibration_bits": float(np.mean(calibration_bits)),
        "calibrations": float(np.mean(calibration_count)),
        "candidate_events": float(np.mean(candidate_events)),
        "accepted_events": float(np.mean(accepted_events)),
        "acceptance_fraction": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_applied_fraction": (
            harmful_events.sum() / total_accepted
            if total_accepted
            else np.nan
        ),
        "slow_candidate_share": (
            client_candidates[:, slow_start:].sum()
            / client_candidates.sum()
            if client_candidates.sum()
            else np.nan
        ),
        "slow_accepted_share": (
            client_accepted[:, slow_start:].sum()
            / client_accepted.sum()
            if client_accepted.sum()
            else np.nan
        ),
    }


def run_basis_aligned_certificate_batch(
    *,
    ensemble: RotatedQuadraticEnsemble,
    calibration_period: int,
    n_ticks: int,
    tail: int,
    seed: int,
    rho: float = 0.999,
    gamma: float = 0.05,
    delta0: float = 0.25,
    noise_std: float = 0.25,
    header_bits: int = 32,
) -> dict[str, float]:
    """Oracle representation control for Experiment 11B.

    The common eigenbasis U is assumed known. Events and the conservative drift
    certificate are both evaluated in that basis. Calibration payload remains a
    full local-gradient vector from every client, so the comparison isolates the
    geometry of the event/certificate representation rather than hiding bits.
    """

    H = ensemble.H
    theta = ensemble.theta
    U = ensemble.U
    periods = ensemble.periods
    weights = ensemble.weights
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    n_clients = ensemble.n_clients
    tail = min(tail, n_ticks)

    local_diag = np.empty((n_runs, n_clients, dimension))
    for run in range(n_runs):
        for client in range(n_clients):
            local_diag[run, client] = np.diag(
                U[run].T @ H[run, client] @ U[run]
            )
    aggregate_diag = np.sum(
        weights[None, :, None] * local_diag, axis=1
    )
    base_scale = ensemble.inherited_scale
    thresholds = np.broadcast_to(
        delta0 * base_scale, (n_runs, dimension)
    )

    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    cal_payload = calibration_payload_bits(ensemble)
    cal_packetized = calibration_packetized_bits(ensemble, header_bits)

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    payload = np.zeros(n_runs, dtype=np.int64)
    packetized = np.zeros(n_runs, dtype=np.int64)
    calibration_bits = np.zeros(n_runs, dtype=np.int64)
    calibration_count = np.zeros(n_runs, dtype=np.int64)
    candidates = np.zeros(n_runs, dtype=np.int64)
    accepted = np.zeros(n_runs, dtype=np.int64)
    harmful = np.zeros(n_runs, dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)

    calibrated_gradient = np.zeros((n_runs, n_clients, dimension))
    calibration_model_latent = np.einsum("rnl,rn->rl", U, w)

    def calibrate() -> None:
        nonlocal calibrated_gradient, calibration_model_latent
        local_native = _local_gradients(w, ensemble)
        calibrated_gradient = np.einsum(
            "rnl,rcn->rcl", U, local_native
        )
        calibration_model_latent = np.einsum("rnl,rn->rl", U, w)
        payload[:] += cal_payload
        packetized[:] += cal_packetized
        calibration_bits[:] += cal_payload
        calibration_count[:] += 1

    calibrate()
    for tick in range(1, n_ticks + 1):
        if calibration_period > 0 and tick % calibration_period == 0:
            calibrate()
        z *= rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            deterministic = np.einsum(
                "rij,rj->ri",
                H[:, client],
                snapshots[client] - theta[:, client],
            )
            noise_latent = rng.normal(
                0.0, noise_std, size=(n_runs, dimension)
            )
            noise_native = np.einsum("rnl,rl->rn", U, noise_latent)
            gradient_native = deterministic + noise_native
            gradient_latent = np.einsum(
                "rnl,rn->rl", U, gradient_native
            )
            rate_weight = weights[client] * periods[client]
            z[client] -= gamma * rate_weight * gradient_latent
            mask = np.abs(z[client]) >= thresholds

            if np.any(mask):
                rows, cols = np.where(mask)
                signs = np.sign(z[client, rows, cols])
                n_events = mask.sum(axis=1)
                candidates += n_events
                cost = n_events * event_bits
                payload += cost
                has_packet = n_events > 0
                packetized += cost + has_packet.astype(np.int64) * header_bits

                center = np.sum(
                    weights[None, :, None] * calibrated_gradient,
                    axis=1,
                )
                current_latent = np.einsum("rnl,rn->rl", U, w)
                delta = np.abs(current_latent - calibration_model_latent)
                radius = aggregate_diag * delta
                lower_alignment = (
                    -signs * center[rows, cols] - radius[rows, cols]
                )
                jumps = np.maximum(lower_alignment, 0.0) / aggregate_diag[
                    rows, cols
                ]
                event_accepted = jumps > 1e-14

                delta_latent = np.zeros((n_runs, dimension))
                np.add.at(delta_latent, (rows, cols), jumps * signs)
                gap_before = rotated_excess_objective(w, ensemble)
                w += np.einsum("rnl,rl->rn", U, delta_latent)
                gap_after = rotated_excess_objective(w, ensemble)
                np.add.at(
                    accepted, rows, event_accepted.astype(np.int64)
                )
                for run in np.unique(rows):
                    if gap_after[run] > gap_before[run] + 1e-12:
                        harmful[run] += int(
                            np.sum(event_accepted[rows == run])
                        )
                z[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        current_gap = rotated_excess_objective(w, ensemble)
        whole_gap += current_gap
        if tick > n_ticks - tail:
            tail_gap += current_gap

    total_candidates = int(candidates.sum())
    total_accepted = int(accepted.sum())
    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload)),
        "packetized_bits": float(np.mean(packetized)),
        "calibration_bits": float(np.mean(calibration_bits)),
        "calibrations": float(np.mean(calibration_count)),
        "candidate_events": float(np.mean(candidates)),
        "accepted_events": float(np.mean(accepted)),
        "acceptance_fraction": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_applied_fraction": (
            harmful.sum() / total_accepted if total_accepted else np.nan
        ),
    }
