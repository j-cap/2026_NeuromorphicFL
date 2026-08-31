from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .rotated_geometry import RotatedQuadraticEnsemble, rotated_excess_objective


CalibrationPolicy = Literal[
    "full",
    "uncertainty_abs",
    "uncertainty_relative",
    "random",
    "round_robin",
]


@dataclass(frozen=True)
class SelectiveCalibrationConfig:
    """Configuration for Experiment 11C selective certificate maintenance.

    Every run starts with one full trustworthy calibration so that all coordinate
    certificates are initialized. Subsequent calibration opportunities refresh only
    ``k`` coordinates, except for ``policy='full'``.

    Communication accounting follows Experiment 11B: requested calibration
    coordinates are assumed known from the server request, so uplink calibration
    payload is ``n_clients * k * 32`` bits. Sparse LIF events still carry only a
    coordinate address and sign. Downlink request traffic is not included, matching
    the existing Experiment-11B convention.
    """

    policy: CalibrationPolicy = "uncertainty_abs"
    k: int = 4
    calibration_period: int = 25
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    noise_std: float = 0.25
    header_bits: int = 32


def _local_gradients(
    w: np.ndarray,
    ensemble: RotatedQuadraticEnsemble,
) -> np.ndarray:
    error = w[:, None, :] - ensemble.theta
    return np.einsum("rcjk,rck->rcj", ensemble.H, error, optimize=True)


def _center_and_radius(
    *,
    w: np.ndarray,
    calibrated_local_gradient: np.ndarray,
    coordinate_reference_model: np.ndarray,
    ensemble: RotatedQuadraticEnsemble,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate stale gradient center and deterministic drift radius for all j."""

    center = np.sum(
        ensemble.weights[None, :, None] * calibrated_local_gradient,
        axis=1,
    )
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    radius = np.zeros((n_runs, dimension))
    abs_H = np.abs(ensemble.H)
    for coord in range(dimension):
        displacement = np.abs(w - coordinate_reference_model[:, coord, :])
        per_client = np.einsum(
            "rck,rk->rc",
            abs_H[:, :, coord, :],
            displacement,
            optimize=True,
        )
        radius[:, coord] = np.sum(
            ensemble.weights[None, :] * per_client,
            axis=1,
        )
    return center, radius


def run_selective_calibration_batch(
    *,
    ensemble: RotatedQuadraticEnsemble,
    config: SelectiveCalibrationConfig,
    n_ticks: int,
    tail: int,
    seed: int,
) -> dict[str, float]:
    """Run Experiment 11C selective maintenance on native coordinate certificates.

    The descent certificate is the same deterministic componentwise certificate as
    Experiment 11B. Only certificate maintenance changes. At each calibration
    opportunity, the selected coordinates are refreshed for every client at the
    current server model. Each coordinate keeps its own reference model because, in
    coupled geometry, different gradient components may have been calibrated at
    different times.

    Certified coordinate events from a packet are applied sequentially, exactly as
    in Experiment 11B, so Hessian cross terms cannot invalidate independently derived
    one-coordinate descent guarantees.
    """

    H = ensemble.H
    theta = ensemble.theta
    Hbar = ensemble.Hbar
    U = ensemble.U
    periods = ensemble.periods
    weights = ensemble.weights
    n_runs = ensemble.n_runs
    dimension = ensemble.dimension
    n_clients = ensemble.n_clients
    tail = min(tail, n_ticks)

    thresholds = config.delta0 * ensemble.diag_scale
    aggregate_diag = np.diagonal(Hbar, axis1=1, axis2=2)
    address_bits = int(np.ceil(np.log2(dimension)))
    event_payload_per_coordinate = address_bits + 1

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    calibration_bits = np.zeros(n_runs, dtype=np.int64)
    event_bits = np.zeros(n_runs, dtype=np.int64)
    calibration_count = np.zeros(n_runs, dtype=np.int64)
    calibrated_coordinate_count = np.zeros(n_runs, dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    accepted_events = np.zeros(n_runs, dtype=np.int64)
    harmful_events = np.zeros(n_runs, dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)

    # Each coordinate has an independently dated calibration reference.
    calibrated_local_gradient = _local_gradients(w, ensemble)
    coordinate_reference_model = np.repeat(
        w[:, None, :], dimension, axis=1
    )
    maintenance_selections = np.zeros((n_runs, dimension), dtype=np.int64)

    # Common initial full calibration. This isolates maintenance efficiency rather
    # than conflating 11C with the separate problem of certificate initialization.
    initial_payload = n_clients * dimension * 32
    payload_bits += initial_payload
    calibration_bits += initial_payload
    packetized_bits += n_clients * (dimension * 32 + config.header_bits)
    calibration_count += 1
    calibrated_coordinate_count += dimension

    round_robin_cursor = 0

    for tick in range(1, n_ticks + 1):
        if (
            config.calibration_period > 0
            and tick % config.calibration_period == 0
        ):
            center, radius = _center_and_radius(
                w=w,
                calibrated_local_gradient=calibrated_local_gradient,
                coordinate_reference_model=coordinate_reference_model,
                ensemble=ensemble,
            )

            if config.policy == "full":
                selected = np.tile(np.arange(dimension), (n_runs, 1))
            else:
                k = min(config.k, dimension)
                if config.policy == "uncertainty_abs":
                    score = radius
                    selected = np.argpartition(score, -k, axis=1)[:, -k:]
                elif config.policy == "uncertainty_relative":
                    # Deliberate control from the original 11C proposal. It favors
                    # sign ambiguity R/(|c|+R), which the experiment shows is not the
                    # right allocation criterion on the rotated benchmark.
                    score = radius / (np.abs(center) + radius + 1e-12)
                    selected = np.argpartition(score, -k, axis=1)[:, -k:]
                elif config.policy == "random":
                    selected = np.stack(
                        [
                            rng.choice(dimension, size=k, replace=False)
                            for _ in range(n_runs)
                        ]
                    )
                elif config.policy == "round_robin":
                    coords = np.array(
                        [
                            (round_robin_cursor + offset) % dimension
                            for offset in range(k)
                        ]
                    )
                    selected = np.tile(coords, (n_runs, 1))
                    round_robin_cursor = (
                        round_robin_cursor + k
                    ) % dimension
                else:
                    raise ValueError(f"unknown calibration policy {config.policy}")

            current_local_gradient = _local_gradients(w, ensemble)
            k_eff = selected.shape[1]
            for run in range(n_runs):
                for coord in selected[run]:
                    calibrated_local_gradient[run, :, coord] = (
                        current_local_gradient[run, :, coord]
                    )
                    coordinate_reference_model[run, coord, :] = w[run]
                    maintenance_selections[run, coord] += 1

            calibration_payload = n_clients * k_eff * 32
            payload_bits += calibration_payload
            calibration_bits += calibration_payload
            packetized_bits += n_clients * (
                k_eff * 32 + config.header_bits
            )
            calibration_count += 1
            calibrated_coordinate_count += k_eff

        z *= config.rho
        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            error = snapshots[client] - theta[:, client, :]
            deterministic_gradient = np.einsum(
                "rij,rj->ri", H[:, client], error, optimize=True
            )
            noise_latent = rng.normal(
                0.0, config.noise_std, size=(n_runs, dimension)
            )
            noise_native = np.einsum(
                "rnl,rl->rn", U, noise_latent, optimize=True
            )
            gradient = deterministic_gradient + noise_native
            rate_weight = weights[client] * periods[client]
            z[client] -= config.gamma * rate_weight * gradient
            mask = np.abs(z[client]) >= thresholds

            if np.any(mask):
                n_events = mask.sum(axis=1)
                candidate_events += n_events
                sparse_payload = n_events * event_payload_per_coordinate
                event_bits += sparse_payload
                payload_bits += sparse_payload
                has_packet = n_events > 0
                packetized_bits += (
                    sparse_payload
                    + has_packet.astype(np.int64) * config.header_bits
                )

                # Sequential-safe server processing from Experiment 11B.
                for run in range(n_runs):
                    for coord in np.flatnonzero(mask[run]):
                        sign = float(np.sign(z[client, run, coord]))
                        center = float(
                            np.sum(
                                weights
                                * calibrated_local_gradient[run, :, coord]
                            )
                        )
                        displacement = np.abs(
                            w[run] - coordinate_reference_model[run, coord]
                        )
                        per_client_radius = np.einsum(
                            "ck,k->c",
                            np.abs(H[run, :, coord, :]),
                            displacement,
                        )
                        radius = float(
                            np.sum(weights * per_client_radius)
                        )
                        lower_alignment = -sign * center - radius

                        if lower_alignment > 0.0:
                            jump = lower_alignment / max(
                                float(aggregate_diag[run, coord]), 1e-12
                            )
                            before = rotated_excess_objective(
                                w[run : run + 1], ensemble
                            )[0]
                            w[run, coord] += sign * jump
                            after = rotated_excess_objective(
                                w[run : run + 1], ensemble
                            )[0]
                            accepted_events[run] += 1
                            if after > before + 1e-12:
                                harmful_events[run] += 1

                z[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        current_gap = rotated_excess_objective(w, ensemble)
        whole_gap += current_gap
        if tick > n_ticks - tail:
            tail_gap += current_gap
            tail_dist2 += np.sum((w - ensemble.wstar) ** 2, axis=1)

    total_candidates = int(candidate_events.sum())
    total_accepted = int(accepted_events.sum())
    maintenance_average = maintenance_selections.mean(axis=0)
    maintenance_mean = float(maintenance_average.mean())

    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(
            np.sqrt(np.mean(tail_dist2 / (tail * dimension)))
        ),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "calibration_bits": float(np.mean(calibration_bits)),
        "event_bits": float(np.mean(event_bits)),
        "calibration_fraction": float(
            calibration_bits.sum() / payload_bits.sum()
        ),
        "calibrations": float(np.mean(calibration_count)),
        "calibrated_coordinates": float(
            np.mean(calibrated_coordinate_count)
        ),
        "candidate_events": float(np.mean(candidate_events)),
        "accepted_events": float(np.mean(accepted_events)),
        "acceptance_fraction": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_applied_fraction": (
            harmful_events.sum() / total_accepted if total_accepted else np.nan
        ),
        "maintenance_coordinate_coverage": float(
            np.mean(maintenance_selections > 0)
        ),
        "maintenance_selection_cv": (
            float(maintenance_average.std() / maintenance_mean)
            if maintenance_mean > 0.0
            else np.nan
        ),
    }
