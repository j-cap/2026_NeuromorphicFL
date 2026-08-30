from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .vector_quadratic import VectorQuadraticEnsemble, excess_objective


TimingMethod = Literal[
    "fixed",
    "local_timing",
    "wall_timing",
    "shuffled_local",
    "relative_local",
    "global_schedule",
    "coordinate_jump",
]


@dataclass(frozen=True)
class VectorTimingConfig:
    method: TimingMethod = "fixed"
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    jump: float = 0.02
    tau_c: float = 50.0
    timing_exponent: float = 0.25
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.25
    relative_tau_c: float = 1.0
    noise_std: float = 0.25
    header_bits: int = 32


def run_vector_timing_batch(
    *,
    ensemble: VectorQuadraticEnsemble,
    config: VectorTimingConfig,
    n_ticks: int,
    tail: int,
    seed: int,
    shuffled_pools: dict[tuple[int, int], np.ndarray] | None = None,
    relative_medians: np.ndarray | None = None,
    record_events: bool = False,
) -> dict[str, object]:
    """Run coordinate LIF with fixed or event-time-aware jump amplitudes.

    Thresholding always uses the curvature-normalized full-reset LIF reference
    from Experiment 10A. Timing changes only the parameter jump applied after an
    event; the event itself carries no additional payload. ``tau_local`` counts
    completed gradients of the corresponding client since the previous event in
    that coordinate. ``tau_wall`` counts wall-clock ticks.
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

    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    thresholds = config.delta0 * ensemble.threshold_scale
    coordinate_jump_scale = 1.0 / np.sqrt(ensemble.threshold_scale)

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    local_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)
    wall_age = np.zeros((n_clients, n_runs, dimension), dtype=np.int64)

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    packets = np.zeros(n_runs, dtype=np.int64)
    events = np.zeros(n_runs, dtype=np.int64)
    coordinate_events = np.zeros((n_runs, dimension), dtype=np.int64)
    client_events = np.zeros((n_runs, n_clients), dtype=np.int64)
    harmful_events = np.zeros(n_runs, dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)
    event_rows: list[dict[str, object]] = []

    for tick in range(1, n_ticks + 1):
        z *= config.rho
        wall_age += 1

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_age[client] += 1
            gradient = h[:, client, :] * (
                snapshots[client] - theta[:, client, :]
            ) + rng.normal(0.0, config.noise_std, size=(n_runs, dimension))
            rate_weight = weights[client] * periods[client]
            z[client] -= config.gamma * rate_weight * gradient

            mask = np.abs(z[client]) >= thresholds[None, :]
            if np.any(mask):
                signs = np.sign(z[client])

                if config.method == "fixed":
                    jump_scale: np.ndarray | float = 1.0
                elif config.method == "local_timing":
                    tau = np.maximum(local_age[client], 1)
                    jump_scale = np.minimum(
                        1.0, (config.tau_c / tau) ** config.timing_exponent
                    )
                elif config.method == "wall_timing":
                    tau = np.maximum(wall_age[client], 1)
                    jump_scale = np.minimum(
                        1.0, (config.tau_c / tau) ** config.timing_exponent
                    )
                elif config.method == "shuffled_local":
                    if shuffled_pools is None:
                        raise ValueError("shuffled_pools required for shuffled_local")
                    jump_scale = np.ones((n_runs, dimension))
                    rows, cols = np.where(mask)
                    for run, coord in zip(rows, cols):
                        pool = shuffled_pools.get((client, int(coord)))
                        if pool is None or len(pool) == 0:
                            sampled = int(local_age[client, run, coord])
                        else:
                            sampled = int(rng.choice(pool))
                        jump_scale[run, coord] = min(
                            1.0,
                            (config.tau_c / max(sampled, 1)) ** config.timing_exponent,
                        )
                elif config.method == "relative_local":
                    if relative_medians is None:
                        raise ValueError("relative_medians required for relative_local")
                    ratio = np.maximum(
                        local_age[client] / relative_medians[client][None, :], 1e-9
                    )
                    jump_scale = np.minimum(
                        1.0,
                        (config.relative_tau_c / ratio) ** config.timing_exponent,
                    )
                elif config.method == "global_schedule":
                    jump_scale = (1.0 + tick / config.schedule_scale) ** (
                        -config.schedule_exponent
                    )
                elif config.method == "coordinate_jump":
                    jump_scale = np.broadcast_to(
                        coordinate_jump_scale, (n_runs, dimension)
                    )
                else:
                    raise ValueError(f"unknown method {config.method}")

                w_before = w.copy()
                update = config.jump * signs * mask * jump_scale

                # For diagonal quadratics an individual coordinate event has the
                # exact aggregate objective change hbar_j*(e_j*dq + 0.5*dq^2).
                rows, cols = np.where(mask)
                if np.isscalar(jump_scale):
                    event_scales = np.full(len(rows), float(jump_scale))
                else:
                    event_scales = np.asarray(jump_scale)[rows, cols]
                dq = config.jump * event_scales * signs[rows, cols]
                error = w_before[rows, cols] - wstar[rows, cols]
                delta_f = hbar[rows, cols] * (error * dq + 0.5 * dq**2)
                for idx, (run, coord) in enumerate(zip(rows, cols)):
                    if delta_f[idx] > 1e-15:
                        harmful_events[run] += 1
                    if record_events:
                        event_rows.append(
                            {
                                "client": client,
                                "coordinate": int(coord),
                                "tau_local": int(local_age[client, run, coord]),
                                "tau_wall": int(wall_age[client, run, coord]),
                                "abs_error": float(abs(error[idx])),
                                "curvature": float(hbar[run, coord]),
                                "harmful": bool(delta_f[idx] > 1e-15),
                            }
                        )

                w += update
                n_events = mask.sum(axis=1)
                events += n_events
                coordinate_events += mask.astype(np.int64)
                client_events[:, client] += n_events
                payload_bits += n_events * event_bits
                has_packet = n_events > 0
                packets += has_packet.astype(np.int64)
                packetized_bits += (
                    n_events * event_bits
                    + has_packet.astype(np.int64) * config.header_bits
                )
                z[client][mask] = 0.0
                local_age[client][mask] = 0
                wall_age[client][mask] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        gap = excess_objective(w, wstar, hbar)
        whole_gap += gap
        if tick > n_ticks - tail:
            tail_gap += gap
            tail_dist2 += np.sum((w - wstar) ** 2, axis=1)

    total_events = int(events.sum())
    slow_start = max(0, n_clients - 2)
    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(np.sqrt(np.mean(tail_dist2 / (tail * dimension)))),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "packets": float(np.mean(packets)),
        "events": float(np.mean(events)),
        "coverage": float(np.mean((coordinate_events > 0).mean(axis=1))),
        "harmful_event_fraction": float(harmful_events.sum() / total_events)
        if total_events
        else np.nan,
        "slow_client_event_share": float(
            client_events[:, slow_start:].sum() / client_events.sum()
        )
        if client_events.sum()
        else np.nan,
        "coordinate_events_mean": coordinate_events.mean(axis=0),
        "event_log": pd.DataFrame(event_rows) if record_events else None,
    }


def timing_calibration(
    event_log: pd.DataFrame, n_clients: int, dimension: int
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Build per-client/coordinate timing pools and medians from a calibration run."""

    pools: dict[tuple[int, int], np.ndarray] = {}
    medians = np.ones((n_clients, dimension), dtype=float)
    for (client, coord), subset in event_log.groupby(["client", "coordinate"]):
        values = subset["tau_local"].to_numpy(dtype=int)
        pools[(int(client), int(coord))] = values
        if len(values):
            medians[int(client), int(coord)] = float(np.median(values))
    return pools, medians
