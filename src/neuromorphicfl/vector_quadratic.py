from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Method = Literal["full", "sign", "lif_global", "lif_norm", "ef_topk"]


@dataclass(frozen=True)
class VectorQuadraticEnsemble:
    h: np.ndarray
    theta: np.ndarray
    hbar: np.ndarray
    wstar: np.ndarray
    w0: np.ndarray
    threshold_scale: np.ndarray
    periods: np.ndarray
    weights: np.ndarray

    @property
    def n_runs(self) -> int:
        return int(self.w0.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.w0.shape[1])

    @property
    def n_clients(self) -> int:
        return int(self.theta.shape[1])


@dataclass(frozen=True)
class VectorRunConfig:
    method: Method
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    jump: float = 0.02
    step: float = 0.01
    topk: int = 4
    noise_std: float = 0.25
    header_bits: int = 32


def make_diagonal_quadratic_ensemble(
    *,
    n_runs: int,
    n_clients: int = 10,
    dimension: int = 20,
    periods: np.ndarray | None = None,
    theta_std: float = 0.25,
    curvature_client_std: float = 0.15,
    curvature_min: float = 0.2,
    curvature_max: float = 5.0,
    initial_offset: float = 0.8,
    seed: int = 222,
) -> VectorQuadraticEnsemble:
    """Create heterogeneous diagonal strongly-convex federated quadratics.

    Local objectives are

        F_i(w) = 0.5 * (w-theta_i)^T H_i (w-theta_i),

    with diagonal positive-definite H_i. Client weights are uniform. The exact
    aggregate optimum is retained for evaluation. The synthetic coordinate
    threshold scale is proportional to the shared base curvature and normalized
    to median one; this is an oracle scale diagnostic, not yet the practical
    normalization rule intended for machine-learning experiments.
    """

    if periods is None:
        if n_clients != 10:
            raise ValueError("provide periods explicitly when n_clients != 10")
        periods = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
    periods = np.asarray(periods, dtype=int)
    if len(periods) != n_clients:
        raise ValueError("period count must equal n_clients")

    weights = np.full(n_clients, 1.0 / n_clients)
    base_h = np.geomspace(curvature_min, curvature_max, dimension)
    rng = np.random.default_rng(seed)
    h = base_h[None, None, :] * np.exp(
        rng.normal(0.0, curvature_client_std, size=(n_runs, n_clients, dimension))
    )
    theta = rng.normal(0.0, theta_std, size=(n_runs, n_clients, dimension))
    hbar = np.sum(weights[None, :, None] * h, axis=1)
    wstar = np.sum(weights[None, :, None] * h * theta, axis=1) / hbar
    w0 = wstar + initial_offset
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


def excess_objective(
    w: np.ndarray,
    wstar: np.ndarray,
    hbar: np.ndarray,
) -> np.ndarray:
    """Exact aggregate objective gap for diagonal quadratics."""

    return 0.5 * np.sum(hbar * (w - wstar) ** 2, axis=1)


def run_vector_batch(
    *,
    ensemble: VectorQuadraticEnsemble,
    config: VectorRunConfig,
    n_ticks: int,
    tail: int,
    seed: int,
) -> dict[str, object]:
    """Run one asynchronous vector-optimization configuration.

    Clients compute on snapshots and return after their configured periods. All
    methods use compute-rate normalization through p_i*T_i so hardware speed
    alone does not redefine the intended aggregate objective.

    Communication accounting is logical payload bits plus a second illustrative
    packetized count with ``header_bits`` added once per nonempty client packet.
    No downlink traffic is counted in Experiment 10A.
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

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    residual = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    packets = np.zeros(n_runs, dtype=np.int64)
    events = np.zeros(n_runs, dtype=np.int64)
    coordinate_events = np.zeros((n_runs, dimension), dtype=np.int64)
    client_events = np.zeros((n_runs, n_clients), dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)

    if config.method == "lif_norm":
        thresholds = config.delta0 * ensemble.threshold_scale
    else:
        thresholds = np.full(dimension, config.delta0)

    for tick in range(1, n_ticks + 1):
        if config.method in ("lif_global", "lif_norm"):
            z *= config.rho

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            gradient = h[:, client, :] * (
                snapshots[client] - theta[:, client, :]
            ) + rng.normal(0.0, config.noise_std, size=(n_runs, dimension))
            rate_weight = weights[client] * periods[client]

            if config.method == "full":
                w -= config.step * rate_weight * gradient
                payload_bits += 32 * dimension
                packetized_bits += 32 * dimension + config.header_bits
                packets += 1

            elif config.method == "sign":
                w -= config.jump * rate_weight * np.sign(gradient)
                payload_bits += dimension
                packetized_bits += dimension + config.header_bits
                packets += 1

            elif config.method in ("lif_global", "lif_norm"):
                z[client] -= config.gamma * rate_weight * gradient
                mask = np.abs(z[client]) >= thresholds[None, :]
                if np.any(mask):
                    signs = np.sign(z[client])
                    w += config.jump * signs * mask
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

            elif config.method == "ef_topk":
                residual[client] -= config.step * rate_weight * gradient
                k = min(config.topk, dimension)
                idx = np.argpartition(np.abs(residual[client]), -k, axis=1)[:, -k:]
                vals = np.take_along_axis(residual[client], idx, axis=1)
                runs = np.arange(n_runs)
                for col in range(k):
                    w[runs, idx[:, col]] += vals[:, col]
                    residual[client, runs, idx[:, col]] = 0.0
                payload_bits += k * (32 + address_bits)
                packetized_bits += k * (32 + address_bits) + config.header_bits
                packets += 1

            else:
                raise ValueError(f"unknown method {config.method}")

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        gap = excess_objective(w, wstar, hbar)
        whole_gap += gap
        if tick > n_ticks - tail:
            tail_gap += gap
            tail_dist2 += np.sum((w - wstar) ** 2, axis=1)

    total_client_events = int(client_events.sum())
    slow_start = max(0, n_clients - 2)
    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(np.sqrt(np.mean(tail_dist2 / (tail * dimension)))),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "packets": float(np.mean(packets)),
        "events": float(np.mean(events)),
        "coverage": float(np.mean((coordinate_events > 0).mean(axis=1)))
        if config.method.startswith("lif")
        else 1.0,
        "slow_client_event_share": float(client_events[:, slow_start:].sum() / total_client_events)
        if total_client_events
        else np.nan,
        "coordinate_events_mean": coordinate_events.mean(axis=0),
    }


def pareto_mask(cost: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Non-dominated mask when both communication cost and error are minimized."""

    cost = np.asarray(cost)
    error = np.asarray(error)
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        dominated = (
            (cost <= cost[i])
            & (error <= error[i])
            & ((cost < cost[i]) | (error < error[i]))
        )
        if np.any(dominated):
            keep[i] = False
    return keep


def check_fullreset_vector_ema_equivalence(
    *,
    rho: float = 0.999,
    gamma: float = 0.05,
    delta0: float = 0.25,
    steps: int = 1000,
    seed: int = 77,
) -> dict[str, float]:
    """Verify coordinatewise full-reset LIF == reset EMA after rescaling."""

    dimension = 20
    base_h = np.geomspace(0.2, 5.0, dimension)
    scale = base_h / np.median(base_h)
    threshold = delta0 * scale
    ema_threshold = (1.0 - rho) * threshold / gamma
    rng = np.random.default_rng(seed)
    z = np.zeros(dimension)
    ema = np.zeros(dimension)
    mismatches = 0
    max_state_error = 0.0

    for _ in range(steps):
        gradient = rng.normal(0.0, 1.0, dimension)
        z = rho * z - gamma * gradient
        ema = rho * ema + (1.0 - rho) * gradient
        fire_z = np.abs(z) >= threshold
        fire_ema = np.abs(ema) >= ema_threshold
        if not np.array_equal(fire_z, fire_ema):
            mismatches += 1
        if np.any(fire_z):
            if not np.array_equal(np.sign(z[fire_z]), -np.sign(ema[fire_z])):
                mismatches += 1
            z[fire_z] = 0.0
            ema[fire_z] = 0.0
        max_state_error = max(
            max_state_error,
            float(np.max(np.abs(ema + (1.0 - rho) * z / gamma))),
        )

    return {
        "mismatching_steps": float(mismatches),
        "max_mapped_state_error": float(max_state_error),
    }
