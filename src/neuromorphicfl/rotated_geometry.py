from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


RotatedMethod = Literal[
    "lif_schedule",
    "lif_basis",
    "global_oracle",
    "ef_topk",
    "full",
]
ThresholdMode = Literal["diag", "uniform", "inherited", "basis"]


@dataclass(frozen=True)
class RotatedQuadraticEnsemble:
    H: np.ndarray
    theta: np.ndarray
    Hbar: np.ndarray
    wstar: np.ndarray
    w0: np.ndarray
    U: np.ndarray
    diag_scale: np.ndarray
    inherited_scale: np.ndarray
    offdiag_ratio: np.ndarray
    condition_number: np.ndarray
    initial_gap: np.ndarray
    periods: np.ndarray
    weights: np.ndarray
    strength: float

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
class RotatedRunConfig:
    method: RotatedMethod
    threshold_mode: ThresholdMode = "diag"
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    jump0: float = 0.02
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.25
    step: float = 0.01
    topk: int = 4
    noise_std: float = 0.25
    header_bits: int = 32


def _rotation_from_plan(
    dimension: int,
    pairs: list[tuple[int, int]],
    angles: np.ndarray,
    strength: float,
) -> np.ndarray:
    U = np.eye(dimension)
    for (a, b), angle in zip(pairs, angles):
        theta = strength * angle
        c, s = np.cos(theta), np.sin(theta)
        col_a = U[:, a].copy()
        col_b = U[:, b].copy()
        U[:, a] = c * col_a + s * col_b
        U[:, b] = -s * col_a + c * col_b
    return U


def make_isospectral_common_rotation_ensemble(
    *,
    n_runs: int,
    strength: float,
    n_clients: int = 10,
    dimension: int = 20,
    periods: np.ndarray | None = None,
    theta_std: float = 0.25,
    curvature_client_std: float = 0.15,
    curvature_min: float = 0.2,
    curvature_max: float = 5.0,
    initial_offset: float = 0.8,
    givens_per_dimension: int = 3,
    seed: int = 222,
) -> RotatedQuadraticEnsemble:
    """Exact isospectral coordinate transform of the Experiment-10A family.

    For each Monte-Carlo problem one orthogonal basis U is shared by all clients:

        H_i = U D_i U^T,  theta_i = U theta_i_tilde.

    The initial error is transformed by the same U. Therefore varying
    ``strength`` changes only the coordinate representation of an otherwise
    equivalent quadratic family. The eigenvalues, aggregate condition number,
    initial objective gap, and latent client heterogeneity are preserved.
    """

    if periods is None:
        if n_clients != 10:
            raise ValueError("provide periods explicitly when n_clients != 10")
        periods = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
    periods = np.asarray(periods, dtype=int)
    if len(periods) != n_clients:
        raise ValueError("period count must equal n_clients")

    weights = np.full(n_clients, 1.0 / n_clients)
    base_eigenvalues = np.geomspace(curvature_min, curvature_max, dimension)
    rng = np.random.default_rng(seed)
    eigenvalues = base_eigenvalues[None, None, :] * np.exp(
        rng.normal(
            0.0,
            curvature_client_std,
            size=(n_runs, n_clients, dimension),
        )
    )
    theta_latent = rng.normal(0.0, theta_std, size=(n_runs, n_clients, dimension))
    hbar_latent = np.sum(weights[None, :, None] * eigenvalues, axis=1)
    wstar_latent = (
        np.sum(weights[None, :, None] * eigenvalues * theta_latent, axis=1)
        / hbar_latent
    )

    n_givens = givens_per_dimension * dimension
    U = np.empty((n_runs, dimension, dimension))
    for run in range(n_runs):
        pairs = [
            tuple(int(x) for x in rng.choice(dimension, 2, replace=False))
            for _ in range(n_givens)
        ]
        angles = rng.uniform(-np.pi / 3.0, np.pi / 3.0, n_givens)
        U[run] = _rotation_from_plan(dimension, pairs, angles, strength)

    H = np.empty((n_runs, n_clients, dimension, dimension))
    theta = np.empty((n_runs, n_clients, dimension))
    for run in range(n_runs):
        theta[run] = theta_latent[run] @ U[run].T
        for client in range(n_clients):
            H[run, client] = (
                U[run]
                @ np.diag(eigenvalues[run, client])
                @ U[run].T
            )

    Hbar = np.einsum("rnl,rl,rml->rnm", U, hbar_latent, U)
    wstar = np.einsum("rnl,rl->rn", U, wstar_latent)
    initial_latent = wstar_latent + initial_offset
    w0 = np.einsum("rnl,rl->rn", U, initial_latent)

    diag_bar = np.diagonal(Hbar, axis1=1, axis2=2)
    diag_scale = diag_bar / np.median(diag_bar, axis=1, keepdims=True)
    inherited_scale = base_eigenvalues / np.median(base_eigenvalues)

    diagonal_part = np.zeros_like(Hbar)
    index = np.arange(dimension)
    diagonal_part[:, index, index] = diag_bar
    offdiag = Hbar - diagonal_part
    offdiag_ratio = np.linalg.norm(offdiag, axis=(1, 2)) / np.linalg.norm(
        Hbar, axis=(1, 2)
    )
    condition_number = np.linalg.cond(Hbar)
    initial_gap = 0.5 * np.sum(
        hbar_latent * (initial_latent - wstar_latent) ** 2,
        axis=1,
    )

    return RotatedQuadraticEnsemble(
        H=H,
        theta=theta,
        Hbar=Hbar,
        wstar=wstar,
        w0=w0,
        U=U,
        diag_scale=diag_scale,
        inherited_scale=inherited_scale,
        offdiag_ratio=offdiag_ratio,
        condition_number=condition_number,
        initial_gap=initial_gap,
        periods=periods,
        weights=weights,
        strength=float(strength),
    )


def rotated_excess_objective(
    w: np.ndarray,
    ensemble: RotatedQuadraticEnsemble,
) -> np.ndarray:
    error = w - ensemble.wstar
    return 0.5 * np.einsum("ri,rij,rj->r", error, ensemble.Hbar, error)


def run_rotated_batch(
    *,
    ensemble: RotatedQuadraticEnsemble,
    config: RotatedRunConfig,
    n_ticks: int,
    tail: int,
    seed: int,
) -> dict[str, float]:
    """Run the Experiment-10D isospectral geometry stress test.

    Gradient noise is sampled in the latent basis and rotated by the same U as
    the objective. Hence full-precision optimization and basis-aligned LIF are
    pathwise invariant to ``strength``. Any degradation of native sparse/event
    methods is attributable to coordinate-dependent communication rather than to
    a changed objective spectrum, initial gap, or stochastic forcing.
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

    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1

    if config.threshold_mode == "diag":
        thresholds = config.delta0 * ensemble.diag_scale
    elif config.threshold_mode == "uniform":
        thresholds = np.full((n_runs, dimension), config.delta0)
    elif config.threshold_mode in ("inherited", "basis"):
        thresholds = np.broadcast_to(
            config.delta0 * ensemble.inherited_scale,
            (n_runs, dimension),
        )
    else:
        raise ValueError(f"unknown threshold mode {config.threshold_mode}")

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
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    accepted_events = np.zeros(n_runs, dtype=np.int64)
    harmful_packets = np.zeros(n_runs, dtype=np.int64)
    coordinate_events = np.zeros((n_runs, dimension), dtype=np.int64)
    client_events = np.zeros((n_runs, n_clients), dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)

    for tick in range(1, n_ticks + 1):
        if config.method in ("lif_schedule", "lif_basis", "global_oracle"):
            z *= config.rho

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            error = snapshots[client] - theta[:, client, :]
            deterministic_gradient = np.einsum(
                "rij,rj->ri", H[:, client], error
            )
            noise_latent = rng.normal(
                0.0, config.noise_std, size=(n_runs, dimension)
            )
            noise_native = np.einsum("rnl,rl->rn", U, noise_latent)
            gradient = deterministic_gradient + noise_native
            rate_weight = weights[client] * periods[client]

            if config.method == "full":
                w -= config.step * rate_weight * gradient
                payload_bits += 32 * dimension
                packetized_bits += 32 * dimension + config.header_bits
                packets += 1

            elif config.method == "ef_topk":
                residual[client] -= config.step * rate_weight * gradient
                k = min(config.topk, dimension)
                idx = np.argpartition(
                    np.abs(residual[client]), -k, axis=1
                )[:, -k:]
                values = np.take_along_axis(residual[client], idx, axis=1)
                runs = np.arange(n_runs)
                for col in range(k):
                    w[runs, idx[:, col]] += values[:, col]
                    residual[client, runs, idx[:, col]] = 0.0
                payload_bits += k * (32 + address_bits)
                packetized_bits += (
                    k * (32 + address_bits) + config.header_bits
                )
                packets += 1

            elif config.method in ("lif_schedule", "lif_basis", "global_oracle"):
                if config.method == "lif_basis":
                    encoded_gradient = np.einsum(
                        "rnl,rn->rl", U, gradient
                    )
                else:
                    encoded_gradient = gradient
                z[client] -= config.gamma * rate_weight * encoded_gradient
                mask = np.abs(z[client]) >= thresholds

                if np.any(mask):
                    n_events = mask.sum(axis=1)
                    candidate_events += n_events
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

                    if config.method == "lif_schedule":
                        gap_before = rotated_excess_objective(w, ensemble)
                        jump = config.jump0 * (
                            1.0 + tick / config.schedule_scale
                        ) ** (-config.schedule_exponent)
                        w += jump * np.sign(z[client]) * mask
                        accepted_events += n_events
                        gap_after = rotated_excess_objective(w, ensemble)
                        harmful_packets += (
                            has_packet & (gap_after > gap_before + 1e-15)
                        ).astype(np.int64)

                    elif config.method == "lif_basis":
                        jump = config.jump0 * (
                            1.0 + tick / config.schedule_scale
                        ) ** (-config.schedule_exponent)
                        delta_latent = jump * np.sign(z[client]) * mask
                        w += np.einsum("rnl,rl->rn", U, delta_latent)
                        accepted_events += n_events

                    else:
                        # Sequential exact coordinate minimization. The trigger is
                        # unchanged; rejected candidates still count as communication.
                        for run in range(n_runs):
                            for coord in np.flatnonzero(mask[run]):
                                sign = np.sign(z[client, run, coord])
                                global_gradient = Hbar[run] @ (
                                    w[run] - wstar[run]
                                )
                                alignment = -sign * global_gradient[coord]
                                if alignment > 0.0:
                                    jump = alignment / Hbar[run, coord, coord]
                                    w[run, coord] += sign * jump
                                    accepted_events[run] += 1

                    z[client][mask] = 0.0

            else:
                raise ValueError(f"unknown method {config.method}")

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        gap = rotated_excess_objective(w, ensemble)
        whole_gap += gap
        if tick > n_ticks - tail:
            tail_gap += gap
            tail_dist2 += np.sum((w - wstar) ** 2, axis=1)

    total_candidates = int(candidate_events.sum())
    total_client_events = int(client_events.sum())
    total_packets = int(packets.sum())
    slow_start = max(0, n_clients - 2)

    return {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(
            np.sqrt(np.mean(tail_dist2 / (tail * dimension)))
        ),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "packets": float(np.mean(packets)),
        "events": float(np.mean(events)),
        "candidate_events": float(np.mean(candidate_events)),
        "accepted_events": float(np.mean(accepted_events)),
        "acceptance_fraction": (
            float(accepted_events.sum() / total_candidates)
            if total_candidates
            else np.nan
        ),
        "harmful_packet_fraction": (
            float(harmful_packets.sum() / total_packets)
            if total_packets
            else np.nan
        ),
        "coverage": (
            float(np.mean((coordinate_events > 0).mean(axis=1)))
            if config.method in ("lif_schedule", "lif_basis", "global_oracle")
            else 1.0
        ),
        "slow_client_event_share": (
            float(client_events[:, slow_start:].sum() / total_client_events)
            if total_client_events
            else np.nan
        ),
    }
