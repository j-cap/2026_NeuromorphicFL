from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .vector_quadratic import VectorQuadraticEnsemble, excess_objective


CompetitiveMethod = Literal[
    "lif_independent",
    "lif_competitive",
    "sign_topk",
    "ef_topk",
]


@dataclass(frozen=True)
class CompetitiveRunConfig:
    method: CompetitiveMethod
    rho: float = 0.999
    gamma: float = 0.05
    delta0: float = 0.25
    jump: float = 0.02
    topk: int = 4
    step: float = 0.01
    noise_std: float = 0.25
    header_bits: int = 32


def _topk_eligible(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    """Select up to k eligible coordinates per run by descending score."""

    n_runs, dimension = scores.shape
    k = min(max(int(k), 1), dimension)
    masked_scores = np.where(eligible, scores, -np.inf)
    idx = np.argpartition(masked_scores, -k, axis=1)[:, -k:]
    runs = np.arange(n_runs)[:, None]
    selected = np.zeros_like(eligible)
    selected[runs, idx] = np.isfinite(masked_scores[runs, idx])
    return selected


def run_competitive_batch(
    *,
    ensemble: VectorQuadraticEnsemble,
    config: CompetitiveRunConfig,
    n_ticks: int,
    tail: int,
    seed: int,
    record_packet_sizes: bool = False,
) -> dict[str, object]:
    """Run competitive coordinate selection for Experiment 10B.

    The two LIF variants use the curvature-normalized thresholds introduced in
    Experiment 10A. ``lif_independent`` emits every coordinate that crosses its
    threshold. ``lif_competitive`` allows at most ``topk`` eligible coordinates
    to fire per client-completion packet, ranked by normalized membrane
    excursion ``|z_j| / Delta_j``. Eligible but unselected coordinates are not
    reset: their evidence is retained for later competitions.

    ``sign_topk`` is an instantaneous sign-only sparse baseline. ``ef_topk`` is
    the conventional float-valued error-feedback Top-K baseline.
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
    tail = min(int(tail), int(n_ticks))

    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    thresholds = config.delta0 * ensemble.threshold_scale

    rng = np.random.default_rng(seed)
    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    residual = np.zeros_like(z)

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    packets = np.zeros(n_runs, dtype=np.int64)
    events = np.zeros(n_runs, dtype=np.int64)
    coordinate_events = np.zeros((n_runs, dimension), dtype=np.int64)
    client_events = np.zeros((n_runs, n_clients), dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    suppressed_candidates = np.zeros(n_runs, dtype=np.int64)
    whole_gap = np.zeros(n_runs)
    tail_gap = np.zeros(n_runs)
    tail_dist2 = np.zeros(n_runs)
    packet_sizes: list[int] = []
    candidate_sizes: list[int] = []

    for tick in range(1, n_ticks + 1):
        if config.method in ("lif_independent", "lif_competitive"):
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

            if config.method in ("lif_independent", "lif_competitive"):
                z[client] -= config.gamma * rate_weight * gradient
                eligible = np.abs(z[client]) >= thresholds[None, :]
                if record_packet_sizes:
                    candidate_sizes.extend(eligible.sum(axis=1).tolist())

                if config.method == "lif_independent":
                    selected = eligible
                else:
                    scores = np.abs(z[client]) / thresholds[None, :]
                    selected = _topk_eligible(scores, eligible, config.topk)
                    candidate_events += eligible.sum(axis=1)
                    suppressed_candidates += eligible.sum(axis=1) - selected.sum(axis=1)

                if np.any(selected):
                    signs = np.sign(z[client])
                    w += config.jump * signs * selected
                    n_events = selected.sum(axis=1)
                    events += n_events
                    coordinate_events += selected.astype(np.int64)
                    client_events[:, client] += n_events
                    payload_bits += n_events * event_bits
                    has_packet = n_events > 0
                    packets += has_packet.astype(np.int64)
                    packetized_bits += (
                        n_events * event_bits
                        + has_packet.astype(np.int64) * config.header_bits
                    )
                    if record_packet_sizes:
                        packet_sizes.extend(n_events[n_events > 0].tolist())
                    # Full reset only for selected neurons. Suppressed candidates retain evidence.
                    z[client][selected] = 0.0

            elif config.method == "sign_topk":
                k = min(max(int(config.topk), 1), dimension)
                idx = np.argpartition(np.abs(gradient), -k, axis=1)[:, -k:]
                runs = np.arange(n_runs)[:, None]
                selected = np.zeros((n_runs, dimension), dtype=bool)
                selected[runs, idx] = True
                w -= config.jump * rate_weight * np.sign(gradient) * selected
                events += k
                coordinate_events += selected.astype(np.int64)
                client_events[:, client] += k
                payload_bits += k * event_bits
                packetized_bits += k * event_bits + config.header_bits
                packets += 1
                if record_packet_sizes:
                    packet_sizes.extend([k] * n_runs)

            elif config.method == "ef_topk":
                residual[client] -= config.step * rate_weight * gradient
                k = min(max(int(config.topk), 1), dimension)
                idx = np.argpartition(np.abs(residual[client]), -k, axis=1)[:, -k:]
                vals = np.take_along_axis(residual[client], idx, axis=1)
                runs = np.arange(n_runs)[:, None]
                np.add.at(w, (runs, idx), vals)
                residual[client][runs, idx] = 0.0
                payload_bits += k * (32 + address_bits)
                packetized_bits += k * (32 + address_bits) + config.header_bits
                packets += 1
                if record_packet_sizes:
                    packet_sizes.extend([k] * n_runs)

            else:
                raise ValueError(f"unknown competitive method {config.method}")

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        current_gap = excess_objective(w, wstar, hbar)
        whole_gap += current_gap
        if tick > n_ticks - tail:
            tail_gap += current_gap
            tail_dist2 += np.sum((w - wstar) ** 2, axis=1)

    total_client_events = int(client_events.sum())
    coordinate_mean = coordinate_events.mean(axis=0)
    candidate_total = int(candidate_events.sum())
    slow_start = max(0, n_clients - 2)
    result: dict[str, object] = {
        "tail_gap": float(np.mean(tail_gap / tail)),
        "tail_rmse_w": float(np.sqrt(np.mean(tail_dist2 / (tail * dimension)))),
        "whole_gap": float(np.mean(whole_gap / n_ticks)),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "packets": float(np.mean(packets)),
        "events": float(np.mean(events)),
        "coverage": float(np.mean((coordinate_events > 0).mean(axis=1)))
        if coordinate_events.sum()
        else 0.0,
        "event_count_cv": float(np.std(coordinate_mean) / np.mean(coordinate_mean))
        if np.mean(coordinate_mean) > 0
        else np.nan,
        "slow_client_event_share": float(
            client_events[:, slow_start:].sum() / total_client_events
        )
        if total_client_events
        else np.nan,
        "candidate_events": float(np.mean(candidate_events)),
        "suppressed_candidates": float(np.mean(suppressed_candidates)),
        "suppression_fraction": float(suppressed_candidates.sum() / candidate_total)
        if candidate_total
        else 0.0,
        "coordinate_events_mean": coordinate_mean,
    }
    if record_packet_sizes:
        sizes = np.asarray(packet_sizes, dtype=float)
        candidates = np.asarray(candidate_sizes, dtype=float)
        result.update(
            {
                "mean_events_per_nonempty_packet": float(np.mean(sizes)) if sizes.size else 0.0,
                "p95_events_per_nonempty_packet": float(np.percentile(sizes, 95)) if sizes.size else 0.0,
                "max_events_per_nonempty_packet": float(np.max(sizes)) if sizes.size else 0.0,
                "max_candidate_count": float(np.max(candidates)) if candidates.size else 0.0,
            }
        )
    return result


def check_competitive_lif_ema_equivalence(
    *,
    rho: float = 0.999,
    gamma: float = 0.05,
    rate_weight: float = 0.5,
    delta0: float = 0.25,
    topk: int = 2,
    steps: int = 2000,
    seed: int = 77,
) -> dict[str, float]:
    """Verify competitive full-reset LIF == thresholded reset-EMA Top-K."""

    dimension = 20
    base_h = np.geomspace(0.2, 5.0, dimension)
    scale = base_h / np.median(base_h)
    threshold = delta0 * scale
    ema_threshold = (1.0 - rho) * threshold / (gamma * rate_weight)
    rng = np.random.default_rng(seed)
    z = np.zeros(dimension)
    ema = np.zeros(dimension)
    mismatches = 0
    max_state_error = 0.0

    for _ in range(steps):
        gradient = rng.normal(0.0, 1.0, dimension)
        z = rho * z - gamma * rate_weight * gradient
        ema = rho * ema + (1.0 - rho) * gradient
        eligible_z = np.abs(z) >= threshold
        eligible_ema = np.abs(ema) >= ema_threshold
        if not np.array_equal(eligible_z, eligible_ema):
            mismatches += 1

        selected_z = _topk_eligible(
            (np.abs(z) / threshold)[None, :], eligible_z[None, :], topk
        )[0]
        selected_ema = _topk_eligible(
            (np.abs(ema) / ema_threshold)[None, :], eligible_ema[None, :], topk
        )[0]
        if not np.array_equal(selected_z, selected_ema):
            mismatches += 1
        if np.any(selected_z):
            if not np.array_equal(
                np.sign(z[selected_z]), -np.sign(ema[selected_z])
            ):
                mismatches += 1
            z[selected_z] = 0.0
            ema[selected_ema] = 0.0

        max_state_error = max(
            max_state_error,
            float(
                np.max(
                    np.abs(
                        ema + (1.0 - rho) * z / (gamma * rate_weight)
                    )
                )
            ),
        )

    return {
        "mismatching_steps": float(mismatches),
        "max_mapped_state_error": float(max_state_error),
    }
