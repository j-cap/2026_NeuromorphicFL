from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Method = Literal[
    "fedlif",
    "lif_full_reset",
    "if_remote_reset",
    "memoryless_pulse",
    "ema_pulse",
    "periodic_sign",
    "full_precision",
]


@dataclass(frozen=True)
class AuditProblem:
    thetas: np.ndarray
    weights: np.ndarray
    periods: np.ndarray
    gamma: float = 0.05
    w0_low: float = 0.93
    w0_high: float = 1.13

    def __post_init__(self):
        if not (len(self.thetas) == len(self.weights) == len(self.periods)):
            raise ValueError("thetas, weights, and periods must have equal length")
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("weights must sum to one")


@dataclass(frozen=True)
class AuditConfig:
    method: Method
    rho: float = 1.0
    threshold: float = 0.5
    jump: float = 0.1
    ema_beta: float = 0.0
    full_precision_step: float = 0.005


def run_audit_batch(
    *,
    problem: AuditProblem,
    config: AuditConfig,
    noise_std: float,
    n_ticks: int,
    n_seeds: int,
    tail: int,
    seed: int,
) -> dict[str, float]:
    """Run one mechanism-comparator configuration.

    All clients compute on a server snapshot and return the gradient only after
    their configured period. Evidence gains are rate normalized as

        gamma_i = gamma * p_i * T_i,

    so the expected evidence injection per wall-clock tick is approximately
    proportional to the intended client weight rather than compute speed.

    Event-based methods apply a fixed model jump ``jump`` per signed event.
    ``mean_events`` therefore provides the scalar communication proxy used in
    Experiment 09. Vector experiments must additionally count coordinate/block
    addresses and packet overhead.
    """

    thetas = np.asarray(problem.thetas, dtype=float)
    weights = np.asarray(problem.weights, dtype=float)
    periods = np.asarray(problem.periods, dtype=int)
    n_clients = len(thetas)
    tail = min(tail, n_ticks)

    rng = np.random.default_rng(seed)
    w = rng.uniform(problem.w0_low, problem.w0_high, size=n_seeds)
    z = np.zeros((n_clients, n_seeds), dtype=float)
    ema = np.zeros((n_clients, n_seeds), dtype=float)
    snapshots = np.tile(w, (n_clients, 1))
    next_completion = periods.copy()
    noise = rng.normal(0.0, noise_std, size=(n_ticks + 1, n_clients, n_seeds))

    events = np.zeros(n_seeds, dtype=int)
    events_per_client = np.zeros((n_clients, n_seeds), dtype=int)
    harmful = np.zeros(n_seeds, dtype=int)
    whole_w2 = np.zeros(n_seeds)
    tail_w2 = np.zeros(n_seeds)

    def apply_event(client: int, signs: np.ndarray, idx: np.ndarray) -> None:
        nonlocal w, z
        w_before = w[idx].copy()
        w_after = w_before + config.jump * signs
        harmful[idx] += (w_after**2 > w_before**2 + 1e-15).astype(int)
        w[idx] = w_after
        events[idx] += 1
        events_per_client[client, idx] += 1
        if config.method == "if_remote_reset":
            for other in range(n_clients):
                if other != client:
                    z[other, idx] = 0.0

    for tick in range(1, n_ticks + 1):
        if config.method in ("fedlif", "lif_full_reset"):
            z *= config.rho

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1 and tick % 2:
            active.reverse()

        for client in active:
            gradient = snapshots[client] - thetas[client] + noise[tick, client]
            gain = problem.gamma * weights[client] * periods[client]
            evidence = -gain * gradient

            if config.method in ("fedlif", "if_remote_reset"):
                z[client] += evidence
                for _ in range(100):
                    mask = np.abs(z[client]) >= config.threshold
                    if not np.any(mask):
                        break
                    signs = np.sign(z[client, mask]).astype(int)
                    z[client, mask] -= config.threshold * signs
                    apply_event(client, signs, np.flatnonzero(mask))
                else:
                    raise RuntimeError("too many threshold crossings in one completion")

            elif config.method == "lif_full_reset":
                z[client] += evidence
                mask = np.abs(z[client]) >= config.threshold
                if np.any(mask):
                    signs = np.sign(z[client, mask]).astype(int)
                    apply_event(client, signs, np.flatnonzero(mask))
                    z[client, mask] = 0.0

            elif config.method in ("memoryless_pulse", "ema_pulse"):
                if config.method == "ema_pulse":
                    ema[client] = (
                        config.ema_beta * ema[client]
                        + (1.0 - config.ema_beta) * gradient
                    )
                    pulse_evidence = -gain * ema[client]
                else:
                    pulse_evidence = evidence

                counts = np.minimum(
                    np.floor(np.abs(pulse_evidence) / config.threshold).astype(int),
                    20,
                )
                for pulse in range(int(counts.max(initial=0))):
                    mask = counts > pulse
                    if np.any(mask):
                        signs = np.sign(pulse_evidence[mask]).astype(int)
                        apply_event(client, signs, np.flatnonzero(mask))

            elif config.method == "periodic_sign":
                signs = -np.sign(gradient).astype(int)
                mask = signs != 0
                if np.any(mask):
                    apply_event(client, signs[mask], np.flatnonzero(mask))

            elif config.method == "full_precision":
                w_before = w.copy()
                step = (
                    config.full_precision_step
                    * weights[client]
                    * periods[client]
                    * gradient
                )
                w -= step
                events += 1
                harmful += (w**2 > w_before**2 + 1e-15).astype(int)

            else:
                raise ValueError(f"unknown method {config.method}")

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        whole_w2 += w**2
        if tick > n_ticks - tail:
            tail_w2 += w**2

    event_count = int(events.sum())
    return {
        "tail_rmse": float(np.sqrt(np.mean(tail_w2 / tail))),
        "whole_mse": float(np.mean(whole_w2 / n_ticks)),
        "mean_events": float(np.mean(events)),
        "mean_slow_events": float(np.mean(events_per_client[0])),
        "harmful_fraction": float(harmful.sum() / event_count)
        if event_count
        else np.nan,
    }


def pareto_mask(events: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Return mask of points non-dominated when both quantities are minimized."""

    events = np.asarray(events)
    error = np.asarray(error)
    keep = np.ones(len(events), dtype=bool)
    for i in range(len(events)):
        dominated = (
            (events <= events[i])
            & (error <= error[i])
            & ((events < events[i]) | (error < error[i]))
        )
        if np.any(dominated):
            keep[i] = False
    return keep


def check_fedlif_error_feedback_equivalence(
    *,
    rho: float = 0.995,
    threshold: float = 0.7,
    jump: float = 0.05,
    gamma: float = 0.05,
    noise_std: float = 0.25,
    theta: float = 0.05,
    steps: int = 5000,
    seed: int = 1,
) -> dict[str, float]:
    """Numerically verify FedLIF == decayed error feedback after rescaling.

    With e=(q/Delta)z, the FedLIF recurrence maps exactly to a decayed
    error-feedback residual in parameter-update units.
    """

    rng = np.random.default_rng(seed)
    z = 0.0
    e = 0.0
    w_lif = 1.0
    w_ef = 1.0
    scale = jump / threshold
    mismatches = 0
    max_state_error = 0.0

    for _ in range(steps):
        gradient = (w_lif - theta) + rng.normal(0.0, noise_std)
        evidence = -gamma * gradient

        z = rho * z + evidence
        while abs(z) >= threshold:
            sign = np.sign(z)
            w_lif += jump * sign
            z -= threshold * sign

        e = rho * e + scale * evidence
        while abs(e) >= jump:
            sign = np.sign(e)
            w_ef += jump * sign
            e -= jump * sign

        mismatches += int(abs(w_lif - w_ef) > 1e-12)
        max_state_error = max(max_state_error, abs(e - scale * z))

    return {
        "mismatching_steps": float(mismatches),
        "max_mapped_state_error": float(max_state_error),
        "final_model_difference": float(abs(w_lif - w_ef)),
    }


def check_fullreset_ema_equivalence(
    *,
    rho: float = 0.999,
    threshold: float = 0.9,
    jump: float = 0.05,
    gain: float = 0.05,
    noise_std: float = 0.25,
    theta: float = 0.05,
    steps: int = 5000,
    seed: int = 2,
) -> dict[str, float]:
    """Verify full-reset LIF == thresholded EMA with reset after rescaling."""

    rng = np.random.default_rng(seed)
    z = 0.0
    ema = 0.0
    w_lif = 1.0
    w_ema = 1.0
    ema_threshold = (1.0 - rho) * threshold / gain
    mismatches = 0
    max_state_error = 0.0

    for _ in range(steps):
        gradient = (w_lif - theta) + rng.normal(0.0, noise_std)

        z = rho * z - gain * gradient
        fired_lif = False
        lif_sign = 0.0
        if abs(z) >= threshold:
            lif_sign = np.sign(z)
            w_lif += jump * lif_sign
            z = 0.0
            fired_lif = True

        ema = rho * ema + (1.0 - rho) * gradient
        fired_ema = False
        ema_sign = 0.0
        if abs(ema) >= ema_threshold:
            ema_sign = -np.sign(ema)
            w_ema += jump * ema_sign
            ema = 0.0
            fired_ema = True

        mismatch = (
            fired_lif != fired_ema
            or (fired_lif and lif_sign != ema_sign)
            or abs(w_lif - w_ema) > 1e-12
        )
        mismatches += int(mismatch)
        max_state_error = max(
            max_state_error,
            abs(ema + (1.0 - rho) * z / gain),
        )

    return {
        "mismatching_steps": float(mismatches),
        "max_mapped_state_error": float(max_state_error),
        "final_model_difference": float(abs(w_lif - w_ema)),
        "mapped_ema_threshold": float(ema_threshold),
    }
