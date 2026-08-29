from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mechanism_audit import AuditProblem


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    """Full-reset LIF with spike-triggered adaptive firing threshold.

    The threshold is

        Delta_i(t) = Delta_0 + a_i(t),

    where the adaptation state relaxes exponentially between events and is
    incremented after every emitted event,

        a_i(t+1) = exp(-1/tau_a) a_i(t),
        a_i^+    = a_i + delta_a.

    Repeated firing therefore raises the threshold temporarily, whereas quiet
    clients relax back to the baseline threshold. Setting ``adapt_increment=0``
    recovers a fixed-threshold full-reset LIF encoder.
    """

    rho: float = 0.999
    base_threshold: float = 0.5
    adapt_increment: float = 0.0
    adapt_tau: float = 200.0
    jump: float = 0.05

    def __post_init__(self):
        if not (0.0 < self.rho <= 1.0):
            raise ValueError("rho must lie in (0, 1]")
        if self.base_threshold <= 0.0:
            raise ValueError("base_threshold must be positive")
        if self.adapt_increment < 0.0:
            raise ValueError("adapt_increment must be nonnegative")
        if self.adapt_tau <= 0.0:
            raise ValueError("adapt_tau must be positive")
        if self.jump <= 0.0:
            raise ValueError("jump must be positive")


@dataclass
class HomeostaticBatchResult:
    tail_rmse: float
    whole_mse: float
    mean_events: float
    mean_slow_events: float
    harmful_fraction: float
    mean_tail_threshold_slow: float
    mean_tail_threshold_fast: float
    pre_switch_event_rate: float
    post_switch_event_rate: float
    threshold_trace: np.ndarray
    event_rate_trace: np.ndarray


def run_homeostatic_batch(
    *,
    problem: AuditProblem,
    config: AdaptiveThresholdConfig,
    noise_std: float,
    n_ticks: int,
    n_seeds: int,
    tail: int,
    seed: int,
    noise_std_after: float | None = None,
    noise_switch_tick: int | None = None,
) -> HomeostaticBatchResult:
    """Run delayed asynchronous full-reset LIF with adaptive thresholds.

    Clients use the same rate-normalized evidence gain as Experiments 8--9,

        gamma_i = gamma * p_i * T_i.

    A gradient computation starts from the current server snapshot and returns
    after the client's compute period. The membrane leaks every wall-clock tick.

    The adaptive threshold implements spike-frequency adaptation rather than a
    fixed nonzero firing-rate setpoint. This matters for optimization because a
    useful encoder should be allowed to become silent near convergence.
    """

    thetas = np.asarray(problem.thetas, dtype=float)
    weights = np.asarray(problem.weights, dtype=float)
    periods = np.asarray(problem.periods, dtype=int)
    n_clients = len(thetas)
    if n_clients != 2:
        raise ValueError("current diagnostics assume two clients for reporting")

    tail = min(tail, n_ticks)
    if noise_switch_tick is None:
        noise_switch_tick = n_ticks // 2

    rng = np.random.default_rng(seed)
    w = rng.uniform(problem.w0_low, problem.w0_high, size=n_seeds)
    z = np.zeros((n_clients, n_seeds), dtype=float)
    adaptation = np.zeros((n_clients, n_seeds), dtype=float)
    snapshots = np.tile(w, (n_clients, 1))
    next_completion = periods.copy()

    noise = rng.normal(
        0.0,
        1.0,
        size=(n_ticks + 1, n_clients, n_seeds),
    )

    events = np.zeros(n_seeds, dtype=int)
    events_per_client = np.zeros((n_clients, n_seeds), dtype=int)
    harmful = np.zeros(n_seeds, dtype=int)
    whole_w2 = np.zeros(n_seeds)
    tail_w2 = np.zeros(n_seeds)

    threshold_trace = np.zeros((n_clients, n_ticks + 1), dtype=float)
    threshold_trace[:, 0] = config.base_threshold
    event_rate_trace = np.zeros(n_ticks + 1, dtype=float)

    adapt_decay = np.exp(-1.0 / config.adapt_tau)

    for tick in range(1, n_ticks + 1):
        z *= config.rho
        adaptation *= adapt_decay
        emitted_this_tick = np.zeros(n_seeds, dtype=float)

        sigma = noise_std
        if noise_std_after is not None and tick > noise_switch_tick:
            sigma = noise_std_after

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1 and tick % 2:
            active.reverse()

        for client in active:
            gradient = (
                snapshots[client]
                - thetas[client]
                + sigma * noise[tick, client]
            )
            gain = problem.gamma * weights[client] * periods[client]
            z[client] -= gain * gradient

            threshold = config.base_threshold + adaptation[client]
            mask = np.abs(z[client]) >= threshold
            if np.any(mask):
                idx = np.flatnonzero(mask)
                signs = np.sign(z[client, mask]).astype(int)

                w_before = w[idx].copy()
                w_after = w_before + config.jump * signs
                harmful[idx] += (
                    w_after**2 > w_before**2 + 1e-15
                ).astype(int)
                w[idx] = w_after

                events[idx] += 1
                events_per_client[client, idx] += 1
                emitted_this_tick[idx] += 1.0

                # Full-reset neuron with spike-triggered threshold adaptation.
                z[client, idx] = 0.0
                adaptation[client, idx] += config.adapt_increment

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        whole_w2 += w**2
        if tick > n_ticks - tail:
            tail_w2 += w**2

        threshold_trace[:, tick] = (
            config.base_threshold + adaptation
        ).mean(axis=1)
        event_rate_trace[tick] = emitted_this_tick.mean()

    event_count = int(events.sum())
    half = noise_switch_tick
    pre_rate = float(event_rate_trace[1 : half + 1].mean()) if half > 0 else np.nan
    post_rate = (
        float(event_rate_trace[half + 1 :].mean())
        if half < n_ticks
        else np.nan
    )

    return HomeostaticBatchResult(
        tail_rmse=float(np.sqrt(np.mean(tail_w2 / tail))),
        whole_mse=float(np.mean(whole_w2 / n_ticks)),
        mean_events=float(np.mean(events)),
        mean_slow_events=float(np.mean(events_per_client[0])),
        harmful_fraction=(
            float(harmful.sum() / event_count) if event_count else np.nan
        ),
        mean_tail_threshold_slow=float(threshold_trace[0, -tail:].mean()),
        mean_tail_threshold_fast=float(threshold_trace[1, -tail:].mean()),
        pre_switch_event_rate=pre_rate,
        post_switch_event_rate=post_rate,
        threshold_trace=threshold_trace,
        event_rate_trace=event_rate_trace,
    )
