from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DelayedAsyncClient:
    """One scalar client with an explicit computation delay.

    The local objective is

        F_i(w) = 0.5 * (w - theta)^2.

    A computation started from a server snapshot returns after ``period``
    wall-clock ticks. ``weight`` is the intended client weight p_i in the
    aggregate objective, not a compute-rate weight.
    """

    theta: float
    period: int
    weight: float


@dataclass(frozen=True)
class DelayedAsyncEvent:
    tick: int
    client: int
    sign: int
    w_before: float
    snapshot: float
    local_stale: bool
    global_harmful: bool


@dataclass
class DelayedAsyncRun:
    w: np.ndarray
    membrane: np.ndarray
    communications: np.ndarray
    event_log: list[DelayedAsyncEvent]
    stale_gradient_results: np.ndarray
    gradient_results_per_client: np.ndarray
    events_per_client: np.ndarray
    locally_stale_per_client: np.ndarray
    globally_harmful_per_client: np.ndarray


def weighted_quadratic_objective(w: float, clients: Sequence[DelayedAsyncClient]) -> float:
    """Evaluate the intended aggregate objective sum_i p_i F_i(w)."""

    return float(
        sum(client.weight * 0.5 * (w - client.theta) ** 2 for client in clients)
    )


def run_delayed_rate_normalized_scalar_federation(
    *,
    clients: Sequence[DelayedAsyncClient],
    n_ticks: int,
    w0: float,
    gamma: float,
    threshold: float,
    jump: float,
    rho: float,
    noise_std: float = 0.0,
    seed: int = 0,
    use_fresh_gradient_oracle: bool = False,
    alternate_tie_order: bool = True,
) -> DelayedAsyncRun:
    """Reference simulator for delayed asynchronous LIF/IF optimization.

    Each client starts a gradient computation from a server-model snapshot.
    The result becomes available only after ``client.period`` wall-clock ticks,
    during which other clients may have changed the server model. This differs
    from sparse sampling at the current server model: the returned gradient is
    genuinely stale.

    Compute-rate normalization
    --------------------------
    A client with period T_i completes approximately once every T_i ticks. Its
    evidence increment is therefore scaled as

        gamma_i = gamma * p_i * T_i.

    For an approximately constant gradient, the average evidence injection per
    wall-clock tick is then -gamma * p_i * g_i, independent of client speed.
    Thus hardware speed does not implicitly redefine the intended aggregate
    objective.

    Event labels
    ------------
    ``local_stale`` means the emitted event sign disagrees with the exact local
    descent direction at the *current* server model. ``global_harmful`` means
    the event increases the intended weighted aggregate objective exactly.

    The fresh-gradient oracle keeps the same completion schedule and rate
    normalization, but evaluates the gradient at the current model when the job
    completes. It therefore removes computation-delay staleness while retaining
    all other algorithmic effects.
    """

    if not clients:
        raise ValueError("At least one client is required.")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1].")
    if gamma <= 0.0 or threshold <= 0.0 or jump <= 0.0:
        raise ValueError("gamma, threshold, and jump must be positive.")
    if n_ticks <= 0:
        raise ValueError("n_ticks must be positive.")
    if any(client.period <= 0 for client in clients):
        raise ValueError("All client periods must be positive integers.")
    if any(client.weight <= 0.0 for client in clients):
        raise ValueError("All client weights must be positive.")

    weight_sum = sum(client.weight for client in clients)
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Client weights must sum to one; got {weight_sum}.")

    rng = np.random.default_rng(seed)
    n_clients = len(clients)
    periods = np.array([client.period for client in clients], dtype=int)
    evidence_gains = gamma * np.array([client.weight for client in clients]) * periods

    w = float(w0)
    z = np.zeros(n_clients, dtype=float)
    snapshots = np.full(n_clients, w, dtype=float)
    next_completion = periods.copy()

    ws = np.empty(n_ticks + 1)
    membranes = np.empty((n_ticks + 1, n_clients))
    communications = np.zeros(n_ticks + 1, dtype=int)
    events_per_client = np.zeros(n_clients, dtype=int)
    locally_stale_per_client = np.zeros(n_clients, dtype=int)
    globally_harmful_per_client = np.zeros(n_clients, dtype=int)
    stale_gradient_results = np.zeros(n_clients, dtype=int)
    gradient_results_per_client = np.zeros(n_clients, dtype=int)
    event_log: list[DelayedAsyncEvent] = []

    ws[0] = w
    membranes[0] = z
    n_comm = 0

    for tick in range(1, n_ticks + 1):
        # Wall-clock leakage: evidence ages while a computation is in flight.
        z *= rho

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if alternate_tie_order and len(active) > 1 and tick % 2 == 1:
            active.reverse()

        for i in active:
            client = clients[i]
            snapshot = float(snapshots[i])
            current_local_gradient = w - client.theta
            snapshot_local_gradient = snapshot - client.theta

            old_descent = -np.sign(snapshot_local_gradient)
            current_descent = -np.sign(current_local_gradient)
            gradient_is_stale = bool(
                old_descent != 0.0
                and current_descent != 0.0
                and old_descent != current_descent
            )
            stale_gradient_results[i] += int(gradient_is_stale)
            gradient_results_per_client[i] += 1

            gradient_point = w if use_fresh_gradient_oracle else snapshot
            gradient = gradient_point - client.theta + rng.normal(0.0, noise_std)
            z[i] -= evidence_gains[i] * gradient

            while abs(z[i]) >= threshold:
                sign = 1 if z[i] > 0.0 else -1
                w_before = w
                current_descent = -np.sign(w_before - client.theta)
                local_stale = bool(current_descent != 0.0 and sign != current_descent)

                w_after = w_before + jump * sign
                f_before = weighted_quadratic_objective(w_before, clients)
                f_after = weighted_quadratic_objective(w_after, clients)
                global_harmful = bool(f_after > f_before + 1e-15)

                w = w_after
                z[i] -= sign * threshold
                n_comm += 1
                events_per_client[i] += 1
                locally_stale_per_client[i] += int(local_stale)
                globally_harmful_per_client[i] += int(global_harmful)
                event_log.append(
                    DelayedAsyncEvent(
                        tick=tick,
                        client=i,
                        sign=sign,
                        w_before=w_before,
                        snapshot=snapshot,
                        local_stale=local_stale,
                        global_harmful=global_harmful,
                    )
                )

            # Immediately start the next local computation from the current
            # server model after this client's returned result/events are handled.
            snapshots[i] = w
            next_completion[i] = tick + client.period

        ws[tick] = w
        membranes[tick] = z
        communications[tick] = n_comm

    return DelayedAsyncRun(
        w=ws,
        membrane=membranes,
        communications=communications,
        event_log=event_log,
        stale_gradient_results=stale_gradient_results,
        gradient_results_per_client=gradient_results_per_client,
        events_per_client=events_per_client,
        locally_stale_per_client=locally_stale_per_client,
        globally_harmful_per_client=globally_harmful_per_client,
    )
