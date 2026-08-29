from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AsyncClient:
    """Configuration for one scalar asynchronous LIF/IF client.

    Parameters
    ----------
    theta:
        Optimum of the local quadratic F_i(w)=0.5*(w-theta)^2.
    period:
        Number of wall-clock ticks between local gradient evaluations.
    phase:
        Optional scheduling offset. The client is active when
        (tick - phase) % period == 0 and tick >= phase.
    """

    theta: float
    period: int
    phase: int = 0

    def active(self, tick: int) -> bool:
        return tick >= self.phase and (tick - self.phase) % self.period == 0


@dataclass
class AsyncRun:
    w: np.ndarray
    membrane: np.ndarray
    communications: np.ndarray
    event_log: list[tuple[int, int, int, float, bool]]
    harmful_events: int
    events_per_client: np.ndarray
    harmful_per_client: np.ndarray


def run_async_scalar_federation(
    *,
    clients: Sequence[AsyncClient],
    n_ticks: int,
    w0: float,
    gamma: float,
    threshold: float,
    jump: float,
    rho: float,
    noise_std: float = 0.0,
    seed: int = 0,
    oracle_reset_others: bool = False,
    active_order: Iterable[int] | None = None,
) -> AsyncRun:
    """Simulate an asynchronous event-driven scalar federation.

    All clients share the current server model `w`, but retain independent
    membrane states. At every wall-clock tick all membranes first decay by
    `rho`, which approximates continuous LIF leakage even for clients that are
    temporarily idle. Active clients then evaluate a noisy local gradient,
    integrate it, and emit fixed signed jumps whenever their membrane crosses
    `threshold`.

    Events are applied to the server immediately. Thus later clients at the
    same tick observe the updated model. Other clients retain their membrane
    state unless `oracle_reset_others=True`, which implements the deliberately
    aggressive diagnostic baseline that invalidates all remote evidence after
    every server update.

    An event is marked harmful when its sign opposes the instantaneous descent
    direction of that client's *current* local quadratic. For homogeneous
    clients with theta=0 this is also a harmful event for the global objective.
    """

    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1].")
    if threshold <= 0.0 or jump <= 0.0 or gamma <= 0.0:
        raise ValueError("gamma, threshold, and jump must be positive.")

    rng = np.random.default_rng(seed)
    n_clients = len(clients)
    w = float(w0)
    z = np.zeros(n_clients, dtype=float)

    ws = np.empty(n_ticks + 1)
    membranes = np.empty((n_ticks + 1, n_clients))
    communications = np.zeros(n_ticks + 1, dtype=int)
    event_log: list[tuple[int, int, int, float, bool]] = []
    events_per_client = np.zeros(n_clients, dtype=int)
    harmful_per_client = np.zeros(n_clients, dtype=int)

    ws[0] = w
    membranes[0] = z
    n_comm = 0

    for tick in range(n_ticks):
        # Wall-clock leak: stored evidence ages even when a client does not
        # compute a gradient at this particular tick.
        z *= rho

        active = [i for i, client in enumerate(clients) if client.active(tick)]
        if active_order is not None:
            rank = {idx: r for r, idx in enumerate(active_order)}
            active.sort(key=lambda idx: rank.get(idx, len(rank) + idx))

        for i in active:
            client = clients[i]
            gradient = (w - client.theta) + rng.normal(0.0, noise_std)
            z[i] -= gamma * gradient

            while abs(z[i]) >= threshold:
                sign = 1 if z[i] > 0.0 else -1
                w_before = w

                true_gradient = w_before - client.theta
                desired_sign = -np.sign(true_gradient) if true_gradient != 0.0 else 0
                harmful = bool(desired_sign != 0 and sign != desired_sign)

                w += jump * sign
                z[i] -= sign * threshold
                n_comm += 1
                events_per_client[i] += 1
                harmful_per_client[i] += int(harmful)
                event_log.append((tick, i, sign, w_before, harmful))

                if oracle_reset_others:
                    for j in range(n_clients):
                        if j != i:
                            z[j] = 0.0

        ws[tick + 1] = w
        membranes[tick + 1] = z
        communications[tick + 1] = n_comm

    return AsyncRun(
        w=ws,
        membrane=membranes,
        communications=communications,
        event_log=event_log,
        harmful_events=int(harmful_per_client.sum()),
        events_per_client=events_per_client,
        harmful_per_client=harmful_per_client,
    )
