from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .optimizers import IFSGD, LIFSGD, SGD, SignSGD


@dataclass
class ScalarRun:
    w: np.ndarray
    target: np.ndarray
    gradient: np.ndarray
    membrane: np.ndarray
    communications: np.ndarray
    event_signs: list[tuple[int, int]]


def run_scalar_optimizer(
    *,
    optimizer,
    n_steps: int,
    w0: float,
    gradient_fn: Callable[[float, int], float],
    target_fn: Callable[[int], float],
    noise_std: float = 0.0,
    seed: int = 0,
    membrane_reset_step: Optional[int] = None,
) -> ScalarRun:
    """Run one stochastic scalar optimization trajectory.

    `gradient_fn(w, k)` should return the deterministic gradient. Gaussian
    gradient noise with standard deviation `noise_std` is then added.
    """

    rng = np.random.default_rng(seed)
    w = float(w0)

    ws = np.empty(n_steps + 1)
    targets = np.empty(n_steps + 1)
    gradients = np.empty(n_steps)
    membranes = np.full(n_steps + 1, np.nan)
    communications = np.zeros(n_steps + 1, dtype=int)
    event_signs: list[tuple[int, int]] = []

    ws[0] = w
    targets[0] = target_fn(0)
    if isinstance(optimizer, (IFSGD, LIFSGD)):
        membranes[0] = optimizer.membrane

    n_comm = 0
    for k in range(n_steps):
        if membrane_reset_step is not None and k == membrane_reset_step:
            if isinstance(optimizer, (IFSGD, LIFSGD)):
                optimizer.reset_membrane()

        g = gradient_fn(w, k) + rng.normal(0.0, noise_std)
        gradients[k] = g
        w, events = optimizer.step(w, g)

        if isinstance(optimizer, SGD):
            n_comm += 1
        elif isinstance(optimizer, SignSGD):
            n_comm += 1
            event_signs.extend((k, s) for _, s in events)
        else:
            n_comm += len(events)
            event_signs.extend((k, s) for _, s in events)

        ws[k + 1] = w
        targets[k + 1] = target_fn(k + 1)
        communications[k + 1] = n_comm
        if isinstance(optimizer, (IFSGD, LIFSGD)):
            membranes[k + 1] = optimizer.membrane

    return ScalarRun(
        w=ws,
        target=targets,
        gradient=gradients,
        membrane=membranes,
        communications=communications,
        event_signs=event_signs,
    )
