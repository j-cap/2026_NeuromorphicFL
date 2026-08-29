from __future__ import annotations

import numpy as np


def mean_absolute_error(w: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ensemble mean absolute tracking error along time axis."""
    return np.mean(np.abs(w - target[None, :]), axis=0)


def rmse(w: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ensemble root mean-square tracking error along time axis."""
    return np.sqrt(np.mean((w - target[None, :]) ** 2, axis=0))


def tail_mse(w: np.ndarray, target: np.ndarray, tail: int) -> float:
    err = w[:, -tail:] - target[None, -tail:]
    return float(np.mean(err ** 2))


def communication_reduction(events: float, periodic_steps: int) -> float:
    return 1.0 - float(events) / float(periodic_steps)
