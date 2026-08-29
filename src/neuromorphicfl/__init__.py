"""Reusable primitives for NeuromorphicFL experiments."""

from .objectives import QuadraticObjective, PiecewiseQuadraticObjective
from .optimizers import SGD, SignSGD, IFSGD, LIFSGD
from .simulation import run_scalar_optimizer
from .async_simulation import AsyncClient, AsyncRun, run_async_scalar_federation

__all__ = [
    "QuadraticObjective",
    "PiecewiseQuadraticObjective",
    "SGD",
    "SignSGD",
    "IFSGD",
    "LIFSGD",
    "run_scalar_optimizer",
    "AsyncClient",
    "AsyncRun",
    "run_async_scalar_federation",
]
