"""Reusable primitives for NeuromorphicFL experiments."""

from .objectives import QuadraticObjective, PiecewiseQuadraticObjective
from .optimizers import SGD, SignSGD, IFSGD, LIFSGD
from .simulation import run_scalar_optimizer
from .async_simulation import AsyncClient, AsyncRun, run_async_scalar_federation
from .delayed_async import (
    DelayedAsyncClient,
    DelayedAsyncEvent,
    DelayedAsyncRun,
    run_delayed_rate_normalized_scalar_federation,
    weighted_quadratic_objective,
)
from .homeostatic import (
    AdaptiveThresholdConfig,
    HomeostaticBatchResult,
    run_homeostatic_batch,
)

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
    "DelayedAsyncClient",
    "DelayedAsyncEvent",
    "DelayedAsyncRun",
    "run_delayed_rate_normalized_scalar_federation",
    "weighted_quadratic_objective",
    "AdaptiveThresholdConfig",
    "HomeostaticBatchResult",
    "run_homeostatic_batch",
]
