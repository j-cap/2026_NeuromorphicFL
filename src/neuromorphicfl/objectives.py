from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuadraticObjective:
    """Scalar quadratic F(w)=0.5*a*(w-theta)^2."""

    a: float = 1.0
    theta: float = 0.0

    def value(self, w: float) -> float:
        e = w - self.theta
        return 0.5 * self.a * e * e

    def gradient(self, w: float) -> float:
        return self.a * (w - self.theta)


@dataclass(frozen=True)
class PiecewiseQuadraticObjective:
    """Quadratic objective with one abrupt optimum change."""

    a: float = 1.0
    theta_before: float = 0.0
    theta_after: float = 2.0
    switch_step: int = 500

    def theta(self, step: int) -> float:
        return self.theta_before if step < self.switch_step else self.theta_after

    def value(self, w: float, step: int) -> float:
        e = w - self.theta(step)
        return 0.5 * self.a * e * e

    def gradient(self, w: float, step: int) -> float:
        return self.a * (w - self.theta(step))
