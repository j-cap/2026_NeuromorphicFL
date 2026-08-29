from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Event = Tuple[int, int]  # (coordinate placeholder for scalar=0, sign)


@dataclass
class SGD:
    learning_rate: float

    def step(self, w: float, gradient: float) -> tuple[float, list[Event]]:
        return w - self.learning_rate * gradient, []


@dataclass
class SignSGD:
    step_size: float

    def step(self, w: float, gradient: float) -> tuple[float, list[Event]]:
        if gradient == 0.0:
            return w, []
        s = -1 if gradient > 0.0 else 1
        return w + self.step_size * s, [(0, s)]


@dataclass
class IFSGD:
    """Integrate-and-fire gradient encoder with subtractive reset."""

    gamma: float
    threshold: float
    jump: float
    membrane: float = 0.0

    def reset_membrane(self) -> None:
        self.membrane = 0.0

    def step(self, w: float, gradient: float) -> tuple[float, list[Event]]:
        self.membrane -= self.gamma * gradient
        events: List[Event] = []

        while abs(self.membrane) >= self.threshold:
            s = 1 if self.membrane > 0.0 else -1
            w += self.jump * s
            self.membrane -= s * self.threshold
            events.append((0, s))

        return w, events


@dataclass
class LIFSGD(IFSGD):
    """Leaky integrate-and-fire SGD.

    The discrete membrane dynamics are
        z_{k+1} = rho z_k - gamma g_k,
    followed by threshold crossings, signed fixed-amplitude parameter jumps,
    and subtractive reset.
    """

    rho: float = 0.98

    def step(self, w: float, gradient: float) -> tuple[float, list[Event]]:
        self.membrane = self.rho * self.membrane - self.gamma * gradient
        events: List[Event] = []

        while abs(self.membrane) >= self.threshold:
            s = 1 if self.membrane > 0.0 else -1
            w += self.jump * s
            self.membrane -= s * self.threshold
            events.append((0, s))

        return w, events
