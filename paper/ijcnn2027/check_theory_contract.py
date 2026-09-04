"""Executable checks for the compact Event-FedAvg theory contract.

This is not a proof assistant. It independently exercises the algebraic
identities used in the paper and guards the correspondence between the paper's
transition and the frozen implementation against accidental drift.
"""

from __future__ import annotations

import math
from pathlib import Path
import random


REPO = Path(__file__).resolve().parents[2]
IMPLEMENTATION = REPO / "src" / "neuromorphicfl" / "final_baseline_campaign.py"
PAPER = REPO / "paper" / "ijcnn2027" / "main.tex"


def check_scalar_encoder() -> None:
    rng = random.Random(20260904)
    for _ in range(500):
        threshold = rng.uniform(0.05, 2.0)
        state = rng.uniform(-0.999 * threshold, 0.999 * threshold)
        initial = state
        event_count = 0
        total_input = 0.0
        for _round in range(200):
            rho = rng.random()
            evidence = rng.gauss(0.0, 0.4)
            total_input += abs(evidence)
            pretrigger = rho * state + evidence
            fired = abs(pretrigger) >= threshold
            event_count += int(fired)
            state = 0.0 if fired else pretrigger
            assert abs(state) < threshold
        assert event_count * threshold <= abs(initial) + total_input + 1e-12


def check_alignment_and_energy() -> None:
    rng = random.Random(20260905)
    for _ in range(500):
        clients = rng.randint(1, 12)
        dimension = rng.randint(1, 40)
        gradient = [rng.uniform(-2.0, 2.0) for _ in range(dimension)]
        pulses = [
            [rng.choice((-1, 0, 0, 0, 1)) for _ in range(dimension)]
            for _ in range(clients)
        ]
        aggregate = [sum(pulse[j] for pulse in pulses) for j in range(dimension)]
        event_count = sum(value * value for pulse in pulses for value in pulse)
        descent = sum(
            abs(gradient[j])
            for pulse in pulses
            for j, value in enumerate(pulse)
            if value * gradient[j] < 0.0
        )
        harmful = sum(
            abs(gradient[j])
            for pulse in pulses
            for j, value in enumerate(pulse)
            if value * gradient[j] > 0.0
        )
        inner = -sum(g * c for g, c in zip(gradient, aggregate))
        assert math.isclose(inner, descent - harmful, rel_tol=1e-12, abs_tol=1e-12)
        energy = sum(value * value for value in aggregate)
        assert energy <= clients * event_count


def check_implementation_correspondence() -> None:
    source = IMPLEMENTATION.read_text(encoding="utf-8")
    required = (
        "evidence_gain = 1.0 / config.local_lr",
        "event_state *= config.rho",
        "event_state[client] += evidence_gain * float(weights[client]) * deltas[client]",
        "mask = np.abs(event_state[client]) >= config.threshold",
        "signs = np.sign(event_state[client, mask]).astype(np.int8)",
        "aggregate[mask] += jump * signs.astype(np.float32)",
        "event_state[client, mask] = 0.0",
    )
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise AssertionError(f"implementation/theory correspondence drift: {missing}")

    paper = PAPER.read_text(encoding="utf-8")
    paper_requirements = (
        "a_i^r=\\frac{p_i}{\\eta_r}\\delta_i^r",
        "C^r=\\sum_{i=1}^M c_i^r",
        "w^{r+1}=w^r+q_rC^r",
        "\\E[A_r\\mid\\mathcal{F}_r]",
        "finite-trajectory diagnostics",
    )
    absent = [snippet for snippet in paper_requirements if snippet not in paper]
    if absent:
        raise AssertionError(f"paper theory contract drift: {absent}")


def main() -> None:
    check_scalar_encoder()
    check_alignment_and_energy()
    check_implementation_correspondence()
    print("Event-FedAvg theory contract checks passed")


if __name__ == "__main__":
    main()
