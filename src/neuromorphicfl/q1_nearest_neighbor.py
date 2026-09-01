from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import (
    LAYOUT as MLP_LAYOUT,
    MulticlassFederation,
    initialize_mlp,
    loss_and_gradient as mlp_loss_and_gradient,
    predictive_metrics as mlp_predictive_metrics,
)
from .fmnist_cnn_benchmark import (
    LAYOUT as CNN_LAYOUT,
    initialize_cnn,
    loss_and_gradient as cnn_loss_and_gradient,
    predictive_metrics as cnn_predictive_metrics,
)


PulseVariant = Literal["strom", "leaky_subtractive", "fullreset_tied"]


def _run_pulse_method(
    *,
    federation: MulticlassFederation,
    architecture: Literal["mlp", "cnn"],
    variant: PulseVariant,
    n_ticks: int,
    seed: int,
    gain: float,
    threshold: float,
    rho: float = 0.999,
    batch_size: int = 32,
    regularization: float = 1e-4,
    init_scale: float = 0.5,
    eval_stride: int = 50,
    record_history: bool = False,
) -> dict[str, object]:
    """Run the nearest-neighbor pulse compressors under the Part-I protocol.

    ``strom`` reproduces the core 1-bit gradient-residual rule of Strom (2015):
    non-leaky accumulation, threshold-sized signed pulse, and subtractive residual
    update.  ``leaky_subtractive`` adds the V1 finite-memory factor but keeps
    residual conservation.  ``fullreset_tied`` replaces residual subtraction by
    a full coordinate reset while keeping the pulse quantum tied to the trigger
    threshold.  Frozen V1 is intentionally run by the canonical benchmark code,
    because its final difference is the independent, annealed pulse quantum.
    """
    if architecture == "mlp":
        layout = MLP_LAYOUT
        w = initialize_mlp(layout=layout, seed=7777, scale=init_scale)
        loss_grad = mlp_loss_and_gradient
        metrics = mlp_predictive_metrics
    elif architecture == "cnn":
        layout = CNN_LAYOUT
        w = initialize_cnn(layout=layout, seed=7777, scale=init_scale)
        loss_grad = cnn_loss_and_gradient
        metrics = cnn_predictive_metrics
    else:
        raise ValueError(f"unknown architecture {architecture}")

    d = int(layout.dimension)
    address_bits = int(math.ceil(math.log2(d)))
    event_bits = address_bits + 1
    rng = np.random.default_rng(seed)

    snapshots = np.repeat(w[None, :], federation.n_clients, axis=0)
    next_completion = federation.periods.copy()
    state = np.zeros((federation.n_clients, d), dtype=np.float32)

    payload_bits = 0
    packetized_bits = 0
    messages = 0
    coordinate_events = 0
    ever_fired = np.zeros(d, dtype=bool)
    history_rows: list[dict[str, float]] = []

    effective_rho = 1.0 if variant == "strom" else float(rho)
    reset_mode = "subtractive" if variant in {"strom", "leaky_subtractive"} else "full"

    for tick in range(1, n_ticks + 1):
        if effective_rho < 1.0:
            state *= effective_rho

        active = [
            client
            for client in range(federation.n_clients)
            if next_completion[client] == tick
        ]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_n = len(federation.client_y[client])
            ids = rng.integers(0, local_n, size=batch_size)
            _, _, gradient = loss_grad(
                snapshots[client],
                federation.client_X[client][ids],
                federation.client_y[client][ids],
                layout=layout,
                regularization=regularization,
                need_gradient=True,
            )
            rate_weight = float(
                federation.weights[client] * federation.periods[client]
            )
            state[client] -= gain * rate_weight * gradient

            mask = np.abs(state[client]) >= threshold
            count = int(np.sum(mask))
            if count:
                signs = np.sign(state[client, mask])
                # Strom's one-bit rule ties the model quantum to the threshold.
                w[mask] += threshold * signs
                if reset_mode == "subtractive":
                    state[client, mask] -= threshold * signs
                else:
                    state[client, mask] = 0.0
                coordinate_events += count
                payload = count * event_bits
                payload_bits += payload
                packetized_bits += payload + 32
                messages += 1
                ever_fired |= mask

            snapshots[client] = w
            next_completion[client] = tick + federation.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train = metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=regularization,
            )
            test = metrics(
                w,
                federation.X_test,
                federation.y_test,
                layout=layout,
                regularization=regularization,
            )
            history_rows.append(
                {
                    "tick": float(tick),
                    "payload_bits": float(payload_bits),
                    "packetized_bits": float(packetized_bits),
                    "train_objective": float(train[0]),
                    "train_ce": float(train[1]),
                    "test_ce": float(test[1]),
                    "test_accuracy": float(test[2]),
                    "worst_class_accuracy": float(test[4]),
                }
            )

    history = pd.DataFrame(history_rows)
    train = metrics(
        w,
        federation.X_train_eval,
        federation.y_train_eval,
        layout=layout,
        regularization=regularization,
    )
    test = metrics(
        w,
        federation.X_test,
        federation.y_test,
        layout=layout,
        regularization=regularization,
    )
    result: dict[str, object] = {
        "architecture": architecture,
        "variant": variant,
        "gain": float(gain),
        "threshold": float(threshold),
        "rho": float(effective_rho),
        "n_ticks": int(n_ticks),
        "final_train_objective": float(train[0]),
        "final_train_ce": float(train[1]),
        "final_test_ce": float(test[1]),
        "final_test_accuracy": float(test[2]),
        "final_worst_class_accuracy": float(test[4]),
        "whole_train_objective": float(history["train_objective"].mean()),
        "payload_bits": int(payload_bits),
        "packetized_bits": int(packetized_bits),
        "messages": int(messages),
        "coordinate_events": int(coordinate_events),
        "events_per_message": float(coordinate_events / max(messages, 1)),
        "ever_fired_fraction": float(np.mean(ever_fired)),
    }
    if record_history:
        result["history"] = history
    return result


def run_mlp_pulse_method(**kwargs) -> dict[str, object]:
    return _run_pulse_method(architecture="mlp", **kwargs)


def run_cnn_pulse_method(**kwargs) -> dict[str, object]:
    return _run_pulse_method(architecture="cnn", **kwargs)
