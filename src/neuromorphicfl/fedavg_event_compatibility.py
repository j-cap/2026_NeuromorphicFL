from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import (
    LAYOUT,
    MulticlassFederation,
    MulticlassMLPLayout,
    initialize_mlp,
    loss_and_gradient,
    predictive_metrics,
)


Method = Literal["event_fedavg", "dense_fedavg", "ef_topk_fedavg", "sign_ef_fedavg"]


@dataclass(frozen=True)
class FedAvgConfig:
    local_steps: int = 5
    local_lr: float = 0.02
    batch_size: int = 32
    regularization: float = 1e-4
    rounds: int = 150
    participation: float = 1.0
    rho: float = 0.999
    threshold: float = 0.025
    jump0: float = 0.0035
    jump_scale: float = 100.0
    jump_exponent: float = 0.1
    encoder_gain_multiplier: float = 1.0
    topk_fraction: float = 0.025
    eval_stride: int = 10


def _local_delta(
    w_server: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    local_steps: int,
    local_lr: float,
    batch_size: int,
    regularization: float,
    rng: np.random.Generator,
    layout: MulticlassMLPLayout,
) -> np.ndarray:
    w_local = w_server.copy()
    n = len(y)
    for _ in range(local_steps):
        ids = rng.integers(0, n, size=batch_size)
        _, _, gradient = loss_and_gradient(
            w_local,
            X[ids],
            y[ids],
            layout=layout,
            regularization=regularization,
            need_gradient=True,
        )
        w_local -= local_lr * gradient
    return (w_local - w_server).astype(np.float32, copy=False)


def run_fedavg_method(
    *,
    federation: MulticlassFederation,
    method: Method,
    config: FedAvgConfig,
    seed: int = 70707,
    layout: MulticlassMLPLayout = LAYOUT,
    init_scale: float = 0.5,
    record_history: bool = False,
):
    if config.local_steps < 1:
        raise ValueError("local_steps must be >=1")
    if config.local_lr <= 0:
        raise ValueError("local_lr must be positive")
    if not (0 < config.participation <= 1):
        raise ValueError("participation must lie in (0,1]")

    rng = np.random.default_rng(seed)
    n_clients = federation.n_clients
    d = layout.dimension
    address_bits = int(math.ceil(math.log2(d)))
    event_bits = address_bits + 1
    topk = max(1, int(round(config.topk_fraction * d)))
    n_participating = max(1, int(round(config.participation * n_clients)))

    w = initialize_mlp(layout=layout, seed=7777, scale=init_scale)
    membrane = np.zeros((n_clients, d), dtype=np.float32)
    residual = np.zeros((n_clients, d), dtype=np.float32)
    last_event_sign = np.zeros((n_clients, d), dtype=np.int8)

    payload_bits = 0
    packetized_bits = 0
    messages = 0
    coordinate_events = 0
    sign_reversals = 0
    repeated_event_coordinates = 0
    client_payload = np.zeros(n_clients, dtype=np.int64)
    client_events = np.zeros(n_clients, dtype=np.int64)
    delta_norms: list[float] = []
    membrane_norms: list[float] = []
    history_rows: list[dict[str, float | int]] = []

    # Delta_i ~= -eta sum_e g_i,e. Dividing by eta makes the encoder state
    # approximately accumulate weighted local gradient evidence while still
    # being driven by the actual multi-step model delta.
    event_gain = config.encoder_gain_multiplier / config.local_lr

    for rnd in range(1, config.rounds + 1):
        if n_participating == n_clients:
            active = np.arange(n_clients, dtype=int)
        else:
            active = np.sort(rng.choice(n_clients, size=n_participating, replace=False))

        deltas: dict[int, np.ndarray] = {}
        for client in active:
            delta = _local_delta(
                w,
                federation.client_X[client],
                federation.client_y[client],
                local_steps=config.local_steps,
                local_lr=config.local_lr,
                batch_size=config.batch_size,
                regularization=config.regularization,
                rng=rng,
                layout=layout,
            )
            deltas[int(client)] = delta
            delta_norms.append(float(np.linalg.norm(delta)))

        # Renormalize the selected-client weights exactly as standard FedAvg.
        active_weights = federation.weights[active].astype(float)
        active_weights /= np.sum(active_weights)

        if method == "dense_fedavg":
            aggregate = np.zeros(d, dtype=np.float32)
            for client, weight in zip(active, active_weights):
                aggregate += float(weight) * deltas[int(client)]
                bits = 32 * d
                payload_bits += bits
                packetized_bits += bits + 64
                messages += 1
                client_payload[int(client)] += bits
            w += aggregate

        elif method == "ef_topk_fedavg":
            aggregate = np.zeros(d, dtype=np.float32)
            for client, weight in zip(active, active_weights):
                ci = int(client)
                residual[ci] += float(weight) * deltas[ci]
                ids_top = np.argpartition(np.abs(residual[ci]), -topk)[-topk:]
                compressed = residual[ci, ids_top].copy()
                aggregate[ids_top] += compressed
                residual[ci, ids_top] = 0.0
                bits = topk * (32 + address_bits)
                payload_bits += bits
                packetized_bits += bits + 64
                messages += 1
                client_payload[ci] += bits
            w += aggregate

        elif method == "sign_ef_fedavg":
            aggregate = np.zeros(d, dtype=np.float32)
            for client, weight in zip(active, active_weights):
                ci = int(client)
                residual[ci] += float(weight) * deltas[ci]
                scale = float(np.mean(np.abs(residual[ci])))
                if scale > 0:
                    compressed = scale * np.sign(residual[ci])
                    aggregate += compressed.astype(np.float32, copy=False)
                    residual[ci] -= compressed.astype(np.float32, copy=False)
                bits = d + 32  # dense sign vector + one float32 scale
                payload_bits += bits
                packetized_bits += bits + 64
                messages += 1
                client_payload[ci] += bits
            w += aggregate

        elif method == "event_fedavg":
            aggregate = np.zeros(d, dtype=np.float32)
            membrane *= config.rho
            jump = config.jump0 * (1.0 + rnd / config.jump_scale) ** (-config.jump_exponent)
            for client, weight in zip(active, active_weights):
                ci = int(client)
                membrane[ci] += event_gain * float(weight) * deltas[ci]
                mask = np.abs(membrane[ci]) >= config.threshold
                count = int(np.sum(mask))
                membrane_norms.append(float(np.linalg.norm(membrane[ci])))
                if count:
                    signs = np.sign(membrane[ci, mask]).astype(np.int8)
                    previous = last_event_sign[ci, mask]
                    repeated = previous != 0
                    repeated_event_coordinates += int(np.sum(repeated))
                    sign_reversals += int(np.sum(repeated & (previous != signs)))
                    last_event_sign[ci, mask] = signs
                    aggregate[mask] += jump * signs.astype(np.float32)
                    membrane[ci, mask] = 0.0
                    coordinate_events += count
                    client_events[ci] += count
                    bits = count * event_bits
                    payload_bits += bits
                    packetized_bits += bits + 64
                    messages += 1
                    client_payload[ci] += bits
            w += aggregate
        else:
            raise ValueError(method)

        if rnd == 1 or rnd % config.eval_stride == 0 or rnd == config.rounds:
            train_obj, train_ce, _, _, _, _ = predictive_metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=config.regularization,
            )
            _, test_ce, test_acc, macro_acc, worst_acc, _ = predictive_metrics(
                w,
                federation.X_test,
                federation.y_test,
                layout=layout,
                regularization=config.regularization,
            )
            history_rows.append({
                "round": rnd,
                "train_objective": train_obj,
                "train_ce": train_ce,
                "test_ce": test_ce,
                "test_accuracy": test_acc,
                "macro_accuracy": macro_acc,
                "worst_class_accuracy": worst_acc,
                "payload_bits": payload_bits,
                "packetized_bits": packetized_bits,
            })

    history = pd.DataFrame(history_rows)
    final_train_obj, final_train_ce, _, _, _, _ = predictive_metrics(
        w,
        federation.X_train_eval,
        federation.y_train_eval,
        layout=layout,
        regularization=config.regularization,
    )
    _, final_test_ce, final_test_acc, final_macro, final_worst, per_class = predictive_metrics(
        w,
        federation.X_test,
        federation.y_test,
        layout=layout,
        regularization=config.regularization,
    )

    nonzero_payload = client_payload[client_payload > 0]
    client_payload_cv = (
        float(np.std(nonzero_payload) / np.mean(nonzero_payload)) if len(nonzero_payload) > 1 else 0.0
    )
    nonzero_events = client_events[client_events > 0]
    client_event_cv = (
        float(np.std(nonzero_events) / np.mean(nonzero_events)) if len(nonzero_events) > 1 else 0.0
    )

    result: dict[str, float | int | str] = {
        "method": method,
        "dimension": d,
        "rounds": config.rounds,
        "local_steps": config.local_steps,
        "local_lr": config.local_lr,
        "participation": config.participation,
        "final_train_objective": final_train_obj,
        "final_train_ce": final_train_ce,
        "final_test_ce": final_test_ce,
        "final_test_accuracy": final_test_acc,
        "final_macro_accuracy": final_macro,
        "final_worst_class_accuracy": final_worst,
        "whole_train_objective": float(history["train_objective"].mean()),
        "payload_bits": int(payload_bits),
        "packetized_bits": int(packetized_bits),
        "messages": int(messages),
        "coordinate_events": int(coordinate_events),
        "events_per_message": coordinate_events / messages if messages else 0.0,
        "mean_delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
        "max_delta_norm": float(np.max(delta_norms)) if delta_norms else 0.0,
        "mean_membrane_norm": float(np.mean(membrane_norms)) if membrane_norms else 0.0,
        "final_membrane_norm": float(np.linalg.norm(membrane)) if method == "event_fedavg" else 0.0,
        "sign_reversal_rate": sign_reversals / repeated_event_coordinates if repeated_event_coordinates else 0.0,
        "client_payload_cv": client_payload_cv,
        "client_event_cv": client_event_cv,
        "event_gain": event_gain if method == "event_fedavg" else 0.0,
        "rho": config.rho,
        "threshold": config.threshold,
        "jump0": config.jump0,
        "jump_exponent": config.jump_exponent,
        "topk_fraction": config.topk_fraction,
    }
    for cls, acc in enumerate(per_class):
        result[f"class_{cls}_accuracy"] = float(acc)
    if record_history:
        result["history"] = history  # type: ignore[assignment]
    return result
