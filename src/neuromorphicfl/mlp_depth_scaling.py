from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .logistic_certificate import LogisticEnsemble

Method = Literal["events", "ef_topk", "full"]


@dataclass(frozen=True)
class NetworkLayout:
    widths: tuple[int, ...]

    @property
    def n_layers(self) -> int:
        return len(self.widths) - 1

    @property
    def dimension(self) -> int:
        return int(sum(
            self.widths[layer] * self.widths[layer + 1]
            + self.widths[layer + 1]
            for layer in range(self.n_layers)
        ))

    def groups(self) -> list[tuple[str, slice, int]]:
        groups: list[tuple[str, slice, int]] = []
        start = 0
        for layer in range(self.n_layers):
            n_in, n_out = self.widths[layer], self.widths[layer + 1]
            w_slice = slice(start, start + n_in * n_out)
            start = w_slice.stop
            b_slice = slice(start, start + n_out)
            start = b_slice.stop
            groups.extend([
                (f"W{layer + 1}", w_slice, n_in * n_out),
                (f"b{layer + 1}", b_slice, n_out),
            ])
        return groups


ARCHITECTURES = {
    "shallow_8": NetworkLayout((19, 8, 1)),
    "shallow_12": NetworkLayout((19, 12, 1)),
    "deep_8x8": NetworkLayout((19, 8, 8, 1)),
    "shallow_32": NetworkLayout((19, 32, 1)),
    "deep_16x16": NetworkLayout((19, 16, 16, 1)),
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def initialize_network(
    layout: NetworkLayout,
    n_runs: int,
    *,
    scale: float,
    seed: int = 7777,
) -> np.ndarray:
    """Initialize every weight tensor with std scale/sqrt(fan_in)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n_runs, layout.dimension))
    groups = layout.groups()
    for layer in range(layout.n_layers):
        n_in, n_out = layout.widths[layer], layout.widths[layer + 1]
        w_slice = groups[2 * layer][1]
        w[:, w_slice] = rng.normal(
            0.0,
            scale / np.sqrt(n_in),
            size=(n_runs, n_in * n_out),
        )
    return w


def _unpack(w: np.ndarray, layout: NetworkLayout):
    layers = []
    groups = layout.groups()
    for layer in range(layout.n_layers):
        n_in, n_out = layout.widths[layer], layout.widths[layer + 1]
        W = w[:, groups[2 * layer][1]].reshape(w.shape[0], n_in, n_out)
        b = w[:, groups[2 * layer + 1][1]]
        layers.append((W, b))
    return layers


def loss_and_gradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    layout: NetworkLayout,
    *,
    regularization: float = 0.02,
    need_gradient: bool = True,
):
    layers = _unpack(w, layout)
    activations = [X]
    current = X
    for layer, (W, b) in enumerate(layers):
        z = np.einsum("rni,rij->rnj", current, W) + b[:, None, :]
        current = np.tanh(z) if layer < layout.n_layers - 1 else z
        activations.append(current)

    logits = activations[-1][..., 0]
    data_loss = np.mean(np.logaddexp(0.0, logits) - y * logits, axis=1)
    objective = data_loss + 0.5 * regularization * np.sum(w * w, axis=1)
    if not need_gradient:
        return objective

    delta = ((_sigmoid(logits) - y) / X.shape[1])[:, :, None]
    layer_gradients = [None] * layout.n_layers
    for layer in range(layout.n_layers - 1, -1, -1):
        previous = activations[layer]
        W, _ = layers[layer]
        grad_W = np.einsum("rni,rnj->rij", previous, delta)
        grad_b = np.sum(delta, axis=1)
        layer_gradients[layer] = (grad_W, grad_b)
        if layer > 0:
            delta = np.einsum("rnj,rij->rni", delta, W)
            delta *= 1.0 - activations[layer] ** 2

    gradient = np.empty_like(w)
    groups = layout.groups()
    for layer, (grad_W, grad_b) in enumerate(layer_gradients):
        gradient[:, groups[2 * layer][1]] = grad_W.reshape(w.shape[0], -1)
        gradient[:, groups[2 * layer + 1][1]] = grad_b
    gradient += regularization * w
    return objective, gradient


def _features(ensemble: LogisticEnsemble, *, test: bool):
    X = ensemble.Xte if test else ensemble.Xtr
    y = ensemble.yte if test else ensemble.ytr
    # The Experiment-12 data contain a constant intercept feature. Neural
    # networks already have trainable biases, so remove it here.
    return X[..., :-1], y


def global_objective(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    layout: NetworkLayout,
    *,
    regularization: float = 0.02,
) -> np.ndarray:
    X, y = _features(ensemble, test=False)
    n_runs, n_clients, n_samples, n_features = X.shape
    return loss_and_gradient(
        w,
        X.reshape(n_runs, n_clients * n_samples, n_features),
        y.reshape(n_runs, n_clients * n_samples),
        layout,
        regularization=regularization,
        need_gradient=False,
    )


def predictive_metrics(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    layout: NetworkLayout,
):
    """Return unregularized test cross-entropy and accuracy.

    Cross-architecture comparisons must use predictive cross-entropy rather
    than the regularized objective, because the latter changes with parameter
    count and parameter norm.
    """
    X, y = _features(ensemble, test=True)
    n_runs, n_clients, n_samples, n_features = X.shape
    X = X.reshape(n_runs, n_clients * n_samples, n_features)
    y = y.reshape(n_runs, n_clients * n_samples)
    current = X
    for layer, (W, b) in enumerate(_unpack(w, layout)):
        z = np.einsum("rni,rij->rnj", current, W) + b[:, None, :]
        current = np.tanh(z) if layer < layout.n_layers - 1 else z
    logits = current[..., 0]
    cross_entropy = np.mean(np.logaddexp(0.0, logits) - y * logits, axis=1)
    accuracy = np.mean((logits >= 0.0) == y, axis=1)
    return cross_entropy, accuracy


def minibatch_gradient(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    client: int,
    rng: np.random.Generator,
    layout: NetworkLayout,
    *,
    batch_size: int = 64,
    regularization: float = 0.02,
) -> np.ndarray:
    n_runs = ensemble.n_runs
    n_samples = ensemble.Xtr.shape[2]
    X = np.empty((n_runs, batch_size, layout.widths[0]))
    y = np.empty((n_runs, batch_size))
    for run in range(n_runs):
        ids = rng.integers(0, n_samples, size=batch_size)
        X[run] = ensemble.Xtr[run, client, ids, :-1]
        y[run] = ensemble.ytr[run, client, ids]
    _, gradient = loss_and_gradient(
        w,
        X,
        y,
        layout,
        regularization=regularization,
        need_gradient=True,
    )
    return gradient


def run_depth_batch(
    *,
    ensemble: LogisticEnsemble,
    layout: NetworkLayout,
    method: Method,
    initialization_scale: float,
    n_ticks: int,
    seed: int = 60606,
    rho: float = 0.999,
    gamma: float = 0.2,
    threshold: float = 0.025,
    jump0: float = 0.015,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.1,
    step: float = 0.04,
    topk_fraction: float = 0.025,
    batch_size: int = 64,
    regularization: float = 0.02,
    eval_stride: int = 60,
    record_history: bool = False,
):
    n_runs = ensemble.n_runs
    n_clients = ensemble.n_clients
    dimension = layout.dimension
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    topk = max(1, int(np.ceil(topk_fraction * dimension)))
    rng = np.random.default_rng(seed)

    w = initialize_network(
        layout, n_runs, scale=initialization_scale
    )
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = ensemble.periods.copy()
    membrane = np.zeros((n_clients, n_runs, dimension))
    residual = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    candidate_events = np.zeros(n_runs, dtype=np.int64)
    group_events = np.zeros((n_runs, len(layout.groups())), dtype=np.int64)
    ever_fired = np.zeros((n_runs, dimension), dtype=bool)
    history_rows = []

    for tick in range(1, n_ticks + 1):
        if method == "events":
            membrane *= rho
        active = [
            client for client in range(n_clients)
            if next_completion[client] == tick
        ]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            gradient = minibatch_gradient(
                snapshots[client], ensemble, client, rng, layout,
                batch_size=batch_size,
                regularization=regularization,
            )
            rate_weight = ensemble.weights[client] * ensemble.periods[client]

            if method == "full":
                w -= step * rate_weight * gradient
                payload_bits += 32 * dimension
            elif method == "ef_topk":
                residual[client] -= step * rate_weight * gradient
                ids = np.argpartition(
                    np.abs(residual[client]), -topk, axis=1
                )[:, -topk:]
                values = np.take_along_axis(residual[client], ids, axis=1)
                runs = np.arange(n_runs)
                for column in range(topk):
                    w[runs, ids[:, column]] += values[:, column]
                    residual[client, runs, ids[:, column]] = 0.0
                payload_bits += topk * (32 + address_bits)
            else:
                membrane[client] -= gamma * rate_weight * gradient
                mask = np.abs(membrane[client]) >= threshold
                if np.any(mask):
                    count = mask.sum(axis=1)
                    candidate_events += count
                    payload_bits += count * event_bits
                    jump = jump0 * (
                        1.0 + tick / schedule_scale
                    ) ** (-schedule_exponent)
                    w += jump * np.sign(membrane[client]) * mask
                    ever_fired |= mask
                    for group_id, (_, group_slice, _) in enumerate(layout.groups()):
                        group_events[:, group_id] += mask[:, group_slice].sum(axis=1)
                    membrane[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + ensemble.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train = global_objective(
                w, ensemble, layout, regularization=regularization
            )
            test_ce, test_accuracy = predictive_metrics(w, ensemble, layout)
            history_rows.append({
                "tick": tick,
                "payload_bits": float(np.mean(payload_bits)),
                "train_objective": float(np.mean(train)),
                "test_cross_entropy": float(np.mean(test_ce)),
                "test_accuracy": float(np.mean(test_accuracy)),
            })

    history = pd.DataFrame(history_rows)
    test_ce, test_accuracy = predictive_metrics(w, ensemble, layout)
    result: dict[str, object] = {
        "method": method,
        "dimension": dimension,
        "topk": topk,
        "initialization_scale": initialization_scale,
        "final_train_objective": float(history.iloc[-1]["train_objective"]),
        "final_test_cross_entropy": float(np.mean(test_ce)),
        "final_test_accuracy": float(np.mean(test_accuracy)),
        "whole_train_objective": float(history["train_objective"].mean()),
        "payload_bits": float(np.mean(payload_bits)),
        "parameter_norm": float(np.mean(np.linalg.norm(w, axis=1))),
        "candidate_events": (
            float(np.mean(candidate_events)) if method == "events" else np.nan
        ),
        "ever_fired_fraction": (
            float(np.mean(ever_fired)) if method == "events" else np.nan
        ),
    }
    if method == "events":
        for group_id, (name, group_slice, n_parameters) in enumerate(layout.groups()):
            result[f"{name}_events_per_parameter"] = float(
                np.mean(group_events[:, group_id]) / n_parameters
            )
            result[f"{name}_never_fired_fraction"] = float(
                1.0 - np.mean(ever_fired[:, group_slice])
            )
    if record_history:
        result["history"] = history
    return result
