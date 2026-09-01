from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math

import numpy as np
import pandas as pd

from .fmnist_event_benchmark import PERIODS, load_fashion_mnist


@dataclass(frozen=True)
class MulticlassFederation:
    client_X: tuple[np.ndarray, ...]
    client_y: tuple[np.ndarray, ...]
    X_train_eval: np.ndarray
    y_train_eval: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    regime: str
    client_class_counts: np.ndarray
    periods: np.ndarray
    weights: np.ndarray

    @property
    def n_clients(self) -> int:
        return len(self.client_X)


@dataclass(frozen=True)
class MulticlassMLPLayout:
    widths: tuple[int, ...] = (784, 32, 16, 10)

    @property
    def n_layers(self) -> int:
        return len(self.widths) - 1

    @property
    def dimension(self) -> int:
        return int(
            sum(
                self.widths[layer] * self.widths[layer + 1]
                + self.widths[layer + 1]
                for layer in range(self.n_layers)
            )
        )

    def groups(self) -> list[tuple[str, slice, int]]:
        groups: list[tuple[str, slice, int]] = []
        start = 0
        for layer in range(self.n_layers):
            n_in, n_out = self.widths[layer], self.widths[layer + 1]
            weight_slice = slice(start, start + n_in * n_out)
            start = weight_slice.stop
            bias_slice = slice(start, start + n_out)
            start = bias_slice.stop
            groups.append((f"W{layer + 1}", weight_slice, n_in * n_out))
            groups.append((f"b{layer + 1}", bias_slice, n_out))
        return groups


LAYOUT = MulticlassMLPLayout()


def class_count_matrix(regime: str) -> np.ndarray:
    """Return an exact 10-client x 10-class balanced partition design.

    Fashion-MNIST contains exactly 6000 training images per class.  Each client
    receives 6000 samples.  The constructions below preserve both the global
    class balance and equal client sample counts exactly.
    """
    if regime == "iid":
        return np.full((10, 10), 600, dtype=int)
    if regime == "moderate":
        counts = np.full((10, 10), 400, dtype=int)
        np.fill_diagonal(counts, 2400)  # 40% dominant class, 6.67% each other class.
        return counts
    if regime == "strong":
        counts = np.full((10, 10), 300, dtype=int)
        np.fill_diagonal(counts, 3300)  # 55% dominant class, 5% each other class.
        return counts
    if regime == "extreme":
        counts = np.full((10, 10), 100, dtype=int)
        np.fill_diagonal(counts, 5100)  # 85% dominant class, 1.67% each other class.
        return counts
    raise ValueError(f"unknown regime {regime}")


def make_multiclass_federation(
    *,
    root: str | Path,
    regime: str = "strong",
    seed: int = 2400,
    train_eval_size: int = 5000,
) -> MulticlassFederation:
    train_images, train_labels, test_images, test_labels = load_fashion_mnist(root)
    counts = class_count_matrix(regime)
    if not np.all(counts.sum(axis=1) == 6000):
        raise RuntimeError("each client must receive exactly 6000 training samples")
    if not np.all(counts.sum(axis=0) == 6000):
        raise RuntimeError("each Fashion-MNIST class must be used exactly once")

    rng = np.random.default_rng(seed)
    per_class_indices: list[np.ndarray] = []
    for cls in range(10):
        ids = np.flatnonzero(train_labels == cls)
        if len(ids) != 6000:
            raise RuntimeError(f"unexpected Fashion-MNIST class count for class {cls}")
        rng.shuffle(ids)
        per_class_indices.append(ids)

    offsets = np.zeros(10, dtype=int)
    clients_X: list[np.ndarray] = []
    clients_y: list[np.ndarray] = []
    global_ids: list[np.ndarray] = []
    for client in range(10):
        ids_parts: list[np.ndarray] = []
        labels_parts: list[np.ndarray] = []
        for cls in range(10):
            n_take = int(counts[client, cls])
            start = int(offsets[cls])
            stop = start + n_take
            ids_cls = per_class_indices[cls][start:stop]
            offsets[cls] = stop
            ids_parts.append(ids_cls)
            labels_parts.append(np.full(n_take, cls, dtype=np.int64))
        ids = np.concatenate(ids_parts)
        labels = np.concatenate(labels_parts)
        order = rng.permutation(len(ids))
        ids = ids[order]
        labels = labels[order]
        clients_X.append(train_images[ids].astype(np.float32) / 127.5 - 1.0)
        clients_y.append(labels)
        global_ids.append(ids)

    if not np.all(offsets == 6000):
        raise RuntimeError("partition did not consume every training sample exactly once")

    all_ids = np.concatenate(global_ids)
    eval_pos = rng.choice(
        len(all_ids), size=min(train_eval_size, len(all_ids)), replace=False
    )
    X_train_eval = train_images[all_ids[eval_pos]].astype(np.float32) / 127.5 - 1.0
    y_train_eval = train_labels[all_ids[eval_pos]].astype(np.int64)

    test_order = rng.permutation(len(test_labels))
    X_test = test_images[test_order].astype(np.float32) / 127.5 - 1.0
    y_test = test_labels[test_order].astype(np.int64)

    return MulticlassFederation(
        client_X=tuple(clients_X),
        client_y=tuple(clients_y),
        X_train_eval=X_train_eval,
        y_train_eval=y_train_eval,
        X_test=X_test,
        y_test=y_test,
        regime=regime,
        client_class_counts=counts,
        periods=PERIODS.copy(),
        weights=np.full(10, 0.1),
    )


def federation_audit(federation: MulticlassFederation) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for client in range(federation.n_clients):
        row: dict[str, float | int] = {
            "client": client,
            "period": int(federation.periods[client]),
            "n_train": int(len(federation.client_y[client])),
            "dominant_class": int(np.argmax(federation.client_class_counts[client])),
            "dominant_fraction": float(
                np.max(federation.client_class_counts[client])
                / federation.client_class_counts[client].sum()
            ),
        }
        for cls in range(10):
            row[f"class_{cls}_count"] = int(federation.client_class_counts[client, cls])
        rows.append(row)
    return pd.DataFrame(rows)


def initialize_mlp(
    *,
    layout: MulticlassMLPLayout = LAYOUT,
    seed: int = 7777,
    scale: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.zeros(layout.dimension, dtype=np.float32)
    groups = layout.groups()
    for layer in range(layout.n_layers):
        n_in = layout.widths[layer]
        weight_slice = groups[2 * layer][1]
        w[weight_slice] = rng.normal(
            0.0,
            scale / math.sqrt(n_in),
            size=weight_slice.stop - weight_slice.start,
        ).astype(np.float32)
    return w


def _unpack(w: np.ndarray, layout: MulticlassMLPLayout = LAYOUT):
    layers = []
    groups = layout.groups()
    for layer in range(layout.n_layers):
        n_in, n_out = layout.widths[layer], layout.widths[layer + 1]
        W = w[groups[2 * layer][1]].reshape(n_in, n_out)
        b = w[groups[2 * layer + 1][1]]
        layers.append((W, b))
    return layers


def _forward_logits(
    w: np.ndarray,
    X: np.ndarray,
    *,
    layout: MulticlassMLPLayout = LAYOUT,
) -> tuple[list[np.ndarray], np.ndarray]:
    activations: list[np.ndarray] = [X]
    a = X
    layers = _unpack(w, layout)
    for layer, (W, b) in enumerate(layers):
        z = a @ W + b
        a = np.tanh(z) if layer < layout.n_layers - 1 else z
        activations.append(a)
    return activations, activations[-1]


def loss_and_gradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    layout: MulticlassMLPLayout = LAYOUT,
    regularization: float = 1e-4,
    need_gradient: bool = True,
):
    activations, logits = _forward_logits(w, X, layout=layout)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(
        logits, axis=1
    )
    predictive = float(np.mean(logsumexp - logits[np.arange(len(y)), y]))
    objective = predictive + 0.5 * regularization * float(w @ w)
    if not need_gradient:
        return objective, predictive

    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    delta = probs
    delta[np.arange(len(y)), y] -= 1.0
    delta /= len(y)

    layers = _unpack(w, layout)
    gradients: list[tuple[np.ndarray, np.ndarray]] = [None] * layout.n_layers  # type: ignore
    for layer in range(layout.n_layers - 1, -1, -1):
        W, _ = layers[layer]
        previous = activations[layer]
        grad_W = previous.T @ delta
        grad_b = np.sum(delta, axis=0)
        gradients[layer] = (grad_W, grad_b)
        if layer > 0:
            delta = (delta @ W.T) * (1.0 - activations[layer] ** 2)

    gradient = np.empty_like(w)
    groups = layout.groups()
    for layer, (grad_W, grad_b) in enumerate(gradients):
        gradient[groups[2 * layer][1]] = grad_W.reshape(-1)
        gradient[groups[2 * layer + 1][1]] = grad_b
    gradient += regularization * w
    return objective, predictive, gradient.astype(np.float32, copy=False)


def predictive_metrics(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    layout: MulticlassMLPLayout = LAYOUT,
    regularization: float = 1e-4,
):
    objective, predictive = loss_and_gradient(
        w,
        X,
        y,
        layout=layout,
        regularization=regularization,
        need_gradient=False,
    )
    _, logits = _forward_logits(w, X, layout=layout)
    predictions = np.argmax(logits, axis=1)
    accuracy = float(np.mean(predictions == y))
    per_class_accuracy = np.array(
        [np.mean(predictions[y == cls] == cls) for cls in range(10)], dtype=float
    )
    macro_accuracy = float(np.mean(per_class_accuracy))
    worst_class_accuracy = float(np.min(per_class_accuracy))
    return objective, predictive, accuracy, macro_accuracy, worst_class_accuracy, per_class_accuracy


def run_federated_method(
    *,
    federation: MulticlassFederation,
    method: Literal["events", "ef_topk", "full"],
    n_ticks: int,
    seed: int = 60606,
    layout: MulticlassMLPLayout = LAYOUT,
    init_scale: float = 0.5,
    batch_size: int = 32,
    regularization: float = 1e-4,
    step: float = 0.02,
    topk_fraction: float = 0.01,
    rho: float = 0.999,
    gamma: float = 0.3,
    threshold: float = 0.025,
    jump0: float = 0.0025,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.1,
    eval_stride: int = 50,
    record_history: bool = False,
):
    rng = np.random.default_rng(seed)
    dimension = layout.dimension
    address_bits = int(math.ceil(math.log2(dimension)))
    event_bits = address_bits + 1
    topk = max(1, int(round(topk_fraction * dimension)))

    w = initialize_mlp(layout=layout, scale=init_scale)
    snapshots = np.repeat(w[None, :], federation.n_clients, axis=0)
    next_completion = federation.periods.copy()
    membrane = np.zeros((federation.n_clients, dimension), dtype=np.float32)
    residual = np.zeros((federation.n_clients, dimension), dtype=np.float32)

    payload_bits = 0
    packetized_bits = 0
    messages = 0
    candidate_events = 0
    group_events = np.zeros(len(layout.groups()), dtype=np.int64)
    ever_fired = np.zeros(dimension, dtype=bool)
    history_rows: list[dict[str, float]] = []

    for tick in range(1, n_ticks + 1):
        if method == "events":
            membrane *= rho

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
            _, _, gradient = loss_and_gradient(
                snapshots[client],
                federation.client_X[client][ids],
                federation.client_y[client][ids],
                layout=layout,
                regularization=regularization,
                need_gradient=True,
            )
            rate_weight = float(federation.weights[client] * federation.periods[client])

            if method == "full":
                w -= step * rate_weight * gradient
                payload_bits += 32 * dimension
                packetized_bits += 32 * dimension + 32
                messages += 1

            elif method == "ef_topk":
                residual[client] -= step * rate_weight * gradient
                ids_top = np.argpartition(np.abs(residual[client]), -topk)[-topk:]
                w[ids_top] += residual[client, ids_top]
                residual[client, ids_top] = 0.0
                payload = topk * (32 + address_bits)
                payload_bits += payload
                packetized_bits += payload + 32
                messages += 1

            else:
                membrane[client] -= gamma * rate_weight * gradient
                mask = np.abs(membrane[client]) >= threshold
                count = int(np.sum(mask))
                if count:
                    jump = jump0 * (1.0 + tick / schedule_scale) ** (-schedule_exponent)
                    w[mask] += jump * np.sign(membrane[client, mask])
                    membrane[client, mask] = 0.0
                    candidate_events += count
                    payload = count * event_bits
                    payload_bits += payload
                    packetized_bits += payload + 32
                    messages += 1
                    ever_fired |= mask
                    for group_id, (_, slc, _) in enumerate(layout.groups()):
                        group_events[group_id] += int(np.sum(mask[slc]))

            snapshots[client] = w
            next_completion[client] = tick + federation.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train_obj, train_ce, _, _, _, _ = predictive_metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=regularization,
            )
            _, test_ce, test_acc, macro_acc, worst_acc, _ = predictive_metrics(
                w,
                federation.X_test,
                federation.y_test,
                layout=layout,
                regularization=regularization,
            )
            history_rows.append(
                {
                    "tick": tick,
                    "payload_bits": payload_bits,
                    "packetized_bits": packetized_bits,
                    "train_objective": train_obj,
                    "train_ce": train_ce,
                    "test_ce": test_ce,
                    "test_accuracy": test_acc,
                    "macro_accuracy": macro_acc,
                    "worst_class_accuracy": worst_acc,
                }
            )

    history = pd.DataFrame(history_rows)
    final_train_obj, final_train_ce, _, _, _, _ = predictive_metrics(
        w,
        federation.X_train_eval,
        federation.y_train_eval,
        layout=layout,
        regularization=regularization,
    )
    (
        _,
        final_test_ce,
        final_test_acc,
        final_macro_acc,
        final_worst_acc,
        final_per_class,
    ) = predictive_metrics(
        w,
        federation.X_test,
        federation.y_test,
        layout=layout,
        regularization=regularization,
    )

    result: dict[str, float | int | str] = {
        "method": method,
        "dimension": dimension,
        "n_ticks": n_ticks,
        "init_scale": init_scale,
        "batch_size": batch_size,
        "regularization": regularization,
        "step": step,
        "topk_fraction": topk_fraction,
        "topk": topk,
        "rho": rho,
        "gamma": gamma,
        "threshold": threshold,
        "jump0": jump0,
        "final_train_objective": final_train_obj,
        "final_train_ce": final_train_ce,
        "final_test_ce": final_test_ce,
        "final_test_accuracy": final_test_acc,
        "final_macro_accuracy": final_macro_acc,
        "final_worst_class_accuracy": final_worst_acc,
        "whole_train_objective": float(history["train_objective"].mean()),
        "payload_bits": payload_bits,
        "packetized_bits": packetized_bits,
        "messages": messages,
        "candidate_events": candidate_events,
        "events_per_message": (candidate_events / messages if messages else 0.0),
        "mean_packet_payload_bits": (payload_bits / messages if messages else 0.0),
        "ever_fired_fraction": (float(np.mean(ever_fired)) if method == "events" else np.nan),
        "parameter_norm": float(np.linalg.norm(w)),
    }
    for cls, acc in enumerate(final_per_class):
        result[f"class_{cls}_accuracy"] = float(acc)
    if method == "events":
        for group_id, (name, slc, n_params) in enumerate(layout.groups()):
            result[f"{name}_events_per_param"] = float(group_events[group_id] / n_params)
            result[f"{name}_never_fired"] = float(1.0 - np.mean(ever_fired[slc]))
    if record_history:
        result["history"] = history  # type: ignore
    return result
