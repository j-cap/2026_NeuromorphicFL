from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import MulticlassFederation


@dataclass(frozen=True)
class CompactCNNLayout:
    conv1_out: int = 8
    conv1_kernel: int = 5
    conv1_stride: int = 2
    conv2_out: int = 16
    conv2_kernel: int = 3
    conv2_stride: int = 2
    hidden: int = 32
    n_classes: int = 10

    @property
    def conv1_hw(self) -> int:
        return (28 - self.conv1_kernel) // self.conv1_stride + 1

    @property
    def conv2_hw(self) -> int:
        return (self.conv1_hw - self.conv2_kernel) // self.conv2_stride + 1

    @property
    def flat_dim(self) -> int:
        return self.conv2_out * self.conv2_hw * self.conv2_hw

    @property
    def dimension(self) -> int:
        return (
            self.conv1_out * self.conv1_kernel * self.conv1_kernel
            + self.conv1_out
            + self.conv2_out * self.conv1_out * self.conv2_kernel * self.conv2_kernel
            + self.conv2_out
            + self.flat_dim * self.hidden
            + self.hidden
            + self.hidden * self.n_classes
            + self.n_classes
        )

    def groups(self) -> list[tuple[str, slice, int]]:
        sizes = [
            ("K1", self.conv1_out * self.conv1_kernel * self.conv1_kernel),
            ("b1", self.conv1_out),
            ("K2", self.conv2_out * self.conv1_out * self.conv2_kernel * self.conv2_kernel),
            ("b2", self.conv2_out),
            ("W3", self.flat_dim * self.hidden),
            ("b3", self.hidden),
            ("W4", self.hidden * self.n_classes),
            ("b4", self.n_classes),
        ]
        out: list[tuple[str, slice, int]] = []
        start = 0
        for name, size in sizes:
            slc = slice(start, start + size)
            out.append((name, slc, size))
            start += size
        return out


LAYOUT = CompactCNNLayout()


def initialize_cnn(
    *, layout: CompactCNNLayout = LAYOUT, seed: int = 7777, scale: float = 0.5
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.zeros(layout.dimension, dtype=np.float32)
    groups = layout.groups()
    fanins = [
        layout.conv1_kernel * layout.conv1_kernel,
        None,
        layout.conv1_out * layout.conv2_kernel * layout.conv2_kernel,
        None,
        layout.flat_dim,
        None,
        layout.hidden,
        None,
    ]
    for gi in [0, 2, 4, 6]:
        _, slc, size = groups[gi]
        fanin = float(fanins[gi])
        w[slc] = rng.normal(0.0, scale / math.sqrt(fanin), size=size).astype(np.float32)
    return w


def _unpack(w: np.ndarray, layout: CompactCNNLayout = LAYOUT):
    g = layout.groups()
    K1 = w[g[0][1]].reshape(layout.conv1_out, 1, layout.conv1_kernel, layout.conv1_kernel)
    b1 = w[g[1][1]]
    K2 = w[g[2][1]].reshape(
        layout.conv2_out, layout.conv1_out, layout.conv2_kernel, layout.conv2_kernel
    )
    b2 = w[g[3][1]]
    W3 = w[g[4][1]].reshape(layout.flat_dim, layout.hidden)
    b3 = w[g[5][1]]
    W4 = w[g[6][1]].reshape(layout.hidden, layout.n_classes)
    b4 = w[g[7][1]]
    return K1, b1, K2, b2, W3, b3, W4, b4


def _conv_forward(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int):
    kh, kw = kernel.shape[-2:]
    patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    patches = patches[:, :, ::stride, ::stride, :, :]
    out = np.einsum("nchwij,ocij->nohw", patches, kernel, optimize=True)
    out += bias[None, :, None, None]
    return out, patches


def _conv_backward(
    dout: np.ndarray,
    x: np.ndarray,
    kernel: np.ndarray,
    patches: np.ndarray,
    stride: int,
):
    dkernel = np.einsum("nchwij,nohw->ocij", patches, dout, optimize=True)
    dbias = np.sum(dout, axis=(0, 2, 3))
    dx = np.zeros_like(x)
    hout, wout = dout.shape[2], dout.shape[3]
    kh, kw = kernel.shape[-2:]
    for i in range(kh):
        for j in range(kw):
            contribution = np.einsum("nohw,oc->nchw", dout, kernel[:, :, i, j], optimize=True)
            dx[:, :, i : i + stride * hout : stride, j : j + stride * wout : stride] += contribution
    return dx, dkernel, dbias


def loss_and_gradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    layout: CompactCNNLayout = LAYOUT,
    regularization: float = 1e-4,
    need_gradient: bool = True,
):
    K1, b1, K2, b2, W3, b3, W4, b4 = _unpack(w, layout)
    x = X.reshape(-1, 1, 28, 28)
    z1, p1 = _conv_forward(x, K1, b1, layout.conv1_stride)
    a1 = np.tanh(z1)
    z2, p2 = _conv_forward(a1, K2, b2, layout.conv2_stride)
    a2 = np.tanh(z2)
    flat = a2.reshape(len(X), -1)
    z3 = flat @ W3 + b3
    a3 = np.tanh(z3)
    logits = a3 @ W4 + b4

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(logits, axis=1)
    predictive = float(np.mean(logsumexp - logits[np.arange(len(y)), y]))
    objective = predictive + 0.5 * regularization * float(w @ w)
    if not need_gradient:
        return objective, predictive

    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    dlogits = probs
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)

    dW4 = a3.T @ dlogits
    db4 = np.sum(dlogits, axis=0)
    dz3 = (dlogits @ W4.T) * (1.0 - a3 * a3)
    dW3 = flat.T @ dz3
    db3 = np.sum(dz3, axis=0)
    da2 = (dz3 @ W3.T).reshape(a2.shape)
    dz2 = da2 * (1.0 - a2 * a2)
    da1, dK2, db2 = _conv_backward(dz2, a1, K2, p2, layout.conv2_stride)
    dz1 = da1 * (1.0 - a1 * a1)
    _, dK1, db1 = _conv_backward(dz1, x, K1, p1, layout.conv1_stride)

    gradient = np.empty_like(w)
    groups = layout.groups()
    gradient[groups[0][1]] = dK1.reshape(-1)
    gradient[groups[1][1]] = db1
    gradient[groups[2][1]] = dK2.reshape(-1)
    gradient[groups[3][1]] = db2
    gradient[groups[4][1]] = dW3.reshape(-1)
    gradient[groups[5][1]] = db3
    gradient[groups[6][1]] = dW4.reshape(-1)
    gradient[groups[7][1]] = db4
    gradient += regularization * w
    return objective, predictive, gradient.astype(np.float32, copy=False)


def predictive_metrics(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    layout: CompactCNNLayout = LAYOUT,
    regularization: float = 1e-4,
):
    K1, b1, K2, b2, W3, b3, W4, b4 = _unpack(w, layout)
    x = X.reshape(-1, 1, 28, 28)
    z1, _ = _conv_forward(x, K1, b1, layout.conv1_stride)
    a1 = np.tanh(z1)
    z2, _ = _conv_forward(a1, K2, b2, layout.conv2_stride)
    a2 = np.tanh(z2)
    a3 = np.tanh(a2.reshape(len(X), -1) @ W3 + b3)
    logits = a3 @ W4 + b4
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(logits, axis=1)
    predictive = float(np.mean(logsumexp - logits[np.arange(len(y)), y]))
    objective = predictive + 0.5 * regularization * float(w @ w)
    prediction = np.argmax(logits, axis=1)
    accuracy = float(np.mean(prediction == y))
    per_class = np.array([np.mean(prediction[y == c] == c) for c in range(10)], dtype=float)
    return objective, predictive, accuracy, float(np.mean(per_class)), float(np.min(per_class)), per_class


def _filter_slices(layout: CompactCNNLayout = LAYOUT):
    groups = layout.groups()
    k1 = groups[0][1]
    k2 = groups[2][1]
    size1 = layout.conv1_kernel * layout.conv1_kernel
    size2 = layout.conv1_out * layout.conv2_kernel * layout.conv2_kernel
    first = [slice(k1.start + f * size1, k1.start + (f + 1) * size1) for f in range(layout.conv1_out)]
    second = [slice(k2.start + f * size2, k2.start + (f + 1) * size2) for f in range(layout.conv2_out)]
    return first, second


def run_federated_cnn(
    *,
    federation: MulticlassFederation,
    method: Literal["events", "ef_topk", "full"],
    n_ticks: int,
    seed: int = 60606,
    layout: CompactCNNLayout = LAYOUT,
    init_scale: float = 0.5,
    batch_size: int = 32,
    regularization: float = 1e-4,
    step: float = 0.02,
    topk_fraction: float = 0.01,
    rho: float = 0.999,
    gamma: float = 1.0,
    threshold: float = 0.025,
    jump0: float = 0.0035,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.1,
    eval_stride: int = 50,
    record_history: bool = False,
):
    rng = np.random.default_rng(seed)
    d = layout.dimension
    address_bits = int(math.ceil(math.log2(d)))
    event_bits = address_bits + 1
    topk = max(1, int(round(topk_fraction * d)))
    w = initialize_cnn(layout=layout, seed=7777, scale=init_scale)
    snapshots = np.repeat(w[None, :], federation.n_clients, axis=0)
    next_completion = federation.periods.copy()
    membrane = np.zeros((federation.n_clients, d), dtype=np.float32)
    residual = np.zeros((federation.n_clients, d), dtype=np.float32)

    payload_bits = 0
    packetized_bits = 0
    messages = 0
    candidate_events = 0
    group_events = np.zeros(len(layout.groups()), dtype=np.int64)
    ever_fired = np.zeros(d, dtype=bool)
    filter1_events = np.zeros(layout.conv1_out, dtype=np.int64)
    filter2_events = np.zeros(layout.conv2_out, dtype=np.int64)
    first_filter_slices, second_filter_slices = _filter_slices(layout)
    history_rows: list[dict[str, float]] = []

    for tick in range(1, n_ticks + 1):
        if method == "events":
            membrane *= rho
        active = [i for i in range(federation.n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]
        for client in active:
            local_n = len(federation.client_y[client])
            ids = rng.integers(0, local_n, size=batch_size)
            _, _, gradient = loss_and_gradient(
                snapshots[client], federation.client_X[client][ids], federation.client_y[client][ids],
                layout=layout, regularization=regularization, need_gradient=True,
            )
            rate_weight = float(federation.weights[client] * federation.periods[client])
            if method == "full":
                w -= step * rate_weight * gradient
                payload_bits += 32 * d
                packetized_bits += 32 * d + 32
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
                    for gi, (_, slc, _) in enumerate(layout.groups()):
                        group_events[gi] += int(np.sum(mask[slc]))
                    for f, slc in enumerate(first_filter_slices):
                        filter1_events[f] += int(np.sum(mask[slc]))
                    for f, slc in enumerate(second_filter_slices):
                        filter2_events[f] += int(np.sum(mask[slc]))
            snapshots[client] = w
            next_completion[client] = tick + federation.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train_obj, train_ce, _, _, _, _ = predictive_metrics(
                w, federation.X_train_eval, federation.y_train_eval,
                layout=layout, regularization=regularization,
            )
            _, test_ce, test_acc, macro_acc, worst_acc, _ = predictive_metrics(
                w, federation.X_test, federation.y_test, layout=layout,
                regularization=regularization,
            )
            history_rows.append({
                "tick": tick, "payload_bits": payload_bits,
                "packetized_bits": packetized_bits, "train_objective": train_obj,
                "train_ce": train_ce, "test_ce": test_ce,
                "test_accuracy": test_acc, "macro_accuracy": macro_acc,
                "worst_class_accuracy": worst_acc,
            })

    history = pd.DataFrame(history_rows)
    final_train_obj, final_train_ce, _, _, _, _ = predictive_metrics(
        w, federation.X_train_eval, federation.y_train_eval,
        layout=layout, regularization=regularization,
    )
    _, final_test_ce, final_test_acc, final_macro, final_worst, per_class = predictive_metrics(
        w, federation.X_test, federation.y_test, layout=layout,
        regularization=regularization,
    )
    result: dict[str, float | int | str] = {
        "method": method, "dimension": d, "n_ticks": n_ticks,
        "init_scale": init_scale, "batch_size": batch_size,
        "regularization": regularization, "step": step,
        "topk_fraction": topk_fraction, "topk": topk,
        "rho": rho, "gamma": gamma, "threshold": threshold, "jump0": jump0,
        "final_train_objective": final_train_obj,
        "final_train_ce": final_train_ce, "final_test_ce": final_test_ce,
        "final_test_accuracy": final_test_acc, "final_macro_accuracy": final_macro,
        "final_worst_class_accuracy": final_worst,
        "whole_train_objective": float(history["train_objective"].mean()),
        "payload_bits": payload_bits, "packetized_bits": packetized_bits,
        "messages": messages, "candidate_events": candidate_events,
        "events_per_message": candidate_events / messages if messages else 0.0,
        "mean_packet_payload_bits": payload_bits / messages if messages else 0.0,
        "ever_fired_fraction": float(np.mean(ever_fired)) if method == "events" else np.nan,
        "parameter_norm": float(np.linalg.norm(w)),
    }
    for cls, acc in enumerate(per_class):
        result[f"class_{cls}_accuracy"] = float(acc)
    if method == "events":
        for gi, (name, slc, n_params) in enumerate(layout.groups()):
            result[f"{name}_events_per_param"] = float(group_events[gi] / n_params)
            result[f"{name}_never_fired"] = float(1.0 - np.mean(ever_fired[slc]))
        size1 = layout.conv1_kernel * layout.conv1_kernel
        size2 = layout.conv1_out * layout.conv2_kernel * layout.conv2_kernel
        for f, slc in enumerate(first_filter_slices):
            result[f"conv1_filter_{f}_events_per_param"] = float(filter1_events[f] / size1)
            result[f"conv1_filter_{f}_never_fired"] = float(1.0 - np.mean(ever_fired[slc]))
        for f, slc in enumerate(second_filter_slices):
            result[f"conv2_filter_{f}_events_per_param"] = float(filter2_events[f] / size2)
            result[f"conv2_filter_{f}_never_fired"] = float(1.0 - np.mean(ever_fired[slc]))
    if record_history:
        result["history"] = history  # type: ignore
    return result
