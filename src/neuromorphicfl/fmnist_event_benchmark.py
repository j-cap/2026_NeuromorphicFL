from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import gzip
import hashlib
import math
import urllib.request

import numpy as np
import pandas as pd


PERIODS = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

FMNIST_FILES = {
    "train-images-idx3-ubyte.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    ),
    "train-labels-idx1-ubyte.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "25c81989df183df01b3e8a0aad5dffbe",
    ),
    "t10k-images-idx3-ubyte.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "bef4ecab320f06d8554ea6380940ec79",
    ),
    "t10k-labels-idx1-ubyte.gz": (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
        "bb300cfdad3c16e7a12a480ee83cd310",
    ),
}


@dataclass(frozen=True)
class BinaryFederation:
    client_X: tuple[np.ndarray, ...]
    client_y: tuple[np.ndarray, ...]
    X_train_eval: np.ndarray
    y_train_eval: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_pair: tuple[int, int]
    regime: str
    client_positive_rates: np.ndarray
    periods: np.ndarray
    weights: np.ndarray

    @property
    def n_clients(self) -> int:
        return len(self.client_X)


@dataclass(frozen=True)
class MLPLayout:
    widths: tuple[int, ...] = (784, 32, 16, 1)

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


LAYOUT = MLPLayout()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_fashion_mnist(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for filename, (url, checksum) in FMNIST_FILES.items():
        target = root / filename
        if not target.exists() or _md5(target) != checksum:
            if target.exists():
                target.unlink()
            urllib.request.urlretrieve(url, target)
        observed = _md5(target)
        if observed != checksum:
            raise RuntimeError(
                f"checksum mismatch for {filename}: {observed} != {checksum}"
            )
    return root


def _load_idx(root: Path, kind: Literal["train", "t10k"]):
    with gzip.open(root / f"{kind}-labels-idx1-ubyte.gz", "rb") as handle:
        labels = np.frombuffer(handle.read(), dtype=np.uint8, offset=8)
    with gzip.open(root / f"{kind}-images-idx3-ubyte.gz", "rb") as handle:
        images = np.frombuffer(handle.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )
    return images, labels


def load_fashion_mnist(root: str | Path):
    root = ensure_fashion_mnist(root)
    return (*_load_idx(root, "train"), *_load_idx(root, "t10k"))


def _rates_for_regime(regime: str) -> np.ndarray:
    if regime == "iid":
        return np.full(10, 0.50)
    if regime == "moderate":
        return np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75])
    if regime == "strong":
        return np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90, 0.95])
    raise ValueError(f"unknown regime {regime}")


def make_binary_federation(
    *,
    root: str | Path,
    class_pair: tuple[int, int] = (2, 4),
    regime: str = "strong",
    seed: int = 1400,
    train_eval_size: int = 2000,
) -> BinaryFederation:
    train_images, train_labels, test_images, test_labels = load_fashion_mnist(root)
    class0, class1 = class_pair
    train0 = np.flatnonzero(train_labels == class0)
    train1 = np.flatnonzero(train_labels == class1)
    test0 = np.flatnonzero(test_labels == class0)
    test1 = np.flatnonzero(test_labels == class1)
    if len(train0) != 6000 or len(train1) != 6000:
        raise RuntimeError("unexpected Fashion-MNIST training class count")
    if len(test0) != 1000 or len(test1) != 1000:
        raise RuntimeError("unexpected Fashion-MNIST test class count")

    rng = np.random.default_rng(seed)
    rng.shuffle(train0)
    rng.shuffle(train1)
    rates = _rates_for_regime(regime)
    n_per_client = 1200
    positive_counts = np.rint(rates * n_per_client).astype(int)
    if int(positive_counts.sum()) != len(train1):
        raise RuntimeError("client rate design must preserve global class balance")
    negative_counts = n_per_client - positive_counts
    if int(negative_counts.sum()) != len(train0):
        raise RuntimeError("client rate design must preserve global class balance")

    clients_X: list[np.ndarray] = []
    clients_y: list[np.ndarray] = []
    start0 = 0
    start1 = 0
    global_indices: list[np.ndarray] = []
    for client in range(10):
        idx0 = train0[start0 : start0 + negative_counts[client]]
        idx1 = train1[start1 : start1 + positive_counts[client]]
        start0 += negative_counts[client]
        start1 += positive_counts[client]
        ids = np.concatenate([idx0, idx1])
        labels = np.concatenate(
            [np.zeros(len(idx0), dtype=np.float32), np.ones(len(idx1), dtype=np.float32)]
        )
        order = rng.permutation(len(ids))
        ids = ids[order]
        labels = labels[order]
        X = train_images[ids].astype(np.float32) / 127.5 - 1.0
        clients_X.append(X)
        clients_y.append(labels)
        global_indices.append(ids)

    all_ids = np.concatenate(global_indices)
    all_y = (train_labels[all_ids] == class1).astype(np.float32)
    eval_ids = rng.choice(len(all_ids), size=min(train_eval_size, len(all_ids)), replace=False)
    X_train_eval = train_images[all_ids[eval_ids]].astype(np.float32) / 127.5 - 1.0
    y_train_eval = all_y[eval_ids]

    test_ids = np.concatenate([test0, test1])
    y_test = (test_labels[test_ids] == class1).astype(np.float32)
    test_order = rng.permutation(len(test_ids))
    test_ids = test_ids[test_order]
    y_test = y_test[test_order]
    X_test = test_images[test_ids].astype(np.float32) / 127.5 - 1.0

    return BinaryFederation(
        client_X=tuple(clients_X),
        client_y=tuple(clients_y),
        X_train_eval=X_train_eval,
        y_train_eval=y_train_eval,
        X_test=X_test,
        y_test=y_test,
        class_pair=class_pair,
        regime=regime,
        client_positive_rates=rates,
        periods=PERIODS.copy(),
        weights=np.full(10, 0.1),
    )


def federation_audit(federation: BinaryFederation) -> pd.DataFrame:
    rows = []
    for client in range(federation.n_clients):
        rows.append(
            {
                "client": client,
                "period": int(federation.periods[client]),
                "n_train": int(len(federation.client_y[client])),
                "positive_rate": float(np.mean(federation.client_y[client])),
                "target_positive_rate": float(federation.client_positive_rates[client]),
            }
        )
    return pd.DataFrame(rows)


def initialize_mlp(
    *,
    layout: MLPLayout = LAYOUT,
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


def _unpack(w: np.ndarray, layout: MLPLayout = LAYOUT):
    layers = []
    groups = layout.groups()
    for layer in range(layout.n_layers):
        n_in, n_out = layout.widths[layer], layout.widths[layer + 1]
        W = w[groups[2 * layer][1]].reshape(n_in, n_out)
        b = w[groups[2 * layer + 1][1]]
        layers.append((W, b))
    return layers


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def loss_and_gradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    layout: MLPLayout = LAYOUT,
    regularization: float = 1e-4,
    need_gradient: bool = True,
):
    layers = _unpack(w, layout)
    activations: list[np.ndarray] = [X]
    a = X
    for layer, (W, b) in enumerate(layers):
        z = a @ W + b
        a = np.tanh(z) if layer < layout.n_layers - 1 else z
        activations.append(a)
    logits = activations[-1][:, 0]
    predictive = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
    objective = predictive + 0.5 * regularization * float(w @ w)
    if not need_gradient:
        return objective, predictive

    delta = ((_sigmoid(logits) - y) / len(y))[:, None]
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
    layout: MLPLayout = LAYOUT,
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
    a = X
    for layer, (W, b) in enumerate(_unpack(w, layout)):
        z = a @ W + b
        a = np.tanh(z) if layer < layout.n_layers - 1 else z
    logits = a[:, 0]
    accuracy = float(np.mean((logits >= 0.0) == y))
    return objective, predictive, accuracy


def run_federated_method(
    *,
    federation: BinaryFederation,
    method: Literal["events", "ef_topk", "full"],
    n_ticks: int,
    seed: int = 60606,
    layout: MLPLayout = LAYOUT,
    init_scale: float = 0.5,
    batch_size: int = 32,
    regularization: float = 1e-4,
    step: float = 0.02,
    topk_fraction: float = 0.01,
    rho: float = 0.999,
    gamma: float = 0.2,
    threshold: float = 0.025,
    jump0: float = 0.005,
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
            train_obj, train_ce, _ = predictive_metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=regularization,
            )
            _, test_ce, test_acc = predictive_metrics(
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
                }
            )

    history = pd.DataFrame(history_rows)
    final_train_obj, final_train_ce, _ = predictive_metrics(
        w,
        np.concatenate(federation.client_X, axis=0),
        np.concatenate(federation.client_y, axis=0),
        layout=layout,
        regularization=regularization,
    )
    _, final_test_ce, final_test_acc = predictive_metrics(
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
    if method == "events":
        for group_id, (name, slc, n_params) in enumerate(layout.groups()):
            result[f"{name}_events_per_param"] = float(group_events[group_id] / n_params)
            result[f"{name}_never_fired"] = float(1.0 - np.mean(ever_fired[slc]))
    if record_history:
        result["history"] = history  # type: ignore
    return result
