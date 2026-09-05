from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import pickle
import tarfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import MulticlassFederation


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CHANNEL_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CHANNEL_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


@dataclass(frozen=True)
class CompactCIFAR10CNNLayout:
    input_channels: int = 3
    input_hw: int = 32
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
        return (self.input_hw - self.conv1_kernel) // self.conv1_stride + 1

    @property
    def conv2_hw(self) -> int:
        return (self.conv1_hw - self.conv2_kernel) // self.conv2_stride + 1

    @property
    def flat_dim(self) -> int:
        return self.conv2_out * self.conv2_hw * self.conv2_hw

    @property
    def dimension(self) -> int:
        return int(sum(size for _, _, size in self.groups()))

    def groups(self) -> list[tuple[str, slice, int]]:
        sizes = [
            ("K1", self.conv1_out * self.input_channels * self.conv1_kernel**2),
            ("b1", self.conv1_out),
            ("K2", self.conv2_out * self.conv1_out * self.conv2_kernel**2),
            ("b2", self.conv2_out),
            ("W3", self.flat_dim * self.hidden),
            ("b3", self.hidden),
            ("W4", self.hidden * self.n_classes),
            ("b4", self.n_classes),
        ]
        groups: list[tuple[str, slice, int]] = []
        start = 0
        for name, size in sizes:
            slc = slice(start, start + size)
            groups.append((name, slc, size))
            start += size
        return groups


LAYOUT = CompactCIFAR10CNNLayout()


def initialize_cnn(
    *, layout: CompactCIFAR10CNNLayout = LAYOUT, seed: int = 7777,
    scale: float = 1.0,
) -> np.ndarray:
    """He initialization for the ReLU compact CNN."""
    rng = np.random.default_rng(seed)
    w = np.zeros(layout.dimension, dtype=np.float32)
    groups = layout.groups()
    fanins = [
        layout.input_channels * layout.conv1_kernel**2,
        None,
        layout.conv1_out * layout.conv2_kernel**2,
        None,
        layout.flat_dim,
        None,
        layout.hidden,
        None,
    ]
    for group_index in (0, 2, 4, 6):
        _, slc, size = groups[group_index]
        std = scale * math.sqrt(2.0 / float(fanins[group_index]))
        w[slc] = rng.normal(0.0, std, size=size).astype(np.float32)
    return w


def _unpack(w: np.ndarray, layout: CompactCIFAR10CNNLayout = LAYOUT):
    groups = layout.groups()
    K1 = w[groups[0][1]].reshape(
        layout.conv1_out, layout.input_channels,
        layout.conv1_kernel, layout.conv1_kernel,
    )
    b1 = w[groups[1][1]]
    K2 = w[groups[2][1]].reshape(
        layout.conv2_out, layout.conv1_out,
        layout.conv2_kernel, layout.conv2_kernel,
    )
    b2 = w[groups[3][1]]
    W3 = w[groups[4][1]].reshape(layout.flat_dim, layout.hidden)
    b3 = w[groups[5][1]]
    W4 = w[groups[6][1]].reshape(layout.hidden, layout.n_classes)
    b4 = w[groups[7][1]]
    return K1, b1, K2, b2, W3, b3, W4, b4


def _normalize(X: np.ndarray) -> np.ndarray:
    x = X.astype(np.float32, copy=False) / 255.0
    return (x - CHANNEL_MEAN[None, :, None, None]) / CHANNEL_STD[None, :, None, None]


def _conv_forward(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray, stride: int):
    kh, kw = kernel.shape[-2:]
    patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    patches = patches[:, :, ::stride, ::stride, :, :]
    out = np.einsum("nchwij,ocij->nohw", patches, kernel, optimize=True)
    out += bias[None, :, None, None]
    return out, patches


def _conv_backward(
    dout: np.ndarray, x: np.ndarray, kernel: np.ndarray,
    patches: np.ndarray, stride: int,
):
    dkernel = np.einsum("nchwij,nohw->ocij", patches, dout, optimize=True)
    dbias = np.sum(dout, axis=(0, 2, 3))
    dx = np.zeros_like(x)
    hout, wout = dout.shape[2:]
    kh, kw = kernel.shape[-2:]
    for i in range(kh):
        for j in range(kw):
            contribution = np.einsum(
                "nohw,oc->nchw", dout, kernel[:, :, i, j], optimize=True
            )
            dx[:, :, i : i + stride * hout : stride,
               j : j + stride * wout : stride] += contribution
    return dx, dkernel, dbias


def _forward(w: np.ndarray, X: np.ndarray, layout: CompactCIFAR10CNNLayout):
    K1, b1, K2, b2, W3, b3, W4, b4 = _unpack(w, layout)
    x = _normalize(X)
    z1, p1 = _conv_forward(x, K1, b1, layout.conv1_stride)
    a1 = np.maximum(z1, 0.0)
    z2, p2 = _conv_forward(a1, K2, b2, layout.conv2_stride)
    a2 = np.maximum(z2, 0.0)
    flat = a2.reshape(len(X), -1)
    z3 = flat @ W3 + b3
    a3 = np.maximum(z3, 0.0)
    logits = a3 @ W4 + b4
    return (x, z1, p1, a1, z2, p2, a2, flat, z3, a3), logits


def loss_and_gradient(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, *,
    layout: CompactCIFAR10CNNLayout = LAYOUT,
    regularization: float = 5e-4, need_gradient: bool = True,
):
    cache, logits = _forward(w, X, layout)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(logits, axis=1)
    predictive = float(np.mean(logsumexp - logits[np.arange(len(y)), y]))
    objective = predictive + 0.5 * regularization * float(w @ w)
    if not need_gradient:
        return objective, predictive

    x, z1, p1, a1, z2, p2, a2, flat, z3, a3 = cache
    K1, _, K2, _, W3, _, W4, _ = _unpack(w, layout)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    probs /= len(y)
    dW4 = a3.T @ probs
    db4 = np.sum(probs, axis=0)
    dz3 = (probs @ W4.T) * (z3 > 0.0)
    dW3 = flat.T @ dz3
    db3 = np.sum(dz3, axis=0)
    dz2 = (dz3 @ W3.T).reshape(a2.shape) * (z2 > 0.0)
    da1, dK2, db2 = _conv_backward(dz2, a1, K2, p2, layout.conv2_stride)
    dz1 = da1 * (z1 > 0.0)
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
    w: np.ndarray, X: np.ndarray, y: np.ndarray, *,
    layout: CompactCIFAR10CNNLayout = LAYOUT,
    regularization: float = 5e-4, batch_size: int = 512,
):
    total_loss = 0.0
    total_correct = 0
    class_correct = np.zeros(layout.n_classes, dtype=np.int64)
    class_count = np.zeros(layout.n_classes, dtype=np.int64)
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        _, logits = _forward(w, X[start:stop], layout)
        labels = y[start:stop]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        total_loss += float(np.sum(
            np.log(np.sum(np.exp(shifted), axis=1))
            + np.max(logits, axis=1) - logits[np.arange(len(labels)), labels]
        ))
        prediction = np.argmax(logits, axis=1)
        total_correct += int(np.sum(prediction == labels))
        for cls in range(layout.n_classes):
            mask = labels == cls
            class_count[cls] += int(np.sum(mask))
            class_correct[cls] += int(np.sum(prediction[mask] == cls))
    predictive = total_loss / len(y)
    objective = predictive + 0.5 * regularization * float(w @ w)
    per_class = class_correct / np.maximum(class_count, 1)
    accuracy = total_correct / len(y)
    return (
        float(objective), float(predictive), float(accuracy),
        float(np.mean(per_class)), float(np.min(per_class)), per_class.astype(float),
    )


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(destination)


def download_cifar10(root: str | Path) -> Path:
    root = Path(root)
    extracted = root / "cifar-10-batches-py"
    if (extracted / "data_batch_1").exists():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        with urlopen(CIFAR10_URL) as response, archive_path.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    digest = hashlib.md5(archive_path.read_bytes()).hexdigest()  # official checksum
    if digest != CIFAR10_MD5:
        raise RuntimeError(f"CIFAR-10 checksum mismatch: {digest}")
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, root)
    return extracted


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(root: str | Path) -> tuple[np.ndarray, ...]:
    data_root = download_cifar10(root)
    train = [_read_batch(data_root / f"data_batch_{index}") for index in range(1, 6)]
    train_X = np.concatenate([part[0] for part in train])
    train_y = np.concatenate([part[1] for part in train])
    test_X, test_y = _read_batch(data_root / "test_batch")
    return train_X, train_y, test_X, test_y


def class_count_matrix(regime: str = "strong") -> np.ndarray:
    if regime == "iid":
        return np.full((10, 10), 500, dtype=int)
    if regime == "strong":
        counts = np.full((10, 10), 250, dtype=int)
        np.fill_diagonal(counts, 2750)
        return counts
    raise ValueError(f"unknown CIFAR-10 regime {regime}")


def make_cifar10_federation(
    *, root: str | Path, regime: str = "strong", seed: int = 3400,
    train_eval_size: int = 5000,
) -> MulticlassFederation:
    train_X, train_y, test_X, test_y = load_cifar10(root)
    counts = class_count_matrix(regime)
    rng = np.random.default_rng(seed)
    per_class = []
    for cls in range(10):
        ids = np.flatnonzero(train_y == cls)
        if len(ids) != 5000:
            raise RuntimeError(f"unexpected CIFAR-10 class count for class {cls}")
        rng.shuffle(ids)
        per_class.append(ids)
    offsets = np.zeros(10, dtype=int)
    client_X, client_y, global_ids = [], [], []
    for client in range(10):
        parts = []
        for cls in range(10):
            take = int(counts[client, cls])
            parts.append(per_class[cls][offsets[cls] : offsets[cls] + take])
            offsets[cls] += take
        ids = np.concatenate(parts)
        ids = ids[rng.permutation(len(ids))]
        client_X.append(train_X[ids])
        client_y.append(train_y[ids])
        global_ids.append(ids)
    if not np.all(offsets == 5000):
        raise RuntimeError("partition did not consume every CIFAR-10 training sample")
    all_ids = np.concatenate(global_ids)
    eval_ids = rng.choice(all_ids, size=min(train_eval_size, len(all_ids)), replace=False)
    test_order = rng.permutation(len(test_y))
    return MulticlassFederation(
        client_X=tuple(client_X), client_y=tuple(client_y),
        X_train_eval=train_X[eval_ids], y_train_eval=train_y[eval_ids],
        X_test=test_X[test_order], y_test=test_y[test_order],
        regime=regime, client_class_counts=counts,
        periods=np.ones(10, dtype=int), weights=np.full(10, 0.1),
    )


def federation_audit(federation: MulticlassFederation) -> pd.DataFrame:
    rows = []
    for client in range(federation.n_clients):
        row = {
            "client": client,
            "n_train": int(len(federation.client_y[client])),
            "dominant_class": int(np.argmax(federation.client_class_counts[client])),
            "dominant_fraction": float(
                np.max(federation.client_class_counts[client])
                / np.sum(federation.client_class_counts[client])
            ),
        }
        row.update({
            f"class_{cls}_count": int(federation.client_class_counts[client, cls])
            for cls in range(10)
        })
        rows.append(row)
    return pd.DataFrame(rows)
