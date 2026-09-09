from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .cifar10_benchmark import CHANNEL_MEAN, CHANNEL_STD


@dataclass(frozen=True)
class CIFAR10ResNet14Layout:
    """CIFAR ResNet-14 (6n+2, n=2) with GroupNorm."""

    input_channels: int = 3
    base_channels: int = 16
    blocks_per_stage: int = 2
    group_norm_groups: int = 4
    n_classes: int = 10

    def tensor_specs(self) -> list[tuple[str, tuple[int, ...]]]:
        specs: list[tuple[str, tuple[int, ...]]] = [
            ("stem.conv", (16, 3, 3, 3)),
            ("stem.gn_weight", (16,)),
            ("stem.gn_bias", (16,)),
        ]
        channels = (16, 32, 64)
        in_channels = 16
        for stage, out_channels in enumerate(channels, start=1):
            for block in range(1, self.blocks_per_stage + 1):
                stride = 2 if stage > 1 and block == 1 else 1
                prefix = f"stage{stage}.block{block}"
                specs.extend([
                    (f"{prefix}.conv1", (out_channels, in_channels, 3, 3)),
                    (f"{prefix}.gn1_weight", (out_channels,)),
                    (f"{prefix}.gn1_bias", (out_channels,)),
                    (f"{prefix}.conv2", (out_channels, out_channels, 3, 3)),
                    (f"{prefix}.gn2_weight", (out_channels,)),
                    (f"{prefix}.gn2_bias", (out_channels,)),
                ])
                if stride != 1 or in_channels != out_channels:
                    specs.extend([
                        (f"{prefix}.shortcut", (out_channels, in_channels, 1, 1)),
                        (f"{prefix}.shortcut_gn_weight", (out_channels,)),
                        (f"{prefix}.shortcut_gn_bias", (out_channels,)),
                    ])
                in_channels = out_channels
        specs.extend([
            ("head.weight", (64, self.n_classes)),
            ("head.bias", (self.n_classes,)),
        ])
        return specs

    def groups(self) -> list[tuple[str, slice, int]]:
        groups: list[tuple[str, slice, int]] = []
        start = 0
        for name, shape in self.tensor_specs():
            size = int(np.prod(shape))
            groups.append((name, slice(start, start + size), size))
            start += size
        return groups

    @property
    def dimension(self) -> int:
        return sum(size for _, _, size in self.groups())

    def activity_groups(self) -> list[tuple[str, slice, int]]:
        """Contiguous semantic groups used for layer-level event diagnostics."""
        tensor_groups = self.groups()
        labels = ["stem"] + [
            f"stage{stage}.block{block}"
            for stage in range(1, 4)
            for block in range(1, self.blocks_per_stage + 1)
        ] + ["head"]
        result: list[tuple[str, slice, int]] = []
        for label in labels:
            selected = [group for group in tensor_groups if group[0].startswith(label)]
            start = selected[0][1].start
            stop = selected[-1][1].stop
            result.append((label, slice(start, stop), stop - start))
        return result


LAYOUT = CIFAR10ResNet14Layout()


def _views(w: np.ndarray, layout: CIFAR10ResNet14Layout) -> dict[str, np.ndarray]:
    return {
        name: w[slc].reshape(shape)
        for (name, shape), (_, slc, _) in zip(layout.tensor_specs(), layout.groups())
    }


def initialize_resnet14(
    *, layout: CIFAR10ResNet14Layout = LAYOUT, seed: int = 7777,
    scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = np.zeros(layout.dimension, dtype=np.float32)
    views = _views(w, layout)
    for name, shape in layout.tensor_specs():
        value = views[name]
        if name.endswith("conv") or ".conv" in name or name.endswith("shortcut"):
            fan_in = shape[1] * shape[2] * shape[3]
            value[:] = rng.normal(
                0.0, scale * math.sqrt(2.0 / fan_in), size=shape
            ).astype(np.float32)
        elif name.endswith("gn_weight"):
            value.fill(1.0)
        elif name == "head.weight":
            value[:] = rng.normal(
                0.0, scale * math.sqrt(2.0 / shape[0]), size=shape
            ).astype(np.float32)
    return w


def _normalize(X: np.ndarray) -> np.ndarray:
    x = X.astype(np.float32, copy=False) / 255.0
    return (x - CHANNEL_MEAN[None, :, None, None]) / CHANNEL_STD[None, :, None, None]


def _conv_forward(x: np.ndarray, kernel: np.ndarray, stride: int, padding: int):
    if padding:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    else:
        x_padded = x
    kh, kw = kernel.shape[-2:]
    patches = np.lib.stride_tricks.sliding_window_view(
        x_padded, (kh, kw), axis=(2, 3)
    )[:, :, ::stride, ::stride, :, :]
    out = np.einsum("nchwij,ocij->nohw", patches, kernel, optimize=True)
    return out, (x.shape, x_padded, patches, stride, padding)


def _conv_backward(dout: np.ndarray, kernel: np.ndarray, cache):
    input_shape, x_padded, patches, stride, padding = cache
    dkernel = np.einsum("nchwij,nohw->ocij", patches, dout, optimize=True)
    dx_padded = np.zeros_like(x_padded)
    hout, wout = dout.shape[2:]
    kh, kw = kernel.shape[-2:]
    for i in range(kh):
        for j in range(kw):
            contribution = np.einsum(
                "nohw,oc->nchw", dout, kernel[:, :, i, j], optimize=True
            )
            dx_padded[:, :, i:i + stride * hout:stride,
                      j:j + stride * wout:stride] += contribution
    if padding:
        dx = dx_padded[:, :, padding:-padding, padding:-padding]
    else:
        dx = dx_padded
    assert dx.shape == input_shape
    return dx, dkernel


def _group_norm_forward(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, groups: int,
    epsilon: float = 1e-5,
):
    n, channels, height, width = x.shape
    if channels % groups:
        raise ValueError(f"{channels} channels are not divisible by {groups} groups")
    grouped = x.reshape(n, groups, channels // groups, height, width)
    mean = grouped.mean(axis=(2, 3, 4), keepdims=True)
    variance = grouped.var(axis=(2, 3, 4), keepdims=True)
    inverse_std = 1.0 / np.sqrt(variance + epsilon)
    normalized_grouped = (grouped - mean) * inverse_std
    normalized = normalized_grouped.reshape(x.shape)
    out = normalized * weight[None, :, None, None] + bias[None, :, None, None]
    return out, (normalized_grouped, inverse_std, weight, x.shape, groups)


def _group_norm_backward(dout: np.ndarray, cache):
    normalized_grouped, inverse_std, weight, shape, groups = cache
    n, channels, height, width = shape
    dxhat = (dout * weight[None, :, None, None]).reshape(normalized_grouped.shape)
    elements = (channels // groups) * height * width
    sum_dxhat = dxhat.sum(axis=(2, 3, 4), keepdims=True)
    sum_product = (dxhat * normalized_grouped).sum(axis=(2, 3, 4), keepdims=True)
    dx_grouped = (
        inverse_std / elements
        * (elements * dxhat - sum_dxhat - normalized_grouped * sum_product)
    )
    dx = dx_grouped.reshape(shape)
    normalized = normalized_grouped.reshape(shape)
    dweight = np.sum(dout * normalized, axis=(0, 2, 3))
    dbias = np.sum(dout, axis=(0, 2, 3))
    return dx, dweight, dbias


def _block_forward(
    x: np.ndarray, params: dict[str, np.ndarray], prefix: str,
    stride: int, groups: int,
):
    z1, conv1_cache = _conv_forward(x, params[f"{prefix}.conv1"], stride, 1)
    n1, gn1_cache = _group_norm_forward(
        z1, params[f"{prefix}.gn1_weight"], params[f"{prefix}.gn1_bias"], groups
    )
    a1 = np.maximum(n1, 0.0)
    z2, conv2_cache = _conv_forward(a1, params[f"{prefix}.conv2"], 1, 1)
    n2, gn2_cache = _group_norm_forward(
        z2, params[f"{prefix}.gn2_weight"], params[f"{prefix}.gn2_bias"], groups
    )
    shortcut_cache = None
    if f"{prefix}.shortcut" in params:
        shortcut, shortcut_conv_cache = _conv_forward(
            x, params[f"{prefix}.shortcut"], stride, 0
        )
        shortcut, shortcut_gn_cache = _group_norm_forward(
            shortcut,
            params[f"{prefix}.shortcut_gn_weight"],
            params[f"{prefix}.shortcut_gn_bias"],
            groups,
        )
        shortcut_cache = (shortcut_conv_cache, shortcut_gn_cache)
    else:
        shortcut = x
    pre_activation = n2 + shortcut
    out = np.maximum(pre_activation, 0.0)
    cache = (
        x, n1, a1, pre_activation, conv1_cache, gn1_cache,
        conv2_cache, gn2_cache, shortcut_cache,
    )
    return out, cache


def _block_backward(
    dout: np.ndarray, params: dict[str, np.ndarray], prefix: str, cache,
):
    x, n1, a1, pre_activation, conv1_cache, gn1_cache, conv2_cache, gn2_cache, shortcut_cache = cache
    dpre = dout * (pre_activation > 0.0)
    dz2, dgn2_weight, dgn2_bias = _group_norm_backward(dpre, gn2_cache)
    da1, dconv2 = _conv_backward(dz2, params[f"{prefix}.conv2"], conv2_cache)
    dn1 = da1 * (n1 > 0.0)
    dz1, dgn1_weight, dgn1_bias = _group_norm_backward(dn1, gn1_cache)
    dx_main, dconv1 = _conv_backward(dz1, params[f"{prefix}.conv1"], conv1_cache)
    grads = {
        f"{prefix}.conv1": dconv1,
        f"{prefix}.gn1_weight": dgn1_weight,
        f"{prefix}.gn1_bias": dgn1_bias,
        f"{prefix}.conv2": dconv2,
        f"{prefix}.gn2_weight": dgn2_weight,
        f"{prefix}.gn2_bias": dgn2_bias,
    }
    if shortcut_cache is None:
        dx_shortcut = dpre
    else:
        shortcut_conv_cache, shortcut_gn_cache = shortcut_cache
        dshortcut, dshortcut_weight, dshortcut_bias = _group_norm_backward(
            dpre, shortcut_gn_cache
        )
        dx_shortcut, dshortcut_kernel = _conv_backward(
            dshortcut, params[f"{prefix}.shortcut"], shortcut_conv_cache
        )
        grads.update({
            f"{prefix}.shortcut": dshortcut_kernel,
            f"{prefix}.shortcut_gn_weight": dshortcut_weight,
            f"{prefix}.shortcut_gn_bias": dshortcut_bias,
        })
    return dx_main + dx_shortcut, grads


def _forward(w: np.ndarray, X: np.ndarray, layout: CIFAR10ResNet14Layout):
    params = _views(w, layout)
    x = _normalize(X)
    stem_z, stem_conv_cache = _conv_forward(x, params["stem.conv"], 1, 1)
    stem_n, stem_gn_cache = _group_norm_forward(
        stem_z, params["stem.gn_weight"], params["stem.gn_bias"],
        layout.group_norm_groups,
    )
    activation = np.maximum(stem_n, 0.0)
    block_caches = []
    for stage, channels in enumerate((16, 32, 64), start=1):
        for block in range(1, layout.blocks_per_stage + 1):
            prefix = f"stage{stage}.block{block}"
            stride = 2 if stage > 1 and block == 1 else 1
            activation, cache = _block_forward(
                activation, params, prefix, stride, layout.group_norm_groups
            )
            block_caches.append((prefix, cache))
    pooled = activation.mean(axis=(2, 3))
    logits = pooled @ params["head.weight"] + params["head.bias"]
    cache = (
        params, stem_n, stem_conv_cache, stem_gn_cache,
        block_caches, activation.shape, pooled,
    )
    return cache, logits


def loss_and_gradient(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, *,
    layout: CIFAR10ResNet14Layout = LAYOUT,
    regularization: float = 5e-4, need_gradient: bool = True,
):
    cache, logits = _forward(w, X, layout)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + np.max(logits, axis=1)
    predictive = float(np.mean(logsumexp - logits[np.arange(len(y)), y]))
    objective = predictive + 0.5 * regularization * float(w @ w)
    if not need_gradient:
        return objective, predictive

    params, stem_n, stem_conv_cache, stem_gn_cache, block_caches, final_shape, pooled = cache
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=1, keepdims=True)
    probs[np.arange(len(y)), y] -= 1.0
    probs /= len(y)
    grads: dict[str, np.ndarray] = {
        "head.weight": pooled.T @ probs,
        "head.bias": probs.sum(axis=0),
    }
    dpooled = probs @ params["head.weight"].T
    height, width = final_shape[2:]
    dactivation = np.broadcast_to(
        dpooled[:, :, None, None] / (height * width), final_shape
    ).copy()
    for prefix, block_cache in reversed(block_caches):
        dactivation, block_grads = _block_backward(
            dactivation, params, prefix, block_cache
        )
        grads.update(block_grads)
    dstem_n = dactivation * (stem_n > 0.0)
    dstem_z, grads["stem.gn_weight"], grads["stem.gn_bias"] = _group_norm_backward(
        dstem_n, stem_gn_cache
    )
    _, grads["stem.conv"] = _conv_backward(
        dstem_z, params["stem.conv"], stem_conv_cache
    )

    gradient = np.empty_like(w)
    for (name, _), (_, slc, _) in zip(layout.tensor_specs(), layout.groups()):
        gradient[slc] = grads[name].reshape(-1)
    gradient += regularization * w
    return objective, predictive, gradient.astype(np.float32, copy=False)


def predictive_metrics(
    w: np.ndarray, X: np.ndarray, y: np.ndarray, *,
    layout: CIFAR10ResNet14Layout = LAYOUT,
    regularization: float = 5e-4, batch_size: int = 256,
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
    return (
        float(objective), float(predictive), float(total_correct / len(y)),
        float(np.mean(per_class)), float(np.min(per_class)), per_class.astype(float),
    )
