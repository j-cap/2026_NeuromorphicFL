from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torchvision.datasets import CIFAR10


CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CHANNEL_STD = (0.2470, 0.2435, 0.2616)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(4, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(4, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = functional.relu(self.gn1(self.conv1(inputs)))
        hidden = self.gn2(self.conv2(hidden))
        return functional.relu(hidden + self.shortcut(inputs))


class CIFARResNet14(nn.Module):
    """CIFAR ResNet-14 with GroupNorm and no convolutional biases."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=False),
        )
        self.stage1 = nn.Sequential(BasicBlock(16, 16, 1), BasicBlock(16, 16, 1))
        self.stage2 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32, 32, 1))
        self.stage3 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64, 1))
        self.head = nn.Linear(64, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(inputs)
        hidden = self.stage1(hidden)
        hidden = self.stage2(hidden)
        hidden = self.stage3(hidden)
        hidden = hidden.mean(dim=(2, 3))
        return self.head(hidden)


@dataclass(frozen=True)
class TorchFederation:
    client_images: tuple[torch.Tensor, ...]
    client_labels: tuple[torch.Tensor, ...]
    train_eval_images: torch.Tensor
    train_eval_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    weights: torch.Tensor
    client_class_counts: np.ndarray

    @property
    def n_clients(self) -> int:
        return len(self.client_labels)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def semantic_parameter_groups(model: nn.Module) -> list[tuple[str, slice, int]]:
    labels = ["stem"] + [
        f"stage{stage}.block{block}"
        for stage in range(1, 4) for block in range(1, 3)
    ] + ["head"]
    converted_prefixes = {
        "stem": "stem.",
        "stage1.block1": "stage1.0.",
        "stage1.block2": "stage1.1.",
        "stage2.block1": "stage2.0.",
        "stage2.block2": "stage2.1.",
        "stage3.block1": "stage3.0.",
        "stage3.block2": "stage3.1.",
        "head": "head.",
    }
    named = list(model.named_parameters())
    result: list[tuple[str, slice, int]] = []
    offset = 0
    parameter_offsets: dict[str, tuple[int, int]] = {}
    for name, parameter in named:
        parameter_offsets[name] = (offset, offset + parameter.numel())
        offset += parameter.numel()
    for label in labels:
        selected = [
            parameter_offsets[name] for name, _ in named
            if name.startswith(converted_prefixes[label])
        ]
        start, stop = selected[0][0], selected[-1][1]
        result.append((label, slice(start, stop), stop - start))
    if sum(size for _, _, size in result) != parameter_count(model):
        raise RuntimeError("semantic parameter groups do not cover the model")
    return result


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CHANNEL_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CHANNEL_STD, device=images.device).view(1, 3, 1, 1)
    return (images.float().div_(255.0) - mean) / std


def class_count_matrix() -> np.ndarray:
    counts = np.full((10, 10), 250, dtype=np.int64)
    np.fill_diagonal(counts, 2750)
    return counts


def load_federation(
    root: str | Path, partition_seed: int, device: torch.device,
    train_eval_size: int = 5000,
) -> TorchFederation:
    train = CIFAR10(root=str(root), train=True, download=True)
    test = CIFAR10(root=str(root), train=False, download=True)
    train_images = np.asarray(train.data).transpose(0, 3, 1, 2).copy()
    train_labels = np.asarray(train.targets, dtype=np.int64)
    test_images = np.asarray(test.data).transpose(0, 3, 1, 2).copy()
    test_labels = np.asarray(test.targets, dtype=np.int64)
    counts = class_count_matrix()
    rng = np.random.default_rng(partition_seed)
    per_class: list[np.ndarray] = []
    for class_index in range(10):
        indices = np.flatnonzero(train_labels == class_index)
        rng.shuffle(indices)
        per_class.append(indices)
    offsets = np.zeros(10, dtype=np.int64)
    client_indices: list[np.ndarray] = []
    for client in range(10):
        parts = []
        for class_index in range(10):
            count = int(counts[client, class_index])
            parts.append(per_class[class_index][offsets[class_index]:offsets[class_index] + count])
            offsets[class_index] += count
        indices = np.concatenate(parts)
        client_indices.append(indices[rng.permutation(len(indices))])
    if not np.all(offsets == 5000):
        raise RuntimeError("partition did not consume all training examples")
    all_indices = np.concatenate(client_indices)
    eval_indices = rng.choice(all_indices, size=train_eval_size, replace=False)
    test_order = rng.permutation(len(test_labels))

    def image_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values).to(device=device, non_blocking=True)

    def label_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(values).to(device=device, non_blocking=True)

    return TorchFederation(
        client_images=tuple(image_tensor(train_images[ids]) for ids in client_indices),
        client_labels=tuple(label_tensor(train_labels[ids]) for ids in client_indices),
        train_eval_images=image_tensor(train_images[eval_indices]),
        train_eval_labels=label_tensor(train_labels[eval_indices]),
        test_images=image_tensor(test_images[test_order]),
        test_labels=label_tensor(test_labels[test_order]),
        weights=torch.full((10,), 0.1, dtype=torch.float32, device=device),
        client_class_counts=counts,
    )
