"""Explicit training-only development and fresh full-training ResNet partitions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import _read_batch, class_count_matrix, download_cifar10

DEV_SEED, SPLIT_SEED = 7000, 17000
CONFIRM_SEEDS = (7100, 7200, 7300)


def ids_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def allocation(labels, *, seed, validation):
    labels = np.asarray(labels)
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected the balanced 50000-image CIFAR training set")
    split_rng, rng = np.random.default_rng(SPLIT_SEED), np.random.default_rng(seed)
    pools, holdout = [], []
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        if validation:
            split_rng.shuffle(ids)
            holdout.extend(ids[:500])
            ids = ids[500:].copy()
        rng.shuffle(ids)
        pools.append(ids)
    counts = class_count_matrix("strong", n_clients=10) * (9 if validation else 10) // 10
    clients, offsets = [], np.zeros(10, dtype=int)
    for client in range(10):
        pieces = []
        for cls in range(10):
            size = int(counts[client, cls])
            pieces.append(pools[cls][offsets[cls]:offsets[cls]+size])
            offsets[cls] += size
        ids = np.concatenate(pieces)
        clients.append(ids[rng.permutation(len(ids))])
    holdout = np.asarray(holdout, dtype=np.int64)
    if validation:
        holdout = holdout[split_rng.permutation(len(holdout))]
    diagnostic = rng.choice(np.concatenate(clients), size=5000, replace=False)
    table = pd.DataFrame({"global_index": np.arange(50000), "class": labels, "split": "train",
                          "client_id": -1, "local_order": -1, "diagnostic_order": -1})
    if validation:
        table.loc[holdout, "split"] = "validation"
        table.loc[holdout, "local_order"] = np.arange(len(holdout))
    for client, ids in enumerate(clients):
        table.loc[ids, "client_id"] = client
        table.loc[ids, "local_order"] = np.arange(len(ids))
    table.loc[diagnostic, "diagnostic_order"] = np.arange(5000)
    if (len(np.unique(np.concatenate([*clients, holdout]))) != 50000
            or sum(map(len, clients)) != (45000 if validation else 50000)
            or len(np.intersect1d(diagnostic, holdout))):
        raise AssertionError("Invalid train/validation/diagnostic allocation")
    manifest = {"partition_seed": seed, "split_seed": SPLIT_SEED if validation else None,
                "training_examples": sum(map(len, clients)), "validation_examples": len(holdout),
                "client_indices_sha256": [ids_hash(ids) for ids in clients], "holdout_indices_sha256": ids_hash(holdout),
                "diagnostic_indices_sha256": ids_hash(diagnostic), "class_counts": counts.tolist(),
                "index_csv_sha256": hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()}
    return tuple(clients), holdout, diagnostic, table, manifest


def load_arrays(root, *, include_test=False):
    extracted = download_cifar10(root)
    batches = [_read_batch(extracted / f"data_batch_{i}") for i in range(1, 6)]
    images, labels = np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches])
    test = _read_batch(extracted / "test_batch") if include_test else None
    return images, labels, test


@dataclass(frozen=True)
class Federation:
    client_images: tuple
    client_labels: tuple
    train_eval_images: object
    train_eval_labels: object
    evaluation_images: object
    evaluation_labels: object
    weights: object
    scope: str
    manifest: dict

    @property
    def n_clients(self):
        return len(self.client_labels)


def federation(images, labels, *, seed, validation, device, test=None):
    import torch
    clients, holdout, diagnostic, table, manifest = allocation(labels, seed=seed, validation=validation)
    if validation:
        if test is not None:
            raise ValueError("Official test arrays are prohibited during development")
        evaluation = images[holdout], labels[holdout]
    else:
        if test is None or len(test[1]) != 10000:
            raise ValueError("Confirmation requires the official 10000-image test set")
        evaluation = test
    convert = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    fed = Federation(tuple(convert(images[ids]) for ids in clients), tuple(convert(labels[ids]) for ids in clients),
                     convert(images[diagnostic]), convert(labels[diagnostic]), convert(evaluation[0]), convert(evaluation[1]),
                     torch.full((10,), .1, dtype=torch.float32, device=device), "validation" if validation else "test", manifest)
    return fed, table
