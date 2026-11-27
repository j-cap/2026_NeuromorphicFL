"""Shared raw-image mmaps and exact fifty-client CIFAR-10 allocations."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
from neuromorphicfl.cifar10_benchmark import download_cifar10, _read_batch
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation

DEV_SEED, SPLIT_SEED = 13000, 23000
SEEDS = tuple(range(13100, 14100, 100))
ACTIVE = (50, 10, 5)
ROUNDS, PILOT_ROUNDS, EVAL_STRIDE = 1800, 300, 60
TEXT_SUFFIXES = {".py", ".csv", ".json", ".md", ".txt", ".tex", ".bib"}
CORE_PROTOCOL = REPO / "experiments/results/core_fresh_comparison_v1/protocol.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path, table):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    try:
        table.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime():
    output = io.StringIO()
    with redirect_stdout(output):
        np.show_config()
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "numpy_build_sha256": hashlib.sha256(output.getvalue().encode()).hexdigest(),
        "platform": platform.system(), "machine": platform.machine(), "blas_threads_per_worker": 1}


def prepare_cache(root, *, include_test=False):
    """Pilot parses only training batches; the CIFAR archive contains both sets."""
    root = Path(root)
    extracted = download_cifar10(root)
    cache = root / "participation_calibrated_cache_v1"
    cache.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CORE_PROTOCOL.read_text())["data_sha256"]
    manifests = {}
    for scope in (("train", "test") if include_test else ("train",)):
        names = [f"data_batch_{i}" for i in range(1, 6)] if scope == "train" else ["test_batch"]
        raw = {name: file_hash(extracted/name) for name in names}
        if any(raw[name] != expected[name] for name in names):
            raise RuntimeError("Official CIFAR batch differs from frozen core evidence")
        path = cache/f"{scope}_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest["raw_sha256"] != raw:
                raise RuntimeError("CIFAR cache provenance differs")
            for name, sha in manifest["cache_sha256"].items():
                if not (cache/name).exists() or file_hash(cache/name) != sha:
                    raise RuntimeError(f"Corrupt mmap cache: {name}")
            manifests[scope] = manifest
            continue
        parts = [_read_batch(extracted/name) for name in names]
        images = np.concatenate([a for a, _ in parts])
        labels = np.concatenate([b for _, b in parts])
        count = 50000 if scope == "train" else 10000
        if (images.shape != (count, 3, 32, 32) or images.dtype != np.uint8
                or labels.shape != (count,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10))):
            raise RuntimeError("CIFAR image/label layout differs")
        hashes = {}
        for kind, values in (("images", images), ("labels", labels)):
            target = cache/f"{scope}_{kind}.npy"
            temporary = target.with_suffix(f".tmp-{os.getpid()}")
            try:
                with temporary.open("wb") as f:
                    np.save(f, values, allow_pickle=False)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            hashes[target.name] = file_hash(target)
        manifest = {"scope": scope, "examples": count, "layout": "uint8_NCHW_frozen_forward_normalizes",
            "raw_sha256": raw, "cache_sha256": hashes}
        atomic_json(path, manifest, immutable=True)
        manifests[scope] = manifest
    return cache, manifests


def open_arrays(cache, *, include_test):
    keys = ["train_images", "train_labels"]+(["test_images", "test_labels"] if include_test else [])
    return {key: np.load(Path(cache)/f"{key}.npy", mmap_mode="r", allow_pickle=False) for key in keys}


class IndexedImages:
    def __init__(self, images, indices):
        self.images, self.indices = images, np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), *images.shape[1:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.images[self.indices[key]]


def allocation(labels, *, seed, validation):
    labels = np.asarray(labels)
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected the balanced original CIFAR training labels")
    rng, split_rng = np.random.default_rng(seed), np.random.default_rng(SPLIT_SEED)
    pools, heldout = [], []
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        if validation:
            split_rng.shuffle(ids)
            heldout.extend(ids[:500])
            ids = ids[500:].copy()
        rng.shuffle(ids)
        pools.append(ids)
    size = 900 if validation else 1000
    counts = np.full((50, 10), size//20, dtype=np.int64)
    counts[np.arange(50), np.arange(50) % 10] = 11*size//20
    offsets, clients = np.zeros(10, dtype=np.int64), []
    for client in range(50):
        ids = []
        for cls in range(10):
            take = int(counts[client, cls])
            ids.extend(pools[cls][offsets[cls]:offsets[cls]+take])
            offsets[cls] += take
        ids = np.asarray(ids, dtype=np.int64)
        clients.append(ids[rng.permutation(size)])
    all_train = np.concatenate(clients)
    heldout = np.asarray(heldout, dtype=np.int64)
    diagnostic = rng.choice(all_train, size=5000, replace=False)
    if not np.array_equal(np.sort(np.r_[all_train, heldout]), np.arange(50000)):
        raise AssertionError("Allocation duplicates or drops training data")
    h = lambda a: hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()
    manifest = {"partition_seed": int(seed), "split_seed": SPLIT_SEED if validation else None,
        "training_examples": len(all_train), "validation_examples": len(heldout),
        "examples_per_client": size, "class_counts": counts.tolist(),
        "client_indices_sha256": [h(ids) for ids in clients],
        "heldout_indices_sha256": h(heldout), "diagnostic_indices_sha256": h(diagnostic)}
    return clients, heldout, diagnostic, manifest


def task_data(arrays, *, seed, validation):
    clients, heldout, diagnostic, manifest = allocation(arrays["train_labels"], seed=seed, validation=validation)
    X, y = arrays["train_images"], arrays["train_labels"]
    X_eval = IndexedImages(X, heldout) if validation else arrays["test_images"]
    y_eval = y[heldout] if validation else arrays["test_labels"]
    federation = MulticlassFederation(tuple(IndexedImages(X, ids) for ids in clients),
        tuple(y[ids] for ids in clients), IndexedImages(X, diagnostic), y[diagnostic],
        X_eval, y_eval, "strong", np.asarray(manifest["class_counts"]),
        np.ones(50, dtype=int), np.full(50, 1/50))
    return federation, manifest


def schedule_for(seed, active_clients, *, rounds=ROUNDS):
    if active_clients not in ACTIVE or rounds <= 0:
        raise ValueError("Unknown participation fraction or horizon")
    # The same permutation stream couples fractions: 5 selected clients are
    # a subset of 10, and those of 50. Sorting fixes aggregation/client order.
    rng = np.random.default_rng(200000+int(seed))
    return np.asarray([np.sort(rng.permutation(50)[:active_clients]) for _ in range(rounds)], dtype=np.int16)


def schedule_manifest(seed):
    schedules = {str(k): schedule_for(seed, k) for k in ACTIVE}
    return {"algorithm": "nested_uniform_permutation_sorted_v1", "selection_seed": 200000+int(seed),
        "rounds": ROUNDS, "sha256": {k: hashlib.sha256(s.astype("<i2").tobytes()).hexdigest() for k, s in schedules.items()},
        "participation_counts": {k: np.bincount(s.ravel(), minlength=50).tolist() for k, s in schedules.items()}}
