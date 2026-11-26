"""Read-only shared Fashion-MNIST cache and deterministic client allocations."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import urllib.request

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from neuromorphicfl.fmnist_event_benchmark import FMNIST_FILES

DEV_SEED, SPLIT_SEED, DEV_TRAIN_SEED = 11000, 21000, 101000
SEEDS = tuple(range(11100, 12100, 100))
TRAIN_SEEDS = {seed: 90000 + seed for seed in SEEDS}
TEXT_SUFFIXES = {".py", ".csv", ".json", ".md", ".txt", ".tex", ".bib"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    h = hashlib.sha256()
    if path.suffix.lower() in TEXT_SUFFIXES:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_raw(root, scope):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = [n for n in FMNIST_FILES if n.startswith("train" if scope == "train" else "t10k")]
    for name in names:
        target = root / name
        _, expected = FMNIST_FILES[name]
        if target.exists() and _md5(target) == expected:
            continue
        urls = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name,
                FMNIST_FILES[name][0]]
        errors = []
        temporary = target.with_suffix(target.suffix + f".download-{os.getpid()}")
        for url in urls:
            try:
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as f:
                    while block := response.read(1 << 20):
                        f.write(block)
                if _md5(temporary) != expected:
                    raise RuntimeError("Official dataset checksum differs")
                os.replace(temporary, target)
                break
            except Exception as exc:
                errors.append(str(exc))
                temporary.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Cannot download {name}: {errors}")
    return names


def prepare_cache(root, *, include_test=False):
    """Development touches only the two training files, even if test exists."""
    root = Path(root)
    cache = root / "mlp_calibrated_cache_v1"
    cache.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for scope in (("train", "test") if include_test else ("train",)):
        names = ensure_raw(root, scope)
        raw = {n: file_hash(root / n) for n in names}
        path = cache / f"{scope}_manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest["raw_sha256"] != raw:
                raise RuntimeError("Cached Fashion-MNIST provenance changed")
            for name, sha in manifest["cache_sha256"].items():
                if not (cache / name).exists() or file_hash(cache / name) != sha:
                    raise RuntimeError(f"Corrupt shared cache: {cache / name}")
            manifests[scope] = manifest
            continue
        prefix, count = ("train", 60000) if scope == "train" else ("t10k", 10000)
        with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
            blob = f.read()
        if struct.unpack(">4I", blob[:16]) != (2051, count, 28, 28) or len(blob) != 16 + count*784:
            raise ValueError("Invalid Fashion-MNIST image IDX")
        images = np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(count, 784).astype(np.float32)/127.5 - 1.0
        with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
            labels_blob = f.read()
        if struct.unpack(">2I", labels_blob[:8]) != (2049, count) or len(labels_blob) != count+8:
            raise ValueError("Invalid Fashion-MNIST label IDX")
        labels = np.frombuffer(labels_blob, dtype=np.uint8, offset=8).astype(np.int64)
        if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10)):
            raise ValueError("Dataset class counts differ")
        hashes = {}
        for kind, array in (("images", images), ("labels", labels)):
            destination = cache / f"{scope}_{kind}.npy"
            temporary = destination.with_suffix(f".tmp-{os.getpid()}")
            with temporary.open("wb") as f:
                np.save(f, array, allow_pickle=False)
            os.replace(temporary, destination)
            hashes[destination.name] = file_hash(destination)
        manifest = {"scope": scope, "examples": count, "normalization": "float32(raw)/127.5-1", "raw_sha256": raw, "cache_sha256": hashes}
        atomic_json(path, manifest, immutable=True)
        manifests[scope] = manifest
    return cache, manifests


class IndexedImages:
    """Materialize only requested minibatches; share the full image mmap."""
    def __init__(self, images, indices):
        self.images, self.indices = images, np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), images.shape[1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.images[self.indices[key]]


def allocation(labels, *, seed, validation):
    labels = np.asarray(labels)
    if labels.shape != (60000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 6000)):
        raise ValueError("Expected balanced original Fashion-MNIST training labels")
    partition_rng, split_rng = np.random.default_rng(seed), np.random.default_rng(SPLIT_SEED)
    pools, heldout = [], []
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        if validation:
            split_rng.shuffle(ids)
            heldout.extend(ids[:600])
            ids = ids[600:].copy()
        partition_rng.shuffle(ids)
        pools.append(ids)
    size = 5400 if validation else 6000
    counts = np.full((10, 10), size//20, dtype=np.int64)
    np.fill_diagonal(counts, size*11//20)
    offsets, clients = np.zeros(10, dtype=np.int64), []
    for client in range(10):
        pieces = []
        for cls in range(10):
            n = int(counts[client, cls])
            pieces.append(pools[cls][offsets[cls]:offsets[cls]+n])
            offsets[cls] += n
        ids = np.concatenate(pieces)
        clients.append(ids[partition_rng.permutation(size)])
    all_train = np.concatenate(clients)
    heldout = np.asarray(heldout, dtype=np.int64)
    diagnostic = partition_rng.choice(all_train, size=5000, replace=False)
    combined = np.concatenate((all_train, heldout))
    if not np.array_equal(np.sort(combined), np.arange(60000)):
        raise AssertionError("Client allocation duplicates or drops examples")
    def ids_hash(ids):
        return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    manifest = {"partition_seed": seed, "split_seed": SPLIT_SEED if validation else None,
                "training_examples": len(all_train), "validation_examples": len(heldout),
                "examples_per_client": size, "class_counts": counts.tolist(),
                "client_indices_sha256": [ids_hash(i) for i in clients],
                "heldout_indices_sha256": ids_hash(heldout), "diagnostic_indices_sha256": ids_hash(diagnostic)}
    return clients, heldout, diagnostic, manifest


def open_arrays(cache, *, include_test):
    cache = Path(cache)
    result = {name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("train_images", "train_labels")}
    if include_test:
        result.update({name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                       for name in ("test_images", "test_labels")})
    return result


def task_data(arrays, *, seed, validation):
    clients, heldout, diagnostic, manifest = allocation(arrays["train_labels"], seed=seed, validation=validation)
    images, labels = arrays["train_images"], arrays["train_labels"]
    if validation:
        X_holdout, y_holdout = IndexedImages(images, heldout), labels[heldout]
    else:
        X_holdout, y_holdout = arrays["test_images"], arrays["test_labels"]
    return {"client_X": tuple(IndexedImages(images, i) for i in clients), "client_y": tuple(labels[i] for i in clients),
            "X_diagnostic": IndexedImages(images, diagnostic), "y_diagnostic": labels[diagnostic],
            "X_holdout": X_holdout, "y_holdout": y_holdout, "scope": "validation" if validation else "test",
            "manifest": manifest}
