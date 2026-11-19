"""Training-only CIFAR split and atomic validation checkpoints for Ström tuning."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
import event_codec_full_horizon as full
import strom_codec_host as strom
import numpy as np
import pandas as pd

from neuromorphicfl.cifar10_benchmark import _read_batch, class_count_matrix, download_cifar10
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation

PARTITION_SEED, SPLIT_SEED, TRAIN_SEED = 4500, 64500, 84500
VALIDATION_PER_CLASS, TRAIN_DIAGNOSTICS = 500, 5000
VALIDATION_FIELDS = ("validation_ce", "validation_accuracy", "validation_macro_accuracy", "validation_worst_class_accuracy")


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def split_spec():
    return {"partition_seed": PARTITION_SEED, "split_seed": SPLIT_SEED,
            "validation_per_class": VALIDATION_PER_CLASS, "training_examples": 45000,
            "validation_examples": 5000, "clients": 10, "examples_per_client": 4500,
            "dominant_class_fraction": .55, "training_diagnostic_examples": TRAIN_DIAGNOSTICS,
            "official_test_batch": "never_parsed_or_evaluated", "official_test_slot": "empty"}


def split_indices(labels):
    labels = np.asarray(labels)
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 5000)):
        raise ValueError("Expected the original balanced 50000 CIFAR training labels")
    split_rng, partition_rng = np.random.default_rng(SPLIT_SEED), np.random.default_rng(PARTITION_SEED)
    pools, validation = [], []
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        split_rng.shuffle(ids)
        validation.extend(ids[:VALIDATION_PER_CLASS])
        remaining = ids[VALIDATION_PER_CLASS:].copy()
        partition_rng.shuffle(remaining)
        pools.append(remaining)
    validation = np.asarray(validation, dtype=np.int64)
    validation = validation[split_rng.permutation(len(validation))]
    counts = class_count_matrix("strong", n_clients=10) * 9 // 10
    offsets = np.zeros(10, dtype=int)
    clients = []
    for client in range(10):
        pieces = []
        for cls in range(10):
            count = int(counts[client, cls])
            pieces.append(pools[cls][offsets[cls]:offsets[cls] + count])
            offsets[cls] += count
        ids = np.concatenate(pieces)
        clients.append(ids[partition_rng.permutation(len(ids))])
    diagnostic = partition_rng.choice(np.concatenate(clients), size=TRAIN_DIAGNOSTICS, replace=False)
    table = pd.DataFrame({"global_index": np.arange(50000), "class": labels, "split": "train",
                          "client_id": -1, "local_order": -1, "training_diagnostic_order": -1})
    table.loc[validation, "split"] = "validation"
    table.loc[validation, "local_order"] = np.arange(len(validation))
    for client, ids in enumerate(clients):
        table.loc[ids, "client_id"] = client
        table.loc[ids, "local_order"] = np.arange(len(ids))
    table.loc[diagnostic, "training_diagnostic_order"] = np.arange(len(diagnostic))
    validate_split(table)
    return table


def ordered_ids(table):
    clients = [table[(table.split == "train") & (table.client_id == client)].sort_values("local_order").global_index.to_numpy(dtype=np.int64)
               for client in range(10)]
    validation = table[table.split == "validation"].sort_values("local_order").global_index.to_numpy(dtype=np.int64)
    diagnostic = table[table.training_diagnostic_order >= 0].sort_values("training_diagnostic_order").global_index.to_numpy(dtype=np.int64)
    return clients, validation, diagnostic


def validate_split(table):
    expected = ["global_index", "class", "split", "client_id", "local_order", "training_diagnostic_order"]
    if list(table.columns) != expected or table.isna().any().any() or not np.array_equal(table.global_index, np.arange(50000)):
        raise ValueError("Invalid development split index coverage/schema")
    if not table.split.isin(("train", "validation")).all() or not table["class"].isin(range(10)).all():
        raise ValueError("Invalid development labels/split names")
    train, validation = table[table.split == "train"], table[table.split == "validation"]
    if len(train) != 45000 or len(validation) != 5000 or not validation.client_id.eq(-1).all():
        raise ValueError("Training/validation sizes or identities differ")
    if not train.client_id.isin(range(10)).all() or not np.array_equal(np.sort(validation.local_order), np.arange(5000)):
        raise ValueError("Invalid development client/validation ordering")
    if not np.array_equal(validation.groupby("class").size().reindex(range(10)), np.full(10, 500)):
        raise ValueError("Validation class balance differs")
    expected_counts = class_count_matrix("strong", n_clients=10) * 9 // 10
    actual_counts = pd.crosstab(train.client_id, train["class"]).reindex(index=range(10), columns=range(10), fill_value=0)
    if not np.array_equal(actual_counts, expected_counts):
        raise ValueError("Training federation class allocation differs")
    for _, group in train.groupby("client_id"):
        if not np.array_equal(np.sort(group.local_order), np.arange(4500)):
            raise ValueError("Training client ordering differs")
    diagnostic = table[table.training_diagnostic_order >= 0]
    if (not np.array_equal(np.sort(diagnostic.training_diagnostic_order), np.arange(TRAIN_DIAGNOSTICS))
            or not diagnostic.split.eq("train").all()
            or not table.training_diagnostic_order.ge(-1).all()):
        raise ValueError("Training diagnostics overlap validation or have invalid ordering")


def split_manifest(table):
    validate_split(table)
    clients, validation, diagnostic = ordered_ids(table)
    serialized = table.to_csv(index=False, lineterminator="\n")
    return {"specification": split_spec(), "index_csv_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "ordered_client_indices_sha256": [ids_hash(ids) for ids in clients],
            "ordered_validation_indices_sha256": ids_hash(validation),
            "ordered_training_diagnostic_indices_sha256": ids_hash(diagnostic)}


@dataclass(frozen=True)
class DevelopmentData:
    federation: MulticlassFederation
    validation_X: np.ndarray
    validation_y: np.ndarray
    indices: pd.DataFrame
    manifest: dict


def federation_from_training(images, labels):
    indices = split_indices(labels)
    clients, validation, diagnostic = ordered_ids(indices)
    counts = class_count_matrix("strong", n_clients=10) * 9 // 10
    # Frozen hosts need only local data and a training-diagnostic slot. Keep
    # their official-test slots empty so accidental test evaluation fails.
    federation = MulticlassFederation(
        client_X=tuple(images[ids] for ids in clients), client_y=tuple(labels[ids] for ids in clients),
        X_train_eval=images[diagnostic], y_train_eval=labels[diagnostic],
        X_test=np.empty((0, *images.shape[1:]), dtype=images.dtype), y_test=np.empty(0, dtype=np.int64),
        regime="strong_training_only_development", client_class_counts=counts,
        periods=np.ones(10, dtype=int), weights=np.full(10, .1))
    return DevelopmentData(federation, images[validation], labels[validation], indices, split_manifest(indices))


def load_data(root):
    extracted = download_cifar10(root)
    batches = [_read_batch(extracted / f"data_batch_{index}") for index in range(1, 6)]
    images = np.concatenate([part[0] for part in batches])
    labels = np.concatenate([part[1] for part in batches])
    return federation_from_training(images, labels)


class ValidationCheckpoint:
    """Save validation metrics atomically with host model/RNG/residual state."""
    def __init__(self, path, provenance, *, data, architecture, config):
        self.base = full.BaselineCheckpoint(path, provenance)
        self.data, self.config = data, config
        self.layout, _, _, self.metrics = full._architecture_ops(architecture)
        self.rows = []

    def load(self, identity):
        saved = self.base.load(identity)
        if saved is None:
            self.rows = []
            return None
        progress = saved["progress"]
        self.rows = progress.get("validation_history", [])
        expected = [r for r in full.sampled_rounds(self.config) if r <= progress["completed_round"]]
        if [r["round"] for r in self.rows] != expected:
            raise ValueError("Validation checkpoint coverage differs")
        records = {r["round"]: r for r in progress["records"]}
        for row in self.rows:
            if (row["model_sha256"] != records[row["round"]]["model_sha256"]
                    or not all(np.isfinite(row[k]) for k in VALIDATION_FIELDS)):
                raise ValueError("Invalid validation checkpoint model/metrics")
        return saved

    def save(self, identity, arrays, progress):
        rnd = int(progress["completed_round"])
        rows = self.rows
        if rnd in full.sampled_rounds(self.config):
            if rows and rows[-1]["round"] >= rnd:
                raise ValueError("Duplicate validation checkpoint")
            model_hash = hashlib.sha256(arrays["model"].astype("<f4").tobytes()).hexdigest()
            if model_hash != progress["records"][-1]["model_sha256"]:
                raise AssertionError("Validation/model fingerprint differs")
            _, ce, accuracy, macro, worst, _ = self.metrics(
                arrays["model"], self.data.validation_X, self.data.validation_y,
                layout=self.layout, regularization=self.config.regularization)
            values = (ce, accuracy, macro, worst)
            if not all(np.isfinite(value) for value in values):
                raise FloatingPointError("Nonfinite validation metrics")
            rows = [*rows, {"round": rnd, **dict(zip(VALIDATION_FIELDS, map(float, values))), "model_sha256": model_hash}]
        self.base.save(identity, arrays, {**progress, "validation_history": rows})
        self.rows = rows


def evaluated_run(*, data, method, config, seed, checkpoint_path, provenance, horizon, progress_callback=None):
    if method not in ("event", "strom") or not 1 <= horizon <= config.rounds:
        raise ValueError("Invalid validation host or requested horizon")
    checkpoint = ValidationCheckpoint(checkpoint_path, provenance, data=data, architecture="cifar_cnn", config=config)
    fn = strom.run_strom_codec if method == "strom" else full.run_codec_pilot
    kwargs = dict(federation=data.federation, architecture="cifar_cnn", config=config, seed=seed,
                  checkpoint=checkpoint)
    if method == "event":
        kwargs.update(method="event", checkpoint_stride=config.eval_stride)

    def progress(row):
        if progress_callback is not None:
            progress_callback(row, checkpoint.rows[-1] if checkpoint.rows else {})

    kwargs["progress_callback"] = progress
    if not checkpoint_path.exists():
        result = fn(**kwargs, stop_after=1)
        if horizon == 1:
            return {**result, "evaluation": result["history"].merge(pd.DataFrame(checkpoint.rows), on="round", validate="one_to_one")}
    result = fn(**kwargs, stop_after=horizon)
    return {**result, "evaluation": result["history"].merge(pd.DataFrame(checkpoint.rows), on="round", validate="one_to_one")}


def trajectory_tables(result, *, method, horizon, n_clients, dimension):
    rounds = result["rounds"][result["rounds"]["round"] <= horizon].copy()
    evaluation = result["evaluation"][result["evaluation"]["round"] <= horizon].copy()
    account = strom.traffic_summary if method == "strom" else full.traffic_summary
    histories = []
    for _, row in evaluation.iterrows():
        table = account(rounds[rounds["round"] <= int(row["round"])], n_clients=n_clients, dimension=dimension)
        for field, value in row.items():
            table[field] = value
        histories.append(table)
    summary = account(rounds, n_clients=n_clients, dimension=dimension)
    for field in ("train_objective", "train_ce", *VALIDATION_FIELDS):
        summary["final_" + field] = float(evaluation.iloc[-1][field])
    summary["model_sha256"] = evaluation.iloc[-1].model_sha256
    return {"rounds": rounds, "evaluation": evaluation, "summary": summary,
            "history": pd.concat(histories, ignore_index=True)}
