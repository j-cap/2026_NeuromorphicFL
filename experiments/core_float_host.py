"""Lossless wire representations for the unchanged dense/EF-TopK FedAvg rules.

This experiment-local host leaves every frozen package source untouched.
Float32 magnitudes, the native TopK selection/reset, client order and dense
gain schedule are preserved. EF-TopK receives support coding and an exact
aggregate/replay downlink portfolio; no extra lossy compressor is introduced.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import struct
import time

import numpy as np
import pandas as pd

import event_codec_full_horizon as full
from neuromorphicfl.event_codec import encode_pulses, decode_pulses
from neuromorphicfl.event_codec_pilot import _same_bits
from neuromorphicfl.final_baseline_campaign import _architecture_ops, _local_delta

MAX_DIMENSION = 1 << 24
DENSE = struct.Struct(">4sI")
SPARSE = struct.Struct(">4sII")
LENGTH = struct.Struct(">I")


def encode_vector(vector, *, dense_only=False):
    vector = np.asarray(vector)
    if vector.dtype != np.float32 or vector.ndim != 1 or not 0 < vector.size <= MAX_DIMENSION or not np.isfinite(vector).all():
        raise ValueError("Expected a finite nonempty float32 vector")
    dense = DENSE.pack(b"F32D", vector.size) + vector.astype("<f4").tobytes()
    if dense_only:
        return dense, "dense_float32"
    # Include negative zero as an explicitly transmitted bit pattern.
    support = vector.view(np.uint32) != 0
    positions = encode_pulses(support.astype(np.int8))
    sparse = SPARSE.pack(b"F32S", vector.size, len(positions.data)) + positions.data + vector[support].astype("<f4").tobytes()
    return (sparse, "support_" + positions.mode) if len(sparse) < len(dense) else (dense, "dense_float32")


def decode_vector(data, *, expected_dimension):
    if len(data) < DENSE.size:
        raise ValueError("Truncated float32 frame")
    magic, dimension = DENSE.unpack_from(data)
    if not 0 < dimension <= MAX_DIMENSION or dimension != expected_dimension:
        raise ValueError("Float32 frame dimension differs")
    if magic == b"F32D":
        if len(data) != DENSE.size + 4 * dimension:
            raise ValueError("Dense float32 frame length differs")
        result = np.frombuffer(data, dtype="<f4", offset=DENSE.size).copy()
    elif magic == b"F32S" and len(data) >= SPARSE.size:
        _, _, size = SPARSE.unpack_from(data)
        if not 0 < size <= len(data) - SPARSE.size:
            raise ValueError("Invalid support frame length")
        support = decode_pulses(data[SPARSE.size:SPARSE.size + size], expected_dimension=dimension)
        if not np.isin(support, [0, 1]).all():
            raise ValueError("Float32 support must be binary")
        count = int(np.count_nonzero(support))
        if len(data) != SPARSE.size + size + 4 * count:
            raise ValueError("Sparse float32 payload length differs")
        result = np.zeros(dimension, dtype=np.float32)
        result[support.astype(bool)] = np.frombuffer(data, dtype="<f4", offset=SPARSE.size + size)
    else:
        raise ValueError("Unknown float32 frame")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite decoded float32 payload")
    return result


def downlink(aggregate, packets, *, gain):
    encoded, vector_mode = encode_vector(aggregate)
    dimension = aggregate.size
    replay = SPARSE.pack(b"F32R", dimension, len(packets)) + b"".join(LENGTH.pack(len(p)) + p for p in packets)
    data, mode = (encoded, "aggregate_" + vector_mode) if len(encoded) <= len(replay) else (replay, "ordered_replay")
    decoded = decode_downlink(data, expected_dimension=dimension, expected_clients=len(packets), gain=gain)
    _same_bits(aggregate, decoded)
    return data, decoded, mode, 8 * len(encoded), 8 * len(replay)


def decode_downlink(data, *, expected_dimension, expected_clients, gain):
    if data[:4] != b"F32R":
        return decode_vector(data, expected_dimension=expected_dimension)
    if len(data) < SPARSE.size:
        raise ValueError("Truncated ordered replay")
    _, dimension, count = SPARSE.unpack_from(data)
    if dimension != expected_dimension or count != expected_clients or count <= 0:
        raise ValueError("Ordered replay metadata differs")
    result, offset = np.zeros(dimension, dtype=np.float32), SPARSE.size
    for _ in range(count):
        if offset + LENGTH.size > len(data):
            raise ValueError("Truncated replay length")
        size, = LENGTH.unpack_from(data, offset)
        offset += LENGTH.size
        if not 0 < size <= len(data) - offset:
            raise ValueError("Invalid replay frame length")
        result += decode_vector(data[offset:offset + size], expected_dimension=dimension)
        offset += size
    if offset != len(data):
        raise ValueError("Trailing ordered replay bytes")
    result *= gain
    return result


def traffic_summary(rounds, *, n_clients, dimension):
    method = rounds.iloc[0].method
    variants = (("dense_float32", "codec_uplink_bits", "exact_downlink_bits", False),) if method == "dense" else (
        ("native_ef_topk", "legacy_uplink_bits", "legacy_server_bits", True),
        ("lossless_ef_topk", "codec_uplink_bits", "exact_downlink_bits", False))
    checkpoint, rows = 32 * dimension + 64, []
    for variant, up, down, fallback in variants:
        upload = int(rounds[up].sum())
        stream = int(np.minimum(rounds[down], checkpoint).sum() if fallback else rounds[down].sum())
        unicast = n_clients * (checkpoint + 32 * len(rounds) + stream)
        rows.append({"variant": variant, "rounds": len(rounds), "uplink_bits": upload, "server_stream_bits": stream,
                     "unicast_downlink_bits": unicast, "unicast_total_bits": upload + unicast,
                     "broadcast_downlink_bits": checkpoint + stream, "broadcast_total_bits": upload + checkpoint + stream,
                     "roundtrip_verified": True})
    table = pd.DataFrame(rows)
    table["unicast_saving_percent"] = 100 * (1 - table.unicast_total_bits / table.iloc[0].unicast_total_bits)
    return table


def run(*, federation, architecture, method, config, seed, checkpoint, progress_callback=None, stop_after=None):
    if method not in ("dense", "ef_topk") or (method != "dense" and config.server_gain_exponent != 0):
        raise ValueError("Only native dense schedules and EF-TopK are supported")
    if (config.rounds <= 0 or config.eval_stride <= 0 or config.local_steps <= 0 or config.batch_size <= 0
            or not np.isfinite([config.server_gain, config.server_gain_exponent, config.server_gain_scale,
                               config.local_lr, config.regularization, config.topk_fraction]).all()
            or config.server_gain <= 0 or config.server_gain_exponent < 0 or config.server_gain_scale <= 0
            or config.local_lr <= 0 or config.regularization < 0 or not 0 < config.topk_fraction <= 1
            or (stop_after is not None and not 1 <= stop_after <= config.rounds)):
        raise ValueError("Invalid dense/EF-TopK configuration")
    layout, initialize, loss_grad, metrics = _architecture_ops(architecture)
    dimension, n = int(layout.dimension), federation.n_clients
    weights = federation.weights.astype(float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    w = initialize(layout=layout, seed=7777, scale=config.init_scale)
    client_model = w.copy()
    residual = np.zeros((n, dimension), dtype=np.float32)
    arrays = {"model": w, "client_model": client_model, "residual": residual}
    identity = {"host": "core_float_host_v1", "method": method, "architecture": architecture, "config": asdict(config),
                "seed": seed, "n_clients": n, "dimension": dimension, "weights": weights.tolist(), "numpy_version": np.__version__}
    records, history, start = [], [], 0
    saved = checkpoint.load(identity)
    if saved is not None:
        if set(saved["arrays"]) != set(arrays):
            raise ValueError("Dense/EF-TopK checkpoint arrays differ")
        for name, target in arrays.items():
            source = saved["arrays"][name]
            if source.dtype != target.dtype or source.shape != target.shape or not np.isfinite(source).all():
                raise ValueError("Invalid dense/EF-TopK checkpoint state")
            target[:] = source
        progress = saved["progress"]
        records, history, start = progress["records"], progress["history"], int(progress["completed_round"])
        if not 0 <= start <= config.rounds or [r["round"] for r in records] != list(range(1, start + 1)):
            raise ValueError("Invalid dense/EF-TopK checkpoint progress")
        rng.bit_generator.state = progress["rng_state"]
        _same_bits(w, client_model)
    topk = max(1, int(round(config.topk_fraction * dimension)))
    for rnd in range(start + 1, config.rounds + 1):
        deltas = [_local_delta(w, federation.client_X[i], federation.client_y[i], config=config, rng=rng,
                               layout=layout, loss_grad=loss_grad) for i in range(n)]
        if not all(np.isfinite(delta).all() for delta in deltas):
            raise FloatingPointError("Nonfinite dense/EF-TopK local delta")
        started = time.perf_counter()
        aggregate, packets, modes = np.zeros(dimension, dtype=np.float32), [], Counter()
        legacy_up = 0
        for i, delta in enumerate(deltas):
            if method == "dense":
                packet, mode = encode_vector(delta, dense_only=True)
                decoded = decode_vector(packet, expected_dimension=dimension)
                _same_bits(delta, decoded)
                aggregate += float(weights[i]) * decoded
                legacy_up += 32 * dimension + 64
            else:
                residual[i] += float(weights[i]) * delta
                ids = np.argpartition(np.abs(residual[i]), -topk)[-topk:]
                message = np.zeros(dimension, dtype=np.float32)
                message[ids] = residual[i, ids]
                packet, mode = encode_vector(message)
                decoded = decode_vector(packet, expected_dimension=dimension)
                _same_bits(message, decoded)
                aggregate[ids] += decoded[ids]
                residual[i, ids] = 0.0
                legacy_up += topk * (32 + (dimension - 1).bit_length()) + 64
            packets.append(packet)
            modes[mode] += 1
        gain = float(config.server_gain) * (1 + rnd / float(config.server_gain_scale)) ** (-float(config.server_gain_exponent))
        aggregate *= gain
        if method == "dense":
            data, mode = encode_vector(aggregate, dense_only=True)
            decoded = decode_vector(data, expected_dimension=dimension)
            _same_bits(aggregate, decoded)
            aggregate_bits, replay_bits = len(data) * 8, 0
        else:
            data, decoded, mode, aggregate_bits, replay_bits = downlink(aggregate, packets, gain=gain)
        w += aggregate
        client_model += decoded
        _same_bits(w, client_model)
        if not np.isfinite(w).all():
            raise FloatingPointError("Nonfinite dense/EF-TopK model")
        records.append({"round": rnd, "method": method, "coordinate_events": n * topk if method == "ef_topk" else 0,
                        "legacy_uplink_bits": legacy_up, "codec_uplink_bits": sum(8 * len(p) for p in packets),
                        "legacy_server_bits": legacy_up, "exact_downlink_bits": 8 * len(data),
                        "aggregate_downlink_bits": aggregate_bits, "ordered_downlink_bits": replay_bits,
                        "active_clients": n, "downlink_mode": mode, "uplink_modes": json.dumps(dict(modes), sort_keys=True),
                        "codec_audit_seconds": time.perf_counter() - started,
                        "model_sha256": hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()})
        if rnd in full.sampled_rounds(config):
            objective, ce, *_ = metrics(w, federation.X_train_eval, federation.y_train_eval,
                                       layout=layout, regularization=config.regularization)
            if not np.isfinite([objective, ce]).all():
                raise FloatingPointError("Nonfinite training diagnostic")
            history.append({"round": rnd, "train_objective": float(objective), "train_ce": float(ce)})
        if rnd % config.eval_stride == 0 or rnd == config.rounds or rnd == stop_after:
            checkpoint.save(identity, arrays, {"completed_round": rnd, "records": records, "history": history,
                                                "rng_state": rng.bit_generator.state})
        if progress_callback is not None:
            progress_callback(records[-1])
        if rnd == stop_after:
            break
    frame = pd.DataFrame(records)
    return {"rounds": frame, "history": pd.DataFrame(history), "model": w.copy(), "residual": residual.copy(),
            "rng_state": rng.bit_generator.state, "summary": traffic_summary(frame, n_clients=n, dimension=dimension)}


def evaluated_run(*, federation, architecture, method, config, seed, checkpoint_path, provenance,
                  progress_callback=None, stop_after=None):
    checkpoint = full.PredictiveCheckpoint(checkpoint_path, provenance, federation=federation, architecture=architecture, config=config)
    def progress(row):
        if progress_callback is not None:
            progress_callback(row, checkpoint.evaluations[-1] if checkpoint.evaluations else {})
    kwargs = dict(federation=federation, architecture=architecture, method=method, config=config, seed=seed,
                  checkpoint=checkpoint, progress_callback=progress)
    if not checkpoint_path.exists():
        result = run(**kwargs, stop_after=1)
        if config.rounds == 1 or stop_after == 1:
            return full.attach_evaluations(result, checkpoint.evaluations)
    result = run(**kwargs, stop_after=stop_after)
    return full.attach_evaluations(result, checkpoint.evaluations)
