"""Matched Torch local learning with native float32 Dense/STC/EF server rules.

Compression/aggregation uses the frozen NumPy baseline conventions. Model
training and application use the same Torch ResNet host as the pulse study.
All primary wire lengths come from actual decoded lossless messages.
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
import core_float_host as wire
from neuromorphicfl.stc import sparse_ternary_compress, STCState
from neuromorphicfl.event_codec import encode_pulses, decode_stc_lossless
from neuromorphicfl.event_codec_pilot import _same_bits

MAIN = {"dense": "dense_float32", "stc": "lossless_stc", "ef_topk": "lossless_ef_topk"}
WIRE_PROTOCOL = "resnet_native_numpy_baselines_lossless_v1"


def ternary(values, sizes, fraction):
    result = np.zeros_like(values)
    start = 0
    for size in sizes:
        result[start:start + size] = sparse_ternary_compress(values[start:start + size], fraction)
        start += size
    # Native Rice decoding canonicalizes transmitted zero entries to +0.
    result[result == 0] = np.float32(0)
    return result


def encode_ternary(values, sizes):
    """Emit the existing SCL1 envelope directly, avoiding a redundant Rice pass."""
    if values.dtype != np.float32 or values.ndim != 1 or sum(sizes) != values.size or not np.isfinite(values).all():
        raise ValueError("Invalid ternary tensor vector")
    scales, start = [], 0
    for size in sizes:
        block = values[start:start + size]
        active = np.abs(block[block != 0])
        scale = float(active[0]) if active.size else 0.
        if active.size and not np.all(active == scale):
            raise ValueError("STC block must have one nonzero magnitude")
        scales.append(scale)
        start += size
    data = (struct.pack(">4sI", b"SCL1", len(sizes)) + struct.pack(">" + "f" * len(sizes), *scales)
            + encode_pulses(np.sign(values).astype(np.int8)).data)
    decoded = decode_stc_lossless(data, tuple(sizes))
    _same_bits(values, decoded)
    return data, decoded


def gain_at(config, rnd):
    return float(config.server_gain) * (1 + rnd / float(config.server_gain_scale)) ** (-float(config.server_gain_exponent))


def native_step(state, deltas, weights, *, method, config, sizes, rnd):
    """Independent native reference, used only by verification, never a campaign."""
    gain = gain_at(config, rnd)
    if method == "stc":
        native = STCState(tuple(sizes), config.stc_upload_fraction, config.stc_download_fraction,
                          state["client_residuals"], state["server_residual"])
        return native.step(deltas, weights, server_gain=gain).server_update
    aggregate = np.zeros(sum(sizes), dtype=np.float32)
    for i, delta in enumerate(deltas):
        if method == "dense":
            aggregate += float(weights[i]) * delta
        elif method == "ef_topk":
            state["client_residuals"][i] += float(weights[i]) * delta
            k = max(1, int(round(config.topk_fraction * aggregate.size)))
            ids = np.argpartition(np.abs(state["client_residuals"][i]), -k)[-k:]
            aggregate[ids] += state["client_residuals"][i, ids]
            state["client_residuals"][i, ids] = 0.
        else:
            raise ValueError("Unknown native baseline")
    aggregate *= gain
    return aggregate


def step(state, deltas, weights, *, method, config, sizes, rnd):
    gain, dimension = gain_at(config, rnd), sum(sizes)
    aggregate = np.zeros(dimension, dtype=np.float32)
    packets, modes, events = [], Counter(), 0
    for i, delta in enumerate(deltas):
        if method == "dense":
            message = delta
            data, mode = wire.encode_vector(message, dense_only=True)
            decoded = wire.decode_vector(data, expected_dimension=dimension)
            aggregate += float(weights[i]) * decoded
        elif method == "stc":
            corrected = state["client_residuals"][i] + delta
            if not np.isfinite(corrected).all():
                raise FloatingPointError("Nonfinite STC client feedback")
            message = ternary(corrected, sizes, config.stc_upload_fraction)
            data, decoded = encode_ternary(message, sizes)
            mode = "lossless_stc"
            state["client_residuals"][i] = corrected - decoded
            aggregate += float(weights[i]) * decoded
        elif method == "ef_topk":
            state["client_residuals"][i] += float(weights[i]) * delta
            if not np.isfinite(state["client_residuals"][i]).all():
                raise FloatingPointError("Nonfinite EF client feedback")
            k = max(1, int(round(config.topk_fraction * dimension)))
            ids = np.argpartition(np.abs(state["client_residuals"][i]), -k)[-k:]
            message = np.zeros(dimension, dtype=np.float32)
            message[ids] = state["client_residuals"][i, ids]
            data, mode = wire.encode_vector(message)
            decoded = wire.decode_vector(data, expected_dimension=dimension)
            aggregate[ids] += decoded[ids]
            state["client_residuals"][i, ids] = 0.
        else:
            raise ValueError("Unknown baseline")
        _same_bits(message, decoded)
        packets.append(data); modes[mode] += 1
        events += int(np.count_nonzero(message))
    aggregate *= gain
    if not np.isfinite(aggregate).all():
        raise FloatingPointError("Nonfinite baseline aggregate")
    if method == "stc":
        corrected = state["server_residual"] + aggregate
        if not np.isfinite(corrected).all():
            raise FloatingPointError("Nonfinite STC server feedback")
        update = ternary(corrected, sizes, config.stc_download_fraction)
        down, received = encode_ternary(update, sizes)
        state["server_residual"][:] = corrected - received
        mode, aggregate_bits, replay_bits = "lossless_stc", 8 * len(down), 0
    elif method == "dense":
        update = aggregate
        down, mode = wire.encode_vector(update, dense_only=True)
        received = wire.decode_vector(down, expected_dimension=dimension)
        aggregate_bits, replay_bits = 8 * len(down), 0
    else:
        update = aggregate
        down, received, mode, aggregate_bits, replay_bits = wire.downlink(update, packets, gain=gain)
    _same_bits(update, received)
    return received, {"method": method, "coordinate_events": events, "codec_uplink_bits": sum(8 * len(p) for p in packets),
                      "exact_downlink_bits": 8 * len(down), "aggregate_downlink_bits": aggregate_bits,
                      "ordered_downlink_bits": replay_bits, "active_clients": len(deltas),
                      "uplink_modes": json.dumps(dict(modes), sort_keys=True), "downlink_mode": mode,
                      "server_gain": gain, "roundtrip_verified": True, "wire_protocol": WIRE_PROTOCOL}


def traffic_summary(rounds, *, n_clients, dimension):
    initial = 32 * dimension + 64
    upload, stream = int(rounds.codec_uplink_bits.sum()), int(rounds.exact_downlink_bits.sum())
    down = n_clients * (initial + 32 * len(rounds) + stream)
    return pd.DataFrame([{"variant": MAIN[rounds.iloc[0].method], "rounds": len(rounds), "uplink_bits": upload,
                          "server_stream_bits": stream, "unicast_downlink_bits": down, "unicast_total_bits": upload + down,
                          "broadcast_downlink_bits": initial + stream, "broadcast_total_bits": upload + initial + stream,
                          "roundtrip_verified": bool(rounds.roundtrip_verified.all())}])


def run(*, federation, method, config, seed, device, checkpoint, eval_batch_size=256, stop_after=None,
        callback=None, native_reference=False):
    import torch
    from torch.nn.utils import parameters_to_vector
    import resnet14_transfer_host as torch_host
    horizon = config.rounds if stop_after is None else stop_after
    numbers = [config.server_gain, config.server_gain_exponent, config.server_gain_scale, config.local_lr,
               config.regularization, config.stc_upload_fraction, config.stc_download_fraction, config.topk_fraction]
    if (method not in MAIN or not 1 <= horizon <= config.rounds or config.eval_stride <= 0 or config.local_steps < 1
            or config.batch_size < 1 or eval_batch_size < 1 or config.init_scale != 1 or not np.isfinite(numbers).all()
            or config.server_gain <= 0 or config.server_gain_exponent < 0 or config.server_gain_scale <= 0
            or (method != "dense" and config.server_gain_exponent != 0) or config.local_lr <= 0 or config.regularization < 0
            or any(not 0 < p <= 1 for p in (config.stc_upload_fraction, config.stc_download_fraction, config.topk_fraction))):
        raise ValueError("Invalid ResNet baseline configuration")
    if federation.scope not in ("validation", "test", "synthetic"):
        raise ValueError("Unknown baseline data scope")
    torch.manual_seed(7777)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7777)
    model, local = torch_host.CIFARResNet14().to(device), torch_host.CIFARResNet14().to(device)
    sizes = tuple(p.numel() for p in model.parameters())
    dimension, clients = sum(sizes), federation.n_clients
    if dimension != 175258:
        raise AssertionError("ResNet tensor layout changed")
    weights = federation.weights.cpu().numpy().astype(np.float64)
    if weights.shape != (clients,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid baseline client weights")
    weights /= weights.sum()
    w = parameters_to_vector(model.parameters()).detach().clone()
    replica = w.clone()
    state = {"client_residuals": np.zeros((clients, dimension), dtype=np.float32),
             "server_residual": np.zeros(dimension, dtype=np.float32)}
    rng = np.random.default_rng(seed)
    identity = {"host": WIRE_PROTOCOL, "method": method, "config": asdict(config), "seed": seed,
                "clients": clients, "tensor_sizes": list(sizes), "weights": weights.tolist(), "scope": federation.scope,
                "data_manifest": federation.manifest, "runtime": torch_host.runtime(device), "eval_batch_size": eval_batch_size,
                "native_reference_only": native_reference}
    records, evaluation, start = [], [], 0
    saved = checkpoint.load(identity)
    if saved is not None:
        arrays, progress = saved["arrays"], saved["progress"]
        for name, tensor in (("model", w), ("client_model", replica)):
            value = arrays[name]
            if value.dtype != np.float32 or value.shape != tuple(tensor.shape) or not np.isfinite(value).all():
                raise ValueError("Invalid baseline model checkpoint")
            tensor.copy_(torch.from_numpy(value).to(device))
        for name, target in state.items():
            value = arrays[name]
            if value.dtype != target.dtype or value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError("Invalid baseline feedback checkpoint")
            target[:] = value
        start, records, evaluation = int(progress["completed_round"]), progress["records"], progress["evaluations"]
        if (not 0 <= start <= horizon or [r["round"] for r in records] != list(range(1, start + 1))
                or [r["round"] for r in evaluation] != [r for r in torch_host.sampled(config, config.rounds) if r <= start]):
            raise ValueError("Invalid baseline checkpoint coverage")
        rng.bit_generator.state = progress["numpy_rng"]
        torch.set_rng_state(torch.from_numpy(arrays["torch_rng"]))
        if device.type == "cuda":
            torch.cuda.set_rng_state(torch.from_numpy(arrays["cuda_rng"]), device)
        _same_bits(w.cpu().numpy(), replica.cpu().numpy())
    prefix = "validation" if federation.scope == "validation" else "test"
    for rnd in range(start + 1, horizon + 1):
        began = time.perf_counter()
        deltas = [torch_host._local_delta(local, w, federation.client_images[i], federation.client_labels[i], config, rng)
                  for i in range(clients)]
        if not all(bool(torch.isfinite(d).all().item()) for d in deltas):
            raise FloatingPointError("Nonfinite baseline local delta")
        training_seconds = time.perf_counter() - began
        began = time.perf_counter()
        values = [d.cpu().numpy() for d in deltas]
        if native_reference:
            update = native_step(state, values, weights, method=method, config=config, sizes=sizes, rnd=rnd)
            row = {"method": method, "wire_protocol": "native_reference_only", "roundtrip_verified": False}
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                update, row = step(state, values, weights, method=method, config=config, sizes=sizes, rnd=rnd)
        w.add_(torch.from_numpy(update).to(device))
        replica.add_(torch.from_numpy(update.copy()).to(device))
        _same_bits(w.cpu().numpy(), replica.cpu().numpy())
        if not bool(torch.isfinite(w).all().item()):
            raise FloatingPointError("Nonfinite baseline global model")
        row.update(round=rnd, model_sha256=hashlib.sha256(w.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
                   training_seconds=training_seconds, codec_seconds=time.perf_counter() - began)
        records.append(row)
        if rnd in torch_host.sampled(config, config.rounds):
            torch_host._copy_vector_to_model(w, model)
            train = torch_host._evaluate(model, federation.train_eval_images, federation.train_eval_labels,
                                         config.regularization, batch_size=eval_batch_size)
            pred = torch_host._evaluate(model, federation.evaluation_images, federation.evaluation_labels,
                                        config.regularization, batch_size=eval_batch_size)
            if not np.isfinite([*train[:5], *pred[:5]]).all():
                raise FloatingPointError("Nonfinite baseline evaluation")
            evaluation.append({"round": rnd, "train_objective": train[0], "train_ce": train[1], prefix + "_ce": pred[1],
                               prefix + "_accuracy": pred[2], prefix + "_macro_accuracy": pred[3],
                               prefix + "_worst_class_accuracy": pred[4], "model_sha256": row["model_sha256"]})
        if rnd == 1 or rnd % config.eval_stride == 0 or rnd == horizon:
            arrays = {"model": w.cpu().numpy(), "client_model": replica.cpu().numpy(), **state,
                      "torch_rng": torch.get_rng_state().cpu().numpy()}
            if device.type == "cuda":
                arrays["cuda_rng"] = torch.cuda.get_rng_state(device).cpu().numpy()
            checkpoint.save(identity, arrays, {"completed_round": rnd, "records": records, "evaluations": evaluation,
                                                "numpy_rng": rng.bit_generator.state})
        if callback is not None:
            callback(row, evaluation[-1] if evaluation else {})
    return {"rounds": pd.DataFrame(records), "evaluation": pd.DataFrame(evaluation), "dimension": dimension}
