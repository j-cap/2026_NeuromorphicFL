"""Torch ResNet-14 FedAvg pulses with real shared codecs and numeric checkpoints."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.nn.utils import parameters_to_vector

from p13_cifar10_resnet14_torch import _copy_vector_to_model, _local_delta, _evaluate
from neuromorphicfl.cifar10_resnet_torch import CIFARResNet14, parameter_count
from neuromorphicfl.event_codec import encode_pulses, decode_pulses, encode_event_downlink, decode_event_downlink
from neuromorphicfl.event_codec_pilot import _same_bits


def configure(requested="auto", threads=4):
    if threads < 1:
        raise ValueError("Positive Torch thread count required")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; check the existing conda environment")
    device = torch.device("cuda:0" if requested == "cuda" else requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def runtime(device):
    return {"python": platform.python_version(), "platform": platform.system(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(), "deterministic": True, "tf32": False, "mixed_precision": False}


def pulse_step(state, deltas, weights, *, method, config, rnd):
    if method not in ("event", "strom"):
        raise ValueError("Only Event and Ström are supported")
    if method == "event":
        state.mul_(config.rho)
    amplitude = config.jump0 * (1+rnd/config.jump_scale)**(-config.jump_exponent) if method == "event" else config.strom_threshold
    pulses = torch.zeros(state.shape, dtype=torch.int8, device=state.device)
    native = torch.zeros(state.shape[1], dtype=torch.float32, device=state.device)
    for i, delta in enumerate(deltas):
        state[i].add_(delta, alpha=(1/config.local_lr)*float(weights[i].item()))
        mask = state[i].abs() >= (config.threshold if method == "event" else config.strom_threshold)
        signs = state[i, mask].sign()
        pulses[i, mask] = signs.to(torch.int8)
        native[mask] = native[mask] + amplitude*signs
        if method == "event":
            state[i, mask] = 0.
        else:
            state[i, mask] = state[i, mask] - config.strom_threshold*signs
    matrix = pulses.cpu().numpy()
    modes, uplink, legacy, active = Counter(), 0, 0, 0
    decoded_matrix = np.zeros_like(matrix)
    for i, vector in enumerate(matrix):
        count = int(np.count_nonzero(vector))
        if count:
            packet = encode_pulses(vector)
            decoded_matrix[i] = decode_pulses(packet.data, expected_dimension=matrix.shape[1])
            if not np.array_equal(decoded_matrix[i], vector):
                raise AssertionError("Torch pulse uplink changed")
            uplink += packet.wire_bits
            legacy += 64 + ((matrix.shape[1]-1).bit_length()+1)*count
            active += 1
            modes[packet.mode] += 1
    packet = encode_event_downlink(decoded_matrix, amplitude)
    decoded = decode_event_downlink(packet.data, amplitude, expected_dimension=matrix.shape[1])
    # Compare against the native Torch summation before applying the gain.
    _same_bits(native.cpu().numpy(), decoded)
    communicated = torch.from_numpy(decoded).to(state.device)
    native.mul_(config.server_gain)
    communicated.mul_(config.server_gain)
    _same_bits(native.cpu().numpy(), communicated.cpu().numpy())
    events = int(np.count_nonzero(matrix))
    counts = matrix.astype(np.int32).sum(axis=0)
    return communicated, {"method": method, "coordinate_events": events, "legacy_uplink_bits": legacy,
                          "codec_uplink_bits": uplink, "legacy_server_bits": legacy, "codec_replay_bits": uplink,
                          "exact_downlink_bits": len(packet.data)*8, "integer_only_downlink_bits": packet.counts.wire_bits,
                          "aggregate_exact_downlink_bits": packet.aggregate_exact_bits, "ordered_downlink_bits": packet.ordered_bits,
                          "correction_nonzeros": packet.correction_nonzeros, "canonical_max_abs_error": packet.canonical_max_abs_error,
                          "aggregate_nonzeros": int(np.count_nonzero(counts)), "aggregate_l1": int(np.abs(counts).sum()),
                          "cancellation_fraction": 1-int(np.abs(counts).sum())/events if events else 0.,
                          "active_clients": active, "uplink_modes": json.dumps(dict(modes), sort_keys=True), "downlink_mode": packet.mode}


def sampled(config, horizon):
    return sorted({1, horizon, *range(config.eval_stride, horizon+1, config.eval_stride)})


def run(*, federation, method, config, seed, device, checkpoint, eval_batch_size=256, stop_after=None, callback=None):
    horizon = config.rounds if stop_after is None else stop_after
    if (method not in ("event", "strom") or not 1 <= horizon <= config.rounds or config.rounds <= 0 or config.eval_stride <= 0
            or eval_batch_size < 1 or config.local_steps < 1 or config.batch_size < 1 or config.local_lr <= 0
            or config.server_gain <= 0 or config.server_gain_exponent != 0 or config.init_scale != 1
            or not np.isfinite(list(asdict(config).get(k) for k in ("threshold", "jump0", "strom_threshold", "server_gain", "local_lr", "regularization"))).all()
            or config.threshold <= 0 or config.jump0 <= 0 or config.strom_threshold <= 0 or config.regularization < 0):
        raise ValueError("Invalid ResNet pulse configuration")
    if federation.scope not in ("validation", "test", "synthetic"):
        raise ValueError("Unknown evaluation scope")
    torch.manual_seed(7777)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7777)
    model, local = CIFARResNet14().to(device), CIFARResNet14().to(device)
    dimension = parameter_count(model)
    if dimension != 175258:
        raise AssertionError("ResNet-14 parameter count changed")
    w = parameters_to_vector(model.parameters()).detach().clone()
    replica = w.clone()
    state = torch.zeros((federation.n_clients, dimension), dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    identity = {"host": "resnet14_transfer_pulse_v1", "method": method, "config": asdict(config), "seed": seed,
                "dimension": dimension, "clients": federation.n_clients, "weights": federation.weights.cpu().tolist(),
                "scope": federation.scope, "data_manifest": federation.manifest, "runtime": runtime(device), "eval_batch_size": eval_batch_size}
    records, evaluations, start = [], [], 0
    saved = checkpoint.load(identity)
    if saved is not None:
        arrays, progress = saved["arrays"], saved["progress"]
        for name, target in (("model", w), ("client_model", replica), ("pulse_state", state)):
            value = arrays[name]
            if value.dtype != np.float32 or value.shape != tuple(target.shape) or not np.isfinite(value).all():
                raise ValueError("Invalid ResNet checkpoint arrays")
            target.copy_(torch.from_numpy(value).to(device))
        start, records, evaluations = int(progress["completed_round"]), progress["records"], progress["evaluations"]
        if (not 0 <= start <= horizon or [r["round"] for r in records] != list(range(1, start+1))
                or [r["round"] for r in evaluations] != [r for r in sampled(config, config.rounds) if r <= start]):
            raise ValueError("ResNet checkpoint sample/round coverage differs")
        rng.bit_generator.state = progress["numpy_rng"]
        torch.set_rng_state(torch.from_numpy(arrays["torch_rng"]))
        if device.type == "cuda":
            torch.cuda.set_rng_state(torch.from_numpy(arrays["cuda_rng"]), device)
        _same_bits(w.cpu().numpy(), replica.cpu().numpy())
    prefix = "validation" if federation.scope == "validation" else "test"
    for rnd in range(start+1, horizon+1):
        started = time.perf_counter()
        deltas = [_local_delta(local, w, federation.client_images[i], federation.client_labels[i], config, rng)
                  for i in range(federation.n_clients)]
        if not all(bool(torch.isfinite(delta).all().item()) for delta in deltas):
            raise FloatingPointError("Nonfinite ResNet local update")
        train_seconds = time.perf_counter()-started
        started = time.perf_counter()
        update, row = pulse_step(state, deltas, federation.weights, method=method, config=config, rnd=rnd)
        w.add_(update)
        replica.add_(update)
        _same_bits(w.cpu().numpy(), replica.cpu().numpy())
        if not bool(torch.isfinite(w).all().item()):
            raise FloatingPointError("Nonfinite ResNet global model")
        row.update(round=rnd, model_sha256=hashlib.sha256(w.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
                   training_seconds=train_seconds, codec_seconds=time.perf_counter()-started)
        records.append(row)
        if rnd in sampled(config, config.rounds):
            _copy_vector_to_model(w, model)
            train = _evaluate(model, federation.train_eval_images, federation.train_eval_labels, config.regularization, batch_size=eval_batch_size)
            prediction = _evaluate(model, federation.evaluation_images, federation.evaluation_labels, config.regularization, batch_size=eval_batch_size)
            if not np.isfinite([*train[:5], *prediction[:5]]).all():
                raise FloatingPointError("Nonfinite ResNet evaluation")
            evaluations.append({"round": rnd, "train_objective": train[0], "train_ce": train[1], prefix+"_ce": prediction[1],
                                prefix+"_accuracy": prediction[2], prefix+"_macro_accuracy": prediction[3],
                                prefix+"_worst_class_accuracy": prediction[4], "model_sha256": row["model_sha256"]})
        if rnd == 1 or rnd % config.eval_stride == 0 or rnd == horizon:
            arrays = {"model": w.cpu().numpy(), "client_model": replica.cpu().numpy(), "pulse_state": state.cpu().numpy(),
                      "torch_rng": torch.get_rng_state().cpu().numpy()}
            if device.type == "cuda":
                arrays["cuda_rng"] = torch.cuda.get_rng_state(device).cpu().numpy()
            checkpoint.save(identity, arrays, {"completed_round": rnd, "records": records, "evaluations": evaluations,
                                                "numpy_rng": rng.bit_generator.state})
        if callback is not None:
            callback(row, evaluations[-1] if evaluations else {})
    return {"rounds": pd.DataFrame(records), "evaluation": pd.DataFrame(evaluations), "dimension": dimension}
