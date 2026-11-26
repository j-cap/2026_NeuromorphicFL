"""NumPy MLP runner: native operators, exact transport, resumable state.

Independent jobs run in separate processes. Local client order, RNG and native
float32 updates remain sequential inside a job, independent of worker count.
"""
from __future__ import annotations

from dataclasses import asdict
from collections import Counter
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments")]
import numpy as np
import pandas as pd
import fast_pulse_codec as fast
import resnet14_baseline_fast_host as baselines
from neuromorphicfl import event_codec as native_wire
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import _local_delta
from neuromorphicfl.fmnist_multiclass_benchmark import LAYOUT, initialize_mlp, loss_and_gradient, _forward_logits
from neuromorphicfl.event_codec_pilot import _same_bits

METHODS = ("event", "strom", "dense", "stc", "ef_topk")
MAIN = {"event": "both_codecs", "strom": "both_codecs", **baselines.MAIN}
IMPLEMENTATION = "MLP_native_float32_fast_bound1_exact_downlink_preselection_v1"


def _private_wire():
    name = "neuromorphicfl._fmnist_mlp_exact_wire"
    spec = importlib.util.spec_from_file_location(name, native_wire.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.encode_pulses, module.decode_pulses = fast.encode_pulses, fast.decode_pulses
    return module


wire = _private_wire()


def encode_downlink(pulses, jump):
    """Exact native bytes; skip serializing an ordered candidate that loses."""
    matrix = np.asarray(pulses)
    if matrix.ndim != 2 or not 0 < len(matrix) <= 32767 or not np.isfinite(jump) or jump <= 0:
        raise ValueError("Invalid pulse matrix/amplitude")
    for row in matrix:
        native_wire._integers(row, 1)
    counts = fast.encode_pulses(matrix.astype(np.int32).sum(axis=0, dtype=np.int32), max_abs=len(matrix))
    canonical = jump * fast.decode_pulses(counts.data).astype(np.float32)
    reference = native_wire.pulse_aggregate(matrix, jump)
    exceptions = reference.view(np.uint32) != canonical.view(np.uint32)
    n = int(exceptions.sum())
    patch = fast.encode_pulses(exceptions.astype(np.int8)).data if n else b""
    corrected = (native_wire._PATCH.pack(b"EVR1", len(counts.data), len(patch)) + counts.data + patch
                 + reference[exceptions].astype(">f4").tobytes())
    ordered_size = native_wire._ORDERED.size + (matrix.shape[1] + 2*int(np.count_nonzero(matrix)) + 7)//8
    if len(corrected) <= ordered_size:
        chosen, mode = corrected, "counts_patch"
    else:
        chosen, mode = native_wire._encode_ordered(matrix), "ordered"
        if len(chosen) != ordered_size:
            raise AssertionError("Ordered candidate size formula differs")
    return native_wire.EventDownlink(chosen, mode, counts, len(corrected)*8, ordered_size*8,
                                    n, float(np.max(np.abs(reference.astype(float)-canonical))))


class NumericalFailure(FloatingPointError):
    def __init__(self, rnd, reason):
        self.round = rnd
        super().__init__(f"Round {rnd}: {reason}")


def pulse_step(state, deltas, weights, *, method, config, rnd):
    if method == "event":
        state *= config.rho
    pulses = np.zeros(state.shape, dtype=np.int8)
    for i, delta in enumerate(deltas):
        state[i] += (1/config.local_lr)*float(weights[i])*delta
        threshold = config.threshold if method == "event" else config.strom_threshold
        mask = np.abs(state[i]) >= threshold
        signs = np.sign(state[i, mask]).astype(np.int8)
        pulses[i, mask] = signs
        if method == "event":
            state[i, mask] = 0.
        else:
            state[i, mask] -= threshold*signs.astype(np.float32)
    upload, modes = 0, Counter()
    decoded_pulses = np.zeros_like(pulses)
    for i, vector in enumerate(pulses):
        if np.any(vector):
            packet = fast.encode_pulses(vector)
            decoded_pulses[i] = fast.decode_pulses(packet.data, expected_dimension=pulses.shape[1])
            if not np.array_equal(decoded_pulses[i], vector):
                raise AssertionError("Pulse upload differs")
            upload += packet.wire_bits
            modes[packet.mode] += 1
    amplitude = (config.jump0*(1+rnd/config.jump_scale)**(-config.jump_exponent) if method == "event"
                 else config.strom_threshold)
    reference = native_wire.pulse_aggregate(pulses, amplitude)
    packet = encode_downlink(decoded_pulses, amplitude)
    received = wire.decode_event_downlink(packet.data, amplitude, expected_dimension=pulses.shape[1])
    reference *= float(config.server_gain)
    received *= float(config.server_gain)
    _same_bits(reference, received)
    return received, {"method": method, "coordinate_events": int(np.count_nonzero(pulses)),
                      "codec_uplink_bits": upload, "exact_downlink_bits": len(packet.data)*8,
                      "active_clients": int(np.any(pulses != 0, axis=1).sum()),
                      "downlink_mode": packet.mode, "aggregate_downlink_bits": packet.aggregate_exact_bits,
                      "ordered_downlink_bits": packet.ordered_bits, "roundtrip_verified": True}


def metrics(w, X, y, *, regularization, batch_size=512):
    """One forward pass per batch; no gradient and no second accuracy pass."""
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty split")
    total, correct = 0., np.zeros(10, dtype=np.int64)
    counts = np.bincount(y, minlength=10)
    for start in range(0, len(y), batch_size):
        labels = y[start:start+batch_size]
        _, logits = _forward_logits(w, X[start:start+batch_size], layout=LAYOUT)
        shifted = logits - logits.max(axis=1, keepdims=True)
        losses = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1) - logits[np.arange(len(labels)), labels]
        total += float(np.sum(losses, dtype=np.float64))
        predictions = logits.argmax(axis=1)
        correct += np.bincount(labels[predictions == labels], minlength=10)
    ce = total/len(y)
    per_class = correct/counts if np.all(counts) else np.divide(correct, counts, out=np.zeros(10), where=counts > 0)
    result = {"objective": ce + .5*regularization*float(w@w), "ce": ce,
              "accuracy": float(correct.sum()/len(y)), "macro_accuracy": float(per_class.mean()),
              "worst_class_accuracy": float(per_class.min())}
    if not all(np.isfinite(list(result.values()))):
        raise FloatingPointError("Nonfinite predictive metric")
    return result


def run(data, *, method, config, seed, checkpoint_path, provenance, stop_after=None, callback=None):
    if method not in METHODS or config.rounds <= 0 or config.eval_stride <= 0:
        raise ValueError("Invalid MLP run")
    goal = config.rounds if stop_after is None else stop_after
    if not 1 <= goal <= config.rounds:
        raise ValueError("Invalid stage horizon")
    dimension, n = LAYOUT.dimension, len(data["client_y"])
    sizes = tuple(size for _, _, size in LAYOUT.groups())
    weights = np.full(n, .1, dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    model = initialize_mlp(layout=LAYOUT, seed=7777, scale=config.init_scale)
    arrays = {"model": model, "client_model": model.copy(), "pulse_state": np.zeros((n, dimension), np.float32),
              "client_residuals": np.zeros((n, dimension), np.float32), "server_residual": np.zeros(dimension, np.float32)}
    identity = {"host": IMPLEMENTATION, "method": method, "config": asdict(config), "seed": seed,
                "numpy_version": np.__version__, "n_clients": n, "weights": weights.tolist()}
    checkpoint = BaselineCheckpoint(Path(checkpoint_path), provenance)
    saved = checkpoint.load(identity)
    records, evaluation, start = [], [], 0
    if saved is not None:
        if set(saved["arrays"]) != set(arrays):
            raise ValueError("Checkpoint state names differ")
        for name, target in arrays.items():
            value = saved["arrays"][name]
            if value.shape != target.shape or value.dtype != target.dtype or not np.isfinite(value).all():
                raise ValueError("Invalid MLP checkpoint state")
            target[:] = value
        progress = saved["progress"]
        records, evaluation, start = progress["records"], progress["evaluation"], int(progress["completed_round"])
        rng.bit_generator.state = progress["rng_state"]
        if not 0 <= start <= goal or [r["round"] for r in records] != list(range(1, start+1)):
            raise ValueError("Invalid MLP checkpoint round coverage")
        expected = [r for r in range(1, start+1) if r == 1 or r % config.eval_stride == 0 or r == config.rounds]
        if [r["round"] for r in evaluation] != expected:
            raise ValueError("Invalid MLP checkpoint evaluation coverage")
        _same_bits(model, arrays["client_model"])
    for rnd in range(start+1, goal+1):
        try:
            tick = time.perf_counter()
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                deltas = [_local_delta(model, data["client_X"][i], data["client_y"][i], config=config,
                                       rng=rng, layout=LAYOUT, loss_grad=loss_and_gradient) for i in range(n)]
                if not all(np.isfinite(d).all() for d in deltas):
                    raise FloatingPointError("Nonfinite local update")
                trained = time.perf_counter()
                if method in ("event", "strom"):
                    update, row = pulse_step(arrays["pulse_state"], deltas, weights, method=method, config=config, rnd=rnd)
                else:
                    update, row = baselines.step(arrays, deltas, weights, method=method, config=config, sizes=sizes, rnd=rnd)
                row.pop("wire_protocol", None)
                arrays["model"] += update
                arrays["client_model"] += update
                _same_bits(arrays["model"], arrays["client_model"])
                if not np.isfinite(model).all():
                    raise FloatingPointError("Nonfinite model")
                encoded = time.perf_counter()
                fingerprint = hashlib.sha256(model.astype("<f4").tobytes()).hexdigest()
                row.update(round=rnd, model_sha256=fingerprint, training_seconds=trained-tick, codec_seconds=encoded-trained)
                if rnd == 1 or rnd % config.eval_stride == 0 or rnd == config.rounds:
                    train = metrics(model, data["X_diagnostic"], data["y_diagnostic"], regularization=config.regularization)
                    held = metrics(model, data["X_holdout"], data["y_holdout"], regularization=config.regularization)
                    ev = {"round": rnd, "model_sha256": fingerprint, "train_objective": train["objective"], "train_ce": train["ce"],
                          "parameter_l2_norm": float(np.linalg.norm(model.astype(float))),
                          **{data["scope"]+"_"+key: held[key] for key in ("ce", "accuracy", "macro_accuracy", "worst_class_accuracy")}}
                    evaluation.append(ev)
                row["evaluation_seconds"] = time.perf_counter()-encoded
                records.append(row)
        except (FloatingPointError, OverflowError) as exc:
            raise NumericalFailure(rnd, str(exc)) from exc
        if rnd % 50 == 0 or rnd == config.rounds or rnd == goal:
            checkpoint.save(identity, arrays, {"completed_round": rnd, "records": records, "evaluation": evaluation,
                                                "rng_state": rng.bit_generator.state})
            if callback:
                callback(rnd)
    return {"rounds": pd.DataFrame(records), "evaluation": pd.DataFrame(evaluation),
            "arrays": {k: v.copy() for k, v in arrays.items()}, "rng_state": rng.bit_generator.state}


def trajectory_tables(result, method, *, n_clients=10):
    rounds, evaluation = result["rounds"], result["evaluation"]
    upload = rounds.codec_uplink_bits.to_numpy(dtype=np.int64).cumsum()
    stream = rounds.exact_downlink_bits.to_numpy(dtype=np.int64).cumsum()
    initial, rows = 32*LAYOUT.dimension+64, []
    for ev in evaluation.to_dict("records"):
        rnd = int(ev["round"])
        up, down = int(upload[rnd-1]), int(stream[rnd-1])
        unicast = n_clients*(initial + 32*rnd + down)
        rows.append({**ev, "variant": MAIN[method], "uplink_bits": up, "server_stream_bits": down,
                     "unicast_downlink_bits": unicast, "unicast_total_bits": up+unicast,
                     "broadcast_total_bits": up+initial+down})
    history = pd.DataFrame(rows)
    last = history.iloc[-1].to_dict()
    summary = {"rounds": int(last.pop("round")), **{("final_"+k if k.startswith(("train_", "test_", "validation_")) else k): v for k, v in last.items()}}
    return {"rounds": rounds, "evaluation": evaluation, "history": history, "summary": pd.DataFrame([summary])}
