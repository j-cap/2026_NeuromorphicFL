"""Lossless codec adapter for the archived round-based Ström-style operator.

Kept outside src/neuromorphicfl to preserve frozen package-source inventories.
The residual and floating-point update order match run_final_baseline('strom').
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

import event_codec_full_horizon as full
from neuromorphicfl.event_codec import (
    decode_event_downlink, decode_pulses, encode_event_downlink, encode_pulses, pulse_aggregate,
)
from neuromorphicfl.event_codec_pilot import _same_bits
from neuromorphicfl.final_baseline_campaign import _architecture_ops, _local_delta


def residual_pulses(state, deltas, weights, *, local_lr, threshold):
    """One signed pulse per coordinate/client/round; retain every overshoot.

    State and threshold have the archived gradient-like units delta/local_lr.
    With server_gain=local_lr*g, the transmitted model increment has the units
    of a FedAvg delta with effective server gain g. No leakage or full reset.
    """
    if not math.isfinite(local_lr) or local_lr <= 0 or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive finite local learning rate and threshold required")
    pulses = np.zeros(state.shape, dtype=np.int8)
    for i, delta in enumerate(deltas):
        # Preserve the native sequence of float32 multiplication/addition.
        state[i] += (1.0 / local_lr) * float(weights[i]) * delta
        mask = np.abs(state[i]) >= threshold
        signs = np.sign(state[i, mask]).astype(np.int8)
        pulses[i, mask] = signs
        state[i, mask] -= threshold * signs.astype(np.float32)
    return pulses


def traffic_summary(rounds, *, n_clients, dimension):
    # Both methods send fixed-amplitude signed pulses. Give Ström exactly the
    # same four representation opportunities and replay/checkpoint fallback.
    return full.traffic_summary(rounds.assign(method="event"), n_clients=n_clients, dimension=dimension)


def trajectory_tables(result, *, n_clients, dimension):
    adapted = {**result, "rounds": result["rounds"].assign(method="event")}
    return full.trajectory_tables(adapted, n_clients=n_clients, dimension=dimension)


def run_strom_codec(*, federation, architecture, config, seed, checkpoint=None,
                    progress_callback=None, stop_after=None):
    if (config.rounds <= 0 or config.eval_stride <= 0 or config.local_steps <= 0
            or config.batch_size <= 0 or not np.isfinite(config.server_gain) or config.server_gain <= 0
            or config.server_gain_exponent != 0 or not np.isfinite(config.regularization)
            or config.regularization < 0):
        raise ValueError("Invalid Ström codec configuration")
    if not np.isfinite(config.local_lr) or config.local_lr <= 0 or not np.isfinite(config.strom_threshold) or config.strom_threshold <= 0:
        raise ValueError("Positive finite local learning rate and threshold required")
    if stop_after is not None and not 1 <= stop_after <= config.rounds:
        raise ValueError("stop_after is outside the configured horizon")
    layout, initialize, loss_grad, metrics = _architecture_ops(architecture)
    d, n = int(layout.dimension), federation.n_clients
    weights = federation.weights.astype(float)
    weights /= weights.sum()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Client weights must be positive and finite")
    rng = np.random.default_rng(seed)
    w = initialize(layout=layout, seed=7777, scale=config.init_scale)
    client_model = w.copy()
    state = np.zeros((n, d), dtype=np.float32)
    arrays = {"model": w, "client_model": client_model, "strom_state": state}
    identity = {"host": "strom_codec_host_v1", "architecture": architecture, "config": asdict(config),
                "seed": seed, "n_clients": n, "dimension": d, "weights": weights.tolist(), "numpy_version": np.__version__}
    records, history, start = [], [], 0
    if checkpoint is not None:
        saved = checkpoint.load(identity)
        if saved is not None:
            if set(saved["arrays"]) != set(arrays):
                raise ValueError("Ström checkpoint array names differ")
            for name, target in arrays.items():
                value = saved["arrays"][name]
                if value.shape != target.shape or value.dtype != target.dtype or not np.isfinite(value).all():
                    raise ValueError("Invalid Ström checkpoint arrays")
                target[:] = value
            progress = saved["progress"]
            start = int(progress["completed_round"])
            records, history = progress["records"], progress["history"]
            rng.bit_generator.state = progress["rng_state"]
            expected_history = [r for r in full.sampled_rounds(config) if r <= start]
            if (not 0 <= start <= config.rounds or [r["round"] for r in records] != list(range(1, start + 1))
                    or [r["round"] for r in history] != expected_history):
                raise ValueError("Ström checkpoint coverage differs")
            _same_bits(w, client_model)
    for rnd in range(start + 1, config.rounds + 1):
        deltas = [_local_delta(w, federation.client_X[i], federation.client_y[i], config=config,
                              rng=rng, layout=layout, loss_grad=loss_grad) for i in range(n)]
        if not all(np.isfinite(delta).all() for delta in deltas):
            raise FloatingPointError("Nonfinite Ström local delta")
        began = time.perf_counter()
        pulses = residual_pulses(state, deltas, weights, local_lr=config.local_lr, threshold=config.strom_threshold)
        modes, legacy, packed, active = Counter(), 0, 0, 0
        for i, vector in enumerate(pulses):
            count = int(np.count_nonzero(vector))
            if count:
                packet = encode_pulses(vector)
                decoded = decode_pulses(packet.data, expected_dimension=d)
                if not np.array_equal(vector, decoded):
                    raise AssertionError("Ström pulse upload changed")
                pulses[i] = decoded
                legacy += 64 + ((d - 1).bit_length() + 1) * count
                packed += packet.wire_bits
                active += 1
                modes[packet.mode] += 1
        tau = float(config.strom_threshold)
        aggregate = pulse_aggregate(pulses, tau)
        packet = encode_event_downlink(pulses, tau)
        decoded = decode_event_downlink(packet.data, tau, expected_dimension=d)
        _same_bits(aggregate, decoded)
        counts = pulses.astype(np.int32).sum(axis=0)
        if not np.array_equal(counts, decode_pulses(packet.counts.data, expected_dimension=d)):
            raise AssertionError("Ström aggregate integer counts changed")
        # The gain is applied once, after the historical ordered accumulation.
        aggregate *= float(config.server_gain)
        decoded *= float(config.server_gain)
        events = int(np.count_nonzero(pulses))
        row = {"round": rnd, "method": "strom", "coordinate_events": events,
               "legacy_uplink_bits": legacy, "codec_uplink_bits": packed, "legacy_server_bits": legacy,
               "codec_replay_bits": packed, "exact_downlink_bits": 8 * len(packet.data),
               "integer_only_downlink_bits": packet.counts.wire_bits,
               "aggregate_exact_downlink_bits": packet.aggregate_exact_bits, "ordered_downlink_bits": packet.ordered_bits,
               "correction_nonzeros": packet.correction_nonzeros, "canonical_max_abs_error": packet.canonical_max_abs_error,
               "aggregate_nonzeros": int(np.count_nonzero(counts)), "aggregate_l1": int(np.abs(counts).sum()),
               "cancellation_fraction": 1 - np.abs(counts).sum() / events if events else 0., "active_clients": active,
               "downlink_mode": packet.mode, "uplink_modes": json.dumps(dict(modes), sort_keys=True),
               "codec_audit_seconds": time.perf_counter() - began,
               "residual_l2_norm": float(np.linalg.norm(state)),
               "residual_max_abs": float(np.max(np.abs(state)))}
        w += aggregate
        client_model += decoded
        _same_bits(w, client_model)
        if not np.isfinite(w).all() or not np.isfinite(state).all():
            raise FloatingPointError("Nonfinite Ström model/residual")
        row["model_sha256"] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
        records.append(row)
        if rnd in full.sampled_rounds(config):
            objective, ce, *_ = metrics(w, federation.X_train_eval, federation.y_train_eval,
                                        layout=layout, regularization=config.regularization)
            history.append({"round": rnd, "train_objective": float(objective), "train_ce": float(ce)})
        if checkpoint is not None and (rnd in full.sampled_rounds(config) or rnd == stop_after):
            checkpoint.save(identity, arrays, {"completed_round": rnd, "records": records,
                                               "history": history, "rng_state": rng.bit_generator.state})
        if progress_callback is not None:
            progress_callback(row)
        if rnd == stop_after:
            break
    records = pd.DataFrame(records)
    return {"model": w, "client_model": client_model, "strom_state": state, "rng_state": rng.bit_generator.state,
            "rounds": records, "history": pd.DataFrame(history),
            "summary": traffic_summary(records, n_clients=n, dimension=d)}


def evaluated_run(*, federation, architecture, config, seed, checkpoint_path, provenance,
                  progress_callback=None, stop_after=None):
    checkpoint = full.PredictiveCheckpoint(checkpoint_path, provenance, federation=federation,
                                          architecture=architecture, config=config)

    def progress(row):
        if progress_callback is not None:
            progress_callback(row, checkpoint.evaluations[-1] if checkpoint.evaluations else {})

    kwargs = dict(federation=federation, architecture=architecture, config=config, seed=seed,
                  checkpoint=checkpoint, progress_callback=progress)
    if not checkpoint_path.exists():
        result = run_strom_codec(**kwargs, stop_after=1)
        if config.rounds == 1 or stop_after == 1:
            return full.attach_evaluations(result, checkpoint.evaluations)
    result = run_strom_codec(**kwargs, stop_after=stop_after)
    return full.attach_evaluations(result, checkpoint.evaluations)
