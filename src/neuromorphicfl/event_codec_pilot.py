"""Short codec audit on unchanged Event/STC trajectories.

This separate host reuses the frozen local optimizer and STC state machine.
It never edits the source files of the ongoing STC confirmation campaign.
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

from .baseline_checkpoint import BaselineCheckpoint
from .event_codec import (
    decode_event_downlink, decode_pulses, decode_stc_lossless,
    encode_event_downlink, encode_pulses, encode_stc_lossless, pulse_aggregate,
)
from .final_baseline_campaign import _architecture_ops, _local_delta
from .stc import STCState, decode_stc


def _same_bits(left, right):
    if not np.array_equal(left.view(np.uint32), right.view(np.uint32)):
        raise AssertionError("decoded message changed the float32 update")


def traffic_summary(rounds: pd.DataFrame, *, n_clients: int, dimension: int) -> pd.DataFrame:
    """Charge complete frames, initialization and requests under both topologies."""
    event = rounds.iloc[0]["method"] == "event"
    variants = (("legacy", "legacy_uplink_bits", "legacy_server_bits"),
                ("uplink_codec_replay", "codec_uplink_bits", "codec_replay_bits"),
                ("aggregate_downlink", "legacy_uplink_bits", "exact_downlink_bits"),
                ("both_codecs", "codec_uplink_bits", "exact_downlink_bits")) if event else (
                    ("native_stc", "legacy_uplink_bits", "legacy_server_bits"),
                    ("lossless_stc", "codec_uplink_bits", "exact_downlink_bits"))
    checkpoint = 32 * dimension + 64
    output = []
    for name, up, down in variants:
        uplink = int(rounds[up].sum())
        # Historical Event chooses replay or full float32 checkpoint. STC and
        # new exact downlinks are their actual native frames, without fallback.
        server = int(np.minimum(rounds[down], checkpoint).sum()) if event and name in ("legacy", "uplink_codec_replay") else int(rounds[down].sum())
        unicast_down = n_clients * (checkpoint + 32 * len(rounds) + server)
        output.append({"variant": name, "rounds": len(rounds), "uplink_bits": uplink,
                       "server_stream_bits": server, "unicast_downlink_bits": unicast_down,
                       "unicast_total_bits": uplink + unicast_down,
                       "broadcast_downlink_bits": checkpoint + server,
                       "broadcast_total_bits": uplink + checkpoint + server,
                       "roundtrip_verified": True})
    table = pd.DataFrame(output)
    baseline = int(table.iloc[0].unicast_total_bits)
    table["unicast_saving_percent"] = 100 * (1 - table.unicast_total_bits / baseline)
    return table


def run_codec_pilot(*, federation, architecture, method, config, seed,
                    checkpoint: BaselineCheckpoint | None = None, checkpoint_stride=30,
                    progress_callback=None, stop_after=None):
    if method not in ("event", "stc") or config.event_memory != "persistent" or config.event_reset != "full" or config.server_gain_exponent != 0:
        raise ValueError("pilot supports the fixed persistent/full-reset Event and native STC protocols")
    if config.rounds <= 0 or config.eval_stride <= 0 or checkpoint_stride <= 0:
        raise ValueError("invalid pilot horizon or stride")
    layout, initialize, loss_grad, metrics = _architecture_ops(architecture)
    sizes = tuple(size for _, _, size in layout.groups())
    d, n = int(layout.dimension), federation.n_clients
    weights = federation.weights.astype(float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    w = initialize(layout=layout, seed=7777, scale=config.init_scale)
    client_model = w.copy()
    event_state = np.zeros((n, d), dtype=np.float32)
    stc = STCState.zeros(n_clients=n, tensor_sizes=sizes,
                        upload_fraction=config.stc_upload_fraction,
                        download_fraction=config.stc_download_fraction) if method == "stc" else None
    arrays = {"model": w, "client_model": client_model, "event_state": event_state}
    if stc is not None:
        arrays.update(stc_client_residuals=stc.client_residuals, stc_server_residual=stc.server_residual)
    identity = {"host": "event_codec_pilot_v1", "architecture": architecture, "method": method,
                "config": asdict(config), "seed": seed, "n_clients": n, "dimension": d,
                "weights": weights.tolist(), "numpy_version": np.__version__}
    records, history, start = [], [], 0
    if checkpoint is not None:
        saved = checkpoint.load(identity)
        if saved is not None:
            if set(saved["arrays"]) != set(arrays):
                raise ValueError("pilot checkpoint array names differ")
            for name, target in arrays.items():
                source = saved["arrays"][name]
                if source.shape != target.shape or source.dtype != target.dtype or not np.all(np.isfinite(source)):
                    raise ValueError("invalid pilot checkpoint array")
                target[:] = source
            start = int(saved["progress"]["completed_round"])
            records, history = saved["progress"]["records"], saved["progress"]["history"]
            rng.bit_generator.state = saved["progress"]["rng_state"]
            if not 0 <= start <= config.rounds or len(records) != start:
                raise ValueError("invalid pilot checkpoint progress")
            _same_bits(w, client_model)

    for rnd in range(start + 1, config.rounds + 1):
        deltas = [_local_delta(w, federation.client_X[i], federation.client_y[i],
                              config=config, rng=rng, layout=layout, loss_grad=loss_grad)
                  for i in range(n)]
        if not all(np.all(np.isfinite(delta)) for delta in deltas):
            raise FloatingPointError("nonfinite local update")
        started = time.perf_counter()
        row = {"round": rnd, "method": method, "coordinate_events": 0,
               "legacy_uplink_bits": 0, "codec_uplink_bits": 0, "legacy_server_bits": 0,
               "codec_replay_bits": 0, "exact_downlink_bits": 0,
               "integer_only_downlink_bits": 0, "aggregate_exact_downlink_bits": 0,
               "ordered_downlink_bits": 0, "correction_nonzeros": 0,
               "canonical_max_abs_error": 0.0, "aggregate_nonzeros": 0,
               "aggregate_l1": 0, "cancellation_fraction": 0.0,
               "active_clients": 0, "downlink_mode": "", "uplink_modes": ""}
        if method == "event":
            event_state *= config.rho
            pulses = np.zeros((n, d), dtype=np.int8)
            modes = Counter()
            for i in range(n):
                event_state[i] += (1.0 / config.local_lr) * float(weights[i]) * deltas[i]
                mask = np.abs(event_state[i]) >= config.threshold
                pulses[i, mask] = np.sign(event_state[i, mask]).astype(np.int8)
                event_state[i, mask] = 0.0
                count = int(np.count_nonzero(pulses[i]))
                if count:
                    packet = encode_pulses(pulses[i])
                    decoded = decode_pulses(packet.data, expected_dimension=d)
                    if not np.array_equal(decoded, pulses[i]):
                        raise AssertionError("uplink pulse reconstruction failed")
                    pulses[i] = decoded
                    row["legacy_uplink_bits"] += 64 + ((d - 1).bit_length() + 1) * count
                    row["codec_uplink_bits"] += packet.wire_bits
                    row["active_clients"] += 1
                    modes[packet.mode] += 1
            jump = config.jump0 * (1.0 + rnd / config.jump_scale) ** (-config.jump_exponent)
            aggregate = pulse_aggregate(pulses, jump)
            packet = encode_event_downlink(pulses, jump)
            decoded = decode_event_downlink(packet.data, jump, expected_dimension=d)
            _same_bits(aggregate, decoded)
            counts = decode_pulses(packet.counts.data, expected_dimension=d)
            if not np.array_equal(counts, pulses.astype(np.int32).sum(axis=0)):
                raise AssertionError("signed integer aggregate changed")
            aggregate *= float(config.server_gain)
            decoded *= float(config.server_gain)
            row.update(coordinate_events=int(np.count_nonzero(pulses)),
                       legacy_server_bits=row["legacy_uplink_bits"],
                       codec_replay_bits=row["codec_uplink_bits"],
                       exact_downlink_bits=8 * len(packet.data),
                       integer_only_downlink_bits=packet.counts.wire_bits,
                       aggregate_exact_downlink_bits=packet.aggregate_exact_bits,
                       ordered_downlink_bits=packet.ordered_bits,
                       correction_nonzeros=packet.correction_nonzeros,
                       canonical_max_abs_error=packet.canonical_max_abs_error,
                       aggregate_nonzeros=int(np.count_nonzero(counts)),
                       aggregate_l1=int(np.abs(counts).sum()), downlink_mode=packet.mode,
                       uplink_modes=json.dumps(dict(modes), sort_keys=True))
            if row["coordinate_events"]:
                row["cancellation_fraction"] = 1 - row["aggregate_l1"] / row["coordinate_events"]
        else:
            communicated = stc.step(deltas, weights, server_gain=float(config.server_gain))
            for native in communicated.uplink:
                encoded = encode_stc_lossless(native.data, sizes)
                _same_bits(decode_stc(native.data, sizes), decode_stc_lossless(encoded, sizes))
                row["legacy_uplink_bits"] += native.wire_bits
                row["codec_uplink_bits"] += 8 * len(encoded)
                row["coordinate_events"] += native.nonzeros
            encoded = encode_stc_lossless(communicated.downlink.data, sizes)
            aggregate = communicated.server_update
            decoded = decode_stc_lossless(encoded, sizes)
            _same_bits(aggregate, decoded)
            row.update(legacy_server_bits=communicated.downlink.wire_bits,
                       exact_downlink_bits=8 * len(encoded), active_clients=n,
                       downlink_mode="lossless_stc", aggregate_nonzeros=int(np.count_nonzero(aggregate)))
        row["codec_audit_seconds"] = time.perf_counter() - started
        w += aggregate
        client_model += decoded
        _same_bits(w, client_model)
        if not np.all(np.isfinite(w)):
            raise FloatingPointError("nonfinite pilot model")
        row["model_sha256"] = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
        records.append(row)
        if rnd == 1 or rnd % config.eval_stride == 0 or rnd == config.rounds:
            objective, ce, *_ = metrics(w, federation.X_train_eval, federation.y_train_eval,
                                       layout=layout, regularization=config.regularization)
            history.append({"round": rnd, "train_objective": float(objective), "train_ce": float(ce)})
        if checkpoint is not None and (rnd % checkpoint_stride == 0 or rnd == config.rounds or rnd == stop_after):
            checkpoint.save(identity, arrays, {"completed_round": rnd, "rng_state": rng.bit_generator.state,
                                                "records": records, "history": history})
        if progress_callback is not None:
            progress_callback(row)
        if rnd == stop_after:
            break
    frame = pd.DataFrame(records)
    result = {"rounds": frame, "history": pd.DataFrame(history), "model": w.copy(),
              "rng_state": rng.bit_generator.state,
              "summary": traffic_summary(frame, n_clients=n, dimension=d)}
    return result
