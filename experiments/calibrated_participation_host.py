"""Five native operators with real wire frames and exact returning-client sync.

Experiment-local: frozen package/hosts are imported without edits. Full
participation preserves their update order and total unicast accounting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments")]
import numpy as np
import pandas as pd
import fast_pulse_codec as codec
import fmnist_mlp_host as pulse_wire
import resnet14_baseline_fast_host as baseline
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.event_codec_pilot import _same_bits
from neuromorphicfl.final_baseline_campaign import _architecture_ops, _local_delta

METHODS = ("event", "strom", "stc", "ef_topk", "dense")
IMPLEMENTATION = "participation_native_operators_exact_frames_v1"
wire = baseline.wire
ternary, encode_ternary = baseline._implementation.ternary, baseline._implementation.encode_ternary


def sha_model(w):
    return hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()


class NumericalFailure(FloatingPointError):
    def __init__(self, rnd, reason):
        self.round = int(rnd)
        super().__init__(f"Round {rnd}: {reason}")


@dataclass
class Frame:
    rnd: int
    data: bytes
    amplitude: float
    gain: float
    active: int
    update: np.ndarray


def decoded_frame(*, rnd, data, method, dimension, sizes, amplitude=1., gain=1., active=1):
    if method in ("event", "strom"):
        update = pulse_wire.wire.decode_event_downlink(data, amplitude, expected_dimension=dimension)
        update *= gain
    elif method == "stc":
        update = codec.decode_stc_lossless(data, sizes)
    else:
        update = wire.decode_downlink(data, expected_dimension=dimension, expected_clients=active, gain=gain)
    if update.dtype != np.float32 or update.shape != (dimension,) or not np.isfinite(update).all():
        raise ValueError("Invalid decoded server frame")
    update.setflags(write=False)
    return Frame(int(rnd), bytes(data), float(amplitude), float(gain), int(active), update)


def step(state, deltas, active, sampling_weights, *, method, config, sizes, rnd):
    """Expose actual packets while retaining the frozen native operators."""
    dimension = sum(sizes)
    gain = float(config.server_gain)*(1+rnd/config.server_gain_scale)**(-config.server_gain_exponent)
    aggregate = np.zeros(dimension, dtype=np.float32)
    packets, events = [], 0
    amplitude = 1.
    if method in ("event", "strom"):
        pulses = np.zeros((len(active), dimension), dtype=np.int8)
        threshold = config.threshold if method == "event" else config.strom_threshold
        for i, (client, delta) in enumerate(zip(active, deltas)):
            if method == "event":
                gap = rnd-int(state["last_state_round"][client])
                state["client_state"][client] *= config.rho**gap
            state["client_state"][client] += (1./config.local_lr)*float(sampling_weights[client])*delta
            if not np.isfinite(state["client_state"][client]).all():
                raise FloatingPointError("Nonfinite pulse evidence/residual")
            mask = np.abs(state["client_state"][client]) >= threshold
            signs = np.sign(state["client_state"][client, mask]).astype(np.int8)
            pulses[i, mask] = signs
            if method == "event":
                state["client_state"][client, mask] = 0.
            else:
                state["client_state"][client, mask] -= threshold*signs.astype(np.float32)
            state["last_state_round"][client] = rnd
            if np.any(pulses[i]):
                packet = codec.encode_pulses(pulses[i])
                received = codec.decode_pulses(packet.data, expected_dimension=dimension)
                if not np.array_equal(pulses[i], received):
                    raise AssertionError("Client pulse upload changed")
                pulses[i] = received
                packets.append(packet.data)
        amplitude = (config.jump0*(1+rnd/config.jump_scale)**(-config.jump_exponent)
                     if method == "event" else config.strom_threshold)
        aggregate = pulse_wire.native_wire.pulse_aggregate(pulses, amplitude)
        packet = pulse_wire.encode_downlink(pulses, amplitude)
        data, mode = packet.data, packet.mode
        aggregate_bits, ordered_bits = packet.aggregate_exact_bits, packet.ordered_bits
        events = int(np.count_nonzero(pulses))
        aggregate *= gain
    else:
        for client, delta in zip(active, deltas):
            if method == "dense":
                message = delta
                data, _ = wire.encode_vector(message, dense_only=True)
                received = wire.decode_vector(data, expected_dimension=dimension)
                aggregate += float(sampling_weights[client])*received
            elif method == "ef_topk":
                state["client_state"][client] += float(sampling_weights[client])*delta
                if not np.isfinite(state["client_state"][client]).all():
                    raise FloatingPointError("Nonfinite EF residual")
                k = max(1, int(round(config.topk_fraction*dimension)))
                ids = np.argpartition(np.abs(state["client_state"][client]), -k)[-k:]
                message = np.zeros(dimension, dtype=np.float32)
                message[ids] = state["client_state"][client, ids]
                data, _ = wire.encode_vector(message)
                received = wire.decode_vector(data, expected_dimension=dimension)
                aggregate[ids] += received[ids]
                state["client_state"][client, ids] = 0.
            elif method == "stc":
                # STC's client residual is UNWEIGHTED, as in its native host.
                corrected = state["client_state"][client]+delta
                if not np.isfinite(corrected).all():
                    raise FloatingPointError("Nonfinite STC client residual")
                message = ternary(corrected, sizes, config.stc_upload_fraction)
                if not np.isfinite(message).all():
                    raise FloatingPointError("Nonfinite STC client ternarization")
                data, received = encode_ternary(message, sizes)
                state["client_state"][client] = corrected-received
                aggregate += float(sampling_weights[client])*received
            else:
                raise ValueError("Unknown operator")
            _same_bits(message, received)
            packets.append(data)
            events += int(np.count_nonzero(message))
        aggregate *= gain
        if not np.isfinite(aggregate).all():
            raise FloatingPointError("Nonfinite server aggregate")
        if method == "stc":
            corrected = state["server_residual"]+aggregate
            if not np.isfinite(corrected).all():
                raise FloatingPointError("Nonfinite STC server residual")
            aggregate = ternary(corrected, sizes, config.stc_download_fraction)
            if not np.isfinite(aggregate).all():
                raise FloatingPointError("Nonfinite STC server ternarization")
            data, received = encode_ternary(aggregate, sizes)
            state["server_residual"][:] = corrected-received
            mode, aggregate_bits, ordered_bits = "lossless_stc", 8*len(data), 0
        elif method == "dense":
            data, mode = wire.encode_vector(aggregate, dense_only=True)
            aggregate_bits, ordered_bits = 8*len(data), 0
        else:
            data, _, mode, aggregate_bits, ordered_bits = wire.downlink(aggregate, packets, gain=gain)
    if not np.isfinite(aggregate).all():
        raise FloatingPointError("Nonfinite model update")
    frame = decoded_frame(rnd=rnd, data=data, method=method, dimension=dimension, sizes=sizes,
                          amplitude=amplitude, gain=gain, active=len(active))
    _same_bits(aggregate, frame.update)
    return frame, {"codec_uplink_bits": sum(8*len(p) for p in packets),
        "exact_downlink_bits": 8*len(data), "aggregate_downlink_bits": int(aggregate_bits),
        "ordered_downlink_bits": int(ordered_bits), "coordinate_events": events,
        "nonempty_uplink_messages": len(packets), "downlink_mode": mode, "server_gain": gain}


class Journal:
    """Ordered server frames, pruned only when a checkpoint is always cheaper."""
    def __init__(self, *, method, dimension, sizes):
        self.method, self.dimension, self.sizes = method, dimension, tuple(sizes)
        self.checkpoint_bits = 32*dimension+64
        self.prefix = [0]
        self.frames = {}

    def append(self, frame):
        if frame.rnd != len(self.prefix):
            raise ValueError("Nonconsecutive journal")
        self.prefix.append(self.prefix[-1]+8*len(frame.data))
        self.frames[frame.rnd] = frame
        while self.frames:
            first = next(iter(self.frames))
            if self.prefix[-1]-self.prefix[first-1] <= self.checkpoint_bits:
                break
            del self.frames[first]

    def synchronize(self, client_model, last, server_model):
        now = len(self.prefix)-1
        if not 0 <= last <= now:
            raise ValueError("Invalid client synchronization cursor")
        missed = self.prefix[now]-self.prefix[last]
        if last == now:
            _same_bits(client_model, server_model)
            return 0, "current"
        if missed <= self.checkpoint_bits:
            for rnd in range(last+1, now+1):
                if rnd not in self.frames:
                    raise AssertionError("Pruned a required replay frame")
                client_model += self.frames[rnd].update
            bits, mode = missed, "replay"
        else:
            data, _ = wire.encode_vector(server_model, dense_only=True)
            decoded = wire.decode_vector(data, expected_dimension=self.dimension)
            _same_bits(server_model, decoded)
            client_model[:] = decoded
            bits, mode = len(data)*8, "checkpoint"
        _same_bits(client_model, server_model)
        return int(bits), mode

    def prune_synchronized(self, cursors):
        """No client can need frames it has already incorporated."""
        minimum = int(np.min(cursors))
        while self.frames and next(iter(self.frames)) <= minimum:
            del self.frames[next(iter(self.frames))]

    def terminal(self, client_models, cursors, server_model):
        """Audit exact final deployment without mutating any training state."""
        now = len(self.prefix)-1
        costs = {"terminal_downlink_bits": 0, "terminal_request_bits": 0,
                 "terminal_replay_clients": 0, "terminal_checkpoint_clients": 0}
        for model, last in zip(client_models, cursors):
            candidate = model.copy()
            bits, mode = self.synchronize(candidate, int(last), server_model)
            costs["terminal_downlink_bits"] += bits
            if last != now:
                costs["terminal_request_bits"] += 32
            if mode != "current":
                costs["terminal_"+mode+"_clients"] += 1
        costs["terminal_catchup_bits"] = costs["terminal_downlink_bits"]+costs["terminal_request_bits"]
        return costs

    def dump(self):
        payload, entries = bytearray(), []
        for frame in self.frames.values():
            entries.append({"round": frame.rnd, "offset": len(payload), "size": len(frame.data),
                "amplitude": frame.amplitude, "gain": frame.gain, "active": frame.active,
                "sha256": hashlib.sha256(frame.data).hexdigest()})
            payload.extend(frame.data)
        return np.frombuffer(bytes(payload), dtype=np.uint8).copy(), {"prefix": self.prefix, "entries": entries}

    def restore(self, raw, metadata):
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("Invalid checkpoint journal bytes")
        self.prefix = [int(x) for x in metadata["prefix"]]
        if not self.prefix or self.prefix[0] != 0 or np.any(np.diff(self.prefix) < 0):
            raise ValueError("Invalid journal prefix")
        self.frames, offset, previous = {}, 0, None
        for row in metadata["entries"]:
            if row["offset"] != offset or row["size"] <= 0:
                raise ValueError("Invalid journal offsets")
            data = raw[offset:offset+row["size"]].tobytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Checkpoint journal changed")
            rnd = int(row["round"])
            if not 1 <= rnd < len(self.prefix) or (previous is not None and rnd != previous+1):
                raise ValueError("Invalid journal round order")
            if self.prefix[rnd]-self.prefix[rnd-1] != len(data)*8:
                raise ValueError("Checkpoint packet length differs")
            self.frames[rnd] = decoded_frame(rnd=rnd, data=data, method=self.method,
                dimension=self.dimension, sizes=self.sizes, amplitude=row["amplitude"],
                gain=row["gain"], active=row["active"])
            offset += row["size"]
            previous = rnd
        if offset != len(raw):
            raise ValueError("Trailing checkpoint journal bytes")


def validate_schedule(schedule, *, n, horizon):
    schedule = np.asarray(schedule)
    if (schedule.ndim != 2 or schedule.shape[0] != horizon or not 1 <= schedule.shape[1] <= n
            or not np.issubdtype(schedule.dtype, np.integer) or np.any(schedule < 0) or np.any(schedule >= n)
            or np.any(np.diff(schedule, axis=1) <= 0)):
        raise ValueError("Expected sorted, distinct in-range clients in every round")
    return schedule.astype(np.int64, copy=False)


def run(*, federation, architecture, method, config, schedule, seed, horizon,
        scope, checkpoint_path=None, provenance=None, stop_after=None, callback=None):
    if method not in METHODS or scope not in ("validation", "test"):
        raise ValueError("Unknown method or predictive scope")
    if (not 1 <= horizon <= config.rounds or config.eval_stride <= 0 or config.local_steps <= 0
            or config.batch_size <= 0):
        raise ValueError("Invalid horizon or local computation")
    numeric = [config.local_lr, config.regularization, config.server_gain, config.server_gain_scale,
        config.server_gain_exponent, config.rho, config.threshold, config.jump0, config.jump_scale,
        config.jump_exponent, config.strom_threshold, config.topk_fraction,
        config.stc_upload_fraction, config.stc_download_fraction]
    if (not np.isfinite(numeric).all() or config.local_lr <= 0 or config.regularization < 0
            or config.server_gain <= 0 or config.server_gain_scale <= 0 or config.server_gain_exponent < 0
            or not 0 <= config.rho <= 1 or min(config.threshold, config.jump0, config.jump_scale, config.strom_threshold) <= 0
            or config.jump_exponent < 0 or not all(0 < f <= 1 for f in
            (config.topk_fraction, config.stc_upload_fraction, config.stc_download_fraction))
            or config.event_memory != "persistent" or config.event_reset != "full"
            or (method != "dense" and config.server_gain_exponent != 0)):
        raise ValueError("Invalid fixed operator parameters")
    if stop_after is not None and not 1 <= stop_after <= horizon:
        raise ValueError("Invalid interruption round")
    layout, initialize, loss_grad, metrics = _architecture_ops(architecture)
    n, d = federation.n_clients, int(layout.dimension)
    sizes = tuple(size for _, _, size in layout.groups())
    schedule = validate_schedule(schedule, n=n, horizon=horizon)
    k = schedule.shape[1]
    weights = np.asarray(federation.weights, dtype=float).copy()
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive client weights required")
    weights /= weights.sum()
    if not np.allclose(weights, 1/n, rtol=0, atol=1e-14):
        raise ValueError("This frozen study uses equal-sized clients only")
    sampling_weights = weights/(k/n)
    rng = np.random.default_rng(seed)
    w = initialize(layout=layout, seed=7777, scale=config.init_scale)
    initial_frame, _ = wire.encode_vector(w, dense_only=True)
    initial_model = wire.decode_vector(initial_frame, expected_dimension=d)
    _same_bits(w, initial_model)
    arrays = {"model": w, "client_models": np.repeat(initial_model[None, :], n, axis=0),
        "client_state": np.zeros((n, d), dtype=np.float32),
        "server_residual": np.zeros(d, dtype=np.float32),
        "last_state_round": np.zeros(n, dtype=np.int64),
        "last_sync_round": np.zeros(n, dtype=np.int64)}
    journal = Journal(method=method, dimension=d, sizes=sizes)
    identity = {"host": IMPLEMENTATION, "architecture": architecture, "method": method,
        "config": asdict(config), "horizon": horizon, "scope": scope, "seed": int(seed),
        "n_clients": n, "dimension": d, "weights": weights.tolist(),
        "schedule_sha256": hashlib.sha256(schedule.astype("<i2").tobytes()).hexdigest(),
        "numpy": np.__version__}
    checkpoint = BaselineCheckpoint(Path(checkpoint_path), provenance or {}) if checkpoint_path else None
    records, evaluations, start = [], [], 0
    expected_evals = sorted({1, horizon, *range(config.eval_stride, horizon+1, config.eval_stride)})
    if checkpoint is not None:
        saved = checkpoint.load(identity)
        if saved is not None:
            if set(saved["arrays"]) != set(arrays)|{"journal_wire"}:
                raise ValueError("Checkpoint state inventory differs")
            for name, dest in arrays.items():
                source = saved["arrays"][name]
                if source.shape != dest.shape or source.dtype != dest.dtype or not np.isfinite(source).all():
                    raise ValueError("Invalid saved "+name)
                dest[:] = source
            p = saved["progress"]
            start, records, evaluations = int(p["completed_round"]), p["records"], p["evaluations"]
            rng.bit_generator.state = p["rng_state"]
            journal.restore(saved["arrays"]["journal_wire"], p["journal"])
            if (not 0 < start <= horizon or len(journal.prefix) != start+1
                    or [r["round"] for r in records] != list(range(1, start+1))
                    or [r["round"] for r in evaluations] != [r for r in expected_evals if r <= start]
                    or records[-1]["model_sha256"] != sha_model(w)
                    or np.any(arrays["last_sync_round"] > start) or np.any(arrays["last_sync_round"] < 0)
                    or np.any(arrays["last_state_round"] > start) or np.any(arrays["last_state_round"] < 0)):
                raise ValueError("Checkpoint progress differs")
            # Reconstruct from every saved cursor before accepting a restart.
            journal.terminal(arrays["client_models"], arrays["last_sync_round"], w)
    totals = ({key: int(records[-1][key]) for key in (
        "cumulative_update_uplink_bits", "cumulative_request_bits", "cumulative_current_downlink_bits",
        "cumulative_catchup_replay_bits", "cumulative_catchup_checkpoint_bits")}
        if records else {key: 0 for key in ("cumulative_update_uplink_bits", "cumulative_request_bits",
            "cumulative_current_downlink_bits", "cumulative_catchup_replay_bits", "cumulative_catchup_checkpoint_bits")})
    initial = n*len(initial_frame)*8
    last_round = min(horizon, stop_after) if stop_after is not None else horizon
    for rnd in range(start+1, last_round+1):
        active = schedule[rnd-1]
        catchup = {"replay": 0, "checkpoint": 0}
        sync_counts = {"replay": 0, "checkpoint": 0, "current": 0}
        for client in active:
            bits, mode = journal.synchronize(arrays["client_models"][client],
                int(arrays["last_sync_round"][client]), w)
            if mode != "current":
                catchup[mode] += bits
            sync_counts[mode] += 1
            arrays["last_sync_round"][client] = rnd-1
        deltas = [_local_delta(arrays["client_models"][client], federation.client_X[client],
            federation.client_y[client], config=config, rng=rng, layout=layout, loss_grad=loss_grad)
            for client in active]
        if not all(np.isfinite(delta).all() for delta in deltas):
            raise NumericalFailure(rnd, "nonfinite local SGD")
        try:
            frame, wire_row = step(arrays, deltas, active, sampling_weights,
                method=method, config=config, sizes=sizes, rnd=rnd)
        except FloatingPointError as exc:
            raise NumericalFailure(rnd, str(exc)) from exc
        w += frame.update
        arrays["client_models"][active] += frame.update
        arrays["last_sync_round"][active] = rnd
        _same_bits(arrays["client_models"][active], np.broadcast_to(w, (k, d)))
        if not np.isfinite(w).all() or not np.isfinite(arrays["client_state"][active]).all():
            raise NumericalFailure(rnd, "nonfinite server or residual state")
        journal.append(frame)
        journal.prune_synchronized(arrays["last_sync_round"])
        totals["cumulative_update_uplink_bits"] += wire_row["codec_uplink_bits"]
        totals["cumulative_request_bits"] += 32*k
        totals["cumulative_current_downlink_bits"] += k*wire_row["exact_downlink_bits"]
        totals["cumulative_catchup_replay_bits"] += catchup["replay"]
        totals["cumulative_catchup_checkpoint_bits"] += catchup["checkpoint"]
        operational = initial+sum(totals.values())
        records.append({"round": rnd, "method": method, **wire_row, **totals,
            "selected_clients": k, "catchup_replay_bits": catchup["replay"],
            "catchup_checkpoint_bits": catchup["checkpoint"], "catchup_replay_clients": sync_counts["replay"],
            "catchup_checkpoint_clients": sync_counts["checkpoint"], "current_clients": sync_counts["current"],
            "request_bits": 32*k, "current_downlink_bits": k*wire_row["exact_downlink_bits"],
            "operational_total_bits": operational, "active_client_rounds": rnd*k,
            "local_sgd_steps": rnd*k*config.local_steps,
            "sampled_training_examples": rnd*k*config.local_steps*config.batch_size,
            "journal_bytes": sum(len(f.data) for f in journal.frames.values()),
            "model_sha256": sha_model(w), "reconstruction_verified": True})
        if rnd in expected_evals:
            train_objective, train_ce, *_ = metrics(w, federation.X_train_eval, federation.y_train_eval,
                layout=layout, regularization=config.regularization)
            _, ce, accuracy, macro, worst, _ = metrics(w, federation.X_test, federation.y_test,
                layout=layout, regularization=config.regularization)
            if not np.isfinite([train_objective, train_ce, ce, accuracy, macro, worst]).all():
                raise NumericalFailure(rnd, "nonfinite predictive endpoint")
            terminal = journal.terminal(arrays["client_models"], arrays["last_sync_round"], w)
            downlink = (initial+totals["cumulative_current_downlink_bits"]
                +totals["cumulative_catchup_replay_bits"]+totals["cumulative_catchup_checkpoint_bits"]
                +terminal["terminal_downlink_bits"])
            uplink = (totals["cumulative_update_uplink_bits"]+totals["cumulative_request_bits"]
                +terminal["terminal_request_bits"])
            if uplink+downlink != operational+terminal["terminal_catchup_bits"]:
                raise AssertionError("Conservative communication components do not balance")
            evaluations.append({"round": rnd, "model_sha256": sha_model(w),
                "train_objective": float(train_objective), "train_ce": float(train_ce),
                scope+"_ce": float(ce), scope+"_accuracy": float(accuracy),
                scope+"_macro_accuracy": float(macro), scope+"_worst_class_accuracy": float(worst),
                **totals, **terminal, "initial_sync_bits": initial, "uplink_bits": uplink,
                "unicast_downlink_bits": downlink, "unicast_total_bits": uplink+downlink,
                "operational_total_bits": operational, "active_client_rounds": rnd*k,
                "local_sgd_steps": rnd*k*config.local_steps, "terminal_verified_clients": n})
        if checkpoint is not None and (rnd in expected_evals or rnd == last_round):
            raw, metadata = journal.dump()
            checkpoint.save(identity, {**arrays, "journal_wire": raw}, {"completed_round": rnd,
                "records": records, "evaluations": evaluations, "rng_state": rng.bit_generator.state,
                "journal": metadata})
        if callback:
            callback(records[-1], evaluations[-1] if evaluations else None)
    return {"rounds": pd.DataFrame(records), "history": pd.DataFrame(evaluations),
        "arrays": {k: a.copy() for k, a in arrays.items()}, "rng_state": rng.bit_generator.state}
