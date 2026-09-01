from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

from .fmnist_multiclass_benchmark import MulticlassFederation
from .fmnist_cnn_benchmark import (
    LAYOUT,
    CompactCNNLayout,
    initialize_cnn,
    loss_and_gradient,
    predictive_metrics,
)


@dataclass
class SparseServerPacket:
    tick: int
    source_client: int
    coordinates: np.ndarray
    deltas: np.ndarray
    payload_bits: int

    def packetized_bits(self, header_bits: int = 64) -> int:
        return int(self.payload_bits + header_bits)


@dataclass
class SparseProtocolTrace:
    method: str
    initial_model: np.ndarray
    final_model: np.ndarray
    packets: list[SparseServerPacket]
    sync_opportunities: list[tuple[int, int, int]]
    metrics: dict[str, float | int | str]
    periods: np.ndarray


def run_sparse_cnn_trace(
    *,
    federation: MulticlassFederation,
    method: Literal["events", "ef_topk"],
    n_ticks: int = 1200,
    seed: int = 60606,
    layout: CompactCNNLayout = LAYOUT,
    init_scale: float = 0.5,
    batch_size: int = 32,
    regularization: float = 1e-4,
    step: float = 0.08,
    topk_fraction: float = 0.025,
    rho: float = 0.999,
    gamma: float = 1.0,
    threshold: float = 0.025,
    jump0: float = 0.01,
    schedule_scale: float = 500.0,
    schedule_exponent: float = 0.5,
    eval_stride: int = 100,
) -> SparseProtocolTrace:
    """Reproduce the 15A sparse learning trajectory while recording server packets.

    The learning dynamics intentionally match ``run_federated_cnn``.  The trace
    adds no feedback into training; it only records the sparse updates from which
    clients could reconstruct the server model.
    """
    rng = np.random.default_rng(seed)
    d = layout.dimension
    address_bits = int(math.ceil(math.log2(d)))
    event_bits = address_bits + 1
    sparse_float_bits = address_bits + 32
    topk = max(1, int(round(topk_fraction * d)))

    w = initialize_cnn(layout=layout, seed=7777, scale=init_scale)
    initial_model = w.copy()
    snapshots = np.repeat(w[None, :], federation.n_clients, axis=0)
    next_completion = federation.periods.copy()
    membrane = np.zeros((federation.n_clients, d), dtype=np.float32)
    residual = np.zeros((federation.n_clients, d), dtype=np.float32)

    packets: list[SparseServerPacket] = []
    sync_opportunities: list[tuple[int, int, int]] = []
    uplink_payload_bits = 0
    uplink_messages = 0
    history_rows: list[dict[str, float]] = []

    for tick in range(1, n_ticks + 1):
        if method == "events":
            membrane *= rho

        active = [i for i in range(federation.n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            local_n = len(federation.client_y[client])
            ids = rng.integers(0, local_n, size=batch_size)
            _, _, gradient = loss_and_gradient(
                snapshots[client],
                federation.client_X[client][ids],
                federation.client_y[client][ids],
                layout=layout,
                regularization=regularization,
                need_gradient=True,
            )
            rate_weight = float(federation.weights[client] * federation.periods[client])

            if method == "events":
                membrane[client] -= gamma * rate_weight * gradient
                mask = np.abs(membrane[client]) >= threshold
                count = int(np.sum(mask))
                if count:
                    jump = jump0 * (1.0 + tick / schedule_scale) ** (-schedule_exponent)
                    coords = np.flatnonzero(mask).astype(np.int32)
                    signs = np.sign(membrane[client, mask]).astype(np.float32)
                    deltas = (jump * signs).astype(np.float32)
                    w[coords] += deltas
                    membrane[client, mask] = 0.0
                    payload = count * event_bits
                    packets.append(
                        SparseServerPacket(
                            tick=tick,
                            source_client=client,
                            coordinates=coords,
                            deltas=deltas,
                            payload_bits=payload,
                        )
                    )
                    uplink_payload_bits += payload
                    uplink_messages += 1
            else:
                residual[client] -= step * rate_weight * gradient
                ids_top = np.argpartition(np.abs(residual[client]), -topk)[-topk:]
                coords = np.sort(ids_top.astype(np.int32))
                deltas = residual[client, coords].copy().astype(np.float32)
                w[coords] += deltas
                residual[client, coords] = 0.0
                payload = topk * sparse_float_bits
                packets.append(
                    SparseServerPacket(
                        tick=tick,
                        source_client=client,
                        coordinates=coords,
                        deltas=deltas,
                        payload_bits=payload,
                    )
                )
                uplink_payload_bits += payload
                uplink_messages += 1

            # This is the same synchronization point implicitly used by the
            # original 15A simulator to start the client's next computation.
            snapshots[client] = w
            sync_opportunities.append((tick, client, len(packets)))
            next_completion[client] = tick + federation.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train_obj, _, _, _, _, _ = predictive_metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=regularization,
            )
            _, test_ce, test_acc, _, worst_acc, _ = predictive_metrics(
                w,
                federation.X_test,
                federation.y_test,
                layout=layout,
                regularization=regularization,
            )
            history_rows.append(
                {
                    "tick": tick,
                    "train_objective": train_obj,
                    "test_ce": test_ce,
                    "test_accuracy": test_acc,
                    "worst_class_accuracy": worst_acc,
                }
            )

    history = pd.DataFrame(history_rows)
    final_train_obj, _, _, _, _, _ = predictive_metrics(
        w,
        federation.X_train_eval,
        federation.y_train_eval,
        layout=layout,
        regularization=regularization,
    )
    _, final_test_ce, final_test_acc, final_macro, final_worst, per_class = predictive_metrics(
        w,
        federation.X_test,
        federation.y_test,
        layout=layout,
        regularization=regularization,
    )
    metrics: dict[str, float | int | str] = {
        "method": method,
        "dimension": d,
        "n_ticks": n_ticks,
        "final_train_objective": final_train_obj,
        "final_test_ce": final_test_ce,
        "final_test_accuracy": final_test_acc,
        "final_macro_accuracy": final_macro,
        "final_worst_class_accuracy": final_worst,
        "whole_train_objective": float(history["train_objective"].mean()),
        "uplink_payload_bits": int(uplink_payload_bits),
        "uplink_messages": int(uplink_messages),
        "n_packets": int(len(packets)),
        "n_sync_opportunities": int(len(sync_opportunities)),
    }
    for cls, acc in enumerate(per_class):
        metrics[f"class_{cls}_accuracy"] = float(acc)

    return SparseProtocolTrace(
        method=method,
        initial_model=initial_model,
        final_model=w.copy(),
        packets=packets,
        sync_opportunities=sync_opportunities,
        metrics=metrics,
        periods=federation.periods.copy(),
    )


def _apply_packets(model: np.ndarray, packets: list[SparseServerPacket], start: int, stop: int) -> None:
    for packet in packets[start:stop]:
        model[packet.coordinates] += packet.deltas


def dense_downlink_accounting(
    trace: SparseProtocolTrace,
    *,
    checkpoint_header_bits: int = 64,
    request_bits: int = 32,
) -> dict[str, float | int | str]:
    """Dense model synchronization at every client completion plus final catch-up."""
    d = len(trace.initial_model)
    checkpoint_payload = 32 * d
    checkpoint_packet = checkpoint_payload + checkpoint_header_bits
    last_packet = np.zeros(len(trace.periods), dtype=np.int64)
    down_payload = 0
    down_packetized = 0
    syncs = 0

    for _, client, packet_index in trace.sync_opportunities:
        down_payload += checkpoint_payload
        down_packetized += checkpoint_packet + request_bits
        last_packet[client] = packet_index
        syncs += 1

    final_index = len(trace.packets)
    final_catchups = 0
    for client in range(len(trace.periods)):
        if last_packet[client] < final_index:
            down_payload += checkpoint_payload
            down_packetized += checkpoint_packet + request_bits
            final_catchups += 1

    return {
        "downlink_mode": "dense_sync",
        "downlink_payload_bits": int(down_payload),
        "downlink_packetized_bits": int(down_packetized),
        "checkpoint_syncs": int(syncs + final_catchups),
        "replay_syncs": 0,
        "empty_syncs": 0,
        "max_sync_error": 0.0,
    }


def simulate_sparse_downlink(
    trace: SparseProtocolTrace,
    *,
    mode: Literal["broadcast", "unicast_replay", "hybrid"],
    packet_header_bits: int = 64,
    checkpoint_header_bits: int = 64,
    request_bits: int = 32,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Simulate client synchronization from the recorded sparse server stream."""
    n_clients = len(trace.periods)
    d = len(trace.initial_model)
    checkpoint_payload = 32 * d
    checkpoint_packet = checkpoint_payload + checkpoint_header_bits

    if mode == "broadcast":
        replay = trace.initial_model.copy()
        _apply_packets(replay, trace.packets, 0, len(trace.packets))
        server_error = float(np.linalg.norm(replay.astype(np.float64) - trace.final_model.astype(np.float64)))
        payload = int(sum(p.payload_bits for p in trace.packets))
        packetized = int(sum(p.packetized_bits(packet_header_bits) for p in trace.packets))
        client_rows = [
            {
                "client": i,
                "period": int(trace.periods[i]),
                "downlink_payload_bits": payload,
                "downlink_packetized_bits": packetized,
                "checkpoints": 0,
                "replays": len(trace.packets),
                "max_sync_error": server_error,
            }
            for i in range(n_clients)
        ]
        # Network cost counts a logical broadcast once, not once per receiving client.
        return (
            {
                "downlink_mode": mode,
                "downlink_payload_bits": payload,
                "downlink_packetized_bits": packetized,
                "checkpoint_syncs": 0,
                "replay_syncs": len(trace.packets),
                "empty_syncs": 0,
                "max_sync_error": server_error,
                "max_replay_packetized_bits": 0,
            },
            pd.DataFrame(client_rows),
        )

    replicas = np.repeat(trace.initial_model[None, :], n_clients, axis=0)
    last_packet = np.zeros(n_clients, dtype=np.int64)
    server_replay = trace.initial_model.copy()
    server_ptr = 0

    down_payload = 0
    down_packetized = 0
    checkpoint_syncs = 0
    replay_syncs = 0
    empty_syncs = 0
    max_error = 0.0
    max_replay_packetized = 0

    client_payload = np.zeros(n_clients, dtype=np.int64)
    client_packetized = np.zeros(n_clients, dtype=np.int64)
    client_checkpoints = np.zeros(n_clients, dtype=np.int64)
    client_replays = np.zeros(n_clients, dtype=np.int64)
    client_errors = np.zeros(n_clients, dtype=float)

    def advance_server(stop: int) -> None:
        nonlocal server_ptr
        _apply_packets(server_replay, trace.packets, server_ptr, stop)
        server_ptr = stop

    def synchronize(client: int, stop: int) -> None:
        nonlocal down_payload, down_packetized, checkpoint_syncs, replay_syncs
        nonlocal empty_syncs, max_error, max_replay_packetized
        advance_server(stop)
        start = int(last_packet[client])
        if start == stop:
            # A small request/ack is still counted for pull-based synchronization.
            down_packetized += request_bits
            client_packetized[client] += request_bits
            empty_syncs += 1
        else:
            missed = trace.packets[start:stop]
            replay_payload = int(sum(p.payload_bits for p in missed))
            replay_packetized = int(sum(p.packetized_bits(packet_header_bits) for p in missed))
            max_replay_packetized = max(max_replay_packetized, replay_packetized)
            use_checkpoint = mode == "hybrid" and replay_packetized > checkpoint_packet
            if use_checkpoint:
                replicas[client] = server_replay
                down_payload += checkpoint_payload
                down_packetized += checkpoint_packet + request_bits
                client_payload[client] += checkpoint_payload
                client_packetized[client] += checkpoint_packet + request_bits
                checkpoint_syncs += 1
                client_checkpoints[client] += 1
            else:
                _apply_packets(replicas[client], trace.packets, start, stop)
                down_payload += replay_payload
                down_packetized += replay_packetized + request_bits
                client_payload[client] += replay_payload
                client_packetized[client] += replay_packetized + request_bits
                replay_syncs += 1
                client_replays[client] += len(missed)
        last_packet[client] = stop
        error = float(np.linalg.norm(replicas[client].astype(np.float64) - server_replay.astype(np.float64)))
        client_errors[client] = max(client_errors[client], error)
        max_error = max(max_error, error)

    for _, client, packet_index in trace.sync_opportunities:
        synchronize(client, packet_index)

    final_index = len(trace.packets)
    advance_server(final_index)
    for client in range(n_clients):
        if last_packet[client] < final_index:
            synchronize(client, final_index)

    final_server_error = float(np.linalg.norm(server_replay.astype(np.float64) - trace.final_model.astype(np.float64)))
    max_error = max(max_error, final_server_error)

    client_rows = []
    for client in range(n_clients):
        client_rows.append(
            {
                "client": client,
                "period": int(trace.periods[client]),
                "downlink_payload_bits": int(client_payload[client]),
                "downlink_packetized_bits": int(client_packetized[client]),
                "checkpoints": int(client_checkpoints[client]),
                "replays": int(client_replays[client]),
                "max_sync_error": float(client_errors[client]),
            }
        )

    return (
        {
            "downlink_mode": mode,
            "downlink_payload_bits": int(down_payload),
            "downlink_packetized_bits": int(down_packetized),
            "checkpoint_syncs": int(checkpoint_syncs),
            "replay_syncs": int(replay_syncs),
            "empty_syncs": int(empty_syncs),
            "max_sync_error": float(max_error),
            "max_replay_packetized_bits": int(max_replay_packetized),
        },
        pd.DataFrame(client_rows),
    )


def q2_total_row(
    trace: SparseProtocolTrace,
    downlink: dict[str, float | int | str],
    *,
    uplink_header_bits: int = 64,
) -> dict[str, float | int | str]:
    uplink_payload = int(trace.metrics["uplink_payload_bits"])
    uplink_packetized = uplink_payload + uplink_header_bits * int(trace.metrics["uplink_messages"])
    down_payload = int(downlink["downlink_payload_bits"])
    down_packetized = int(downlink["downlink_packetized_bits"])
    return {
        "method": trace.method,
        "downlink_mode": str(downlink["downlink_mode"]),
        "final_test_ce": float(trace.metrics["final_test_ce"]),
        "final_test_accuracy": float(trace.metrics["final_test_accuracy"]),
        "final_worst_class_accuracy": float(trace.metrics["final_worst_class_accuracy"]),
        "uplink_payload_bits": uplink_payload,
        "uplink_packetized_bits": int(uplink_packetized),
        "downlink_payload_bits": down_payload,
        "downlink_packetized_bits": down_packetized,
        "total_payload_bits": int(uplink_payload + down_payload),
        "total_packetized_bits": int(uplink_packetized + down_packetized),
        "checkpoint_syncs": int(downlink.get("checkpoint_syncs", 0)),
        "replay_syncs": int(downlink.get("replay_syncs", 0)),
        "empty_syncs": int(downlink.get("empty_syncs", 0)),
        "max_sync_error": float(downlink.get("max_sync_error", 0.0)),
    }
