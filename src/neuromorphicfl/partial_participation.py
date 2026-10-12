"""Synchronous partial-participation runner with explicit client catch-up.

This module extends the frozen compact-CNN protocol without changing its local
SGD, compressors, or optimizer parameters.  Only selected clients compute and
communicate in a round.  Returning clients are charged for exact ordered replay
of every missed server packet, or for a dense checkpoint when that is smaller.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import numpy as np
import pandas as pd

from .final_baseline_campaign import (
    FinalBaselineConfig,
    _architecture_ops,
    _local_delta,
)
from .fmnist_multiclass_benchmark import MulticlassFederation


PartialMethod = Literal["event", "ef_topk", "dense"]


def participation_schedule(
    *, n_rounds: int, n_clients: int, active_clients: int, seed: int
) -> np.ndarray:
    """Create a deterministic uniform-without-replacement round schedule."""
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    if not 1 <= active_clients <= n_clients:
        raise ValueError("active_clients must lie in [1, n_clients]")
    if active_clients == n_clients:
        return np.tile(
            np.arange(n_clients, dtype=np.int16), (n_rounds, 1)
        )
    rng = np.random.default_rng(seed)
    schedule = np.empty((n_rounds, active_clients), dtype=np.int16)
    for round_index in range(n_rounds):
        schedule[round_index] = np.sort(
            rng.choice(n_clients, size=active_clients, replace=False)
        )
    return schedule


def schedule_sha256(schedule: np.ndarray) -> str:
    normalized = np.asarray(schedule, dtype="<i2", order="C")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _validate_schedule(
    schedule: np.ndarray, *, n_rounds: int, n_clients: int
) -> np.ndarray:
    selected = np.asarray(schedule, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[0] != n_rounds:
        raise ValueError("schedule must have shape (rounds, active_clients)")
    if selected.shape[1] == 0:
        raise ValueError("schedule must select at least one client per round")
    if np.any(selected < 0) or np.any(selected >= n_clients):
        raise ValueError("schedule contains an invalid client index")
    if any(len(np.unique(row)) != len(row) for row in selected):
        raise ValueError("schedule selects a client more than once in a round")
    return selected


def _hybrid_sync_cost(
    replay_packetized_bits: int, checkpoint_packet_bits: int
) -> tuple[int, str]:
    if replay_packetized_bits <= 0:
        return 0, "empty"
    if replay_packetized_bits <= checkpoint_packet_bits:
        return int(replay_packetized_bits), "replay"
    return int(checkpoint_packet_bits), "checkpoint"


def _terminal_catchup_cost(
    *,
    prefix_replay_bits: list[int],
    client_last_sync_round: np.ndarray,
    checkpoint_packet_bits: int,
    request_bits: int,
) -> tuple[int, int, int]:
    current_round = len(prefix_replay_bits) - 1
    total = 0
    replay_syncs = 0
    checkpoint_syncs = 0
    for last_sync in client_last_sync_round:
        if int(last_sync) >= current_round:
            continue
        missed = (
            prefix_replay_bits[current_round]
            - prefix_replay_bits[int(last_sync)]
        )
        data_bits, mode = _hybrid_sync_cost(missed, checkpoint_packet_bits)
        total += request_bits + data_bits
        replay_syncs += int(mode == "replay")
        checkpoint_syncs += int(mode == "checkpoint")
    return total, replay_syncs, checkpoint_syncs


def run_partial_participation(
    *,
    federation: MulticlassFederation,
    architecture: Literal["cifar_cnn"],
    method: PartialMethod,
    config: FinalBaselineConfig,
    schedule: np.ndarray,
    seed: int,
    record_history: bool = True,
    record_activity: bool = False,
) -> dict[str, float | int | str | pd.DataFrame]:
    """Run one matched partial-participation experiment.

    The estimator uses ``p_i / pi_i`` before compression.  With equal client
    weights and uniform fixed-size sampling this is ``1 / active_clients``.
    Event states are decayed lazily by ``rho ** gap`` when a client returns.
    The reported ``unicast_hybrid_total_bits`` includes a terminal catch-up so
    every client can reconstruct the exact final server model.  The operational
    total before that final audit synchronization is reported separately.
    """
    if method not in ("event", "ef_topk", "dense"):
        raise ValueError(f"unsupported partial-participation method: {method}")
    if not np.isfinite(config.server_gain) or config.server_gain <= 0.0:
        raise ValueError("server_gain must be finite and positive")

    layout, initialize, loss_grad, metrics = _architecture_ops(architecture)
    n_clients = federation.n_clients
    selected = _validate_schedule(
        schedule, n_rounds=config.rounds, n_clients=n_clients
    )
    active_clients = int(selected.shape[1])
    inclusion_probability = active_clients / n_clients
    weights = federation.weights.astype(float)
    weights /= np.sum(weights)
    sampling_weights = weights / inclusion_probability

    d = int(layout.dimension)
    address_bits = int(math.ceil(math.log2(d)))
    pulse_bits = address_bits + 1
    topk = max(1, int(round(config.topk_fraction * d)))
    checkpoint_packet_bits = 32 * d + 64
    request_bits = 32
    evidence_gain = 1.0 / config.local_lr

    rng = np.random.default_rng(seed)
    w = initialize(layout=layout, seed=7777, scale=config.init_scale)
    event_state = np.zeros((n_clients, d), dtype=np.float32)
    residual = np.zeros((n_clients, d), dtype=np.float32)
    last_state_round = np.zeros(n_clients, dtype=np.int64)
    last_active_round = np.zeros(n_clients, dtype=np.int64)
    client_last_sync_round = np.zeros(n_clients, dtype=np.int64)

    # The initial dense model is distributed once to every registered client.
    initial_sync_bits = n_clients * checkpoint_packet_bits
    operational_downlink_bits = int(initial_sync_bits)
    request_total_bits = 0
    replay_downlink_bits = 0
    checkpoint_downlink_bits = int(initial_sync_bits)
    catchup_replay_syncs = 0
    catchup_checkpoint_syncs = 0
    round_replay_syncs = 0
    round_checkpoint_syncs = 0
    empty_syncs = 0

    uplink_payload_bits = 0
    uplink_packetized_bits = 0
    messages = 0
    coordinate_events = 0
    active_client_rounds = 0
    delta_norms: list[float] = []
    prefix_replay_bits = [0]
    history_rows: list[dict[str, float | int]] = []
    activity_rows: list[dict[str, float | int | str]] = []
    return_event_counts: list[int] = []
    consecutive_event_counts: list[int] = []

    for rnd in range(1, config.rounds + 1):
        active = selected[rnd - 1]
        active_client_rounds += active_clients
        round_request = active_clients * request_bits
        request_total_bits += round_request
        operational_downlink_bits += round_request

        pre_sync_bits: dict[int, int] = {}
        pre_sync_modes: dict[int, str] = {}
        gaps: dict[int, int] = {}
        first_participation: dict[int, bool] = {}
        for client_value in active:
            client = int(client_value)
            missed = (
                prefix_replay_bits[rnd - 1]
                - prefix_replay_bits[int(client_last_sync_round[client])]
            )
            sync_bits, sync_mode = _hybrid_sync_cost(
                missed, checkpoint_packet_bits
            )
            operational_downlink_bits += sync_bits
            pre_sync_bits[client] = sync_bits
            pre_sync_modes[client] = sync_mode
            replay_downlink_bits += sync_bits if sync_mode == "replay" else 0
            checkpoint_downlink_bits += (
                sync_bits if sync_mode == "checkpoint" else 0
            )
            catchup_replay_syncs += int(sync_mode == "replay")
            catchup_checkpoint_syncs += int(sync_mode == "checkpoint")
            empty_syncs += int(sync_mode == "empty")
            client_last_sync_round[client] = rnd - 1

            previous = int(last_active_round[client])
            gap = rnd - previous
            gaps[client] = gap
            first_participation[client] = previous == 0

        deltas: dict[int, np.ndarray] = {}
        for client_value in active:
            client = int(client_value)
            delta = _local_delta(
                w,
                federation.client_X[client],
                federation.client_y[client],
                config=config,
                rng=rng,
                layout=layout,
                loss_grad=loss_grad,
            )
            deltas[client] = delta
            delta_norms.append(float(np.linalg.norm(delta)))

        aggregate = np.zeros(d, dtype=np.float32)
        client_coordinate_counts: dict[int, int] = {}
        client_uplink_bits: dict[int, int] = {}
        replay_this_round = 0
        round_messages = 0
        round_coordinate_events = 0

        if method == "event":
            jump = config.jump0 * (
                1.0 + rnd / config.jump_scale
            ) ** (-config.jump_exponent)
            for client_value in active:
                client = int(client_value)
                decay_gap = rnd - int(last_state_round[client])
                event_state[client] *= config.rho ** decay_gap
                event_state[client] += (
                    evidence_gain * sampling_weights[client] * deltas[client]
                )
                mask = np.abs(event_state[client]) >= config.threshold
                count = int(np.sum(mask))
                packet_bits = 0
                if count:
                    signs = np.sign(event_state[client, mask]).astype(np.int8)
                    aggregate[mask] += jump * signs.astype(np.float32)
                    event_state[client, mask] = 0.0
                    payload_bits = count * pulse_bits
                    packet_bits = payload_bits + 64
                    uplink_payload_bits += payload_bits
                    uplink_packetized_bits += packet_bits
                    replay_this_round += packet_bits
                    messages += 1
                    round_messages += 1
                    coordinate_events += count
                    round_coordinate_events += count
                client_coordinate_counts[client] = count
                client_uplink_bits[client] = packet_bits
                last_state_round[client] = rnd

                if not first_participation[client]:
                    if gaps[client] > 1:
                        return_event_counts.append(count)
                    else:
                        consecutive_event_counts.append(count)

        elif method == "ef_topk":
            for client_value in active:
                client = int(client_value)
                residual[client] += sampling_weights[client] * deltas[client]
                ids = np.argpartition(np.abs(residual[client]), -topk)[-topk:]
                values = residual[client, ids].copy()
                aggregate[ids] += values
                residual[client, ids] = 0.0
                payload_bits = topk * (32 + address_bits)
                packet_bits = payload_bits + 64
                uplink_payload_bits += payload_bits
                uplink_packetized_bits += packet_bits
                replay_this_round += packet_bits
                messages += 1
                round_messages += 1
                client_coordinate_counts[client] = topk
                client_uplink_bits[client] = packet_bits

        else:
            for client_value in active:
                client = int(client_value)
                aggregate += sampling_weights[client] * deltas[client]
                payload_bits = 32 * d
                packet_bits = payload_bits + 64
                uplink_payload_bits += payload_bits
                uplink_packetized_bits += packet_bits
                replay_this_round += packet_bits
                messages += 1
                round_messages += 1
                client_coordinate_counts[client] = d
                client_uplink_bits[client] = packet_bits

        aggregate *= float(config.server_gain)
        w += aggregate
        prefix_replay_bits.append(
            prefix_replay_bits[-1] + int(replay_this_round)
        )

        post_sync_bits, post_sync_mode = _hybrid_sync_cost(
            replay_this_round, checkpoint_packet_bits
        )
        operational_downlink_bits += active_clients * post_sync_bits
        replay_downlink_bits += (
            active_clients * post_sync_bits if post_sync_mode == "replay" else 0
        )
        checkpoint_downlink_bits += (
            active_clients * post_sync_bits
            if post_sync_mode == "checkpoint"
            else 0
        )
        round_replay_syncs += (
            active_clients if post_sync_mode == "replay" else 0
        )
        round_checkpoint_syncs += (
            active_clients if post_sync_mode == "checkpoint" else 0
        )
        empty_syncs += active_clients if post_sync_mode == "empty" else 0

        for client_value in active:
            client = int(client_value)
            client_last_sync_round[client] = rnd
            if record_activity:
                activity_rows.append(
                    {
                        "round": rnd,
                        "client": client,
                        "first_participation": int(first_participation[client]),
                        "participation_gap": gaps[client],
                        "inactive_rounds": max(0, gaps[client] - 1),
                        "coordinate_count": client_coordinate_counts[client],
                        "uplink_packetized_bits": client_uplink_bits[client],
                        "catchup_mode": pre_sync_modes[client],
                        "catchup_bits": pre_sync_bits[client],
                        "round_downlink_mode": post_sync_mode,
                        "round_downlink_bits": post_sync_bits,
                    }
                )
            last_active_round[client] = rnd

        if rnd == 1 or rnd % config.eval_stride == 0 or rnd == config.rounds:
            train_obj, train_ce, _, _, _, _ = metrics(
                w,
                federation.X_train_eval,
                federation.y_train_eval,
                layout=layout,
                regularization=config.regularization,
            )
            _, test_ce, test_acc, macro_acc, worst_acc, _ = metrics(
                w,
                federation.X_test,
                federation.y_test,
                layout=layout,
                regularization=config.regularization,
            )
            stop_catchup, _, _ = _terminal_catchup_cost(
                prefix_replay_bits=prefix_replay_bits,
                client_last_sync_round=client_last_sync_round,
                checkpoint_packet_bits=checkpoint_packet_bits,
                request_bits=request_bits,
            )
            history_rows.append(
                {
                    "round": rnd,
                    "train_objective": train_obj,
                    "train_ce": train_ce,
                    "test_ce": test_ce,
                    "test_accuracy": test_acc,
                    "macro_accuracy": macro_acc,
                    "worst_class_accuracy": worst_acc,
                    "uplink_packetized_bits": uplink_packetized_bits,
                    "operational_unicast_hybrid_downlink_bits": (
                        operational_downlink_bits
                    ),
                    "terminal_catchup_downlink_bits_if_stop": stop_catchup,
                    "operational_total_bits": (
                        uplink_packetized_bits + operational_downlink_bits
                    ),
                    "unicast_hybrid_total_bits": (
                        uplink_packetized_bits
                        + operational_downlink_bits
                        + stop_catchup
                    ),
                    "coordinate_events": coordinate_events,
                    "round_coordinate_events": round_coordinate_events,
                    "round_messages": round_messages,
                    "round_replay_packetized_bits": replay_this_round,
                    "server_update_norm": float(np.linalg.norm(aggregate)),
                    "active_clients": active_clients,
                }
            )

    history = pd.DataFrame(history_rows)
    train_obj, train_ce, _, _, _, _ = metrics(
        w,
        federation.X_train_eval,
        federation.y_train_eval,
        layout=layout,
        regularization=config.regularization,
    )
    _, test_ce, test_acc, macro_acc, worst_acc, per_class = metrics(
        w,
        federation.X_test,
        federation.y_test,
        layout=layout,
        regularization=config.regularization,
    )

    terminal_catchup_bits, terminal_replay_syncs, terminal_checkpoint_syncs = (
        _terminal_catchup_cost(
            prefix_replay_bits=prefix_replay_bits,
            client_last_sync_round=client_last_sync_round,
            checkpoint_packet_bits=checkpoint_packet_bits,
            request_bits=request_bits,
        )
    )
    operational_total_bits = (
        uplink_packetized_bits + operational_downlink_bits
    )
    exact_final_total_bits = operational_total_bits + terminal_catchup_bits
    participation_counts = np.bincount(
        selected.reshape(-1), minlength=n_clients
    )

    activity = pd.DataFrame(activity_rows)
    mean_inactive_rounds = (
        float(activity["inactive_rounds"].mean())
        if len(activity)
        else float("nan")
    )
    max_inactive_rounds = (
        int(activity["inactive_rounds"].max()) if len(activity) else 0
    )
    mean_return_events = (
        float(np.mean(return_event_counts)) if return_event_counts else 0.0
    )
    mean_consecutive_events = (
        float(np.mean(consecutive_event_counts))
        if consecutive_event_counts
        else 0.0
    )

    result: dict[str, float | int | str | pd.DataFrame] = {
        "architecture": architecture,
        "method": method,
        "dimension": d,
        "n_clients": n_clients,
        "active_clients": active_clients,
        "participation_rate": float(inclusion_probability),
        "sampling_estimator": "p_i_over_pi_i",
        "selection_schedule_sha256": schedule_sha256(selected),
        "rounds": config.rounds,
        "local_steps": config.local_steps,
        "local_lr": config.local_lr,
        "final_train_objective": float(train_obj),
        "final_train_ce": float(train_ce),
        "final_test_ce": float(test_ce),
        "final_test_accuracy": float(test_acc),
        "final_macro_accuracy": float(macro_acc),
        "final_worst_class_accuracy": float(worst_acc),
        "whole_train_objective": float(history["train_objective"].mean()),
        "uplink_payload_bits": int(uplink_payload_bits),
        "uplink_packetized_bits": int(uplink_packetized_bits),
        "initial_sync_downlink_bits": int(initial_sync_bits),
        "request_bits": int(request_total_bits),
        "replay_downlink_bits": int(replay_downlink_bits),
        "checkpoint_downlink_bits": int(checkpoint_downlink_bits),
        "operational_unicast_hybrid_downlink_bits": int(
            operational_downlink_bits
        ),
        "terminal_catchup_downlink_bits": int(terminal_catchup_bits),
        "unicast_hybrid_downlink_bits": int(
            operational_downlink_bits + terminal_catchup_bits
        ),
        "operational_total_bits": int(operational_total_bits),
        "unicast_hybrid_total_bits": int(exact_final_total_bits),
        "messages": int(messages),
        "coordinate_events": int(coordinate_events),
        "events_per_message": (
            float(coordinate_events / messages) if messages else 0.0
        ),
        "events_per_active_client_round": float(
            coordinate_events / active_client_rounds
        ),
        "active_client_rounds": int(active_client_rounds),
        "mean_delta_norm": float(np.mean(delta_norms)),
        "catchup_replay_syncs": int(catchup_replay_syncs),
        "catchup_checkpoint_syncs": int(catchup_checkpoint_syncs),
        "round_replay_syncs": int(round_replay_syncs),
        "round_checkpoint_syncs": int(round_checkpoint_syncs),
        "terminal_replay_syncs": int(terminal_replay_syncs),
        "terminal_checkpoint_syncs": int(terminal_checkpoint_syncs),
        "empty_syncs": int(empty_syncs),
        "mean_inactive_rounds_before_selection": mean_inactive_rounds,
        "max_inactive_rounds_before_selection": max_inactive_rounds,
        "min_client_participations": int(np.min(participation_counts)),
        "max_client_participations": int(np.max(participation_counts)),
        "mean_return_coordinate_events": mean_return_events,
        "mean_consecutive_coordinate_events": mean_consecutive_events,
        "return_burst_ratio": (
            mean_return_events / mean_consecutive_events
            if mean_consecutive_events > 0.0
            else float("nan")
        ),
        "rho": float(config.rho),
        "threshold": float(config.threshold),
        "jump0": float(config.jump0),
        "jump_exponent": float(config.jump_exponent),
        "topk_fraction": float(config.topk_fraction),
        "server_gain": float(config.server_gain),
    }
    for cls, accuracy in enumerate(per_class):
        result[f"class_{cls}_accuracy"] = float(accuracy)
    if record_history:
        result["history"] = history
    if record_activity:
        result["activity"] = activity
    return result
