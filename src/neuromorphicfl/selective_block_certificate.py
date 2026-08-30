from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .logistic_certificate import (
    LogisticEnsemble,
    aggregate_componentwise_bound,
    componentwise_curvature_bound,
    global_gradient,
    minibatch_gradient,
    test_metrics,
    train_loss,
)


PartitionKind = Literal["contiguous", "coupling"]
SummaryKind = Literal["full", "block_residual"]
RefreshPolicy = Literal["on_demand", "round_robin"]


@dataclass(frozen=True)
class SelectiveCertificateConfig:
    block_size: int = 1
    partition: PartitionKind = "contiguous"
    summary_kind: SummaryKind = "block_residual"
    refresh_policy: RefreshPolicy = "on_demand"
    min_refresh_gap: int = 50
    round_robin_every: int = 3
    rho: float = 0.999
    gamma: float = 0.1
    threshold: float = 0.05
    batch_size: int = 64
    header_bits: int = 32
    verify_harm: bool = False


def contiguous_blocks(dimension: int, block_size: int) -> list[list[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [
        list(range(start, min(dimension, start + block_size)))
        for start in range(0, dimension, block_size)
    ]


def block_index(blocks: list[list[int]], dimension: int) -> np.ndarray:
    index = np.empty(dimension, dtype=int)
    assigned = np.zeros(dimension, dtype=bool)
    for block_id, block in enumerate(blocks):
        for coordinate in block:
            if coordinate < 0 or coordinate >= dimension:
                raise ValueError("block contains an invalid coordinate")
            if assigned[coordinate]:
                raise ValueError("coordinate assigned to multiple blocks")
            index[coordinate] = block_id
            assigned[coordinate] = True
    if not np.all(assigned):
        raise ValueError("blocks must cover every coordinate")
    return index


def coupling_greedy_blocks(
    mean_bound: np.ndarray,
    block_size: int,
) -> list[list[int]]:
    """Greedy equal-size grouping using normalized curvature coupling."""

    dimension = int(mean_bound.shape[0])
    diagonal = np.sqrt(np.maximum(np.diag(mean_bound), 1e-15))
    coupling = mean_bound / (diagonal[:, None] * diagonal[None, :])
    np.fill_diagonal(coupling, 0.0)

    unassigned = set(range(dimension))
    blocks: list[list[int]] = []
    while unassigned:
        if len(unassigned) <= block_size:
            blocks.append(sorted(unassigned))
            break
        coordinates = sorted(unassigned)
        pair = max(
            (
                (a, b)
                for position, a in enumerate(coordinates)
                for b in coordinates[position + 1 :]
            ),
            key=lambda pair_: coupling[pair_[0], pair_[1]],
        )
        block = [pair[0], pair[1]]
        unassigned.remove(pair[0])
        unassigned.remove(pair[1])
        while len(block) < block_size and unassigned:
            coordinate = max(
                unassigned,
                key=lambda candidate: float(
                    np.mean(coupling[candidate, block])
                ),
            )
            block.append(coordinate)
            unassigned.remove(coordinate)
        blocks.append(sorted(block))
    return blocks


def make_blocks(
    *,
    aggregate_bound: np.ndarray,
    block_size: int,
    partition: PartitionKind,
) -> list[list[int]]:
    dimension = int(aggregate_bound.shape[-1])
    if partition == "contiguous":
        return contiguous_blocks(dimension, block_size)
    if partition == "coupling":
        return coupling_greedy_blocks(
            np.mean(aggregate_bound, axis=0),
            block_size,
        )
    raise ValueError(f"unknown partition {partition}")


def residual_coupling_fraction(
    mean_bound: np.ndarray,
    blocks: list[list[int]],
) -> float:
    dimension = int(mean_bound.shape[0])
    ids = block_index(blocks, dimension)
    total = 0.0
    outside = 0.0
    for coordinate in range(dimension):
        total += float(np.sum(mean_bound[coordinate]))
        outside += float(
            np.sum(mean_bound[coordinate, ids != ids[coordinate]])
        )
    return outside / total if total > 0.0 else 0.0


def curvature_summary_bits(
    *,
    blocks: list[list[int]],
    kind: SummaryKind,
    n_clients: int,
    dimension: int,
) -> int:
    mapping_bits = dimension * int(
        np.ceil(np.log2(max(1, len(blocks))))
    )
    if kind == "full":
        return 32 * n_clients * dimension * dimension
    if kind == "block_residual":
        ids = block_index(blocks, dimension)
        floats_per_client = sum(
            len(blocks[ids[coordinate]]) + 1
            for coordinate in range(dimension)
        )
        return 32 * n_clients * floats_per_client + mapping_bits
    raise ValueError(f"unknown summary kind {kind}")


def _radius(
    *,
    delta_abs: np.ndarray,
    bound: np.ndarray,
    coordinate: int,
    blocks: list[list[int]],
    ids: np.ndarray,
    kind: SummaryKind,
) -> float:
    if kind == "full":
        return float(bound[coordinate] @ delta_abs)

    block = blocks[ids[coordinate]]
    within = float(bound[coordinate, block] @ delta_abs[block])
    outside = np.flatnonzero(ids != ids[coordinate])
    if len(outside) == 0:
        return within
    residual = float(np.sum(bound[coordinate, outside]))
    return within + residual * float(np.max(delta_abs[outside]))


def _global_gradient_block_run(
    *,
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    run: int,
    block: list[int],
) -> np.ndarray:
    output = np.zeros(len(block))
    for client in range(ensemble.n_clients):
        X = ensemble.Xtr[run, client]
        y = ensemble.ytr[run, client]
        logits = X @ w
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        residual = probability - y
        output += ensemble.weights[client] * (
            X[:, block].T @ residual / len(y)
            + ensemble.regularization * w[block]
        )
    return output


def _train_loss_run(
    *,
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    run: int,
) -> float:
    objective = 0.0
    for client in range(ensemble.n_clients):
        X = ensemble.Xtr[run, client]
        y = ensemble.ytr[run, client]
        logits = X @ w
        objective += ensemble.weights[client] * float(
            np.mean(np.logaddexp(0.0, logits) - y * logits)
        )
    return objective + 0.5 * ensemble.regularization * float(w @ w)


def run_selective_certificate(
    *,
    ensemble: LogisticEnsemble,
    config: SelectiveCertificateConfig,
    n_ticks: int,
    seed: int,
    eval_stride: int = 30,
    tail_fraction: float = 0.25,
    record_history: bool = False,
) -> dict[str, object]:
    """Run the Experiment-12B selective calibrated certificate.

    Every coordinate keeps its own calibrated gradient component and its own
    full model snapshot.  A block refresh therefore changes only the
    calibration age of the coordinates contained in that block.  Candidate
    events are processed sequentially so that each accepted jump satisfies the
    certificate at the actual current model.
    """

    n_runs = ensemble.n_runs
    n_clients = ensemble.n_clients
    dimension = ensemble.dimension
    periods = ensemble.periods
    weights = ensemble.weights
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    rng = np.random.default_rng(seed)

    client_bounds = componentwise_curvature_bound(ensemble)
    aggregate_bound = aggregate_componentwise_bound(
        ensemble,
        client_bounds,
    )
    coordinate_L = np.diagonal(
        aggregate_bound,
        axis1=1,
        axis2=2,
    )
    blocks = make_blocks(
        aggregate_bound=aggregate_bound,
        block_size=config.block_size,
        partition=config.partition,
    )
    ids = block_index(blocks, dimension)
    n_blocks = len(blocks)

    static_bits = curvature_summary_bits(
        blocks=blocks,
        kind=config.summary_kind,
        n_clients=n_clients,
        dimension=dimension,
    )

    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    membrane = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.full(n_runs, static_bits, dtype=np.int64)
    packetized_bits = payload_bits + n_clients * config.header_bits
    refresh_bits = np.zeros(n_runs, dtype=np.int64)
    events = np.zeros(n_runs, dtype=np.int64)
    candidates = np.zeros(n_runs, dtype=np.int64)
    accepted = np.zeros(n_runs, dtype=np.int64)
    harmful = np.zeros(n_runs, dtype=np.int64)
    refresh_count = np.zeros((n_runs, n_blocks), dtype=np.int64)
    client_accepted = np.zeros((n_runs, n_clients), dtype=np.int64)

    calibrated_gradient = global_gradient(w, ensemble)
    calibration_model = np.repeat(w[:, None, :], dimension, axis=1)
    last_refresh = np.zeros((n_runs, n_blocks), dtype=int)

    # Initial calibration contains every gradient coordinate once.
    initial_bits = n_clients * dimension * 32
    payload_bits += initial_bits
    packetized_bits += n_clients * (
        dimension * 32 + config.header_bits
    )
    refresh_bits += initial_bits
    refresh_count += 1

    def refresh_block(run: int, block_id_: int, tick: int) -> None:
        block = blocks[block_id_]
        gradient = _global_gradient_block_run(
            w=w[run],
            ensemble=ensemble,
            run=run,
            block=block,
        )
        calibrated_gradient[run, block] = gradient
        for coordinate in block:
            calibration_model[run, coordinate] = w[run]
        bits = n_clients * len(block) * 32
        payload_bits[run] += bits
        packetized_bits[run] += n_clients * (
            len(block) * 32 + config.header_bits
        )
        refresh_bits[run] += bits
        refresh_count[run, block_id_] += 1
        last_refresh[run, block_id_] = tick

    history_rows: list[dict[str, float | int]] = []
    for tick in range(1, n_ticks + 1):
        if (
            config.refresh_policy == "round_robin"
            and tick % config.round_robin_every == 0
        ):
            block_id_ = (
                tick // config.round_robin_every - 1
            ) % n_blocks
            for run in range(n_runs):
                refresh_block(run, block_id_, tick)

        membrane *= config.rho
        active = [
            client
            for client in range(n_clients)
            if next_completion[client] == tick
        ]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            gradient = minibatch_gradient(
                snapshots[client],
                ensemble,
                client,
                rng,
                config.batch_size,
            )
            rate_weight = weights[client] * periods[client]
            membrane[client] -= config.gamma * rate_weight * gradient
            mask = np.abs(membrane[client]) >= config.threshold
            if np.any(mask):
                count = mask.sum(axis=1)
                events += count
                candidates += count
                payload_bits += count * event_bits
                has_packet = count > 0
                packetized_bits += (
                    count * event_bits
                    + has_packet.astype(np.int64) * config.header_bits
                )

                for run in range(n_runs):
                    for coordinate in np.flatnonzero(mask[run]):
                        sign = float(
                            np.sign(membrane[client, run, coordinate])
                        )
                        block_id_ = ids[coordinate]
                        delta_abs = np.abs(
                            w[run]
                            - calibration_model[run, coordinate]
                        )
                        radius = _radius(
                            delta_abs=delta_abs,
                            bound=aggregate_bound[run],
                            coordinate=coordinate,
                            blocks=blocks,
                            ids=ids,
                            kind=config.summary_kind,
                        )
                        lower_alignment = (
                            -sign * calibrated_gradient[run, coordinate]
                            - radius
                        )

                        if (
                            lower_alignment <= 0.0
                            and config.refresh_policy == "on_demand"
                            and tick - last_refresh[run, block_id_]
                            >= max(1, config.min_refresh_gap)
                        ):
                            refresh_block(run, block_id_, tick)
                            delta_abs = np.abs(
                                w[run]
                                - calibration_model[run, coordinate]
                            )
                            radius = _radius(
                                delta_abs=delta_abs,
                                bound=aggregate_bound[run],
                                coordinate=coordinate,
                                blocks=blocks,
                                ids=ids,
                                kind=config.summary_kind,
                            )
                            lower_alignment = (
                                -sign
                                * calibrated_gradient[run, coordinate]
                                - radius
                            )

                        if lower_alignment > 0.0:
                            jump = lower_alignment / max(
                                coordinate_L[run, coordinate],
                                1e-12,
                            )
                            if config.verify_harm:
                                before = _train_loss_run(
                                    w=w[run],
                                    ensemble=ensemble,
                                    run=run,
                                )
                            w[run, coordinate] += jump * sign
                            accepted[run] += 1
                            client_accepted[run, client] += 1
                            if config.verify_harm:
                                after = _train_loss_run(
                                    w=w[run],
                                    ensemble=ensemble,
                                    run=run,
                                )
                                if after > before + 1e-10:
                                    harmful[run] += 1

                membrane[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train = train_loss(w, ensemble)
            test_loss_, test_accuracy = test_metrics(w, ensemble)
            history_rows.append(
                {
                    "tick": tick,
                    "payload_bits": float(np.mean(payload_bits)),
                    "train_loss": float(np.mean(train)),
                    "test_loss": float(np.mean(test_loss_)),
                    "test_accuracy": float(np.mean(test_accuracy)),
                }
            )

    history = pd.DataFrame(history_rows)
    tail = history[history["tick"] >= (1.0 - tail_fraction) * n_ticks]
    total_accepted = int(accepted.sum())
    total_candidates = int(candidates.sum())

    result: dict[str, object] = {
        "partition": config.partition,
        "summary_kind": config.summary_kind,
        "refresh_policy": config.refresh_policy,
        "block_size": config.block_size,
        "n_blocks": n_blocks,
        "min_refresh_gap": config.min_refresh_gap,
        "round_robin_every": config.round_robin_every,
        "residual_coupling_fraction": residual_coupling_fraction(
            np.mean(aggregate_bound, axis=0),
            blocks,
        ),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_test_loss": float(history.iloc[-1]["test_loss"]),
        "final_test_accuracy": float(
            history.iloc[-1]["test_accuracy"]
        ),
        "whole_train_loss": float(history["train_loss"].mean()),
        "tail_test_loss": float(tail["test_loss"].mean()),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "curvature_bits": float(static_bits),
        "refresh_bits": float(np.mean(refresh_bits)),
        "event_bits": float(
            np.mean(payload_bits) - static_bits - np.mean(refresh_bits)
        ),
        "events": float(np.mean(events)),
        "candidate_events": float(np.mean(candidates)),
        "accepted_events": float(np.mean(accepted)),
        "acceptance": (
            total_accepted / total_candidates
            if total_candidates
            else np.nan
        ),
        "harmful_fraction": (
            float(harmful.sum() / total_accepted)
            if config.verify_harm and total_accepted
            else np.nan
        ),
        "mean_refreshes": float(
            np.mean(np.sum(refresh_count, axis=1))
        ),
        "slow_accepted_share": (
            float(
                client_accepted[:, -2:].sum()
                / client_accepted.sum()
            )
            if client_accepted.sum()
            else np.nan
        ),
    }
    result["history"] = history if record_history else None
    return result
