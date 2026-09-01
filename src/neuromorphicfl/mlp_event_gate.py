from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from .logistic_certificate import LogisticEnsemble

MLPMethod = Literal["schedule", "packet_descent_oracle", "ef_topk", "full"]
NormalizationKind = Literal["raw", "layer", "coordinate"]


@dataclass(frozen=True)
class MLPLayout:
    input_dim: int = 19
    hidden_dim: int = 8

    @property
    def dimension(self) -> int:
        return self.input_dim * self.hidden_dim + 2 * self.hidden_dim + 1

    def slices(self) -> tuple[slice, slice, slice, slice]:
        start = 0
        w1 = slice(start, start + self.input_dim * self.hidden_dim)
        start = w1.stop
        b1 = slice(start, start + self.hidden_dim)
        start = b1.stop
        w2 = slice(start, start + self.hidden_dim)
        start = w2.stop
        b2 = slice(start, start + 1)
        return w1, b1, w2, b2


@dataclass(frozen=True)
class MLPEventConfig:
    method: MLPMethod
    normalization: NormalizationKind = "raw"
    rho: float = 0.999
    gamma: float = 0.2
    threshold: float = 0.025
    jump0: float = 0.015
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.1
    step: float = 0.04
    topk: int = 4
    batch_size: int = 64
    regularization: float = 0.02
    header_bits: int = 32
    verify_harm: bool = False


LAYOUT = MLPLayout()


def initialize_mlp(n_runs: int, seed: int = 7777, scale: float = 0.18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p, h = LAYOUT.input_dim, LAYOUT.hidden_dim
    output = np.zeros((n_runs, LAYOUT.dimension))
    sw1, _, sw2, _ = LAYOUT.slices()
    output[:, sw1] = rng.normal(
        0.0, scale / np.sqrt(p), size=(n_runs, p, h)
    ).reshape(n_runs, -1)
    output[:, sw2] = rng.normal(
        0.0, scale / np.sqrt(h), size=(n_runs, h)
    )
    return output


def _unpack(w: np.ndarray):
    n_runs = w.shape[0]
    p, h = LAYOUT.input_dim, LAYOUT.hidden_dim
    sw1, sb1, sw2, sb2 = LAYOUT.slices()
    return (
        w[:, sw1].reshape(n_runs, p, h),
        w[:, sb1],
        w[:, sw2],
        w[:, sb2].reshape(n_runs),
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def loss_and_gradient(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    regularization: float = 0.02,
    need_gradient: bool = True,
):
    W1, b1, W2, b2 = _unpack(w)
    hidden_pre = np.einsum("rnp,rph->rnh", X, W1) + b1[:, None, :]
    hidden = np.tanh(hidden_pre)
    logits = np.einsum("rnh,rh->rn", hidden, W2) + b2[:, None]
    loss = (
        np.mean(np.logaddexp(0.0, logits) - y * logits, axis=1)
        + 0.5 * regularization * np.sum(w * w, axis=1)
    )
    if not need_gradient:
        return loss

    dlogit = (_sigmoid(logits) - y) / X.shape[1]
    grad_W2 = np.einsum("rnh,rn->rh", hidden, dlogit)
    grad_b2 = np.sum(dlogit, axis=1)
    dz = dlogit[:, :, None] * W2[:, None, :] * (1.0 - hidden * hidden)
    grad_W1 = np.einsum("rnp,rnh->rph", X, dz)
    grad_b1 = np.sum(dz, axis=1)

    gradient = np.empty_like(w)
    sw1, sb1, sw2, sb2 = LAYOUT.slices()
    gradient[:, sw1] = grad_W1.reshape(w.shape[0], -1)
    gradient[:, sb1] = grad_b1
    gradient[:, sw2] = grad_W2
    gradient[:, sb2] = grad_b2[:, None]
    gradient += regularization * w
    return loss, gradient


def _input_data(ensemble: LogisticEnsemble, *, test: bool):
    X = ensemble.Xte if test else ensemble.Xtr
    y = ensemble.yte if test else ensemble.ytr
    # Experiment 12 has an explicit intercept feature. The MLP has trainable
    # biases, so the final constant feature is removed.
    return X[..., :-1], y


def global_loss(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    *,
    test: bool = False,
    regularization: float = 0.02,
) -> np.ndarray:
    X, y = _input_data(ensemble, test=test)
    n_runs, n_clients, n_samples, _ = X.shape
    return loss_and_gradient(
        w,
        X.reshape(n_runs, n_clients * n_samples, LAYOUT.input_dim),
        y.reshape(n_runs, n_clients * n_samples),
        regularization=regularization,
        need_gradient=False,
    )


def test_metrics(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    *,
    regularization: float = 0.02,
):
    X, y = _input_data(ensemble, test=True)
    n_runs, n_clients, n_samples, _ = X.shape
    X = X.reshape(n_runs, n_clients * n_samples, LAYOUT.input_dim)
    y = y.reshape(n_runs, n_clients * n_samples)
    W1, b1, W2, b2 = _unpack(w)
    hidden = np.tanh(np.einsum("rnp,rph->rnh", X, W1) + b1[:, None, :])
    logits = np.einsum("rnh,rh->rn", hidden, W2) + b2[:, None]
    loss = (
        np.mean(np.logaddexp(0.0, logits) - y * logits, axis=1)
        + 0.5 * regularization * np.sum(w * w, axis=1)
    )
    accuracy = np.mean((logits >= 0.0) == y, axis=1)
    return loss, accuracy


def client_full_gradient(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    client: int,
    *,
    regularization: float = 0.02,
) -> np.ndarray:
    _, gradient = loss_and_gradient(
        w,
        ensemble.Xtr[:, client, :, :-1],
        ensemble.ytr[:, client],
        regularization=regularization,
        need_gradient=True,
    )
    return gradient


def minibatch_gradient(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    client: int,
    rng: np.random.Generator,
    *,
    batch_size: int,
    regularization: float,
) -> np.ndarray:
    n_runs = ensemble.n_runs
    n_samples = ensemble.Xtr.shape[2]
    X = np.empty((n_runs, batch_size, LAYOUT.input_dim))
    y = np.empty((n_runs, batch_size))
    for run in range(n_runs):
        ids = rng.integers(0, n_samples, size=batch_size)
        X[run] = ensemble.Xtr[run, client, ids, :-1]
        y[run] = ensemble.ytr[run, client, ids]
    _, gradient = loss_and_gradient(
        w,
        X,
        y,
        regularization=regularization,
        need_gradient=True,
    )
    return gradient


def threshold_scale(
    ensemble: LogisticEnsemble,
    w0: np.ndarray,
    *,
    kind: NormalizationKind,
    regularization: float,
):
    dimension = LAYOUT.dimension
    if kind == "raw":
        return np.ones(dimension), 0

    gradients = np.stack(
        [
            client_full_gradient(
                w0,
                ensemble,
                client,
                regularization=regularization,
            )
            for client in range(ensemble.n_clients)
        ],
        axis=1,
    )
    if kind == "coordinate":
        rms = np.sqrt(np.mean(gradients * gradients, axis=(0, 1))) + 1e-12
        scale = np.clip(rms / np.median(rms), 0.25, 4.0)
        return scale, ensemble.n_clients * dimension * 32

    if kind == "layer":
        sw1, sb1, sw2, sb2 = LAYOUT.slices()
        groups = [
            np.arange(sw1.start, sw1.stop),
            np.arange(sb1.start, sb1.stop),
            np.arange(sw2.start, sw2.stop),
            np.arange(sb2.start, sb2.stop),
        ]
        values = [
            float(np.sqrt(np.mean(gradients[:, :, group] ** 2)) + 1e-12)
            for group in groups
        ]
        median = float(np.median(values))
        scale = np.ones(dimension)
        for group, value in zip(groups, values):
            scale[group] = np.clip(value / median, 0.25, 4.0)
        return scale, ensemble.n_clients * len(groups) * 32

    raise ValueError(f"unknown normalization kind {kind}")


def _single_run(ensemble: LogisticEnsemble, run: int) -> LogisticEnsemble:
    return replace(
        ensemble,
        Xtr=ensemble.Xtr[run : run + 1],
        ytr=ensemble.ytr[run : run + 1],
        Xte=ensemble.Xte[run : run + 1],
        yte=ensemble.yte[run : run + 1],
        w0=ensemble.w0[run : run + 1],
        positive_rates=ensemble.positive_rates[run : run + 1],
    )


def run_mlp_batch(
    *,
    ensemble: LogisticEnsemble,
    config: MLPEventConfig,
    n_ticks: int,
    seed: int,
    eval_stride: int = 60,
    record_history: bool = False,
):
    n_runs = ensemble.n_runs
    n_clients = ensemble.n_clients
    dimension = LAYOUT.dimension
    periods = ensemble.periods
    weights = ensemble.weights
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    rng = np.random.default_rng(seed)

    w = initialize_mlp(n_runs)
    scale, normalization_bits = threshold_scale(
        ensemble,
        w,
        kind=config.normalization,
        regularization=config.regularization,
    )
    thresholds = config.threshold * scale

    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    membrane = np.zeros((n_clients, n_runs, dimension))
    residual = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.full(n_runs, normalization_bits, dtype=np.int64)
    packetized_bits = payload_bits.copy()
    candidates = np.zeros(n_runs, dtype=np.int64)
    accepted = np.zeros(n_runs, dtype=np.int64)
    harmful = np.zeros(n_runs, dtype=np.int64)
    group_events = np.zeros((n_runs, 4), dtype=np.int64)

    history_rows: list[dict[str, float]] = []
    sw1, sb1, sw2, sb2 = LAYOUT.slices()

    for tick in range(1, n_ticks + 1):
        if config.method in {"schedule", "packet_descent_oracle"}:
            membrane *= config.rho

        active = [
            client for client in range(n_clients)
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
                batch_size=config.batch_size,
                regularization=config.regularization,
            )
            rate_weight = weights[client] * periods[client]

            if config.method == "full":
                w -= config.step * rate_weight * gradient
                payload_bits += 32 * dimension
                packetized_bits += 32 * dimension + config.header_bits

            elif config.method == "ef_topk":
                residual[client] -= config.step * rate_weight * gradient
                k = min(config.topk, dimension)
                ids = np.argpartition(
                    np.abs(residual[client]), -k, axis=1
                )[:, -k:]
                values = np.take_along_axis(residual[client], ids, axis=1)
                runs = np.arange(n_runs)
                for column in range(k):
                    w[runs, ids[:, column]] += values[:, column]
                    residual[client, runs, ids[:, column]] = 0.0
                payload_bits += k * (32 + address_bits)
                packetized_bits += (
                    k * (32 + address_bits) + config.header_bits
                )

            else:
                membrane[client] -= config.gamma * rate_weight * gradient
                mask = np.abs(membrane[client]) >= thresholds[None, :]
                if np.any(mask):
                    count = mask.sum(axis=1)
                    candidates += count
                    payload_bits += count * event_bits
                    has_packet = count > 0
                    packetized_bits += (
                        count * event_bits
                        + has_packet.astype(np.int64) * config.header_bits
                    )
                    group_events[:, 0] += mask[:, sw1].sum(axis=1)
                    group_events[:, 1] += mask[:, sb1].sum(axis=1)
                    group_events[:, 2] += mask[:, sw2].sum(axis=1)
                    group_events[:, 3] += mask[:, sb2].sum(axis=1)

                    jump = config.jump0 * (
                        1.0 + tick / config.schedule_scale
                    ) ** (-config.schedule_exponent)
                    if config.method == "schedule":
                        before = (
                            global_loss(
                                w,
                                ensemble,
                                regularization=config.regularization,
                            )
                            if config.verify_harm
                            else None
                        )
                        w += jump * np.sign(membrane[client]) * mask
                        accepted += count
                        if config.verify_harm:
                            after = global_loss(
                                w,
                                ensemble,
                                regularization=config.regularization,
                            )
                            harmful += (
                                (after > before + 1e-12) * count
                            ).astype(np.int64)

                    elif config.method == "packet_descent_oracle":
                        # Expensive diagnostic only. The sparse packet direction is
                        # retained, while exact full-objective backtracking chooses
                        # the scalar jump and rejects non-descending packets.
                        direction = np.sign(membrane[client]) * mask
                        base = global_loss(
                            w,
                            ensemble,
                            regularization=config.regularization,
                        )
                        q = np.full(n_runs, jump)
                        done = count <= 0
                        for _ in range(7):
                            trial = w + q[:, None] * direction
                            value = global_loss(
                                trial,
                                ensemble,
                                regularization=config.regularization,
                            )
                            good = (~done) & (value < base - 1e-12)
                            if np.any(good):
                                w[good] = trial[good]
                                accepted[good] += count[good]
                                done[good] = True
                            q[~done] *= 0.5
                            if np.all(done):
                                break

                    membrane[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train = global_loss(
                w,
                ensemble,
                regularization=config.regularization,
            )
            test_loss_, accuracy = test_metrics(
                w,
                ensemble,
                regularization=config.regularization,
            )
            history_rows.append(
                {
                    "tick": float(tick),
                    "payload_bits": float(np.mean(payload_bits)),
                    "train_loss": float(np.mean(train)),
                    "test_loss": float(np.mean(test_loss_)),
                    "test_accuracy": float(np.mean(accuracy)),
                }
            )

    history = pd.DataFrame(history_rows)
    total_candidates = int(candidates.sum())
    total_accepted = int(accepted.sum())
    test_loss_, accuracy = test_metrics(
        w,
        ensemble,
        regularization=config.regularization,
    )
    total_events = int(group_events.sum())

    result: dict[str, object] = {
        "method": config.method,
        "normalization": config.normalization,
        "normalization_bits": float(normalization_bits),
        "final_train_loss": float(np.mean(global_loss(
            w, ensemble, regularization=config.regularization
        ))),
        "final_test_loss": float(np.mean(test_loss_)),
        "final_test_accuracy": float(np.mean(accuracy)),
        "whole_train_loss": float(history["train_loss"].mean()),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "candidate_events": float(np.mean(candidates)),
        "accepted_events": float(np.mean(accepted)),
        "acceptance_fraction": (
            total_accepted / total_candidates if total_candidates else np.nan
        ),
        "harmful_packet_event_fraction": (
            float(harmful.sum() / total_candidates)
            if config.verify_harm and total_candidates
            else np.nan
        ),
        "event_share_W1": (
            float(group_events[:, 0].sum() / total_events)
            if total_events else np.nan
        ),
        "event_share_b1": (
            float(group_events[:, 1].sum() / total_events)
            if total_events else np.nan
        ),
        "event_share_W2": (
            float(group_events[:, 2].sum() / total_events)
            if total_events else np.nan
        ),
        "event_share_b2": (
            float(group_events[:, 3].sum() / total_events)
            if total_events else np.nan
        ),
    }
    if record_history:
        result["history"] = history
    return result
