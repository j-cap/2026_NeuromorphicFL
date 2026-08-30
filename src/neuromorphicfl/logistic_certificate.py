from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


LogisticMethod = Literal[
    "schedule",
    "certificate",
    "naive_certificate",
    "global_oracle",
    "ef_topk",
    "full",
]
SummaryKind = Literal[
    "full",
    "block",
    "diag_residual",
    "spectral",
    "diag_naive",
    "none",
]


@dataclass(frozen=True)
class LogisticEnsemble:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    w0: np.ndarray
    periods: np.ndarray
    weights: np.ndarray
    positive_rates: np.ndarray
    heterogeneity: str
    regularization: float

    @property
    def n_runs(self) -> int:
        return int(self.w0.shape[0])

    @property
    def n_clients(self) -> int:
        return int(self.Xtr.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.Xtr.shape[-1])


@dataclass(frozen=True)
class LogisticRunConfig:
    method: LogisticMethod
    summary_kind: SummaryKind = "none"
    calibration_period: int = 100
    block_size: int = 4
    rho: float = 0.999
    gamma: float = 0.1
    threshold: float = 0.05
    jump0: float = 0.01
    schedule_scale: float = 500.0
    schedule_exponent: float = 0.1
    step: float = 0.04
    topk: int = 1
    batch_size: int = 64
    header_bits: int = 32
    verify_harm: bool = False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic_logistic_ensemble(
    *,
    n_runs: int,
    heterogeneity: str,
    n_clients: int = 10,
    dimension: int = 20,
    n_train: int = 300,
    n_test: int = 150,
    periods: np.ndarray | None = None,
    feature_correlation: float = 0.45,
    regularization: float = 0.05,
    seed: int = 1300,
) -> LogisticEnsemble:
    """Controlled non-IID binary logistic-regression family.

    The final coordinate is an intercept.  Client heterogeneity is introduced
    through client-specific feature means and latent label biases, while the
    underlying feature covariance and global separating direction are shared.
    """

    if periods is None:
        if n_clients != 10:
            raise ValueError("provide periods explicitly when n_clients != 10")
        periods = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
    periods = np.asarray(periods, dtype=int)
    if len(periods) != n_clients:
        raise ValueError("period count must equal client count")

    raw_dimension = dimension - 1
    weights = np.full(n_clients, 1.0 / n_clients)
    rng = np.random.default_rng(seed)

    if heterogeneity == "iid":
        mean_std, bias_std = 0.0, 0.0
    elif heterogeneity == "moderate":
        mean_std, bias_std = 0.25, 0.5
    elif heterogeneity == "strong":
        mean_std, bias_std = 0.6, 1.0
    else:
        raise ValueError(f"unknown heterogeneity regime {heterogeneity}")

    index = np.arange(raw_dimension)
    covariance = feature_correlation ** np.abs(index[:, None] - index[None, :])
    chol = np.linalg.cholesky(covariance)

    true_direction = rng.normal(size=(n_runs, raw_dimension))
    true_direction = (
        2.0 * true_direction / np.linalg.norm(true_direction, axis=1, keepdims=True)
    )
    client_means = rng.normal(
        0.0, mean_std, size=(n_runs, n_clients, raw_dimension)
    )
    client_bias = rng.normal(0.0, bias_std, size=(n_runs, n_clients))

    Xtr = np.empty((n_runs, n_clients, n_train, dimension))
    ytr = np.empty((n_runs, n_clients, n_train))
    Xte = np.empty((n_runs, n_clients, n_test, dimension))
    yte = np.empty((n_runs, n_clients, n_test))
    positive_rates = np.empty((n_runs, n_clients))

    for run in range(n_runs):
        for client in range(n_clients):
            raw_train = (
                rng.normal(size=(n_train, raw_dimension)) @ chol.T
                + client_means[run, client]
            )
            raw_test = (
                rng.normal(size=(n_test, raw_dimension)) @ chol.T
                + client_means[run, client]
            )
            Xtr[run, client, :, :-1] = raw_train
            Xtr[run, client, :, -1] = 1.0
            Xte[run, client, :, :-1] = raw_test
            Xte[run, client, :, -1] = 1.0

            train_prob = _sigmoid(
                raw_train @ true_direction[run] + client_bias[run, client]
            )
            test_prob = _sigmoid(
                raw_test @ true_direction[run] + client_bias[run, client]
            )
            ytr[run, client] = (
                rng.random(n_train) < train_prob
            ).astype(float)
            yte[run, client] = (
                rng.random(n_test) < test_prob
            ).astype(float)
            positive_rates[run, client] = ytr[run, client].mean()

    return LogisticEnsemble(
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        w0=np.zeros((n_runs, dimension)),
        periods=periods,
        weights=weights,
        positive_rates=positive_rates,
        heterogeneity=heterogeneity,
        regularization=float(regularization),
    )


def local_full_grad(w: np.ndarray, ensemble: LogisticEnsemble) -> np.ndarray:
    logits = np.einsum("rcnd,rd->rcn", ensemble.Xtr, w)
    residual = _sigmoid(logits) - ensemble.ytr
    gradient = (
        np.einsum("rcnd,rcn->rcd", ensemble.Xtr, residual)
        / ensemble.Xtr.shape[2]
    )
    gradient += ensemble.regularization * w[:, None, :]
    return gradient


def global_gradient(w: np.ndarray, ensemble: LogisticEnsemble) -> np.ndarray:
    return np.sum(
        ensemble.weights[None, :, None] * local_full_grad(w, ensemble),
        axis=1,
    )


def minibatch_gradient(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    client: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    n_runs = ensemble.n_runs
    n_samples = ensemble.Xtr.shape[2]
    output = np.empty((n_runs, ensemble.dimension))
    for run in range(n_runs):
        ids = rng.integers(0, n_samples, size=batch_size)
        X = ensemble.Xtr[run, client, ids]
        y = ensemble.ytr[run, client, ids]
        residual = _sigmoid(X @ w[run]) - y
        output[run] = (
            X.T @ residual / batch_size
            + ensemble.regularization * w[run]
        )
    return output


def train_loss(w: np.ndarray, ensemble: LogisticEnsemble) -> np.ndarray:
    logits = np.einsum("rcnd,rd->rcn", ensemble.Xtr, w)
    data_loss = np.logaddexp(0.0, logits) - ensemble.ytr * logits
    return (
        np.mean(data_loss, axis=(1, 2))
        + 0.5 * ensemble.regularization * np.sum(w * w, axis=1)
    )


def test_metrics(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.einsum("rcnd,rd->rcn", ensemble.Xte, w)
    data_loss = np.logaddexp(0.0, logits) - ensemble.yte * logits
    loss = (
        np.mean(data_loss, axis=(1, 2))
        + 0.5 * ensemble.regularization * np.sum(w * w, axis=1)
    )
    accuracy = np.mean((logits >= 0.0) == ensemble.yte, axis=(1, 2))
    return loss, accuracy


def componentwise_curvature_bound(ensemble: LogisticEnsemble) -> np.ndarray:
    """Return M_i with |[H_i(w)]_jk| <= M_i,jk for every w."""

    X = np.abs(ensemble.Xtr)
    M = 0.25 * np.einsum("rcnj,rcnk->rcjk", X, X) / X.shape[2]
    index = np.arange(ensemble.dimension)
    M[:, :, index, index] += ensemble.regularization
    return M


def aggregate_componentwise_bound(
    ensemble: LogisticEnsemble,
    M: np.ndarray | None = None,
) -> np.ndarray:
    if M is None:
        M = componentwise_curvature_bound(ensemble)
    return np.sum(ensemble.weights[None, :, None, None] * M, axis=1)


def aggregate_spectral_bound(ensemble: LogisticEnsemble) -> np.ndarray:
    n_runs = ensemble.n_runs
    L = np.empty((n_runs, ensemble.n_clients))
    for run in range(n_runs):
        for client in range(ensemble.n_clients):
            smax = np.linalg.svd(
                ensemble.Xtr[run, client], compute_uv=False
            )[0]
            L[run, client] = (
                0.25 * smax * smax / ensemble.Xtr.shape[2]
                + ensemble.regularization
            )
    return np.sum(ensemble.weights[None, :] * L, axis=1)


def curvature_summary_bits(
    *,
    kind: SummaryKind,
    n_clients: int,
    dimension: int,
    block_size: int = 4,
) -> int:
    if kind == "full":
        floats = n_clients * dimension * dimension
    elif kind == "block":
        floats = n_clients * dimension * (block_size + 1)
    elif kind == "diag_residual":
        floats = n_clients * 2 * dimension
    elif kind == "spectral":
        floats = n_clients
    elif kind == "diag_naive":
        floats = n_clients * dimension
    elif kind == "none":
        floats = 0
    else:
        raise ValueError(f"unknown summary kind {kind}")
    return 32 * floats


def _radius_single(
    *,
    delta_abs: np.ndarray,
    Mbar: np.ndarray,
    coordinate: int,
    kind: SummaryKind,
    block_size: int,
    spectral_bound: float | None,
) -> float:
    j = coordinate
    if kind == "full":
        return float(Mbar[j] @ delta_abs)
    if kind == "diag_residual":
        diagonal = Mbar[j, j]
        off_sum = Mbar[j].sum() - diagonal
        outside = np.delete(delta_abs, j)
        outside_max = float(np.max(outside)) if len(outside) else 0.0
        return float(diagonal * delta_abs[j] + off_sum * outside_max)
    if kind == "block":
        start = (j // block_size) * block_size
        stop = min(len(delta_abs), start + block_size)
        within = float(Mbar[j, start:stop] @ delta_abs[start:stop])
        residual = float(Mbar[j].sum() - Mbar[j, start:stop].sum())
        outside = np.concatenate((delta_abs[:start], delta_abs[stop:]))
        outside_max = float(np.max(outside)) if len(outside) else 0.0
        return within + residual * outside_max
    if kind == "spectral":
        if spectral_bound is None:
            raise ValueError("spectral bound required")
        return float(spectral_bound * np.linalg.norm(delta_abs))
    if kind == "diag_naive":
        # Deliberately unsafe control: ignores cross-coordinate coupling.
        return float(Mbar[j, j] * delta_abs[j])
    raise ValueError(f"unknown summary kind {kind}")


def _global_gradient_component_run(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    run: int,
    coordinate: int,
) -> float:
    local = np.empty(ensemble.n_clients)
    for client in range(ensemble.n_clients):
        X = ensemble.Xtr[run, client]
        y = ensemble.ytr[run, client]
        residual = _sigmoid(X @ w) - y
        local[client] = (
            X[:, coordinate] @ residual / len(y)
            + ensemble.regularization * w[coordinate]
        )
    return float(ensemble.weights @ local)


def _train_loss_run(
    w: np.ndarray,
    ensemble: LogisticEnsemble,
    run: int,
) -> float:
    loss = 0.0
    for client in range(ensemble.n_clients):
        X = ensemble.Xtr[run, client]
        y = ensemble.ytr[run, client]
        logits = X @ w
        loss += ensemble.weights[client] * np.mean(
            np.logaddexp(0.0, logits) - y * logits
        )
    return float(loss + 0.5 * ensemble.regularization * (w @ w))


def run_logistic_batch(
    *,
    ensemble: LogisticEnsemble,
    config: LogisticRunConfig,
    n_ticks: int,
    seed: int,
    eval_stride: int = 30,
    tail_fraction: float = 0.25,
    record_history: bool = False,
) -> dict[str, object]:
    """Run Experiment 12A.

    Valid certificate summaries are ``full``, ``block``,
    ``diag_residual``, and ``spectral``. ``diag_naive`` is retained only as an
    unsafe diagnostic control and must not be interpreted as a certificate.
    """

    n_runs = ensemble.n_runs
    n_clients = ensemble.n_clients
    dimension = ensemble.dimension
    periods = ensemble.periods
    weights = ensemble.weights
    address_bits = int(np.ceil(np.log2(dimension)))
    event_bits = address_bits + 1
    rng = np.random.default_rng(seed)

    w = ensemble.w0.copy()
    snapshots = np.repeat(w[None, :, :], n_clients, axis=0)
    next_completion = periods.copy()
    z = np.zeros((n_clients, n_runs, dimension))
    residual = np.zeros((n_clients, n_runs, dimension))

    payload_bits = np.zeros(n_runs, dtype=np.int64)
    packetized_bits = np.zeros(n_runs, dtype=np.int64)
    events = np.zeros(n_runs, dtype=np.int64)
    candidates = np.zeros(n_runs, dtype=np.int64)
    accepted = np.zeros(n_runs, dtype=np.int64)
    harmful = np.zeros(n_runs, dtype=np.int64)
    calibrations = np.zeros(n_runs, dtype=np.int64)
    client_candidates = np.zeros((n_runs, n_clients), dtype=np.int64)
    client_accepted = np.zeros((n_runs, n_clients), dtype=np.int64)

    certificate_method = config.method in ("certificate", "naive_certificate")
    summary_kind: SummaryKind = config.summary_kind
    if config.method == "naive_certificate":
        summary_kind = "diag_naive"

    M = componentwise_curvature_bound(ensemble)
    Mbar = aggregate_componentwise_bound(ensemble, M)
    coordinate_L = np.diagonal(Mbar, axis1=1, axis2=2)
    spectral_L = (
        aggregate_spectral_bound(ensemble)
        if summary_kind == "spectral"
        else None
    )

    static_bits = (
        curvature_summary_bits(
            kind=summary_kind,
            n_clients=n_clients,
            dimension=dimension,
            block_size=config.block_size,
        )
        if certificate_method
        else 0
    )
    if static_bits:
        payload_bits += static_bits
        packetized_bits += static_bits + n_clients * config.header_bits

    calibration_gradient = np.zeros((n_runs, dimension))
    calibration_model = w.copy()

    def calibrate() -> None:
        nonlocal calibration_gradient, calibration_model
        local = local_full_grad(w, ensemble)
        calibration_gradient = np.sum(
            weights[None, :, None] * local, axis=1
        )
        calibration_model = w.copy()
        payload_bits[:] += n_clients * dimension * 32
        packetized_bits[:] += n_clients * (
            dimension * 32 + config.header_bits
        )
        calibrations[:] += 1

    if certificate_method:
        calibrate()

    history_rows: list[dict[str, float]] = []

    for tick in range(1, n_ticks + 1):
        if (
            certificate_method
            and config.calibration_period > 0
            and tick % config.calibration_period == 0
        ):
            calibrate()

        if config.method in (
            "schedule",
            "certificate",
            "naive_certificate",
            "global_oracle",
        ):
            z *= config.rho

        active = [i for i in range(n_clients) if next_completion[i] == tick]
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

            if config.method == "full":
                w -= config.step * rate_weight * gradient
                payload_bits += 32 * dimension
                packetized_bits += 32 * dimension + config.header_bits

            elif config.method == "ef_topk":
                residual[client] -= config.step * rate_weight * gradient
                k = min(config.topk, dimension)
                idx = np.argpartition(
                    np.abs(residual[client]), -k, axis=1
                )[:, -k:]
                values = np.take_along_axis(residual[client], idx, axis=1)
                runs = np.arange(n_runs)
                for column in range(k):
                    w[runs, idx[:, column]] += values[:, column]
                    residual[client, runs, idx[:, column]] = 0.0
                payload_bits += k * (32 + address_bits)
                packetized_bits += (
                    k * (32 + address_bits) + config.header_bits
                )

            else:
                z[client] -= config.gamma * rate_weight * gradient
                mask = np.abs(z[client]) >= config.threshold
                if np.any(mask):
                    n_events = mask.sum(axis=1)
                    candidates += n_events
                    events += n_events
                    client_candidates[:, client] += n_events
                    payload_bits += n_events * event_bits
                    has_packet = n_events > 0
                    packetized_bits += (
                        n_events * event_bits
                        + has_packet.astype(np.int64) * config.header_bits
                    )

                    if config.method == "schedule":
                        jump = config.jump0 * (
                            1.0 + tick / config.schedule_scale
                        ) ** (-config.schedule_exponent)
                        if config.verify_harm:
                            before = train_loss(w, ensemble)
                        w += jump * np.sign(z[client]) * mask
                        accepted += n_events
                        client_accepted[:, client] += n_events
                        if config.verify_harm:
                            after = train_loss(w, ensemble)
                            for run in np.flatnonzero(has_packet):
                                if after[run] > before[run] + 1e-12:
                                    harmful[run] += n_events[run]

                    else:
                        # Sequential processing is required: once one coordinate
                        # changes, the uncertainty radius for the next candidate
                        # must be recomputed at the new model.
                        for run in range(n_runs):
                            for coord in np.flatnonzero(mask[run]):
                                sign = float(np.sign(z[client, run, coord]))
                                if config.method == "global_oracle":
                                    current_gradient = _global_gradient_component_run(
                                        w[run], ensemble, run, coord
                                    )
                                    lower_alignment = -sign * current_gradient
                                    L = coordinate_L[run, coord]
                                else:
                                    radius = _radius_single(
                                        delta_abs=np.abs(
                                            w[run] - calibration_model[run]
                                        ),
                                        Mbar=Mbar[run],
                                        coordinate=coord,
                                        kind=summary_kind,
                                        block_size=config.block_size,
                                        spectral_bound=(
                                            None
                                            if spectral_L is None
                                            else float(spectral_L[run])
                                        ),
                                    )
                                    lower_alignment = (
                                        -sign * calibration_gradient[run, coord]
                                        - radius
                                    )
                                    L = (
                                        float(spectral_L[run])
                                        if summary_kind == "spectral"
                                        else coordinate_L[run, coord]
                                    )

                                if lower_alignment > 0.0:
                                    jump = lower_alignment / max(float(L), 1e-12)
                                    if config.verify_harm:
                                        before = _train_loss_run(
                                            w[run], ensemble, run
                                        )
                                    w[run, coord] += sign * jump
                                    accepted[run] += 1
                                    client_accepted[run, client] += 1
                                    if config.verify_harm:
                                        after = _train_loss_run(
                                            w[run], ensemble, run
                                        )
                                        if after > before + 1e-10:
                                            harmful[run] += 1

                    z[client][mask] = 0.0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            train = train_loss(w, ensemble)
            test_loss_value, accuracy = test_metrics(w, ensemble)
            history_rows.append(
                {
                    "tick": float(tick),
                    "payload_bits": float(np.mean(payload_bits)),
                    "train_loss": float(np.mean(train)),
                    "test_loss": float(np.mean(test_loss_value)),
                    "test_accuracy": float(np.mean(accuracy)),
                }
            )

    history = pd.DataFrame(history_rows)
    tail_start = n_ticks * (1.0 - tail_fraction)
    tail_history = history[history["tick"] >= tail_start]
    total_candidates = int(candidates.sum())
    total_accepted = int(accepted.sum())
    test_loss_value, accuracy = test_metrics(w, ensemble)

    result: dict[str, object] = {
        "method": config.method,
        "summary_kind": summary_kind,
        "calibration_period": config.calibration_period,
        "block_size": config.block_size,
        "final_train_loss": float(np.mean(train_loss(w, ensemble))),
        "final_test_loss": float(np.mean(test_loss_value)),
        "final_test_accuracy": float(np.mean(accuracy)),
        "tail_test_loss": float(tail_history["test_loss"].mean()),
        "tail_test_accuracy": float(tail_history["test_accuracy"].mean()),
        "whole_train_loss": float(history["train_loss"].mean()),
        "payload_bits": float(np.mean(payload_bits)),
        "packetized_bits": float(np.mean(packetized_bits)),
        "curvature_bits": float(static_bits),
        "calibration_bits": float(
            np.mean(calibrations) * n_clients * dimension * 32
        ),
        "calibrations": float(np.mean(calibrations)),
        "events": float(np.mean(events)),
        "candidate_events": float(np.mean(candidates)),
        "accepted_events": float(np.mean(accepted)),
        "acceptance_fraction": (
            float(total_accepted / total_candidates)
            if total_candidates
            else np.nan
        ),
        "harmful_fraction": (
            float(harmful.sum() / total_accepted)
            if config.verify_harm and total_accepted
            else np.nan
        ),
        "slow_candidate_share": (
            float(client_candidates[:, -2:].sum() / client_candidates.sum())
            if client_candidates.sum()
            else np.nan
        ),
        "slow_accepted_share": (
            float(client_accepted[:, -2:].sum() / client_accepted.sum())
            if client_accepted.sum()
            else np.nan
        ),
        "label_rate_std": float(
            np.mean(np.std(ensemble.positive_rates, axis=1))
        ),
    }
    if record_history:
        result["history"] = history
    return result
