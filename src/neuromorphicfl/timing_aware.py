from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Weighting = Literal[
    "fixed",
    "inverse_local",
    "inverse_wall",
    "shuffled_local",
    "shuffled_wall_within_client",
    "static_client",
    "global_wall_schedule",
]


@dataclass(frozen=True)
class TimingProblem:
    thetas: np.ndarray
    weights: np.ndarray
    periods: np.ndarray
    gamma: float = 0.05
    rho: float = 0.999
    threshold: float = 1.5
    jump: float = 0.05
    w0_low: float = 0.93
    w0_high: float = 1.13

    def __post_init__(self):
        if not (len(self.thetas) == len(self.weights) == len(self.periods)):
            raise ValueError("thetas, weights, and periods must have equal length")
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("weights must sum to one")


def run_timing_batch(
    *,
    problem: TimingProblem,
    noise_std: float,
    n_ticks: int,
    n_seeds: int,
    tail: int,
    seed: int,
    weighting: Weighting = "fixed",
    tau_c: float = 250.0,
    alpha: float = 0.5,
    shuffled_local_pool: np.ndarray | None = None,
    shuffled_wall_pools: dict[int, np.ndarray] | None = None,
    static_client_weights: dict[int, float] | None = None,
    schedule_scale: float = 5000.0,
    schedule_exponent: float = 1.0,
    log_events: bool = False,
) -> dict:
    """Run full-reset LIF with optional event-time-aware jump scaling.

    Clients compute on delayed server snapshots and return after ``periods[i]``
    wall-clock ticks. Evidence is rate normalized using

        gamma_i = gamma * p_i * T_i.

    The baseline event encoder is a full-reset LIF state. Timing-aware variants
    retain the same event trigger and communication payload; only the server-side
    jump magnitude is changed. Thus event timing is treated as metadata already
    present in the arrival stream rather than an additional transmitted value.

    ``inverse_local`` uses the number of client-local gradient completions since
    the previous event. ``inverse_wall`` uses raw wall-clock inter-event time.
    The latter intentionally retains compute-rate information and therefore must
    be interpreted with client-speed and optimization-stage confounder controls.
    """

    thetas = np.asarray(problem.thetas, dtype=float)
    weights = np.asarray(problem.weights, dtype=float)
    periods = np.asarray(problem.periods, dtype=int)
    n_clients = len(thetas)
    tail = min(tail, n_ticks)

    rng = np.random.default_rng(seed)
    w = rng.uniform(problem.w0_low, problem.w0_high, size=n_seeds)
    z = np.zeros((n_clients, n_seeds), dtype=float)
    snapshots = np.tile(w, (n_clients, 1))
    next_completion = periods.copy()

    local_age = np.zeros((n_clients, n_seeds), dtype=int)
    wall_age = np.zeros((n_clients, n_seeds), dtype=int)
    event_index = np.zeros(n_seeds, dtype=int)

    events = np.zeros(n_seeds, dtype=int)
    events_per_client = np.zeros((n_clients, n_seeds), dtype=int)
    harmful = np.zeros(n_seeds, dtype=int)
    local_wrong = np.zeros(n_seeds, dtype=int)
    whole_w2 = np.zeros(n_seeds)
    tail_w2 = np.zeros(n_seeds)

    event_rows: list[dict] = []

    for tick in range(1, n_ticks + 1):
        z *= problem.rho
        wall_age += 1

        active = [i for i in range(n_clients) if next_completion[i] == tick]
        if len(active) > 1 and tick % 2:
            active.reverse()

        for client in active:
            local_age[client] += 1
            gradient = snapshots[client] - thetas[client] + rng.normal(
                0.0, noise_std, size=n_seeds
            )
            gain = problem.gamma * weights[client] * periods[client]
            z[client] -= gain * gradient

            mask = np.abs(z[client]) >= problem.threshold
            if np.any(mask):
                idx = np.flatnonzero(mask)
                signs = np.sign(z[client, mask]).astype(int)
                tau_local = local_age[client, idx].copy()
                tau_wall = wall_age[client, idx].copy()

                if weighting == "fixed":
                    jump_weight = np.ones(len(idx))
                elif weighting == "inverse_local":
                    jump_weight = np.minimum(
                        1.0,
                        (tau_c / np.maximum(tau_local.astype(float), 1.0)) ** alpha,
                    )
                elif weighting == "inverse_wall":
                    jump_weight = np.minimum(
                        1.0,
                        (tau_c / np.maximum(tau_wall.astype(float), 1.0)) ** alpha,
                    )
                elif weighting == "shuffled_local":
                    if shuffled_local_pool is None or len(shuffled_local_pool) == 0:
                        raise ValueError("shuffled_local_pool is required")
                    sampled = rng.choice(shuffled_local_pool, size=len(idx), replace=True)
                    jump_weight = np.minimum(
                        1.0, (tau_c / np.maximum(sampled.astype(float), 1.0)) ** alpha
                    )
                elif weighting == "shuffled_wall_within_client":
                    if shuffled_wall_pools is None or client not in shuffled_wall_pools:
                        raise ValueError("shuffled_wall_pools must contain every client")
                    sampled = rng.choice(
                        shuffled_wall_pools[client], size=len(idx), replace=True
                    )
                    jump_weight = np.minimum(
                        1.0, (tau_c / np.maximum(sampled.astype(float), 1.0)) ** alpha
                    )
                elif weighting == "static_client":
                    if static_client_weights is None or client not in static_client_weights:
                        raise ValueError("static_client_weights must contain every client")
                    jump_weight = np.full(len(idx), static_client_weights[client])
                elif weighting == "global_wall_schedule":
                    jump_weight = np.full(
                        len(idx),
                        (1.0 + tick / schedule_scale) ** (-schedule_exponent),
                    )
                else:
                    raise ValueError(f"unknown weighting {weighting}")

                w_before = w[idx].copy()
                current_local_descent = -np.sign(w_before - thetas[client])
                is_local_wrong = (
                    (current_local_descent != 0) & (signs != current_local_descent)
                )

                w_after = w_before + problem.jump * jump_weight * signs
                is_harmful = w_after**2 > w_before**2 + 1e-15

                if log_events:
                    for j, run_idx in enumerate(idx):
                        event_rows.append(
                            {
                                "tick": tick,
                                "client": client,
                                "run": int(run_idx),
                                "tau_local": int(tau_local[j]),
                                "tau_wall": int(tau_wall[j]),
                                "sign": int(signs[j]),
                                "jump_weight": float(jump_weight[j]),
                                "w_before": float(w_before[j]),
                                "local_wrong": bool(is_local_wrong[j]),
                                "global_harmful": bool(is_harmful[j]),
                            }
                        )

                w[idx] = w_after
                events[idx] += 1
                events_per_client[client, idx] += 1
                harmful[idx] += is_harmful.astype(int)
                local_wrong[idx] += is_local_wrong.astype(int)
                event_index[idx] += 1

                # Full-reset LIF.
                z[client, idx] = 0.0
                local_age[client, idx] = 0
                wall_age[client, idx] = 0

            snapshots[client] = w.copy()
            next_completion[client] = tick + periods[client]

        whole_w2 += w**2
        if tick > n_ticks - tail:
            tail_w2 += w**2

    event_count = int(events.sum())
    return {
        "tail_rmse": float(np.sqrt(np.mean(tail_w2 / tail))),
        "whole_mse": float(np.mean(whole_w2 / n_ticks)),
        "mean_events": float(np.mean(events)),
        "mean_slow_events": float(np.mean(events_per_client[0])),
        "global_harm_fraction": float(harmful.sum() / event_count)
        if event_count
        else np.nan,
        "local_wrong_fraction": float(local_wrong.sum() / event_count)
        if event_count
        else np.nan,
        "event_log": pd.DataFrame(event_rows) if log_events else None,
    }


def timing_quality_tables(event_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize event quality by local-completion interval and by client."""

    log = event_log.copy()
    bins = [0, 2, 5, 10, 20, 50, 100, 250, 1000, np.inf]
    labels = [
        "1-2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        "101-250",
        "251-1000",
        ">1000",
    ]
    log["local_bin"] = pd.cut(log["tau_local"], bins=bins, labels=labels)
    quality = (
        log.groupby("local_bin", observed=False)
        .agg(
            events=("tau_local", "size"),
            mean_tau_local=("tau_local", "mean"),
            p_local_wrong=("local_wrong", "mean"),
            p_global_harm=("global_harmful", "mean"),
            mean_abs_w=("w_before", lambda x: np.mean(np.abs(x))),
        )
        .reset_index()
    )

    rows = []
    for client, sub in log.groupby("client"):
        rows.append(
            {
                "client": int(client),
                "events": len(sub),
                "tau_local_harm_spearman": sub["tau_local"].corr(
                    sub["global_harmful"].astype(float), method="spearman"
                ),
                "tau_local_wrong_spearman": sub["tau_local"].corr(
                    sub["local_wrong"].astype(float), method="spearman"
                ),
                "tau_wall_harm_spearman": sub["tau_wall"].corr(
                    sub["global_harmful"].astype(float), method="spearman"
                ),
                "tau_wall_absw_spearman": sub["tau_wall"].corr(
                    np.abs(sub["w_before"]), method="spearman"
                ),
                "mean_tau_local": sub["tau_local"].mean(),
                "mean_tau_wall": sub["tau_wall"].mean(),
                "harm_fraction": sub["global_harmful"].mean(),
            }
        )
    return quality, pd.DataFrame(rows)
