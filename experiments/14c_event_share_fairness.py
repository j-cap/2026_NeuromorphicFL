from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    LAYOUT,
    MulticlassFederation,
    initialize_mlp,
    loss_and_gradient,
    make_multiclass_federation,
    predictive_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14c_event_share_fairness"

HETERO_PERIODS = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
EQUAL_PERIODS = np.full(10, 3, dtype=int)


def reorder_clients(fed: MulticlassFederation, order: np.ndarray) -> MulticlassFederation:
    """Reassign semantic client datasets to compute slots while preserving periods."""
    order = np.asarray(order, dtype=int)
    return replace(
        fed,
        client_X=tuple(fed.client_X[int(i)] for i in order),
        client_y=tuple(fed.client_y[int(i)] for i in order),
        client_class_counts=fed.client_class_counts[order].copy(),
    )


def with_periods(fed: MulticlassFederation, periods: np.ndarray) -> MulticlassFederation:
    return replace(fed, periods=np.asarray(periods, dtype=int).copy())


def run_event_diagnostic(
    fed: MulticlassFederation,
    *,
    n_ticks: int = 650,
    seed: int = 60606,
    rho: float = 0.999,
    gamma: float = 1.0,
    threshold: float = 0.025,
    jump0: float = 0.0035,
    batch_size: int = 32,
    regularization: float = 1e-4,
    eval_stride: int = 50,
):
    rng = np.random.default_rng(seed)
    d = LAYOUT.dimension
    w = initialize_mlp(layout=LAYOUT, scale=0.5)
    snapshots = np.repeat(w[None, :], fed.n_clients, axis=0)
    next_completion = fed.periods.copy()
    membrane = np.zeros((fed.n_clients, d), dtype=np.float32)

    completions = np.zeros(fed.n_clients, dtype=np.int64)
    event_counts = np.zeros(fed.n_clients, dtype=np.int64)
    update_mass = np.zeros(fed.n_clients, dtype=np.float64)
    message_counts = np.zeros(fed.n_clients, dtype=np.int64)
    output_events = np.zeros(10, dtype=np.int64)
    output_mass = np.zeros(10, dtype=np.float64)
    history_rows: list[dict[str, float]] = []

    groups = LAYOUT.groups()
    w3 = groups[-2][1]
    b3 = groups[-1][1]
    w3_shape = (LAYOUT.widths[-2], LAYOUT.widths[-1])

    for tick in range(1, n_ticks + 1):
        membrane *= rho
        active = [i for i in range(fed.n_clients) if next_completion[i] == tick]
        if len(active) > 1:
            shift = tick % len(active)
            active = active[shift:] + active[:shift]

        for client in active:
            completions[client] += 1
            local_n = len(fed.client_y[client])
            ids = rng.integers(0, local_n, size=batch_size)
            _, _, grad = loss_and_gradient(
                snapshots[client],
                fed.client_X[client][ids],
                fed.client_y[client][ids],
                layout=LAYOUT,
                regularization=regularization,
                need_gradient=True,
            )
            rate_weight = float(fed.weights[client] * fed.periods[client])
            membrane[client] -= gamma * rate_weight * grad
            mask = np.abs(membrane[client]) >= threshold
            count = int(mask.sum())
            if count:
                q = jump0 * (1.0 + tick / 500.0) ** (-0.1)
                signed = np.sign(membrane[client, mask])
                w[mask] += q * signed
                membrane[client, mask] = 0.0
                event_counts[client] += count
                update_mass[client] += q * count
                message_counts[client] += 1

                w3_mask = mask[w3].reshape(w3_shape)
                b3_mask = mask[b3]
                class_counts = w3_mask.sum(axis=0).astype(np.int64) + b3_mask.astype(np.int64)
                output_events += class_counts
                output_mass += q * class_counts

            snapshots[client] = w
            next_completion[client] = tick + fed.periods[client]

        if tick == 1 or tick % eval_stride == 0 or tick == n_ticks:
            _, test_ce, test_acc, _, worst, per_class = predictive_metrics(
                w, fed.X_test, fed.y_test, layout=LAYOUT, regularization=regularization
            )
            row: dict[str, float] = {
                "tick": float(tick),
                "test_ce": float(test_ce),
                "test_accuracy": float(test_acc),
                "worst_class_accuracy": float(worst),
            }
            for cls, acc in enumerate(per_class):
                row[f"class_{cls}_accuracy"] = float(acc)
            history_rows.append(row)

    _, test_ce, test_acc, _, worst, per_class = predictive_metrics(
        w, fed.X_test, fed.y_test, layout=LAYOUT, regularization=regularization
    )

    total_events = max(1, int(event_counts.sum()))
    total_mass = max(np.finfo(float).eps, float(update_mass.sum()))
    total_completions = max(1, int(completions.sum()))
    client_rows = []
    for i in range(fed.n_clients):
        dominant_class = int(np.argmax(fed.client_class_counts[i]))
        client_rows.append({
            "client": i,
            "period": int(fed.periods[i]),
            "dominant_class": dominant_class,
            "dominant_fraction": float(np.max(fed.client_class_counts[i]) / fed.client_class_counts[i].sum()),
            "completions": int(completions[i]),
            "completion_share": float(completions[i] / total_completions),
            "events": int(event_counts[i]),
            "event_share": float(event_counts[i] / total_events),
            "events_per_completion": float(event_counts[i] / max(1, completions[i])),
            "update_mass": float(update_mass[i]),
            "update_mass_share": float(update_mass[i] / total_mass),
            "messages": int(message_counts[i]),
        })

    output_total = max(1, int(output_events.sum()))
    output_mass_total = max(np.finfo(float).eps, float(output_mass.sum()))
    class_rows = []
    for cls in range(10):
        dom = int(np.argmax(fed.client_class_counts[:, cls]))
        class_rows.append({
            "class": cls,
            "final_accuracy": float(per_class[cls]),
            "output_events": int(output_events[cls]),
            "output_event_share": float(output_events[cls] / output_total),
            "output_update_mass": float(output_mass[cls]),
            "output_mass_share": float(output_mass[cls] / output_mass_total),
            "dominant_client": dom,
            "dominant_client_period": int(fed.periods[dom]),
            "dominant_client_event_share": float(event_counts[dom] / total_events),
            "dominant_client_mass_share": float(update_mass[dom] / total_mass),
        })

    summary = {
        "test_ce": float(test_ce),
        "test_accuracy": float(test_acc),
        "worst_class_accuracy": float(worst),
        "total_events": int(event_counts.sum()),
        "total_messages": int(message_counts.sum()),
        "total_completions": int(completions.sum()),
        "event_share_cv": float(np.std(event_counts / total_events) / np.mean(event_counts / total_events)),
        "mass_share_cv": float(np.std(update_mass / total_mass) / np.mean(update_mass / total_mass)),
        "events_per_completion_cv": float(np.std(event_counts / np.maximum(completions, 1)) / np.mean(event_counts / np.maximum(completions, 1))),
        "output_event_share_cv": float(np.std(output_events / output_total) / np.mean(output_events / output_total)),
    }
    return summary, pd.DataFrame(client_rows), pd.DataFrame(class_rows), pd.DataFrame(history_rows)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    scenarios = []
    client_all = []
    class_all = []
    history_all = []

    orders = {
        "identity": np.arange(10),
        "reverse": np.arange(9, -1, -1),
        "mixed": np.array([0, 9, 1, 8, 2, 7, 3, 6, 4, 5]),
    }

    # 2x2 factorial at one fixed partition seed.
    for regime in ("iid", "strong"):
        base = make_multiclass_federation(root=DATA_ROOT, regime=regime, seed=2400)
        for compute_name, periods in (("equal", EQUAL_PERIODS), ("heterogeneous", HETERO_PERIODS)):
            fed = with_periods(base, periods)
            name = f"{regime}_{compute_name}_identity"
            summary, clients, classes, hist = run_event_diagnostic(fed)
            classes["scenario"] = name
            clients["scenario"] = name
            hist["scenario"] = name
            summary.update({"scenario": name, "regime": regime, "compute": compute_name, "assignment": "identity", "seed": 2400})
            summary["rho_output_share_accuracy"] = spearman(classes["output_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
            summary["rho_dom_client_share_accuracy"] = spearman(classes["dominant_client_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
            scenarios.append(summary); client_all.append(clients); class_all.append(classes); history_all.append(hist)

    # Class-to-period rotations under strong skew and heterogeneous compute.
    base = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    for assignment, order in orders.items():
        if assignment == "identity":
            continue  # already included in factorial
        fed = with_periods(reorder_clients(base, order), HETERO_PERIODS)
        name = f"strong_heterogeneous_{assignment}"
        summary, clients, classes, hist = run_event_diagnostic(fed)
        classes["scenario"] = name
        clients["scenario"] = name
        hist["scenario"] = name
        summary.update({"scenario": name, "regime": "strong", "compute": "heterogeneous", "assignment": assignment, "seed": 2400})
        summary["rho_output_share_accuracy"] = spearman(classes["output_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
        summary["rho_dom_client_share_accuracy"] = spearman(classes["dominant_client_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
        scenarios.append(summary); client_all.append(clients); class_all.append(classes); history_all.append(hist)

    # Reproduce the problematic strong/heterogeneous seed 2500 for causal checking.
    base = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2500)
    for compute_name, periods in (("equal", EQUAL_PERIODS), ("heterogeneous", HETERO_PERIODS)):
        fed = with_periods(base, periods)
        name = f"strong_{compute_name}_seed2500"
        summary, clients, classes, hist = run_event_diagnostic(fed)
        classes["scenario"] = name
        clients["scenario"] = name
        hist["scenario"] = name
        summary.update({"scenario": name, "regime": "strong", "compute": compute_name, "assignment": "identity", "seed": 2500})
        summary["rho_output_share_accuracy"] = spearman(classes["output_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
        summary["rho_dom_client_share_accuracy"] = spearman(classes["dominant_client_event_share"].to_numpy(), classes["final_accuracy"].to_numpy())
        scenarios.append(summary); client_all.append(clients); class_all.append(classes); history_all.append(hist)

    summary_df = pd.DataFrame(scenarios)
    client_df = pd.concat(client_all, ignore_index=True)
    class_df = pd.concat(class_all, ignore_index=True)
    history_df = pd.concat(history_all, ignore_index=True)

    summary_df.to_csv(RESULT_ROOT / "scenario_summary.csv", index=False)
    client_df.to_csv(RESULT_ROOT / "client_diagnostics.csv", index=False)
    class_df.to_csv(RESULT_ROOT / "class_diagnostics.csv", index=False)
    history_df.to_csv(RESULT_ROOT / "class_history.csv", index=False)

    print("=== 14C SCENARIO SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("=== 14C CLIENT DIAGNOSTICS ===")
    print(client_df.to_string(index=False))
    print("=== 14C CLASS DIAGNOSTICS ===")
    print(class_df.to_string(index=False))
    print("=== 14C RESULT JSON ===")
    print(json.dumps({"scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
