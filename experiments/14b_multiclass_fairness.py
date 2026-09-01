from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from neuromorphicfl.fmnist_event_benchmark import PERIODS, load_fashion_mnist
from neuromorphicfl.fmnist_multiclass_benchmark import (
    MulticlassFederation,
    class_count_matrix,
    make_multiclass_federation,
    run_federated_method,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14b_fmnist_multiclass"


def make_permuted_federation(
    *, regime: str, seed: int, dominant_classes: tuple[int, ...], train_eval_size: int = 5000
) -> MulticlassFederation:
    if sorted(dominant_classes) != list(range(10)):
        raise ValueError("dominant_classes must be a permutation of 0,...,9")
    train_images, train_labels, test_images, test_labels = load_fashion_mnist(DATA_ROOT)
    base = class_count_matrix(regime)
    if regime == "iid":
        counts = base.copy()
    else:
        dominant = int(np.max(base[0]))
        off = int(np.min(base[0]))
        counts = np.full((10, 10), off, dtype=int)
        for client, cls in enumerate(dominant_classes):
            counts[client, cls] = dominant
    if not np.all(counts.sum(axis=0) == 6000) or not np.all(counts.sum(axis=1) == 6000):
        raise RuntimeError("permuted count matrix must preserve exact balance")

    rng = np.random.default_rng(seed)
    per_class_indices = []
    for cls in range(10):
        ids = np.flatnonzero(train_labels == cls)
        rng.shuffle(ids)
        per_class_indices.append(ids)

    offsets = np.zeros(10, dtype=int)
    client_X = []
    client_y = []
    global_ids = []
    for client in range(10):
        ids_parts = []
        label_parts = []
        for cls in range(10):
            n = int(counts[client, cls])
            start = int(offsets[cls])
            stop = start + n
            ids_parts.append(per_class_indices[cls][start:stop])
            label_parts.append(np.full(n, cls, dtype=np.int64))
            offsets[cls] = stop
        ids = np.concatenate(ids_parts)
        labels = np.concatenate(label_parts)
        order = rng.permutation(len(ids))
        ids = ids[order]
        labels = labels[order]
        client_X.append(train_images[ids].astype(np.float32) / 127.5 - 1.0)
        client_y.append(labels)
        global_ids.append(ids)

    all_ids = np.concatenate(global_ids)
    eval_pos = rng.choice(len(all_ids), size=train_eval_size, replace=False)
    X_train_eval = train_images[all_ids[eval_pos]].astype(np.float32) / 127.5 - 1.0
    y_train_eval = train_labels[all_ids[eval_pos]].astype(np.int64)
    test_order = rng.permutation(len(test_labels))
    X_test = test_images[test_order].astype(np.float32) / 127.5 - 1.0
    y_test = test_labels[test_order].astype(np.int64)
    return MulticlassFederation(
        client_X=tuple(client_X),
        client_y=tuple(client_y),
        X_train_eval=X_train_eval,
        y_train_eval=y_train_eval,
        X_test=X_test,
        y_test=y_test,
        regime=regime,
        client_class_counts=counts,
        periods=PERIODS.copy(),
        weights=np.full(10, 0.1),
    )


def selected_baselines(federation, n_ticks: int):
    rows = []
    for name, method, kwargs in [
        ("full", "full", {"step": 0.08}),
        ("ef_0p5", "ef_topk", {"step": 0.08, "topk_fraction": 0.005}),
        ("ef_2p5", "ef_topk", {"step": 0.08, "topk_fraction": 0.025}),
    ]:
        row = run_federated_method(
            federation=federation, method=method, n_ticks=n_ticks, eval_stride=100, **kwargs
        )
        row["configuration"] = name
        rows.append(row)
    return rows


def main() -> None:
    base = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)

    # Focused event refinement around the monotone gamma trend seen in the 650-tick audit.
    tuning_rows = []
    for gamma in (0.6, 0.8, 1.0):
        for jump0 in (0.0035, 0.0050, 0.0075):
            row = run_federated_method(
                federation=base,
                method="events",
                n_ticks=650,
                gamma=gamma,
                threshold=0.025,
                jump0=jump0,
                eval_stride=100,
            )
            row["configuration"] = f"g{gamma}_q{jump0}"
            tuning_rows.append(row)
    tuning = pd.DataFrame(tuning_rows)
    best = tuning.sort_values("final_train_objective").iloc[0]
    best_gamma = float(best["gamma"])
    best_jump = float(best["jump0"])
    print("=== EVENT REFINEMENT ===")
    print(tuning[[
        "configuration", "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "candidate_events", "events_per_message",
        "ever_fired_fraction"
    ]].to_string(index=False))
    print(f"SELECTED_EVENT gamma={best_gamma} jump0={best_jump}")

    # Long horizon: distinguish finite-horizon class imbalance from persistent failure.
    long_rows = selected_baselines(base, 1200)
    event_long = run_federated_method(
        federation=base,
        method="events",
        n_ticks=1200,
        gamma=best_gamma,
        threshold=0.025,
        jump0=best_jump,
        eval_stride=100,
    )
    event_long["configuration"] = "events_selected"
    long_rows.append(event_long)
    long_df = pd.DataFrame(long_rows)
    print("=== LONG HORIZON ===")
    long_cols = [
        "configuration", "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "whole_train_objective", "payload_bits", "candidate_events",
        "events_per_message", "ever_fired_fraction"
    ] + [f"class_{c}_accuracy" for c in range(10)]
    print(long_df[long_cols].to_string(index=False))

    # Rotate which semantic class is assigned to fast/slow dominant clients.
    assignments = {
        "identity": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        "reverse": (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "mixed": (4, 7, 1, 9, 0, 6, 3, 8, 5, 2),
    }
    assignment_rows = []
    for name, assignment in assignments.items():
        fed = make_permuted_federation(
            regime="strong", seed=2400, dominant_classes=assignment
        )
        for row in selected_baselines(fed, 650):
            row["assignment"] = name
            assignment_rows.append(row)
        row = run_federated_method(
            federation=fed,
            method="events",
            n_ticks=650,
            gamma=best_gamma,
            threshold=0.025,
            jump0=best_jump,
            eval_stride=100,
        )
        row["configuration"] = "events_selected"
        row["assignment"] = name
        assignment_rows.append(row)
    assignment_df = pd.DataFrame(assignment_rows)
    print("=== DOMINANT-CLASS / COMPUTE-PERIOD ROTATION ===")
    print(assignment_df[[
        "assignment", "configuration", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "whole_train_objective"
    ]].to_string(index=False))

    # Label-skew severity sweep with fixed, selected parameters.
    heterogeneity_rows = []
    for regime in ("iid", "moderate", "strong", "extreme"):
        fed = make_multiclass_federation(root=DATA_ROOT, regime=regime, seed=2400)
        for row in selected_baselines(fed, 650):
            row["regime"] = regime
            heterogeneity_rows.append(row)
        row = run_federated_method(
            federation=fed,
            method="events",
            n_ticks=650,
            gamma=best_gamma,
            threshold=0.025,
            jump0=best_jump,
            eval_stride=100,
        )
        row["configuration"] = "events_selected"
        row["regime"] = regime
        heterogeneity_rows.append(row)
    heterogeneity_df = pd.DataFrame(heterogeneity_rows)
    print("=== LABEL-SKEW SWEEP ===")
    print(heterogeneity_df[[
        "regime", "configuration", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "whole_train_objective"
    ]].to_string(index=False))

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(RESULT_ROOT / "event_refinement.csv", index=False)
    long_df.to_csv(RESULT_ROOT / "long_horizon.csv", index=False)
    assignment_df.to_csv(RESULT_ROOT / "period_class_rotation.csv", index=False)
    heterogeneity_df.to_csv(RESULT_ROOT / "heterogeneity.csv", index=False)


if __name__ == "__main__":
    main()
