from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    make_multiclass_federation,
    run_federated_method,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14b_fmnist_multiclass"


def main() -> None:
    federation = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    rows = []
    for step in (0.02, 0.04, 0.08):
        row = run_federated_method(
            federation=federation,
            method="full",
            n_ticks=650,
            step=step,
            eval_stride=100,
        )
        row["configuration"] = f"full_step{step}"
        rows.append(row)

    for fraction in (0.005, 0.025):
        for step in (0.04, 0.08):
            row = run_federated_method(
                federation=federation,
                method="ef_topk",
                n_ticks=650,
                step=step,
                topk_fraction=fraction,
                eval_stride=100,
            )
            row["configuration"] = f"ef_{fraction}_step{step}"
            rows.append(row)

    event_configs = [
        ("event_g04_q25", 0.4, 0.025, 0.0025),
        ("event_g05_q25", 0.5, 0.025, 0.0025),
        ("event_g06_q25", 0.6, 0.025, 0.0025),
        ("event_g04_q5", 0.4, 0.025, 0.0050),
        ("event_g05_q5", 0.5, 0.025, 0.0050),
        ("event_g06_q5", 0.6, 0.025, 0.0050),
        ("event_g06_t05_q5", 0.6, 0.050, 0.0050),
    ]
    for name, gamma, threshold, jump0 in event_configs:
        row = run_federated_method(
            federation=federation,
            method="events",
            n_ticks=650,
            gamma=gamma,
            threshold=threshold,
            jump0=jump0,
            eval_stride=100,
        )
        row["configuration"] = name
        rows.append(row)

    results = pd.DataFrame(rows)
    cols = [
        "configuration",
        "method",
        "final_train_objective",
        "final_test_ce",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "whole_train_objective",
        "payload_bits",
        "messages",
        "candidate_events",
        "events_per_message",
        "ever_fired_fraction",
        "step",
        "topk_fraction",
        "gamma",
        "threshold",
        "jump0",
    ]
    print("=== MULTICLASS REFINEMENT ===")
    print(results[cols].to_string(index=False))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_ROOT / "refinement.csv", index=False)


if __name__ == "__main__":
    main()
