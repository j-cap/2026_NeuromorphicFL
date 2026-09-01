from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    LAYOUT,
    federation_audit,
    make_multiclass_federation,
    run_federated_method,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14b_fmnist_multiclass"


def audit() -> None:
    federation = make_multiclass_federation(
        root=DATA_ROOT, regime="strong", seed=2400
    )
    print("=== PARTITION AUDIT ===")
    print(federation_audit(federation).to_string(index=False))
    print(f"dimension={LAYOUT.dimension}")

    rows = []
    for step in (0.01, 0.02, 0.04):
        row = run_federated_method(
            federation=federation,
            method="full",
            n_ticks=250,
            step=step,
            eval_stride=50,
        )
        row["configuration"] = f"full_step{step}"
        rows.append(row)

    for fraction in (0.005, 0.025):
        for step in (0.01, 0.02, 0.04):
            row = run_federated_method(
                federation=federation,
                method="ef_topk",
                n_ticks=250,
                step=step,
                topk_fraction=fraction,
                eval_stride=50,
            )
            row["configuration"] = f"ef_{fraction}_step{step}"
            rows.append(row)

    event_configs = [
        ("event_a14", 0.3, 0.025, 0.0025),
        ("event_g02", 0.2, 0.025, 0.0025),
        ("event_g04", 0.4, 0.025, 0.0025),
        ("event_thr05", 0.3, 0.05, 0.0025),
        ("event_smallq", 0.3, 0.025, 0.00125),
        ("event_thr05_q5", 0.3, 0.05, 0.005),
    ]
    for name, gamma, threshold, jump0 in event_configs:
        row = run_federated_method(
            federation=federation,
            method="events",
            n_ticks=250,
            gamma=gamma,
            threshold=threshold,
            jump0=jump0,
            eval_stride=50,
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
        "final_macro_accuracy",
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
    print("=== MULTICLASS AUDIT ===")
    print(results[cols].to_string(index=False))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    federation_audit(federation).to_csv(RESULT_ROOT / "partition_audit.csv", index=False)
    results.to_csv(RESULT_ROOT / "audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["audit"], default="audit")
    args = parser.parse_args()
    if args.phase == "audit":
        audit()


if __name__ == "__main__":
    main()
