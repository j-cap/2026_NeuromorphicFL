from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation, run_federated_method
from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14b_fmnist_multiclass"

# Reuse the exact dominant-class permutation constructor from the fairness audit.
spec = spec_from_file_location("fairness14b", ROOT / "experiments" / "14b_multiclass_fairness.py")
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
make_permuted_federation = mod.make_permuted_federation

EVENT = {"gamma": 1.0, "threshold": 0.025, "jump0": 0.0035}


def main() -> None:
    # 1) Long-horizon tuning audit: constant dense/EF steps may degrade at 1200 ticks.
    fed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    long_rows = []
    for step in (0.02, 0.04, 0.08):
        row = run_federated_method(federation=fed, method="full", n_ticks=1200, step=step, eval_stride=100)
        row["configuration"] = f"full_{step}"
        long_rows.append(row)
    for fraction in (0.005, 0.025):
        for step in (0.04, 0.08):
            row = run_federated_method(
                federation=fed, method="ef_topk", n_ticks=1200, step=step,
                topk_fraction=fraction, eval_stride=100
            )
            row["configuration"] = f"ef_{fraction}_{step}"
            long_rows.append(row)
    row = run_federated_method(
        federation=fed, method="events", n_ticks=1200, eval_stride=100, **EVENT
    )
    row["configuration"] = "events_selected"
    long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    print("=== LONG-HORIZON BASELINE AUDIT ===")
    print(long_df[[
        "configuration", "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "whole_train_objective", "payload_bits"
    ]].to_string(index=False))

    # 2) Independent strong partitions, frozen hyperparameters.
    robust_rows = []
    for seed in (2400, 2500, 2600):
        fed_seed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=seed)
        configs = [
            ("full", "full", {"step": 0.08}),
            ("ef_0p5", "ef_topk", {"step": 0.08, "topk_fraction": 0.005}),
            ("ef_2p5", "ef_topk", {"step": 0.08, "topk_fraction": 0.025}),
            ("events_selected", "events", EVENT),
        ]
        for name, method, kwargs in configs:
            row = run_federated_method(
                federation=fed_seed, method=method, n_ticks=650, eval_stride=100, **kwargs
            )
            row["configuration"] = name
            row["data_seed"] = seed
            robust_rows.append(row)
    robust_df = pd.DataFrame(robust_rows)
    print("=== STRONG-PARTITION ROBUSTNESS ===")
    print(robust_df[[
        "data_seed", "configuration", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "whole_train_objective"
    ]].to_string(index=False))

    # 3) Extreme-skew rotations: test whether 2.4% worst-class accuracy is just period/class confounding.
    assignments = {
        "identity": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        "reverse": (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "mixed": (4, 7, 1, 9, 0, 6, 3, 8, 5, 2),
    }
    extreme_rows = []
    for assignment_name, assignment in assignments.items():
        fed_extreme = make_permuted_federation(
            regime="extreme", seed=2400, dominant_classes=assignment
        )
        for name, method, kwargs in [
            ("full", "full", {"step": 0.08}),
            ("ef_2p5", "ef_topk", {"step": 0.08, "topk_fraction": 0.025}),
            ("events_selected", "events", EVENT),
        ]:
            row = run_federated_method(
                federation=fed_extreme, method=method, n_ticks=650, eval_stride=100, **kwargs
            )
            row["configuration"] = name
            row["assignment"] = assignment_name
            extreme_rows.append(row)
    extreme_df = pd.DataFrame(extreme_rows)
    print("=== EXTREME-SKEW ROTATIONS ===")
    print(extreme_df[[
        "assignment", "configuration", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "whole_train_objective"
    ]].to_string(index=False))

    # 4) Extreme long horizon for persistence, with class-wise diagnostics.
    fed_extreme = make_multiclass_federation(root=DATA_ROOT, regime="extreme", seed=2400)
    persistence_rows = []
    for name, method, kwargs in [
        ("full_0p04", "full", {"step": 0.04}),
        ("ef_2p5_0p08", "ef_topk", {"step": 0.08, "topk_fraction": 0.025}),
        ("events_selected", "events", EVENT),
    ]:
        row = run_federated_method(
            federation=fed_extreme, method=method, n_ticks=1200, eval_stride=100, **kwargs
        )
        row["configuration"] = name
        persistence_rows.append(row)
    persistence_df = pd.DataFrame(persistence_rows)
    print("=== EXTREME-SKEW LONG HORIZON ===")
    cols = [
        "configuration", "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
        "payload_bits", "whole_train_objective"
    ] + [f"class_{c}_accuracy" for c in range(10)]
    print(persistence_df[cols].to_string(index=False))

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(RESULT_ROOT / "final_long_horizon_baseline_audit.csv", index=False)
    robust_df.to_csv(RESULT_ROOT / "final_strong_robustness.csv", index=False)
    extreme_df.to_csv(RESULT_ROOT / "final_extreme_rotation.csv", index=False)
    persistence_df.to_csv(RESULT_ROOT / "final_extreme_long_horizon.csv", index=False)


if __name__ == "__main__":
    main()
