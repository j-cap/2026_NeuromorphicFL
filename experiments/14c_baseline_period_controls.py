from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    make_multiclass_federation,
    run_federated_method,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14c_event_share_fairness"
HETERO = np.array([1, 1, 2, 2, 5, 5, 10, 10, 20, 20], dtype=int)
EQUAL = np.full(10, 3, dtype=int)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime, seed in (("iid", 2400), ("strong", 2400), ("strong", 2500)):
        base = make_multiclass_federation(root=DATA_ROOT, regime=regime, seed=seed)
        for compute, periods in (("equal", EQUAL), ("heterogeneous", HETERO)):
            fed = replace(base, periods=periods.copy())
            for method in ("full", "ef_topk"):
                kwargs = dict(
                    federation=fed,
                    method=method,
                    n_ticks=650,
                    eval_stride=50,
                    step=0.08,
                )
                if method == "ef_topk":
                    kwargs["topk_fraction"] = 0.025
                result = run_federated_method(**kwargs)
                rows.append({
                    "regime": regime,
                    "seed": seed,
                    "compute": compute,
                    "method": method,
                    "test_ce": result["final_test_ce"],
                    "test_accuracy": result["final_test_accuracy"],
                    "worst_class_accuracy": result["final_worst_class_accuracy"],
                    "whole_train_objective": result["whole_train_objective"],
                    "payload_bits": result["payload_bits"],
                    **{f"class_{c}_accuracy": result[f"class_{c}_accuracy"] for c in range(10)},
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / "baseline_period_controls.csv", index=False)
    print("=== 14C BASELINE PERIOD CONTROLS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
