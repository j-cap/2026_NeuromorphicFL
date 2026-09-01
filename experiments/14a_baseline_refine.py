from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_event_benchmark import make_binary_federation, run_federated_method

RESULT_DIR = Path("experiments/results/14a_fmnist_binary")

if __name__ == "__main__":
    fed = make_binary_federation(
        root="/tmp/fashion-mnist",
        class_pair=(2, 4),
        regime="strong",
        seed=1400,
    )
    rows = []
    for step in [0.005, 0.01, 0.02, 0.04, 0.08]:
        result = run_federated_method(
            federation=fed,
            method="full",
            n_ticks=450,
            seed=60606,
            init_scale=0.5,
            step=step,
            batch_size=32,
            eval_stride=50,
        )
        rows.append({"configuration": f"full_step{step}", **result})

    for fraction in [0.005, 0.025]:
        for step in [0.005, 0.01, 0.02, 0.04, 0.08]:
            result = run_federated_method(
                federation=fed,
                method="ef_topk",
                n_ticks=450,
                seed=60606,
                init_scale=0.5,
                step=step,
                topk_fraction=fraction,
                batch_size=32,
                eval_stride=50,
            )
            rows.append(
                {"configuration": f"ef_{fraction}_step{step}", **result}
            )
    frame = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / "baseline_refine.csv", index=False)
    print("=== BASELINE REFINEMENT ===")
    print(frame.to_string(index=False))
