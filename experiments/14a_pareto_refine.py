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

    event_settings = [
        ("event_hi", 0.4, 0.0125, 0.0025),
        ("event_m1", 0.3, 0.0250, 0.0050),
        ("event_m2", 0.4, 0.0250, 0.0025),
        ("event_m3", 0.3, 0.0250, 0.0025),
        ("event_low", 0.2, 0.0250, 0.0050),
    ]
    for label, gamma, threshold, jump0 in event_settings:
        result = run_federated_method(
            federation=fed,
            method="events",
            n_ticks=450,
            seed=60606,
            init_scale=0.5,
            step=0.02,
            gamma=gamma,
            threshold=threshold,
            jump0=jump0,
            batch_size=32,
            eval_stride=50,
        )
        rows.append({"configuration": label, **result})

    for fraction in [0.001, 0.0025, 0.005, 0.01, 0.025]:
        result = run_federated_method(
            federation=fed,
            method="ef_topk",
            n_ticks=450,
            seed=60606,
            init_scale=0.5,
            step=0.02,
            topk_fraction=fraction,
            batch_size=32,
            eval_stride=50,
        )
        rows.append({"configuration": f"ef_{fraction:.4f}", **result})

    frame = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / "pareto_refine.csv", index=False)
    print("=== PARETO REFINEMENT ===")
    print(frame.to_string(index=False))
