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
    settings = [
        (0.4, 0.0250, 0.0050),
        (0.6, 0.0250, 0.0050),
        (0.4, 0.0125, 0.0025),
        (0.4, 0.0125, 0.0050),
        (0.6, 0.0125, 0.0025),
        (0.6, 0.0125, 0.0050),
    ]
    rows = []
    for gamma, threshold, jump0 in settings:
        rows.append(
            run_federated_method(
                federation=fed,
                method="events",
                n_ticks=160,
                seed=60606,
                init_scale=0.5,
                gamma=gamma,
                threshold=threshold,
                jump0=jump0,
                batch_size=32,
                eval_stride=40,
            )
        )
    frame = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULT_DIR / "event_refine.csv", index=False)
    print("=== EVENT REFINEMENT ===")
    print(frame.to_string(index=False))
