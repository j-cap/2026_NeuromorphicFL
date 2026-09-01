from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import LAYOUT, run_federated_cnn


ROOT = Path("data/fashion-mnist")
OUT = Path("experiments/results/15a_fmnist_cnn")
OUT.mkdir(parents=True, exist_ok=True)

fed = make_multiclass_federation(root=ROOT, regime="strong", seed=2400)
rows = []

# Method-neutral dense trainability audit.
for step in [0.01, 0.02, 0.04, 0.08]:
    r = run_federated_cnn(
        federation=fed, method="full", n_ticks=250, step=step, eval_stride=50
    )
    rows.append({"configuration": f"full_{step:g}", **r})

# Sparse conventional reference audit.
for step in [0.02, 0.04, 0.08]:
    for frac in [0.005, 0.025]:
        r = run_federated_cnn(
            federation=fed, method="ef_topk", n_ticks=250,
            step=step, topk_fraction=frac, eval_stride=50,
        )
        rows.append({"configuration": f"ef_{frac:g}_{step:g}", **r})

# Event viability / scale audit. Keep threshold common initially.
for gamma in [0.3, 0.6, 1.0]:
    for q0 in [0.0015, 0.0025, 0.0035]:
        r = run_federated_cnn(
            federation=fed, method="events", n_ticks=250,
            rho=0.999, gamma=gamma, threshold=0.025, jump0=q0,
            eval_stride=50,
        )
        rows.append({"configuration": f"event_g{gamma:g}_q{q0:g}", **r})

frame = pd.DataFrame(rows)
frame.to_csv(OUT / "audit_250.csv", index=False)
cols = [
    "configuration", "method", "final_train_objective", "final_test_ce",
    "final_test_accuracy", "final_worst_class_accuracy", "payload_bits",
    "candidate_events", "events_per_message", "ever_fired_fraction",
]
print("CNN layout dimension:", LAYOUT.dimension)
print(frame[cols].to_string(index=False))
