from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation, run_federated_method
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn
from neuromorphicfl.q1_nearest_neighbor import run_mlp_pulse_method, run_cnn_pulse_method

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "q1_nearest_neighbor_focused"

VARIANTS = ("strom", "leaky_subtractive", "fullreset_tied")
GAINS = (0.3, 1.0)
THRESHOLDS = (0.0025, 0.005, 0.01)


def decorate(row, configuration, family):
    row = dict(row)
    row["configuration"] = configuration
    row["family"] = family
    return row


def main(architecture: str) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    federation = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    rows = []

    runner = run_mlp_pulse_method if architecture == "mlp" else run_cnn_pulse_method
    for variant in VARIANTS:
        for gain in GAINS:
            for threshold in THRESHOLDS:
                row = runner(
                    federation=federation,
                    variant=variant,
                    n_ticks=650,
                    seed=60606,
                    gain=gain,
                    threshold=threshold,
                    rho=0.999,
                    eval_stride=50,
                )
                rows.append(decorate(row, f"{variant}_g{gain:g}_t{threshold:g}", variant))

    if architecture == "mlp":
        v1 = run_federated_method(
            federation=federation, method="events", n_ticks=650, seed=60606,
            rho=0.999, gamma=1.0, threshold=0.025, jump0=0.0035,
            schedule_exponent=0.1, eval_stride=50,
        )
        ef = run_federated_method(
            federation=federation, method="ef_topk", n_ticks=650, seed=60606,
            step=0.02, topk_fraction=0.025, eval_stride=50,
        )
        dense = run_federated_method(
            federation=federation, method="full", n_ticks=650, seed=60606,
            step=0.02, eval_stride=50,
        )
        rows.append(decorate(v1, "v1_g1_t0.025_q0.0035_p0.1", "v1"))
    else:
        v1 = run_federated_cnn(
            federation=federation, method="events", n_ticks=650, seed=60606,
            rho=0.999, gamma=1.0, threshold=0.025, jump0=0.0075,
            schedule_exponent=0.1, eval_stride=50,
        )
        ef = run_federated_cnn(
            federation=federation, method="ef_topk", n_ticks=650, seed=60606,
            step=0.08, topk_fraction=0.025, eval_stride=50,
        )
        dense = run_federated_cnn(
            federation=federation, method="full", n_ticks=650, seed=60606,
            step=0.12, eval_stride=50,
        )
        rows.append(decorate(v1, "v1_g1_t0.025_q0.0075_p0.1", "v1"))

    rows.append(decorate(ef, "ef2p5", "ef_topk"))
    rows.append(decorate(dense, "dense", "dense"))

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT / f"{architecture}_650.csv", index=False)
    cols = ["configuration", "family", "final_train_objective", "final_test_ce",
            "final_test_accuracy", "final_worst_class_accuracy", "whole_train_objective",
            "payload_bits"]
    print(f"=== Q1 FOCUSED {architecture.upper()} ===")
    print(df[cols].sort_values("final_train_objective").to_string(index=False))
    best = df.sort_values("final_train_objective").groupby("family", as_index=False).first()
    print(f"=== Q1 FOCUSED {architecture.upper()} BEST BY FAMILY ===")
    print(best[cols].sort_values("final_train_objective").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["mlp", "cnn"], required=True)
    args = parser.parse_args()
    main(args.architecture)
