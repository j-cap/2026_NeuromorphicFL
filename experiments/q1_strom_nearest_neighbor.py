from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    make_multiclass_federation,
    run_federated_method,
)
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn
from neuromorphicfl.q1_nearest_neighbor import (
    run_mlp_pulse_method,
    run_cnn_pulse_method,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "q1_nearest_neighbor"


VARIANTS = ("strom", "leaky_subtractive", "fullreset_tied")
MLP_GAINS = (0.1, 0.3, 1.0, 3.0)
MLP_THRESHOLDS = (0.00125, 0.0025, 0.005, 0.01, 0.02)
CNN_GAINS = (0.3, 1.0, 3.0)
CNN_THRESHOLDS = (0.0025, 0.005, 0.01, 0.02)


def _decorate(row: dict[str, object], configuration: str, family: str) -> dict[str, object]:
    row = dict(row)
    row["configuration"] = configuration
    row["family"] = family
    return row


def run_mlp(federation) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for gain in MLP_GAINS:
            for threshold in MLP_THRESHOLDS:
                row = run_mlp_pulse_method(
                    federation=federation,
                    variant=variant,
                    n_ticks=650,
                    seed=60606,
                    gain=gain,
                    threshold=threshold,
                    rho=0.999,
                    eval_stride=50,
                )
                rows.append(
                    _decorate(
                        row,
                        f"{variant}_g{gain:g}_t{threshold:g}",
                        variant,
                    )
                )

    v1 = run_federated_method(
        federation=federation,
        method="events",
        n_ticks=650,
        seed=60606,
        rho=0.999,
        gamma=1.0,
        threshold=0.025,
        jump0=0.0035,
        schedule_exponent=0.1,
        eval_stride=50,
    )
    rows.append(_decorate(v1, "v1_g1_t0.025_q0.0035_p0.1", "v1"))

    for step in (0.01, 0.02, 0.04):
        dense = run_federated_method(
            federation=federation,
            method="full",
            n_ticks=650,
            seed=60606,
            step=step,
            eval_stride=50,
        )
        rows.append(_decorate(dense, f"dense_step{step:g}", "dense"))
        ef = run_federated_method(
            federation=federation,
            method="ef_topk",
            n_ticks=650,
            seed=60606,
            step=step,
            topk_fraction=0.025,
            eval_stride=50,
        )
        rows.append(_decorate(ef, f"ef2p5_step{step:g}", "ef_topk"))
    return pd.DataFrame(rows)


def run_cnn(federation) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for gain in CNN_GAINS:
            for threshold in CNN_THRESHOLDS:
                row = run_cnn_pulse_method(
                    federation=federation,
                    variant=variant,
                    n_ticks=650,
                    seed=60606,
                    gain=gain,
                    threshold=threshold,
                    rho=0.999,
                    eval_stride=50,
                )
                rows.append(
                    _decorate(
                        row,
                        f"{variant}_g{gain:g}_t{threshold:g}",
                        variant,
                    )
                )

    v1 = run_federated_cnn(
        federation=federation,
        method="events",
        n_ticks=650,
        seed=60606,
        rho=0.999,
        gamma=1.0,
        threshold=0.025,
        jump0=0.0075,
        schedule_exponent=0.1,
        eval_stride=50,
    )
    rows.append(_decorate(v1, "v1_g1_t0.025_q0.0075_p0.1", "v1"))

    for step in (0.04, 0.08, 0.12):
        dense = run_federated_cnn(
            federation=federation,
            method="full",
            n_ticks=650,
            seed=60606,
            step=step,
            eval_stride=50,
        )
        rows.append(_decorate(dense, f"dense_step{step:g}", "dense"))
        ef = run_federated_cnn(
            federation=federation,
            method="ef_topk",
            n_ticks=650,
            seed=60606,
            step=step,
            topk_fraction=0.025,
            eval_stride=50,
        )
        rows.append(_decorate(ef, f"ef2p5_step{step:g}", "ef_topk"))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, architecture: str) -> None:
    cols = [
        "configuration",
        "family",
        "final_train_objective",
        "final_test_ce",
        "final_test_accuracy",
        "final_worst_class_accuracy",
        "whole_train_objective",
        "payload_bits",
    ]
    print(f"=== Q1 {architecture.upper()} ALL CONFIGS ===")
    print(df[cols].sort_values("final_train_objective").to_string(index=False))
    print(f"=== Q1 {architecture.upper()} BEST BY FAMILY (TRAIN OBJECTIVE) ===")
    best = (
        df.sort_values("final_train_objective")
        .groupby("family", as_index=False)
        .first()
    )
    print(best[cols].sort_values("final_train_objective").to_string(index=False))


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    federation = make_multiclass_federation(
        root=DATA_ROOT,
        regime="strong",
        seed=2400,
    )
    mlp = run_mlp(federation)
    mlp.to_csv(RESULT_ROOT / "mlp_650.csv", index=False)
    summarize(mlp, "mlp")

    cnn = run_cnn(federation)
    cnn.to_csv(RESULT_ROOT / "cnn_650.csv", index=False)
    summarize(cnn, "cnn")


if __name__ == "__main__":
    main()
