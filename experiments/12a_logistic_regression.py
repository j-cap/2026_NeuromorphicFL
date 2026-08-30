from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from neuromorphicfl.logistic_certificate import (
    LogisticEnsemble,
    LogisticRunConfig,
    global_gradient,
    make_synthetic_logistic_ensemble,
    run_logistic_batch,
    test_metrics,
    train_loss,
)


RESULT_DIR = Path("experiments/results/12a_logistic_regression")


def _single_run(ensemble: LogisticEnsemble, run: int) -> LogisticEnsemble:
    return replace(
        ensemble,
        Xtr=ensemble.Xtr[run : run + 1],
        ytr=ensemble.ytr[run : run + 1],
        Xte=ensemble.Xte[run : run + 1],
        yte=ensemble.yte[run : run + 1],
        w0=ensemble.w0[run : run + 1],
        positive_rates=ensemble.positive_rates[run : run + 1],
    )


def centralized_reference(ensemble: LogisticEnsemble) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for run in range(ensemble.n_runs):
        one = _single_run(ensemble, run)

        def objective(w: np.ndarray) -> float:
            return float(train_loss(w[None, :], one)[0])

        def gradient(w: np.ndarray) -> np.ndarray:
            return global_gradient(w[None, :], one)[0]

        solution = minimize(
            objective,
            np.zeros(ensemble.dimension),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 500, "gtol": 1e-10, "ftol": 1e-12},
        )
        loss, accuracy = test_metrics(solution.x[None, :], one)
        rows.append(
            {
                "run": run,
                "train_optimum": float(solution.fun),
                "test_loss": float(loss[0]),
                "test_accuracy": float(accuracy[0]),
                "gradient_norm": float(np.linalg.norm(solution.jac)),
                "success": bool(solution.success),
            }
        )
    return pd.DataFrame(rows)


def _configurations() -> list[tuple[str, LogisticRunConfig]]:
    return [
        (
            "scheduled_events",
            LogisticRunConfig(
                method="schedule",
                jump0=0.01,
                schedule_exponent=0.1,
            ),
        ),
        (
            "scheduled_events_fast",
            LogisticRunConfig(
                method="schedule",
                jump0=0.04,
                schedule_exponent=0.5,
            ),
        ),
        ("global_descent_oracle", LogisticRunConfig(method="global_oracle")),
        ("full_precision", LogisticRunConfig(method="full", step=0.04)),
        (
            "ef_topk_k1",
            LogisticRunConfig(method="ef_topk", topk=1, step=0.04),
        ),
        (
            "ef_topk_k2",
            LogisticRunConfig(method="ef_topk", topk=2, step=0.04),
        ),
        (
            "certificate_full_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="full",
                calibration_period=25,
            ),
        ),
        (
            "certificate_full_C50",
            LogisticRunConfig(
                method="certificate",
                summary_kind="full",
                calibration_period=50,
            ),
        ),
        (
            "certificate_block4_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="block",
                block_size=4,
                calibration_period=25,
            ),
        ),
        (
            "certificate_block10_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="block",
                block_size=10,
                calibration_period=25,
            ),
        ),
        (
            "certificate_diag_residual_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="diag_residual",
                calibration_period=25,
            ),
        ),
        (
            "certificate_spectral_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="spectral",
                calibration_period=25,
            ),
        ),
        (
            "unsafe_diagonal_C50",
            LogisticRunConfig(
                method="naive_certificate",
                summary_kind="diag_naive",
                calibration_period=50,
            ),
        ),
    ]


def run_campaign(*, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if quick:
        n_runs, n_ticks, eval_stride = 3, 400, 20
    else:
        n_runs, n_ticks, eval_stride = 8, 1200, 30

    rows: list[dict[str, object]] = []
    reference_rows: list[pd.DataFrame] = []
    strong_histories: dict[str, pd.DataFrame] = {}

    selected_history_labels = {
        "scheduled_events",
        "global_descent_oracle",
        "full_precision",
        "ef_topk_k1",
        "certificate_full_C25",
        "certificate_block10_C25",
    }

    for regime in ["iid", "moderate", "strong"]:
        ensemble = make_synthetic_logistic_ensemble(
            n_runs=n_runs,
            heterogeneity=regime,
            seed=1300,
        )
        reference = centralized_reference(ensemble)
        reference.insert(0, "heterogeneity", regime)
        reference_rows.append(reference)

        for label, config in _configurations():
            record_history = regime == "strong" and label in selected_history_labels
            result = run_logistic_batch(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                seed=60606,
                eval_stride=eval_stride,
                record_history=record_history,
            )
            history = result.pop("history", None)
            rows.append(
                {
                    "heterogeneity": regime,
                    "configuration": label,
                    **result,
                }
            )
            if history is not None:
                strong_histories[label] = history

    campaign = pd.DataFrame(rows)
    references = pd.concat(reference_rows, ignore_index=True)

    # Event-level safety audit on the strong split.
    strong = make_synthetic_logistic_ensemble(
        n_runs=n_runs,
        heterogeneity="strong",
        seed=1300,
    )
    safety_configs = [
        (
            "scheduled_events",
            LogisticRunConfig(
                method="schedule",
                jump0=0.01,
                schedule_exponent=0.1,
                verify_harm=True,
            ),
        ),
        (
            "certificate_full_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="full",
                calibration_period=25,
                verify_harm=True,
            ),
        ),
        (
            "certificate_block10_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="block",
                block_size=10,
                calibration_period=25,
                verify_harm=True,
            ),
        ),
        (
            "certificate_diag_residual_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="diag_residual",
                calibration_period=25,
                verify_harm=True,
            ),
        ),
        (
            "unsafe_diagonal_C50",
            LogisticRunConfig(
                method="naive_certificate",
                summary_kind="diag_naive",
                calibration_period=50,
                verify_harm=True,
            ),
        ),
    ]
    safety_rows: list[dict[str, object]] = []
    for label, config in safety_configs:
        result = run_logistic_batch(
            ensemble=strong,
            config=config,
            n_ticks=n_ticks,
            seed=60606,
            eval_stride=eval_stride,
        )
        safety_rows.append({"configuration": label, **result})
    safety = pd.DataFrame(safety_rows)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    campaign.to_csv(RESULT_DIR / "campaign.csv", index=False)
    references.to_csv(RESULT_DIR / "centralized_reference.csv", index=False)
    safety.to_csv(RESULT_DIR / "safety_audit.csv", index=False)
    for label, history in strong_histories.items():
        history.to_csv(RESULT_DIR / f"history_{label}.csv", index=False)

    _make_plots(campaign, strong_histories)
    return campaign, references, safety


def _make_plots(
    campaign: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
) -> None:
    display_names = {
        "scheduled_events": "Scheduled events",
        "global_descent_oracle": "Global descent oracle",
        "full_precision": "Full precision",
        "ef_topk_k1": "EF-TopK k=1",
        "certificate_full_C25": "Full certificate C=25",
        "certificate_block10_C25": "Block-10 certificate C=25",
    }

    plt.figure(figsize=(9, 6))
    for label, history in histories.items():
        plt.plot(
            history["payload_bits"],
            history["test_loss"],
            marker="o",
            markersize=3,
            label=display_names[label],
        )
    plt.xscale("log")
    plt.xlabel("Cumulative mean payload bits")
    plt.ylabel("Mean test loss")
    plt.title("Experiment 12A: strong non-IID test loss vs communication")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "test_loss_vs_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for label, history in histories.items():
        plt.plot(
            history["payload_bits"],
            history["test_accuracy"],
            marker="o",
            markersize=3,
            label=display_names[label],
        )
    plt.xscale("log")
    plt.xlabel("Cumulative mean payload bits")
    plt.ylabel("Mean test accuracy")
    plt.title("Experiment 12A: strong non-IID test accuracy vs communication")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "accuracy_vs_bits.png", dpi=180)
    plt.close()

    configs = [
        "scheduled_events",
        "global_descent_oracle",
        "certificate_full_C25",
        "certificate_block4_C25",
        "ef_topk_k1",
        "full_precision",
    ]
    plt.figure(figsize=(9, 5))
    for configuration in configs:
        sub = campaign[campaign["configuration"] == configuration].sort_values(
            "label_rate_std"
        )
        plt.plot(
            sub["label_rate_std"],
            sub["final_test_loss"],
            marker="o",
            label=configuration,
        )
    plt.xlabel("Mean client positive-rate standard deviation")
    plt.ylabel("Final mean test loss")
    plt.title("Experiment 12A: robustness to client heterogeneity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "heterogeneity_test_loss.png", dpi=180)
    plt.close()

    strong = campaign[campaign["heterogeneity"] == "strong"]
    certificate_configs = [
        "certificate_full_C25",
        "certificate_block4_C25",
        "certificate_diag_residual_C25",
        "certificate_spectral_C25",
    ]
    rows = []
    for configuration in certificate_configs:
        row = strong[strong["configuration"] == configuration].iloc[0]
        rows.append(
            {
                "configuration": configuration,
                "event_bits": row["payload_bits"]
                - row["curvature_bits"]
                - row["calibration_bits"],
                "curvature_bits": row["curvature_bits"],
                "calibration_bits": row["calibration_bits"],
            }
        )
    composition = pd.DataFrame(rows)
    x = np.arange(len(composition))
    bottom = np.zeros(len(composition))
    plt.figure(figsize=(9, 5))
    for column in ["event_bits", "curvature_bits", "calibration_bits"]:
        plt.bar(x, composition[column], bottom=bottom, label=column)
        bottom += composition[column].to_numpy()
    plt.xticks(x, composition["configuration"], rotation=25, ha="right")
    plt.ylabel("Mean payload bits")
    plt.title("Experiment 12A: certificate communication composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "certificate_bits.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    campaign_df, reference_df, safety_df = run_campaign(quick=args.quick)
    print(campaign_df.to_string(index=False))
    print("\nCentralized reference")
    print(reference_df.groupby("heterogeneity").mean(numeric_only=True).to_string())
    print("\nSafety audit")
    print(safety_df.to_string(index=False))
