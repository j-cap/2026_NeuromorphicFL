from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from neuromorphicfl.logistic_certificate import (
    LogisticRunConfig,
    global_gradient,
    make_synthetic_logistic_ensemble,
    run_logistic_batch,
    test_metrics,
    train_loss,
)
from neuromorphicfl.selective_block_certificate import (
    SelectiveCertificateConfig,
    coupling_greedy_blocks,
    make_blocks,
    residual_coupling_fraction,
    run_selective_certificate,
)


RESULT_DIR = Path("experiments/results/12b_selective_block_certificate")


def _single_run(ensemble, run):
    return replace(
        ensemble,
        Xtr=ensemble.Xtr[run : run + 1],
        ytr=ensemble.ytr[run : run + 1],
        Xte=ensemble.Xte[run : run + 1],
        yte=ensemble.yte[run : run + 1],
        w0=ensemble.w0[run : run + 1],
        positive_rates=ensemble.positive_rates[run : run + 1],
    )


def centralized_reference(ensemble) -> pd.DataFrame:
    rows = []
    for run in range(ensemble.n_runs):
        one = _single_run(ensemble, run)

        def objective(w):
            return float(train_loss(w[None, :], one)[0])

        def gradient(w):
            return global_gradient(w[None, :], one)[0]

        result = minimize(
            objective,
            np.zeros(ensemble.dimension),
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": 500, "gtol": 1e-10, "ftol": 1e-12},
        )
        loss, accuracy = test_metrics(result.x[None, :], one)
        rows.append(
            {
                "run": run,
                "train_optimum": float(result.fun),
                "test_loss": float(loss[0]),
                "test_accuracy": float(accuracy[0]),
                "gradient_norm": float(np.linalg.norm(result.jac)),
                "success": bool(result.success),
            }
        )
    return pd.DataFrame(rows)


def _reference_configs():
    return [
        (
            "scheduled_events",
            LogisticRunConfig(
                method="schedule",
                jump0=0.01,
                schedule_exponent=0.1,
            ),
        ),
        ("global_descent_oracle", LogisticRunConfig(method="global_oracle")),
        (
            "certificate_full_C25",
            LogisticRunConfig(
                method="certificate",
                summary_kind="full",
                calibration_period=25,
            ),
        ),
        (
            "ef_topk_k1",
            LogisticRunConfig(method="ef_topk", topk=1, step=0.04),
        ),
        ("full_precision", LogisticRunConfig(method="full", step=0.04)),
    ]


def _selective_configs():
    configs = []
    for gap in [25, 50, 100]:
        configs.append(
            (
                f"selective_coord_gap{gap}",
                SelectiveCertificateConfig(
                    block_size=1,
                    partition="contiguous",
                    summary_kind="block_residual",
                    refresh_policy="on_demand",
                    min_refresh_gap=gap,
                ),
            )
        )
    for block_size, gap in [(5, 25), (5, 50), (10, 25)]:
        configs.append(
            (
                f"selective_block{block_size}_gap{gap}",
                SelectiveCertificateConfig(
                    block_size=block_size,
                    partition="contiguous",
                    summary_kind="block_residual",
                    refresh_policy="on_demand",
                    min_refresh_gap=gap,
                ),
            )
        )
    configs.extend(
        [
            (
                "selective_coord_fullM_gap50",
                SelectiveCertificateConfig(
                    block_size=1,
                    partition="contiguous",
                    summary_kind="full",
                    refresh_policy="on_demand",
                    min_refresh_gap=50,
                ),
            ),
            (
                "coupling_block5_gap25",
                SelectiveCertificateConfig(
                    block_size=5,
                    partition="coupling",
                    summary_kind="block_residual",
                    refresh_policy="on_demand",
                    min_refresh_gap=25,
                ),
            ),
            (
                "coupling_block10_gap25",
                SelectiveCertificateConfig(
                    block_size=10,
                    partition="coupling",
                    summary_kind="block_residual",
                    refresh_policy="on_demand",
                    min_refresh_gap=25,
                ),
            ),
        ]
    )
    for every in [2, 3, 4, 5]:
        configs.append(
            (
                f"round_robin_coord_every{every}",
                SelectiveCertificateConfig(
                    block_size=1,
                    partition="contiguous",
                    summary_kind="block_residual",
                    refresh_policy="round_robin",
                    round_robin_every=every,
                ),
            )
        )
    return configs


def run_campaign(*, quick: bool):
    if quick:
        n_runs, n_ticks, eval_stride = 3, 400, 20
        robustness_seeds = [1300]
    else:
        n_runs, n_ticks, eval_stride = 8, 1200, 30
        robustness_seeds = [1300, 1500, 1700]

    strong = make_synthetic_logistic_ensemble(
        n_runs=n_runs,
        heterogeneity="strong",
        seed=1300,
    )

    rows = []
    histories = {}
    for label, config in _reference_configs():
        result = run_logistic_batch(
            ensemble=strong,
            config=config,
            n_ticks=n_ticks,
            seed=60606,
            eval_stride=eval_stride,
            record_history=label in {
                "scheduled_events",
                "global_descent_oracle",
                "ef_topk_k1",
                "full_precision",
            },
        )
        history = result.pop("history", None)
        rows.append({"configuration": label, "family": "reference", **result})
        if history is not None:
            histories[label] = history

    for label, config in _selective_configs():
        result = run_selective_certificate(
            ensemble=strong,
            config=config,
            n_ticks=n_ticks,
            seed=60606,
            eval_stride=eval_stride,
            record_history=label == "selective_coord_gap50",
        )
        history = result.pop("history", None)
        rows.append({"configuration": label, "family": "selective", **result})
        if history is not None:
            histories[label] = history

    campaign = pd.DataFrame(rows)

    # Explicit event-level safety verification for the leading valid points.
    safety_rows = []
    for label, config in [
        (
            "selective_coord_gap50",
            SelectiveCertificateConfig(
                block_size=1,
                summary_kind="block_residual",
                min_refresh_gap=50,
                verify_harm=True,
            ),
        ),
        (
            "selective_coord_gap100",
            SelectiveCertificateConfig(
                block_size=1,
                summary_kind="block_residual",
                min_refresh_gap=100,
                verify_harm=True,
            ),
        ),
        (
            "coupling_block5_gap25",
            SelectiveCertificateConfig(
                block_size=5,
                partition="coupling",
                summary_kind="block_residual",
                min_refresh_gap=25,
                verify_harm=True,
            ),
        ),
    ]:
        result = run_selective_certificate(
            ensemble=strong,
            config=config,
            n_ticks=n_ticks,
            seed=60606,
            eval_stride=max(eval_stride, 60),
        )
        result.pop("history", None)
        safety_rows.append({"configuration": label, **result})
    safety = pd.DataFrame(safety_rows)

    # Robustness across heterogeneity and independent strong partitions.
    robustness_rows = []
    for regime in ["iid", "moderate", "strong"]:
        seeds = robustness_seeds if regime == "strong" else [1300]
        for data_seed in seeds:
            ensemble = make_synthetic_logistic_ensemble(
                n_runs=n_runs,
                heterogeneity=regime,
                seed=data_seed,
            )
            config = SelectiveCertificateConfig(
                block_size=1,
                summary_kind="block_residual",
                min_refresh_gap=50,
            )
            result = run_selective_certificate(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                seed=60606,
                eval_stride=eval_stride,
            )
            result.pop("history", None)
            reference = centralized_reference(ensemble)
            robustness_rows.append(
                {
                    "heterogeneity": regime,
                    "data_seed": data_seed,
                    "label_rate_std": float(
                        np.mean(np.std(ensemble.positive_rates, axis=1))
                    ),
                    "central_test_loss": float(reference["test_loss"].mean()),
                    "central_test_accuracy": float(
                        reference["test_accuracy"].mean()
                    ),
                    **result,
                }
            )
    robustness = pd.DataFrame(robustness_rows)
    robustness["test_loss_excess"] = (
        robustness["final_test_loss"] - robustness["central_test_loss"]
    )

    return campaign, safety, robustness, histories


def save_outputs(campaign, safety, robustness, histories):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    campaign.to_csv(RESULT_DIR / "campaign.csv", index=False)
    safety.to_csv(RESULT_DIR / "safety.csv", index=False)
    robustness.to_csv(RESULT_DIR / "robustness.csv", index=False)

    selected = campaign[
        campaign["configuration"].isin(
            [
                "scheduled_events",
                "global_descent_oracle",
                "certificate_full_C25",
                "selective_coord_gap50",
                "selective_coord_gap100",
                "ef_topk_k1",
                "full_precision",
            ]
        )
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(selected["payload_bits"], selected["final_test_loss"])
    for _, row in selected.iterrows():
        ax.annotate(
            row["configuration"],
            (row["payload_bits"], row["final_test_loss"]),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total payload bits")
    ax.set_ylabel("Final mean test loss")
    ax.set_title("Experiment 12B: selective calibration Pareto trade-off")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test_loss_vs_bits.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(selected["payload_bits"], selected["whole_train_loss"])
    for _, row in selected.iterrows():
        ax.annotate(
            row["configuration"],
            (row["payload_bits"], row["whole_train_loss"]),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Total payload bits")
    ax.set_ylabel("Whole-run mean training loss")
    ax.set_title("Experiment 12B: transient performance vs communication")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "transient_vs_bits.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [
        f"{row.heterogeneity}-{int(row.data_seed)}"
        for _, row in robustness.iterrows()
    ]
    ax.bar(labels, robustness["test_loss_excess"])
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Test-loss excess vs centralized optimum")
    ax.set_title("Experiment 12B: robustness of selective certification")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "robustness.png", dpi=180)
    plt.close(fig)

    for label, history in histories.items():
        history.to_csv(RESULT_DIR / f"history_{label}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    campaign_, safety_, robustness_, histories_ = run_campaign(quick=args.quick)
    save_outputs(campaign_, safety_, robustness_, histories_)
    print(campaign_.sort_values(["final_test_loss", "payload_bits"]).to_string(index=False))
    print("\nSafety:\n", safety_.to_string(index=False))
    print("\nRobustness:\n", robustness_.to_string(index=False))
