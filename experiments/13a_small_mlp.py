from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from neuromorphicfl.logistic_certificate import make_synthetic_logistic_ensemble
from neuromorphicfl.mlp_event_gate import (
    LAYOUT,
    MLPEventConfig,
    initialize_mlp,
    loss_and_gradient,
    run_mlp_batch,
    test_metrics,
)

RESULT_DIR = Path("experiments/results/13a_small_mlp")


def centralized_reference(ensemble) -> pd.DataFrame:
    rows = []
    w0 = initialize_mlp(ensemble.n_runs)
    for run in range(ensemble.n_runs):
        one = replace(
            ensemble,
            Xtr=ensemble.Xtr[run : run + 1],
            ytr=ensemble.ytr[run : run + 1],
            Xte=ensemble.Xte[run : run + 1],
            yte=ensemble.yte[run : run + 1],
            w0=ensemble.w0[run : run + 1],
            positive_rates=ensemble.positive_rates[run : run + 1],
        )
        X = one.Xtr[..., :-1].reshape(1, -1, LAYOUT.input_dim)
        y = one.ytr.reshape(1, -1)

        def fg(w):
            loss, gradient = loss_and_gradient(w[None, :], X, y)
            return float(loss[0]), gradient[0]

        solution = minimize(
            lambda w: fg(w)[0],
            w0[run],
            jac=lambda w: fg(w)[1],
            method="L-BFGS-B",
            options={"maxiter": 300, "gtol": 1e-8, "ftol": 1e-10},
        )
        loss, accuracy = test_metrics(solution.x[None, :], one)
        rows.append(
            {
                "run": run,
                "train_loss": float(solution.fun),
                "test_loss": float(loss[0]),
                "test_accuracy": float(accuracy[0]),
                "gradient_norm": float(np.linalg.norm(solution.jac)),
                "iterations": int(solution.nit),
                "success": bool(solution.success),
            }
        )
    return pd.DataFrame(rows)


def run_campaign(*, quick: bool):
    if quick:
        n_runs, n_ticks, eval_stride = 3, 400, 40
        robustness_seeds = [1500]
    else:
        n_runs, n_ticks, eval_stride = 8, 1200, 60
        robustness_seeds = [1500, 1700]

    strong = make_synthetic_logistic_ensemble(
        n_runs=n_runs, heterogeneity="strong", seed=1300
    )
    rows = []
    histories = {}
    configs = [
        ("events_raw_q015", MLPEventConfig(method="schedule", normalization="raw", jump0=0.015)),
        ("events_raw_q020", MLPEventConfig(method="schedule", normalization="raw", jump0=0.020)),
        ("ef_topk_k1", MLPEventConfig(method="ef_topk", topk=1, step=0.04)),
        ("ef_topk_k4", MLPEventConfig(method="ef_topk", topk=4, step=0.04)),
        ("ef_topk_k8", MLPEventConfig(method="ef_topk", topk=8, step=0.04)),
        ("ef_topk_k16", MLPEventConfig(method="ef_topk", topk=16, step=0.04)),
        ("full_precision", MLPEventConfig(method="full", step=0.04)),
    ]
    for label, config in configs:
        result = run_mlp_batch(
            ensemble=strong,
            config=config,
            n_ticks=n_ticks,
            seed=60606,
            eval_stride=eval_stride,
            record_history=label in {
                "events_raw_q015", "ef_topk_k4", "ef_topk_k8", "full_precision"
            },
        )
        history = result.pop("history", None)
        rows.append({"configuration": label, **result})
        if history is not None:
            histories[label] = history
    campaign = pd.DataFrame(rows)

    normalization_rows = []
    for kind in ["raw", "layer", "coordinate"]:
        for jump0 in [0.015, 0.020]:
            result = run_mlp_batch(
                ensemble=strong,
                config=MLPEventConfig(
                    method="schedule", normalization=kind, jump0=jump0
                ),
                n_ticks=n_ticks,
                seed=60606,
                eval_stride=eval_stride,
            )
            normalization_rows.append({"jump0": jump0, **result})
    normalization = pd.DataFrame(normalization_rows)

    heterogeneity_rows = []
    central_rows = []
    for regime in ["iid", "moderate", "strong"]:
        ensemble = make_synthetic_logistic_ensemble(
            n_runs=n_runs, heterogeneity=regime, seed=1300
        )
        central = centralized_reference(ensemble)
        central.insert(0, "heterogeneity", regime)
        central_rows.append(central)
        for label, config in [
            ("events_raw_q015", MLPEventConfig(method="schedule", normalization="raw", jump0=0.015)),
            ("ef_topk_k4", MLPEventConfig(method="ef_topk", topk=4, step=0.04)),
            ("full_precision", MLPEventConfig(method="full", step=0.04)),
        ]:
            result = run_mlp_batch(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                seed=60606,
                eval_stride=eval_stride,
            )
            heterogeneity_rows.append(
                {
                    "heterogeneity": regime,
                    "configuration": label,
                    "label_rate_std": float(
                        np.mean(np.std(ensemble.positive_rates, axis=1))
                    ),
                    **result,
                }
            )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    centralized = pd.concat(central_rows, ignore_index=True)
    central_mean = centralized.groupby("heterogeneity")["test_loss"].mean()
    heterogeneity["central_test_loss"] = heterogeneity["heterogeneity"].map(central_mean)
    heterogeneity["test_loss_excess"] = (
        heterogeneity["final_test_loss"] - heterogeneity["central_test_loss"]
    )

    oracle_ensemble = make_synthetic_logistic_ensemble(
        n_runs=1 if quick else 2,
        heterogeneity="strong",
        seed=1300,
    )
    oracle_rows = []
    oracle_ticks = 120 if quick else 250
    for label, config in [
        ("scheduled_events", MLPEventConfig(method="schedule", normalization="raw", jump0=0.020)),
        ("packet_descent_oracle", MLPEventConfig(method="packet_descent_oracle", normalization="raw", jump0=0.020)),
        ("ef_topk_k8", MLPEventConfig(method="ef_topk", topk=8, step=0.04)),
        ("full_precision", MLPEventConfig(method="full", step=0.04)),
    ]:
        result = run_mlp_batch(
            ensemble=oracle_ensemble,
            config=config,
            n_ticks=oracle_ticks,
            seed=60606,
            eval_stride=25,
        )
        oracle_rows.append({"configuration": label, **result})
    oracle = pd.DataFrame(oracle_rows)

    robustness_rows = []
    for data_seed in robustness_seeds:
        ensemble = make_synthetic_logistic_ensemble(
            n_runs=n_runs, heterogeneity="strong", seed=data_seed
        )
        for label, config in [
            ("events_raw_q015", MLPEventConfig(method="schedule", normalization="raw", jump0=0.015)),
            ("ef_topk_k4", MLPEventConfig(method="ef_topk", topk=4, step=0.04)),
            ("full_precision", MLPEventConfig(method="full", step=0.04)),
        ]:
            result = run_mlp_batch(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                seed=60606,
                eval_stride=eval_stride,
            )
            robustness_rows.append(
                {"data_seed": data_seed, "configuration": label, **result}
            )
    robustness = pd.DataFrame(robustness_rows)

    long_rows = []
    long_ticks = 800 if quick else 2400
    for label, config in [
        ("events_raw_q015", MLPEventConfig(method="schedule", normalization="raw", jump0=0.015)),
        ("ef_topk_k4", MLPEventConfig(method="ef_topk", topk=4, step=0.04)),
        ("ef_topk_k8", MLPEventConfig(method="ef_topk", topk=8, step=0.04)),
        ("full_precision", MLPEventConfig(method="full", step=0.04)),
    ]:
        result = run_mlp_batch(
            ensemble=strong,
            config=config,
            n_ticks=long_ticks,
            seed=60606,
            eval_stride=max(eval_stride, 120),
        )
        long_rows.append({"configuration": label, **result})
    long_horizon = pd.DataFrame(long_rows)

    return campaign, normalization, heterogeneity, centralized, oracle, robustness, long_horizon, histories


def save_outputs(outputs):
    campaign, normalization, heterogeneity, centralized, oracle, robustness, long_horizon, histories = outputs
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    campaign.to_csv(RESULT_DIR / "campaign_strong.csv", index=False)
    normalization.to_csv(RESULT_DIR / "normalization_audit.csv", index=False)
    heterogeneity.to_csv(RESULT_DIR / "heterogeneity.csv", index=False)
    centralized.to_csv(RESULT_DIR / "centralized_reference.csv", index=False)
    oracle.to_csv(RESULT_DIR / "packet_oracle_diagnostic.csv", index=False)
    robustness.to_csv(RESULT_DIR / "robustness.csv", index=False)
    long_horizon.to_csv(RESULT_DIR / "long_horizon.csv", index=False)
    for label, history in histories.items():
        history.to_csv(RESULT_DIR / f"history_{label}.csv", index=False)

    selected = campaign[campaign["configuration"].isin(
        ["events_raw_q015", "ef_topk_k4", "ef_topk_k8", "full_precision"]
    )]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(selected["payload_bits"], selected["final_test_loss"])
    for _, row in selected.iterrows():
        ax.annotate(row["configuration"], (row["payload_bits"], row["final_test_loss"]), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Mean logical payload bits")
    ax.set_ylabel("Final mean test loss")
    ax.set_title("Experiment 13A: nonconvex MLP performance vs communication")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test_loss_vs_bits.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, history in histories.items():
        ax.plot(history["payload_bits"], history["test_loss"], marker="o", markersize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative mean payload bits")
    ax.set_ylabel("Mean test loss")
    ax.set_title("Experiment 13A: learning trajectory vs communication")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "history_test_loss_vs_bits.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    outputs_ = run_campaign(quick=args.quick)
    save_outputs(outputs_)
    names = ["campaign", "normalization", "heterogeneity", "centralized", "oracle", "robustness", "long_horizon"]
    for name, table in zip(names, outputs_[:-1]):
        print(f"\n{name}\n")
        print(table.to_string(index=False))
