from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from neuromorphicfl.logistic_certificate import make_synthetic_logistic_ensemble
from neuromorphicfl.mlp_depth_scaling import ARCHITECTURES, run_depth_batch

RESULT_DIR = Path("experiments/results/13b_depth_scaling")


def initialization_scale(name: str) -> float:
    # Experiment 13A's small initialization remains suitable for one-hidden-
    # layer networks. A common 0.5/sqrt(fan_in) scale is used for both deeper
    # networks after the method-independent depth initialization audit.
    return 0.18 if ARCHITECTURES[name].n_layers == 2 else 0.5


def run_campaign(*, quick: bool):
    if quick:
        n_runs, n_ticks, eval_stride = 3, 500, 50
        robustness_runs = 3
        robustness_seeds = [1500]
        long_ticks = 800
    else:
        n_runs, n_ticks, eval_stride = 8, 1200, 60
        robustness_runs = 6
        robustness_seeds = [1500, 1700]
        long_ticks = 2400

    strong = make_synthetic_logistic_ensemble(
        n_runs=n_runs, heterogeneity="strong", seed=1300
    )

    rows = []
    histories = {}
    for name, layout in ARCHITECTURES.items():
        scale = initialization_scale(name)
        for method in ["events", "ef_topk", "full"]:
            result = run_depth_batch(
                ensemble=strong,
                layout=layout,
                method=method,
                initialization_scale=scale,
                n_ticks=n_ticks,
                eval_stride=eval_stride,
                record_history=method == "events",
            )
            history = result.pop("history", None)
            rows.append({"architecture": name, **result})
            if history is not None:
                histories[name] = history
    primary = pd.DataFrame(rows)

    # Diagnostic: the frozen 13A initialization is too small after adding an
    # extra tanh layer. This audit is deliberately common to all optimizers.
    initialization_rows = []
    for name in ["deep_8x8", "deep_16x16"]:
        layout = ARCHITECTURES[name]
        for scale in [0.18, 0.5, 1.0]:
            for method in ["events", "ef_topk", "full"]:
                result = run_depth_batch(
                    ensemble=make_synthetic_logistic_ensemble(
                        n_runs=3, heterogeneity="strong", seed=1300
                    ),
                    layout=layout,
                    method=method,
                    initialization_scale=scale,
                    n_ticks=500,
                    eval_stride=50,
                )
                initialization_rows.append({"architecture": name, **result})
    initialization = pd.DataFrame(initialization_rows)

    robustness_rows = []
    for data_seed in robustness_seeds:
        ensemble = make_synthetic_logistic_ensemble(
            n_runs=robustness_runs,
            heterogeneity="strong",
            seed=data_seed,
        )
        for name in ["shallow_12", "deep_8x8", "shallow_32", "deep_16x16"]:
            layout = ARCHITECTURES[name]
            scale = initialization_scale(name)
            for method in ["events", "ef_topk"]:
                result = run_depth_batch(
                    ensemble=ensemble,
                    layout=layout,
                    method=method,
                    initialization_scale=scale,
                    n_ticks=n_ticks,
                    eval_stride=max(eval_stride, 120),
                )
                robustness_rows.append(
                    {"data_seed": data_seed, "architecture": name, **result}
                )
    robustness = pd.DataFrame(robustness_rows)

    long_rows = []
    for name in ["deep_8x8", "deep_16x16"]:
        layout = ARCHITECTURES[name]
        for method in ["events", "ef_topk", "full"]:
            result = run_depth_batch(
                ensemble=strong,
                layout=layout,
                method=method,
                initialization_scale=0.5,
                n_ticks=long_ticks,
                eval_stride=120,
            )
            long_rows.append({"architecture": name, **result})
    long_horizon = pd.DataFrame(long_rows)

    return primary, initialization, robustness, long_horizon, histories


def save_outputs(outputs):
    primary, initialization, robustness, long_horizon, histories = outputs
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    primary.to_csv(RESULT_DIR / "primary.csv", index=False)
    initialization.to_csv(RESULT_DIR / "initialization_audit.csv", index=False)
    robustness.to_csv(RESULT_DIR / "robustness.csv", index=False)
    long_horizon.to_csv(RESULT_DIR / "long_horizon.csv", index=False)
    for name, history in histories.items():
        history.to_csv(RESULT_DIR / f"history_events_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, sub in primary.groupby("method"):
        ax.scatter(sub["payload_bits"], sub["final_test_cross_entropy"], label=method)
    for _, row in primary[primary["method"] == "events"].iterrows():
        ax.annotate(
            row["architecture"],
            (row["payload_bits"], row["final_test_cross_entropy"]),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean logical payload bits")
    ax.set_ylabel("Final predictive test cross-entropy")
    ax.set_title("Experiment 13B: depth/dimension scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test_ce_vs_bits.png", dpi=180)
    plt.close(fig)

    event_rows = primary[primary["method"] == "events"]
    tensor_rows = []
    for _, row in event_rows[event_rows["architecture"].isin(["deep_8x8", "deep_16x16"])].iterrows():
        for tensor in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            tensor_rows.append({
                "architecture": row["architecture"],
                "tensor": tensor,
                "events_per_parameter": row[f"{tensor}_events_per_parameter"],
                "never_fired_fraction": row[f"{tensor}_never_fired_fraction"],
            })
    pd.DataFrame(tensor_rows).to_csv(RESULT_DIR / "layer_diagnostics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    outputs_ = run_campaign(quick=args.quick)
    save_outputs(outputs_)
    for name, table in zip(
        ["primary", "initialization", "robustness", "long_horizon"],
        outputs_[:-1],
    ):
        print(f"\n{name}\n")
        print(table.to_string(index=False))
