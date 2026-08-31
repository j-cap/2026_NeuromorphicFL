from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from neuromorphicfl.rotated_geometry import (
    RotatedRunConfig,
    make_isospectral_common_rotation_ensemble,
    run_rotated_batch,
)
from neuromorphicfl.selective_calibration import (
    SelectiveCalibrationConfig,
    run_selective_calibration_batch,
)


RESULT_DIR = Path("experiments/results/11c_selective_calibration")


def _row(
    *,
    geometry: str,
    label: str,
    family: str,
    result: dict[str, float],
    k: int | None = None,
    calibration_period: int | None = None,
    policy: str | None = None,
    data_seed: int = 222,
) -> dict[str, object]:
    return {
        "geometry": geometry,
        "configuration": label,
        "family": family,
        "policy": policy,
        "k": k,
        "calibration_period": calibration_period,
        "data_seed": data_seed,
        **result,
    }


def run_campaign(*, quick: bool):
    if quick:
        n_runs, n_ticks, tail = 6, 600, 150
        robustness_runs = 4
        robustness_seeds = [333]
    else:
        # The main run is supplemented by two independent held-out ensembles below.
        n_runs, n_ticks, tail = 20, 2000, 500
        robustness_runs = 10
        robustness_seeds = [333, 444]

    rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Diagonal control: certificate maintenance is degenerate because each
    # component evolves independently. One initial full calibration is enough.
    # ------------------------------------------------------------------
    diagonal = make_isospectral_common_rotation_ensemble(
        n_runs=n_runs,
        strength=0.0,
        seed=222,
    )
    for label, config in [
        (
            "initial calibration only",
            SelectiveCalibrationConfig(
                policy="full",
                k=diagonal.dimension,
                calibration_period=0,
            ),
        ),
        (
            "full C=1000",
            SelectiveCalibrationConfig(
                policy="full",
                k=diagonal.dimension,
                calibration_period=1000,
            ),
        ),
    ]:
        result = run_selective_calibration_batch(
            ensemble=diagonal,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        rows.append(
            _row(
                geometry="diagonal",
                label=label,
                family="certificate",
                result=result,
                k=config.k,
                calibration_period=config.calibration_period,
                policy=config.policy,
            )
        )

    # ------------------------------------------------------------------
    # Primary discriminating geometry: strong common rotation from 10D/11B.
    # ------------------------------------------------------------------
    rotated = make_isospectral_common_rotation_ensemble(
        n_runs=n_runs,
        strength=1.0,
        seed=222,
    )

    full_configs = [25, 50, 75, 100]
    for period in full_configs:
        config = SelectiveCalibrationConfig(
            policy="full",
            k=rotated.dimension,
            calibration_period=period,
        )
        result = run_selective_calibration_batch(
            ensemble=rotated,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        rows.append(
            _row(
                geometry="rotated",
                label=f"full C={period}",
                family="full calibration",
                result=result,
                k=rotated.dimension,
                calibration_period=period,
                policy="full",
            )
        )

    selective = [
        ("abs-R K=4 C=25", "uncertainty_abs", 4, 25),
        ("abs-R K=8 C=25", "uncertainty_abs", 8, 25),
        ("abs-R K=12 C=25", "uncertainty_abs", 12, 25),
        ("abs-R K=8 C=50", "uncertainty_abs", 8, 50),
        ("round-robin K=8 C=25", "round_robin", 8, 25),
        ("random K=8 C=25", "random", 8, 25),
        # Original proposal retained explicitly as a negative control.
        ("relative-R K=8 C=25", "uncertainty_relative", 8, 25),
    ]
    for label, policy, k, period in selective:
        config = SelectiveCalibrationConfig(
            policy=policy,
            k=k,
            calibration_period=period,
        )
        result = run_selective_calibration_batch(
            ensemble=rotated,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        rows.append(
            _row(
                geometry="rotated",
                label=label,
                family="selective calibration",
                result=result,
                k=k,
                calibration_period=period,
                policy=policy,
            )
        )

    # Conventional references on exactly the same controlled geometry.
    for label, config in [
        (
            "q(t) events",
            RotatedRunConfig(
                method="lif_schedule",
                threshold_mode="diag",
                jump0=0.05,
                schedule_scale=500.0,
                schedule_exponent=0.5,
            ),
        ),
        (
            "EF-TopK k=4",
            RotatedRunConfig(method="ef_topk", step=0.01, topk=4),
        ),
        (
            "full precision",
            RotatedRunConfig(method="full", step=0.02),
        ),
        (
            "global descent oracle",
            RotatedRunConfig(method="global_oracle", threshold_mode="diag"),
        ),
    ]:
        result = run_rotated_batch(
            ensemble=rotated,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        rows.append(
            _row(
                geometry="rotated",
                label=label,
                family="reference",
                result=result,
            )
        )

    campaign = pd.DataFrame(rows)

    # Independent strong-rotation ensembles for the main selective claims.
    robustness_rows: list[dict[str, object]] = []
    for data_seed in robustness_seeds:
        ensemble = make_isospectral_common_rotation_ensemble(
            n_runs=robustness_runs,
            strength=1.0,
            seed=data_seed,
        )
        configs = [
            ("full C=50", "full", ensemble.dimension, 50),
            ("full C=75", "full", ensemble.dimension, 75),
            ("abs-R K=4 C=25", "uncertainty_abs", 4, 25),
            ("abs-R K=8 C=25", "uncertainty_abs", 8, 25),
            ("round-robin K=8 C=25", "round_robin", 8, 25),
            ("random K=8 C=25", "random", 8, 25),
        ]
        for label, policy, k, period in configs:
            config = SelectiveCalibrationConfig(
                policy=policy,
                k=k,
                calibration_period=period,
            )
            result = run_selective_calibration_batch(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
            )
            robustness_rows.append(
                _row(
                    geometry="rotated",
                    label=label,
                    family="robustness",
                    result=result,
                    k=k,
                    calibration_period=period,
                    policy=policy,
                    data_seed=data_seed,
                )
            )

    return campaign, pd.DataFrame(robustness_rows)


def save_outputs(campaign: pd.DataFrame, robustness: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    campaign.to_csv(RESULT_DIR / "campaign.csv", index=False)
    robustness.to_csv(RESULT_DIR / "robustness.csv", index=False)

    rotated = campaign[campaign["geometry"] == "rotated"].copy()
    selected = rotated[
        rotated["configuration"].isin(
            [
                "full C=25",
                "full C=50",
                "full C=75",
                "abs-R K=4 C=25",
                "abs-R K=8 C=25",
                "abs-R K=12 C=25",
                "round-robin K=8 C=25",
                "random K=8 C=25",
                "relative-R K=8 C=25",
                "q(t) events",
                "EF-TopK k=4",
                "full precision",
            ]
        )
    ].copy()
    selected["tail_plot"] = selected["tail_gap"].clip(lower=1e-8)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(selected["payload_bits"], selected["tail_plot"])
    for _, row in selected.iterrows():
        ax.annotate(
            row["configuration"],
            (row["payload_bits"], row["tail_plot"]),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean logical payload bits")
    ax.set_ylabel("Tail excess objective")
    ax.set_title("Experiment 11C: selective calibration trade-off")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "error_vs_bits.png", dpi=180)
    plt.close(fig)

    policies = rotated[
        rotated["configuration"].isin(
            [
                "abs-R K=8 C=25",
                "round-robin K=8 C=25",
                "random K=8 C=25",
                "relative-R K=8 C=25",
            ]
        )
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(policies["payload_bits"], policies["tail_gap"])
    for _, row in policies.iterrows():
        ax.annotate(
            row["configuration"],
            (row["payload_bits"], row["tail_gap"]),
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Mean logical payload bits")
    ax.set_ylabel("Tail excess objective")
    ax.set_title("Experiment 11C: refresh-policy comparison")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "policy_comparison.png", dpi=180)
    plt.close(fig)

    composition = rotated[
        rotated["configuration"].isin(
            [
                "full C=50",
                "full C=75",
                "abs-R K=4 C=25",
                "abs-R K=8 C=25",
                "abs-R K=12 C=25",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        composition["configuration"],
        composition["calibration_bits"],
        label="calibration",
    )
    ax.bar(
        composition["configuration"],
        composition["event_bits"],
        bottom=composition["calibration_bits"],
        label="sparse events",
    )
    ax.set_ylabel("Mean logical payload bits")
    ax.set_title("Experiment 11C: calibration remains the dominant payload")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "bit_composition.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    campaign_, robustness_ = run_campaign(quick=args.quick)
    save_outputs(campaign_, robustness_)

    columns = [
        "geometry",
        "configuration",
        "tail_gap",
        "whole_gap",
        "payload_bits",
        "calibration_bits",
        "event_bits",
        "calibration_fraction",
        "accepted_events",
        "harmful_applied_fraction",
        "maintenance_coordinate_coverage",
        "maintenance_selection_cv",
    ]
    available = [c for c in columns if c in campaign_.columns]
    print(campaign_[available].to_string(index=False))
    print("\nIndependent-fleet robustness:\n")
    print(robustness_[available].to_string(index=False))
