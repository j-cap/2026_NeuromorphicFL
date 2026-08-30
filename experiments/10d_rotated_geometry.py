from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.rotated_geometry import (
    RotatedRunConfig,
    make_isospectral_common_rotation_ensemble,
    run_rotated_batch,
)


RESULT_DIR = Path("experiments/results/10d_rotated_geometry")


def run_primary(*, quick: bool) -> pd.DataFrame:
    if quick:
        n_runs, n_ticks, tail = 12, 500, 125
        strengths = [0.0, 0.5, 1.0]
    else:
        n_runs, n_ticks, tail = 40, 2000, 500
        strengths = [0.0, 0.25, 0.5, 1.0]

    rows: list[dict[str, float | str]] = []
    for strength in strengths:
        ensemble = make_isospectral_common_rotation_ensemble(
            n_runs=n_runs, strength=strength, seed=222
        )
        geometry = {
            "strength": strength,
            "offdiag_ratio": float(np.mean(ensemble.offdiag_ratio)),
            "condition_number": float(np.mean(ensemble.condition_number)),
            "initial_gap": float(np.mean(ensemble.initial_gap)),
        }
        configs = [
            (
                "LIF-native-diag",
                RotatedRunConfig(method="lif_schedule", threshold_mode="diag"),
            ),
            (
                "LIF-native-uniform",
                RotatedRunConfig(method="lif_schedule", threshold_mode="uniform"),
            ),
            (
                "LIF-basis-oracle",
                RotatedRunConfig(method="lif_basis", threshold_mode="basis"),
            ),
            (
                "EF-TopK",
                RotatedRunConfig(method="ef_topk", step=0.01, topk=4),
            ),
            (
                "Full precision",
                RotatedRunConfig(method="full", step=0.02),
            ),
            (
                "Global descent oracle",
                RotatedRunConfig(method="global_oracle", threshold_mode="diag"),
            ),
        ]
        for label, config in configs:
            result = run_rotated_batch(
                ensemble=ensemble,
                config=config,
                n_ticks=n_ticks,
                tail=tail,
                seed=30303,
            )
            rows.append({"label": label, **geometry, **result})
    return pd.DataFrame(rows)


def run_normalization_check(*, quick: bool) -> pd.DataFrame:
    n_runs, n_ticks, tail = (12, 500, 125) if quick else (40, 2000, 500)
    ensemble = make_isospectral_common_rotation_ensemble(
        n_runs=n_runs, strength=1.0, seed=222
    )
    rows = []
    for label, mode in [
        ("Inherited eigenvalue-index", "inherited"),
        ("Oracle native diag(Hbar)", "diag"),
        ("Uniform", "uniform"),
    ]:
        result = run_rotated_batch(
            ensemble=ensemble,
            config=RotatedRunConfig(
                method="lif_schedule", threshold_mode=mode
            ),
            n_ticks=n_ticks,
            tail=tail,
            seed=30303,
        )
        rows.append({"normalization": label, **result})
    return pd.DataFrame(rows)


def run_schedule_tuning(*, quick: bool) -> pd.DataFrame:
    if quick:
        n_runs, n_ticks, tail = 10, 500, 125
        jumps = [0.03, 0.05]
        scales = [500.0]
        exponents = [0.25, 0.5]
        threshold_modes = ["diag"]
    else:
        n_runs, n_ticks, tail = 20, 2000, 500
        jumps = [0.03, 0.04, 0.05, 0.06]
        scales = [500.0, 1000.0]
        exponents = [0.25, 0.5]
        threshold_modes = ["diag", "uniform"]

    ensemble = make_isospectral_common_rotation_ensemble(
        n_runs=n_runs, strength=1.0, seed=222
    )
    rows = []
    for threshold_mode in threshold_modes:
        for jump in jumps:
            for scale in scales:
                for exponent in exponents:
                    config = RotatedRunConfig(
                        method="lif_schedule",
                        threshold_mode=threshold_mode,
                        jump0=jump,
                        schedule_scale=scale,
                        schedule_exponent=exponent,
                    )
                    result = run_rotated_batch(
                        ensemble=ensemble,
                        config=config,
                        n_ticks=n_ticks,
                        tail=tail,
                        seed=30303,
                    )
                    rows.append(
                        {
                            "threshold_mode": threshold_mode,
                            "jump0": jump,
                            "schedule_scale": scale,
                            "schedule_exponent": exponent,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def run_long_horizon(*, quick: bool) -> pd.DataFrame:
    if quick:
        n_runs, n_ticks, tail = 10, 800, 200
    else:
        n_runs, n_ticks, tail = 40, 4000, 1000
    ensemble = make_isospectral_common_rotation_ensemble(
        n_runs=n_runs, strength=1.0, seed=222
    )
    configs = [
        (
            "Native LIF retuned",
            RotatedRunConfig(
                method="lif_schedule",
                threshold_mode="diag",
                jump0=0.05,
                schedule_scale=500.0,
                schedule_exponent=0.5,
            ),
        ),
        (
            "Basis-aligned LIF oracle",
            RotatedRunConfig(method="lif_basis", threshold_mode="basis"),
        ),
        (
            "EF-TopK k=4",
            RotatedRunConfig(method="ef_topk", step=0.01, topk=4),
        ),
    ]
    rows = []
    for label, config in configs:
        result = run_rotated_batch(
            ensemble=ensemble,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=30303,
        )
        rows.append({"label": label, "ticks": n_ticks, **result})
    return pd.DataFrame(rows)


def make_figures(primary: pd.DataFrame, validation: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    for label in [
        "LIF-native-diag",
        "LIF-basis-oracle",
        "EF-TopK",
        "Full precision",
        "Global descent oracle",
    ]:
        subset = primary[primary["label"] == label]
        plt.plot(
            subset["offdiag_ratio"],
            subset["tail_gap"],
            marker="o",
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("Aggregate off-diagonal energy ratio")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 10D: native coordinate coding is not rotation invariant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "geometry_tail.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for label in ["LIF-native-diag", "LIF-basis-oracle", "EF-TopK"]:
        subset = primary[primary["label"] == label]
        plt.plot(
            subset["offdiag_ratio"],
            subset["payload_bits"],
            marker="o",
            label=label,
        )
    plt.yscale("log")
    plt.xlabel("Aggregate off-diagonal energy ratio")
    plt.ylabel("Mean logical payload bits")
    plt.title("Experiment 10D: misalignment increases native event traffic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "geometry_bits.png", dpi=180)
    plt.close()

    strong = primary[primary["strength"] == primary["strength"].max()]
    selected = strong[strong["label"].isin(["LIF-native-diag", "LIF-basis-oracle", "EF-TopK"])]
    plt.figure(figsize=(8, 5))
    plt.scatter(selected["payload_bits"], selected["tail_gap"], s=70)
    for _, row in selected.iterrows():
        plt.annotate(row["label"], (row["payload_bits"], row["tail_gap"]), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 10D: full-rotation trade-off")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "full_rotation_tradeoff.png", dpi=180)
    plt.close()

    if len(validation):
        validation.to_csv(RESULT_DIR / "long_horizon_validation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    primary = run_primary(quick=args.quick)
    normalization = run_normalization_check(quick=args.quick)
    tuning = run_schedule_tuning(quick=args.quick)
    validation = run_long_horizon(quick=args.quick)

    primary.to_csv(RESULT_DIR / "primary_geometry_sweep.csv", index=False)
    normalization.to_csv(RESULT_DIR / "normalization_check.csv", index=False)
    tuning.to_csv(RESULT_DIR / "strong_rotation_schedule_tuning.csv", index=False)
    validation.to_csv(RESULT_DIR / "long_horizon_validation.csv", index=False)
    make_figures(primary, validation)

    best_tuning = tuning.sort_values("tail_gap").head(10)
    best_tuning.to_csv(RESULT_DIR / "best_schedule_points.csv", index=False)

    print("Experiment 10D complete")
    print(
        primary[
            [
                "strength",
                "offdiag_ratio",
                "label",
                "tail_gap",
                "whole_gap",
                "payload_bits",
                "events",
                "acceptance_fraction",
            ]
        ].to_string(index=False)
    )
    print("\nBest full-rotation schedule points")
    print(
        best_tuning[
            [
                "threshold_mode",
                "jump0",
                "schedule_scale",
                "schedule_exponent",
                "tail_gap",
                "whole_gap",
                "payload_bits",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
