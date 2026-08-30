from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.calibrated_certificate import (
    CalibratedCertificateConfig,
    estimate_first_passage_error_margins,
    run_basis_aligned_certificate_batch,
    run_calibrated_certificate_batch,
)
from neuromorphicfl.rotated_geometry import (
    RotatedRunConfig,
    make_isospectral_common_rotation_ensemble,
    run_rotated_batch,
)


RESULT_DIR = Path("experiments/results/11b_calibrated_certificate")


def _record(
    rows: list[dict[str, object]],
    *,
    geometry: str,
    method: str,
    result: dict[str, float],
    calibration_period: float = np.nan,
    refresh_gap: float = np.nan,
) -> None:
    rows.append(
        {
            "geometry": geometry,
            "method": method,
            "calibration_period": calibration_period,
            "refresh_gap": refresh_gap,
            **result,
        }
    )


def run_campaign(*, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quick:
        n_runs, n_ticks, tail = 10, 500, 125
        periods = [50, 100, 250]
        transition = [25, 50, 100]
        calibration_runs = 8
        calibration_ticks = 400
    else:
        n_runs, n_ticks, tail = 40, 2000, 500
        periods = [50, 100, 200, 500, 1000]
        transition = [25, 40, 50, 60, 75, 100]
        calibration_runs = 20
        calibration_ticks = 1200

    rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []

    for strength, geometry in [(0.0, "diagonal"), (1.0, "rotated")]:
        ensemble = make_isospectral_common_rotation_ensemble(
            n_runs=n_runs,
            strength=strength,
            seed=222,
        )
        calibration_ensemble = make_isospectral_common_rotation_ensemble(
            n_runs=calibration_runs,
            strength=strength,
            seed=111,
        )
        margins99 = estimate_first_passage_error_margins(
            ensemble=calibration_ensemble,
            quantile=0.99,
            n_ticks=calibration_ticks,
        )
        margins999 = estimate_first_passage_error_margins(
            ensemble=calibration_ensemble,
            quantile=0.999,
            n_ticks=calibration_ticks,
        )
        for period in sorted(margins99):
            margin_rows.append(
                {
                    "geometry": geometry,
                    "client_period": period,
                    "margin_q99": margins99[period],
                    "margin_q999": margins999[period],
                }
            )

        schedule = (
            CalibratedCertificateConfig(
                method="schedule",
                jump0=0.02,
                schedule_scale=500.0,
                schedule_exponent=0.25,
            )
            if geometry == "diagonal"
            else CalibratedCertificateConfig(
                method="schedule",
                jump0=0.05,
                schedule_scale=500.0,
                schedule_exponent=0.5,
            )
        )
        result = run_calibrated_certificate_batch(
            ensemble=ensemble,
            config=schedule,
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        _record(rows, geometry=geometry, method="q(t) baseline", result=result)

        result = run_calibrated_certificate_batch(
            ensemble=ensemble,
            config=CalibratedCertificateConfig(method="global_oracle"),
            n_ticks=n_ticks,
            tail=tail,
            seed=50505,
        )
        _record(rows, geometry=geometry, method="global oracle", result=result)

        for calibration_period in periods:
            result = run_calibrated_certificate_batch(
                ensemble=ensemble,
                config=CalibratedCertificateConfig(
                    method="periodic_cert",
                    calibration_period=calibration_period,
                    bound_mode="componentwise",
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
            )
            _record(
                rows,
                geometry=geometry,
                method="periodic certificate",
                calibration_period=calibration_period,
                result=result,
            )

        for calibration_period in [periods[1], periods[-2], periods[-1]]:
            result = run_calibrated_certificate_batch(
                ensemble=ensemble,
                config=CalibratedCertificateConfig(
                    method="periodic_event_cert",
                    calibration_period=calibration_period,
                    bound_mode="componentwise",
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
                event_error_margins=margins99,
            )
            _record(
                rows,
                geometry=geometry,
                method="event-tightened q99",
                calibration_period=calibration_period,
                result=result,
            )
            result = run_calibrated_certificate_batch(
                ensemble=ensemble,
                config=CalibratedCertificateConfig(
                    method="periodic_event_cert",
                    calibration_period=calibration_period,
                    bound_mode="componentwise",
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
                event_error_margins=margins999,
            )
            _record(
                rows,
                geometry=geometry,
                method="event-tightened q999",
                calibration_period=calibration_period,
                result=result,
            )

        for refresh_gap in ([50, 100, 200] if not quick else [50, 100]):
            result = run_calibrated_certificate_batch(
                ensemble=ensemble,
                config=CalibratedCertificateConfig(
                    method="adaptive_cert",
                    min_refresh_gap=refresh_gap,
                    bound_mode="componentwise",
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
            )
            _record(
                rows,
                geometry=geometry,
                method="adaptive refresh",
                refresh_gap=refresh_gap,
                result=result,
            )

        for calibration_period in ([200, 1000] if not quick else [100, 250]):
            result = run_calibrated_certificate_batch(
                ensemble=ensemble,
                config=CalibratedCertificateConfig(
                    method="periodic_naive",
                    calibration_period=calibration_period,
                ),
                n_ticks=n_ticks,
                tail=tail,
                seed=50505,
            )
            _record(
                rows,
                geometry=geometry,
                method="naive stale calibration",
                calibration_period=calibration_period,
                result=result,
            )

        if geometry == "rotated":
            for calibration_period in ([100, 500, 1000] if not quick else [100, 250]):
                result = run_basis_aligned_certificate_batch(
                    ensemble=ensemble,
                    calibration_period=calibration_period,
                    n_ticks=n_ticks,
                    tail=tail,
                    seed=50505,
                )
                _record(
                    rows,
                    geometry=geometry,
                    method="basis-aligned certificate oracle",
                    calibration_period=calibration_period,
                    result=result,
                )

            for calibration_period in transition:
                result = run_calibrated_certificate_batch(
                    ensemble=ensemble,
                    config=CalibratedCertificateConfig(
                        method="periodic_cert",
                        calibration_period=calibration_period,
                        bound_mode="componentwise",
                    ),
                    n_ticks=n_ticks,
                    tail=tail,
                    seed=50505,
                )
                _record(
                    rows,
                    geometry=geometry,
                    method="rotated transition",
                    calibration_period=calibration_period,
                    result=result,
                )

            # Conventional references from the exact same controlled geometry.
            for label, config in [
                (
                    "EF-TopK k=4",
                    RotatedRunConfig(method="ef_topk", step=0.01, topk=4),
                ),
                (
                    "full precision",
                    RotatedRunConfig(method="full", step=0.02),
                ),
            ]:
                result = run_rotated_batch(
                    ensemble=ensemble,
                    config=config,
                    n_ticks=n_ticks,
                    tail=tail,
                    seed=50505,
                )
                _record(rows, geometry=geometry, method=label, result=result)

    return pd.DataFrame(rows), pd.DataFrame(margin_rows)


def save_plots(results: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rotated = results[results["geometry"] == "rotated"].copy()
    transition = rotated[rotated["method"] == "rotated transition"].sort_values(
        "calibration_period"
    )

    selected_labels = {
        "q(t) baseline",
        "EF-TopK k=4",
        "full precision",
        "basis-aligned certificate oracle",
    }
    selected = rotated[rotated["method"].isin(selected_labels)].copy()
    selected = pd.concat(
        [
            selected,
            transition.assign(
                plot_label=transition["calibration_period"].map(
                    lambda x: f"native cert C={int(x)}"
                )
            ),
        ],
        ignore_index=True,
    )
    if "plot_label" not in selected:
        selected["plot_label"] = selected["method"]
    selected["plot_label"] = selected["plot_label"].fillna(selected["method"])
    selected["tail_plot"] = selected["tail_gap"].clip(lower=1e-6)

    plt.figure(figsize=(9, 6))
    plt.scatter(selected["payload_bits"], selected["tail_plot"], s=60)
    for _, row in selected.iterrows():
        plt.annotate(
            row["plot_label"],
            (row["payload_bits"], row["tail_plot"]),
            fontsize=8,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total payload bits: sparse events + dense calibration")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11B: calibrated certificate accuracy vs communication")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "error_vs_total_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        transition["calibration_period"],
        transition["tail_gap"],
        marker="o",
    )
    plt.yscale("log")
    plt.xlabel("Calibration period C [wall-clock ticks]")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 11B: native rotated calibration transition")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "calibration_period.png", dpi=180)
    plt.close()

    comp = transition[transition["calibration_period"].isin([25, 50, 75, 100])]
    if not comp.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(
            comp["calibration_period"],
            comp["calibration_bits"],
            marker="o",
            label="calibration bits",
        )
        plt.plot(
            comp["calibration_period"],
            comp["payload_bits"] - comp["calibration_bits"],
            marker="o",
            label="event bits",
        )
        plt.xlabel("Calibration period C")
        plt.ylabel("Payload bits")
        plt.title("Experiment 11B: calibration dominates certificate traffic")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "bit_composition.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    results, margins = run_campaign(quick=args.quick)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_DIR / "summary.csv", index=False)
    margins.to_csv(RESULT_DIR / "event_error_margins.csv", index=False)
    save_plots(results)

    cols = [
        "geometry",
        "method",
        "calibration_period",
        "refresh_gap",
        "tail_gap",
        "whole_gap",
        "payload_bits",
        "calibration_bits",
        "acceptance_fraction",
        "harmful_applied_fraction",
        "slow_accepted_share",
    ]
    print(results[cols].to_string(index=False))
    print("\nFirst-passage empirical margins:\n")
    print(margins.to_string(index=False))


if __name__ == "__main__":
    main()
