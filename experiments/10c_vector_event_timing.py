from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.vector_quadratic import make_diagonal_quadratic_ensemble, pareto_mask
from neuromorphicfl.vector_timing import (
    VectorTimingConfig,
    run_vector_timing_batch,
    timing_calibration,
)


RESULT_DIR = Path("experiments/results/10c_vector_event_timing")


def _record(
    rows: list[dict[str, object]],
    *,
    label: str,
    ensemble,
    config: VectorTimingConfig,
    n_ticks: int,
    tail: int,
    pools=None,
    medians=None,
    seed: int = 40404,
) -> None:
    result = run_vector_timing_batch(
        ensemble=ensemble,
        config=config,
        n_ticks=n_ticks,
        tail=tail,
        seed=seed,
        shuffled_pools=pools,
        relative_medians=medians,
    )
    rows.append(
        {
            "label": label,
            "method": config.method,
            "q": config.jump,
            "tau_c": config.tau_c,
            "timing_exponent": config.timing_exponent,
            "schedule_scale": config.schedule_scale,
            "schedule_exponent": config.schedule_exponent,
            "relative_tau_c": config.relative_tau_c,
            **{k: v for k, v in result.items() if k not in ("coordinate_events_mean", "event_log")},
        }
    )


def _diagnostics(event_log: pd.DataFrame, medians: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "timing": "tau_local",
                "corr_abs_error": event_log["tau_local"].corr(event_log["abs_error"], method="spearman"),
                "corr_harmful": event_log["tau_local"].corr(event_log["harmful"].astype(float), method="spearman"),
                "corr_curvature": event_log["tau_local"].corr(event_log["curvature"], method="spearman"),
            },
            {
                "timing": "tau_wall",
                "corr_abs_error": event_log["tau_wall"].corr(event_log["abs_error"], method="spearman"),
                "corr_harmful": event_log["tau_wall"].corr(event_log["harmful"].astype(float), method="spearman"),
                "corr_curvature": event_log["tau_wall"].corr(event_log["curvature"], method="spearman"),
            },
        ]
    )

    rows = []
    for (client, coordinate), subset in event_log.groupby(["client", "coordinate"]):
        if len(subset) < 50 or subset["tau_local"].nunique() < 2:
            continue
        rows.append(
            {
                "client": client,
                "coordinate": coordinate,
                "events": len(subset),
                "corr_tau_local_abs_error": subset["tau_local"].corr(subset["abs_error"], method="spearman"),
                "corr_tau_local_harmful": subset["tau_local"].corr(subset["harmful"].astype(float), method="spearman"),
            }
        )
    conditional = pd.DataFrame(rows)

    ratios = []
    for row in event_log.itertuples(index=False):
        ratios.append(row.tau_local / medians[int(row.client), int(row.coordinate)])
    event_log = event_log.copy()
    event_log["relative_tau"] = ratios
    event_log["relative_tau_bin"] = pd.cut(
        event_log["relative_tau"],
        bins=[0, 0.5, 1.0, 2.0, 4.0, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2-4", ">4"],
        include_lowest=True,
    )
    binned = (
        event_log.groupby("relative_tau_bin", observed=False)
        .agg(
            events=("relative_tau", "size"),
            mean_relative_tau=("relative_tau", "mean"),
            mean_abs_error=("abs_error", "mean"),
            harmful_fraction=("harmful", "mean"),
        )
        .reset_index()
    )
    return overall, conditional, binned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.quick:
        cal_runs, cal_ticks, cal_tail = 8, 300, 75
        n_runs, n_ticks, tail = 12, 400, 100
        fixed_q = [0.015, 0.02]
        timing_q = [0.015, 0.02]
        timing_tau = [20, 50]
        timing_alpha = [0.25]
        wall_tau = [100, 250]
        schedule_q = [0.02, 0.04]
        schedules = [(500.0, 0.25)]
        coordinate_q = [0.02, 0.03]
    else:
        cal_runs, cal_ticks, cal_tail = 30, 1600, 400
        n_runs, n_ticks, tail = 80, 2000, 500
        fixed_q = [0.01, 0.015, 0.02, 0.03, 0.04]
        timing_q = [0.015, 0.02, 0.03, 0.04]
        timing_tau = [20, 50]
        timing_alpha = [0.25, 0.50]
        wall_tau = [100, 250, 500]
        schedule_q = [0.015, 0.02, 0.03, 0.04]
        schedules = [(500.0, 0.25), (2000.0, 0.50), (5000.0, 1.00)]
        coordinate_q = [0.015, 0.02, 0.025, 0.03, 0.04]

    # Independent calibration set: provides timing distributions only.
    calibration_ensemble = make_diagonal_quadratic_ensemble(n_runs=cal_runs, seed=222)
    calibration = run_vector_timing_batch(
        ensemble=calibration_ensemble,
        config=VectorTimingConfig(method="fixed", jump=0.015),
        n_ticks=cal_ticks,
        tail=cal_tail,
        seed=30303,
        record_events=True,
    )
    event_log = calibration["event_log"]
    pools, medians = timing_calibration(
        event_log,
        calibration_ensemble.n_clients,
        calibration_ensemble.dimension,
    )
    overall, conditional, binned = _diagnostics(event_log, medians)
    overall.to_csv(RESULT_DIR / "timing_correlations.csv", index=False)
    conditional.to_csv(RESULT_DIR / "conditional_timing_correlations.csv", index=False)
    binned.to_csv(RESULT_DIR / "relative_timing_quality.csv", index=False)

    # Held-out evaluation ensemble.
    ensemble = make_diagonal_quadratic_ensemble(n_runs=n_runs, seed=333)
    rows: list[dict[str, object]] = []

    for q in fixed_q:
        _record(
            rows,
            label=f"fixed_q{q}",
            ensemble=ensemble,
            config=VectorTimingConfig(method="fixed", jump=q),
            n_ticks=n_ticks,
            tail=tail,
        )

    for q in timing_q:
        for tau_c in timing_tau:
            for alpha in timing_alpha:
                _record(
                    rows,
                    label=f"local_q{q}_tau{tau_c}_a{alpha}",
                    ensemble=ensemble,
                    config=VectorTimingConfig(
                        method="local_timing",
                        jump=q,
                        tau_c=tau_c,
                        timing_exponent=alpha,
                    ),
                    n_ticks=n_ticks,
                    tail=tail,
                )
        _record(
            rows,
            label=f"shuffle_q{q}",
            ensemble=ensemble,
            config=VectorTimingConfig(
                method="shuffled_local", jump=q, tau_c=50, timing_exponent=0.25
            ),
            n_ticks=n_ticks,
            tail=tail,
            pools=pools,
        )
        for tau_c in wall_tau:
            _record(
                rows,
                label=f"wall_q{q}_tau{tau_c}",
                ensemble=ensemble,
                config=VectorTimingConfig(
                    method="wall_timing", jump=q, tau_c=tau_c, timing_exponent=0.25
                ),
                n_ticks=n_ticks,
                tail=tail,
            )

    for q in [0.015, 0.02, 0.03] if not args.quick else timing_q:
        for relative_tau_c in [1.0, 2.0]:
            _record(
                rows,
                label=f"relative_q{q}_tc{relative_tau_c}",
                ensemble=ensemble,
                config=VectorTimingConfig(
                    method="relative_local",
                    jump=q,
                    relative_tau_c=relative_tau_c,
                    timing_exponent=0.25,
                ),
                n_ticks=n_ticks,
                tail=tail,
                medians=medians,
            )

    for q in schedule_q:
        for scale, exponent in schedules:
            _record(
                rows,
                label=f"schedule_q{q}_S{scale}_a{exponent}",
                ensemble=ensemble,
                config=VectorTimingConfig(
                    method="global_schedule",
                    jump=q,
                    schedule_scale=scale,
                    schedule_exponent=exponent,
                ),
                n_ticks=n_ticks,
                tail=tail,
            )

    for q in coordinate_q:
        _record(
            rows,
            label=f"coordinate_jump_q{q}",
            ensemble=ensemble,
            config=VectorTimingConfig(method="coordinate_jump", jump=q),
            n_ticks=n_ticks,
            tail=tail,
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "all_results.csv", index=False)
    pareto = results[
        pareto_mask(results["payload_bits"].to_numpy(), results["tail_gap"].to_numpy())
    ].sort_values("payload_bits")
    pareto.to_csv(RESULT_DIR / "tail_pareto.csv", index=False)

    family_rows = []
    for method, subset in results.groupby("method"):
        best = subset.loc[subset["tail_gap"].idxmin()]
        family_rows.append(
            {
                "method": method,
                "label": best.label,
                "tail_gap": best.tail_gap,
                "tail_rmse_w": best.tail_rmse_w,
                "whole_gap": best.whole_gap,
                "payload_bits": best.payload_bits,
                "packetized_bits": best.packetized_bits,
                "events": best.events,
                "harmful_event_fraction": best.harmful_event_fraction,
            }
        )
    pd.DataFrame(family_rows).to_csv(RESULT_DIR / "family_best.csv", index=False)

    plt.figure(figsize=(9, 6))
    for method, subset in results.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["tail_gap"], label=method, alpha=0.65)
    plt.plot(pareto["payload_bits"], pareto["tail_gap"], linewidth=2, label="global Pareto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 10C: coordinate event timing and jump resolution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "tail_gap_vs_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(binned["relative_tau_bin"].astype(str), binned["harmful_fraction"], marker="o")
    plt.xticks(rotation=30)
    plt.xlabel("Inter-event interval / client-coordinate median")
    plt.ylabel("Harmful-event fraction")
    plt.title("Experiment 10C: timing is informative, but decoder utility is separate")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "relative_timing_quality.png", dpi=180)
    plt.close()

    print("Experiment 10C complete")
    print(pd.DataFrame(family_rows).to_string(index=False))


if __name__ == "__main__":
    main()
