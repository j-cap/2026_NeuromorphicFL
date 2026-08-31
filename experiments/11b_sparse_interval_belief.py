from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuromorphicfl.descent_aware import DescentAwareConfig, run_descent_aware_batch
from neuromorphicfl.sparse_interval_certificate import (
    SparseIntervalConfig,
    build_sparse_interval_calibration,
    collect_first_passage_calibration,
    run_sparse_interval_certificate,
)
from neuromorphicfl.vector_quadratic import make_diagonal_quadratic_ensemble


def run_campaign(quick: bool = False) -> pd.DataFrame:
    n_cal = 20 if quick else 80
    n_eval = 20 if quick else 80
    ticks = 700 if quick else 2000
    tail = 175 if quick else 500

    calibration_ensemble = make_diagonal_quadratic_ensemble(
        n_runs=n_cal, seed=11001
    )
    calibration_log = collect_first_passage_calibration(
        ensemble=calibration_ensemble,
        n_ticks=700 if quick else 1800,
        seed=12001,
    )
    evaluation_ensemble = make_diagonal_quadratic_ensemble(
        n_runs=n_eval, seed=21001
    )

    rows: list[dict[str, object]] = []
    schedule = run_descent_aware_batch(
        ensemble=evaluation_ensemble,
        config=DescentAwareConfig(method="global_schedule"),
        n_ticks=ticks,
        tail=tail,
        seed=22001,
    )
    rows.append({"method": "global_schedule", "confidence": None, **{k: v for k, v in schedule.items() if k != "estimator_log"}})
    oracle = run_descent_aware_batch(
        ensemble=evaluation_ensemble,
        config=DescentAwareConfig(method="global_oracle", eta=1.0),
        n_ticks=ticks,
        tail=tail,
        seed=22001,
    )
    rows.append({"method": "global_oracle", "confidence": None, **{k: v for k, v in oracle.items() if k != "estimator_log"}})

    confidences = [0.8, 0.9] if quick else [0.6, 0.8, 0.9, 0.95, 0.975, 0.99]
    for confidence in confidences:
        calibration = build_sparse_interval_calibration(
            ensemble=calibration_ensemble,
            event_log=calibration_log,
            confidence=confidence,
        )
        result = run_sparse_interval_certificate(
            ensemble=evaluation_ensemble,
            calibration=calibration,
            config=SparseIntervalConfig(confidence=confidence),
            n_ticks=ticks,
            tail=tail,
            seed=22001,
            record_coverage=True,
        )
        rows.append({"method": "timing_only_interval", **result})

    return pd.DataFrame(rows)


def run_robustness(quick: bool = False) -> pd.DataFrame:
    if quick:
        return pd.DataFrame()

    calibration_ensemble = make_diagonal_quadratic_ensemble(
        n_runs=80, seed=11001
    )
    calibration_log = collect_first_passage_calibration(
        ensemble=calibration_ensemble, n_ticks=1800, seed=12001
    )
    rows: list[dict[str, object]] = []

    for confidence in [0.6, 0.8, 0.9]:
        calibration = build_sparse_interval_calibration(
            ensemble=calibration_ensemble,
            event_log=calibration_log,
            confidence=confidence,
        )
        for label, n_ticks, tail, ensemble_seed, sim_seed in [
            ("long_horizon", 4000, 800, 21001, 22001),
            ("independent_eval", 2000, 500, 31001, 32001),
        ]:
            ensemble = make_diagonal_quadratic_ensemble(
                n_runs=80, seed=ensemble_seed
            )
            result = run_sparse_interval_certificate(
                ensemble=ensemble,
                calibration=calibration,
                config=SparseIntervalConfig(confidence=confidence),
                n_ticks=n_ticks,
                tail=tail,
                seed=sim_seed,
                record_coverage=True,
            )
            rows.append({"case": label, "ensemble_seed": ensemble_seed, **result})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="experiments/results/11b_sparse_interval")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    primary = run_campaign(quick=args.quick)
    primary.to_csv(out / "primary.csv", index=False)
    print("\nPrimary held-out sparse interval audit")
    print(primary.to_string(index=False))

    robustness = run_robustness(quick=args.quick)
    if not robustness.empty:
        robustness.to_csv(out / "robustness.csv", index=False)
        print("\nHorizon / independent-fleet robustness")
        print(robustness.to_string(index=False))


if __name__ == "__main__":
    main()
