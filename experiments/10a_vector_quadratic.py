from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.vector_quadratic import (
    VectorRunConfig,
    check_fullreset_vector_ema_equivalence,
    make_diagonal_quadratic_ensemble,
    pareto_mask,
    run_vector_batch,
)


RESULT_DIR = Path("experiments/results/10a_vector_quadratic")


def _run_campaign(*, quick: bool) -> tuple[pd.DataFrame, object]:
    if quick:
        n_runs, n_ticks, tail = 12, 400, 100
        full_steps = [0.01, 0.02]
        sign_jumps = [0.005, 0.01]
        rhos = [0.999, 1.0]
        deltas = [0.15, 0.4]
        lif_jumps = [0.02, 0.04]
        topks = [1, 4]
        ef_steps = [0.01]
    else:
        n_runs, n_ticks, tail = 80, 2000, 500
        full_steps = [0.008, 0.012, 0.016, 0.020, 0.025]
        sign_jumps = [0.003, 0.005, 0.008, 0.012, 0.020]
        rhos = [0.995, 0.999, 1.0]
        deltas = [0.10, 0.15, 0.25, 0.40]
        lif_jumps = [0.02, 0.04]
        topks = [1, 2, 4, 8, 20]
        ef_steps = [0.005, 0.010, 0.020]

    ensemble = make_diagonal_quadratic_ensemble(
        n_runs=n_runs,
        seed=222,
    )
    rows: list[dict[str, float | str]] = []

    def record(method: str, config: VectorRunConfig, **parameters: float) -> None:
        result = run_vector_batch(
            ensemble=ensemble,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=30303,
        )
        row = {
            "method": method,
            **{k: v for k, v in result.items() if k != "coordinate_events_mean"},
            "rho": np.nan,
            "delta0": np.nan,
            "q": np.nan,
            "eta": np.nan,
            "k": np.nan,
        }
        row.update(parameters)
        rows.append(row)

    for eta in full_steps:
        record("full", VectorRunConfig(method="full", step=eta), eta=eta)

    for q in sign_jumps:
        record("sign", VectorRunConfig(method="sign", jump=q), q=q)

    for method in ("lif_global", "lif_norm"):
        for rho in rhos:
            for delta0 in deltas:
                for q in lif_jumps:
                    record(
                        method,
                        VectorRunConfig(
                            method=method,
                            rho=rho,
                            delta0=delta0,
                            jump=q,
                        ),
                        rho=rho,
                        delta0=delta0,
                        q=q,
                    )

    for k in topks:
        for eta in ef_steps:
            record(
                "ef_topk",
                VectorRunConfig(method="ef_topk", topk=k, step=eta),
                k=float(k),
                eta=eta,
            )

    return pd.DataFrame(rows), ensemble


def _selected_diagnostics(ensemble) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {
        "Global threshold": VectorRunConfig(
            method="lif_global", rho=1.0, delta0=0.15, jump=0.02
        ),
        "Normalized threshold": VectorRunConfig(
            method="lif_norm", rho=0.999, delta0=0.25, jump=0.02
        ),
    }
    results = {}
    for name, config in selected.items():
        results[name] = run_vector_batch(
            ensemble=ensemble,
            config=config,
            n_ticks=2000,
            tail=500,
            seed=30303,
        )

    base_h = np.geomspace(0.2, 5.0, ensemble.dimension)
    coord = pd.DataFrame(
        {
            "coordinate": np.arange(ensemble.dimension),
            "base_curvature": base_h,
            "global_threshold_events": results["Global threshold"]["coordinate_events_mean"],
            "normalized_threshold_events": results["Normalized threshold"]["coordinate_events_mean"],
        }
    )
    summary_rows = []
    for name, result in results.items():
        events = np.asarray(result["coordinate_events_mean"])
        summary_rows.append(
            {
                "variant": name,
                "tail_gap": result["tail_gap"],
                "whole_gap": result["whole_gap"],
                "payload_bits": result["payload_bits"],
                "packetized_bits": result["packetized_bits"],
                "events": result["events"],
                "coverage": result["coverage"],
                "slow_client_event_share": result["slow_client_event_share"],
                "corr_curvature_event_count": float(np.corrcoef(base_h, events)[0, 1]),
                "event_count_cv": float(np.std(events) / np.mean(events)),
            }
        )
    return coord, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results, ensemble = _run_campaign(quick=args.quick)
    results.to_csv(RESULT_DIR / "all_results.csv", index=False)

    tail_pareto = results[
        pareto_mask(results["payload_bits"].to_numpy(), results["tail_gap"].to_numpy())
    ].sort_values("payload_bits")
    transient_pareto = results[
        pareto_mask(results["payload_bits"].to_numpy(), results["whole_gap"].to_numpy())
    ].sort_values("payload_bits")
    tail_pareto.to_csv(RESULT_DIR / "tail_pareto.csv", index=False)
    transient_pareto.to_csv(RESULT_DIR / "transient_pareto.csv", index=False)

    family_rows = []
    for method, subset in results.groupby("method"):
        best_tail = subset.loc[subset["tail_gap"].idxmin()]
        best_whole = subset.loc[subset["whole_gap"].idxmin()]
        family_rows.append(
            {
                "method": method,
                "best_tail_gap": best_tail.tail_gap,
                "tail_rmse_at_best_tail": best_tail.tail_rmse_w,
                "payload_bits_best_tail": best_tail.payload_bits,
                "packetized_bits_best_tail": best_tail.packetized_bits,
                "events_best_tail": best_tail.events,
                "coverage_best_tail": best_tail.coverage,
                "best_tail_eta": best_tail.eta,
                "best_tail_q": best_tail.q,
                "best_tail_delta": best_tail.delta0,
                "best_tail_rho": best_tail.rho,
                "best_tail_k": best_tail.k,
                "best_whole_gap": best_whole.whole_gap,
                "payload_bits_best_whole": best_whole.payload_bits,
            }
        )
    pd.DataFrame(family_rows).to_csv(RESULT_DIR / "family_best.csv", index=False)

    if not args.quick:
        coord, firing = _selected_diagnostics(ensemble)
        coord.to_csv(RESULT_DIR / "coordinate_firing.csv", index=False)
        firing.to_csv(RESULT_DIR / "firing_balance.csv", index=False)
        pd.DataFrame([check_fullreset_vector_ema_equivalence()]).to_csv(
            RESULT_DIR / "ema_equivalence.csv", index=False
        )

        plt.figure(figsize=(9, 5))
        plt.plot(
            coord["base_curvature"],
            coord["global_threshold_events"],
            marker="o",
            label="Global threshold",
        )
        plt.plot(
            coord["base_curvature"],
            coord["normalized_threshold_events"],
            marker="o",
            label="Normalized threshold",
        )
        plt.xscale("log")
        plt.xlabel("Base coordinate curvature")
        plt.ylabel("Mean coordinate events")
        plt.title("Experiment 10A: threshold normalization and coordinate firing")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "coordinate_firing.png", dpi=180)
        plt.close()

    plt.figure(figsize=(9, 6))
    for method, subset in results.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["tail_gap"], label=method, alpha=0.65)
    plt.plot(tail_pareto["payload_bits"], tail_pareto["tail_gap"], linewidth=2, label="global Pareto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 10A: vector optimization error vs communication")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "tail_gap_vs_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    for method, subset in results.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["whole_gap"], label=method, alpha=0.65)
    plt.plot(
        transient_pareto["payload_bits"],
        transient_pareto["whole_gap"],
        linewidth=2,
        label="global Pareto",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Whole-run mean excess objective")
    plt.title("Experiment 10A: transient cost vs communication")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "whole_gap_vs_bits.png", dpi=180)
    plt.close()

    print("Experiment 10A complete")
    print(tail_pareto[["method", "tail_gap", "payload_bits", "packetized_bits"]].to_string(index=False))


if __name__ == "__main__":
    main()
