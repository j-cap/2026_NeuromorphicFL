from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neuromorphicfl.competitive_vector import (
    CompetitiveRunConfig,
    check_competitive_lif_ema_equivalence,
    run_competitive_batch,
)
from neuromorphicfl.vector_quadratic import make_diagonal_quadratic_ensemble, pareto_mask


RESULT_DIR = Path("experiments/results/10b_competitive_coordinates")


def _run_campaign(*, quick: bool) -> tuple[pd.DataFrame, object]:
    if quick:
        n_runs, n_ticks, tail = 12, 400, 100
        k_values = [1, 2, 4]
        sign_k = [1, 4]
        sign_jumps = [0.005, 0.01]
        ef_k = [1, 4]
        ef_steps = [0.01]
    else:
        n_runs, n_ticks, tail = 80, 2000, 500
        k_values = [1, 2, 4, 8, 20]
        sign_k = [1, 2, 4, 8, 20]
        sign_jumps = [0.003, 0.005, 0.010, 0.020]
        ef_k = [1, 2, 4, 8, 20]
        ef_steps = [0.005, 0.010, 0.020]

    ensemble = make_diagonal_quadratic_ensemble(n_runs=n_runs, seed=222)
    rows: list[dict[str, float | str]] = []

    def record(label: str, config: CompetitiveRunConfig, **params: float) -> None:
        result = run_competitive_batch(
            ensemble=ensemble,
            config=config,
            n_ticks=n_ticks,
            tail=tail,
            seed=30303,
        )
        row = {
            "label": label,
            "method": config.method,
            **{k: v for k, v in result.items() if k != "coordinate_events_mean"},
            "rho": np.nan,
            "delta0": np.nan,
            "q": np.nan,
            "eta": np.nan,
            "k": np.nan,
        }
        row.update(params)
        rows.append(row)

    # Experiment-10A reference and competition-budget sweep.
    reference = dict(rho=0.999, delta0=0.25, jump=0.02)
    record(
        "independent_reference",
        CompetitiveRunConfig(method="lif_independent", **reference),
        rho=0.999,
        delta0=0.25,
        q=0.02,
    )
    for k in k_values:
        record(
            f"competitive_K{k}",
            CompetitiveRunConfig(method="lif_competitive", topk=k, **reference),
            rho=0.999,
            delta0=0.25,
            q=0.02,
            k=float(k),
        )

    if not quick:
        # Local robustness around the reference rather than a prohibitively large grid.
        for delta0 in [0.15, 0.40]:
            for k in [2, 4]:
                record(
                    f"competitive_delta{delta0}_K{k}",
                    CompetitiveRunConfig(
                        method="lif_competitive",
                        rho=0.999,
                        delta0=delta0,
                        jump=0.02,
                        topk=k,
                    ),
                    rho=0.999,
                    delta0=delta0,
                    q=0.02,
                    k=float(k),
                )
        for k in [2, 4]:
            record(
                f"competitive_q0.04_K{k}",
                CompetitiveRunConfig(
                    method="lif_competitive",
                    rho=0.999,
                    delta0=0.25,
                    jump=0.04,
                    topk=k,
                ),
                rho=0.999,
                delta0=0.25,
                q=0.04,
                k=float(k),
            )
            record(
                f"competitive_IF_K{k}",
                CompetitiveRunConfig(
                    method="lif_competitive",
                    rho=1.0,
                    delta0=0.25,
                    jump=0.02,
                    topk=k,
                ),
                rho=1.0,
                delta0=0.25,
                q=0.02,
                k=float(k),
            )

    for k in sign_k:
        for q in sign_jumps:
            record(
                f"sign_topk_K{k}_q{q}",
                CompetitiveRunConfig(method="sign_topk", topk=k, jump=q),
                q=q,
                k=float(k),
            )

    for k in ef_k:
        for eta in ef_steps:
            record(
                f"ef_topk_K{k}_eta{eta}",
                CompetitiveRunConfig(method="ef_topk", topk=k, step=eta),
                eta=eta,
                k=float(k),
            )

    return pd.DataFrame(rows), ensemble


def _packet_diagnostics(ensemble, *, quick: bool) -> pd.DataFrame:
    n_ticks, tail = (400, 100) if quick else (2000, 500)
    rows = []
    for name, method, k in [
        ("Independent", "lif_independent", 20),
        ("Competitive K=1", "lif_competitive", 1),
        ("Competitive K=2", "lif_competitive", 2),
        ("Competitive K=4", "lif_competitive", 4),
    ]:
        result = run_competitive_batch(
            ensemble=ensemble,
            config=CompetitiveRunConfig(
                method=method,
                rho=0.999,
                delta0=0.25,
                jump=0.02,
                topk=k,
            ),
            n_ticks=n_ticks,
            tail=tail,
            seed=30303,
            record_packet_sizes=True,
        )
        rows.append(
            {
                "variant": name,
                "mean_events_per_nonempty_packet": result["mean_events_per_nonempty_packet"],
                "p95_events_per_nonempty_packet": result["p95_events_per_nonempty_packet"],
                "max_events_per_nonempty_packet": result["max_events_per_nonempty_packet"],
                "max_candidate_count": result["max_candidate_count"],
                "packetized_bits": result["packetized_bits"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke campaign")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results, ensemble = _run_campaign(quick=args.quick)
    results.to_csv(RESULT_DIR / "all_results.csv", index=False)

    pareto = results[
        pareto_mask(results["payload_bits"].to_numpy(), results["tail_gap"].to_numpy())
    ].sort_values("payload_bits")
    pareto.to_csv(RESULT_DIR / "tail_pareto.csv", index=False)

    reference = results[results["label"] == "independent_reference"].iloc[0]
    k_sweep = results[results["label"].str.startswith("competitive_K")].sort_values("k")
    selected = pd.concat([reference.to_frame().T, k_sweep], ignore_index=True)
    selected.to_csv(RESULT_DIR / "competition_k_sweep.csv", index=False)

    packet = _packet_diagnostics(ensemble, quick=args.quick)
    packet.to_csv(RESULT_DIR / "packet_diagnostics.csv", index=False)
    pd.DataFrame([check_competitive_lif_ema_equivalence()]).to_csv(
        RESULT_DIR / "competitive_ema_equivalence.csv", index=False
    )

    plt.figure(figsize=(9, 6))
    for method, subset in results.groupby("method"):
        plt.scatter(subset["payload_bits"], subset["tail_gap"], label=method, alpha=0.65)
    plt.plot(pareto["payload_bits"], pareto["tail_gap"], linewidth=2, label="global Pareto")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mean logical payload bits")
    plt.ylabel("Tail excess objective")
    plt.title("Experiment 10B: competitive coordinate selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "tail_gap_vs_bits.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_sweep["k"], k_sweep["payload_bits"], marker="o", label="Payload bits")
    plt.plot(k_sweep["k"], k_sweep["packetized_bits"], marker="o", label="Packetized bits")
    plt.axhline(reference["payload_bits"], linestyle="--", label="Independent payload")
    plt.axhline(reference["packetized_bits"], linestyle=":", label="Independent packetized")
    plt.xscale("log", base=2)
    plt.xlabel("Competition budget K")
    plt.ylabel("Communication bits")
    plt.title("Experiment 10B: competition and packet overhead")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "competition_bits.png", dpi=180)
    plt.close()

    print("Experiment 10B complete")
    print(selected[[
        "label", "k", "tail_gap", "whole_gap", "payload_bits",
        "packetized_bits", "suppression_fraction", "slow_client_event_share"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
