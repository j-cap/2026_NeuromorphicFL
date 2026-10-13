"""Publication-oriented P20 participation robustness figure.

Usage: python experiments/p20_plot_participation.py
Produces three independent panel PDFs/PNGs, two legend PDFs/PNGs, and a CSV audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ROOT = Path(__file__).resolve().parent / "results" / "p20_cifar10_partial_participation"
METHODS = ("dense", "ef_topk", "event")
LABELS = {"dense": "Dense FedAvg", "ef_topk": "EF-TopK", "event": "Event-FedAvg"}
COLORS = {"dense": "#65758b", "ef_topk": "#da8c35", "event": "#167f85"}
MARKERS = {"dense": "s", "ef_topk": "^", "event": "o"}
TRAFFIC = "unicast_hybrid_total_bits"


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for n_active, filename in ((10, "ten_seed_runs.csv"), (5, "stress_ten_seed_runs.csv")):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen results: {path}")
        df = pd.read_csv(path)
        required = {"method", "seed_index", "final_test_accuracy", TRAFFIC,
                    "uplink_packetized_bits", "checkpoint_downlink_bits",
                    "terminal_catchup_downlink_bits"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        df = df.copy()
        df["active_clients"] = n_active
        frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    expected = {(n, method, seed) for n in (5, 10)
                for method in METHODS for seed in range(10)}
    observed = set(zip(runs.active_clients, runs.method, runs.seed_index))
    if observed != expected or len(runs) != len(expected):
        raise ValueError("Expected exactly 60 unique runs (2 x 3 x 10)")
    for col in ("final_test_accuracy", TRAFFIC, "uplink_packetized_bits",
                "checkpoint_downlink_bits", "terminal_catchup_downlink_bits"):
        if not np.isfinite(pd.to_numeric(runs[col], errors="coerce")).all():
            raise ValueError(f"Nonfinite input: {col}")
    if (runs[TRAFFIC] <= 0).any():
        raise ValueError("Nonpositive traffic")
    return runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (active, method), frame in runs.groupby(["active_clients", "method"]):
        acc = frame["final_test_accuracy"].to_numpy(dtype=float) * 100
        bits = frame[TRAFFIC].to_numpy(dtype=float) / 1e9
        row = {"active_clients": int(active), "participation_pct": 2 * int(active),
               "method": method, "n_seeds": len(frame),
               "accuracy_mean_pct": acc.mean(), "accuracy_sd_pp": acc.std(ddof=1),
               "traffic_mean_gbit": bits.mean(), "traffic_sd_gbit": bits.std(ddof=1)}
        if method == "event":
            for col, dest in (("uplink_packetized_bits", "uplink_gbit"),
                              ("checkpoint_downlink_bits", "checkpoint_downlink_gbit"),
                              ("terminal_catchup_downlink_bits", "terminal_gbit")):
                row[dest] = frame[col].mean() / 1e9
            # Exact traffic minus displayed components is the remaining
            # synchronization/request/header/replay traffic (not an uplink).
            row["other_gbit"] = (row["traffic_mean_gbit"] - row["uplink_gbit"]
                                - row["checkpoint_downlink_gbit"] - row["terminal_gbit"])
            if row["other_gbit"] < -1e-7:
                raise ValueError("Traffic components exceed exact total")
            row["other_gbit"] = max(0.0, row["other_gbit"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["active_clients", "method"])


def draw(summary: pd.DataFrame, root: Path, dpi: int) -> None:
    """Export independent vector panels and legends for LaTeX subfigures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.65, "savefig.facecolor": "white",
    })
    stem = "p20_participation_"
    def export(fig, name):
        fig.savefig(root / f"{stem}{name}.pdf", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(root / f"{stem}{name}.png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)

    method_handles = []
    for metric, ylabel, filename in (
        ("accuracy_mean_pct", "Test accuracy (%)", "accuracy"),
        ("traffic_mean_gbit", "Exact traffic (Gbit)", "communication"),
    ):
        fig, ax = plt.subplots(figsize=(2.30, 1.85))
        fig.subplots_adjust(left=0.24, right=0.97, bottom=0.25, top=0.97)
        for method in METHODS:
            sub = summary[summary.method == method].sort_values("participation_pct")
            x = sub.participation_pct.to_numpy()
            uncertainty = ("accuracy_sd_pp" if metric == "accuracy_mean_pct"
                           else "traffic_sd_gbit")
            handle = ax.errorbar(
                x, sub[metric], yerr=sub[uncertainty],
                color=COLORS[method], marker=MARKERS[method],
                linewidth=1.25, markersize=4, capsize=2,
                elinewidth=0.9, markeredgecolor="white",
                markeredgewidth=0.4, label=LABELS[method])
            if metric == "accuracy_mean_pct":
                method_handles.append(handle)
        ax.set_xticks([10, 20])
        ax.set_xlim(7.5, 22.5)
        ax.set_xlabel("Participation (%)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dedede", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        export(fig, filename)

    event = summary[summary.method == "event"].sort_values("participation_pct")
    fig, ax = plt.subplots(figsize=(2.48, 1.85))
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.25, top=0.97)
    x = np.arange(2)
    bottom = np.zeros(2)
    components = [
        ("uplink_gbit", "Uplink", "#167f85"),
        ("checkpoint_downlink_gbit", "Checkpoints", "#da8c35"),
        ("other_gbit", "Other sync.", "#8c92a2"),
        ("terminal_gbit", "Final catch-up", "#b7c7d1"),
    ]
    component_handles = []
    for col, name, color in components:
        values = event[col].to_numpy(dtype=float)
        handle = ax.bar(x, values, bottom=bottom, width=0.57,
                        color=color, edgecolor="white", linewidth=0.35, label=name)
        component_handles.append(handle)
        bottom += values
    np.testing.assert_allclose(
        bottom, event.traffic_mean_gbit.to_numpy(), rtol=1e-10, atol=1e-7)
    ax.set_xticks(x, ["10", "20"])
    ax.set_xlabel("Participation (%)")
    ax.set_ylabel("Event traffic (Gbit)")
    ax.grid(axis="y", color="#dedede", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    export(fig, "composition")

    # These independent legends are placed and scaled by LaTeX, as with
    # the manuscript's communication-frontier and convergence figures.
    def legend_pdf(handles, labels, name, width, ncol):
        fig = plt.figure(figsize=(width, 0.28))
        fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
                   fontsize=7, columnspacing=1.45, handlelength=1.7)
        export(fig, name)
    legend_pdf(method_handles, [LABELS[m] for m in METHODS],
               "method_legend", 5.1, 3)
    legend_pdf(component_handles, [entry[1] for entry in components],
               "traffic_legend", 6.0, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    runs = load_runs(root)
    summary = summarize(runs)
    summary.to_csv(root / "p20_participation_figure_audit.csv", index=False,
                   float_format="%.10g")
    draw(summary, root, args.dpi)
    print(summary[["participation_pct", "method", "accuracy_mean_pct",
                   "accuracy_sd_pp", "traffic_mean_gbit"]].to_string(index=False))
    print(f"Saved three panel PDFs/PNGs, two legends and audit CSV in {root}")


if __name__ == "__main__":
    main()
