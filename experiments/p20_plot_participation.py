"""Publication-oriented P20 participation robustness figure.

Usage: python experiments/p20_plot_participation.py
Produces PDF, PNG and a CSV audit next to the frozen P20 data.
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
    """Render at final IEEE two-column width; typography is not scaled down later."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.65, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 2.75),
        gridspec_kw={"width_ratios": [1.04, 1.04, 1.1]},
    )
    # Reserve top margin for a common method legend, and bottom for traffic legend.
    fig.subplots_adjust(left=0.072, right=0.99, top=0.77, bottom=0.34,
                        wspace=0.43)
    handles = []
    for method in METHODS:
        sub = summary[summary.method == method].sort_values("participation_pct")
        x = sub.participation_pct.to_numpy()
        style = dict(color=COLORS[method], marker=MARKERS[method],
                     linewidth=1.25, markersize=4, capsize=2,
                     elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.4)
        handle = axes[0].errorbar(
            x, sub.accuracy_mean_pct, yerr=sub.accuracy_sd_pp,
            label=LABELS[method], **style
        )
        axes[1].errorbar(x, sub.traffic_mean_gbit,
                         yerr=sub.traffic_sd_gbit, **style)
        handles.append(handle)
    for ax in axes[:2]:
        ax.set_xticks([10, 20], ["10", "20"])
        ax.set_xlim(7.5, 22.5)
        ax.grid(axis="y", color="#dedede", linewidth=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].set_ylabel("Exact traffic (Gbit)")
    event = summary[summary.method == "event"].sort_values("participation_pct")
    x = np.arange(2)
    bottom = np.zeros(2)
    # The other-sync component is a residual: never label it as measured replay.
    components = [
        ("uplink_gbit", "Uplink", "#167f85"),
        ("checkpoint_downlink_gbit", "Checkpoints", "#da8c35"),
        ("other_gbit", "Other sync.", "#8c92a2"),
        ("terminal_gbit", "Final catch-up", "#b7c7d1"),
    ]
    for col, name, color in components:
        values = event[col].to_numpy(dtype=float)
        axes[2].bar(x, values, bottom=bottom, width=0.57, label=name,
                    color=color, edgecolor="white", linewidth=0.35)
        bottom += values
    # Assert plot decomposition is consistent with the audit total.
    np.testing.assert_allclose(bottom, event.traffic_mean_gbit.to_numpy(),
                               rtol=1e-10, atol=1e-7)
    axes[2].set_xticks(x, ["10", "20"])
    axes[2].set_ylabel("Event traffic (Gbit)")
    axes[2].grid(axis="y", color="#dedede", linewidth=0.55)
    axes[2].set_axisbelow(True)
    for ax, heading in zip(axes, (
        "(a) Test accuracy", "(b) Total communication",
        "(c) Traffic composition",
    )):
        ax.set_title(heading, loc="left", fontweight="bold", pad=7)
        ax.set_xlabel("Participation (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles, [LABELS[m] for m in METHODS], loc="upper center",
               bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False,
               columnspacing=2.5, handlelength=2)
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.33),
                   ncol=2, frameon=False, handlelength=1.3,
                   columnspacing=0.65, labelspacing=0.5)
    out = root / "p20_participation_robustness"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


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
    print(f"Saved PDF, PNG and audit CSV in {root}")


if __name__ == "__main__":
    main()
