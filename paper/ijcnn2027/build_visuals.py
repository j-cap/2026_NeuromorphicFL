"""Generate and validate the frozen IJCNN visual argument.

The two vector figures and compact main-results table are derived only from
the paper evidence CSVs and the authoritative T4 full-gradient audit.  Run
with ``--check`` to detect source, selection-rule, or rendered-artifact drift.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
FMNIST = PAPER / "evidence" / "fmnist_master_results.csv"
CIFAR10 = PAPER / "evidence" / "cifar10_master_results.csv"
ALIGNMENT = PAPER / "evidence" / "t4_alignment_audit.csv"

FIGURES = PAPER / "figures"
METHOD_FIGURE = FIGURES / "event_fedavg_method.pdf"
FRONTIER_FIGURE = FIGURES / "communication_frontier.pdf"
MAIN_TABLE = PAPER / "generated" / "main_results_table.tex"

PDF_METADATA = {
    "Title": None,
    "Author": None,
    "Subject": None,
    "Keywords": None,
    "Creator": "NeuromorphicFL build_visuals.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

METHODS = {
    "event": {"label": "Event-FedAvg", "marker": "*", "color": "#b2182b"},
    "strom": {"label": "Strom", "marker": "s", "color": "#2166ac"},
    "ef_topk": {"label": "EF-TopK", "marker": "^", "color": "#1b7837"},
    "sign_ef": {"label": "Sign-EF", "marker": "D", "color": "#762a83"},
    "dense": {"label": "Dense FedAvg", "marker": "o", "color": "#4d4d4d"},
}

PANELS = [
    ("fmnist_mlp", "Fashion-MNIST\nMLP", "mlp", (78.0, 84.0)),
    ("fmnist_cnn", "Fashion-MNIST\nCNN", "cnn", (74.0, 84.0)),
    ("cifar10_cnn", "CIFAR-10\ncompact CNN", "cifar_cnn", (32.0, 50.5)),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty visual evidence source: {path}")
    return rows


def benchmark_rows() -> dict[str, list[dict[str, str]]]:
    rows = read_csv(FMNIST) + read_csv(CIFAR10)
    required = {
        "comparison",
        "architecture",
        "method",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "unicast_total_Mbit_mean",
        "unicast_total_Mbit_std",
    }
    if required.difference(rows[0]):
        raise ValueError("paper evidence lacks columns required by P5 visuals")

    grouped: dict[str, list[dict[str, str]]] = {}
    for key, _title, architecture, _ylim in PANELS:
        selected = [row for row in rows if row["architecture"] == architecture]
        if len(selected) != 8:
            raise ValueError(f"{key} does not contain the frozen eight visual rows")
        grouped[key] = selected
    return grouped


def value(row: dict[str, str], field: str, scale: float = 1.0) -> float:
    return scale * float(row[field])


def unique_points(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop the duplicated Event-FedAvg row shared by both selection views."""

    points: list[dict[str, str]] = []
    seen: set[tuple[str, float, float]] = set()
    for row in rows:
        key = (
            row["method"],
            value(row, "unicast_total_Mbit_mean"),
            value(row, "final_test_accuracy_mean"),
        )
        if key not in seen:
            points.append(row)
            seen.add(key)
    return points


def nondominated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return points not beaten by lower/equal traffic and higher/equal accuracy."""

    points = unique_points(rows)
    frontier = []
    for candidate in points:
        cx = value(candidate, "unicast_total_Mbit_mean")
        cy = value(candidate, "final_test_accuracy_mean")
        dominated = any(
            value(other, "unicast_total_Mbit_mean") <= cx
            and value(other, "final_test_accuracy_mean") >= cy
            and (
                value(other, "unicast_total_Mbit_mean") < cx
                or value(other, "final_test_accuracy_mean") > cy
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: value(row, "unicast_total_Mbit_mean"))


def validate_visual_contract(grouped: dict[str, list[dict[str, str]]]) -> None:
    markers = [style["marker"] for style in METHODS.values()]
    if len(set(markers)) != len(markers):
        raise ValueError("method markers must remain distinct for grayscale output")

    for key, rows in grouped.items():
        expected = {
            ("quality-selected", method)
            for method in ("event", "strom", "ef_topk", "sign_ef", "dense")
        } | {
            ("traffic-matched", method)
            for method in ("event", "strom", "ef_topk")
        }
        observed = {(row["comparison"], row["method"]) for row in rows}
        if observed != expected:
            raise ValueError(f"{key} selection rows drifted from the frozen design")
        if "event" not in {row["method"] for row in nondominated(rows)}:
            raise ValueError(f"Event-FedAvg is no longer nondominated in {key}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )


def save_pdf(fig: plt.Figure) -> bytes:
    stream = io.BytesIO()
    fig.savefig(
        stream,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        metadata=PDF_METADATA,
    )
    plt.close(fig)
    return stream.getvalue()


def method_figure() -> bytes:
    fig, ax = plt.subplots(figsize=(7.08, 1.68))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [0.095, 0.295, 0.495, 0.695, 0.895]
    widths = [0.16, 0.18, 0.18, 0.16, 0.16]
    titles = [
        "1  Local SGD",
        "2  Leaky state",
        "3  Threshold + reset",
        "4  Server update",
        "5  Exact sync",
    ]
    formulas = [
        "$\\delta_i^r=-\\eta_r\\sum_e g_{i,e}^r$",
        "$\\widetilde z_i^r=\\rho_r z_i^r$\n$+p_i\\delta_i^r/\\eta_r$",
        "$|\\widetilde z_{ij}^r|\\geq\\vartheta$\n$\\Rightarrow c_{ij}^r=\\pm1,\\; z_{ij}^{r+1}=0$",
        "$w^{r+1}=w^r$\n$+q_r\\sum_i c_i^r$",
        "ordered sparse replay\nor dense checkpoint",
    ]
    fills = ["#f7f7f7", "#e6e6e6", "#d9d9d9", "#f2f2f2", "#ffffff"]

    for x, width, title, formula, fill in zip(centers, widths, titles, formulas, fills):
        box = FancyBboxPatch(
            (x - width / 2, 0.34),
            width,
            0.40,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#202020",
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, 0.665, title, ha="center", va="center", fontsize=6.5, weight="bold")
        ax.text(x, 0.495, formula, ha="center", va="center", fontsize=6.4, linespacing=1.35)

    for index in range(4):
        start = centers[index] + widths[index] / 2 + 0.006
        end = centers[index + 1] - widths[index + 1] / 2 - 0.006
        ax.add_patch(
            FancyArrowPatch(
                (start, 0.54),
                (end, 0.54),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#202020",
            )
        )

    # Two feedback paths make the persistent encoder and exact next-round model explicit.
    ax.add_patch(
        FancyArrowPatch(
            (centers[2], 0.33),
            (centers[1], 0.33),
            connectionstyle="arc3,rad=-0.35",
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.7,
            color="#4d4d4d",
        )
    )
    ax.text(
        0.395, 0.19, "$z_i^{r+1}$ persists", ha="center", fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax.add_patch(
        FancyArrowPatch(
            (centers[4], 0.33),
            (centers[0], 0.33),
            connectionstyle="arc3,rad=-0.18",
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.7,
            color="#4d4d4d",
        )
    )
    ax.text(
        0.62, 0.045, "exact server model starts the next round", ha="center", fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )

    ax.text(0.30, 0.88, "client $i$", ha="center", weight="bold", fontsize=6.4)
    ax.plot([0.01, 0.585], [0.83, 0.83], color="#737373", linewidth=0.6)
    ax.text(0.695, 0.88, "server", ha="center", weight="bold", fontsize=6.4)
    ax.plot([0.605, 0.785], [0.83, 0.83], color="#737373", linewidth=0.6)
    ax.text(0.895, 0.88, "clients", ha="center", weight="bold", fontsize=6.4)
    ax.plot([0.805, 0.985], [0.83, 0.83], color="#737373", linewidth=0.6)
    ax.text(
        0.50,
        0.96,
        "Trigger resolution $\\vartheta$ and model-update resolution $q_r$ are independent",
        ha="center",
        va="center",
        fontsize=6.5,
    )
    return save_pdf(fig)


def frontier_figure(grouped: dict[str, list[dict[str, str]]]) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.58))

    for ax, (key, title, _architecture, ylim) in zip(axes, PANELS):
        rows = grouped[key]
        frontier = nondominated(rows)
        ax.plot(
            [value(row, "unicast_total_Mbit_mean") for row in frontier],
            [value(row, "final_test_accuracy_mean", 100.0) for row in frontier],
            color="#737373",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

        for row in rows:
            # Event is identical in both selection views and is drawn once.
            if row["method"] == "event" and row["comparison"] == "traffic-matched":
                continue
            style = METHODS[row["method"]]
            quality = row["comparison"] == "quality-selected"
            face = style["color"] if quality else "white"
            size = 7.8 if row["method"] == "event" else 5.5
            ax.errorbar(
                value(row, "unicast_total_Mbit_mean"),
                value(row, "final_test_accuracy_mean", 100.0),
                xerr=value(row, "unicast_total_Mbit_std"),
                yerr=value(row, "final_test_accuracy_std", 100.0),
                fmt=style["marker"],
                markersize=size,
                markerfacecolor=face,
                markeredgecolor="#111111",
                markeredgewidth=0.65,
                color="#4d4d4d",
                ecolor="#8c8c8c",
                elinewidth=0.65,
                capsize=1.6,
                zorder=4 if row["method"] == "event" else 3,
            )

        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_title(title, weight="bold", pad=3)
        ax.set_xlabel("Total bidirectional traffic [Mbit]")
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.45)
        ax.grid(True, which="minor", axis="x", color="#eeeeee", linewidth=0.35)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test accuracy [%]")

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="none",
            markersize=6.0 if method != "event" else 7.5,
            markerfacecolor=style["color"],
            markeredgecolor="#111111",
            markeredgewidth=0.65,
            label=style["label"],
        )
        for method, style in METHODS.items()
    ]
    selection_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.0,
            markerfacecolor="#737373", markeredgecolor="#111111",
            label="quality-selected",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.0,
            markerfacecolor="white", markeredgecolor="#111111",
            label="nearest-traffic",
        ),
    ]
    fig.legend(
        handles=method_handles + selection_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        handletextpad=0.35,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.066, right=0.995, top=0.88, bottom=0.28, wspace=0.28)
    return save_pdf(fig)


def selected_rows(
    rows: list[dict[str, str]], comparison: str
) -> tuple[dict[str, str], dict[str, str]]:
    event = next(
        row
        for row in rows
        if row["comparison"] == "quality-selected" and row["method"] == "event"
    )
    controls = [
        row
        for row in rows
        if row["comparison"] == comparison and row["method"] != "event"
    ]
    if comparison == "quality-selected":
        selected = max(controls, key=lambda row: value(row, "final_test_accuracy_mean"))
    elif comparison == "traffic-matched":
        event_traffic = value(event, "unicast_total_Mbit_mean")
        selected = min(
            controls,
            key=lambda row: abs(math.log(value(row, "unicast_total_Mbit_mean") / event_traffic)),
        )
    else:
        raise ValueError(comparison)
    return event, selected


def pm(row: dict[str, str], mean: str, std: str, scale: float, digits: int) -> str:
    return (
        f"{scale * value(row, mean):.{digits}f}"
        f"$\\pm${scale * value(row, std):.{digits}f}"
    )


def alignment_metrics() -> tuple[float, float, float, float]:
    rows = read_csv(ALIGNMENT)
    by_metric = {row["metric"]: row for row in rows}
    required = {"weighted_alignment_ratio", "objective_decrease_fraction"}
    if set(by_metric) != required or len(rows) != len(required):
        raise ValueError("authoritative T4 baseline metrics are missing")
    for row in rows:
        counts = (
            int(row["n_partitions"]),
            int(row["snapshots_per_partition"]),
            int(row["total_snapshots"]),
        )
        if counts != (3, 31, 93):
            raise ValueError("T4 alignment-audit sample counts drifted")
        if not row["source_run"].endswith("/33807650381"):
            raise ValueError("T4 alignment-audit source run drifted")
    alignment = by_metric["weighted_alignment_ratio"]
    descent = by_metric["objective_decrease_fraction"]
    return (
        float(alignment["mean"]),
        float(alignment["std"]),
        float(descent["mean"]),
        float(descent["std"]),
    )


def main_table(grouped: dict[str, list[dict[str, str]]]) -> str:
    dataset_labels = {
        "fmnist_mlp": "Fashion-MNIST MLP",
        "fmnist_cnn": "Fashion-MNIST CNN",
        "cifar10_cnn": "CIFAR-10 CNN",
    }
    lines = [
        "% Generated by paper/ijcnn2027/build_visuals.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Headline held-out comparisons under conservative bidirectional "
        "unicast accounting. Event-FedAvg is nondominated in all three settings. "
        "It leads the strongest quality-selected control on Fashion-MNIST MLP and "
        "CIFAR-10; on Fashion-MNIST CNN, Strom gains 0.90 accuracy points but uses "
        "4.9$\\times$ more traffic. Values are mean $\\pm$ sample standard deviation "
        "over three independently seeded data realizations.}",
        "\\label{tab:main-results}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular*}{0.99\\textwidth}{@{\\extracolsep{\\fill}}lllrrr@{}}",
        "\\toprule",
        "Benchmark & Selection rule & Method & Accuracy [\\%] & Worst [\\%] & Total [Mbit] \\\\",
        "\\midrule",
    ]
    for index, (key, _title, _architecture, _ylim) in enumerate(PANELS):
        rows = grouped[key]
        event, quality = selected_rows(rows, "quality-selected")
        _event, traffic = selected_rows(rows, "traffic-matched")
        entries = [
            ("Event operating point", event),
            ("Strongest quality control", quality),
            ("Nearest-traffic control", traffic),
        ]
        for row_index, (selection, row) in enumerate(entries):
            benchmark = dataset_labels[key] if row_index == 0 else ""
            lines.append(
                f"{benchmark} & {selection} & {METHODS[row['method']]['label']} & "
                f"{pm(row, 'final_test_accuracy_mean', 'final_test_accuracy_std', 100.0, 2)} & "
                f"{pm(row, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', 100.0, 1)} & "
                f"{pm(row, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', 1.0, 1)} \\\\"
            )
        if index != len(PANELS) - 1:
            lines.append("\\addlinespace[1pt]")

    alignment_mean, alignment_std, descent_mean, descent_std = alignment_metrics()
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\vspace{2pt}",
            "\\parbox{0.99\\textwidth}{\\footnotesize \\emph{Class-wise qualification:} "
            "on CIFAR-10, quality-selected EF-TopK has the highest mean worst-class "
            "accuracy (26.7$\\pm$0.8\\%), versus 24.6$\\pm$4.6\\% for Event-FedAvg. "
            "\\emph{Theory-interface audit "
            "(Fashion-MNIST MLP):} the $q_r$-weighted aggregate-alignment ratio is "
            f"{alignment_mean:.2f}$\\pm${alignment_std:.2f}, and the objective decreases on "
            f"{100 * descent_mean:.2f}$\\pm${100 * descent_std:.2f}\\% of 31 independently "
            "replayed snapshots per partition (93 total). These finite trajectories assess "
            "the conditional alignment assumption; they do not prove it.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def products() -> dict[Path, bytes]:
    configure_matplotlib()
    grouped = benchmark_rows()
    validate_visual_contract(grouped)
    return {
        METHOD_FIGURE: method_figure(),
        FRONTIER_FIGURE: frontier_figure(grouped),
        MAIN_TABLE: main_table(grouped).encode("utf-8"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed visuals against frozen source evidence",
    )
    args = parser.parse_args()

    expected = products()
    stale: list[str] = []
    for path, content in expected.items():
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                stale.append(path.relative_to(REPO).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    if stale:
        raise SystemExit("stale or missing generated visuals: " + ", ".join(stale))
    action = "validated" if args.check else "generated"
    print(f"{action} {len(expected)} visual products")


if __name__ == "__main__":
    main()
