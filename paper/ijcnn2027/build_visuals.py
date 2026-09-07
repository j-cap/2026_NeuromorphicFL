"""Generate and validate the frozen IJCNN comparison figure and main table.

Figure 1 is maintained as an editable Draw.io diagram. Figure 2 uses the full
frozen comparison campaign, while Table 1 uses the later ten-seed headline
campaign. Run with ``--check`` to detect source, selection-rule, or
rendered-artifact drift.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
HEADLINE = PAPER / "evidence" / "p12_headline_ten_seed.csv"
FMNIST = PAPER / "evidence" / "fmnist_master_results.csv"
CIFAR10 = PAPER / "evidence" / "cifar10_master_results.csv"

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
    ("cifar_cnn", "CIFAR-10\ncompact CNN", "cifar_cnn", (32.0, 50.5)),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty visual evidence source: {path}")
    return rows


def headline_rows() -> dict[str, list[dict[str, str]]]:
    source = read_csv(HEADLINE)
    required = {
        "point_id",
        "benchmark",
        "role",
        "method",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "unicast_hybrid_total_bits_mean",
        "unicast_hybrid_total_bits_std",
    }
    if required.difference(source[0]):
        raise ValueError("P12 evidence lacks columns required by the visuals")

    grouped: dict[str, list[dict[str, str]]] = {}
    for key, _title, _architecture, _ylim in PANELS:
        selected = []
        for row in source:
            if row["benchmark"] != key or row["role"] not in {
                "event", "quality", "traffic"
            }:
                continue
            converted = dict(row)
            converted["comparison"] = row["role"]
            converted["unicast_total_Mbit_mean"] = str(
                float(row["unicast_hybrid_total_bits_mean"]) / 1e6
            )
            converted["unicast_total_Mbit_std"] = str(
                float(row["unicast_hybrid_total_bits_std"]) / 1e6
            )
            selected.append(converted)
        if len(selected) != 3:
            raise ValueError(f"{key} does not contain the three frozen P12 points")
        grouped[key] = selected
    return grouped


def frontier_rows() -> dict[str, list[dict[str, str]]]:
    """Return the complete frozen comparison campaign used in Figure 2."""

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
        raise ValueError("paper evidence lacks columns required by Figure 2")

    grouped: dict[str, list[dict[str, str]]] = {}
    for key, _title, architecture, _ylim in PANELS:
        selected = [row for row in rows if row["architecture"] == architecture]
        if len(selected) != 8:
            raise ValueError(f"{key} does not contain the frozen eight visual rows")
        grouped[key] = selected
    return grouped


def value(row: dict[str, str], field: str, scale: float = 1.0) -> float:
    return scale * float(row[field])


def nondominated(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return points not beaten by lower/equal traffic and higher/equal accuracy."""

    points = rows
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


def validate_headline_contract(grouped: dict[str, list[dict[str, str]]]) -> None:
    markers = [style["marker"] for style in METHODS.values()]
    if len(set(markers)) != len(markers):
        raise ValueError("method markers must remain distinct for grayscale output")

    for key, rows in grouped.items():
        quality_method = "ef_topk" if key == "fmnist_mlp" else "strom"
        expected = {
            ("event", "event"),
            ("quality", quality_method),
            ("traffic", "strom"),
        }
        observed = {(row["comparison"], row["method"]) for row in rows}
        if observed != expected:
            raise ValueError(f"{key} selection rows drifted from the frozen design")
        if "event" not in {row["method"] for row in nondominated(rows)}:
            raise ValueError(f"Event-FedAvg is no longer nondominated in {key}")


def validate_frontier_contract(grouped: dict[str, list[dict[str, str]]]) -> None:
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
            raise ValueError(f"{key} comparison rows drifted from the frozen design")
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
    event = next(row for row in rows if row["comparison"] == "event")
    selected = next(row for row in rows if row["comparison"] == comparison)
    return event, selected


def pm(
    row: dict[str, str], mean: str, std: str, scale: float, digits: int,
    bold: bool = False,
) -> str:
    rendered = (
        f"{scale * value(row, mean):.{digits}f}"
        f"$\\pm${scale * value(row, std):.{digits}f}"
    )
    if bold:
        rendered = rendered.replace("$\\pm$", "$\\boldsymbol{\\pm}$")
        return f"\\textbf{{{rendered}}}"
    return rendered


def main_table(grouped: dict[str, list[dict[str, str]]]) -> str:
    dataset_labels = {
        "fmnist_mlp": "Fashion-MNIST MLP",
        "fmnist_cnn": "Fashion-MNIST CNN",
        "cifar_cnn": "CIFAR-10 CNN",
    }
    cnn_event, cnn_quality = selected_rows(grouped["fmnist_cnn"], "quality")
    cnn_gap = 100.0 * (
        value(cnn_quality, "final_test_accuracy_mean")
        - value(cnn_event, "final_test_accuracy_mean")
    )
    cnn_ratio = value(cnn_quality, "unicast_total_Mbit_mean") / value(
        cnn_event, "unicast_total_Mbit_mean"
    )
    cifar_ef = next(
        row for row in read_csv(HEADLINE) if row["point_id"] == "cifar_ef_quality"
    )
    cifar_event = next(
        row for row in read_csv(HEADLINE) if row["point_id"] == "cifar_event"
    )
    lines = [
        "% Generated by paper/ijcnn2027/build_visuals.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Headline held-out comparisons under conservative bidirectional "
        "unicast accounting. Event-FedAvg is nondominated in all three settings. "
        "It leads the frozen quality baseline on Fashion-MNIST MLP and CIFAR-10. "
        f"On Fashion-MNIST CNN, Strom gains {cnn_gap:.2f} accuracy points but uses "
        f"{cnn_ratio:.1f}$\\times$ more traffic. Values are mean $\\pm$ sample "
        "standard deviation over ten paired data and training seeds. Bold marks "
        "the best mean in each benchmark and metric; lower is better for traffic.}",
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
        event, quality = selected_rows(rows, "quality")
        _event, traffic = selected_rows(rows, "traffic")
        entries = [
            ("Event operating point", event),
            ("Frozen quality baseline", quality),
            ("Nearest-traffic baseline", traffic),
        ]
        best_accuracy = max(value(row, "final_test_accuracy_mean") for _, row in entries)
        best_worst = max(value(row, "final_worst_class_accuracy_mean") for _, row in entries)
        best_traffic = min(value(row, "unicast_total_Mbit_mean") for _, row in entries)
        for row_index, (selection, row) in enumerate(entries):
            benchmark = dataset_labels[key] if row_index == 0 else ""
            lines.append(
                f"{benchmark} & {selection} & {METHODS[row['method']]['label']} & "
                f"{pm(row, 'final_test_accuracy_mean', 'final_test_accuracy_std', 100.0, 2, value(row, 'final_test_accuracy_mean') == best_accuracy)} & "
                f"{pm(row, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', 100.0, 1, value(row, 'final_worst_class_accuracy_mean') == best_worst)} & "
                f"{pm(row, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', 1.0, 1, value(row, 'unicast_total_Mbit_mean') == best_traffic)} \\\\"
            )
        if index != len(PANELS) - 1:
            lines.append("\\addlinespace[1pt]")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\vspace{2pt}",
            "\\parbox{0.99\\textwidth}{\\footnotesize \\emph{Baseline sources:} "
            "dense FedAvg~\\cite{mcmahan2017fedavg}, Sign-EF based on sign "
            "quantization and error feedback~\\cite{seide2014onebit,bernstein2018signsgd}, "
            "EF-TopK with memory~\\cite{stich2018memory,richtarik2021ef21}, and "
            "Strom threshold pulses~\\cite{strom2015distributed}. "
            "\\emph{Class-wise qualification:} "
            "on CIFAR-10, Event-FedAvg reaches "
            f"{100 * float(cifar_event['final_worst_class_accuracy_mean']):.1f}"
            f"$\\pm${100 * float(cifar_event['final_worst_class_accuracy_std']):.1f}\\% "
            "mean worst-class accuracy, versus "
            f"{100 * float(cifar_ef['final_worst_class_accuracy_mean']):.1f}"
            f"$\\pm${100 * float(cifar_ef['final_worst_class_accuracy_std']):.1f}\\% "
            "for the frozen EF-TopK quality point.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def products() -> dict[Path, bytes]:
    configure_matplotlib()
    headline = headline_rows()
    frontier = frontier_rows()
    validate_headline_contract(headline)
    validate_frontier_contract(frontier)
    return {
        FRONTIER_FIGURE: frontier_figure(frontier),
        MAIN_TABLE: main_table(headline).encode("utf-8"),
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
