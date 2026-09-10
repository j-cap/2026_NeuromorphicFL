"""Generate and validate the frozen IJCNN comparison figure and main table.

Figure 1 is maintained as an editable Draw.io diagram. Figure 2 uses the P14
ten-seed extension plus the provisional P13 ResNet-14 development run. Table 1
uses the P12 headline campaign, P14 dense references, and the provisional P13
block.
Run with ``--check`` to detect source, selection-rule, or rendered-artifact
drift.
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
P14 = PAPER / "evidence" / "p14_figure2_ten_seed.csv"
RESNET14 = PAPER / "evidence" / "p13_resnet14_development.csv"

FIGURES = PAPER / "figures"
METHOD_FIGURE = FIGURES / "event_fedavg_method.pdf"
FRONTIER_PANEL_FILES = {
    "fmnist_mlp": FIGURES / "communication_frontier_fmnist_mlp.pdf",
    "fmnist_cnn": FIGURES / "communication_frontier_fmnist_cnn.pdf",
    "cifar_cnn": FIGURES / "communication_frontier_cifar_cnn.pdf",
    "cifar_resnet14": FIGURES / "communication_frontier_cifar_resnet14.pdf",
}
FRONTIER_LEGEND = FIGURES / "communication_frontier_legend.pdf"
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
    # Okabe--Ito-derived, color-vision-deficiency-safe method palette. Distinct
    # marker shapes remain the primary redundant encoding for grayscale print.
    "event": {"label": "Event-FedAvg", "marker": "*", "color": "#D55E00"},
    "strom": {"label": "Strom", "marker": "s", "color": "#0072B2"},
    "ef_topk": {"label": "EF-TopK", "marker": "^", "color": "#009E73"},
    "sign_ef": {"label": "Sign-EF", "marker": "D", "color": "#CC79A7"},
    "dense": {"label": "Dense FedAvg", "marker": "o", "color": "#595959"},
}

# Explicit log-axis limits and labelled ticks keep communication scale readable
# in the narrow four-panel layout while retaining every observed point.
TRAFFIC_AXES = {
    "fmnist_mlp": {"limits": (150.0, 3500.0), "ticks": (200.0, 1000.0)},
    "fmnist_cnn": {"limits": (50.0, 1000.0), "ticks": (100.0, 500.0)},
    "cifar_cnn": {"limits": (150.0, 2200.0), "ticks": (200.0, 1000.0)},
    "cifar_resnet14": {"limits": (7000.0, 300000.0), "ticks": (10000.0, 100000.0)},
}

CORE_PANELS = [
    ("fmnist_mlp", "Fashion-MNIST\nMLP", "mlp", (78.0, 84.0)),
    ("fmnist_cnn", "Fashion-MNIST\nCNN", "cnn", (74.0, 84.0)),
    ("cifar_cnn", "CIFAR-10\ncompact CNN", "cifar_cnn", (32.0, 50.5)),
]
PANELS = CORE_PANELS + [
    ("cifar_resnet14", "CIFAR-10\nResNet-14 ($n=1$)", "cifar_resnet14", (40.0, 78.0)),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty visual evidence source: {path}")
    return rows


def normalized_headline_row(row: dict[str, str]) -> dict[str, str]:
    """Expose the P12 traffic fields through the plotting/table field names."""

    converted = dict(row)
    converted["comparison"] = row["role"]
    converted["unicast_total_Mbit_mean"] = str(
        float(row["unicast_hybrid_total_bits_mean"]) / 1e6
    )
    converted["unicast_total_Mbit_std"] = str(
        float(row["unicast_hybrid_total_bits_std"]) / 1e6
    )
    return converted


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
    for key, _title, _architecture, _ylim in CORE_PANELS:
        selected = []
        for row in source:
            if row["benchmark"] != key or row["role"] not in {
                "event", "quality", "traffic"
            }:
                continue
            selected.append(normalized_headline_row(row))
        if len(selected) != 3:
            raise ValueError(f"{key} does not contain the three frozen P12 points")
        grouped[key] = selected
    return grouped


def frontier_rows() -> dict[str, list[dict[str, str]]]:
    """Return the complete ten-seed frozen comparison used in Figure 2."""

    source = read_csv(P14)
    rows = []
    for row in source:
        converted = dict(row)
        converted["unicast_total_Mbit_mean"] = str(
            float(row["unicast_hybrid_total_bits_mean"]) / 1e6
        )
        converted["unicast_total_Mbit_std"] = str(
            float(row["unicast_hybrid_total_bits_std"]) / 1e6
        )
        rows.append(converted)
    required = {
        "comparison",
        "benchmark",
        "method",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "unicast_total_Mbit_mean",
        "unicast_total_Mbit_std",
    }
    if required.difference(rows[0]):
        raise ValueError("paper evidence lacks columns required by Figure 2")

    grouped: dict[str, list[dict[str, str]]] = {}
    for key, _title, architecture, _ylim in CORE_PANELS:
        selected = [row for row in rows if row["benchmark"] == key]
        if len(selected) != 7:
            raise ValueError(f"{key} does not contain the seven unique visual rows")
        grouped[key] = selected
    resnet = read_csv(RESNET14)
    required_resnet = {
        "benchmark", "comparison", "method", "n_seeds",
        "final_test_accuracy_mean", "final_test_accuracy_std",
        "final_worst_class_accuracy_mean", "final_worst_class_accuracy_std",
        "unicast_hybrid_total_bits_mean", "unicast_hybrid_total_bits_std",
    }
    if required_resnet.difference(resnet[0]) or len(resnet) != 5:
        raise ValueError("provisional ResNet-14 evidence must contain five rows")
    for row in resnet:
        if row["benchmark"] != "cifar_resnet14" or int(row["n_seeds"]) != 1:
            raise ValueError("ResNet-14 panel must remain a single development seed")
        row["unicast_total_Mbit_mean"] = str(
            float(row["unicast_hybrid_total_bits_mean"]) / 1e6
        )
        row["unicast_total_Mbit_std"] = "0"
    grouped["cifar_resnet14"] = resnet
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
        if key == "cifar_resnet14":
            expected = {
                ("quality-selected", method) for method in METHODS
            }
            observed = {(row["comparison"], row["method"]) for row in rows}
            if observed != expected:
                raise ValueError("ResNet-14 provisional rows drifted")
            if "event" not in {row["method"] for row in nondominated(rows)}:
                raise ValueError("Event-FedAvg is not nondominated on ResNet-14")
            continue
        expected = {
            ("quality-selected", method)
            for method in ("event", "strom", "ef_topk", "sign_ef", "dense")
        } | {
            ("traffic-matched", method)
            for method in ("strom", "ef_topk")
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


def save_pdf(fig: plt.Figure, *, tight: bool = True) -> bytes:
    stream = io.BytesIO()
    options = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(stream, format="pdf", metadata=PDF_METADATA, **options)
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


def frontier_panel(key: str, rows: list[dict[str, str]], ylim: tuple[float, float]) -> bytes:
    """Render one title-free panel for assembly and labelling in LaTeX."""
    fig, ax = plt.subplots(figsize=(1.77, 2.02))
    frontier = nondominated(rows)
    ax.plot(
        [value(row, "unicast_total_Mbit_mean") for row in frontier],
        [value(row, "final_test_accuracy_mean", 100.0) for row in frontier],
        color="#737373", linestyle=":", linewidth=0.9, zorder=1,
    )
    for row in rows:
        style = METHODS[row["method"]]
        quality = row["comparison"] == "quality-selected"
        ax.errorbar(
            value(row, "unicast_total_Mbit_mean"),
            value(row, "final_test_accuracy_mean", 100.0),
            xerr=value(row, "unicast_total_Mbit_std"),
            yerr=value(row, "final_test_accuracy_std", 100.0),
            fmt=style["marker"],
            markersize=7.8 if row["method"] == "event" else 5.5,
            markerfacecolor=style["color"] if quality else "white",
            markeredgecolor="#111111", markeredgewidth=0.65,
            color="#4d4d4d", ecolor="#8c8c8c", elinewidth=0.65,
            capsize=1.6, zorder=4 if row["method"] == "event" else 3,
        )
    ax.set_xscale("log")
    ax.set_xlim(*TRAFFIC_AXES[key]["limits"])
    ax.set_xticks(TRAFFIC_AXES[key]["ticks"])
    ax.set_xticklabels([f"{tick:g}" for tick in TRAFFIC_AXES[key]["ticks"]])
    ax.set_ylim(*ylim)
    ax.set_xlabel("Traffic [Mbit]")
    ax.set_ylabel("Test accuracy [%]")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.45)
    ax.grid(True, which="minor", axis="x", color="#eeeeee", linewidth=0.35)
    ax.set_axisbelow(True)
    if key == "cifar_resnet14":
        for spine in ax.spines.values():
            spine.set_color("#b2182b")
    fig.subplots_adjust(left=0.26, right=0.98, top=0.98, bottom=0.19)
    return save_pdf(fig, tight=False)


def frontier_legend() -> bytes:
    """Render the method and operating-point legend shared by all panels."""
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
    fig = plt.figure(figsize=(7.08, 0.24))
    fig.legend(
        handles=method_handles + selection_handles,
        loc="center",
        ncol=7,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.75,
    )
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


def provisional_value(row: dict[str, str], field: str, scale: float, digits: int) -> str:
    return f"\\prov{{{scale * value(row, field):.{digits}f}}}"


def main_table(grouped: dict[str, list[dict[str, str]]]) -> str:
    dataset_labels = {
        "fmnist_mlp": "Fashion-MNIST MLP",
        "fmnist_cnn": "Fashion-MNIST CNN",
        "cifar_cnn": "CIFAR-10 CNN",
        "cifar_resnet14": "CIFAR-10 ResNet-14",
    }
    cnn_event, cnn_quality = selected_rows(grouped["fmnist_cnn"], "quality")
    cnn_gap = 100.0 * (
        value(cnn_quality, "final_test_accuracy_mean")
        - value(cnn_event, "final_test_accuracy_mean")
    )
    cnn_ratio = value(cnn_quality, "unicast_total_Mbit_mean") / value(
        cnn_event, "unicast_total_Mbit_mean"
    )
    cifar_ef = normalized_headline_row(next(
        row for row in read_csv(HEADLINE) if row["point_id"] == "cifar_ef_quality"
    ))
    cifar_event = normalized_headline_row(next(
        row for row in read_csv(HEADLINE) if row["point_id"] == "cifar_event"
    ))
    p14_rows = frontier_rows()
    dense_references = {
        key: next(row for row in p14_rows[key] if row["method"] == "dense")
        for key, _title, _architecture, _ylim in CORE_PANELS
    }
    resnet_rows = p14_rows["cifar_resnet14"]
    lines = [
        "% Generated by paper/ijcnn2027/build_visuals.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\rev{All frozen headline configurations under conservative "
        "bidirectional unicast accounting. Values are mean "
        "$\\pm$ sample standard deviation over ten matched seeds, including a dense "
        "reference for every benchmark. Bold marks the best mean per benchmark and "
        "metric; lower traffic is better.} \\prov{The ResNet-14 block is provisional "
        "development-partition evidence ($n=1$); no uncertainty or confirmatory "
        "claim is attached to it.}}",
        "\\label{tab:main-results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{0.99\\textwidth}{@{\\extracolsep{\\fill}}lllrrr@{}}",
        "\\toprule",
        "Benchmark & Selection rule & Method & Accuracy [\\%] & Worst [\\%] & Total [Mbit] \\\\",
        "\\midrule",
    ]
    for index, (key, _title, _architecture, _ylim) in enumerate(CORE_PANELS):
        rows = grouped[key]
        event, quality = selected_rows(rows, "quality")
        _event, traffic = selected_rows(rows, "traffic")
        entries = [("Event operating point", event)]
        entries.append(("Dense reference", dense_references[key]))
        entries.append(("Frozen quality baseline", quality))
        if key == "cifar_cnn":
            entries.append(("Class-wise reference", cifar_ef))
        entries.append(("Nearest-traffic baseline", traffic))
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
        lines.append("\\midrule")

    entries = [
        ("Event operating point", next(row for row in resnet_rows if row["method"] == "event")),
        ("Dense reference", next(row for row in resnet_rows if row["method"] == "dense")),
        ("Frozen quality baseline", next(row for row in resnet_rows if row["method"] == "ef_topk")),
        ("Frozen quality baseline", next(row for row in resnet_rows if row["method"] == "sign_ef")),
        ("Frozen quality baseline", next(row for row in resnet_rows if row["method"] == "strom")),
    ]
    for row_index, (selection, row) in enumerate(entries):
        benchmark = dataset_labels["cifar_resnet14"] if row_index == 0 else ""
        lines.append(
            f"\\prov{{{benchmark}}} & \\prov{{{selection}}} & "
            f"\\prov{{{METHODS[row['method']]['label']}}} & "
            f"{provisional_value(row, 'final_test_accuracy_mean', 100.0, 2)} & "
            f"{provisional_value(row, 'final_worst_class_accuracy_mean', 100.0, 1)} & "
            f"{provisional_value(row, 'unicast_total_Mbit_mean', 1.0, 1)} \\\\"
        )

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
    products = {
        FRONTIER_PANEL_FILES[key]: frontier_panel(key, frontier[key], ylim)
        for key, _title, _architecture, ylim in PANELS
    }
    products[FRONTIER_LEGEND] = frontier_legend()
    products[MAIN_TABLE] = main_table(headline).encode("utf-8")
    return products


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
