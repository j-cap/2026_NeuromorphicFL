"""Build and validate the frozen Fashion-MNIST evidence used by the paper.

The committed aggregate CSVs are the numerical source of truth.  This script
validates their expected campaign structure, checks communication-accounting
invariants, and generates both a machine-readable master table and LaTeX table
fragments.  Run with ``--check`` in CI or before submission to detect drift.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
HELDOUT = (
    REPO
    / "experiments"
    / "results"
    / "final_baseline_campaign"
    / "observed_heldout_summary.csv"
)
TRAFFIC = (
    REPO
    / "experiments"
    / "results"
    / "final_baseline_campaign"
    / "observed_traffic_matched_summary.csv"
)

MASTER = PAPER / "evidence" / "fmnist_master_results.csv"
QUALITY_TEX = PAPER / "generated" / "fmnist_quality_table.tex"
TRAFFIC_TEX = PAPER / "generated" / "fmnist_traffic_table.tex"

PARTITION_SEEDS = "2500;2600;2700"
TRAIN_SEEDS = "72500;72600;72700"
ARCH_ORDER = {"mlp": 0, "cnn": 1}
QUALITY_METHOD_ORDER = {
    "event": 0,
    "ef_topk": 1,
    "sign_ef": 2,
    "strom": 3,
    "dense": 4,
}
TRAFFIC_METHOD_ORDER = {"event": 0, "strom": 1, "ef_topk": 2}

METRIC_COLUMNS = [
    "final_test_ce_mean",
    "final_test_ce_std",
    "final_test_accuracy_mean",
    "final_test_accuracy_std",
    "final_worst_class_accuracy_mean",
    "final_worst_class_accuracy_std",
    "uplink_Mbit_mean",
    "uplink_Mbit_std",
    "broadcast_total_Mbit_mean",
    "broadcast_total_Mbit_std",
    "unicast_total_Mbit_mean",
    "unicast_total_Mbit_std",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"architecture", "method", "configuration_value", *METRIC_COLUMNS}
    if not rows:
        raise ValueError(f"empty evidence source: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return rows


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["architecture"], row["method"]


def validate(heldout: list[dict[str, str]], traffic: list[dict[str, str]]) -> None:
    expected_heldout = {
        (arch, method)
        for arch in ARCH_ORDER
        for method in QUALITY_METHOD_ORDER
    }
    expected_traffic = {
        (arch, method)
        for arch in ARCH_ORDER
        for method in TRAFFIC_METHOD_ORDER
    }
    if {key(row) for row in heldout} != expected_heldout:
        raise ValueError("quality-selected campaign rows do not match the frozen design")
    if {key(row) for row in traffic} != expected_traffic:
        raise ValueError("traffic-matched campaign rows do not match the frozen design")

    expected_configuration = {
        ("mlp", "ef_topk", "quality-selected"): "0.05",
        ("cnn", "ef_topk", "quality-selected"): "0.05",
        ("mlp", "strom", "quality-selected"): "0.00125",
        ("cnn", "strom", "quality-selected"): "0.005",
        ("mlp", "ef_topk", "traffic-matched"): "0.01",
        ("cnn", "ef_topk", "traffic-matched"): "0.01",
        ("mlp", "strom", "traffic-matched"): "0.02",
        ("cnn", "strom", "traffic-matched"): "0.02",
    }

    for comparison, rows in (
        ("quality-selected", heldout),
        ("traffic-matched", traffic),
    ):
        for row in rows:
            lookup = (row["architecture"], row["method"], comparison)
            if lookup in expected_configuration:
                observed = str(float(row["configuration_value"]))
                expected = str(float(expected_configuration[lookup]))
                if observed != expected:
                    raise ValueError(f"unexpected configuration for {lookup}: {observed}")

            for column in METRIC_COLUMNS:
                value = float(row[column])
                if value < 0:
                    raise ValueError(f"negative {column} for {comparison} {key(row)}")
            uplink = float(row["uplink_Mbit_mean"])
            broadcast = float(row["broadcast_total_Mbit_mean"])
            unicast = float(row["unicast_total_Mbit_mean"])
            if not uplink < broadcast < unicast:
                raise ValueError(
                    f"communication totals violate uplink < broadcast < unicast for "
                    f"{comparison} {key(row)}"
                )

    quality_event = {key(row): row for row in heldout if row["method"] == "event"}
    traffic_event = {key(row): row for row in traffic if row["method"] == "event"}
    for event_key, quality_row in quality_event.items():
        traffic_row = traffic_event[event_key]
        for column in METRIC_COLUMNS:
            if quality_row[column] != traffic_row[column]:
                raise ValueError(
                    f"Event-FedAvg drift between comparisons for {event_key}, {column}"
                )


def sorted_rows(
    rows: list[dict[str, str]], order: dict[str, int]
) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (ARCH_ORDER[row["architecture"]], order[row["method"]]))


def master_csv(
    heldout: list[dict[str, str]], traffic: list[dict[str, str]]
) -> str:
    columns = [
        "comparison",
        "architecture",
        "method",
        "configuration_value",
        "n_seeds",
        "partition_seeds",
        "training_seeds",
        *METRIC_COLUMNS,
        "source_artifact",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for comparison, rows, source, order in (
        ("quality-selected", heldout, HELDOUT, QUALITY_METHOD_ORDER),
        ("traffic-matched", traffic, TRAFFIC, TRAFFIC_METHOD_ORDER),
    ):
        for row in sorted_rows(rows, order):
            output = {
                "comparison": comparison,
                "architecture": row["architecture"],
                "method": row["method"],
                "configuration_value": row["configuration_value"],
                "n_seeds": "3",
                "partition_seeds": PARTITION_SEEDS,
                "training_seeds": TRAIN_SEEDS,
                "source_artifact": source.relative_to(REPO).as_posix(),
            }
            output.update({column: row[column] for column in METRIC_COLUMNS})
            writer.writerow(output)
    return stream.getvalue()


def method_label(row: dict[str, str]) -> str:
    method = row["method"]
    value = row["configuration_value"]
    if method == "event":
        return "Event-FedAvg"
    if method == "ef_topk":
        return f"EF-TopK {100 * float(value):g}\\%"
    if method == "sign_ef":
        return "Sign-EF"
    if method == "strom":
        return f"Strom ($\\tau={float(value):g}$)"
    if method == "dense":
        return "Dense FedAvg"
    raise ValueError(method)


def pm(row: dict[str, str], mean: str, std: str, scale: float, digits: int) -> str:
    return (
        f"{scale * float(row[mean]):.{digits}f}"
        f"$\\pm${scale * float(row[std]):.{digits}f}"
    )


def latex_table(
    rows: list[dict[str, str]], *, comparison: str, label: str, caption: str,
    order: dict[str, int]
) -> str:
    lines = [
        "% Generated by paper/ijcnn2027/build_evidence.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Architecture & Method & Test CE & Accuracy [\\%] & Worst class [\\%] & Total [Mbit] \\\\",
        "\\midrule",
    ]
    ordered = sorted_rows(rows, order)
    previous_arch = None
    for row in ordered:
        arch = row["architecture"]
        if previous_arch is not None and arch != previous_arch:
            lines.append("\\midrule")
        lines.append(
            f"{arch.upper()} & {method_label(row)} & "
            f"{pm(row, 'final_test_ce_mean', 'final_test_ce_std', 1.0, 4)} & "
            f"{pm(row, 'final_test_accuracy_mean', 'final_test_accuracy_std', 100.0, 2)} & "
            f"{pm(row, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', 100.0, 1)} & "
            f"{pm(row, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', 1.0, 1)} \\\\"
        )
        previous_arch = arch
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def products(
    heldout: list[dict[str, str]], traffic: list[dict[str, str]]
) -> dict[Path, str]:
    return {
        MASTER: master_csv(heldout, traffic),
        QUALITY_TEX: latex_table(
            heldout,
            comparison="quality-selected",
            label="tab:fmnist-quality-selected",
            caption=(
                "Fashion-MNIST quality-selected comparison over three held-out "
                "partitions. Values are mean $\\pm$ sample standard deviation; "
                "communication is conservative bidirectional unicast traffic."
            ),
            order=QUALITY_METHOD_ORDER,
        ),
        TRAFFIC_TEX: latex_table(
            traffic,
            comparison="traffic-matched",
            label="tab:fmnist-traffic-matched",
            caption=(
                "Fashion-MNIST traffic-matched comparison over the same three "
                "held-out partitions."
            ),
            order=TRAFFIC_METHOD_ORDER,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that committed products match the source CSVs",
    )
    args = parser.parse_args()

    heldout = read_rows(HELDOUT)
    traffic = read_rows(TRAFFIC)
    validate(heldout, traffic)
    expected = products(heldout, traffic)

    stale: list[str] = []
    for path, content in expected.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(REPO).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    if stale:
        raise SystemExit("stale or missing generated evidence: " + ", ".join(stale))
    action = "validated" if args.check else "generated"
    print(f"{action} {len(expected)} evidence products")


if __name__ == "__main__":
    main()
