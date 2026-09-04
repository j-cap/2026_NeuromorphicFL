"""Build and validate the frozen Fashion-MNIST and CIFAR-10 paper evidence.

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
CIFAR_ROOT = REPO / "experiments" / "results" / "p3_cifar10"
CIFAR_RUNS = CIFAR_ROOT / "heldout_runs.csv"
CIFAR_SOURCE = CIFAR_ROOT / "master_results.csv"

MASTER = PAPER / "evidence" / "fmnist_master_results.csv"
QUALITY_TEX = PAPER / "generated" / "fmnist_quality_table.tex"
TRAFFIC_TEX = PAPER / "generated" / "fmnist_traffic_table.tex"
CIFAR_MASTER = PAPER / "evidence" / "cifar10_master_results.csv"
CIFAR_TEX = PAPER / "generated" / "cifar10_table.tex"

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


def read_cifar_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "comparison",
        "method",
        "config_name",
        "final_test_ce_mean",
        "final_test_ce_std",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "final_worst_class_accuracy_mean",
        "final_worst_class_accuracy_std",
        "uplink_packetized_bits_mean",
        "uplink_packetized_bits_std",
        "broadcast_total_bits_mean",
        "broadcast_total_bits_std",
        "unicast_hybrid_total_bits_mean",
        "unicast_hybrid_total_bits_std",
    }
    if not rows:
        raise ValueError(f"empty evidence source: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return rows


def validate_cifar(rows: list[dict[str, str]]) -> None:
    expected = {
        ("quality", "event", "event_t025_q005"),
        ("quality", "strom", "strom_t0025"),
        ("quality", "ef_topk", "ef_k05"),
        ("quality", "sign_ef", "sign_ef"),
        ("quality", "dense", "dense"),
        ("traffic", "event", "event_t025_q005"),
        ("traffic", "strom", "strom_t01"),
        ("traffic", "ef_topk", "ef_k01"),
    }
    observed = {
        (row["comparison"], row["method"], row["config_name"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("CIFAR-10 summary rows do not match the frozen P3 design")

    numeric = [
        "final_test_ce_mean",
        "final_test_ce_std",
        "final_test_accuracy_mean",
        "final_test_accuracy_std",
        "final_worst_class_accuracy_mean",
        "final_worst_class_accuracy_std",
        "uplink_packetized_bits_mean",
        "uplink_packetized_bits_std",
        "broadcast_total_bits_mean",
        "broadcast_total_bits_std",
        "unicast_hybrid_total_bits_mean",
        "unicast_hybrid_total_bits_std",
    ]
    for row in rows:
        if any(float(row[column]) < 0 for column in numeric):
            raise ValueError(f"negative CIFAR-10 metric for {row['comparison']} {row['method']}")
        uplink = float(row["uplink_packetized_bits_mean"])
        broadcast = float(row["broadcast_total_bits_mean"])
        unicast = float(row["unicast_hybrid_total_bits_mean"])
        if not uplink < broadcast < unicast:
            raise ValueError(
                "CIFAR-10 communication totals violate uplink < broadcast < "
                f"unicast for {row['comparison']} {row['method']}"
            )

    by_key = {(row["comparison"], row["method"]): row for row in rows}
    for column in numeric:
        if by_key[("quality", "event")][column] != by_key[("traffic", "event")][column]:
            raise ValueError(f"CIFAR-10 Event-FedAvg drift between comparisons: {column}")

    with CIFAR_RUNS.open(newline="", encoding="utf-8") as stream:
        runs = list(csv.DictReader(stream))
    expected_seeds = {"3500", "3600", "3700"}
    if len(runs) != 21 or {row["partition_seed"] for row in runs} != expected_seeds:
        raise ValueError("CIFAR-10 held-out runs do not contain the frozen 21-run design")
    for row in runs:
        if row["train_seed"] != str(80000 + int(row["partition_seed"])):
            raise ValueError("CIFAR-10 partition and training seed mapping drifted")


def cifar_configuration(row: dict[str, str]) -> str:
    return {
        "event_t025_q005": "tau=0.025;q0=0.005",
        "strom_t0025": "tau=0.0025",
        "strom_t01": "tau=0.01",
        "ef_k05": "k=0.05",
        "ef_k01": "k=0.01",
        "sign_ef": "-",
        "dense": "-",
    }[row["config_name"]]


def cifar_master_csv(rows: list[dict[str, str]]) -> str:
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
    order = {"event": 0, "strom": 1, "ef_topk": 2, "sign_ef": 3, "dense": 4}
    for row in sorted(rows, key=lambda item: (item["comparison"] != "quality", order[item["method"]])):
        output = {
            "comparison": "quality-selected" if row["comparison"] == "quality" else "traffic-matched",
            "architecture": "cifar_cnn",
            "method": row["method"],
            "configuration_value": cifar_configuration(row),
            "n_seeds": "3",
            "partition_seeds": "3500;3600;3700",
            "training_seeds": "83500;83600;83700",
            "source_artifact": CIFAR_SOURCE.relative_to(REPO).as_posix(),
        }
        mapping = {
            "final_test_ce_mean": "final_test_ce_mean",
            "final_test_ce_std": "final_test_ce_std",
            "final_test_accuracy_mean": "final_test_accuracy_mean",
            "final_test_accuracy_std": "final_test_accuracy_std",
            "final_worst_class_accuracy_mean": "final_worst_class_accuracy_mean",
            "final_worst_class_accuracy_std": "final_worst_class_accuracy_std",
            "uplink_Mbit_mean": "uplink_packetized_bits_mean",
            "uplink_Mbit_std": "uplink_packetized_bits_std",
            "broadcast_total_Mbit_mean": "broadcast_total_bits_mean",
            "broadcast_total_Mbit_std": "broadcast_total_bits_std",
            "unicast_total_Mbit_mean": "unicast_hybrid_total_bits_mean",
            "unicast_total_Mbit_std": "unicast_hybrid_total_bits_std",
        }
        for destination, source in mapping.items():
            scale = 1e6 if "Mbit" in destination else 1.0
            output[destination] = str(float(row[source]) / scale)
        writer.writerow(output)
    return stream.getvalue()


def cifar_method_label(row: dict[str, str]) -> str:
    return {
        "event": "Event-FedAvg",
        "strom": "Strom",
        "ef_topk": "EF-TopK",
        "sign_ef": "Sign-EF",
        "dense": "Dense FedAvg",
    }[row["method"]]


def cifar_latex_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "% Generated by paper/ijcnn2027/build_evidence.py; do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{CIFAR-10 results over three held-out partitions. Values are mean "
        "$\\pm$ sample standard deviation; total is conservative bidirectional "
        "unicast traffic. Quality-selected and traffic-matched rows are separated.}",
        "\\label{tab:cifar10-results}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Selection & Method & Test CE & Accuracy [\\%] & Worst class [\\%] & Total [Mbit] \\\\",
        "\\midrule",
    ]
    order = {"event": 0, "strom": 1, "ef_topk": 2, "sign_ef": 3, "dense": 4}
    ordered = sorted(rows, key=lambda item: (item["comparison"] != "quality", order[item["method"]]))
    previous = None
    for row in ordered:
        if previous is not None and row["comparison"] != previous:
            lines.append("\\midrule")
        selection = "Quality" if row["comparison"] == "quality" else "Traffic"
        lines.append(
            f"{selection} & {cifar_method_label(row)} & "
            f"{float(row['final_test_ce_mean']):.4f}$\\pm${float(row['final_test_ce_std']):.4f} & "
            f"{100 * float(row['final_test_accuracy_mean']):.2f}$\\pm${100 * float(row['final_test_accuracy_std']):.2f} & "
            f"{100 * float(row['final_worst_class_accuracy_mean']):.1f}$\\pm${100 * float(row['final_worst_class_accuracy_std']):.1f} & "
            f"{float(row['unicast_hybrid_total_bits_mean']) / 1e6:.1f}$\\pm$"
            f"{float(row['unicast_hybrid_total_bits_std']) / 1e6:.1f} \\\\"
        )
        previous = row["comparison"]
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


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
    cifar = read_cifar_rows(CIFAR_SOURCE)
    validate_cifar(cifar)
    expected.update(
        {
            CIFAR_MASTER: cifar_master_csv(cifar),
            CIFAR_TEX: cifar_latex_table(cifar),
        }
    )

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
