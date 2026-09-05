"""Check empirical manuscript prose against the frozen evidence CSVs.

Generated tables and figures are checked by their builders.  This guard covers
the rounded values and derived comparisons that are written directly in
``main.tex`` so a source update cannot silently leave stale narrative claims.
"""

from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
MANUSCRIPT = PAPER / "main.tex"
FMNIST = PAPER / "evidence" / "fmnist_master_results.csv"
CIFAR10 = PAPER / "evidence" / "cifar10_master_results.csv"
P8 = PAPER / "evidence" / "p8_targeted_revision.csv"
ALIGNMENT = PAPER / "evidence" / "t4_alignment_audit.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise AssertionError(f"empty evidence file: {path.relative_to(REPO)}")
    return result


def select(
    source: list[dict[str, str]],
    *,
    comparison: str,
    architecture: str,
    method: str,
) -> dict[str, str]:
    matches = [
        row
        for row in source
        if row["comparison"] == comparison
        and row["architecture"] == architecture
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {comparison}/{architecture}/{method} row, found {len(matches)}"
        )
    return matches[0]


def p8_select(source: list[dict[str, str]], config_name: str) -> dict[str, str]:
    matches = [row for row in source if row["config_name"] == config_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one P8 {config_name} row, found {len(matches)}")
    return matches[0]


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def pm(
    row: dict[str, str],
    mean: str,
    std: str,
    *,
    scale: float,
    digits: int,
) -> str:
    return (
        f"{scale * number(row, mean):.{digits}f}"
        f"\\pm{scale * number(row, std):.{digits}f}"
    )


def require(text: str, label: str, fragment: str) -> None:
    if fragment not in text:
        raise AssertionError(f"stale or missing manuscript claim ({label}): {fragment}")


def main() -> None:
    manuscript = " ".join(MANUSCRIPT.read_text(encoding="utf-8").split())
    fmnist = rows(FMNIST)
    cifar = rows(CIFAR10)
    mechanism = rows(P8)
    alignment = {row["metric"]: row for row in rows(ALIGNMENT)}

    mlp_event = select(
        fmnist, comparison="quality-selected", architecture="mlp", method="event"
    )
    mlp_topk = select(
        fmnist, comparison="quality-selected", architecture="mlp", method="ef_topk"
    )
    mlp_near = select(
        fmnist, comparison="traffic-matched", architecture="mlp", method="strom"
    )
    cnn_event = select(
        fmnist, comparison="quality-selected", architecture="cnn", method="event"
    )
    cnn_strom = select(
        fmnist, comparison="quality-selected", architecture="cnn", method="strom"
    )
    cnn_near = select(
        fmnist, comparison="traffic-matched", architecture="cnn", method="strom"
    )
    c_event = select(
        cifar, comparison="quality-selected", architecture="cifar_cnn", method="event"
    )
    c_dense = select(
        cifar, comparison="quality-selected", architecture="cifar_cnn", method="dense"
    )
    c_strom = select(
        cifar, comparison="quality-selected", architecture="cifar_cnn", method="strom"
    )
    c_near_strom = select(
        cifar, comparison="traffic-matched", architecture="cifar_cnn", method="strom"
    )
    c_near_topk = select(
        cifar, comparison="traffic-matched", architecture="cifar_cnn", method="ef_topk"
    )

    c_accuracy_gap = 100 * (
        number(c_event, "final_test_accuracy_mean")
        - number(c_dense, "final_test_accuracy_mean")
    )
    c_traffic_fraction = 100 * (
        number(c_event, "unicast_total_Mbit_mean")
        / number(c_dense, "unicast_total_Mbit_mean")
    )
    c_traffic_fold = (
        number(c_dense, "unicast_total_Mbit_mean")
        / number(c_event, "unicast_total_Mbit_mean")
    )

    claims = [
        (
            "abstract CIFAR Event point",
            f"${pm(c_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ accuracy at "
            f"${pm(c_event, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', scale=1, digits=1)}$ Mbit",
        ),
        (
            "abstract tuned dense point",
            f"${pm(c_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
            f"${number(c_dense, 'unicast_total_Mbit_mean'):.1f}$ Mbit",
        ),
        ("introduction traffic fold", f"${c_traffic_fold:.1f}\\times$ less traffic"),
        ("introduction accuracy gap", f"${c_accuracy_gap:.2f}$ percentage points"),
        (
            "MLP Event point",
            f"${pm(mlp_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
            f"${pm(mlp_event, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', scale=1, digits=1)}$ Mbit",
        ),
        (
            "MLP quality control",
            f"${pm(mlp_topk, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ using "
            f"${number(mlp_topk, 'unicast_total_Mbit_mean'):.1f}$ Mbit",
        ),
        (
            "MLP nearest-traffic control",
            f"${pm(mlp_near, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
            f"${pm(mlp_near, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', scale=1, digits=1)}$ Mbit",
        ),
        (
            "CNN quality qualification",
            f"Strom gains ${100 * (number(cnn_strom, 'final_test_accuracy_mean') - number(cnn_event, 'final_test_accuracy_mean')):.2f}$ accuracy points, but requires "
            f"${pm(cnn_strom, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', scale=1, digits=1)}$ Mbit versus "
            f"${pm(cnn_event, 'unicast_total_Mbit_mean', 'unicast_total_Mbit_std', scale=1, digits=1)}$ Mbit",
        ),
        (
            "CNN nearest-traffic gap",
            f"Event-FedAvg exceeds Strom by ${100 * (number(cnn_event, 'final_test_accuracy_mean') - number(cnn_near, 'final_test_accuracy_mean')):.2f}$ points",
        ),
        (
            "CIFAR tuned-dense comparison",
            f"its ${pm(c_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ accuracy is "
            f"${c_accuracy_gap:.2f}$ points higher at only ${c_traffic_fraction:.1f}\\%$ of the traffic. Tuned dense FedAvg reaches "
            f"${pm(c_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$",
        ),
        (
            "CIFAR quality Strom",
            f"${pm(c_strom, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ but uses "
            f"${number(c_strom, 'unicast_total_Mbit_mean') / number(c_event, 'unicast_total_Mbit_mean'):.1f}\\times$ more traffic",
        ),
        (
            "CIFAR nearest Strom",
            f"uses ${100 * (number(c_near_strom, 'unicast_total_Mbit_mean') / number(c_event, 'unicast_total_Mbit_mean') - 1):.1f}\\%$ more traffic and trails by "
            f"${100 * (number(c_event, 'final_test_accuracy_mean') - number(c_near_strom, 'final_test_accuracy_mean')):.2f}$ points",
        ),
        (
            "CIFAR nearest EF-TopK",
            f"uses ${100 * (1 - number(c_near_topk, 'unicast_total_Mbit_mean') / number(c_event, 'unicast_total_Mbit_mean')):.1f}\\%$ less traffic but trails by "
            f"${100 * (number(c_event, 'final_test_accuracy_mean') - number(c_near_topk, 'final_test_accuracy_mean')):.2f}$ points",
        ),
        (
            "CIFAR worst-class qualification",
            f"(${100 * number(select(cifar, comparison='quality-selected', architecture='cifar_cnn', method='ef_topk'), 'final_worst_class_accuracy_mean'):.1f}\\%$) than Event-FedAvg "
            f"(${100 * number(c_event, 'final_worst_class_accuracy_mean'):.1f}\\%$). Tuned dense FedAvg reaches "
            f"${100 * number(c_dense, 'final_worst_class_accuracy_mean'):.1f}\\%$",
        ),
    ]

    frozen = p8_select(mechanism, "event_frozen")
    no_leak = p8_select(mechanism, "event_no_leak")
    coupled = p8_select(mechanism, "event_coupled_quantum")
    for label, row, traffic_digits in (
        ("P8 frozen rerun", frozen, 1),
        ("P8 no-leak ablation", no_leak, 1),
        ("P8 coupled-resolution ablation", coupled, 1),
    ):
        claims.append(
            (
                label,
                f"${pm(row, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
                f"${pm(row, 'unicast_hybrid_total_bits_mean', 'unicast_hybrid_total_bits_std', scale=1e-6, digits=traffic_digits)}$ Mbit",
            )
        )

    ratio = alignment["weighted_alignment_ratio"]
    descent = alignment["objective_decrease_fraction"]
    claims.extend(
        [
            (
                "finite-trajectory alignment ratio",
                f"${pm(ratio, 'mean', 'std', scale=1, digits=2)}$",
            ),
            (
                "finite-trajectory descent frequency",
                f"${pm(descent, 'mean', 'std', scale=100, digits=2)}\\%$ of audited rounds",
            ),
        ]
    )

    for label, fragment in claims:
        require(manuscript, label, fragment)
    print(f"validated {len(claims)} empirical manuscript claims against frozen CSVs")


if __name__ == "__main__":
    main()
