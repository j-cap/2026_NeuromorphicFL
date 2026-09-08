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
P12 = PAPER / "evidence" / "p12_headline_ten_seed.csv"
P12_PAIRED = PAPER / "evidence" / "p12_paired_differences.csv"
P8 = PAPER / "evidence" / "p8_targeted_revision.csv"
P11 = PAPER / "evidence" / "p11_alignment_factorial.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise AssertionError(f"empty evidence file: {path.relative_to(REPO)}")
    return result


def point(source: list[dict[str, str]], point_id: str) -> dict[str, str]:
    matches = [row for row in source if row["point_id"] == point_id]
    if len(matches) != 1:
        raise AssertionError(f"expected one P12 {point_id} row, found {len(matches)}")
    return matches[0]


def paired(
    source: list[dict[str, str]], comparison_id: str, subset: str
) -> dict[str, str]:
    matches = [
        row
        for row in source
        if row["comparison_id"] == comparison_id and row["subset"] == subset
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one P12 {comparison_id}/{subset} row, found {len(matches)}"
        )
    return matches[0]


def p8_select(source: list[dict[str, str]], config_name: str) -> dict[str, str]:
    matches = [row for row in source if row["config_name"] == config_name]
    if len(matches) != 1:
        raise AssertionError(f"expected one P8 {config_name} row, found {len(matches)}")
    return matches[0]


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def traffic_mbit(row: dict[str, str], statistic: str = "mean") -> float:
    return number(row, f"unicast_hybrid_total_bits_{statistic}") / 1e6


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
    headline = rows(P12)
    differences = rows(P12_PAIRED)
    mechanism = rows(P8)
    factorial = rows(P11)

    mlp_event = point(headline, "fmnist_mlp_event")
    mlp_topk = point(headline, "fmnist_mlp_ef_quality")
    mlp_near = point(headline, "fmnist_mlp_strom_near")
    cnn_event = point(headline, "fmnist_cnn_event")
    cnn_strom = point(headline, "fmnist_cnn_strom_quality")
    cnn_near = point(headline, "fmnist_cnn_strom_near")
    c_event = point(headline, "cifar_event")
    c_dense = point(headline, "cifar_dense_gain2")
    c_strom = point(headline, "cifar_strom_quality")
    c_near_strom = point(headline, "cifar_strom_near")
    c_topk = point(headline, "cifar_ef_quality")

    c_accuracy_gap = 100 * (
        number(c_event, "final_test_accuracy_mean")
        - number(c_dense, "final_test_accuracy_mean")
    )
    c_traffic_fraction = 100 * (
        traffic_mbit(c_event)
        / traffic_mbit(c_dense)
    )
    c_traffic_fold = (
        traffic_mbit(c_dense)
        / traffic_mbit(c_event)
    )

    claims = [
        (
            "abstract CIFAR Event point",
            f"${pm(c_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ accuracy at "
            f"${pm(c_event, 'unicast_hybrid_total_bits_mean', 'unicast_hybrid_total_bits_std', scale=1e-6, digits=1)}$ Mbit",
        ),
        (
            "abstract tuned dense point",
            f"${pm(c_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
            f"${traffic_mbit(c_dense):.1f}$ Mbit",
        ),
        ("introduction traffic fold", f"${c_traffic_fold:.1f}\\times$ less traffic"),
        ("introduction accuracy gap", f"${c_accuracy_gap:.2f}$ percentage points"),
        (
            "MLP quality tradeoff",
            f"gains ${100 * (number(mlp_event, 'final_test_accuracy_mean') - number(mlp_topk, 'final_test_accuracy_mean')):.2f}$ points over the quality baseline with "
            f"${traffic_mbit(mlp_topk) / traffic_mbit(mlp_event):.1f}\\times$ less traffic",
        ),
        (
            "MLP nearest-traffic gap",
            f"gains ${100 * (number(mlp_event, 'final_test_accuracy_mean') - number(mlp_near, 'final_test_accuracy_mean')):.2f}$ points over Strom at comparable traffic",
        ),
        (
            "CNN quality qualification",
            f"Strom gains ${100 * (number(cnn_strom, 'final_test_accuracy_mean') - number(cnn_event, 'final_test_accuracy_mean')):.2f}$ points but uses "
            f"${traffic_mbit(cnn_strom) / traffic_mbit(cnn_event):.1f}\\times$ more traffic",
        ),
        (
            "CNN nearest-traffic gap",
            f"Event-FedAvg gains ${100 * (number(cnn_event, 'final_test_accuracy_mean') - number(cnn_near, 'final_test_accuracy_mean')):.2f}$ points at nearby traffic",
        ),
        (
            "CIFAR tuned-dense comparison",
            f"leads tuned dense FedAvg by ${c_accuracy_gap:.2f}$ points using only "
            f"${c_traffic_fraction:.1f}\\%$ of its traffic",
        ),
        (
            "CIFAR quality Strom",
            f"leads quality-selected Strom by ${100 * (number(c_event, 'final_test_accuracy_mean') - number(c_strom, 'final_test_accuracy_mean')):.2f}$ points with "
            f"${traffic_mbit(c_strom) / traffic_mbit(c_event):.1f}\\times$ less traffic",
        ),
        (
            "CIFAR worst-class qualification",
            f"has ${pm(c_event, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', scale=100, digits=1)}\\%$ mean worst-class accuracy and frozen EF-TopK has "
            f"${pm(c_topk, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', scale=100, digits=1)}\\%$",
        ),
    ]

    paired_specs = (
        ("fmnist_mlp_event_minus_quality_accuracy", "+0.69", "[0.25,1.12]"),
        ("fmnist_cnn_event_minus_quality_accuracy", "-0.86", "[-1.66,-0.05]"),
        ("cifar_event_minus_quality_accuracy", "+1.14", "[0.09,2.19]"),
    )
    for comparison_id, displayed_mean, displayed_interval in paired_specs:
        row = paired(differences, comparison_id, "new_seven")
        expected_mean = f"{number(row, 'mean_difference_points'):+.2f}"
        expected_interval = (
            f"[{number(row, 'ci95_low_points'):.2f},"
            f"{number(row, 'ci95_high_points'):.2f}]"
        )
        if (displayed_mean, displayed_interval) != (expected_mean, expected_interval):
            raise AssertionError(f"hard-coded P12 rounding drift for {comparison_id}")
        claims.append((f"P12 paired mean {comparison_id}", f"${displayed_mean}$"))
        claims.append(
            (f"P12 paired interval {comparison_id}", f"${displayed_interval}$")
        )

    worst_difference = paired(
        differences, "cifar_event_minus_ef_worst_class", "all_ten"
    )
    claims.append(
        (
            "CIFAR worst-class paired interval",
            f"$[{number(worst_difference, 'ci95_low_points'):.2f},"
            f"{number(worst_difference, 'ci95_high_points'):.2f}]$",
        )
    )

    frozen = p8_select(mechanism, "event_frozen")
    no_leak = p8_select(mechanism, "event_no_leak")
    coupled = p8_select(mechanism, "event_coupled_quantum")
    for label, row in (
        ("P8 frozen rerun", frozen),
        ("P8 no-leak ablation", no_leak),
        ("P8 coupled-resolution ablation", coupled),
    ):
        claims.append(
            (
                label,
                f"${pm(row, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$",
            )
        )

    total_factorial_snapshots = sum(
        int(row["n_seeds"]) * int(row["snapshots_per_seed"])
        for row in factorial
    )
    claims.append(
        (
            "P11 audited snapshot count",
            f"${total_factorial_snapshots:,}$".replace(",", "{,}")
            + " snapshots",
        )
    )

    for label, fragment in claims:
        require(manuscript, label, fragment)
    print(f"validated {len(claims)} empirical manuscript claims against frozen CSVs")


if __name__ == "__main__":
    main()
