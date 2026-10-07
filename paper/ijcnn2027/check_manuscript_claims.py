"""Check empirical manuscript prose against the frozen evidence CSVs.

Generated tables and figures are checked by their builders.  This guard covers
the rounded values and derived comparisons that are written directly in
``main.tex`` so a source update cannot silently leave stale narrative claims.
"""

from __future__ import annotations

import csv
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
MANUSCRIPT = PAPER / "main.tex"
P12 = PAPER / "evidence" / "p12_headline_ten_seed.csv"
P12_PAIRED = PAPER / "evidence" / "p12_paired_differences.csv"
P11 = PAPER / "evidence" / "p11_alignment_factorial.csv"
P13 = PAPER / "evidence" / "p13_resnet14_ten_seed.csv"
P13_PAIRED = PAPER / "evidence" / "p13_resnet14_paired.csv"
P14 = PAPER / "evidence" / "p14_figure2_ten_seed.csv"
P17_ROOT = REPO / "experiments" / "results" / "p17_fmnist_final_horizons"
P17 = P17_ROOT / "aggregate.csv"
P19_ROOT = REPO / "experiments" / "results" / "p19_cifar10_cnn_final_horizons"
P19 = P19_ROOT / "aggregate.csv"
P19_PAIRED = P19_ROOT / "paired_differences.csv"
P19_HISTORIES = P19_ROOT / "histories.csv"
P13_ROOT = REPO / "experiments" / "results" / "p13_cifar10_resnet14"
P8_SUMMARY = (
    REPO
    / "experiments"
    / "results"
    / "p8_targeted_revision"
    / "heldout_summary.csv"
)
P15_ROOT = REPO / "experiments" / "results" / "p15_operator_ablation"
P15_SUMMARY = P15_ROOT / "summary.csv"
P15_PAIRED = P15_ROOT / "paired.csv"
P15_TABLE = PAPER / "generated" / "p15_operator_ablation_table.tex"


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


def paired_point(
    source: list[dict[str, str]], comparison_id: str
) -> dict[str, str]:
    matches = [row for row in source if row["comparison_id"] == comparison_id]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {comparison_id} paired row, found {len(matches)}"
        )
    return matches[0]


def history_mean(
    source: list[dict[str, str]], point_id: str, round_number: int, field: str
) -> float:
    values = [
        number(row, field)
        for row in source
        if row["point_id"] == point_id and int(row["round"]) == round_number
    ]
    if not values:
        raise AssertionError(
            f"no history rows for {point_id} at round {round_number}"
        )
    return statistics.mean(values)


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def traffic_mbit(row: dict[str, str], statistic: str = "mean") -> float:
    return number(row, f"unicast_hybrid_total_bits_{statistic}") / 1e6


def mean_traffic_to_accuracy(config_name: str, target_percent: float) -> float:
    histories: list[list[dict[str, str]]] = []
    for seed in range(4200, 5200, 100):
        path = P13_ROOT / f"{config_name}_p{seed}_heldout-r3000_history.csv"
        histories.append(rows(path))
    reference_rounds = [int(row["round"]) for row in histories[0]]
    if any(
        [int(row["round"]) for row in history] != reference_rounds
        for history in histories[1:]
    ):
        raise AssertionError("P13 histories do not use matched evaluation rounds")
    for index in range(len(reference_rounds)):
        mean_accuracy = statistics.mean(
            100 * float(history[index]["test_accuracy"]) for history in histories
        )
        if mean_accuracy >= target_percent:
            return statistics.mean(
                (
                    float(history[index]["uplink_packetized_bits"])
                    + float(history[index]["unicast_hybrid_downlink_bits"])
                )
                / 1e9
                for history in histories
            )
    raise AssertionError(f"{config_name} never reaches {target_percent}% mean accuracy")


def resnet_mean(config_name: str, field: str) -> float:
    values = []
    for seed in range(4200, 5200, 100):
        path = P13_ROOT / f"{config_name}_p{seed}_heldout-r3000.csv"
        values.append(float(rows(path)[0][field]))
    return statistics.mean(values)


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
    manuscript = " ".join(
        (
            MANUSCRIPT.read_text(encoding="utf-8")
            + " "
            + P15_TABLE.read_text(encoding="utf-8")
        ).split()
    )
    headline = rows(P12)
    differences = rows(P12_PAIRED)
    factorial = rows(P11)
    resnet = rows(P13)
    resnet_paired = rows(P13_PAIRED)
    figure2 = rows(P14)
    p17 = rows(P17)
    p19 = rows(P19)
    p19_paired = rows(P19_PAIRED)
    p19_histories = rows(P19_HISTORIES)
    p8_summary = rows(P8_SUMMARY)
    p15_summary = rows(P15_SUMMARY)
    p15_paired = rows(P15_PAIRED)

    mlp_event = point(headline, "fmnist_mlp_event")
    mlp_topk = point(headline, "fmnist_mlp_ef_quality")
    mlp_near = point(headline, "fmnist_mlp_strom_near")
    cnn_event = point(headline, "fmnist_cnn_event")
    cnn_strom = point(headline, "fmnist_cnn_strom_quality")
    cnn_near = point(headline, "fmnist_cnn_strom_near")
    c_event = point(p19, "cifar_event")
    c_dense = point(p19, "cifar_dense_gain2")
    c_strom = point(p19, "cifar_strom_quality")
    c_near_strom = point(p19, "cifar_strom_near")
    c_topk = point(p19, "cifar_ef_quality")
    resnet_event = next(row for row in resnet if row["method"] == "event")
    resnet_dense = next(row for row in resnet if row["method"] == "dense")
    resnet_difference = next(
        row for row in resnet_paired if row["comparison"] == "event_minus_dense"
    )
    p8_frozen = next(
        row for row in p8_summary if row["config_name"] == "event_frozen"
    )
    p8_no_leak = next(
        row for row in p8_summary if row["config_name"] == "event_no_leak"
    )

    accounting_ratios = []
    for source, event_id, dense_id in (
        (p17, "fmnist_mlp_event_quality", "fmnist_mlp_dense_quality"),
        (p17, "fmnist_cnn_event_quality", "fmnist_cnn_dense_quality"),
        (p19, "cifar_event", "cifar_dense_gain2"),
    ):
        event_row = point(source, event_id)
        dense_row = point(source, dense_id)
        accounting_ratios.append(
            (
                100
                * number(event_row, "broadcast_total_bits_mean")
                / number(dense_row, "broadcast_total_bits_mean"),
                100
                * number(event_row, "unicast_hybrid_total_bits_mean")
                / number(dense_row, "unicast_hybrid_total_bits_mean"),
            )
        )
    accounting_ratios.append(
        (
            100
            * resnet_mean("event_t0125_q005", "broadcast_total_bits")
            / resnet_mean("dense_g15", "broadcast_total_bits"),
            100
            * resnet_mean("event_t0125_q005", "unicast_hybrid_total_bits")
            / resnet_mean("dense_g15", "unicast_hybrid_total_bits"),
        )
    )

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
    c_dense_difference = paired_point(
        p19_paired, "cifar_event_minus_cifar_dense_gain2"
    )
    c_topk_difference = paired_point(
        p19_paired, "cifar_event_minus_cifar_ef_quality"
    )
    c_accuracy_1200 = 100 * history_mean(
        p19_histories, "cifar_event", 1200, "test_accuracy"
    )
    c_accuracy_1800 = 100 * history_mean(
        p19_histories, "cifar_event", 1800, "test_accuracy"
    )

    claims = [
        (
            "broadcast-accounting sensitivity",
            "they are "
            + ", ".join(f"${broadcast:.1f}\\%$" for broadcast, _ in accounting_ratios[:-1])
            + f", and ${accounting_ratios[-1][0]:.1f}\\%$",
        ),
        (
            "unicast-accounting sensitivity",
            "compared with "
            + ", ".join(f"${unicast:.1f}\\%$" for _, unicast in accounting_ratios[:-1])
            + f", and ${accounting_ratios[-1][1]:.1f}\\%$ under conservative unicast",
        ),
        (
            "abstract ResNet Event point",
            f"${pm(resnet_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ accuracy at "
            f"${pm(resnet_event, 'unicast_hybrid_total_bits_mean', 'unicast_hybrid_total_bits_std', scale=1e-9, digits=2)}$ Gbit",
        ),
        (
            "abstract ResNet dense point",
            f"${pm(resnet_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ at "
            f"${number(resnet_dense, 'unicast_hybrid_total_bits_mean') / 1e9:.2f}$ Gbit",
        ),
        (
            "introduction ResNet traffic fold",
            f"${number(resnet_dense, 'unicast_hybrid_total_bits_mean') / number(resnet_event, 'unicast_hybrid_total_bits_mean'):.1f}\\times$ less traffic",
        ),
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
            "compact-CNN Event endpoint",
            f"Event-FedAvg reaches ${pm(c_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ accuracy and "
            f"${pm(c_event, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', scale=100, digits=1)}\\%$ worst-class "
            f"accuracy at ${pm(c_event, 'unicast_hybrid_total_bits_mean', 'unicast_hybrid_total_bits_std', scale=1e-9, digits=2)}$ Gbit",
        ),
        (
            "compact-CNN dense endpoint",
            f"Dense FedAvg reaches ${pm(c_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ and "
            f"${pm(c_dense, 'final_worst_class_accuracy_mean', 'final_worst_class_accuracy_std', scale=100, digits=1)}\\%$ at "
            f"${number(c_dense, 'unicast_hybrid_total_bits_mean') / 1e9:.2f}$ Gbit",
        ),
        (
            "compact-CNN paired dense comparison",
            f"paired accuracy advantage is ${number(c_dense_difference, 'mean_difference_points'):.2f}$ points "
            f"with a 95\\% interval of $[{number(c_dense_difference, 'ci95_low_points'):.2f},"
            f"{number(c_dense_difference, 'ci95_high_points'):.2f}]$ points",
        ),
        (
            "compact-CNN dense traffic fraction",
            f"uses ${c_traffic_fraction:.1f}\\%$ of dense traffic",
        ),
        (
            "compact-CNN EF-TopK comparison",
            f"its ${number(c_topk_difference, 'mean_difference_points'):.2f}$-point advantage has interval "
            f"$[{number(c_topk_difference, 'ci95_low_points'):.2f},"
            f"{number(c_topk_difference, 'ci95_high_points'):.2f}]$ while using "
            f"${100 * traffic_mbit(c_event) / traffic_mbit(c_topk):.1f}\\%$ of the traffic",
        ),
        (
            "compact-CNN convergence plateau",
            f"already ${c_accuracy_1200:.2f}\\%$ at round 1,200 and remains "
            f"${c_accuracy_1800:.2f}\\%$ at round 1,800",
        ),
        (
            "compact-CNN Strom variability",
            f"large ${100 * number(c_near_strom, 'final_test_accuracy_std'):.1f}$-point standard deviation",
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

    claims.extend(
        [
            (
                "P8 leakage-audit accuracy",
                f"${pm(p8_frozen, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ for $\\rho=0.999$ and "
                f"${pm(p8_no_leak, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$ for $\\rho=1$",
            ),
            (
                "P8 leakage-audit traffic",
                f"at ${traffic_mbit(p8_frozen):.1f}$ and ${traffic_mbit(p8_no_leak):.1f}$ Mbit",
            ),
            (
                "ResNet Event accuracy",
                f"${pm(resnet_event, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$",
            ),
            (
                "ResNet dense accuracy",
                f"${pm(resnet_dense, 'final_test_accuracy_mean', 'final_test_accuracy_std', scale=100, digits=2)}\\%$",
            ),
            (
                "ResNet paired accuracy difference",
                f"${number(resnet_difference, 'mean_accuracy_difference_points'):.2f}$ points",
            ),
            (
                "ResNet paired accuracy interval",
                f"$[{number(resnet_difference, 'ci95_low_points'):.2f},{number(resnet_difference, 'ci95_high_points'):.2f}]$ points",
            ),
        ]
    )

    for label, config_name in (
        ("Event-FedAvg", "event_t0125_q005"),
        ("EF-TopK", "ef_k025"),
        ("dense FedAvg", "dense_g15"),
    ):
        claims.append(
            (
                f"ResNet trajectory traffic for {label}",
                f"{mean_traffic_to_accuracy(config_name, 65):.2f} Gbit",
            )
        )

    p15_by_family = {row["family"]: row for row in p15_summary}
    p15_by_comparison = {row["comparison"]: row for row in p15_paired}
    memoryless = p15_by_comparison["memoryless_minus_frozen"]
    subtractive = p15_by_comparison["subtractive_minus_frozen"]
    for family, label in (
        ("frozen", "P15 frozen accuracy"),
        ("memoryless", "P15 memoryless accuracy"),
        ("subtractive", "P15 subtractive accuracy"),
    ):
        row = p15_by_family[family]
        claims.append(
            (
                label,
                f"{100 * number(row, 'final_test_accuracy_mean'):.2f}"
                f"$\\pm${100 * number(row, 'final_test_accuracy_std'):.2f}",
            )
        )
    claims.extend(
        [
            (
                "P15 memoryless paired difference",
                f"${abs(number(memoryless, 'mean_accuracy_difference_points')):.2f}$ points",
            ),
            (
                "P15 memoryless paired interval",
                f"$[{number(memoryless, 'ci95_low_points'):.2f},{number(memoryless, 'ci95_high_points'):.2f}]$ points",
            ),
            (
                "P15 subtractive paired difference",
                f"$+{number(subtractive, 'mean_accuracy_difference_points'):.2f}$ points",
            ),
            (
                "P15 subtractive paired interval",
                f"$[{number(subtractive, 'ci95_low_points'):.2f},{number(subtractive, 'ci95_high_points'):.2f}]$ points",
            ),
        ]
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

    for index, row in enumerate(factorial):
        claims.extend(
            [
                (
                    f"P11 kappa-condition fraction cell {index}",
                    f"${pm(row, 'kappa_condition_fraction_mean', 'kappa_condition_fraction_std', scale=100, digits=1)}\\%$",
                ),
                (
                    f"P11 defect contribution cell {index}",
                    f"${pm(row, 'sampled_defect_contribution_mean', 'sampled_defect_contribution_std', scale=1, digits=3)}$",
                ),
                (
                    f"P11 event-curvature factor cell {index}",
                    f"${pm(row, 'sampled_event_curvature_factor_mean', 'sampled_event_curvature_factor_std', scale=1, digits=1)}$",
                ),
            ]
        )

    direct_prose_labels = {
        "abstract ResNet Event point",
        "abstract ResNet dense point",
        "introduction ResNet traffic fold",
        "compact-CNN Event endpoint",
        "compact-CNN dense endpoint",
        "compact-CNN paired dense comparison",
        "compact-CNN dense traffic fraction",
        "compact-CNN EF-TopK comparison",
        "compact-CNN convergence plateau",
        "compact-CNN Strom variability",
        "ResNet Event accuracy",
        "ResNet dense accuracy",
        "ResNet paired accuracy difference",
        "ResNet paired accuracy interval",
        "P11 audited snapshot count",
        "broadcast-accounting sensitivity",
        "unicast-accounting sensitivity",
        "P8 leakage-audit accuracy",
        "P8 leakage-audit traffic",
        "P15 frozen accuracy",
        "P15 memoryless accuracy",
        "P15 subtractive accuracy",
        "P15 memoryless paired difference",
        "P15 memoryless paired interval",
        "P15 subtractive paired difference",
        "P15 subtractive paired interval",
    }
    checked = [(label, fragment) for label, fragment in claims if label in direct_prose_labels]
    for label, fragment in checked:
        require(manuscript, label, fragment)
    print(f"validated {len(checked)} empirical manuscript claims against frozen CSVs")


if __name__ == "__main__":
    main()
