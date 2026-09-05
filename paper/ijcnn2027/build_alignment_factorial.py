"""Validate P11 alignment-factorial results and build the paper table."""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import statistics


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper" / "ijcnn2027"
ROOT = REPO / "experiments" / "results" / "p11_alignment_factorial"
SUMMARY = ROOT / "combined_summary.csv"
ROUNDS = ROOT / "combined_rounds.csv"
AGGREGATES = ROOT / "aggregate_metrics.csv"
PAIRED = ROOT / "paired_effects.csv"
EVIDENCE = PAPER / "evidence" / "p11_alignment_factorial.csv"
TABLE = PAPER / "generated" / "p11_alignment_table.tex"
REGIMES = ("iid", "strong")
LOCAL_STEPS = (1, 5)
SEEDS = tuple(range(2500, 3500, 100))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError(f"empty P11 artifact: {path.relative_to(REPO)}")
    return rows


def cell_key(row: dict[str, str]) -> tuple[str, int]:
    return row["regime"], int(row["local_steps"])


def run_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["regime"], int(row["local_steps"]), int(row["partition_seed"])


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def validate_design(
    summaries: list[dict[str, str]], rounds: list[dict[str, str]]
) -> None:
    expected = {
        (regime, steps, seed)
        for regime in REGIMES
        for steps in LOCAL_STEPS
        for seed in SEEDS
    }
    expected_runs = len(REGIMES) * len(LOCAL_STEPS) * len(SEEDS)
    if {run_key(row) for row in summaries} != expected or len(summaries) != expected_runs:
        raise AssertionError(f"P11 does not contain the complete {expected_runs}-run design")
    counts: dict[tuple[str, int, int], int] = {key: 0 for key in expected}
    for row in rounds:
        key = run_key(row)
        if key not in counts:
            raise AssertionError(f"unexpected P11 round key: {key}")
        counts[key] += 1
        if float(row["alignment_identity_error"]) >= 1e-4:
            raise AssertionError("P11 alignment identity failed")
        if float(row["decomposition_identity_error"]) >= 1e-4:
            raise AssertionError("P11 decomposition identity failed")
        if float(row["update_reconstruction_error"]) >= 1e-6:
            raise AssertionError("P11 update reconstruction failed")
        if float(row["defect_bound_slack"]) < -1e-5:
            raise AssertionError("P11 defect upper bound failed")
        if float(row["memory_opposition_penalty"]) != 0.0:
            raise AssertionError("P11 unexpectedly contains memory-opposed pulses")
    expected_snapshots = expected_runs * 31
    if len(rounds) != expected_snapshots or set(counts.values()) != {31}:
        raise AssertionError(
            f"P11 must contain 31 snapshots for each of {expected_runs} runs"
        )

    indexed = {run_key(row): row for row in summaries}
    for steps in LOCAL_STEPS:
        for seed in SEEDS:
            iid = indexed[("iid", steps, seed)]
            strong = indexed[("strong", steps, seed)]
            if not float(strong["heterogeneity_ratio"]) < float(
                iid["heterogeneity_ratio"]
            ):
                raise AssertionError("strong non-IID did not make B more adverse")
    for seed in SEEDS:
        iid_e1 = indexed[("iid", 1, seed)]
        iid_e5 = indexed[("iid", 5, seed)]
        strong_e1 = indexed[("strong", 1, seed)]
        strong_e5 = indexed[("strong", 5, seed)]
        if min(
            float(row["weighted_alignment_ratio"])
            for row in (iid_e1, iid_e5, strong_e1, strong_e5)
        ) <= 0.0:
            raise AssertionError("an individual P11 trajectory has nonpositive alignment")
        if not float(iid_e5["local_drift_ratio"]) < float(
            iid_e1["local_drift_ratio"]
        ):
            raise AssertionError("IID local-depth effect changed direction")
        if not float(strong_e5["local_drift_ratio"]) > float(
            strong_e1["local_drift_ratio"]
        ):
            raise AssertionError("non-IID local-depth effect changed direction")
        if not float(strong_e5["objective_decrease_fraction"]) < float(
            strong_e1["objective_decrease_fraction"]
        ):
            raise AssertionError("non-IID descent-depth effect changed direction")
        if not float(strong_e5["mean_curvature_remainder"]) > float(
            strong_e1["mean_curvature_remainder"]
        ):
            raise AssertionError("non-IID curvature-depth effect changed direction")


def build_cells(summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = (
        "weighted_alignment_ratio",
        "positive_alignment_fraction",
        "objective_decrease_fraction",
        "positive_without_descent_fraction",
        "harmful_mass_share",
        "ideal_local_ratio",
        "local_drift_ratio",
        "heterogeneity_ratio",
        "mean_curvature_remainder",
        "mean_coordinate_events_per_audit",
        "mean_cancellation_fraction",
        "final_test_accuracy",
        "unicast_hybrid_total_bits",
    )
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {
        (regime, steps): [] for regime in REGIMES for steps in LOCAL_STEPS
    }
    for row in summaries:
        grouped[cell_key(row)].append(row)
    cells: list[dict[str, str]] = []
    for regime in REGIMES:
        for steps in LOCAL_STEPS:
            group = grouped[(regime, steps)]
            if len(group) != len(SEEDS):
                raise AssertionError(f"each P11 cell requires {len(SEEDS)} seeds")
            output = {
                "regime": regime,
                "local_steps": str(steps),
                "n_seeds": str(len(SEEDS)),
                "partition_seeds": ";".join(str(seed) for seed in SEEDS),
                "snapshots_per_seed": "31",
            }
            for metric in metrics:
                mean, std = mean_std([float(row[metric]) for row in group])
                output[f"{metric}_mean"] = f"{mean:.17g}"
                output[f"{metric}_std"] = f"{std:.17g}"
            cells.append(output)
    if min(float(row["weighted_alignment_ratio_mean"]) for row in cells) <= 0.0:
        raise AssertionError("P11 mean aggregate alignment is not positive in every cell")
    return cells


def validate_aggregates(cells: list[dict[str, str]]) -> None:
    rows = read_rows(AGGREGATES)
    lookup = {
        (row["regime"], int(row["local_steps"]), row["metric"]): row
        for row in rows
    }
    for cell in cells:
        for metric in (
            "weighted_alignment_ratio",
            "positive_alignment_fraction",
            "objective_decrease_fraction",
            "local_drift_ratio",
            "heterogeneity_ratio",
        ):
            source = lookup[(cell["regime"], int(cell["local_steps"]), metric)]
            observed = float(cell[f"{metric}_mean"])
            if not math.isclose(observed, float(source["mean"]), rel_tol=1e-12):
                raise AssertionError(f"P11 aggregate drift for {cell_key(cell)} {metric}")


def validate_paired_effects() -> None:
    rows = read_rows(PAIRED)
    lookup = {
        (row["factor"], row["from"], row["to"], row["metric"]): row
        for row in rows
    }
    directional_contract = (
        ("heterogeneity", "iid_e1", "strong_e1", "heterogeneity_ratio", "negative_pairs"),
        ("heterogeneity", "iid_e5", "strong_e5", "heterogeneity_ratio", "negative_pairs"),
        ("local_steps", "iid_e1", "iid_e5", "local_drift_ratio", "negative_pairs"),
        ("local_steps", "strong_e1", "strong_e5", "local_drift_ratio", "positive_pairs"),
        ("local_steps", "strong_e1", "strong_e5", "objective_decrease_fraction", "negative_pairs"),
        ("local_steps", "strong_e1", "strong_e5", "mean_curvature_remainder", "positive_pairs"),
    )
    for factor, source, target, metric, direction in directional_contract:
        row = lookup[(factor, source, target, metric)]
        if int(row["n_pairs"]) != len(SEEDS) or int(row[direction]) != len(SEEDS):
            raise AssertionError(
                f"P11 paired direction failed for {source}->{target} {metric}"
            )
        low = float(row["paired_t95_low"])
        high = float(row["paired_t95_high"])
        if low <= 0.0 <= high:
            raise AssertionError(
                f"P11 paired interval crosses zero for {source}->{target} {metric}"
            )


def evidence_csv(cells: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    columns = list(cells[0])
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(cells)
    return stream.getvalue()


def pm(cell: dict[str, str], metric: str, scale: float, digits: int) -> str:
    mean = scale * float(cell[f"{metric}_mean"])
    std = scale * float(cell[f"{metric}_std"])
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def table_tex(cells: list[dict[str, str]]) -> str:
    labels = {"iid": "IID", "strong": "Strong non-IID"}
    lines = [
        "% Generated by paper/ijcnn2027/build_alignment_factorial.py; do not edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Finite-trajectory alignment factorial. Values are mean $\\pm$ sample standard deviation over ten seeds and 31 independently replayed snapshots per seed.}",
        "\\label{tab:alignment-factorial}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Partition & $E$ & $\\widehat{\\kappa}_{\\mathcal A}$ & $A_r>0$ [\\%] & $\\Delta F_r<0$ [\\%] \\\\",
        "\\midrule",
    ]
    for cell in cells:
        lines.append(
            f"{labels[cell['regime']]} & {cell['local_steps']} & "
            f"{pm(cell, 'weighted_alignment_ratio', 1.0, 2)} & "
            f"{pm(cell, 'positive_alignment_fraction', 100.0, 1)} & "
            f"{pm(cell, 'objective_decrease_fraction', 100.0, 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def check_or_write(path: Path, expected: str, check: bool) -> None:
    if check:
        observed = path.read_text(encoding="utf-8")
        if observed != expected:
            raise AssertionError(f"stale generated P11 product: {path.relative_to(REPO)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summaries = read_rows(SUMMARY)
    rounds = read_rows(ROUNDS)
    validate_design(summaries, rounds)
    cells = build_cells(summaries)
    validate_aggregates(cells)
    validate_paired_effects()
    check_or_write(EVIDENCE, evidence_csv(cells), args.check)
    check_or_write(TABLE, table_tex(cells), args.check)
    print(
        f"validated P11: {len(REGIMES) * len(LOCAL_STEPS) * len(SEEDS)} runs, "
        f"{len(REGIMES) * len(LOCAL_STEPS) * len(SEEDS) * 31} snapshots, "
        "4 paper cells"
    )


if __name__ == "__main__":
    main()
