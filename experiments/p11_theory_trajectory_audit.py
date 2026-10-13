"""P11 theory-to-trajectory audit from saved checkpoints; no training.

Run:
  python experiments/p11_theory_trajectory_audit.py
  python experiments/p11_theory_trajectory_audit.py --check-only

Outputs in experiments/results/p11_alignment_factorial/theory_trajectory/:
checkpoint_diagnostics.csv, seed_metrics.csv, condition_summary.csv,
paired_contrasts.csv, audit_report.md, theory_trajectory.pdf/.png.
These are sampled checkpoint diagnostics, NOT a theorem certificate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "results" / "p11_alignment_factorial"
REQUIRED = {
    "round", "jump", "objective_change", "net_alignment",
    "gradient_sq_norm", "coordinate_events", "curvature_remainder",
    "regime", "local_steps", "partition_seed", "train_seed",
}
CONDITIONS = (("iid", 1), ("iid", 5), ("strong", 1), ("strong", 5))
M = 10
KAPPA_REF = 8.0
T_CRIT = 2.2621571627409915  # 95%, df=9, only used for ten seeds


def read_validate(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    df = df.copy()
    for col in ("round", "jump", "objective_change", "net_alignment",
                "gradient_sq_norm", "coordinate_events", "curvature_remainder",
                "local_steps", "partition_seed", "train_seed"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {col}")
    if (df["jump"] <= 0).any() or (df["coordinate_events"] < 0).any():
        raise ValueError("Invalid quantum or event count")
    df["regime"] = df["regime"].astype(str).str.lower()
    if set(zip(df.regime, df.local_steps.astype(int))) != set(CONDITIONS):
        raise ValueError("Unexpected or missing factorial conditions")
    keys = ["regime", "local_steps", "partition_seed", "round"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate condition/seed/round rows")
    for condition, sub in df.groupby(["regime", "local_steps"]):
        if sub.partition_seed.nunique() != 10:
            raise ValueError(f"{condition}: expected ten distinct seeds")
        rounds = {tuple(sorted(x["round"].astype(int))) for _, x in
                  sub.groupby("partition_seed")}
        if len(rounds) != 1:
            raise ValueError(f"{condition}: unequal audited-round schedules")
    seeds = {
        (regime, steps): set(sub.partition_seed.astype(int))
        for (regime, steps), sub in df.groupby(["regime", "local_steps"])
    }
    if any(v != seeds[CONDITIONS[0]] for v in seeds.values()):
        raise ValueError("Factorial cells do not share partition seeds")
    return df


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    q, align, n = (x["jump"], x["net_alignment"], x["coordinate_events"])
    x["first_order_prediction"] = -q * align
    x["remainder_computed"] = x["objective_change"] + q * align
    x["event_scale"] = q ** 2 * M * n
    x["positive_alignment"] = align > 0
    x["objective_decrease"] = x["objective_change"] < 0
    x["positive_alignment_non_descent"] = x.positive_alignment & ~x.objective_decrease
    x["positive_remainder"] = np.maximum(x.remainder_computed, 0)
    x["reference_defect"] = np.maximum(
        KAPPA_REF * x.gradient_sq_norm - align, 0)
    x["reference_defect_q"] = q * x.reference_defect
    x["curvature_scale_q"] = q ** 2 * n * M
    # The following is a pointwise empirical required L, not a global estimate.
    x["local_L_requirement"] = np.where(
        x.event_scale > 0, 2 * x.positive_remainder / x.event_scale, np.nan)
    residual = np.abs(x.remainder_computed - x.curvature_remainder)
    if residual.max() > 1e-5:
        raise ValueError(f"Stored remainder mismatch: max {residual.max():.3g}")
    return x


def seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (regime, steps, seed), group in df.groupby(
            ["regime", "local_steps", "partition_seed"], sort=True):
        qsum = float(group.jump.sum())
        ref = float(group.reference_defect_q.sum())
        curve = float(group.curvature_scale_q.sum())
        pearson = group[["event_scale", "remainder_computed"]].corr().iloc[0, 1]
        result.append(dict(
            regime=regime, local_steps=int(steps), partition_seed=int(seed),
            n_checkpoints=len(group),
            alignment_positive_pct=100 * group.positive_alignment.mean(),
            objective_decrease_pct=100 * group.objective_decrease.mean(),
            aligned_non_descent_pct=100 * group.positive_alignment_non_descent.mean(),
            conditional_non_descent_pct=100 * (
                group.positive_alignment_non_descent.sum()
                / max(1, group.positive_alignment.sum())),
            normalized_sample_defect=ref / (KAPPA_REF * qsum),
            normalized_sample_event_scale=curve / qsum,
            mean_curvature_remainder=float(group.remainder_computed.mean()),
            positive_remainder_pct=100 * (group.remainder_computed > 0).mean(),
            event_remainder_pearson=float(pearson) if np.isfinite(pearson) else np.nan,
            positive_defect_pct=100 * (group.reference_defect > 0).mean(),
        ))
    return pd.DataFrame(result)


def aggregate(seeds: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in seeds.columns
               if c not in ("regime", "local_steps", "partition_seed", "n_checkpoints")]
    rows = []
    for (regime, steps), part in seeds.groupby(["regime", "local_steps"]):
        row = {"regime": regime, "local_steps": int(steps), "n_seeds": len(part)}
        for m in metrics:
            vals = part[m].dropna().to_numpy(dtype=float)
            row[m + "_mean"] = vals.mean() if len(vals) else np.nan
            row[m + "_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("alignment_positive_pct", "objective_decrease_pct",
                   "aligned_non_descent_pct", "normalized_sample_defect",
                   "normalized_sample_event_scale", "mean_curvature_remainder"):
        indexed = seeds.set_index(["regime", "local_steps", "partition_seed"])
        for name, left, right in (
            ("strong_minus_iid_E1", ("strong", 1), ("iid", 1)),
            ("strong_minus_iid_E5", ("strong", 5), ("iid", 5)),
            ("E5_minus_E1_iid", ("iid", 5), ("iid", 1)),
            ("E5_minus_E1_strong", ("strong", 5), ("strong", 1)),
        ):
            l = indexed.loc[left, metric].sort_index()
            r = indexed.loc[right, metric].sort_index()
            diff = (l - r).to_numpy(dtype=float)
            mean, sd = diff.mean(), diff.std(ddof=1)
            half = T_CRIT * sd / np.sqrt(len(diff))
            rows.append(dict(comparison=name, metric=metric, n=len(diff),
                             mean_difference=mean, sd_difference=sd,
                             ci95_low=mean-half, ci95_high=mean+half))
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, agg: pd.DataFrame, destination: Path) -> None:
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.0))
    combos = list(CONDITIONS)
    for regime, steps in combos:
        g = df[(df.regime == regime) & (df.local_steps == steps)]
        label = f"{'IID' if regime == 'iid' else 'Strong'}; E={steps}"
        axes[0].scatter(g.event_scale, g.remainder_computed,
                        s=10, alpha=0.35, label=label)
    axes[0].axhline(0, color="0.5", linewidth=0.7)
    axes[0].set_xlabel(r"$M q_r^2 N_r$")
    axes[0].set_ylabel(r"$\Delta F_r+q_r A_r$")
    axes[0].set_title("(a) Realized curvature remainder")
    axes[0].legend(fontsize=6, frameon=False)
    xs = np.arange(4)
    labels = ["IID/1", "IID/5", "Strong/1", "Strong/5"]
    ordered = agg.set_index(["regime", "local_steps"]).loc[combos]
    for ax, field, title, ylabel in [
        (axes[1], "aligned_non_descent_pct", "(b) Alignment without descent", "Audited rounds (%)"),
        (axes[2], "normalized_sample_defect", "(c) Sampled reference defect", "Sampled normalized defect"),
    ]:
        y = ordered[field + "_mean"].to_numpy()
        err = ordered[field + "_sd"].to_numpy()
        ax.errorbar(xs, y, yerr=err, fmt="o", capsize=3, linewidth=1)
        ax.set_xticks(xs, labels, rotation=25)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "theory_trajectory.pdf")
    fig.savefig(destination / "theory_trajectory.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "combined_rounds.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "theory_trajectory")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    x = diagnostics(read_validate(args.source))
    if args.check_only:
        print(f"Verified {len(x)} audited checkpoints, four conditions, ten seeds each")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = seed_metrics(x)
    agg = aggregate(seeds)
    contrasts = paired(seeds)
    x.to_csv(args.output / "checkpoint_diagnostics.csv", index=False)
    seeds.to_csv(args.output / "seed_metrics.csv", index=False)
    agg.to_csv(args.output / "condition_summary.csv", index=False)
    contrasts.to_csv(args.output / "paired_contrasts.csv", index=False)
    make_figure(x, agg, args.output)
    report = [
        "# P11 Theory-to-Trajectory Audit", "",
        "These are **31 selected audit checkpoints per seed**, not all training rounds.",
        "The normalized sums below are **sampled diagnostics**, not full-horizon theorem terms.",
        "No conditional expectation or global smoothness constant is certified.", "",
        "## Condition-level mean metrics", "",
        agg.to_markdown(index=False, floatfmt=".4g"), "",
        "## Paired seed contrasts (95% Student-t intervals)", "",
        contrasts.to_markdown(index=False, floatfmt=".4g"), "",
        "Interpretation: a positive-alignment round can increase the objective;",
        "the realized curvature remainder is an exact algebraic difference.",
        "Its relation to $M q_r^2 N_r$ is descriptive, not an identified",
        "smoothness bound. Seed-paired intervals describe the audited seeds",
        "and should not be interpreted as independent-checkpoint inference.",
    ]
    (args.output / "audit_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(agg[["regime", "local_steps", "alignment_positive_pct_mean",
               "objective_decrease_pct_mean", "normalized_sample_defect_mean",
               "normalized_sample_event_scale_mean"]].to_string(index=False))
    print(f"Outputs saved in {args.output}")


if __name__ == "__main__":
    main()
