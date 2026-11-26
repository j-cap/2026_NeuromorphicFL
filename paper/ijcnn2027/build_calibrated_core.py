"""Rebuild the frozen, calibrated CIFAR panels without data or training.

Run from any directory with Python, numpy and matplotlib.  --check validates
all source hashes, recomputes the paired statistics and compares every product
byte for byte.  No experiment result is ever written by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
FREEZE = PAPER / "evidence/calibrated_core_freeze.json"
PANELS = {
    "compact_cnn": ("core_fresh_comparison_v1", "main_runs.csv", "comparison_history.csv", "point", "Compact CNN"),
    "resnet14": ("resnet14_core_confirmation_v1", "confirmation_runs.csv", "confirmation_history.csv", "comparison_id", "ResNet-14"),
}
METHODS = ("event", "strom", "stc", "ef_topk", "dense")
LABELS = {"event": "Event-FedAvg", "strom": "Ström-style", "stc": "STC", "ef_topk": "EF-TopK", "dense": "Dense FedAvg"}
TEX = dict(LABELS, strom=r'Str\"om-style')
COLORS = {"event": "#D55E00", "strom": "#0072B2", "stc": "#CC79A7", "ef_topk": "#009E73", "dense": "#595959"}
MARKERS = {"event": "*", "strom": "s", "stc": "D", "ef_topk": "^", "dense": "o"}
STYLES = {"event": "-", "strom": "--", "stc": ":", "ef_topk": "-.", "dense": (0, (5, 1, 1, 1))}
VARIANTS = {"event": "both_codecs", "strom": "both_codecs", "stc": "lossless_stc", "ef_topk": "lossless_ef_topk", "dense": "dense_float32"}
TARGETS = (0.45, 0.50, 0.55)
METADATA = {"Creator": "Event-FedAvg calibrated core", "CreationDate": None, "ModDate": None}


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_freeze():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, digest in freeze["files"].items():
        data = (REPO / name).read_bytes()
        if Path(name).suffix.lower() in {".py", ".csv", ".json", ".md", ".txt", ".tex", ".bib"}:
            data = data.replace(b"\r\n", b"\n")
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise AssertionError(f"Frozen panel changed: {name}")
    return freeze


def paired(a, b, log=False):
    x = np.log(np.asarray(a) / np.asarray(b)) if log else np.asarray(a) - np.asarray(b)
    n = len(x)
    # Two-sided 95% Student-t critical values; only n=9 or n=10 occurs.
    critical = {9: 2.306004135204166, 10: 2.2621571628540993}[n]
    mean = float(x.mean())
    half = critical * float(x.std(ddof=1)) / np.sqrt(n)
    return tuple(float(np.exp(v)) if log else v for v in (mean, mean-half, mean+half))


def load_panel(key):
    directory, run_name, history_name, field, _ = PANELS[key]
    root = REPO / "experiments/results" / directory
    runs = read(root / run_name)
    history = [r for r in read(root / history_name) if r["variant"] == VARIANTS[r[field]]]
    assert len(runs) == 50 and len(history) == 1550
    indexed = {}
    curves = {}
    hits = {}
    for method in METHODS:
        rows = sorted([r for r in runs if r[field] == method], key=lambda r: int(r["partition_seed"]))
        assert len(rows) == 10 and all(int(r["rounds"]) == 1800 for r in rows)
        indexed[method] = rows
        seeds = [int(r["partition_seed"]) for r in rows]
        assert seeds == list(range(6000, 7000, 100) if key == "compact_cnn" else range(7100, 8100, 100))
        for row in rows:
            assert row["variant"] == VARIANTS[method]
            assert abs(float(row["uplink_bits"]) + float(row["unicast_downlink_bits"]) - float(row["unicast_total_bits"])) < 1
        per_seed = []
        for seed in seeds:
            curve = sorted([r for r in history if r[field] == method and int(r["partition_seed"]) == seed], key=lambda r: int(r["round"]))
            assert [int(r["round"]) for r in curve] == [1] + list(range(60, 1801, 60))
            assert curve[-1]["model_sha256"] == rows[seeds.index(seed)]["model_sha256"]
            assert np.all(np.diff([float(r["unicast_total_bits"]) for r in curve]) >= 0)
            per_seed.append(curve)
            for target in TARGETS:
                hits[method, seed, target] = next((r for r in curve if float(r["test_accuracy"]) >= target), None)
        curves[method] = per_seed
    return indexed, curves, hits


def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    # Twelve significant digits retain far more precision than any paper
    # statistic while avoiding irrelevant NumPy reduction last-bit differences.
    writer.writerows({key: format(value, ".12g") if isinstance(value, float) else value
                      for key, value in row.items()} for row in rows)
    return out.getvalue().encode()


def pdf_bytes(fig):
    out = io.BytesIO()
    fig.savefig(out, format="pdf", metadata=METADATA, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return out.getvalue()


def products():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    endpoints, effects, target_rows, history_rows = [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=False)
    ratios, ratio_axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, ratio_ax, (panel, (_, _, _, _, title)) in zip(axes, ratio_axes, PANELS.items()):
        runs, curves, hits = load_panel(panel)
        for method in METHODS:
            rows = runs[method]
            e = {"panel": panel, "method": method, "n": len(rows)}
            for name in ("final_test_accuracy", "final_test_ce", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"):
                x = np.array([float(r[name]) for r in rows])
                e[name + "_mean"] = float(x.mean())
                e[name + "_std"] = float(x.std(ddof=1))
            endpoints.append(e)
            seed_curves = curves[method]
            x = np.array([[float(r["unicast_total_bits"]) / 1e9 for r in c] for c in seed_curves])
            y = np.array([[100*float(r["test_accuracy"]) for r in c] for c in seed_curves])
            ax.plot(x.mean(axis=0), y.mean(axis=0), color=COLORS[method], ls=STYLES[method],
                    lw=2.0 if method == "event" else 1.1, marker=MARKERS[method], markevery=5,
                    ms=5 if method == "event" else 3, zorder=6 if method == "event" else 3)
            # Shading describes variation among runs, not a confidence band.
            ax.fill_between(x.mean(axis=0), y.mean(axis=0)-y.std(axis=0, ddof=1), y.mean(axis=0)+y.std(axis=0, ddof=1), color=COLORS[method], alpha=.08)
            for j, round_number in enumerate([1] + list(range(60, 1801, 60))):
                history_rows.append({"panel": panel, "method": method, "round": round_number,
                                     "mean_unicast_Gbit": float(x[:, j].mean()), "mean_test_accuracy_percent": float(y[:, j].mean()),
                                     "sd_test_accuracy_percent": float(y[:, j].std(ddof=1))})
            if method == "event":
                continue
            for metric, source, scale, log in (("accuracy_difference_pp", "final_test_accuracy", 100, False),
                                               ("ce_difference", "final_test_ce", 1, False),
                                               ("unicast_ratio", "unicast_total_bits", 1, True)):
                a = [scale*float(r[source]) for r in runs["event"]]
                b = [scale*float(r[source]) for r in rows]
                estimate, low, high = paired(a, b, log)
                if metric == "unicast_ratio":
                    assert all(x < y for x, y in zip(a, b)), (panel, method, "final cost win")
                effects.append(dict(panel=panel, reference=method, metric=metric, n=10, estimate=estimate, ci95_low=low, ci95_high=high))
            for ti, target in enumerate(TARGETS):
                pairs = [(hits["event", int(r["partition_seed"]), target], hits[method, int(r["partition_seed"]), target]) for r in rows]
                common = [(a, b) for a, b in pairs if a is not None and b is not None]
                estimate, low, high = paired([float(a["unicast_total_bits"]) for a, b in common], [float(b["unicast_total_bits"]) for a, b in common], True)
                target_rows.append(dict(panel=panel, reference=method, target_percent=int(round(100*target)),
                    event_reached=sum(a is not None for a, b in pairs), reference_reached=sum(b is not None for a, b in pairs),
                    common_hits=len(common), cost_wins=sum(float(a["unicast_total_bits"]) < float(b["unicast_total_bits"]) for a, b in common), ratio=estimate, ratio_ci95_low=low, ratio_ci95_high=high,
                    saving_percent=100*(1-estimate), saving_ci95_low=100*(1-high), saving_ci95_high=100*(1-low),
                    earlier=sum(int(a["round"]) < int(b["round"]) for a, b in common),
                    same=sum(int(a["round"]) == int(b["round"]) for a, b in common),
                    later=sum(int(a["round"]) > int(b["round"]) for a, b in common)))
                pos = list(METHODS[1:]).index(method) + (ti-1)*.19
                ratio_ax.errorbar(pos, estimate, yerr=[[estimate-low], [high-estimate]], marker=("o", "s", "^")[ti],
                                  ms=3.5, capsize=2, color=("#0072B2", "#D55E00", "#009E73")[ti], lw=1)
        ax.set(title=title, xlabel="Cumulative total unicast [Gbit]", ylabel="Test accuracy [%]")
        ax.set_xscale("log")
        ax.set_ylim((9, 62) if panel == "compact_cnn" else (9, 81))
        ax.grid(alpha=.2, axis="both")
        ratio_ax.set(title=title, ylabel="Event / reference traffic", ylim=(.006, 1.4), yscale="log")
        ratio_ax.set_xticks(range(4), [LABELS[m] for m in METHODS[1:]], rotation=15, ha="right")
        ratio_ax.axhline(1, ls="--", color=".4", lw=.8)
        ratio_ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], marker=MARKERS[m], label=LABELS[m], ms=5) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=1)
    ratios.legend(handles=[Line2D([], [], color=c, marker=m, label=f"{t}% accuracy") for t, c, m in zip((45, 50, 55), ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^"))], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
    ratios.tight_layout(rect=(0, 0, 1, .91), w_pad=1)
    output = {
        "figures/calibrated_core_trajectories.pdf": pdf_bytes(fig),
        "figures/calibrated_core_target_ratios.pdf": pdf_bytes(ratios),
        "evidence/calibrated_core_endpoints.csv": csv_bytes(endpoints),
        "evidence/calibrated_core_endpoint_effects.csv": csv_bytes(effects),
        "evidence/calibrated_core_targets.csv": csv_bytes(target_rows),
        "evidence/calibrated_core_history.csv": csv_bytes(history_rows),
    }
    # Independently reconstructed summaries must match the original study CSVs.
    for panel, (directory, _, _, field, _) in PANELS.items():
        source = read(REPO / "experiments/results" / directory / "endpoint_summary.csv")
        for row in source:
            item = next(x for x in endpoints if x["panel"] == panel and x["method"] == row[field])
            for metric in ("final_test_accuracy", "final_test_ce", "unicast_total_bits"):
                for statistic in ("mean", "std"):
                    key = metric + "_" + statistic
                    assert np.isclose(item[key], float(row[key]), rtol=1e-11, atol=1e-9), (panel, key)
        original = read(REPO / "experiments/results" / directory / "paired_target_summary.csv")
        for row in original:
            item = next(x for x in target_rows if x["panel"] == panel and x["reference"] == row["reference"] and abs(x["target_percent"]-100*float(row["target_accuracy"])) < 1e-8)
            assert item["common_hits"] == int(row["n_common_hits"])
            for new, old in (("ratio", "estimate"), ("ratio_ci95_low", "ci95_low"), ("ratio_ci95_high", "ci95_high")):
                assert np.isclose(item[new], float(row[old]), rtol=1e-9, atol=1e-11), (panel, new)
    output["generated/calibrated_core_results.tex"] = result_table(endpoints).encode()
    output["generated/calibrated_core_targets.tex"] = target_table(target_rows).encode()
    output["figures/calibrated_event_protocol.pdf"] = method_figure()
    return output


def result_table(rows):
    s = [r"% Generated by build_calibrated_core.py; do not edit.", r"\begin{table*}[t]", r"\centering\coreColor\footnotesize",
         r"\caption{Frozen CIFAR-10 panels at 1,800 rounds. Accuracy and test cross-entropy (CE): mean $\pm$ sample SD over ten paired seeds. Traffic: mean Gbit ($10^9$ bits), including all recipient copies. Each method uses architecture-specific development-selected settings.}",
         r"\label{tab:calibrated-core}", r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{llrrrrr}\toprule",
         r"Model & Method & Accuracy [\%] & Test CE & Uplink & Downlink & Total \\\midrule"]
    for panel, (_, _, _, _, title) in PANELS.items():
        selected = [x for x in rows if x["panel"] == panel]
        best_acc = max(x["final_test_accuracy_mean"] for x in selected)
        best_ce = min(x["final_test_ce_mean"] for x in selected)
        for j, row in enumerate(selected):
            method = row["method"]
            acc = f"{100*row['final_test_accuracy_mean']:.2f}\\pm{100*row['final_test_accuracy_std']:.2f}"
            ce = f"{row['final_test_ce_mean']:.3f}\\pm{row['final_test_ce_std']:.3f}"
            if row["final_test_accuracy_mean"] == best_acc: acc = r"\mathbf{"+acc+"}"
            if row["final_test_ce_mean"] == best_ce: ce = r"\mathbf{"+ce+"}"
            total = f"{row['unicast_total_bits_mean']/1e9:.3f}"
            if method == "event": total = r"\mathbf{"+total+"}"
            s.append(f"{title if j == 0 else ''} & {TEX[method]} & ${acc}$ & ${ce}$ & {row['uplink_bits_mean']/1e9:.3f} & {row['unicast_downlink_bits_mean']/1e9:.3f} & ${total}$" + r" \\")
        s.append(r"\midrule")
    s[-1] = r"\bottomrule"
    s += [r"\end{tabular}", r"\par\smallskip\parbox{.97\textwidth}{\footnotesize Bold identifies the largest mean accuracy, smallest mean CE, and smallest total traffic within each panel; it does not imply statistical superiority. STC retains all coordinates ($p=1$) and ternarizes them, with gain 2 (compact CNN) or 4 (ResNet-14). EF-TopK retains 5\% or 1\%, respectively.}", r"\end{table*}"]
    return "\n".join(s)+"\n"


def target_table(rows):
    s = [r"% Generated by build_calibrated_core.py; do not edit.", r"\begin{table}[t]\centering\coreColor\footnotesize",
         r"\caption{Event's unicast saving to 50\% test accuracy. All methods reach this target in 10/10 runs. Savings and descriptive 95\% intervals use paired log cost ratios.}",
         r"\label{tab:calibrated-targets}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llr}\toprule",
         r"Model & Reference & Saving [\%] (95\% CI) \\\midrule"]
    for panel, (_, _, _, _, title) in PANELS.items():
        selected = [x for x in rows if x["panel"] == panel and x["target_percent"] == 50]
        for j, row in enumerate(selected):
            s.append(f"{title if j == 0 else ''} & {TEX[row['reference']]} & {row['saving_percent']:.1f} [{row['saving_ci95_low']:.1f}, {row['saving_ci95_high']:.1f}]" + r" \\")
        s.append(r"\midrule")
    s[-1] = r"\bottomrule"
    s += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(s)+"\n"


def method_figure():
    fig, ax = plt.subplots(figsize=(7, 1.65))
    ax.set(xlim=(0, 10), ylim=(0, 2)); ax.axis("off")
    boxes = [(0, .65, 2.15, "Client local SGD\nweighted delta / learning rate"),
             (2.65, .65, 2.15, "Leaky evidence state\nthreshold + full reset"),
             (5.3, .65, 2.15, "Server pulse sum\nindependent quantum $q_r$"),
             (7.95, .65, 2.05, "Exact client synchronization\naggregate counts + patch")]
    for x, y, w, text in boxes:
        ax.add_patch(FancyBboxPatch((x+.04, y), w-.08, .75, boxstyle="round,pad=.025", facecolor="#fff2e8" if x == 2.65 else "#edf3f7", edgecolor="#555555", linewidth=.8))
        ax.text(x+w/2, y+.375, text, ha="center", va="center", fontsize=7)
    for x, label in [(2.15, "local evidence"), (4.8, "encoded support\n+ signs"), (7.45, "lossless\ndownlink")]:
        ax.add_patch(FancyArrowPatch((x, 1.02), (x+.5, 1.02), arrowstyle="-|>", mutation_scale=10, color="#555555"))
        ax.text(x+.25, 1.49, label, ha="center", va="bottom", fontsize=6.5)
    ax.annotate("Next synchronous round: every client starts from the same server model", xy=(1, .35), xytext=(8.9, .35), ha="right", va="center", fontsize=7, arrowprops={"arrowstyle": "->", "lw": .8, "color": "#555555"})
    return pdf_bytes(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    freeze = check_freeze()
    built = products()
    for name, content in built.items():
        target = PAPER / name
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                raise AssertionError(f"Stale manuscript product: {name}; regenerate first")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f"{'Validated' if args.check else 'Rebuilt'} {len(built)} manuscript products; {len(freeze['files'])} frozen inputs; 100 runs; no training.")


if __name__ == "__main__":
    main()
