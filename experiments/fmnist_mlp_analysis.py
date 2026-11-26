"""Audit saved MLP outcomes and generate the complete five-method comparison.

This module is deliberately separate from training source contracts so figure
presentation can evolve without rewriting the frozen optimizer or its results.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import fmnist_mlp_calibrated as study
from fmnist_mlp_data import atomic_json, digest, file_hash

T_CRITICAL = {2: 12.706204736432, 3: 4.302652729696, 4: 3.182446305284, 5: 2.776445105198,
              6: 2.570581835636, 7: 2.446911851145, 8: 2.364624251593, 9: 2.306004135204,
              10: 2.262157162854}
LABELS = {"event": "Event-FedAvg", "strom": "Ström-style", "dense": "Dense FedAvg", "stc": "STC", "ef_topk": "EF-TopK"}
COLORS = {"event": "#D55E00", "strom": "#0072B2", "dense": "#595959", "stc": "#CC79A7", "ef_topk": "#009E73"}
MARKERS = {"event": "*", "strom": "s", "dense": "o", "stc": "D", "ef_topk": "^"}


def interval(a, b, *, ratio=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid paired observations")
    n = len(a)
    if not n:
        return None, None, None
    if ratio and (np.any(a <= 0) or np.any(b <= 0)):
        raise ValueError("Cost ratios require positive costs")
    x = np.log(a/b) if ratio else a-b
    mean = float(x.mean())
    estimate = float(np.exp(mean)) if ratio else mean
    if n == 1:
        return estimate, None, None
    half = T_CRITICAL[n]*float(x.std(ddof=1))/np.sqrt(n)
    return (estimate, float(np.exp(mean-half)), float(np.exp(mean+half))) if ratio else (estimate, mean-half, mean+half)


def validate_tables(tables, *, method, seed, config, horizon):
    rounds, evaluation, history, summary = (tables[k] for k in ("rounds", "evaluation", "history", "summary"))
    expected_rounds = list(range(1, horizon+1))
    expected_evaluation = [r for r in expected_rounds if r == 1 or r % config.eval_stride == 0 or r == config.rounds]
    if rounds["round"].tolist() != expected_rounds or evaluation["round"].tolist() != expected_evaluation:
        raise RuntimeError("Saved round/evaluation coverage differs")
    if history["round"].tolist() != expected_evaluation or len(summary) != 1:
        raise RuntimeError("Saved history/endpoint coverage differs")
    if not rounds.method.eq(method).all() or not rounds.partition_seed.eq(seed).all() or not rounds.roundtrip_verified.eq(True).all():
        raise RuntimeError("Saved model identities/round trips differ")
    if (rounds.codec_uplink_bits < 0).any() or (rounds.exact_downlink_bits <= 0).any():
        raise RuntimeError("Invalid wire lengths")
    initial = 32*study.host.LAYOUT.dimension + 64
    numeric_fields = ("uplink_bits", "server_stream_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits")
    evaluation_by_round = evaluation.set_index("round")
    for row in history.to_dict("records"):
        rnd = int(row["round"])
        prefix = rounds[rounds["round"] <= rnd]
        # Independent Python integer accumulation, not the training host's
        # vectorized prefix builder, checks every saved accounting checkpoint.
        upload = sum(int(v) for v in prefix.codec_uplink_bits)
        stream = sum(int(v) for v in prefix.exact_downlink_bits)
        downlink = 10*(initial + 32*rnd + stream)
        expected = dict(uplink_bits=upload, server_stream_bits=stream, unicast_downlink_bits=downlink,
                        unicast_total_bits=upload+downlink, broadcast_total_bits=upload+initial+stream)
        for field, value in expected.items():
            if int(row[field]) != value:
                raise RuntimeError(f"Independent cumulative accounting differs: {field}")
        for field in ("train_ce", "train_objective", "test_ce", "test_accuracy", "test_macro_accuracy", "test_worst_class_accuracy", "parameter_l2_norm"):
            if field in row and float(row[field]) != float(evaluation_by_round.loc[rnd, field]):
                raise RuntimeError(f"Evaluation/history metric differs: {field}")
    for field in numeric_fields:
        if int(summary.iloc[0][field]) != int(history.iloc[-1][field]):
            raise RuntimeError(f"Final accounting differs: {field}")
    for field in ("train_ce", "train_objective", "test_ce", "test_accuracy", "test_macro_accuracy", "test_worst_class_accuracy"):
        if field in history and float(summary.iloc[0]["final_"+field]) != float(history.iloc[-1][field]):
            raise RuntimeError(f"Final metric differs: {field}")
    fingerprints = rounds.set_index("round").model_sha256
    if evaluation.model_sha256.tolist() != fingerprints.loc[expected_evaluation].tolist():
        raise RuntimeError("Saved evaluation model fingerprints differ")
    if summary.iloc[0].model_sha256 != fingerprints.iloc[-1]:
        raise RuntimeError("Saved final fingerprint differs")


def analyze(out=study.OUT):
    out = Path(out)
    science = study.load_contract(out, check_runtime=False)
    selection = study.verify_selection(science, out)
    selection_sha = digest(selection)
    coverage, runs, histories = [], [], []
    for method in study.METHODS:
        p = study.BY_ID[selection["selected"][method][0]]
        for seed in study.SEEDS:
            receipt = study.read_receipt(p, "confirmation", seed, study.FINAL, science, out, selection_sha=selection_sha)
            coverage.append({"method": method, "partition_seed": seed,
                             "state": "missing" if receipt is None else receipt["state"]})
            if receipt is None or receipt["state"] != "complete":
                continue
            paths = study.stage_paths(p, "confirmation", seed, study.FINAL, out)
            tables = {k: pd.read_csv(paths[k], float_precision="round_trip", dtype={"model_sha256": str}) for k in ("rounds", "evaluation", "history", "summary")}
            validate_tables(tables, method=method, seed=seed, config=study.FinalBaselineConfig(**science["protocol"]["configs"][p.point_id]), horizon=study.FINAL)
            runs.append(tables["summary"])
            histories.append(tables["history"])
    complete = all(r["state"] == "complete" for r in coverage)
    study.atomic_csv(out / "confirmation_coverage.csv", pd.DataFrame(coverage))
    atomic_json(out / "confirmation_completion.json", {"complete": complete, "expected": 50,
                 "finished": len(runs), "states": coverage, "contract_sha256": digest(science), "selection_sha256": selection_sha})
    if not complete:
        raise RuntimeError("Confirmation is incomplete; coverage saved, no paired effect analysis written")
    runs, history = pd.concat(runs, ignore_index=True), pd.concat(histories, ignore_index=True)
    study.atomic_csv(out / "confirmation_runs.csv", runs)
    study.atomic_csv(out / "confirmation_history.csv", history)
    metrics = ["final_test_accuracy", "final_test_ce", "final_test_worst_class_accuracy", "final_train_ce",
               "final_train_objective", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"]
    summary = runs.groupby("method")[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    study.atomic_csv(out / "endpoint_summary.csv", summary.reset_index())
    effects = []
    event = runs[runs.method == "event"].set_index("partition_seed").sort_index()
    for method in study.METHODS:
        if method == "event":
            continue
        reference = runs[runs.method == method].set_index("partition_seed").sort_index()
        for metric, column, scale, ratio in (("accuracy_difference_pp", "final_test_accuracy", 100, False),
                                             ("ce_difference", "final_test_ce", 1, False),
                                             ("unicast_ratio", "unicast_total_bits", 1, True),
                                             ("upload_ratio", "uplink_bits", 1, True),
                                             ("download_ratio", "unicast_downlink_bits", 1, True),
                                             ("broadcast_ratio", "broadcast_total_bits", 1, True)):
            estimate, low, high = interval(scale*event[column], scale*reference[column], ratio=ratio)
            effects.append({"reference": method, "metric": metric, "n": len(event), "estimate": estimate,
                            "ci95_low": low, "ci95_high": high,
                            "saving_percent": 100*(1-estimate) if ratio else None,
                            "saving_ci95_low": 100*(1-high) if ratio else None,
                            "saving_ci95_high": 100*(1-low) if ratio else None})
    study.atomic_csv(out / "paired_endpoints.csv", pd.DataFrame(effects))
    hits, targets = [], []
    for method in study.METHODS:
        for seed in study.SEEDS:
            curve = history[(history.method == method) & (history.partition_seed == seed)].sort_values("round")
            for target in study.TARGETS:
                crossing = curve[curve.test_accuracy >= target]
                row = None if crossing.empty else crossing.iloc[0]
                hits.append({"method": method, "partition_seed": seed, "target_accuracy": target,
                             "reached": row is not None, "round": None if row is None else int(row["round"]),
                             "unicast_total_bits": None if row is None else int(row.unicast_total_bits)})
    hit_frame = pd.DataFrame(hits)
    study.atomic_csv(out / "target_crossings.csv", hit_frame)
    for method in study.METHODS:
        if method == "event":
            continue
        for target in study.TARGETS:
            a = hit_frame[(hit_frame.method == "event") & (hit_frame.target_accuracy == target)].set_index("partition_seed").sort_index()
            b = hit_frame[(hit_frame.method == method) & (hit_frame.target_accuracy == target)].set_index("partition_seed").sort_index()
            common = a.reached & b.reached
            estimate, low, high = interval(a.loc[common, "unicast_total_bits"], b.loc[common, "unicast_total_bits"], ratio=True)
            targets.append({"reference": method, "target_accuracy": target, "n_total": len(a),
                            "event_reached": int(a.reached.sum()), "reference_reached": int(b.reached.sum()),
                            "common_hits": int(common.sum()), "estimate": estimate, "ci95_low": low, "ci95_high": high,
                            "saving_percent": None if estimate is None else 100*(1-estimate),
                            "saving_ci95_low": None if high is None else 100*(1-high),
                            "saving_ci95_high": None if low is None else 100*(1-low),
                            "event_fewer_bits": int((a.loc[common, "unicast_total_bits"] < b.loc[common, "unicast_total_bits"]).sum()),
                            "earlier": int((a.loc[common, "round"] < b.loc[common, "round"]).sum()),
                            "same_sampled_round": int((a.loc[common, "round"] == b.loc[common, "round"]).sum()),
                            "later": int((a.loc[common, "round"] > b.loc[common, "round"]).sum())})
    study.atomic_csv(out / "paired_targets.csv", pd.DataFrame(targets))
    plot(history, out)
    atomic_json(out / "analysis_audit.json", {"contract_sha256": digest(science), "selection_sha256": selection_sha,
                "runs": len(runs), "round_records": len(runs)*study.FINAL, "sampled_fingerprints": len(history),
                "raw_receipts_unchanged": True, "statistics": "descriptive_unadjusted_paired_Student_t_log_ratios",
                "frozen_selection_sha256": file_hash(out / "frozen_selection.json")})
    print("Validated 50 MLP outcomes and exact cumulative accounting; comparison tables/plots saved.")


def plot(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for method in study.METHODS:
        frame = history[history.method == method].groupby("round")
        mean = frame[["test_accuracy", "unicast_total_bits"]].mean()
        sd = frame.test_accuracy.std()
        kwargs = {"label": LABELS[method], "color": COLORS[method], "marker": MARKERS[method],
                  "markevery": 5, "markersize": 5, "linewidth": 2 if method == "event" else 1.2}
        axes[0].plot(mean.index, 100*mean.test_accuracy, **kwargs)
        axes[1].plot(mean.unicast_total_bits/1e9, 100*mean.test_accuracy, **kwargs)
        axes[0].fill_between(mean.index, 100*(mean.test_accuracy-sd), 100*(mean.test_accuracy+sd), alpha=.08, color=COLORS[method])
    axes[0].set(xlabel="Communication round", ylabel="Test accuracy [%]", title="Fixed-horizon predictive performance")
    axes[1].set(xlabel="Cumulative total unicast [Gbit]", ylabel="Test accuracy [%]", xscale="log", title="Accuracy against actual encoded traffic")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Fashion-MNIST MLP · frozen validation-selected five-method panel")
    fig.savefig(Path(out) / "comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(Path(out) / "comparison.png", dpi=180)
    plt.close(fig)
