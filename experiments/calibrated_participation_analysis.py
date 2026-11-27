"""Read-only campaign audit and derived summaries; never trains or tunes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import calibrated_participation as study
import calibrated_participation_data as data

T95 = {2: 12.706204736432095, 3: 4.302652729696142, 4: 3.182446305284263,
    5: 2.7764451051977987, 6: 2.570581835636314, 7: 2.4469118511449692,
    8: 2.3646242515927844, 9: 2.306004135204166, 10: 2.2621571628540993}
COLORS = {"event": "#D55E00", "strom": "#0072B2", "stc": "#CC79A7", "ef_topk": "#009E73", "dense": "#595959"}
LABELS = {"event": "Event-FedAvg", "strom": "Ström-style", "stc": "STC", "ef_topk": "EF-TopK", "dense": "Dense"}
STYLES = {"event": "-", "strom": "--", "stc": ":", "ef_topk": "-.", "dense": (0, (5, 1, 1, 1))}


def paired(a, b, *, log=False):
    if len(a) != len(b) or len(a) > 10:
        raise ValueError("Invalid paired denominator")
    if not len(a):
        return None, None, None
    values = np.log(np.asarray(a)/np.asarray(b)) if log else np.asarray(a)-np.asarray(b)
    mean = float(values.mean())
    estimate = float(np.exp(mean)) if log else mean
    if len(values) == 1:
        return estimate, None, None
    half = T95[len(values)]*float(values.std(ddof=1))/np.sqrt(len(values))
    return tuple(float(np.exp(x)) if log else float(x) for x in (mean, mean-half, mean+half))


def summarize(runs, history, coverage, phase):
    scope = "validation" if phase == "pilot" else "test"
    expected = 1 if phase == "pilot" else 10
    seeds = (data.DEV_SEED,) if phase == "pilot" else data.SEEDS
    endpoints, effects, crossings, targets = [], [], [], []
    for active in data.ACTIVE:
        indexed = {}
        hits = {}
        for method in study.METHODS:
            group = runs[(runs.active_clients == active) & (runs.method == method)].sort_values("partition_seed")
            indexed[method] = group.set_index("partition_seed")
            state = coverage[(coverage.active_clients == active) & (coverage.method == method)]
            row = {"phase": phase, "active_clients": active, "method": method, "expected": expected,
                "complete": len(group), "numerical_failure": int(state.state.eq("numerical_failure").sum()),
                "pending": int(state.state.eq("pending").sum()), "full_seed_panel": len(group) == expected}
            for metric in ("final_"+scope+"_accuracy", "final_"+scope+"_ce", "unicast_total_bits",
                           "operational_total_bits", "uplink_bits", "unicast_downlink_bits",
                           "initial_sync_bits", "cumulative_catchup_replay_bits",
                           "cumulative_catchup_checkpoint_bits", "terminal_catchup_bits", "local_sgd_steps"):
                values = group[metric].to_numpy(dtype=float)
                row[metric+"_mean"] = float(values.mean()) if len(values) == expected else None
                row[metric+"_std"] = float(values.std(ddof=1)) if len(values) == expected and expected > 1 else None
            endpoints.append(row)
            for seed in seeds:
                curve = history[(history.active_clients == active) & (history.method == method)
                    & (history.partition_seed == seed)].sort_values("round")
                for target in study.TARGETS:
                    first = curve[curve[scope+"_accuracy"] >= target].head(1)
                    hit = first.iloc[0].to_dict() if len(first) else None
                    hits[method, seed, target] = hit
                    crossing = {"phase": phase, "active_clients": active, "method": method,
                        "partition_seed": seed, "target_accuracy": target,
                        "run_complete": seed in indexed[method].index, "target_reached": hit is not None}
                    crossing.update({name: hit[name] if hit else None for name in (
                        "round", "unicast_total_bits", "operational_total_bits", "local_sgd_steps")})
                    crossings.append(crossing)
        for method in study.METHODS:
            if method == "event":
                continue
            event, reference = indexed["event"], indexed[method]
            common = sorted(set(event.index)&set(reference.index))
            full = len(common) == expected
            for metric, field, scale, log in (
                ("accuracy_difference_pp", "final_"+scope+"_accuracy", 100, False),
                ("ce_difference", "final_"+scope+"_ce", 1, False),
                ("unicast_ratio", "unicast_total_bits", 1, True),
                ("operational_ratio", "operational_total_bits", 1, True),
                ("uplink_ratio", "uplink_bits", 1, True),
                ("downlink_ratio", "unicast_downlink_bits", 1, True)):
                a = [scale*float(event.loc[s, field]) for s in common]
                b = [scale*float(reference.loc[s, field]) for s in common]
                estimate, low, high = paired(a, b, log=log) if full else (None, None, None)
                effects.append({"active_clients": active, "reference": method, "metric": metric,
                    "n_total": expected, "n_complete_pairs": len(common),
                    "status": "complete" if full else "blocked_missing_or_failed_runs",
                    "estimate": estimate, "ci95_low": low, "ci95_high": high,
                    "event_fewer_bits": sum(x < y for x, y in zip(a, b)) if log and full else None,
                    "saving_percent": 100*(1-estimate) if log and estimate is not None else None,
                    "saving_ci95_low": 100*(1-high) if log and high is not None else None,
                    "saving_ci95_high": 100*(1-low) if log and low is not None else None})
            for target in study.TARGETS:
                pairs = [(hits["event", seed, target], hits[method, seed, target]) for seed in seeds]
                common_hits = [(a, b) for a, b in pairs if a is not None and b is not None]
                estimate, low, high = paired([a["unicast_total_bits"] for a, _ in common_hits],
                    [b["unicast_total_bits"] for _, b in common_hits], log=True) if full else (None, None, None)
                targets.append({"active_clients": active, "reference": method, "target_accuracy": target,
                    "n_total": expected, "n_complete_pairs": len(common),
                    "event_reached": sum(a is not None for a, _ in pairs),
                    "reference_reached": sum(b is not None for _, b in pairs),
                    "common_hits": len(common_hits),
                    "status": "conditional_on_common_hits" if full else "blocked_missing_or_failed_runs",
                    "estimate": estimate, "ci95_low": low, "ci95_high": high,
                    "saving_percent": 100*(1-estimate) if estimate is not None else None,
                    "saving_ci95_low": 100*(1-high) if high is not None else None,
                    "saving_ci95_high": 100*(1-low) if low is not None else None,
                    "event_fewer_bits": sum(a["unicast_total_bits"] < b["unicast_total_bits"] for a, b in common_hits),
                    "earlier": sum(a["round"] < b["round"] for a, b in common_hits),
                    "same_sampled_round": sum(a["round"] == b["round"] for a, b in common_hits),
                    "later": sum(a["round"] > b["round"] for a, b in common_hits)})
    return {name: pd.DataFrame(rows) for name, rows in (
        ("endpoint_summary", endpoints), ("paired_endpoints", effects),
        ("target_crossings", crossings), ("paired_targets", targets))}


def plot(history, phase, out, *, complete, expected):
    if history.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), sharey=True)
    scope = "validation" if phase == "pilot" else "test"
    for col, active in enumerate(data.ACTIVE):
        for method in study.METHODS:
            group = history[(history.active_clients == active) & (history.method == method)]
            if group.empty:
                continue
            stats = group.groupby("round").agg(accuracy=(scope+"_accuracy", "mean"),
                sd=(scope+"_accuracy", "std"), cost=("unicast_total_bits", "mean"))
            for ax, x in ((axes[0, col], stats.index), (axes[1, col], stats.cost/1e9)):
                ax.plot(x, 100*stats.accuracy, label=LABELS[method], color=COLORS[method], ls=STYLES[method],
                        lw=2 if method == "event" else 1.2)
                ax.fill_between(x, 100*(stats.accuracy-stats.sd.fillna(0)),
                    100*(stats.accuracy+stats.sd.fillna(0)), color=COLORS[method], alpha=.08)
        axes[0, col].set_title(f"{active}/50 clients ({2*active}%)")
        axes[0, col].set_xlabel("Communication round")
        axes[1, col].set_xlabel("Total unicast, incl. final catch-up [Gbit]")
        axes[1, col].set_xscale("log")
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel(scope.capitalize()+" accuracy [%]")
    axes[1, 0].set_ylabel(scope.capitalize()+" accuracy [%]")
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        for a, b in zip(h, l):
            if b not in labels:
                handles.append(a); labels.append(b)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, .96))
    fig.suptitle(f"{phase.capitalize()}: {complete}/{expected} fits complete; bands are sample SD")
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out/f"{phase}_comparison.pdf", metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight")
    fig.savefig(out/f"{phase}_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze(phase, out=study.OUT):
    out = Path(out)
    science = study.load_contract(out)
    coverage = study.coverage(phase, science, out)
    summaries, histories, inputs = [], [], {}
    for job in study.jobs_for(phase):
        receipt = study.read_receipt(job, science, out)
        if receipt is None:
            continue
        inputs[study.stem(job)] = {"receipt": data.file_hash(study.paths(job, out)["receipt"]),
                                  "state": receipt["state"]}
        if receipt["state"] == "complete":
            tables = study.read_tables(job, science, out)
            summaries.append(tables["summary"]); histories.append(tables["history"])
    scope = "validation" if phase == "pilot" else "test"
    empty = ["active_clients", "method", "partition_seed", "round",
        "final_"+scope+"_accuracy", "final_"+scope+"_ce", "unicast_total_bits",
        "operational_total_bits", "uplink_bits", "unicast_downlink_bits", "initial_sync_bits",
        "cumulative_catchup_replay_bits", "cumulative_catchup_checkpoint_bits",
        "terminal_catchup_bits", "local_sgd_steps", scope+"_accuracy"]
    runs = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=empty)
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(columns=empty)
    products = {"coverage": coverage, "runs": runs, "history": history,
                **summarize(runs, history, coverage, phase)}
    for name, frame in products.items():
        data.atomic_csv(out/f"{phase}_{name}.csv", frame)
    complete = int(coverage.state.eq("complete").sum())
    failures = int(coverage.state.eq("numerical_failure").sum())
    pending = int(coverage.state.eq("pending").sum())
    completion = {"phase": phase, "contract_sha256": data.digest(science), "expected": len(coverage),
        "complete": complete, "numerical_failure": failures, "pending": pending,
        "all_jobs_terminal": pending == 0, "all_fits_complete_finite": complete == len(coverage),
        "selection_or_training_in_analysis": False,
        "failed_or_missing_seed_pairs": "reported_explicitly_endpoint_effects_blocked_not_silently_dropped"}
    data.atomic_json(out/f"{phase}_completion.json", completion)
    data.atomic_json(out/f"{phase}_analysis_audit.json", {"contract_sha256": data.digest(science),
        "original_receipts": inputs, "derived_csv_sha256": {
            name: data.file_hash(out/f"{phase}_{name}.csv") for name in products},
        "round_fingerprints_prefixes_frames_reconstruction_and_terminal_components_verified": True,
        "evaluation_scope_read_from_result_files": scope,
        "dataset_images_loaded_or_model_evaluation_run": False, "training": False})
    plot(history, phase, out, complete=complete, expected=len(coverage))
    print(f"{phase}: {complete}/{len(coverage)} complete, {failures} numerical failures, {pending} pending.")
    if phase == "pilot":
        print("Validation only, no tuning or ranking gate. Review protocol/runtime before the long confirmation.")
    if pending or failures:
        print("Incomplete/failed panels stay visible; affected endpoint comparisons have no estimates.")
    return completion
