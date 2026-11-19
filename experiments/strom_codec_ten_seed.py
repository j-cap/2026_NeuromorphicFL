"""Frozen calibrated Ström on ten matched full-training seeds; reuse Event/STC.

python experiments/strom_codec_ten_seed.py verify
python experiments/strom_codec_ten_seed.py preflight
python experiments/strom_codec_ten_seed.py run --workers 2
python experiments/strom_codec_ten_seed.py analyze
git add experiments/results/strom_codec_ten_seed_v1
git commit -m "Add calibrated Strom ten-seed confirmation results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import event_codec_ten_seed as previous
import strom_codec_development as development
import strom_codec_development_audit as development_audit
import strom_codec_host as host
from neuromorphicfl.confirmation_metrics import add_regularization_columns, interval, target_crossings

full = previous.full
OUT = REPO / "experiments/results/strom_codec_ten_seed_v1"
SEEDS, ROUNDS, TARGETS = previous.SEEDS, 1800, previous.TARGETS
POINTS = ("event", "strom", "stc_primary", "stc_sparse")
COMPARISONS = (("event", "strom"), ("strom", "stc_primary"), ("strom", "stc_sparse"),
               ("event", "stc_primary"), ("event", "stc_sparse"))
MAIN_VARIANT = {**full.MAIN_VARIANT, "strom": "both_codecs"}
METRIC_FIELDS = ("train_objective", "train_ce", "test_ce", "test_accuracy", "macro_accuracy", "worst_class_accuracy")
atomic_json, atomic_csv, digest, text_hash = full.atomic_json, full.atomic_csv, full.digest, full.text_hash
train_seed = previous.train_seed


def config_for(point="strom"):
    if point in previous.POINTS:
        return previous.config_for(point)
    if point != "strom":
        raise ValueError("Unknown confirmation point")
    return replace(full.config_for("event"), rounds=ROUNDS, eval_stride=60,
                   strom_threshold=.05, server_gain=.10)


def protocol_spec():
    return {"campaign": "strom_codec_ten_seed_v1", "architecture": "cifar_cnn", "rounds": ROUNDS,
            "dimension": full.LAYOUT.dimension, "n_clients": 10, "regime": "strong",
            "training_examples": 50000, "test_examples": 10000, "train_diagnostic_examples": 5000,
            "partition_seeds": list(SEEDS), "training_seeds": [train_seed(i) for i in range(len(SEEDS))],
            "points": {p: asdict(config_for(p)) for p in POINTS}, "main_variants": MAIN_VARIANT,
            "evaluation_rounds": list(full.sampled_rounds(config_for())),
            "new_training_runs": 10, "reused_Event_STC_results": 30, "expected_point_seed_results": 40,
            "development_partition_seed": 4500, "development_training_examples": 45000,
            "development_checkpoint_reuse": False, "seed_3500_requires_new_Strom_run": True,
            "primary_endpoint": "Event minus calibrated Strom round-1800 test accuracy in percentage points",
            "comparisons": [list(pair) for pair in COMPARISONS],
            "secondary_endpoints": ["test_CE", "regularized_training_diagnostics", "upload", "download",
                                    "unicast_primary_broadcast_supplementary", "accuracy_target_traffic_and_hit_coverage"],
            "accuracy_targets": list(TARGETS), "target_rule": "first_saved_crossing_no_interpolation",
            "unreached_targets": "censored_no_cost_ratio_unless_both_reach_report_all_coverage",
            "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_on_log_scale",
            "scopes": {"all_10": "ten matched seeds including previously inspected pilot 3500",
                       "additional_9": "exclude the previously inspected pilot 3500"},
            "sensitivity_excludes_seed": 3500, "equivalence_or_noninferiority_claim": False,
            "uncertainty_scope": "partition_and_SGD_variation_on_fixed_previously_inspected_test_dataset",
            "test_tuning_or_early_stopping": False, "endpoint": "round_1800_not_best_test_checkpoint",
            "checkpoint_stride": 60, "accounting": "frozen_exact_frames_conservative_unicast_primary",
            "source_policy": "frozen_native_Strom_adapter_and_shared_lossless_codecs",
            "hash_format": "sha256_utf8_lf_v1"}


def selection_provenance():
    science, _, frozen = development_audit.validate()
    if (frozen["selection"]["quality"]["point_id"] != "strom_t0p05_g2"
            or frozen["selected_configs"]["quality"] != asdict(config_for())
            or frozen["selected_configs"]["within_event_budget"] != asdict(config_for())):
        raise RuntimeError("Frozen development setting differs; do not retune this confirmation")
    files = [development.OUT / name for name in ("protocol.json", "frozen_selection.json", "screening_selection.json",
                                                "split_manifest.json", "split_indices.csv")]
    files += [development.paths(development.strom_point(.05, 2), 1800)["receipt"]]
    return {"contract_sha256": digest(science), "config": asdict(config_for()),
            "files": {p.relative_to(REPO).as_posix(): text_hash(p) for p in files}}


def reference_files(point, index):
    files = previous.paths(point, index)
    if index == 0:
        return {"reference_receipt": files["receipt"], **full.paths(point)}
    return files


def fingerprint_reference(point, index):
    return {kind: {"path": file.relative_to(REPO).as_posix(), "sha256": text_hash(file)}
            for kind, file in reference_files(point, index).items()}


def reference_key(point, index):
    return f"{point}_s{index}"


def build_science():
    selected = selection_provenance()  # Freeze validation selection before reading comparison scores.
    old = previous.load_contract()
    references = {}
    for index in range(len(SEEDS)):
        for point in previous.POINTS:
            previous.verified_tables(point, index, old)
            references[reference_key(point, index)] = fingerprint_reference(point, index)
    return {"protocol": protocol_spec(), "development_selection": selected,
            "source_hashes": {**previous.source_hashes(), **development.build_science()["source_hashes"],
                              "experiments/strom_codec_development_audit.py": text_hash(Path(development_audit.__file__)),
                              "experiments/strom_codec_ten_seed.py": text_hash(Path(__file__))},
            "runtime": old["runtime"], "data_sha256": old["data_sha256"],
            "reference_contract_sha256": digest(old), "reference_protocol_sha256": text_hash(previous.OUT / "protocol.json"),
            "full_horizon_contract_sha256": digest(full.load_contract()), "references": references}


def build_contract():
    science = build_science()
    if full.development.runtime_manifest() != science["runtime"]:
        raise RuntimeError("Use the recorded Windows environment before confirmation training")
    if digest(full.build_contract()) != science["full_horizon_contract_sha256"]:
        raise RuntimeError("Execution source/runtime/raw CIFAR batches differ from the reference campaign")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Confirmation protocol, selection, sources or reference evidence changed")
    return science


def metadata(point, index, science):
    if point not in POINTS:
        raise ValueError("Unknown confirmation point")
    seed = train_seed(index)  # Also rejects negative/out-of-range indices.
    return {"point": point, "seed_index": index, "partition_seed": SEEDS[index], "train_seed": seed,
            "config_sha256": digest(asdict(config_for(point))), "contract_sha256": digest(science),
            "evaluation_scope": "calibrated_Strom_full_training_ten_seed_confirmation",
            "evidence_origin": "new_calibrated_Strom_run" if point == "strom" else "reused_Event_STC_codec_run",
            "source_contract_sha256": digest(science) if point == "strom" else science["reference_contract_sha256"]}


def paths(point, index, out=OUT):
    train_seed(index)
    stem = f"{point}_s{index}_p{SEEDS[index]}_r{ROUNDS}"
    return {kind: out / f"{stem}_{kind}.{'json' if kind == 'receipt' else 'csv'}"
            for kind in ("summary", "rounds", "evaluation", "history", "receipt")}


def read_receipt(point, index, science, out=OUT):
    files = paths(point, index, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != metadata(point, index, science):
        raise RuntimeError("Confirmation receipt identity differs")
    state = receipt.get("state")
    if point != "strom":
        expected = science["references"][reference_key(point, index)]
        if state != "reused" or receipt.get("source") != expected:
            raise RuntimeError("Event/STC must reference the frozen completed evidence")
        for info in expected.values():
            if text_hash(REPO / info["path"]) != info["sha256"]:
                raise RuntimeError("Reused reference evidence changed")
    elif state == "complete":
        for kind in ("summary", "rounds", "evaluation", "history"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing or changed Strom evidence: seed {index}, {kind}")
    elif state != "numerical_failure":
        raise RuntimeError("Every Strom seed requires a new full-training run; invalid completion state")
    return receipt


def register_reuse(science, out=OUT):
    for index in range(len(SEEDS)):
        for point in previous.POINTS:
            file = paths(point, index, out)["receipt"]
            with full.development.point_lock(out / "locks" / (file.stem + ".lock")):
                if read_receipt(point, index, science, out) is None:
                    atomic_json({"identity": metadata(point, index, science), "state": "reused",
                                 "source": science["references"][reference_key(point, index)]}, file)


def trajectory_tables(point, result, science):
    account = host.traffic_summary if point == "strom" else full.traffic_summary
    result = {**result, "summary": account(result["rounds"], n_clients=science["protocol"]["n_clients"],
                                          dimension=science["protocol"]["dimension"])}
    assemble = host.trajectory_tables if point == "strom" else full.trajectory_tables
    summary, history = assemble(result, n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    return {"summary": summary, "history": history, "rounds": result["rounds"], "evaluation": result["evaluation"]}


def save_result(index, science, result, out=OUT):
    tables = trajectory_tables("strom", result, science)
    files = paths("strom", index, out)
    for kind, frame in tables.items():
        atomic_csv(frame.assign(**metadata("strom", index, science)), files[kind])
    atomic_json({"identity": metadata("strom", index, science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def verified_tables(point, index, science, out=OUT):
    receipt = read_receipt(point, index, science, out)
    if receipt is None or receipt["state"] not in ("complete", "reused"):
        raise RuntimeError("Incomplete confirmation point")
    if point == "strom":
        tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
                  for kind, file in paths(point, index, out).items() if kind != "receipt"}
    else:
        old = previous.load_contract()
        if digest(old) != science["reference_contract_sha256"]:
            raise RuntimeError("Reference contract differs")
        tables = {kind: frame.assign(**metadata(point, index, science))
                  for kind, frame in previous.verified_tables(point, index, old).items()}
    for kind, frame in tables.items():
        for field, value in metadata(point, index, science).items():
            if not frame[field].eq(value).all():
                raise RuntimeError(f"Confirmation metadata differs: {point}/{index}/{kind}/{field}")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    if (rounds["round"].tolist() != list(range(1, ROUNDS + 1))
            or evaluation["round"].tolist() != science["protocol"]["evaluation_rounds"]):
        raise RuntimeError("Confirmation horizon/evaluation coverage differs")
    method = point if point in ("event", "strom") else "stc"
    if not rounds.method.eq(method).all():
        raise RuntimeError("Confirmation training method differs")
    if (not np.isfinite(evaluation[list(METRIC_FIELDS)]).all().all()
            or not evaluation[["test_accuracy", "macro_accuracy", "worst_class_accuracy"]].apply(lambda c: c.between(0, 1)).all().all()
            or not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"])):
        raise RuntimeError("Invalid confirmation predictions/model fingerprints")
    rebuilt = trajectory_tables(point, {"rounds": rounds, "evaluation": evaluation[["round", *METRIC_FIELDS, "model_sha256"]]}, science)
    for kind in ("summary", "history"):
        for field in rebuilt[kind]:
            actual, expected = tables[kind][field], rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=1e-12, atol=1e-12)):
                raise RuntimeError(f"Reconstructed confirmation evidence differs: {point}/{index}/{kind}/{field}")
    return tables


def run_point(index, science, out=OUT):
    train_seed(index)
    lock = out / "locks" / (paths("strom", index, out)["receipt"].stem + ".lock")
    with full.development.point_lock(lock):
        receipt = read_receipt("strom", index, science, out)
        if receipt is not None:
            return f"reuse strom p{SEEDS[index]}: {receipt['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/runtime/data differs from the frozen confirmation")
        federation = full.make_cifar10_federation(root=full.DATA, regime="strong", seed=SEEDS[index],
                                                 train_eval_size=5000, n_clients=10)

        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"strom p{SEEDS[index]}: {row['round']}/{ROUNDS}, "
                      f"test_acc={evaluation['test_accuracy']:.4f}, test_ce={evaluation['test_ce']:.6f}", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = host.evaluated_run(federation=federation, architecture="cifar_cnn", config=config_for(),
                                            seed=train_seed(index), checkpoint_path=out / "checkpoints" / (lock.stem + ".npz"),
                                            provenance=metadata("strom", index, science), progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": metadata("strom", index, science), "state": "numerical_failure", "error": str(exc)},
                        paths("strom", index, out)["receipt"])
            return f"numerical failure strom p{SEEDS[index]}: {exc}"
        save_result(index, science, result, out)
        return f"done strom p{SEEDS[index]}"


def run(workers, seed_index=None, out=OUT):
    if not 1 <= workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    if seed_index is not None:
        train_seed(seed_index)
    with full.development.point_lock(out / "locks" / "controller.lock"):
        science = build_contract()
        protocol = out / "protocol.json"
        if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != science:
            raise RuntimeError("Existing confirmation contract differs; preserve its evidence")
        if not protocol.exists():
            atomic_json(science, protocol)
        register_reuse(science, out)
        indices = range(len(SEEDS)) if seed_index is None else (seed_index,)
        jobs = [i for i in indices if read_receipt("strom", i, science, out) is None]
        print(f"30 Event/STC runs reused; {len(jobs)} calibrated Strom runs pending in this selection", flush=True)
        if workers == 1:
            for index in jobs:
                print(run_point(index, science, out), flush=True)
        elif jobs:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(run_point, index, science, out) for index in jobs]
                for future in as_completed(futures):
                    print(future.result(), flush=True)


def optimized(frame):
    return frame[[variant == MAIN_VARIANT[point] for point, variant in zip(frame.point, frame.variant)]].copy()


def indexed(frame, point, expected_seeds):
    rows = frame[frame.point == point].set_index("partition_seed")
    if rows.index.has_duplicates or set(rows.index) != set(expected_seeds):
        raise ValueError(f"Missing or duplicate paired seeds for {point}")
    return rows.sort_index()


def paired_endpoints(runs):
    rows, summaries = [], []
    for candidate, reference in COMPARISONS:
        a, b = (indexed(runs, p, SEEDS) for p in (candidate, reference))
        for seed in SEEDS:
            for name, field, ratio, scale in previous.METRICS:
                value = float(a.loc[seed, field] / b.loc[seed, field]) if ratio else float(scale * (a.loc[seed, field] - b.loc[seed, field]))
                rows.append({"candidate": candidate, "reference": reference, "partition_seed": seed,
                             "metric": name, "value": value})
    pairs = pd.DataFrame(rows)
    for scope in ("all_10", "additional_9"):
        selected = pairs if scope == "all_10" else pairs[pairs.partition_seed != SEEDS[0]]
        for (candidate, reference, metric), group in selected.groupby(["candidate", "reference", "metric"], sort=False):
            summaries.append({"scope": scope, "candidate": candidate, "reference": reference, "metric": metric,
                              **interval(group.value, ratio=metric.endswith("_ratio")),
                              "direction": "candidate_over_reference" if metric.endswith("_ratio") else "candidate_minus_reference",
                              "primary_endpoint": (candidate, reference, metric) == ("event", "strom", "test_accuracy_difference_pp")})
    return pairs, pd.DataFrame(summaries)


def target_outcomes(history):
    rows = []
    for (point, seed), frame in history.groupby(["point", "partition_seed"], sort=False):
        frame = frame.sort_values("round").copy()
        frame["uplink_packetized_bits"] = frame.uplink_bits
        for row in target_crossings(frame, targets=TARGETS, horizon=ROUNDS, comparison_id=point, partition_seed=int(seed)):
            row["point"] = point
            row["accuracy_at_crossing"] = (float(frame.loc[frame["round"] == row["first_sampled_round"], "test_accuracy"].iloc[0])
                                           if row["reached"] else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_target_summary(crossings):
    rows = []
    for scope in ("all_10", "additional_9"):
        selected = crossings if scope == "all_10" else crossings[crossings.partition_seed != SEEDS[0]]
        seeds = SEEDS if scope == "all_10" else SEEDS[1:]
        for candidate, reference in COMPARISONS:
            for target in TARGETS:
                frame = selected[selected.target_accuracy == target]
                a, b = (indexed(frame, p, seeds) for p in (candidate, reference))
                common = a.reached & b.reached
                ratios = a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"]
                rows.append({"scope": scope, "candidate": candidate, "reference": reference, "target_accuracy": target,
                             "n_total": len(seeds), "n_candidate_reached": int(a.reached.sum()),
                             "n_reference_reached": int(b.reached.sum()), "n_common_hits": int(common.sum()),
                             **interval(ratios, ratio=True), "direction": "candidate_over_reference",
                             "estimand": "unicast_ratio_conditional_on_both_reaching_target"})
    return pd.DataFrame(rows)


def plot(history, paired, targets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"event": "#333333", "strom": "#009E73", "stc_primary": "#D55E00", "stc_sparse": "#0072B2"}
    labels = {"event": "Event", "strom": "Calibrated Ström", "stc_primary": "STC p=1", "stc_sparse": "STC p=0.25"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for point in POINTS:
        frame = history[history.point == point].groupby("round")
        acc, ce, bits = frame.test_accuracy.mean(), frame.test_ce.mean(), frame.unicast_total_bits.mean()
        axes[0, 0].plot(acc.index, 100 * acc, color=colors[point], label=labels[point])
        axes[0, 1].plot(bits / 1e6, 100 * acc, color=colors[point], label=labels[point])
        axes[1, 0].plot(ce.index, ce, color=colors[point], label=labels[point])
    chosen = targets[(targets.scope == "all_10") & (targets.candidate == "event") & (targets.reference == "strom")]
    for x, (_, row) in enumerate(chosen.sort_values("target_accuracy").iterrows()):
        if np.isfinite(row.estimate):
            error = ([[max(0, row.estimate - row.ci95_low)], [max(0, row.ci95_high - row.estimate)]]
                     if np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high) else None)
            axes[1, 1].errorbar(x, row.estimate, yerr=error, fmt="o", color=colors["strom"], capsize=3)
        axes[1, 1].annotate(f"{int(row.n_common_hits)}/{int(row.n_total)}", (x, row.ci95_high if np.isfinite(row.ci95_high) else 1),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].set(xlabel="Communication rounds", ylabel="Mean test accuracy (%)", title="Ten-seed descriptive mean")
    axes[0, 1].set(xlabel="Mean total unicast traffic (Mbit)", ylabel="Mean test accuracy (%)", title="Communication to predictive quality")
    axes[1, 0].set(xlabel="Communication rounds", ylabel="Mean test cross-entropy", title="Fixed-horizon predictive loss")
    axes[1, 1].axhline(1, color="#777777", ls="--", lw=.8)
    axes[1, 1].set_xticks(range(len(TARGETS)), [f"{100 * t:g}%" for t in TARGETS])
    axes[1, 1].set_xlim(-.25, len(TARGETS) - .75)
    axes[1, 1].set(xlabel="Target test accuracy (labels: common hits)", ylabel="Event / Ström unicast traffic",
                   title="Target costs: conditional ratios and 95% intervals")
    axes[1, 1].margins(y=.18)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.legend(fontsize=8)
    fig.suptitle("Frozen calibrated Ström: ten matched full-training seeds; Event/STC reused", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "confirmation.png", dpi=180)
    fig.savefig(out / "confirmation.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    completion = []
    for index in range(len(SEEDS)):
        for point in POINTS:
            try:
                receipt = read_receipt(point, index, science, out)
                completion.append({"point": point, "partition_seed": SEEDS[index],
                                   "state": receipt["state"] if receipt else "pending", "error": receipt.get("error", "") if receipt else ""})
            except (RuntimeError, OSError) as exc:
                completion.append({"point": point, "partition_seed": SEEDS[index], "state": "invalid", "error": str(exc)})
    atomic_json({"expected": len(SEEDS) * len(POINTS), "results": completion}, out / "completion_audit.json")
    if any(row["state"] not in ("complete", "reused") for row in completion):
        raise RuntimeError("Confirmation grid incomplete or failed; no seed is dropped. See completion_audit.json")
    tables = [verified_tables(point, index, science, out) for index in range(len(SEEDS)) for point in POINTS]
    summary = pd.concat([t["summary"] for t in tables], ignore_index=True)
    history = pd.concat([t["history"] for t in tables], ignore_index=True)
    runs = add_regularization_columns(optimized(summary), config_for().regularization)
    main_history = optimized(history)
    if len(runs) != len(SEEDS) * len(POINTS) or runs.duplicated(["point", "partition_seed"]).any():
        raise RuntimeError("Confirmation endpoint grid differs")
    pairs, paired = paired_endpoints(runs)
    targets = target_outcomes(main_history)
    target_summary = paired_target_summary(targets)
    outputs = {"comparison.csv": summary, "comparison_history.csv": history, "main_runs.csv": runs,
               "endpoint_summary.csv": previous.descriptive_summary(runs), "paired_per_seed.csv": pairs,
               "paired_summary.csv": paired, "traffic_to_target.csv": targets, "paired_target_summary.csv": target_summary}
    for name, frame in outputs.items():
        atomic_csv(frame, out / name)
    atomic_json({"seed_pairs": len(SEEDS), "new_Strom_training_runs": len(SEEDS), "reused_Event_STC_results": 30,
                 "development_checkpoint_reused": False, "frozen_development_selection_sha256": digest(science["development_selection"]),
                 "source_receipts_model_fingerprints_and_prefix_costs_verified": True,
                 "primary_endpoint": protocol_spec()["primary_endpoint"], "intervals": protocol_spec()["intervals"],
                 "scopes": protocol_spec()["scopes"], "equivalence_or_noninferiority_claim": False,
                 "test_tuning_or_early_stopping": False, "target_coverage_required_for_interpretation": True}, out / "analysis_audit.json")
    plot(main_history, paired, target_summary, out)
    print(paired[(paired.scope == "all_10") & (paired.candidate == "event") & (paired.reference == "strom")].to_string(index=False))
    print(target_summary[(target_summary.scope == "all_10") & (target_summary.candidate == "event")
                         & (target_summary.reference == "strom")].to_string(index=False))


def preflight():
    science = build_science()
    print("Frozen validation-selected Strom: threshold 0.05, equivalent gain 2, native gain 0.10")
    print("30 completed Event/STC results verified for reuse; TEN new Strom runs, including seed 3500")
    print("Full 50000-image training set; official 10000-image test; 1800 rounds; no retuning")
    print("Development 45000/5000 checkpoint is excluded; partition 4500 is not a confirmation seed")
    print("Matched partition seeds:", ", ".join(map(str, SEEDS)))
    print("Runtime matches recorded Windows environment:", full.development.runtime_manifest() == science["runtime"])
    print("Both all-ten and excluding-pilot-nine paired summaries will be reported")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("Calibrated Strom ten-seed confirmation not started; ten full-training runs required")
        return
    science = load_contract(out)
    for index, seed in enumerate(SEEDS):
        values = []
        for point in POINTS:
            receipt = read_receipt(point, index, science, out)
            value = receipt["state"] if receipt else "pending"
            checkpoint = out / "checkpoints" / (paths(point, index, out)["receipt"].stem + ".npz")
            if receipt is None and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"].item()))
                if meta["provenance"] != metadata(point, index, science):
                    raise RuntimeError("Checkpoint provenance differs")
                value = f"checkpoint {meta['progress']['completed_round']}"
            values.append(f"{point}: {value}")
        print(seed, "; ".join(values))


def smoke_worker(index):
    from stc_stage1 import smoke_federation
    config = replace(config_for(), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
    with tempfile.TemporaryDirectory() as folder:
        kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", config=config, seed=train_seed(index),
                      provenance={"synthetic_seed_index": index, "config": asdict(config)})
        uninterrupted = host.evaluated_run(**kwargs, checkpoint_path=Path(folder) / "full.npz")
        path = Path(folder) / "resumed.npz"
        host.evaluated_run(**kwargs, checkpoint_path=path, stop_after=2)
        resumed = host.evaluated_run(**kwargs, checkpoint_path=path)
        for field in ("model", "client_model", "strom_state"):
            np.testing.assert_array_equal(uninterrupted[field].view(np.uint32), resumed[field].view(np.uint32))
        pd.testing.assert_frame_equal(uninterrupted["evaluation"], resumed["evaluation"], check_exact=True)
        columns = [c for c in uninterrupted["rounds"] if c != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(uninterrupted["rounds"][columns], resumed["rounds"][columns], check_exact=True)
        if uninterrupted["rng_state"] != resumed["rng_state"]:
            raise AssertionError("Resumed RNG state differs")
        return {"seed_index": index, "train_seed": train_seed(index),
                "model_sha256": resumed["evaluation"].iloc[-1].model_sha256,
                "summary_sha256": digest(resumed["summary"].to_dict("records"))}


def smoke():
    indices = (0, 9)
    expected = [smoke_worker(i) for i in indices]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = list(pool.map(smoke_worker, indices))
    if actual != expected:
        raise AssertionError("Spawned calibrated Strom seed/model/traffic mapping differs")
    print("Synthetic first/last seeds: exact model, residual, RNG, prediction and traffic resume; spawned workers match")
    print("No CIFAR research training was run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--seed-index", type=int, choices=range(len(SEEDS)))
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_strom_codec_ten_seed.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke()
    elif args.command == "smoke":
        smoke()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        run(args.workers, args.seed_index)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
