"""Thirty selected ResNet baseline fits and twenty verified pulse reuses.

python experiments/resnet14_core_confirmation.py verify --device cuda
python experiments/resnet14_core_confirmation.py preflight --device cuda
python experiments/resnet14_core_confirmation.py run --device cuda
git add experiments/results/resnet14_core_confirmation_v1
git commit -m "Add selected ResNet14 common-seed baseline confirmation"
git push origin main
"""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_baseline_development as development
from resnet14_baseline_development_audit import independent_traffic as baseline_traffic
from resnet14_transfer_audit import independent_traffic as pulse_traffic
import numpy as np
import pandas as pd

parent, pulse, host = development.parent, development.pulse, development.host
OUT = REPO / "experiments/results/resnet14_core_confirmation_v1"
SEEDS, ROUNDS = pulse.SEEDS, 1800
REUSED, NEW = ("event", "strom"), ("dense", "ef_topk", "stc")
METHODS = (*REUSED, *NEW)
POINT_IDS = {"dense": "dense_g1p5_a0", "ef_topk": "ef_topk_p0p01_g4", "stc": "stc_u1_d1_g4"}
MAIN = {"event": "both_codecs", "strom": "both_codecs", **host.MAIN}
LABELS = {"event": "Event", "strom": "Ström", "dense": "Dense gain 1.5",
          "ef_topk": "EF-TopK p=0.01, gain 4", "stc": "STC p=1, gain 4"}
EVIDENCE_COMMIT = "a181e4ebab3df94e5add5a8ad566ca24ca540c6d"
METRICS = ("train_objective", "train_ce", "test_ce", "test_accuracy", "test_macro_accuracy", "test_worst_class_accuracy")


def point_for(method):
    if method in NEW:
        return development.BY_ID[POINT_IDS[method]]
    if method in REUSED:
        return pulse.point_for(method)
    raise ValueError("Unknown confirmation method")


def config_for(method):
    p = point_for(method)
    return development.config_for(p) if method in NEW else parent.config_for(p)


def protocol_spec():
    return {
        "campaign": "resnet14_core_confirmation_v1", "architecture": "CIFAR_ResNet14_GroupNorm_Torch",
        "dimension": 175258, "n_clients": 10, "regime": "strong_55pct_dominant", "rounds": ROUNDS,
        "partition_seeds": list(SEEDS), "training_seeds": [parent.train_seed(s) for s in SEEDS],
        "new_methods": list(NEW), "reused_methods": list(REUSED), "main_variants": MAIN,
        "point_ids": {m: point_for(m).point_id for m in METHODS}, "configs": {m: asdict(config_for(m)) for m in METHODS},
        "new_training_runs": len(NEW) * len(SEEDS), "reused_training_results": len(REUSED) * len(SEEDS),
        "expected_outcomes": len(METHODS) * len(SEEDS), "additional_training_rounds": len(NEW) * len(SEEDS) * ROUNDS,
        "additional_evaluation_points": len(NEW) * len(SEEDS) * 31,
        "total_verified_rounds": len(METHODS) * len(SEEDS) * ROUNDS,
        "total_sampled_model_fingerprints": len(METHODS) * len(SEEDS) * 31,
        "training_examples": 50000, "test_examples": 10000, "training_diagnostic_examples": 5000,
        "eval_stride": 60, "initialization_seed": 7777, "matched_minibatches_and_partitions": True,
        "selection": "frozen_round1800_validation_CE_winners_after_600_round_screen_top3_continuation",
        "selection_reference_commit": EVIDENCE_COMMIT, "new_tuning": False, "automatic_boundary_extension": False,
        "sparse_STC_secondary_runs": 0, "automatic_extra_methods_or_seed_replacement": False,
        "development_checkpoint_reuse": False, "test_based_selection_or_early_stopping": False,
        "primary_predictive_endpoint": "Event_minus_each_reference_round1800_test_accuracy_pp",
        "secondary_endpoints": ["test_CE", "worst_class_accuracy", "training_CE_objective_L2_diagnostics",
                                "upload", "download", "total_unicast", "logical_broadcast", "traffic_to_fixed_accuracy_targets"],
        "comparisons": [["event", m] for m in METHODS if m != "event"],
        "primary_communication_accounting": "actual_lossless_frames_initialization_requests_conservative_unicast",
        "accuracy_targets": list(parent.TARGETS), "target_rule": "first_saved_crossing_no_interpolation_no_stability_requirement",
        "unreached_targets": "censored_at1800_ratio_on_common_hits_full_coverage_reported",
        "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_on_log_scale",
        "baseline_arithmetic": "frozen_native_NumPy_float32_compressors_and_server_Torch_local_SGD_and_model_application",
        "codec": host.WIRE_PROTOCOL, "execution": "sequential_single_device_same_recorded_runtime_as_completed_ResNet_pulses",
        "reuse_policy": "all_twenty_completed_Event_Strom_outcomes_read_only_source_receipts_and_hashes_verified",
        "numerical_failure": "terminal_receipt_retained_continue_other_runs_final_analysis_requires_all_fifty_outcomes",
        "uncertainty_scope": "partition_SGD_variation_fixed_initialization_and_previously_inspected_CIFAR10_test_set",
        "scope": "selected_five_method_common_seed_ResNet_confirmation_no_retuning_or_sparse_STC_extension",
    }


def selection_provenance():
    # Reproduce the validation-only choices before opening any pulse test results.
    science = development.load_contract()
    screen = development.ledger(development.POINTS, development.SCREEN, science)
    promotion = development.decision(screen, 3, development.SCREEN, science)
    if promotion != json.loads((development.OUT / "promotion.json").read_text()):
        raise RuntimeError("Baseline validation promotion changed")
    points = tuple(development.BY_ID[k] for m in development.METHODS for k in promotion["selected"][m])
    final = development.ledger(points, development.FINAL, science)
    development.validate_prefixes(points, science)
    chosen = development.choose(final, 1)
    frozen = {**development.decision(final, 1, development.FINAL, science),
              "configs": {m: asdict(development.config_for(development.BY_ID[ids[0]])) for m, ids in chosen.items()},
              "promotion_sha256": parent.text_hash(development.OUT / "promotion.json"), "confirmation_launched": False,
              "development_checkpoint_reuse_for_confirmation": False, "automatic_boundary_extension": False}
    if (frozen != json.loads((development.OUT / "frozen_selection.json").read_text())
            or chosen != {m: [POINT_IDS[m]] for m in NEW}):
        raise RuntimeError("Expected exactly the frozen Dense, EF-TopK and p1/gain4 STC choices")
    common = ("rounds", "eval_stride", "local_steps", "local_lr", "batch_size", "regularization", "init_scale")
    for method in NEW:
        if any(getattr(config_for(method), k) != getattr(config_for("event"), k) for k in common):
            raise RuntimeError("Baseline and pulse local-learning protocols differ")
    paths = [development.OUT / n for n in ("protocol.json", "promotion.json", "frozen_selection.json", "selected.csv")]
    paths.extend(path for m in NEW for path in development.paths(point_for(m), development.FINAL).values())
    return science, {"calibration_contract_sha256": parent.digest(science), "selected": chosen,
                     "configs": frozen["configs"], "files": {p.relative_to(REPO).as_posix(): parent.text_hash(p) for p in paths}}


def reference_key(method, seed):
    return f"{method}_p{seed}"


def reference_files(method, seed):
    source = parent.OUT if seed in pulse.PILOT_SEEDS else pulse.OUT
    files = {**parent.paths(pulse.point_for(method), "confirmation", seed, ROUNDS, source),
             "partition": source / f"confirmation_partition_p{seed}.json"}
    if seed in pulse.PILOT_SEEDS:
        files["pulse_reuse_receipt"] = pulse.reuse_path(method, seed)
    return files


def fingerprint_reference(method, seed):
    return {k: {"path": p.relative_to(REPO).as_posix(), "sha256": parent.text_hash(p)}
            for k, p in reference_files(method, seed).items()}


def science_spec():
    calibration, selection = selection_provenance()
    prior = pulse.load_contract()
    if calibration["execution"] != prior["execution"] or calibration["data_sha256"] != prior["data_sha256"]:
        raise RuntimeError("Baseline calibration and pulse confirmation runtime/data differ")
    references = {}
    for seed in SEEDS:
        manifest = prior["partition_manifests"][str(seed)]
        if manifest["training_examples"] != 50000 or manifest["validation_examples"] != 0:
            raise RuntimeError("Confirmation requires full-training allocations")
        for method in REUSED:
            receipt = pulse.read_outcome(method, seed, prior)
            if receipt is None or receipt["state"] not in ("complete", "reused"):
                raise RuntimeError("All twenty completed pulse outcomes are required")
            table = pulse.verified_tables(method, seed, prior)
            pulse_traffic(table, clients=prior["protocol"]["n_clients"], dimension=prior["protocol"]["dimension"])
            references[reference_key(method, seed)] = fingerprint_reference(method, seed)
    sources = {}
    for name, sha in (*calibration["source_hashes"].items(), *prior["source_hashes"].items()):
        name = name.replace("\\", "/")
        if name in sources and sources[name] != sha:
            raise RuntimeError("Conflicting baseline/pulse dependency identity")
        sources[name] = sha
    for name in ("experiments/resnet14_core_confirmation.py", "experiments/resnet14_baseline_development_audit.py",
                 "experiments/resnet14_transfer_audit.py"):
        sources[name] = parent.text_hash(REPO / name)
    return {"protocol": protocol_spec(), "selection": selection, "source_hashes": sources,
            "execution": prior["execution"], "data_sha256": prior["data_sha256"],
            "reference_contract_sha256": parent.digest(prior), "references": references,
            "partition_manifests": prior["partition_manifests"]}


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text())
    if science != science_spec():
        raise RuntimeError("ResNet core protocol, sources, frozen choices or reused outcomes changed")
    return science


def prepare(device_name, threads, eval_batch_size, out=OUT):
    import resnet14_transfer_host as torch_host
    science = science_spec()
    device = torch_host.configure(device_name, threads)
    pulse.check_runtime(science, device, eval_batch_size)
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != science:
        raise RuntimeError("Existing ResNet core campaign differs; preserve its identity")
    root = parent.data_tools.download_cifar10(parent.DATA)
    if {n: parent.full.development.file_hash(root / n) for n in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR batch fingerprints changed")
    parent.immutable_json(science, out / "protocol.json")
    return science, device


def metadata(method, seed, science):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown confirmation method/seed")
    return {"comparison_id": method, "point_id": point_for(method).point_id, "method": point_for(method).method,
            "phase": "confirmation", "partition_seed": seed, "train_seed": parent.train_seed(seed),
            "config_sha256": parent.digest(asdict(config_for(method))), "contract_sha256": parent.digest(science),
            "stage_horizon": ROUNDS}


def paths(method, seed, out=OUT):
    stem = f"{method}_p{seed}_confirmation_r{ROUNDS}"
    return {k: out / f"{stem}_{k}.{'json' if k == 'receipt' else 'csv'}"
            for k in ("rounds", "evaluation", "summary", "history", "receipt")}


def checkpoint_path(method, seed, out=OUT):
    if method not in NEW or seed not in SEEDS:
        raise ValueError("Only new baseline fits have checkpoints in this campaign")
    return out / "checkpoints" / f"{method}_p{seed}_confirmation.npz"


def read_outcome(method, seed, science, out=OUT):
    expected_identity = metadata(method, seed, science)
    files = paths(method, seed, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text())
    if receipt.get("identity") != expected_identity:
        raise RuntimeError("ResNet core receipt identity differs")
    if method in REUSED:
        expected = science["references"][reference_key(method, seed)]
        if (receipt.get("state") != "reused" or receipt.get("references") != expected
                or fingerprint_reference(method, seed) != expected
                or receipt.get("source_contract_sha256") != science["reference_contract_sha256"]):
            raise RuntimeError("Read-only pulse outcome or reference fingerprints changed")
    elif receipt.get("state") == "complete":
        if any(parent.text_hash(files[k]) != receipt["sha256"][k] for k in ("rounds", "evaluation", "summary", "history")):
            raise RuntimeError("New ResNet core evidence changed")
    elif receipt.get("state") != "numerical_failure":
        raise RuntimeError("New baselines require new fits, not reused outcomes")
    return receipt


def register_reuse(science, out=OUT):
    for seed in SEEDS:
        for method in REUSED:
            value = {"identity": metadata(method, seed, science), "state": "reused",
                     "source_contract_sha256": science["reference_contract_sha256"],
                     "references": science["references"][reference_key(method, seed)]}
            parent.immutable_json(value, paths(method, seed, out)["receipt"])
            read_outcome(method, seed, science, out)


def save_result(method, seed, science, result, manifest, out=OUT):
    tables = development.trajectory_tables(result, science)
    files = paths(method, seed, out)
    for kind, frame in tables.items():
        parent.atomic_csv(frame.assign(**metadata(method, seed, science)), files[kind])
    parent.atomic_json({"identity": metadata(method, seed, science), "state": "complete", "data_manifest": manifest,
                        "sha256": {k: parent.text_hash(files[k]) for k in tables}}, files["receipt"])


def verified_tables(method, seed, science, out=OUT):
    receipt = read_outcome(method, seed, science, out)
    if receipt is None or receipt["state"] not in ("complete", "reused"):
        raise RuntimeError("Incomplete ResNet core outcome")
    if method in REUSED:
        prior = json.loads((pulse.OUT / "protocol.json").read_text())
        if parent.digest(prior) != science["reference_contract_sha256"]:
            raise RuntimeError("Pulse source contract differs")
        tables = pulse.verified_tables(method, seed, prior)
        pulse_traffic(tables, clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    else:
        files = paths(method, seed, out)
        tables = {k: pd.read_csv(path, float_precision="round_trip", dtype={"model_sha256": str})
                  for k, path in files.items() if k != "receipt"}
        manifest = science["partition_manifests"][str(seed)]
        if (receipt["data_manifest"] != manifest
                or json.loads((out / f"confirmation_partition_p{seed}.json").read_text()) != manifest):
            raise RuntimeError("New confirmation partition/diagnostic allocation differs")
        for frame in tables.values():
            for key, value in metadata(method, seed, science).items():
                if not frame[key].eq(value).all():
                    raise RuntimeError("New confirmation table metadata differs")
        r, e = tables["rounds"], tables["evaluation"]
        cfg = config_for(method)
        if (r["round"].tolist() != list(range(1, ROUNDS + 1))
                or e["round"].tolist() != sorted({1, ROUNDS, *range(cfg.eval_stride, ROUNDS + 1, cfg.eval_stride)})
                or any(c.startswith("validation_") for c in e) or not np.isfinite(e[list(METRICS)]).all().all()
                or not e[list(METRICS[3:])].apply(lambda s: s.between(0, 1)).all().all()
                or not np.array_equal(e.model_sha256, r.set_index("round").loc[e["round"], "model_sha256"])):
            raise RuntimeError("Test metric, horizon or sampled model coverage differs")
        rebuilt = development.trajectory_tables({"rounds": r, "evaluation": e[["round", *METRICS, "model_sha256"]]}, science)
        for kind in ("summary", "history"):
            for key in rebuilt[kind]:
                a, b = tables[kind][key], rebuilt[kind][key]
                exact = key.endswith("bits") or key in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
                if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1e-12, atol=1e-12)):
                    raise RuntimeError("Reconstructed confirmation metrics/accounting differ")
        baseline_traffic(tables, point_for(method), clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    return {kind: frame.assign(comparison_id=method, core_evidence_origin="verified_reuse" if method in REUSED else "new_run",
                               core_contract_sha256=parent.digest(science)) for kind, frame in tables.items()}


def train_point(method, seed, science, federation, device, out=OUT):
    if method not in NEW:
        raise ValueError("Completed pulse controls are read-only")
    with parent.full.development.point_lock(out / "locks" / f"{method}_p{seed}.lock"):
        receipt = read_outcome(method, seed, science, out)
        if receipt is not None:
            print("skip", method, seed, receipt["state"], flush=True)
            return
        import resnet14_transfer_host as torch_host
        if (federation.scope != "test" or federation.manifest != science["partition_manifests"][str(seed)]
                or torch_host.runtime(device) != science["execution"]["runtime"]):
            raise RuntimeError("Baseline confirmation requires the pinned full-training federation/runtime")
        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{method} p{seed} {row['round']}/{ROUNDS}: test CE={evaluation['test_ce']:.6f}, acc={evaluation['test_accuracy']:.4f}", flush=True)
        try:
            result = host.run(federation=federation, method=point_for(method).method, config=config_for(method),
                              seed=parent.train_seed(seed), device=device,
                              checkpoint=parent.BaselineCheckpoint(checkpoint_path(method, seed, out), metadata(method, seed, science)),
                              eval_batch_size=science["execution"]["eval_batch_size"], callback=progress)
        except FloatingPointError as exc:
            parent.atomic_json({"identity": metadata(method, seed, science), "state": "numerical_failure", "error": str(exc),
                                "data_manifest": federation.manifest}, paths(method, seed, out)["receipt"])
            print("numerical_failure", method, seed, str(exc), flush=True)
            return
        save_result(method, seed, science, result, federation.manifest, out)
        verified_tables(method, seed, science, out)


def run(device_name="auto", threads=4, eval_batch_size=256, out=OUT):
    with parent.full.development.point_lock(out / "locks/controller.lock"):
        science, device = prepare(device_name, threads, eval_batch_size, out)
        register_reuse(science, out)
        pending = [s for s in SEEDS if any(read_outcome(m, s, science, out) is None for m in NEW)]
        if pending:
            # Choices, full-training partitions and all twenty read-only controls are pinned first.
            images, labels, test = parent.data_tools.load_arrays(parent.DATA, include_test=True)
            for seed in pending:
                fed, _ = parent.data_tools.federation(images, labels, seed=seed, validation=False, device=device, test=test)
                if fed.scope != "test" or fed.manifest != science["partition_manifests"][str(seed)]:
                    raise RuntimeError("Baseline fit requires the pinned full 50000-image training allocation")
                parent.immutable_json(fed.manifest, out / f"confirmation_partition_p{seed}.json")
                for method in NEW:
                    train_point(method, seed, science, fed, device, out)
                del fed
        analyze(out)


def paired_analysis(runs, targets):
    pairs, effects, target_effects = [], [], []
    event = runs[runs.comparison_id == "event"].set_index("partition_seed").sort_index()
    if event.index.has_duplicates or set(event.index) != set(SEEDS):
        raise RuntimeError("Event common-seed coverage differs")
    definitions = (("accuracy_difference_pp", "final_test_accuracy", False, 100), ("test_ce_difference", "final_test_ce", False, 1),
                   ("unicast_ratio", "unicast_total_bits", True, 1), ("upload_ratio", "uplink_bits", True, 1),
                   ("download_ratio", "unicast_downlink_bits", True, 1), ("broadcast_ratio", "broadcast_total_bits", True, 1))
    for method in METHODS[1:]:
        ref = runs[runs.comparison_id == method].set_index("partition_seed").sort_index()
        if ref.index.has_duplicates or set(ref.index) != set(SEEDS):
            raise RuntimeError("Missing or duplicated matched reference seed")
        for metric, field, ratio, scale in definitions:
            values = event[field] / ref[field] if ratio else scale * (event[field] - ref[field])
            pairs.extend({"reference": method, "partition_seed": int(s), "metric": metric, "value": float(v)} for s, v in values.items())
            effect = {"reference": method, "metric": metric, **parent.interval(values, ratio=ratio),
                      "direction": "Event_over_reference" if ratio else "Event_minus_reference"}
            if ratio:
                effect.update(Event_saving_percent=100 * (1 - effect["estimate"]),
                              Event_saving_ci95_low=100 * (1 - effect["ci95_high"]), Event_saving_ci95_high=100 * (1 - effect["ci95_low"]))
            effects.append(effect)
        for target in parent.TARGETS:
            frame = targets[targets.target_accuracy == target]
            a, b = (frame[frame.comparison_id == m].set_index("partition_seed").sort_index() for m in ("event", method))
            if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(SEEDS) or set(b.index) != set(SEEDS):
                raise RuntimeError("Target coverage differs from the declared common seed set")
            common = a.reached & b.reached
            estimate = parent.interval(a.loc[common, "unicast_total_bits"] / b.loc[common, "unicast_total_bits"], ratio=True)
            target_effects.append({"reference": method, "target_accuracy": target, "n_total": len(SEEDS),
                                   "n_event_reached": int(a.reached.sum()), "n_reference_reached": int(b.reached.sum()),
                                   "n_common_hits": int(common.sum()), **estimate,
                                   "Event_saving_percent": 100 * (1 - estimate["estimate"]),
                                   "Event_saving_ci95_low": 100 * (1 - estimate["ci95_high"]),
                                   "Event_saving_ci95_high": 100 * (1 - estimate["ci95_low"])})
    return pd.DataFrame(pairs), pd.DataFrame(effects), pd.DataFrame(target_effects)


def analyze(out=OUT):
    science = load_contract(out)
    completion, summaries, histories = [], [], []
    round_count, model_count = 0, 0
    for seed in SEEDS:
        for method in METHODS:
            receipt = read_outcome(method, seed, science, out)
            state = receipt["state"] if receipt else "pending"
            completion.append({"method": method, "partition_seed": seed, "state": state,
                               "origin": "verified_reuse" if method in REUSED else "new_run"})
            if state in ("complete", "reused"):
                table = verified_tables(method, seed, science, out)
                summaries.append(table["summary"][table["summary"].variant == MAIN[method]])
                histories.append(table["history"][table["history"].variant == MAIN[method]])
                round_count += len(table["rounds"]); model_count += len(table["evaluation"])
    expected = len(METHODS) * len(SEEDS)
    parent.atomic_json({"expected": expected, "expected_new": len(NEW) * len(SEEDS), "expected_reused": len(REUSED) * len(SEEDS),
                        "results": completion}, out / "confirmation_completion.json")
    if any(r["state"] not in ("complete", "reused") for r in completion):
        raise RuntimeError("All fifty outcomes are required; pending/failed baselines remain in the completion audit")
    expected_models = expected * len(sorted({1, ROUNDS, *range(config_for("dense").eval_stride, ROUNDS + 1, config_for("dense").eval_stride)}))
    if round_count != expected * ROUNDS or model_count != expected_models:
        raise RuntimeError("Round/evaluated-model coverage differs")
    runs = parent.add_regularization_columns(pd.concat(summaries, ignore_index=True), config_for("event").regularization)
    history = pd.concat(histories, ignore_index=True)
    crossings = []
    for (method, seed), curve in history.groupby(["comparison_id", "partition_seed"]):
        curve = curve.sort_values("round").copy(); curve["uplink_packetized_bits"] = curve.uplink_bits
        crossings.extend(parent.target_crossings(curve, targets=parent.TARGETS, horizon=ROUNDS, comparison_id=method, partition_seed=int(seed)))
    targets = pd.DataFrame(crossings)
    pairs, effects, target_effects = paired_analysis(runs, targets)
    fields = ["final_test_accuracy", "final_test_ce", "final_test_worst_class_accuracy", "final_train_objective", "final_train_ce",
              "final_train_l2_penalty", "final_parameter_l2_norm", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"]
    descriptive = runs.groupby("comparison_id", sort=False)[fields].agg(["mean", "std", "count"])
    descriptive.columns = ["_".join(c) for c in descriptive.columns]
    for name, frame in (("confirmation_runs.csv", runs), ("confirmation_history.csv", history), ("endpoint_summary.csv", descriptive.reset_index()),
                        ("paired_per_seed.csv", pairs), ("paired_summary.csv", effects), ("traffic_to_target.csv", targets),
                        ("paired_target_summary.csv", target_effects)):
        parent.atomic_csv(frame, out / name)
    parent.atomic_json({"new_runs": len(NEW) * len(SEEDS), "reused_runs": len(REUSED) * len(SEEDS), "seed_pairs": len(SEEDS),
                        "rounds_verified": round_count, "sampled_model_fingerprints": model_count,
                        "all_outcomes_retained": True, "selection_sha256": parent.digest(science["selection"]),
                        "test_based_tuning_or_stopping": False, "development_checkpoint_reuse": False,
                        "extra_sparse_STC_runs": 0, "unreached_targets_censored": True,
                        "scope": science["protocol"]["scope"]}, out / "analysis_audit.json")
    plot(history, out)
    print(descriptive[[c for c in descriptive if c.startswith(("final_test_accuracy_", "unicast_total_bits_"))]].to_string())
    print(effects.to_string(index=False))
    print(target_effects.to_string(index=False))


def plot(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"event": "#D55E00", "strom": "#0072B2", "dense": "#666666", "ef_topk": "#CC79A7", "stc": "#009E73"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for method in METHODS:
            g = history[history.comparison_id == method].groupby("round")
            acc, ce, bits = g.test_accuracy.mean(), g.test_ce.mean(), g.unicast_total_bits.mean() / 1e9
            axes[0].plot(acc.index, 100 * acc, color=colors[method], label=LABELS[method])
            axes[0].fill_between(acc.index, 100 * (acc - g.test_accuracy.std()), 100 * (acc + g.test_accuracy.std()), color=colors[method], alpha=.12)
            axes[1].plot(bits, 100 * acc, color=colors[method])
            axes[2].plot(ce.index, ce, color=colors[method])
        axes[0].set(xlabel="Communication rounds", ylabel="Mean test accuracy (%)")
        axes[1].set(xlabel="Mean total unicast traffic (Gbit)", ylabel="Mean test accuracy (%)", xscale="log")
        axes[2].set(xlabel="Communication rounds", ylabel="Mean test cross-entropy")
        for axis in axes:
            axis.grid(alpha=.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle("ResNet-14: ten common seeds; thirty new fits, twenty reused pulse outcomes; band = sample SD", fontsize=10)
        fig.tight_layout(rect=(0, .13, 1, .94))
        fig.savefig(out / "confirmation.png", dpi=180)
        fig.savefig(out / "confirmation.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def preflight(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as torch_host
    science = science_spec()
    device = torch_host.configure(device_name, threads)
    pulse.check_runtime(science, device, eval_batch_size)
    print(json.dumps(science["execution"], indent=2))
    print("Verified selections and twenty completed pulse outcomes; 30 new fits / 54000 rounds / 930 new evaluations.")
    print("All ten partition seeds:", SEEDS)
    for method in NEW:
        print(method, POINT_IDS[method], "fresh full-training fit from initialization on every seed")
    print("No extra sparse-STC points, tuning, seed replacement or development checkpoint reuse.")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("ResNet core confirmation not started")
        return
    science = load_contract(out)
    for seed in SEEDS:
        for method in METHODS:
            receipt = read_outcome(method, seed, science, out)
            label = receipt["state"] if receipt else "pending"
            if method in NEW and receipt is None and checkpoint_path(method, seed, out).exists():
                with np.load(checkpoint_path(method, seed, out), allow_pickle=False) as saved:
                    label = "checkpoint:" + str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"])
            print(method, seed, label)


def smoke(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as torch_host
    device = torch_host.configure(device_name, threads)
    fed = parent.synthetic_federation(device, examples=4, evaluation=4)
    for method in NEW:
        cfg = replace(config_for(method), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            full, resumed, native = (Path(folder) / (n + ".npz") for n in ("full", "resumed", "native"))
            kwargs = dict(federation=fed, method=point_for(method).method, config=cfg, seed=parent.train_seed(SEEDS[0]), device=device, eval_batch_size=eval_batch_size)
            checkpoint = lambda p: parent.BaselineCheckpoint(p, {"synthetic_core_confirmation": method})
            a = host.run(**kwargs, checkpoint=checkpoint(full))
            reference = host.run(**kwargs, checkpoint=checkpoint(native), native_reference=True)
            host.run(**kwargs, checkpoint=checkpoint(resumed), stop_after=2)
            b = host.run(**kwargs, checkpoint=checkpoint(resumed))
            with np.load(full, allow_pickle=False) as x, np.load(resumed, allow_pickle=False) as y, np.load(native, allow_pickle=False) as z:
                if set(x.files) != set(y.files) or set(x.files) != set(z.files):
                    raise AssertionError("ResNet smoke checkpoint arrays differ")
                for name in x.files:
                    if name != "metadata":
                        np.testing.assert_array_equal(x[name].view(np.uint8), y[name].view(np.uint8))
                        np.testing.assert_array_equal(x[name].view(np.uint8), z[name].view(np.uint8))
                rng = lambda archive: json.loads(str(archive["metadata"].item()))["progress"]["numpy_rng"]
                if rng(x) != rng(y) or rng(x) != rng(z):
                    raise AssertionError("ResNet smoke NumPy RNG differs")
            fields = [k for k in a["rounds"] if k not in ("training_seconds", "codec_seconds")]
            pd.testing.assert_frame_equal(a["rounds"][fields], b["rounds"][fields], check_exact=True)
            pd.testing.assert_frame_equal(a["evaluation"], b["evaluation"], check_exact=True)
            pd.testing.assert_frame_equal(a["evaluation"], reference["evaluation"], check_exact=True)
        print(method, "synthetic ResNet: exact native/encoded models, feedback, RNG and predictions; exact resumed traffic", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.threads < 1 or args.eval_batch_size < 1:
        parser.error("threads and evaluation batch size must be positive")
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        for pattern in ("test_resnet14_core_confirmation.py", "test_resnet14_baseline.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"], cwd=REPO, env=env, check=True)
        smoke(args.device, args.threads, args.eval_batch_size)
    elif args.command == "smoke":
        smoke(args.device, args.threads, args.eval_batch_size)
    elif args.command == "preflight":
        preflight(args.device, args.threads, args.eval_batch_size)
    elif args.command == "run":
        run(args.device, args.threads, args.eval_batch_size)
    elif args.command == "status":
        status()
    else:
        analyze()


if __name__ == "__main__":
    main()
