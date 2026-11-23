"""Frozen ResNet Event/Ström confirmation: reuse three pairs, add seven.

python experiments/resnet14_ten_seed.py verify --device cuda
python experiments/resnet14_ten_seed.py preflight --device cuda
python experiments/resnet14_ten_seed.py run --device cuda
git add experiments/results/resnet14_ten_seed_v1
git commit -m "Add ResNet14 ten-seed Event Strom confirmation"
git push origin main
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_transfer as parent
import resnet14_boundary as boundary
from resnet14_boundary_audit import verify_portable_contract, best
import numpy as np
import pandas as pd

OUT = REPO / "experiments/results/resnet14_ten_seed_v1"
ROUNDS = 1800
PILOT_SEEDS = (7100, 7200, 7300)
NEW_SEEDS = tuple(range(7400, 8100, 100))
SEEDS = PILOT_SEEDS + NEW_SEEDS
SCOPES = {"all_10": SEEDS, "additional_7": NEW_SEEDS}
METHODS = ("event", "strom")
POINT_IDS = {"event": "event_t0p01_q0p0025", "strom": "strom_t0p005_g2"}
BOUNDARY_SELECTION_SHA256 = "ead10713fa42e500d6aedb3d15296999c41437ae7899ba19f982e2b862f66d01"
EVIDENCE_COMMIT = "bd729848a9fcb89edd4e40f71ed09c7eb461372c"


def point_for(method):
    return parent.BY_ID[POINT_IDS[method]]


def protocol_spec():
    if len(SEEDS) != 10 or len(set(SEEDS)) != 10 or set(NEW_SEEDS) & set(range(2500, 7400, 100)):
        raise RuntimeError("Expected ten distinct pairs and seven fresh partition seeds")
    return {
        "campaign": "resnet14_ten_seed_v1", "architecture": "CIFAR_ResNet14_GroupNorm_Torch",
        "dimension": 175258, "n_clients": 10, "regime": "strong_55pct_dominant", "rounds": ROUNDS,
        "partition_seeds": list(SEEDS), "training_seeds": [parent.train_seed(s) for s in SEEDS],
        "reused_pilot_partition_seeds": list(PILOT_SEEDS), "additional_partition_seeds": list(NEW_SEEDS),
        "configs": {m: asdict(parent.config_for(point_for(m))) for m in METHODS},
        "training_examples": 50000, "test_examples": 10000, "training_diagnostic_examples": 5000,
        "eval_stride": 60, "initialization_seed": 7777, "matched_minibatches_and_partitions": True,
        "new_training_runs": 14, "reused_pilot_runs": 6, "expected_outcomes": 20,
        "additional_training_rounds": 25200, "additional_evaluation_points": 434,
        "total_verified_rounds": 36000, "total_evaluated_model_fingerprints": 620,
        "scopes": {k: list(v) for k, v in SCOPES.items()},
        "primary_endpoint": "Event_minus_Strom_round_1800_test_accuracy_percentage_points",
        "secondary_endpoints": ["test_CE", "training_CE_objective_L2_diagnostics", "upload", "download",
                                "conservative_unicast_primary_logical_broadcast_supplementary", "fixed_accuracy_target_cost_and_coverage"],
        "accuracy_targets": list(parent.TARGETS), "target_rule": "first_saved_crossing_no_interpolation",
        "unreached_targets": "censored_at_1800_ratios_only_on_common_hits_report_full_coverage",
        "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_on_log_scale",
        "selection": "unchanged_validation_frozen_choices_after_bounded_boundary_check",
        "development_checkpoint_reuse": False, "test_based_selection_or_early_stopping": False,
        "all_fourteen_new_runs_even_if_Event_loses": True, "automatic_retuning_or_seed_replacement": False,
        "numerical_failures": "terminal_receipts_retained_final_analysis_requires_all_twenty_outcomes",
        "execution": "sequential_one_GPU_same_runtime_as_pilot_numeric_atomic_checkpoints",
        "reuse_policy": "six_verified_read_only_pilot_references_never_relabelled_as_new_runs",
        "path_format": "repository_relative_POSIX_new_provenance_preserve_original_source_digests",
        "seed_freshness_snapshot_commit": EVIDENCE_COMMIT,
        "uncertainty_scope": "partition_SGD_variation_fixed_initialization_configs_previously_inspected_CIFAR10_test_set",
        "equivalence_or_global_SOTA_claim": False,
    }


def selection_provenance():
    recorded = json.loads((boundary.OUT / "protocol.json").read_text())
    bscience = verify_portable_contract(recorded, boundary.science_spec())
    screen, _, finalists = boundary.promoted_points(bscience)
    final = boundary.ledger(finalists, boundary.FINAL, bscience)
    parent.validate_prefixes(finalists, bscience, boundary.OUT)
    prior = parent.load_contract()
    old = parent.ledger(tuple(point_for(m) for m in METHODS), ROUNDS, prior, parent.OUT)
    frozen_file = boundary.OUT / "frozen_selection.json"
    frozen = json.loads(frozen_file.read_text())
    expected = {m: [POINT_IDS[m]] for m in METHODS}
    receipts = {row.point_id: parent.text_hash(parent.paths(boundary.BY_ID[row.point_id], "development", 7000, ROUNDS, source)["receipt"])
                for frame, source in ((old, parent.OUT), (final, boundary.OUT)) for _, row in frame.iterrows()}
    if (parent.text_hash(frozen_file) != BOUNDARY_SELECTION_SHA256 or best(pd.concat([old, final])) != expected
            or frozen["selected"] != expected or frozen["receipt_sha256"] != receipts
            or frozen["contract_sha256"] != parent.digest(bscience) or any(frozen["changed"].values())
            or frozen["configs"] != {m: asdict(parent.config_for(point_for(m))) for m in METHODS}):
        raise RuntimeError("Expected unchanged frozen boundary choices; do not retune this confirmation")
    files = {p.replace("\\", "/"): sha for p, sha in bscience["parent_evidence_sha256"].items()}
    names = ("protocol.json", "promotion.json", "frozen_selection.json", "screening.csv", "finalists.csv",
             "comparison.csv", "boundary_summary.csv", "analysis_audit.json")
    paths = [boundary.OUT / n for n in names]
    for horizon, points in ((boundary.SCREEN, boundary.POINTS), (boundary.FINAL, finalists)):
        for p in points:
            paths.extend(parent.paths(p, "development", 7000, horizon, boundary.OUT).values())
    files.update({p.relative_to(REPO).as_posix(): parent.text_hash(p) for p in paths})
    return prior, bscience, {"parent_contract_sha256": parent.digest(prior), "boundary_contract_sha256": parent.digest(bscience),
                            "selected": expected, "configs": frozen["configs"], "files": files}


def reference_key(method, seed):
    return f"{method}_p{seed}"


def reference_files(method, seed):
    return {**parent.paths(point_for(method), "confirmation", seed, ROUNDS, parent.OUT),
            "partition": parent.OUT / f"confirmation_partition_p{seed}.json"}


def fingerprint_reference(method, seed):
    return {kind: {"path": p.relative_to(REPO).as_posix(), "sha256": parent.text_hash(p)}
            for kind, p in reference_files(method, seed).items()}


def science_spec():
    # Freeze and verify validation choices before reading any pilot test metrics.
    prior, bscience, selected = selection_provenance()
    labels = pd.read_csv(parent.OUT / "split_indices.csv")["class"].to_numpy()
    manifests = {str(s): parent.data_tools.allocation(labels, seed=s, validation=False)[-1] for s in SEEDS}
    references = {}
    for seed in PILOT_SEEDS:
        for method in METHODS:
            p = point_for(method)
            parent.verified_tables(p, "confirmation", seed, ROUNDS, prior, parent.OUT)
            receipt = parent.read_receipt(p, "confirmation", seed, ROUNDS, prior, parent.OUT)
            sidecar = json.loads((parent.OUT / f"confirmation_partition_p{seed}.json").read_text())
            if receipt["data_manifest"] != manifests[str(seed)] or sidecar != manifests[str(seed)]:
                raise RuntimeError("Pilot reuse requires the exact full-training partition")
            references[reference_key(method, seed)] = fingerprint_reference(method, seed)
    sources = {**bscience["source_hashes"]}
    for name in ("experiments/resnet14_boundary_audit.py", "experiments/resnet14_ten_seed.py"):
        sources[name] = parent.text_hash(REPO / name)
    return {"protocol": protocol_spec(), "selection": selected, "execution": prior["execution"],
            "data_sha256": prior["data_sha256"], "source_hashes": sources,
            "reference_contract_sha256": parent.digest(prior), "references": references, "partition_manifests": manifests}


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text())
    if science != science_spec():
        raise RuntimeError("Ten-seed protocol, sources, selection, runtime or reused evidence changed")
    return science


def check_runtime(science, device, eval_batch_size):
    import resnet14_transfer_host as host
    if {"runtime": host.runtime(device), "eval_batch_size": eval_batch_size} != science["execution"]:
        raise RuntimeError("Use the recorded pilot runtime/device, thread count and evaluation batch size")


def prepare(device_name, threads, eval_batch_size, out=OUT):
    import resnet14_transfer_host as host
    science = science_spec()
    device = host.configure(device_name, threads)
    check_runtime(science, device, eval_batch_size)
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != science:
        raise RuntimeError("Existing ten-seed campaign differs; preserve its recorded identity")
    root = parent.data_tools.download_cifar10(parent.DATA)
    if {n: parent.full.development.file_hash(root / n) for n in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR fingerprints differ from the frozen pilot")
    parent.immutable_json(science, out / "protocol.json")
    return science, device


def reuse_path(method, seed, out=OUT):
    return out / f"{method}_p{seed}_reuse.json"


def reuse_receipt(method, seed, science):
    return {"identity": parent.metadata(point_for(method), "confirmation", seed, ROUNDS, science), "state": "reused",
            "source_contract_sha256": science["reference_contract_sha256"],
            "source": science["references"][reference_key(method, seed)]}


def write_reuse(method, seed, science, out=OUT):
    parent.immutable_json(reuse_receipt(method, seed, science), reuse_path(method, seed, out))


def read_outcome(method, seed, science, out=OUT):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown method/seed outcome")
    if seed in PILOT_SEEDS:
        path = reuse_path(method, seed, out)
        if not path.exists():
            return None
        receipt = json.loads(path.read_text())
        if receipt != reuse_receipt(method, seed, science) or fingerprint_reference(method, seed) != receipt["source"]:
            raise RuntimeError("Pilot reuse receipt or source fingerprints changed")
        return receipt
    return parent.read_receipt(point_for(method), "confirmation", seed, ROUNDS, science, out)


def verified_tables(method, seed, science, out=OUT):
    receipt = read_outcome(method, seed, science, out)
    if receipt is None or receipt["state"] not in ("complete", "reused"):
        raise RuntimeError("Incomplete ten-seed outcome")
    if seed in PILOT_SEEDS:
        prior = parent.load_contract()
        if parent.digest(prior) != science["reference_contract_sha256"]:
            raise RuntimeError("Pilot source contract differs")
        source, source_science = parent.OUT, prior
    else:
        source, source_science = out, science
    original = parent.read_receipt(point_for(method), "confirmation", seed, ROUNDS, source_science, source)
    sidecar = json.loads((source / f"confirmation_partition_p{seed}.json").read_text())
    if original["data_manifest"] != science["partition_manifests"][str(seed)] or sidecar != science["partition_manifests"][str(seed)]:
        raise RuntimeError("Confirmation requires the pinned full-training partition and diagnostic set")
    table = parent.verified_tables(point_for(method), "confirmation", seed, ROUNDS, source_science, source)
    return {kind: frame.assign(evidence_origin="reused_pilot" if seed in PILOT_SEEDS else "new_additional",
                               source_contract_sha256=parent.digest(source_science), analysis_contract_sha256=parent.digest(science))
            for kind, frame in table.items()}


def run(device_name="auto", threads=4, eval_batch_size=256, out=OUT):
    with parent.full.development.point_lock(out / "locks/controller.lock"):
        science, device = prepare(device_name, threads, eval_batch_size, out)
        for seed in PILOT_SEEDS:
            for method in METHODS:
                write_reuse(method, seed, science, out)
        pending = [s for s in NEW_SEEDS if any(read_outcome(m, s, science, out) is None for m in METHODS)]
        if pending:
            # Validation choices and six pilot references are pinned before official test images are parsed.
            images, labels, test = parent.data_tools.load_arrays(parent.DATA, include_test=True)
            for seed in pending:
                fed, _ = parent.data_tools.federation(images, labels, seed=seed, validation=False, device=device, test=test)
                if fed.scope != "test" or fed.manifest != science["partition_manifests"][str(seed)]:
                    raise RuntimeError("New confirmation federation differs from its pinned full-training allocation")
                parent.immutable_json(fed.manifest, out / f"confirmation_partition_p{seed}.json")
                for method in METHODS:
                    parent.train_point(point_for(method), "confirmation", seed, ROUNDS, science, fed, device, out)
                del fed
        analyze(out)


def analyze(out=OUT):
    science = load_contract(out)
    completion, summaries, histories = [], [], []
    rounds, models = 0, 0
    for seed in SEEDS:
        for method in METHODS:
            receipt = read_outcome(method, seed, science, out)
            state = receipt["state"] if receipt else "pending"
            completion.append({"method": method, "partition_seed": seed, "state": state,
                               "origin": "reused_pilot" if seed in PILOT_SEEDS else "new_additional"})
            if state in ("reused", "complete"):
                table = verified_tables(method, seed, science, out)
                summaries.append(table["summary"]); histories.append(table["history"])
                rounds += len(table["rounds"]); models += len(table["evaluation"])
    parent.atomic_json({"expected": 20, "expected_new": 14, "expected_reused": 6, "results": completion}, out / "confirmation_completion.json")
    if any(r["state"] not in ("complete", "reused") for r in completion):
        raise RuntimeError("All ten pairs are required; pending or failed outcomes cannot be omitted")
    summary, history = pd.concat(summaries, ignore_index=True), pd.concat(histories, ignore_index=True)
    runs = parent.add_regularization_columns(summary[summary.variant == "both_codecs"].copy(), parent.BASE.regularization)
    a, b = (runs[runs.method == m].set_index("partition_seed").sort_index() for m in METHODS)
    if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(SEEDS) or set(b.index) != set(SEEDS):
        raise RuntimeError("Missing or duplicated matched seed pair")
    definitions = (("accuracy_difference_pp", "final_test_accuracy", False, 100), ("test_ce_difference", "final_test_ce", False, 1),
                   ("unicast_ratio", "unicast_total_bits", True, 1), ("upload_ratio", "uplink_bits", True, 1),
                   ("download_ratio", "unicast_downlink_bits", True, 1), ("broadcast_ratio", "broadcast_total_bits", True, 1))
    pairs, effects = [], []
    for metric, field, ratio, scale in definitions:
        values = a[field] / b[field] if ratio else scale * (a[field] - b[field])
        pairs.extend({"partition_seed": int(s), "metric": metric, "value": float(v),
                      "origin": "reused_pilot" if s in PILOT_SEEDS else "new_additional"} for s, v in values.items())
        for scope, seeds in SCOPES.items():
            effects.append({"scope": scope, "metric": metric, **parent.interval(values.loc[list(seeds)], ratio=ratio),
                            "direction": "Event_over_Strom" if ratio else "Event_minus_Strom"})
    crossings = []
    main_history = history[history.variant == "both_codecs"]
    for (method, seed), curve in main_history.groupby(["method", "partition_seed"]):
        curve = curve.sort_values("round").copy(); curve["uplink_packetized_bits"] = curve.uplink_bits
        crossings.extend(parent.target_crossings(curve, targets=parent.TARGETS, horizon=ROUNDS, comparison_id=method, partition_seed=int(seed)))
    targets, target_summary = pd.DataFrame(crossings), []
    for scope, seeds in SCOPES.items():
        for target, frame in targets[targets.partition_seed.isin(seeds)].groupby("target_accuracy"):
            x, y = (frame[frame.comparison_id == m].set_index("partition_seed").sort_index() for m in METHODS)
            common = x.reached & y.reached
            target_summary.append({"scope": scope, "target_accuracy": target, "n_total": len(seeds),
                                   "n_event_reached": int(x.reached.sum()), "n_strom_reached": int(y.reached.sum()),
                                   "n_common_hits": int(common.sum()),
                                   **parent.interval(x.loc[common, "unicast_total_bits"] / y.loc[common, "unicast_total_bits"], ratio=True)})
    for name, frame in (("confirmation_runs.csv", runs), ("confirmation_history.csv", history),
                        ("paired_per_seed.csv", pd.DataFrame(pairs)), ("paired_summary.csv", pd.DataFrame(effects)),
                        ("traffic_to_target.csv", targets), ("paired_target_summary.csv", pd.DataFrame(target_summary))):
        parent.atomic_csv(frame, out / name)
    parent.atomic_json({"new_runs": 14, "reused_runs": 6, "seed_pairs": 10, "additional_pairs": 7,
                        "rounds_verified": rounds, "sampled_model_fingerprints": models,
                        "all_pairs_retained": True, "test_based_selection_or_stopping": False,
                        "selection_sha256": parent.digest(science["selection"]), "scopes": list(SCOPES),
                        "unreached_targets_censored": True, "scope": "fixed_configurations_ten_matched_ResNet_pairs_including_three_inspected_pilot_pairs"},
                       out / "analysis_audit.json")
    plot(main_history, out)
    print(pd.DataFrame(effects).to_string(index=False))
    print(pd.DataFrame(target_summary).to_string(index=False))


def plot(history, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for method, color in (("event", "#D55E00"), ("strom", "#0072B2")):
        g = history[history.method == method].groupby("round")
        acc, ce, bits = g.test_accuracy.mean(), g.test_ce.mean(), g.unicast_total_bits.mean()
        axes[0].plot(acc.index, 100 * acc, color=color, label=method)
        axes[0].fill_between(acc.index, 100 * (acc - g.test_accuracy.std()), 100 * (acc + g.test_accuracy.std()), color=color, alpha=.15)
        axes[1].plot(bits / 1e6, 100 * acc, color=color, label=method)
        axes[2].plot(ce.index, ce, color=color, label=method)
    axes[0].set(xlabel="Communication rounds", ylabel="Mean test accuracy (%)")
    axes[1].set(xlabel="Mean total unicast traffic (Mbit)", ylabel="Mean test accuracy (%)")
    axes[2].set(xlabel="Communication rounds", ylabel="Mean test cross-entropy")
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle("ResNet-14: ten matched pairs (three reused, seven additional); accuracy band = sample SD")
    fig.tight_layout(); fig.savefig(out / "confirmation.png", dpi=180)
    fig.savefig(out / "confirmation.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def preflight(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as host
    science = science_spec()
    device = host.configure(device_name, threads)
    check_runtime(science, device, eval_batch_size)
    print(json.dumps(science["execution"], indent=2))
    print("Six verified pilot outcomes reused; fourteen new runs, 25200 additional rounds, sequential one-device execution.")
    print("New partition seeds:", NEW_SEEDS, "new SGD seeds:", tuple(parent.train_seed(s) for s in NEW_SEEDS))
    print("Fixed Event threshold .01 / quantum .0025; Strom threshold .005 / native gain .1.")
    print("Automatic final analysis requires all twenty outcomes; all-ten and prospective-seven summaries are separate.")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("ResNet ten-seed confirmation not started")
        return
    science = load_contract(out)
    for seed in SEEDS:
        for method in METHODS:
            receipt = read_outcome(method, seed, science, out)
            label = receipt["state"] if receipt else "pending"
            checkpoint = parent.checkpoint_path(point_for(method), "confirmation", seed, out)
            if receipt is None and seed in NEW_SEEDS and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    label = "checkpoint:" + str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"])
            print(method, seed, label)


def smoke(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as host
    device = host.configure(device_name, threads)
    fed = parent.synthetic_federation(device, examples=4, evaluation=4)
    for method in METHODS:
        config = replace(parent.config_for(point_for(method)), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "full.npz", Path(folder) / "resumed.npz"
            kwargs = dict(federation=fed, method=method, config=config, seed=parent.train_seed(NEW_SEEDS[0]), device=device, eval_batch_size=eval_batch_size)
            a = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(left, {"synthetic_ten_seed": method}))
            host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(right, {"synthetic_ten_seed": method}), stop_after=2)
            b = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(right, {"synthetic_ten_seed": method}))
            with np.load(left, allow_pickle=False) as before, np.load(right, allow_pickle=False) as after:
                if set(before.files) != set(after.files):
                    raise AssertionError("Ten-seed smoke checkpoint arrays differ")
                for name in before.files:
                    if name != "metadata":
                        np.testing.assert_array_equal(before[name].view(np.uint8), after[name].view(np.uint8))
            fields = [k for k in a["rounds"] if k not in ("training_seconds", "codec_seconds")]
            pd.testing.assert_frame_equal(a["rounds"][fields], b["rounds"][fields], check_exact=True)
            pd.testing.assert_frame_equal(a["evaluation"], b["evaluation"], check_exact=True)
    print("Synthetic selected ResNet configurations: native codec fidelity and exact uninterrupted/resumed model, state, RNG, metrics and traffic.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.threads < 1 or args.eval_batch_size < 1:
        parser.error("threads and eval batch size must be positive")
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_resnet14*.py", "-v"], cwd=REPO, env=env, check=True)
        smoke(args.device, args.threads, args.eval_batch_size)
    elif args.command in ("status", "analyze"):
        globals()[args.command]()
    else:
        globals()[args.command](args.device, args.threads, args.eval_batch_size)


if __name__ == "__main__":
    main()
