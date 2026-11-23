"""Bounded validation-only follow-up at the ResNet-14 calibration boundaries.

python experiments/resnet14_boundary.py verify --device cuda
python experiments/resnet14_boundary.py preflight --device cuda
python experiments/resnet14_boundary.py run --device cuda
git add experiments/results/resnet14_boundary_v1
git commit -m "Add bounded ResNet14 validation boundary check"
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
import numpy as np
import pandas as pd

OUT = REPO / "experiments/results/resnet14_boundary_v1"
SCREEN, FINAL = parent.SCREEN, parent.FINAL
PARENT_RESULTS_COMMIT = "f8e740b1a888eee83200d82f9a92333543b16bba"
PARENT_SELECTION_SHA256 = "0a9f189cf0499975ae4bfa6df14c8ed3e95d8ff556738f91fe53af928c20c7cc"
INCUMBENT_IDS = {"event": "event_t0p01_q0p0025", "strom": "strom_t0p005_g2"}
POINTS = tuple(parent.point("event", t, .00125) for t in (.005, .01, .0125)) + (
    parent.point("strom", .0025, 2.),
    parent.point("strom", .005, 1.),
    parent.point("strom", .0025, 1.),
)
BY_ID = {p.point_id: p for p in (*POINTS, *(parent.BY_ID[k] for k in INCUMBENT_IDS.values()))}


def parent_evidence():
    """Read only the frozen parent's development evidence, never its test results."""
    science = parent.load_contract()
    frozen = parent.frozen_selection(science)
    if (parent.text_hash(parent.OUT / "frozen_selection.json") != PARENT_SELECTION_SHA256
            or frozen["selected"] != {m: [k] for m, k in INCUMBENT_IDS.items()}):
        raise RuntimeError("Expected the frozen ResNet development choices from " + PARENT_RESULTS_COMMIT)
    names = {"protocol.json", "split_indices.csv", "split_manifest.json", "promotion.json", "frozen_selection.json"}
    for horizon, ids in ((SCREEN, parent.BY_ID), (FINAL, frozen["receipt_sha256"])):
        for key in ids:
            files = parent.paths(parent.BY_ID[key], "development", parent.data_tools.DEV_SEED, horizon)
            names.add(files["receipt"].name)
            if horizon == FINAL and key in INCUMBENT_IDS.values():
                names.update(f.name for f in files.values())
    hashes = {str((parent.OUT / name).relative_to(REPO)): parent.text_hash(parent.OUT / name) for name in sorted(names)}
    return science, hashes


def protocol_spec():
    return {
        "campaign": "resnet14_boundary_v1", "architecture": "CIFAR_ResNet14_GroupNorm_Torch",
        "dimension": 175258, "n_clients": 10, "regime": "strong_55pct_dominant",
        "parent_results_commit": PARENT_RESULTS_COMMIT, "parent_selected": INCUMBENT_IDS,
        "development_partition_seed": parent.data_tools.DEV_SEED, "split_seed": parent.data_tools.SPLIT_SEED,
        "development_training_seed": parent.train_seed(parent.data_tools.DEV_SEED),
        "training_examples": 45000, "validation_examples": 5000, "training_diagnostic_examples": 5000,
        "points": [asdict(p) for p in POINTS], "configs": {p.point_id: asdict(parent.config_for(p)) for p in POINTS},
        "screen_rounds": SCREEN, "final_rounds": FINAL, "screen_candidates_per_method": 3,
        "max_promoted_per_method": 1, "max_additional_training_rounds": 6000,
        "max_new_stage_receipts": 8, "max_fresh_trajectories": 6,
        "max_unique_evaluation_points": 106, "new_confirmation_runs": 0,
        "selection_metric": "final_validation_ce", "selection_ties": ["unicast_total_bits", "point_id"],
        "final_selection_scope": "each_incumbent_and_its_one_screen_promoted_new_finalist",
        "incumbent_reuse": "verified_1800_round_development_only_read_only_parent",
        "checkpoint_identity": "1800_round_config_from_start_exact_600_to_1800_continuation",
        "shared_learning_protocol": "unchanged_parent_local_SGD_regularization_initialization_and_codecs",
        "execution": "same_runtime_threads_device_and_evaluation_batch_size_as_parent",
        "official_test_images_loaded": False, "official_test_metrics_read": False,
        "test_based_tuning_or_stopping": False, "automatic_confirmation": False,
        "automatic_further_boundary_extension": False,
        "all_new_screen_candidates_failed": "stop_explicitly_no_fallback_or_extra_candidates",
        "finalist_numerical_failure": "retain_terminal_failure_compare_incumbent_without_extra_promotion",
        "changed_choice_requires_fresh_matched_confirmation": True,
        "unchanged_choice_does_not_add_new_confirmation_seed_pairs": True,
        "scope": "single_development_split_boundary_sensitivity_not_optimal_tuning_or_test_confirmation",
    }


def science_spec():
    prior, hashes = parent_evidence()
    return {
        "protocol": protocol_spec(), "execution": prior["execution"], "data_sha256": prior["data_sha256"],
        "source_hashes": {**prior["source_hashes"], "experiments/resnet14_boundary.py": parent.text_hash(Path(__file__))},
        "parent_contract_sha256": parent.digest(prior), "parent_evidence_sha256": hashes,
    }


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text())
    if science != science_spec():
        raise RuntimeError("Boundary protocol, sources, parent evidence or runtime changed; preserve this campaign")
    return science


def check_runtime(science, device, eval_batch_size):
    import resnet14_transfer_host as host
    if {"runtime": host.runtime(device), "eval_batch_size": eval_batch_size} != science["execution"]:
        raise RuntimeError("Use the recorded parent runtime/device, thread count and evaluation batch size for this boundary check")


def prepare(device_name, threads, eval_batch_size, out=OUT):
    import resnet14_transfer_host as host
    science = science_spec()
    device = host.configure(device_name, threads)
    check_runtime(science, device, eval_batch_size)
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != science:
        raise RuntimeError("Existing boundary campaign differs; preserve the recorded campaign")
    root = parent.data_tools.download_cifar10(parent.DATA)
    if {n: parent.full.development.file_hash(root / n) for n in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR fingerprints differ from the frozen parent")
    images, labels, test = parent.data_tools.load_arrays(parent.DATA, include_test=False)
    if test is not None:
        raise RuntimeError("Official test data are prohibited in this validation-only study")
    fed, _ = parent.data_tools.federation(images, labels, seed=parent.data_tools.DEV_SEED, validation=True, device=device)
    if fed.scope != "validation" or fed.manifest != json.loads((parent.OUT / "split_manifest.json").read_text()):
        raise RuntimeError("Boundary training/validation allocation differs from the frozen parent")
    parent.immutable_json(science, out / "protocol.json")
    return science, device, fed


def checked_tables(p, horizon, science, out=OUT):
    receipt = parent.read_receipt(p, "development", parent.data_tools.DEV_SEED, horizon, science, out)
    if receipt is None:
        raise RuntimeError("Boundary stage is incomplete: " + p.point_id)
    if receipt["state"] != "complete":
        return None
    if receipt["data_manifest"] != json.loads((parent.OUT / "split_manifest.json").read_text()):
        raise RuntimeError("Saved boundary data manifest differs from the parent holdout")
    return parent.verified_tables(p, "development", parent.data_tools.DEV_SEED, horizon, science, out)


def ledger(points, horizon, science, out=OUT):
    for p in points:
        checked_tables(p, horizon, science, out)
    return parent.ledger(points, horizon, science, out)


def decision(frame, science, horizon, out=OUT):
    return {
        "contract_sha256": parent.digest(science), "horizon": horizon,
        "selection_metric": "final_validation_ce", "selected": parent.choose(frame, 1),
        "receipt_sha256": {key: parent.text_hash(parent.paths(BY_ID[key], "development", parent.data_tools.DEV_SEED, horizon, out)["receipt"])
                           for key in frame.point_id}, "official_test_evaluations": 0,
    }


def promoted_points(science, out=OUT):
    screen = ledger(POINTS, SCREEN, science, out)
    promoted = decision(screen, science, SCREEN, out)
    if json.loads((out / "promotion.json").read_text()) != promoted:
        raise RuntimeError("Frozen boundary screen promotion changed")
    points = tuple(BY_ID[promoted["selected"][m][0]] for m in ("event", "strom"))
    return screen, promoted, points


def analyze(out=OUT):
    science = load_contract(out)
    screen, promoted, points = promoted_points(science, out)
    final = ledger(points, FINAL, science, out)
    parent.validate_prefixes(points, science, out)
    prior = parent.load_contract()
    incumbents = tuple(BY_ID[INCUMBENT_IDS[m]] for m in ("event", "strom"))
    old = parent.ledger(incumbents, FINAL, prior, parent.OUT)
    old["origin"], final["origin"] = "parent_incumbent", "boundary_finalist"
    final = final.reindex(columns=old.columns)
    comparison = pd.concat([old, final], ignore_index=True)
    receipts = {}
    for _, row in comparison.iterrows():
        source = parent.OUT if row.origin == "parent_incumbent" else out
        receipts[row.point_id] = parent.text_hash(parent.paths(BY_ID[row.point_id], "development", parent.data_tools.DEV_SEED, FINAL, source)["receipt"])
    selected = parent.choose(comparison, 1)
    changed = {m: selected[m][0] != INCUMBENT_IDS[m] for m in ("event", "strom")}
    frozen = {
        "contract_sha256": parent.digest(science), "selection_metric": "final_validation_ce", "horizon": FINAL,
        "selected": selected, "configs": {m: asdict(parent.config_for(BY_ID[ids[0]])) for m, ids in selected.items()},
        "parent_selected": INCUMBENT_IDS, "changed": changed, "receipt_sha256": receipts,
        "promotion_sha256": parent.text_hash(out / "promotion.json"), "official_test_evaluations": 0,
        "fresh_matched_confirmation_required": any(changed.values()), "confirmation_launched": False,
        "automatic_further_boundary_extension": False,
    }
    parent.immutable_json(frozen, out / "frozen_selection.json")
    changes = []
    for method in ("event", "strom"):
        a, b = old[old.method == method].iloc[0], final[final.method == method].iloc[0]
        changes.append({
            "method": method, "incumbent_id": a.point_id, "new_finalist_id": b.point_id,
            "new_finalist_state": b.state, "incumbent_validation_ce": a.final_validation_ce,
            "new_validation_ce": b.final_validation_ce, "validation_ce_difference": b.final_validation_ce - a.final_validation_ce,
            "incumbent_validation_accuracy": a.final_validation_accuracy, "new_validation_accuracy": b.final_validation_accuracy,
            "new_over_incumbent_unicast_ratio": b.unicast_total_bits / a.unicast_total_bits,
            "selected_id": selected[method][0], "changed": changed[method],
        })
    complete_tables = [checked_tables(p, h, science, out) for h, candidates in ((SCREEN, POINTS), (FINAL, points)) for p in candidates]
    complete = [t for t in complete_tables if t is not None]
    for name, frame in (("screening.csv", screen), ("finalists.csv", final), ("comparison.csv", comparison),
                        ("boundary_summary.csv", pd.DataFrame(changes))):
        parent.atomic_csv(frame, out / name)
    parent.atomic_json({
        "complete_new_stage_receipts": len(complete), "failed_new_stage_receipts": len(complete_tables) - len(complete),
        "reused_parent_incumbents": 2, "verified_new_archived_round_rows": sum(len(t["rounds"]) for t in complete),
        "verified_new_archived_evaluated_models": sum(len(t["evaluation"]) for t in complete),
        "verified_exact_screen_prefixes": int(final.state.eq("complete").sum()),
        "max_unique_additional_training_rounds": 6000, "max_unique_new_evaluation_points": 106,
        "parent_sources_and_development_evidence_verified": True, "official_test_images_or_metrics_read": False,
        "all_new_candidate_outcomes_retained": True, "selection_sha256": parent.digest(frozen),
        "automatic_confirmation": False, "fresh_matched_confirmation_required": any(changed.values()),
        "scope": "single_split_validation_boundary_check_not_predictive_confirmation",
    }, out / "analysis_audit.json")
    print(pd.DataFrame(changes).to_string(index=False))
    print("Fresh matched confirmation required:", any(changed.values()))
    print("Validation-only boundary check complete. No official test evaluation or further search is launched.")
    return frozen


def run(device_name="auto", threads=4, eval_batch_size=256, out=OUT):
    with parent.full.development.point_lock(out / "locks/controller.lock"):
        science, device, fed = prepare(device_name, threads, eval_batch_size, out)
        if fed.scope != "validation":
            raise RuntimeError("Boundary work requires validation scope")
        print("Stage 1: three new settings per method, 600 rounds each", flush=True)
        for p in POINTS:
            parent.train_point(p, "development", parent.data_tools.DEV_SEED, SCREEN, science, fed, device, out)
        screen = ledger(POINTS, SCREEN, science, out)
        parent.atomic_csv(screen, out / "screening.csv")
        parent.immutable_json(decision(screen, science, SCREEN, out), out / "promotion.json")
        _, _, points = promoted_points(science, out)
        print("Stage 2: exact continuation of one new finalist per method to 1800 rounds", flush=True)
        for p in points:
            parent.train_point(p, "development", parent.data_tools.DEV_SEED, FINAL, science, fed, device, out)
        return analyze(out)


def preflight(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as host
    science = science_spec()
    device = host.configure(device_name, threads)
    check_runtime(science, device, eval_batch_size)
    print(json.dumps(science["execution"], indent=2))
    print("Six new validation-only screens; one exact 1800-round finalist per method; two verified reused incumbents.")
    print("Maximum 6000 additional training rounds / 106 evaluation points. No test data, automatic confirmation or further extension.")
    print(pd.DataFrame([asdict(p) for p in POINTS]).to_string(index=False))


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("ResNet boundary check not started")
        return
    science = load_contract(out)
    for p in POINTS:
        labels = []
        for horizon in (SCREEN, FINAL):
            receipt = parent.read_receipt(p, "development", parent.data_tools.DEV_SEED, horizon, science, out)
            if receipt:
                labels.append(f"r{horizon}:{receipt['state']}")
        checkpoint = parent.checkpoint_path(p, "development", parent.data_tools.DEV_SEED, out)
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                labels.append("checkpoint:" + str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"]))
        print(p.point_id, ", ".join(labels) or "pending")
    if (out / "frozen_selection.json").exists():
        frozen = json.loads((out / "frozen_selection.json").read_text())
        print("Selection:", frozen["selected"], "changed:", frozen["changed"])


def smoke(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as host
    device = host.configure(device_name, threads)
    fed = parent.synthetic_federation(device, examples=4, evaluation=4)
    for p in (POINTS[1], POINTS[-1]):
        config = replace(parent.config_for(p), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "full.npz", Path(folder) / "resumed.npz"
            kwargs = dict(federation=fed, method=p.method, config=config, seed=97000, device=device, eval_batch_size=eval_batch_size)
            a = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(left, {"boundary_smoke": p.point_id}))
            host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(right, {"boundary_smoke": p.point_id}), stop_after=2)
            b = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(right, {"boundary_smoke": p.point_id}))
            with np.load(left, allow_pickle=False) as before, np.load(right, allow_pickle=False) as after:
                if set(before.files) != set(after.files):
                    raise AssertionError("Boundary checkpoint arrays differ")
                for name in before.files:
                    if name != "metadata":
                        np.testing.assert_array_equal(before[name].view(np.uint8), after[name].view(np.uint8))
            fields = [k for k in a["rounds"] if k not in ("training_seconds", "codec_seconds")]
            pd.testing.assert_frame_equal(a["rounds"][fields], b["rounds"][fields], check_exact=True)
            pd.testing.assert_frame_equal(a["evaluation"], b["evaluation"], check_exact=True)
    print("Synthetic ResNet boundary settings: exact native codec updates and uninterrupted/resumed model, state, RNG, metrics and traffic.")


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
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_resnet14*.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke(args.device, args.threads, args.eval_batch_size)
    elif args.command in ("status", "analyze"):
        globals()[args.command]()
    else:
        globals()[args.command](args.device, args.threads, args.eval_batch_size)


if __name__ == "__main__":
    main()
