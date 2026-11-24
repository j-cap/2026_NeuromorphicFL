"""Validation-only ResNet calibration of Dense FedAvg, STC and EF-TopK.

python experiments/resnet14_baseline_development.py verify --device cuda
python experiments/resnet14_baseline_development.py preflight --device cuda
python experiments/resnet14_baseline_development.py run --device cuda
git add experiments/results/resnet14_baseline_development_v1
git commit -m "Add ResNet14 baseline validation calibration"
git push origin main
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_transfer as parent
import resnet14_ten_seed as pulse
import resnet14_baseline_host as host
import numpy as np
import pandas as pd

OUT = REPO / "experiments/results/resnet14_baseline_development_v1"
METHODS = ("dense", "stc", "ef_topk")
SCREEN, FINAL = 600, 1800


@dataclass(frozen=True)
class Point:
    point_id: str
    method: str
    gain: float
    fraction: float = 1.
    download_fraction: float = 1.
    exponent: float = 0.


def make_point(method, gain, fraction=1., download=None, exponent=0.):
    label = lambda x: f"{x:g}".replace(".", "p")
    down = fraction if download is None else download
    suffix = (f"g{label(gain)}_a{label(exponent)}" if method == "dense" else
              f"u{label(fraction)}_d{label(down)}_g{label(gain)}" if method == "stc" else
              f"p{label(fraction)}_g{label(gain)}")
    return Point(method + "_" + suffix, method, float(gain), float(fraction), float(down), float(exponent))


POINTS = (
    tuple(make_point("dense", g, exponent=a) for g in (.25, .5, 1., 1.5, 2., 4.) for a in (0., .2, .3))
    + tuple(make_point("stc", g, p) for p in (.01, .05, .25, 1.) for g in (.5, 1., 2., 4.))
    + (make_point("stc", 1., .01, download=1.), make_point("stc", 1., .25, download=1.))
    + tuple(make_point("ef_topk", g, p) for p in (.01, .025, .05, .1) for g in (.5, 1., 2., 4.))
    + (make_point("ef_topk", 1., .005), make_point("ef_topk", 1., .25))
)
BY_ID = {p.point_id: p for p in POINTS}


def config_for(p):
    return replace(parent.BASE, rounds=FINAL, server_gain=p.gain, server_gain_exponent=p.exponent,
                   server_gain_scale=100., stc_upload_fraction=p.fraction, stc_download_fraction=p.download_fraction,
                   topk_fraction=p.fraction)


def protocol_spec():
    return {
        "campaign": "resnet14_baseline_development_v1", "architecture": "CIFAR_ResNet14_GroupNorm_Torch",
        "dimension": 175258, "n_clients": 10, "development_partition_seed": 7000, "split_seed": 17000,
        "training_seed": 97000, "training_examples": 45000, "validation_examples": 5000,
        "training_diagnostic_examples": 5000, "initialization_seed": 7777,
        "points": [asdict(p) for p in POINTS], "configs": {p.point_id: asdict(config_for(p)) for p in POINTS},
        "screen_rounds": SCREEN, "final_rounds": FINAL, "screen_candidates_per_method": 18,
        "max_promoted_per_method": 3, "max_new_stage_receipts": 63,
        "max_unique_training_rounds": 43200, "max_unique_evaluation_points": 774,
        "budget_reference": "Event_and_Strom_each_used_18_screens_and_3_continuations_across_transfer_and_boundary_studies",
        "budget_scope": "same_total_new_ResNet_screen_and_continuation_work_not_identical_search_order_or_geometry",
        "selection_metric": "final_validation_ce", "selection_ties": ["unicast_total_bits", "point_id"],
        "final_selection_scope": "best_of_three_screen_promoted_finalists_per_method",
        "STC_operator": "per_parameter_tensor_exact_k_mean_magnitude_ternary_with_both_error_feedback_states",
        "STC_grid": "16_symmetric_fraction_gain_points_plus_two_full_server_fraction_anchors",
        "EF_operator": "global_round_p_times_dimension_TopK_weighted_delta_client_error_feedback_float32_values",
        "EF_grid": "16_fraction_gain_points_plus_0p5pct_and_25pct_at_gain1_identity_fraction_is_verification_only",
        "Dense_grid": "six_initial_server_gains_times_three_decay_exponents_scale100",
        "shared_learning_fixed": ["local_steps", "local_lr", "batch_size", "regularization", "initialization", "data", "SGD_seed"],
        "server_arithmetic": "native_baseline_NumPy_float32_with_normalized_float64_weights_Torch_local_SGD_and_model_application",
        "codec": host.WIRE_PROTOCOL, "primary_variants": host.MAIN,
        "STC_wire": "direct_existing_SCL1_bytes_proven_equal_to_native_Rice_then_lossless_reencode_no_requantization",
        "accounting": "actual_frames_initialization_requests_recipient_replication_conservative_unicast",
        "official_test_images_loaded": False, "saved_official_test_metrics_read": False,
        "new_confirmation_runs": 0, "automatic_confirmation": False, "automatic_boundary_extension": False,
        "numerical_failure": "terminal_receipt_retained_stop_if_no_finite_candidate_no_replacement_settings",
        "checkpoint_identity": "1800_round_configuration_from_start_exact_600_to_1800_continuation",
        "scope": "single_validation_split_bounded_calibration_then_stop_for_analysis_not_predictive_confirmation",
    }


def science_spec():
    prior, previous, selection = pulse.selection_provenance()  # Validation files only; no pilot test scores.
    sources = {**previous["source_hashes"]}
    for name in ("experiments/resnet14_baseline_development.py", "experiments/resnet14_baseline_host.py",
                 "experiments/core_float_host.py", "experiments/resnet14_ten_seed.py", "experiments/resnet14_boundary_audit.py"):
        sources[name] = parent.text_hash(REPO / name)
    return {"protocol": protocol_spec(), "source_hashes": sources, "execution": prior["execution"],
            "data_sha256": prior["data_sha256"], "pulse_validation_reference": selection}


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text())
    if science != science_spec():
        raise RuntimeError("Baseline calibration sources/configuration/validation references changed")
    return science


def prepare(device_name, threads, eval_batch_size, out=OUT):
    import resnet14_transfer_host as torch_host
    science = science_spec()
    device = torch_host.configure(device_name, threads)
    if {"runtime": torch_host.runtime(device), "eval_batch_size": eval_batch_size} != science["execution"]:
        raise RuntimeError("Use the recorded pulse-study environment/device, threads and evaluation batch size")
    if (out / "protocol.json").exists() and json.loads((out / "protocol.json").read_text()) != science:
        raise RuntimeError("Existing baseline calibration differs; preserve the campaign")
    root = parent.data_tools.download_cifar10(parent.DATA)
    if {n: parent.full.development.file_hash(root / n) for n in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR fingerprints differ")
    images, labels, test = parent.data_tools.load_arrays(parent.DATA, include_test=False)
    if test is not None:
        raise RuntimeError("Official test arrays are prohibited during calibration")
    fed, _ = parent.data_tools.federation(images, labels, seed=7000, validation=True, device=device)
    if fed.scope != "validation" or fed.manifest != json.loads((parent.OUT / "split_manifest.json").read_text()):
        raise RuntimeError("Baseline validation allocation differs from the pulse study")
    parent.immutable_json(science, out / "protocol.json")
    return science, device, fed


def paths(p, horizon, out=OUT):
    stem = f"{p.point_id}_development_r{horizon}"
    return {k: out / f"{stem}_{k}.{'json' if k == 'receipt' else 'csv'}" for k in ("rounds", "evaluation", "summary", "history", "receipt")}


def checkpoint_path(p, out=OUT):
    return out / "checkpoints" / (p.point_id + ".npz")


def provenance(p, science):
    return {"point_id": p.point_id, "method": p.method, "phase": "development", "partition_seed": 7000,
            "train_seed": 97000, "config_sha256": parent.digest(asdict(config_for(p))), "contract_sha256": parent.digest(science)}


def metadata(p, horizon, science):
    return {**provenance(p, science), "stage_horizon": horizon}


def read_receipt(p, horizon, science, out=OUT):
    files = paths(p, horizon, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text())
    if receipt["identity"] != metadata(p, horizon, science):
        raise RuntimeError("Baseline stage identity differs")
    if receipt["state"] == "complete":
        for k in ("rounds", "evaluation", "summary", "history"):
            if parent.text_hash(files[k]) != receipt["sha256"][k]:
                raise RuntimeError("Baseline stage evidence changed")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unexpected baseline stage state")
    return receipt


def trajectory_tables(result, science):
    rounds, evaluation = result["rounds"], result["evaluation"]
    kwargs = {"n_clients": science["protocol"]["n_clients"], "dimension": science["protocol"]["dimension"]}
    summary = host.traffic_summary(rounds, **kwargs)
    history = pd.concat([host.traffic_summary(rounds[rounds["round"] <= int(row["round"])], **kwargs).assign(**row.to_dict())
                         for _, row in evaluation.iterrows()], ignore_index=True)
    for field, value in evaluation.iloc[-1].items():
        if field not in ("round", "model_sha256"):
            summary["final_" + field] = value
    summary["model_sha256"] = evaluation.iloc[-1].model_sha256
    return {"rounds": rounds, "evaluation": evaluation, "summary": summary, "history": history}


def verified_tables(p, horizon, science, out=OUT):
    receipt = read_receipt(p, horizon, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError("Incomplete baseline calibration point")
    if receipt["data_manifest"] != json.loads((parent.OUT / "split_manifest.json").read_text()):
        raise RuntimeError("Saved baseline split manifest differs")
    tables = {k: pd.read_csv(path, float_precision="round_trip", dtype={"model_sha256": str}) for k, path in paths(p, horizon, out).items() if k != "receipt"}
    for frame in tables.values():
        for key, value in metadata(p, horizon, science).items():
            if not frame[key].eq(value).all():
                raise RuntimeError("Baseline table metadata differs")
    r, e = tables["rounds"], tables["evaluation"]
    cfg = config_for(p)
    fields = ("train_objective", "train_ce", "validation_ce", "validation_accuracy", "validation_macro_accuracy", "validation_worst_class_accuracy")
    if (r["round"].tolist() != list(range(1, horizon + 1)) or not r.method.eq(p.method).all()
            or not r.wire_protocol.eq(host.WIRE_PROTOCOL).all() or not r.roundtrip_verified.all()
            or e["round"].tolist() != sorted({1, horizon, *range(cfg.eval_stride, horizon + 1, cfg.eval_stride)})
            or any(c.startswith("test_") for c in e) or not np.isfinite(e[list(fields)]).all().all()
            or not e[list(fields[3:])].apply(lambda s: s.between(0, 1)).all().all()
            or not np.array_equal(e.model_sha256, r.set_index("round").loc[e["round"], "model_sha256"])):
        raise RuntimeError("Baseline horizon, codec or validation/model coverage differs")
    rebuilt = trajectory_tables({"rounds": r, "evaluation": e[["round", *fields, "model_sha256"]]}, science)
    for kind in ("summary", "history"):
        for key in rebuilt[kind]:
            a, b = tables[kind][key], rebuilt[kind][key]
            exact = key.endswith("bits") or key in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1e-12, atol=1e-12)):
                raise RuntimeError("Reconstructed baseline accounting/metrics differ")
    return tables


def train_point(p, horizon, science, fed, device, out=OUT):
    files = paths(p, horizon, out)
    with parent.full.development.point_lock(out / "locks" / (files["receipt"].stem + ".lock")):
        receipt = read_receipt(p, horizon, science, out)
        if receipt is not None:
            print("skip", p.point_id, horizon, receipt["state"], flush=True)
            return
        if horizon == FINAL and not checkpoint_path(p, out).exists():
            raise RuntimeError("Finalist requires its local exact 600-round checkpoint")
        import resnet14_transfer_host as torch_host
        if fed.scope != "validation" or torch_host.runtime(device) != science["execution"]["runtime"]:
            raise RuntimeError("Baseline runtime or validation scope differs")
        def progress(row, evaluation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{p.point_id} {row['round']}/{horizon}: validation CE={evaluation['validation_ce']:.6f}, acc={evaluation['validation_accuracy']:.4f}", flush=True)
        try:
            result = host.run(federation=fed, method=p.method, config=config_for(p), seed=97000, device=device,
                              checkpoint=parent.BaselineCheckpoint(checkpoint_path(p, out), provenance(p, science)),
                              eval_batch_size=science["execution"]["eval_batch_size"], stop_after=horizon, callback=progress)
        except FloatingPointError as exc:
            parent.atomic_json({"identity": metadata(p, horizon, science), "state": "numerical_failure", "error": str(exc)}, files["receipt"])
            return
        tables = trajectory_tables(result, science)
        for k, frame in tables.items():
            parent.atomic_csv(frame.assign(**metadata(p, horizon, science)), files[k])
        parent.atomic_json({"identity": metadata(p, horizon, science), "state": "complete", "data_manifest": fed.manifest,
                            "sha256": {k: parent.text_hash(files[k]) for k in tables}}, files["receipt"])


def ledger(points, horizon, science, out=OUT):
    rows = []
    for p in points:
        receipt = read_receipt(p, horizon, science, out)
        if receipt is None:
            raise RuntimeError("Baseline calibration grid incomplete: " + p.point_id)
        row = {**asdict(p), "state": receipt["state"], "error": receipt.get("error", "")}
        if receipt["state"] == "complete":
            row.update(verified_tables(p, horizon, science, out)["summary"].iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def choose(frame, count):
    selected = {}
    for method in METHODS:
        rows = frame[(frame.method == method) & (frame.state == "complete")]
        if rows.empty:
            raise RuntimeError("All " + method + " candidates failed; no automatic fallback")
        if not np.isfinite(rows.final_validation_ce).all():
            raise RuntimeError("Nonfinite baseline validation selection score")
        selected[method] = rows.sort_values(["final_validation_ce", "unicast_total_bits", "point_id"]).head(count).point_id.tolist()
    return selected


def decision(frame, count, horizon, science, out=OUT):
    return {"contract_sha256": parent.digest(science), "horizon": horizon, "selection_metric": "final_validation_ce",
            "selected": choose(frame, count), "receipt_sha256": {key: parent.text_hash(paths(BY_ID[key], horizon, out)["receipt"]) for key in frame.point_id},
            "official_test_evaluations": 0}


def validate_prefixes(points, science, out=OUT):
    for p in points:
        receipt = read_receipt(p, FINAL, science, out)
        if receipt is None or receipt["state"] != "complete":
            continue
        before, after = verified_tables(p, SCREEN, science, out), verified_tables(p, FINAL, science, out)
        for kind in ("rounds", "evaluation", "history"):
            fields = [f for f in before[kind] if f not in ("stage_horizon", "training_seconds", "codec_seconds")]
            pd.testing.assert_frame_equal(before[kind][fields].reset_index(drop=True),
                                          after[kind][after[kind]["round"] <= SCREEN][fields].reset_index(drop=True),
                                          check_dtype=False, check_exact=True)


def analyze(out=OUT):
    science = load_contract(out)
    screen = ledger(POINTS, SCREEN, science, out)
    promoted = decision(screen, 3, SCREEN, science, out)
    if json.loads((out / "promotion.json").read_text()) != promoted:
        raise RuntimeError("Frozen baseline promotion changed")
    points = tuple(BY_ID[k] for m in METHODS for k in promoted["selected"][m])
    final = ledger(points, FINAL, science, out)
    validate_prefixes(points, science, out)
    parent.atomic_csv(final, out / "finalists.csv")
    chosen = choose(final, 1)
    frozen = {**decision(final, 1, FINAL, science, out), "configs": {m: asdict(config_for(BY_ID[ids[0]])) for m, ids in chosen.items()},
              "promotion_sha256": parent.text_hash(out / "promotion.json"), "confirmation_launched": False,
              "development_checkpoint_reuse_for_confirmation": False, "automatic_boundary_extension": False}
    parent.immutable_json(frozen, out / "frozen_selection.json")
    selected = final[final.point_id.isin(k[0] for k in chosen.values())].copy()
    selected["origin"] = "new_baseline_choice"
    prior = parent.load_contract()
    controls = parent.ledger(tuple(pulse.point_for(m) for m in pulse.METHODS), FINAL, prior, parent.OUT)
    controls["origin"] = "read_only_pulse_incumbent"
    comparison = pd.concat([controls, selected], ignore_index=True)
    for name, frame in (("screening.csv", screen), ("finalists.csv", final), ("selected.csv", selected), ("validation_comparison.csv", comparison)):
        parent.atomic_csv(frame, out / name)
    parent.atomic_json({"screen_candidates": len(POINTS), "screen_complete": int(screen.state.eq("complete").sum()),
                        "screen_failed": int(screen.state.eq("numerical_failure").sum()), "promoted": len(points),
                        "final_complete": int(final.state.eq("complete").sum()), "final_failed": int(final.state.eq("numerical_failure").sum()),
                        "selection_sha256": parent.digest(frozen), "max_unique_training_rounds": 43200,
                        "official_test_images_or_metrics_read": False, "new_confirmation_runs": 0,
                        "scope": "single_split_validation_only_baseline_calibration_not_predictive_confirmation"}, out / "analysis_audit.json")
    print(comparison[["point_id", "method", "origin", "final_validation_ce", "final_validation_accuracy", "unicast_total_bits"]].to_string(index=False))
    print("Calibration complete; three baseline configurations frozen. Stop for analysis before the thirty full-training confirmation runs.")
    return frozen


def run(device_name="auto", threads=4, eval_batch_size=256, out=OUT):
    with parent.full.development.point_lock(out / "locks/controller.lock"):
        science, device, fed = prepare(device_name, threads, eval_batch_size, out)
        if fed.scope != "validation":
            raise RuntimeError("Calibration requires validation scope")
        for p in POINTS:
            train_point(p, SCREEN, science, fed, device, out)
        screen = ledger(POINTS, SCREEN, science, out)
        parent.atomic_csv(screen, out / "screening.csv")
        promoted = decision(screen, 3, SCREEN, science, out)
        parent.immutable_json(promoted, out / "promotion.json")
        for method in METHODS:
            for key in promoted["selected"][method]:
                train_point(BY_ID[key], FINAL, science, fed, device, out)
        return analyze(out)


def preflight(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as torch_host
    science = science_spec()
    device = torch_host.configure(device_name, threads)
    if {"runtime": torch_host.runtime(device), "eval_batch_size": eval_batch_size} != science["execution"]:
        raise RuntimeError("Use the recorded pulse-study environment/device, threads and evaluation batch size")
    print(json.dumps(science["execution"], indent=2))
    print("18 settings and 3 exact finalist continuations per method; maximum 43200 new calibration rounds, 774 evaluations.")
    print("Dense: gain/schedule. STC: upload/download fractions and gain. EF-TopK: global fraction and gain.")
    print("Torch local SGD; native NumPy baseline compressor/aggregation conventions; real lossless wire frames. No official-test data or confirmation.")


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("ResNet baseline calibration not started")
        return
    science = load_contract(out)
    for p in POINTS:
        labels = []
        for horizon in (SCREEN, FINAL):
            receipt = read_receipt(p, horizon, science, out)
            if receipt:
                labels.append(f"r{horizon}:{receipt['state']}")
        if checkpoint_path(p, out).exists():
            with np.load(checkpoint_path(p, out), allow_pickle=False) as saved:
                labels.append("checkpoint:" + str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"]))
        print(p.point_id, ", ".join(labels) or "pending")


def smoke(device_name="auto", threads=4, eval_batch_size=256):
    import resnet14_transfer_host as torch_host
    device = torch_host.configure(device_name, threads)
    fed = parent.synthetic_federation(device, examples=4, evaluation=4)
    examples = (make_point("dense", 2., exponent=.3), make_point("stc", 2., .25, download=1.), make_point("ef_topk", .5, .05))
    for p in examples:
        cfg = replace(config_for(p), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            files = {k: Path(folder) / (k + ".npz") for k in ("native", "full", "resumed")}
            kwargs = dict(federation=fed, method=p.method, config=cfg, seed=97000, device=device, eval_batch_size=eval_batch_size)
            reference = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(files["native"], {}), native_reference=True)
            actual = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(files["full"], {}))
            host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(files["resumed"], {}), stop_after=2)
            resumed = host.run(**kwargs, checkpoint=parent.BaselineCheckpoint(files["resumed"], {}))
            with np.load(files["native"], allow_pickle=False) as native, np.load(files["full"], allow_pickle=False) as full, np.load(files["resumed"], allow_pickle=False) as after:
                rng_states = [json.loads(str(item["metadata"].item()))["progress"]["numpy_rng"] for item in (native, full, after)]
                if not rng_states[0] == rng_states[1] == rng_states[2]:
                    raise AssertionError("Baseline minibatch RNG differs")
                for key in full.files:
                    if key != "metadata":
                        np.testing.assert_array_equal(full[key].view(np.uint8), after[key].view(np.uint8))
                        np.testing.assert_array_equal(full[key].view(np.uint8), native[key].view(np.uint8))
            pd.testing.assert_frame_equal(reference["evaluation"], actual["evaluation"], check_exact=True)
            pd.testing.assert_frame_equal(actual["evaluation"], resumed["evaluation"], check_exact=True)
            fields = [k for k in actual["rounds"] if k not in ("training_seconds", "codec_seconds")]
            pd.testing.assert_frame_equal(actual["rounds"][fields], resumed["rounds"][fields], check_exact=True)
    print("Synthetic actual ResNet: all three lossless hosts match native models, residuals, RNG and predictions; resume also preserves exact traffic.")


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
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_resnet14_baseline*.py", "-v"], cwd=REPO, env=env, check=True)
        smoke(args.device, args.threads, args.eval_batch_size)
    elif args.command in ("status", "analyze"):
        globals()[args.command]()
    else:
        globals()[args.command](args.device, args.threads, args.eval_batch_size)


if __name__ == "__main__":
    main()
