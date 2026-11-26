"""Parallel validation tuning and ten-seed Fashion-MNIST MLP confirmation.

python experiments/fmnist_mlp_calibrated.py verify
python experiments/fmnist_mlp_calibrated.py preflight --workers auto
python experiments/fmnist_mlp_calibrated.py run --workers auto
git add experiments/results/fmnist_mlp_calibrated_v1
git commit -m "Add calibrated Fashion-MNIST MLP five-method results"
git push origin main
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO / "src"), str(REPO)]
import numpy as np
import pandas as pd
import fmnist_mlp_data as data_tools
import fmnist_mlp_host as host
from fmnist_mlp_data import DEV_SEED, SPLIT_SEED, DEV_TRAIN_SEED, SEEDS, TRAIN_SEEDS, atomic_json, digest, file_hash
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig

OUT = REPO / "experiments/results/fmnist_mlp_calibrated_v1"
DATA = REPO / "data/fashion-mnist"
METHODS = host.METHODS
SCREEN, FINAL, PROMOTE = 500, 1500, 3
TARGETS = (.80, .85, .87)
BASE = FinalBaselineConfig(rounds=FINAL, eval_stride=50, local_steps=5, local_lr=.1,
                           batch_size=32, regularization=.0001, init_scale=.5,
                           rho=.999, jump_scale=100., server_gain_scale=100.)


@dataclass(frozen=True)
class Point:
    point_id: str
    method: str
    gain: float = 1.
    threshold: float = .025
    quantum: float = .005
    exponent: float = 0.
    fraction: float = 1.
    download_fraction: float = 1.


def point(method, *, gain=1., threshold=.025, quantum=.005, exponent=0., fraction=1., download=None):
    label = lambda value: f"{value:g}".replace(".", "p")
    down = fraction if download is None else download
    suffix = (f"t{label(threshold)}_q{label(quantum)}_a{label(exponent)}" if method == "event" else
              f"t{label(threshold)}_g{label(gain)}" if method == "strom" else
              f"g{label(gain)}_a{label(exponent)}" if method == "dense" else
              f"u{label(fraction)}_d{label(down)}_g{label(gain)}" if method == "stc" else
              f"p{label(fraction)}_g{label(gain)}")
    return Point(method+"_"+suffix, method, gain, threshold, quantum, exponent, fraction, down)


POINTS = (
    tuple(point("event", threshold=t, quantum=q, exponent=a) for t in (.0125, .025, .05)
          for q in (.0025, .005, .01) for a in (.1, .2))
    + tuple(point("strom", threshold=t, gain=g) for t in (.000625, .00125, .0025, .005, .01, .02)
            for g in (.1, .3, 1.))
    + tuple(point("dense", gain=g, exponent=a) for g in (.25, .5, 1., 1.5, 2., 4.) for a in (0., .1, .3))
    + tuple(point("stc", fraction=p, gain=g) for p in (.01, .05, .25, 1.) for g in (.5, 1., 2., 4.))
    + (point("stc", fraction=.01, download=1.), point("stc", fraction=.25, download=1.))
    + tuple(point("ef_topk", fraction=p, gain=g) for p in (.01, .025, .05, .1) for g in (.5, 1., 2., 4.))
    + (point("ef_topk", fraction=.005), point("ef_topk", fraction=.25))
)
BY_ID = {p.point_id: p for p in POINTS}


def config_for(p):
    return replace(BASE, server_gain=p.gain,
                   threshold=p.threshold, strom_threshold=p.threshold,
                   jump0=p.quantum, jump_exponent=p.exponent if p.method == "event" else BASE.jump_exponent,
                   server_gain_exponent=p.exponent if p.method == "dense" else 0.,
                   topk_fraction=p.fraction, stc_upload_fraction=p.fraction, stc_download_fraction=p.download_fraction)


def worker_count(value="auto"):
    if str(value).lower() == "auto":
        # Approximate physical cores on common SMT desktops, capped to keep the
        # desktop responsive. Explicit --workers overrides this heuristic.
        return max(1, min(12, (os.cpu_count() or 2)//2))
    result = int(value)
    if result < 1:
        raise ValueError("--workers must be auto or a positive integer")
    return result


def runtime():
    output = io.StringIO()
    with redirect_stdout(output):
        np.show_config()
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.system(), "machine": platform.machine(), "processor": platform.processor(),
            "numpy_build_sha256": digest(output.getvalue()), "blas_threads_per_worker": 1}


def protocol_spec():
    return {"campaign": "fmnist_mlp_calibrated_v1", "dataset": "Fashion-MNIST", "architecture": list(host.LAYOUT.widths),
            "parameters": host.LAYOUT.dimension, "clients": 10, "participation": 1., "dominant_class_fraction": .55,
            "development_seed": DEV_SEED, "split_seed": SPLIT_SEED, "development_train_seed": DEV_TRAIN_SEED,
            "development_training_examples": 54000, "development_validation_examples": 6000,
            "confirmation_training_examples": 60000, "test_examples": 10000, "train_diagnostic_examples": 5000,
            "partition_seeds": list(SEEDS), "training_seeds": [TRAIN_SEEDS[s] for s in SEEDS],
            "points": [asdict(p) for p in POINTS], "configs": {p.point_id: asdict(config_for(p)) for p in POINTS},
            "screen_rounds": SCREEN, "final_rounds": FINAL, "screen_candidates_per_method": 18,
            "continuations_per_method": PROMOTE, "maximum_unique_tuning_rounds_per_method": 12000,
            "maximum_new_unique_rounds": 135000, "maximum_stage_receipts": 155, "confirmation_outcomes": 50,
            "selection": "final_validation_CE_then_total_unicast_bits_then_point_id",
            "selection_scope": "best_of_three_screen_promoted_finalists_per_family",
            "initialization_seed": 7777, "matched_minibatches": True, "main_variants": host.MAIN,
            "accuracy_targets": list(TARGETS), "target_rule": "first_saved_per_run_crossing_no_interpolation_or_stability_requirement",
            "missed_target": "right_censored_at1500_cost_ratios_only_on_common_hits_report_all_coverage",
            "intervals": "descriptive_unadjusted_paired_95pct_Student_t_log_cost_ratios",
            "operator": "native_float32_FedAvg_STC_two_residuals_global_EF_TopK_and_residual_conserving_Strom_adaptation",
            "Strom_gain_units": "native_gain=local_lr*FedAvg_equivalent_gain; grid_native_0.1_0.3_1.0_means_effective_1_3_10",
            "event_state": "rho0.999_persistent_full_reset_gain1_independent_quantum",
            "codec_backend": host.IMPLEMENTATION, "byte_equivalent_optimization": "bound1_vectorization_and_exact_ordered_length_preselection",
            "primary_accounting": "actual_upload_and_server_frames_padding_initial_sync_32bit_requests_ten_recipient_copies",
            "logical_broadcast": "secondary_server_payload_once",
            "evaluation": "round1_and_every50_fixed_round1500_endpoint_single_forward_per_batch512",
            "evaluation_reduction": "float32_logits_float64_loss_sum_no_effect_on_local_SGD_or_encoder_state",
            "parallelism": "spawn_independent_jobs_one_BLAS_thread_shared_readonly_mmaps_no_client_parallel_reduction",
            "resume": "model_client_reconstruction_all_residuals_RNG_counters_evaluations_every50rounds",
            "continuation": "same1500round_config_from_start_500round_checkpoint_preserves_exact_prefix",
            "confirmation": "fresh_full_training_initialization_no_development_checkpoint_reuse",
            "official_test_during_tuning": "neither_downloaded_by_this_campaign_nor_parsed_or_evaluated",
            "freezing": "automatic_validation_only_promotion_and_immutable_selection_before_test_access",
            "numerical_failure": "terminal_receipt_no_replacement_candidate_or_seed_incomplete_confirmation_blocks_effect_analysis",
            "automatic_boundary_extension": False, "hyperparameter_optimality_claim": False,
            "previous_test_exposure": "official_FashionMNIST_test_previously_inspected_in_archived_studies_new_seeds_not_new_dataset"}


def source_hashes():
    paths = list((REPO / "src/neuromorphicfl").glob("*.py"))
    names = ("fmnist_mlp_calibrated.py", "fmnist_mlp_data.py", "fmnist_mlp_host.py", "resnet14_baseline_fast_host.py",
             "resnet14_baseline_host.py", "core_float_host.py", "fast_pulse_codec.py", "event_codec_full_horizon.py")
    paths.extend(REPO / "experiments" / name for name in names)
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(paths)}


def known_seeds(out):
    seen = set()
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("partition_seed", "development_partition_seed") and isinstance(child, int):
                    seen.add(child)
                elif key in ("partition_seeds", "selected_partition_seeds") and isinstance(child, list):
                    seen.update(x for x in child if isinstance(x, int))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for p in (REPO / "experiments/results").rglob("*.json"):
        if p.is_relative_to(out):
            continue
        try:
            visit(json.loads(p.read_text()))
        except (json.JSONDecodeError, UnicodeError):
            continue
    return seen


def prepare(out=OUT, root=DATA):
    out, root = Path(out), Path(root)
    cache, manifests = data_tools.prepare_cache(root, include_test=False)
    science = {"protocol": protocol_spec(), "source_hashes": source_hashes(), "runtime": runtime(),
               "training_data": manifests["train"]}
    path = out / "protocol.json"
    atomic_json(path, science, immutable=True)
    audit_path = out / "fresh_seed_audit.json"
    if not audit_path.exists():
        previous = known_seeds(out)
        selected = {DEV_SEED, *SEEDS}
        overlap = sorted(selected & previous)
        if overlap:
            raise RuntimeError(f"Planned partition seeds occur in earlier evidence: {overlap}")
        atomic_json(audit_path, {"selected_partition_seeds": sorted(selected), "overlap": overlap,
                                "prior_inventory_sha256": digest(sorted(previous)), "prior_unique_seed_count": len(previous)}, immutable=True)
    arrays = data_tools.open_arrays(cache, include_test=False)
    _, _, _, manifest = data_tools.allocation(arrays["train_labels"], seed=DEV_SEED, validation=True)
    atomic_json(out / "development_partition.json", manifest, immutable=True)
    return science, cache


def load_contract(out=OUT, *, check_runtime=True):
    science = json.loads((Path(out) / "protocol.json").read_text())
    if science["protocol"] != protocol_spec() or science["source_hashes"] != source_hashes() or (check_runtime and science["runtime"] != runtime()):
        raise RuntimeError("Frozen MLP source/configuration/runtime differs; preserve the campaign instead of mixing runs")
    return science


def stage_paths(p, phase, seed, horizon, out):
    stem = f"{p.point_id}_{phase}_p{seed}_r{horizon}"
    return {key: Path(out) / f"{stem}_{key}.{'json' if key == 'receipt' else 'csv'}"
            for key in ("rounds", "evaluation", "history", "summary", "receipt")}


def identity(p, phase, seed, horizon, science, *, selection_sha=None):
    return {"point_id": p.point_id, "method": p.method, "dataset": "fashion_mnist", "architecture": "mlp_784_32_16_10",
            "phase": phase, "partition_seed": seed,
            "train_seed": DEV_TRAIN_SEED if phase == "development" else TRAIN_SEEDS[seed], "stage_horizon": horizon,
            "config_sha256": digest(science["protocol"]["configs"][p.point_id]), "contract_sha256": digest(science),
            "selection_sha256": selection_sha}


def read_receipt(p, phase, seed, horizon, science, out, *, selection_sha=None):
    paths = stage_paths(p, phase, seed, horizon, out)
    if not paths["receipt"].exists():
        return None
    receipt = json.loads(paths["receipt"].read_text())
    if receipt["identity"] != identity(p, phase, seed, horizon, science, selection_sha=selection_sha):
        raise RuntimeError("MLP stage identity differs")
    if receipt["state"] == "complete":
        for key in ("rounds", "evaluation", "history", "summary"):
            if file_hash(paths[key]) != receipt["sha256"][key]:
                raise RuntimeError(f"MLP stage evidence changed: {paths[key]}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown stage receipt state")
    return receipt


_ARRAYS = None
_CACHE = None


def worker_init(cache, include_test):
    global _ARRAYS, _CACHE
    _CACHE = Path(cache)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _ARRAYS = data_tools.open_arrays(cache, include_test=include_test)


class StopRequested(RuntimeError):
    pass


def atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(f".csv.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_job(job):
    p, phase, seed, horizon, science, out, selection_sha = job
    if "runtime" in science and science["runtime"] != runtime():
        raise RuntimeError("Worker numerical runtime differs from the campaign")
    if "source_hashes" in science and science["source_hashes"] != source_hashes():
        raise RuntimeError("Worker training source files differ from the campaign")
    out = Path(out)
    if (out / ".stop_requested").exists():
        raise StopRequested("Coordinator requested stop")
    existing = read_receipt(p, phase, seed, horizon, science, out, selection_sha=selection_sha)
    if existing is not None:
        return existing["state"], p.point_id, seed
    data = data_tools.task_data(_ARRAYS, seed=seed, validation=phase == "development")
    manifest = {**data["manifest"], "scope": data["scope"]}
    if phase == "confirmation":
        manifest["test_manifest_sha256"] = digest(json.loads((_CACHE / "test_manifest.json").read_text()))
        atomic_json(out / f"confirmation_partition_p{seed}.json", manifest, immutable=True)
    else:
        manifest["scope"] = "validation"
    meta = identity(p, phase, seed, horizon, science, selection_sha=selection_sha)
    provenance = {k: v for k, v in meta.items() if k != "stage_horizon"}
    provenance["data_manifest_sha256"] = digest(manifest)
    checkpoint = out / "checkpoints" / f"{p.point_id}_{phase}_p{seed}.npz"
    progress_path = out / "progress" / f"{p.point_id}_{phase}_p{seed}.json"
    def progress(rnd):
        atomic_json(progress_path, {"round": rnd, "goal": horizon, "pid": os.getpid()})
        if (out / ".stop_requested").exists():
            raise StopRequested("Saved checkpoint before stopping")
    tick = time.perf_counter()
    paths = stage_paths(p, phase, seed, horizon, out)
    try:
        result = host.run(data, method=p.method, config=FinalBaselineConfig(**science["protocol"]["configs"][p.point_id]), seed=meta["train_seed"],
                          checkpoint_path=checkpoint, provenance=provenance, stop_after=horizon, callback=progress)
        tables = host.trajectory_tables(result, p.method)
        if tables["summary"].iloc[0]["rounds"] != horizon or len(tables["rounds"]) != horizon:
            raise RuntimeError("MLP stage did not reach its horizon")
        for key, frame in tables.items():
            for k, v in meta.items():
                frame[k] = v if v is not None else ""
            atomic_csv(paths[key], frame)
        receipt = {"identity": meta, "state": "complete", "data_manifest": manifest,
                   "sha256": {k: file_hash(paths[k]) for k in tables}, "invocation_wall_seconds": time.perf_counter()-tick}
    except host.NumericalFailure as exc:
        receipt = {"identity": meta, "state": "numerical_failure", "data_manifest": manifest,
                   "failed_round": exc.round, "reason": str(exc), "invocation_wall_seconds": time.perf_counter()-tick}
    atomic_json(paths["receipt"], receipt, immutable=True)
    return receipt["state"], p.point_id, seed


def run_job(job):
    p, phase, seed, *_ = job
    out = Path(job[5])
    folder = out / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / f"{p.point_id}_{phase}_p{seed}.claim"
    record = {"pid": os.getpid(), "host": socket.gethostname()}
    for _ in range(2):
        try:
            with claim.open("x") as f:
                json.dump(record, f)
            break
        except FileExistsError:
            previous = json.loads(claim.read_text())
            if previous["host"] != record["host"] or _pid_alive(previous["pid"]):
                raise RuntimeError("A live worker already owns this model checkpoint")
            claim.unlink()
    else:
        raise RuntimeError("Unable to acquire model checkpoint claim")
    try:
        return _run_job(job)
    finally:
        if claim.exists() and json.loads(claim.read_text()) == record:
            claim.unlink()


def run_jobs(jobs, *, cache, workers, out, label, include_test=False):
    pending = []
    for job in jobs:
        p, phase, seed, horizon, science, _, selection_sha = job
        if read_receipt(p, phase, seed, horizon, science, out, selection_sha=selection_sha) is None:
            pending.append(job)
    print(f"{label}: {len(pending)} pending / {len(jobs)} jobs; {min(workers, len(pending))} workers, one BLAS thread each", flush=True)
    if not pending:
        return
    executor = ProcessPoolExecutor(max_workers=min(workers, len(pending)), mp_context=multiprocessing.get_context("spawn"),
                                   initializer=worker_init, initargs=(str(cache), include_test))
    futures = {executor.submit(run_job, j): j for j in pending}
    tick, completed = time.perf_counter(), 0
    try:
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                print(f"{label}: {completed}/{len(pending)} newly finished; {len(futures)} queued/running; {(time.perf_counter()-tick)/60:.1f} min", flush=True)
            for future in done:
                state, point_id, seed = future.result()
                del futures[future]
                completed += 1
                print(f"{label} {completed}/{len(pending)} {state}: {point_id}, p{seed}", flush=True)
    except BaseException:
        (Path(out) / ".stop_requested").touch()
        for future in futures:
            future.cancel()
        print("Stopping workers at their next saved checkpoint; completed receipts are retained.", flush=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def ledger(points, horizon, science, out):
    rows = []
    for p in points:
        receipt = read_receipt(p, "development", DEV_SEED, horizon, science, out)
        if receipt is None:
            raise RuntimeError("Development stage is incomplete")
        row = {"point_id": p.point_id, "method": p.method, "state": receipt["state"]}
        if receipt["state"] == "complete":
            values = pd.read_csv(stage_paths(p, "development", DEV_SEED, horizon, out)["summary"], float_precision="round_trip", dtype={"model_sha256": str}).iloc[0]
            row.update(validation_ce=float(values.final_validation_ce), validation_accuracy=float(values.final_validation_accuracy),
                       unicast_total_bits=int(values.unicast_total_bits))
        rows.append(row)
    return rows


def choose(rows, count):
    result = {}
    for method in METHODS:
        finite = [r for r in rows if r["method"] == method and r["state"] == "complete" and np.isfinite(r["validation_ce"])]
        finite.sort(key=lambda r: (r["validation_ce"], r["unicast_total_bits"], r["point_id"]))
        if not finite:
            raise RuntimeError(f"No finite validation candidate for {method}; no replacements are introduced")
        result[method] = [r["point_id"] for r in finite[:count]]
    return result


def promotion_record(science, out):
    rows = ledger(POINTS, SCREEN, science, out)
    return {"contract_sha256": digest(science), "criterion": "validation_ce_then_unicast_then_id",
            "selected": choose(rows, PROMOTE), "screen_receipts": {p.point_id: file_hash(stage_paths(p, "development", DEV_SEED, SCREEN, out)["receipt"]) for p in POINTS}}


def selection_record(science, promotion, out):
    candidates = [BY_ID[name] for method in METHODS for name in promotion["selected"][method]]
    rows = ledger(candidates, FINAL, science, out)
    # Verify exact screen prefixes even though the current checkpoint is final.
    for p in candidates:
        receipt = read_receipt(p, "development", DEV_SEED, FINAL, science, out)
        if receipt["state"] != "complete":
            continue
        a = pd.read_csv(stage_paths(p, "development", DEV_SEED, SCREEN, out)["rounds"], float_precision="round_trip", dtype={"model_sha256": str})
        b = pd.read_csv(stage_paths(p, "development", DEV_SEED, FINAL, out)["rounds"], float_precision="round_trip", dtype={"model_sha256": str}).iloc[:SCREEN]
        pd.testing.assert_frame_equal(a.drop(columns="stage_horizon"), b.drop(columns="stage_horizon"), check_exact=True)
    winners = choose(rows, 1)
    return {"contract_sha256": digest(science), "promotion_sha256": digest(promotion), "selected": winners,
            "configs": {m: science["protocol"]["configs"][names[0]] for m, names in winners.items()},
            "final_receipts": {p.point_id: file_hash(stage_paths(p, "development", DEV_SEED, FINAL, out)["receipt"]) for p in candidates},
            "official_test_used_for_selection": False, "development_checkpoint_reused_for_confirmation": False}


def tune(science, cache, workers, out):
    jobs = [(p, "development", DEV_SEED, SCREEN, science, str(out), None) for p in POINTS]
    run_jobs(jobs, cache=cache, workers=workers, out=out, label="Screen")
    rows = ledger(POINTS, SCREEN, science, out)
    atomic_csv(Path(out) / "screen_summary.csv", pd.DataFrame(rows))
    promotion = promotion_record(science, out)
    atomic_json(Path(out) / "promotion.json", promotion, immutable=True)
    promoted = [BY_ID[name] for m in METHODS for name in promotion["selected"][m]]
    run_jobs([(p, "development", DEV_SEED, FINAL, science, str(out), None) for p in promoted],
             cache=cache, workers=workers, out=out, label="Continue")
    atomic_csv(Path(out) / "finalist_summary.csv", pd.DataFrame(ledger(promoted, FINAL, science, out)))
    selection = selection_record(science, promotion, out)
    atomic_json(Path(out) / "frozen_selection.json", selection, immutable=True)
    atomic_csv(Path(out) / "selected.csv", pd.DataFrame([asdict(BY_ID[selection["selected"][m][0]]) for m in METHODS]))
    print("Frozen all five validation selections before official test access.", flush=True)
    return selection


def verify_selection(science, out):
    promotion = promotion_record(science, out)
    if promotion != json.loads((Path(out) / "promotion.json").read_text()):
        raise RuntimeError("Promotion record differs")
    selection = selection_record(science, promotion, out)
    if selection != json.loads((Path(out) / "frozen_selection.json").read_text()):
        raise RuntimeError("Frozen validation selection differs")
    return selection


def confirm(science, root, workers, out):
    selection = verify_selection(science, out)  # Must finish before opening test files.
    cache, manifests = data_tools.prepare_cache(root, include_test=True)
    if manifests["train"] != science["training_data"]:
        raise RuntimeError("Training data changed after selection")
    atomic_json(Path(out) / "confirmation_test_manifest.json", manifests["test"], immutable=True)
    points = [BY_ID[selection["selected"][method][0]] for method in METHODS]
    jobs = [(p, "confirmation", seed, FINAL, science, str(out), digest(selection)) for seed in SEEDS for p in points]
    run_jobs(jobs, cache=cache, workers=workers, out=out, label="Confirm", include_test=True)


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() != 87  # Invalid PID means dead; access denied is conservative.
        code = wintypes.DWORD()
        try:
            return (not kernel.GetExitCodeProcess(handle, ctypes.byref(code))) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def campaign_lock(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ".campaign.lock"
    record = {"pid": os.getpid(), "host": socket.gethostname()}
    for _ in range(2):
        try:
            with path.open("x") as f:
                json.dump(record, f)
            break
        except FileExistsError:
            old = json.loads(path.read_text())
            if old["host"] != record["host"] or _pid_alive(old["pid"]):
                raise RuntimeError("Another coordinator owns this campaign; use status or wait for it to finish")
            path.unlink()
    else:
        raise RuntimeError("Unable to acquire campaign lock")
    (out / ".stop_requested").unlink(missing_ok=True)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()) == record:
            path.unlink()


def status(out):
    counts = Counter()
    for p in Path(out).glob("*_receipt.json"):
        row = json.loads(p.read_text())
        counts[(row["identity"]["phase"], row["identity"]["stage_horizon"], row["state"])] += 1
    for key, count in sorted(counts.items()):
        print(*key, count)
    if not counts:
        print("No completed stages yet")
    for p in sorted((Path(out) / "progress").glob("*.json")):
        row = json.loads(p.read_text())
        if row["round"] < row["goal"]:
            print(p.stem, f"checkpoint {row['round']}/{row['goal']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "preflight", "tune", "confirm", "run", "analyze", "status"))
    parser.add_argument("--workers", default="auto", help="auto: approximately physical cores, capped at 12; or an explicit count")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "verify":
        env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + str(REPO))
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_fmnist_mlp_calibrated.py", "-v"],
                       cwd=REPO, env=env, check=True)
        return
    if args.command == "status":
        status(out)
        return
    if args.command == "analyze":
        import fmnist_mlp_analysis as analysis
        analysis.analyze(out)
        return
    workers = worker_count(args.workers)
    with campaign_lock(out):
        science, cache = prepare(out, args.data_root)
        print(f"Fashion-MNIST MLP: {len(POINTS)} screens; up to 15 continuations; 50 fresh confirmation fits.", flush=True)
        print(f"Workers={workers}; logical CPUs={os.cpu_count()}; BLAS threads=1; shared mmap cache; resumable every 50 rounds.", flush=True)
        if args.command == "preflight":
            print("Preflight complete: training-only dataset/cache, source/runtime contract and exact development split verified.")
            return
        if args.command in ("tune", "run"):
            tune(science, cache, workers, out)
        if args.command in ("confirm", "run"):
            confirm(science, args.data_root, workers, out)
            import fmnist_mlp_analysis as analysis
            analysis.analyze(out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped after saving worker checkpoints. Repeat the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
