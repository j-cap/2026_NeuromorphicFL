"""Frozen-setting fifty-client participation study: prepare, pilot, confirm.

python experiments/calibrated_participation.py verify
python experiments/calibrated_participation.py pilot --workers auto
python experiments/calibrated_participation.py analyze --phase pilot
git add experiments/results/calibrated_participation_v1
git commit -m "Add calibrated participation development pilot"
git push origin main

The separate confirm command prepares 150 fresh fits, after all fifteen pilot
jobs have terminal receipts. No method-ranking or Event-win gate is applied.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/"src"), str(REPO/"experiments")]
import numpy as np
import pandas as pd
import calibrated_participation_data as data
import calibrated_participation_host as host
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig
from stc_cifar10_development import point_lock

OUT = REPO/"experiments/results/calibrated_participation_v1"
DATA = REPO/"data/cifar-10"
METHODS, ACTIVE, SEEDS = host.METHODS, data.ACTIVE, data.SEEDS
TARGETS = (.45, .50, .55)
CORE = data.CORE_PROTOCOL
ANCESTOR = REPO/"paper/ijcnn2027/evidence/calibrated_core_freeze.json"
EXTRA_SOURCES = (
    "experiments/calibrated_participation.py", "experiments/calibrated_participation_host.py",
    "experiments/calibrated_participation_data.py", "experiments/calibrated_participation_analysis.py",
    "tests/test_calibrated_participation.py",
    "experiments/fast_pulse_codec.py", "experiments/fmnist_mlp_host.py",
    "experiments/resnet14_baseline_host.py", "experiments/resnet14_baseline_fast_host.py",
    "experiments/core_float_host.py", "experiments/event_codec_full_horizon.py",
    "experiments/stc_cifar10_confirmation.py", "experiments/stc_cifar10_development.py",
)


def configs():
    ancestor = json.loads(ANCESTOR.read_text())
    relative = CORE.relative_to(REPO).as_posix()
    if data.file_hash(CORE) != ancestor["files"][relative]:
        raise RuntimeError("Frozen full-participation core protocol changed")
    points = json.loads(CORE.read_text())["protocol"]["points"]
    if set(points) != set(METHODS):
        raise RuntimeError("The five frozen methods are required")
    result = {m: FinalBaselineConfig(**points[m]) for m in METHODS}
    for m, cfg in result.items():
        if (cfg.rounds, cfg.eval_stride, cfg.local_steps, cfg.batch_size, cfg.local_lr, cfg.regularization) != (1800, 60, 5, 32, .05, .0005):
            raise RuntimeError("Frozen local-learning protocol differs: "+m)
    return result


def source_hashes():
    previous = json.loads(CORE.read_text())["source_hashes"]
    for name, sha in previous.items():
        if data.file_hash(REPO/name) != sha:
            raise RuntimeError("Frozen original scientific source changed: "+name)
    names = set(previous)|set(EXTRA_SOURCES)
    names.update(p.relative_to(REPO).as_posix() for p in (REPO/"src/neuromorphicfl").glob("*.py"))
    return {name: data.file_hash(REPO/name) for name in sorted(names)}


def specification():
    return {"campaign": "calibrated_participation_v1", "architecture": "cifar_cnn",
        "parameters": 20570, "clients": 50, "active_clients": list(ACTIVE),
        "methods": list(METHODS), "configs": {m: asdict(c) for m, c in configs().items()},
        "fixed_settings": "full_participation_selected_points_transferred_without_retuning",
        "new_partial_participation_optimality_claim": False,
        "pilot": {"partition_seed": data.DEV_SEED, "split_seed": data.SPLIT_SEED,
            "train_seed": 100000+data.DEV_SEED, "rounds": data.PILOT_ROUNDS,
            "train_examples": 45000, "validation_examples": 5000, "new_fits": 15,
            "official_test_parsed_or_evaluated": False, "hyperparameter_selection": False},
        "confirmation": {"partition_seeds": list(SEEDS), "train_seeds": [100000+s for s in SEEDS],
            "rounds": data.ROUNDS, "train_examples": 50000, "test_examples": 10000, "new_fits": 150,
            "development_checkpoint_reuse": False, "initialization_seed": 7777},
        "sampling": "uniform_without_replacement_nested_permutation_sort_each_active_subset",
        "selection_seed_offset": 200000, "data_skew": "55pct_dominant_5pct_each_other_class",
        "inclusion_weights": "p_i/pi_i_with_equal_clients_1_over_K",
        "weight_placement": {"event": "gradient_scale_evidence_before_threshold",
            "strom": "gradient_scale_residual_before_threshold", "ef_topk": "weighted_client_delta_residual",
            "stc": "unweighted_client_delta_residual_weight_decoded_upload_once_before_server_residual",
            "dense": "weighted_decoded_local_delta"},
        "inactive_state": {"event": "rho_power_global_round_gap_lazy_decay",
            "strom_stc_ef_topk": "retain_client_residual_unchanged"},
        "synchronization": "initial_all_clients_then_selected_clients_exact_catchup_and_current_server_frame",
        "catchup": "ordered_actual_server_frames_or_smaller_exact_dense_model_checkpoint",
        "post_round_sync": "current_native_lossless_frame_no_extra_lossy_operator",
        "primary_cost": "conservative_unicast_including_final_exact_sync_all_fifty_clients",
        "secondary_cost": "operational_unicast_before_final_deployment_sync",
        "requests": "one_actual_32bit_request_per_selected_client_round_plus_terminal_requests",
        "directions": "requests_count_as_uplink_total_matches_old_accounting_at_full_participation",
        "compute": "fixed_rounds_with_recorded_active_client_rounds_local_SGD_steps_and_examples",
        "evaluation": "round_one_every_sixty_and_fixed_endpoint_no_best_test_checkpoint",
        "targets": list(TARGETS), "target_rule": "first_saved_crossing_no_interpolation_no_stability_requirement",
        "target_failure": "censored_never_zero_cost_report_total_and_common_hit_denominators",
        "numerical_failure": "terminal_receipt_no_seed_or_method_replacement_no_silent_exclusion",
        "pilot_gate": "terminal_receipts_and_protocol_checks_only_not_predictive_ranking",
        "intervals": "descriptive_unadjusted_paired_95pct_Student_t_log_ratios",
        "previous_test_exposure": "official_test_previously_inspected_new_seeds_not_new_dataset",
        "parallelism": "spawn_independent_runs_one_BLAS_thread_readonly_shared_uint8_image_mmaps",
        "checkpoint": "atomic_model_all_client_models_states_residuals_rng_actual_frame_journal_counters_metrics",
        "frozen_core_protocol_sha256": data.file_hash(CORE), "hash_format": "sha256_utf8_LF"}


def prepare(out=OUT, root=DATA):
    out, root = Path(out), Path(root)
    with point_lock(out/"locks/prepare.lock"):
        cfg, hashes = specification(), source_hashes()
        cache, manifests = data.prepare_cache(root, include_test=False)
        science = {"protocol": cfg, "source_hashes": hashes, "runtime": data.runtime(),
            "training_data": manifests["train"]}
        data.atomic_json(out/"protocol.json", science, immutable=True)
        arrays = data.open_arrays(cache, include_test=False)
        for seed in (data.DEV_SEED, *SEEDS):
            _, _, _, manifest = data.allocation(arrays["train_labels"], seed=seed, validation=seed == data.DEV_SEED)
            data.atomic_json(out/f"partition_p{seed}.json", manifest, immutable=True)
            data.atomic_json(out/f"schedule_p{seed}.json", data.schedule_manifest(seed), immutable=True)
    print(f"Prepared immutable contract {data.digest(science)}; 15 validation pilots / 150 fresh confirmations.", flush=True)
    return cache, science


def load_contract(out=OUT, *, check_runtime=False):
    out = Path(out)
    science = json.loads((out/"protocol.json").read_text())
    if science["protocol"] != specification() or science["source_hashes"] != source_hashes():
        raise RuntimeError("Study settings/sources changed; never mix campaign versions")
    if check_runtime and science["runtime"] != data.runtime():
        raise RuntimeError("Use the same recorded NumPy/Python/BLAS environment to train or resume")
    return science


def jobs_for(phase):
    if phase not in ("pilot", "confirmation"):
        raise ValueError("Unknown phase")
    seeds = (data.DEV_SEED,) if phase == "pilot" else SEEDS
    return [(phase, k, method, seed) for seed in seeds for k in ACTIVE for method in METHODS]


def stem(job):
    phase, active, method, seed = job
    return f"{phase}_k{active:02d}_{method}_p{seed}"


def paths(job, out=OUT):
    out = Path(out)
    return {kind: out/f"{stem(job)}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("rounds", "history", "summary", "receipt")}


def identity(job, science, out=OUT):
    phase, active, method, seed = job
    out = Path(out)
    return {"phase": phase, "method": method, "active_clients": active, "partition_seed": seed,
        "train_seed": 100000+seed, "selection_seed": 200000+seed,
        "evaluation_scope": "validation" if phase == "pilot" else "test",
        "horizon": data.PILOT_ROUNDS if phase == "pilot" else data.ROUNDS,
        "contract_sha256": data.digest(science), "config_sha256": data.digest(asdict(configs()[method])),
        "partition_manifest_sha256": data.file_hash(out/f"partition_p{seed}.json"),
        "schedule_sha256": json.loads((out/f"schedule_p{seed}.json").read_text())["sha256"][str(active)],
        "confirmation_freeze_sha256": data.file_hash(out/"confirmation_freeze.json") if phase == "confirmation" else "",
        "test_manifest_sha256": data.file_hash(out/"confirmation_test_manifest.json") if phase == "confirmation" else ""}


def read_receipt(job, science, out=OUT):
    files = paths(job, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text())
    if receipt["identity"] != identity(job, science, out):
        raise RuntimeError("Receipt identity changed: "+stem(job))
    if receipt["state"] == "complete":
        for kind in ("rounds", "history", "summary"):
            if not files[kind].exists() or data.file_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError("Output file changed: "+stem(job)+"/"+kind)
    elif receipt["state"] == "numerical_failure":
        if not 1 <= receipt["failed_round"] <= receipt["identity"]["horizon"]:
            raise RuntimeError("Invalid numerical-failure receipt")
    else:
        raise RuntimeError("Unknown terminal state")
    return receipt


def validate_tables(tables, job, *, expected_identity=None):
    phase, k, method, _ = job
    horizon = data.PILOT_ROUNDS if phase == "pilot" else data.ROUNDS
    scope = "validation" if phase == "pilot" else "test"
    rounds, history, summary = (tables[name] for name in ("rounds", "history", "summary"))
    if (len(summary) != 1 or rounds["round"].tolist() != list(range(1, horizon+1))
            or history["round"].tolist() != sorted({1, horizon, *range(60, horizon+1, 60)})):
        raise RuntimeError("Incomplete time/endpoint coverage")
    for table in tables.values():
        if not np.isfinite(table.select_dtypes(include=np.number)).all().all():
            raise RuntimeError("Nonfinite evidence")
        if expected_identity:
            for key, value in expected_identity.items():
                if value == "":
                    if not table[key].fillna("").eq("").all():
                        raise RuntimeError("Unexpected evidence identity")
                elif not table[key].eq(value).all():
                    raise RuntimeError("Evidence identity changed: "+key)
    if not rounds.method.eq(method).all() or not rounds.selected_clients.eq(k).all():
        raise RuntimeError("Method/participation mismatch")
    if not rounds.reconstruction_verified.eq(True).all():
        raise RuntimeError("Missing exact reconstruction proof")
    if not rounds.current_downlink_bits.eq(k*rounds.exact_downlink_bits).all():
        raise RuntimeError("Recipient-copy accounting differs")
    if not (rounds.codec_uplink_bits.mod(8).eq(0).all() and rounds.exact_downlink_bits.mod(8).eq(0).all()):
        raise RuntimeError("Serialized frame padding not charged")
    if not rounds.request_bits.eq(32*k).all():
        raise RuntimeError("Selected-client request accounting differs")
    if not (rounds.current_clients+rounds.catchup_replay_clients+rounds.catchup_checkpoint_clients).eq(k).all():
        raise RuntimeError("Catch-up client counts differ")
    fields = {"cumulative_update_uplink_bits": "codec_uplink_bits", "cumulative_request_bits": "request_bits",
        "cumulative_current_downlink_bits": "current_downlink_bits",
        "cumulative_catchup_replay_bits": "catchup_replay_bits",
        "cumulative_catchup_checkpoint_bits": "catchup_checkpoint_bits"}
    initial = 50*(32*20570+64)
    for cumulative, increment in fields.items():
        if not np.array_equal(rounds[cumulative], rounds[increment].cumsum()):
            raise RuntimeError("Traffic prefix differs")
    if not rounds.operational_total_bits.eq(initial+rounds[list(fields)].sum(axis=1)).all():
        raise RuntimeError("Operational components do not balance")
    if not rounds.local_sgd_steps.eq(rounds["round"]*k*5).all():
        raise RuntimeError("Computation accounting differs")
    indexed = rounds.set_index("round")
    for row in history.to_dict("records"):
        rnd = int(row["round"])
        if (row["model_sha256"] != indexed.loc[rnd, "model_sha256"]
                or row["operational_total_bits"] != indexed.loc[rnd, "operational_total_bits"]
                or row["unicast_total_bits"] != row["uplink_bits"]+row["unicast_downlink_bits"]
                or row["unicast_total_bits"] != row["operational_total_bits"]+row["terminal_catchup_bits"]
                or row["terminal_verified_clients"] != 50):
            raise RuntimeError("Saved model/terminal accounting differs")
    for column in (scope+"_accuracy", scope+"_macro_accuracy", scope+"_worst_class_accuracy"):
        if not history[column].between(0, 1).all():
            raise RuntimeError("Invalid predictive accuracy")
    if phase == "pilot" and any((c.startswith("test_") and c != "test_manifest_sha256")
            or c.startswith("final_test_") for table in tables.values() for c in table):
        raise RuntimeError("Pilot may contain validation metrics only")
    last = history.iloc[-1]
    metric_fields = {"train_objective", "train_ce", scope+"_ce", scope+"_accuracy",
                     scope+"_macro_accuracy", scope+"_worst_class_accuracy"}
    expected = {("final_"+key if key in metric_fields else key): value
                for key, value in last.to_dict().items() if key != "round"}
    expected["rounds"] = horizon
    for key, value in expected.items():
        actual = summary.iloc[0][key]
        if isinstance(value, (float, np.floating)):
            if not np.isclose(actual, value, rtol=1e-12, atol=1e-12):
                raise RuntimeError("Final summary differs: "+key)
        elif actual != value:
            raise RuntimeError("Final summary differs: "+key)


_ARRAYS = None


def worker_init(cache, include_test):
    global _ARRAYS
    _ARRAYS = data.open_arrays(cache, include_test=include_test)
    worker_data.cache_clear()


@lru_cache(maxsize=2)
def worker_data(seed, validation):
    return data.task_data(_ARRAYS, seed=seed, validation=validation)


def run_job(payload):
    job, science, out_name = payload
    out = Path(out_name)
    with point_lock(out/"locks"/(stem(job)+".lock")):
        previous = read_receipt(job, science, out)
        if previous is not None:
            return stem(job)+": existing "+previous["state"]
        phase, active, method, seed = job
        meta = identity(job, science, out)
        federation, manifest = worker_data(seed, phase == "pilot")
        if manifest != json.loads((out/f"partition_p{seed}.json").read_text()):
            raise RuntimeError("Worker client allocation differs")
        schedule = data.schedule_for(seed, active)
        if hashlib.sha256(schedule.astype("<i2").tobytes()).hexdigest() != meta["schedule_sha256"]:
            raise RuntimeError("Worker schedule differs")
        started = time.perf_counter()
        def progress(row, evaluation):
            if row["round"] % 60 == 0 or row["round"] == 1 or row["round"] == meta["horizon"]:
                data.atomic_json(out/"progress"/(stem(job)+".json"),
                    {"identity": meta, "round": row["round"], "goal": meta["horizon"],
                     "elapsed_seconds_this_attempt": time.perf_counter()-started,
                     "latest_metric": evaluation})
        try:
            result = host.run(federation=federation, architecture="cifar_cnn", method=method,
                config=configs()[method], schedule=schedule[:meta["horizon"]], seed=meta["train_seed"],
                horizon=meta["horizon"], scope=meta["evaluation_scope"],
                checkpoint_path=out/"checkpoints"/(stem(job)+".npz"), provenance=meta, callback=progress)
        except host.NumericalFailure as exc:
            data.atomic_json(paths(job, out)["receipt"], {"identity": meta, "state": "numerical_failure",
                "failed_round": exc.round, "reason": str(exc)})
            return stem(job)+": NUMERICAL FAILURE (retained)"
        history = result["history"]
        last = history.iloc[-1].to_dict()
        metric_fields = {"train_objective", "train_ce", meta["evaluation_scope"]+"_ce",
            meta["evaluation_scope"]+"_accuracy", meta["evaluation_scope"]+"_macro_accuracy",
            meta["evaluation_scope"]+"_worst_class_accuracy"}
        summary = {("final_"+key if key in metric_fields else key): value
                   for key, value in last.items() if key != "round"}
        summary["rounds"] = int(last["round"])
        tables = {"rounds": result["rounds"], "history": history, "summary": pd.DataFrame([summary])}
        validate_tables(tables, job)
        files = paths(job, out)
        for kind, table in tables.items():
            data.atomic_csv(files[kind], table.assign(**meta))
        data.atomic_json(files["receipt"], {"identity": meta, "state": "complete",
            "sha256": {kind: data.file_hash(files[kind]) for kind in tables},
            "elapsed_seconds_this_attempt": time.perf_counter()-started})
        return stem(job)+f": complete {100*float(last[meta['evaluation_scope']+'_accuracy']):.2f}%"


def read_tables(job, science, out=OUT):
    receipt = read_receipt(job, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError("Complete receipt required")
    tables = {kind: pd.read_csv(paths(job, out)[kind], float_precision="round_trip", keep_default_na=False)
              for kind in ("rounds", "history", "summary")}
    validate_tables(tables, job, expected_identity=receipt["identity"])
    return tables


def coverage(phase, science, out=OUT):
    rows = []
    for job in jobs_for(phase):
        receipt = read_receipt(job, science, out)
        progress = Path(out)/"progress"/(stem(job)+".json")
        latest = json.loads(progress.read_text()).get("round", 0) if progress.exists() else 0
        rows.append({"phase": phase, "active_clients": job[1], "method": job[2], "partition_seed": job[3],
            "state": receipt["state"] if receipt else "pending", "latest_checkpoint_round": latest,
            "failed_round": receipt.get("failed_round", "") if receipt else ""})
    return pd.DataFrame(rows)


def confirmation_freeze(science, out):
    receipts = {}
    for job in jobs_for("pilot"):
        receipt = read_receipt(job, science, out)
        if receipt is None:
            raise RuntimeError("Finish all fifteen pilot jobs before confirmation (no ranking gate)")
        if receipt["state"] == "complete":
            read_tables(job, science, out)
        receipts[stem(job)] = data.file_hash(paths(job, out)["receipt"])
    value = {"contract_sha256": data.digest(science), "pilot_receipts_sha256": receipts,
        "criterion": "all_pilot_jobs_terminal_and_protocol_valid_no_method_ranking_gate",
        "frozen_configs": science["protocol"]["configs"], "targets": list(TARGETS),
        "official_test_parsed_by_campaign_before_this_freeze": False,
        "confirmation_development_checkpoint_reuse": False}
    data.atomic_json(Path(out)/"confirmation_freeze.json", value, immutable=True)


def run_jobs(phase, *, workers, out=OUT, root=DATA, method=None, seed_index=None, active=None):
    out = Path(out)
    cache, science = prepare(out, root)
    load_contract(out, check_runtime=True)
    if phase == "confirmation":
        confirmation_freeze(science, out)
        cache, manifests = data.prepare_cache(root, include_test=True)
        data.atomic_json(out/"confirmation_test_manifest.json", manifests["test"], immutable=True)
    jobs = jobs_for(phase)
    if method is not None:
        jobs = [j for j in jobs if j[2] == method]
    if active is not None:
        jobs = [j for j in jobs if j[1] == active]
    if seed_index is not None:
        if phase == "pilot" or not 0 <= seed_index < len(SEEDS):
            raise ValueError("Confirmation seed index must be 0..9")
        jobs = [j for j in jobs if j[3] == SEEDS[seed_index]]
    jobs = [j for j in jobs if read_receipt(j, science, out) is None]
    workers = resolve_workers(workers)
    print(f"{phase}: {len(jobs)} pending jobs; up to {min(workers, len(jobs))} processes, one BLAS thread each.", flush=True)
    if jobs:
        errors = []
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                mp_context=multiprocessing.get_context("spawn"), initializer=worker_init,
                initargs=(str(cache), phase == "confirmation")) as pool:
            futures = {pool.submit(run_job, (job, science, str(out))): job for job in jobs}
            for future in as_completed(futures):
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    errors.append(stem(futures[future])+": "+repr(exc))
        if errors:
            raise RuntimeError("Worker errors (checkpoints preserved; completed outputs retained):\n"+"\n".join(errors))
    from calibrated_participation_analysis import analyze
    analyze(phase, out)


def resolve_workers(value):
    if value == "auto":
        return max(1, min(8, (os.cpu_count() or 2)//2))
    number = int(value)
    if number <= 0:
        raise ValueError("workers must be positive or auto")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("verify", "prepare", "pilot", "confirm", "analyze", "status"))
    parser.add_argument("--workers", default="auto", help="Independent processes; auto uses half the CPU threads, capped at eight")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--active-clients", type=int, choices=ACTIVE)
    args = parser.parse_args()
    if args.command == "verify":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
            "-p", "test_calibrated_participation.py", "-v"], cwd=REPO, check=True)
        print("Synthetic verification only; no dataset download or paper evidence.")
    elif args.command == "prepare":
        prepare(args.out, args.data)
    elif args.command in ("pilot", "confirm"):
        run_jobs("pilot" if args.command == "pilot" else "confirmation", workers=args.workers,
            out=args.out, root=args.data, method=args.method, seed_index=args.seed_index,
            active=args.active_clients)
    elif args.command == "analyze":
        from calibrated_participation_analysis import analyze
        analyze(args.phase, args.out)
    elif not (args.out/"protocol.json").exists():
        print("Not prepared. Run prepare or pilot.")
    else:
        science = load_contract(args.out)
        for phase in ("pilot", "confirmation"):
            if phase == "confirmation" and not (args.out/"confirmation_freeze.json").exists():
                print("confirmation: not frozen/started")
                continue
            rows = coverage(phase, science, args.out)
            print(phase+"\n"+rows.groupby(["active_clients", "method", "state"]).size().to_string())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
