"""120-round development-only Event/STC lossless-message audit.

python experiments/event_codec_pilot.py verify
python experiments/event_codec_pilot.py preflight
python experiments/event_codec_pilot.py run --workers 2
python experiments/event_codec_pilot.py analyze
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
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

import stc_cifar10_development as dev
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.cifar10_benchmark import download_cifar10, make_cifar10_federation
from neuromorphicfl.event_codec import PROTOCOL
from neuromorphicfl.event_codec_pilot import run_codec_pilot, traffic_summary
from p12_headline_ten_seed import CIFAR_BASE
from stc_stage1 import smoke_federation

OUT = REPO / "experiments/results/event_codec_pilot_v1"
DATA = REPO / "data/cifar-10"
ROUNDS = 120
PARTITION_SEED, TRAIN_SEED = 4500, 84500
POINTS = ("event", "stc_primary", "stc_sparse")


def config_for(point):
    if point not in POINTS:
        raise ValueError(point)
    fraction = 1.0 if point == "stc_primary" else 0.25
    return replace(CIFAR_BASE, rounds=ROUNDS, eval_stride=30,
                   server_gain=1.0 if point == "event" else 2.0,
                   stc_upload_fraction=fraction, stc_download_fraction=fraction)


def text_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    table.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(temporary, path)


def spec():
    return {"campaign": "event_codec_pilot_v1", "codec": PROTOCOL, "rounds": ROUNDS,
            "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
            "architecture": "cifar_cnn", "dimension": 20570, "clients": 10, "regime": "strong",
            "scope": "codec_correctness_and_early_traffic_not_convergence",
            "evaluation": "training_subset_only_no_test_accuracy",
            "points": {point: asdict(config_for(point)) for point in POINTS},
            "checkpoint_stride": 30, "header_bits_new_integer_frame": 128,
            "legacy_event_uplink": "analytic_64_bit_header_plus_address_and_sign",
            "new_messages": "actual_bytes_including_all_codec_metadata_and_padding",
            "downlink": "shortest_exact_counts_with_float32_patches_or_ordered_pulses",
            "integer_only_cost": "diagnostic_not_claimed_bitwise_legacy_equivalent",
            "stc_control": "native_quantization_and_residuals_unchanged_lossless_reencoding_only",
            "timing": "audit_overhead_includes_reference_codecs_and_reconstruction_checks",
            "hash_format": "sha256_utf8_lf_v1"}


def sources():
    paths = list((REPO / "src/neuromorphicfl").glob("*.py")) + [
        Path(__file__), REPO / "experiments/p12_headline_ten_seed.py",
        REPO / "experiments/stc_cifar10_development.py"]
    return {path.relative_to(REPO).as_posix(): text_hash(path) for path in sorted(paths)}


def contract():
    extracted = download_cifar10(DATA)
    return {"protocol": spec(), "sources": sources(), "runtime": dev.runtime_manifest(),
            "data_sha256": {name: dev.file_hash(extracted / name) for name in
                            [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]}}


def paths(point, out=OUT):
    return {kind: out / f"{point}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("summary", "rounds", "history", "receipt")}


def read_receipt(point, science, out=OUT):
    files = paths(point, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("contract_sha256") != digest(science) or receipt.get("point") != point:
        raise RuntimeError("Pilot receipt identity changed")
    if receipt["state"] == "complete":
        for kind in ("summary", "rounds", "history"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Pilot output changed: {point}/{kind}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown pilot receipt state")
    return receipt


def run_point(point, science, out=OUT):
    with dev.point_lock(out / "locks" / f"{point}.lock"):
        old = read_receipt(point, science, out)
        if old is not None:
            return f"reuse {point}: {old['state']}"
        if contract() != science:
            raise RuntimeError("Pilot source/data/runtime changed")
        federation = make_cifar10_federation(root=DATA, regime="strong", seed=PARTITION_SEED,
                                            train_eval_size=5000, n_clients=10)
        federation = replace(federation, X_test=federation.X_train_eval, y_test=federation.y_train_eval)
        checkpoint = BaselineCheckpoint(out / "checkpoints" / f"{point}.npz",
                                        {"contract_sha256": digest(science), "point": point})

        def progress(row):
            if row["round"] == 1 or row["round"] % 30 == 0:
                print(f"{point}: {row['round']}/{ROUNDS}; upload={row['codec_uplink_bits']} bits; "
                      f"server={row['exact_downlink_bits']} bits", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = run_codec_pilot(federation=federation, architecture="cifar_cnn",
                                         method="event" if point == "event" else "stc",
                                         config=config_for(point), seed=TRAIN_SEED,
                                         checkpoint=checkpoint, progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"point": point, "contract_sha256": digest(science),
                         "state": "numerical_failure", "error": str(exc)}, paths(point, out)["receipt"])
            return f"numerical failure {point}: {exc}"
        save_result(point, science, result, out)
        return f"done {point}: {ROUNDS} rounds, all roundtrips exact"


def save_result(point, science, result, out):
    files = paths(point, out)
    final = result["history"].iloc[-1]
    meta = {"point": point, "partition_seed": PARTITION_SEED, "train_seed": TRAIN_SEED,
            "evaluation_scope": "development_training_only", "contract_sha256": digest(science)}
    summary = result["summary"].assign(**meta, final_train_objective=float(final.train_objective),
                                       final_train_ce=float(final.train_ce),
                                       model_sha256=result["rounds"].iloc[-1].model_sha256)
    for kind, frame in (("summary", summary), ("rounds", result["rounds"]), ("history", result["history"])):
        atomic_csv(frame.assign(**meta), files[kind])
    atomic_json({"point": point, "contract_sha256": digest(science), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in ("summary", "rounds", "history")}}, files["receipt"])


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science["protocol"] != spec() or science["sources"] != sources():
        raise RuntimeError("Pilot results belong to different source/protocol versions")
    return science


def run(workers, point=None):
    science = contract()
    if (OUT / "protocol.json").exists():
        if json.loads((OUT / "protocol.json").read_text(encoding="utf-8")) != science:
            raise RuntimeError("Existing pilot source/data/runtime differs; preserve its evidence")
    else:
        atomic_json(science, OUT / "protocol.json")
    points = (point,) if point else POINTS
    if workers == 1:
        for name in points:
            print(run_point(name, science), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_point, name, science) for name in points]
            for future in as_completed(jobs):
                print(future.result(), flush=True)


def analyze(out=OUT):
    science = load_contract(out)
    tables, audit = [], []
    for point in POINTS:
        receipt = read_receipt(point, science, out)
        if receipt is None or receipt["state"] != "complete":
            raise RuntimeError(f"Pilot incomplete: {point}; no point is silently omitted")
        files = paths(point, out)
        rounds = pd.read_csv(files["rounds"], float_precision="round_trip")
        if rounds["round"].tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError("Pilot round coverage differs")
        summary = pd.read_csv(files["summary"], float_precision="round_trip")
        rebuilt = traffic_summary(rounds, n_clients=science["protocol"]["clients"],
                                  dimension=science["protocol"]["dimension"])
        for field in ("variant", "uplink_bits", "server_stream_bits", "unicast_downlink_bits",
                      "unicast_total_bits", "broadcast_downlink_bits", "broadcast_total_bits"):
            if summary[field].tolist() != rebuilt[field].tolist():
                raise RuntimeError(f"Traffic reconstruction mismatch: {point}/{field}")
        tables.append(summary)
        audit.append({"point": point, "rounds": ROUNDS, "all_frames_verified_at_execution": True,
                      "model_sha256": rounds.iloc[-1].model_sha256,
                      "total_roundoff_corrections": int(rounds.correction_nonzeros.sum()),
                      "max_canonical_error": float(rounds.canonical_max_abs_error.max()),
                      "codec_audit_seconds": float(rounds.codec_audit_seconds.sum())})
    combined = pd.concat(tables, ignore_index=True)
    atomic_csv(combined, out / "comparison.csv")
    atomic_json({"scope": spec()["scope"], "points": audit,
                 "convergence_or_accuracy_ranking": False}, out / "analysis_audit.json")
    print(combined[["point", "variant", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "unicast_saving_percent"]].to_string(index=False))
    print("120-round codec audit only; no convergence or test-accuracy conclusion.")


def preflight():
    archived = REPO / "experiments/results/stc_cifar10_development_v1/protocol.json"
    if dev.source_hashes() != json.loads(archived.read_text(encoding="utf-8"))["source_hashes"]:
        raise RuntimeError("Frozen STC development sources changed")
    print("Frozen STC development sources: unchanged")
    print("Pilot: compact CIFAR CNN; 10 clients; partition 4500; 120 rounds per point")
    print("Points: Event, frozen STC p=1/gain=2, frozen STC p=.25/gain=2")
    print("No held-out test evaluation or parameter selection")
    print("Existing CIFAR data:", (DATA / "cifar-10-batches-py/data_batch_1").exists())


def smoke():
    with tempfile.TemporaryDirectory() as folder:
        for point in POINTS:
            config = replace(config_for(point), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
            kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn",
                          method="event" if point == "event" else "stc", config=config, seed=TRAIN_SEED)
            uninterrupted = run_codec_pilot(**kwargs)
            checkpoint = BaselineCheckpoint(Path(folder) / f"{point}.npz", {"synthetic": True})
            run_codec_pilot(**kwargs, checkpoint=checkpoint, stop_after=1)
            resumed = run_codec_pilot(**kwargs, checkpoint=checkpoint)
            np.testing.assert_array_equal(uninterrupted["model"].view(np.uint32), resumed["model"].view(np.uint32))
            pd.testing.assert_frame_equal(uninterrupted["summary"], resumed["summary"])
            print(f"smoke {point}: exact upload/downlink reconstruction and resume")
    print("Synthetic smoke only; not paper evidence.")


def spawn_smoke_worker():
    config = replace(config_for("event"), rounds=1, eval_stride=1, local_steps=1, batch_size=2)
    result = run_codec_pilot(federation=smoke_federation(), architecture="cifar_cnn",
                             method="event", config=config, seed=TRAIN_SEED)
    return result["rounds"].iloc[-1].model_sha256


def spawn_smoke():
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        actual = pool.submit(spawn_smoke_worker).result()
    if actual != spawn_smoke_worker():
        raise AssertionError("spawned pilot differs from sequential pilot")
    print("Spawned-worker smoke: matching model checksum")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "status", "analyze"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--point", choices=POINTS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("workers must be 1, 2 or 3")
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_event_codec*.py", "-v"],
                       cwd=REPO, env=env, check=True)
        smoke()
        spawn_smoke()
    elif args.command == "smoke":
        smoke()
        spawn_smoke()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        run(args.workers, args.point)
    elif args.command == "status":
        if not (OUT / "protocol.json").exists():
            print("Pilot not started")
        else:
            science = load_contract()
            for point in POINTS:
                receipt = read_receipt(point, science)
                print(point, receipt["state"] if receipt else "pending/checkpoint")
    else:
        analyze()


if __name__ == "__main__":
    main()
