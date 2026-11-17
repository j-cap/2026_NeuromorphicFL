"""Stage 2: fixed, development-only STC search on the compact CIFAR-10 CNN.

From the research-code repository root (Windows and Linux):
    python experiments/stc_cifar10_development.py smoke
    python experiments/stc_cifar10_development.py plan
    python experiments/stc_cifar10_development.py develop --workers 2
    python experiments/stc_cifar10_development.py verify
    python experiments/stc_cifar10_development.py select
    python experiments/stc_cifar10_development.py analyze

Restart develop to skip completed points and resume 60-round checkpoints.
No held-out evaluation command is provided at this stage.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for key in THREAD_KEYS:
    os.environ[key] = "1"

import numpy as np
import pandas as pd

from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.cifar10_benchmark import download_cifar10, make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from neuromorphicfl.stc import STC_PROTOCOL
from p12_headline_ten_seed import CIFAR_BASE

CAMPAIGN = "stc_cifar10_development_v1"
DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / CAMPAIGN
ROUNDS = 1800
DEV_PARTITION = 4500
TRAIN_SEED = 80000 + DEV_PARTITION
EVAL_PARTITIONS = tuple(range(3500, 4500, 100))
FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
GAINS = (1.0, 2.0)
SOURCE_FILES = (
    "experiments/stc_cifar10_development.py",
    "experiments/p12_headline_ten_seed.py",
    "src/neuromorphicfl/final_baseline_campaign.py",
    "src/neuromorphicfl/baseline_checkpoint.py",
    "src/neuromorphicfl/stc.py",
    "src/neuromorphicfl/cifar10_benchmark.py",
)


@dataclass(frozen=True)
class Point:
    point_id: str
    method: str
    upload_fraction: float = 0.01
    download_fraction: float = 0.01
    server_gain: float = 1.0


def label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


PAIRS = tuple((p, p) for p in FRACTIONS) + ((0.0025, 0.01), (0.01, 0.0025))
CANDIDATES = tuple(
    Point(f"stc_u{label(up)}_d{label(down)}_g{label(gain)}", "stc", up, down, gain)
    for up, down in PAIRS for gain in GAINS
)
REFERENCE = Point("event_reference", "event")
POINTS = (REFERENCE,) + CANDIDATES
POINT_BY_ID = {point.point_id: point for point in POINTS}


def config_for(point: Point):
    return replace(
        CIFAR_BASE, rounds=ROUNDS, eval_stride=60,
        stc_upload_fraction=point.upload_fraction,
        stc_download_fraction=point.download_fraction,
        server_gain=point.server_gain, server_gain_exponent=0.0,
    )


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def protocol_spec() -> dict:
    assert DEV_PARTITION not in EVAL_PARTITIONS
    return {
        "campaign": CAMPAIGN, "stc_protocol": STC_PROTOCOL,
        "architecture": "cifar_cnn", "regime": "strong", "n_clients": 10,
        "partition_seed": DEV_PARTITION, "train_seed": TRAIN_SEED,
        "excluded_evaluation_partitions": list(EVAL_PARTITIONS),
        "rounds": ROUNDS, "train_eval_size": 5000, "checkpoint_stride": 60,
        "base_config": asdict(config_for(REFERENCE)),
        "points": [asdict(point) for point in POINTS],
        "selection_metric": "final_train_objective",
        "selection_ties": ["unicast_hybrid_total_bits", "point_id"],
        "budget_rule": "best training objective at or below event_reference unicast bits",
        "nearest_traffic_rule": "minimum absolute log traffic ratio; diagnostic only",
        "evaluation_scope": "development_training_only",
    }


def source_hashes() -> dict:
    # Normalize newlines so Git's Windows checkout policy is immaterial.
    return {
        name: hashlib.sha256((REPO / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()
        for name in SOURCE_FILES
    }


def runtime_manifest() -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "system": platform.system(),
        "machine": platform.machine(), "byteorder": sys.byteorder,
        "numpy_build_sha256": hashlib.sha256(buffer.getvalue().encode()).hexdigest(),
        "thread_limits": {key: os.environ.get(key) for key in THREAD_KEYS},
    }


def build_contract(data_root: Path) -> dict:
    extracted = download_cifar10(data_root)
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    return {
        "protocol": protocol_spec(), "source_hashes": source_hashes(),
        "runtime": runtime_manifest(),
        "data_sha256": {name: file_hash(extracted / name) for name in names},
    }


def ensure_contract(out: Path, contract: dict) -> None:
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Campaign source, data, runtime or protocol changed. Preserve the existing results and investigate before starting a new campaign version.")
    else:
        atomic_json(contract, path)


def load_contract(out: Path = OUT) -> dict:
    path = out / "protocol.json"
    if not path.exists():
        raise RuntimeError("No campaign protocol found; run develop first.")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract["protocol"] != protocol_spec() or contract["source_hashes"] != source_hashes():
        raise RuntimeError("Results belong to a different protocol or source version.")
    return contract


def paths(point: Point, out: Path = OUT) -> tuple[Path, Path, Path]:
    stem = f"{point.point_id}_p{DEV_PARTITION}_r{ROUNDS}"
    return tuple(out / (stem + suffix) for suffix in ("_summary.csv", "_history.csv", "_receipt.json"))


def job_identity(point: Point, contract: dict) -> dict:
    return {
        "campaign_sha256": digest(contract), "point": asdict(point),
        "config_sha256": digest(asdict(config_for(point))),
    }


def read_receipt(point: Point, contract: dict, out: Path = OUT) -> dict | None:
    summary, history, receipt_path = paths(point, out)
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("identity") != job_identity(point, contract):
        raise RuntimeError(f"Result identity mismatch: {point.point_id}")
    if receipt.get("state") == "complete":
        for kind, path in (("summary", summary), ("history", history)):
            if not path.exists() or file_hash(path) != receipt.get(f"{kind}_sha256"):
                raise RuntimeError(f"Missing or changed {kind}: {point.point_id}")
    elif receipt.get("state") != "numerical_failure":
        raise RuntimeError(f"Unknown completion state: {point.point_id}")
    return receipt


@contextmanager
def point_lock(path: Path):
    """OS locks release on process exit, including interrupted Windows runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError(f"Point is already running: {path.stem}") from exc
        try:
            yield
        finally:
            release()


def save_result(point: Point, contract: dict, result: dict, out: Path, elapsed: float) -> None:
    history = result.pop("history")
    # The secondary evaluation slot also contains training samples in this
    # campaign. Remove its generic test/accuracy labels from exported evidence.
    result = {key: value for key, value in result.items() if "accuracy" not in key and "test" not in key}
    history = history[[name for name in history if "accuracy" not in name and "test" not in name]].copy()
    meta = {
        "point_id": point.point_id, "campaign_sha256": digest(contract),
        "config_sha256": digest(asdict(config_for(point))),
        "partition_seed": DEV_PARTITION, "train_seed": TRAIN_SEED,
        "evaluation_scope": "development_training_only",
    }
    result.update(meta, elapsed_seconds_this_attempt=elapsed)
    summary_path, history_path, receipt_path = paths(point, out)
    atomic_csv(pd.DataFrame([result]), summary_path)
    atomic_csv(history.assign(**meta), history_path)
    # Written last: presence alone never makes half-written outputs complete.
    atomic_json({
        "state": "complete", "identity": job_identity(point, contract),
        "summary_sha256": file_hash(summary_path),
        "history_sha256": file_hash(history_path),
    }, receipt_path)


def numerical_failure(point: Point, contract: dict, out: Path, error: Exception, started: float) -> str:
    atomic_json({
        "state": "numerical_failure", "identity": job_identity(point, contract),
        "error": str(error), "elapsed_seconds_this_attempt": time.monotonic() - started,
    }, paths(point, out)[2])
    return f"numerical failure {point.point_id}: {error}"


def run_point(point: Point, contract: dict, data_root: Path = DATA, out: Path = OUT) -> str:
    with point_lock(out / "locks" / (point.point_id + ".lock")):
        receipt = read_receipt(point, contract, out)
        if receipt is not None:
            return f"reuse {point.point_id}: {receipt['state']}"
        if build_contract(data_root) != contract:
            raise RuntimeError("Worker source/data/runtime differs from the frozen campaign.")
        fed = make_cifar10_federation(
            root=data_root, regime="strong", seed=DEV_PARTITION,
            train_eval_size=5000, n_clients=10,
        )
        # Never evaluate CIFAR-10 test images while selecting configurations.
        fed = replace(fed, X_test=fed.X_train_eval, y_test=fed.y_train_eval)
        checkpoint = BaselineCheckpoint(
            out / "checkpoints" / (point.point_id + ".npz"), job_identity(point, contract),
        )
        started = time.monotonic()

        def progress(row):
            rnd = int(row["round"])
            if rnd == 1 or rnd % 180 == 0 or rnd == ROUNDS:
                print(f"{point.point_id}: {rnd}/{ROUNDS}, train={row['train_objective']:.6f}", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = run_final_baseline(
                    federation=fed, architecture="cifar_cnn", method=point.method,
                    config=config_for(point), seed=TRAIN_SEED, record_history=True,
                    checkpoint=checkpoint, checkpoint_stride=60,
                    progress_callback=progress,
                )
            if not np.isfinite(result["final_train_objective"]):
                raise FloatingPointError("nonfinite final training objective")
        except FloatingPointError as exc:
            return numerical_failure(point, contract, out, exc, started)
        save_result(point, contract, result, out, time.monotonic() - started)
        return f"done {point.point_id}: train={result['final_train_objective']:.6f}"


def develop(workers: int, only: str | None = None) -> None:
    contract = build_contract(DATA)
    ensure_contract(OUT, contract)
    points = (POINT_BY_ID[only],) if only else POINTS
    pending = [point for point in points if read_receipt(point, contract) is None]
    print(f"{len(points) - len(pending)}/{len(points)} attempted; {len(pending)} pending", flush=True)
    if workers == 1:
        for point in pending:
            print(run_point(point, contract), flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = {pool.submit(run_point, point, contract): point for point in pending}
            for future in as_completed(jobs):
                print(future.result(), flush=True)


def verified_rows(contract: dict, out: Path = OUT) -> pd.DataFrame:
    rows = []
    for point in POINTS:
        receipt = read_receipt(point, contract, out)
        if receipt is None:
            raise RuntimeError(f"Incomplete development grid: {point.point_id}")
        if receipt["state"] == "numerical_failure":
            rows.append({"point_id": point.point_id, "method": point.method,
                         "state": "numerical_failure", "error": receipt["error"]})
            continue
        summary, history, _ = paths(point, out)
        frame = pd.read_csv(summary, float_precision="round_trip")
        trajectory = pd.read_csv(history, float_precision="round_trip")
        if len(frame) != 1:
            raise RuntimeError(f"Expected one summary row: {point.point_id}")
        row = frame.iloc[0].to_dict()
        expected = {
            "point_id": point.point_id, "method": point.method,
            "rounds": ROUNDS, "partition_seed": DEV_PARTITION,
            "train_seed": TRAIN_SEED, "campaign_sha256": digest(contract),
            "config_sha256": digest(asdict(config_for(point))),
            "evaluation_scope": "development_training_only",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Summary metadata drift: {point.point_id}")
        expected_rounds = [1] + list(range(60, ROUNDS + 1, 60))
        if trajectory["round"].tolist() != expected_rounds:
            raise RuntimeError(f"History horizon/stride mismatch: {point.point_id}")
        for key in ("point_id", "campaign_sha256", "config_sha256", "partition_seed", "train_seed", "evaluation_scope"):
            if key not in trajectory or not trajectory[key].eq(expected[key]).all():
                raise RuntimeError(f"History metadata drift: {point.point_id}")
        if any("accuracy" in key or "test" in key for key in frame.columns):
            raise RuntimeError("Development output unexpectedly contains test/accuracy fields")
        if not np.isfinite(trajectory[["train_objective", "train_ce"]].to_numpy()).all():
            raise RuntimeError(f"Nonfinite development history: {point.point_id}")
        score, bits = float(row["final_train_objective"]), float(row["unicast_hybrid_total_bits"])
        if not np.isfinite(score) or not np.isfinite(bits) or bits <= 0:
            raise RuntimeError(f"Invalid objective/traffic: {point.point_id}")
        if row["uplink_packetized_bits"] + row["unicast_hybrid_downlink_bits"] != bits:
            raise RuntimeError(f"Endpoint traffic mismatch: {point.point_id}")
        if not np.isclose(trajectory.iloc[-1]["train_objective"], score, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Endpoint/history objective mismatch: {point.point_id}")
        if point.method == "stc" and trajectory.iloc[-1]["unicast_total_bits"] != bits:
            raise RuntimeError(f"Native STC traffic history mismatch: {point.point_id}")
        rows.append(dict(row, state="complete"))
    frame = pd.DataFrame(rows)
    reference = frame[frame["point_id"] == REFERENCE.point_id].iloc[0]
    if reference["state"] != "complete":
        raise RuntimeError("Event development reference failed; a budget cannot be selected.")
    return frame


def choose(frame: pd.DataFrame) -> dict:
    valid = frame[(frame["method"] == "stc") & (frame["state"] == "complete")].copy()
    if valid.empty:
        raise RuntimeError("No finite STC candidate completed")
    ordering = ["final_train_objective", "unicast_hybrid_total_bits", "point_id"]
    ranked = valid.sort_values(ordering)
    budget = float(frame[frame["point_id"] == REFERENCE.point_id].iloc[0]["unicast_hybrid_total_bits"])
    feasible = ranked[ranked["unicast_hybrid_total_bits"] <= budget]
    near = valid.assign(log_gap=np.abs(np.log(valid["unicast_hybrid_total_bits"] / budget)))
    nearest = near.sort_values(["log_gap"] + ordering).iloc[0]

    def choice(row):
        point = POINT_BY_ID[str(row["point_id"])]
        bits = float(row["unicast_hybrid_total_bits"])
        return {
            **asdict(point), "final_train_objective": float(row["final_train_objective"]),
            "unicast_bits": int(bits), "traffic_ratio_to_event": bits / budget,
        }

    return {
        "quality": choice(ranked.iloc[0]),
        "within_event_budget": choice(feasible.iloc[0]) if not feasible.empty else None,
        "event_reference_unicast_bits": int(budget),
        "nearest_traffic_diagnostic": {
            **choice(nearest),
            "relative_budget_gap": abs(float(nearest["unicast_hybrid_total_bits"]) / budget - 1),
            "within_ten_percent": bool(abs(float(nearest["unicast_hybrid_total_bits"]) / budget - 1) <= 0.1),
        },
    }


def select(out: Path = OUT) -> dict:
    contract = load_contract(out)
    frame = verified_rows(contract, out)
    selection = {
        "campaign_sha256": digest(contract), "selection_metric": "final_train_objective",
        "partition_seed": DEV_PARTITION, "train_seed": TRAIN_SEED,
        "rounds": ROUNDS, "choices": choose(frame),
        "numerical_failures": sorted(frame[frame["state"] == "numerical_failure"]["point_id"].tolist()),
        "receipt_sha256": {p.point_id: file_hash(paths(p, out)[2]) for p in POINTS},
    }
    path = out / "frozen_selection.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != selection:
        raise RuntimeError("A different frozen selection already exists; investigate the changed evidence.")
    atomic_json(selection, path)
    print(json.dumps(selection["choices"], indent=2), flush=True)
    return selection


def analyze(out: Path = OUT) -> None:
    contract = load_contract(out)
    frame = verified_rows(contract, out)
    atomic_csv(frame, out / "development_summary.csv")
    ranking = frame[(frame["method"] == "stc") & (frame["state"] == "complete")].sort_values(
        ["final_train_objective", "unicast_hybrid_total_bits", "point_id"]
    )
    atomic_csv(ranking, out / "development_ranking.csv")
    finite = frame[frame["state"] == "complete"]
    frontier = []
    for _, row in finite.iterrows():
        cheaper = finite["unicast_hybrid_total_bits"] <= row["unicast_hybrid_total_bits"]
        better = finite["final_train_objective"] <= row["final_train_objective"]
        strict = (finite["unicast_hybrid_total_bits"] < row["unicast_hybrid_total_bits"]) | (finite["final_train_objective"] < row["final_train_objective"])
        if not (cheaper & better & strict).any():
            frontier.append(row.to_dict())
    atomic_csv(pd.DataFrame(frontier), out / "development_frontier.csv")
    print(ranking[["point_id", "final_train_objective", "unicast_hybrid_total_bits"]].to_string(index=False))
    print("Development training objectives only; final evaluation is a later stage.")


def worker_signature() -> dict:
    return {"runtime": runtime_manifest(), "protocol": digest(protocol_spec()),
            "sources": source_hashes()}


def smoke() -> None:
    from stc_stage1 import smoke_federation
    fed = smoke_federation()
    config = replace(config_for(CANDIDATES[0]), rounds=4, eval_stride=1, local_steps=1, batch_size=2)
    kwargs = dict(federation=fed, architecture="cifar_cnn", method="stc", config=config, seed=TRAIN_SEED, record_history=True)
    continuous = run_final_baseline(**kwargs)
    with tempfile.TemporaryDirectory(prefix="stc-resume-smoke-") as directory:
        store = BaselineCheckpoint(Path(directory) / "state.npz", {"scope": "synthetic-smoke"})

        def interrupt(row):
            if row["round"] == 2:
                raise InterruptedError("intentional smoke interruption")

        try:
            run_final_baseline(**kwargs, checkpoint=store, checkpoint_stride=2, progress_callback=interrupt)
        except InterruptedError:
            pass
        resumed = run_final_baseline(**kwargs, checkpoint=store, checkpoint_stride=2)
    for key, value in continuous.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, resumed[key], check_exact=True)
        elif value != resumed[key]:
            raise RuntimeError(f"Resume smoke mismatch: {key}")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        signatures = [pool.submit(worker_signature) for _ in range(2)]
        if any(future.result() != worker_signature() for future in signatures):
            raise RuntimeError("Spawned worker runtime/source signature differs")
    print("Stage 2 checkpoint/resume and spawned-worker smoke passed; no data downloaded or campaign outputs written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "smoke", "prepare-data", "status", "verify", "select", "analyze"):
        sub.add_parser(command)
    develop_parser = sub.add_parser("develop")
    develop_parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    point_parser = sub.add_parser("point")
    point_parser.add_argument("--point-id", choices=POINT_BY_ID, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "plan":
        print(f"{CAMPAIGN}: {len(CANDIDATES)} STC configurations + one Event development reference")
        print(f"{ROUNDS} rounds each, partition={DEV_PARTITION}, SGD seed={TRAIN_SEED}; no evaluation-seed runs")
        print(pd.DataFrame([asdict(point) for point in POINTS]).to_string(index=False))
    elif args.command == "prepare-data":
        contract = build_contract(DATA)
        print(json.dumps(contract["data_sha256"], indent=2))
    elif args.command == "develop":
        develop(args.workers)
    elif args.command == "point":
        develop(1, args.point_id)
    elif args.command == "status":
        contract = load_contract()
        counts = {"complete": 0, "numerical_failure": 0, "pending": 0}
        for point in POINTS:
            receipt = read_receipt(point, contract)
            state = receipt["state"] if receipt else "pending"
            counts[state] += 1
            print(f"{point.point_id}: {state}")
        print(counts)
    elif args.command == "verify":
        frame = verified_rows(load_contract())
        print(f"Verified all {len(frame)} development attempts: {frame['state'].value_counts().to_dict()}")
    elif args.command == "select":
        select()
    elif args.command == "analyze":
        analyze()


if __name__ == "__main__":
    main()
