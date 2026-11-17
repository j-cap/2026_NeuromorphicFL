"""Fixed two-point, ten-seed STC confirmation; no tuning or winner gate.

Commands from j-cap/2026_NeuromorphicFL:
    python experiments/stc_cifar10_confirmation.py smoke
    python experiments/stc_cifar10_confirmation.py preflight
    python experiments/stc_cifar10_confirmation.py confirm --workers 2
    python experiments/stc_cifar10_confirmation.py verify
    python experiments/stc_cifar10_confirmation.py analyze
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

import stc_cifar10_development as dev
from stc_cifar10_development_audit import validated_evidence
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.cifar10_benchmark import LAYOUT, make_cifar10_federation
from neuromorphicfl.confirmation_metrics import (
    add_regularization_columns, legacy_traffic_history, paired_endpoints,
    paired_targets, target_crossings, validate_native_traffic,
)
from neuromorphicfl.final_baseline_campaign import run_final_baseline

CAMPAIGN = "stc_cifar10_confirmation_v1"
OUT = REPO / "experiments" / "results" / CAMPAIGN
SEEDS = tuple(range(3500, 4500, 100))
ROUNDS = 1800
TARGETS = (0.45, 0.50, 0.55)
EXPECTED_ROUNDS = (1,) + tuple(range(60, ROUNDS + 1, 60))
SOURCE_FILES = tuple(sorted(set(dev.SOURCE_FILES) | {
    "experiments/stc_cifar10_confirmation.py",
    "experiments/stc_cifar10_development_audit.py",
    "experiments/p19_cifar10_cnn_final_horizons.py",
    "experiments/m3_dense_schedule.py",
    "src/neuromorphicfl/confirmation_metrics.py",
}))


@dataclass(frozen=True)
class Point:
    point_id: str
    role: str
    development_point_id: str


POINTS = (
    Point("stc_primary", "primary_frozen_selection", "stc_u1_d1_g2"),
    Point("stc_sparse_secondary", "secondary_predeclared", "stc_u0p25_d0p25_g2"),
)
POINT_BY_ID = {point.point_id: point for point in POINTS}
REFERENCE_IDS = ("event", "dense_constant", "dense_scheduled", "ef_topk", "strom_style")
P19_NAMES = {"event": "cifar_event", "dense_constant": "cifar_dense_gain2",
             "ef_topk": "cifar_ef_quality", "strom_style": "cifar_strom_quality"}


def config_for(point: Point):
    return dev.config_for(dev.POINT_BY_ID[point.development_point_id])


def reference_config(name: str):
    base = dev.config_for(dev.REFERENCE)
    if name == "event":
        return base
    if name == "dense_constant":
        return replace(base, server_gain=2.0)
    if name == "dense_scheduled":
        return replace(base, server_gain=4.0, server_gain_exponent=0.3, server_gain_scale=100.0)
    if name == "ef_topk":
        return replace(base, topk_fraction=0.05)
    if name == "strom_style":
        return replace(base, strom_threshold=0.0025)
    raise ValueError(name)


def text_hash(path: Path) -> str:
    """SHA-256 over UTF-8 text with LF newlines; no numeric reformatting."""
    value = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(temporary, path)


def source_hashes() -> dict:
    return {name: text_hash(REPO / name) for name in SOURCE_FILES}


def protocol_spec() -> dict:
    return {
        "campaign": CAMPAIGN, "architecture": "cifar_cnn", "n_clients": 10,
        "regime": "strong", "rounds": ROUNDS, "eval_stride": 60,
        "partition_seeds": list(SEEDS), "training_seeds": [80000 + seed for seed in SEEDS],
        "excluded_development_seed": dev.DEV_PARTITION,
        "points": [{**asdict(point), "config": asdict(config_for(point))} for point in POINTS],
        "new_training_runs": len(POINTS) * len(SEEDS),
        "secondary_rationale": "development-only near-primary objective at substantially lower traffic",
        "primary_endpoint": "stc_primary minus event final test accuracy in percentage points",
        "secondary_endpoints": ["final_test_ce", "first_sampled_accuracy_target_traffic",
                                "upload_bits", "download_bits", "broadcast_and_unicast_total_bits"],
        "diagnostics": ["training_objective", "training_cross_entropy", "L2_penalty", "parameter_L2_norm"],
        "accuracy_targets": list(TARGETS), "target_crossing": "first_saved_sample_no_interpolation",
        "unreached_targets": "censored_at_horizon_not_achievements",
        "paired_intervals": "descriptive_unadjusted_95_percent_Student_t_across_seed_pairs",
        "traffic_ratios": "geometric_mean_with_paired_log_scale_interval",
        "equivalence_claim": "none_from_intervals_spanning_zero",
        "references": list(REFERENCE_IDS), "checkpoint_stride": 60,
        "receipt_hash_format": "sha256_utf8_lf_v1",
    }


def reference_paths(name: str, seed_index: int) -> tuple[Path, Path]:
    seed = SEEDS[seed_index]
    if name == "dense_scheduled":
        root = REPO / "experiments/results/m3_dense_schedule"
        stem = f"m3_scheduled_dense_p{seed}_r{ROUNDS}"
    else:
        root = REPO / "experiments/results/p19_cifar10_cnn_final_horizons"
        stem = f"{P19_NAMES[name]}_s{seed_index}_p{seed}_r{ROUNDS}"
    return root / (stem + "_summary.csv"), root / (stem + "_history.csv")


def load_reference(name: str, seed_index: int):
    summary_path, history_path = reference_paths(name, seed_index)
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    history = pd.read_csv(history_path, float_precision="round_trip")
    if len(summary) != 1:
        raise RuntimeError(f"Invalid reference row count: {summary_path}")
    row = summary.iloc[0].to_dict()
    seed, config = SEEDS[seed_index], reference_config(name)
    method = "dense" if name.startswith("dense") else "strom" if name == "strom_style" else name
    expected = {"method": method, "architecture": "cifar_cnn", "dimension": LAYOUT.dimension,
                "point_id": "m3_scheduled_dense" if name == "dense_scheduled" else P19_NAMES[name],
                "partition_seed": seed, "train_seed": 80000 + seed, "rounds": ROUNDS,
                "local_steps": config.local_steps, "local_lr": config.local_lr,
                "server_gain": config.server_gain, "init_scale": config.init_scale}
    expected.update({field: getattr(config, field) for field in (
        "rho", "threshold", "jump0", "jump_exponent", "topk_fraction", "strom_threshold",
        "event_memory", "event_reset",
    )})
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Archived reference metadata mismatch: {name}, seed {seed}")
    if row.get("server_gain_exponent", 0.0) != config.server_gain_exponent:
        raise RuntimeError("Archived gain schedule differs")
    if name == "dense_scheduled":
        if row.get("server_gain_scale") != 100 or row.get("gain_exponent") != 0.3:
            raise RuntimeError("Archived dense schedule differs from its frozen selection")
    if tuple(history["round"]) != EXPECTED_ROUNDS:
        raise RuntimeError("Archived history horizon/stride differs")
    for field in ("point_id", "partition_seed", "train_seed", "rounds"):
        if field not in history or not history[field].eq(expected[field]).all():
            raise RuntimeError(f"Archived history metadata differs: {name}, {field}")
    if not np.isfinite(history[["train_objective", "train_ce", "test_ce", "test_accuracy"]]).all().all():
        raise RuntimeError("Nonfinite archived history")
    for final, recorded in (("final_train_objective", "train_objective"),
                            ("final_train_ce", "train_ce"), ("final_test_ce", "test_ce"),
                            ("final_test_accuracy", "test_accuracy"),
                            ("uplink_packetized_bits", "uplink_packetized_bits")):
        if not np.isclose(row[final], history.iloc[-1][recorded], rtol=1e-8, atol=1e-9):
            raise RuntimeError("Archived endpoint/history mismatch")
    history = legacy_traffic_history(row, history)
    row = add_regularization_columns(pd.DataFrame([row]), config.regularization).iloc[0].to_dict()
    row.update(comparison_id=name, role="archived_reference", regularization=config.regularization,
               batch_size=config.batch_size)
    provenance = {
        "reference": name, "partition_seed": seed,
        "summary_path": summary_path.relative_to(REPO).as_posix(),
        "history_path": history_path.relative_to(REPO).as_posix(),
        "summary_text_sha256": text_hash(summary_path), "history_text_sha256": text_hash(history_path),
        "traffic_history_source": history.traffic_history_source.iloc[0],
        "declared_config": asdict(config),
        "missing_per_run_config_fields": [field for field in asdict(config) if field not in summary],
        "runtime_and_raw_data_provenance": "not_recorded_per_run_in_legacy_archive",
        "compatibility_scope": "recorded_metadata_and_declared_archived_driver_protocol",
    }
    return row, history, provenance


def build_science() -> dict:
    _, selection, audit = validated_evidence()
    if selection["choices"]["quality"]["point_id"] != POINTS[0].development_point_id:
        raise RuntimeError("Primary differs from the original frozen development selection")
    dense = json.loads((REPO / "experiments/results/m3_dense_schedule/frozen_selection.json").read_text())
    if (dense["gain0"], dense["gain_exponent"], dense["gain_scale"], dense["rounds"]) != (4, 0.3, 100, ROUNDS):
        raise RuntimeError("Frozen dense schedule changed")
    references = [load_reference(name, i)[2] for name in REFERENCE_IDS for i in range(len(SEEDS))]
    return {
        "protocol": protocol_spec(), "source_hashes": source_hashes(),
        "development_campaign_sha256": audit["campaign_sha256"],
        "development_selection_sha256": dev.digest(selection),
        "frozen_dense_selection_sha256": dev.digest(dense),
        "reference_provenance": references,
    }


def prepare_execution(science: dict) -> dict:
    development = dev.load_contract()
    runtime = dev.runtime_manifest()
    if runtime != development["runtime"]:
        raise RuntimeError("Training runtime differs from the recorded development environment. Use the original neuromorphicfl-resnet14 environment; read-only analysis remains available on other machines.")
    current = dev.build_contract(dev.DATA)
    if current["data_sha256"] != development["data_sha256"]:
        raise RuntimeError("CIFAR-10 batches differ from development inputs")
    return {"science": science, "runtime": runtime, "data_sha256": current["data_sha256"]}


def ensure_execution(contract: dict, out: Path = OUT) -> None:
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Existing confirmation contract differs; do not mix sources, parameters or environments")
    if not path.exists():
        atomic_json(contract, path)


def load_execution(out: Path = OUT) -> dict:
    contract = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if contract["science"] != build_science():
        raise RuntimeError("Confirmation scientific protocol, sources or archived references changed")
    if contract["data_sha256"] != dev.load_contract()["data_sha256"]:
        raise RuntimeError("Confirmation dataset identity differs")
    return contract


def paths(point: Point, seed_index: int, out: Path = OUT) -> tuple[Path, Path, Path]:
    stem = f"{point.point_id}_s{seed_index}_p{SEEDS[seed_index]}_r{ROUNDS}"
    return tuple(out / (stem + suffix) for suffix in ("_summary.csv", "_history.csv", "_receipt.json"))


def identity(point: Point, seed_index: int, contract: dict) -> dict:
    return {"campaign_sha256": dev.digest(contract), "point": asdict(point),
            "partition_seed": SEEDS[seed_index], "train_seed": 80000 + SEEDS[seed_index],
            "config_sha256": dev.digest(asdict(config_for(point)))}


def metadata(point: Point, seed_index: int, contract: dict) -> dict:
    config = config_for(point)
    return {"point_id": point.point_id, "comparison_id": point.point_id, "role": point.role,
            "development_point_id": point.development_point_id, "seed_index": seed_index,
            "partition_seed": SEEDS[seed_index], "train_seed": 80000 + SEEDS[seed_index],
            "campaign_sha256": dev.digest(contract),
            "config_sha256": dev.digest(asdict(config)),
            "regularization": config.regularization, "batch_size": config.batch_size,
            "evaluation_scope": "paired_confirmation_no_tuning"}


def read_receipt(point: Point, seed_index: int, contract: dict, out: Path = OUT) -> dict | None:
    summary, history, receipt_path = paths(point, seed_index, out)
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("identity") != identity(point, seed_index, contract) or receipt.get("hash_format") != "sha256_utf8_lf_v1":
        raise RuntimeError("Confirmation result identity/hash format mismatch")
    if receipt["state"] == "complete":
        for name, path in (("summary", summary), ("history", history)):
            if not path.exists() or text_hash(path) != receipt.get(name + "_text_sha256"):
                raise RuntimeError(f"Missing or changed confirmation {name}: {point.point_id}, seed {seed_index}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown confirmation completion state")
    return receipt


def save_result(point: Point, seed_index: int, contract: dict, result: dict, out: Path, elapsed: float) -> None:
    history = add_regularization_columns(result.pop("history"), config_for(point).regularization)
    result = add_regularization_columns(pd.DataFrame([result]), config_for(point).regularization)
    meta = metadata(point, seed_index, contract)
    summary_path, history_path, receipt_path = paths(point, seed_index, out)
    atomic_csv(result.assign(**meta, elapsed_seconds_this_attempt=elapsed), summary_path)
    atomic_csv(history.assign(**meta), history_path)
    atomic_json({"state": "complete", "identity": identity(point, seed_index, contract),
                 "hash_format": "sha256_utf8_lf_v1", "summary_text_sha256": text_hash(summary_path),
                 "history_text_sha256": text_hash(history_path)}, receipt_path)


def run_job(point: Point, seed_index: int, contract: dict, out: Path = OUT) -> str:
    tag = f"{point.point_id}_s{seed_index}"
    with dev.point_lock(out / "locks" / (tag + ".lock")):
        receipt = read_receipt(point, seed_index, contract, out)
        if receipt is not None:
            return f"reuse {tag}: {receipt['state']}"
        if source_hashes() != contract["science"]["source_hashes"] or dev.runtime_manifest() != contract["runtime"]:
            raise RuntimeError("Worker source/runtime differs from the confirmation contract")
        if dev.build_contract(dev.DATA)["data_sha256"] != contract["data_sha256"]:
            raise RuntimeError("Worker data differs from the confirmation contract")
        seed = SEEDS[seed_index]
        federation = make_cifar10_federation(root=dev.DATA, regime="strong", seed=seed, train_eval_size=5000, n_clients=10)
        store = BaselineCheckpoint(out / "checkpoints" / (tag + ".npz"), identity(point, seed_index, contract))
        started = time.monotonic()

        def progress(row):
            rnd = int(row["round"])
            if rnd == 1 or rnd % 180 == 0 or rnd == ROUNDS:
                print(f"{tag}: {rnd}/{ROUNDS}, train={row['train_objective']:.6f}", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = run_final_baseline(
                    federation=federation, architecture="cifar_cnn", method="stc",
                    config=config_for(point), seed=80000 + seed, record_history=True,
                    checkpoint=store, checkpoint_stride=60, progress_callback=progress,
                )
            for field in ("final_train_objective", "final_test_ce", "final_test_accuracy"):
                if not np.isfinite(result[field]):
                    raise FloatingPointError(f"nonfinite confirmation endpoint: {field}")
        except FloatingPointError as exc:
            atomic_json({"state": "numerical_failure", "identity": identity(point, seed_index, contract),
                         "hash_format": "sha256_utf8_lf_v1", "error": str(exc)}, paths(point, seed_index, out)[2])
            return f"numerical failure {tag}: {exc}"
        score = result["final_test_accuracy"]
        save_result(point, seed_index, contract, result, out, time.monotonic() - started)
        return f"done {tag}: test_accuracy={score:.4f}"


def confirm(workers: int, point_id: str | None = None, seed_index: int | None = None) -> None:
    contract = prepare_execution(build_science())
    ensure_execution(contract)
    jobs = [(point, i) for point in POINTS for i in range(len(SEEDS))
            if (point_id is None or point.point_id == point_id) and (seed_index is None or i == seed_index)]
    pending = [(point, i) for point, i in jobs if read_receipt(point, i, contract) is None]
    print(f"{len(jobs)-len(pending)}/{len(jobs)} attempted; {len(pending)} pending", flush=True)
    if workers == 1:
        for point, i in pending:
            print(run_job(point, i, contract), flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(run_job, point, i, contract) for point, i in pending]
            for future in as_completed(futures):
                print(future.result(), flush=True)


def verified_runs(contract: dict, out: Path = OUT) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows, histories, states = [], {}, []
    for point in POINTS:
        for i, seed in enumerate(SEEDS):
            receipt = read_receipt(point, i, contract, out)
            if receipt is None:
                raise RuntimeError(f"Missing confirmation: {point.point_id}, seed {seed}")
            states.append({"point_id": point.point_id, "partition_seed": seed,
                           "state": receipt["state"], "error": receipt.get("error", "")})
            if receipt["state"] != "complete":
                continue
            summary_path, history_path, _ = paths(point, i, out)
            summary = pd.read_csv(summary_path, float_precision="round_trip")
            history = pd.read_csv(history_path, float_precision="round_trip")
            if len(summary) != 1 or tuple(history["round"]) != EXPECTED_ROUNDS:
                raise RuntimeError("Confirmation summary/history horizon differs")
            row = summary.iloc[0].to_dict()
            expected = {**metadata(point, i, contract), "method": "stc", "architecture": "cifar_cnn",
                        "rounds": ROUNDS, "dimension": LAYOUT.dimension,
                        "local_steps": config_for(point).local_steps, "local_lr": config_for(point).local_lr,
                        "stc_upload_fraction": config_for(point).stc_upload_fraction,
                        "stc_download_fraction": config_for(point).stc_download_fraction,
                        "server_gain": 2.0, "downlink_protocol": "stc_compressed_server_update"}
            if any(row.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Confirmation metadata differs from its declared point/seed")
            for key, value in metadata(point, i, contract).items():
                if key not in history or not history[key].eq(value).all():
                    raise RuntimeError("Confirmation history metadata differs")
            numeric = history[["train_objective", "train_ce", "test_ce", "test_accuracy", "unicast_total_bits"]]
            if not np.isfinite(numeric).all().all() or not history.test_accuracy.between(0, 1).all():
                raise RuntimeError("Invalid confirmation history metrics")
            for final, field in (("final_train_objective", "train_objective"), ("final_train_ce", "train_ce"),
                                 ("final_test_ce", "test_ce"), ("final_test_accuracy", "test_accuracy"),
                                 ("unicast_hybrid_total_bits", "unicast_total_bits")):
                if not np.isclose(row[final], history.iloc[-1][field], rtol=1e-12, atol=1e-12):
                    raise RuntimeError("Confirmation endpoint/history differs")
            if int(row["uplink_packetized_bits"] + row["unicast_hybrid_downlink_bits"]) != int(row["unicast_hybrid_total_bits"]):
                raise RuntimeError("Confirmation endpoint traffic does not add up")
            if row["stc_downlink_messages"] != ROUNDS:
                raise RuntimeError("Missing native STC downlink messages")
            validate_native_traffic(row, history)
            derived = add_regularization_columns(summary, config_for(point).regularization).iloc[0]
            for field in ("final_train_l2_penalty", "final_parameter_l2_norm"):
                if not np.isclose(row[field], derived[field], rtol=1e-12, atol=1e-12):
                    raise RuntimeError("Regularization diagnostics do not reproduce")
            rows.append(row)
            histories[(point.point_id, seed)] = history
    return pd.DataFrame(rows), histories, pd.DataFrame(states)


def analyze(out: Path = OUT) -> None:
    contract = load_execution(out)
    confirmation, histories, states = verified_runs(contract, out)
    atomic_csv(states, out / "completion_audit.csv")
    if not states.state.eq("complete").all():
        raise RuntimeError("Numerical failures are recorded in completion_audit.csv; no fixed-horizon paired inference is generated by dropping failed seeds")
    references = []
    for name in REFERENCE_IDS:
        for i, seed in enumerate(SEEDS):
            row, history, _ = load_reference(name, i)
            references.append(row)
            histories[(name, seed)] = history
    runs = pd.concat([confirmation, pd.DataFrame(references)], ignore_index=True)
    atomic_csv(confirmation, out / "confirmation_runs.csv")
    atomic_csv(runs, out / "comparison_runs.csv")
    fields = ["final_test_accuracy", "final_test_ce", "final_train_objective", "final_train_ce",
              "final_train_l2_penalty", "final_parameter_l2_norm", "uplink_packetized_bits",
              "unicast_hybrid_downlink_bits", "unicast_hybrid_total_bits", "broadcast_total_bits"]
    summary = runs.groupby("comparison_id")[fields].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    atomic_csv(summary.reset_index(), out / "endpoint_summary.csv")
    points = tuple(point.point_id for point in POINTS)
    per_seed, paired = paired_endpoints(runs, points, REFERENCE_IDS)
    atomic_csv(per_seed, out / "paired_per_seed.csv")
    atomic_csv(paired, out / "paired_summary.csv")
    crossings = pd.DataFrame([row for (name, seed), history in histories.items()
                             for row in target_crossings(history, targets=TARGETS, horizon=ROUNDS,
                                                         comparison_id=name, partition_seed=seed)])
    atomic_csv(crossings, out / "traffic_to_target.csv")
    atomic_csv(paired_targets(crossings, points, REFERENCE_IDS), out / "paired_target_summary.csv")
    shown = paired[(paired.reference == "event") & paired.metric.isin(["test_accuracy_difference_pp", "test_ce_difference", "unicast_traffic_ratio"])]
    print(shown.to_string(index=False))
    print("Intervals are descriptive and unadjusted; an interval spanning zero does not establish equivalence.")


def worker_signature() -> dict:
    return {"runtime": dev.runtime_manifest(), "sources": source_hashes(), "spec": protocol_spec()}


def smoke() -> None:
    from stc_stage1 import smoke_federation
    for point in POINTS:
        config = replace(config_for(point), rounds=4, eval_stride=1, local_steps=1, batch_size=2)
        kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="stc",
                      config=config, seed=83500, record_history=True)
        continuous = run_final_baseline(**kwargs)
        with tempfile.TemporaryDirectory(prefix="stc-confirmation-smoke-") as directory:
            store = BaselineCheckpoint(Path(directory) / "state.npz", {"scope": "synthetic-confirmation-smoke"})

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
                raise RuntimeError(f"Confirmation resume differs: {point.point_id}, {key}")
        validate_native_traffic(continuous, continuous["history"], n_clients=kwargs["federation"].n_clients)
        metrics = add_regularization_columns(continuous["history"], config.regularization)
        if not metrics.train_l2_penalty.ge(0).all():
            raise RuntimeError("Invalid smoke regularization diagnostics")
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker_signature) for _ in range(2)]
        if any(future.result() != worker_signature() for future in futures):
            raise RuntimeError("Confirmation worker signature differs")
    print("Both STC points pass synthetic checkpoint/resume and spawn checks; no research data loaded or outputs written.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "preflight", "status", "verify", "analyze"):
        sub.add_parser(name)
    run = sub.add_parser("confirm")
    run.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    point = sub.add_parser("point")
    point.add_argument("--point-id", choices=POINT_BY_ID, required=True)
    point.add_argument("--seed-index", type=int, choices=range(10), required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "preflight":
        science = build_science()
        print(f"Verified 25 development runs and {len(science['reference_provenance'])} archived references.")
        print(f"Fixed confirmation: {len(POINTS)*len(SEEDS)} STC runs; regularization={config_for(POINTS[0]).regularization}.")
        print("Training runtime matches development:", dev.runtime_manifest() == dev.load_contract()["runtime"])
        print("Legacy runtime/raw-batch provenance is unavailable per run; reference reuse checks recorded metadata and declared drivers.")
    elif args.command == "confirm":
        confirm(args.workers)
    elif args.command == "point":
        confirm(1, args.point_id, args.seed_index)
    elif args.command == "status":
        if not (OUT / "protocol.json").exists():
            print("Confirmation not started: 20 pending runs")
            return
        contract = load_execution()
        counts = {"complete": 0, "numerical_failure": 0, "pending": 0}
        for point in POINTS:
            for i in range(len(SEEDS)):
                receipt = read_receipt(point, i, contract)
                state = receipt["state"] if receipt else "pending"
                counts[state] += 1
                print(f"{point.point_id} seed={SEEDS[i]}: {state}")
        print(counts)
    elif args.command == "verify":
        _, _, states = verified_runs(load_execution())
        print("Verified confirmation attempts:", states.state.value_counts().to_dict())
    elif args.command == "analyze":
        analyze()


if __name__ == "__main__":
    main()
