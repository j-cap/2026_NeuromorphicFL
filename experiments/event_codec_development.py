"""Bounded validation-only Event threshold/quantum calibration against fixed Ström.

python experiments/event_codec_development.py verify
python experiments/event_codec_development.py preflight
python experiments/event_codec_development.py run --workers 2
python experiments/event_codec_development.py analyze
git add experiments/results/event_codec_development_v1
git commit -m "Add validation-only Event threshold quantum development results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
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
import strom_codec_development as previous

full, data_tools = previous.full, previous.data_tools
atomic_csv, atomic_json, digest, text_hash = previous.atomic_csv, previous.atomic_json, previous.digest, previous.text_hash
immutable_json = previous.immutable_json
EVAL_FIELDS = previous.EVAL_FIELDS
OUT = REPO / "experiments/results/event_codec_development_v1"
SCREEN, FINAL = 600, 1800
THRESHOLDS, QUANTA = (.0125, .025, .05, .1, .2), (.0025, .005, .01)
STROM_ID = "strom_reference"
STROM_POINT = previous.strom_point(.05, 2.)


@dataclass(frozen=True)
class Point:
    point_id: str
    threshold: float
    jump0: float


def event_point(threshold, jump0):
    label = lambda value: f"{value:g}".replace(".", "p")
    return Point(f"event_t{label(threshold)}_q{label(jump0)}", float(threshold), float(jump0))


POINTS = tuple(event_point(threshold, jump0) for threshold in THRESHOLDS for jump0 in QUANTA)
POINT_BY_ID = {p.point_id: p for p in POINTS}
CURRENT = event_point(.025, .005)


def config_for(point):
    if point.threshold <= 0 or point.jump0 <= 0 or not np.isfinite([point.threshold, point.jump0]).all():
        raise ValueError("Positive finite Event threshold and quantum required")
    return replace(previous.config_for(previous.EVENT), rounds=FINAL, threshold=point.threshold, jump0=point.jump0)


def config_for_id(point_id):
    return previous.config_for(STROM_POINT) if point_id == STROM_ID else config_for(POINT_BY_ID[point_id])


def method_for(point_id):
    return "strom" if point_id == STROM_ID else "event"


def protocol_spec():
    return {"campaign": "event_codec_development_v1", "architecture": "cifar_cnn", "dimension": 20570,
            "n_clients": 10, "screen_rounds": SCREEN, "final_rounds": FINAL,
            "split": data_tools.split_spec(), "training_seed": data_tools.TRAIN_SEED,
            "points": [asdict(p) for p in POINTS], "configs": {p.point_id: asdict(config_for(p)) for p in POINTS},
            "current_Event_id": CURRENT.point_id, "fixed_Strom_config": asdict(config_for_id(STROM_ID)),
            "new_screen_training_runs": 14, "reused_Event_candidates": 1, "screen_Event_candidates": 15,
            "reused_controls_at_both_horizons": [CURRENT.point_id, STROM_ID],
            "max_promoted_Event_candidates": 2, "automatic_boundary_extension": False,
            "selection_metric": "final_validation_ce", "selection_ties": ["unicast_total_bits", "point_id"],
            "quality_rule": "minimum_Event_validation_CE_at_fixed_horizon",
            "budget_rule": "minimum_Event_validation_CE_at_or_below_same_horizon_frozen_Strom_unicast_cost",
            "infeasible_budget": "explicit_null_no_nearest_candidate_promoted_as_fallback",
            "promotion": "distinct_screen_quality_and_feasible_budget_choices_only",
            "final_scope": "among_screen_promoted_Event_candidates_not_full_1800_grid_optimum",
            "current_Event_at_1800": "control_only_unless_promoted_by_the_screening_rule",
            "checkpoint_config": "1800_round_identity_from_start_exact_600_to_1800_continuation",
            "accounting": "frozen_actual_frames_both_codecs_conservative_unicast_primary",
            "fixed_parameters": ["rho", "event_memory", "event_reset", "jump_exponent", "jump_scale",
                                 "server_gain", "local_lr", "local_steps", "batch_size", "regularization"],
            "validation_only": True, "official_test_images_loaded": False, "test_based_selection": False,
            "additional_development_after_inspected_confirmation": True, "confirmation_included": False,
            "hash_format": "sha256_utf8_lf_v1"}


def reference_key(point_id, horizon):
    return f"{point_id}_r{horizon}"


def frozen_references():
    """Validate the training-holdout study only; never open confirmation test curves."""
    science = previous.load_contract()
    split = previous.load_split(science)
    points, screening = previous.verify_screening(science, split)
    by_id = {p.point_id: p for p in points}
    promoted = (previous.EVENT, *[by_id[p] for p in screening["promoted_strom_ids"]])
    selected = previous.select_points(previous.ledger(promoted, previous.FINAL, science, split))
    expected = {"contract_sha256": digest(science), "split_sha256": digest(split), "screen_horizon": previous.SCREEN,
                "final_horizon": previous.FINAL, "selection_metric": "final_validation_ce", "selection": selected,
                "scope": "two_stage_development_selection_among_promoted_finalists",
                "screening_selection_sha256": text_hash(previous.OUT / "screening_selection.json"),
                "final_receipts": previous.receipt_fingerprints(promoted, previous.FINAL, previous.OUT),
                "selected_configs": {role: asdict(previous.config_for(by_id[choice["point_id"]])) if choice else None
                                     for role, choice in (("quality", selected["quality"]), ("within_event_budget", selected["within_event_budget"]))},
                "official_test_evaluations": 0, "confirmation_included": False}
    saved = json.loads((previous.OUT / "frozen_selection.json").read_text(encoding="utf-8"))
    if (saved != expected or selected["quality"]["point_id"] != STROM_POINT.point_id
            or saved["selected_configs"]["quality"] != asdict(config_for_id(STROM_ID))
            or asdict(config_for(CURRENT)) != asdict(previous.config_for(previous.EVENT))):
        raise RuntimeError("Frozen Ström choice or current Event reference changed")
    references = {}
    for point_id, source in ((CURRENT.point_id, previous.EVENT), (STROM_ID, STROM_POINT)):
        for horizon in (SCREEN, FINAL):
            previous.verified_tables(source, horizon, science, split)
            references[reference_key(point_id, horizon)] = {
                "source_point": asdict(source), "source_horizon": horizon,
                "files": {kind: {"path": file.relative_to(REPO).as_posix(), "sha256": text_hash(file)}
                          for kind, file in previous.paths(source, horizon).items()}}
    return science, split, references


def build_science():
    old, split, references = frozen_references()
    return {"protocol": protocol_spec(), "source_hashes": {**old["source_hashes"],
            "experiments/event_codec_development.py": text_hash(Path(__file__))},
            "runtime": old["runtime"], "training_batch_sha256": old["training_batch_sha256"],
            "split": split, "reference_contract_sha256": digest(old), "references": references,
            "frozen_Strom_selection_sha256": text_hash(previous.OUT / "frozen_selection.json")}


def build_contract():
    science = build_science()
    if full.development.runtime_manifest() != science["runtime"]:
        raise RuntimeError("Use the recorded Windows environment before development training")
    root = full.download_cifar10(full.DATA)
    if {name: full.development.file_hash(root / name) for name in science["training_batch_sha256"]} != science["training_batch_sha256"]:
        raise RuntimeError("CIFAR training batch fingerprints differ")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Event development protocol, sources or reference evidence changed")
    return science


def load_split(science, out=OUT):
    indices = pd.read_csv(out / "split_indices.csv")
    split = data_tools.split_manifest(indices)
    saved = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    if split != science["split"] or saved != {"contract_sha256": digest(science), "split": split}:
        raise RuntimeError("Event development training/validation split changed")
    return split


def checkpoint_identity(point_id, science, split):
    return {"point_id": point_id, "config_sha256": digest(asdict(config_for_id(point_id))),
            "contract_sha256": digest(science), "split_sha256": digest(split), "training_seed": data_tools.TRAIN_SEED}


def metadata(point_id, horizon, science, split):
    return {**checkpoint_identity(point_id, science, split), "stage_horizon": horizon,
            "evaluation_scope": "training_holdout_validation_only", "method": method_for(point_id),
            "evidence_origin": "reused_validation_reference" if point_id in (CURRENT.point_id, STROM_ID) else "new_Event_validation_run"}


def paths(point_id, horizon, out=OUT):
    return {kind: out / f"{point_id}_r{horizon}_{kind}.{'json' if kind == 'receipt' else 'csv'}"
            for kind in ("summary", "history", "rounds", "evaluation", "receipt")}


def read_receipt(point_id, horizon, science, split, out=OUT):
    files = paths(point_id, horizon, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != metadata(point_id, horizon, science, split):
        raise RuntimeError("Event development receipt identity differs")
    if point_id in (CURRENT.point_id, STROM_ID):
        expected = science["references"][reference_key(point_id, horizon)]
        if receipt.get("state") != "reused" or receipt.get("source") != expected:
            raise RuntimeError("Validation reference reuse identity differs")
        for info in expected["files"].values():
            if text_hash(REPO / info["path"]) != info["sha256"]:
                raise RuntimeError("Reused validation reference evidence changed")
    elif receipt.get("state") == "complete":
        for kind in ("summary", "history", "rounds", "evaluation"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing/changed Event development evidence: {point_id}/{horizon}/{kind}")
    elif receipt.get("state") != "numerical_failure":
        raise RuntimeError("Unknown Event development completion state")
    return receipt


def register_reuse(science, split, out=OUT):
    for point_id in (CURRENT.point_id, STROM_ID):
        with full.development.point_lock(out / "locks" / (point_id + ".lock")):
            for horizon in (SCREEN, FINAL):
                if read_receipt(point_id, horizon, science, split, out) is None:
                    atomic_json({"identity": metadata(point_id, horizon, science, split), "state": "reused",
                                 "source": science["references"][reference_key(point_id, horizon)]}, paths(point_id, horizon, out)["receipt"])


def prepare(out=OUT):
    science = build_contract()
    immutable_json(science, out / "protocol.json")
    indices = pd.read_csv(previous.OUT / "split_indices.csv")
    if data_tools.split_manifest(indices) != science["split"]:
        raise RuntimeError("Original development split differs")
    if (out / "split_indices.csv").exists():
        if text_hash(out / "split_indices.csv") != science["split"]["index_csv_sha256"]:
            raise RuntimeError("Existing split indices differ")
    else:
        atomic_csv(indices, out / "split_indices.csv")
    immutable_json({"contract_sha256": digest(science), "split": science["split"]}, out / "split_manifest.json")
    split = load_split(science, out)
    register_reuse(science, split, out)
    return science, split


def save_result(point_id, horizon, science, split, result, out=OUT):
    if point_id in (CURRENT.point_id, STROM_ID):
        raise ValueError("Current Event and calibrated Ström are reused, never retrained")
    tables = data_tools.trajectory_tables(result, method="event", horizon=horizon,
                                          n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    files = paths(point_id, horizon, out)
    for kind, frame in tables.items():
        atomic_csv(frame.assign(**metadata(point_id, horizon, science, split)), files[kind])
    atomic_json({"identity": metadata(point_id, horizon, science, split), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def verified_tables(point_id, horizon, science, split, out=OUT):
    receipt = read_receipt(point_id, horizon, science, split, out)
    if receipt is None or receipt["state"] not in ("complete", "reused"):
        raise RuntimeError("Incomplete Event development point")
    if receipt["state"] == "reused":
        old = previous.load_contract()
        if digest(old) != science["reference_contract_sha256"] or previous.load_split(old) != split:
            raise RuntimeError("Reference contract/split differs")
        source = previous.Point(**receipt["source"]["source_point"])
        tables = {kind: frame.assign(**metadata(point_id, horizon, science, split))
                  for kind, frame in previous.verified_tables(source, horizon, old, split).items()}
    else:
        tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
                  for kind, file in paths(point_id, horizon, out).items() if kind != "receipt"}
    for frame in tables.values():
        if any("test_" in field for field in frame):
            raise RuntimeError("Test metrics are prohibited in Event development evidence")
        for field, value in metadata(point_id, horizon, science, split).items():
            if not frame[field].eq(value).all():
                raise RuntimeError("Event development result metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    samples = [r for r in full.sampled_rounds(config_for_id(point_id)) if r <= horizon]
    if rounds["round"].tolist() != list(range(1, horizon + 1)) or evaluation["round"].tolist() != samples:
        raise RuntimeError("Event development horizon/sample coverage differs")
    if (not rounds.method.eq(method_for(point_id)).all() or not np.isfinite(evaluation[list(EVAL_FIELDS[1:-1])]).all().all()
            or not evaluation[list(data_tools.VALIDATION_FIELDS[1:])].apply(lambda c: c.between(0, 1)).all().all()
            or not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"])):
        raise RuntimeError("Invalid validation method/metrics/model fingerprints")
    rebuilt = data_tools.trajectory_tables({"rounds": rounds, "evaluation": evaluation[list(EVAL_FIELDS)]},
                                           method=method_for(point_id), horizon=horizon,
                                           n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind in ("summary", "history"):
        for field in rebuilt[kind]:
            actual, expected = tables[kind][field], rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=1e-12, atol=1e-12)):
                raise RuntimeError("Reconstructed Event development traffic/metrics differ")
    return tables


def verify_screen_prefix(point_id, result, science, split, out=OUT):
    before = verified_tables(point_id, SCREEN, science, split, out)
    after = data_tools.trajectory_tables(result, method="event", horizon=SCREEN,
                                         n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind in ("rounds", "evaluation"):
        fields = [c for c in after[kind] if c != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(before[kind][fields].reset_index(drop=True), after[kind][fields].reset_index(drop=True),
                                      check_dtype=False, check_exact=True)


def run_point(point_id, horizon, science, split, out=OUT):
    if horizon not in (SCREEN, FINAL):
        raise ValueError("Unsupported development horizon")
    with full.development.point_lock(out / "locks" / (point_id + ".lock")):
        receipt = read_receipt(point_id, horizon, science, split, out)
        if receipt is not None:
            return f"reuse {point_id}/{horizon}: {receipt['state']}"
        if point_id in (CURRENT.point_id, STROM_ID):
            raise ValueError("Reference reuse must be registered before running")
        if build_contract() != science:
            raise RuntimeError("Worker source/runtime/data changed")
        data = data_tools.load_data(full.DATA)
        if data.manifest != split:
            raise RuntimeError("Worker training/validation split differs")

        def progress(row, validation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point_id}: {row['round']}/{horizon}, validation_ce={validation['validation_ce']:.6f}", flush=True)

        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = data_tools.evaluated_run(data=data, method="event", config=config_for_id(point_id), seed=data_tools.TRAIN_SEED,
                                                  checkpoint_path=out / "checkpoints" / (point_id + ".npz"),
                                                  provenance=checkpoint_identity(point_id, science, split), horizon=horizon, progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": metadata(point_id, horizon, science, split), "state": "numerical_failure", "error": str(exc)},
                        paths(point_id, horizon, out)["receipt"])
            return f"numerical failure {point_id}/{horizon}: {exc}"
        if horizon == FINAL:
            verify_screen_prefix(point_id, result, science, split, out)
        save_result(point_id, horizon, science, split, result, out)
        return f"done {point_id}/{horizon}"


def execute(point_ids, horizon, workers, science, split, out=OUT):
    jobs = [key for key in point_ids if read_receipt(key, horizon, science, split, out) is None]
    print(f"{len(jobs)} new Event development runs pending to round {horizon}", flush=True)
    if workers == 1:
        for key in jobs:
            print(run_point(key, horizon, science, split, out), flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for job in as_completed([pool.submit(run_point, key, horizon, science, split, out) for key in jobs]):
                print(job.result(), flush=True)


def ledger(point_ids, horizon, science, split, out=OUT):
    rows = []
    for key in point_ids:
        receipt = read_receipt(key, horizon, science, split, out)
        if receipt is None:
            raise RuntimeError("Development grid incomplete; no pending point is dropped")
        config = config_for_id(key)
        row = {"point_id": key, "method": method_for(key), "threshold": config.strom_threshold if key == STROM_ID else config.threshold,
               "jump0": np.nan if key == STROM_ID else config.jump0, "horizon": horizon,
               "state": receipt["state"], "error": receipt.get("error", "")}
        if receipt["state"] in ("complete", "reused"):
            endpoint = verified_tables(key, horizon, science, split, out)["summary"].set_index("variant").loc["both_codecs"]
            for field in ("final_train_objective", "final_train_ce", *["final_" + name for name in data_tools.VALIDATION_FIELDS],
                          "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"):
                row[field] = endpoint[field]
        rows.append(row)
    return pd.DataFrame(rows)


def select_points(frame):
    valid = frame.state.isin(("complete", "reused"))
    reference = frame[(frame.point_id == STROM_ID) & valid]
    if len(reference) != 1:
        raise RuntimeError("A complete frozen Ström budget reference is required")
    budget = int(reference.iloc[0].unicast_total_bits)
    candidates = frame[(frame.method == "event") & valid].copy()
    if candidates.empty:
        raise RuntimeError("No finite Event candidate")
    if (not np.isfinite(candidates[["final_validation_ce", "unicast_total_bits"]]).all().all()
            or candidates.unicast_total_bits.le(0).any() or budget <= 0):
        raise ValueError("Invalid completed validation candidate")
    candidates = candidates.sort_values(["final_validation_ce", "unicast_total_bits", "point_id"])

    def choice(row):
        return {"point_id": row.point_id, "threshold": float(row.threshold), "jump0": float(row.jump0),
                "validation_ce": float(row.final_validation_ce), "unicast_bits": int(row.unicast_total_bits)}

    feasible = candidates[candidates.unicast_total_bits <= budget]
    return {"strom_unicast_budget_bits": budget, "quality": choice(candidates.iloc[0]),
            "within_strom_budget": choice(feasible.iloc[0]) if len(feasible) else None}


def shortlist(selection):
    return sorted({choice["point_id"] for role in ("quality", "within_strom_budget") if (choice := selection[role]) is not None})


def receipt_fingerprints(point_ids, horizon, out):
    return {key: text_hash(paths(key, horizon, out)["receipt"]) for key in point_ids}


def screening_decision(science, split, out=OUT):
    ids = (STROM_ID, *POINT_BY_ID)
    rows = ledger(ids, SCREEN, science, split, out)
    selected = select_points(rows)
    decision = {"contract_sha256": digest(science), "split_sha256": digest(split), "horizon": SCREEN,
                "selection": selected, "promoted_Event_ids": shortlist(selected),
                "all_screen_receipts": receipt_fingerprints(ids, SCREEN, out), "automatic_extension": False}
    return rows, decision


def verify_screening(science, split, out=OUT):
    rows, decision = screening_decision(science, split, out)
    if json.loads((out / "screening_selection.json").read_text(encoding="utf-8")) != decision:
        raise RuntimeError("Frozen Event screening selection changed")
    return rows, decision


def screen(workers, science, split, out=OUT):
    execute(tuple(POINT_BY_ID), SCREEN, workers, science, split, out)
    rows, decision = screening_decision(science, split, out)
    atomic_csv(rows, out / "screening.csv")
    immutable_json(decision, out / "screening_selection.json")
    return decision


def final_decision(science, split, out=OUT):
    _, screened = verify_screening(science, split, out)
    ids = (STROM_ID, *screened["promoted_Event_ids"])
    rows = ledger(ids, FINAL, science, split, out)
    selected = select_points(rows)
    frozen = {"contract_sha256": digest(science), "split_sha256": digest(split), "screen_horizon": SCREEN,
              "final_horizon": FINAL, "selection_metric": "final_validation_ce", "selection": selected,
              "scope": "additional_development_among_screen_promoted_Event_candidates",
              "screening_selection_sha256": text_hash(out / "screening_selection.json"),
              "final_receipts": receipt_fingerprints(ids, FINAL, out),
              "selected_configs": {role: asdict(config_for_id(choice["point_id"])) if choice else None
                                   for role, choice in (("quality", selected["quality"]), ("within_strom_budget", selected["within_strom_budget"]))},
              "official_test_evaluations": 0, "confirmation_included": False,
              "fresh_paired_confirmation_required": True}
    return rows, frozen


def finalize(workers, science, split, out=OUT):
    _, decision = verify_screening(science, split, out)
    execute(decision["promoted_Event_ids"], FINAL, workers, science, split, out)
    rows, frozen = final_decision(science, split, out)
    atomic_csv(rows, out / "finalists.csv")
    immutable_json(frozen, out / "frozen_selection.json")
    print(json.dumps(frozen["selection"], indent=2), flush=True)
    return frozen


def run(workers, phase="all", out=OUT):
    if not 1 <= workers <= 8 or phase not in ("all", "screen", "finalize"):
        raise ValueError("Invalid worker count or development phase")
    with full.development.point_lock(out / "locks" / "controller.lock"):
        science, split = prepare(out)
        if phase in ("all", "screen"):
            screen(workers, science, split, out)
        if phase in ("all", "finalize"):
            finalize(workers, science, split, out)


def plot(screened, finals, controls, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for frame, axis, horizon in ((screened, axes[0], SCREEN), (finals, axes[1], FINAL)):
        if frame is not None:
            events = frame[(frame.method == "event") & frame.state.isin(("complete", "reused"))]
            axis.scatter(events.unicast_total_bits / 1e6, events.final_validation_ce, color="#0072B2", s=38, label="Event candidates")
        else:
            axis.text(.5, .5, "Finalists pending", transform=axis.transAxes, ha="center")
        for key, label, marker, color in ((STROM_ID, "Fixed calibrated Ström", "*", "#009E73"),
                                           (CURRENT.point_id, "Current Event control", "X", "#D55E00")):
            point = controls[(controls.point_id == key) & (controls.horizon == horizon)]
            axis.scatter(point.unicast_total_bits / 1e6, point.final_validation_ce, marker=marker, color=color, s=85, label=label)
        axis.set(title=f"{horizon}-round validation {'screen' if horizon == SCREEN else 'finalists'}",
                 xlabel="Total unicast traffic (Mbit)", ylabel="Validation cross-entropy")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Event threshold/quantum development; training holdout validation only", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "development_frontier.png", dpi=180)
    fig.savefig(out / "development_frontier.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def analyze(out=OUT):
    science = load_contract(out)
    split = load_split(science, out)
    screened, decision = verify_screening(science, split, out)
    atomic_csv(screened, out / "screening.csv")
    controls = pd.concat([ledger((STROM_ID, CURRENT.point_id), horizon, science, split, out) for horizon in (SCREEN, FINAL)], ignore_index=True)
    atomic_csv(controls, out / "controls.csv")
    finals = None
    if (out / "frozen_selection.json").exists():
        finals, expected = final_decision(science, split, out)
        if json.loads((out / "frozen_selection.json").read_text(encoding="utf-8")) != expected:
            raise RuntimeError("Frozen final Event selection changed")
        atomic_csv(finals, out / "finalists.csv")
    plot(screened, finals, controls, out)
    atomic_json({"validation_only": True, "official_test_evaluations": 0, "split_disjointness_verified": True,
                 "screen_Event_points": len(POINTS), "new_screen_training_runs": len(POINTS) - 1,
                 "reused_reference_stages": 4, "promoted_Event_points": len(decision["promoted_Event_ids"]),
                 "final_selection_available": finals is not None, "selection_metric": "final_validation_ce",
                 "test_curves_used_for_selection": False, "confirmation_included": False}, out / "analysis_audit.json")
    print(screened[["point_id", "state", "final_validation_ce", "unicast_total_bits"]].to_string(index=False))


def smoke_worker(index):
    from stc_stage1 import smoke_federation
    federation = smoke_federation()
    data = data_tools.DevelopmentData(replace(federation, X_test=np.empty((0, 3, 32, 32), dtype=np.uint8), y_test=np.empty(0, dtype=int)),
                                      federation.X_test, federation.y_test, pd.DataFrame(), {"synthetic": True})
    point = (POINTS[0], POINTS[-1])[index]
    config = replace(config_for(point), rounds=4, eval_stride=1, local_steps=1, batch_size=2)
    kwargs = dict(data=data, method="event", config=config, seed=data_tools.TRAIN_SEED,
                  provenance={"synthetic_point": asdict(point)})
    with tempfile.TemporaryDirectory() as folder:
        baseline_path = Path(folder) / "full.npz"
        baseline = data_tools.evaluated_run(**kwargs, checkpoint_path=baseline_path, horizon=4)
        path = Path(folder) / "resumed.npz"
        data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=2)
        resumed = data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=4)
        with np.load(baseline_path, allow_pickle=False) as before, np.load(path, allow_pickle=False) as after:
            for field in ("model", "client_model", "event_state"):
                np.testing.assert_array_equal(before[field].view(np.uint32), after[field].view(np.uint32))
        pd.testing.assert_frame_equal(baseline["evaluation"], resumed["evaluation"], check_exact=True)
        fields = [c for c in baseline["rounds"] if c != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(baseline["rounds"][fields], resumed["rounds"][fields], check_exact=True)
        if baseline["rng_state"] != resumed["rng_state"]:
            raise AssertionError("Resumed RNG differs")
        return {"point_id": point.point_id, "model_sha256": resumed["evaluation"].iloc[-1].model_sha256,
                "summary_sha256": digest(resumed["summary"].to_dict("records"))}


def preflight():
    science = build_science()
    print("Frozen Event/Ström hosts and validation references verified; official test curves not loaded")
    print("45000 training / 5000 validation; same split, partition 4500 and SGD 84500")
    print("15 Event candidates: 14 new screens plus reused current Event; no automatic grid extension")
    print("600-round screen; at most two Event finalists continue exactly to 1800")
    print("Calibrated Ström and current Event controls reused at both horizons; only threshold/quantum vary")
    print("Runtime matches recorded Windows environment:", full.development.runtime_manifest() == science["runtime"])


def status(out=OUT):
    if not (out / "protocol.json").exists():
        print("Event threshold/quantum development not started")
        return
    science = load_contract(out)
    split = load_split(science, out)
    for key in (STROM_ID, *POINT_BY_ID):
        labels = []
        for horizon in (SCREEN, FINAL):
            receipt = read_receipt(key, horizon, science, split, out)
            labels.append(f"r{horizon}: {receipt['state'] if receipt else 'pending/not promoted'}")
        checkpoint = out / "checkpoints" / (key + ".npz")
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                meta = json.loads(str(saved["metadata"].item()))
            if meta.get("provenance") != checkpoint_identity(key, science, split):
                raise RuntimeError("Checkpoint provenance differs")
            labels.append(f"checkpoint {meta['progress']['completed_round']}")
        print(key, "; ".join(labels))
    print("Frozen final selection:", (out / "frozen_selection.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "screen", "finalize", "analyze", "status"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_event_codec_development.py", "-v"],
                       cwd=REPO, env=env, check=True)
    if args.command in ("verify", "smoke"):
        expected = [smoke_worker(i) for i in (0, 1)]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            if list(pool.map(smoke_worker, (0, 1))) != expected:
                raise AssertionError("Spawned Event development workers differ")
        print("Synthetic extremes: exact continuation and spawned workers; no CIFAR research training.")
    elif args.command == "preflight":
        preflight()
    elif args.command in ("run", "screen", "finalize"):
        run(args.workers, "all" if args.command == "run" else args.command)
    elif args.command == "analyze":
        analyze()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
