"""Bounded validation-only Ström calibration; local research training is explicit.

python experiments/strom_codec_development.py verify
python experiments/strom_codec_development.py preflight
python experiments/strom_codec_development.py run --workers 2
python experiments/strom_codec_development.py analyze
git add experiments/results/strom_codec_development_v1
git commit -m "Add validation-only Strom development results"
git push origin main
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import math
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

import event_codec_full_horizon as full
import strom_development_data as data_tools
import numpy as np
import pandas as pd

OUT = REPO / "experiments/results/strom_codec_development_v1"
SCREEN, FINAL = 600, 1800
THRESHOLDS, GAINS = (.0025, .01, .025, .05, .1), (1., 2., 4.)
atomic_csv, atomic_json, digest, text_hash = full.atomic_csv, full.atomic_json, full.digest, full.text_hash
EVAL_FIELDS = ("round", "train_objective", "train_ce", *data_tools.VALIDATION_FIELDS, "model_sha256")


@dataclass(frozen=True)
class Point:
    point_id: str
    method: str
    threshold: float = 0.
    equivalent_gain: float = 1.


def strom_point(threshold, gain):
    label = lambda value: f"{value:g}".replace(".", "p")
    return Point(f"strom_t{label(threshold)}_g{label(gain)}", "strom", float(threshold), float(gain))


EVENT = Point("event_reference", "event")
INITIAL = tuple(strom_point(tau, gain) for tau in THRESHOLDS for gain in GAINS)


def config_for(point):
    config = replace(full.config_for("event"), rounds=FINAL, eval_stride=60)
    if point.method == "event":
        return config
    if point.method != "strom" or point.threshold <= 0 or point.equivalent_gain <= 0:
        raise ValueError("Invalid development point")
    return replace(config, strom_threshold=point.threshold, server_gain=config.local_lr * point.equivalent_gain)


def protocol_spec():
    return {"campaign": "strom_codec_development_v1", "architecture": "cifar_cnn", "dimension": 20570,
            "n_clients": 10, "screen_rounds": SCREEN, "final_rounds": FINAL, "split": data_tools.split_spec(),
            "training_seed": data_tools.TRAIN_SEED, "initial_points": [asdict(p) for p in INITIAL],
            "initial_configs": {p.point_id: asdict(config_for(p)) for p in (EVENT, *INITIAL)},
            "selection_metric": "final_validation_ce", "selection_ties": ["unicast_total_bits", "point_id"],
            "quality_rule": "minimum_validation_CE_at_fixed_horizon",
            "budget_rule": "minimum_validation_CE_at_or_below_same_horizon_Event_unicast_total",
            "infeasible_budget": "explicit_null_nearest_traffic_is_diagnostic_only",
            "boundary_rule": "one_pass_adjacent_factor2_extension_of_each_initial_quality_budget_or_nearest_winner_include_corner",
            "max_extension_points": 6, "max_screen_strom_points": 21, "max_final_strom_points": 2,
            "promotion": "screen_quality_and_budget_or_nearest_after_extension_deduplicated",
            "final_selection_scope": "among_promoted_finalists_not_full_1800_grid_optimum",
            "checkpoint_config": "1800_round_identity_from_start_exact_600_to_1800_continuation",
            "accounting": "frozen_actual_frame_conservative_unicast_primary",
            "validation_only": True, "official_test_images_loaded": False, "test_based_selection": False,
            "residual_diagnostics": "reported_never_used_as_performance_filter",
            "confirmation_included": False, "hash_format": "sha256_utf8_lf_v1"}


def frozen_base():
    # Inspect protocol/source identity only. Do not load old official-test curves.
    original = json.loads((full.OUT / "protocol.json").read_text(encoding="utf-8"))
    if original["protocol"] != full.protocol_spec() or original["source_hashes"] != full.source_hashes():
        raise RuntimeError("Frozen base sources/protocol changed")
    return original


def build_science():
    original = frozen_base()
    sources = {**full.source_hashes(), **{f"experiments/{name}": text_hash(REPO / "experiments" / name)
               for name in ("strom_codec_host.py", "strom_development_data.py", "strom_codec_development.py")}}
    return {"protocol": protocol_spec(), "source_hashes": sources, "runtime": original["runtime"],
            "training_batch_sha256": {f"data_batch_{i}": original["data_sha256"][f"data_batch_{i}"] for i in range(1, 6)}}


def build_contract():
    science = build_science()
    if full.development.runtime_manifest() != science["runtime"]:
        raise RuntimeError("Use the recorded Windows environment before starting development training")
    root = full.download_cifar10(full.DATA)
    actual = {name: full.development.file_hash(root / name) for name in science["training_batch_sha256"]}
    if actual != science["training_batch_sha256"]:
        raise RuntimeError("CIFAR training batch fingerprints differ")
    return science


def load_contract(out=OUT):
    science = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if science != build_science():
        raise RuntimeError("Development protocol or sources changed")
    return science


def immutable_json(value, path):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing frozen decision differs: {path.name}")
    else:
        atomic_json(value, path)


def load_split(science, out=OUT):
    saved = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    indices = pd.read_csv(out / "split_indices.csv")
    actual = data_tools.split_manifest(indices)
    if saved != {"contract_sha256": digest(science), "split": actual}:
        raise RuntimeError("Saved training/validation split changed")
    return actual


def prepare(out=OUT):
    science = build_contract()
    immutable_json(science, out / "protocol.json")
    data = data_tools.load_data(full.DATA)
    split_file = out / "split_indices.csv"
    if split_file.exists():
        if text_hash(split_file) != data.manifest["index_csv_sha256"]:
            raise RuntimeError("Existing split indices differ")
    else:
        atomic_csv(data.indices, split_file)
    immutable_json({"contract_sha256": digest(science), "split": data.manifest}, out / "split_manifest.json")
    return science, load_split(science, out)


def checkpoint_identity(point, science, split):
    return {"point_id": point.point_id, "config_sha256": digest(asdict(config_for(point))),
            "contract_sha256": digest(science), "split_sha256": digest(split), "training_seed": data_tools.TRAIN_SEED}


def metadata(point, horizon, science, split):
    return {**checkpoint_identity(point, science, split), "stage_horizon": horizon,
            "evaluation_scope": "training_holdout_validation_only", "method": point.method}


def paths(point, horizon, out=OUT):
    return {kind: out / f"{point.point_id}_r{horizon}_{kind}.{('json' if kind == 'receipt' else 'csv')}"
            for kind in ("summary", "history", "rounds", "evaluation", "receipt")}


def read_receipt(point, horizon, science, split, out=OUT):
    files = paths(point, horizon, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text(encoding="utf-8"))
    if receipt.get("identity") != metadata(point, horizon, science, split):
        raise RuntimeError("Development result identity differs")
    if receipt["state"] == "complete":
        for kind in ("summary", "history", "rounds", "evaluation"):
            if not files[kind].exists() or text_hash(files[kind]) != receipt["sha256"][kind]:
                raise RuntimeError(f"Missing/changed development evidence: {point.point_id}/{horizon}/{kind}")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unknown development completion state")
    return receipt


def save_result(point, horizon, science, split, result, out=OUT):
    tables = data_tools.trajectory_tables(result, method=point.method, horizon=horizon,
                                          n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    files = paths(point, horizon, out)
    for kind, table in tables.items():
        atomic_csv(table.assign(**metadata(point, horizon, science, split)), files[kind])
    atomic_json({"identity": metadata(point, horizon, science, split), "state": "complete",
                 "sha256": {kind: text_hash(files[kind]) for kind in tables}}, files["receipt"])


def verified_tables(point, horizon, science, split, out=OUT):
    receipt = read_receipt(point, horizon, science, split, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError("Incomplete development point")
    tables = {kind: pd.read_csv(file, float_precision="round_trip", dtype={"model_sha256": str})
              for kind, file in paths(point, horizon, out).items() if kind != "receipt"}
    for table in tables.values():
        if any("test_" in field for field in table.columns):
            raise RuntimeError("Test metrics are prohibited in development evidence")
        for field, value in metadata(point, horizon, science, split).items():
            if not table[field].eq(value).all():
                raise RuntimeError("Development result metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    expected = [r for r in full.sampled_rounds(config_for(point)) if r <= horizon]
    if rounds["round"].tolist() != list(range(1, horizon + 1)) or evaluation["round"].tolist() != expected:
        raise RuntimeError("Development horizon/sample coverage differs")
    if (not np.isfinite(evaluation[list(EVAL_FIELDS[1:-1])]).all().all()
            or not evaluation.validation_accuracy.between(0, 1).all()
            or not np.array_equal(evaluation.model_sha256, rounds.set_index("round").loc[evaluation["round"], "model_sha256"])):
        raise RuntimeError("Invalid validation evidence/model fingerprints")
    result = {"rounds": rounds, "evaluation": evaluation[list(EVAL_FIELDS)]}
    rebuilt = data_tools.trajectory_tables(result, method=point.method, horizon=horizon,
                                           n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind in ("summary", "history"):
        for field in rebuilt[kind]:
            a, b = tables[kind][field], rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round", "rounds", "variant", "model_sha256", "roundtrip_verified")
            if not (np.array_equal(a, b) if exact else np.allclose(a, b, rtol=1e-12, atol=1e-12)):
                raise RuntimeError("Reconstructed development traffic/metrics differ")
    return tables


def verify_screen_prefix(point, result, science, split, out=OUT):
    old = verified_tables(point, SCREEN, science, split, out)
    rebuilt = data_tools.trajectory_tables(result, method=point.method, horizon=SCREEN,
                                           n_clients=science["protocol"]["n_clients"], dimension=science["protocol"]["dimension"])
    for kind in ("rounds", "evaluation"):
        columns = [field for field in rebuilt[kind] if field != "codec_audit_seconds"]
        pd.testing.assert_frame_equal(old[kind][columns].reset_index(drop=True), rebuilt[kind][columns].reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def run_point(point, horizon, science, split, out=OUT):
    with full.development.point_lock(out / "locks" / (point.point_id + ".lock")):
        receipt = read_receipt(point, horizon, science, split, out)
        if receipt is not None:
            return f"reuse {point.point_id}/{horizon}: {receipt['state']}"
        if build_contract() != science:
            raise RuntimeError("Worker source/runtime/data changed")
        data = data_tools.load_data(full.DATA)
        if data.manifest != split:
            raise RuntimeError("Worker training/validation split differs")

        def progress(row, validation):
            if row["round"] == 1 or row["round"] % 60 == 0:
                print(f"{point.point_id}: {row['round']}/{horizon}, validation_ce={validation['validation_ce']:.6f}", flush=True)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = data_tools.evaluated_run(data=data, method=point.method, config=config_for(point), seed=data_tools.TRAIN_SEED,
                                                  checkpoint_path=out / "checkpoints" / (point.point_id + ".npz"),
                                                  provenance=checkpoint_identity(point, science, split), horizon=horizon, progress_callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity": metadata(point, horizon, science, split), "state": "numerical_failure", "error": str(exc)},
                        paths(point, horizon, out)["receipt"])
            return f"numerical failure {point.point_id}/{horizon}: {exc}"
        if horizon == FINAL:
            verify_screen_prefix(point, result, science, split, out)
        save_result(point, horizon, science, split, result, out)
        return f"done {point.point_id}/{horizon}"


def execute(points, horizon, workers, science, split, out=OUT):
    jobs = [point for point in points if read_receipt(point, horizon, science, split, out) is None]
    print(f"{len(jobs)} development runs pending to round {horizon}", flush=True)
    if workers == 1:
        for point in jobs:
            print(run_point(point, horizon, science, split, out), flush=True)
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for job in as_completed([pool.submit(run_point, point, horizon, science, split, out) for point in jobs]):
                print(job.result(), flush=True)


def ledger(points, horizon, science, split, out=OUT):
    rows = []
    for point in points:
        receipt = read_receipt(point, horizon, science, split, out)
        if receipt is None:
            raise RuntimeError("Development grid incomplete; no pending point is dropped")
        row = {"point_id": point.point_id, "method": point.method, "threshold": point.threshold,
               "equivalent_gain": point.equivalent_gain if point.method == "strom" else np.nan,
               "native_gain": config_for(point).server_gain, "horizon": horizon,
               "state": receipt["state"], "error": receipt.get("error", "")}
        if receipt["state"] == "complete":
            tables = verified_tables(point, horizon, science, split, out)
            endpoint = tables["summary"].set_index("variant").loc["both_codecs"]
            for field in ("final_train_objective", "final_train_ce", *["final_" + name for name in data_tools.VALIDATION_FIELDS],
                          "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"):
                row[field] = endpoint[field]
            row["final_residual_l2_norm"] = tables["rounds"].iloc[-1].get("residual_l2_norm", np.nan)
            row["final_residual_max_abs"] = tables["rounds"].iloc[-1].get("residual_max_abs", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def select_points(frame):
    event = frame[(frame.method == "event") & (frame.state == "complete")]
    if len(event) != 1:
        raise RuntimeError("A complete Event development budget reference is required")
    budget = float(event.iloc[0].unicast_total_bits)
    candidates = frame[(frame.method == "strom") & (frame.state == "complete")].copy()
    if candidates.empty:
        raise RuntimeError("No finite Ström candidate; inspect the complete failure ledger")
    if (not np.isfinite(candidates[["final_validation_ce", "unicast_total_bits"]]).all().all()
            or candidates.unicast_total_bits.le(0).any()):
        raise ValueError("Invalid completed validation candidate")
    order = ["final_validation_ce", "unicast_total_bits", "point_id"]

    def choice(row):
        return {"point_id": row.point_id, "threshold": float(row.threshold), "equivalent_gain": float(row.equivalent_gain),
                "validation_ce": float(row.final_validation_ce), "unicast_bits": int(row.unicast_total_bits)}

    quality = choice(candidates.sort_values(order).iloc[0])
    eligible = candidates[candidates.unicast_total_bits <= budget]
    constrained = choice(eligible.sort_values(order).iloc[0]) if len(eligible) else None
    candidates["budget_distance"] = np.abs(np.log(candidates.unicast_total_bits / budget))
    nearest = choice(candidates.sort_values(["budget_distance", *order]).iloc[0])
    return {"event_unicast_budget_bits": int(budget), "quality": quality,
            "within_event_budget": constrained, "nearest_traffic_diagnostic": nearest}


def shortlist(selection):
    other = selection["within_event_budget"] or selection["nearest_traffic_diagnostic"]
    return sorted({selection["quality"]["point_id"], other["point_id"]})


def boundary_points(selection):
    chosen = {entry["point_id"]: entry for entry in (selection["quality"],
              selection["within_event_budget"] or selection["nearest_traffic_diagnostic"])}
    initial_ids = {p.point_id for p in INITIAL}
    proposals = {}
    for row in chosen.values():
        tau, gain = row["threshold"], row["equivalent_gain"]
        taus, gains = [tau], [gain]
        if tau == min(THRESHOLDS):
            taus.append(tau / 2)
        elif tau == max(THRESHOLDS):
            taus.append(tau * 2)
        if gain == min(GAINS):
            gains.append(gain / 2)
        elif gain == max(GAINS):
            gains.append(gain * 2)
        for new_tau in taus:
            for new_gain in gains:
                point = strom_point(new_tau, new_gain)
                if point.point_id not in initial_ids:
                    proposals[point.point_id] = point
    if len(proposals) > 6:
        raise AssertionError("Boundary extension exceeded its declared cap")
    return tuple(proposals[name] for name in sorted(proposals))


def receipt_fingerprints(points, horizon, out):
    return {p.point_id: text_hash(paths(p, horizon, out)["receipt"]) for p in points}


def screen(workers, science, split, out=OUT):
    initial = (EVENT, *INITIAL)
    execute(initial, SCREEN, workers, science, split, out)
    base = ledger(initial, SCREEN, science, split, out)
    atomic_csv(base, out / "screening_initial.csv")
    initial_selection = select_points(base)
    extensions = boundary_points(initial_selection)
    boundary = {"contract_sha256": digest(science), "split_sha256": digest(split),
                "initial_selection": initial_selection, "initial_receipts": receipt_fingerprints(initial, SCREEN, out),
                "additional_points": [asdict(p) for p in extensions], "extension_passes": 1}
    immutable_json(boundary, out / "boundary_extension.json")
    execute(extensions, SCREEN, workers, science, split, out)
    points = (*initial, *extensions)
    rows = ledger(points, SCREEN, science, split, out)
    atomic_csv(rows, out / "screening.csv")
    selected = select_points(rows)
    decision = {"contract_sha256": digest(science), "split_sha256": digest(split), "horizon": SCREEN,
                "selection": selected, "promoted_strom_ids": shortlist(selected),
                "all_screen_receipts": receipt_fingerprints(points, SCREEN, out),
                "budget_fallback_is_diagnostic": selected["within_event_budget"] is None}
    immutable_json(decision, out / "screening_selection.json")
    return points, decision


def verify_screening(science, split, out=OUT):
    base_points = (EVENT, *INITIAL)
    base_selection = select_points(ledger(base_points, SCREEN, science, split, out))
    extensions = boundary_points(base_selection)
    expected_boundary = {"contract_sha256": digest(science), "split_sha256": digest(split),
                         "initial_selection": base_selection, "initial_receipts": receipt_fingerprints(base_points, SCREEN, out),
                         "additional_points": [asdict(p) for p in extensions], "extension_passes": 1}
    if json.loads((out / "boundary_extension.json").read_text()) != expected_boundary:
        raise RuntimeError("Frozen boundary decision changed")
    points = (*base_points, *extensions)
    selected = select_points(ledger(points, SCREEN, science, split, out))
    expected = {"contract_sha256": digest(science), "split_sha256": digest(split), "horizon": SCREEN,
                "selection": selected, "promoted_strom_ids": shortlist(selected),
                "all_screen_receipts": receipt_fingerprints(points, SCREEN, out),
                "budget_fallback_is_diagnostic": selected["within_event_budget"] is None}
    if json.loads((out / "screening_selection.json").read_text()) != expected:
        raise RuntimeError("Frozen validation screening selection changed")
    return points, expected


def finalize(workers, science, split, out=OUT):
    points, decision = verify_screening(science, split, out)
    by_id = {p.point_id: p for p in points}
    promoted = (EVENT, *[by_id[name] for name in decision["promoted_strom_ids"]])
    execute(promoted, FINAL, workers, science, split, out)
    rows = ledger(promoted, FINAL, science, split, out)
    atomic_csv(rows, out / "finalists.csv")
    selected = select_points(rows)
    frozen = {"contract_sha256": digest(science), "split_sha256": digest(split), "screen_horizon": SCREEN,
              "final_horizon": FINAL, "selection_metric": "final_validation_ce", "selection": selected,
              "scope": "two_stage_development_selection_among_promoted_finalists",
              "screening_selection_sha256": text_hash(out / "screening_selection.json"),
              "final_receipts": receipt_fingerprints(promoted, FINAL, out),
              "selected_configs": {role: asdict(config_for(by_id[value["point_id"]])) if value else None
                                   for role, value in (("quality", selected["quality"]), ("within_event_budget", selected["within_event_budget"]))},
              "official_test_evaluations": 0, "confirmation_included": False}
    immutable_json(frozen, out / "frozen_selection.json")
    print(json.dumps(frozen["selection"], indent=2), flush=True)
    return frozen


def run(workers, phase="all", out=OUT):
    with full.development.point_lock(out / "locks" / "controller.lock"):
        science, split = prepare(out)
        if phase in ("all", "screen"):
            screen(workers, science, split, out)
        if phase in ("all", "finalize"):
            finalize(workers, science, split, out)


def analyze(out=OUT):
    science = load_contract(out)
    split = load_split(science, out)
    points, decision = verify_screening(science, split, out)
    screened = ledger(points, SCREEN, science, split, out)
    atomic_csv(screened, out / "screening.csv")
    by_id = {p.point_id: p for p in points}
    promoted = (EVENT, *[by_id[name] for name in decision["promoted_strom_ids"]])
    finals = None
    if (out / "frozen_selection.json").exists():
        finals = ledger(promoted, FINAL, science, split, out)
        saved = json.loads((out / "frozen_selection.json").read_text())
        if (saved["selection"] != select_points(finals) or saved["final_receipts"] != receipt_fingerprints(promoted, FINAL, out)
                or saved["screening_selection_sha256"] != text_hash(out / "screening_selection.json")
                or saved["contract_sha256"] != digest(science) or saved["split_sha256"] != digest(split)):
            raise RuntimeError("Final validation selection or source evidence changed")
        for role, choice in (("quality", saved["selection"]["quality"]), ("within_event_budget", saved["selection"]["within_event_budget"])):
            expected = asdict(config_for(by_id[choice["point_id"]])) if choice else None
            if saved["selected_configs"][role] != expected:
                raise RuntimeError("Selected final configuration changed")
        atomic_csv(finals, out / "finalists.csv")
    plot(screened, finals, out)
    atomic_json({"validation_only": True, "official_test_evaluations": 0, "split_disjointness_verified": True,
                 "screened_strom_points": len(points) - 1, "boundary_passes": 1,
                 "promoted_strom_points": len(promoted) - 1, "full_horizon_selection_available": finals is not None,
                 "selection_metric": "final_validation_ce", "confirmation_included": False}, out / "analysis_audit.json")
    print(screened[["point_id", "state", "final_validation_ce", "unicast_total_bits"]].to_string(index=False))


def plot(screened, finals, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for frame, axis, title in ((screened, axes[0], "600-round validation screening"), (finals, axes[1], "1800-round promoted finalists")):
        if frame is None:
            axis.text(.5, .5, "Finalists not completed", transform=axis.transAxes, ha="center")
        else:
            for method, marker, color in (("strom", "o", "#0072B2"), ("event", "*", "#D55E00")):
                rows = frame[(frame.method == method) & (frame.state == "complete")]
                axis.scatter(rows.unicast_total_bits / 1e6, rows.final_validation_ce, marker=marker, color=color,
                             s=80 if method == "event" else 35, label="Event reference" if method == "event" else "Ström candidates")
            axis.legend(fontsize=8)
        axis.set(title=title, xlabel="Total unicast traffic (Mbit)", ylabel="Validation cross-entropy")
        axis.grid(alpha=.2)
    fig.suptitle("Training-derived validation only; lower CE and lower traffic are preferable")
    fig.tight_layout()
    fig.savefig(out / "development_frontier.png", dpi=180)
    fig.savefig(out / "development_frontier.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def smoke_worker(method):
    from dataclasses import replace as dc_replace
    from stc_stage1 import smoke_federation
    fed = smoke_federation()
    data = data_tools.DevelopmentData(dc_replace(fed, X_test=np.empty((0, 3, 32, 32), dtype=np.uint8), y_test=np.empty(0, dtype=int)),
                                      fed.X_test, fed.y_test, pd.DataFrame(), {"synthetic": True})
    point = EVENT if method == "event" else INITIAL[1]
    config = replace(config_for(point), rounds=4, eval_stride=2, local_steps=1, batch_size=2)
    kwargs = dict(data=data, method=method, config=config, seed=84500, provenance={"synthetic": True})
    with tempfile.TemporaryDirectory() as folder:
        a = data_tools.evaluated_run(**kwargs, checkpoint_path=Path(folder) / "full.npz", horizon=4)
        path = Path(folder) / "promote.npz"
        data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=2)
        b = data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=4)
        np.testing.assert_array_equal(a["model"].view(np.uint32), b["model"].view(np.uint32))
        pd.testing.assert_frame_equal(a["evaluation"], b["evaluation"])
        if method == "strom":
            np.testing.assert_array_equal(a["strom_state"].view(np.uint32), b["strom_state"].view(np.uint32))
        return a["evaluation"].iloc[-1].model_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "smoke", "preflight", "run", "screen", "finalize", "analyze", "status"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if args.command == "verify":
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")])
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_strom_development.py", "-v"],
                       cwd=REPO, env=env, check=True)
    if args.command in ("verify", "smoke"):
        methods = ("event", "strom")
        expected = [smoke_worker(method) for method in methods]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            if list(pool.map(smoke_worker, methods)) != expected:
                raise AssertionError("Spawned validation workers differ")
        print("Synthetic validation-only runs: exact promotion/resume and spawned workers; no CIFAR research training.")
    elif args.command == "preflight":
        science = build_science()
        print("Frozen codecs/hosts unchanged; official test curves not loaded")
        print("45000 training / 5000 genuine validation images; development partition 4500, SGD 84500")
        print("15 initial Ström settings + fixed Event; at most 6 boundary points and 2 Ström finalists")
        print("600-round screen; exact continuation of finalists and Event to 1800")
        print("Runtime matches original environment:", full.development.runtime_manifest() == science["runtime"])
    elif args.command in ("run", "screen", "finalize"):
        run(args.workers, "all" if args.command == "run" else args.command)
    elif args.command == "analyze":
        analyze()
    elif args.command == "status":
        if not (OUT / "protocol.json").exists():
            print("Ström development study not started")
            return
        science, split = load_contract(), load_split(load_contract())
        points = [EVENT, *INITIAL]
        if (OUT / "boundary_extension.json").exists():
            points += [Point(**p) for p in json.loads((OUT / "boundary_extension.json").read_text())["additional_points"]]
        for point in points:
            states = []
            for horizon in (SCREEN, FINAL):
                receipt = read_receipt(point, horizon, science, split)
                states.append(f"r{horizon}: {receipt['state'] if receipt else 'pending/not promoted'}")
            path = OUT / "checkpoints" / (point.point_id + ".npz")
            if path.exists():
                with np.load(path, allow_pickle=False) as archive:
                    states.append("checkpoint " + str(json.loads(str(archive["metadata"].item()))["progress"]["completed_round"]))
            print(point.point_id, "; ".join(states))
        print("Frozen 1800-round selection:", (OUT / "frozen_selection.json").exists())


if __name__ == "__main__":
    main()
