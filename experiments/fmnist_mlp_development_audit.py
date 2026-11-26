"""Audit the completed MLP calibration from saved evidence, without training.

Original campaign files are read only. New derivative products use the
analysis_development_ prefix; --check validates those products without writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fmnist_mlp_calibrated as study
from fmnist_mlp_data import digest, file_hash
import numpy as np
import pandas as pd

PREFIX = "analysis_development_"


def inventory(out):
    metadata = {"protocol.json", "development_partition.json", "fresh_seed_audit.json", "screen_summary.csv",
                "promotion.json", "finalist_summary.csv", "frozen_selection.json", "selected.csv"}
    # Keep this validation audit valid when the separate confirmation phase
    # subsequently adds its own files to the same campaign directory.
    return {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()
            and not p.name.startswith(PREFIX) and ("_development_" in p.name or p.name in metadata)}


def csv_bytes(frame):
    return frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()


def audit(out):
    original = inventory(out)
    science = study.load_contract(out, check_runtime=False)
    selected = study.verify_selection(science, out)
    expected_manifest = {**json.loads((out/"development_partition.json").read_text()), "scope": "validation"}
    counts = np.array(expected_manifest["class_counts"])
    expected_counts = np.full((10, 10), 270)
    np.fill_diagonal(expected_counts, 2970)
    if (not np.array_equal(counts, expected_counts) or expected_manifest["training_examples"] != 54000
            or expected_manifest["validation_examples"] != 6000):
        raise RuntimeError("Development split/counts differ")
    initial = 32*25818+64
    stages, rounds_count, sampled_count = [], 0, 0
    for path in sorted(out.glob("*_receipt.json")):
        receipt = json.loads(path.read_text())
        meta = receipt["identity"]
        if meta["phase"] != "development":
            continue
        if receipt["state"] != "complete" or receipt["data_manifest"] != expected_manifest:
            raise RuntimeError("Non-complete or differently allocated development outcome")
        p, horizon = study.BY_ID[meta["point_id"]], meta["stage_horizon"]
        files = study.stage_paths(p, "development", study.DEV_SEED, horizon, out)
        for key in ("rounds", "evaluation", "history", "summary"):
            if file_hash(files[key]) != receipt["sha256"][key]:
                raise RuntimeError("Receipt/CSV hash differs")
        tables = {key: pd.read_csv(files[key], float_precision="round_trip", dtype={"model_sha256": str})
                  for key in ("rounds", "evaluation", "history", "summary")}
        records, evaluation, history, summary = (tables[k] for k in ("rounds", "evaluation", "history", "summary"))
        expected_rounds = [1]+list(range(50, horizon+1, 50))
        if (records["round"].tolist() != list(range(1, horizon+1))
                or evaluation["round"].tolist() != expected_rounds or history["round"].tolist() != expected_rounds):
            raise RuntimeError("Development stage coverage differs")
        if any(c.startswith("test_") for c in evaluation):
            raise RuntimeError("Official-test metric found in validation stage")
        if not records.method.eq(p.method).all() or not records.partition_seed.eq(study.DEV_SEED).all():
            raise RuntimeError("Development model/seed identity differs")
        if not records.model_sha256.str.fullmatch("[a-f0-9]{64}").all():
            raise RuntimeError("Invalid model fingerprint")
        if not records.roundtrip_verified.eq(True).all():
            raise RuntimeError("Missing exact round-trip verification")
        up, down = records.codec_uplink_bits, records.exact_downlink_bits
        if (up.lt(0).any() or down.le(0).any() or (up % 8).ne(0).any() or (down % 8).ne(0).any()
                or records.active_clients.lt(0).any() or records.active_clients.gt(10).any()):
            raise RuntimeError("Invalid packet/counter constraints")
        if p.method == "dense":
            if not up.eq(10*initial).all() or not down.eq(initial).all():
                raise RuntimeError("Dense frame formula differs")
        elif p.method in ("event", "strom"):
            if (not up.ge(128*records.active_clients).all()
                    or not down.le(records.aggregate_downlink_bits).all()
                    or not down.le(records.ordered_downlink_bits).all()):
                raise RuntimeError("Pulse frame portfolio/header bound differs")
        fingerprints = records.set_index("round").model_sha256
        if evaluation.model_sha256.tolist() != fingerprints.loc[expected_rounds].tolist():
            raise RuntimeError("Audited model fingerprint differs")
        evaluation_index = evaluation.set_index("round")
        for row in history.to_dict("records"):
            rnd = int(row["round"])
            # Independent integer sums rather than the runner's prefix builder.
            upload = sum(int(v) for v in up.iloc[:rnd])
            stream = sum(int(v) for v in down.iloc[:rnd])
            unicast_down = 10*(initial+32*rnd+stream)
            expected = {"uplink_bits": upload, "server_stream_bits": stream, "unicast_downlink_bits": unicast_down,
                        "unicast_total_bits": upload+unicast_down, "broadcast_total_bits": upload+initial+stream}
            for field, value in expected.items():
                if int(row[field]) != value:
                    raise RuntimeError(f"Independent prefix accounting differs: {field}")
            for field in ("validation_ce", "validation_accuracy", "validation_macro_accuracy", "validation_worst_class_accuracy", "train_ce", "train_objective"):
                if not np.isfinite(row[field]) or row[field] != evaluation_index.loc[rnd, field]:
                    raise RuntimeError(f"Validation/history metric differs: {field}")
            if not 0 <= row["validation_accuracy"] <= 1 or row["validation_ce"] < 0:
                raise RuntimeError("Invalid predictive metric")
        last = summary.iloc[0]
        if int(last["rounds"]) != horizon or last.model_sha256 != fingerprints.iloc[-1]:
            raise RuntimeError("Final development horizon/model differs")
        for field in ("uplink_bits", "server_stream_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits"):
            if int(last[field]) != int(history.iloc[-1][field]):
                raise RuntimeError("Final accounting differs")
        for field in ("validation_ce", "validation_accuracy", "validation_worst_class_accuracy", "train_ce", "train_objective"):
            if last["final_"+field] != history.iloc[-1][field]:
                raise RuntimeError("Final development metric differs")
        stages.append({"point_id": p.point_id, "method": p.method, "horizon": horizon,
                       "validation_ce": float(last.final_validation_ce),
                       "validation_accuracy_percent": 100*float(last.final_validation_accuracy),
                       "uplink_Gbit": float(last.uplink_bits)/1e9, "downlink_Gbit": float(last.unicast_downlink_bits)/1e9,
                       "total_Gbit": float(last.unicast_total_bits)/1e9, "broadcast_Gbit": float(last.broadcast_total_bits)/1e9})
        rounds_count += len(records)
        sampled_count += len(evaluation)
    stages = pd.DataFrame(stages)
    if len(stages) != 105 or stages.groupby(["method", "horizon"]).size().to_dict() != {
            (method, horizon): count for method in study.METHODS for horizon, count in ((500, 18), (1500, 3))}:
        raise RuntimeError("Expected exactly 90 screens and 15 continuations")
    selected_rows = stages[(stages.horizon == 1500) & stages.point_id.isin([v[0] for v in selected["selected"].values()])].copy()
    selected_rows = selected_rows.set_index("method").loc[list(study.METHODS)].reset_index()
    event = selected_rows[selected_rows.method == "event"].iloc[0]
    comparisons = []
    for row in selected_rows.to_dict("records"):
        if row["method"] == "event":
            continue
        comparisons.append({"reference": row["method"], "n_development_splits": 1,
                            "event_accuracy_difference_pp": event.validation_accuracy_percent-row["validation_accuracy_percent"],
                            "event_CE_difference": event.validation_ce-row["validation_ce"],
                            "event_unicast_saving_percent": 100*(1-event.total_Gbit/row["total_Gbit"]),
                            "event_uplink_saving_percent": 100*(1-event.uplink_Gbit/row["uplink_Gbit"]),
                            "event_downlink_saving_percent": 100*(1-event.downlink_Gbit/row["downlink_Gbit"]),
                            "inference": "single_development_split_no_confidence_interval_no_confirmatory_claim"})
    after = inventory(out)
    if original != after:
        raise RuntimeError("An original campaign file changed during audit")
    audit_record = {"scope": "completed_validation_calibration_only", "contract_sha256": digest(science),
                    "selection_sha256": digest(selected), "source_hashes_checked": len(science["source_hashes"]),
                    "screens": 90, "continuations": 15, "numerical_failures": 0, "unique_training_rounds": 60000,
                    "saved_round_records_with_screen_prefixes": rounds_count,
                    "saved_validation_fingerprints_with_screen_prefixes": sampled_count,
                    "confirmation_results": "outside_this_validation_only_audit",
                    "official_test_metrics_used_for_selection": False,
                    "hyperparameter_selection_reproduced": True, "exact_continuation_prefixes_reproduced": True,
                    "independent_integer_accounting": True, "raw_files_unchanged": True,
                    "original_file_sha256": original}
    return {PREFIX+"stages.csv": csv_bytes(stages), PREFIX+"selected.csv": csv_bytes(selected_rows),
            PREFIX+"comparisons.csv": csv_bytes(pd.DataFrame(comparisons)),
            PREFIX+"audit.json": (json.dumps(audit_record, indent=2, sort_keys=True)+"\n").encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    products = audit(args.out)
    for name, value in products.items():
        path = args.out/name
        if args.check:
            if not path.exists() or path.read_bytes() != value:
                raise RuntimeError(f"Stale calibration audit product: {name}")
        else:
            path.write_bytes(value)
    print("Validated 105 calibration stages, 67500 saved rounds, 1455 validation fingerprints; no training or test inference.")


if __name__ == "__main__":
    main()
