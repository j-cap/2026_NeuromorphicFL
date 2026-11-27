"""Independently audit the fifteen committed pilot fits without training.

Preserves all original files. Reproduces the original numerical analysis in a
temporary copy; independently rebuilds client catch-up and terminal accounting
from packet lengths, saved schedules and synchronization cursors. Adds only
analysis_pilot_independent_* products. --check verifies those products.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import calibrated_participation as study
import calibrated_participation_analysis as original
import calibrated_participation_data as data

PREFIX = "analysis_pilot_independent_"
RESULTS_COMMIT = "563f4a534e5ae950bc0fd9596b1b39bd374fffd7"
CSV_PRODUCTS = ("coverage", "runs", "history", "endpoint_summary", "paired_endpoints",
                "target_crossings", "paired_targets")


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def original_paths(out):
    names = {"protocol.json"}
    for seed in (data.DEV_SEED, *data.SEEDS):
        names.update((f"partition_p{seed}.json", f"schedule_p{seed}.json"))
    for job in study.jobs_for("pilot"):
        names.update(p.name for p in study.paths(job, out).values())
    names.update("pilot_"+name+".csv" for name in CSV_PRODUCTS)
    names.update(("pilot_completion.json", "pilot_analysis_audit.json", "pilot_comparison.pdf", "pilot_comparison.png"))
    return [out/name for name in sorted(names)]


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-11):
        raise AssertionError(f"Numerical mismatch: {a} != {b}")


def compare_csv(path, rebuilt):
    a = pd.read_csv(path, float_precision="round_trip", keep_default_na=False)
    b = pd.read_csv(rebuilt, float_precision="round_trip", keep_default_na=False)
    # Progress JSON is deliberately local/ignored; this display-only field is
    # not reconstructible from a Git checkout. Receipts provide completeness.
    if path.name == "pilot_coverage.csv":
        a = a.drop(columns="latest_checkpoint_round")
        b = b.drop(columns="latest_checkpoint_round")
    assert a.shape == b.shape and a.columns.tolist() == b.columns.tolist(), path.name
    for key in a:
        if pd.api.types.is_numeric_dtype(a[key]) and pd.api.types.is_numeric_dtype(b[key]):
            assert np.allclose(a[key], b[key], rtol=1e-10, atol=1e-11), (path.name, key)
        else:
            assert a[key].astype(str).tolist() == b[key].astype(str).tolist(), (path.name, key)


def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: format(v, ".12g") if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    return out.getvalue().encode()


def audit(out):
    originals = {p.name: data.file_hash(p) for p in original_paths(out)}
    assert len(originals) == 94
    science = study.load_contract(out)
    assert science["protocol"]["pilot"]["official_test_parsed_or_evaluated"] is False
    assert science["protocol"]["fixed_settings"] == "full_participation_selected_points_transferred_without_retuning"
    for seed in (data.DEV_SEED, *data.SEEDS):
        manifest = json.loads((out/f"partition_p{seed}.json").read_text())
        size = 900 if seed == data.DEV_SEED else 1000
        counts = np.full((50, 10), size//20)
        counts[np.arange(50), np.arange(50) % 10] = 11*size//20
        assert np.array_equal(counts, manifest["class_counts"])
        assert manifest["training_examples"] == 50*size
        assert manifest["validation_examples"] == (5000 if seed == data.DEV_SEED else 0)
        assert manifest["split_seed"] == (23000 if seed == data.DEV_SEED else None)
        assert json.loads((out/f"schedule_p{seed}.json").read_text()) == data.schedule_manifest(seed)
    with tempfile.TemporaryDirectory(prefix="participation_pilot_rebuild_") as folder:
        shadow = Path(folder)
        for p in original_paths(out):
            shutil.copy2(p, shadow/p.name)
        with redirect_stdout(io.StringIO()):
            original.analyze("pilot", shadow)
        for name in CSV_PRODUCTS:
            compare_csv(out/f"pilot_{name}.csv", shadow/f"pilot_{name}.csv")
        assert json.loads((out/"pilot_completion.json").read_text()) == json.loads((shadow/"pilot_completion.json").read_text())
        old_audit = json.loads((out/"pilot_analysis_audit.json").read_text())
        new_audit = json.loads((shadow/"pilot_analysis_audit.json").read_text())
        for key in old_audit:
            if key != "derived_csv_sha256":
                assert old_audit[key] == new_audit[key], key
    endpoints, costs, crossings, effects = [], [], [], []
    fingerprints, raw_count, sampled_count = set(), 0, 0
    checkpoint, initial = 32*20570+64, 50*(32*20570+64)
    source_targets = read(out/"pilot_paired_targets.csv")
    source_effects = read(out/"pilot_paired_endpoints.csv")
    for job in study.jobs_for("pilot"):
        receipt = study.read_receipt(job, science, out)
        assert receipt is not None and receipt["state"] == "complete", job
        study.read_tables(job, science, out)
        _, active, method, seed = job
        files = study.paths(job, out)
        rounds, history, summary = read(files["rounds"]), read(files["history"]), read(files["summary"])[0]
        expected_schedule = data.schedule_for(seed, active)[:300]
        cursors = [0]*50
        prefix = [0]
        cumulative = {"cumulative_update_uplink_bits": 0, "cumulative_request_bits": 0,
            "cumulative_current_downlink_bits": 0, "cumulative_catchup_replay_bits": 0,
            "cumulative_catchup_checkpoint_bits": 0}
        by_round = {int(r["round"]): r for r in history}
        for row in rounds:
            rnd = int(row["round"])
            expected = {"catchup_replay_bits": 0, "catchup_checkpoint_bits": 0,
                        "catchup_replay_clients": 0, "catchup_checkpoint_clients": 0, "current_clients": 0}
            for client in expected_schedule[rnd-1]:
                client = int(client)
                if cursors[client] == rnd-1:
                    expected["current_clients"] += 1
                else:
                    missed = prefix[-1]-prefix[cursors[client]]
                    kind = "replay" if missed <= checkpoint else "checkpoint"
                    expected["catchup_"+kind+"_bits"] += min(missed, checkpoint)
                    expected["catchup_"+kind+"_clients"] += 1
                cursors[client] = rnd
            for key, value in expected.items():
                assert int(row[key]) == value, (job, rnd, key)
            up, down = int(row["codec_uplink_bits"]), int(row["exact_downlink_bits"])
            assert up % 8 == down % 8 == 0
            assert int(row["coordinate_events"]) <= active*20570
            if method == "dense":
                assert up == active*checkpoint and down == checkpoint
            elif method in ("event", "strom", "ef_topk"):
                assert down == min(int(row["aggregate_downlink_bits"]), int(row["ordered_downlink_bits"]))
                if method != "ef_topk":
                    assert up >= 128*int(row["nonempty_uplink_messages"])
            prefix.append(prefix[-1]+down)
            cumulative["cumulative_update_uplink_bits"] += up
            cumulative["cumulative_request_bits"] += 32*active
            cumulative["cumulative_current_downlink_bits"] += active*down
            cumulative["cumulative_catchup_replay_bits"] += expected["catchup_replay_bits"]
            cumulative["cumulative_catchup_checkpoint_bits"] += expected["catchup_checkpoint_bits"]
            for key, value in cumulative.items():
                assert int(row[key]) == value, (job, rnd, key)
            operational = initial+sum(cumulative.values())
            assert int(row["operational_total_bits"]) == operational
            assert int(row["local_sgd_steps"]) == rnd*active*5
            assert int(row["sampled_training_examples"]) == rnd*active*5*32
            if rnd in by_round:
                h = by_round[rnd]
                terminal = {"terminal_downlink_bits": 0, "terminal_request_bits": 0,
                            "terminal_replay_clients": 0, "terminal_checkpoint_clients": 0}
                for last in cursors:
                    if last != rnd:
                        missed = prefix[-1]-prefix[last]
                        kind = "replay" if missed <= checkpoint else "checkpoint"
                        terminal["terminal_downlink_bits"] += min(missed, checkpoint)
                        terminal["terminal_request_bits"] += 32
                        terminal["terminal_"+kind+"_clients"] += 1
                for key, value in terminal.items():
                    assert int(h[key]) == value, (job, rnd, key)
                terminal_total = terminal["terminal_downlink_bits"]+terminal["terminal_request_bits"]
                assert int(h["terminal_catchup_bits"]) == terminal_total
                assert int(h["unicast_total_bits"]) == operational+terminal_total
                assert int(h["terminal_verified_clients"]) == 50
                assert h["model_sha256"] == row["model_sha256"]
                sampled_count += 1
            fingerprints.add(row["model_sha256"])
        raw_count += len(rounds)
        final = history[-1]
        endpoints.append({"active_clients": active, "method": method,
            "validation_accuracy_percent": 100*float(summary["final_validation_accuracy"]),
            "validation_ce": float(summary["final_validation_ce"]),
            "worst_class_accuracy_percent": 100*float(summary["final_validation_worst_class_accuracy"]),
            "total_unicast_Gbit": int(summary["unicast_total_bits"])/1e9,
            "operational_Gbit": int(summary["operational_total_bits"])/1e9,
            "local_sgd_steps": int(summary["local_sgd_steps"])})
        costs.append({"active_clients": active, "method": method,
            **{key: int(final[key]) for key in ("initial_sync_bits", *cumulative,
                "terminal_downlink_bits", "terminal_request_bits", "terminal_catchup_bits", "unicast_total_bits")},
            "catchup_share_percent": 100*(cumulative["cumulative_catchup_replay_bits"]+
                cumulative["cumulative_catchup_checkpoint_bits"])/int(final["unicast_total_bits"]),
            "terminal_share_percent": 100*int(final["terminal_catchup_bits"])/int(final["unicast_total_bits"]),
            "catchup_checkpoint_clients": sum(int(r["catchup_checkpoint_clients"]) for r in rounds),
            "catchup_replay_clients": sum(int(r["catchup_replay_clients"]) for r in rounds),
            "current_clients": sum(int(r["current_clients"]) for r in rounds),
            "recorded_attempt_seconds": float(receipt["elapsed_seconds_this_attempt"])})
        for target in study.TARGETS:
            hit = next((h for h in history if float(h["validation_accuracy"]) >= target), None)
            crossings.append({"active_clients": active, "method": method, "target_percent": 100*target,
                "reached": hit is not None, "first_saved_round": int(hit["round"]) if hit else None,
                "conservative_bits": int(hit["unicast_total_bits"]) if hit else None,
                "operational_bits": int(hit["operational_total_bits"]) if hit else None})
        assert all(float(h["validation_accuracy"]) >= 0 for h in history)
    for active in data.ACTIVE:
        e = next(r for r in endpoints if r["active_clients"] == active and r["method"] == "event")
        for method in study.METHODS:
            if method == "event":
                continue
            r = next(r for r in endpoints if r["active_clients"] == active and r["method"] == method)
            ratio = e["total_unicast_Gbit"]/r["total_unicast_Gbit"]
            effects.append({"active_clients": active, "reference": method, "n": 1,
                "accuracy_difference_pp": e["validation_accuracy_percent"]-r["validation_accuracy_percent"],
                "ce_difference": e["validation_ce"]-r["validation_ce"], "final_cost_ratio": ratio,
                "final_saving_percent": 100*(1-ratio)})
            for metric, estimate in (("accuracy_difference_pp", effects[-1]["accuracy_difference_pp"]),
                    ("ce_difference", effects[-1]["ce_difference"]), ("unicast_ratio", ratio)):
                source = next(x for x in source_effects if int(x["active_clients"]) == active
                    and x["reference"] == method and x["metric"] == metric)
                close(estimate, source["estimate"])
                assert source["ci95_low"] == source["ci95_high"] == ""
            for target in study.TARGETS:
                a = next(x for x in crossings if x["active_clients"] == active and x["method"] == "event"
                         and abs(x["target_percent"]-100*target) < 1e-8)
                b = next(x for x in crossings if x["active_clients"] == active and x["method"] == method
                         and abs(x["target_percent"]-100*target) < 1e-8)
                source = next(x for x in source_targets if int(x["active_clients"]) == active and x["reference"] == method
                              and abs(float(x["target_accuracy"])-target) < 1e-8)
                assert int(source["common_hits"]) == int(a["reached"] and b["reached"])
                if a["reached"] and b["reached"]:
                    close(a["conservative_bits"]/b["conservative_bits"], source["estimate"])
                else:
                    assert source["estimate"] == "", "Censored crossings must not receive zero/imputed ratios"
    assert raw_count == 4500 and sampled_count == 90
    assert originals == {p.name: data.file_hash(p) for p in original_paths(out)}, "Original campaign files changed"
    evidence = {"status": "pilot_protocol_and_accounting_verified", "original_results_commit": RESULTS_COMMIT,
        "contract_sha256": data.digest(science), "source_hashes_verified": len(science["source_hashes"]),
        "complete_fits": 15, "raw_round_records": raw_count, "sampled_fingerprints": sampled_count,
        "distinct_round_fingerprints": len(fingerprints), "seed_pairs_per_cell": 1,
        "old_numerical_csv_products_reproduced": list(CSV_PRODUCTS),
        "transient_progress_field_excluded": "pilot_coverage.latest_checkpoint_round",
        "independent_round_by_round_catchup_and_all_sampled_terminal_costs_verified": True,
        "model_reconstruction_scope": "source-pinned runner assertions and recorded flags; no model arrays/wire journal available in Git",
        "actual_model_training_or_dataset_loading": False, "all_original_files_unchanged": True,
        "original_files_sha256": originals}
    products = {PREFIX+name+".csv": csv_bytes(rows) for name, rows in (
        ("endpoints", endpoints), ("cost_breakdown", costs), ("crossings", crossings), ("effects", effects))}
    products[PREFIX+"audit.json"] = (json.dumps(evidence, indent=2, sort_keys=True)+"\n").encode()
    return products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=study.OUT)
    args = parser.parse_args()
    products = audit(args.out)
    for name, content in products.items():
        path = args.out/name
        if args.check:
            assert path.read_bytes() == content, f"Stale derivative: {name}"
        else:
            path.write_bytes(content)
    print("Verified 15 pilot fits, 4500 rounds, 90 sampled fingerprints, all numerical products and exact catch-up/terminal accounting; original files unchanged.")


if __name__ == "__main__":
    main()
