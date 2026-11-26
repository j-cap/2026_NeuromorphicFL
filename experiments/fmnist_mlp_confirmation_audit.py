"""Verify the closed MLP panel without training, images or checkpoints.

Rebuilds the nine original numerical products in a temporary copy; checks raw
receipts, source/selection identities, prefixes, frame constraints and data
manifests; independently recomputes paired effects and target crossings.
Original files remain unchanged. --check also checks the new audit products.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import fmnist_mlp_calibrated as study
import fmnist_mlp_analysis as original_analysis
from fmnist_mlp_data import digest, file_hash

PREFIX = "analysis_confirmation_"
RESULTS_COMMIT = "b2e2ce8226cacefa8c36855f474f0769e87b20ad"
NUMERICAL = ("confirmation_coverage.csv", "confirmation_completion.json", "confirmation_runs.csv",
             "confirmation_history.csv", "endpoint_summary.csv", "paired_endpoints.csv",
             "target_crossings.csv", "paired_targets.csv", "analysis_audit.json")
T95 = {2: 12.706204736432095, 3: 4.302652729696142, 4: 3.182446305284263,
       5: 2.7764451051977987, 6: 2.570581835636314, 7: 2.4469118511449692,
       8: 2.3646242515927844, 9: 2.306004135204166, 10: 2.2621571628540993}


def read(path):
    return pd.read_csv(path, float_precision="round_trip", dtype={"model_sha256": str})


def inventory(out):
    return {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file() and not p.name.startswith(PREFIX)}


def close(actual, expected):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-11, equal_nan=True):
        raise RuntimeError(f"Numerical reproduction differs: {actual} / {expected}")


def paired(a, b, *, ratio=False):
    if len(a) != len(b):
        raise ValueError("Unpaired observations")
    if not len(a):
        return None, None, None
    values = [math.log(x/y) if ratio else x-y for x, y in zip(a, b)]
    mean = statistics.fmean(values)
    estimate = math.exp(mean) if ratio else mean
    if len(values) == 1:
        return estimate, None, None
    half = T95[len(values)]*statistics.stdev(values)/math.sqrt(len(values))
    return (estimate, math.exp(mean-half), math.exp(mean+half)) if ratio else (estimate, mean-half, mean+half)


def compare_file(actual, expected):
    if actual.suffix == ".json":
        if json.loads(actual.read_text()) != json.loads(expected.read_text()):
            raise RuntimeError(f"JSON reproduction differs: {actual.name}")
        return
    a, b = read(actual), read(expected)
    if a.columns.tolist() != b.columns.tolist() or a.shape != b.shape:
        raise RuntimeError(f"CSV reproduction schema differs: {actual.name}")
    for column in a:
        x, y = a[column], b[column]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if pd.api.types.is_integer_dtype(x) and pd.api.types.is_integer_dtype(y):
                if not np.array_equal(x, y):
                    raise RuntimeError(f"Integer reproduction differs: {actual.name}/{column}")
            else:
                close(x, y)
        elif x.fillna("<missing>").astype(str).tolist() != y.fillna("<missing>").astype(str).tolist():
            raise RuntimeError(f"String reproduction differs: {actual.name}/{column}")


def csv_bytes(frame):
    return frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()


def audit(out):
    originals = inventory(out)
    science = study.load_contract(out, check_runtime=False)
    selection = study.verify_selection(science, out)
    selected_sha = digest(selection)
    test_manifest = json.loads((out/"confirmation_test_manifest.json").read_text())
    if test_manifest["scope"] != "test" or test_manifest["examples"] != 10000:
        raise RuntimeError("Confirmation test scope differs")
    with tempfile.TemporaryDirectory(prefix="mlp_confirmation_rebuild_") as folder:
        shadow = Path(folder)
        for name in originals:
            if name not in (*NUMERICAL, "comparison.pdf", "comparison.png"):
                shutil.copy2(out/name, shadow/name)
        with redirect_stdout(io.StringIO()):
            original_analysis.analyze(shadow)
        for name in NUMERICAL:
            compare_file(out/name, shadow/name)
    endpoints, histories, round_count = [], [], 0
    initial = 32*25818+64
    expected_counts = np.full((10, 10), 300)
    np.fill_diagonal(expected_counts, 3300)
    for method in study.METHODS:
        p = study.BY_ID[selection["selected"][method][0]]
        for seed in study.SEEDS:
            receipt = study.read_receipt(p, "confirmation", seed, 1500, science, out, selection_sha=selected_sha)
            if receipt is None or receipt["state"] != "complete":
                raise RuntimeError("All fifty complete fits are required")
            manifest = json.loads((out/f"confirmation_partition_p{seed}.json").read_text())
            if (receipt["data_manifest"] != manifest or manifest["training_examples"] != 60000
                    or manifest["validation_examples"] != 0 or manifest["examples_per_client"] != 6000
                    or manifest["scope"] != "test" or manifest["split_seed"] is not None
                    or manifest["test_manifest_sha256"] != digest(test_manifest)
                    or not np.array_equal(manifest["class_counts"], expected_counts)):
                raise RuntimeError("Full-training/test allocation differs")
            files = study.stage_paths(p, "confirmation", seed, 1500, out)
            tables = {key: read(files[key]) for key in ("rounds", "evaluation", "history", "summary")}
            original_analysis.validate_tables(tables, method=method, seed=seed,
                    config=study.FinalBaselineConfig(**selection["configs"][method]), horizon=1500)
            rounds = tables["rounds"]
            up, down = rounds.codec_uplink_bits, rounds.exact_downlink_bits
            if ((up % 8).ne(0).any() or (down % 8).ne(0).any()
                    or not rounds.model_sha256.str.fullmatch("[a-f0-9]{64}").all()):
                raise RuntimeError("Frame alignment/fingerprint differs")
            if method == "dense" and (not up.eq(10*initial).all() or not down.eq(initial).all()):
                raise RuntimeError("Dense actual-frame accounting differs")
            if method in ("event", "strom") and (not up.ge(128*rounds.active_clients).all()
                    or not down.eq(np.minimum(rounds.aggregate_downlink_bits, rounds.ordered_downlink_bits)).all()):
                raise RuntimeError("Pulse header/portfolio constraints differ")
            if method == "ef_topk" and not down.eq(np.minimum(rounds.aggregate_downlink_bits, rounds.ordered_downlink_bits)).all():
                raise RuntimeError("Float32 aggregate/replay selection differs")
            endpoints.append(tables["summary"].iloc[0].to_dict())
            histories.extend(tables["history"].to_dict("records"))
            round_count += len(rounds)
    runs = pd.DataFrame(endpoints)
    history = pd.DataFrame(histories)
    endpoint_summary = read(out/"endpoint_summary.csv").set_index("method")
    metrics = ("final_test_accuracy", "final_test_ce", "final_test_worst_class_accuracy", "final_train_ce",
               "final_train_objective", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits")
    for method in study.METHODS:
        group = runs[runs.method == method]
        for metric in metrics:
            values = [float(v) for v in group[metric]]
            close(endpoint_summary.loc[method, metric+"_mean"], statistics.fmean(values))
            close(endpoint_summary.loc[method, metric+"_std"], statistics.stdev(values))
            if endpoint_summary.loc[method, metric+"_count"] != 10:
                raise RuntimeError("Endpoint seed denominator differs")
    effects, counts = [], []
    event = runs[runs.method == "event"].set_index("partition_seed").sort_index()
    for method in study.METHODS:
        if method == "event":
            continue
        reference = runs[runs.method == method].set_index("partition_seed").sort_index()
        for metric, field, scale, ratio in (("accuracy_difference_pp", "final_test_accuracy", 100, False),
                                           ("ce_difference", "final_test_ce", 1, False),
                                           ("unicast_ratio", "unicast_total_bits", 1, True),
                                           ("upload_ratio", "uplink_bits", 1, True),
                                           ("download_ratio", "unicast_downlink_bits", 1, True),
                                           ("broadcast_ratio", "broadcast_total_bits", 1, True)):
            estimate, low, high = paired([scale*float(x) for x in event[field]], [scale*float(x) for x in reference[field]], ratio=ratio)
            effects.append(dict(reference=method, metric=metric, n=10, estimate=estimate, ci95_low=low, ci95_high=high,
                                saving_percent=100*(1-estimate) if ratio else None,
                                saving_ci95_low=100*(1-high) if ratio else None,
                                saving_ci95_high=100*(1-low) if ratio else None))
        counts.append({"reference": method, "n": 10,
                       "event_lower_final_cost": int((event.unicast_total_bits < reference.unicast_total_bits).sum()),
                       "event_higher_accuracy": int((event.final_test_accuracy > reference.final_test_accuracy).sum()),
                       "equal_accuracy": int((event.final_test_accuracy == reference.final_test_accuracy).sum()),
                       "event_lower_CE": int((event.final_test_ce < reference.final_test_ce).sum())})
    effects = pd.DataFrame(effects)
    original_effects = read(out/"paired_endpoints.csv")
    for row in effects.to_dict("records"):
        original = original_effects[(original_effects.reference == row["reference"]) & (original_effects.metric == row["metric"])].iloc[0]
        close([row[k] for k in ("estimate", "ci95_low", "ci95_high")], [original[k] for k in ("estimate", "ci95_low", "ci95_high")])
    hits, descriptions = {}, []
    original_crossings = read(out/"target_crossings.csv").set_index(["method", "partition_seed", "target_accuracy"])
    for method in study.METHODS:
        for target in study.TARGETS:
            achieved = []
            for seed in study.SEEDS:
                curve = history[(history.method == method) & (history.partition_seed == seed)].sort_values("round")
                first = next((r for r in curve.to_dict("records") if r["test_accuracy"] >= target), None)
                hits[method, seed, target] = first
                original = original_crossings.loc[(method, seed, target)]
                if bool(original.reached) != (first is not None):
                    raise RuntimeError("Target censoring differs")
                if first is not None:
                    if int(first["round"]) != int(original["round"]) or int(first["unicast_total_bits"]) != int(original.unicast_total_bits):
                        raise RuntimeError("First sampled crossing differs")
                    achieved.append(first)
            descriptions.append({"method": method, "target_accuracy": target, "reached": len(achieved), "total": 10,
                                 "mean_first_sampled_round": statistics.fmean(r["round"] for r in achieved) if achieved else None,
                                 "mean_unicast_bits": statistics.fmean(r["unicast_total_bits"] for r in achieved) if achieved else None,
                                 "missed_partition_seeds": ";".join(str(s) for s in study.SEEDS if hits[method, s, target] is None)})
    targets = read(out/"paired_targets.csv")
    for row in targets.to_dict("records"):
        method, target = row["reference"], row["target_accuracy"]
        common = [(hits["event", s, target], hits[method, s, target]) for s in study.SEEDS
                  if hits["event", s, target] is not None and hits[method, s, target] is not None]
        estimate, low, high = paired([a["unicast_total_bits"] for a, b in common], [b["unicast_total_bits"] for a, b in common], ratio=True)
        close([estimate, low, high], [row["estimate"], row["ci95_low"], row["ci95_high"]])
        for field, expected in {"common_hits": len(common), "event_reached": sum(hits["event", s, target] is not None for s in study.SEEDS),
                               "reference_reached": sum(hits[method, s, target] is not None for s in study.SEEDS),
                               "event_fewer_bits": sum(a["unicast_total_bits"] < b["unicast_total_bits"] for a, b in common),
                               "earlier": sum(a["round"] < b["round"] for a, b in common),
                               "same_sampled_round": sum(a["round"] == b["round"] for a, b in common),
                               "later": sum(a["round"] > b["round"] for a, b in common)}.items():
            if row[field] != expected:
                raise RuntimeError(f"Target denominator/count differs: {field}")
    if inventory(out) != originals:
        raise RuntimeError("An original campaign file changed")
    record = {"status": "complete_five_method_panel_verified", "results_commit": RESULTS_COMMIT,
              "contract_sha256": digest(science), "selection_sha256": selected_sha,
              "source_hashes_checked": len(science["source_hashes"]), "complete_new_fits": len(runs),
              "round_records": round_count, "sampled_model_fingerprints": len(history),
              "unique_final_model_fingerprints": int(runs.model_sha256.nunique()),
              "numerical_products_reproduced_in_temporary_copy": list(NUMERICAL),
              "independent_Python_paired_statistics": True, "first_saved_crossings_and_censoring_verified": True,
              "raw_files_unchanged": True, "training_or_image_loading": False,
              "original_file_sha256": originals}
    final = endpoint_summary.reset_index().copy()
    final["accuracy_percent_mean"] = 100*final.final_test_accuracy_mean
    final["accuracy_percent_sd"] = 100*final.final_test_accuracy_std
    for source, destination in (("uplink_bits_mean", "uplink_Gbit"), ("unicast_downlink_bits_mean", "downlink_Gbit"),
                                ("unicast_total_bits_mean", "unicast_Gbit"), ("broadcast_total_bits_mean", "broadcast_Gbit")):
        final[destination] = final[source]/1e9
    return {PREFIX+"endpoints.csv": csv_bytes(final), PREFIX+"effects.csv": csv_bytes(effects),
            PREFIX+"pair_counts.csv": csv_bytes(pd.DataFrame(counts)), PREFIX+"target_descriptive.csv": csv_bytes(pd.DataFrame(descriptions)),
            PREFIX+"audit.json": (json.dumps(record, indent=2, sort_keys=True)+"\n").encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=study.OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    products = audit(args.out)
    for name, content in products.items():
        destination = args.out/name
        if args.check:
            if not destination.exists() or destination.read_bytes() != content:
                raise RuntimeError(f"Stale confirmation audit product: {name}")
        else:
            destination.write_bytes(content)
    print("Verified 50 complete MLP fits, 75000 rounds, 1550 sampled fingerprints and all original numerical products; original files unchanged.")


if __name__ == "__main__":
    main()
