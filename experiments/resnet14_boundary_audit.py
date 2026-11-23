"""Read-only audit of completed ResNet boundary evidence; no Torch or images.

python experiments/resnet14_boundary_audit.py
git add experiments/results/resnet14_boundary_v1
git commit -m "Refresh ResNet14 boundary evidence audit"
git push origin main
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import resnet14_boundary as study
from resnet14_transfer_audit import independent_traffic
from event_strom_fresh_confirmation_audit import evidence_hash
import pandas as pd

parent = study.parent
ORIGINAL = ("protocol.json", "screening.csv", "promotion.json", "finalists.csv", "comparison.csv",
            "boundary_summary.csv", "frozen_selection.json", "analysis_audit.json")


def verify_portable_contract(recorded, expected):
    """Normalize dependency key separators only; retain the recorded identity."""
    def canonical(science):
        dependencies = {}
        for path, value in science["parent_evidence_sha256"].items():
            key = path.replace("\\", "/")
            if key in dependencies:
                raise RuntimeError("Duplicate canonical parent dependency path")
            dependencies[key] = value
        return {**science, "parent_evidence_sha256": dependencies}
    if canonical(recorded) != canonical(expected):
        raise RuntimeError("Boundary protocol, source/data hashes, runtime or parent evidence differ")
    return recorded


def compare_saved(name, frame):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip", dtype={"model_sha256": str})
    if "error" in saved:
        saved["error"] = saved["error"].fillna("")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def best(frame):
    selected = {}
    for method in ("event", "strom"):
        rows = [r for r in frame.to_dict("records") if r["method"] == method and r["state"] == "complete"]
        if not rows:
            raise RuntimeError("No complete boundary candidate for " + method)
        selected[method] = [min(rows, key=lambda r: (r["final_validation_ce"], r["unicast_total_bits"], r["point_id"]))["point_id"]]
    return selected


def main():
    originals = {n: evidence_hash(study.OUT / n) for n in ORIGINAL}
    recorded = json.loads((study.OUT / "protocol.json").read_text())
    science = verify_portable_contract(recorded, study.science_spec())
    screen = study.ledger(study.POINTS, study.SCREEN, science)
    compare_saved("screening.csv", screen)
    promoted = {
        "contract_sha256": parent.digest(science), "horizon": study.SCREEN,
        "selection_metric": "final_validation_ce", "selected": best(screen),
        "receipt_sha256": {p.point_id: parent.text_hash(parent.paths(p, "development", 7000, study.SCREEN, study.OUT)["receipt"])
                           for p in study.POINTS}, "official_test_evaluations": 0,
    }
    if promoted != json.loads((study.OUT / "promotion.json").read_text()):
        raise RuntimeError("Independent boundary screen ranking/receipt reconstruction differs")
    points = tuple(study.BY_ID[promoted["selected"][m][0]] for m in ("event", "strom"))
    final = study.ledger(points, study.FINAL, science)
    parent.validate_prefixes(points, science, study.OUT)
    prior = parent.load_contract()
    incumbents = tuple(study.BY_ID[study.INCUMBENT_IDS[m]] for m in ("event", "strom"))
    old = parent.ledger(incumbents, study.FINAL, prior, parent.OUT)
    old["origin"], final["origin"] = "parent_incumbent", "boundary_finalist"
    final = final.reindex(columns=old.columns)
    comparison = pd.concat([old, final], ignore_index=True)
    compare_saved("finalists.csv", final)
    compare_saved("comparison.csv", comparison)
    stages, hashes = {}, {}
    for horizon, candidates in ((study.SCREEN, study.POINTS), (study.FINAL, points)):
        for p in candidates:
            table = study.checked_tables(p, horizon, science)
            if table is None:
                raise RuntimeError("This completed-evidence audit requires all eight stages complete")
            independent_traffic(table)
            stages[p.point_id, horizon] = table
            for path in parent.paths(p, "development", 7000, horizon, study.OUT).values():
                hashes[path.name] = evidence_hash(path)
    manifest = json.loads((parent.OUT / "split_manifest.json").read_text())
    for p in incumbents:
        receipt = parent.read_receipt(p, "development", 7000, study.FINAL, prior, parent.OUT)
        if receipt["data_manifest"] != manifest:
            raise RuntimeError("Reused parent development manifest differs")
        independent_traffic(parent.verified_tables(p, "development", 7000, study.FINAL, prior, parent.OUT))
    if (len(stages), sum(len(t["rounds"]) for t in stages.values()),
            sum(len(t["evaluation"]) for t in stages.values())) != (8, 7200, 128):
        raise RuntimeError("Expected six screens, two finalists and both duplicated prefixes")
    if comparison.model_sha256.nunique() != 4:
        raise RuntimeError("Expected four distinct full-horizon development models")
    selected = best(comparison)
    changed = {m: selected[m][0] != study.INCUMBENT_IDS[m] for m in ("event", "strom")}
    receipts = {}
    for _, row in comparison.iterrows():
        source = parent.OUT if row.origin == "parent_incumbent" else study.OUT
        receipts[row.point_id] = parent.text_hash(parent.paths(study.BY_ID[row.point_id], "development", 7000, study.FINAL, source)["receipt"])
    frozen = {
        "contract_sha256": parent.digest(science), "selection_metric": "final_validation_ce", "horizon": study.FINAL,
        "selected": selected, "configs": {m: asdict(parent.config_for(study.BY_ID[k[0]])) for m, k in selected.items()},
        "parent_selected": study.INCUMBENT_IDS, "changed": changed, "receipt_sha256": receipts,
        "promotion_sha256": parent.text_hash(study.OUT / "promotion.json"), "official_test_evaluations": 0,
        "fresh_matched_confirmation_required": any(changed.values()), "confirmation_launched": False,
        "automatic_further_boundary_extension": False,
    }
    if frozen != json.loads((study.OUT / "frozen_selection.json").read_text()):
        raise RuntimeError("Independent final validation selection differs")
    changes = []
    for method in ("event", "strom"):
        a, b = old[old.method == method].iloc[0], final[final.method == method].iloc[0]
        changes.append({
            "method": method, "incumbent_id": a.point_id, "new_finalist_id": b.point_id,
            "new_finalist_state": b.state, "incumbent_validation_ce": a.final_validation_ce,
            "new_validation_ce": b.final_validation_ce, "validation_ce_difference": b.final_validation_ce - a.final_validation_ce,
            "incumbent_validation_accuracy": a.final_validation_accuracy, "new_validation_accuracy": b.final_validation_accuracy,
            "new_over_incumbent_unicast_ratio": b.unicast_total_bits / a.unicast_total_bits,
            "selected_id": selected[method][0], "changed": changed[method],
        })
    changes = pd.DataFrame(changes)
    compare_saved("boundary_summary.csv", changes)
    expected_audit = {
        "complete_new_stage_receipts": 8, "failed_new_stage_receipts": 0, "reused_parent_incumbents": 2,
        "verified_new_archived_round_rows": 7200, "verified_new_archived_evaluated_models": 128,
        "verified_exact_screen_prefixes": 2, "max_unique_additional_training_rounds": 6000,
        "max_unique_new_evaluation_points": 106, "parent_sources_and_development_evidence_verified": True,
        "official_test_images_or_metrics_read": False, "all_new_candidate_outcomes_retained": True,
        "selection_sha256": parent.digest(frozen), "automatic_confirmation": False,
        "fresh_matched_confirmation_required": any(changed.values()),
        "scope": "single_split_validation_boundary_check_not_predictive_confirmation",
    }
    if expected_audit != json.loads((study.OUT / "analysis_audit.json").read_text()):
        raise RuntimeError("Saved completion/coverage audit differs")
    endpoint = parent.add_regularization_columns(comparison, parent.BASE.regularization)
    endpoint["validation_accuracy_percent"] = 100 * endpoint.final_validation_accuracy
    for field in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        endpoint[field + "_Gbit"] = endpoint[field + "_bits"] / 1e9
    effects = changes.copy()
    effects["validation_accuracy_difference_pp"] = 100 * (effects.new_validation_accuracy - effects.incumbent_validation_accuracy)
    effects["new_traffic_increase_percent"] = 100 * (effects.new_over_incumbent_unicast_ratio - 1)
    for name, frame in (("analysis_endpoints.csv", endpoint), ("analysis_boundary_effects.csv", effects)):
        parent.atomic_csv(frame, study.OUT / name)
    if originals != {n: evidence_hash(study.OUT / n) for n in ORIGINAL}:
        raise RuntimeError("Original boundary selection/analysis changed")
    if hashes != {n: evidence_hash(study.OUT / n) for n in hashes}:
        raise RuntimeError("Original boundary stage evidence changed")
    result_commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(study.OUT / "protocol.json")], cwd=REPO, text=True).strip()
    parent.atomic_json({
        "results_commit": result_commit, "contract_sha256": parent.digest(science),
        "audit_source_sha256": parent.text_hash(Path(__file__)),
        "helper_sha256": {"experiments/resnet14_transfer_audit.py": parent.text_hash(REPO / "experiments/resnet14_transfer_audit.py")},
        "new_complete_stage_receipts_verified": 8, "reused_parent_incumbents_verified": 2,
        "new_archived_round_rows_verified": 7200, "new_sampled_model_fingerprints_verified": 128,
        "exact_600_to_1800_prefixes_verified": 2, "unique_additional_training_rounds": 6000,
        "unique_new_evaluation_points": 106, "distinct_full_horizon_development_models": 4,
        "original_analysis_csvs_reproduced": 4, "original_analysis_sha256": originals,
        "original_stage_evidence_sha256": hashes, "original_evidence_preserved": True,
        "independent_traffic_prefix_and_endpoint_checks": True, "independent_validation_rankings_verified": True,
        "windows_posix_dependency_key_separators_normalized_for_comparison_only": True,
        "recorded_contract_and_receipt_identities_preserved": True, "selected": selected, "changed": changed,
        "official_test_images_or_metrics_read": False, "new_confirmation_pairs": 0,
        "limitations": ["one development split and initialization", "one new finalist per family after 600-round screening",
                        "bounded local sensitivity, no global optimum or new predictive confirmation",
                        "saved evidence and hashes only; no packet/model arrays or training replay"],
    }, study.OUT / "analysis_independent_audit.json")
    print("Verified eight new stages, two reused incumbents, 7200 archived rounds, 128 sampled models and two exact prefixes.")
    print(effects.to_string(index=False))
    print("Selections unchanged:", not any(changed.values()))


if __name__ == "__main__":
    main()
