"""Verify saved Event development evidence without overwriting original analysis.

python experiments/event_codec_development_audit.py
git add experiments/results/event_codec_development_v1
git commit -m "Refresh Event threshold quantum development analysis"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import event_codec_development as study
from strom_codec_development_audit import independent_counts
import numpy as np
import pandas as pd

TARGETS = (.45, .50, .55)
ORIGINAL_ANALYSIS = ("screening.csv", "finalists.csv", "controls.csv", "analysis_audit.json",
                     "development_frontier.png", "development_frontier.pdf")


def compare_saved(name, frame):
    saved = pd.read_csv(study.OUT / name, float_precision="round_trip")
    if "error" in saved:
        saved["error"] = saved["error"].fillna("")
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), frame.reset_index(drop=True), check_dtype=False, check_exact=True)


def validate():
    science = study.load_contract()
    split = study.load_split(science)
    screened, decision = study.verify_screening(science, split)
    finals, frozen = study.final_decision(science, split)
    if json.loads((study.OUT / "frozen_selection.json").read_text(encoding="utf-8")) != frozen:
        raise RuntimeError("Frozen Event final selection differs")
    controls = pd.concat([study.ledger((study.STROM_ID, study.CURRENT.point_id), h, science, split)
                          for h in (study.SCREEN, study.FINAL)], ignore_index=True)
    for name, table in (("screening.csv", screened), ("finalists.csv", finals), ("controls.csv", controls)):
        compare_saved(name, table)
    tables = {}
    for horizon, ids in ((study.SCREEN, (study.STROM_ID, *study.POINT_BY_ID)),
                          (study.FINAL, (study.STROM_ID, *decision["promoted_Event_ids"]))):
        for key in ids:
            receipt = study.read_receipt(key, horizon, science, split)
            expected = "reused" if key in (study.STROM_ID, study.CURRENT.point_id) else "complete"
            if receipt is None or receipt["state"] != expected:
                raise RuntimeError(f"Missing expected {expected} result: {key}/{horizon}")
            tables[key, horizon] = study.verified_tables(key, horizon, science, split)
            independent_counts(tables[key, horizon], clients=10, dimension=20570)
    prefixes = []
    for key, horizon in tables:
        if horizon != study.FINAL:
            continue
        before, after = tables[key, study.SCREEN], tables[key, study.FINAL]
        for kind in ("rounds", "evaluation", "history"):
            fields = [c for c in before[kind] if c not in ("stage_horizon", "codec_audit_seconds")]
            prefix = after[kind][after[kind]["round"] <= study.SCREEN]
            pd.testing.assert_frame_equal(before[kind][fields].reset_index(drop=True), prefix[fields].reset_index(drop=True),
                                          check_dtype=False, check_exact=True)
        prefixes.append(key)
    counts = (len(tables), sum(len(t["rounds"]) for t in tables.values()),
              sum(len(t["evaluation"]) for t in tables.values()))
    if counts != (19, 15000, 269) or len(prefixes) != 3:
        raise RuntimeError(f"Completed study coverage differs: {counts}, prefixes={prefixes}")
    return science, tables, frozen, prefixes


def summarize(tables, frozen):
    endpoints, targets, codec, curves = [], [], [], []
    winner = frozen["selection"]["quality"]["point_id"]
    for (key, horizon), table in tables.items():
        config = study.config_for_id(key)
        summary = table["summary"].set_index("variant")
        endpoint = summary.loc["both_codecs"].to_dict()
        penalty = endpoint["final_train_objective"] - endpoint["final_train_ce"]
        endpoint.update(point_id=key, horizon=horizon, threshold=config.strom_threshold if key==study.STROM_ID else config.threshold,
                        jump0=np.nan if key==study.STROM_ID else config.jump0, final_train_l2_penalty=penalty,
                        inferred_parameter_l2_norm=np.sqrt(2*penalty/config.regularization),
                        validation_accuracy_percent=100*endpoint["final_validation_accuracy"],
                        validation_train_ce_gap=endpoint["final_validation_ce"]-endpoint["final_train_ce"])
        endpoints.append(endpoint)
        history = table["history"][table["history"].variant == "both_codecs"].copy()
        history["horizon"] = horizon
        if horizon == study.FINAL:
            curves.append(history)
        for target in TARGETS:
            hits = history[history.validation_accuracy >= target]
            row = {"point_id":key,"horizon":horizon,"target_validation_accuracy":target,"reached":not hits.empty,
                   "first_saved_round":np.nan,"accuracy_at_crossing":np.nan,"unicast_total_bits":np.nan,
                   "censor_round":horizon if hits.empty else np.nan}
            if not hits.empty:
                hit = hits.iloc[0]
                row.update(first_saved_round=int(hit["round"]),accuracy_at_crossing=hit.validation_accuracy,
                           unicast_total_bits=int(hit.unicast_total_bits))
            targets.append(row)
        rounds = table["rounds"]
        events = int(rounds.coordinate_events.sum())
        overhead = int(10*(rounds.exact_downlink_bits-rounds.integer_only_downlink_bits).sum())
        codec.append({"point_id":key,"horizon":horizon,"coordinate_events":events,
                      "late_120_firing_fraction":rounds.tail(120).coordinate_events.mean()/(10*20570),
                      "counts_patch_rounds":int(rounds.downlink_mode.eq("counts_patch").sum()),
                      "ordered_fallback_rounds":int(rounds.downlink_mode.eq("ordered").sum()),
                      "exact_fidelity_overhead_bits":overhead,"exact_fidelity_overhead_percent":100*overhead/endpoint["unicast_total_bits"],
                      "canonical_roundoff_max":rounds.canonical_max_abs_error.max(),
                      "legacy_unicast_bits":int(summary.loc["legacy","unicast_total_bits"]),
                      "codec_unicast_reduction_percent":100*(1-endpoint["unicast_total_bits"]/summary.loc["legacy","unicast_total_bits"])})
    endpoints, targets = pd.DataFrame(endpoints), pd.DataFrame(targets)
    contrasts, paired_targets = [], []
    for _, row in endpoints[endpoints.method=="event"].iterrows():
        for reference in (study.CURRENT.point_id,study.STROM_ID):
            if row.point_id==reference:
                continue
            baseline = endpoints[(endpoints.point_id==reference)&(endpoints.horizon==row.horizon)].iloc[0]
            record = {"point_id":row.point_id,"reference":reference,"horizon":row.horizon,
                      "candidate_minus_reference_accuracy_pp":100*(row.final_validation_accuracy-baseline.final_validation_accuracy),
                      "candidate_minus_reference_validation_ce":row.final_validation_ce-baseline.final_validation_ce,
                      "candidate_minus_reference_train_objective":row.final_train_objective-baseline.final_train_objective}
            for field in ("uplink_bits","unicast_downlink_bits","unicast_total_bits","broadcast_total_bits"):
                record["candidate_reduction_percent_"+field] = 100*(1-row[field]/baseline[field])
            contrasts.append(record)
            if row.point_id != winner:
                continue
            for target in TARGETS:
                hit = targets[(targets.point_id==row.point_id)&(targets.horizon==row.horizon)&(targets.target_validation_accuracy==target)].iloc[0]
                ref = targets[(targets.point_id==reference)&(targets.horizon==row.horizon)&(targets.target_validation_accuracy==target)].iloc[0]
                both = bool(hit.reached and ref.reached)
                paired_targets.append({"point_id":row.point_id,"reference":reference,"horizon":row.horizon,
                                       "target_validation_accuracy":target,"candidate_reached":hit.reached,"reference_reached":ref.reached,
                                       "candidate_round":hit.first_saved_round,"reference_round":ref.first_saved_round,
                                       "candidate_accuracy_at_crossing":hit.accuracy_at_crossing,"reference_accuracy_at_crossing":ref.accuracy_at_crossing,
                                       "candidate_unicast_bits":hit.unicast_total_bits,"reference_unicast_bits":ref.unicast_total_bits,
                                       "candidate_reduction_percent":100*(1-hit.unicast_total_bits/ref.unicast_total_bits) if both else np.nan})
    return {"analysis_endpoint_comparison.csv":endpoints,"analysis_endpoint_contrasts.csv":pd.DataFrame(contrasts),
            "analysis_traffic_to_targets.csv":targets,"analysis_target_comparison.csv":pd.DataFrame(paired_targets),
            "analysis_codec_diagnostics.csv":pd.DataFrame(codec),"analysis_finalist_curves.csv":pd.concat(curves,ignore_index=True)}


def plot(curves, winner):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(10.5,6.6))
    for key,label,color in ((study.CURRENT.point_id,"Current Event","#666666"),(winner,"Selected Event","#D55E00"),
                            (study.STROM_ID,"Calibrated Ström","#0072B2")):
        rows = curves[(curves.point_id==key)&(curves["round"]>=60)]
        for axis,xfield,metric,ylabel,scale in (
            (axes[0,0],"round","validation_ce","Validation cross-entropy",1),
            (axes[0,1],"unicast_total_bits","validation_ce","Validation cross-entropy",1),
            (axes[1,0],"round","validation_accuracy","Validation accuracy (%)",100),
            (axes[1,1],"unicast_total_bits","validation_accuracy","Validation accuracy (%)",100)):
            x = rows[xfield]/(1e6 if xfield.endswith("bits") else 1)
            axis.plot(x,scale*rows[metric],marker="o",markersize=2.8,linewidth=1.3,color=color,label=label)
            axis.set(xlabel="Total unicast traffic (Mbit)" if xfield.endswith("bits") else "Communication rounds",ylabel=ylabel)
            axis.grid(alpha=.2)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    fig.suptitle("Event threshold/quantum development\nOne validation seed; samples from round 60",fontsize=12)
    fig.tight_layout()
    fig.savefig(study.OUT/"analysis_trajectories.png",dpi=180)
    fig.savefig(study.OUT/"analysis_trajectories.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def main():
    original = {name: study.full.development.file_hash(study.OUT/name) for name in ORIGINAL_ANALYSIS}
    science,tables,frozen,prefixes = validate()
    outputs = summarize(tables,frozen)
    for name,frame in outputs.items():
        study.atomic_csv(frame,study.OUT/name)
    plot(outputs["analysis_finalist_curves.csv"],frozen["selection"]["quality"]["point_id"])
    if original != {name: study.full.development.file_hash(study.OUT/name) for name in ORIGINAL_ANALYSIS}:
        raise RuntimeError("Original analysis unexpectedly modified")
    result_commit = subprocess.check_output(["git","log","-1","--format=%H","--",str(study.OUT/"protocol.json")],cwd=REPO,text=True).strip()
    winner = frozen["selection"]["quality"]
    study.atomic_json({"results_commit":result_commit,"contract_sha256":study.digest(science),
                       "audit_source_sha256":study.text_hash(Path(__file__)),"independent_counter_helper_sha256":study.text_hash(REPO/"experiments/strom_codec_development_audit.py"),
                       "new_completion_receipts_verified":15,"reused_reference_stages_verified":4,
                       "archived_round_records":15000,"archived_evaluated_model_fingerprints":269,
                       "unique_trajectory_rounds":13200,"unique_evaluated_models":236,
                       "exact_600_round_prefixes":prefixes,"reproduced_original_ledger_csvs":3,
                       "original_analysis_preserved":True,"original_analysis_raw_byte_sha256":original,
                       "independent_prefix_and_endpoint_traffic_reconstruction":True,"frozen_validation_selection_verified":True,
                       "official_test_evaluations":0,"confirmation_included":False,
                       "winner_initial_quantum_on_upper_grid_boundary":winner["jump0"]==max(study.QUANTA),
                       "limits":["one development partition/SGD seed with validation-based selection; no confidence intervals",
                                 "only screen-promoted Event candidates evaluated to 1800; no full-grid optimum claim",
                                 "accuracy and CE can favour different algorithms; report both",
                                 "first saved accuracy crossings are retrospective diagnostics, without interpolation",
                                 "raw images, packet bytes, NPZ model weights and Event/residual arrays are not reloaded",
                                 "additional development after inspected test confirmation; fresh paired confirmation required"]},
                      study.OUT/"analysis_independent_audit.json")
    print("Verified 15 new completion receipts, four reused stages, 15000 archived rounds and 269 sampled model fingerprints.")
    print("All three full trajectories reproduce their 600-round prefixes exactly; original analysis preserved.")
    print(outputs["analysis_endpoint_comparison.csv"].query("horizon==1800")[["point_id","validation_accuracy_percent","final_validation_ce","unicast_total_bits"]].to_string(index=False))


if __name__=="__main__":
    main()
