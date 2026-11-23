"""Audit the completed ResNet transfer from saved evidence without training.

python experiments/resnet14_transfer_audit.py
git add experiments/results/resnet14_transfer_v1
git commit -m "Refresh ResNet14 transfer evidence audit"
git push origin main
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"experiments"))
import resnet14_transfer as study
from event_strom_fresh_confirmation_audit import evidence_hash
from strom_codec_ten_seed_audit import independent_interval
import numpy as np
import pandas as pd

ORIGINAL=("screening.csv","promotion.json","finalists.csv","frozen_selection.json","confirmation_completion.json",
          "confirmation_runs.csv","confirmation_history.csv","paired_per_seed.csv","paired_summary.csv",
          "traffic_to_target.csv","paired_target_summary.csv","analysis_audit.json","confirmation.png","confirmation.pdf")
ANALYSIS_CSV=("screening.csv","finalists.csv","confirmation_runs.csv","confirmation_history.csv",
              "paired_per_seed.csv","paired_summary.csv","traffic_to_target.csv","paired_target_summary.csv")


def compare_saved(name,frame):
    saved=pd.read_csv(study.OUT/name,float_precision="round_trip",dtype={"model_sha256":str})
    if "error" in saved:saved["error"]=saved["error"].fillna("")
    tol=1e-10 if name in ("paired_summary.csv","paired_target_summary.csv") else 1e-12
    pd.testing.assert_frame_equal(saved.reset_index(drop=True),frame.reset_index(drop=True),check_dtype=False,
                                  check_exact=False,rtol=tol,atol=tol)


def independent_traffic(table,*,clients=10,dimension=175258):
    rounds=table["rounds"];initial=32*dimension+64
    if (not rounds.active_clients.between(0,clients).all()
            or not np.array_equal(rounds.active_clients,[sum(json.loads(v).values()) for v in rounds.uplink_modes])
            or not np.array_equal(rounds.legacy_uplink_bits,((dimension-1).bit_length()+1)*rounds.coordinate_events+64*rounds.active_clients)
            or not np.array_equal(rounds.legacy_server_bits,rounds.legacy_uplink_bits)
            or not np.array_equal(rounds.ordered_downlink_bits,128+8*((dimension+2*rounds.coordinate_events+7)//8))
            or not np.array_equal(rounds.exact_downlink_bits,np.minimum(rounds.aggregate_exact_downlink_bits,rounds.ordered_downlink_bits))):
        raise RuntimeError("ResNet pulse frame metadata/length selection differs")
    variants=(("legacy","legacy_uplink_bits","legacy_server_bits",True),
              ("uplink_codec_replay","codec_uplink_bits","codec_replay_bits",True),
              ("aggregate_downlink","legacy_uplink_bits","exact_downlink_bits",False),
              ("both_codecs","codec_uplink_bits","exact_downlink_bits",False))
    for variant,up,down,fallback in variants:
        values=rounds[[up,down]].to_numpy()
        if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
            raise RuntimeError("Invalid ResNet bit counters")
        upload=rounds[up].cumsum().to_numpy()
        stream=(np.minimum(rounds[down],initial) if fallback else rounds[down]).cumsum().to_numpy()
        unicast=clients*(initial+32*rounds["round"].to_numpy()+stream)
        totals={"uplink_bits":upload,"server_stream_bits":stream,"unicast_downlink_bits":unicast,
                "unicast_total_bits":upload+unicast,"broadcast_downlink_bits":initial+stream,
                "broadcast_total_bits":upload+initial+stream}
        history=table["history"][table["history"].variant==variant]
        indices=history["round"].to_numpy(dtype=int)-1
        summary=table["summary"][table["summary"].variant==variant].iloc[0]
        for field,array in totals.items():
            if not np.array_equal(history[field],array[indices]) or int(summary[field])!=int(array[-1]):
                raise RuntimeError("Independent ResNet prefix/endpoint accounting differs: "+field)


def validate():
    science=study.load_contract()
    frozen=study.frozen_selection(science)
    promotion=json.loads((study.OUT/"promotion.json").read_text())
    finalists=tuple(study.BY_ID[k] for m in ("event","strom") for k in promotion["selected"][m])
    screen,final=study.ledger(study.POINTS,study.SCREEN,science),study.ledger(finalists,study.FINAL,science)
    compare_saved("screening.csv",screen);compare_saved("finalists.csv",final)
    development,confirmation={},{}
    for horizon,points in ((study.SCREEN,study.POINTS),(study.FINAL,finalists)):
        for p in points:
            receipt=study.read_receipt(p,"development",study.data_tools.DEV_SEED,horizon,science)
            if receipt is None or receipt["state"]!="complete":raise RuntimeError("Expected complete declared development point")
            table=study.verified_tables(p,"development",study.data_tools.DEV_SEED,horizon,science)
            independent_traffic(table)
            development[p.point_id,horizon]=table
    completed=[]
    for seed in study.data_tools.CONFIRM_SEEDS:
        for method in ("event","strom"):
            p=study.BY_ID[frozen["selected"][method][0]]
            receipt=study.read_receipt(p,"confirmation",seed,study.FINAL,science)
            if receipt is None or receipt["state"]!="complete":raise RuntimeError("All six confirmation results are required")
            if receipt["data_manifest"]!=json.loads((study.OUT/f"confirmation_partition_p{seed}.json").read_text()):
                raise RuntimeError("Confirmation data manifest differs")
            table=study.verified_tables(p,"confirmation",seed,study.FINAL,science)
            independent_traffic(table)
            confirmation[method,seed]=table
            completed.append({"method":method,"partition_seed":seed,"state":"complete"})
    if json.loads((study.OUT/"confirmation_completion.json").read_text())!={"expected":6,"results":completed}:
        raise RuntimeError("Confirmation completion coverage differs")
    counts=(len(development),len(confirmation),sum(len(t["rounds"]) for t in development.values()),
            sum(len(t["evaluation"]) for t in development.values()),sum(len(t["rounds"]) for t in confirmation.values()),
            sum(len(t["evaluation"]) for t in confirmation.values()))
    if counts!=(34,6,25200,454,10800,186):raise RuntimeError("Completed ResNet coverage differs: "+str(counts))
    summary=pd.concat([t["summary"] for t in confirmation.values()],ignore_index=True)
    history=pd.concat([t["history"] for t in confirmation.values()],ignore_index=True)
    runs=study.add_regularization_columns(summary[summary.variant=="both_codecs"].copy(),study.BASE.regularization)
    if runs.model_sha256.nunique()!=6:
        raise RuntimeError("Expected six distinct final confirmation models")
    main_history=history[history.variant=="both_codecs"].copy()
    compare_saved("confirmation_runs.csv",runs);compare_saved("confirmation_history.csv",history)
    a,b=(runs[runs.method==m].set_index("partition_seed").sort_index() for m in ("event","strom"))
    definitions=(("accuracy_difference_pp","final_test_accuracy",False,100),("test_ce_difference","final_test_ce",False,1),
                 ("unicast_ratio","unicast_total_bits",True,1),("upload_ratio","uplink_bits",True,1),
                 ("download_ratio","unicast_downlink_bits",True,1),("broadcast_ratio","broadcast_total_bits",True,1))
    pairs,effects=[],[]
    for metric,field,ratio,scale in definitions:
        values=a[field]/b[field] if ratio else scale*(a[field]-b[field])
        pairs.extend({"partition_seed":int(s),"metric":metric,"value":float(v)} for s,v in values.items())
        estimate,low,high=independent_interval(values,ratio)
        effects.append({"metric":metric,"n":len(values),"estimate":estimate,"ci95_low":low,"ci95_high":high,
                        "direction":"Event_over_Strom" if ratio else "Event_minus_Strom"})
    pairs,effects=pd.DataFrame(pairs),pd.DataFrame(effects)
    compare_saved("paired_per_seed.csv",pairs);compare_saved("paired_summary.csv",effects)
    crossings=[]
    for (method,seed),table in sorted(confirmation.items()):
        curve=table["history"][table["history"].variant=="both_codecs"]
        for target in study.TARGETS:
            hits=table["evaluation"][table["evaluation"].test_accuracy>=target]
            row={"comparison_id":method,"partition_seed":seed,"target_accuracy":target,"reached":not hits.empty,
                 "first_sampled_round":np.nan,"censor_round":np.nan if not hits.empty else study.FINAL,
                 "upload_bits":np.nan,"unicast_total_bits":np.nan,"broadcast_total_bits":np.nan}
            if not hits.empty:
                first=hits.iloc[0];cost=curve[curve["round"]==int(first["round"])].iloc[0]
                row.update(first_sampled_round=int(first["round"]),upload_bits=int(cost.uplink_bits),
                           unicast_total_bits=int(cost.unicast_total_bits),broadcast_total_bits=int(cost.broadcast_total_bits))
            crossings.append(row)
    targets=pd.DataFrame(crossings);compare_saved("traffic_to_target.csv",targets)
    target_intervals=[]
    for target,frame in targets.groupby("target_accuracy"):
        x,y=(frame[frame.comparison_id==m].set_index("partition_seed").sort_index() for m in ("event","strom"))
        common=x.reached&y.reached;ratios=x.loc[common,"unicast_total_bits"]/y.loc[common,"unicast_total_bits"]
        estimate,low,high=independent_interval(ratios,True)
        target_intervals.append({"target_accuracy":target,"n_total":3,"n_event_reached":int(x.reached.sum()),
                                 "n_strom_reached":int(y.reached.sum()),"n_common_hits":int(common.sum()),"n":len(ratios),
                                 "estimate":estimate,"ci95_low":low,"ci95_high":high})
    target_intervals=pd.DataFrame(target_intervals);compare_saved("paired_target_summary.csv",target_intervals)
    if json.loads((study.OUT/"analysis_audit.json").read_text())!={"new_confirmation_runs":6,"seed_pairs":3,"rounds_verified":10800,
            "sampled_model_fingerprints":186,"frozen_selection_sha256":study.digest(frozen),"test_based_selection_or_stopping":False,
            "all_pairs_retained":True,"scope":"three_pair_initial_architecture_transfer_not_ten_seed_confirmation"}:
        raise RuntimeError("Original analysis audit differs")
    return science,frozen,development,confirmation,runs,main_history,targets,effects,target_intervals


def add_savings(frame):
    result=frame.copy()
    for i,row in result.iterrows():
        if "metric" not in result or row.metric.endswith("_ratio"):
            result.loc[i,"Event_saving_percent"]=100*(1-row.estimate)
            result.loc[i,"Event_saving_ci95_low"]=100*(1-row.ci95_high)
            result.loc[i,"Event_saving_ci95_high"]=100*(1-row.ci95_low)
    return result


def main():
    originals={n:evidence_hash(study.OUT/n) for n in ORIGINAL}
    science,frozen,development,confirmation,runs,history,targets,effects,target_intervals=validate()
    endpoint=runs.copy()
    endpoint["test_accuracy_percent"]=100*endpoint.final_test_accuracy
    for field in ("uplink","unicast_downlink","unicast_total","broadcast_total"):
        endpoint[field+"_Mbit"]=endpoint[field+"_bits"]/1e6
    effects=add_savings(effects);target_effects=add_savings(target_intervals)
    for i,row in target_effects.iterrows():
        frame=targets[targets.target_accuracy==row.target_accuracy]
        a,b=(frame[frame.comparison_id==m].set_index("partition_seed").sort_index() for m in ("event","strom"))
        common=a.reached&b.reached
        target_effects.loc[i,"Event_fewer_rounds"]=int((a.loc[common,"first_sampled_round"]<b.loc[common,"first_sampled_round"]).sum())
        target_effects.loc[i,"same_sampled_round"]=int((a.loc[common,"first_sampled_round"]==b.loc[common,"first_sampled_round"]).sum())
        target_effects.loc[i,"Event_fewer_bits"]=int((a.loc[common,"unicast_total_bits"]<b.loc[common,"unicast_total_bits"]).sum())
    diagnostics=[]
    for (method,seed),table in confirmation.items():
        round_frame=table["rounds"];summary=table["summary"].set_index("variant");main=summary.loc["both_codecs"]
        correction=10*int((round_frame.exact_downlink_bits-round_frame.integer_only_downlink_bits).sum())
        diagnostics.append({"method":method,"partition_seed":seed,"coordinate_events":int(round_frame.coordinate_events.sum()),
                            "late_120_firing_fraction":round_frame.tail(120).coordinate_events.mean()/(10*175258),
                            "counts_patch_rounds":int(round_frame.downlink_mode.eq("counts_patch").sum()),
                            "ordered_rounds":int(round_frame.downlink_mode.eq("ordered").sum()),"exact_fidelity_overhead_bits":correction,
                            "exact_fidelity_overhead_percent":100*correction/main.unicast_total_bits,
                            "legacy_unicast_bits":int(summary.loc["legacy","unicast_total_bits"]),"main_unicast_bits":int(main.unicast_total_bits)})
    boundaries={"event_quantum_on_lower_grid_boundary":frozen["configs"]["event"]["jump0"]==min(study.QUANTA),
                "strom_threshold_on_lower_grid_boundary":frozen["configs"]["strom"]["strom_threshold"]==min(study.THRESHOLDS),
                "strom_gain_on_lower_grid_boundary":frozen["configs"]["strom"]["server_gain"]==study.BASE.local_lr*min(study.GAINS),
                "automatic_extension_performed":False,"claim":"best_promoted_finalists_in_declared_bounded_grid_not_global_optima"}
    for name,frame in (("analysis_endpoints.csv",endpoint),("analysis_endpoint_effects.csv",effects),
                       ("analysis_target_effects.csv",target_effects),("analysis_codec_diagnostics.csv",pd.DataFrame(diagnostics))):
        study.atomic_csv(frame,study.OUT/name)
    if originals!={n:evidence_hash(study.OUT/n) for n in ORIGINAL}:raise RuntimeError("Original ResNet analysis changed")
    result_commit=subprocess.check_output(["git","log","-1","--format=%H","--",str(study.OUT/"protocol.json")],cwd=REPO,text=True).strip()
    study.atomic_json({"results_commit":result_commit,"contract_sha256":study.digest(science),"audit_source_sha256":study.text_hash(Path(__file__)),
                       "helper_sha256":{p:study.text_hash(REPO/p) for p in ("experiments/strom_codec_ten_seed_audit.py","experiments/event_strom_fresh_confirmation_audit.py")},
                       "complete_receipts_verified":40,"development_receipts":34,"confirmation_receipts":6,
                       "archived_round_records_verified":36000,"archived_evaluated_model_fingerprints":640,
                       "exact_600_to_1800_prefixes_verified":4,"unique_scheduled_training_rounds":33600,
                       "distinct_confirmation_final_models":int(runs.model_sha256.nunique()),
                       "unique_scheduled_evaluations":596,"original_analysis_csvs_reproduced":8,
                       "original_analysis_sha256":originals,"original_analysis_preserved":True,
                       "independent_prefix_and_endpoint_accounting":True,"independent_paired_intervals":True,
                       "first_crossings_rebuilt_from_raw_evaluations":True,"all_three_confirmation_pairs_retained":True,
                       "boundary_diagnostics":boundaries,"official_test_used_for_selection":False,
                       "runtime":science["execution"],"limits":["three pairs on one fixed previously inspected CIFAR10 test set",
                         "selected configurations touch search boundaries; no optimal-tuning claim",
                         "Torch backend and architecture change together relative to compact CNN",
                         "no raw images, packet bytes, NPZ model/state arrays or runtime replay loaded",
                         "secondary intervals are descriptive and unadjusted; no field-wide SOTA or causal mechanism claim"]},
                      study.OUT/"analysis_independent_audit.json")
    print("Verified 40 complete receipts, 36000 archived rounds, 640 sampled model fingerprints, four exact prefixes and eight original analysis CSVs.")
    print(effects.to_string(index=False));print(target_effects.to_string(index=False));print(boundaries)


if __name__=="__main__":main()
