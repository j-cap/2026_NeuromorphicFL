"""Bounded ResNet-14 Event/Ström tuning followed by three paired confirmations.

python experiments/resnet14_transfer.py verify
python experiments/resnet14_transfer.py preflight --device auto
python experiments/resnet14_transfer.py benchmark --device auto
python experiments/resnet14_transfer.py run --device auto
git add experiments/results/resnet14_transfer_v1
git commit -m "Add bounded ResNet14 pulse transfer results"
git push origin main
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"experiments"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import resnet14_transfer_data as data_tools
import event_codec_full_horizon as full
import strom_codec_host as strom
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.confirmation_metrics import interval, target_crossings, add_regularization_columns

OUT = REPO/"experiments/results/resnet14_transfer_v1"
DATA = REPO/"data/cifar-10"
SCREEN, FINAL, STRIDE = 600, 1800, 60
TARGETS = (.45, .50, .55)
THRESHOLDS, QUANTA, GAINS = (.005, .01, .0125, .025, .05), (.0025, .005, .01), (2., 10., 20.)
BASE = FinalBaselineConfig(local_steps=5, local_lr=.05, batch_size=32, regularization=.0005, rounds=FINAL,
                           eval_stride=STRIDE, rho=.999, threshold=.05, jump0=.01, jump_scale=100., jump_exponent=.2,
                           init_scale=1., server_gain=1.)
atomic_json, atomic_csv, digest, text_hash = full.atomic_json, full.atomic_csv, full.digest, full.text_hash
from strom_codec_development import immutable_json


@dataclass(frozen=True)
class Point:
    point_id: str
    method: str
    threshold: float
    quantum_or_gain: float


def point(method, threshold, value):
    label = lambda x: f"{x:g}".replace(".", "p")
    return Point(f"{method}_t{label(threshold)}_{'q' if method=='event' else 'g'}{label(value)}", method, threshold, value)


POINTS = tuple(point(m, t, v) for m, values in (("event", QUANTA), ("strom", GAINS)) for t in THRESHOLDS for v in values)
BY_ID = {p.point_id: p for p in POINTS}


def config_for(p):
    if p.method == "event":
        return replace(BASE, threshold=p.threshold, jump0=p.quantum_or_gain)
    if p.method == "strom":
        return replace(BASE, strom_threshold=p.threshold, server_gain=BASE.local_lr*p.quantum_or_gain)
    raise ValueError("Unknown pulse method")


def train_seed(seed):
    return 90000+seed


def protocol_spec():
    prior=[*range(2500,5200,100),*range(6000,7000,100)]
    proposed=(data_tools.DEV_SEED,*data_tools.CONFIRM_SEEDS)
    if len(set(proposed))!=4 or set(proposed).intersection(prior):
        raise RuntimeError("New development/confirmation seeds must be distinct from the recorded prior seed grid")
    return {"campaign": "resnet14_transfer_v1", "architecture": "CIFAR_ResNet14_GroupNorm_Torch", "dimension": 175258,
            "n_clients": 10, "regime": "strong_55pct_dominant", "screen_rounds": SCREEN, "final_rounds": FINAL,
            "development_partition_seed": data_tools.DEV_SEED, "split_seed": data_tools.SPLIT_SEED,
            "confirmation_partition_seeds": list(data_tools.CONFIRM_SEEDS),
            "confirmation_training_seeds": [train_seed(s) for s in data_tools.CONFIRM_SEEDS],
            "development_training_examples": 45000, "validation_examples": 5000, "confirmation_training_examples": 50000,
            "official_test_examples": 10000, "training_diagnostic_examples": 5000, "eval_stride": STRIDE,
            "points": [asdict(p) for p in POINTS], "configs": {p.point_id: asdict(config_for(p)) for p in POINTS},
            "screen_candidates_per_method": 15, "max_promoted_per_method": 2, "automatic_boundary_extension": False,
            "grid_anchors": "compact_selected_theta0p05_q0p01_Strom_gamma0p1_and_prior_ResNet_theta0p0125_q0p005_Strom_tau0p01_gamma1",
            "selection_metric": "final_validation_ce", "selection_ties": ["unicast_total_bits", "point_id"],
            "final_selection_scope": "best_validation_CE_among_screen_promoted_finalists",
            "development_official_test_images_loaded": False, "development_checkpoint_reused_for_confirmation": False,
            "new_confirmation_runs": 6, "max_training_rounds": 33600, "max_stage_receipts": 40,
            "confirmation_runs_after_frozen_selection_even_if_Event_loses": True,
            "numerical_failure": "explicit_terminal_receipt_in_ledger_never_retried_with_changed_settings",
            "promotion": "best_two_finite_complete_candidates_per_method_or_one_if_only_one_finite",
            "all_candidates_failed": "stop_with_explicit_error_no_fallback_winner",
            "accounting": "real_shared_position_and_exact_aggregate_frames_conservative_unicast_primary",
            "checkpoint_identity": "1800_round_config_from_start_exact_600_to_1800_continuation",
            "confirmation_endpoint": "round_1800_not_best_test_checkpoint", "accuracy_targets": list(TARGETS),
            "target_rule": "first_saved_crossing_no_interpolation_censored_if_unreached",
            "intervals": "descriptive_unadjusted_paired_95pct_Student_t_ratios_log_scale_n3_pilot",
            "test_tuning_or_early_stopping": False, "equivalence_or_noninferiority_claim": False,
            "backend_scope": "native_Torch_pulse_arithmetic_checked_against_shared_lossless_codecs_not_cross_backend_bit_parity",
            "seed_freshness_snapshot": {"evidence_commit": "4640f164ccafd84f95364545ed3bb3945a5fc14e",
                                        "prior_recorded_partition_seeds": [*range(2500, 5200, 100), *range(6000, 7000, 100)], "overlap": []}}


def science_spec():
    parent_file = REPO/"experiments/results/core_fresh_comparison_v1/protocol.json"
    parent = json.loads(parent_file.read_text())
    sources = dict(parent["source_hashes"])
    files = ("experiments/resnet14_transfer.py", "experiments/resnet14_transfer_host.py", "experiments/resnet14_transfer_data.py",
             "experiments/p13_cifar10_resnet14_torch.py", "experiments/p13_cifar10_resnet14.py", "experiments/strom_codec_development.py")
    for path in files:
        sources[path] = text_hash(REPO/path)
    for path, sha in sources.items():
        if text_hash(REPO/path) != sha:
            raise RuntimeError("Frozen inherited source changed: "+path)
    return {"protocol": protocol_spec(), "source_hashes": sources, "data_sha256": parent["data_sha256"],
            "core_protocol_sha256": text_hash(parent_file)}


def load_contract(out=OUT):
    science = json.loads((out/"protocol.json").read_text())
    if {k:v for k,v in science.items() if k != "execution"} != science_spec():
        raise RuntimeError("ResNet protocol, configs or sources changed")
    table = pd.read_csv(out/"split_indices.csv")
    _, _, _, rebuilt, manifest = data_tools.allocation(table["class"].to_numpy(), seed=data_tools.DEV_SEED, validation=True)
    if text_hash(out/"split_indices.csv") != manifest["index_csv_sha256"] or json.loads((out/"split_manifest.json").read_text()) != manifest:
        raise RuntimeError("Development split changed")
    return science


def prepare(device_name, threads, eval_batch_size, out=OUT):
    import resnet14_transfer_host as host
    device = host.configure(device_name, threads)
    science = {**science_spec(), "execution": {"runtime": host.runtime(device), "eval_batch_size": eval_batch_size}}
    if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text()) != science:
        raise RuntimeError("Existing campaign sources/runtime/settings differ; preserve the recorded campaign")
    root = data_tools.download_cifar10(DATA)
    if {name: full.development.file_hash(root/name) for name in science["data_sha256"]} != science["data_sha256"]:
        raise RuntimeError("Raw CIFAR batch fingerprints differ")
    images, labels, _ = data_tools.load_arrays(DATA, include_test=False)
    _, _, _, table, manifest = data_tools.allocation(labels, seed=data_tools.DEV_SEED, validation=True)
    immutable_json(science, out/"protocol.json")
    if (out/"split_indices.csv").exists():
        if text_hash(out/"split_indices.csv") != manifest["index_csv_sha256"]:
            raise RuntimeError("Existing development indices differ")
    else:
        atomic_csv(table, out/"split_indices.csv")
    immutable_json(manifest, out/"split_manifest.json")
    load_contract(out)
    return science, device, images, labels


def stem(p, phase, seed, horizon):
    return f"{p.point_id}_development_r{horizon}" if phase == "development" else f"{p.method}_p{seed}_confirmation_r{horizon}"


def paths(p, phase, seed, horizon, out=OUT):
    name = stem(p, phase, seed, horizon)
    return {kind: out/f"{name}_{kind}.{'json' if kind=='receipt' else 'csv'}" for kind in ("rounds", "evaluation", "summary", "history", "receipt")}


def checkpoint_path(p, phase, seed, out=OUT):
    return out/"checkpoints"/(f"{p.point_id}_development.npz" if phase == "development" else f"{p.method}_p{seed}_confirmation.npz")


def provenance(p, phase, seed, science):
    return {"point_id": p.point_id, "method": p.method, "phase": phase, "partition_seed": seed,
            "train_seed": train_seed(seed), "config_sha256": digest(asdict(config_for(p))), "contract_sha256": digest(science)}


def metadata(p, phase, seed, horizon, science):
    return {**provenance(p, phase, seed, science), "stage_horizon": horizon}


def read_receipt(p, phase, seed, horizon, science, out=OUT):
    files = paths(p, phase, seed, horizon, out)
    if not files["receipt"].exists():
        return None
    receipt = json.loads(files["receipt"].read_text())
    if receipt.get("identity") != metadata(p, phase, seed, horizon, science):
        raise RuntimeError("ResNet receipt identity differs")
    if receipt["state"] == "complete":
        for k in ("rounds", "evaluation", "summary", "history"):
            if text_hash(files[k]) != receipt["sha256"][k]:
                raise RuntimeError("ResNet evidence changed")
    elif receipt["state"] != "numerical_failure":
        raise RuntimeError("Unexpected ResNet completion state")
    return receipt


def trajectory_tables(result, *, p, clients, dimension):
    account = strom.traffic_summary if p.method == "strom" else full.traffic_summary
    rounds, evaluation = result["rounds"], result["evaluation"]
    summary = account(rounds, n_clients=clients, dimension=dimension)
    history = pd.concat([account(rounds[rounds["round"] <= int(row["round"])], n_clients=clients, dimension=dimension)
                         .assign(**row.to_dict()) for _,row in evaluation.iterrows()], ignore_index=True)
    for field, value in evaluation.iloc[-1].items():
        if field not in ("round", "model_sha256"):
            summary["final_"+field] = value
    summary["model_sha256"] = evaluation.iloc[-1].model_sha256
    return {"rounds": rounds, "evaluation": evaluation, "summary": summary, "history": history}


def verified_tables(p, phase, seed, horizon, science, out=OUT):
    receipt = read_receipt(p, phase, seed, horizon, science, out)
    if receipt is None or receipt["state"] != "complete":
        raise RuntimeError("Incomplete ResNet point")
    tables = {k: pd.read_csv(v, float_precision="round_trip", dtype={"model_sha256":str}) for k,v in paths(p,phase,seed,horizon,out).items() if k != "receipt"}
    for frame in tables.values():
        for k,v in metadata(p,phase,seed,horizon,science).items():
            if not frame[k].eq(v).all():
                raise RuntimeError("ResNet table metadata differs")
    rounds, evaluation = tables["rounds"], tables["evaluation"]
    config = config_for(p)
    expected = sorted({1,horizon,*range(config.eval_stride,horizon+1,config.eval_stride)})
    prefix = "validation" if phase == "development" else "test"
    fields = ("train_objective", "train_ce", prefix+"_ce", prefix+"_accuracy", prefix+"_macro_accuracy", prefix+"_worst_class_accuracy")
    if (rounds["round"].tolist() != list(range(1,horizon+1)) or not rounds.method.eq(p.method).all()
            or evaluation["round"].tolist() != expected or not np.isfinite(evaluation[list(fields)]).all().all()
            or not evaluation[[prefix+"_accuracy",prefix+"_macro_accuracy",prefix+"_worst_class_accuracy"]].apply(lambda c:c.between(0,1)).all().all()
            or not np.array_equal(evaluation.model_sha256,rounds.set_index("round").loc[evaluation["round"],"model_sha256"])):
        raise RuntimeError("ResNet round/evaluation/model coverage differs")
    rebuilt = trajectory_tables({"rounds":rounds,"evaluation":evaluation[["round",*fields,"model_sha256"]]},p=p,
                                 clients=science["protocol"]["n_clients"],dimension=science["protocol"]["dimension"])
    for kind in ("summary","history"):
        for field in rebuilt[kind]:
            a,b = tables[kind][field],rebuilt[kind][field]
            exact = field.endswith("bits") or field in ("round","rounds","variant","model_sha256","roundtrip_verified")
            if not (np.array_equal(a,b) if exact else np.allclose(a,b,rtol=1e-12,atol=1e-12)):
                raise RuntimeError("Reconstructed ResNet traffic/metrics differ")
    return tables


def train_point(p, phase, seed, horizon, science, federation, device, out=OUT):
    import resnet14_transfer_host as host
    files = paths(p,phase,seed,horizon,out)
    with full.development.point_lock(out/"locks"/(files["receipt"].stem+".lock")):
        receipt = read_receipt(p,phase,seed,horizon,science,out)
        if receipt is not None:
            print("skip",files["receipt"].stem,receipt["state"],flush=True)
            return
        if phase=="development" and horizon==FINAL and not checkpoint_path(p,phase,seed,out).exists():
            raise RuntimeError("Finalist continuation requires its local 600-round NPZ checkpoint")
        if (federation.scope != ("validation" if phase=="development" else "test")
                or host.runtime(device) != science["execution"]["runtime"]):
            raise RuntimeError("ResNet data scope/runtime differs")
        def progress(row,evaluation):
            if row["round"]==1 or row["round"]%STRIDE==0:
                prefix = "validation" if phase=="development" else "test"
                print(f"{p.point_id} {phase} p{seed}: {row['round']}/{horizon}, {prefix}_acc={evaluation[prefix+'_accuracy']:.4f}, "
                      f"CE={evaluation[prefix+'_ce']:.6f}",flush=True)
        try:
            result = host.run(federation=federation,method=p.method,config=config_for(p),seed=train_seed(seed),device=device,
                              checkpoint=BaselineCheckpoint(checkpoint_path(p,phase,seed,out),provenance(p,phase,seed,science)),
                              eval_batch_size=science["execution"]["eval_batch_size"],stop_after=horizon,callback=progress)
        except FloatingPointError as exc:
            atomic_json({"identity":metadata(p,phase,seed,horizon,science),"state":"numerical_failure","error":str(exc)},files["receipt"])
            print("numerical failure",p.point_id,phase,str(exc),flush=True)
            return
        tables = trajectory_tables(result,p=p,clients=science["protocol"]["n_clients"],dimension=science["protocol"]["dimension"])
        for k,frame in tables.items():
            atomic_csv(frame.assign(**metadata(p,phase,seed,horizon,science)),files[k])
        atomic_json({"identity":metadata(p,phase,seed,horizon,science),"state":"complete",
                     "sha256":{k:text_hash(files[k]) for k in tables},"data_manifest":federation.manifest},files["receipt"])


def ledger(points, horizon, science, out=OUT):
    rows=[]
    for p in points:
        receipt=read_receipt(p,"development",data_tools.DEV_SEED,horizon,science,out)
        if receipt is None:
            raise RuntimeError("Development grid incomplete: "+p.point_id)
        row={"point_id":p.point_id,"method":p.method,"threshold":p.threshold,"quantum_or_gain":p.quantum_or_gain,
             "state":receipt["state"],"error":receipt.get("error","")}
        if receipt["state"]=="complete":
            table=verified_tables(p,"development",data_tools.DEV_SEED,horizon,science,out)["summary"]
            row.update(table[table.variant=="both_codecs"].iloc[0].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def choose(frame, count):
    selected={}
    for method in ("event","strom"):
        rows=frame[(frame.method==method)&(frame.state=="complete")].copy()
        if rows.empty:
            raise RuntimeError("All "+method+" candidates failed; no automatic fallback")
        if not np.isfinite(rows.final_validation_ce).all():
            raise RuntimeError("Nonfinite development selection score")
        selected[method]=rows.sort_values(["final_validation_ce","unicast_total_bits","point_id"]).head(count).point_id.tolist()
    return selected


def decision(frame, count, science, horizon, out=OUT):
    selected=choose(frame,count)
    receipts={p: text_hash(paths(BY_ID[p],"development",data_tools.DEV_SEED,horizon,out)["receipt"]) for p in frame.point_id}
    return {"contract_sha256":digest(science),"selection_metric":"final_validation_ce","horizon":horizon,
            "selected":selected,"receipt_sha256":receipts,"official_test_evaluations":0}


def frozen_selection(science,out=OUT):
    screen=ledger(POINTS,SCREEN,science,out)
    promoted=decision(screen,2,science,SCREEN,out)
    if json.loads((out/"promotion.json").read_text()) != promoted:
        raise RuntimeError("Screen promotion changed")
    points=tuple(BY_ID[k] for method in ("event","strom") for k in promoted["selected"][method])
    finals=ledger(points,FINAL,science,out)
    validate_prefixes(points,science,out)
    frozen={**decision(finals,1,science,FINAL,out),"configs":{m:asdict(config_for(BY_ID[v[0]])) for m,v in choose(finals,1).items()},
            "promotion_sha256":text_hash(out/"promotion.json"),"confirmation_checkpoint_reuse":False}
    if json.loads((out/"frozen_selection.json").read_text()) != frozen:
        raise RuntimeError("Frozen ResNet selection changed")
    return frozen


def validate_prefixes(points,science,out=OUT):
    for p in points:
        receipt=read_receipt(p,"development",data_tools.DEV_SEED,FINAL,science,out)
        if receipt is None or receipt["state"]!="complete":
            continue
        before=verified_tables(p,"development",data_tools.DEV_SEED,SCREEN,science,out)
        after=verified_tables(p,"development",data_tools.DEV_SEED,FINAL,science,out)
        for kind in ("rounds","evaluation","history"):
            fields=[f for f in before[kind] if f not in ("stage_horizon","training_seconds","codec_seconds")]
            prefix=after[kind][after[kind]["round"]<=SCREEN]
            pd.testing.assert_frame_equal(before[kind][fields].reset_index(drop=True),prefix[fields].reset_index(drop=True),
                                          check_dtype=False,check_exact=True)


def run(device_name="auto",threads=4,eval_batch_size=256,out=OUT):
    with full.development.point_lock(out/"locks/controller.lock"):
        science,device,images,labels=prepare(device_name,threads,eval_batch_size,out)
        dev,_=data_tools.federation(images,labels,seed=data_tools.DEV_SEED,validation=True,device=device)
        print("Stage 1: thirty validation-only 600-round screens",flush=True)
        for p in POINTS:
            train_point(p,"development",data_tools.DEV_SEED,SCREEN,science,dev,device,out)
        screen=ledger(POINTS,SCREEN,science,out)
        atomic_csv(screen,out/"screening.csv")
        promoted=decision(screen,2,science,SCREEN,out)
        immutable_json(promoted,out/"promotion.json")
        finalists=tuple(BY_ID[k] for m in ("event","strom") for k in promoted["selected"][m])
        print("Stage 2: exact continuation of at most four finalists to 1800 rounds",flush=True)
        for p in finalists:
            train_point(p,"development",data_tools.DEV_SEED,FINAL,science,dev,device,out)
        final=ledger(finalists,FINAL,science,out)
        atomic_csv(final,out/"finalists.csv")
        validate_prefixes(finalists,science,out)
        choices=choose(final,1)
        frozen={**decision(final,1,science,FINAL,out),"configs":{m:asdict(config_for(BY_ID[v[0]])) for m,v in choices.items()},
                "promotion_sha256":text_hash(out/"promotion.json"),"confirmation_checkpoint_reuse":False}
        immutable_json(frozen,out/"frozen_selection.json")
        frozen_selection(science,out)
        del dev,images,labels
        # This is the first call that parses official test images.
        print("Stage 3: selections frozen; six new full-training confirmation runs",flush=True)
        images,labels,test=data_tools.load_arrays(DATA,include_test=True)
        for seed in data_tools.CONFIRM_SEEDS:
            fed,indices=data_tools.federation(images,labels,seed=seed,validation=False,device=device,test=test)
            immutable_json(fed.manifest,out/f"confirmation_partition_p{seed}.json")
            for method in ("event","strom"):
                train_point(BY_ID[choices[method][0]],"confirmation",seed,FINAL,science,fed,device,out)
            del fed
        analyze(out)


def analyze(out=OUT):
    science=load_contract(out)
    frozen=frozen_selection(science,out)
    summaries,histories,completion=[],[],[]
    verified_rounds,verified_models=0,0
    for seed in data_tools.CONFIRM_SEEDS:
        for method in ("event","strom"):
            p=BY_ID[frozen["selected"][method][0]]
            receipt=read_receipt(p,"confirmation",seed,FINAL,science,out)
            completion.append({"method":method,"partition_seed":seed,"state":receipt["state"] if receipt else "pending"})
            if receipt is not None and receipt["state"]=="complete":
                table=verified_tables(p,"confirmation",seed,FINAL,science,out)
                summaries.append(table["summary"]);histories.append(table["history"])
                verified_rounds+=len(table["rounds"]);verified_models+=len(table["evaluation"])
    atomic_json({"expected":6,"results":completion},out/"confirmation_completion.json")
    if any(r["state"]!="complete" for r in completion):
        raise RuntimeError("Confirmation incomplete or failed; all three pairs are required")
    summary,history=pd.concat(summaries,ignore_index=True),pd.concat(histories,ignore_index=True)
    runs=add_regularization_columns(summary[summary.variant=="both_codecs"].copy(),BASE.regularization)
    pairs,effects=[],[]
    definitions=(("accuracy_difference_pp","final_test_accuracy",False,100),("test_ce_difference","final_test_ce",False,1),
                 ("unicast_ratio","unicast_total_bits",True,1),("upload_ratio","uplink_bits",True,1),
                 ("download_ratio","unicast_downlink_bits",True,1),("broadcast_ratio","broadcast_total_bits",True,1))
    event,baseline=(runs[runs.method==m].set_index("partition_seed").sort_index() for m in ("event","strom"))
    if (event.index.has_duplicates or baseline.index.has_duplicates or set(event.index)!=set(data_tools.CONFIRM_SEEDS)
            or set(baseline.index)!=set(data_tools.CONFIRM_SEEDS)):
        raise RuntimeError("Missing confirmation seed pair")
    for name,field,ratio,scale in definitions:
        values=event[field]/baseline[field] if ratio else scale*(event[field]-baseline[field])
        pairs.extend({"partition_seed":int(s),"metric":name,"value":float(v)} for s,v in values.items())
        effects.append({"metric":name,**interval(values,ratio=ratio),"direction":"Event_over_Strom" if ratio else "Event_minus_Strom"})
    crossings=[]
    main_history=history[history.variant=="both_codecs"]
    for (method,seed),curve in main_history.groupby(["method","partition_seed"]):
        curve=curve.sort_values("round").copy();curve["uplink_packetized_bits"]=curve.uplink_bits
        crossings.extend(target_crossings(curve,targets=TARGETS,horizon=FINAL,comparison_id=method,partition_seed=int(seed)))
    targets=pd.DataFrame(crossings);target_summary=[]
    for target,frame in targets.groupby("target_accuracy"):
        a,b=(frame[frame.comparison_id==m].set_index("partition_seed").sort_index() for m in ("event","strom"))
        common=a.reached&b.reached
        target_summary.append({"target_accuracy":target,"n_total":3,"n_event_reached":int(a.reached.sum()),
                               "n_strom_reached":int(b.reached.sum()),"n_common_hits":int(common.sum()),
                               **interval(a.loc[common,"unicast_total_bits"]/b.loc[common,"unicast_total_bits"],ratio=True)})
    for name,frame in (("confirmation_runs.csv",runs),("confirmation_history.csv",history),("paired_per_seed.csv",pd.DataFrame(pairs)),
                       ("paired_summary.csv",pd.DataFrame(effects)),("traffic_to_target.csv",targets),("paired_target_summary.csv",pd.DataFrame(target_summary))):
        atomic_csv(frame,out/name)
    atomic_json({"new_confirmation_runs":6,"seed_pairs":3,"rounds_verified":verified_rounds,"sampled_model_fingerprints":verified_models,
                 "frozen_selection_sha256":digest(frozen),"test_based_selection_or_stopping":False,"all_pairs_retained":True,
                 "scope":"three_pair_initial_architecture_transfer_not_ten_seed_confirmation"},out/"analysis_audit.json")
    plot(main_history,pd.DataFrame(target_summary),out)
    print(pd.DataFrame(effects).to_string(index=False))
    print(pd.DataFrame(target_summary).to_string(index=False))


def plot(history,targets,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for method,color in (("event","#D55E00"),("strom","#0072B2")):
        grouped=history[history.method==method].groupby("round")
        accuracy,ce,bits=grouped.test_accuracy.mean(),grouped.test_ce.mean(),grouped.unicast_total_bits.mean()
        axes[0].plot(accuracy.index,100*accuracy,label=method,color=color)
        axes[1].plot(bits/1e6,100*accuracy,label=method,color=color)
        axes[2].plot(ce.index,ce,label=method,color=color)
    axes[0].set(xlabel="Communication rounds",ylabel="Mean test accuracy (%)")
    axes[1].set(xlabel="Mean total unicast traffic (Mbit)",ylabel="Mean test accuracy (%)")
    axes[2].set(xlabel="Communication rounds",ylabel="Mean test cross-entropy")
    for axis in axes:
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle("ResNet-14: frozen selections, three fresh matched pairs")
    fig.tight_layout();fig.savefig(out/"confirmation.png",dpi=180)
    fig.savefig(out/"confirmation.pdf",metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def preflight(device_name="auto",threads=4,eval_batch_size=256):
    import resnet14_transfer_host as host
    device=host.configure(device_name,threads)
    science_spec()
    print(json.dumps(host.runtime(device),indent=2))
    print("ResNet14/GroupNorm: 175258 parameters; 15 settings per method; two finalists per method; three paired confirmations")
    print("Maximum 33600 training rounds; 596 saved evaluations; sequential one-device execution; no test-based tuning")
    print("Development seed 7000; confirmation seeds 7100/7200/7300; local lr .05, five steps, batch32; final horizon1800")
    print("Run benchmark for a machine-specific planning estimate; it uses synthetic images only.")


def status(out=OUT):
    if not (out/"protocol.json").exists():
        print("ResNet transfer not started")
        return
    science=load_contract(out)
    for p in POINTS:
        labels=[]
        for horizon in (SCREEN,FINAL):
            receipt=read_receipt(p,"development",data_tools.DEV_SEED,horizon,science,out)
            if receipt:labels.append(f"r{horizon}:{receipt['state']}")
        checkpoint=checkpoint_path(p,"development",data_tools.DEV_SEED,out)
        if checkpoint.exists():
            with np.load(checkpoint,allow_pickle=False) as saved:
                labels.append("checkpoint:"+str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"]))
        print(p.point_id,", ".join(labels) or "pending")
    if (out/"frozen_selection.json").exists():
        frozen=frozen_selection(science,out)
        for seed in data_tools.CONFIRM_SEEDS:
            for method in ("event","strom"):
                p=BY_ID[frozen["selected"][method][0]];receipt=read_receipt(p,"confirmation",seed,FINAL,science,out)
                checkpoint=checkpoint_path(p,"confirmation",seed,out)
                label=receipt["state"] if receipt else "pending"
                if receipt is None and checkpoint.exists():
                    with np.load(checkpoint,allow_pickle=False) as saved:
                        label="checkpoint:"+str(json.loads(str(saved["metadata"].item()))["progress"]["completed_round"])
                print("confirmation",method,seed,label)


def synthetic_federation(device,clients=2,examples=8,evaluation=8):
    import torch
    rng=np.random.default_rng(51)
    images=lambda n:torch.from_numpy(rng.integers(0,256,(n,3,32,32),dtype=np.uint8)).to(device)
    labels=lambda n:torch.from_numpy(rng.integers(0,10,n,dtype=np.int64)).to(device)
    return data_tools.Federation(tuple(images(examples) for _ in range(clients)),tuple(labels(examples) for _ in range(clients)),
                                 images(evaluation),labels(evaluation),images(evaluation),labels(evaluation),
                                 torch.full((clients,),1/clients,dtype=torch.float32,device=device),"synthetic",{"synthetic":True})


def smoke(device_name="auto",threads=4,eval_batch_size=256):
    import resnet14_transfer_host as host
    device=host.configure(device_name,threads)
    fed=synthetic_federation(device)
    for method in ("event","strom"):
        p=next(p for p in POINTS if p.method==method)
        config=replace(config_for(p),rounds=3,eval_stride=1,local_steps=1,batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/"full.npz",Path(folder)/"resumed.npz"
            kwargs=dict(federation=fed,method=method,config=config,seed=97000,device=device,eval_batch_size=eval_batch_size)
            left=host.run(**kwargs,checkpoint=BaselineCheckpoint(a,{}))
            host.run(**kwargs,checkpoint=BaselineCheckpoint(b,{}),stop_after=2)
            right=host.run(**kwargs,checkpoint=BaselineCheckpoint(b,{}))
            with np.load(a,allow_pickle=False) as before,np.load(b,allow_pickle=False) as after:
                if set(before.files)!=set(after.files):raise AssertionError("ResNet checkpoint arrays differ")
                for k in before.files:
                    if k!="metadata":np.testing.assert_array_equal(before[k].view(np.uint8),after[k].view(np.uint8))
            fields=[k for k in left["rounds"] if k not in ("training_seconds","codec_seconds")]
            pd.testing.assert_frame_equal(left["rounds"][fields],right["rounds"][fields],check_exact=True)
            pd.testing.assert_frame_equal(left["evaluation"],right["evaluation"],check_exact=True)
    print("Actual synthetic ResNet14: both methods decode native Torch updates exactly and resume model/state/RNG/metrics/traffic exactly.")


def benchmark(device_name="auto",threads=4,eval_batch_size=256):
    import resnet14_transfer_host as host
    device=host.configure(device_name,threads)
    fed=synthetic_federation(device,clients=10,examples=64,evaluation=256)
    timings=[]
    with tempfile.TemporaryDirectory() as folder:
        for method in ("event","strom"):
            p=next(p for p in POINTS if p.method==method)
            config=replace(config_for(p),rounds=2,eval_stride=1)
            result=host.run(federation=fed,method=method,config=config,seed=97000,device=device,
                            checkpoint=BaselineCheckpoint(Path(folder)/(method+".npz"),{}),eval_batch_size=eval_batch_size)
            row=result["rounds"].iloc[-1]
            timings.append(float(row.training_seconds+row.codec_seconds))
            print(method,"synthetic training+codec seconds/round:",round(timings[-1],3))
    estimate=np.mean(timings)*33600/3600
    print(f"Planning estimate for maximum training+codec work: {estimate:.1f} hours; excludes full-data evaluations/checkpoints/I/O.")
    print("Synthetic firing patterns and system load affect this estimate; it is not a finish-time guarantee or paper timing evidence.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("verify","smoke","preflight","benchmark","run","status","analyze"))
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--eval-batch-size",type=int,default=256)
    args=parser.parse_args()
    if args.threads<1 or args.eval_batch_size<1:parser.error("threads and eval batch size must be positive")
    if args.command=="verify":
        env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join([str(REPO/"src"),str(REPO),env.get("PYTHONPATH","")])
        subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_resnet14_transfer.py","-v"],cwd=REPO,env=env,check=True)
        smoke(args.device,args.threads,args.eval_batch_size)
    elif args.command in ("status","analyze"):
        globals()[args.command]()
    else:
        globals()[args.command](args.device,args.threads,args.eval_batch_size)


if __name__=="__main__":
    main()
