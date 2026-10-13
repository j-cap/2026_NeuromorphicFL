"""M3: development-selected scheduled-gain dense FedAvg fairness control.

Commands (repo root):
  python experiments/m3_dense_schedule.py smoke
  python experiments/m3_dense_schedule.py develop --workers 2
  python experiments/m3_dense_schedule.py select
  python experiments/m3_dense_schedule.py confirm --workers 2
  python experiments/m3_dense_schedule.py verify
  python experiments/m3_dense_schedule.py analyze

Frozen full 1800-round development selection uses partition 4500, outside the
final ten-seed evaluation set (3500...4400). Select by final TRAIN objective
only; never rank candidates by development test accuracy. The frozen selected
configuration is then run on the ten P19-matched seeds, with existing P19
Event-FedAvg and constant dense outputs used for paired comparisons.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import numpy as np
import pandas as pd
from neuromorphicfl.cifar10_benchmark import download_cifar10, make_cifar10_federation
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from p12_headline_ten_seed import CIFAR_BASE

DATA = REPO / "data" / "cifar-10"
OUT = REPO / "experiments" / "results" / "m3_dense_schedule"
P19 = REPO / "experiments" / "results" / "p19_cifar10_cnn_final_horizons"
ROUNDS=1800
DEV_PARTITION=4500
EVAL_SEEDS=tuple(range(3500,4500,100))
CANDIDATES=tuple((gain,alpha) for gain in (1.0,2.0,4.0)
                 for alpha in (0.0,0.1,0.2,0.3,0.5))
T95=2.2621571627409915


def label(gain,alpha):
    return f"g{gain:g}_a{alpha:.2f}".replace(".","p")


def config(gain,alpha):
    return replace(CIFAR_BASE, rounds=ROUNDS, eval_stride=60,
                   server_gain=gain, server_gain_exponent=alpha,
                   server_gain_scale=100.0)


def paths(name, seed):
    stem=f"{name}_p{seed}_r{ROUNDS}"
    return OUT/(stem+"_summary.csv"),OUT/(stem+"_history.csv")


def atomic_csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".tmp-{os.getpid()}")
    frame.to_csv(tmp,index=False,float_format="%.12g")
    os.replace(tmp,path)


def run(name,gain,alpha,partition,force=False):
    summary,history=paths(name,partition)
    if summary.exists() and history.exists() and not force:
        return f"reuse {name} seed={partition}"
    fed=make_cifar10_federation(root=DATA,regime="strong",seed=partition)
    result=run_final_baseline(federation=fed,architecture="cifar_cnn",
                              method="dense",config=config(gain,alpha),
                              seed=80000+partition,record_history=True)
    hist=result.pop("history")
    if not isinstance(hist,pd.DataFrame):
        raise RuntimeError("missing history")
    meta={"point_id":name,"partition_seed":partition,
          "train_seed":80000+partition,"rounds":ROUNDS,
          "gain0":gain,"gain_exponent":alpha}
    result.update(meta)
    atomic_csv(pd.DataFrame([result]),summary)
    atomic_csv(hist.assign(**meta),history)
    return f"done {name} seed={partition}: train={result['final_train_objective']:.5f}"


def jobs_run(jobs,workers,force=False):
    download_cifar10(DATA)
    pending=[job for job in jobs if force or not all(p.exists() for p in paths(job[0],job[3]))]
    print(f"{len(jobs)-len(pending)}/{len(jobs)} complete; {len(pending)} pending")
    if workers==1:
        for job in pending:
            print(run(*job,force=force),flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run,*job,force):job for job in pending}
        for future in as_completed(futures):
            print(future.result(),flush=True)


def develop(workers,force=False):
    jobs=[(label(g,a),g,a,DEV_PARTITION) for g,a in CANDIDATES]
    jobs_run(jobs,workers,force)


def select():
    rows=[]
    for g,a in CANDIDATES:
        p,h=paths(label(g,a),DEV_PARTITION)
        if not p.exists() or not h.exists():
            raise RuntimeError(f"Missing development result: {p}")
        d=pd.read_csv(p).iloc[0]
        hdata=pd.read_csv(h)
        if int(d["rounds"])!=ROUNDS or int(hdata["round"].max())!=ROUNDS:
            raise RuntimeError(f"development horizon mismatch: {p}")
        val=float(d["final_train_objective"])
        if not np.isfinite(val):
            raise RuntimeError(f"nonfinite training objective: {p}")
        rows.append(dict(point_id=label(g,a),gain0=g,gain_exponent=a,
                         final_train_objective=val))
    frame=pd.DataFrame(rows).sort_values(["final_train_objective","gain0","gain_exponent"])
    chosen=frame.iloc[0]
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_csv(frame,OUT/"development_ranking.csv")
    frozen={"point_id":str(chosen.point_id),"gain0":float(chosen.gain0),
            "gain_exponent":float(chosen.gain_exponent),"gain_scale":100.0,
            "selection_metric":"final_train_objective",
            "development_partition_seed":DEV_PARTITION,
            "rounds":ROUNDS}
    path=OUT/"frozen_selection.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old!=frozen:
            raise RuntimeError("Frozen choice exists and differs; investigate before modifying")
    else:
        path.write_text(json.dumps(frozen,indent=2)+"\n")
    print("Frozen scheduled dense:",frozen)


def selected():
    p=OUT/"frozen_selection.json"
    if not p.exists(): raise RuntimeError("Run develop and select first")
    d=json.loads(p.read_text())
    return d["point_id"],float(d["gain0"]),float(d["gain_exponent"])


def confirm(workers,force=False):
    _,g,a=selected()
    jobs=[("m3_scheduled_dense",g,a,seed) for seed in EVAL_SEEDS]
    jobs_run(jobs,workers,force)


def load_confirm():
    records=[]
    name,g,a=selected()
    for i,seed in enumerate(EVAL_SEEDS):
        path,hist=paths("m3_scheduled_dense",seed)
        if not path.exists() or not hist.exists():
            raise RuntimeError(f"missing confirmation seed={seed}")
        d=pd.read_csv(path).iloc[0]
        h=pd.read_csv(hist)
        if int(d["rounds"])!=ROUNDS or int(h["round"].max())!=ROUNDS:
            raise RuntimeError("horizon drift")
        if int(d["train_seed"])!=80000+seed or int(d["partition_seed"])!=seed:
            raise RuntimeError("seed drift")
        if abs(float(d["gain0"])-g)>1e-12 or abs(float(d["gain_exponent"])-a)>1e-12:
            raise RuntimeError("scheduled gain drift")
        records.append(d.to_dict())
    return pd.DataFrame(records).sort_values("partition_seed")


def reference(point,seed):
    pattern=f"{point}_s{EVAL_SEEDS.index(seed)}_p{seed}_r{ROUNDS}_summary.csv"
    path=P19/pattern
    if not path.exists(): raise RuntimeError(f"Missing frozen P19 reference: {path}")
    d=pd.read_csv(path)
    if len(d)!=1: raise RuntimeError(f"invalid reference {path}")
    row=d.iloc[0]
    if int(row["partition_seed"])!=seed or int(row["train_seed"])!=80000+seed:
        raise RuntimeError(f"reference seed mismatch {path}")
    return row


def analyze():
    df=load_confirm()
    pairs=[]
    for _,row in df.iterrows():
        seed=int(row["partition_seed"])
        for point,alias in (("cifar_dense_gain2","constant_dense"),
                            ("cifar_event","event"),
                            ("cifar_ef_quality","ef_topk")):
            ref=reference(point,seed)
            pairs.append(dict(partition_seed=seed,comparator=alias,
                accuracy_diff=float(row["final_test_accuracy"])-float(ref["final_test_accuracy"]),
                traffic_diff=float(row["unicast_hybrid_total_bits"])-float(ref["unicast_hybrid_total_bits"]),
                comparator_acc=float(ref["final_test_accuracy"]),
                comparator_bits=float(ref["unicast_hybrid_total_bits"])))
    paired=pd.DataFrame(pairs)
    summary=[]
    for comp,d in paired.groupby("comparator"):
        for metric in ("accuracy_diff","traffic_diff"):
            values=d[metric].to_numpy()
            mean=float(values.mean())
            half=T95*float(values.std(ddof=1))/np.sqrt(len(values))
            summary.append(dict(comparator=comp,metric=metric,n=len(values),
                                mean_difference=mean,ci95_low=mean-half,
                                ci95_high=mean+half))
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_csv(df,OUT/"confirmation_runs.csv")
    atomic_csv(paired,OUT/"paired_per_seed.csv")
    atomic_csv(pd.DataFrame(summary),OUT/"paired_summary.csv")
    print(pd.DataFrame(summary).to_string(index=False))
    print("No statistical equivalence claim follows from intervals spanning zero.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("smoke")
    for name in ("develop","confirm"):
        q=sub.add_parser(name)
        q.add_argument("--workers",type=int,choices=range(1,9),default=2)
        q.add_argument("--force",action="store_true")
    for name in ("select","verify","analyze","status"):
        sub.add_parser(name)
    args=p.parse_args()
    if args.command=="smoke":
        download_cifar10(DATA)
        fed=make_cifar10_federation(root=DATA,regime="strong",seed=DEV_PARTITION)
        c=replace(config(2.0,0.2),rounds=2,eval_stride=1)
        result=run_final_baseline(federation=fed,architecture="cifar_cnn",
                                  method="dense",config=c,seed=84500)
        assert np.isfinite(float(result["final_train_objective"]))
        print("M3 scheduled dense smoke passed")
    elif args.command=="develop": develop(args.workers,args.force)
    elif args.command=="select": select()
    elif args.command=="confirm": confirm(args.workers,args.force)
    elif args.command=="verify":
        print(f"M3 verified {len(load_confirm())} frozen matched confirmation runs")
    elif args.command=="analyze": analyze()
    elif args.command=="status":
        for stage,seeds in (("development",[DEV_PARTITION]),("confirmation",EVAL_SEEDS)):
            names=[label(g,a) for g,a in CANDIDATES] if stage=="development" else ["m3_scheduled_dense"]
            finished=sum(all(x.exists() for x in paths(n,seed)) for n in names for seed in seeds)
            print(f"{stage}: {finished}/{len(names)*len(seeds)}")
    else: raise RuntimeError(args.command)


if __name__=="__main__":
    main()
