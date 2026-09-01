from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fedavg_event_compatibility import FedAvgConfig, run_fedavg_method
from neuromorphicfl.fedavg_cnn_compatibility import run_fedavg_cnn

ROOT=Path("data/fashion-mnist")
OUT=Path("experiments/results/q3_fedavg_compatibility")
OUT.mkdir(parents=True,exist_ok=True)


def run_mlp(E: int):
    fed=make_multiclass_federation(root=ROOT,regime="strong",seed=2400)
    rows=[]
    base=dict(local_steps=E,local_lr=0.1,rounds=150,eval_stride=15)
    cfg=FedAvgConfig(**base,threshold=0.025,jump0=0.005,jump_exponent=0.1,jump_scale=100.0)
    rows.append({"configuration":"events",**run_fedavg_method(federation=fed,method="event_fedavg",config=cfg)})
    cfg=FedAvgConfig(**base)
    rows.append({"configuration":"dense",**run_fedavg_method(federation=fed,method="dense_fedavg",config=cfg)})
    for frac in [0.005,0.025]:
        cfg=FedAvgConfig(**base,topk_fraction=frac)
        rows.append({"configuration":f"ef_{frac:g}",**run_fedavg_method(federation=fed,method="ef_topk_fedavg",config=cfg)})
    cfg=FedAvgConfig(**base)
    rows.append({"configuration":"sign_ef",**run_fedavg_method(federation=fed,method="sign_ef_fedavg",config=cfg)})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/f"final_mlp_E{E}.csv",index=False)
    print(f"=== Q3 FINAL MLP E={E} ===")
    print(frame[["configuration","final_train_objective","final_test_ce","final_test_accuracy","final_worst_class_accuracy","payload_bits","coordinate_events","events_per_message","mean_delta_norm","final_membrane_norm","sign_reversal_rate"]].sort_values("final_train_objective").to_string(index=False))


def run_cnn():
    fed=make_multiclass_federation(root=ROOT,regime="strong",seed=2400)
    # Small local-LR audit first, then the matched communication methods at the selected LR.
    rows=[]
    E=5; rounds=80
    dense=[]
    for lr in [0.02,0.05,0.08,0.1]:
        cfg=FedAvgConfig(local_steps=E,local_lr=lr,rounds=rounds,eval_stride=10)
        r=run_fedavg_cnn(federation=fed,method="dense_fedavg",config=cfg)
        row={"configuration":f"dense_lr{lr:g}",**r}; rows.append(row); dense.append(row)
    best=min(dense,key=lambda r:r["final_train_objective"]); lr=float(best["local_lr"])
    for q in [0.005,0.0075,0.01]:
        cfg=FedAvgConfig(local_steps=E,local_lr=lr,rounds=rounds,eval_stride=10,threshold=0.025,jump0=q,jump_exponent=0.3,jump_scale=100.0)
        rows.append({"configuration":f"events_q{q:g}",**run_fedavg_cnn(federation=fed,method="event_fedavg",config=cfg)})
    for frac in [0.005,0.025]:
        cfg=FedAvgConfig(local_steps=E,local_lr=lr,rounds=rounds,eval_stride=10,topk_fraction=frac)
        rows.append({"configuration":f"ef_{frac:g}",**run_fedavg_cnn(federation=fed,method="ef_topk_fedavg",config=cfg)})
    cfg=FedAvgConfig(local_steps=E,local_lr=lr,rounds=rounds,eval_stride=10)
    rows.append({"configuration":"sign_ef",**run_fedavg_cnn(federation=fed,method="sign_ef_fedavg",config=cfg)})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"final_cnn_E5.csv",index=False)
    print(f"=== Q3 FINAL CNN E=5; selected local lr={lr:g} ===")
    print(frame[["configuration","final_train_objective","final_test_ce","final_test_accuracy","final_worst_class_accuracy","payload_bits","coordinate_events","events_per_message","mean_delta_norm","final_membrane_norm","sign_reversal_rate"]].sort_values("final_train_objective").to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["mlp1","mlp5","mlp10","cnn"],required=True); a=p.parse_args()
    if a.mode.startswith("mlp"): run_mlp(int(a.mode[3:]))
    else: run_cnn()
