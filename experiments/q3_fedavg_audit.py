from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fedavg_event_compatibility import FedAvgConfig, run_fedavg_method

ROOT=Path("data/fashion-mnist")
OUT=Path("experiments/results/q3_fedavg_compatibility")
OUT.mkdir(parents=True,exist_ok=True)


def main(local_steps: int) -> None:
    fed=make_multiclass_federation(root=ROOT,regime="strong",seed=2400)
    rows=[]
    rounds=70

    dense_lrs=[0.01,0.02,0.05,0.1]
    for lr in dense_lrs:
        cfg=FedAvgConfig(local_steps=local_steps,local_lr=lr,rounds=rounds,eval_stride=10)
        r=run_fedavg_method(federation=fed,method="dense_fedavg",config=cfg)
        rows.append({"configuration":f"dense_lr{lr:g}",**r})
    dense_rows=[r for r in rows if r["method"]=="dense_fedavg"]
    best_dense=min(dense_rows,key=lambda r:r["final_train_objective"])
    lr=float(best_dense["local_lr"])

    # Small communication-side audit. The encoder gain is 1/lr so the
    # evidence scale connects continuously to the E=1 gradient-driven case.
    event_grid=[
        (0.025,0.0025,0.1),
        (0.025,0.0035,0.1),
        (0.025,0.0050,0.1),
        (0.05,0.0035,0.1),
        (0.025,0.0050,0.3),
    ]
    for threshold,jump0,p in event_grid:
        cfg=FedAvgConfig(
            local_steps=local_steps,local_lr=lr,rounds=rounds,eval_stride=10,
            threshold=threshold,jump0=jump0,jump_exponent=p,
        )
        r=run_fedavg_method(federation=fed,method="event_fedavg",config=cfg)
        rows.append({"configuration":f"event_t{threshold:g}_q{jump0:g}_p{p:g}",**r})

    for frac in [0.005,0.025]:
        cfg=FedAvgConfig(
            local_steps=local_steps,local_lr=lr,rounds=rounds,eval_stride=10,
            topk_fraction=frac,
        )
        r=run_fedavg_method(federation=fed,method="ef_topk_fedavg",config=cfg)
        rows.append({"configuration":f"ef_topk_{frac:g}",**r})

    cfg=FedAvgConfig(local_steps=local_steps,local_lr=lr,rounds=rounds,eval_stride=10)
    r=run_fedavg_method(federation=fed,method="sign_ef_fedavg",config=cfg)
    rows.append({"configuration":"sign_ef",**r})

    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/f"audit_E{local_steps}.csv",index=False)
    cols=["configuration","method","local_steps","local_lr","final_train_objective",
          "final_test_ce","final_test_accuracy","final_worst_class_accuracy",
          "payload_bits","coordinate_events","events_per_message","mean_delta_norm",
          "final_membrane_norm","sign_reversal_rate","client_payload_cv"]
    print(f"=== Q3 FEDAVG AUDIT E={local_steps}; selected local lr={lr:g} ===")
    print(frame[cols].sort_values("final_train_objective").to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--local-steps",type=int,required=True)
    args=parser.parse_args()
    main(args.local_steps)
