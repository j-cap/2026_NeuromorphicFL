from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn

ROOT=Path("data/fashion-mnist")
OUT=Path("experiments/results/15a_fmnist_cnn")
OUT.mkdir(parents=True,exist_ok=True)
fed=make_multiclass_federation(root=ROOT,regime="strong",seed=2400)
rows=[]

# Two identical event calls establish within-environment hybrid reproducibility.
for rep in [1,2]:
    r=run_federated_cnn(
        federation=fed,method="events",n_ticks=1200,rho=0.999,gamma=1.0,
        threshold=0.025,jump0=0.0100,schedule_exponent=0.5,eval_stride=100,
    )
    rows.append({"configuration":f"events_rep{rep}",**r})

for step in [0.04,0.08]:
    r=run_federated_cnn(federation=fed,method="full",n_ticks=1200,step=step,eval_stride=100)
    rows.append({"configuration":f"full_{step:g}",**r})
for step in [0.04,0.08]:
    r=run_federated_cnn(
        federation=fed,method="ef_topk",n_ticks=1200,step=step,
        topk_fraction=0.025,eval_stride=100,
    )
    rows.append({"configuration":f"ef_2p5_{step:g}",**r})

frame=pd.DataFrame(rows)
frame.to_csv(OUT/"final_validation_1200.csv",index=False)
cols=["configuration","method","final_train_objective","final_test_ce","final_test_accuracy",
      "final_worst_class_accuracy","whole_train_objective","payload_bits","candidate_events","events_per_message"]
print("=== 15A PAIRED FINAL VALIDATION 1200 ===")
print(frame[cols].to_string(index=False))
