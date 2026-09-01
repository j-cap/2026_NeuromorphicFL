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
curve_rows=[]
configs=[
    (0.0075,0.10),
    (0.0075,0.30),
    (0.0075,0.50),
    (0.0100,0.30),
    (0.0100,0.50),
    (0.0125,0.50),
]
for q0,p in configs:
    r=run_federated_cnn(
        federation=fed,method="events",n_ticks=1200,rho=0.999,gamma=1.0,
        threshold=0.025,jump0=q0,schedule_exponent=p,eval_stride=50,
        record_history=True,
    )
    hist=r.pop("history")
    label=f"event_q{q0:g}_p{p:g}"
    rows.append({"configuration":label,**r})
    h=hist.copy(); h.insert(0,"configuration",label); curve_rows.append(h)
summary=pd.DataFrame(rows)
curves=pd.concat(curve_rows,ignore_index=True)
summary.to_csv(OUT/"schedule_refinement_1200.csv",index=False)
curves.to_csv(OUT/"schedule_refinement_history.csv",index=False)
cols=["configuration","final_train_objective","final_test_ce","final_test_accuracy",
      "final_worst_class_accuracy","whole_train_objective","payload_bits","candidate_events","events_per_message"]
print("=== 15A EVENT SCHEDULE REFINEMENT ===")
print(summary[cols].to_string(index=False))
print("=== 650/1200 HISTORY POINTS ===")
print(curves[curves.tick.isin([650,1200])][["configuration","tick","train_objective","test_ce","test_accuracy","worst_class_accuracy","payload_bits"]].to_string(index=False))
