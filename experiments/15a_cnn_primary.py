from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn

ROOT = Path("data/fashion-mnist")
OUT = Path("experiments/results/15a_fmnist_cnn")
OUT.mkdir(parents=True, exist_ok=True)
fed = make_multiclass_federation(root=ROOT, regime="strong", seed=2400)
rows=[]

# Horizon-specific baseline audit.
for step in [0.04, 0.08, 0.12]:
    r=run_federated_cnn(federation=fed,method="full",n_ticks=650,step=step,eval_stride=50)
    rows.append({"configuration":f"full_{step:g}",**r})
for frac in [0.005,0.025]:
    for step in [0.08,0.12]:
        r=run_federated_cnn(federation=fed,method="ef_topk",n_ticks=650,step=step,topk_fraction=frac,eval_stride=50)
        rows.append({"configuration":f"ef_{frac:g}_{step:g}",**r})

# Focused event refinement. The 250-tick audit showed monotone benefit up to gamma=1.
for gamma,q0 in [
    (1.0,0.0035),(1.0,0.0050),(1.0,0.0075),
    (1.5,0.0035),(1.5,0.0050),(2.0,0.0035),
]:
    r=run_federated_cnn(
        federation=fed,method="events",n_ticks=650,rho=0.999,gamma=gamma,
        threshold=0.025,jump0=q0,eval_stride=50,
    )
    rows.append({"configuration":f"event_g{gamma:g}_q{q0:g}",**r})

frame=pd.DataFrame(rows)
frame.to_csv(OUT/"primary_650.csv",index=False)
cols=["configuration","method","final_train_objective","final_test_ce","final_test_accuracy",
      "final_worst_class_accuracy","whole_train_objective","payload_bits","candidate_events",
      "events_per_message","ever_fired_fraction"]
print("=== 15A PRIMARY 650 ===")
print(frame[cols].to_string(index=False))
print("=== EVENT TENSOR DIAGNOSTICS ===")
events=frame[frame.method=="events"]
diag_cols=[c for c in frame.columns if c.endswith("events_per_param") or c.endswith("never_fired")]
print(events[["configuration"]+diag_cols].to_string(index=False))
