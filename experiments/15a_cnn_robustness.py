from __future__ import annotations

from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn

ROOT=Path("data/fashion-mnist")
OUT=Path("experiments/results/15a_fmnist_cnn")
OUT.mkdir(parents=True,exist_ok=True)
rows=[]

def run_case(regime,seed):
    fed=make_multiclass_federation(root=ROOT,regime=regime,seed=seed)
    methods=[
        ("events",dict(rho=0.999,gamma=1.0,threshold=0.025,jump0=0.0075)),
        ("ef_0p5",dict(method="ef_topk",step=0.08,topk_fraction=0.005)),
        ("ef_2p5",dict(method="ef_topk",step=0.08,topk_fraction=0.025)),
        ("full",dict(method="full",step=0.12)),
    ]
    for name,kw in methods:
        method=kw.pop("method", "events")
        r=run_federated_cnn(federation=fed,method=method,n_ticks=650,eval_stride=50,**kw)
        rows.append({"regime":regime,"seed":seed,"configuration":name,**r})

# independent strong partitions plus transfer to easier heterogeneity regimes
for seed in [2500,2600]:
    run_case("strong",seed)
for regime in ["iid","moderate"]:
    run_case(regime,2400)

frame=pd.DataFrame(rows)
frame.to_csv(OUT/"robustness_650.csv",index=False)
cols=["regime","seed","configuration","final_train_objective","final_test_ce",
      "final_test_accuracy","final_worst_class_accuracy","whole_train_objective",
      "payload_bits","candidate_events","events_per_message"]
print("=== 15A ROBUSTNESS / TRANSFER ===")
print(frame[cols].to_string(index=False))
