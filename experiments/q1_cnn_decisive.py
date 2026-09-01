from __future__ import annotations
from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import run_federated_cnn
from neuromorphicfl.q1_nearest_neighbor import run_cnn_pulse_method

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "q1_nearest_neighbor_focused"


def add(rows, row, name, family):
    row = dict(row); row["configuration"] = name; row["family"] = family; rows.append(row)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    rows=[]
    # MLP-selected nearest-neighbor operating regions.
    for variant, gain, threshold in [
        ("strom", 0.3, 0.0025),
        ("leaky_subtractive", 0.3, 0.0025),
        ("fullreset_tied", 0.3, 0.005),
        ("fullreset_tied", 1.0, 0.0025),
    ]:
        r=run_cnn_pulse_method(federation=fed, variant=variant, n_ticks=650, seed=60606,
                               gain=gain, threshold=threshold, rho=0.999, eval_stride=50)
        add(rows,r,f"{variant}_g{gain:g}_t{threshold:g}",variant)
    v1=run_federated_cnn(federation=fed,method="events",n_ticks=650,seed=60606,
                         rho=0.999,gamma=1.0,threshold=0.025,jump0=0.0075,
                         schedule_exponent=0.1,eval_stride=50)
    add(rows,v1,"v1","v1")
    ef=run_federated_cnn(federation=fed,method="ef_topk",n_ticks=650,seed=60606,
                         step=0.08,topk_fraction=0.025,eval_stride=50)
    add(rows,ef,"ef2p5","ef_topk")
    dense=run_federated_cnn(federation=fed,method="full",n_ticks=650,seed=60606,
                            step=0.12,eval_stride=50)
    add(rows,dense,"dense","dense")
    df=pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT/"cnn_decisive_650.csv",index=False)
    cols=["configuration","family","final_train_objective","final_test_ce",
          "final_test_accuracy","final_worst_class_accuracy","whole_train_objective","payload_bits"]
    print(df[cols].sort_values("final_train_objective").to_string(index=False))

if __name__ == "__main__": main()
