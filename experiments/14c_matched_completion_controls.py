from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation, run_federated_method

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14c_event_share_fairness"
EQUAL = np.full(10, 3, dtype=int)

spec = spec_from_file_location("diag14c", ROOT / "experiments" / "14c_event_share_fairness.py")
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for regime,seed in (("iid",2400),("strong",2400),("strong",2500)):
        base=make_multiclass_federation(root=DATA_ROOT,regime=regime,seed=seed)
        fed=replace(base,periods=EQUAL.copy())
        event,*_=mod.run_event_diagnostic(fed,n_ticks=722,eval_stride=100)
        rows.append({"regime":regime,"seed":seed,"method":"events","test_ce":event["test_ce"],"test_accuracy":event["test_accuracy"],"worst_class_accuracy":event["worst_class_accuracy"],"total_completions":event["total_completions"],"event_share_cv":event["event_share_cv"],"events_per_completion_cv":event["events_per_completion_cv"]})
        for method in ("full","ef_topk"):
            kwargs=dict(federation=fed,method=method,n_ticks=722,eval_stride=100,step=0.08)
            if method=="ef_topk": kwargs["topk_fraction"]=0.025
            r=run_federated_method(**kwargs)
            rows.append({"regime":regime,"seed":seed,"method":method,"test_ce":r["final_test_ce"],"test_accuracy":r["final_test_accuracy"],"worst_class_accuracy":r["final_worst_class_accuracy"],"total_completions":2400,"event_share_cv":np.nan,"events_per_completion_cv":np.nan})
    df=pd.DataFrame(rows)
    df.to_csv(RESULT_ROOT/"matched_completion_controls.csv",index=False)
    print("=== 14C MATCHED COMPLETION CONTROLS ===")
    print(df.to_string(index=False))

if __name__=="__main__": main()
