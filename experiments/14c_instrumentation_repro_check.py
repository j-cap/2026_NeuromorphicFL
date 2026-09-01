from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation, run_federated_method

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
spec = spec_from_file_location("diag14c", ROOT / "experiments" / "14c_event_share_fairness.py")
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def main() -> None:
    fed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)
    canonical = run_federated_method(
        federation=fed,
        method="events",
        n_ticks=650,
        gamma=1.0,
        threshold=0.025,
        jump0=0.0035,
        eval_stride=100,
    )
    inst1, *_ = mod.run_event_diagnostic(fed, n_ticks=650, eval_stride=100)
    inst2, *_ = mod.run_event_diagnostic(fed, n_ticks=650, eval_stride=100)
    print("CANONICAL", canonical["final_test_ce"], canonical["final_test_accuracy"], canonical["final_worst_class_accuracy"], canonical["candidate_events"])
    print("INSTRUMENTED_1", inst1)
    print("INSTRUMENTED_2", inst2)
    print("INSTRUMENTED_EXACT_REPEAT", inst1 == inst2)


if __name__ == "__main__":
    main()
