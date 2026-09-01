from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import (
    make_multiclass_federation,
    run_federated_method,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "fashion-mnist"
RESULT_ROOT = ROOT / "experiments" / "results" / "14b_fmnist_multiclass"

spec = spec_from_file_location(
    "fairness14b", ROOT / "experiments" / "14b_multiclass_fairness.py"
)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
make_permuted_federation = mod.make_permuted_federation


def run_named(federation, name: str, method: str, n_ticks: int, **kwargs):
    r = run_federated_method(
        federation=federation,
        method=method,
        n_ticks=n_ticks,
        eval_stride=100,
        **kwargs,
    )
    r["configuration"] = name
    return r


def best_by_train(rows: list[dict], method: str, fraction: float | None = None):
    candidates = [r for r in rows if r["method"] == method]
    if fraction is not None:
        candidates = [r for r in candidates if abs(float(r["topk_fraction"]) - fraction) < 1e-12]
    return min(candidates, key=lambda r: float(r["final_train_objective"]))


def print_table(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"=== {title} ===")
    print(df[cols].to_csv(index=False).strip())


def main() -> None:
    strong = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)

    # 1) Event-scale selection at the primary 650-tick horizon.
    event_rows: list[dict] = []
    for gamma in (0.6, 0.8, 1.0):
        for jump0 in (0.0035, 0.0050, 0.0075):
            event_rows.append(
                run_named(
                    strong,
                    f"event_g{gamma}_q{jump0}",
                    "events",
                    650,
                    gamma=gamma,
                    threshold=0.025,
                    jump0=jump0,
                )
            )
    event_df = pd.DataFrame(event_rows)
    selected_event = min(event_rows, key=lambda r: float(r["final_train_objective"]))
    event_kwargs = {
        "gamma": float(selected_event["gamma"]),
        "threshold": float(selected_event["threshold"]),
        "jump0": float(selected_event["jump0"]),
    }
    print_table(
        "DETERMINISTIC EVENT REFINEMENT",
        event_df,
        [
            "configuration", "final_train_objective", "final_test_ce",
            "final_test_accuracy", "final_worst_class_accuracy", "payload_bits",
            "candidate_events", "events_per_message", "ever_fired_fraction",
        ],
    )
    print(
        "SELECTED_EVENT,"
        f"gamma={event_kwargs['gamma']},threshold={event_kwargs['threshold']},"
        f"jump0={event_kwargs['jump0']}"
    )

    # 2) Baseline selection at the same horizon, by training objective.
    base650: list[dict] = []
    for step in (0.02, 0.04, 0.08):
        base650.append(run_named(strong, f"full_{step}", "full", 650, step=step))
    for fraction in (0.005, 0.025):
        for step in (0.04, 0.08):
            base650.append(
                run_named(
                    strong,
                    f"ef_{fraction}_{step}",
                    "ef_topk",
                    650,
                    step=step,
                    topk_fraction=fraction,
                )
            )
    base650_df = pd.DataFrame(base650)
    full650 = best_by_train(base650, "full")
    ef05_650 = best_by_train(base650, "ef_topk", 0.005)
    ef25_650 = best_by_train(base650, "ef_topk", 0.025)
    print_table(
        "DETERMINISTIC 650 BASELINE AUDIT",
        base650_df,
        [
            "configuration", "final_train_objective", "final_test_ce",
            "final_test_accuracy", "final_worst_class_accuracy", "payload_bits",
            "whole_train_objective",
        ],
    )

    # 3) Exact repeatability check inside the single-thread environment.
    repeat_rows = []
    for rep in range(2):
        rr = run_named(strong, f"repeat_{rep}", "events", 650, **event_kwargs)
        repeat_rows.append(rr)
    repeat_df = pd.DataFrame(repeat_rows)
    check_cols = [
        "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "payload_bits", "candidate_events",
        "events_per_message",
    ]
    exact_repeat = all(
        repeat_df.loc[0, c] == repeat_df.loc[1, c] for c in check_cols
    )
    print_table("DETERMINISTIC REPEATABILITY", repeat_df, ["configuration"] + check_cols)
    print(f"EXACT_REPEATABILITY={exact_repeat}")

    selected650 = {
        "full": ("full", {"step": float(full650["step"])}),
        "ef_0p5": (
            "ef_topk",
            {"step": float(ef05_650["step"]), "topk_fraction": 0.005},
        ),
        "ef_2p5": (
            "ef_topk",
            {"step": float(ef25_650["step"]), "topk_fraction": 0.025},
        ),
        "events": ("events", event_kwargs),
    }

    # 4) Independent strong partitions.
    robust_rows: list[dict] = []
    # Use the already-computed selected seed-2400 rows rather than rerunning them.
    seed2400_rows = {
        "full": full650,
        "ef_0p5": ef05_650,
        "ef_2p5": ef25_650,
        "events": selected_event,
    }
    for name, r0 in seed2400_rows.items():
        rr = dict(r0)
        rr["configuration"] = name
        rr["data_seed"] = 2400
        robust_rows.append(rr)
    for data_seed in (2500, 2600):
        fed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=data_seed)
        for name, (method, kwargs) in selected650.items():
            rr = run_named(fed, name, method, 650, **kwargs)
            rr["data_seed"] = data_seed
            robust_rows.append(rr)
    robust_df = pd.DataFrame(robust_rows)
    print_table(
        "DETERMINISTIC STRONG ROBUSTNESS",
        robust_df,
        [
            "data_seed", "configuration", "final_test_ce", "final_test_accuracy",
            "final_worst_class_accuracy", "payload_bits", "whole_train_objective",
        ],
    )

    # 5) Heterogeneity sweep, with the strong seed-2400 points reused above.
    hetero_rows: list[dict] = []
    for name, r0 in seed2400_rows.items():
        rr = dict(r0)
        rr["configuration"] = name
        rr["regime"] = "strong"
        hetero_rows.append(rr)
    for regime in ("iid", "moderate", "extreme"):
        fed = make_multiclass_federation(root=DATA_ROOT, regime=regime, seed=2400)
        for name, (method, kwargs) in selected650.items():
            rr = run_named(fed, name, method, 650, **kwargs)
            rr["regime"] = regime
            hetero_rows.append(rr)
    hetero_df = pd.DataFrame(hetero_rows)
    regime_order = {"iid": 0, "moderate": 1, "strong": 2, "extreme": 3}
    hetero_df["_order"] = hetero_df["regime"].map(regime_order)
    hetero_df = hetero_df.sort_values(["_order", "configuration"]).drop(columns=["_order"])
    print_table(
        "DETERMINISTIC HETEROGENEITY",
        hetero_df,
        [
            "regime", "configuration", "final_test_ce", "final_test_accuracy",
            "final_worst_class_accuracy", "payload_bits", "whole_train_objective",
        ],
    )

    # 6) Horizon-specific long-run baseline audit on strong seed 2400.
    long_rows: list[dict] = []
    for step in (0.02, 0.04, 0.08):
        long_rows.append(run_named(strong, f"full_{step}", "full", 1200, step=step))
    for fraction in (0.005, 0.025):
        for step in (0.04, 0.08):
            long_rows.append(
                run_named(
                    strong,
                    f"ef_{fraction}_{step}",
                    "ef_topk",
                    1200,
                    step=step,
                    topk_fraction=fraction,
                )
            )
    long_rows.append(run_named(strong, "events", "events", 1200, **event_kwargs))
    long_df = pd.DataFrame(long_rows)
    full1200 = best_by_train(long_rows, "full")
    ef05_1200 = best_by_train(long_rows, "ef_topk", 0.005)
    ef25_1200 = best_by_train(long_rows, "ef_topk", 0.025)
    event1200 = [r for r in long_rows if r["configuration"] == "events"][0]
    print_table(
        "DETERMINISTIC LONG HORIZON",
        long_df,
        [
            "configuration", "final_train_objective", "final_test_ce",
            "final_test_accuracy", "final_worst_class_accuracy", "payload_bits",
            "whole_train_objective",
        ] + [f"class_{c}_accuracy" for c in range(10)],
    )

    # 7) Semantic class / compute-period rotations at strong and extreme skew.
    assignments = {
        "identity": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        "reverse": (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "mixed": (4, 7, 1, 9, 0, 6, 3, 8, 5, 2),
    }
    rotation_rows: list[dict] = []
    for regime in ("strong", "extreme"):
        for assignment_name, assignment in assignments.items():
            if regime == "strong" and assignment_name == "identity":
                fed = strong
            else:
                fed = make_permuted_federation(
                    regime=regime,
                    seed=2400,
                    dominant_classes=assignment,
                )
            for name in ("full", "ef_2p5", "events"):
                method, kwargs = selected650[name]
                rr = run_named(fed, name, method, 650, **kwargs)
                rr["regime"] = regime
                rr["assignment"] = assignment_name
                rotation_rows.append(rr)
    rotation_df = pd.DataFrame(rotation_rows)
    print_table(
        "DETERMINISTIC PERIOD-CLASS ROTATIONS",
        rotation_df,
        [
            "regime", "assignment", "configuration", "final_test_ce",
            "final_test_accuracy", "final_worst_class_accuracy", "payload_bits",
            "whole_train_objective",
        ],
    )

    # 8) Extreme-skew persistence using the long-horizon baseline selections.
    extreme = make_multiclass_federation(root=DATA_ROOT, regime="extreme", seed=2400)
    extreme_long_rows = [
        run_named(extreme, "full", "full", 1200, step=float(full1200["step"])),
        run_named(
            extreme,
            "ef_2p5",
            "ef_topk",
            1200,
            step=float(ef25_1200["step"]),
            topk_fraction=0.025,
        ),
        run_named(extreme, "events", "events", 1200, **event_kwargs),
    ]
    extreme_long_df = pd.DataFrame(extreme_long_rows)
    print_table(
        "DETERMINISTIC EXTREME LONG HORIZON",
        extreme_long_df,
        [
            "configuration", "final_test_ce", "final_test_accuracy",
            "final_worst_class_accuracy", "payload_bits", "whole_train_objective",
        ] + [f"class_{c}_accuracy" for c in range(10)],
    )

    # 9) Layer traffic from the deterministic selected event point.
    layer_cols = [
        "W1_events_per_param", "W1_never_fired", "b1_events_per_param", "b1_never_fired",
        "W2_events_per_param", "W2_never_fired", "b2_events_per_param", "b2_never_fired",
        "W3_events_per_param", "W3_never_fired", "b3_events_per_param", "b3_never_fired",
    ]
    print("=== DETERMINISTIC SELECTED LAYER ACTIVITY ===")
    print(
        pd.DataFrame([{c: selected_event[c] for c in layer_cols}]).to_csv(index=False).strip()
    )

    # Persist locally for interactive/local reruns. CI logs are the immutable observed source for this run.
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(RESULT_ROOT / "det_event_refinement.csv", index=False)
    base650_df.to_csv(RESULT_ROOT / "det_baseline_650.csv", index=False)
    robust_df.to_csv(RESULT_ROOT / "det_strong_robustness.csv", index=False)
    hetero_df.to_csv(RESULT_ROOT / "det_heterogeneity.csv", index=False)
    long_df.to_csv(RESULT_ROOT / "det_long_horizon.csv", index=False)
    rotation_df.to_csv(RESULT_ROOT / "det_period_class_rotations.csv", index=False)
    extreme_long_df.to_csv(RESULT_ROOT / "det_extreme_long.csv", index=False)


if __name__ == "__main__":
    main()
