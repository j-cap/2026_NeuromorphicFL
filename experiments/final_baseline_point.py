from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig, run_final_baseline


DATA = Path("data/fashion-mnist")
OUT = Path("experiments/results/final_baseline_campaign")
OUT.mkdir(parents=True, exist_ok=True)


def build_config(architecture: str, method: str, value: float | None) -> FinalBaselineConfig:
    if architecture == "mlp":
        base = dict(
            local_steps=5,
            local_lr=0.1,
            rounds=150,
            eval_stride=15,
            threshold=0.025,
            jump0=0.005,
            jump_scale=100.0,
            jump_exponent=0.1,
        )
    else:
        base = dict(
            local_steps=5,
            local_lr=0.1,
            rounds=80,
            eval_stride=10,
            threshold=0.025,
            jump0=0.01,
            jump_scale=100.0,
            jump_exponent=0.3,
        )
    if method == "ef_topk":
        base["topk_fraction"] = float(value)
    if method == "strom":
        base["strom_threshold"] = float(value)
    return FinalBaselineConfig(**base)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=["mlp", "cnn"], required=True)
    p.add_argument("--method", choices=["event", "strom", "ef_topk", "sign_ef", "dense"], required=True)
    p.add_argument("--value", type=float, default=None)
    p.add_argument("--partition-seed", type=int, default=2400)
    p.add_argument("--train-seed", type=int, default=None)
    p.add_argument("--tag", default="point")
    a = p.parse_args()

    train_seed = int(a.train_seed if a.train_seed is not None else 70000 + a.partition_seed)
    fed = make_multiclass_federation(root=DATA, regime="strong", seed=a.partition_seed)
    cfg = build_config(a.architecture, a.method, a.value)
    result = run_final_baseline(
        federation=fed,
        architecture=a.architecture,
        method=a.method,
        config=cfg,
        seed=train_seed,
    )
    result["partition_seed"] = a.partition_seed
    result["train_seed"] = train_seed
    result["configuration_value"] = a.value if a.value is not None else float("nan")
    result["tag"] = a.tag

    frame = pd.DataFrame([result])
    suffix = f"{a.architecture}_{a.method}_{a.value if a.value is not None else 'fixed'}_p{a.partition_seed}_t{train_seed}_{a.tag}"
    path = OUT / f"{suffix}.csv"
    frame.to_csv(path, index=False)

    cols = [
        "architecture", "method", "configuration_value", "partition_seed", "train_seed",
        "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "uplink_packetized_bits",
        "broadcast_total_bits", "unicast_hybrid_total_bits",
        "replay_rounds", "checkpoint_rounds",
    ]
    print(frame[cols].to_string(index=False))


if __name__ == "__main__":
    main()
