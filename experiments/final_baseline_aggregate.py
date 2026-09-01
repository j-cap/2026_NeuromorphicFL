from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ROOT = Path("experiments/results/final_baseline_campaign")


def load(tag: str) -> pd.DataFrame:
    files = sorted(ROOT.glob(f"*_{tag}.csv"))
    if not files:
        raise RuntimeError(f"no files found for tag {tag}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    df = load(a.tag)
    df["uplink_Mbit"] = df.uplink_packetized_bits / 1e6
    df["broadcast_total_Mbit"] = df.broadcast_total_bits / 1e6
    df["unicast_total_Mbit"] = df.unicast_hybrid_total_bits / 1e6

    print("=== ALL RUNS ===")
    cols = [
        "architecture", "method", "configuration_value", "partition_seed",
        "final_train_objective", "final_test_ce", "final_test_accuracy",
        "final_worst_class_accuracy", "uplink_Mbit", "broadcast_total_Mbit",
        "unicast_total_Mbit", "replay_rounds", "checkpoint_rounds",
    ]
    print(df[cols].sort_values(["architecture", "method", "partition_seed", "final_train_objective"]).to_string(index=False))

    if a.tag == "tune":
        print("\n=== BEST DEVELOPMENT CONFIG BY FAMILY ===")
        best = (
            df.sort_values("final_train_objective")
            .groupby(["architecture", "method"], as_index=False)
            .first()
        )
        print(best[cols].sort_values(["architecture", "final_train_objective"]).to_string(index=False))
    else:
        print("\n=== HELD-OUT AGGREGATES ===")
        metrics = [
            "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
            "uplink_Mbit", "broadcast_total_Mbit", "unicast_total_Mbit",
        ]
        agg = df.groupby(["architecture", "method"])[metrics].agg(["mean", "std"])
        print(agg.to_string())

    df.to_csv(ROOT / f"combined_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
