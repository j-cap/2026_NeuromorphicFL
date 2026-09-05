from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ROOT = Path("experiments/results/final_baseline_campaign")

SUMMARY_NAMES = {
    "eval": "observed_heldout_summary.csv",
    "traffic": "observed_traffic_matched_summary.csv",
}

METRICS = [
    "final_test_ce", "final_test_accuracy", "final_worst_class_accuracy",
    "uplink_Mbit", "broadcast_total_Mbit", "unicast_total_Mbit",
]


def load(tag: str) -> pd.DataFrame:
    files = sorted(ROOT.glob(f"*_{tag}.csv"))
    if not files:
        raise RuntimeError(f"no files found for tag {tag}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def write_heldout_summary(df: pd.DataFrame, tag: str) -> Path:
    """Write the aggregate artifact consumed by the paper evidence build."""
    counts = df.groupby(["architecture", "method"], dropna=False).size()
    if not (counts == 3).all():
        raise RuntimeError(
            f"{tag} requires exactly three held-out rows per architecture/method; "
            f"observed {counts.to_dict()}"
        )
    aggregate = (
        df.groupby(["architecture", "method"], as_index=False, dropna=False)
        .agg(
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in METRICS
                for stat in ("mean", "std")
            },
            configuration_value=("configuration_value", "first"),
        )
        .sort_values(["architecture", "method"])
    )
    if tag == "traffic":
        heldout_path = ROOT / SUMMARY_NAMES["eval"]
        if not heldout_path.exists():
            raise RuntimeError(
                "traffic summary requires the frozen held-out summary so the "
                "same Event-FedAvg rows can be included without rerunning them"
            )
        event_rows = pd.read_csv(heldout_path).query("method == 'event'")
        aggregate = (
            pd.concat([aggregate, event_rows], ignore_index=True)
            .sort_values(["architecture", "method"])
        )
    output = ROOT / SUMMARY_NAMES[tag]
    aggregate.to_csv(output, index=False, float_format="%.9f")
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, choices=["tune", "eval", "traffic"])
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
        agg = df.groupby(["architecture", "method"])[METRICS].agg(["mean", "std"])
        print(agg.to_string())
        summary = write_heldout_summary(df, a.tag)
        print(f"\nwrote {summary}")

    df.to_csv(ROOT / f"combined_{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
