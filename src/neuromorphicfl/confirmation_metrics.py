"""Descriptive paired metrics for the fixed STC confirmation experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def add_regularization_columns(frame: pd.DataFrame, regularization: float) -> pd.DataFrame:
    result = frame.copy()
    prefix = "final_" if "final_train_objective" in result else ""
    penalty = result[prefix + "train_objective"] - result[prefix + "train_ce"]
    if (penalty < -1e-8).any():
        raise ValueError("Training objective is smaller than cross-entropy")
    result[prefix + "train_l2_penalty"] = np.maximum(penalty, 0)
    result[prefix + "parameter_l2_norm"] = np.sqrt(2 * np.maximum(penalty, 0) / regularization)
    return result


def validate_native_traffic(summary: dict, history: pd.DataFrame, *, n_clients: int = 10) -> None:
    initial = 32 * int(summary["dimension"]) + 64
    for field in ("round", "uplink_packetized_bits", "stc_downlink_packetized_bits",
                  "unicast_total_bits", "broadcast_total_bits"):
        values = history[field].to_numpy()
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError("Native traffic counters must be finite nonnegative integers")
        if np.any(np.diff(values) < 0):
            raise ValueError("Native traffic counters decreased")
    upload = history.uplink_packetized_bits.astype(np.int64)
    server = history.stc_downlink_packetized_bits.astype(np.int64)
    rounds = history["round"].astype(np.int64)
    unicast = upload + n_clients * (initial + 32 * rounds + server)
    broadcast = upload + initial + server
    if not np.array_equal(history.unicast_total_bits, unicast) or not np.array_equal(history.broadcast_total_bits, broadcast):
        raise ValueError("Native STC prefix traffic does not reproduce")
    if (int(summary["uplink_packetized_bits"]) != int(upload.iloc[-1])
            or int(summary["stc_downlink_packetized_bits"]) != int(server.iloc[-1])
            or int(summary["unicast_hybrid_total_bits"]) != int(unicast.iloc[-1])
            or int(summary["broadcast_total_bits"]) != int(broadcast.iloc[-1])):
        raise ValueError("Native STC endpoint traffic does not reproduce")


def legacy_traffic_history(
    summary: dict, history: pd.DataFrame, *, n_clients: int = 10,
) -> pd.DataFrame:
    """Recover exact prefix costs only when the archived sync mode is uniform.

    With replay in every round, cumulative upload equals the replay stream.
    With checkpoints in every round, downlink length is constant. Mixed-mode
    archives lack per-round choices and cannot be reconstructed exactly.
    """
    result = history.copy()
    rounds = int(summary["rounds"])
    replay, checkpoints = int(summary["replay_rounds"]), int(summary["checkpoint_rounds"])
    initial = 32 * int(summary["dimension"]) + 64
    if replay == rounds and checkpoints == 0:
        stream = result["uplink_packetized_bits"].astype(np.int64)
        mode = "exact_all_replay"
    elif checkpoints == rounds and replay == 0:
        stream = result["round"].astype(np.int64) * initial
        mode = "exact_all_checkpoints"
    else:
        raise ValueError("Mixed replay/checkpoint archive has no exact prefix traffic record")
    result["unicast_downlink_bits"] = n_clients * (
        initial + 32 * result["round"].astype(np.int64) + stream
    )
    result["unicast_total_bits"] = result["uplink_packetized_bits"] + result["unicast_downlink_bits"]
    result["broadcast_total_bits"] = result["uplink_packetized_bits"] + initial + stream
    result["traffic_history_source"] = mode
    for field, original in (("unicast_total_bits", "unicast_hybrid_total_bits"),
                            ("broadcast_total_bits", "broadcast_total_bits")):
        if int(result.iloc[-1][field]) != int(summary[original]):
            raise ValueError(f"Archived endpoint disagrees with recovered {field}")
    return result


def interval(values, *, ratio: bool = False) -> dict:
    """Unadjusted paired Student-t interval; ratios use the log scale."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (ratio and np.any(values <= 0)):
        raise ValueError("Paired values must be finite; ratios must be positive")
    n = values.size
    if n == 0:
        return {"n": 0, "estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    work = np.log(values) if ratio else values
    mean = float(work.mean())
    half = float(t.ppf(0.975, n - 1) * work.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    convert = np.exp if ratio else float
    return {"n": int(n), "estimate": float(convert(mean)),
            "ci95_low": float(convert(mean - half)), "ci95_high": float(convert(mean + half))}


def paired_endpoints(runs: pd.DataFrame, points: tuple[str, ...], references: tuple[str, ...]):
    rows = []
    metrics = (
        ("test_accuracy_difference_pp", "final_test_accuracy", False, 100),
        ("test_ce_difference", "final_test_ce", False, 1),
        ("train_objective_difference", "final_train_objective", False, 1),
        ("train_l2_penalty_difference", "final_train_l2_penalty", False, 1),
        ("unicast_traffic_ratio", "unicast_hybrid_total_bits", True, 1),
        ("broadcast_traffic_ratio", "broadcast_total_bits", True, 1),
        ("upload_traffic_ratio", "uplink_packetized_bits", True, 1),
        ("downlink_traffic_ratio", "unicast_hybrid_downlink_bits", True, 1),
    )
    for point in points:
        candidate = runs[runs.comparison_id == point].set_index("partition_seed")
        if candidate.index.has_duplicates:
            raise ValueError("Duplicate confirmation seed")
        for reference in references:
            baseline = runs[runs.comparison_id == reference].set_index("partition_seed")
            if baseline.index.has_duplicates or set(candidate.index) != set(baseline.index):
                raise ValueError("Paired seed sets differ")
            for seed in sorted(candidate.index):
                a, b = candidate.loc[seed], baseline.loc[seed]
                for name, column, ratio, scale in metrics:
                    value = float(a[column] / b[column]) if ratio else float(scale * (a[column] - b[column]))
                    rows.append({"point_id": point, "reference": reference,
                                 "partition_seed": seed, "metric": name, "value": value})
    pairs = pd.DataFrame(rows)
    summaries = []
    for (point, reference, metric), group in pairs.groupby(["point_id", "reference", "metric"], sort=False):
        ratio = metric.endswith("_ratio")
        summaries.append({"point_id": point, "reference": reference, "metric": metric,
                          **interval(group.value, ratio=ratio),
                          "scale": "geometric_mean_ratio" if ratio else "mean_paired_difference",
                          "primary_endpoint": point == "stc_primary" and reference == "event" and metric == "test_accuracy_difference_pp",
                          "interval_scope": "descriptive_paired_t_unadjusted"})
    return pairs, pd.DataFrame(summaries)


def target_crossings(
    history: pd.DataFrame, *, targets: tuple[float, ...], horizon: int,
    comparison_id: str, partition_seed: int,
) -> list[dict]:
    if not history["round"].is_monotonic_increasing or history["round"].duplicated().any():
        raise ValueError("Accuracy history rounds must be strictly increasing")
    rows = []
    for target in targets:
        hits = history[history.test_accuracy >= target]
        row = {"comparison_id": comparison_id, "partition_seed": partition_seed,
               "target_accuracy": target, "reached": not hits.empty,
               "first_sampled_round": float("nan"), "censor_round": horizon if hits.empty else float("nan"),
               "upload_bits": float("nan"), "unicast_total_bits": float("nan"),
               "broadcast_total_bits": float("nan")}
        if not hits.empty:
            hit = hits.iloc[0]
            row.update(first_sampled_round=int(hit["round"]),
                       upload_bits=int(hit["uplink_packetized_bits"]),
                       unicast_total_bits=int(hit["unicast_total_bits"]),
                       broadcast_total_bits=int(hit["broadcast_total_bits"]))
        rows.append(row)
    return rows


def paired_targets(crossings: pd.DataFrame, points: tuple[str, ...], references: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for point in points:
        for reference in references:
            for target in sorted(crossings.target_accuracy.unique()):
                a = crossings[(crossings.comparison_id == point) & (crossings.target_accuracy == target)].set_index("partition_seed")
                b = crossings[(crossings.comparison_id == reference) & (crossings.target_accuracy == target)].set_index("partition_seed")
                if a.index.has_duplicates or b.index.has_duplicates or set(a.index) != set(b.index):
                    raise ValueError("Target-comparison seed sets differ")
                joined = a.join(b, lsuffix="_stc", rsuffix="_ref")
                common = joined[joined.reached_stc & joined.reached_ref]
                ratios = common.unicast_total_bits_stc / common.unicast_total_bits_ref
                rows.append({"point_id": point, "reference": reference, "target_accuracy": target,
                             "n_total": len(joined), "n_stc_reached": int(a.reached.sum()),
                             "n_reference_reached": int(b.reached.sum()), "n_common_hits": len(common),
                             **interval(ratios, ratio=True),
                             "estimand": "unicast_ratio_conditional_on_both_reaching_target",
                             "interval_scope": "descriptive_paired_t_unadjusted"})
    return pd.DataFrame(rows)
