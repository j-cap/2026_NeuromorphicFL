from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from neuromorphicfl.bidirectional_protocol import (
    dense_downlink_accounting,
    q2_total_row,
    run_sparse_cnn_trace,
    simulate_sparse_downlink,
)
from neuromorphicfl.fmnist_multiclass_benchmark import make_multiclass_federation
from neuromorphicfl.fmnist_cnn_benchmark import LAYOUT, run_federated_cnn


DATA_ROOT = Path("data/fashion-mnist")
OUT = Path("experiments/results/q2_bidirectional_protocol")
OUT.mkdir(parents=True, exist_ok=True)

fed = make_multiclass_federation(root=DATA_ROOT, regime="strong", seed=2400)

# Frozen 15A/Q1-compatible sparse methods.
event_trace = run_sparse_cnn_trace(
    federation=fed,
    method="events",
    n_ticks=1200,
    rho=0.999,
    gamma=1.0,
    threshold=0.025,
    jump0=0.01,
    schedule_exponent=0.5,
)
ef_trace = run_sparse_cnn_trace(
    federation=fed,
    method="ef_topk",
    n_ticks=1200,
    step=0.08,
    topk_fraction=0.025,
)

# Shadow runs verify that protocol instrumentation does not change the learning trajectory.
event_shadow = run_federated_cnn(
    federation=fed,
    method="events",
    n_ticks=1200,
    rho=0.999,
    gamma=1.0,
    threshold=0.025,
    jump0=0.01,
    schedule_exponent=0.5,
    eval_stride=100,
)
ef_shadow = run_federated_cnn(
    federation=fed,
    method="ef_topk",
    n_ticks=1200,
    step=0.08,
    topk_fraction=0.025,
    eval_stride=100,
)

shadow = pd.DataFrame(
    [
        {
            "method": "events",
            "trace_test_ce": event_trace.metrics["final_test_ce"],
            "shadow_test_ce": event_shadow["final_test_ce"],
            "trace_accuracy": event_trace.metrics["final_test_accuracy"],
            "shadow_accuracy": event_shadow["final_test_accuracy"],
            "abs_ce_difference": abs(float(event_trace.metrics["final_test_ce"]) - float(event_shadow["final_test_ce"])),
            "abs_accuracy_difference": abs(float(event_trace.metrics["final_test_accuracy"]) - float(event_shadow["final_test_accuracy"])),
        },
        {
            "method": "ef_topk",
            "trace_test_ce": ef_trace.metrics["final_test_ce"],
            "shadow_test_ce": ef_shadow["final_test_ce"],
            "trace_accuracy": ef_trace.metrics["final_test_accuracy"],
            "shadow_accuracy": ef_shadow["final_test_accuracy"],
            "abs_ce_difference": abs(float(ef_trace.metrics["final_test_ce"]) - float(ef_shadow["final_test_ce"])),
            "abs_accuracy_difference": abs(float(ef_trace.metrics["final_test_accuracy"]) - float(ef_shadow["final_test_accuracy"])),
        },
    ]
)
shadow.to_csv(OUT / "shadow_reproduction.csv", index=False)

rows: list[dict[str, float | int | str]] = []
client_frames: list[pd.DataFrame] = []

for trace in (event_trace, ef_trace):
    dense_down = dense_downlink_accounting(trace)
    rows.append(q2_total_row(trace, dense_down))

    for mode in ("broadcast", "unicast_replay", "hybrid"):
        down, client = simulate_sparse_downlink(trace, mode=mode)
        rows.append(q2_total_row(trace, down))
        client.insert(0, "method", trace.method)
        client.insert(1, "downlink_mode", mode)
        client_frames.append(client)

# Dense reference.  The completion schedule is identical for all methods.
dense = run_federated_cnn(
    federation=fed,
    method="full",
    n_ticks=1200,
    step=0.08,
    eval_stride=100,
)

sync_ops = event_trace.sync_opportunities
n_completions = len(sync_ops)
last_completion = np.zeros(fed.n_clients, dtype=np.int64)
for sequence, (_, client, _) in enumerate(sync_ops, start=1):
    last_completion[client] = sequence
final_stale_clients = int(np.sum(last_completion < n_completions))
checkpoint_payload = 32 * LAYOUT.dimension
checkpoint_packet = checkpoint_payload + 64
n_dense_syncs = n_completions + final_stale_clients

dense_uplink_payload = int(dense["payload_bits"])
dense_uplink_packetized = dense_uplink_payload + 64 * int(dense["messages"])
dense_down_payload = n_dense_syncs * checkpoint_payload
dense_down_packetized = n_dense_syncs * (checkpoint_packet + 32)
rows.append(
    {
        "method": "full",
        "downlink_mode": "dense_sync",
        "final_test_ce": float(dense["final_test_ce"]),
        "final_test_accuracy": float(dense["final_test_accuracy"]),
        "final_worst_class_accuracy": float(dense["final_worst_class_accuracy"]),
        "uplink_payload_bits": dense_uplink_payload,
        "uplink_packetized_bits": int(dense_uplink_packetized),
        "downlink_payload_bits": int(dense_down_payload),
        "downlink_packetized_bits": int(dense_down_packetized),
        "total_payload_bits": int(dense_uplink_payload + dense_down_payload),
        "total_packetized_bits": int(dense_uplink_packetized + dense_down_packetized),
        "checkpoint_syncs": int(n_dense_syncs),
        "replay_syncs": 0,
        "empty_syncs": 0,
        "max_sync_error": 0.0,
    }
)

summary = pd.DataFrame(rows)
summary["uplink_Mbit"] = summary["uplink_packetized_bits"] / 1e6
summary["downlink_Mbit"] = summary["downlink_packetized_bits"] / 1e6
summary["total_Mbit"] = summary["total_packetized_bits"] / 1e6
summary.to_csv(OUT / "q2_summary.csv", index=False)

clients = pd.concat(client_frames, ignore_index=True)
clients["downlink_Mbit"] = clients["downlink_packetized_bits"] / 1e6
clients.to_csv(OUT / "q2_client_breakdown.csv", index=False)

# Relative total-traffic factors against the strongest directly comparable EF modes.
def total(method: str, mode: str) -> float:
    row = summary[(summary.method == method) & (summary.downlink_mode == mode)].iloc[0]
    return float(row.total_packetized_bits)

comparisons = pd.DataFrame(
    [
        {
            "comparison": "EF dense sync / V1 dense sync",
            "traffic_ratio": total("ef_topk", "dense_sync") / total("events", "dense_sync"),
        },
        {
            "comparison": "EF sparse broadcast / V1 sparse broadcast",
            "traffic_ratio": total("ef_topk", "broadcast") / total("events", "broadcast"),
        },
        {
            "comparison": "EF unicast replay / V1 unicast replay",
            "traffic_ratio": total("ef_topk", "unicast_replay") / total("events", "unicast_replay"),
        },
        {
            "comparison": "EF hybrid / V1 hybrid",
            "traffic_ratio": total("ef_topk", "hybrid") / total("events", "hybrid"),
        },
        {
            "comparison": "Dense total / V1 hybrid",
            "traffic_ratio": total("full", "dense_sync") / total("events", "hybrid"),
        },
    ]
)
comparisons.to_csv(OUT / "q2_comparisons.csv", index=False)

print("=== Q2 SHADOW CHECK ===")
print(shadow.to_string(index=False))
print("\n=== Q2 BIDIRECTIONAL SUMMARY ===")
print(
    summary[
        [
            "method",
            "downlink_mode",
            "final_test_ce",
            "final_test_accuracy",
            "final_worst_class_accuracy",
            "uplink_Mbit",
            "downlink_Mbit",
            "total_Mbit",
            "checkpoint_syncs",
            "replay_syncs",
            "max_sync_error",
        ]
    ].to_string(index=False)
)
print("\n=== Q2 TRAFFIC RATIOS ===")
print(comparisons.to_string(index=False))
print("\n=== Q2 HYBRID CLIENT BREAKDOWN ===")
print(clients[clients.downlink_mode == "hybrid"].to_string(index=False))
