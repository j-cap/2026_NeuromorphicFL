"""Reproduce the short codec analysis without changing its frozen sources.

python experiments/event_codec_pilot_audit.py
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

import event_codec_pilot as pilot
import stc_cifar10_development as development
from stc_cifar10_development_audit import recover_recorded_bytes
import numpy as np
import pandas as pd

ARCHIVED_POINTS = {"event": "event_reference", "stc_primary": "stc_u1_d1_g2",
                   "stc_sparse": "stc_u0p25_d0p25_g2"}


def pilot_semantic_audit(science):
    expected = {"partition_seed": 4500, "train_seed": 84500,
                "evaluation_scope": "development_training_only", "contract_sha256": pilot.digest(science)}
    for point in pilot.POINTS:
        tables = {kind: pd.read_csv(path, float_precision="round_trip")
                  for kind, path in pilot.paths(point).items() if kind != "receipt"}
        for kind, table in tables.items():
            for field, value in {**expected, "point": point}.items():
                if not np.all(table[field] == value):
                    raise RuntimeError(f"Pilot metadata mismatch: {point}/{kind}/{field}")
        if tables["history"]["round"].tolist() != [1, 30, 60, 90, 120]:
            raise RuntimeError("Training diagnostic coverage differs")
        if not np.all(tables["summary"].model_sha256 == tables["rounds"].iloc[-1].model_sha256):
            raise RuntimeError("Summary and last-round model fingerprints differ")
        if not np.all(tables["summary"]["rounds"] == 120) or not tables["summary"].roundtrip_verified.all():
            raise RuntimeError("Summary horizon or reconstruction status differs")


def archived_prefix_audit(science):
    root = development.OUT
    original = development.load_contract(root)
    if science["data_sha256"] != original["data_sha256"] or science["runtime"] != original["runtime"]:
        raise RuntimeError("Pilot and archived development data/runtime records differ")
    frozen = json.loads((root / "frozen_selection.json").read_text())
    output = []
    for point, old_id in ARCHIVED_POINTS.items():
        old_point = development.POINT_BY_ID[old_id]
        summary_path, history_path, receipt_path = development.paths(old_point)
        receipt_bytes, _ = recover_recorded_bytes(receipt_path.read_bytes(), frozen["receipt_sha256"][old_id])
        receipt = json.loads(receipt_bytes)
        for kind, path in (("summary", summary_path), ("history", history_path)):
            recover_recorded_bytes(path.read_bytes(), receipt[kind + "_sha256"])
        config = science["protocol"]["points"][point]
        prior_config = development.asdict(development.config_for(old_point))
        omitted = {"rounds", "eval_stride"}
        if point == "event":
            omitted |= {"stc_upload_fraction", "stc_download_fraction"}
        if any(config[key] != value for key, value in prior_config.items() if key not in omitted):
            raise RuntimeError("Pilot changed a relevant archived training setting")
        old = pd.read_csv(history_path, float_precision="round_trip")
        history = pd.read_csv(pilot.paths(point)["history"], float_precision="round_trip")
        rounds = pd.read_csv(pilot.paths(point)["rounds"], float_precision="round_trip")
        shared = history.merge(old, on="round", suffixes=("_pilot", "_archive"))
        if shared["round"].tolist() != [1, 60, 120]:
            raise RuntimeError("Unexpected archived prefix coverage")
        for field in ("train_objective", "train_ce"):
            if not np.array_equal(shared[field + "_pilot"].to_numpy(), shared[field + "_archive"].to_numpy()):
                raise RuntimeError("Pilot training diagnostic differs from archived prefix")
        for rnd in shared["round"]:
            prefix = rounds[rounds["round"] <= rnd]
            prior = old[old["round"] == rnd].iloc[0]
            if int(prefix.legacy_uplink_bits.sum()) != int(prior.uplink_packetized_bits):
                raise RuntimeError("Native upload prefix differs from archive")
            if point != "event" and int(prefix.legacy_server_bits.sum()) != int(prior.stc_downlink_packetized_bits):
                raise RuntimeError("Native STC download prefix differs from archive")
        output.append({"point": point, "archived_point": old_id, "matching_rounds": [1, 60, 120],
                       "training_diagnostics_exact": True, "native_traffic_exact": True,
                       "original_receipt_and_file_hashes_verified": True})
    return output


def archived_event_ceiling():
    row = pd.read_csv(development.paths(development.REFERENCE)[0], float_precision="round_trip").iloc[0]
    d, rounds, packets, events = (int(row.dimension), int(row["rounds"]), int(row.messages), int(row.coordinate_events))
    # Fixed b=4 Rice: sum zero gaps <= packets*d-events; each pulse
    # adds a unary terminator, 4 remainder bits and one sign bit.
    up_payload = (packets * d - events) // 16 + 6 * events
    upload = 8 * ((up_payload + 128 * packets + 7 * packets) // 8)
    # The selected exact downlink cannot exceed its serialized ordered fallback.
    server = 8 * ((rounds * d + 2 * events + 128 * rounds + 7 * rounds) // 8)
    fixed = 10 * (32 * d + 64) + 32 * 10 * rounds
    total = upload + 10 * server + fixed
    return {"status": "calculated_format_ceiling_not_measured_reencoding",
            "rounds": rounds, "coordinate_events": events, "uplink_upper_bits": upload,
            "server_stream_upper_bits": server, "fixed_bits": fixed,
            "unicast_total_upper_bits": total, "unicast_saving_lower_percent": 100 * (1 - total / int(row.unicast_hybrid_total_bits))}


def plot(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    selections = [
        ([table[(table.point == "event") & (table.variant == value)].iloc[0]
          for value in ("legacy", "uplink_codec_replay", "aggregate_downlink", "both_codecs")],
         ["Original", "Uplink\ncodec", "Aggregate\ndownlink", "Both\ncodecs"], "Event: encoding ablation"),
        ([table[(table.point == point) & (table.variant == value)].iloc[0]
          for point in ("stc_primary", "stc_sparse") for value in ("native_stc", "lossless_stc")],
         ["p=1\nnative", "p=1\nrecoded", "p=0.25\nnative", "p=0.25\nrecoded"], "STC: lossless coding controls"),
    ]
    fixed = (10 * (32 * 20570 + 64) + 32 * 10 * 120) / 1e6
    for axis, (rows, labels, title) in zip(axes, selections):
        x = np.arange(len(rows))
        up = np.array([r.uplink_bits for r in rows]) / 1e6
        server = np.array([r.unicast_downlink_bits for r in rows]) / 1e6 - fixed
        axis.bar(x, up, color="#0072B2", label="Uplink")
        axis.bar(x, server, bottom=up, color="#E69F00", label="Replicated server stream")
        axis.bar(x, np.full(len(rows), fixed), bottom=up + server, color="#BBBBBB", label="Initialization + requests")
        totals = up + server + fixed
        for j, total in enumerate(totals):
            axis.text(j, total + max(totals) * .025, f"{total:.2f}", ha="center", fontsize=9)
        axis.set_xticks(x, labels)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Total unicast traffic (Mbit)")
        axis.set_ylim(0, max(totals) * 1.15)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Codec pilot: 120 rounds on one development partition", fontsize=11, y=.98)
    fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.23, wspace=.3)
    fig.savefig(out / "analysis_codec_traffic.png", dpi=180)
    fig.savefig(out / "analysis_codec_traffic.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    pilot.analyze()
    out = pilot.OUT
    science = pilot.load_contract()
    pilot_semantic_audit(science)
    prefixes = archived_prefix_audit(science)
    event = pd.read_csv(pilot.paths("event")["rounds"], float_precision="round_trip")
    modes = Counter()
    for encoded in event.uplink_modes:
        modes.update(json.loads(encoded))
    events = int(event.coordinate_events.sum())
    support = int(event.aggregate_nonzeros.sum())
    mass = int(event.aggregate_l1.sum())
    if int(event.active_clients.sum()) != 1200 or sum(modes.values()) != 1200:
        raise RuntimeError("Client-message coverage differs")
    if not np.array_equal(event.legacy_uplink_bits, 16 * event.coordinate_events + 64 * event.active_clients):
        raise RuntimeError("Historical Event packet accounting differs")
    ordered = 128 + 8 * np.ceil((20570 + 2 * event.coordinate_events) / 8).astype(np.int64)
    if not np.array_equal(event.ordered_downlink_bits, ordered):
        raise RuntimeError("Ordered-pulse frame length differs")
    if not np.array_equal(event.exact_downlink_bits, np.minimum(event.aggregate_exact_downlink_bits, event.ordered_downlink_bits)):
        raise RuntimeError("Exact server-message selection differs")
    if not np.all(event.downlink_mode == "counts_patch") or not np.all(event.aggregate_l1 >= event.aggregate_nonzeros):
        raise RuntimeError("Unexpected aggregate encoding or mass")
    if not np.all(event.aggregate_l1 <= event.coordinate_events) or not np.allclose(
            event.cancellation_fraction, 1 - event.aggregate_l1 / event.coordinate_events, rtol=1e-14, atol=1e-14):
        raise RuntimeError("Pulse cancellation accounting differs")
    weighted_cancellation = 1 - mass / events
    extra = int((event.exact_downlink_bits - event.integer_only_downlink_bits).sum())
    table = pd.read_csv(out / "comparison.csv", float_precision="round_trip")
    keys = table[["point", "variant", "uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits", "unicast_saving_percent"]].copy()
    for column in ("uplink", "unicast_downlink", "unicast_total", "broadcast_total"):
        keys[column + "_Mbit"] = keys[column + "_bits"] / 1e6
    pilot.atomic_csv(keys, out / "analysis_key_points.csv")
    windows = []
    for start in (1, 31, 61, 91):
        rows = event[event["round"].between(start, start + 29)]
        old_up = int(rows.legacy_uplink_bits.sum())
        windows.append({"first_round": start, "last_round": start + 29,
                        "legacy_uplink_bits": old_up, "codec_uplink_bits": int(rows.codec_uplink_bits.sum()),
                        "exact_server_stream_bits": int(rows.exact_downlink_bits.sum()),
                        "upload_saving_percent": 100 * (1 - rows.codec_uplink_bits.sum() / old_up),
                        "server_stream_saving_percent": 100 * (1 - rows.exact_downlink_bits.sum() / old_up),
                        "weighted_cancellation_fraction": 1 - rows.aggregate_l1.sum() / rows.coordinate_events.sum()})
    pilot.atomic_csv(pd.DataFrame(windows), out / "analysis_windows.csv")
    exact_total = int(table[(table.point == "event") & (table.variant == "both_codecs")].iloc[0].unicast_total_bits)
    audit = {"pilot_source_and_receipt_verification": True, "matching_recorded_data_and_runtime": True,
             "archived_prefixes": prefixes, "event_upload_modes": dict(modes),
             "coordinate_events": events, "aggregate_nonzeros": support, "aggregate_l1": mass,
             "weighted_cancellation_fraction": weighted_cancellation,
             "bits_per_uplink_pulse": float(event.codec_uplink_bits.sum() / events),
             "exact_aggregate_downlink_rounds": 120, "roundoff_corrected_coordinates": int(event.correction_nonzeros.sum()),
             "max_canonical_float32_error": float(event.canonical_max_abs_error.max()),
             "exactness_extra_server_bits": extra,
             "exactness_extra_fraction_of_total_unicast": 10 * extra / exact_total,
             "archived_1800_round_event_format_ceiling": archived_event_ceiling(),
             "limits": ["one development partition and first 120 rounds", "no test-accuracy or convergence conclusion",
                        "packet decoding checked during recorded execution; full packet streams not saved",
                        "recorded raw-data hashes match archive; data not independently downloaded by this analysis"]}
    pilot.atomic_json(audit, out / "analysis_codec_audit.json")
    plot(table, out)
    print("Archived prefixes and independent Event frame-accounting identities match.")
    print("Weighted pulse cancellation:", weighted_cancellation)


if __name__ == "__main__":
    main()
