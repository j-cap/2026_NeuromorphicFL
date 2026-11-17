from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import itertools
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.event_codec_pilot import config_for, POINTS
from experiments.stc_stage1 import smoke_federation
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.event_codec import (
    decode_event_downlink, decode_pulses, decode_stc_lossless,
    encode_event_downlink, encode_pulses, encode_stc_lossless, pulse_aggregate,
)
from neuromorphicfl.event_codec_pilot import run_codec_pilot, traffic_summary
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from neuromorphicfl.stc import compress_encode_stc, decode_stc


class PulseCodecTests(unittest.TestCase):
    def test_hand_encoded_rice_message_and_complete_wire_cost(self):
        values = np.array([0, -1, 0, 0, 1, 0, 0, 0], dtype=np.int8)
        packet = encode_pulses(values, force_mode="rice")
        # b=0: gaps 1,2 -> 10/sign1 and 110/sign0 -> 1011100 + padding0.
        self.assertEqual(packet.data, struct.pack(">4sIIHBB", b"EVS1", 8, 2, 1, 1, 0) + b"\xb8")
        self.assertEqual((packet.payload_bits, packet.padding_bits, packet.wire_bits), (7, 1, 136))
        np.testing.assert_array_equal(decode_pulses(packet.data), values)

    def test_empty_and_full_support_need_no_position_stream(self):
        empty = encode_pulses(np.zeros(9, dtype=np.int8))
        full = encode_pulses(np.array([1, -1] * 4, dtype=np.int8))
        self.assertEqual(empty.wire_bits, 128)
        self.assertTrue(full.mode.startswith("full:"))
        self.assertEqual(full.wire_bits, 136)

    def test_signed_counts_have_hand_encoded_dense_unary_representation(self):
        values = np.array([0, 2, 0, -1], dtype=np.int8)
        packet = encode_pulses(values, max_abs=3, force_mode="dense_unary")
        self.assertEqual(packet.data[-2:], b"\x62\x80")
        self.assertEqual(packet.payload_bits, 9)
        np.testing.assert_array_equal(decode_pulses(packet.data), values)

    def test_all_modes_roundtrip_and_portfolio_selects_shortest_byte_length(self):
        rng = np.random.default_rng(900)
        for d, bound in ((1, 1), (7, 3), (33, 10), (257, 50)):
            for density in (0, 0.01, 0.3, 1):
                values = rng.integers(-bound, bound + 1, size=d, dtype=np.int32)
                values[rng.random(d) >= density] = 0
                candidates = []
                for mode in ("empty", "rice", "indices", "bitmap", "full", "dense_unary", "dense_fixed"):
                    try:
                        packet = encode_pulses(values, max_abs=bound, force_mode=mode)
                    except ValueError:
                        continue
                    np.testing.assert_array_equal(decode_pulses(packet.data, expected_dimension=d), values)
                    candidates.append(packet.wire_bits)
                self.assertEqual(encode_pulses(values, max_abs=bound).wire_bits, min(candidates))

    def test_invalid_metadata_truncation_padding_and_trailing_bytes_fail(self):
        packet = encode_pulses(np.array([0, -1, 0, 0, 1, 0, 0, 0]), force_mode="rice")
        variants = [b"", b"FAIL" + packet.data[4:], packet.data[:-1], packet.data + b"\0",
                    packet.data[:-1] + bytes([packet.data[-1] | 1]),
                    struct.pack(">4sIIHBB", b"EVS1", 8, 9, 1, 1, 0) + packet.data[16:],
                    packet.data[:14] + b"\x71" + packet.data[15:]]
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_pulses(data)
        with self.assertRaises(ValueError):
            decode_pulses(packet.data, expected_dimension=9)
        for values in ([], [1.5], [2], [[1]], [np.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                encode_pulses(values)

    def test_dense_support_bypass_improves_stc_without_changing_values(self):
        vector = np.array([1, -2, 3, -4] * 64, dtype=np.float32)
        for fraction in (0.01, 0.25, 1.0):
            native = compress_encode_stc(vector, (128, 128), fraction)
            recoded = encode_stc_lossless(native.data, (128, 128))
            np.testing.assert_array_equal(decode_stc(native.data, (128, 128)).view(np.uint32),
                                          decode_stc_lossless(recoded, (128, 128)).view(np.uint32))
            if fraction == 1:
                self.assertLess(len(recoded), len(native.data))


class EventDownlinkTests(unittest.TestCase):
    def test_aggregate_preserves_multiplicity_and_cancels_opposing_pulses(self):
        pulses = np.array([[1, 1, 0], [1, -1, 0], [1, 0, -1]], dtype=np.int8)
        message = encode_event_downlink(pulses, 0.005)
        np.testing.assert_array_equal(decode_pulses(message.counts.data), [3, 0, -1])
        np.testing.assert_array_equal(decode_event_downlink(message.data, .005, expected_dimension=3).view(np.uint32),
                                      pulse_aggregate(pulses, .005).view(np.uint32))

    def test_all_ten_client_patterns_preserve_historical_float32_arithmetic(self):
        pulses = np.array(list(itertools.product((-1, 0, 1), repeat=10)), dtype=np.int8).T
        jump = .005 * (1 + 1 / 100) ** (-.2)
        packet = encode_event_downlink(pulses, jump)
        self.assertGreater(packet.correction_nonzeros, 0)
        np.testing.assert_array_equal(decode_event_downlink(packet.data, jump, expected_dimension=pulses.shape[1]).view(np.uint32),
                                      pulse_aggregate(pulses, jump).view(np.uint32))
        # Either exact representation can be selected; neither costs more than
        # the fully serialized ordered-pulse reference.
        self.assertLessEqual(8 * len(packet.data), packet.ordered_bits)

    def test_downlink_rejects_wrong_shape_truncated_and_nonzero_padding(self):
        packet = encode_event_downlink(np.array([[1, 0, -1]], dtype=np.int8), .01)
        for data in (b"", packet.data[:-1], packet.data + b"\0"):
            with self.assertRaises(ValueError):
                decode_event_downlink(data, .01, expected_dimension=3)
        with self.assertRaises(ValueError):
            decode_event_downlink(packet.data, .01, expected_dimension=4)


class PilotTests(unittest.TestCase):
    def test_short_protocol_is_frozen_at_development_seed_and_not_convergence(self):
        from experiments import event_codec_pilot as driver
        self.assertEqual((driver.ROUNDS, driver.PARTITION_SEED, driver.TRAIN_SEED), (120, 4500, 84500))
        self.assertEqual(len(driver.POINTS), 3)
        self.assertEqual(driver.spec()["evaluation"], "training_subset_only_no_test_accuracy")
        self.assertEqual(config_for("event").regularization, .0005)

    def test_host_matches_frozen_runner_model_history_and_native_traffic(self):
        with tempfile.TemporaryDirectory() as folder:
            for point in POINTS:
                config = replace(config_for(point), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
                kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn",
                              method="event" if point == "event" else "stc", config=config, seed=84500)
                frozen_path = Path(folder) / f"{point}.npz"
                frozen = run_final_baseline(**kwargs, record_history=True,
                                             checkpoint=BaselineCheckpoint(frozen_path, {"test": True}))
                pilot = run_codec_pilot(**kwargs)
                with np.load(frozen_path, allow_pickle=False) as archive:
                    np.testing.assert_array_equal(pilot["model"].view(np.uint32), archive["model"].view(np.uint32))
                pd.testing.assert_frame_equal(pilot["history"], frozen["history"][["round", "train_objective", "train_ce"]])
                row = pilot["summary"].iloc[0]
                self.assertEqual(int(row.uplink_bits), frozen["uplink_packetized_bits"])
                self.assertEqual(int(row.unicast_total_bits), frozen["unicast_hybrid_total_bits"])
                self.assertEqual(int(row.broadcast_total_bits), frozen["broadcast_total_bits"])

    def test_exact_resume_preserves_model_rng_and_traffic(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="event",
                          config=replace(config_for("event"), rounds=3, eval_stride=1, local_steps=1, batch_size=2), seed=84500)
            full = run_codec_pilot(**kwargs)
            checkpoint = BaselineCheckpoint(Path(folder) / "resume.npz", {"test": True})
            run_codec_pilot(**kwargs, checkpoint=checkpoint, stop_after=1)
            resumed = run_codec_pilot(**kwargs, checkpoint=checkpoint)
            np.testing.assert_array_equal(full["model"].view(np.uint32), resumed["model"].view(np.uint32))
            self.assertEqual(full["rng_state"], resumed["rng_state"])
            pd.testing.assert_frame_equal(full["summary"], resumed["summary"])
            pd.testing.assert_frame_equal(full["history"], resumed["history"])

    def test_traffic_charges_initialization_requests_and_each_receiver(self):
        rounds = pd.DataFrame([{"method": "event", "legacy_uplink_bits": 100,
                                "codec_uplink_bits": 50, "legacy_server_bits": 100,
                                "codec_replay_bits": 50, "exact_downlink_bits": 40}])
        table = traffic_summary(rounds, n_clients=2, dimension=10).set_index("variant")
        # 384-bit initialization + 32-bit request + 40-bit exact server frame.
        self.assertEqual(table.loc["both_codecs", "unicast_total_bits"], 50 + 2 * (384 + 32 + 40))
        self.assertEqual(table.loc["both_codecs", "broadcast_total_bits"], 50 + 384 + 40)


class EvidenceTests(unittest.TestCase):
    def test_receipts_analysis_and_cross_platform_newlines(self):
        from experiments import event_codec_pilot as driver
        original_spec = driver.spec
        with tempfile.TemporaryDirectory() as folder, patch.object(driver, "ROUNDS", 2):
            tiny_spec = {**original_spec(), "clients": 2, "dimension": 4}
            with patch.object(driver, "spec", return_value=tiny_spec):
                out = Path(folder)
                science = {"protocol": driver.spec(), "sources": driver.sources(), "runtime": {"synthetic": True}}
                driver.atomic_json(science, out / "protocol.json")
                for point in POINTS:
                    rows = pd.DataFrame([{
                        "round": r, "method": "event" if point == "event" else "stc",
                        "legacy_uplink_bits": 100, "codec_uplink_bits": 80,
                        "legacy_server_bits": 100, "codec_replay_bits": 80,
                        "exact_downlink_bits": 40, "model_sha256": "a" * 64,
                        "correction_nonzeros": 0, "canonical_max_abs_error": 0.,
                        "codec_audit_seconds": .01} for r in (1, 2)])
                    result = {"rounds": rows,
                              "history": pd.DataFrame({"round": [1, 2], "train_objective": [1., .9], "train_ce": [.9, .8]}),
                              "summary": traffic_summary(rows, n_clients=2, dimension=4)}
                    driver.save_result(point, science, result, out)
                with redirect_stdout(io.StringIO()):
                    driver.analyze(out)
                self.assertEqual(len(pd.read_csv(out / "comparison.csv")), 8)
                # Git may change LF to CRLF on Windows. Scientific text hashes
                # deliberately tolerate only this newline representation change.
                file = driver.paths("event", out)["rounds"]
                file.write_bytes(file.read_bytes().replace(b"\n", b"\r\n"))
                driver.read_receipt("event", science, out)
                with redirect_stdout(io.StringIO()):
                    driver.analyze(out)
                file.write_bytes(file.read_bytes().replace(b"100", b"101", 1))
                with self.assertRaisesRegex(RuntimeError, "output changed"):
                    driver.read_receipt("event", science, out)

    def test_analysis_accounts_for_failed_points(self):
        from experiments import event_codec_pilot as driver
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            science = {"protocol": driver.spec(), "sources": driver.sources(), "runtime": {"synthetic": True}}
            driver.atomic_json(science, out / "protocol.json")
            driver.atomic_json({"point": "event", "state": "numerical_failure",
                                "contract_sha256": driver.digest(science), "error": "synthetic failure"},
                               driver.paths("event", out)["receipt"])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                driver.analyze(out)


if __name__ == "__main__":
    unittest.main()
