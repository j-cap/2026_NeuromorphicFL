from __future__ import annotations

from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

import numpy as np

from neuromorphicfl.final_baseline_campaign import (
    FinalBaselineConfig, run_final_baseline,
)
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation
from neuromorphicfl.stc import (
    STCState, compress_encode_stc, decode_stc, sparse_ternary_compress,
)


class CompressorTests(unittest.TestCase):
    def test_topk_uses_selected_mean_and_original_sign(self) -> None:
        actual = sparse_ternary_compress(np.array([4, -2, 1, 0]), 0.5)
        np.testing.assert_array_equal(actual, [3, -3, 0, 0])

    def test_ties_are_deterministic_and_exact_k(self) -> None:
        actual = sparse_ternary_compress(np.array([2, -2, 2, -2]), 0.5)
        np.testing.assert_array_equal(actual, [2, -2, 0, 0])

    def test_zero_tiny_and_full_support(self) -> None:
        np.testing.assert_array_equal(sparse_ternary_compress(np.zeros(5), 0.01), 0)
        np.testing.assert_array_equal(sparse_ternary_compress(np.array([1, -3]), 1e-9), [0, -3])
        actual = sparse_ternary_compress(np.array([1, 3, 0, -2]), 1.0)
        np.testing.assert_array_equal(actual, [1.5, 1.5, 0, -1.5])
        self.assertFalse(np.array_equal(actual, [1, 3, 0, -2]))

    def test_invalid_inputs_fail_before_training(self) -> None:
        for fraction in (0, -1, 1.01, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                sparse_ternary_compress(np.ones(3), fraction)
        for vector in (np.array([]), np.ones((2, 2)), np.array([np.nan])):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                sparse_ternary_compress(vector, 0.5)


class CodecTests(unittest.TestCase):
    def test_hand_encoded_positions_signs_and_bit_count(self) -> None:
        # n=8, p=1/4 gives Rice b=1. Indices 1,4 have zero-based gaps
        # 1,2, hence codes 01 and 100. Interleaved signs 1,0 give
        # 0111000, followed by one zero padding bit: byte 0x70.
        values = np.array([0, -2, 0, 0, 2, 0, 0, 0], dtype=np.float32)
        packet = compress_encode_stc(values, (8,), 0.25)
        expected = b"STC1" + struct.pack(">IIfB", 1, 2, 2.0, 1) + b"\x70"
        self.assertEqual(packet.data, expected)
        self.assertEqual(packet.wire_bits, 144)
        self.assertEqual(packet.payload_bits, 80)
        self.assertEqual(packet.padding_bits, 1)
        self.assertEqual(packet.nonzeros, 2)
        np.testing.assert_array_equal(decode_stc(expected, (8,)), values)

    def test_zero_message_metadata_is_charged(self) -> None:
        packet = compress_encode_stc(np.zeros(8), (8,), 0.25)
        self.assertEqual(packet.wire_bits, 64 + 72)
        self.assertEqual(packet.nonzeros, 0)
        np.testing.assert_array_equal(decode_stc(packet.data, (8,)), 0)

    def test_roundtrip_multiple_tensors_and_fraction_extremes(self) -> None:
        rng = np.random.default_rng(53)
        sizes = (1, 5, 32, 333)
        vector = rng.normal(size=sum(sizes)).astype(np.float32)
        vector[20:35] = 0
        for fraction in (1e-9, 0.0025, 0.01, 0.3, 1.0):
            with self.subTest(fraction=fraction):
                packet = compress_encode_stc(vector, sizes, fraction)
                start, pieces = 0, []
                for size in sizes:
                    pieces.append(sparse_ternary_compress(vector[start:start + size], fraction))
                    start += size
                np.testing.assert_array_equal(decode_stc(packet.data, sizes), np.concatenate(pieces))
                self.assertEqual(packet.wire_bits, 8 * len(packet.data))

    def test_decoder_rejects_bad_metadata_truncation_and_padding(self) -> None:
        packet = compress_encode_stc(np.array([0, -2, 0, 0, 2, 0, 0, 0]), (8,), 0.25)
        malformed = (
            b"", b"FAIL" + packet.data[4:], packet.data[:-1], packet.data + b"\x00",
            packet.data[:-1] + bytes([packet.data[-1] | 1]),
            packet.data[:8] + struct.pack(">IfB", 9, 2.0, 1) + packet.data[17:],
        )
        for data in malformed:
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_stc(data, (8,))
        with self.assertRaises(ValueError):
            decode_stc(packet.data, (4, 4))


class ResidualProtocolTests(unittest.TestCase):
    def test_two_round_hand_reference_preserves_both_residuals(self) -> None:
        state = STCState.zeros(n_clients=2, tensor_sizes=(4,), upload_fraction=0.5, download_fraction=0.5)
        deltas = np.array([[4, 2, 1, -0.5], [-2, 0, 3, 1]], dtype=np.float32)
        weights = np.array([0.25, 0.75])
        first = state.step(deltas, weights)
        np.testing.assert_array_equal(first.server_update, [-1.5, 0, 1.5, 0])
        np.testing.assert_array_equal(state.client_residuals, [[1, -1, 1, -0.5], [0.5, 0, 0.5, 1]])
        np.testing.assert_array_equal(state.server_residual, [0.375, 0.75, 0.375, 0])
        second = state.step(np.zeros_like(deltas), weights)
        np.testing.assert_array_equal(second.server_update, [0.875, 0, 0, 0.875])
        np.testing.assert_array_equal(state.server_residual, [0.3125, 0.5, 0.375, -0.3125])

    def test_residual_balance_and_client_model_reconstruction(self) -> None:
        rng = np.random.default_rng(59)
        sizes = (3, 7)
        state = STCState.zeros(n_clients=3, tensor_sizes=sizes, upload_fraction=0.3, download_fraction=0.2)
        weights = np.array([0.2, 0.3, 0.5])
        gain = 1.7
        server = np.zeros(10, dtype=np.float32)
        clients = np.zeros((3, 10), dtype=np.float32)
        cumulative_local = np.zeros((3, 10), dtype=np.float64)
        cumulative_applied = np.zeros(10, dtype=np.float64)
        for _ in range(15):
            deltas = rng.normal(0, 0.1, size=(3, 10)).astype(np.float32)
            old_clients = state.client_residuals.copy()
            old_server = state.server_residual.copy()
            communicated = state.step(deltas, weights, server_gain=gain)
            uploads = np.stack([decode_stc(packet.data, sizes) for packet in communicated.uplink])
            np.testing.assert_allclose(old_clients + deltas, uploads + state.client_residuals, atol=3e-8, rtol=1e-6)
            corrected = old_server + gain * np.sum(weights[:, None] * uploads, axis=0)
            np.testing.assert_allclose(corrected, communicated.server_update + state.server_residual, atol=1e-7, rtol=1e-6)
            server += communicated.server_update
            for client in clients:
                client += decode_stc(communicated.downlink.data, sizes)
                np.testing.assert_array_equal(client, server)
            cumulative_local += deltas
            cumulative_applied += communicated.server_update
        # End-to-end accounting: the untransmitted information consists exactly
        # of weighted client residuals and the separate server residual.
        expected = gain * np.sum(weights[:, None] * cumulative_local, axis=0)
        retained = gain * np.sum(weights[:, None] * state.client_residuals, axis=0) + state.server_residual
        np.testing.assert_allclose(expected, cumulative_applied + retained, atol=5e-7, rtol=1e-6)

    def test_nonunit_gain_is_applied_before_server_compression_once(self) -> None:
        state = STCState.zeros(n_clients=1, tensor_sizes=(4,), upload_fraction=0.5, download_fraction=0.5)
        result = state.step([np.array([4, -2, 1, 0])], np.array([1.0]), server_gain=2)
        np.testing.assert_array_equal(result.server_update, [6, -6, 0, 0])
        np.testing.assert_array_equal(state.client_residuals[0], [1, 1, 1, 0])


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_applies_native_server_message_and_counts_its_downlink(self) -> None:
        class Layout:
            dimension = 4

            def groups(self):
                return [("weights", slice(0, 4), 4)]

        observed_models = []

        def initialize(**kwargs):
            return np.zeros(4, dtype=np.float32)

        def metrics(model, *args, **kwargs):
            observed_models.append(model.copy())
            return 0.0, 0.0, 0.5, 0.5, 0.5, [0.5]

        federation = MulticlassFederation(
            client_X=(np.zeros((1, 1)),) * 2,
            client_y=(np.zeros(1),) * 2,
            X_train_eval=np.zeros((2, 1)), y_train_eval=np.zeros(2),
            X_test=np.zeros((1, 1)), y_test=np.zeros(1), regime="test",
            client_class_counts=np.ones((2, 1)), periods=np.ones(2),
            weights=np.array([0.25, 0.75]),
        )
        deltas = [np.array([4, 2, 1, -0.5]), np.array([-2, 0, 3, 1])]
        config = FinalBaselineConfig(rounds=1, eval_stride=1, stc_upload_fraction=0.5, stc_download_fraction=0.5)
        with patch("neuromorphicfl.final_baseline_campaign._architecture_ops", return_value=(Layout(), initialize, None, metrics)), patch("neuromorphicfl.final_baseline_campaign._local_delta", side_effect=deltas):
            result = run_final_baseline(federation=federation, architecture="mlp", method="stc", config=config, record_history=True)
        for model in observed_models:
            np.testing.assert_array_equal(model, [-1.5, 0, 1.5, 0])
        # Each 4-coordinate message has 64 frame + 72 metadata + one symbol byte.
        self.assertEqual(result["uplink_packetized_bits"], 2 * 144)
        self.assertEqual(result["stc_downlink_packetized_bits"], 144)
        self.assertEqual(result["broadcast_downlink_bits"], 192 + 144)
        self.assertEqual(result["unicast_hybrid_downlink_bits"], 2 * (192 + 32 + 144))
        self.assertEqual(result["unicast_total_bits"], 1024)
        self.assertEqual(result["replay_rounds"], 0)
        self.assertEqual(result["checkpoint_rounds"], 0)
        self.assertEqual(result["history"].iloc[-1]["unicast_total_bits"], 1024)
        self.assertEqual(result["stc_downlink_messages"], 1)
        observed_models.clear()
        with patch("neuromorphicfl.final_baseline_campaign._architecture_ops", return_value=(Layout(), initialize, None, metrics)), patch("neuromorphicfl.final_baseline_campaign._local_delta", side_effect=deltas):
            run_final_baseline(federation=federation, architecture="mlp", method="stc", config=replace(config, server_gain=2.0))
        for model in observed_models:
            np.testing.assert_array_equal(model, [-3, 0, 3, 0])
        with self.assertRaisesRegex(ValueError, "scheduled server gain"):
            run_final_baseline(federation=federation, architecture="mlp", method="stc", config=replace(config, server_gain_exponent=0.1))


if __name__ == "__main__":
    unittest.main()
