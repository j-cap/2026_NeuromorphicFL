from __future__ import annotations

from copy import deepcopy
import struct
import unittest

import numpy as np
from experiments import fast_pulse_codec as fast
from experiments import resnet14_baseline_fast_host as fast_host
from experiments import resnet14_baseline_development as development
from neuromorphicfl import event_codec as native


class PulseCodecTests(unittest.TestCase):
    def test_all_valid_modes_exact_bytes_metadata_and_both_decoders(self):
        rng = np.random.default_rng(412)
        for dimension in (1, 2, 7, 8, 9, 31, 127, 1024):
            for density in (0., .01, .25, .75, 1.):
                vector = rng.choice([-1, 1], dimension).astype(np.int8)
                vector[rng.random(dimension) >= density] = 0
                modes = [None, "indices", "bitmap", "rice", "dense_unary", "dense_fixed"]
                modes += ["empty"] if not np.any(vector) else []
                modes += ["full"] if np.count_nonzero(vector) == dimension else []
                for mode in modes:
                    with self.subTest(d=dimension, density=density, mode=mode):
                        try:
                            expected = native.encode_pulses(vector, force_mode=mode)
                        except ValueError:
                            with self.assertRaises(ValueError):
                                fast.encode_pulses(vector, force_mode=mode)
                            continue
                        actual = fast.encode_pulses(vector, force_mode=mode)
                        self.assertEqual(actual, expected)
                        np.testing.assert_array_equal(fast.decode_pulses(expected.data, expected_dimension=dimension), vector)
                        np.testing.assert_array_equal(native.decode_pulses(actual.data, expected_dimension=dimension), vector)

    def test_resnet_sized_full_nearly_full_and_sparse_messages(self):
        rng = np.random.default_rng(77)
        for density in (1., .9999, .01):
            vector = rng.choice([-1, 1], 175258).astype(np.int8)
            vector[rng.random(vector.size) >= density] = 0
            expected, actual = native.encode_pulses(vector), fast.encode_pulses(vector)
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(fast.decode_pulses(expected.data), vector)

    def test_larger_integer_bounds_use_exact_frozen_fallback(self):
        for bound in (2, 10, 32767):
            vector = np.array([0, bound, -bound, 1, -1, 0], dtype=np.int32)
            expected = native.encode_pulses(vector, max_abs=bound)
            self.assertEqual(fast.encode_pulses(vector, max_abs=bound), expected)
            np.testing.assert_array_equal(fast.decode_pulses(expected.data), vector)

    def test_handwritten_unary_value_codes_and_invalid_magnitude(self):
        for mode in (1, 2, 3, 4):
            vector = np.array([-1, 1, -1, 1, -1] if mode == 4 else [-1, 0, 1, -1, 0], dtype=np.int8)
            indices = np.flatnonzero(vector)
            writer = native.BitWriter()
            if mode == 3:
                for value in vector:
                    writer.write(int(value != 0), 1)
            previous = -1
            for index in indices:
                if mode == 1:
                    writer.unary(int(index) - previous - 1)
                elif mode == 2:
                    writer.write(int(index), (vector.size - 1).bit_length())
                writer.unary(0)
                writer.write(int(vector[index] < 0), 1)
                previous = int(index)
            data = struct.pack(">4sIIHBB", b"EVS1", vector.size, len(indices), 1, mode | 16, 0) + writer.finish()
            np.testing.assert_array_equal(native.decode_pulses(data), vector)
            np.testing.assert_array_equal(fast.decode_pulses(data), vector)
        invalid = struct.pack(">4sIIHBB", b"EVS1", 2, 2, 1, 4 | 16, 0) + b"\x80"
        for decoder in (native.decode_pulses, fast.decode_pulses):
            with self.assertRaises(ValueError):
                decoder(invalid)

    def test_padding_truncation_metadata_and_arbitrary_small_packets_match_rejection(self):
        rng = np.random.default_rng(19)
        corpus = []
        for mode in (None, "rice", "bitmap", "indices", "full"):
            packet = native.encode_pulses(np.ones(9, dtype=np.int8), force_mode=mode).data
            corpus += [packet[:-1], packet + b"\x00", packet[:-1] + bytes([packet[-1] | 1])]
        for _ in range(400):
            dimension = int(rng.integers(1, 33))
            count = int(rng.integers(0, dimension + 1))
            mode = int(rng.integers(0, 8))
            code = int(rng.integers(0, 3))
            exponent = int(rng.integers(0, 7))
            header = struct.pack(">4sIIHBB", b"EVS1", dimension, count, 1, mode | (code << 4), exponent)
            corpus.append(header + rng.integers(0, 256, size=int(rng.integers(0, 12)), dtype=np.uint8).tobytes())
        for data in corpus:
            try:
                expected = native.decode_pulses(data)
            except ValueError:
                with self.assertRaises(ValueError):
                    fast.decode_pulses(data)
            else:
                np.testing.assert_array_equal(fast.decode_pulses(data), expected)
        with self.assertRaises(ValueError):
            fast.decode_pulses(native.encode_pulses(np.ones(3, dtype=np.int8)).data, expected_dimension=4)


class HostFidelityTests(unittest.TestCase):
    def test_original_modules_are_not_mutated(self):
        self.assertIs(development.host.encode_pulses, native.encode_pulses)
        self.assertIs(development.host.wire.encode_pulses, native.encode_pulses)
        self.assertIs(development.host.decode_stc_lossless, native.decode_stc_lossless)
        self.assertIsNot(fast_host.wire, development.host.wire)

    def test_stc_envelope_roundtrips_zero_tiny_and_mixed_tensor_scales(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        for values in (np.zeros(8, dtype=np.float32), np.array([tiny, -tiny, 0, 0, 1, -1, 0, 1], dtype=np.float32)):
            expected, native_decoded = development.host.encode_ternary(values, (4, 4))
            actual, decoded = fast_host._implementation.encode_ternary(values, (4, 4))
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(decoded.view(np.uint32), native_decoded.view(np.uint32))

    def test_all_selected_rules_match_original_models_residuals_and_every_counter(self):
        rng = np.random.default_rng(110)
        sizes, weights = (4, 4), np.array([.5, .5])
        points = [development.BY_ID[k] for k in ("dense_g1p5_a0", "ef_topk_p0p01_g4", "stc_u1_d1_g4")]
        for point in points:
            state = {"client_residuals": np.zeros((2, 8), dtype=np.float32), "server_residual": np.zeros(8, dtype=np.float32)}
            reference = deepcopy(state)
            for rnd in range(1, 5):
                deltas = rng.normal(size=(2, 8)).astype(np.float32)
                deltas[:, ::3] = 0
                expected, row_expected = development.host.step(reference, deltas, weights, method=point.method,
                                                              config=development.config_for(point), sizes=sizes, rnd=rnd)
                actual, row = fast_host.step(state, deltas, weights, method=point.method,
                                            config=development.config_for(point), sizes=sizes, rnd=rnd)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                self.assertEqual(row, row_expected)
                for name in state:
                    np.testing.assert_array_equal(state[name].view(np.uint32), reference[name].view(np.uint32))


if __name__ == "__main__":
    unittest.main()
