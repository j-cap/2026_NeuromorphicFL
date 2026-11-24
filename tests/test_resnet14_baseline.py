from __future__ import annotations

from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from experiments import resnet14_baseline_development as study
from neuromorphicfl.stc import compress_encode_stc, decode_stc
from neuromorphicfl.event_codec import encode_stc_lossless

parent, host = study.parent, study.host
MANIFEST = {"synthetic": True, "training_examples": 12, "validation_examples": 4}


def write_point(p, horizon, science, out, *, failure=False, prefix_change=False, reference_wire=False):
    files = study.paths(p, horizon, out)
    receipt = {"identity": study.metadata(p, horizon, science), "state": "numerical_failure" if failure else "complete"}
    if failure:
        parent.atomic_json({**receipt, "error": "synthetic numerical failure"}, files["receipt"])
        return
    index = study.POINTS.index(p)
    hashes = [f"{index * 10 + i:064x}" for i in range(1, horizon + 1)]
    rounds = pd.DataFrame({"round": range(1, horizon + 1), "method": p.method, "codec_uplink_bits": 80,
                           "exact_downlink_bits": 64, "roundtrip_verified": True, "model_sha256": hashes,
                           "wire_protocol": "native_reference_only" if reference_wire else host.WIRE_PROTOCOL})
    evaluation = pd.DataFrame({"round": range(1, horizon + 1), "train_objective": 1.2, "train_ce": 1.,
                               "validation_ce": [1.5 + (.1 if prefix_change else 0), 1. + index * .001, .8 + index * .001][:horizon],
                               "validation_accuracy": .6, "validation_macro_accuracy": .6,
                               "validation_worst_class_accuracy": .2, "model_sha256": hashes})
    tables = study.trajectory_tables({"rounds": rounds, "evaluation": evaluation}, science)
    for k, frame in tables.items():
        parent.atomic_csv(frame.assign(**receipt["identity"]), files[k])
    parent.atomic_json({**receipt, "data_manifest": MANIFEST, "sha256": {k: parent.text_hash(files[k]) for k in tables}}, files["receipt"])


@contextmanager
def campaign(*, interrupt_after=None, failed_screen_family=None, failed_final_family=None):
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        root = Path(folder)
        out, prior_out = root / "baseline", root / "parent"
        science = {"protocol": {"n_clients": 2, "dimension": 8}, "synthetic": True}
        for obj, key, value in ((study, "SCREEN", 2), (study, "FINAL", 3), (parent, "OUT", prior_out),
                                (parent, "BASE", replace(parent.BASE, rounds=3, eval_stride=1))):
            stack.enter_context(patch.object(obj, key, value))
        parent.atomic_json(MANIFEST, prior_out / "split_manifest.json")
        stack.enter_context(patch.object(study, "load_contract", return_value=science))
        stack.enter_context(patch.object(parent, "load_contract", return_value={"synthetic": "prior"}))
        controls = pd.DataFrame([{"point_id": key, "method": m, "state": "complete", "final_validation_ce": .75,
                                  "final_validation_accuracy": .7, "unicast_total_bits": 1000}
                                 for m, key in study.pulse.POINT_IDS.items()])
        stack.enter_context(patch.object(parent, "ledger", return_value=controls))
        calls, interrupted = [], [False]

        def prepare(device_name, threads, batch, output):
            parent.immutable_json(science, output / "protocol.json")
            return science, "device", SimpleNamespace(scope="validation", manifest=MANIFEST)

        def train(p, horizon, contract, fed, device, output):
            assert fed.scope == "validation"
            if study.read_receipt(p, horizon, contract, output) is not None:
                return
            if interrupt_after is not None and len(calls) == interrupt_after and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt("synthetic interruption")
            calls.append((p.point_id, horizon))
            fail = (horizon == 2 and p.method == failed_screen_family) or (horizon == 3 and p.method == failed_final_family)
            write_point(p, horizon, contract, output, failure=fail)

        stack.enter_context(patch.object(study, "prepare", side_effect=prepare))
        stack.enter_context(patch.object(study, "train_point", side_effect=train))
        stack.enter_context(patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("no official test or image loader in orchestration")))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(out=out, prior_out=prior_out, science=science, calls=calls)


class NativeServerTests(unittest.TestCase):
    def test_direct_ternary_envelope_equals_native_rice_then_lossless_bytes(self):
        rng = np.random.default_rng(10)
        sizes = (4, 3, 1, 16)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        vectors = [rng.normal(size=sum(sizes)).astype(np.float32), np.zeros(sum(sizes), dtype=np.float32),
                   np.array([-tiny] + [0] * (sum(sizes) - 1), dtype=np.float32)]
        for values in vectors:
            for fraction in (.01, .25, 1.):
                native = compress_encode_stc(values, sizes, fraction)
                expected = decode_stc(native.data, sizes)
                actual = host.ternary(values, sizes, fraction)
                np.testing.assert_array_equal(expected.view(np.uint32), actual.view(np.uint32))
                data, decoded = host.encode_ternary(actual, sizes)
                self.assertEqual(data, encode_stc_lossless(native.data, sizes))
                np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
        values = np.arange(1, 25, dtype=np.float32)
        self.assertFalse(np.array_equal(host.ternary(values, sizes, 1.), values))

    def test_invalid_ternary_blocks_and_dtype_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one nonzero magnitude"):
            host.encode_ternary(np.array([1, 2], dtype=np.float32), (2,))
        with self.assertRaises(ValueError):
            host.encode_ternary(np.array([1, 1], dtype=np.float64), (2,))

    def test_three_server_rules_match_independent_native_steps_and_charge_frames(self):
        rng, sizes = np.random.default_rng(21), (4, 4)
        weights = np.array([.25, .75], dtype=np.float64)
        examples = [study.make_point("dense", 2., exponent=.3), study.make_point("stc", 2., .25, download=1.),
                    study.make_point("ef_topk", .5, .25)]
        for p in examples:
            state = {"client_residuals": np.zeros((2, 8), dtype=np.float32), "server_residual": np.zeros(8, dtype=np.float32)}
            reference, rows = deepcopy(state), []
            cfg = study.config_for(p)
            for rnd in (1, 2):
                deltas = rng.normal(size=(2, 8)).astype(np.float32)
                native = host.native_step(reference, deltas, weights, method=p.method, config=cfg, sizes=sizes, rnd=rnd)
                actual, row = host.step(state, deltas, weights, method=p.method, config=cfg, sizes=sizes, rnd=rnd)
                np.testing.assert_array_equal(actual.view(np.uint32), native.view(np.uint32))
                for name in state:
                    np.testing.assert_array_equal(state[name].view(np.uint32), reference[name].view(np.uint32))
                self.assertTrue(row["roundtrip_verified"])
                self.assertGreater(row["codec_uplink_bits"], 0)
                self.assertGreater(row["exact_downlink_bits"], 0)
                rows.append({**row, "round": rnd})
            traffic = host.traffic_summary(pd.DataFrame(rows), n_clients=2, dimension=8).iloc[0]
            uplink = sum(r["codec_uplink_bits"] for r in rows)
            stream = sum(r["exact_downlink_bits"] for r in rows)
            self.assertEqual(traffic.unicast_total_bits, uplink + 2 * (32 * 8 + 64 + 32 * 2 + stream))
            self.assertEqual(traffic.broadcast_total_bits, uplink + 32 * 8 + 64 + stream)

    def test_full_fraction_ef_is_dense_update_and_zero_messages_still_cost_bits(self):
        rng, weights = np.random.default_rng(12), np.array([.5, .5])
        state = lambda: {"client_residuals": np.zeros((2, 8), dtype=np.float32), "server_residual": np.zeros(8, dtype=np.float32)}
        dense, ef = state(), state()
        for rnd in (1, 2):
            deltas = rng.normal(size=(2, 8)).astype(np.float32)
            left, _ = host.step(dense, deltas, weights, method="dense", config=study.config_for(study.make_point("dense", 1.)), sizes=(4, 4), rnd=rnd)
            right, _ = host.step(ef, deltas, weights, method="ef_topk", config=study.config_for(study.make_point("ef_topk", 1., 1.)), sizes=(4, 4), rnd=rnd)
            np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
            self.assertFalse(np.any(ef["client_residuals"]))
        for method in study.METHODS:
            p = next(p for p in study.POINTS if p.method == method)
            _, row = host.step(state(), np.zeros((2, 8), dtype=np.float32), weights, method=method,
                               config=study.config_for(p), sizes=(4, 4), rnd=1)
            self.assertGreater(row["codec_uplink_bits"], 0)
            self.assertGreater(row["exact_downlink_bits"], 0)


class ProtocolTests(unittest.TestCase):
    def test_equal_accumulated_resnet_budget_native_anchors_and_fixed_learning(self):
        self.assertEqual(len(study.BY_ID), 54)
        self.assertEqual([sum(p.method == m for p in study.POINTS) for m in study.METHODS], [18, 18, 18])
        for key in ("dense_g1_a0", "dense_g4_a0p3", "stc_u1_d1_g2", "stc_u0p25_d0p25_g1", "ef_topk_p0p05_g1", "ef_topk_p0p005_g1", "ef_topk_p0p25_g1"):
            self.assertIn(key, study.BY_ID)
        self.assertNotIn("ef_topk_p1_g1", study.BY_ID)  # Identity is a native-parity test, not a second Dense comparator.
        for p in study.POINTS:
            cfg = study.config_for(p)
            self.assertEqual((cfg.local_steps, cfg.local_lr, cfg.batch_size, cfg.regularization), (5, .05, 32, .0005))
        spec = study.protocol_spec()
        self.assertEqual(spec["max_unique_training_rounds"], 54 * 600 + 9 * 1200)
        self.assertEqual(spec["max_unique_evaluation_points"], 54 * 11 + 9 * 20)
        self.assertEqual(spec["new_confirmation_runs"], 0)
        self.assertFalse(spec["automatic_boundary_extension"])

    def test_source_reference_reads_validation_only_without_loading_images(self):
        original = parent.paths
        def paths(p, phase, *args, **kwargs):
            self.assertEqual(phase, "development")
            return original(p, phase, *args, **kwargs)
        with patch.object(parent, "paths", side_effect=paths), \
                patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("no images")):
            science = study.science_spec()
        self.assertIn("experiments/resnet14_baseline_host.py", science["source_hashes"])
        self.assertFalse(science["protocol"]["saved_official_test_metrics_read"])
        self.assertTrue(all("\\" not in p for p in science["pulse_validation_reference"]["files"]))

    def test_selection_uses_validation_ce_then_cost_not_accuracy_or_test_scores(self):
        rows = []
        for method in study.METHODS:
            for i, p in enumerate([p for p in study.POINTS if p.method == method][:3]):
                rows.append({"point_id": p.point_id, "method": method, "state": "complete" if i < 2 else "numerical_failure",
                             "final_validation_ce": 1., "unicast_total_bits": 200 if i == 0 else 100,
                             "final_validation_accuracy": .99 if i == 0 else .5, "final_test_ce": 0 if i == 0 else 10})
        frame = pd.DataFrame(rows)
        chosen = study.choose(frame, 1)
        for method in study.METHODS:
            self.assertEqual(chosen[method], [frame[(frame.method == method) & (frame.unicast_total_bits == 100) & (frame.state == "complete")].iloc[0].point_id])
        frame.loc[0, "final_validation_ce"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            study.choose(frame, 1)


class WorkflowTests(unittest.TestCase):
    def test_full_grid_exact_continuations_three_choices_and_no_test_loading(self):
        with campaign() as c:
            frozen = study.run(out=c.out)
            self.assertEqual(len(c.calls), 63)
            self.assertEqual(sum(h == 2 for _, h in c.calls), 54)
            self.assertEqual(sum(h == 3 for _, h in c.calls), 9)
            self.assertEqual(set(frozen["selected"]), set(study.METHODS))
            self.assertTrue(all(len(v) == 1 for v in frozen["selected"].values()))
            self.assertFalse(frozen["confirmation_launched"])
            self.assertEqual(len(pd.read_csv(c.out / "validation_comparison.csv")), 5)

    def test_interruption_and_repeat_skip_completed_work(self):
        with campaign(interrupt_after=3) as c:
            with self.assertRaises(KeyboardInterrupt):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 3)
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 63)
            self.assertEqual(len(set(c.calls)), 63)
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 63)

    def test_failed_family_stops_and_preserves_screening_or_final_ledger(self):
        with campaign(failed_screen_family="stc") as c:
            with self.assertRaisesRegex(RuntimeError, "All stc candidates failed"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 54)
            self.assertEqual(int(pd.read_csv(c.out / "screening.csv").state.eq("numerical_failure").sum()), 18)
        with campaign(failed_final_family="stc") as c:
            with self.assertRaisesRegex(RuntimeError, "All stc candidates failed"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 63)
            self.assertEqual(int(pd.read_csv(c.out / "finalists.csv").state.eq("numerical_failure").sum()), 3)
            self.assertFalse((c.out / "frozen_selection.json").exists())

    def test_rehashed_prefix_change_and_reference_only_wire_are_rejected(self):
        with campaign() as c:
            frozen = study.run(out=c.out)
            key = frozen["selected"]["dense"][0]
            p = study.BY_ID[key]
            write_point(p, 3, c.science, c.out, prefix_change=True)
            with self.assertRaises(AssertionError):
                study.analyze(c.out)
            write_point(p, 3, c.science, c.out, reference_wire=True)
            with self.assertRaisesRegex(RuntimeError, "codec or validation/model coverage differs"):
                study.verified_tables(p, 3, c.science, c.out)

    def test_wrong_split_or_changed_stage_identity_is_rejected(self):
        with campaign() as c:
            study.run(out=c.out)
            p = study.POINTS[0]
            file = study.paths(p, 2, c.out)["receipt"]
            receipt = json.loads(file.read_text())
            receipt["data_manifest"] = {"wrong": True}
            parent.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "split manifest differs"):
                study.verified_tables(p, 2, c.science, c.out)
            receipt["identity"]["partition_seed"] = 8000
            parent.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "stage identity differs"):
                study.read_receipt(p, 2, c.science, c.out)

    def test_prepare_never_requests_official_test_arrays(self):
        science = {"data_sha256": {"data_batch_1": "hash", "test_batch": "hash"},
                   "execution": {"runtime": {"synthetic": True}, "eval_batch_size": 4}}
        fake_host = SimpleNamespace(configure=lambda *_: "device", runtime=lambda _: {"synthetic": True})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent.atomic_json(MANIFEST, root / "prior/split_manifest.json")
            fed = SimpleNamespace(scope="validation", manifest=MANIFEST)
            with patch.dict(sys.modules, {"resnet14_transfer_host": fake_host}), patch.object(parent, "OUT", root / "prior"), \
                    patch.object(study, "science_spec", return_value=science), \
                    patch.object(parent.data_tools, "download_cifar10", return_value=Path("raw")), \
                    patch.object(parent.full.development, "file_hash", return_value="hash"), \
                    patch.object(parent.data_tools, "load_arrays", return_value=("images", "labels", None)) as loader, \
                    patch.object(parent.data_tools, "federation", return_value=(fed, pd.DataFrame())):
                study.prepare("cpu", 2, 4, root / "new")
                loader.assert_called_once_with(parent.DATA, include_test=False)
                loader.return_value = ("images", "labels", "prohibited_test_arrays")
                with self.assertRaisesRegex(RuntimeError, "Official test arrays are prohibited"):
                    study.prepare("cpu", 2, 4, root / "new")


if __name__ == "__main__":
    unittest.main()
