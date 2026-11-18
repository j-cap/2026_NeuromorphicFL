from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import event_codec_full_horizon as full
from experiments.stc_stage1 import smoke_federation
from neuromorphicfl.event_codec_pilot import run_codec_pilot, traffic_summary


def short_config(point="event", rounds=3, stride=2):
    return replace(full.config_for(point), rounds=rounds, eval_stride=stride, local_steps=1, batch_size=2)


def fake_science():
    protocol = {**full.protocol_spec(), "rounds": 2, "evaluation_rounds": [1, 2],
                "n_clients": 2, "dimension": 20570,
                "points": {point: full.asdict(short_config(point, 2, 1)) for point in full.POINTS}}
    return {"protocol": protocol, "source_hashes": full.source_hashes(), "runtime": {"synthetic": True},
            "archived_event_reference": full.event_reference()[2],
            "pilot_protocol_sha256": full.text_hash(full.pilot.OUT / "protocol.json")}


class ProtocolTests(unittest.TestCase):
    def test_fixed_seed_horizon_and_comparator_settings(self):
        self.assertEqual((full.ROUNDS, full.PARTITION_SEED, full.TRAIN_SEED), (1800, 3500, 83500))
        self.assertEqual(full.TARGETS, (.45, .50, .55))
        self.assertEqual(full.sampled_rounds(full.config_for("event")), (1,) + tuple(range(60, 1801, 60)))
        for point, gain, fraction in (("event", 1, .01), ("stc_primary", 2, 1), ("stc_sparse", 2, .25)):
            config = full.config_for(point)
            self.assertEqual((config.rounds, config.eval_stride, config.local_steps, config.local_lr, config.regularization),
                             (1800, 60, 5, .05, .0005))
            self.assertEqual((config.server_gain, config.stc_upload_fraction, config.stc_download_fraction),
                             (gain, fraction, fraction))

    def test_existing_pilot_and_archived_event_still_validate(self):
        full.pilot.load_contract()
        row, history, provenance = full.event_reference()
        self.assertEqual((row["partition_seed"], row["train_seed"], len(history)), (3500, 83500, 31))
        self.assertEqual(provenance["traffic_history_source"], "exact_all_replay")

    def test_runtime_mismatch_is_rejected_before_any_download(self):
        with patch.object(full.development, "runtime_manifest", return_value={"different": True}), patch.object(full, "download_cifar10") as download:
            with self.assertRaisesRegex(RuntimeError, "same recorded environment"):
                full.build_contract()
            download.assert_not_called()


class PredictiveCheckpointTests(unittest.TestCase):
    def test_evaluator_reads_real_test_slot_and_saves_metrics_with_model(self):
        fed = smoke_federation()
        config = short_config(rounds=2, stride=1)
        w = np.array([1., 2.], dtype=np.float32)
        model_hash = hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = full.PredictiveCheckpoint(Path(folder) / "model.npz", {"test": True},
                                                   federation=fed, architecture="cifar_cnn", config=config)
            seen = []

            def metrics(weights, X, y, **kwargs):
                seen.append((X, y))
                return 1., .7, .6, .5, .4, np.zeros(10)

            checkpoint.metrics = metrics
            checkpoint.save({"identity": 1}, {"model": w}, {"completed_round": 1,
                            "records": [{"round": 1, "model_sha256": model_hash}]})
            self.assertIs(seen[0][0], fed.X_test)
            self.assertIs(seen[0][1], fed.y_test)
            saved = checkpoint.load({"identity": 1})
            self.assertEqual(saved["progress"]["predictive_history"][0]["test_ce"], .7)
            self.assertEqual(saved["progress"]["predictive_history"][0]["model_sha256"], model_hash)

    def test_failed_checkpoint_write_retains_previous_model_and_prediction_pair(self):
        fed = smoke_federation()
        config = short_config(rounds=2, stride=1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = full.PredictiveCheckpoint(Path(folder) / "model.npz", {"test": True},
                                                   federation=fed, architecture="cifar_cnn", config=config)
            checkpoint.metrics = lambda *a, **k: (1., .7, .6, .5, .4, np.zeros(10))
            records = []
            for rnd in (1, 2):
                w = np.full(2, rnd, dtype=np.float32)
                records.append({"round": rnd, "model_sha256": hashlib.sha256(w.astype("<f4").tobytes()).hexdigest()})
                if rnd == 1:
                    checkpoint.save({}, {"model": w}, {"completed_round": rnd, "records": records.copy()})
                else:
                    with patch.object(full.BaselineCheckpoint, "save", side_effect=OSError("synthetic write failure")):
                        with self.assertRaises(OSError):
                            checkpoint.save({}, {"model": w}, {"completed_round": rnd, "records": records.copy()})
            saved = checkpoint.load({})
            np.testing.assert_array_equal(saved["arrays"]["model"], [1, 1])
            self.assertEqual([r["round"] for r in checkpoint.evaluations], [1])

    def test_missing_prediction_history_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = full.PredictiveCheckpoint(Path(folder) / "bad.npz", {}, federation=smoke_federation(),
                                                   architecture="cifar_cnn", config=short_config())
            checkpoint.base.save({}, {"model": np.ones(2, dtype=np.float32)},
                                 {"completed_round": 1, "records": [{"round": 1, "model_sha256": "x"}]})
            with self.assertRaisesRegex(ValueError, "coverage"):
                checkpoint.load({})

    def test_prediction_evaluation_preserves_original_host_for_all_points(self):
        with tempfile.TemporaryDirectory() as folder:
            for point in full.POINTS:
                config = short_config(point)
                fed = smoke_federation()
                kwargs = dict(federation=fed, architecture="cifar_cnn", method="event" if point == "event" else "stc",
                              config=config, seed=full.TRAIN_SEED)
                original = run_codec_pilot(**kwargs)
                actual = full.evaluated_run(**kwargs, checkpoint_path=Path(folder) / f"{point}.npz", provenance={})
                np.testing.assert_array_equal(original["model"].view(np.uint32), actual["model"].view(np.uint32))
                self.assertEqual(original["rng_state"], actual["rng_state"])
                pd.testing.assert_frame_equal(original["summary"], actual["summary"])
                self.assertEqual(actual["evaluation"]["round"].tolist(), [1, 2, 3])
                _, _, _, metrics = full._architecture_ops("cifar_cnn")
                expected = metrics(actual["model"], fed.X_test, fed.y_test, layout=full.LAYOUT, regularization=config.regularization)
                self.assertEqual(actual["evaluation"].iloc[-1].test_ce, expected[1])
                self.assertEqual(actual["evaluation"].iloc[-1].test_accuracy, expected[2])

    def test_resume_keeps_all_sampled_predictions_without_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            for point in full.POINTS:
                kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="event" if point == "event" else "stc",
                              config=short_config(point, rounds=4), seed=83500, provenance={"point": point})
                uninterrupted = full.evaluated_run(**kwargs, checkpoint_path=Path(folder) / f"{point}_full.npz")
                path = Path(folder) / f"{point}_resume.npz"
                full.evaluated_run(**kwargs, checkpoint_path=path, stop_after=2)
                resumed = full.evaluated_run(**kwargs, checkpoint_path=path)
                pd.testing.assert_frame_equal(uninterrupted["evaluation"], resumed["evaluation"])
                self.assertEqual(resumed["evaluation"]["round"].tolist(), [1, 2, 4])


class MetricTests(unittest.TestCase):
    def test_regression_tolerates_archive_print_precision_and_rejects_changed_accuracy(self):
        reference = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1., .9],
                                  "test_ce": [1.1, 1.], "test_accuracy": [.4, .5], "macro_accuracy": [.4, .5],
                                  "worst_class_accuracy": [.1, .2], "uplink_packetized_bits": [100, 200]})
        evaluation = reference.drop(columns="uplink_packetized_bits").copy()
        evaluation["test_ce"] += 1e-10
        tables = {"evaluation": evaluation, "rounds": pd.DataFrame({"legacy_uplink_bits": [100, 100]})}
        with patch.object(full, "event_reference", return_value=({}, reference, {})):
            self.assertEqual(full.reference_check("event", tables)["status"], "matched")
            tables["evaluation"].loc[1, "test_accuracy"] = .51
            with self.assertRaisesRegex(RuntimeError, "regression differs"):
                full.reference_check("event", tables)

    def test_prefix_traffic_uses_actual_replay_checkpoint_decisions(self):
        rounds = pd.DataFrame([{"round": r, "method": "event", "legacy_uplink_bits": k,
                                "legacy_server_bits": k, "codec_uplink_bits": 40,
                                "codec_replay_bits": 40, "exact_downlink_bits": 32}
                               for r, k in ((1, 100), (2, 1000))])
        evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1., .9], "train_ce": [.9, .8],
                                   "test_ce": [1., .9], "test_accuracy": [.4, .5], "macro_accuracy": [.4, .5],
                                   "worst_class_accuracy": [.1, .2], "model_sha256": ["a", "b"]})
        result = {"rounds": rounds, "evaluation": evaluation,
                  "summary": traffic_summary(rounds, n_clients=2, dimension=4)}
        _, history = full.trajectory_tables(result, n_clients=2, dimension=4)
        legacy = history[history.variant == "legacy"]
        self.assertEqual(legacy.server_stream_bits.tolist(), [100, 292])
        self.assertEqual(legacy.unicast_total_bits.tolist(), [100 + 2 * (192 + 32 + 100),
                                                           1100 + 2 * (192 + 64 + 292)])

    def test_targets_use_first_sample_and_censor_unreached_targets(self):
        history = pd.DataFrame({"point": "event", "variant": "both_codecs", "round": [1, 60, 120, 1800],
                                "test_accuracy": [.44, .51, .49, .54], "uplink_bits": [10, 20, 30, 40],
                                "unicast_downlink_bits": [90, 180, 270, 360],
                                "unicast_total_bits": [100, 200, 300, 400], "broadcast_total_bits": [80, 160, 240, 320]})
        result = full.targets_table(history)
        self.assertEqual(result.reached.tolist(), [True, True, False])
        self.assertEqual(result.first_sampled_round.iloc[:2].tolist(), [60, 60])
        self.assertTrue(np.isnan(result.iloc[-1].unicast_total_bits))
        self.assertEqual(result.iloc[-1].censor_round, 1800)


class EvidenceTests(unittest.TestCase):
    def test_complete_synthetic_results_reconstruct_and_analyze(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(full, "ROUNDS", 2), patch.object(full, "protocol_spec", return_value=science["protocol"]):
            out = Path(folder)
            full.atomic_json(science, out / "protocol.json")
            for point in full.POINTS:
                result = full.evaluated_run(federation=smoke_federation(), architecture="cifar_cnn",
                                            method="event" if point == "event" else "stc",
                                            config=short_config(point, 2, 1), seed=83500,
                                            checkpoint_path=out / f"{point}.npz", provenance={"synthetic": True})
                full.save_result(point, science, result, out)
                full.verified_point(point, science, out)
            with patch.object(full, "reference_check", return_value={"status": "synthetic"}), patch.object(full, "plot"), redirect_stdout(io.StringIO()):
                full.analyze(out)
            self.assertEqual(len(pd.read_csv(out / "comparison.csv")), 8)
            self.assertEqual(len(pd.read_csv(out / "traffic_to_target.csv")), 24)
            self.assertEqual(len(pd.read_csv(out / "comparison_history.csv")), 16)
            file = full.paths("event", out)["evaluation"]
            file.write_bytes(file.read_bytes().replace(b"\n", b"\r\n"))
            full.read_receipt("event", science, out)
            # Even a rehashed CSV must retain the declared seed metadata.
            table = pd.read_csv(file)
            table["partition_seed"] = 999
            full.atomic_csv(table, file)
            receipt_file = full.paths("event", out)["receipt"]
            receipt = json.loads(receipt_file.read_text())
            receipt["sha256"]["evaluation"] = full.text_hash(file)
            full.atomic_json(receipt, receipt_file)
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                full.verified_point("event", science, out)

    def test_failed_point_blocks_analysis(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(full, "protocol_spec", return_value=science["protocol"]):
            out = Path(folder)
            full.atomic_json(science, out / "protocol.json")
            full.atomic_json({"point": "event", "contract_sha256": full.digest(science), "state": "numerical_failure"},
                             full.paths("event", out)["receipt"])
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                full.analyze(out)

    def test_worker_keeps_test_slot_and_reuses_completed_receipt(self):
        science = fake_science()
        fed = smoke_federation()
        with tempfile.TemporaryDirectory() as folder, patch.object(full, "build_contract", return_value=science), patch.object(full, "make_cifar10_federation", return_value=fed), patch.object(full, "config_for", return_value=short_config(rounds=2, stride=1)), redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn("done event", full.run_point("event", science, out))
            saved = pd.read_csv(full.paths("event", out)["evaluation"])
            self.assertEqual(len(saved), 2)
            with patch.object(full, "evaluated_run", side_effect=AssertionError("must reuse")):
                self.assertIn("reuse event", full.run_point("event", science, out))


if __name__ == "__main__":
    unittest.main()
