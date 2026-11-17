from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from neuromorphicfl.confirmation_metrics import (
    add_regularization_columns, interval, legacy_traffic_history, paired_endpoints,
    paired_targets, target_crossings, validate_native_traffic,
)
from experiments import stc_cifar10_confirmation as c
from experiments.stc_stage1 import smoke_federation


def fake_result(point) -> dict:
    rounds = np.asarray(c.EXPECTED_ROUNDS, dtype=np.int64)
    initial = 32 * c.LAYOUT.dimension + 64
    upload, server = rounds * 10_000, rounds * 800
    unicast = upload + 10 * (initial + 32 * rounds + server)
    broadcast = upload + initial + server
    config = c.config_for(point)
    history = pd.DataFrame({
        "round": rounds, "train_objective": 1.0, "train_ce": 0.9,
        "test_ce": 1.05, "test_accuracy": 0.55,
        "uplink_packetized_bits": upload, "stc_downlink_packetized_bits": server,
        "unicast_total_bits": unicast, "broadcast_total_bits": broadcast,
    })
    return {
        "architecture": "cifar_cnn", "method": "stc", "dimension": c.LAYOUT.dimension,
        "rounds": c.ROUNDS, "local_steps": config.local_steps, "local_lr": config.local_lr,
        "stc_upload_fraction": config.stc_upload_fraction,
        "stc_download_fraction": config.stc_download_fraction, "server_gain": 2.0,
        "final_train_objective": 1.0, "final_train_ce": 0.9,
        "final_test_ce": 1.05, "final_test_accuracy": 0.55,
        "uplink_packetized_bits": int(upload[-1]),
        "stc_downlink_packetized_bits": int(server[-1]),
        "unicast_hybrid_downlink_bits": int(unicast[-1] - upload[-1]),
        "unicast_hybrid_total_bits": int(unicast[-1]),
        "broadcast_total_bits": int(broadcast[-1]), "stc_downlink_messages": c.ROUNDS,
        "downlink_protocol": "stc_compressed_server_update", "history": history,
    }


class ProtocolTests(unittest.TestCase):
    def test_two_predeclared_points_and_ten_paired_seeds(self):
        spec = c.protocol_spec()
        self.assertEqual(spec["new_training_runs"], 20)
        self.assertEqual(c.SEEDS, tuple(range(3500, 4500, 100)))
        self.assertNotIn(c.dev.DEV_PARTITION, c.SEEDS)
        self.assertEqual(c.TARGETS, (0.45, 0.50, 0.55))
        for point, fraction in zip(c.POINTS, (1.0, 0.25)):
            config = c.config_for(point)
            self.assertEqual((config.stc_upload_fraction, config.stc_download_fraction), (fraction, fraction))
            self.assertEqual((config.server_gain, config.regularization, config.local_steps, config.local_lr), (2, 0.0005, 5, 0.05))

    def test_real_archives_have_exact_prefix_traffic_and_matching_seeds(self):
        for name in c.REFERENCE_IDS:
            for i in range(10):
                row, history, provenance = c.load_reference(name, i)
                self.assertEqual(row["partition_seed"], c.SEEDS[i])
                self.assertEqual(row["train_seed"], 80000 + c.SEEDS[i])
                self.assertEqual(int(history.iloc[-1].unicast_total_bits), int(row["unicast_hybrid_total_bits"]))
                self.assertIn(provenance["traffic_history_source"], ("exact_all_replay", "exact_all_checkpoints"))


class MetricsTests(unittest.TestCase):
    def test_objective_and_cross_entropy_remain_separate(self):
        result = add_regularization_columns(pd.DataFrame({"final_train_objective": [1.4], "final_train_ce": [0.9]}), 0.0005)
        self.assertAlmostEqual(result.final_train_l2_penalty.iloc[0], 0.5)
        self.assertAlmostEqual(result.final_parameter_l2_norm.iloc[0], np.sqrt(2000))

    def test_exact_archived_replay_and_checkpoint_costs(self):
        history = pd.DataFrame({"round": [1, 2], "uplink_packetized_bits": [80, 160]})
        row = {"rounds": 2, "dimension": 4, "replay_rounds": 2, "checkpoint_rounds": 0,
               "unicast_hybrid_total_bits": 992, "broadcast_total_bits": 512}
        replay = legacy_traffic_history(row, history, n_clients=2)
        self.assertEqual(replay.unicast_total_bits.tolist(), [688, 992])
        row.update(replay_rounds=0, checkpoint_rounds=2, unicast_hybrid_total_bits=1440, broadcast_total_bits=736)
        checkpoint = legacy_traffic_history(row, history, n_clients=2)
        self.assertEqual(checkpoint.unicast_total_bits.tolist(), [912, 1440])
        row.update(replay_rounds=1, checkpoint_rounds=1)
        with self.assertRaisesRegex(ValueError, "Mixed"):
            legacy_traffic_history(row, history, n_clients=2)

    def test_native_traffic_catches_wrong_downlink_reconstruction(self):
        result = fake_result(c.POINTS[0])
        history = result.pop("history")
        validate_native_traffic(result, history)
        history.loc[0, "unicast_total_bits"] += 1
        with self.assertRaisesRegex(ValueError, "prefix"):
            validate_native_traffic(result, history)

    def test_targets_use_first_saved_crossing_and_preserve_censoring(self):
        history = pd.DataFrame({"round": [1, 60, 120, 180], "test_accuracy": [0.4, 0.5, 0.49, 0.56],
                                "uplink_packetized_bits": [10, 20, 30, 40],
                                "unicast_total_bits": [100, 200, 300, 400], "broadcast_total_bits": [50, 60, 70, 80]})
        rows = target_crossings(history, targets=(0.5, 0.55, 0.6), horizon=180, comparison_id="a", partition_seed=1)
        self.assertEqual(rows[0]["first_sampled_round"], 60)
        self.assertEqual(rows[0]["unicast_total_bits"], 200)
        self.assertEqual(rows[1]["first_sampled_round"], 180)
        self.assertFalse(rows[2]["reached"])
        self.assertTrue(np.isnan(rows[2]["first_sampled_round"]))
        self.assertTrue(np.isnan(rows[2]["unicast_total_bits"]))
        self.assertEqual(rows[2]["censor_round"], 180)

    def test_ratio_interval_uses_geometric_mean_and_has_no_single_pair_ci(self):
        result = interval([1, 2, 4], ratio=True)
        self.assertAlmostEqual(result["estimate"], 2)
        self.assertLess(result["ci95_low"], 2)
        self.assertGreater(result["ci95_high"], 2)
        self.assertTrue(np.isnan(interval([2], ratio=True)["ci95_low"]))

    def test_endpoint_pairs_align_by_seed_rather_than_row_order(self):
        common = {"final_test_ce": 1.0, "final_train_objective": 1.1, "final_train_l2_penalty": 0.1,
                  "unicast_hybrid_total_bits": 200, "broadcast_total_bits": 100,
                  "uplink_packetized_bits": 80, "unicast_hybrid_downlink_bits": 120}
        rows = [{**common, "comparison_id": "stc_primary", "partition_seed": 1, "final_test_accuracy": 0.6},
                {**common, "comparison_id": "stc_primary", "partition_seed": 2, "final_test_accuracy": 0.7},
                {**common, "comparison_id": "event", "partition_seed": 2, "final_test_accuracy": 0.65},
                {**common, "comparison_id": "event", "partition_seed": 1, "final_test_accuracy": 0.55}]
        pairs, summary = paired_endpoints(pd.DataFrame(rows), ("stc_primary",), ("event",))
        differences = pairs[pairs.metric == "test_accuracy_difference_pp"]
        np.testing.assert_allclose(differences.value, [5, 5])
        row = summary[summary.metric == "test_accuracy_difference_pp"].iloc[0]
        self.assertAlmostEqual(row.estimate, 5)
        self.assertTrue(row.primary_endpoint)
        with self.assertRaisesRegex(ValueError, "seed sets"):
            paired_endpoints(pd.DataFrame(rows[:-1]), ("stc_primary",), ("event",))

    def test_target_pair_summary_reports_missing_hits(self):
        rows = [{"comparison_id": "stc_primary", "partition_seed": i, "target_accuracy": 0.5,
                 "reached": i == 1, "unicast_total_bits": 100 if i == 1 else np.nan} for i in (1, 2)]
        rows += [{"comparison_id": "event", "partition_seed": i, "target_accuracy": 0.5,
                  "reached": True, "unicast_total_bits": 200} for i in (1, 2)]
        result = paired_targets(pd.DataFrame(rows), ("stc_primary",), ("event",)).iloc[0]
        self.assertEqual((result.n_total, result.n_stc_reached, result.n_reference_reached, result.n_common_hits), (2, 1, 2, 1))
        self.assertAlmostEqual(result.estimate, 0.5)


class EvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture(out):
        contract = {"scope": "synthetic_evidence_test"}
        for point in c.POINTS:
            for i in range(10):
                c.save_result(point, i, contract, fake_result(point), out, 0.01)
        return contract

    def test_full_grid_verifies_and_crlf_checkout_preserves_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            rows, histories, states = c.verified_runs(contract, out)
            self.assertEqual((len(rows), len(histories), len(states)), (20, 20, 20))
            for path in out.iterdir():
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            self.assertEqual(len(c.verified_runs(contract, out)[0]), 20)

    def test_changed_numeric_value_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            summary = c.paths(c.POINTS[0], 0, out)[0]
            frame = pd.read_csv(summary)
            frame["final_test_accuracy"] = 0.99
            c.atomic_csv(frame, summary)
            with self.assertRaisesRegex(RuntimeError, "changed confirmation"):
                c.read_receipt(c.POINTS[0], 0, contract, out)

    def test_seed_mislabel_is_rejected_even_if_receipt_checksum_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            summary, _, receipt_path = c.paths(c.POINTS[0], 0, out)
            frame = pd.read_csv(summary)
            frame["partition_seed"] = 4500
            c.atomic_csv(frame, summary)
            receipt = json.loads(receipt_path.read_text())
            receipt["summary_text_sha256"] = c.text_hash(summary)
            c.atomic_json(receipt, receipt_path)
            with self.assertRaisesRegex(RuntimeError, "metadata differs"):
                c.verified_runs(contract, out)

    def test_numerical_failure_blocks_inference_instead_of_dropping_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            receipt_path = c.paths(c.POINTS[0], 0, out)[2]
            receipt = json.loads(receipt_path.read_text())
            receipt.update(state="numerical_failure", error="test overflow")
            c.atomic_json(receipt, receipt_path)
            with patch.object(c, "load_execution", return_value=contract):
                with self.assertRaisesRegex(RuntimeError, "dropping failed seeds"):
                    c.analyze(out)
            self.assertTrue((out / "completion_audit.csv").exists())
            self.assertFalse((out / "paired_summary.csv").exists())

    def test_complete_synthetic_evidence_exercises_the_full_analysis_pipeline(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            contract = self.fixture(out)
            with patch.object(c, "load_execution", return_value=contract):
                c.analyze(out)
            pairs = pd.read_csv(out / "paired_summary.csv")
            targets = pd.read_csv(out / "traffic_to_target.csv")
            self.assertEqual(len(pairs), 2 * 5 * 8)
            self.assertEqual(len(targets), 7 * 10 * 3)
            self.assertEqual(int(pairs.primary_endpoint.sum()), 1)

    def test_existing_different_execution_contract_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c.ensure_execution({"version": 1}, out)
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                c.ensure_execution({"version": 2}, out)


class WorkerTests(unittest.TestCase):
    def test_worker_keeps_real_test_slot_and_reuses_completed_result(self):
        fed = smoke_federation()
        point = c.POINTS[0]
        runtime = c.dev.runtime_manifest()
        contract = {"science": {"source_hashes": c.source_hashes()}, "runtime": runtime, "data_sha256": {}}

        def fake_run(**kwargs):
            self.assertIs(kwargs["federation"].X_test, fed.X_test)
            self.assertIsNot(kwargs["federation"].X_test, fed.X_train_eval)
            self.assertEqual(kwargs["seed"], 83500)
            self.assertEqual(kwargs["config"], c.config_for(point))
            return fake_result(point)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(c.dev, "build_contract", return_value={"data_sha256": {}}), patch.object(c, "make_cifar10_federation", return_value=fed), patch.object(c, "run_final_baseline", side_effect=fake_run) as runner:
                self.assertIn("done", c.run_job(point, 0, contract, out))
                self.assertIn("reuse", c.run_job(point, 0, contract, out))
                self.assertEqual(runner.call_count, 1)


if __name__ == "__main__":
    unittest.main()
