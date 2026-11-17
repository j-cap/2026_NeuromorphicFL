from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig, run_final_baseline
from experiments import stc_cifar10_development as campaign
from experiments.stc_stage1 import smoke_federation


class CheckpointTests(unittest.TestCase):
    def test_interrupted_and_resumed_runs_match_continuous_for_all_methods(self) -> None:
        fed = smoke_federation()
        base = FinalBaselineConfig(
            local_steps=1, local_lr=0.05, batch_size=2, rounds=4,
            eval_stride=1, init_scale=1.0, server_gain=2.0,
        )
        for method in ("stc", "event", "strom", "ef_topk", "sign_ef", "dense"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                kwargs = dict(federation=fed, architecture="cifar_cnn", method=method,
                              config=base, seed=83500, record_history=True)
                expected = run_final_baseline(**kwargs)
                store = BaselineCheckpoint(Path(directory) / "state.npz", {"dataset": "synthetic"})

                def interrupt(row):
                    if row["round"] == 2:
                        raise InterruptedError("simulated interruption")

                with self.assertRaises(InterruptedError):
                    run_final_baseline(**kwargs, checkpoint=store, checkpoint_stride=2, progress_callback=interrupt)
                with np.load(store.path, allow_pickle=False) as archive:
                    saved = json.loads(str(archive["metadata"].item()))
                self.assertEqual(saved["progress"]["completed_round"], 2)
                actual = run_final_baseline(**kwargs, checkpoint=store, checkpoint_stride=2)
                # A fully completed checkpoint must also be reusable exactly.
                again = run_final_baseline(**kwargs, checkpoint=store, checkpoint_stride=2)
                for key, value in expected.items():
                    if isinstance(value, pd.DataFrame):
                        pd.testing.assert_frame_equal(value, actual[key], check_exact=True)
                        pd.testing.assert_frame_equal(value, again[key], check_exact=True)
                    else:
                        self.assertEqual(value, actual[key], key)
                        self.assertEqual(value, again[key], key)

    def test_changed_configuration_or_provenance_cannot_resume(self) -> None:
        config = FinalBaselineConfig(rounds=1, local_steps=1, batch_size=2, eval_stride=1)
        kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="stc", seed=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            store = BaselineCheckpoint(path, {"source": "original"})
            run_final_baseline(**kwargs, config=config, checkpoint=store)
            with self.assertRaisesRegex(ValueError, "identity/provenance"):
                run_final_baseline(**kwargs, config=replace(config, stc_upload_fraction=0.1), checkpoint=store)
            with self.assertRaisesRegex(ValueError, "identity/provenance"):
                run_final_baseline(**kwargs, config=config, checkpoint=BaselineCheckpoint(path, {"source": "changed"}))

    def test_failed_write_preserves_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BaselineCheckpoint(Path(directory) / "state.npz", {})
            store.save({}, {"model": np.ones(2, dtype=np.float32)}, {"round": 1})
            before = store.path.read_bytes()
            with patch("neuromorphicfl.baseline_checkpoint.np.savez_compressed", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    store.save({}, {"model": np.zeros(2, dtype=np.float32)}, {"round": 2})
            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(list(Path(directory).glob("*.tmp-*")), [])

    def test_nonfinite_delta_is_reported_as_numerical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BaselineCheckpoint(Path(directory) / "state.npz", {})
            with patch("neuromorphicfl.final_baseline_campaign._local_delta", return_value=np.array([np.inf], dtype=np.float32)):
                with self.assertRaisesRegex(FloatingPointError, "nonfinite local delta"):
                    run_final_baseline(
                        federation=smoke_federation(), architecture="cifar_cnn", method="stc",
                        config=FinalBaselineConfig(rounds=1), checkpoint=store,
                    )


class SelectionTests(unittest.TestCase):
    @staticmethod
    def rows(scores=(0.8, 0.9, 1.0), bits=(120, 90, 50)):
        rows = [{"point_id": campaign.REFERENCE.point_id, "method": "event",
                 "state": "complete", "final_train_objective": 1.2,
                 "unicast_hybrid_total_bits": 100}]
        for point, score, traffic in zip(campaign.CANDIDATES, scores, bits):
            rows.append({"point_id": point.point_id, "method": "stc", "state": "complete",
                         "final_train_objective": score, "unicast_hybrid_total_bits": traffic})
        return pd.DataFrame(rows)

    def test_quality_and_budget_choices_use_training_objective(self) -> None:
        frame = self.rows()
        selected = campaign.choose(frame)
        self.assertEqual(selected["quality"]["point_id"], campaign.CANDIDATES[0].point_id)
        self.assertEqual(selected["within_event_budget"]["point_id"], campaign.CANDIDATES[1].point_id)
        self.assertTrue(selected["nearest_traffic_diagnostic"]["within_ten_percent"])
        frame["final_test_accuracy"] = [0.99, 0.01, 0.1, 1.0]
        self.assertEqual(campaign.choose(frame), selected)

    def test_no_feasible_point_is_reported_and_not_called_matched(self) -> None:
        selected = campaign.choose(self.rows(bits=(300, 200, 150)))
        self.assertIsNone(selected["within_event_budget"])
        self.assertFalse(selected["nearest_traffic_diagnostic"]["within_ten_percent"])
        self.assertEqual(selected["nearest_traffic_diagnostic"]["relative_budget_gap"], 0.5)

    def test_ties_favor_lower_traffic_and_failed_points_are_excluded(self) -> None:
        frame = self.rows(scores=(1.0, 1.0, 0.1))
        frame.loc[3, "state"] = "numerical_failure"
        selected = campaign.choose(frame)
        self.assertEqual(selected["quality"]["point_id"], campaign.CANDIDATES[1].point_id)

    def test_same_point_can_fill_both_roles(self) -> None:
        selected = campaign.choose(self.rows(scores=(1, 0.5, 1)))
        self.assertEqual(selected["quality"], selected["within_event_budget"])

    def test_protocol_excludes_evaluation_seeds_and_has_fixed_search(self) -> None:
        spec = campaign.protocol_spec()
        self.assertEqual(len(campaign.CANDIDATES), 24)
        self.assertEqual(len(campaign.POINT_BY_ID), 25)
        self.assertNotIn(campaign.DEV_PARTITION, campaign.EVAL_PARTITIONS)
        self.assertEqual(spec["selection_metric"], "final_train_objective")
        self.assertEqual(spec["evaluation_scope"], "development_training_only")
        self.assertEqual(campaign.config_for(campaign.REFERENCE).local_steps, 5)
        self.assertEqual(campaign.config_for(campaign.REFERENCE).local_lr, 0.05)
        self.assertTrue(any(p.upload_fraction != p.download_fraction for p in campaign.CANDIDATES))


class EvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture(out: Path) -> dict:
        contract = {"protocol": campaign.protocol_spec(), "source_hashes": campaign.source_hashes(),
                    "runtime": {"scope": "test"}, "data_sha256": {}}
        campaign.ensure_contract(out, contract)
        for index, point in enumerate(campaign.POINTS):
            score = 1.0 + index / 100
            history = pd.DataFrame({
                "round": [1] + list(range(60, campaign.ROUNDS + 1, 60)),
                "train_objective": score, "train_ce": score,
                "test_accuracy": 0.123, "unicast_total_bits": 200,
            })
            result = {"method": point.method, "rounds": campaign.ROUNDS,
                      "final_train_objective": score, "final_train_ce": score,
                      "final_test_accuracy": 0.123, "final_test_ce": 2.0,
                      "uplink_packetized_bits": 100, "unicast_hybrid_downlink_bits": 100,
                      "unicast_hybrid_total_bits": 200, "history": history}
            campaign.save_result(point, contract, result, out, 0.01)
        return contract

    def test_receipts_validate_grid_and_selection_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            out = Path(directory)
            contract = self.fixture(out)
            frame = campaign.verified_rows(contract, out)
            self.assertEqual(len(frame), 25)
            self.assertFalse(any("test" in key or "accuracy" in key for key in frame.columns))
            selection = campaign.select(out)
            self.assertEqual(campaign.select(out), selection)
            frozen = out / "frozen_selection.json"
            frozen.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different frozen selection"):
                campaign.select(out)

    def test_corrupted_csv_is_rejected_instead_of_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            point = campaign.CANDIDATES[0]
            campaign.paths(point, out)[0].write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Missing or changed summary"):
                campaign.read_receipt(point, contract, out)

    def test_mislabeled_seed_is_rejected_even_with_updated_receipt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            contract = self.fixture(out)
            point = campaign.CANDIDATES[0]
            summary, _, receipt_path = campaign.paths(point, out)
            frame = pd.read_csv(summary)
            frame["partition_seed"] = 3500
            campaign.atomic_csv(frame, summary)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["summary_sha256"] = campaign.file_hash(summary)
            campaign.atomic_json(receipt, receipt_path)
            with self.assertRaisesRegex(RuntimeError, "Summary metadata drift"):
                campaign.verified_rows(contract, out)

    def test_changed_contract_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            campaign.ensure_contract(out, {"original": True})
            with self.assertRaisesRegex(RuntimeError, "changed"):
                campaign.ensure_contract(out, {"original": False})

    def test_duplicate_worker_cannot_acquire_same_point_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.lock"
            with campaign.point_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with campaign.point_lock(path):
                        self.fail("second lock must not be acquired")
            with campaign.point_lock(path):
                pass


class WorkerTests(unittest.TestCase):
    def test_worker_evaluates_training_only_and_reuses_completed_receipt(self) -> None:
        fed = smoke_federation()
        point = campaign.CANDIDATES[0]
        contract = {"scope": "worker-test"}

        def fake_run(**kwargs):
            self.assertIs(kwargs["federation"].X_test, fed.X_train_eval)
            self.assertIs(kwargs["federation"].y_test, fed.y_train_eval)
            self.assertEqual(kwargs["seed"], campaign.TRAIN_SEED)
            self.assertEqual(kwargs["config"], campaign.config_for(point))
            return {"final_train_objective": 1.0, "final_test_accuracy": 0.5,
                    "history": pd.DataFrame({"round": [1], "train_objective": [1.0], "test_accuracy": [0.5]})}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(campaign, "build_contract", return_value=contract), patch.object(campaign, "make_cifar10_federation", return_value=fed), patch.object(campaign, "run_final_baseline", side_effect=fake_run) as runner:
                self.assertIn("done", campaign.run_point(point, contract, out=out))
                self.assertIn("reuse", campaign.run_point(point, contract, out=out))
                self.assertEqual(runner.call_count, 1)
            summary = pd.read_csv(campaign.paths(point, out)[0])
            self.assertNotIn("final_test_accuracy", summary)

    def test_numerical_failure_is_retained_and_not_retried_silently(self) -> None:
        point = campaign.CANDIDATES[0]
        contract = {"scope": "worker-test"}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(campaign, "build_contract", return_value=contract), patch.object(campaign, "make_cifar10_federation", return_value=smoke_federation()), patch.object(campaign, "run_final_baseline", side_effect=FloatingPointError("diverged")) as runner:
                self.assertIn("numerical failure", campaign.run_point(point, contract, out=out))
                self.assertIn("reuse", campaign.run_point(point, contract, out=out))
                self.assertEqual(runner.call_count, 1)
            receipt = campaign.read_receipt(point, contract, out)
            self.assertEqual(receipt["state"], "numerical_failure")
            self.assertFalse(campaign.paths(point, out)[0].exists())

    def test_codec_error_is_fatal_and_not_recorded_as_numerical_failure(self) -> None:
        point = campaign.CANDIDATES[0]
        contract = {"scope": "worker-test"}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(campaign, "build_contract", return_value=contract), patch.object(campaign, "make_cifar10_federation", return_value=smoke_federation()), patch.object(campaign, "run_final_baseline", side_effect=ValueError("invalid STC tensor metadata")):
                with self.assertRaisesRegex(ValueError, "invalid STC tensor"):
                    campaign.run_point(point, contract, out=out)
            self.assertFalse(campaign.paths(point, out)[2].exists())


if __name__ == "__main__":
    unittest.main()
