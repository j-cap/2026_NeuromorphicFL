from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import event_strom_fresh_confirmation as fresh
from experiments.stc_stage1 import smoke_federation


def fake_science():
    return {"protocol": {"rounds": 2, "evaluation_rounds": [1, 2], "n_clients": 2, "dimension": 4},
            "selection_provenance": {"synthetic": True}, "seed_freshness": fresh.seed_freshness()}


def fake_result(point, index=0):
    rounds = pd.DataFrame({"round": [1, 2], "method": point,
                           "legacy_uplink_bits": [160, 160], "codec_uplink_bits": [80, 80],
                           "legacy_server_bits": [160, 160], "codec_replay_bits": [80, 80],
                           "exact_downlink_bits": [64, 64],
                           "model_sha256": [f"{index * 10 + r:064x}" for r in (1, 2)]})
    evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1.1, 1.0],
                               "test_ce": [1.2, 1.1], "test_accuracy": [.46, .56], "macro_accuracy": [.46, .56],
                               "worst_class_accuracy": [.2, .3], "model_sha256": rounds.model_sha256})
    return {"rounds": rounds, "evaluation": evaluation}


def paired_runs():
    rows = []
    for i, seed in enumerate(fresh.SEEDS):
        for point in fresh.POINTS:
            multiplier = 2 if point == "event" else 1
            rows.append({"point": point, "partition_seed": seed,
                         "final_test_accuracy": .6 + .001 * i + (.01 if point == "strom" else 0),
                         "final_test_ce": 1.1, "final_train_objective": 1.2, "final_train_l2_penalty": .1,
                         **{field: multiplier * (1000 + 100 * i) for field in
                            ("uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits")}})
    return pd.DataFrame(rows)


class ProtocolTests(unittest.TestCase):
    def test_frozen_methods_fresh_paired_seed_grid_and_no_reuse(self):
        event, strom = (fresh.config_for(p) for p in fresh.POINTS)
        self.assertEqual((event.threshold, event.jump0, event.rounds), (.05, .01, 1800))
        self.assertEqual((strom.strom_threshold, strom.server_gain, strom.local_lr, strom.rounds), (.05, .10, .05, 1800))
        self.assertEqual(asdict(event), asdict(fresh.development.config_for_id(fresh.EVENT_ID)))
        self.assertEqual(asdict(strom), asdict(fresh.development.config_for_id(fresh.development.STROM_ID)))
        self.assertEqual(fresh.SEEDS, tuple(range(6000, 7000, 100)))
        self.assertEqual([fresh.train_seed(i) for i in range(10)], list(range(86000, 87000, 100)))
        protocol = fresh.protocol_spec()
        self.assertEqual((protocol["new_training_runs"], protocol["reused_training_results"]), (20, 0))
        self.assertFalse(protocol["development_checkpoint_reuse"])
        self.assertEqual(len(protocol["evaluation_rounds"]), 31)
        self.assertEqual(fresh.seed_freshness()["overlap"], [])
        self.assertFalse(set(fresh.SEEDS) & set(fresh.seed_freshness()["recorded_prior_partition_seeds"]))
        with patch.object(fresh, "SEEDS", (4600,)):
            with self.assertRaisesRegex(RuntimeError, "overlaps"):
                fresh.seed_freshness()
        for index in (-1, 10):
            with self.assertRaises(ValueError):
                fresh.train_seed(index)

    def test_real_selection_and_science_do_not_open_old_test_curves_or_images(self):
        with patch.object(fresh.full, "load_contract", side_effect=AssertionError("no old test curves")), \
                patch.object(fresh.full, "build_contract", side_effect=AssertionError("no old test curves")), \
                patch.object(fresh.full, "download_cifar10", side_effect=AssertionError("no image load")), \
                patch.object(fresh.full, "make_cifar10_federation", side_effect=AssertionError("no image load")):
            science = fresh.build_science()
        self.assertEqual(len(science["data_sha256"]), 6)
        self.assertEqual(science["selection_provenance"]["configs"]["event"]["jump0"], .01)
        self.assertIn("experiments/event_strom_fresh_confirmation.py", science["source_hashes"])

    def test_changed_selection_and_runtime_mismatch_fail_before_download(self):
        source = {"protocol": {"fixed_Strom_config": asdict(fresh.config_for("strom"))}}
        frozen = {"selection": {"quality": {"point_id": "wrong"}}}
        with patch.object(fresh.development_audit, "validate", return_value=(source, {}, frozen, [])):
            with self.assertRaisesRegex(RuntimeError, "Frozen Event/Ström selection differs"):
                fresh.selection_provenance()
        with patch.object(fresh, "build_science", return_value={"runtime": {"wrong": True}}), \
                patch.object(fresh.full, "download_cifar10") as download:
            with self.assertRaisesRegex(RuntimeError, "recorded Windows"):
                fresh.build_contract()
            download.assert_not_called()

    def test_default_queue_trains_both_methods_for_all_ten_seeds_including_first(self):
        science, jobs = fake_science(), []
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "build_contract", return_value=science), \
                patch.object(fresh, "run_point", side_effect=lambda p, i, s, out: jobs.append((p, i)) or "synthetic"), \
                redirect_stdout(io.StringIO()):
            fresh.run(1, out=Path(folder))
        self.assertEqual(jobs, [(p, i) for i in range(10) for p in fresh.POINTS])
        self.assertEqual(len(set(jobs)), 20)

    def test_subset_queue_and_immutable_existing_contract(self):
        science, jobs = fake_science(), []
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "build_contract", return_value=science), \
                patch.object(fresh, "run_point", side_effect=lambda p, i, s, out: jobs.append((p, i)) or "synthetic") as worker, \
                redirect_stdout(io.StringIO()):
            out = Path(folder)
            fresh.run(1, point="strom", seed_index=9, out=out)
            self.assertEqual(jobs, [("strom", 9)])
            original = {"different": "campaign"}
            fresh.atomic_json(original, out / "protocol.json")
            worker.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "Existing frozen decision differs"):
                fresh.run(1, out=out)
            self.assertEqual(json.loads((out / "protocol.json").read_text()), original)
            worker.assert_not_called()


class EvidenceTests(unittest.TestCase):
    def test_both_methods_roundtrip_crlf_and_relabelled_metadata_rejection(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "ROUNDS", 2):
            out = Path(folder)
            for point in fresh.POINTS:
                fresh.save_result(point, 0, science, fake_result(point), out)
                fresh.verified_tables(point, 0, science, out)
                file = fresh.paths(point, 0, out)["evaluation"]
                file.write_bytes(file.read_bytes().replace(b"\n", b"\r\n"))
                fresh.read_receipt(point, 0, science, out)
                table = pd.read_csv(file)
                table["partition_seed"] = 3500
                fresh.atomic_csv(table, file)
                receipt_file = fresh.paths(point, 0, out)["receipt"]
                receipt = json.loads(receipt_file.read_text())
                receipt["sha256"]["evaluation"] = fresh.text_hash(file)
                fresh.atomic_json(receipt, receipt_file)
                with self.assertRaisesRegex(RuntimeError, "metadata"):
                    fresh.verified_tables(point, 0, science, out)

    def test_reused_receipts_and_changed_config_identity_are_rejected(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            for point in fresh.POINTS:
                file = fresh.paths(point, 0, out)["receipt"]
                receipt = {"identity": fresh.metadata(point, 0, science), "state": "reused"}
                fresh.atomic_json(receipt, file)
                with self.assertRaisesRegex(RuntimeError, "reuse is prohibited"):
                    fresh.read_receipt(point, 0, science, out)
                receipt["state"] = "numerical_failure"
                receipt["identity"]["config_sha256"] = "other-setting"
                fresh.atomic_json(receipt, file)
                with self.assertRaisesRegex(RuntimeError, "identity differs"):
                    fresh.read_receipt(point, 0, science, out)

    def test_model_mismatch_and_rehashed_incorrect_traffic_are_rejected(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "ROUNDS", 2):
            out = Path(folder)
            for kind, column, value, error in (("evaluation", "model_sha256", "a" * 64, "model fingerprints"),
                                               ("summary", "unicast_total_bits", 1, "Reconstructed")):
                fresh.save_result("event", 0, science, fake_result("event"), out)
                file = fresh.paths("event", 0, out)[kind]
                frame = pd.read_csv(file, dtype={"model_sha256": str})
                frame[column] = value
                fresh.atomic_csv(frame, file)
                receipt_file = fresh.paths("event", 0, out)["receipt"]
                receipt = json.loads(receipt_file.read_text())
                receipt["sha256"][kind] = fresh.text_hash(file)
                fresh.atomic_json(receipt, receipt_file)
                with self.assertRaisesRegex(RuntimeError, error):
                    fresh.verified_tables("event", 0, science, out)

    def test_workers_use_fresh_seeds_and_skip_verified_complete_runs(self):
        science = fake_science()
        science["protocol"]["dimension"] = 20570
        configs = {p: replace(fresh.config_for(p), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
                   for p in fresh.POINTS}
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "ROUNDS", 2), \
                patch.object(fresh, "build_contract", return_value=science), \
                patch.object(fresh, "config_for", side_effect=configs.__getitem__), \
                patch.object(fresh.full, "make_cifar10_federation", return_value=smoke_federation()) as make, \
                redirect_stdout(io.StringIO()):
            out = Path(folder)
            for point, index in (("event", 0), ("strom", 9)):
                self.assertIn(f"done {point} p{fresh.SEEDS[index]}", fresh.run_point(point, index, science, out))
                self.assertEqual(make.call_args.kwargs["seed"], fresh.SEEDS[index])
                self.assertEqual(make.call_args.kwargs["n_clients"], 10)
                self.assertEqual(make.call_args.kwargs["regime"], "strong")
                saved = fresh.verified_tables(point, index, science, out)
                self.assertTrue(saved["evaluation"].train_seed.eq(fresh.train_seed(index)).all())
                host = fresh.full if point == "event" else fresh.strom
                with patch.object(host, "evaluated_run", side_effect=AssertionError("no retraining")):
                    self.assertIn("complete", fresh.run_point(point, index, science, out))

    def test_numerical_failure_is_visible_and_not_retried(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "build_contract", return_value=science), \
                patch.object(fresh.full, "make_cifar10_federation", return_value=smoke_federation()), \
                patch.object(fresh.strom, "evaluated_run", side_effect=FloatingPointError("synthetic")) as run, \
                redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn("numerical failure", fresh.run_point("strom", 9, science, out))
            self.assertIn("numerical_failure", fresh.run_point("strom", 9, science, out))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(fresh.read_receipt("strom", 9, science, out)["error"], "synthetic")


class StatisticsTests(unittest.TestCase):
    def test_unfavourable_pairs_have_correct_directions_and_keep_all_ten(self):
        pairs, summary = fresh.paired_endpoints(paired_runs().sample(frac=1, random_state=3))
        self.assertEqual((len(pairs), len(summary)), (80, 8))
        row = summary.set_index("metric").loc["test_accuracy_difference_pp"]
        self.assertAlmostEqual(row.estimate, -1)
        self.assertEqual(row.n, 10)
        self.assertTrue(row.primary_predictive_endpoint)
        ratio = summary.set_index("metric").loc["unicast_traffic_ratio"]
        self.assertAlmostEqual(ratio.estimate, 2)
        modified = paired_runs()
        modified.loc[(modified.point == "event") & (modified.partition_seed == fresh.SEEDS[0]), "final_test_accuracy"] += .5
        _, other = fresh.paired_endpoints(modified)
        self.assertAlmostEqual(other.set_index("metric").loc["test_accuracy_difference_pp", "estimate"], 4)
        self.assertTrue(other.n.eq(10).all())

    def test_missing_or_duplicate_pair_is_rejected(self):
        frame = paired_runs()
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                fresh.paired_endpoints(bad)

    def test_target_censoring_reports_all_denominators_and_true_seed_labels(self):
        rows = []
        for i, seed in enumerate(fresh.SEEDS):
            for point in fresh.POINTS:
                for target in fresh.TARGETS:
                    reached = not (point == "strom" and i == 9 and target == .55)
                    rows.append({"point": point, "partition_seed": seed, "target_accuracy": target, "reached": reached,
                                 "first_sampled_round": (120 if point == "event" else 180) if reached else np.nan,
                                 "unicast_total_bits": (1000 if point == "event" else 500) if reached else np.nan})
        frame = pd.DataFrame(rows)
        row = fresh.paired_targets(frame).set_index("target_accuracy").loc[.55]
        self.assertEqual((row.n_total, row.n_event_reached, row.n_strom_reached, row.n_common_hits), (10, 10, 9, 9))
        self.assertEqual((row.event_fewer_rounds, row.same_sampled_round, row.event_fewer_bits), (9, 0, 0))
        self.assertAlmostEqual(row.estimate, 2)
        missing = (frame.point == "strom") & (frame.target_accuracy == .55)
        frame.loc[missing, "reached"] = False
        frame.loc[missing, ["unicast_total_bits", "first_sampled_round"]] = np.nan
        row = fresh.paired_targets(frame).set_index("target_accuracy").loc[.55]
        self.assertEqual((row.n_total, row.n_strom_reached, row.n_common_hits), (10, 0, 0))
        self.assertTrue(np.isnan(row.estimate))
        history = fresh.optimized(fresh.trajectory_tables("strom", fake_result("strom"), fake_science())["history"])
        targets = fresh.target_outcomes(history.assign(point="strom", partition_seed=6900))
        self.assertTrue(targets.partition_seed.eq(6900).all())
        self.assertEqual(targets.first_sampled_round.tolist(), [1, 2, 2])


class PipelineTests(unittest.TestCase):
    def test_complete_grid_analysis_and_no_silent_missing_or_failed_seed(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(fresh, "ROUNDS", 2), \
                patch.object(fresh, "build_science", return_value=science), patch.object(fresh, "plot"), \
                redirect_stdout(io.StringIO()):
            out = Path(folder)
            fresh.atomic_json(science, out / "protocol.json")
            for i in range(10):
                for point in fresh.POINTS:
                    fresh.save_result(point, i, science, fake_result(point, i), out)
            fresh.analyze(out)
            for file, count in (("comparison.csv", 80), ("comparison_history.csv", 160), ("main_runs.csv", 20),
                                ("paired_per_seed.csv", 80), ("paired_summary.csv", 8),
                                ("traffic_to_target.csv", 60), ("paired_target_summary.csv", 3)):
                self.assertEqual(len(pd.read_csv(out / file)), count)
            audit = json.loads((out / "analysis_audit.json").read_text())
            self.assertEqual((audit["fresh_seed_pairs"], audit["new_training_runs"], audit["reused_training_results"]), (10, 20, 0))
            file = fresh.paths("strom", 9, out)["receipt"]
            receipt = json.loads(file.read_text())
            receipt.update(state="numerical_failure", error="synthetic")
            fresh.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "no seed is dropped"):
                fresh.analyze(out)
            completion = json.loads((out / "completion_audit.json").read_text())
            self.assertEqual(len(completion["results"]), 20)
            self.assertEqual(sum(r["state"] == "numerical_failure" for r in completion["results"]), 1)
            file.unlink()
            with self.assertRaisesRegex(RuntimeError, "no seed is dropped"):
                fresh.analyze(out)
            completion = json.loads((out / "completion_audit.json").read_text())
            self.assertEqual(sum(r["state"] == "pending" for r in completion["results"]), 1)


if __name__ == "__main__":
    unittest.main()
