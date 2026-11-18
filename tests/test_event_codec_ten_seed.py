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

from experiments import event_codec_ten_seed as ten
from experiments.stc_stage1 import smoke_federation


def synthetic_science():
    protocol = {**ten.protocol_spec(), "rounds": 2, "dimension": 4, "n_clients": 2,
                "evaluation_rounds": [1, 2]}
    return {"protocol": protocol, "runtime": {"synthetic": True}, "source_hashes": {},
            "seed_zero": {"contract_sha256": "synthetic_original", "points": {
                point: {"files": {}, "source_contract_sha256": "synthetic_original"} for point in ten.POINTS}}}


def synthetic_result(point, index):
    offset = 0 if point == "event" else -.01 if point == "stc_primary" else -.02
    records = pd.DataFrame([{
        "round": rnd, "method": "event" if point == "event" else "stc",
        "legacy_uplink_bits": 160, "codec_uplink_bits": 80,
        "legacy_server_bits": 160, "codec_replay_bits": 80, "exact_downlink_bits": 64,
        "model_sha256": f"{index * 10 + rnd:064x}"} for rnd in (1, 2)])
    evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1.1, 1.],
                               "test_ce": [1.2, 1.1], "test_accuracy": [.46 + offset, .56 + offset],
                               "macro_accuracy": [.46 + offset, .56 + offset], "worst_class_accuracy": [.2, .3],
                               "model_sha256": records.model_sha256})
    return {"rounds": records, "evaluation": evaluation,
            "summary": ten.full.traffic_summary(records, n_clients=2, dimension=4)}


def paired_runs():
    rows = []
    for i, seed in enumerate(ten.SEEDS):
        for point in ten.POINTS:
            multiplier = 1 if point == "event" else 2 if point == "stc_primary" else 3
            rows.append({"point": point, "partition_seed": seed,
                         "final_test_accuracy": .60 + .005 * i - (0 if point == "event" else .01 if point == "stc_primary" else .02),
                         "final_test_ce": 1. + .01 * i + (0 if point == "event" else .02),
                         "final_train_objective": 1.2, "final_train_l2_penalty": .1,
                         "unicast_total_bits": multiplier * (1000 + 100 * i),
                         "broadcast_total_bits": multiplier * (500 + 50 * i),
                         "uplink_bits": multiplier * (100 + 10 * i),
                         "unicast_downlink_bits": multiplier * (900 + 90 * i)})
    return pd.DataFrame(rows)


class ProtocolTests(unittest.TestCase):
    def test_fixed_grid_reuses_first_seed_and_has_27_new_runs(self):
        self.assertEqual(ten.SEEDS, tuple(range(3500, 4500, 100)))
        self.assertEqual([ten.train_seed(i) for i in range(10)], list(range(83500, 84500, 100)))
        spec = ten.protocol_spec()
        self.assertEqual((spec["expected_point_seed_results"], spec["reused_results"], spec["new_training_runs"]), (30, 3, 27))
        self.assertEqual(spec["sensitivity_excludes_seed"], 3500)
        self.assertEqual(spec["accuracy_targets"], [.45, .50, .55])
        self.assertFalse(spec["equivalence_or_noninferiority_claim"])
        for index in (-1, 10):
            with self.assertRaises(ValueError):
                ten.train_seed(index)

    def test_real_seed_zero_reuse_verifies_original_files_without_training(self):
        science = ten.build_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten.full, "evaluated_run", side_effect=AssertionError("must not retrain")):
            out = Path(folder)
            ten.register_reuse(science, out)
            self.assertEqual(len(list(out.glob('*.csv'))), 0)
            self.assertEqual(len(list(out.glob('*_receipt.json'))), 3)
            for point in ten.POINTS:
                self.assertEqual(ten.read_receipt(point, 0, science, out)["state"], "reused")
                tables = ten.verified_tables(point, 0, science, out)
                self.assertTrue(tables["summary"].evidence_origin.eq("reused_full_horizon").all())
                self.assertTrue(tables["summary"].partition_seed.eq(3500).all())

    def test_reused_source_tampering_is_rejected(self):
        science = synthetic_science()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / "original.csv"
            source.write_text('a\n1\n', encoding='utf-8')
            science["seed_zero"]["points"]["event"]["files"] = {
                "summary": {"path": str(source), "sha256": ten.text_hash(source)}}
            ten.register_reuse(science, out)
            source.write_text('a\n2\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, "Reused"):
                ten.read_receipt("event", 0, science, out)

    def test_default_queue_never_schedules_seed_zero(self):
        science = synthetic_science()
        jobs = []
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, "OUT", Path(folder)), \
                patch.object(ten, "build_contract", return_value=science), patch.object(ten, "register_reuse"), \
                patch.object(ten, "read_receipt", return_value=None), \
                patch.object(ten, "run_point", side_effect=lambda p, i, s: jobs.append((p, i)) or 'synthetic'), redirect_stdout(io.StringIO()):
            ten.run(1)
        self.assertEqual(len(jobs), 27)
        self.assertEqual({i for _, i in jobs}, set(range(1, 10)))
        self.assertEqual(len(set(jobs)), 27)
        with self.assertRaisesRegex(ValueError, "reused"):
            ten.run_point("event", 0, science)


class StatisticsTests(unittest.TestCase):
    def test_pairs_match_by_seed_and_use_event_minus_or_over_reference(self):
        runs = paired_runs().sample(frac=1, random_state=32).reset_index(drop=True)
        pairs, summary = ten.paired_endpoints(runs)
        self.assertEqual(len(pairs), 160)
        self.assertEqual(len(summary), 32)
        a = summary[(summary.scope == "all_10") & (summary.reference == "stc_primary") & (summary.metric == "test_accuracy_difference_pp")].iloc[0]
        self.assertEqual(a.n, 10)
        self.assertAlmostEqual(a.estimate, 1.)
        self.assertAlmostEqual(a.ci95_low, 1.)
        self.assertAlmostEqual(a.ci95_high, 1.)
        ratio = summary[(summary.scope == "all_10") & (summary.reference == "stc_primary") & (summary.metric == "unicast_traffic_ratio")].iloc[0]
        self.assertAlmostEqual(ratio.estimate, .5)
        self.assertAlmostEqual(ratio.ci95_low, .5)

    def test_additional_nine_sensitivity_excludes_observed_pilot(self):
        runs = paired_runs()
        runs.loc[runs.point == "event", "final_test_accuracy"] = .6
        runs.loc[runs.point == "stc_primary", "final_test_accuracy"] = .6
        runs.loc[(runs.point == "event") & (runs.partition_seed == 3500), "final_test_accuracy"] = .7
        _, result = ten.paired_endpoints(runs)
        selected = result[(result.reference == "stc_primary") & (result.metric == "test_accuracy_difference_pp")].set_index("scope")
        self.assertAlmostEqual(selected.loc['all_10', 'estimate'], 1.)
        self.assertEqual(selected.loc['additional_9', 'n'], 9)
        self.assertAlmostEqual(selected.loc['additional_9', 'estimate'], 0.)

    def test_unfavourable_event_results_are_retained(self):
        runs = paired_runs()
        runs.loc[runs.point == "event", "final_test_accuracy"] = .4
        _, result = ten.paired_endpoints(runs)
        row = result[(result.scope == "all_10") & (result.reference == "stc_primary") & (result.metric == "test_accuracy_difference_pp")].iloc[0]
        self.assertLess(row.estimate, 0)

    def test_missing_or_duplicate_seed_cannot_form_pairs(self):
        runs = paired_runs()
        with self.assertRaises(ValueError):
            ten.paired_endpoints(runs.drop(runs[(runs.point == 'stc_primary') & (runs.partition_seed == 3500)].index))
        with self.assertRaises(ValueError):
            ten.paired_endpoints(pd.concat([runs, runs.iloc[[0]]], ignore_index=True))

    def test_target_ratios_preserve_all_seed_hit_counts(self):
        rows = []
        for i, seed in enumerate(ten.SEEDS):
            for point in ten.POINTS:
                reached = i != 9 if point == "event" else i < 8 if point == "stc_primary" else i not in (0, 9)
                for target in ten.TARGETS:
                    rows.append({"point": point, "partition_seed": seed, "target_accuracy": target,
                                 "reached": reached, "unicast_total_bits": (100 if point == "event" else 200) if reached else float('nan')})
        result = ten.paired_target_summary(pd.DataFrame(rows))
        selected = result[(result.scope == 'all_10') & (result.reference == 'stc_primary') & (result.target_accuracy == .55)].iloc[0]
        self.assertEqual((selected.n_total, selected.n_event_reached, selected.n_reference_reached, selected.n_common_hits, selected.n), (10, 9, 8, 8, 8))
        self.assertAlmostEqual(selected.estimate, .5)
        nine = result[(result.scope == 'additional_9') & (result.reference == 'stc_primary') & (result.target_accuracy == .55)].iloc[0]
        self.assertEqual((nine.n_total, nine.n_common_hits), (9, 7))

    def test_target_rows_use_their_actual_seed_not_3500(self):
        history = pd.DataFrame({"point": "event", "variant": "both_codecs", "partition_seed": 3900,
                                "round": [1, 60, 1800], "test_accuracy": [.44, .51, .54],
                                "uplink_bits": [10, 20, 30], "unicast_downlink_bits": [90, 180, 270],
                                "unicast_total_bits": [100, 200, 300], "broadcast_total_bits": [60, 120, 180]})
        result = ten.target_outcomes(history)
        self.assertTrue(result.partition_seed.eq(3900).all())
        self.assertEqual(result.reached.tolist(), [True, True, False])
        self.assertEqual(result.iloc[0].first_sampled_round, 60)
        self.assertTrue(np.isnan(result.iloc[-1].unicast_total_bits))


class EvidenceTests(unittest.TestCase):
    def test_new_seed_write_verification_and_seed_metadata_rejection(self):
        science = synthetic_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'ROUNDS', 2):
            out = Path(folder)
            ten.save_result('event', 1, science, synthetic_result('event', 1), out)
            ten.verified_tables('event', 1, science, out)
            file = ten.paths('event', 1, out)['evaluation']
            file.write_bytes(file.read_bytes().replace(b'\n', b'\r\n'))
            ten.read_receipt('event', 1, science, out)
            table = pd.read_csv(file)
            table['partition_seed'] = 3700
            ten.atomic_csv(table, file)
            receipt_file = ten.paths('event', 1, out)['receipt']
            receipt = json.loads(receipt_file.read_text())
            receipt['sha256']['evaluation'] = ten.text_hash(file)
            ten.atomic_json(receipt, receipt_file)
            with self.assertRaisesRegex(RuntimeError, 'metadata'):
                ten.verified_tables('event', 1, science, out)

    def test_complete_synthetic_grid_exercises_full_analysis(self):
        science = synthetic_science()

        def reused(point, original):
            result = synthetic_result(point, 0)
            summary, history = ten.full.trajectory_tables(result, n_clients=2, dimension=4)
            return {'rounds': result['rounds'], 'evaluation': result['evaluation'], 'summary': summary, 'history': history}

        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'ROUNDS', 2), \
                patch.object(ten, 'build_science', return_value=science), patch.object(ten.full, 'verified_point', side_effect=reused), \
                patch.object(ten, 'regression_check', return_value={'status': 'synthetic'}), patch.object(ten, 'plot'), redirect_stdout(io.StringIO()):
            out = Path(folder)
            ten.atomic_json(science, out/'protocol.json')
            ten.register_reuse(science, out)
            for i in range(1, 10):
                for point in ten.POINTS:
                    ten.save_result(point, i, science, synthetic_result(point, i), out)
            ten.analyze(out)
            self.assertEqual(len(pd.read_csv(out/'comparison.csv')), 80)
            self.assertEqual(len(pd.read_csv(out/'comparison_history.csv')), 160)
            self.assertEqual(len(pd.read_csv(out/'main_runs.csv')), 30)
            self.assertEqual(len(pd.read_csv(out/'paired_per_seed.csv')), 160)
            self.assertEqual(len(pd.read_csv(out/'paired_summary.csv')), 32)
            self.assertEqual(len(pd.read_csv(out/'traffic_to_target.csv')), 240)
            self.assertEqual(len(pd.read_csv(out/'paired_target_summary.csv')), 12)
            receipt_file = ten.paths('event', 9, out)['receipt']
            receipt = json.loads(receipt_file.read_text())
            receipt.update(state='numerical_failure', error='synthetic failure')
            ten.atomic_json(receipt, receipt_file)
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                ten.analyze(out)
            completion = json.loads((out/'completion_audit.json').read_text())
            self.assertEqual(len(completion['results']), 30)
            self.assertEqual(sum(row['state']=='numerical_failure' for row in completion['results']), 1)

    def test_worker_uses_matched_new_seed_and_reuses_completion(self):
        science = synthetic_science()
        # The actual synthetic federation has 20570 parameters; retain that
        # dimension in its output identity and prefix accounting.
        science['protocol']['dimension'] = 20570
        config = replace(ten.config_for('event'), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
        fed = smoke_federation()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'build_contract', return_value=science), \
                patch.object(ten, 'config_for', return_value=config), \
                patch.object(ten.full, 'make_cifar10_federation', return_value=fed) as make, redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn('done event', ten.run_point('event', 1, science, out))
            self.assertEqual(make.call_args.kwargs['seed'], 3600)
            saved = pd.read_csv(ten.paths('event', 1, out)['evaluation'])
            self.assertTrue(saved.train_seed.eq(83600).all())
            with patch.object(ten.full, 'evaluated_run', side_effect=AssertionError('must reuse')):
                self.assertIn('reuse event', ten.run_point('event', 1, science, out))


if __name__ == '__main__':
    unittest.main()
