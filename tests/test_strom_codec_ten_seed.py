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

from experiments import strom_codec_ten_seed as ten
from experiments.stc_stage1 import smoke_federation
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import run_final_baseline


def fake_science():
    return {"protocol": {**ten.protocol_spec(), "rounds": 2, "evaluation_rounds": [1, 2], "n_clients": 2, "dimension": 4},
            "reference_contract_sha256": "synthetic-reference", "development_selection": {"synthetic": True},
            "references": {ten.reference_key(p, i): {} for i in range(10) for p in ten.previous.POINTS}}


def fake_result(point="strom", index=0):
    records = pd.DataFrame({"round": [1, 2], "method": point if point in ("event", "strom") else "stc",
                            "legacy_uplink_bits": [160, 160], "codec_uplink_bits": [80, 80],
                            "legacy_server_bits": [160, 160], "codec_replay_bits": [80, 80],
                            "exact_downlink_bits": [64, 64], "model_sha256": [f"{index*10+r:064x}" for r in (1, 2)]})
    evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1.1, 1.0],
                               "test_ce": [1.2, 1.1], "test_accuracy": [.46, .56], "macro_accuracy": [.46, .56],
                               "worst_class_accuracy": [.2, .3], "model_sha256": records.model_sha256})
    return {"rounds": records, "evaluation": evaluation}


def fake_reference_tables(point, index, science):
    return ten.trajectory_tables(point, fake_result(point, index), science)


def paired_runs():
    rows = []
    for i, seed in enumerate(ten.SEEDS):
        for point in ten.POINTS:
            multiplier = {"event": 2, "strom": 1, "stc_primary": 3, "stc_sparse": 4}[point]
            rows.append({"point": point, "partition_seed": seed, "final_test_accuracy": .6 + .001*i + (.01 if point == "strom" else 0),
                         "final_test_ce": 1.1, "final_train_objective": 1.2, "final_train_l2_penalty": .1,
                         **{field: multiplier*(1000+100*i) for field in ("uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits")}})
    return pd.DataFrame(rows)


class ProtocolTests(unittest.TestCase):
    def test_frozen_config_refits_all_ten_full_training_seeds(self):
        config = ten.config_for()
        self.assertEqual((config.strom_threshold, config.server_gain, config.local_lr, config.rounds), (.05, .10, .05, 1800))
        self.assertEqual(config.server_gain/config.local_lr, 2)
        self.assertEqual(ten.SEEDS, tuple(range(3500, 4500, 100)))
        self.assertEqual([ten.train_seed(i) for i in range(10)], list(range(83500, 84500, 100)))
        spec = ten.protocol_spec()
        self.assertEqual((spec["new_training_runs"], spec["reused_Event_STC_results"], spec["expected_point_seed_results"]), (10, 30, 40))
        self.assertEqual((spec["training_examples"], spec["test_examples"]), (50000, 10000))
        self.assertFalse(spec["development_checkpoint_reuse"])
        self.assertNotIn(4500, spec["partition_seeds"])
        for index in (-1, 10):
            with self.assertRaises(ValueError):
                ten.train_seed(index)

    def test_real_frozen_selection_and_all_thirty_controls_validate_without_training(self):
        with patch.object(ten.host, "evaluated_run", side_effect=AssertionError("no training")):
            science = ten.build_science()
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                ten.register_reuse(science, out)
                self.assertEqual(len(list(out.glob('*receipt.json'))), 30)
                self.assertEqual(len(list(out.glob('*.csv'))), 0)
                for point, index in (("event", 0), ("stc_primary", 1), ("stc_sparse", 9)):
                    tables = ten.verified_tables(point, index, science, out)
                    self.assertTrue(tables["evaluation"].partition_seed.eq(ten.SEEDS[index]).all())
                    self.assertTrue(tables["evaluation"].evidence_origin.eq('reused_Event_STC_codec_run').all())
                self.assertIsNone(ten.read_receipt('strom', 0, science, out))
        self.assertEqual(science['development_selection']['config'], asdict(ten.config_for()))
        self.assertEqual(len(science['references']), 30)

    def test_changed_selected_configuration_and_wrong_runtime_fail_before_training(self):
        original = ten.development_audit.validate
        science, tables, frozen = original()
        changed = json.loads(json.dumps(frozen))
        changed['selected_configs']['quality']['strom_threshold'] = .1
        with patch.object(ten.development_audit, 'validate', return_value=(science, tables, changed)):
            with self.assertRaisesRegex(RuntimeError, 'Frozen development setting'):
                ten.selection_provenance()
        with patch.object(ten, 'build_science', return_value={'runtime': {'wrong': True}}), patch.object(ten.full, 'build_contract') as build:
            with self.assertRaisesRegex(RuntimeError, 'recorded Windows'):
                ten.build_contract()
            build.assert_not_called()

    def test_default_queue_includes_seed_zero_and_only_strom_is_trained(self):
        science, jobs = fake_science(), []
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'build_contract', return_value=science), \
                patch.object(ten, 'register_reuse') as reused, \
                patch.object(ten, 'run_point', side_effect=lambda i, s, out: jobs.append(i) or 'synthetic'), redirect_stdout(io.StringIO()):
            ten.run(1, out=Path(folder))
            reused.assert_called_once()
        self.assertEqual(jobs, list(range(10)))
        self.assertEqual(len(set(jobs)), 10)

    def test_existing_different_protocol_is_preserved_without_scheduling(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'build_contract', return_value=science), \
                patch.object(ten, 'register_reuse') as reused, patch.object(ten, 'run_point') as worker:
            out = Path(folder)
            original = {'different': 'campaign'}
            ten.atomic_json(original, out/'protocol.json')
            with self.assertRaisesRegex(RuntimeError, 'Existing confirmation contract differs'):
                ten.run(1, out=out)
            self.assertEqual(json.loads((out/'protocol.json').read_text()), original)
            reused.assert_not_called()
            worker.assert_not_called()


class EvidenceTests(unittest.TestCase):
    def test_selected_setting_matches_native_model_residual_rng_and_legacy_accounting(self):
        config = replace(ten.config_for(), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
        fed = smoke_federation()
        with tempfile.TemporaryDirectory() as folder:
            native_path = Path(folder)/'native.npz'
            native = run_final_baseline(federation=fed, architecture='cifar_cnn', method='strom', config=config,
                                        seed=ten.train_seed(0), record_history=True, checkpoint=BaselineCheckpoint(native_path, {}), checkpoint_stride=1)
            actual = ten.host.evaluated_run(federation=fed, architecture='cifar_cnn', config=config, seed=ten.train_seed(0),
                                            checkpoint_path=Path(folder)/'codec.npz', provenance={})
            with np.load(native_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['model'].view(np.uint32), actual['model'].view(np.uint32))
                np.testing.assert_array_equal(saved['strom_state'].view(np.uint32), actual['strom_state'].view(np.uint32))
                self.assertEqual(json.loads(str(saved['metadata'].item()))['progress']['rng_state'], actual['rng_state'])
            for field in ten.METRIC_FIELDS:
                np.testing.assert_array_equal(native['history'][field], actual['evaluation'][field])
            legacy = actual['summary'].set_index('variant').loc['legacy']
            self.assertEqual(legacy.unicast_total_bits, native['unicast_hybrid_total_bits'])

    def test_new_result_write_crlf_reuse_and_relabelled_metadata_rejection(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'ROUNDS', 2):
            out = Path(folder)
            ten.save_result(0, science, fake_result(), out)
            ten.verified_tables('strom', 0, science, out)
            file = ten.paths('strom', 0, out)['evaluation']
            file.write_bytes(file.read_bytes().replace(b'\n', b'\r\n'))
            ten.read_receipt('strom', 0, science, out)
            table = pd.read_csv(file)
            table['partition_seed'] = 4500
            ten.atomic_csv(table, file)
            receipt_file = ten.paths('strom', 0, out)['receipt']
            receipt = json.loads(receipt_file.read_text())
            receipt['sha256']['evaluation'] = ten.text_hash(file)
            ten.atomic_json(receipt, receipt_file)
            with self.assertRaisesRegex(RuntimeError, 'metadata'):
                ten.verified_tables('strom', 0, science, out)

    def test_reused_source_tampering_and_reusing_development_as_strom_are_rejected(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = out/'reference.csv'
            original.write_text('a\n1\n')
            science['references']['event_s0'] = {'summary': {'path': str(original), 'sha256': ten.text_hash(original)}}
            ten.register_reuse(science, out)
            original.write_text('a\n2\n')
            with self.assertRaisesRegex(RuntimeError, 'reference evidence changed'):
                ten.read_receipt('event', 0, science, out)
            ten.atomic_json({'identity': ten.metadata('strom', 0, science), 'state': 'reused'}, ten.paths('strom', 0, out)['receipt'])
            with self.assertRaisesRegex(RuntimeError, 'new full-training run'):
                ten.read_receipt('strom', 0, science, out)

    def test_worker_matches_full_training_loader_seed_and_skips_completed_result(self):
        science = fake_science()
        science['protocol']['dimension'] = 20570
        config = replace(ten.config_for(), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'ROUNDS', 2), \
                patch.object(ten, 'build_contract', return_value=science), patch.object(ten, 'config_for', return_value=config), \
                patch.object(ten.full, 'make_cifar10_federation', return_value=smoke_federation()) as make, redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn('done strom p3500', ten.run_point(0, science, out))
            self.assertEqual(make.call_args.kwargs['seed'], 3500)
            self.assertEqual(make.call_args.kwargs['n_clients'], 10)
            self.assertEqual(make.call_args.kwargs['regime'], 'strong')
            saved = ten.verified_tables('strom', 0, science, out)
            self.assertTrue(saved['evaluation'].train_seed.eq(83500).all())
            with patch.object(ten.host, 'evaluated_run', side_effect=AssertionError('no retraining')):
                self.assertIn('reuse strom p3500: complete', ten.run_point(0, science, out))

    def test_numerical_failure_receipt_is_retained_and_not_retried_automatically(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'build_contract', return_value=science), \
                patch.object(ten.full, 'make_cifar10_federation', return_value=smoke_federation()), \
                patch.object(ten.host, 'evaluated_run', side_effect=FloatingPointError('synthetic')) as run, redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn('numerical failure', ten.run_point(9, science, out))
            self.assertIn('numerical_failure', ten.run_point(9, science, out))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(ten.read_receipt('strom', 9, science, out)['error'], 'synthetic')


class StatisticsTests(unittest.TestCase):
    def test_unfavourable_event_pairs_have_correct_direction_and_seed_scope(self):
        pairs, summary = ten.paired_endpoints(paired_runs().sample(frac=1, random_state=3))
        self.assertEqual(len(pairs), 400)
        self.assertEqual(len(summary), 80)
        row = summary.query("scope == 'all_10' and candidate == 'event' and reference == 'strom' and metric == 'test_accuracy_difference_pp'").iloc[0]
        self.assertAlmostEqual(row.estimate, -1)
        self.assertTrue(row.primary_endpoint)
        ratio = summary.query("scope == 'all_10' and candidate == 'event' and reference == 'strom' and metric == 'unicast_traffic_ratio'").iloc[0]
        self.assertAlmostEqual(ratio.estimate, 2)
        self.assertTrue(summary.query("scope == 'additional_9'").n.eq(9).all())
        modified = paired_runs()
        modified.loc[(modified.point=='event') & (modified.partition_seed==3500), 'final_test_accuracy'] += .5
        _, other = ten.paired_endpoints(modified)
        pd.testing.assert_frame_equal(summary.query("scope == 'additional_9'").reset_index(drop=True),
                                      other.query("scope == 'additional_9'").reset_index(drop=True))

    def test_missing_or_duplicate_seed_blocks_paired_endpoints(self):
        frame = paired_runs()
        for bad in (frame.iloc[1:], pd.concat([frame, frame.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                ten.paired_endpoints(bad)

    def test_target_censoring_keeps_denominators_and_actual_seed_labels(self):
        rows = []
        for i, seed in enumerate(ten.SEEDS):
            for point in ten.POINTS:
                for target in ten.TARGETS:
                    reached = not (point=='strom' and i==9 and target==.55)
                    rows.append({'point': point, 'partition_seed': seed, 'target_accuracy': target,
                                 'reached': reached, 'unicast_total_bits': (1000 if point=='event' else 500) if reached else np.nan})
        summary = ten.paired_target_summary(pd.DataFrame(rows))
        row = summary.query("scope == 'all_10' and candidate == 'event' and reference == 'strom' and target_accuracy == .55").iloc[0]
        self.assertEqual((row.n_total, row.n_candidate_reached, row.n_reference_reached, row.n_common_hits), (10, 10, 9, 9))
        self.assertAlmostEqual(row.estimate, 2)
        no_hits = pd.DataFrame(rows)
        missing = (no_hits.point=='strom') & (no_hits.target_accuracy==.55)
        no_hits.loc[missing, 'reached'] = False
        no_hits.loc[missing, 'unicast_total_bits'] = np.nan
        zero = ten.paired_target_summary(no_hits).query("scope == 'all_10' and candidate == 'event' and reference == 'strom' and target_accuracy == .55").iloc[0]
        self.assertEqual((zero.n_total, zero.n_reference_reached, zero.n_common_hits), (10, 0, 0))
        self.assertTrue(np.isnan(zero.estimate))
        result = fake_result()
        science = fake_science()
        history = ten.optimized(ten.trajectory_tables('strom', result, science)['history'].assign(point='strom', partition_seed=4400))
        targets = ten.target_outcomes(history)
        self.assertTrue(targets.partition_seed.eq(4400).all())
        self.assertTrue(targets.reached.all())


class PipelineTests(unittest.TestCase):
    def test_complete_grid_analysis_and_failure_never_silently_drops_seed(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(ten, 'ROUNDS', 2), \
                patch.object(ten, 'build_science', return_value=science), patch.object(ten, 'plot'), \
                patch.object(ten.previous, 'load_contract', return_value={}), \
                redirect_stdout(io.StringIO()):
            out = Path(folder)
            science['reference_contract_sha256'] = ten.digest({})
            ten.atomic_json(science, out/'protocol.json')
            ten.register_reuse(science, out)
            for i in range(10):
                ten.save_result(i, science, fake_result(index=i), out)

            def previous_tables(point, index, old):
                return fake_reference_tables(point, index, science)

            with patch.object(ten.previous, 'verified_tables', side_effect=previous_tables):
                ten.analyze(out)
                self.assertEqual(len(pd.read_csv(out/'main_runs.csv')), 40)
                self.assertEqual(len(pd.read_csv(out/'paired_per_seed.csv')), 400)
                self.assertEqual(len(pd.read_csv(out/'paired_summary.csv')), 80)
                self.assertEqual(len(pd.read_csv(out/'traffic_to_target.csv')), 120)
                self.assertEqual(len(pd.read_csv(out/'paired_target_summary.csv')), 30)
                file = ten.paths('strom', 9, out)['receipt']
                receipt = json.loads(file.read_text())
                receipt.update(state='numerical_failure', error='synthetic')
                ten.atomic_json(receipt, file)
                with self.assertRaisesRegex(RuntimeError, 'no seed is dropped'):
                    ten.analyze(out)
            completion = json.loads((out/'completion_audit.json').read_text())
            self.assertEqual(len(completion['results']), 40)
            self.assertEqual(sum(r['state']=='numerical_failure' for r in completion['results']), 1)


if __name__ == '__main__':
    unittest.main()
