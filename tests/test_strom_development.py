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

from experiments import strom_codec_development as dev
from experiments.stc_stage1 import smoke_federation

data_tools = dev.data_tools


def synthetic_data():
    fed = smoke_federation()
    return data_tools.DevelopmentData(replace(fed, X_test=np.empty((0, 3, 32, 32), dtype=np.uint8), y_test=np.empty(0, dtype=int)),
                                      fed.X_test, fed.y_test, pd.DataFrame(), {"synthetic": True})


def choice_frame():
    return pd.DataFrame([
        {"point_id": "event_reference", "method": "event", "state": "complete", "unicast_total_bits": 100., "final_validation_ce": 1.},
        {"point_id": "a", "method": "strom", "state": "complete", "threshold": .025, "equivalent_gain": 2.,
         "unicast_total_bits": 150., "final_validation_ce": .8, "final_residual_l2_norm": 1e9},
        {"point_id": "b", "method": "strom", "state": "complete", "threshold": .05, "equivalent_gain": 1.,
         "unicast_total_bits": 90., "final_validation_ce": .9, "final_residual_l2_norm": .1},
        {"point_id": "failed", "method": "strom", "state": "numerical_failure", "threshold": .1,
         "equivalent_gain": 4., "unicast_total_bits": np.nan, "final_validation_ce": np.nan}])


class SplitTests(unittest.TestCase):
    def test_genuine_holdout_balanced_clients_disjoint_indices_and_empty_test_slots(self):
        labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
        images = np.arange(50000)
        data = data_tools.federation_from_training(images, labels)
        fed = data.federation
        train = np.concatenate(fed.client_X)
        self.assertEqual((len(train), len(data.validation_X), len(fed.X_train_eval)), (45000, 5000, 5000))
        self.assertEqual(len(np.unique(train)), 45000)
        self.assertEqual(len(np.intersect1d(train, data.validation_X)), 0)
        self.assertEqual(len(np.intersect1d(fed.X_train_eval, data.validation_X)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, data.validation_X]), np.arange(50000))
        np.testing.assert_array_equal(np.bincount(data.validation_y), np.full(10, 500))
        self.assertEqual(len(fed.X_test), 0)
        for i in range(10):
            self.assertEqual(len(fed.client_y[i]), 4500)
            self.assertEqual(np.bincount(fed.client_y[i], minlength=10)[i], 2475)
        again = data_tools.split_indices(labels)
        pd.testing.assert_frame_equal(again, data.indices)
        self.assertEqual(data_tools.split_manifest(again), data.manifest)

    def test_loader_parses_only_the_five_training_batches(self):
        labels = np.repeat(np.arange(10, dtype=np.int64), 5000)
        names = []

        def read(file):
            names.append(file.name)
            if file.name == 'test_batch':
                raise AssertionError('Official test batch must never be parsed')
            i = int(file.name[-1]) - 1
            return np.arange(i * 10000, (i + 1) * 10000), labels[i * 10000:(i + 1) * 10000]

        with patch.object(data_tools, 'download_cifar10', return_value=Path('synthetic')), patch.object(data_tools, '_read_batch', side_effect=read):
            data = data_tools.load_data('synthetic')
        self.assertEqual(names, [f'data_batch_{i}' for i in range(1, 6)])
        self.assertEqual(len(data.validation_y), 5000)

    def test_modified_assignment_or_diagnostic_overlap_is_rejected(self):
        frame = data_tools.split_indices(np.repeat(np.arange(10, dtype=np.int64), 5000))
        a = frame.copy()
        a.loc[0, 'global_index'] = a.loc[1, 'global_index']
        with self.assertRaises(ValueError):
            data_tools.validate_split(a)
        b = frame.copy()
        validation_index = b[b.split == 'validation'].index[0]
        train_index = b[b.training_diagnostic_order == 0].index[0]
        b.loc[train_index, 'training_diagnostic_order'] = -1
        b.loc[validation_index, 'training_diagnostic_order'] = 0
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data_tools.validate_split(b)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation_ce_and_budget_not_training_test_or_residual(self):
        frame = choice_frame()
        frame['final_train_objective'] = [0, 100, 0, np.nan]
        frame['final_test_accuracy'] = [0, 0, 1, np.nan]  # Deliberate decoy, never an input to the rule.
        selected = dev.select_points(frame.sample(frac=1, random_state=12))
        self.assertEqual(selected['quality']['point_id'], 'a')
        self.assertEqual(selected['within_event_budget']['point_id'], 'b')
        self.assertEqual(dev.shortlist(selected), ['a', 'b'])

    def test_infeasible_budget_is_null_and_nearest_remains_diagnostic(self):
        frame = choice_frame()
        frame.loc[frame.point_id == 'event_reference', 'unicast_total_bits'] = 80
        selected = dev.select_points(frame)
        self.assertIsNone(selected['within_event_budget'])
        self.assertEqual(selected['nearest_traffic_diagnostic']['point_id'], 'b')
        self.assertEqual(dev.shortlist(selected), ['a', 'b'])

    def test_ties_are_deterministic_and_failed_points_are_not_silent_winners(self):
        frame = choice_frame()
        extra = frame[frame.point_id == 'a'].copy()
        extra['point_id'] = 'aa'
        extra['unicast_total_bits'] = 140
        self.assertEqual(dev.select_points(pd.concat([frame, extra]))['quality']['point_id'], 'aa')
        all_failed = frame[frame.point_id.isin(['event_reference', 'failed'])]
        with self.assertRaisesRegex(RuntimeError, 'No finite'):
            dev.select_points(all_failed)

    def test_one_pass_boundary_extension_has_fixed_cap_and_corner_coverage(self):
        low = {'point_id': dev.strom_point(.0025, 1).point_id, 'threshold': .0025, 'equivalent_gain': 1.}
        high = {'point_id': dev.strom_point(.1, 4).point_id, 'threshold': .1, 'equivalent_gain': 4.}
        selected = {'quality': low, 'within_event_budget': high, 'nearest_traffic_diagnostic': high}
        points = dev.boundary_points(selected)
        self.assertEqual(len(points), 6)
        self.assertIn(dev.strom_point(.00125, .5), points)
        self.assertIn(dev.strom_point(.2, 8), points)
        self.assertTrue(set(p.point_id for p in points).isdisjoint(p.point_id for p in dev.INITIAL))
        self.assertEqual(len(dev.INITIAL), 15)

    def test_frozen_decision_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'selection.json'
            dev.immutable_json({'point': 'a'}, file)
            dev.immutable_json({'point': 'a'}, file)
            with self.assertRaisesRegex(RuntimeError, 'frozen decision'):
                dev.immutable_json({'point': 'b'}, file)

    def test_preflight_has_no_dependency_on_old_test_curves_or_test_images(self):
        with patch.object(dev.full, 'event_reference', side_effect=AssertionError('no archived test curves')), \
                patch.object(dev.full, 'load_contract', side_effect=AssertionError('no old evaluation data')):
            science = dev.build_science()
        self.assertEqual(set(science['training_batch_sha256']), {f'data_batch_{i}' for i in range(1, 6)})
        self.assertFalse(science['protocol']['official_test_images_loaded'])
        with patch.object(dev.full.development, 'runtime_manifest', return_value={'wrong': True}), patch.object(dev.full, 'download_cifar10') as download:
            with self.assertRaisesRegex(RuntimeError, 'recorded Windows'):
                dev.build_contract()
            download.assert_not_called()


class ValidationCheckpointTests(unittest.TestCase):
    def test_only_validation_arrays_are_evaluated_and_failed_save_is_atomic(self):
        data = synthetic_data()
        config = replace(dev.config_for(dev.INITIAL[0]), rounds=2, eval_stride=1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = data_tools.ValidationCheckpoint(Path(folder) / 'model.npz', {}, data=data, architecture='cifar_cnn', config=config)
            seen = []

            def metric(weights, X, y, **kwargs):
                seen.append((X, y))
                return 1., .8, .6, .5, .4, np.zeros(10)

            checkpoint.metrics = metric
            records = []
            for rnd in (1, 2):
                model = np.full(2, rnd, dtype=np.float32)
                sha = data_tools.hashlib.sha256(model.astype('<f4').tobytes()).hexdigest()
                records.append({'round': rnd, 'model_sha256': sha})
                args = ({}, {'model': model}, {'completed_round': rnd, 'records': records.copy()})
                if rnd == 1:
                    checkpoint.save(*args)
                else:
                    with patch.object(checkpoint.base.__class__, 'save', side_effect=OSError('synthetic failure')):
                        with self.assertRaises(OSError):
                            checkpoint.save(*args)
            saved = checkpoint.load({})
            self.assertEqual(len(checkpoint.rows), 1)
            self.assertEqual(saved['progress']['validation_history'][0]['validation_ce'], .8)
            for X, y in seen:
                self.assertIs(X, data.validation_X)
                self.assertIs(y, data.validation_y)
                self.assertIsNot(X, data.federation.X_test)

    def test_exact_screen_to_final_promotion_and_receipt_reuse_for_both_hosts(self):
        data = synthetic_data()
        science = {'protocol': {'n_clients': 2, 'dimension': 20570}}
        split = {'synthetic': True}
        with tempfile.TemporaryDirectory() as folder, patch.object(dev, 'SCREEN', 2), patch.object(dev, 'FINAL', 4):
            out = Path(folder)
            original_config = dev.config_for

            def config(point):
                return replace(original_config(point), eval_stride=1, local_steps=1, batch_size=2)

            with patch.object(dev, 'config_for', side_effect=config):
                for point in (dev.EVENT, dev.INITIAL[1]):
                    kwargs = dict(data=data, method=point.method, config=config(point), seed=84500,
                                  provenance=dev.checkpoint_identity(point, science, split))
                    uninterrupted = data_tools.evaluated_run(**kwargs, checkpoint_path=out/(point.point_id+'_full.npz'), horizon=4)
                    path = out/(point.point_id+'_resume.npz')
                    early = data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=2)
                    dev.save_result(point, 2, science, split, early, out)
                    late = data_tools.evaluated_run(**kwargs, checkpoint_path=path, horizon=4)
                    dev.verify_screen_prefix(point, late, science, split, out)
                    dev.save_result(point, 4, science, split, late, out)
                    tables = dev.verified_tables(point, 4, science, split, out)
                    self.assertFalse(any('test_' in key for key in tables['evaluation']))
                    np.testing.assert_array_equal(uninterrupted['model'].view(np.uint32), late['model'].view(np.uint32))
                    pd.testing.assert_frame_equal(uninterrupted['evaluation'], late['evaluation'])
                    if point.method == 'strom':
                        np.testing.assert_array_equal(uninterrupted['strom_state'].view(np.uint32), late['strom_state'].view(np.uint32))
                    with self.assertRaisesRegex(ValueError, 'identity/provenance'):
                        data_tools.evaluated_run(**{**kwargs, 'provenance': {'different': True}}, checkpoint_path=path, horizon=4)


class PipelineTests(unittest.TestCase):
    def test_worker_promotes_the_same_checkpoint_and_reuses_completed_receipts(self):
        data = synthetic_data()
        split = data.manifest
        science = {'protocol': {'n_clients': 2, 'dimension': 20570}}
        original_config = dev.config_for
        point = dev.INITIAL[1]

        def config(p):
            return replace(original_config(p), rounds=4, eval_stride=1, local_steps=1, batch_size=2)

        with tempfile.TemporaryDirectory() as folder, patch.object(dev, 'SCREEN', 2), patch.object(dev, 'FINAL', 4), \
                patch.object(dev, 'config_for', side_effect=config), patch.object(dev, 'build_contract', return_value=science), \
                patch.object(data_tools, 'load_data', return_value=data), redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn('done', dev.run_point(point, 2, science, split, out))
            self.assertIn('done', dev.run_point(point, 4, science, split, out))
            self.assertEqual(len(dev.verified_tables(point, 4, science, split, out)['evaluation']), 4)
            with patch.object(data_tools, 'evaluated_run', side_effect=AssertionError('must reuse')):
                self.assertIn('reuse', dev.run_point(point, 4, science, split, out))

    def test_full_bounded_pipeline_freezes_validation_choices_without_test_metrics(self):
        quality = dev.strom_point(.025, 2)
        budget = dev.strom_point(.05, 1)
        original_config = dev.config_for
        executed = []

        def config(point):
            return replace(original_config(point), rounds=4, eval_stride=1, local_steps=1, batch_size=2)

        def result(point, horizon):
            # Toy byte counters deliberately separate quality and budget roles.
            upload = 100 if point.method == 'event' else 300 if point == quality else 60
            records = pd.DataFrame([{'round': rnd, 'method': point.method, 'legacy_uplink_bits': upload * 2,
                                      'codec_uplink_bits': upload, 'legacy_server_bits': upload * 2,
                                      'codec_replay_bits': upload, 'exact_downlink_bits': upload,
                                      'model_sha256': 'a' * 62 + f'{rnd:02x}'} for rnd in range(1, horizon + 1)])
            ce = 1.0 if point == quality else 1.05 if point == budget else 1.5
            if horizon == 4 and point == budget:
                ce = .9
            evaluation = pd.DataFrame({'round': range(1, horizon + 1), 'train_objective': 1.7, 'train_ce': 1.6,
                                       'validation_ce': ce, 'validation_accuracy': .5, 'validation_macro_accuracy': .5,
                                       'validation_worst_class_accuracy': .2, 'model_sha256': records.model_sha256})
            return {'rounds': records, 'evaluation': evaluation}

        def execute(points, horizon, workers, science, split, out):
            for point in points:
                if dev.read_receipt(point, horizon, science, split, out) is None:
                    executed.append((point.point_id, horizon))
                    if point == dev.INITIAL[-1]:
                        dev.atomic_json({'identity': dev.metadata(point, horizon, science, split), 'state': 'numerical_failure', 'error': 'synthetic'},
                                        dev.paths(point, horizon, out)['receipt'])
                    else:
                        dev.save_result(point, horizon, science, split, result(point, horizon), out)

        with tempfile.TemporaryDirectory() as folder, patch.object(dev, 'SCREEN', 2), patch.object(dev, 'FINAL', 4), \
                patch.object(dev, 'config_for', side_effect=config), patch.object(dev, 'execute', side_effect=execute), \
                patch.object(dev, 'plot'), redirect_stdout(io.StringIO()):
            out = Path(folder)
            science = {'protocol': {'n_clients': 2, 'dimension': 4}}
            indices = data_tools.split_indices(np.repeat(np.arange(10, dtype=np.int64), 5000))
            split = data_tools.split_manifest(indices)
            dev.atomic_json(science, out/'protocol.json')
            dev.atomic_csv(indices, out/'split_indices.csv')
            dev.atomic_json({'contract_sha256': dev.digest(science), 'split': split}, out/'split_manifest.json')
            points, decision = dev.screen(1, science, split, out)
            self.assertLessEqual(len(points), 22)
            self.assertEqual(set(decision['promoted_strom_ids']), {quality.point_id, budget.point_id})
            frozen = dev.finalize(1, science, split, out)
            self.assertEqual(frozen['selection']['quality']['point_id'], budget.point_id)
            self.assertEqual(frozen['official_test_evaluations'], 0)
            self.assertEqual(len(frozen['final_receipts']), 3)
            previous = executed.copy()
            dev.screen(1, science, split, out)
            dev.finalize(1, science, split, out)
            self.assertEqual(executed, previous)
            with patch.object(dev, 'build_science', return_value=science):
                dev.analyze(out)
            initial = pd.read_csv(out/'screening_initial.csv')
            self.assertEqual(len(initial), 16)
            self.assertEqual(initial.state.eq('numerical_failure').sum(), 1)
            self.assertFalse(any('test_' in key for key in initial.columns))
            manifest = json.loads((out/'frozen_selection.json').read_text())
            manifest['selection']['quality']['point_id'] = quality.point_id
            dev.atomic_json(manifest, out/'frozen_selection.json')
            with patch.object(dev, 'build_science', return_value=science):
                with self.assertRaisesRegex(RuntimeError, 'selection'):
                    dev.analyze(out)


if __name__ == '__main__':
    unittest.main()
