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

from experiments import event_codec_development as dev
from experiments.stc_stage1 import smoke_federation

REAL_CONFIG = dev.config_for_id


def short_config(point_id):
    return replace(REAL_CONFIG(point_id), rounds=4, eval_stride=1, local_steps=1, batch_size=2)


def synthetic_data():
    fed = smoke_federation()
    return dev.data_tools.DevelopmentData(replace(fed, X_test=np.empty((0, 3, 32, 32), dtype=np.uint8), y_test=np.empty(0, dtype=int)),
                                          fed.X_test, fed.y_test, pd.DataFrame(), {"synthetic": True})


def fake_science():
    return {"protocol": {"n_clients": 2, "dimension": 4}, "reference_contract_sha256": dev.digest({}),
            "references": {dev.reference_key(key, horizon): {
                "source_point": asdict(dev.previous.EVENT if key == dev.CURRENT.point_id else dev.STROM_POINT),
                "source_horizon": horizon, "files": {}}
                for key in (dev.CURRENT.point_id, dev.STROM_ID) for horizon in (dev.SCREEN, dev.FINAL)}}


def choice_frame():
    return pd.DataFrame([
        {"point_id": dev.STROM_ID, "method": "strom", "state": "reused", "unicast_total_bits": 100., "final_validation_ce": 1.},
        {"point_id": "a", "method": "event", "state": "complete", "threshold": .025, "jump0": .005,
         "unicast_total_bits": 150., "final_validation_ce": .8},
        {"point_id": "b", "method": "event", "state": "complete", "threshold": .05, "jump0": .005,
         "unicast_total_bits": 90., "final_validation_ce": .9},
        {"point_id": "failed", "method": "event", "state": "numerical_failure", "threshold": .1,
         "jump0": .01, "unicast_total_bits": 1., "final_validation_ce": .01}])


def fake_result(key, horizon):
    quality, budget = dev.event_point(.05, .005).point_id, dev.event_point(.1, .005).point_id
    upload = 100 if key == dev.STROM_ID else 140 if key == quality else 60 if key == budget else 120 if key == dev.CURRENT.point_id else 80
    method = dev.method_for(key)
    records = pd.DataFrame([{"round": rnd, "method": method, "legacy_uplink_bits": 2*upload,
                            "codec_uplink_bits": upload, "legacy_server_bits": 2*upload,
                            "codec_replay_bits": upload, "exact_downlink_bits": upload,
                            "model_sha256": f"{rnd:064x}"} for rnd in range(1, horizon+1)])
    screen_ce = .8 if key == quality else .9 if key == budget else 1.5
    final_ce = .85 if key == quality else .6 if key == budget else .1 if key == dev.CURRENT.point_id else 1.4
    evaluation = pd.DataFrame({"round": range(1, horizon+1), "train_objective": 1.7, "train_ce": 1.6,
                               "validation_ce": [screen_ce if rnd <= dev.SCREEN else final_ce for rnd in range(1,horizon+1)],
                               "validation_accuracy": .5, "validation_macro_accuracy": .5,
                               "validation_worst_class_accuracy": .2, "model_sha256": records.model_sha256})
    return {"rounds": records, "evaluation": evaluation}


class ProtocolTests(unittest.TestCase):
    def test_fifteen_candidates_vary_only_threshold_and_quantum_and_include_current(self):
        self.assertEqual(len(dev.POINTS), 15)
        self.assertEqual(len(dev.POINT_BY_ID), 15)
        self.assertIn(dev.CURRENT, dev.POINTS)
        self.assertEqual(dev.THRESHOLDS, (.0125, .025, .05, .1, .2))
        self.assertEqual(dev.QUANTA, (.0025, .005, .01))
        base = asdict(dev.previous.config_for(dev.previous.EVENT))
        for point in dev.POINTS:
            config = asdict(dev.config_for(point))
            self.assertTrue({k for k in config if config[k]!=base[k]}.issubset({'threshold','jump0'}))
            self.assertEqual(config['rounds'], 1800)
            self.assertEqual(config['server_gain'], 1)
        spec = dev.protocol_spec()
        self.assertEqual((spec['new_screen_training_runs'],spec['max_promoted_Event_candidates']), (14,2))
        self.assertFalse(spec['automatic_boundary_extension'])
        self.assertFalse(spec['official_test_images_loaded'])
        self.assertFalse(spec['confirmation_included'])

    def test_real_references_reuse_exact_validation_evidence_without_reading_test_curves(self):
        with patch.object(dev.full, 'load_contract', side_effect=AssertionError('no test curves')), \
                patch.object(dev.full, 'event_reference', side_effect=AssertionError('no archived test reference')), \
                patch.object(dev.data_tools, '_read_batch', side_effect=AssertionError('no raw images')):
            science = dev.build_science()
            split = science['split']
            self.assertEqual(len(science['references']), 4)
            self.assertEqual(set(science['training_batch_sha256']), {f'data_batch_{i}' for i in range(1,6)})
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                dev.register_reuse(science, split, out)
                self.assertEqual(len(list(out.glob('*receipt.json'))), 4)
                self.assertEqual(len(list(out.glob('*.csv'))), 0)
                for key in (dev.CURRENT.point_id, dev.STROM_ID):
                    for horizon in (600,1800):
                        table = dev.verified_tables(key, horizon, science, split, out)
                        self.assertTrue(table['evaluation'].evidence_origin.eq('reused_validation_reference').all())
                        self.assertFalse(any('test_' in c for c in table['evaluation']))
                current = dev.verified_tables(dev.CURRENT.point_id, 600, science, split, out)
                strom = dev.verified_tables(dev.STROM_ID, 600, science, split, out)
                self.assertEqual(current['summary'].query("variant == 'both_codecs'").iloc[0].unicast_total_bits, 234420328)
                self.assertEqual(strom['summary'].query("variant == 'both_codecs'").iloc[0].unicast_total_bits, 158684888)

    def test_wrong_runtime_fails_before_downloading_or_loading_training_images(self):
        with patch.object(dev, 'build_science', return_value={'runtime': {'wrong': True}}), \
                patch.object(dev.full, 'download_cifar10') as download, patch.object(dev.data_tools, 'load_data') as load:
            with self.assertRaisesRegex(RuntimeError, 'recorded Windows'):
                dev.build_contract()
            download.assert_not_called()
            load.assert_not_called()

    def test_changed_reused_reference_is_rejected(self):
        science, split = fake_science(), {'synthetic': True}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = out/'original.csv'
            original.write_text('a\n1\n')
            science['references'][dev.reference_key(dev.CURRENT.point_id,dev.SCREEN)]['files'] = {
                'summary': {'path': str(original), 'sha256': dev.text_hash(original)}}
            dev.register_reuse(science, split, out)
            original.write_text('a\n2\n')
            with self.assertRaisesRegex(RuntimeError, 'reference evidence changed'):
                dev.read_receipt(dev.CURRENT.point_id, dev.SCREEN, science, split, out)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_validation_ce_and_same_horizon_strom_budget(self):
        frame = choice_frame()
        frame['final_test_accuracy'] = [1.,0.,1.,1.]  # Deliberate decoy, never used by selection.
        frame['final_train_objective'] = [0.,100.,0.,0.]
        choice = dev.select_points(frame.sample(frac=1, random_state=2))
        self.assertEqual(choice['quality']['point_id'], 'a')
        self.assertEqual(choice['within_strom_budget']['point_id'], 'b')
        self.assertEqual(choice['strom_unicast_budget_bits'], 100)
        self.assertEqual(dev.shortlist(choice), ['a','b'])

    def test_infeasible_budget_is_null_and_no_nearest_fallback_is_promoted(self):
        frame = choice_frame()
        frame.loc[frame.point_id==dev.STROM_ID, 'unicast_total_bits'] = 80
        selected = dev.select_points(frame)
        self.assertIsNone(selected['within_strom_budget'])
        self.assertEqual(dev.shortlist(selected), ['a'])

    def test_reused_current_setting_can_win_and_exact_ties_use_cost_then_id(self):
        frame = choice_frame()
        frame.loc[frame.point_id=='a', ['point_id','state']] = [dev.CURRENT.point_id, 'reused']
        self.assertEqual(dev.select_points(frame)['quality']['point_id'], dev.CURRENT.point_id)
        extra = frame[frame.point_id==dev.CURRENT.point_id].copy()
        extra['point_id'], extra['unicast_total_bits'] = 'aa', 140
        self.assertEqual(dev.select_points(pd.concat([frame,extra]))['quality']['point_id'], 'aa')
        extra2 = extra.copy(); extra2['point_id'] = 'ab'
        self.assertEqual(dev.select_points(pd.concat([frame,extra2,extra]))['quality']['point_id'], 'aa')
        frame.loc[frame.point_id==dev.STROM_ID, 'state'] = 'numerical_failure'
        with self.assertRaisesRegex(RuntimeError, 'budget reference'):
            dev.select_points(frame)


class ContinuationTests(unittest.TestCase):
    def test_worker_continues_exact_checkpoint_and_reuses_completed_results(self):
        data = synthetic_data()
        science = {'protocol': {'n_clients': 2, 'dimension': 20570}}
        split = data.manifest
        point = dev.event_point(.05,.005).point_id
        with tempfile.TemporaryDirectory() as folder, patch.object(dev, 'SCREEN', 2), patch.object(dev, 'FINAL', 4), \
                patch.object(dev, 'config_for_id', side_effect=short_config), patch.object(dev, 'build_contract', return_value=science), \
                patch.object(dev.data_tools, 'load_data', return_value=data), redirect_stdout(io.StringIO()):
            out = Path(folder)
            kwargs = dict(data=data, method='event', config=short_config(point), seed=dev.data_tools.TRAIN_SEED,
                          provenance=dev.checkpoint_identity(point,science,split))
            uninterrupted = dev.data_tools.evaluated_run(**kwargs, checkpoint_path=out/'full.npz', horizon=4)
            self.assertIn('done', dev.run_point(point,2,science,split,out))
            before = dev.read_receipt(point,2,science,split,out)
            self.assertIn('done', dev.run_point(point,4,science,split,out))
            dev.verified_tables(point,4,science,split,out)
            self.assertEqual(before,dev.read_receipt(point,2,science,split,out))
            with np.load(out/'full.npz',allow_pickle=False) as full, np.load(out/'checkpoints'/(point+'.npz'),allow_pickle=False) as actual:
                for field in ('model','client_model','event_state'):
                    np.testing.assert_array_equal(full[field].view(np.uint32),actual[field].view(np.uint32))
                self.assertEqual(json.loads(str(full['metadata'].item()))['progress']['rng_state'],
                                 json.loads(str(actual['metadata'].item()))['progress']['rng_state'])
            with patch.object(dev.data_tools,'evaluated_run',side_effect=AssertionError('must reuse')):
                self.assertIn('reuse',dev.run_point(point,4,science,split,out))
            with self.assertRaisesRegex(ValueError,'identity/provenance'):
                dev.data_tools.evaluated_run(**{**kwargs,'provenance':{'wrong':True}},checkpoint_path=out/'checkpoints'/(point+'.npz'),horizon=4)

    def test_numerical_failure_is_saved_and_not_retried_with_altered_settings(self):
        data, science = synthetic_data(), {'protocol': {'n_clients': 2,'dimension':20570}}
        key = dev.POINTS[0].point_id
        with tempfile.TemporaryDirectory() as folder, patch.object(dev,'build_contract',return_value=science), \
                patch.object(dev.data_tools,'load_data',return_value=data), \
                patch.object(dev.data_tools,'evaluated_run',side_effect=FloatingPointError('synthetic')) as run, redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertIn('numerical failure',dev.run_point(key,600,science,data.manifest,out))
            self.assertIn('numerical_failure',dev.run_point(key,600,science,data.manifest,out))
            self.assertEqual(run.call_count,1)

    def test_recomputed_traffic_rejects_even_a_rehashed_incorrect_summary(self):
        key = dev.POINTS[0].point_id
        science, split = fake_science(), {'synthetic':True}
        with tempfile.TemporaryDirectory() as folder, patch.object(dev,'SCREEN',2), patch.object(dev,'FINAL',4), \
                patch.object(dev,'config_for_id',side_effect=short_config):
            out = Path(folder)
            dev.save_result(key,2,science,split,fake_result(key,2),out)
            dev.verified_tables(key,2,science,split,out)
            file = dev.paths(key,2,out)['summary']
            frame = pd.read_csv(file); frame['unicast_total_bits'] += 8
            dev.atomic_csv(frame,file)
            receipt_file = dev.paths(key,2,out)['receipt']
            receipt = json.loads(receipt_file.read_text()); receipt['sha256']['summary'] = dev.text_hash(file)
            dev.atomic_json(receipt,receipt_file)
            with self.assertRaisesRegex(RuntimeError,'Reconstructed'):
                dev.verified_tables(key,2,science,split,out)


class PipelineTests(unittest.TestCase):
    def test_screen_and_final_selection_only_promote_declared_candidates_and_preserve_all_failures(self):
        quality = dev.event_point(.05,.005).point_id
        budget = dev.event_point(.1,.005).point_id
        executed = []
        with tempfile.TemporaryDirectory() as folder, patch.object(dev,'SCREEN',2), patch.object(dev,'FINAL',4), \
                patch.object(dev,'config_for_id',side_effect=short_config), patch.object(dev,'plot'), \
                patch.object(dev.previous,'load_contract',return_value={}), redirect_stdout(io.StringIO()):
            out = Path(folder)
            science = fake_science()
            indices = dev.data_tools.split_indices(np.repeat(np.arange(10,dtype=int),5000))
            split = dev.data_tools.split_manifest(indices)
            science['split'] = split

            def source_tables(source,horizon,*args):
                key = dev.CURRENT.point_id if source.method=='event' else dev.STROM_ID
                return dev.data_tools.trajectory_tables(fake_result(key,horizon),method=source.method,horizon=horizon,n_clients=2,dimension=4)

            def execute(ids,horizon,workers,science,split,out):
                for key in ids:
                    if dev.read_receipt(key,horizon,science,split,out) is not None:
                        continue
                    executed.append((key,horizon))
                    if key==dev.POINTS[-1].point_id:
                        dev.atomic_json({'identity':dev.metadata(key,horizon,science,split),'state':'numerical_failure','error':'synthetic'},dev.paths(key,horizon,out)['receipt'])
                    else:
                        result = fake_result(key,horizon)
                        if horizon==4:
                            dev.verify_screen_prefix(key,result,science,split,out)
                        dev.save_result(key,horizon,science,split,result,out)

            dev.atomic_json(science,out/'protocol.json')
            dev.atomic_csv(indices,out/'split_indices.csv')
            dev.atomic_json({'contract_sha256':dev.digest(science),'split':split},out/'split_manifest.json')
            with patch.object(dev.previous,'load_split',return_value=split), patch.object(dev.previous,'verified_tables',side_effect=source_tables), \
                    patch.object(dev,'execute',side_effect=execute), patch.object(dev,'build_science',return_value=science):
                dev.register_reuse(science,split,out)
                decision = dev.screen(1,science,split,out)
                self.assertEqual(set(decision['promoted_Event_ids']),{quality,budget})
                frozen = dev.finalize(1,science,split,out)
                self.assertEqual(frozen['selection']['quality']['point_id'],budget)
                self.assertEqual(frozen['selection']['within_strom_budget']['point_id'],budget)
                self.assertEqual(frozen['official_test_evaluations'],0)
                self.assertNotIn(dev.CURRENT.point_id,frozen['final_receipts'])
                self.assertEqual(len([job for job in executed if job[1]==2]),14)
                self.assertEqual(len([job for job in executed if job[1]==4]),2)
                previous = executed.copy()
                dev.screen(1,science,split,out); dev.finalize(1,science,split,out)
                self.assertEqual(executed,previous)
                dev.analyze(out)
                rows = pd.read_csv(out/'screening.csv')
                self.assertEqual(len(rows),16)
                self.assertEqual(rows.state.eq('numerical_failure').sum(),1)
                self.assertFalse(any('test_' in c for c in rows))
                # The unpromoted current control is deliberately better at the final horizon.
                # It cannot be introduced into selection after inspecting that outcome.
                controls = pd.read_csv(out/'controls.csv')
                self.assertEqual(controls.query("horizon==4 and method=='event'").iloc[0].final_validation_ce,.1)
                frozen['selection']['quality']['point_id']=dev.CURRENT.point_id
                dev.atomic_json(frozen,out/'frozen_selection.json')
                with self.assertRaisesRegex(RuntimeError,'final Event selection'):
                    dev.analyze(out)


if __name__=='__main__':
    unittest.main()
