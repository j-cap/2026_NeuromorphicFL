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

from experiments import strom_codec_pilot as pilot
from experiments.stc_stage1 import smoke_federation
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import run_final_baseline

host = pilot.host


def short_config(point=pilot.POINTS[0], rounds=3):
    return replace(pilot.config_for(point), rounds=rounds, eval_stride=1, local_steps=1, batch_size=2)


def fake_science():
    return {"protocol": {**pilot.protocol_spec(), "rounds": 2, "evaluation_rounds": [1, 2],
                         "dimension": 4, "n_clients": 2}, "source_hashes": {}, "runtime": {"synthetic": True}}


def fake_result(method="strom"):
    rounds = pd.DataFrame([{"round": r, "method": method, "legacy_uplink_bits": 160,
                            "codec_uplink_bits": 80, "legacy_server_bits": 160, "codec_replay_bits": 80,
                            "exact_downlink_bits": 64, "model_sha256": "a" * 63 + str(r)} for r in (1, 2)])
    evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1.1, 1.],
                               "test_ce": [1.2, 1.1], "test_accuracy": [.46, .51], "macro_accuracy": [.46, .51],
                               "worst_class_accuracy": [.2, .3], "model_sha256": rounds.model_sha256})
    accounting = host.traffic_summary if method == "strom" else pilot.full.traffic_summary
    return {"rounds": rounds, "evaluation": evaluation, "summary": accounting(rounds, n_clients=2, dimension=4)}


class ResidualTests(unittest.TestCase):
    def test_two_round_reference_retains_overshoot_and_emits_at_most_one_pulse(self):
        state = np.zeros((1, 3), dtype=np.float32)
        first = np.array([.35, -.25, .04], dtype=np.float32)
        pulses = host.residual_pulses(state, [first], [1.], local_lr=1., threshold=.1)
        np.testing.assert_array_equal(pulses, [[1, -1, 0]])
        np.testing.assert_allclose(state, [[.25, -.15, .04]], atol=2e-8)
        second = np.array([0., 0., .07], dtype=np.float32)
        pulses2 = host.residual_pulses(state, [second], [1.], local_lr=1., threshold=.1)
        np.testing.assert_array_equal(pulses2, [[1, -1, 1]])
        np.testing.assert_allclose(state, [[.15, -.05, .01]], atol=2e-8)
        np.testing.assert_allclose(state[0] + .1 * (pulses[0] + pulses2[0]), first + second, atol=3e-8)

    def test_gradient_state_and_parameter_updates_have_explicit_learning_rate_units(self):
        # Exact binary values make conservation and gain scaling hand-checkable.
        eta, tau = .125, .25
        delta = np.array([.03125, -.03125], dtype=np.float32)
        state = np.zeros((1, 2), dtype=np.float32)
        pulses = host.residual_pulses(state, [delta], [1.], local_lr=eta, threshold=tau)
        np.testing.assert_array_equal(state, [[0, 0]])
        for fedavg_gain in (1., 2., 20.):
            update = pilot.host.pulse_aggregate(pulses, tau) * (eta * fedavg_gain)
            np.testing.assert_array_equal(update, fedavg_gain * delta)

    def test_invalid_configuration_fails_before_local_training(self):
        base = short_config()
        for changes in ({"strom_threshold": 0}, {"strom_threshold": float('nan')}, {"local_lr": 0},
                        {"server_gain": -1}, {"server_gain_exponent": .2}, {"rounds": 0}):
            with patch.object(host, '_local_delta') as local:
                with self.assertRaises(ValueError):
                    host.run_strom_codec(federation=smoke_federation(), architecture='cifar_cnn',
                                         config=replace(base, **changes), seed=83500)
                local.assert_not_called()


class NativeEquivalenceTests(unittest.TestCase):
    def test_all_six_settings_match_native_models_residuals_rng_metrics_and_legacy_costs(self):
        fed = smoke_federation()
        with tempfile.TemporaryDirectory() as folder:
            for point in pilot.POINTS:
                config = short_config(point)
                native_path = Path(folder) / (point.point_id + '_native.npz')
                original = run_final_baseline(federation=fed, architecture='cifar_cnn', method='strom', config=config,
                                              seed=83500, record_history=True,
                                              checkpoint=BaselineCheckpoint(native_path, {}), checkpoint_stride=1)
                with np.load(native_path, allow_pickle=False) as data:
                    native_model, native_residual = data['model'].copy(), data['strom_state'].copy()
                    native_rng = json.loads(str(data['metadata'].item()))['progress']['rng_state']
                actual = host.evaluated_run(federation=fed, architecture='cifar_cnn', config=config, seed=83500,
                                            checkpoint_path=Path(folder) / (point.point_id + '_codec.npz'), provenance={})
                np.testing.assert_array_equal(actual['model'].view(np.uint32), native_model.view(np.uint32))
                np.testing.assert_array_equal(actual['strom_state'].view(np.uint32), native_residual.view(np.uint32))
                np.testing.assert_array_equal(actual['model'].view(np.uint32), actual['client_model'].view(np.uint32))
                self.assertEqual(actual['rng_state'], native_rng)
                for field in pilot.EVAL_FIELDS[1:-1]:
                    np.testing.assert_array_equal(actual['evaluation'][field], original['history'][field])
                legacy = actual['summary'].set_index('variant').loc['legacy']
                self.assertEqual(legacy.uplink_bits, original['uplink_packetized_bits'])
                self.assertEqual(legacy.unicast_total_bits, original['unicast_hybrid_total_bits'])
                self.assertEqual(legacy.broadcast_total_bits, original['broadcast_total_bits'])
                expected_uploads = 16 * actual['rounds'].coordinate_events + 64 * actual['rounds'].active_clients
                np.testing.assert_array_equal(actual['rounds'].legacy_uplink_bits, expected_uploads)

    def test_resume_preserves_model_predictions_and_overshoot_state(self):
        config = short_config(pilot.POINTS[1], rounds=4)
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(federation=smoke_federation(), architecture='cifar_cnn', config=config, seed=83500, provenance={'id': 'fixed'})
            full = host.evaluated_run(**kwargs, checkpoint_path=Path(folder) / 'full.npz')
            path = Path(folder) / 'resume.npz'
            host.evaluated_run(**kwargs, checkpoint_path=path, stop_after=2)
            resumed = host.evaluated_run(**kwargs, checkpoint_path=path)
            for field in ('model', 'client_model', 'strom_state'):
                np.testing.assert_array_equal(full[field].view(np.uint32), resumed[field].view(np.uint32))
            pd.testing.assert_frame_equal(full['evaluation'], resumed['evaluation'])
            self.assertEqual(full['rng_state'], resumed['rng_state'])
            with self.assertRaisesRegex(ValueError, 'identity/provenance'):
                host.evaluated_run(**{**kwargs, 'provenance': {'id': 'different'}}, checkpoint_path=path)

    def test_decoding_error_is_fatal_and_never_applied_as_a_different_update(self):
        real_decode = host.decode_event_downlink

        def corrupt(*args, **kwargs):
            result = real_decode(*args, **kwargs)
            result[0] += np.float32(.1)
            return result

        with patch.object(host, 'decode_event_downlink', side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError, 'decoded message'):
                host.run_strom_codec(federation=smoke_federation(), architecture='cifar_cnn',
                                     config=short_config(rounds=1), seed=83500)


class ProtocolTests(unittest.TestCase):
    def test_grid_preserves_two_legacy_controls_and_exposes_gain_units(self):
        self.assertEqual((pilot.ROUNDS, pilot.PARTITION_SEED, pilot.TRAIN_SEED), (600, 3500, 83500))
        self.assertEqual(len(pilot.POINTS), 6)
        self.assertEqual(sum(p.native_gain == 1 for p in pilot.POINTS), 2)
        self.assertEqual({p.threshold for p in pilot.POINTS}, {.0025, .01})
        self.assertEqual({p.native_gain / pilot.config_for(p).local_lr for p in pilot.POINTS}, {1., 2., 20.})
        self.assertEqual(pilot.protocol_spec()['evaluation_rounds'], [1, *range(60, 601, 60)])
        self.assertFalse(pilot.protocol_spec()['test_based_selection'])
        self.assertFalse(pilot.protocol_spec()['intervals'])

    def test_existing_frozen_sources_and_archives_validate_and_prefixes_need_no_training(self):
        science = pilot.build_science()
        for tau in (.0025, .01):
            history, _ = pilot.archived_reference(tau)
            self.assertEqual(len(history), 11)
        with patch.object(pilot.full, 'evaluated_run', side_effect=AssertionError('must reuse')):
            for point in pilot.REFERENCE_POINTS:
                tables = pilot.reused_tables(point, science)
                self.assertEqual(len(tables['rounds']), 600)
                self.assertEqual(tables['evaluation']['round'].tolist(), [1, *range(60, 601, 60)])
                self.assertTrue(tables['summary'].evidence_origin.eq('reused_1800_round_prefix').all())

    def test_legacy_curve_regression_checks_metric_precision_and_exact_upload_prefixes(self):
        result = fake_result()
        reference = result['evaluation'].copy()
        reference['test_ce'] += 1e-10
        reference['uplink_packetized_bits'] = [160, 320]
        point = pilot.POINTS[2].point_id
        with patch.object(pilot, 'archived_reference', return_value=(reference, {})):
            self.assertEqual(pilot.regression_check(point, result)['status'], 'matched')
            result['rounds'].loc[0, 'legacy_uplink_bits'] += 8
            with self.assertRaisesRegex(RuntimeError, 'upload prefixes'):
                pilot.regression_check(point, result)


class EvidenceTests(unittest.TestCase):
    def test_receipts_allow_crlf_but_reject_relabelled_or_corrupted_evidence(self):
        science = fake_science()
        point = pilot.POINTS[0].point_id
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            pilot.save_result(point, science, fake_result(), out)
            pilot.verified_tables(point, science, out)
            file = pilot.paths(point, out)['evaluation']
            file.write_bytes(file.read_bytes().replace(b'\n', b'\r\n'))
            pilot.read_receipt(point, science, out)
            table = pd.read_csv(file)
            table['partition_seed'] = 4500
            pilot.atomic_csv(table, file)
            with self.assertRaisesRegex(RuntimeError, 'Changed Ström evidence'):
                pilot.read_receipt(point, science, out)
            receipt_file = pilot.paths(point, out)['receipt']
            receipt = json.loads(receipt_file.read_text())
            receipt['sha256']['evaluation'] = pilot.text_hash(file)
            pilot.atomic_json(receipt, receipt_file)
            with self.assertRaisesRegex(RuntimeError, 'metadata'):
                pilot.verified_tables(point, science, out)

    def test_complete_grid_analyzes_all_settings_and_failure_blocks_analysis(self):
        science = fake_science()

        def reused(point, contract):
            result = fake_result('event' if point == 'event' else 'stc')
            summary, history = pilot.full.trajectory_tables(result, n_clients=2, dimension=4)
            return {kind: table.assign(**pilot.metadata(point, contract, reused=True)) for kind, table in
                    (('summary', summary), ('history', history), ('rounds', result['rounds']), ('evaluation', result['evaluation']))}

        with tempfile.TemporaryDirectory() as folder, patch.object(pilot, 'build_science', return_value=science), \
                patch.object(pilot, 'ROUNDS', 2), patch.object(pilot, 'reused_tables', side_effect=reused), \
                patch.object(pilot, 'regression_check', return_value={'status': 'synthetic'}), \
                patch.object(pilot, 'plot'), redirect_stdout(io.StringIO()):
            out = Path(folder)
            pilot.atomic_json(science, out / 'protocol.json')
            for point in pilot.POINT_BY_ID:
                pilot.save_result(point, science, fake_result(), out)
            pilot.analyze(out)
            self.assertEqual(len(pd.read_csv(out / 'comparison.csv')), 32)
            self.assertEqual(len(pd.read_csv(out / 'comparison_history.csv')), 64)
            self.assertEqual(len(pd.read_csv(out / 'main_runs.csv')), 9)
            targets = pd.read_csv(out / 'traffic_to_target.csv')
            self.assertEqual(len(targets), 96)
            self.assertEqual(targets.reached.sum(), 64)
            self.assertTrue(targets.loc[~targets.reached, 'unicast_total_bits'].isna().all())
            self.assertEqual(set(pd.read_csv(out / 'main_runs.csv').point), set(pilot.POINT_BY_ID) | set(pilot.REFERENCE_POINTS))
            point = pilot.POINTS[0].point_id
            pilot.atomic_json({'identity': pilot.identity(point, science), 'state': 'numerical_failure', 'error': 'synthetic'},
                              pilot.paths(point, out)['receipt'])
            with self.assertRaisesRegex(RuntimeError, 'incomplete or failed'):
                pilot.analyze(out)
            self.assertEqual(len(json.loads((out / 'completion_audit.json').read_text())['results']), 6)

    def test_worker_uses_matched_seed_and_completed_receipt_without_retraining(self):
        science = fake_science()
        science['protocol']['dimension'] = 20570
        point = pilot.POINTS[0].point_id
        with tempfile.TemporaryDirectory() as folder, patch.object(pilot, 'build_contract', return_value=science), \
                patch.object(pilot, 'config_for', return_value=short_config(rounds=2)), \
                patch.object(pilot.full, 'make_cifar10_federation', return_value=smoke_federation()) as make, redirect_stdout(io.StringIO()):
            out = Path(folder)
            self.assertEqual(pilot.run_point(point, science, out), f'done {point}')
            self.assertEqual(make.call_args.kwargs['seed'], 3500)
            with patch.object(pilot.host, 'evaluated_run', side_effect=AssertionError('must reuse')):
                self.assertIn('reuse', pilot.run_point(point, science, out))


if __name__ == '__main__':
    unittest.main()
