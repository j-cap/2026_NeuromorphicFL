from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from experiments import resnet14_boundary as study

parent = study.parent
MANIFEST = {"synthetic": True, "training_examples": 12, "validation_examples": 4}


def write_point(p, horizon, science, out, *, screen_ce=.8, final_ce=.9, accuracy=.5, failure=False):
    files = parent.paths(p, "development", parent.data_tools.DEV_SEED, horizon, out)
    receipt = {"identity": parent.metadata(p, "development", parent.data_tools.DEV_SEED, horizon, science),
               "state": "numerical_failure" if failure else "complete"}
    if failure:
        parent.atomic_json({**receipt, "error": "synthetic numerical failure"}, files["receipt"])
        return
    hashes = [f"{i:064x}" for i in range(1, horizon + 1)]
    rounds = pd.DataFrame({"round": range(1, horizon + 1), "method": p.method,
                           "legacy_uplink_bits": 160, "codec_uplink_bits": 80, "legacy_server_bits": 160,
                           "codec_replay_bits": 80, "exact_downlink_bits": 64, "model_sha256": hashes})
    evaluation = pd.DataFrame({"round": range(1, horizon + 1), "train_objective": 1.2, "train_ce": 1.,
                               "validation_ce": [1.5, screen_ce, final_ce][:horizon], "validation_accuracy": accuracy,
                               "validation_macro_accuracy": accuracy, "validation_worst_class_accuracy": .2,
                               "model_sha256": hashes})
    tables = parent.trajectory_tables({"rounds": rounds, "evaluation": evaluation}, p=p, clients=2, dimension=8)
    for k, frame in tables.items():
        parent.atomic_csv(frame.assign(**receipt["identity"]), files[k])
    parent.atomic_json({**receipt, "sha256": {k: parent.text_hash(files[k]) for k in tables},
                        "data_manifest": MANIFEST}, files["receipt"])


@contextmanager
def campaign(*, failed_finalists=False, failed_event_screens=False):
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        root = Path(folder)
        out, prior_out = root / "boundary", root / "parent"
        science = {"protocol": {"n_clients": 2, "dimension": 8}, "synthetic_campaign": "boundary"}
        prior = {"protocol": {"n_clients": 2, "dimension": 8}, "synthetic_campaign": "parent"}
        for obj, attr, value in ((study, "SCREEN", 2), (study, "FINAL", 3),
                                 (parent, "SCREEN", 2), (parent, "FINAL", 3),
                                 (parent, "BASE", replace(parent.BASE, rounds=3, eval_stride=1)),
                                 (parent, "OUT", prior_out)):
            stack.enter_context(patch.object(obj, attr, value))
        stack.enter_context(patch.object(study, "load_contract", return_value=science))
        stack.enter_context(patch.object(parent, "load_contract", return_value=prior))
        parent.atomic_json(MANIFEST, prior_out / "split_manifest.json")
        for method, key in study.INCUMBENT_IDS.items():
            write_point(study.BY_ID[key], 3, prior, prior_out, final_ce=1., accuracy=.7)
        fed = SimpleNamespace(scope="validation", manifest=MANIFEST)
        stack.enter_context(patch.object(study, "prepare", return_value=(science, "synthetic_device", fed)))
        calls = []

        def train(p, phase, seed, horizon, contract, federation, device, output):
            assert phase == "development" and seed == 7000 and federation.scope == "validation"
            calls.append((p.point_id, phase, seed, horizon))
            index = [v for v in study.POINTS if v.method == p.method].index(p)
            write_point(p, horizon, contract, output, screen_ce=.8 + index * .1,
                        final_ce=1.2 if p.method == "event" else .9,
                        accuracy=.99 if p.method == "event" else .5,
                        failure=(failed_finalists and horizon == 3)
                        or (failed_event_screens and p.method == "event" and horizon == 2))

        stack.enter_context(patch.object(parent, "train_point", side_effect=train))
        stack.enter_context(patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("no image or test loading in orchestration")))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(out=out, prior_out=prior_out, science=science, calls=calls, fed=fed)


class ProtocolTests(unittest.TestCase):
    def test_bounded_new_candidates_equal_budget_and_native_gain_mapping(self):
        self.assertEqual([sum(p.method == m for p in study.POINTS) for m in ("event", "strom")], [3, 3])
        self.assertFalse(set(p.point_id for p in study.POINTS) & set(parent.BY_ID))
        self.assertEqual({(p.threshold, p.quantum_or_gain) for p in study.POINTS if p.method == "event"},
                         {(.005, .00125), (.01, .00125), (.0125, .00125)})
        self.assertEqual({(p.threshold, p.quantum_or_gain) for p in study.POINTS if p.method == "strom"},
                         {(.0025, 2.), (.005, 1.), (.0025, 1.)})
        self.assertEqual(parent.config_for(study.BY_ID["strom_t0p0025_g1"]).server_gain, .05)
        self.assertEqual(parent.config_for(study.BY_ID["strom_t0p0025_g2"]).server_gain, .1)
        self.assertEqual(parent.config_for(study.POINTS[1]).jump_exponent, .2)
        protocol = study.protocol_spec()
        self.assertEqual(protocol["max_additional_training_rounds"], 6 * 600 + 2 * 1200)
        self.assertEqual(protocol["max_unique_evaluation_points"], 6 * 11 + 2 * 20)
        self.assertEqual(protocol["new_confirmation_runs"], 0)
        self.assertFalse(protocol["automatic_further_boundary_extension"])

    def test_frozen_parent_sources_and_development_only_dependencies(self):
        original = parent.paths
        phases = []

        def paths(p, phase, *args, **kwargs):
            phases.append(phase)
            self.assertEqual(phase, "development")
            return original(p, phase, *args, **kwargs)

        with patch.object(parent, "paths", side_effect=paths), \
                patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("no images")):
            science = study.science_spec()
        self.assertTrue(phases)
        self.assertIn("experiments/resnet14_boundary.py", science["source_hashes"])
        self.assertTrue(all("confirmation" not in p for p in science["parent_evidence_sha256"]))
        self.assertFalse(science["protocol"]["official_test_metrics_read"])

    def test_altered_contract_blocks_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(study, "science_spec", return_value={"changed": True}):
            out = Path(folder)
            parent.atomic_json({"changed": False}, out / "protocol.json")
            with self.assertRaisesRegex(RuntimeError, "sources, parent evidence or runtime changed"):
                study.load_contract(out)


class IsolationTests(unittest.TestCase):
    def test_prepare_matches_parent_holdout_and_never_requests_test_arrays(self):
        science = {"data_sha256": {"data_batch_1": "hash", "test_batch": "hash"},
                   "execution": {"runtime": {"synthetic": True}, "eval_batch_size": 4}}
        host = SimpleNamespace(configure=lambda *_: "synthetic_device", runtime=lambda _: {"synthetic": True})
        with tempfile.TemporaryDirectory() as folder:
            prior_out, out = Path(folder) / "parent", Path(folder) / "boundary"
            parent.atomic_json(MANIFEST, prior_out / "split_manifest.json")
            fed = SimpleNamespace(scope="validation", manifest=MANIFEST)
            with patch.dict(sys.modules, {"resnet14_transfer_host": host}), \
                    patch.object(study, "science_spec", return_value=science), patch.object(parent, "OUT", prior_out), \
                    patch.object(parent.data_tools, "download_cifar10", return_value=Path("raw")), \
                    patch.object(parent.full.development, "file_hash", return_value="hash"), \
                    patch.object(parent.data_tools, "load_arrays", return_value=("images", "labels", None)) as loader, \
                    patch.object(parent.data_tools, "federation", return_value=(fed, pd.DataFrame())) as federation:
                result = study.prepare("cpu", 2, 4, out)
                self.assertIs(result[2], fed)
                loader.assert_called_once_with(parent.DATA, include_test=False)
                self.assertTrue(federation.call_args.kwargs["validation"])
                self.assertNotIn("test", federation.call_args.kwargs)
                with self.assertRaisesRegex(RuntimeError, "recorded parent runtime"):
                    study.check_runtime(science, "synthetic_device", 8)
                loader.return_value = ("images", "labels", ("test_images", "test_labels"))
                with self.assertRaisesRegex(RuntimeError, "test data are prohibited"):
                    study.prepare("cpu", 2, 4, out)

    def test_orchestrator_refuses_test_scope_before_training(self):
        with campaign() as c:
            with patch.object(study, "prepare", return_value=(c.science, "device", SimpleNamespace(scope="test"))):
                with self.assertRaisesRegex(RuntimeError, "requires validation scope"):
                    study.run(out=c.out)
            self.assertEqual(c.calls, [])


class WorkflowTests(unittest.TestCase):
    def test_exact_budget_ce_selection_and_parent_preservation(self):
        with campaign() as c:
            before = {p.name: p.read_bytes() for p in c.prior_out.iterdir()}
            chosen = study.run(out=c.out)
            self.assertEqual(len(c.calls), 8)
            self.assertEqual(sum(v[3] == 2 for v in c.calls), 6)
            self.assertEqual(sum(v[3] == 3 for v in c.calls), 2)
            self.assertEqual({v[2] for v in c.calls}, {7000})
            self.assertEqual(chosen["selected"]["event"], [study.INCUMBENT_IDS["event"]])
            self.assertEqual(chosen["selected"]["strom"], ["strom_t0p0025_g2"])
            self.assertEqual(chosen["changed"], {"event": False, "strom": True})
            self.assertTrue(chosen["fresh_matched_confirmation_required"])
            self.assertFalse(chosen["confirmation_launched"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in c.prior_out.iterdir()})
            comparison = pd.read_csv(c.out / "comparison.csv")
            self.assertEqual(len(comparison), 4)
            audit = json.loads((c.out / "analysis_audit.json").read_text())
            self.assertEqual(audit["verified_new_archived_round_rows"], 18)
            self.assertEqual(audit["verified_exact_screen_prefixes"], 2)
            self.assertFalse(audit["official_test_images_or_metrics_read"])

    def test_failed_finalists_keep_incumbents_without_extra_promotion(self):
        with campaign(failed_finalists=True) as c:
            chosen = study.run(out=c.out)
            self.assertEqual(chosen["changed"], {"event": False, "strom": False})
            self.assertFalse(chosen["fresh_matched_confirmation_required"])
            self.assertEqual(len(c.calls), 8)
            self.assertEqual(int(pd.read_csv(c.out / "finalists.csv").state.eq("numerical_failure").sum()), 2)
            audit = json.loads((c.out / "analysis_audit.json").read_text())
            self.assertEqual(audit["failed_new_stage_receipts"], 2)

    def test_all_new_screens_failed_stops_and_keeps_failure_ledger(self):
        with campaign(failed_event_screens=True) as c:
            with self.assertRaisesRegex(RuntimeError, "All event candidates failed"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 6)
            self.assertEqual(int(pd.read_csv(c.out / "screening.csv").state.eq("numerical_failure").sum()), 3)
            self.assertFalse((c.out / "promotion.json").exists())

    def test_missing_stage_blocks_final_analysis(self):
        with campaign() as c:
            study.run(out=c.out)
            parent.paths(study.POINTS[0], "development", 7000, study.SCREEN, c.out)["receipt"].unlink()
            with self.assertRaisesRegex(RuntimeError, "stage is incomplete"):
                study.analyze(c.out)

    def test_rehashed_changed_prefix_is_rejected(self):
        with campaign() as c:
            study.run(out=c.out)
            p = study.BY_ID["event_t0p005_q0p00125"]
            write_point(p, 3, c.science, c.out, screen_ce=.7, final_ce=1.2, accuracy=.99)
            with self.assertRaises(AssertionError):
                study.analyze(c.out)

    def test_saved_data_manifest_and_immutable_promotion_are_checked(self):
        with campaign() as c:
            study.run(out=c.out)
            p = study.POINTS[0]
            file = parent.paths(p, "development", 7000, 2, c.out)["receipt"]
            receipt = json.loads(file.read_text())
            receipt["data_manifest"] = {"wrong_holdout": True}
            parent.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "data manifest differs"):
                study.analyze(c.out)
            receipt["data_manifest"] = MANIFEST
            parent.atomic_json(receipt, file)
            promotion = json.loads((c.out / "promotion.json").read_text())
            promotion["selected"]["event"] = [study.POINTS[1].point_id]
            parent.atomic_json(promotion, c.out / "promotion.json")
            with self.assertRaisesRegex(RuntimeError, "screen promotion changed"):
                study.analyze(c.out)


if __name__ == "__main__":
    unittest.main()
