from __future__ import annotations

from contextlib import contextmanager, ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from experiments import resnet14_ten_seed as study

parent = study.parent


def write_point(method, seed, science, out, manifests, *, failure=False):
    p = study.point_for(method)
    files = parent.paths(p, "confirmation", seed, study.ROUNDS, out)
    receipt = {"identity": parent.metadata(p, "confirmation", seed, study.ROUNDS, science),
               "state": "numerical_failure" if failure else "complete"}
    if failure:
        parent.atomic_json({**receipt, "error": "synthetic failure"}, files["receipt"])
        return
    hashes = [f"{seed * 100 + (0 if method == 'event' else 10) + i:064x}" for i in range(1, 4)]
    rounds = pd.DataFrame({"round": [1, 2, 3], "method": method, "legacy_uplink_bits": 160,
                           "codec_uplink_bits": 80, "legacy_server_bits": 160, "codec_replay_bits": 80,
                           "exact_downlink_bits": 64, "model_sha256": hashes})
    acc = [.46, .51, .54 if method == "strom" and seed == 8000 else .56]
    evaluation = pd.DataFrame({"round": [1, 2, 3], "train_objective": 1.2, "train_ce": 1.,
                               "test_ce": 1.1 if method == "event" else 1., "test_accuracy": acc,
                               "test_macro_accuracy": acc, "test_worst_class_accuracy": .2,
                               "model_sha256": hashes})
    tables = parent.trajectory_tables({"rounds": rounds, "evaluation": evaluation}, p=p, clients=2, dimension=8)
    for k, frame in tables.items():
        parent.atomic_csv(frame.assign(**receipt["identity"]), files[k])
    parent.atomic_json({**receipt, "sha256": {k: parent.text_hash(files[k]) for k in tables},
                        "data_manifest": manifests[str(seed)]}, files["receipt"])


@contextmanager
def campaign(*, interrupt_after=None, failure=None):
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        root = Path(folder)
        out, prior_out = root / "extension", root / "parent"
        manifests = {str(s): {"seed": s, "full_training": True, "synthetic": True} for s in study.SEEDS}
        prior = {"protocol": {"n_clients": 2, "dimension": 8}, "synthetic": "prior"}
        science = {"protocol": {"n_clients": 2, "dimension": 8}, "synthetic": "extension",
                   "selection": {"selected": study.POINT_IDS}, "partition_manifests": manifests,
                   "reference_contract_sha256": parent.digest(prior), "references": {}}
        for obj, key, value in ((study, "ROUNDS", 3), (study, "REPO", root), (parent, "OUT", prior_out),
                                (parent, "BASE", replace(parent.BASE, rounds=3, eval_stride=1))):
            stack.enter_context(patch.object(obj, key, value))
        stack.enter_context(patch.object(parent, "load_contract", return_value=prior))
        stack.enter_context(patch.object(study, "load_contract", return_value=science))
        for seed in study.PILOT_SEEDS:
            parent.atomic_json(manifests[str(seed)], prior_out / f"confirmation_partition_p{seed}.json")
            for method in study.METHODS:
                write_point(method, seed, prior, prior_out, manifests)
                science["references"][study.reference_key(method, seed)] = study.fingerprint_reference(method, seed)
        calls, loads, interrupted = [], [], [False]

        def prepare(device_name, threads, eval_batch_size, output):
            parent.immutable_json(science, output / "protocol.json")
            return science, "synthetic_device"

        def load(root, include_test=False):
            assert include_test and (out / "protocol.json").exists()
            assert all(study.reuse_path(m, s, out).exists() for s in study.PILOT_SEEDS for m in study.METHODS)
            loads.append(include_test)
            return "training_images", "labels", ("test_images", "test_labels")

        def federation(images, labels, *, seed, validation, device, test):
            assert seed in study.NEW_SEEDS and not validation and test == ("test_images", "test_labels")
            return SimpleNamespace(scope="test", manifest=manifests[str(seed)]), pd.DataFrame()

        def train(p, phase, seed, horizon, contract, fed, device, output):
            assert phase == "confirmation" and seed in study.NEW_SEEDS and horizon == 3 and fed.scope == "test"
            if parent.read_receipt(p, phase, seed, horizon, contract, output) is not None:
                return
            if interrupt_after is not None and len(calls) == interrupt_after and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt("synthetic interruption")
            calls.append((p.method, seed, horizon))
            write_point(p.method, seed, contract, output, manifests, failure=failure == (p.method, seed))

        stack.enter_context(patch.object(study, "prepare", side_effect=prepare))
        stack.enter_context(patch.object(parent.data_tools, "load_arrays", side_effect=load))
        stack.enter_context(patch.object(parent.data_tools, "federation", side_effect=federation))
        stack.enter_context(patch.object(parent, "train_point", side_effect=train))
        stack.enter_context(patch.object(study, "plot"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(out=out, prior_out=prior_out, science=science, calls=calls, loads=loads, manifests=manifests)


class ProtocolTests(unittest.TestCase):
    def test_fixed_configurations_fresh_seeds_and_exact_budget(self):
        self.assertEqual(study.SEEDS, tuple(range(7100, 8100, 100)))
        self.assertEqual(study.NEW_SEEDS, tuple(range(7400, 8100, 100)))
        self.assertEqual([parent.train_seed(s) for s in study.NEW_SEEDS], list(range(97400, 98100, 100)))
        event, strom = (parent.config_for(study.point_for(m)) for m in study.METHODS)
        self.assertEqual((event.threshold, event.jump0, event.jump_exponent), (.01, .0025, .2))
        self.assertEqual((strom.strom_threshold, strom.server_gain), (.005, .1))
        spec = study.protocol_spec()
        self.assertEqual(spec["additional_training_rounds"], 14 * 1800)
        self.assertEqual(spec["additional_evaluation_points"], 14 * 31)
        self.assertEqual(spec["total_evaluated_model_fingerprints"], 20 * 31)
        self.assertEqual(spec["scopes"]["additional_7"], list(study.NEW_SEEDS))
        self.assertFalse(spec["development_checkpoint_reuse"])
        self.assertFalse(spec["automatic_retuning_or_seed_replacement"])

    def test_real_selection_reuse_manifests_and_paths_require_no_images(self):
        with patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("no image loading during scientific pinning")):
            science = study.science_spec()
        self.assertEqual(len(science["references"]), 6)
        self.assertEqual(len(science["partition_manifests"]), 10)
        self.assertEqual(science["selection"]["selected"], {m: [k] for m, k in study.POINT_IDS.items()})
        self.assertTrue(all("\\" not in path for path in science["selection"]["files"]))
        self.assertTrue(all("\\" not in info["path"] for ref in science["references"].values() for info in ref.values()))
        self.assertTrue(all(m["training_examples"] == 50000 and m["validation_examples"] == 0
                            for m in science["partition_manifests"].values()))

    def test_altered_protocol_source_or_reference_identity_blocks_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(study, "science_spec", return_value={"identity": "new"}):
            out = Path(folder)
            parent.atomic_json({"identity": "old"}, out / "protocol.json")
            with self.assertRaisesRegex(RuntimeError, "sources, selection, runtime or reused evidence changed"):
                study.load_contract(out)


class WorkflowTests(unittest.TestCase):
    def test_reuses_six_trains_fourteen_and_reports_both_scopes_and_censoring(self):
        with campaign() as c:
            original = {p.name: p.read_bytes() for p in c.prior_out.iterdir()}
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 14)
            self.assertEqual(set(c.calls), {(m, s, 3) for m in study.METHODS for s in study.NEW_SEEDS})
            self.assertEqual(original, {p.name: p.read_bytes() for p in c.prior_out.iterdir()})
            self.assertEqual(c.loads, [True])
            runs = pd.read_csv(c.out / "confirmation_runs.csv")
            self.assertEqual(runs.evidence_origin.value_counts().to_dict(), {"new_additional": 14, "reused_pilot": 6})
            for scope, n in (("all_10", 10), ("additional_7", 7)):
                effects = pd.read_csv(c.out / "paired_summary.csv")
                self.assertTrue(effects[effects.scope == scope].n.eq(n).all())
            targets = pd.read_csv(c.out / "paired_target_summary.csv")
            rows = targets[targets.target_accuracy == .55].set_index("scope")
            self.assertEqual((rows.loc["all_10", "n_total"], rows.loc["all_10", "n_common_hits"]), (10, 9))
            self.assertEqual((rows.loc["additional_7", "n_total"], rows.loc["additional_7", "n_common_hits"]), (7, 6))
            crossing = pd.read_csv(c.out / "traffic_to_target.csv")
            censored = crossing[(crossing.partition_seed == 8000) & (crossing.comparison_id == "strom") & (crossing.target_accuracy == .55)].iloc[0]
            self.assertFalse(censored.reached)
            self.assertEqual(censored.censor_round, 3)
            self.assertTrue(pd.isna(censored.unicast_total_bits))
            audit = json.loads((c.out / "analysis_audit.json").read_text())
            self.assertEqual((audit["rounds_verified"], audit["sampled_model_fingerprints"]), (60, 60))

    def test_interruption_resumes_pending_work_without_retraining_pilot_or_completed_new_runs(self):
        with campaign(interrupt_after=3) as c:
            with self.assertRaises(KeyboardInterrupt):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 3)
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 14)
            self.assertEqual(len(set(c.calls)), 14)
            self.assertEqual(len(c.loads), 2)
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 14)
            self.assertEqual(len(c.loads), 2)

    def test_failure_is_retained_all_new_runs_attempted_and_final_summary_is_blocked(self):
        with campaign(failure=("event", 8000)) as c:
            with self.assertRaisesRegex(RuntimeError, "All ten pairs are required"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 14)
            results = json.loads((c.out / "confirmation_completion.json").read_text())["results"]
            self.assertEqual(len(results), 20)
            self.assertEqual(sum(r["state"] == "numerical_failure" for r in results), 1)
            self.assertFalse((c.out / "paired_summary.csv").exists())

    def test_missing_reuse_outcome_cannot_be_dropped_from_ten_seed_analysis(self):
        with campaign() as c:
            study.run(out=c.out)
            study.reuse_path("event", 7100, c.out).unlink()
            with self.assertRaisesRegex(RuntimeError, "All ten pairs are required"):
                study.analyze(c.out)

    def test_changed_pilot_metric_source_is_rejected_before_reuse(self):
        with campaign() as c:
            study.run(out=c.out)
            file = study.reference_files("event", 7100)["evaluation"]
            file.write_text(file.read_text() + "changed\n")
            with self.assertRaisesRegex(RuntimeError, "source fingerprints changed"):
                study.read_outcome("event", 7100, c.science, c.out)

    def test_new_run_cannot_be_marked_as_reused_or_relabelled_as_another_configuration(self):
        with campaign() as c:
            study.run(out=c.out)
            file = parent.paths(study.point_for("event"), "confirmation", 7400, 3, c.out)["receipt"]
            receipt = json.loads(file.read_text())
            receipt["state"] = "reused"
            parent.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "Unexpected ResNet completion state"):
                study.read_outcome("event", 7400, c.science, c.out)
            receipt["state"] = "complete"
            receipt["identity"]["point_id"] = "other_configuration"
            parent.atomic_json(receipt, file)
            with self.assertRaisesRegex(RuntimeError, "receipt identity differs"):
                study.read_outcome("event", 7400, c.science, c.out)

    def test_wrong_full_training_partition_sidecar_blocks_analysis(self):
        with campaign() as c:
            study.run(out=c.out)
            parent.atomic_json({"wrong_partition": True}, c.out / "confirmation_partition_p7400.json")
            with self.assertRaisesRegex(RuntimeError, "pinned full-training partition"):
                study.analyze(c.out)


if __name__ == "__main__":
    unittest.main()
