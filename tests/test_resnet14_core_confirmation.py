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

import numpy as np
import pandas as pd
from experiments import resnet14_core_confirmation as study

parent = study.parent


def synthetic_result(method, seed, *, pulse=False, censored=False):
    native_method = method
    hashes = [f"{seed * 100 + study.METHODS.index(method) * 10 + i:064x}" for i in range(1, 4)]
    if pulse:
        rows = {"round": [1, 2, 3], "method": method, "coordinate_events": 8, "active_clients": 2,
                "uplink_modes": json.dumps({"rice:fixed:0": 2}), "legacy_uplink_bits": 160, "codec_uplink_bits": 80,
                "legacy_server_bits": 160, "codec_replay_bits": 80, "exact_downlink_bits": 64,
                "integer_only_downlink_bits": 64, "aggregate_exact_downlink_bits": 64, "ordered_downlink_bits": 152,
                "downlink_mode": "counts_patch", "model_sha256": hashes}
    else:
        up, down, replay = (640, 320, 0) if method == "dense" else (256, 128, 0) if method == "stc" else (192, 128, 352)
        mode = {"dense": "dense_float32", "stc": "lossless_stc", "ef_topk": "support_rice:fixed:0"}[method]
        rows = {"round": [1, 2, 3], "method": native_method, "coordinate_events": 8, "active_clients": 2,
                "codec_uplink_bits": up, "exact_downlink_bits": down, "aggregate_downlink_bits": down,
                "ordered_downlink_bits": replay, "uplink_modes": json.dumps({mode: 2}),
                "downlink_mode": "aggregate_" + mode if method == "ef_topk" else mode,
                "server_gain": study.config_for(method).server_gain, "roundtrip_verified": True,
                "wire_protocol": study.host.WIRE_PROTOCOL, "model_sha256": hashes}
    acc = [.46, .51, .54 if censored else .58 if method == "stc" else .56]
    evaluation = pd.DataFrame({"round": [1, 2, 3], "train_objective": 1.2, "train_ce": 1., "test_ce": 1.1,
                               "test_accuracy": acc, "test_macro_accuracy": acc, "test_worst_class_accuracy": .2,
                               "model_sha256": hashes})
    return {"rounds": pd.DataFrame(rows), "evaluation": evaluation}


@contextmanager
def campaign(*, interrupt_after=None, failure=None):
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        root = Path(folder)
        out, prior_out = root / "core", root / "pulse"
        manifests = {str(s): {"seed": s, "training_examples": 50000, "validation_examples": 0} for s in study.SEEDS}
        prior = {"synthetic": "pulse"}
        science = {"protocol": {"dimension": 8, "n_clients": 2, "scope": "synthetic_core"},
                   "selection": {"selected": study.POINT_IDS}, "partition_manifests": manifests,
                   "reference_contract_sha256": parent.digest(prior), "references": {}}
        configs = {m: replace(study.config_for(m), rounds=3, eval_stride=1) for m in study.METHODS}
        stack.enter_context(patch.object(study, "ROUNDS", 3))
        stack.enter_context(patch.object(study, "config_for", side_effect=lambda m: configs[m]))
        stack.enter_context(patch.object(study, "REPO", root))
        stack.enter_context(patch.object(study.pulse, "OUT", prior_out))
        stack.enter_context(patch.object(study, "science_spec", return_value=science))
        parent.atomic_json(prior, prior_out / "protocol.json")
        source_tables, source_paths = {}, {}
        for seed in study.SEEDS:
            for method in study.REUSED:
                result = synthetic_result(method, seed, pulse=True)
                table = {k: frame.assign(method=method, partition_seed=seed, train_seed=parent.train_seed(seed),
                                         point_id=study.point_for(method).point_id, phase="confirmation")
                         for k, frame in parent.trajectory_tables(result, p=study.point_for(method), clients=2, dimension=8).items()}
                paths = {k: prior_out / f"{method}_p{seed}_{k}.csv" for k in table}
                for k, frame in table.items():
                    parent.atomic_csv(frame, paths[k])
                source_tables[method, seed] = table
                source_paths[method, seed] = paths
        stack.enter_context(patch.object(study, "reference_files", side_effect=lambda m, s: source_paths[m, s]))
        stack.enter_context(patch.object(study.pulse, "verified_tables", side_effect=lambda m, s, c: source_tables[m, s]))
        for seed in study.SEEDS:
            for method in study.REUSED:
                science["references"][study.reference_key(method, seed)] = study.fingerprint_reference(method, seed)
        calls, loads, interrupted = [], [], [False]

        def prepare(device_name, threads, batch, output):
            parent.immutable_json(science, output / "protocol.json")
            return science, "synthetic_device"

        def load(root, include_test=False):
            assert include_test and (out / "protocol.json").exists()
            assert all(study.paths(m, s, out)["receipt"].exists() for s in study.SEEDS for m in study.REUSED)
            loads.append(include_test)
            return "training_images", "labels", ("test_images", "test_labels")

        def federation(images, labels, *, seed, validation, device, test):
            assert not validation and test == ("test_images", "test_labels")
            return SimpleNamespace(scope="test", manifest=manifests[str(seed)]), pd.DataFrame()

        def train(method, seed, contract, fed, device, output):
            assert method in study.NEW and fed.scope == "test" and fed.manifest == manifests[str(seed)]
            if study.read_outcome(method, seed, contract, output) is not None:
                return
            if interrupt_after is not None and len(calls) == interrupt_after and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt("synthetic interruption")
            calls.append((method, seed))
            if failure == (method, seed):
                parent.atomic_json({"identity": study.metadata(method, seed, contract), "state": "numerical_failure",
                                    "error": "synthetic failure", "data_manifest": fed.manifest}, study.paths(method, seed, output)["receipt"])
                return
            result = synthetic_result(method, seed, censored=method == "ef_topk" and seed == study.SEEDS[-1])
            study.save_result(method, seed, contract, result, fed.manifest, output)

        stack.enter_context(patch.object(study, "prepare", side_effect=prepare))
        stack.enter_context(patch.object(parent.data_tools, "load_arrays", side_effect=load))
        stack.enter_context(patch.object(parent.data_tools, "federation", side_effect=federation))
        stack.enter_context(patch.object(study, "train_point", side_effect=train))
        stack.enter_context(patch.object(study, "plot"))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(out=out, prior_out=prior_out, science=science, calls=calls, loads=loads, sources=source_paths)


class ProtocolTests(unittest.TestCase):
    def test_only_authorized_selected_settings_and_common_seeds(self):
        self.assertEqual(study.NEW, ("dense", "ef_topk", "stc"))
        self.assertEqual(study.SEEDS, tuple(range(7100, 8100, 100)))
        self.assertEqual([parent.train_seed(s) for s in study.SEEDS], list(range(97100, 98100, 100)))
        self.assertEqual((study.config_for("dense").server_gain, study.config_for("dense").server_gain_exponent), (1.5, 0.))
        self.assertEqual((study.config_for("ef_topk").topk_fraction, study.config_for("ef_topk").server_gain), (.01, 4.))
        self.assertEqual((study.config_for("stc").stc_upload_fraction, study.config_for("stc").stc_download_fraction, study.config_for("stc").server_gain), (1., 1., 4.))
        p = study.protocol_spec()
        self.assertEqual((p["new_training_runs"], p["reused_training_results"], p["additional_training_rounds"]), (30, 20, 54000))
        self.assertEqual((p["expected_outcomes"], p["additional_evaluation_points"], p["total_sampled_model_fingerprints"]), (50, 930, 1550))
        self.assertEqual(p["sparse_STC_secondary_runs"], 0)
        self.assertFalse(p["development_checkpoint_reuse"])
        self.assertFalse(p["new_tuning"])
        self.assertFalse(p["automatic_extra_methods_or_seed_replacement"])

    def test_real_choices_twenty_reuses_and_full_training_manifests_without_images(self):
        with patch.object(parent.data_tools, "load_arrays", side_effect=AssertionError("scientific pinning must not parse images")):
            science = study.science_spec()
        self.assertEqual(len(science["references"]), 20)
        self.assertEqual(len(science["partition_manifests"]), 10)
        self.assertEqual(science["selection"]["selected"], {m: [k] for m, k in study.POINT_IDS.items()})
        self.assertTrue(all(m["training_examples"] == 50000 and m["validation_examples"] == 0 for m in science["partition_manifests"].values()))
        self.assertTrue(all("\\" not in p for p in science["source_hashes"]))
        self.assertTrue(all("\\" not in entry["path"] for ref in science["references"].values() for entry in ref.values()))
        self.assertTrue(all("development" not in entry["path"] for ref in science["references"].values() for entry in ref.values()))

    def test_new_full_training_checkpoints_cannot_alias_development_or_pulse_checkpoints(self):
        files = {study.checkpoint_path(m, s) for m in study.NEW for s in study.SEEDS}
        self.assertEqual(len(files), 30)
        for path in files:
            self.assertTrue(path.is_relative_to(study.OUT))
            self.assertNotIn("development", path.name)
        with self.assertRaises(ValueError):
            study.checkpoint_path("event", study.SEEDS[0])

    def test_real_reused_contrast_reproduces_completed_pulse_statistics(self):
        prior = study.pulse.load_contract()
        summaries, crossings = [], []
        for seed in study.SEEDS:
            for method in study.REUSED:
                table = study.pulse.verified_tables(method, seed, prior)
                summaries.append(table["summary"][table["summary"].variant == "both_codecs"].assign(comparison_id=method))
                curve = table["history"][table["history"].variant == "both_codecs"].copy()
                curve["uplink_packetized_bits"] = curve.uplink_bits
                crossings.extend(parent.target_crossings(curve, targets=parent.TARGETS, horizon=study.ROUNDS,
                                                          comparison_id=method, partition_seed=seed))
        with patch.object(study, "METHODS", study.REUSED):
            _, effects, target_effects = study.paired_analysis(pd.concat(summaries, ignore_index=True), pd.DataFrame(crossings))
        original = pd.read_csv(study.pulse.OUT / "paired_summary.csv").query('scope == "all_10"').set_index("metric")
        effects = effects.set_index("metric")
        fields = ["n", "estimate", "ci95_low", "ci95_high"]
        np.testing.assert_allclose(effects.loc[original.index, fields], original[fields], rtol=1e-10, atol=1e-10)
        original = pd.read_csv(study.pulse.OUT / "paired_target_summary.csv").query('scope == "all_10"').set_index("target_accuracy")
        target_effects = target_effects.set_index("target_accuracy")
        np.testing.assert_allclose(target_effects.loc[original.index, fields], original[fields], rtol=1e-10, atol=1e-10)


class WorkflowTests(unittest.TestCase):
    def test_full_grid_exact_reuse_and_no_retraining_on_repeat(self):
        with campaign() as c:
            before = {str(p): p.read_bytes() for paths in c.sources.values() for p in paths.values()}
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 30)
            self.assertEqual(len(set(c.calls)), 30)
            self.assertEqual(c.loads, [True])
            complete = json.loads((c.out / "confirmation_completion.json").read_text())
            self.assertEqual(len(complete["results"]), 50)
            self.assertEqual(sum(r["state"] == "reused" for r in complete["results"]), 20)
            runs = pd.read_csv(c.out / "confirmation_runs.csv")
            self.assertEqual(len(runs), 50)
            self.assertEqual(set(runs.comparison_id), set(study.METHODS))
            for method in study.REUSED:
                for seed in study.SEEDS:
                    table = study.verified_tables(method, seed, c.science, c.out)
                    self.assertEqual(table["summary"].core_evidence_origin.unique().tolist(), ["verified_reuse"])
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 30)
            self.assertEqual(c.loads, [True])
            self.assertEqual(before, {str(p): p.read_bytes() for paths in c.sources.values() for p in paths.values()})

    def test_interruption_resume_skips_complete_fits_and_preserves_exact_identity(self):
        with campaign(interrupt_after=4) as c:
            with self.assertRaises(KeyboardInterrupt):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 4)
            study.run(out=c.out)
            self.assertEqual(len(c.calls), 30)
            self.assertEqual(len(set(c.calls)), 30)
            self.assertEqual(c.loads, [True, True])

    def test_failure_retained_other_fits_finish_and_final_analysis_blocks(self):
        failed = ("dense", study.SEEDS[0])
        with campaign(failure=failed) as c:
            with self.assertRaisesRegex(RuntimeError, "All fifty outcomes"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 30)
            complete = json.loads((c.out / "confirmation_completion.json").read_text())
            self.assertEqual(sum(r["state"] == "numerical_failure" for r in complete["results"]), 1)
            with self.assertRaisesRegex(RuntimeError, "All fifty outcomes"):
                study.run(out=c.out)
            self.assertEqual(len(c.calls), 30)

    def test_censored_targets_and_negative_Event_effects_are_not_omitted(self):
        with campaign() as c:
            study.run(out=c.out)
            targets = pd.read_csv(c.out / "traffic_to_target.csv")
            row = targets[(targets.comparison_id == "ef_topk") & (targets.partition_seed == study.SEEDS[-1]) & (targets.target_accuracy == .55)].iloc[0]
            self.assertFalse(row.reached)
            self.assertEqual(row.censor_round, 3)
            self.assertTrue(np.isnan(row.unicast_total_bits))
            summary = pd.read_csv(c.out / "paired_target_summary.csv")
            row = summary[(summary.reference == "ef_topk") & (summary.target_accuracy == .55)].iloc[0]
            self.assertEqual((row.n_total, row.n_event_reached, row.n_reference_reached, row.n_common_hits), (10, 10, 9, 9))
            effects = pd.read_csv(c.out / "paired_summary.csv")
            value = effects[(effects.reference == "stc") & (effects.metric == "accuracy_difference_pp")].iloc[0].estimate
            self.assertAlmostEqual(value, -2.)

    def test_changed_new_and_reused_evidence_blocks_resume(self):
        for reused in (False, True):
            with self.subTest(reused=reused), campaign() as c:
                study.run(out=c.out)
                path = c.sources["event", study.SEEDS[0]]["evaluation"] if reused else study.paths("dense", study.SEEDS[0], c.out)["evaluation"]
                frame = pd.read_csv(path)
                frame.loc[0, "test_accuracy"] += .01
                parent.atomic_csv(frame, path)
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    study.run(out=c.out)

    def test_changed_source_contract_blocks_resume(self):
        with campaign() as c:
            parent.immutable_json({**c.science, "tampered": True}, c.out / "protocol.json")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                study.load_contract(c.out)

    def test_wrong_training_allocation_and_sampled_model_identity_are_rejected(self):
        for kind in ("manifest", "fingerprint"):
            with self.subTest(kind=kind), campaign() as c:
                study.run(out=c.out)
                files = study.paths("dense", study.SEEDS[0], c.out)
                receipt = json.loads(files["receipt"].read_text())
                if kind == "manifest":
                    receipt["data_manifest"]["training_examples"] = 45000
                else:
                    e = pd.read_csv(files["evaluation"], dtype={"model_sha256": str})
                    e.loc[0, "model_sha256"] = "f" * 64
                    parent.atomic_csv(e, files["evaluation"])
                    receipt["sha256"]["evaluation"] = parent.text_hash(files["evaluation"])
                parent.atomic_json(receipt, files["receipt"])
                with self.assertRaises(RuntimeError):
                    study.verified_tables("dense", study.SEEDS[0], c.science, c.out)


if __name__ == "__main__":
    unittest.main()
