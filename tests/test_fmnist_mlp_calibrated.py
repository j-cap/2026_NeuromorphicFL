from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "experiments")]
import fmnist_mlp_calibrated as study
import fmnist_mlp_data as data_tools
import fmnist_mlp_host as host
import fmnist_mlp_analysis as analysis
import numpy as np
import pandas as pd
from neuromorphicfl import event_codec as native
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import run_final_baseline
from neuromorphicfl.fmnist_multiclass_benchmark import MulticlassFederation, predictive_metrics


def synthetic():
    rng = np.random.default_rng(631)
    X = rng.normal(0, .2, (90, 784)).astype(np.float32)
    y = np.tile(np.arange(10, dtype=np.int64), 9)
    data = {"client_X": (X[:30], X[30:60], X[60:90]), "client_y": (y[:30], y[30:60], y[60:90]),
            "X_diagnostic": X, "y_diagnostic": y, "X_holdout": X[:40], "y_holdout": y[:40], "scope": "test"}
    fed = MulticlassFederation(data["client_X"], data["client_y"], X, y, X[:40], y[:40], "synthetic",
                              np.full((3, 10), 3), np.ones(3, int), np.full(3, 1/3))
    return data, fed


def tiny_config(method):
    return replace(study.BASE, rounds=4, eval_stride=1, local_steps=1, threshold=.0125, jump0=.003,
                   jump_exponent=.2, strom_threshold=.0025, server_gain=.2,
                   server_gain_exponent=.1 if method == "dense" else 0., topk_fraction=.05,
                   stc_upload_fraction=.25, stc_download_fraction=.05)


class SplitAndDesignTests(unittest.TestCase):
    def test_exact_disjoint_development_and_confirmation_partitions(self):
        labels = np.repeat(np.arange(10), 6000)
        for validation, seed in ((True, study.DEV_SEED), (False, study.SEEDS[0])):
            clients, held, diag, manifest = data_tools.allocation(labels, seed=seed, validation=validation)
            expected_size = 5400 if validation else 6000
            combined = np.concatenate([*clients, held])
            np.testing.assert_array_equal(np.sort(combined), np.arange(60000))
            self.assertEqual(len(np.unique(combined)), 60000)
            self.assertEqual(len(held), 6000 if validation else 0)
            for client, ids in enumerate(clients):
                expected = np.full(10, expected_size//20)
                expected[client] = expected_size*11//20
                np.testing.assert_array_equal(np.bincount(labels[ids], minlength=10), expected)
            self.assertFalse(np.intersect1d(diag, held).size)
            self.assertEqual(len(diag), 5000)
            self.assertEqual(manifest, data_tools.allocation(labels, seed=seed, validation=validation)[3])

    def test_validation_loader_never_opens_official_test_arrays(self):
        seen = []
        def load(path, **kwargs):
            seen.append(Path(path).name)
            return np.zeros((1, 784), np.float32) if "images" in str(path) else np.zeros(1, np.int64)
        with patch.object(data_tools.np, "load", side_effect=load):
            data_tools.open_arrays(Path("unused"), include_test=False)
        self.assertEqual(seen, ["train_images.npy", "train_labels.npy"])

    def test_equal_bounded_grids_and_distinct_common_seeds(self):
        self.assertEqual(host.LAYOUT.widths, (784, 32, 16, 10))
        self.assertEqual(host.LAYOUT.dimension, 25818)
        self.assertEqual(len(study.BY_ID), 90)
        for method in study.METHODS:
            self.assertEqual(sum(p.method == method for p in study.POINTS), 18)
        self.assertEqual(len(set(study.SEEDS)), 10)
        self.assertNotIn(study.DEV_SEED, study.SEEDS)
        self.assertEqual(len(set(study.TRAIN_SEEDS.values())), 10)
        self.assertEqual(study.SCREEN*18+(study.FINAL-study.SCREEN)*3, 12000)
        self.assertEqual(study.BASE.rounds, 1500)

    def test_selection_is_validation_only_and_ties_charge_total_bits(self):
        rows = []
        for method in study.METHODS:
            rows.extend([dict(method=method, point_id=method+"_a", state="complete", validation_ce=.3,
                              unicast_total_bits=100, test_accuracy=.99),
                         dict(method=method, point_id=method+"_b", state="complete", validation_ce=.2,
                              unicast_total_bits=200, test_accuracy=.01),
                         dict(method=method, point_id=method+"_c", state="complete", validation_ce=.2,
                              unicast_total_bits=150, test_accuracy=.0),
                         dict(method=method, point_id=method+"_failed", state="numerical_failure")])
        self.assertEqual(study.choose(rows, 1), {m: [m+"_c"] for m in study.METHODS})
        with self.assertRaises(RuntimeError):
            study.choose([r for r in rows if r["method"] != "stc"], 1)

    def test_shared_view_materializes_the_same_minibatch(self):
        images = np.arange(100*784, dtype=np.float32).reshape(100, 784)
        ids = np.arange(99, -1, -2)
        view = data_tools.IndexedImages(images, ids)
        np.testing.assert_array_equal(view[np.array([1, 3, 7])], images[ids[[1, 3, 7]]])
        np.testing.assert_array_equal(view[4:10], images[ids[4:10]])

    def test_test_download_is_gated_by_frozen_validation_selection(self):
        with patch.object(study, "verify_selection", side_effect=RuntimeError("not selected")), \
             patch.object(study.data_tools, "prepare_cache") as prepare:
            with self.assertRaises(RuntimeError):
                study.confirm({}, Path("unused"), 2, Path("unused"))
            prepare.assert_not_called()

    def test_readonly_audit_is_portable_but_training_runtime_is_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            science = {"protocol": study.protocol_spec(), "source_hashes": study.source_hashes(),
                       "runtime": {"different": "training machine"}}
            data_tools.atomic_json(Path(folder)/"protocol.json", science)
            with self.assertRaises(RuntimeError):
                study.load_contract(folder)
            self.assertEqual(study.load_contract(folder, check_runtime=False), science)


class OperatorAndResumeTests(unittest.TestCase):
    def test_downlink_preselection_matches_every_native_byte(self):
        rng = np.random.default_rng(58)
        modes = set()
        for n in (1, 3, 10):
            for d in (1, 19, 89, 4096):
                for density in (0., .05, .8, 1.):
                    pulses = rng.choice([-1, 0, 1], (n, d)).astype(np.int8)
                    pulses[rng.random(pulses.shape) >= density] = 0
                    expected, actual = native.encode_event_downlink(pulses, .1), host.encode_downlink(pulses, .1)
                    self.assertEqual(actual.data, expected.data)
                    self.assertEqual(actual.mode, expected.mode)
                    self.assertEqual(actual.counts, expected.counts)
                    self.assertEqual(actual.ordered_bits, expected.ordered_bits)
                    self.assertEqual(actual.aggregate_exact_bits, expected.aggregate_exact_bits)
                    self.assertEqual(actual.correction_nonzeros, expected.correction_nonzeros)
                    modes.add(actual.mode)
        self.assertEqual(modes, {"counts_patch", "ordered"})
        self.assertIsNot(host.wire.encode_pulses, native.encode_pulses)
        self.assertEqual(native.encode_pulses.__module__, "neuromorphicfl.event_codec")

    def test_all_five_models_and_residuals_match_the_frozen_native_runner(self):
        data, fed = synthetic()
        with tempfile.TemporaryDirectory() as folder:
            for method in study.METHODS:
                with self.subTest(method=method):
                    config = tiny_config(method)
                    result = host.run(data, method=method, config=config, seed=73,
                                      checkpoint_path=Path(folder)/f"new_{method}.npz", provenance={"test": True})
                    path = Path(folder)/f"reference_{method}.npz"
                    run_final_baseline(federation=fed, architecture="mlp", method=method, config=config, seed=73,
                                       checkpoint=BaselineCheckpoint(path, {"test": True}), checkpoint_stride=1)
                    with np.load(path, allow_pickle=False) as native_state:
                        np.testing.assert_array_equal(result["arrays"]["model"].view(np.uint32), native_state["model"].view(np.uint32))
                        if method in ("event", "strom"):
                            name = "event_state" if method == "event" else "strom_state"
                            np.testing.assert_array_equal(result["arrays"]["pulse_state"], native_state[name])
                        elif method == "ef_topk":
                            np.testing.assert_array_equal(result["arrays"]["client_residuals"], native_state["residual"])
                        elif method == "stc":
                            np.testing.assert_array_equal(result["arrays"]["client_residuals"], native_state["stc_client_residuals"])
                            np.testing.assert_array_equal(result["arrays"]["server_residual"], native_state["stc_server_residual"])

    def test_every_method_resumes_exactly_without_rng_or_metric_restarts(self):
        data, _ = synthetic()
        with tempfile.TemporaryDirectory() as folder:
            for method in study.METHODS:
                with self.subTest(method=method):
                    config = tiny_config(method)
                    kwargs = dict(data=data, method=method, config=config, seed=918, provenance={"test": True})
                    a = host.run(**kwargs, checkpoint_path=Path(folder)/f"full_{method}.npz")
                    path = Path(folder)/f"resumed_{method}.npz"
                    first = host.run(**kwargs, checkpoint_path=path, stop_after=2)
                    b = host.run(**kwargs, checkpoint_path=path)
                    for name in a["arrays"]:
                        np.testing.assert_array_equal(a["arrays"][name].view(np.uint32), b["arrays"][name].view(np.uint32))
                    self.assertEqual(a["rng_state"], b["rng_state"])
                    pd.testing.assert_frame_equal(a["evaluation"], b["evaluation"], check_exact=True)
                    cols = [c for c in a["rounds"] if not c.endswith("seconds")]
                    pd.testing.assert_frame_equal(a["rounds"][cols], b["rounds"][cols], check_exact=True)
                    pd.testing.assert_frame_equal(first["rounds"], b["rounds"].iloc[:2], check_exact=True)
                    with self.assertRaises(ValueError):
                        host.run(**{**kwargs, "provenance": {"changed": True}}, checkpoint_path=path)

    def test_single_pass_metrics_agree_with_native_metrics(self):
        data, _ = synthetic()
        w = host.initialize_mlp(seed=7777, scale=.5)
        expected = predictive_metrics(w, data["X_holdout"], data["y_holdout"])
        actual = host.metrics(w, data["X_holdout"], data["y_holdout"], regularization=.0001, batch_size=13)
        np.testing.assert_allclose([actual[k] for k in ("objective", "ce", "accuracy", "macro_accuracy", "worst_class_accuracy")],
                                   expected[:5], atol=2e-6, rtol=2e-6)

    def test_intervals_preserve_unreached_and_single_pair_cases(self):
        self.assertEqual(analysis.interval([], [], ratio=True), (None, None, None))
        value, low, high = analysis.interval([2], [4], ratio=True)
        self.assertAlmostEqual(value, .5)
        self.assertIsNone(low)
        self.assertIsNone(high)
        value, low, high = analysis.interval([1, 2, 3], [2, 4, 6], ratio=True)
        np.testing.assert_allclose([value, low, high], [.5, .5, .5])

    def test_divergence_is_reported_as_a_numerical_failure(self):
        data, _ = synthetic()
        config = replace(tiny_config("dense"), server_gain=1e38)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(host.NumericalFailure) as caught:
                host.run(data, method="dense", config=config, seed=73,
                         checkpoint_path=Path(folder)/"diverged.npz", provenance={"test": True})
            self.assertGreaterEqual(caught.exception.round, 1)


class ParallelCampaignTests(unittest.TestCase):
    def test_duplicate_coordinator_and_live_model_claim_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with study.campaign_lock(out):
                with self.assertRaises(RuntimeError):
                    with study.campaign_lock(out):
                        self.fail("Acquired an already-owned coordinator lock")
            self.assertFalse((out/".campaign.lock").exists())
            p = study.POINTS[0]
            checkpoint_folder = out/"checkpoints"
            checkpoint_folder.mkdir()
            claim = checkpoint_folder/f"{p.point_id}_development_p{study.DEV_SEED}.claim"
            claim.write_text(json.dumps({"pid": os.getpid(), "host": study.socket.gethostname()}))
            with self.assertRaises(RuntimeError):
                study.run_job((p, "development", study.DEV_SEED, 2, {}, str(out), None))

    def test_spawn_workers_preserve_outputs_continuation_and_receipt_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root/"cache"
            cache.mkdir()
            # A sparse zero-filled .npy has the real dimensions without writing
            # a full dataset; this is synthetic infrastructure verification.
            mmap = np.lib.format.open_memmap(cache/"train_images.npy", mode="w+", dtype=np.float32, shape=(60000, 784))
            mmap.flush()
            del mmap
            np.save(cache/"train_labels.npy", np.repeat(np.arange(10), 6000))
            points = [next(p for p in study.POINTS if p.method == m) for m in study.METHODS]
            configs = {p.point_id: asdict(replace(study.config_for(p), rounds=4, eval_stride=1, local_steps=1)) for p in points}
            science = {"protocol": {"configs": configs}, "synthetic_only": True,
                       "runtime": study.runtime(), "source_hashes": study.source_hashes()}
            for name, workers in (("serial", 1), ("parallel", 2)):
                out = root/name
                out.mkdir()
                jobs = [(p, "development", study.DEV_SEED, 2, science, str(out), None) for p in points]
                study.run_jobs(jobs, cache=cache, workers=workers, out=out, label="Synthetic screen")
                # Repeating completed work must skip rather than retrain.
                before = {p.name: data_tools.file_hash(p) for p in out.glob("*_receipt.json")}
                study.run_jobs(jobs, cache=cache, workers=workers, out=out, label="Synthetic skip")
                self.assertEqual(before, {p.name: data_tools.file_hash(p) for p in out.glob("*_receipt.json")})
            for p in points:
                files_a = study.stage_paths(p, "development", study.DEV_SEED, 2, root/"serial")
                files_b = study.stage_paths(p, "development", study.DEV_SEED, 2, root/"parallel")
                a, b = pd.read_csv(files_a["history"]), pd.read_csv(files_b["history"])
                pd.testing.assert_frame_equal(a, b, check_exact=True)
            out = root/"parallel"
            jobs = [(p, "development", study.DEV_SEED, 4, science, str(out), None) for p in points]
            study.run_jobs(jobs, cache=cache, workers=2, out=out, label="Synthetic continue")
            with patch.object(study, "POINTS", tuple(points)), patch.object(study, "SCREEN", 2), patch.object(study, "FINAL", 4):
                promotion = study.promotion_record(science, out)
                selected = study.selection_record(science, promotion, out)
                self.assertFalse(selected["official_test_used_for_selection"])
                self.assertFalse(selected["development_checkpoint_reused_for_confirmation"])
                self.assertEqual(set(selected["selected"]), set(study.METHODS))
            p = points[0]
            path = study.stage_paths(p, "development", study.DEV_SEED, 2, out)["summary"]
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(RuntimeError):
                study.read_receipt(p, "development", study.DEV_SEED, 2, science, out)


class SavedAnalysisTests(unittest.TestCase):
    def test_fifty_saved_outcomes_target_censoring_and_independent_accounting(self):
        """An independent two-round fixture tests reporting, not training quality."""
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            points = {m: next(p for p in study.POINTS if p.method == m) for m in study.METHODS}
            science = {"protocol": {"configs": {p.point_id: asdict(replace(study.config_for(p), rounds=2, eval_stride=1))
                                                for p in points.values()}}, "synthetic_only": True}
            selection = {"selected": {m: [p.point_id] for m, p in points.items()}}
            data_tools.atomic_json(out/"frozen_selection.json", selection)
            selection_sha = data_tools.digest(selection)
            initial = 32*host.LAYOUT.dimension+64
            for mi, (method, p) in enumerate(points.items()):
                for si, seed in enumerate(study.SEEDS):
                    final_accuracy = {"event": .87 if si < 9 else .86, "strom": .86,
                                      "dense": .85, "stc": .88, "ef_topk": .88}[method]
                    meta = study.identity(p, "confirmation", seed, 2, science, selection_sha=selection_sha)
                    round_rows, eval_rows, history_rows = [], [], []
                    upload_per_round, stream_per_round = 2048+mi*512, 1024+mi*128
                    for rnd in (1, 2):
                        sha = f"{mi*1000+si*10+rnd:064x}"
                        ev = {"round": rnd, "model_sha256": sha, "train_ce": .3, "train_objective": .31,
                              "parameter_l2_norm": 2., "test_ce": .4, "test_accuracy": .7 if rnd == 1 else final_accuracy,
                              "test_macro_accuracy": .7 if rnd == 1 else final_accuracy, "test_worst_class_accuracy": .5}
                        rr = {"round": rnd, "method": method, "model_sha256": sha, "roundtrip_verified": True,
                              "codec_uplink_bits": upload_per_round, "exact_downlink_bits": stream_per_round, **meta}
                        round_rows.append(rr)
                        eval_rows.append({**ev, **meta})
                        up, stream = rnd*upload_per_round, rnd*stream_per_round
                        down = 10*(initial+32*rnd+stream)
                        history_rows.append({**ev, "variant": host.MAIN[method], "uplink_bits": up,
                                             "server_stream_bits": stream, "unicast_downlink_bits": down,
                                             "unicast_total_bits": up+down, "broadcast_total_bits": up+initial+stream, **meta})
                    last = history_rows[-1]
                    summary = {"rounds": 2, "model_sha256": last["model_sha256"], "variant": host.MAIN[method], **meta}
                    for name, value in last.items():
                        if name.startswith(("train_", "test_")) and name != "train_seed":
                            summary["final_"+name] = value
                        elif name.endswith("_bits") or name == "parameter_l2_norm":
                            summary[name] = value
                    tables = {"rounds": pd.DataFrame(round_rows), "evaluation": pd.DataFrame(eval_rows),
                              "history": pd.DataFrame(history_rows), "summary": pd.DataFrame([summary])}
                    paths = study.stage_paths(p, "confirmation", seed, 2, out)
                    for key, frame in tables.items():
                        study.atomic_csv(paths[key], frame)
                    data_tools.atomic_json(paths["receipt"], {"identity": meta, "state": "complete",
                                           "sha256": {k: data_tools.file_hash(paths[k]) for k in tables}})
            with patch.object(study, "FINAL", 2), patch.object(study, "load_contract", return_value=science), \
                 patch.object(study, "verify_selection", return_value=selection):
                analysis.analyze(out)
                targets = pd.read_csv(out/"paired_targets.csv")
                absent = targets[(targets.reference == "strom") & (targets.target_accuracy == .87)].iloc[0]
                self.assertEqual(absent.event_reached, 9)
                self.assertEqual(absent.reference_reached, 0)
                self.assertEqual(absent.common_hits, 0)
                self.assertTrue(pd.isna(absent.estimate))
                common = targets[(targets.reference == "stc") & (targets.target_accuracy == .87)].iloc[0]
                self.assertEqual(common.common_hits, 9)
                expected_event = 2*2048+10*(initial+64+2*1024)
                expected_stc = 2*(2048+3*512)+10*(initial+64+2*(1024+3*128))
                self.assertAlmostEqual(common.estimate, expected_event/expected_stc)
                self.assertEqual(json.loads((out/"confirmation_completion.json").read_text())["finished"], 50)
                self.assertTrue((out/"comparison.png").exists())
                path = study.stage_paths(points["event"], "confirmation", study.SEEDS[0], 2, out)["receipt"]
                path.unlink()
                with self.assertRaises(RuntimeError):
                    analysis.analyze(out)
                self.assertFalse(json.loads((out/"confirmation_completion.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
