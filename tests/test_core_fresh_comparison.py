from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments import core_fresh_comparison as core
from experiments.stc_stage1 import smoke_federation
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import run_final_baseline


def fake_science():
    old = json.loads((core.previous.OUT / "protocol.json").read_text())
    return {"protocol": {"rounds": 2, "evaluation_rounds": [1, 2], "dimension": 4, "n_clients": 2,
                         "intervals": "synthetic"}, "reference_contract_sha256": core.digest(old),
            "references": {core.reference_key(p, i): {} for i in range(10) for p in core.REUSED}}


def fake_result(point, index=0):
    rounds = pd.DataFrame({"round": [1, 2], "method": point, "legacy_uplink_bits": [160, 160],
                           "codec_uplink_bits": [80, 80], "legacy_server_bits": [160, 160],
                           "codec_replay_bits": [80, 80], "exact_downlink_bits": [64, 64],
                           "model_sha256": [f"{10*index+r:064x}" for r in (1, 2)]})
    evaluation = pd.DataFrame({"round": [1, 2], "train_objective": [1.2, 1.1], "train_ce": [1.1, 1.0],
                               "test_ce": [1.2, 1.1], "test_accuracy": [.46, .56], "macro_accuracy": [.46, .56],
                               "worst_class_accuracy": [.2, .3], "model_sha256": rounds.model_sha256})
    return {"rounds": rounds, "evaluation": evaluation}


def paired_runs():
    rows = []
    for i, seed in enumerate(core.SEEDS):
        for point in core.POINTS:
            multiplier = {"event": 2, "strom": 1, "dense": 3, "stc": 4, "ef_topk": 5}[point]
            rows.append({"point": point, "partition_seed": seed, "final_test_accuracy": .6 + .001*i + (.01 if point != "event" else 0),
                         "final_test_ce": 1.1, "final_train_objective": 1.2, "final_train_l2_penalty": .1,
                         **{f: multiplier * (1000+100*i) for f in
                            ("uplink_bits", "unicast_downlink_bits", "unicast_total_bits", "broadcast_total_bits")}})
    return pd.DataFrame(rows)


class CodecTests(unittest.TestCase):
    def test_float32_bytes_sparse_dense_ties_signed_zero_and_subnormal(self):
        values = [np.zeros(64, dtype=np.float32), np.ones(64, dtype=np.float32),
                  np.array([0, -0., 1e-45, -1e-45] + [0]*60, dtype=np.float32),
                  np.arange(1, 65, dtype=np.float32)]
        for vector in values:
            for dense in (False, True):
                data, _ = core.host.encode_vector(vector, dense_only=dense)
                actual = core.host.decode_vector(data, expected_dimension=vector.size)
                np.testing.assert_array_equal(actual.view(np.uint32), vector.view(np.uint32))
        sparse, mode = core.host.encode_vector(values[2])
        self.assertTrue(mode.startswith("support_"))
        self.assertLess(len(sparse), 8 + 4*64)
        dense, mode = core.host.encode_vector(values[-1])
        self.assertEqual((mode, len(dense)), ("dense_float32", 8+4*64))
        for bad in (np.array([np.nan], dtype=np.float32), np.array([1.], dtype=np.float64), np.empty(0, dtype=np.float32)):
            with self.assertRaises(ValueError):
                core.host.encode_vector(bad)

    def test_bad_dimensions_metadata_truncation_trailing_bytes_and_nonfinite_payload(self):
        vector = np.zeros(64, dtype=np.float32)
        vector[3] = 1
        sparse, _ = core.host.encode_vector(vector)
        dense, _ = core.host.encode_vector(vector, dense_only=True)
        for data in (b"", sparse[:-1], sparse+b"x", dense[:-1], dense+b"x", b"BAD!"+dense[4:],
                     core.host.SPARSE.pack(b"F32S", 64, 100000)+sparse[12:],
                     core.host.DENSE.pack(b"F32D", 1)+np.array([np.inf], dtype="<f4").tobytes()):
            with self.assertRaises(ValueError):
                core.host.decode_vector(data, expected_dimension=64)
        with self.assertRaisesRegex(ValueError, "dimension"):
            core.host.decode_vector(dense, expected_dimension=63)

    def test_aggregate_and_ordered_downlink_apply_gain_once_and_charge_metadata(self):
        vectors = [np.array([0, 1, -2, 0], dtype=np.float32), np.array([0, -1, 3, 0], dtype=np.float32)]
        packets = [core.host.encode_vector(v)[0] for v in vectors]
        aggregate = np.zeros(4, dtype=np.float32)
        for v in vectors:
            aggregate += v
        aggregate *= 2.3
        data, decoded, mode, aggregate_bits, replay_bits = core.host.downlink(aggregate, packets, gain=2.3)
        np.testing.assert_array_equal(decoded.view(np.uint32), aggregate.view(np.uint32))
        self.assertEqual(len(data)*8, min(aggregate_bits, replay_bits))
        replay = core.host.SPARSE.pack(b"F32R", 4, 2) + b"".join(struct.pack(">I", len(p))+p for p in packets)
        decoded = core.host.decode_downlink(replay, expected_dimension=4, expected_clients=2, gain=2.3)
        np.testing.assert_array_equal(decoded.view(np.uint32), aggregate.view(np.uint32))
        self.assertEqual(replay_bits, 8*(12+8+sum(map(len, packets))))
        for bad in (replay[:-1], replay+b"x"):
            with self.assertRaises(ValueError):
                core.host.decode_downlink(bad, expected_dimension=4, expected_clients=2, gain=2.3)
        with self.assertRaises(ValueError):
            core.host.decode_downlink(replay, expected_dimension=4, expected_clients=3, gain=2.3)


class HostTests(unittest.TestCase):
    def test_all_three_match_native_models_residuals_rng_metrics_and_native_cost(self):
        fed = smoke_federation()
        for point in core.NEW:
            config = replace(core.config_for(point), rounds=3, eval_stride=1, local_steps=1, batch_size=2)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"native.npz"
                native = run_final_baseline(federation=fed, architecture="cifar_cnn", method=point, config=config,
                                            seed=core.train_seed(0), record_history=True,
                                            checkpoint=BaselineCheckpoint(path, {}), checkpoint_stride=1)
                fn = core.full.evaluated_run if point == "stc" else core.host.evaluated_run
                actual_path = Path(folder)/"codec.npz"
                actual = fn(federation=fed, architecture="cifar_cnn", method=point, config=config,
                            seed=core.train_seed(0), checkpoint_path=actual_path, provenance={})
                with np.load(path, allow_pickle=False) as a, np.load(actual_path, allow_pickle=False) as b:
                    for name in ("model", "stc_client_residuals", "stc_server_residual") if point == "stc" else ("model", "residual"):
                        np.testing.assert_array_equal(a[name].view(np.uint32), b[name].view(np.uint32))
                    self.assertEqual(json.loads(str(a["metadata"].item()))["progress"]["rng_state"], actual["rng_state"])
                for field in core.previous.METRIC_FIELDS:
                    np.testing.assert_array_equal(native["history"][field], actual["evaluation"][field])
                legacy = {"dense": "dense_float32", "stc": "native_stc", "ef_topk": "native_ef_topk"}[point]
                self.assertEqual(actual["summary"].set_index("variant").loc[legacy, "unicast_total_bits"], native["unicast_hybrid_total_bits"])

    def test_changed_checkpoint_config_is_rejected(self):
        config = replace(core.config_for("ef_topk"), rounds=2, eval_stride=1, local_steps=1, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            kwargs = dict(federation=smoke_federation(), architecture="cifar_cnn", method="ef_topk", seed=86000,
                          checkpoint_path=Path(folder)/"run.npz", provenance={})
            core.host.evaluated_run(**kwargs, config=config, stop_after=1)
            with self.assertRaises(ValueError):
                core.host.evaluated_run(**kwargs, config=replace(config, topk_fraction=.1))


class ProtocolTests(unittest.TestCase):
    def test_frozen_configs_seed_mapping_and_only_thirty_new_runs(self):
        self.assertEqual(core.SEEDS, tuple(range(6000, 7000, 100)))
        self.assertEqual([core.train_seed(i) for i in range(10)], list(range(86000, 87000, 100)))
        dense, stc, ef = (core.config_for(p) for p in core.NEW)
        self.assertEqual((dense.server_gain, dense.server_gain_exponent, dense.server_gain_scale), (4, .3, 100))
        self.assertEqual((stc.stc_upload_fraction, stc.stc_download_fraction, stc.server_gain), (1, 1, 2))
        self.assertEqual((ef.topk_fraction, ef.server_gain), (.05, 1))
        for point in core.REUSED:
            self.assertEqual(asdict(core.config_for(point)), asdict(core.previous.config_for(point)))
        protocol = core.protocol_spec()
        self.assertEqual((protocol["new_training_runs"], protocol["reused_training_results"]), (30, 20))
        self.assertFalse(protocol["equal_tuning_budget_claim"])

    def test_real_frozen_sources_selections_and_twenty_references_without_image_load(self):
        with patch.object(core.full, "download_cifar10", side_effect=AssertionError("no images")), \
                patch.object(core.full, "make_cifar10_federation", side_effect=AssertionError("no images")):
            science = core.build_science()
        self.assertEqual(len(science["references"]), 20)
        self.assertTrue(all(len(files)==5 for files in science["references"].values()))
        self.assertIn("experiments/core_float_host.py", science["source_hashes"])
        with patch.object(core, "build_science", return_value={"runtime": {"wrong": True}}), \
                patch.object(core.full, "download_cifar10") as download:
            with self.assertRaisesRegex(RuntimeError, "recorded Windows"):
                core.build_contract()
            download.assert_not_called()

    def test_queue_contains_only_three_new_baselines_and_reuses_both_pulse_controls(self):
        jobs, science = [], fake_science()
        worker_fn = core.run_point
        with tempfile.TemporaryDirectory() as folder, patch.object(core, "build_contract", return_value=science), \
                patch.object(core, "run_point", side_effect=lambda p,i,s,out: jobs.append((p,i)) or "synthetic"), redirect_stdout(io.StringIO()):
            out = Path(folder)
            core.run(1, out=out)
            self.assertEqual(jobs, [(p,i) for i in range(10) for p in core.NEW])
            self.assertEqual(len(list(out.glob("*_receipt.json"))), 20)
            for point in core.REUSED:
                self.assertEqual(core.read_receipt(point, 0, science, out)["state"], "reused")
            for point in core.REUSED:
                with self.assertRaises(ValueError):
                    worker_fn(point, 0, science, out)

    def test_immutable_protocol_and_subset_queue(self):
        science, jobs = fake_science(), []
        with tempfile.TemporaryDirectory() as folder, patch.object(core, "build_contract", return_value=science), \
                patch.object(core, "run_point", side_effect=lambda p,i,s,out: jobs.append((p,i)) or "synthetic") as worker, redirect_stdout(io.StringIO()):
            out = Path(folder)
            core.run(1, point="stc", seed_index=9, out=out)
            self.assertEqual(jobs, [("stc",9)])
            core.atomic_json({"different": True}, out/"protocol.json")
            worker.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "Existing frozen decision differs"):
                core.run(1, out=out)
            worker.assert_not_called()


class EvidenceTests(unittest.TestCase):
    def test_new_tables_roundtrip_crlf_and_relabelled_metadata_rejection(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(core, "ROUNDS", 2):
            out = Path(folder)
            for point in core.NEW:
                core.save_result(point, 0, science, fake_result(point), out)
                core.verified_tables(point, 0, science, out)
                files = core.paths(point, 0, out)
                files["evaluation"].write_bytes(files["evaluation"].read_bytes().replace(b"\n", b"\r\n"))
                core.read_receipt(point, 0, science, out)
                frame = pd.read_csv(files["evaluation"])
                frame["partition_seed"] = 3500
                core.atomic_csv(frame, files["evaluation"])
                receipt = json.loads(files["receipt"].read_text())
                receipt["sha256"]["evaluation"] = core.text_hash(files["evaluation"])
                core.atomic_json(receipt, files["receipt"])
                with self.assertRaisesRegex(RuntimeError, "metadata"):
                    core.verified_tables(point, 0, science, out)

    def test_changed_reused_evidence_and_reused_new_baseline_are_rejected(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            original = out/"source.csv"
            original.write_text("a\n1\n")
            science["references"]["event_s0"] = {"summary": {"path": str(original), "sha256": core.text_hash(original)}}
            core.register_reuse(science, out)
            original.write_text("a\n2\n")
            with self.assertRaisesRegex(RuntimeError, "Reused pulse evidence changed"):
                core.read_receipt("event", 0, science, out)
            core.atomic_json({"identity": core.metadata("stc", 0, science), "state": "reused"}, core.paths("stc",0,out)["receipt"])
            with self.assertRaisesRegex(RuntimeError, "cannot reuse"):
                core.read_receipt("stc", 0, science, out)

    def test_three_workers_use_matched_seeds_and_complete_runs_are_skipped(self):
        science = fake_science()
        science["protocol"]["dimension"] = 20570
        configs = {p: replace(core.config_for(p), rounds=2, eval_stride=1, local_steps=1, batch_size=2) for p in core.NEW}
        with tempfile.TemporaryDirectory() as folder, patch.object(core, "ROUNDS", 2), \
                patch.object(core, "config_for", side_effect=configs.__getitem__), patch.object(core, "build_contract", return_value=science), \
                patch.object(core.full, "make_cifar10_federation", return_value=smoke_federation()) as make, redirect_stdout(io.StringIO()):
            out = Path(folder)
            for point, index in (("dense",0), ("stc",9), ("ef_topk",0)):
                self.assertIn("done", core.run_point(point,index,science,out))
                self.assertEqual((make.call_args.kwargs["seed"], make.call_args.kwargs["n_clients"]), (core.SEEDS[index],10))
                saved = core.verified_tables(point,index,science,out)
                self.assertTrue(saved["evaluation"].train_seed.eq(core.train_seed(index)).all())
                module = core.full if point == "stc" else core.host
                with patch.object(module,"evaluated_run",side_effect=AssertionError("no retraining")):
                    self.assertIn("complete", core.run_point(point,index,science,out))

    def test_numerical_failure_remains_visible_without_auto_retry(self):
        science = fake_science()
        with tempfile.TemporaryDirectory() as folder, patch.object(core,"build_contract",return_value=science), \
                patch.object(core.full,"make_cifar10_federation",return_value=smoke_federation()), \
                patch.object(core.host,"evaluated_run",side_effect=FloatingPointError("synthetic")) as worker:
            out = Path(folder)
            self.assertIn("numerical failure",core.run_point("ef_topk",9,science,out))
            self.assertIn("numerical_failure",core.run_point("ef_topk",9,science,out))
            self.assertEqual(worker.call_count,1)


class StatisticsTests(unittest.TestCase):
    def test_four_paired_contrasts_keep_all_ten_and_unfavourable_event_results(self):
        pairs, summary = core.paired_endpoints(paired_runs().sample(frac=1,random_state=3))
        self.assertEqual((len(pairs),len(summary)),(320,32))
        self.assertTrue(summary.n.eq(10).all())
        self.assertTrue(np.allclose(summary[summary.metric=="test_accuracy_difference_pp"].estimate,-1))
        row = summary[(summary.reference=="strom") & (summary.metric=="unicast_traffic_ratio")].iloc[0]
        self.assertAlmostEqual(row.estimate,2)
        for bad in (paired_runs().iloc[1:],pd.concat([paired_runs(),paired_runs().iloc[:1]])):
            with self.assertRaisesRegex(ValueError,"Missing or duplicate"):
                core.paired_endpoints(bad)

    def test_target_censoring_retains_full_denominators_including_zero_common_hits(self):
        rows=[]
        for p in core.POINTS:
            for i,seed in enumerate(core.SEEDS):
                for target in core.TARGETS:
                    hit=not (p=="stc" and i==9 and target==.55) and not (p=="dense" and target==.45)
                    rows.append({"point":p,"partition_seed":seed,"target_accuracy":target,"reached":hit,
                                 "unicast_total_bits":(1000 if p=="event" else 500) if hit else np.nan,
                                 "first_sampled_round":(120 if p=="event" else 180) if hit else np.nan})
        summary=core.paired_targets(pd.DataFrame(rows))
        self.assertEqual(len(summary),12)
        row=summary[(summary.reference=="stc") & (summary.target_accuracy==.55)].iloc[0]
        self.assertEqual((row.n_total,row.n_candidate_reached,row.n_reference_reached,row.n_common_hits),(10,10,9,9))
        self.assertAlmostEqual(row.estimate,2)
        row=summary[(summary.reference=="dense") & (summary.target_accuracy==.45)].iloc[0]
        self.assertEqual((row.n_total,row.n_reference_reached,row.n_common_hits),(10,0,0))
        self.assertTrue(np.isnan(row.estimate))


class PipelineTests(unittest.TestCase):
    def test_complete_fifty_result_grid_and_missing_failure_blocks_without_exclusion(self):
        science=fake_science()
        def reused(p,i,old):
            return core.previous.trajectory_tables(p,fake_result(p,i),science)
        with tempfile.TemporaryDirectory() as folder, patch.object(core,"ROUNDS",2), \
                patch.object(core,"build_science",return_value=science),patch.object(core,"plot"), \
                patch.object(core.previous,"verified_tables",side_effect=reused),redirect_stdout(io.StringIO()):
            out=Path(folder)
            core.atomic_json(science,out/"protocol.json")
            core.register_reuse(science,out)
            for i in range(10):
                for p in core.NEW:
                    core.save_result(p,i,science,fake_result(p,i),out)
            core.analyze(out)
            for name,count in (("comparison.csv",130),("comparison_history.csv",260),("main_runs.csv",50),
                               ("paired_per_seed.csv",320),("paired_summary.csv",32),("traffic_to_target.csv",150),
                               ("paired_target_summary.csv",12)):
                self.assertEqual(len(pd.read_csv(out/name)),count)
            file=core.paths("stc",9,out)["receipt"]
            value=json.loads(file.read_text());value.update(state="numerical_failure",error="synthetic")
            core.atomic_json(value,file)
            with self.assertRaisesRegex(RuntimeError,"no seed is dropped"):
                core.analyze(out)
            self.assertEqual(len(json.loads((out/"completion_audit.json").read_text())["results"]),50)
            file.unlink()
            with self.assertRaisesRegex(RuntimeError,"no seed is dropped"):
                core.analyze(out)


if __name__ == "__main__":
    unittest.main()
