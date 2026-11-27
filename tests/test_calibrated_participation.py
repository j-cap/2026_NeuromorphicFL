"""Independent native references, exact replay/resume and campaign invariants."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import io
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"experiments"), str(ROOT/"src")]
import calibrated_participation as study
import calibrated_participation_data as data
import calibrated_participation_host as host
import calibrated_participation_analysis as analysis
import core_float_host as floating
import strom_codec_host as strom
import numpy as np
import pandas as pd
from neuromorphicfl import event_codec_pilot as native
from neuromorphicfl.baseline_checkpoint import BaselineCheckpoint
from neuromorphicfl.final_baseline_campaign import FinalBaselineConfig
from neuromorphicfl.stc import STCState
from neuromorphicfl.event_codec_pilot import _same_bits


class Layout:
    def __init__(self, dimension=32):
        self.dimension = dimension

    def groups(self):
        half = self.dimension//2
        return [("a", slice(0, half), half),
                ("b", slice(half, self.dimension), self.dimension-half)]


def fake_ops(dimension=32):
    layout = Layout(dimension)
    def initialize(*, layout, seed, scale):
        return np.linspace(-.02, .02, layout.dimension, dtype=np.float32)*scale
    def loss(w, X, y, *, layout, regularization, need_gradient):
        gradient = (.03*w-.4*float(np.mean(X))).astype(np.float32)
        return float(w@w)+1, float(w@w)+.9, gradient
    def metrics(w, X, y, *, layout, regularization):
        ce = 1.+float(w@w)
        return ce+.5*regularization*float(w@w), ce, .5, .5, .4, np.full(10, .5)
    return layout, initialize, loss, metrics


def federation(n=4):
    return SimpleNamespace(n_clients=n,
        client_X=tuple(np.array([[.1+i*.03], [.2-i*.01]], dtype=np.float32) for i in range(n)),
        client_y=tuple(np.array([0, 1]) for _ in range(n)),
        X_train_eval=np.array([[.1]], dtype=np.float32), y_train_eval=np.array([0]),
        X_test=np.array([[.2]], dtype=np.float32), y_test=np.array([1]), weights=np.full(n, 1/n))


def config(**kwargs):
    return replace(FinalBaselineConfig(rounds=7, eval_stride=3, local_steps=2, batch_size=3,
        local_lr=.1, threshold=.005, jump0=.0007, jump_exponent=.2, rho=.9,
        strom_threshold=.004, topk_fraction=.25, stc_upload_fraction=.25,
        stc_download_fraction=.5), **kwargs)


def state(n=4, d=32):
    return {"client_state": np.zeros((n, d), dtype=np.float32),
        "server_residual": np.zeros(d, dtype=np.float32), "last_state_round": np.zeros(n, dtype=np.int64)}


def synthetic_worker(payload):
    """Exercise Windows-compatible spawn and the real save/receipt workflow."""
    cache, job, science, out = payload
    with patch.object(data, "PILOT_ROUNDS", 4), patch.object(host, "_architecture_ops", return_value=fake_ops(20570)):
        study.worker_init(cache, False)
        return study.run_job((job, science, out))


class NativeEquivalenceTests(unittest.TestCase):
    def test_full_participation_matches_all_five_current_hosts_and_total_bits(self):
        fed, cfg = federation(3), config()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            ops = fake_ops()
            for module in (host, native, floating, strom):
                stack.enter_context(patch.object(module, "_architecture_ops", return_value=ops))
            for method in study.METHODS:
                with self.subTest(method=method):
                    tuned = (replace(cfg,server_gain=4.,server_gain_exponent=.3) if method == "dense"
                             else replace(cfg,server_gain=2.) if method == "stc"
                             else replace(cfg,server_gain=.1) if method == "strom" else cfg)
                    kwargs = dict(federation=fed, architecture="synthetic", method=method, config=tuned, seed=17)
                    if method in ("event", "stc"):
                        reference = native.run_codec_pilot(**kwargs)
                    elif method == "strom":
                        reference = strom.run_strom_codec(**{k:v for k,v in kwargs.items() if k != "method"})
                    else:
                        reference = floating.run(**kwargs, checkpoint=BaselineCheckpoint(Path(folder)/method, {}))
                    result = host.run(**kwargs, schedule=np.tile(np.arange(3), (7, 1)), horizon=7, scope="test")
                    _same_bits(reference["model"], result["arrays"]["model"])
                    self.assertEqual(reference["rounds"].model_sha256.tolist(), result["rounds"].model_sha256.tolist())
                    for name in ("codec_uplink_bits", "exact_downlink_bits"):
                        np.testing.assert_array_equal(reference["rounds"][name], result["rounds"][name])
                    expected = (3*(32*32+64)+3*32*7+
                        int(reference["rounds"].codec_uplink_bits.sum())+
                        3*int(reference["rounds"].exact_downlink_bits.sum()))
                    self.assertEqual(int(result["history"].iloc[-1].unicast_total_bits), expected)
                    self.assertEqual(int(result["history"].iloc[-1].terminal_catchup_bits), 0)

    def test_stc_native_weight_placement_both_residuals_and_inactive_client(self):
        cfg, s = config(server_gain=2.), state()
        s["client_state"][:] = np.linspace(-.13, .17, 128, dtype=np.float32).reshape(4,32)
        s["server_residual"][:] = .01
        before = s["client_state"].copy()
        active = np.array([1, 3])
        deltas = [np.linspace(-.09, .08, 32, dtype=np.float32), np.linspace(.1, -.04, 32, dtype=np.float32)]
        expected = STCState((16,16), .25, .5, before[active].copy(), s["server_residual"].copy())
        reference = expected.step(deltas, np.array([.5,.5]), server_gain=2.)
        frame, _ = host.step(s, deltas, active, np.full(4,.5), method="stc", config=cfg, sizes=(16,16), rnd=1)
        _same_bits(frame.update, reference.server_update)
        _same_bits(s["client_state"][active], expected.client_residuals)
        _same_bits(s["server_residual"], expected.server_residual)
        _same_bits(s["client_state"][[0,2]], before[[0,2]])

    def test_strom_subtractive_residual_and_inactive_state(self):
        cfg, s = config(), state()
        s["client_state"][2] = .012
        before = s["client_state"].copy()
        active, delta = np.array([1,3]), [np.full(32,.003,dtype=np.float32), np.full(32,-.002,dtype=np.float32)]
        expected = before[active].copy()
        pulses = strom.residual_pulses(expected, delta, np.array([.5,.5]), local_lr=.1, threshold=.004)
        frame, _ = host.step(s, delta, active, np.full(4,.5), method="strom", config=cfg, sizes=(16,16), rnd=3)
        _same_bits(s["client_state"][active], expected)
        _same_bits(s["client_state"][[0,2]], before[[0,2]])
        ref = host.pulse_wire.native_wire.pulse_aggregate(pulses, cfg.strom_threshold)
        _same_bits(frame.update, ref)

    def test_event_global_round_leak_and_full_reset(self):
        cfg, s = config(), state()
        s["client_state"][1] = .004
        s["client_state"][0] = .011
        s["last_state_round"][1] = 1
        before = s["client_state"].copy()
        host.step(s, [np.zeros(32,dtype=np.float32)], np.array([1]), np.ones(4),
                  method="event", config=cfg, sizes=(16,16), rnd=4)
        _same_bits(s["client_state"][1], (before[1]*(.9**3)).astype(np.float32))
        _same_bits(s["client_state"][0], before[0])
        s["client_state"][1] = .017
        host.step(s, [np.zeros(32,dtype=np.float32)], np.array([1]), np.ones(4),
                  method="event", config=cfg, sizes=(16,16), rnd=5)
        np.testing.assert_array_equal(s["client_state"][1], 0)

    def test_native_cifar_sgd_and_mmap_style_image_view_smoke(self):
        rng = np.random.default_rng(4)
        images = rng.integers(0,256,(12,3,32,32),dtype=np.uint8)
        fed = SimpleNamespace(n_clients=3, weights=np.full(3,1/3),
            client_X=tuple(data.IndexedImages(images,np.arange(i*4,(i+1)*4)) for i in range(3)),
            client_y=tuple(np.arange(4) for _ in range(3)), X_train_eval=images[:4],
            y_train_eval=np.arange(4), X_test=images[4:8], y_test=np.arange(4))
        result = host.run(federation=fed, architecture="cifar_cnn", method="event",
            config=config(rounds=2, local_steps=1, batch_size=2), schedule=np.array([[0,1],[1,2]]),
            seed=12, horizon=2, scope="validation")
        self.assertEqual(result["arrays"]["model"].size,20570)
        self.assertTrue(result["rounds"].reconstruction_verified.all())
        self.assertTrue("validation_ce" in result["history"])
        self.assertFalse(any(c.startswith("test_") for c in result["history"]))


class ReplayResumeTests(unittest.TestCase):
    def test_exact_ordered_replay_cannot_be_replaced_by_summing_updates(self):
        d = 128
        journal = host.Journal(method="ef_topk", dimension=d, sizes=(64,64))
        initial = np.zeros(d, dtype=np.float32); initial[0] = 1e20
        server = initial.copy()
        for rnd, value in enumerate((-1e20, 3.),1):
            delta = np.zeros(d,dtype=np.float32);delta[0] = value
            wire, _ = host.wire.encode_vector(delta)
            frame = host.decoded_frame(rnd=rnd,data=wire,method="ef_topk",dimension=d,sizes=(64,64))
            journal.append(frame);server += frame.update
        client = initial.copy()
        bits, mode = journal.synchronize(client,0,server)
        self.assertEqual(mode,"replay");self.assertEqual(bits,journal.prefix[-1])
        self.assertEqual(float(client[0]),3.)
        self.assertNotEqual(float((initial+sum((f.update for f in journal.frames.values()),np.zeros(d,dtype=np.float32)))[0]),3.)

    def test_checkpoint_fallback_pruning_and_dump_restore(self):
        d = 32;journal = host.Journal(method="dense",dimension=d,sizes=(16,16))
        initial = np.zeros(d,dtype=np.float32);server=initial.copy()
        for rnd in range(1,4):
            delta=np.full(d,rnd/10,dtype=np.float32)
            wire,_=host.wire.encode_vector(delta,dense_only=True)
            frame=host.decoded_frame(rnd=rnd,data=wire,method="dense",dimension=d,sizes=(16,16))
            journal.append(frame);server+=frame.update
        self.assertEqual(list(journal.frames),[3])
        client=initial.copy();bits,mode=journal.synchronize(client,0,server)
        self.assertEqual((bits,mode),(32*d+64,"checkpoint"))
        raw,meta=journal.dump();restored=host.Journal(method="dense",dimension=d,sizes=(16,16))
        restored.restore(raw,meta);_same_bits(restored.frames[3].update,journal.frames[3].update)
        corrupted=raw.copy();corrupted[-1]^=1
        with self.assertRaises(ValueError):restored.restore(corrupted,meta)

    def test_partial_resume_is_identical_for_all_five_methods(self):
        schedule=np.array([[0,1],[2,3],[0,2],[1,3],[0,3],[1,2]])
        with tempfile.TemporaryDirectory() as folder,patch.object(host,"_architecture_ops",return_value=fake_ops()):
            for method in study.METHODS:
                with self.subTest(method=method):
                    kwargs=dict(federation=federation(),architecture="synthetic",method=method,
                        config=config(rounds=6,eval_stride=2),schedule=schedule,seed=3,horizon=6,scope="test")
                    one=host.run(**kwargs)
                    path=Path(folder)/(method+".npz")
                    host.run(**kwargs,checkpoint_path=path,stop_after=3)
                    resumed=host.run(**kwargs,checkpoint_path=path)
                    for name in one["arrays"]:np.testing.assert_array_equal(one["arrays"][name],resumed["arrays"][name])
                    pd.testing.assert_frame_equal(one["rounds"],resumed["rounds"])
                    pd.testing.assert_frame_equal(one["history"],resumed["history"])
                    self.assertEqual(one["rng_state"],resumed["rng_state"])
                    self.assertEqual(int(resumed["history"].iloc[-1].terminal_verified_clients),4)
                    self.assertGreater(int(resumed["history"].iloc[-1].terminal_catchup_bits),0)

    def test_invalid_schedule_and_nonfinite_configuration_rejected(self):
        kwargs=dict(federation=federation(),architecture="synthetic",method="event",
            config=config(rounds=2),schedule=np.array([[0,0],[1,2]]),seed=1,horizon=2,scope="test")
        with patch.object(host,"_architecture_ops",return_value=fake_ops()):
            with self.assertRaises(ValueError):host.run(**kwargs)
            kwargs["schedule"]=np.array([[0,1],[1,2]])
            kwargs["config"]=config(rounds=2,rho=float("nan"))
            with self.assertRaises(ValueError):host.run(**kwargs)

    def test_nonfinite_local_update_is_an_explicit_round_failure(self):
        with patch.object(host,"_architecture_ops",return_value=fake_ops()),\
             patch.object(host,"_local_delta",return_value=np.full(32,np.nan,dtype=np.float32)):
            with self.assertRaises(host.NumericalFailure) as failure:
                host.run(federation=federation(),architecture="synthetic",method="event",config=config(),
                    schedule=np.tile(np.array([0,1]),(7,1)),seed=1,horizon=7,scope="test")
        self.assertEqual(failure.exception.round,1)


class CampaignTests(unittest.TestCase):
    def test_spawn_workers_save_identical_auditable_outputs(self):
        job = ("pilot",5,"dense",data.DEV_SEED)
        with tempfile.TemporaryDirectory() as folder,patch.object(data,"PILOT_ROUNDS",4):
            root=Path(folder);cache=root/"cache";cache.mkdir()
            np.save(cache/"train_images.npy",np.full((50000,1),.1,dtype=np.float32))
            np.save(cache/"train_labels.npy",np.repeat(np.arange(10),5000))
            science={"protocol":study.specification(),"source_hashes":study.source_hashes(),
                "runtime":data.runtime(),"training_data":{"synthetic_verification":True}}
            payloads=[]
            for suffix in ("a","b"):
                out=root/suffix;out.mkdir()
                data.atomic_json(out/"protocol.json",science)
                _,_,_,manifest=data.allocation(np.repeat(np.arange(10),5000),seed=data.DEV_SEED,validation=True)
                data.atomic_json(out/f"partition_p{data.DEV_SEED}.json",manifest)
                data.atomic_json(out/f"schedule_p{data.DEV_SEED}.json",data.schedule_manifest(data.DEV_SEED))
                payloads.append((str(cache),job,science,str(out)))
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
                results=list(pool.map(synthetic_worker,payloads))
            self.assertTrue(all("complete" in r for r in results))
            for kind in ("rounds","history","summary"):
                a=study.paths(job,root/"a")[kind];b=study.paths(job,root/"b")[kind]
                self.assertEqual(a.read_bytes(),b.read_bytes())
            tables=study.read_tables(job,science,root/"a")
            self.assertEqual(int(tables["summary"].iloc[0].rounds),4)
            with redirect_stdout(io.StringIO()):
                completion=analysis.analyze("pilot",root/"a")
            self.assertEqual(completion["complete"],1)
            self.assertEqual(completion["pending"],14)
            self.assertTrue((root/"a/pilot_comparison.pdf").exists())

    def test_cached_pilot_does_not_read_or_parse_official_test_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=root/"participation_calibrated_cache_v1";cache.mkdir()
            expected=json.loads(data.CORE_PROTOCOL.read_text())["data_sha256"]
            manifest={"raw_sha256":{f"data_batch_{i}":expected[f"data_batch_{i}"] for i in range(1,6)},
                      "cache_sha256":{},"scope":"train"}
            data.atomic_json(cache/"train_manifest.json",manifest)
            touched=[]
            def source_hash(path):
                touched.append(Path(path).name)
                self.assertNotEqual(Path(path).name,"test_batch")
                return expected[Path(path).name]
            with patch.object(data,"download_cifar10",return_value=root),\
                 patch.object(data,"file_hash",side_effect=source_hash),\
                 patch.object(data,"_read_batch",side_effect=AssertionError("Unexpected image parsing")):
                _,manifests=data.prepare_cache(root,include_test=False)
            self.assertEqual(set(manifests),{"train"})
            self.assertEqual(len(touched),5)

    def test_fifty_client_allocations_exact_and_validation_disjoint(self):
        labels=np.repeat(np.arange(10),5000)
        for validation,size in ((True,900),(False,1000)):
            clients,heldout,diag,manifest=data.allocation(labels,seed=13000,validation=validation)
            self.assertEqual(len(clients),50);self.assertTrue(all(len(ids)==size for ids in clients))
            np.testing.assert_array_equal(np.sort(np.r_[np.concatenate(clients),heldout]),np.arange(50000))
            counts=np.array([np.bincount(labels[ids],minlength=10) for ids in clients])
            np.testing.assert_array_equal(counts,np.array(manifest["class_counts"]))
            np.testing.assert_array_equal(counts.max(axis=1),11*size//20)
            self.assertEqual(len(diag),5000)
            self.assertFalse(np.intersect1d(diag,heldout).size)

    def test_nested_uniform_schedule_and_fresh_seed_job_grid(self):
        schedules={k:data.schedule_for(13100,k,rounds=20) for k in data.ACTIVE}
        for a,b in zip(schedules[5],schedules[10]):self.assertTrue(set(a)<=set(b))
        np.testing.assert_array_equal(schedules[10],data.schedule_for(13100,10,rounds=20))
        self.assertFalse(np.array_equal(schedules[5],data.schedule_for(13200,5,rounds=20)))
        self.assertEqual(len(study.jobs_for("pilot")),15)
        self.assertEqual(len(study.jobs_for("confirmation")),150)
        self.assertNotIn(data.DEV_SEED,data.SEEDS)
        self.assertEqual(set(study.configs()),set(study.METHODS))
        self.assertEqual(study.configs()["event"].threshold,.05)
        self.assertEqual(study.configs()["strom"].strom_threshold,.05)
        self.assertEqual(study.configs()["dense"].server_gain_exponent,.3)

    def test_one_seed_has_no_confidence_interval_and_empty_target_is_censored(self):
        self.assertEqual(analysis.paired([],[],log=True),(None,None,None))
        self.assertEqual(analysis.paired([2.],[4.],log=True),(.5,None,None))
        est,lo,hi=analysis.paired([2.]*10,[4.]*10,log=True)
        self.assertAlmostEqual(est,.5);self.assertAlmostEqual(lo,.5);self.assertAlmostEqual(hi,.5)

    def test_failed_confirmation_pair_is_not_silently_excluded(self):
        runs=[];histories=[];coverage=[]
        for job in study.jobs_for("confirmation"):
            phase,k,method,seed=job
            failed=method=="stc" and seed==data.SEEDS[0]
            coverage.append(dict(phase=phase,active_clients=k,method=method,partition_seed=seed,
                state="numerical_failure" if failed else "complete"))
            if failed:continue
            row=dict(active_clients=k,method=method,partition_seed=seed,round=1800,
                final_test_accuracy=.6,final_test_ce=1.,unicast_total_bits=100.,
                operational_total_bits=90.,uplink_bits=30.,unicast_downlink_bits=70.,
                initial_sync_bits=10.,cumulative_catchup_replay_bits=10.,
                cumulative_catchup_checkpoint_bits=10.,terminal_catchup_bits=10.,local_sgd_steps=9000.)
            runs.append(row);histories.append({**row,"test_accuracy":.6})
        products=analysis.summarize(pd.DataFrame(runs),pd.DataFrame(histories),pd.DataFrame(coverage),"confirmation")
        blocked=products["paired_endpoints"].query("reference == 'stc'")
        self.assertTrue(blocked.estimate.isna().all())
        self.assertTrue(blocked.n_complete_pairs.eq(9).all())
        self.assertTrue(products["paired_targets"].query("reference == 'stc'").estimate.isna().all())

    def test_saved_prefix_validation_detects_accounting_corruption(self):
        cfg=replace(study.configs()["dense"],rounds=4)
        with patch.object(host,"_architecture_ops",return_value=fake_ops(20570)),patch.object(data,"PILOT_ROUNDS",4):
            result=host.run(federation=federation(50),architecture="synthetic",method="dense",config=cfg,
                schedule=np.tile(np.arange(5),(4,1)),seed=1,horizon=4,scope="validation")
            hist=result["history"];last=hist.iloc[-1].to_dict()
            fields={"train_ce","train_objective","validation_ce","validation_accuracy","validation_macro_accuracy","validation_worst_class_accuracy"}
            summary={("final_"+k if k in fields else k):v for k,v in last.items() if k!="round"}
            summary["rounds"]=4
            tables={"rounds":result["rounds"],"history":hist,"summary":pd.DataFrame([summary])}
            study.validate_tables(tables,("pilot",5,"dense",13000))
            tables["rounds"].loc[1,"cumulative_catchup_checkpoint_bits"]+=8
            with self.assertRaises(RuntimeError):study.validate_tables(tables,("pilot",5,"dense",13000))

    def test_confirmation_freeze_requires_every_pilot_not_an_event_win(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(study,"read_receipt",return_value=None):
                with self.assertRaises(RuntimeError):study.confirmation_freeze({},Path(folder))
            with patch.object(study,"read_receipt",return_value={"state":"numerical_failure"}),\
                 patch.object(data,"file_hash",return_value="a"*64):
                study.confirmation_freeze({"protocol":{"configs":{"event":"not_a_ranking_gate"}}},Path(folder))
            frozen=json.loads((Path(folder)/"confirmation_freeze.json").read_text())
            self.assertEqual(len(frozen["pilot_receipts_sha256"]),15)
            self.assertFalse(frozen["official_test_parsed_by_campaign_before_this_freeze"])
            self.assertFalse(frozen["confirmation_development_checkpoint_reuse"])


if __name__ == "__main__":
    unittest.main()
