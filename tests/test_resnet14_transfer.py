from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from experiments import resnet14_transfer as study

HAS_TORCH=importlib.util.find_spec("torch") is not None


def fake_result(p,horizon,phase="development"):
    prefix="validation" if phase=="development" else "test"
    rounds=pd.DataFrame({"round":range(1,horizon+1),"method":p.method,"legacy_uplink_bits":160,
                         "codec_uplink_bits":80,"legacy_server_bits":160,"codec_replay_bits":80,
                         "exact_downlink_bits":64,"model_sha256":[f"{i:064x}" for i in range(1,horizon+1)]})
    evaluation=pd.DataFrame({"round":range(1,horizon+1),"train_objective":1.2,"train_ce":1.,prefix+"_ce":1.1,
                             prefix+"_accuracy":[.46,.51,.56][:horizon],prefix+"_macro_accuracy":[.46,.51,.56][:horizon],
                             prefix+"_worst_class_accuracy":.2,"model_sha256":rounds.model_sha256})
    return {"rounds":rounds,"evaluation":evaluation}


def write_result(p,phase,seed,horizon,science,out,result=None):
    result=fake_result(p,horizon,phase) if result is None else result
    tables=study.trajectory_tables(result,p=p,clients=2,dimension=8)
    files=study.paths(p,phase,seed,horizon,out)
    for k,frame in tables.items():study.atomic_csv(frame.assign(**study.metadata(p,phase,seed,horizon,science)),files[k])
    study.atomic_json({"identity":study.metadata(p,phase,seed,horizon,science),"state":"complete",
                       "sha256":{k:study.text_hash(files[k]) for k in tables}},files["receipt"])


class ProtocolTests(unittest.TestCase):
    def test_equal_bounded_grids_include_compact_and_previous_resnet_anchors(self):
        self.assertEqual(len(study.POINTS),30)
        self.assertEqual([sum(p.method==m for p in study.POINTS) for m in ("event","strom")],[15,15])
        self.assertIn("event_t0p05_q0p01",study.BY_ID)
        self.assertIn("event_t0p0125_q0p005",study.BY_ID)
        self.assertIn("strom_t0p01_g20",study.BY_ID)
        self.assertEqual(study.config_for(study.BY_ID["strom_t0p01_g20"]).server_gain,1.)
        self.assertEqual(study.config_for(study.BY_ID["strom_t0p05_g2"]).server_gain,.1)
        self.assertEqual(study.data_tools.CONFIRM_SEEDS,(7100,7200,7300))
        self.assertEqual([study.train_seed(s) for s in study.data_tools.CONFIRM_SEEDS],[97100,97200,97300])
        self.assertEqual(study.protocol_spec()["max_training_rounds"],33600)
        self.assertFalse(study.protocol_spec()["development_official_test_images_loaded"])
        self.assertFalse(study.protocol_spec()["automatic_boundary_extension"])
        self.assertEqual(study.BASE.jump_exponent,.2)

    def test_source_contract_requires_no_torch_or_images_and_preserves_old_sources(self):
        with patch.object(study.data_tools,"load_arrays",side_effect=AssertionError("no images")):
            science=study.science_spec()
        self.assertEqual(len(science["data_sha256"]),6)
        self.assertIn("experiments/resnet14_transfer_host.py",science["source_hashes"])
        self.assertIn("experiments/p13_cifar10_resnet14_torch.py",science["source_hashes"])

    def test_selection_uses_validation_ce_then_cost_and_retains_failure_status(self):
        rows=[]
        for method in ("event","strom"):
            points=[p for p in study.POINTS if p.method==method][:3]
            for i,p in enumerate(points):
                rows.append({"point_id":p.point_id,"method":method,"state":"numerical_failure" if i==2 else "complete",
                             "final_validation_ce":np.nan if i==2 else 1.,"unicast_total_bits":200 if i==0 else 100,
                             "final_test_ce":0 if i==0 else 10,"final_validation_accuracy":.99 if i==0 else .5})
        frame=pd.DataFrame(rows)
        selected=study.choose(frame,2)
        for method in ("event","strom"):
            self.assertEqual(selected[method][0],frame[(frame.method==method)&(frame.unicast_total_bits==100)&(frame.state=="complete")].iloc[0].point_id)
        self.assertEqual(int(frame.state.eq("numerical_failure").sum()),2)
        frame.loc[frame.method=="event","state"]="numerical_failure"
        with self.assertRaisesRegex(RuntimeError,"All event candidates failed"):
            study.choose(frame,1)


class SplitTests(unittest.TestCase):
    def test_validation_balance_no_leakage_full_partition_and_reproducibility(self):
        labels=np.repeat(np.arange(10),5000)
        clients,holdout,diagnostic,table,manifest=study.data_tools.allocation(labels,seed=7000,validation=True)
        self.assertEqual([len(ids) for ids in clients],[4500]*10)
        self.assertEqual(len(holdout),5000)
        np.testing.assert_array_equal(np.bincount(labels[holdout],minlength=10),np.full(10,500))
        self.assertEqual(len(np.intersect1d(np.concatenate(clients),holdout)),0)
        self.assertEqual(len(np.intersect1d(diagnostic,holdout)),0)
        self.assertEqual(len(np.unique(np.concatenate([*clients,holdout]))),50000)
        _,_,_,again,same=study.data_tools.allocation(labels,seed=7000,validation=True)
        pd.testing.assert_frame_equal(table,again);self.assertEqual(manifest,same)
        for seed in (7100,7200,7300):
            c,v,d,t,m=study.data_tools.allocation(labels,seed=seed,validation=False)
            self.assertEqual([len(ids) for ids in c],[5000]*10)
            self.assertEqual(len(v),0);self.assertEqual(len(np.unique(np.concatenate(c))),50000)
            self.assertNotEqual(m["client_indices_sha256"],manifest["client_indices_sha256"])

    def test_development_loader_parses_only_five_training_batches(self):
        names=[]
        def read(path):
            names.append(path.name)
            return np.zeros((2,3,32,32),dtype=np.uint8),np.array([0,1],dtype=np.int64)
        with patch.object(study.data_tools,"download_cifar10",return_value=Path("synthetic")),patch.object(study.data_tools,"_read_batch",side_effect=read):
            _,_,test=study.data_tools.load_arrays(Path("unused"),include_test=False)
            self.assertIsNone(test)
            self.assertEqual(names,[f"data_batch_{i}" for i in range(1,6)])
            names.clear();study.data_tools.load_arrays(Path("unused"),include_test=True)
            self.assertEqual(names[-1],"test_batch")


class EvidenceTests(unittest.TestCase):
    def test_roundtrip_crlf_relabelled_metadata_and_no_reuse_state(self):
        p=study.POINTS[0];science={"protocol":{"n_clients":2,"dimension":8}}
        with tempfile.TemporaryDirectory() as folder,patch.object(study,"BASE",replace(study.BASE,rounds=3,eval_stride=1)):
            out=Path(folder);write_result(p,"development",7000,2,science,out)
            study.verified_tables(p,"development",7000,2,science,out)
            files=study.paths(p,"development",7000,2,out)
            files["evaluation"].write_bytes(files["evaluation"].read_bytes().replace(b"\n",b"\r\n"))
            study.read_receipt(p,"development",7000,2,science,out)
            frame=pd.read_csv(files["evaluation"]);frame["partition_seed"]=4500
            study.atomic_csv(frame,files["evaluation"])
            receipt=json.loads(files["receipt"].read_text());receipt["sha256"]["evaluation"]=study.text_hash(files["evaluation"])
            study.atomic_json(receipt,files["receipt"])
            with self.assertRaisesRegex(RuntimeError,"metadata"):
                study.verified_tables(p,"development",7000,2,science,out)
            receipt["state"]="reused";study.atomic_json(receipt,files["receipt"])
            with self.assertRaisesRegex(RuntimeError,"Unexpected"):
                study.read_receipt(p,"development",7000,2,science,out)
            self.assertNotEqual(study.checkpoint_path(p,"development",7000,out),study.checkpoint_path(p,"confirmation",7100,out))

    def test_finalist_prefix_is_verified_even_when_all_csv_hashes_are_recomputed(self):
        p=study.POINTS[0];science={"protocol":{"n_clients":2,"dimension":8}}
        with tempfile.TemporaryDirectory() as folder,patch.object(study,"SCREEN",2),patch.object(study,"FINAL",3), \
                patch.object(study,"BASE",replace(study.BASE,rounds=3,eval_stride=1)):
            out=Path(folder)
            write_result(p,"development",7000,2,science,out);write_result(p,"development",7000,3,science,out)
            study.validate_prefixes((p,),science,out)
            changed=fake_result(p,3);changed["evaluation"].loc[0,"validation_ce"]+=.2
            write_result(p,"development",7000,3,science,out,result=changed)
            study.verified_tables(p,"development",7000,3,science,out)
            with self.assertRaises(AssertionError):study.validate_prefixes((p,),science,out)

    def test_confirmation_analysis_keeps_three_pairs_and_blocks_failure(self):
        science={"protocol":{"n_clients":2,"dimension":8}}
        chosen={m:[next(p.point_id for p in study.POINTS if p.method==m)] for m in ("event","strom")}
        with tempfile.TemporaryDirectory() as folder,patch.object(study,"FINAL",3), \
                patch.object(study,"BASE",replace(study.BASE,rounds=3,eval_stride=1)), \
                patch.object(study,"load_contract",return_value=science),patch.object(study,"frozen_selection",return_value={"selected":chosen}), \
                patch.object(study,"plot"),redirect_stdout(io.StringIO()):
            out=Path(folder)
            for seed in study.data_tools.CONFIRM_SEEDS:
                for m in ("event","strom"):write_result(study.BY_ID[chosen[m][0]],"confirmation",seed,3,science,out)
            study.analyze(out)
            self.assertEqual(len(pd.read_csv(out/"confirmation_runs.csv")),6)
            self.assertTrue(pd.read_csv(out/"paired_summary.csv").n.eq(3).all())
            self.assertEqual(len(pd.read_csv(out/"traffic_to_target.csv")),18)
            audit=json.loads((out/"analysis_audit.json").read_text());self.assertEqual(audit["rounds_verified"],18)
            p=study.BY_ID[chosen["event"][0]];file=study.paths(p,"confirmation",7100,3,out)["receipt"]
            value=json.loads(file.read_text());value.update(state="numerical_failure",error="synthetic");study.atomic_json(value,file)
            with self.assertRaisesRegex(RuntimeError,"all three pairs"):
                study.analyze(out)


class AutomationTests(unittest.TestCase):
    def test_full_pipeline_freezes_before_loading_test_and_confirms_even_when_event_loses(self):
        calls=[];science={"synthetic":True}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def fake_train(p,phase,seed,horizon,sc,fed,device,output):
                calls.append((p.point_id,phase,seed,horizon))
                study.atomic_json({"synthetic":p.point_id},study.paths(p,phase,seed,horizon,output)["receipt"])
            def fake_ledger(points,horizon,sc,output):
                return pd.DataFrame([{"point_id":p.point_id,"method":p.method,"state":"complete",
                                      "final_validation_ce":(2 if p.method=="event" else 1)+i*.001,
                                      "unicast_total_bits":1000+i} for i,p in enumerate(points)])
            def load(root,include_test=False):
                self.assertTrue(include_test)
                frozen=json.loads((out/"frozen_selection.json").read_text())
                self.assertEqual(frozen["official_test_evaluations"],0)
                self.assertFalse(frozen["confirmation_checkpoint_reuse"])
                return "images","labels",("official_images","official_labels")
            def fed(images,labels,seed,validation,device,test=None):
                return SimpleNamespace(scope="validation" if validation else "test",manifest={"seed":seed}),pd.DataFrame()
            with patch.object(study,"prepare",return_value=(science,"cpu","images","labels")), \
                    patch.object(study,"train_point",side_effect=fake_train),patch.object(study,"ledger",side_effect=fake_ledger), \
                    patch.object(study,"validate_prefixes"),patch.object(study,"frozen_selection"), \
                    patch.object(study.data_tools,"federation",side_effect=fed),patch.object(study.data_tools,"load_arrays",side_effect=load) as loader, \
                    patch.object(study,"analyze") as analyze,redirect_stdout(io.StringIO()):
                study.run(out=out)
            self.assertEqual(len(calls),40)
            self.assertEqual(sum(c[1]=="development" and c[3]==600 for c in calls),30)
            self.assertEqual(sum(c[1]=="development" and c[3]==1800 for c in calls),4)
            confirmation=[c for c in calls if c[1]=="confirmation"]
            self.assertEqual(len(confirmation),6)
            self.assertEqual({c[2] for c in confirmation},{7100,7200,7300})
            self.assertEqual(sum(c[0].startswith("event") for c in confirmation),3)
            loader.assert_called_once();analyze.assert_called_once_with(out)


@unittest.skipUnless(HAS_TORCH,"Torch is required for actual ResNet host checks")
class TorchHostTests(unittest.TestCase):
    def test_pulse_rules_match_hand_native_torch_and_preserve_subtractive_overshoot(self):
        import torch
        from experiments import resnet14_transfer_host as host
        host.configure("cpu",2)
        for method in ("event","strom"):
            config=replace(study.BASE,rho=.5,threshold=.1,strom_threshold=.1,jump0=.03,server_gain=.3,local_lr=.05)
            state=torch.tensor([[.3,-.3,.01,0.],[.4,-.2,0.,.1]],dtype=torch.float32)
            expected=state.clone();weights=torch.tensor([.5,.5]);deltas=[torch.tensor([.02,-.02,0.,0.]),torch.zeros(4)]
            native=torch.zeros(4)
            if method=="event":expected.mul_(config.rho)
            amp=config.jump0*(1+1/config.jump_scale)**(-config.jump_exponent) if method=="event" else .1
            for i,delta in enumerate(deltas):
                expected[i].add_(delta,alpha=(1/config.local_lr)*float(weights[i]))
                mask=expected[i].abs()>=.1;signs=expected[i,mask].sign();native[mask]=native[mask]+amp*signs
                if method=="event":expected[i,mask]=0
                else:expected[i,mask]=expected[i,mask]-.1*signs
            native.mul_(config.server_gain)
            actual,row=host.pulse_step(state,deltas,weights,method=method,config=config,rnd=1)
            np.testing.assert_array_equal(state.numpy().view(np.uint32),expected.numpy().view(np.uint32))
            np.testing.assert_array_equal(actual.numpy().view(np.uint32),native.numpy().view(np.uint32))
            self.assertGreater(row["exact_downlink_bits"],0)
            if method=="strom":self.assertGreater(float(state.abs().max()),.1)

    def test_actual_resnet_checkpoint_rejects_config_and_provenance_changes(self):
        from experiments import resnet14_transfer_host as host
        device=host.configure("cpu",2);fed=study.synthetic_federation(device)
        config=replace(study.BASE,rounds=2,eval_stride=1,local_steps=1,batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"run.npz"
            kwargs=dict(federation=fed,method="event",seed=97000,device=device,eval_batch_size=4)
            host.run(**kwargs,config=config,checkpoint=study.BaselineCheckpoint(path,{"scope":"synthetic"}),stop_after=1)
            with self.assertRaises(ValueError):host.run(**kwargs,config=replace(config,jump0=.02),checkpoint=study.BaselineCheckpoint(path,{"scope":"synthetic"}))
            with self.assertRaises(ValueError):host.run(**kwargs,config=config,checkpoint=study.BaselineCheckpoint(path,{"scope":"other"}))


if __name__=="__main__":unittest.main()
