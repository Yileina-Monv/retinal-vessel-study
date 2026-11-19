"""Tiny CPU fixtures test the production loop; no retinal model or GPU updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,unittest,random,io,contextlib
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from retinal_m0.maples_soft_cldice_engineering_core import (Budget,run_segment,pair_record,empty_budget)
from retinal_m0.maples_g1_safety import GateError,atomic_json
from torch.utils.data._utils.collate import default_collate

torch.set_num_threads(2)

class TinyDataset:
    def __len__(self):return 200
    def __getitem__(self,i):
        g=np.random.default_rng(500+i);x=g.normal(size=(3,16,16)).astype(np.float32)
        y=(g.random((1,16,16))>.65).astype(np.float32)
        return {'image':torch.tensor(x),'mask':torch.tensor(y),'fov':torch.ones(1,16,16),'draw':i,'filename':f'fixture_{i}',
                'geometry':torch.tensor([i,0,0,0,0]),'input_sha256':f'fixture_hash_{i}'}

class EngineeringTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def budget(self,name='budget'):
        b=Budget(self.root/(name+'.json'),'fixture');b.initialize_fixture();return b
    def reserve_ready(self,b,track='CONT',step=1):
        b.before_forward(track,step)
        for _ in range(3):b.before_gradient(track,step)
        b.reserve(track,step,1)
    def model(self):
        torch.manual_seed(2026);m=torch.nn.Conv2d(3,1,1)
        return m,torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False),torch.amp.GradScaler('cpu',enabled=False)
    def settings(self):
        ds=TinyDataset();pairs=[pair_record(default_collate([ds[i] for i in range(s*4,s*4+4)]),s+1) for s in range(50)]
        return ds,pairs

    def test_production_loop_exact_50_vs_25_restore25(self):
        ds,pairs=self.settings();budget=self.budget();m,o,s=self.model();random.seed(2026);np.random.seed(2026)
        common=dict(dataset=ds,budget=budget,binding_sha='fixture',expected_pairs=pairs,device='cpu',eta=.15115714003619224,k=2)
        run_segment(model=m,optimizer=o,scaler=s,track='CONT',start=0,end=50,out=self.root/'CONT',**common)
        m,o,s=self.model();random.seed(2026);np.random.seed(2026)
        run_segment(model=m,optimizer=o,scaler=s,track='SPLIT',start=0,end=25,out=self.root/'SPLIT',**common)
        m,o,s=self.model();random.seed(12);np.random.seed(19);torch.manual_seed(33)
        run_segment(model=m,optimizer=o,scaler=s,track='SPLIT',start=25,end=50,out=self.root/'SPLIT',**common)
        a=[json.loads(l) for l in (self.root/'CONT/train.jsonl').read_text().splitlines()]
        b=[json.loads(l) for l in (self.root/'SPLIT/train.jsonl').read_text().splitlines()]
        for x,y in zip(a,b):
            for key in ['pair','loss','audit','state_digests','preclip_total_L2']:self.assertEqual(x[key],y[key])
        z=budget.snapshot();self.assertEqual(z['verified_successes'],100);self.assertEqual(z['forward_calls'],100);self.assertEqual(z['gradient_calls'],300)

    def test_no_budget_reset(self):
        b=self.budget()
        with self.assertRaises(GateError):b.initialize_fixture()

    def test_failure_and_unknown_are_charged_and_stop(self):
        for outcome in ['failed','uncertain']:
            b=self.budget(outcome);self.reserve_ready(b);b.before_optimizer('CONT',1);b.resolve('CONT',outcome)
            z=b.snapshot();self.assertEqual(z['charged_events'],1);self.assertEqual(z['verified_successes'],0)
            with self.assertRaises(GateError):b.before_forward('CONT',1)

    def test_unreserved_optimizer_denied(self):
        b=self.budget()
        with self.assertRaises(GateError):b.before_optimizer('CONT',1)
        self.assertEqual(b.snapshot()['charged_events'],0)

    def test_duplicate_optimizer_pre_hook_denied(self):
        b=self.budget();self.reserve_ready(b);b.before_optimizer('CONT',1)
        with self.assertRaises(GateError):b.before_optimizer('CONT',1)

    def test_compute_cannot_be_repeated_or_grad_exceeded(self):
        b=self.budget();b.before_forward('CONT',1)
        with self.assertRaises(GateError):b.before_forward('CONT',1)
        for _ in range(3):b.before_gradient('CONT',1)
        with self.assertRaises(GateError):b.before_gradient('CONT',1)

    def test_caps_50_per_track100_total(self):
        b=self.budget()
        for track in ['CONT','SPLIT']:
            for step in range(1,51):self.reserve_ready(b,track,step);b.before_optimizer(track,step);b.resolve(track,'success')
        self.assertEqual(b.snapshot()['charged_events'],100)
        with self.assertRaises(GateError):b.before_forward('CONT',51)
        with self.assertRaises(GateError):b.before_forward('SPLIT',51)

    def test_amp_overflow_stops_without_retry_or_refund(self):
        b=self.budget();self.reserve_ready(b);b.resolve('CONT','amp_overflow',amp_retry=True)
        s=b.snapshot();self.assertEqual(s['forward_calls'],1);self.assertEqual(s['failed_compute'],1);self.assertEqual(s['charged_events'],0)
        with self.assertRaises(GateError):b.before_forward('CONT',1)

    def test_bad_pair_stops_before_forward(self):
        ds,pairs=self.settings();pairs[0]['input_sha256'][0]='wrong';budget=self.budget();m,o,s=self.model()
        with self.assertRaises(GateError):run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='CONT',start=0,end=50,out=self.root/'bad',budget=budget,binding_sha='fixture',expected_pairs=pairs,device='cpu',eta=.1,k=2)
        self.assertEqual(budget.snapshot()['forward_calls'],0);self.assertEqual(budget.snapshot()['verified_successes'],0)

    def test_scientific_segment_not_allowed(self):
        ds,pairs=self.settings();b=self.budget();m,o,s=self.model()
        with self.assertRaises(GateError):run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='CONT',start=0,end=2000,out=self.root/'bad',budget=b,binding_sha='fixture',expected_pairs=pairs,device='cpu',eta=.1,k=2)

    def test_cli_budget_seed_device_overrides_denied(self):
        import retinal_m0.maples_soft_cldice_engineering as entry
        for args in [[],['--run','--steps','2000'],['--run','--device','cuda:1'],['--run','--seed','2027']]:
            with mock.patch('sys.argv',['eng',*args]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as e:entry.main()
            self.assertEqual(e.exception.code,2)

    def test_worker_needs_active_authorized_parent_before_GPU(self):
        import retinal_m0.maples_soft_cldice_engineering as entry
        with mock.patch.object(entry,'authority',return_value={'eta':.1,'k':26}),mock.patch.object(entry,'read_json',side_effect=FileNotFoundError('no session')):
            with self.assertRaises(FileNotFoundError):entry.worker_authority('CONT',0,50)

if __name__=='__main__':unittest.main(verbosity=2)
