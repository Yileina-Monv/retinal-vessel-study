"""Isolated CPU regression and real subprocess tests for the production G2 loop."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from retinal_m0 import maples_g2_engineering as ge, maples_g2_runtime as rt, maples_g2_worker as worker

torch.set_num_threads(1)

class TinyData:
    def __len__(self):return 800
    def __getitem__(self,draw):
        x=torch.rand((1,4,4),generator=torch.Generator().manual_seed(draw+4))
        y=(x>.4).float();f=torch.ones_like(y);s=torch.zeros_like(y)
        if draw%4:s[0,1,1]=y[0,1,1]
        return {'image':x,'mask':y,'fov':f,'weight':f,'selected':s,'draw':draw,'filename':str(draw),
            'geometry':torch.tensor([draw,0,0,0,0]),'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            'target_sha256':hashlib.sha256(s.numpy().tobytes()).hexdigest(),'target_count':int(s.sum())}

def fixture_objects():
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    model=torch.nn.Sequential(torch.nn.Conv2d(1,2,1),torch.nn.Dropout2d(.1),torch.nn.Conv2d(2,1,1))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False)
    scaler=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=4)
    ds=TinyData();pairs=[]
    from torch.utils.data import default_collate
    for step in range(1,201):pairs.append(worker.pair_record(default_collate([ds[d] for d in range((step-1)*4,step*4)]),step))
    return model,opt,scaler,ds,pairs

def fixture(root,track,start,end):
    if torch.cuda.is_initialized():raise RuntimeError('CPU fixture initialized CUDA')
    m,o,s,ds,pairs=fixture_objects();budget=ge.Budget(Path(root)/'ledger.json','a'*64)
    return worker.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track=track,start=start,end=end,
        out=Path(root)/track,budget=budget,binding_sha='a'*64,expected_pairs=pairs,device='cpu',authorize=lambda:None)

class RepairedAccountingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.b=ge.Budget(self.root/'ledger.json','a'*64);self.b.initialize_isolated()
    def tearDown(self):self.tmp.cleanup()
    def test_pending_blocks_other_track(self):
        self.b.reserve('B1_CONT',1,1)
        with self.assertRaisesRegex(ge.GateError,'unresolved'):self.b.reserve('B2_CONT',1,1)
    def test_nonfinite_update_charged_failed_not_success(self):
        m=torch.nn.Linear(1,1);o=torch.optim.SGD(m.parameters(),lr=.1)
        for p in m.parameters():p.grad=torch.full_like(p,float('inf'))
        with self.assertRaisesRegex(ge.GateError,'nonfinite'):ge.optimizer_update(self.b,'B1_CONT',1,1,o,None,amp_enabled=False)
        s=self.b.snapshot();self.assertEqual((s['charged_events'],s['verified_successes'],s['failed_events']),(1,0,1));self.assertEqual(s['status'],'FAILED_STOP')
    def test_exception_during_optimizer_is_uncertain(self):
        m=torch.nn.Linear(1,1);o=torch.optim.SGD(m.parameters(),lr=.1)
        with patch.object(o,'step',side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError):ge.optimizer_update(self.b,'B1_CONT',1,1,o,None,amp_enabled=False)
        s=self.b.snapshot();self.assertEqual(s['uncertain_events'],1);self.assertEqual(s['charged_events'],1)
    def test_compute_count_precedes_body_and_rejects_second_forward(self):
        self.b.before_forward('B1_CONT',1,1)
        with self.assertRaises(ge.GateError):self.b.before_forward('B1_CONT',1,1)
        self.assertEqual(self.b.snapshot()['forward_calls'],1)
    def test_gradient_limit_and_wrong_identity(self):
        self.b.before_forward('B1_CONT',1,1)
        with self.assertRaises(ge.GateError):self.b.before_gradient('B2_CONT',1,1)
        for _ in range(3):self.b.before_gradient('B1_CONT',1,1)
        with self.assertRaises(ge.GateError):self.b.before_gradient('B1_CONT',1,1)
    def test_real_scaler_overflow_and_same_draw_retry(self):
        m=torch.nn.Linear(1,1);o=torch.optim.SGD(m.parameters(),lr=.1);s=torch.amp.GradScaler('cpu',init_scale=128)
        for attempt,overflow in [(1,True),(2,False)]:
            self.b.before_forward('B1_CONT',1,attempt);o.zero_grad(set_to_none=True)
            self.b.before_gradient('B1_CONT',1,attempt);s.scale(m(torch.ones(1,1)).square().mean()).backward()
            if overflow:
                for p in m.parameters():p.grad.fill_(float('inf'))
            s.unscale_(o)
            result=ge.optimizer_update(self.b,'B1_CONT',1,attempt,o,s,amp_enabled=True)
            self.assertEqual(result['outcome'],'amp_overflow' if overflow else 'success')
        state=self.b.snapshot();self.assertEqual((state['verified_successes'],state['amp_retries'],state['forward_calls']),(1,1,2));self.assertEqual(s.get_scale(),64)
    def test_ratio_pause_prevents_actual_update(self):
        m,o,s,ds,pairs=fixture_objects()
        with patch.object(worker,'same_forward_gradient_audit',return_value={'pause_ratio_gt_2':True}):
            with self.assertRaisesRegex(ge.GateError,'requires Sol review'):
                worker.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='B1_CONT',start=0,end=50,out=self.root/'B1_CONT',budget=self.b,binding_sha='a'*64,expected_pairs=pairs,device='cpu',authorize=lambda:None)
        state=self.b.snapshot();self.assertEqual(state['verified_successes'],0);self.assertEqual(state['forward_calls'],1);self.assertEqual(state['status'],'FAILED_STOP')
    def test_missing_ledger_does_not_create_replacement(self):
        with self.assertRaises(ge.GateError):ge.Budget(self.root/'missing','a'*64).snapshot()
    def test_production_loop_same_batch_backoff_and_ninth_overflow_stop(self):
        m,o,s,ds,pairs=fixture_objects()
        counts=[0]
        def once(g):
            counts[0]+=1
            return torch.full_like(g,float('inf')) if counts[0]==1 else g
        handle=next(m.parameters()).register_hook(once)
        worker.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='R0_LAMBDA0',start=0,end=200,out=self.root/'R0_LAMBDA0',budget=self.b,binding_sha='a'*64,expected_pairs=pairs,device='cpu',authorize=lambda:None)
        handle.remove()
        rows=[json.loads(x) for x in (self.root/'R0_LAMBDA0/attempts.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['input'],rows[1]['input']);self.assertEqual(rows[0]['outcome'],'amp_overflow')
        self.assertEqual(rows[0]['scale_after'],rows[0]['scale_before']/2)
        self.assertEqual(self.b.snapshot()['verified_successes'],200);self.assertEqual(self.b.snapshot()['forward_calls'],201)
        bad=ge.Budget(self.root/'bad.json','a'*64);bad.initialize_isolated();m,o,s,ds,pairs=fixture_objects()
        handle=next(m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        with self.assertRaisesRegex(ge.GateError,'eight retries'):
            worker.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='R0_LAMBDA0',start=0,end=200,out=self.root/'bad_run',budget=bad,binding_sha='a'*64,expected_pairs=pairs,device='cpu',authorize=lambda:None)
        handle.remove();state=bad.snapshot();self.assertEqual(state['forward_calls'],9);self.assertEqual(state['amp_retries'],9);self.assertEqual(state['optimizer_events'],0)
    def test_counted_parameter_gradients_are_unscaled(self):
        m=torch.nn.Linear(1,1,bias=False);z=m(torch.ones(1,1));s=torch.amp.GradScaler('cpu',init_scale=1024)
        a=ge.same_forward_gradient_audit(m,z,z.sum()*.002,z.sum()*.001,scaler=s,audit_step=1)
        self.assertAlmostEqual(a['base_grad_l2'],.002,places=8);self.assertAlmostEqual(a['weighted_aux_grad_l2'],.001,places=8)
        self.assertAlmostEqual(a['aux_to_base_l2'],.5);self.assertAlmostEqual(a['cosine'],1)
    def test_audit_does_not_change_actual_amp_update(self):
        m,o,s,_,_=fixture_objects();n=copy.deepcopy(m);p=torch.optim.AdamW(n.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False);t=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=4)
        m.eval();n.eval();x=torch.ones(4,1,4,4)
        for model,opt,scaler,audit in [(m,o,s,False),(n,p,t,True)]:
            z=model(x);base=z.square().mean();aux=.1*torch.nn.functional.softplus(-z).mean()
            if audit:ge.same_forward_gradient_audit(model,z,base,aux,scaler=scaler,audit_step=1)
            scaler.scale(base+aux).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update()
        self.assertTrue(worker.exact_equal(m.state_dict(),n.state_dict()));self.assertTrue(worker.exact_equal(o.state_dict(),p.state_dict()));self.assertEqual(s.state_dict(),t.state_dict())

class AuthorityTests(unittest.TestCase):
    def state(self,**change):
        s={'policy_id':'P1_B1_B2_G2_ENGINEERING_EXCEPTION_01_20261021','policy_sha256':rt.POLICY_SHA,'phase':'G2_ONLY',
           'user_approved':True,'GPU_execution_conditionally_authorized':True,'charged_update_ceiling':400,'scientific_authorized':False,
           'state':'HOLD_CPU_PREFLIGHT','GPU_released':False}
        s.update(change);return '<!-- P1_G2_RELEASE_CURRENT -->'+json.dumps(s)+'<!-- /P1_G2_RELEASE_CURRENT -->'
    def test_wrong_phase_permission_and_cap(self):
        for change in [{'phase':'G1_ONLY'},{'user_approved':False},{'charged_update_ceiling':999},{'charged_update_ceiling':400.0},{'scientific_authorized':True}]:
            with self.subTest(change=change),self.assertRaises(ge.GateError):rt.current_g2_state(self.state(**change))
    def test_duplicate_fields_rejected(self):
        text=self.state().replace('"phase": "G2_ONLY"','"phase":"G1_ONLY","phase":"G2_ONLY"')
        with self.assertRaises(ge.GateError):rt.current_g2_state(text)
    def test_missing_release_denies_before_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'status').write_text(self.state())
            with patch.object(rt,'STATUS',p/'status'),patch.object(rt,'G2_RELEASE',p/'absent'):
                with self.assertRaisesRegex(ge.GateError,'Sol-issued'):rt.run_g2()
    def test_release_integrity_and_evidence_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'code';src.write_text('bound')
            evidence=p/'test.log';evidence.write_text('CPU test passed')
            binding={'phase':'G2_ONLY','production_ledger':str(p/'ledger'),'gpu_uuid':rt.GPU_UUID,'source_hashes':{str(src):ge.file_sha(src)}}
            ge.atomic_json(p/'ready',binding);digest=ge.canonical_sha(binding)
            receipt={'status':'CPU_VALIDATED_GPU_HELD','all_passed':True,'binding_sha256':digest,'evidence_hashes':{str(evidence):ge.file_sha(evidence)}}
            ge.atomic_json(p/'receipt',receipt)
            release={'phase':'G2_ONLY','review_owner':'Sol','policy_sha256':rt.POLICY_SHA,'charged_update_ceiling':400,'scientific_authorized':False,
                'release_id':'fixture','binding_sha256':digest,'binding_file_sha256':ge.file_sha(p/'ready'),'cpu_receipt_sha256':ge.file_sha(p/'receipt')}
            ge.atomic_json(p/'release',release)
            (p/'status').write_text(self.state(state='RELEASED',GPU_released=True,release_id='fixture',release_file_sha256=ge.file_sha(p/'release'),binding_sha256=digest))
            with patch.multiple(rt,STATUS=p/'status',G2_RELEASE=p/'release',READY=p/'ready',RECEIPT=p/'receipt',G2OUT=p,G2_LEDGER=p/'ledger'):
                self.assertEqual(rt.verify_release(),binding)
                evidence.write_text('changed')
                with self.assertRaisesRegex(ge.GateError,'evidence drift'):rt.verify_release()
    def test_private_segment_cannot_run_without_gpu_mask(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(ge.GateError,'GPU0'):worker.gpu_worker(0)

class RealProcessRecoveryTests(unittest.TestCase):
    def test_b1_b2_real_amp_process_restart_and_all_state_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=ge.Budget(root/'ledger.json','a'*64);b.initialize_isolated()
            for method in ('B1','B2'):
                for track,start,end in [(method+'_CONT',0,50),(method+'_SPLIT',0,25),(method+'_SPLIT',25,50)]:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
                    r=subprocess.run([sys.executable,'-B','-m',__name__ if __name__!='__main__' else 'tests.test_maples_g2_repaired','--fixture',str(root),track,str(start),str(end)],env=env,capture_output=True,text=True,timeout=300)
                    self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                for n in (25,50):
                    a=torch.load(root/f'{method}_CONT/step_{n:06d}.pt',weights_only=True)
                    c=torch.load(root/f'{method}_SPLIT/step_{n:06d}.pt',weights_only=True)
                    worker.compare_checkpoint(a,c)
                    self.assertNotEqual(a['pid'],c['pid'])
                    self.assertEqual(c['scaler']['_growth_tracker'],n%4)
                for name in ('train.jsonl','attempts.jsonl'):
                    self.assertEqual((root/f'{method}_CONT'/name).read_bytes(),(root/f'{method}_SPLIT'/name).read_bytes())
                restore=ge.read_json(root/f'{method}_SPLIT/restore_000025.json')
                self.assertNotEqual(restore['pid'],restore['previous_pid'])
            state=b.snapshot();self.assertEqual(state['verified_successes'],200);self.assertEqual(state['forward_calls'],200);self.assertEqual(state['gradient_calls'],240)
            self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture(sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
    else:unittest.main()
