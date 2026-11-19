"""V4 isolated CPU validation; uses the same production loop and actual GradScaler."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from retinal_m0 import maples_g2_v4 as v, maples_g2_v4_worker as w
from retinal_m0.maples_g2_engineering import GateError,read_json,atomic_json,optimizer_update
from retinal_m0.maples_branch_bce import branch_bce_loss,normalized_positive_bce
from retinal_m0.maples_core import weighted_masked_loss
from tests.test_maples_g2_repaired import fixture_objects

def parent_fixture():return read_json(v.previous.G2_LEDGER)

def fixture(root,track,start,end):
    m,o,s,ds,pairs=fixture_objects()
    return w.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track=track,start=start,end=end,
        out=Path(root)/track,budget=v.Budget(Path(root)/'ledger.json','a'*64),binding_sha='a'*64,expected_pairs=pairs,
        device='cpu',authorize=lambda:None)

class LossTests(unittest.TestCase):
    def test_exact_zero_lambda_and_003_derivative_empty_and_batch4(self):
        z=torch.linspace(-5,5,64).reshape(4,1,4,4).requires_grad_();y=torch.ones_like(z);f=torch.ones_like(z);s=torch.zeros_like(z)
        for i,n in enumerate([0,1,5,16]):s[i].view(-1)[:n]=1
        original=weighted_masked_loss(z,y,f,f);zero=branch_bce_loss(z,y,f,s,lam=0)
        self.assertTrue(torch.equal(zero,original));self.assertTrue(torch.equal(torch.autograd.grad(zero,z,retain_graph=True)[0],torch.autograd.grad(original,z,retain_graph=True)[0]))
        loss=v.LAMBDA*normalized_positive_bce(z,y,f,s)
        actual=torch.autograd.grad(loss,z)[0]
        expected=v.LAMBDA/4*s*(z.detach().sigmoid()-1)/s.sum((1,2,3)).clamp_min(1)[:,None,None,None]
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-5));self.assertEqual(float(actual[0].abs().sum()),0)
    def test_audit45_real_unscaled_gradient_and_side_effect_free_update(self):
        m,o,s,_,_=fixture_objects();n=copy.deepcopy(m);p=torch.optim.AdamW(n.parameters(),lr=1e-4,weight_decay=1e-4,foreach=False);t=torch.amp.GradScaler('cpu',init_scale=128,growth_interval=4)
        m.eval();n.eval();x=torch.ones(4,1,4,4)
        for model,opt,scaler,audit in [(m,o,s,False),(n,p,t,True)]:
            z=model(x);base=z.square().mean();aux=.03*torch.nn.functional.softplus(-z).mean()
            if audit:
                result=v.gradient_audit(model,z,base,aux,scaler=scaler,audit_step=45)
                self.assertTrue(result['unscaled']);self.assertEqual(result['audit_step'],45)
            scaler.scale(base+aux).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update()
        self.assertTrue(w.exact_equal(m.state_dict(),n.state_dict()));self.assertTrue(w.exact_equal(o.state_dict(),p.state_dict()));self.assertEqual(s.state_dict(),t.state_dict())
    def test_ratio_pause_and_zero_base_guard(self):
        m=torch.nn.Linear(1,1,bias=False);z=m(torch.ones(1,1));s=torch.amp.GradScaler('cpu')
        for base in [z.sum()*.001,z.sum()*0]:
            a=v.gradient_audit(m,z,base,.03*z.sum(),scaler=s,audit_step=10);self.assertTrue(a['pause_ratio_gt_2'])

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.parent=parent_fixture()
        self.b=v.Budget(self.root/'ledger.json','a'*64);self.b.initialize_migration(self.parent)
    def tearDown(self):self.tmp.cleanup()
    def test209_and_old_compute_pending_retained_without_extra_update_charge(self):
        s=self.b.snapshot();self.assertEqual(s['charged_events'],209);self.assertEqual(s['forward_calls'],210);self.assertEqual(s['gradient_calls'],216)
        self.assertEqual(s['inherited_ledger'],self.parent);self.assertEqual(s['inherited_ledger']['compute_pending']['step'],10);self.assertIsNone(s['compute_pending'])
        with self.assertRaises(GateError):self.b.initialize_migration(self.parent)
    def test_all180_updates_stop389_and_do_not_use_remaining11(self):
        for track in v.CAPS:
            for step in range(1,46):
                self.b.before_forward(track,step,1);self.b.reserve(track,step,1);self.b.resolve(track,'success')
        s=self.b.snapshot();self.assertEqual(s['charged_events'],389);self.assertEqual(s['verified_successes'],389)
        self.assertEqual(s['forward_calls'],390)
        with self.assertRaises(GateError):self.b.before_forward('B2_SPLIT',46,1)
    def test_failed_and_uncertain_consumption_never_refunded(self):
        self.b.before_forward('B1_CONT',1,1);self.b.reserve('B1_CONT',1,1);self.b.mark_crash_uncertain('B1_CONT')
        s=self.b.snapshot();self.assertEqual(s['charged_events'],210);self.assertEqual(s['verified_successes'],209);self.assertEqual(s['uncertain_events'],1)
        with self.assertRaises(GateError):self.b.before_forward('B2_CONT',1,1)
    def test_pending_optimizer_blocks_all_tracks(self):
        self.b.before_forward('B1_CONT',1,1);self.b.reserve('B1_CONT',1,1)
        with self.assertRaises(GateError):self.b.before_forward('B2_CONT',1,1)
    def test_parent_or_binding_tamper_rejected(self):
        s=self.b.snapshot();s['inherited_ledger']['charged_events']=0;atomic_json(self.b.path,s)
        with self.assertRaises(GateError):self.b.snapshot()
    def test_nonfinite_optimizer_call_charged_as_failure(self):
        m=torch.nn.Linear(1,1);o=torch.optim.SGD(m.parameters(),lr=.1)
        for p in m.parameters():p.grad=torch.full_like(p,float('inf'))
        self.b.before_forward('B1_CONT',1,1)
        with self.assertRaises(GateError):optimizer_update(self.b,'B1_CONT',1,1,o,None,amp_enabled=False)
        s=self.b.snapshot();self.assertEqual(s['charged_events'],210);self.assertEqual(s['failed_events'],1);self.assertEqual(s['verified_successes'],209)
    def test_ninth_amp_overflow_stops_without_draw_advance(self):
        for attempt in range(1,10):
            self.b.before_forward('B1_CONT',1,attempt);self.b.reserve('B1_CONT',1,attempt);self.b.resolve('B1_CONT','amp_overflow',amp_retry=True)
        s=self.b.snapshot();self.assertEqual(s['amp_retries'],9);self.assertEqual(s['charged_events'],209);self.assertEqual(s['status'],'FAILED_STOP')
    def test_ratio_pause_saves_previous_complete_last(self):
        m,o,s,ds,pairs=fixture_objects();real=v.gradient_audit
        def pause_at10(*args,**kwargs):
            a=real(*args,**kwargs)
            if kwargs['audit_step']==10:a['pause_ratio_gt_2']=True
            return a
        with patch.object(w,'same_forward_gradient_audit',side_effect=pause_at10):
            with self.assertRaisesRegex(GateError,'requires Sol review'):
                w.run_segment(model=m,optimizer=o,scaler=s,dataset=ds,track='B1_CONT',start=0,end=45,out=self.root/'B1_CONT',budget=self.b,binding_sha='a'*64,expected_pairs=pairs,device='cpu',authorize=lambda:None)
        last=torch.load(self.root/'B1_CONT/last.pt',weights_only=True);self.assertEqual(last['step'],9);self.assertEqual(last['next_global_draw'],36)
        self.assertEqual(self.b.snapshot()['charged_events'],218);self.assertEqual(self.b.snapshot()['forward_calls'],220)
        rows=[json.loads(x) for x in (self.root/'B1_CONT/train.jsonl').read_text().splitlines()]
        from retinal_m0.state import state_digest
        for key in ['model','optimizer','scaler','rng']:self.assertEqual(rows[-1]['state_digests'][key],state_digest(last[key]))

class EntryTests(unittest.TestCase):
    def test_default_no_training_and_unreleased_rejected(self):
        with self.assertRaises(SystemExit):v.main([])
        with self.assertRaises(GateError):v.run()
    def test_no_gpu1_or_unmasked_private_worker(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(GateError,'GPU0'):w.gpu_worker(0)
    def test_release_current_revision_required(self):
        with patch.object(v.previous,'current_g2_state',return_value={'revision_id':'old','planned_total_charged':389}):
            with self.assertRaises(GateError):v.active()
    def test_missing_ledger_cannot_reset(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(GateError):v.Budget(Path(t)/'absent','a'*64).snapshot()
    def test_valid_release_and_scope_hash_evidence_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src=p/'code';src.write_text('bound');log=p/'test.log';log.write_text('passed')
            b={'schema':4,'revision_id':v.REVISION,'phase':'G2_ONLY','lambda':.03,'global_cap':400,'planned_total':389,
                'ledger_path':str(p/'ledger'),'gpu_uuid':v.GPU_UUID,'sources':{str(src):v.file_sha(src)}}
            atomic_json(p/'ready',b);h=v.canonical_sha(b)
            c={'all_passed':True,'binding_sha256':h,'R0_reuse_verified':True,'evidence_hashes':{str(log):v.file_sha(log)}};atomic_json(p/'receipt',c)
            r={'phase':'G2_ONLY','revision_id':v.REVISION,'release_id':'fixture','scientific_authorized':False,'review_owner':'Sol',
                'binding_sha256':h,'binding_file_sha256':v.file_sha(p/'ready'),'cpu_receipt_sha256':v.file_sha(p/'receipt'),'charged_update_ceiling':400,'planned_total':389,'lambda':.03}
            atomic_json(p/'release',r)
            s={'state':'RELEASED','GPU_released':True,'release_id':'fixture','release_file_sha256':v.file_sha(p/'release'),'binding_sha256':h}
            with patch.multiple(v,OUT=p,READY=p/'ready',RECEIPT=p/'receipt',RELEASE=p/'release',LEDGER=p/'ledger'),patch.object(v,'active',return_value=s):
                self.assertEqual(v.verify_release(),b)
                for key,value in [('phase','G1_ONLY'),('scientific_authorized',True),('planned_total',400),('lambda',.1)]:
                    bad=dict(r);bad[key]=value;atomic_json(p/'release',bad);s['release_file_sha256']=v.file_sha(p/'release')
                    with self.subTest(key=key),self.assertRaises(GateError):v.verify_release()
                atomic_json(p/'release',r);s['release_file_sha256']=v.file_sha(p/'release');log.write_text('changed')
                with self.assertRaisesRegex(GateError,'evidence drift'):v.verify_release()

class ProcessRecoveryTests(unittest.TestCase):
    def test_real25_plus20_both_methods_all_states_and_logs_equal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=v.Budget(root/'ledger.json','a'*64);b.initialize_migration(parent_fixture())
            for method in ['B1','B2']:
                for track,start,end in [(method+'_CONT',0,45),(method+'_SPLIT',0,25),(method+'_SPLIT',25,45)]:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
                    r=subprocess.run([sys.executable,'-B','-m','tests.test_maples_g2_v4','--fixture',str(root),track,str(start),str(end)],env=env,capture_output=True,text=True,timeout=300)
                    self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                for step in [25,45]:
                    a=torch.load(root/f'{method}_CONT/step_{step:06d}.pt',weights_only=True);c=torch.load(root/f'{method}_SPLIT/step_{step:06d}.pt',weights_only=True)
                    w.compare_checkpoint(a,c);self.assertNotEqual(a['pid'],c['pid'])
                for name in ['train.jsonl','attempts.jsonl']:
                    self.assertEqual((root/f'{method}_CONT'/name).read_bytes(),(root/f'{method}_SPLIT'/name).read_bytes())
                last=torch.load(root/f'{method}_SPLIT/last.pt',weights_only=True);self.assertEqual(last['step'],45)
            s=b.snapshot();self.assertEqual(s['charged_events'],389);self.assertEqual(s['forward_calls'],390);self.assertEqual(s['gradient_calls'],436)
            self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture(sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
    else:unittest.main()
