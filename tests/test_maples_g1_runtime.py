"""CPU-only isolated session/CLI tests; no real release, retinal model, or CUDA."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from retinal_m0 import maples_g1_runtime as rt
from retinal_m0.maples_g1_safety import *


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.body_calls=0
    def forward(self,x):self.body_calls+=1;return x*self.weight


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        self.model=Tiny();self.optimizer=torch.optim.AdamW(self.model.parameters())
        self.plan=[{'combo_id':str(i),'step':i//4+1,'method':('B1' if i%4<2 else 'B2'),
                    'precision':('FP32' if i%2==0 else 'AMP')} for i in range(32)]
        self.batch=[{'step_original_T0_one_based':step,'draws':[{'draw':step,
                    'input_sha256_original_T0':'input','target_crop_uint8_C_order_sha256':{'B1':'target','B2':'target'}}]}
                    for step in range(1,9)]
        path=self.out/'ledger.json';atomic_json(path,{'schema':1,'cap':32,'combinations':{},
                   'autograd_calls':0,'optimizer_step_attempts':0})
        self.ledger=Ledger(path,'binding',[p['combo_id'] for p in self.plan]);self.ledger.initialize_empty_legacy(self.out/'backup.json')
        class Provider:
            def get(_,batch,method):
                step=batch['step_original_T0_one_based']
                return {**{k:torch.ones(1,1,2,2) for k in ('x','y','fov','selected')},
                    'actual_input_digest':canonical_sha([{'draw':step,'sha256':'input'}]),
                    'actual_target_digest':canonical_sha([{'draw':step,'sha256':'target'}])}
        self.provider=Provider()

    def tearDown(self):self.tmp.cleanup()

    def diagnostic(self,model,x,*args,record,**kwargs):
        model(x)
        for _ in range(4):record('autograd_call',{})
        return {**{k:1. for k in ('base_loss','aux_unweighted','grad_l2_base',
                'grad_l2_aux_weighted','grad_l2_combined')},'gradient_review_trigger':False}

    def session(self,diagnostic=None,authorize=lambda:None):
        with patch('retinal_m0.maples_g1_diagnostics.diagnose',side_effect=diagnostic or self.diagnostic):
            return rt._session(self.model,self.optimizer,self.provider,self.ledger,self.plan,self.batch,
                               self.out,authorize,lambda _:None)

    def test_32_actual_tiny_forwards_zero_updates(self):
        result=self.session();state=ledger_snapshot(self.ledger.path)
        self.assertEqual(self.model.body_calls,32);self.assertEqual(state['forward_starts'],32)
        self.assertEqual(state['autograd_calls'],128);self.assertEqual(state['optimizer_step_attempts'],0)
        self.assertEqual(len(result['FP32_vs_AMP']),16);self.assertFalse(result['G2_authorized'])
        self.assertEqual(self.model.weight.item(),1)

    def test_33rd_forward_never_enters_model_body(self):
        calls=[0]
        def diagnostic(*a,**k):
            result=self.diagnostic(*a,**k);calls[0]+=1
            if calls[0]==32:a[0](a[1])
            return result
        with self.assertRaises(GateError):self.session(diagnostic)
        state=ledger_snapshot(self.ledger.path)
        self.assertEqual(self.model.body_calls,32);self.assertEqual(len(state['combinations']),32)
        self.assertEqual(state['blocked_forward_attempts'],1)
        self.assertEqual(state['combinations']['31']['status'],'failed')
        self.assertEqual(read_json(self.out/'31.json')['status'],'FAILED')

    def test_optimizer_call_blocked_before_update(self):
        def diagnostic(*a,**k):
            result=self.diagnostic(*a,**k);self.optimizer.step();return result
        with self.assertRaises(GateError):self.session(diagnostic)
        state=ledger_snapshot(self.ledger.path);self.assertEqual(state['optimizer_step_attempts'],1)
        self.assertEqual(self.model.weight.item(),1);self.assertEqual(len(state['combinations']),1)

    def test_nested_nan_fails_and_partial_evidence_survives(self):
        def diagnostic(*a,record,**k):
            m=self.diagnostic(*a,record=record,**k);m['per_layer']={'bad':[float('nan')]}
            record('metrics_observed',m);return m
        with self.assertRaises(GateError):self.session(diagnostic)
        report=read_json(self.out/'0.json')
        self.assertEqual(report['status'],'FAILED')
        self.assertIn('NONFINITE',report['metrics']['per_layer']['bad'][0])
        self.assertEqual(len(ledger_snapshot(self.ledger.path)['combinations']),1)

    def test_revocation_before_model_body_is_charged(self):
        n=[0]
        def authorize():
            n[0]+=1
            if n[0]>1:raise GateError('revoked')
        with self.assertRaises(GateError):self.session(authorize=authorize)
        self.assertEqual(self.model.body_calls,0)
        state=ledger_snapshot(self.ledger.path)
        self.assertEqual(len(state['combinations']),1);self.assertEqual(state['forward_starts'],0)

    def test_actual_input_fingerprint_mismatch_stops(self):
        with patch.object(self.provider,'get',return_value={**{k:torch.ones(1,1,2,2) for k in ('x','y','fov','selected')},
                          'actual_input_digest':'wrong','actual_target_digest':'wrong'}):
            with self.assertRaises(GateError):self.session()
        self.assertEqual(self.model.body_calls,0)

    def test_no_args_and_no_production_ledger_override(self):
        for args in ([],['--run-g1','--ledger-path','other.json'],['--run-g1','--cpu-preflight']):
            with contextlib.redirect_stderr(io.StringIO()),patch.object(rt,'run_gpu') as gpu:
                with self.assertRaises(SystemExit):rt.main(args)
                gpu.assert_not_called()

    def test_hold_rejected_before_cuda_or_model_initialization(self):
        status=self.out/'STATUS.md';status.write_text(START+'\n{"state":"HOLD","phase":"G1_ONLY"}\n'+END)
        with patch.object(rt,'STATUS',status),patch.object(rt,'build_binding',return_value=({'source_hashes':{}},{})),\
             patch('torch.cuda.device_count',side_effect=AssertionError('must not call CUDA')),\
             patch('torch.load',side_effect=AssertionError('must not load M0')),\
             patch('retinal_m0.model.UNet',side_effect=AssertionError('must not construct U-Net')):
            with self.assertRaisesRegex(GateError,'not RELEASED'):rt.run_gpu()
        self.assertFalse(torch.cuda.is_initialized())

    def test_resource_probe_queries_only_physical_gpu0(self):
        class Result:
            stdout='0, '+GPU_UUID+', 11178, 115, 0\n'
        with patch.object(rt.subprocess,'run',return_value=Result()) as run:
            snap=rt.probe_resources()
        self.assertFalse(snap['GPU1_queried'])
        self.assertEqual(run.call_args.args[0][1:3],['-i','0'])

    def test_busy_gpu_and_wrong_uuid_rejected(self):
        for line in ('0, '+GPU_UUID+', 11178, 2000, 0\n','0, WRONG, 11178, 115, 0\n'):
            class Result:stdout=line
            with patch.object(rt.subprocess,'run',return_value=Result()):
                with self.assertRaises(GateError):rt.probe_resources(require_idle=True)

    def test_initialization_failure_is_journaled_before_any_forward(self):
        import os
        attempt=self.out/'gpu_simulation';attempt.mkdir()
        legacy=self.out/'legacy.json'
        atomic_json(legacy,{'schema':1,'cap':32,'combinations':{},'autograd_calls':0,'optimizer_step_attempts':0})
        binding={'source_hashes':{},'fixed_plan':self.plan}
        with patch.object(rt,'build_binding',return_value=(binding,{})),\
             patch.object(rt,'verify_authority',return_value={'authorized':True}),\
             patch.object(rt,'STATUS',self.out/'status'),\
             patch.object(rt,'LEDGER',legacy),\
             patch.object(rt,'RUN_LOCK',self.out/'run.lock'),\
             patch.object(rt,'GPU_LOCK',self.out/'gpu.lock'),\
             patch.object(rt,'_next_output',return_value=attempt),\
             patch.object(rt,'probe_resources',return_value={'GPU1_queried':False}),\
             patch.object(rt,'_authorized_gpu_body',side_effect=RuntimeError('simulated initialization failure')),\
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':GPU_UUID}):
            (self.out/'status').write_text('isolated authority stub, not a release')
            with self.assertRaisesRegex(RuntimeError,'initialization failure'):rt.run_gpu()
        self.assertEqual(read_json(attempt/'job_failure.json')['status'],'FAILED_NO_AUTOMATIC_RESTART')
        self.assertEqual(read_json(attempt/'job_consumption.json')['ledger']['combinations'],{})
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_device_mask_refused_before_lease_or_cuda(self):
        import os
        status=self.out/'status';status.write_text('isolated authority stub, not a release')
        with patch.object(rt,'build_binding',return_value=({'source_hashes':{}},{})),\
             patch.object(rt,'verify_authority',return_value={'authorized':True}),\
             patch.object(rt,'STATUS',status),\
             patch.object(rt,'exclusive_lock',side_effect=AssertionError('no lease should be acquired')),\
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
            with self.assertRaisesRegex(GateError,'mask exactly'):rt.run_gpu()


if __name__=='__main__':unittest.main()
