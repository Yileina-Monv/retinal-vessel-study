"""CPU-only production ledger, release, resource and numeric boundary tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile,unittest
from pathlib import Path
from unittest import mock

from retinal_m0 import maples_focal_dice_gpu_diag as d


class DiagnosticSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        patches={k:root/name for k,name in [('CODE','code'),('OUT','out'),('C_STAGE','C'),
                 ('BINDING','binding.json'),('RELEASE','release.json'),('RECEIPT','receipt.json'),('LEDGER_PATH','ledger.json')]}
        self.patch=mock.patch.multiple(d,**patches);self.patch.start()
        for q in [d.CODE,d.OUT,d.C_STAGE]:q.mkdir()
        self.source=root/'source.txt';self.source.write_text('fixed-source')
        self.evidence=root/'cpu.txt';self.evidence.write_text('CPU passed')
        self.binding={'scope':d.scope(),'plan':d.plan(),'rules':d.rules(),
                      'source_sha256':{str(self.source):d.file_sha(self.source)},
                      'runtime_guard_paths':[str(self.source)],'C_dependency_binding_sha256':'C_binding'}
        d.atomic_json(d.BINDING,self.binding)
        d.atomic_json(d.RECEIPT,{'status':'CPU_PREFLIGHT_ACCEPTED_GPU_NOT_EXECUTED',
                     'binding_sha256':d.canonical_sha(self.binding),'all_tests_passed':True,
                     'evidence_sha256':{str(self.evidence):d.file_sha(self.evidence)}})
        self.release={'phase':d.PHASE,'authorized':True,'scope':d.scope(),
                      'binding_sha256':d.canonical_sha(self.binding),'binding_file_sha256':d.file_sha(d.BINDING),
                      'CPU_review_sha256':d.file_sha(d.RECEIPT),'ledger_path':str(d.LEDGER_PATH),'release_id':'test_D_only'}
        d.atomic_json(d.RELEASE,self.release);self.save_status()
        d.atomic_json(d.LEDGER_PATH,d.empty_ledger(d.canonical_sha(self.binding)))

    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def save_status(self,**changes):
        state={'state':'RELEASED','phase':d.PHASE,'release_id':self.release['release_id'],
               'binding_sha256':d.canonical_sha(self.binding),'release_file_sha256':d.file_sha(d.RELEASE)}
        state.update(changes)
        (d.CODE/'STATUS.md').write_text('history\n'+d.START+'\n'+json.dumps(state)+'\n'+d.END+'\nC unchanged')

    def ledger(self):return d.DiagnosticLedger(d.LEDGER_PATH,d.canonical_sha(self.binding))

    def full_combo(self,combo):
        ledger=self.ledger();ledger.reserve(combo);ledger.forward(combo)
        for _ in range(6):ledger.gradient(combo)
        ledger.finish(combo)

    def test_exact16_plan_and_precision_order(self):
        plan=d.plan();self.assertEqual(len(plan),16);self.assertEqual(len({x['id'] for x in plan}),16)
        self.assertEqual([(x['batch_index'],x['precision']) for x in plan],[(i,p) for i in range(8) for p in ['FP32','AMP']])

    def test_valid_release_bound_to_CPU_and_status(self):self.assertEqual(d.verify_authority(),self.binding)

    def test_missing_release_refused(self):
        d.RELEASE.unlink()
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_hold_not_authorization(self):
        self.save_status(state='HOLD')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_consumed_not_relaunchable(self):
        self.save_status(state='CONSUMED')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_other_phase_not_accepted(self):
        self.save_status(phase='C_SCIENCE')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_source_drift_refused(self):
        self.source.write_text('changed')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_CPU_evidence_drift_refused(self):
        self.evidence.write_text('changed')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_release_scope_mismatch_refused(self):
        q=d.read_json(d.RELEASE);q['scope']['parameter_updates']=1;d.atomic_json(d.RELEASE,q)
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_binding_digest_mismatch_refused(self):
        q=d.read_json(d.BINDING);q['extra']='changed';d.atomic_json(d.BINDING,q)
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_exact_cap_persists_across_instances(self):
        for combo in d.plan():self.full_combo(combo['id'])
        ledger=self.ledger();state=ledger.snapshot()
        self.assertEqual((state['forward_starts'],state['autograd_calls']),(16,96))
        with self.assertRaises(d.GateError):ledger.reserve('b00_FP32')
        self.assertEqual((self.ledger().snapshot()['forward_starts'],self.ledger().snapshot()['autograd_calls']),(16,96))

    def test_unknown_or_duplicate_combo_refused(self):
        with self.assertRaises(d.GateError):self.ledger().reserve('not_planned')
        self.full_combo('b00_FP32')
        with self.assertRaises(d.GateError):self.ledger().reserve('b00_FP32')

    def test_second_model_forward_blocked_and_recorded(self):
        l=self.ledger();l.reserve('b00_FP32');l.forward('b00_FP32')
        with self.assertRaises(d.GateError):l.forward('b00_FP32')
        self.assertEqual(l.snapshot()['forward_starts'],1);self.assertEqual(l.snapshot()['blocked_forward_attempts'],1)

    def test_seventh_gradient_blocked(self):
        l=self.ledger();l.reserve('b00_FP32');l.forward('b00_FP32')
        for _ in range(6):l.gradient('b00_FP32')
        with self.assertRaises(d.GateError):l.gradient('b00_FP32')
        self.assertEqual(l.snapshot()['autograd_calls'],6)

    def test_failure_preserves_counts_and_stops(self):
        l=self.ledger();l.reserve('b00_FP32');l.forward('b00_FP32');l.gradient('b00_FP32');l.fail('b00_FP32','test failure')
        self.assertEqual((l.snapshot()['forward_starts'],l.snapshot()['autograd_calls']),(1,1))
        with self.assertRaises(d.GateError):self.ledger().reserve('b00_AMP')
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_counter_tamper_refused(self):
        self.full_combo('b00_FP32');s=d.read_json(d.LEDGER_PATH);s['forward_starts']=0;d.atomic_json(d.LEDGER_PATH,s)
        with self.assertRaises(d.GateError):self.ledger().snapshot()

    def test_boolean_or_negative_counter_refused(self):
        saved=d.read_json(d.LEDGER_PATH)
        for value in [False,-1]:
            altered=dict(saved,forward_starts=value);d.atomic_json(d.LEDGER_PATH,altered)
            with self.assertRaises(d.GateError):self.ledger().snapshot()

    def test_duplicate_JSON_status_key_refused(self):
        (d.CODE/'STATUS.md').write_text(d.START+'{"state":"HOLD","state":"RELEASED"}'+d.END)
        with self.assertRaises(d.GateError):d.current_status((d.CODE/'STATUS.md').read_text())

    def test_no_implicit_ledger_reset(self):
        d.LEDGER_PATH.unlink()
        with self.assertRaises(d.GateError):d.verify_authority()
        self.assertFalse(d.LEDGER_PATH.exists())

    def test_failed_pending_blocks_other_combinations(self):
        l=self.ledger();l.reserve('b00_FP32')
        with self.assertRaises(d.GateError):l.reserve('b00_AMP')

    def test_optimizer_guard_prevents_real_CPU_update(self):
        import torch
        p=torch.nn.Parameter(torch.tensor(2.));opt=torch.optim.SGD([p],lr=1);p.grad=torch.tensor(3.)
        with d.block_optimizers(torch,self.ledger()):
            with self.assertRaises(d.GateError):opt.step()
        self.assertEqual(float(p),2.);self.assertEqual(self.ledger().snapshot()['optimizer_step_attempts'],1)

    def test_duplicate_STATUS_blocks_refused(self):
        p=d.CODE/'STATUS.md';p.write_text(p.read_text()+d.START+d.END)
        with self.assertRaises(d.GateError):d.verify_authority()

    def test_resource_query_scoped_only_GPU0_and_busy_not_killed(self):
        outputs=['0, '+d.GPU_UUID+', 11264, 7984, 99\n','1234\n']
        with mock.patch.object(d.subprocess,'check_output',side_effect=outputs) as run:
            self.assertIsNone(d.resources(True))
        self.assertTrue(all('--id=0' in call.args[0] for call in run.call_args_list))

    def test_wrong_GPU_UUID_rejected(self):
        with mock.patch.object(d.subprocess,'check_output',return_value='0, wrong, 11264, 100, 0'):
            with self.assertRaises(d.GateError):d.resources(True)

    def test_AMP_tolerance_boundaries_and_near_zero(self):
        q={'reference_loss':1.,'loss_absolute':.001,'loss_relative':.001,'total_gradient':{'reference_L2':1.,'current_L2':1.,'difference_L2':.01,'relative_L2':.01,'cosine':.999}}
        d.validate_amp(q)
        q['total_gradient']['relative_L2']=.11
        with self.assertRaises(d.GateError):d.validate_amp(q)
        q={'reference_loss':0.,'loss_absolute':1e-8,'loss_relative':1e20,'total_gradient':{'reference_L2':0.,'current_L2':0.,'difference_L2':0.,'relative_L2':0.,'cosine':None}}
        d.validate_amp(q)

    def seed_C(self,done=False):
        d.atomic_json(d.C_STAGE/'RUN_SESSION.json',{'binding_sha256':'C_binding','controller_pid':123,'controller_token':'123:99'})
        d.atomic_json(d.C_STAGE/'heartbeat.json',{'status':'COMPLETE' if done else 'RUNNING','step':2000 if done else 746})
        if done:d.atomic_json(d.C_STAGE/'completion.json',{'status':'C2000_TRAINED_AND_EVALUATED','updates':2000,'binding_sha256':'C_binding'})

    def test_wait_for_C_without_GPU_calls(self):
        self.seed_C()
        with mock.patch.object(d,'owner_alive',return_value=True),mock.patch.object(d,'resources') as resource:
            self.assertFalse(d.dependency_ready(self.binding)[0]);resource.assert_not_called()

    def test_C_normal_completion_requires_controller_exit(self):
        self.seed_C(True)
        with mock.patch.object(d,'owner_alive',return_value=True):self.assertFalse(d.dependency_ready(self.binding)[0])
        with mock.patch.object(d,'owner_alive',return_value=False):self.assertTrue(d.dependency_ready(self.binding)[0])

    def test_C_disappears_without_completion_stops_queue(self):
        self.seed_C()
        with mock.patch.object(d,'owner_alive',return_value=False):
            with self.assertRaises(d.GateError):d.dependency_ready(self.binding)

    def test_status_update_preserves_other_text(self):
        original=(d.CODE/'STATUS.md').read_text();d.update_status(run_state='WAITING')
        changed=(d.CODE/'STATUS.md').read_text()
        self.assertTrue(changed.startswith('history\n') and changed.endswith('C unchanged'))
        self.assertEqual(d.current_status(changed)['run_state'],'WAITING')


if __name__=='__main__':unittest.main(verbosity=2)
