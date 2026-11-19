"""CPU-only tests of candidate-C scope, release binding, counters, and guards."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest import mock
from retinal_m0 import maples_soft_cldice_gpu_diag as d
from retinal_m0.maples_g1_safety import Ledger,GateError,atomic_json,canonical_sha,file_sha,ledger_snapshot


class CandidateDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.paths={k:self.root/v for k,v in {'CODE':'code','CPU':'cpu','BINDING':'binding.json','RELEASE':'release.json','RECEIPT':'receipt.json','LEDGER':'ledger.json','RETAINED':'retained.json'}.items()}
        self.paths['CODE'].mkdir()
        self.patch=mock.patch.multiple(d,**self.paths);self.patch.start()
        d.CPU.mkdir();atomic_json(d.CPU/'candidate_C_engineering_config.json',{'eta':.15115714003619224,'k':26})
        self.source=self.root/'source.txt';self.source.write_text('original')
        self.evidence=self.root/'test_evidence.txt';self.evidence.write_text('passed')
        self.binding={'phase':d.PHASE,'plan':d.plan(),'parameter_updates':0,'max_forwards':32,'gpu_uuid':d.GPU_UUID,
                      'execution_plan':[x for x in d.plan() if x['id']!=d.ACKNOWLEDGED_OOM],
                      'sources_sha256':{str(self.source):file_sha(self.source)},'runtime_guard_paths':[str(self.source)],
                      'eta':.15115714003619224,'k':26,'precision_rules':d.precision_rules(),'allocator_memory_cap_bytes':d.MEMORY_CAP,'wall_seconds_cap':d.WALL_SECONDS}
        atomic_json(d.BINDING,self.binding)
        atomic_json(d.RECEIPT,{'status':'CPU_PREFLIGHT_ACCEPTED_GPU_NOT_YET_EXECUTED','binding_sha256':canonical_sha(self.binding),'tests_passed':True,'test_evidence_sha256':{str(self.evidence):file_sha(self.evidence)}})
        self.release={'release_id':'sol_C_cldice_gpu_diag_20261023_02','phase':d.PHASE,'authorized':True,'max_forwards':32,'parameter_updates':0,
                      'gpu_uuid':d.GPU_UUID,'binding_sha256':canonical_sha(self.binding),'binding_file_sha256':file_sha(d.BINDING),'cpu_preflight_review_sha256':file_sha(d.RECEIPT),'ledger_path':str(d.LEDGER)}
        atomic_json(d.RELEASE,self.release);self.save_status();self.empty_ledger()
        self.retained_fixture()

    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def save_status(self):
        state={'state':'RELEASED','phase':d.PHASE,'release_id':self.release['release_id'],'binding_sha256':canonical_sha(self.binding),'release_file_sha256':file_sha(d.RELEASE)}
        (d.CODE/'STATUS.md').write_text(d.START+'\n'+json.dumps(state)+'\n'+d.END)

    def empty_ledger(self):
        atomic_json(d.LEDGER,{'schema':2,'cap':32,'binding_id':canonical_sha(self.binding),'combinations':{},'forward_starts':0,'autograd_calls':0,'optimizer_step_attempts':0,'blocked_forward_attempts':0})

    def retained_fixture(self):
        s=d.read_json(d.LEDGER);s['forward_starts']=1;s['autograd_calls']=3
        s['combinations']={d.ACKNOWLEDGED_OOM:{'status':'failed','owner':{'pid':1,'start_time':'fixture'},'forward_starts':1,'autograd_calls':3,'failure':'retained diagnostic OOM'}}
        atomic_json(d.LEDGER,s);atomic_json(d.RETAINED,s)

    def ledger(self):
        self.empty_ledger()
        return Ledger(d.LEDGER,canonical_sha(self.binding),[x['id'] for x in d.plan()])

    def test_plan_exact32_unique_and_ordered(self):
        p=d.plan();self.assertEqual(len({x['id'] for x in p}),32)
        self.assertEqual([(x['precision'],x['mode']) for x in p[:4]],[('FP32','eta0'),('FP32','candidate'),('AMP','eta0'),('AMP','candidate')])

    def test_valid_authority(self):self.assertEqual(d.verify_authority(),self.binding)

    def test_missing_duplicate_status_rejected(self):
        for text in ['',d.START+d.END+d.START+d.END]:
            with self.subTest(text=text),self.assertRaises(GateError):d.status_release(text)

    def test_duplicate_JSON_key_rejected(self):
        with self.assertRaises(GateError):d.status_release(d.START+'{"state":"HOLD","state":"RELEASED"}'+d.END)

    def test_eta_k_and_numeric_policy_drift_rejected(self):
        for field,value in [('eta',.03),('k',10),('wall_seconds_cap',999999),('allocator_memory_cap_bytes',999999999999),('precision_rules',{})]:
            bad=dict(self.binding);bad[field]=value;atomic_json(d.BINDING,bad)
            with self.subTest(field=field),self.assertRaises(GateError):d.verify_authority()
        atomic_json(d.BINDING,self.binding)

    def test_scope_stage_budget_update_rejected(self):
        for field,value in [('phase','G2'),('max_forwards',33),('parameter_updates',1),('gpu_uuid','GPU1'),('authorized',False)]:
            bad=dict(self.release);bad[field]=value;atomic_json(d.RELEASE,bad)
            with self.subTest(field=field),self.assertRaises(GateError):d.verify_authority()
        atomic_json(d.RELEASE,self.release)

    def test_stale_release_status_rejected(self):
        self.release['release_id']='old_G1_release';atomic_json(d.RELEASE,self.release)
        with self.assertRaises(GateError):d.verify_authority()

    def test_source_and_test_evidence_drift_rejected(self):
        self.source.write_text('changed')
        with self.assertRaises(GateError):d.verify_authority()
        self.source.write_text('original');self.evidence.write_text('changed')
        with self.assertRaises(GateError):d.verify_authority()

    def test_CPU_receipt_not_passed_rejected(self):
        rec=d.read_json(d.RECEIPT);rec['tests_passed']=False;atomic_json(d.RECEIPT,rec)
        self.release['cpu_preflight_review_sha256']=file_sha(d.RECEIPT);atomic_json(d.RELEASE,self.release);self.save_status()
        with self.assertRaises(GateError):d.verify_authority()

    def test_failure_prevents_continuation_and_preserves_counts(self):
        l=self.ledger();p=d.plan();l.reserve(p[0]['id']);l.before_forward(p[0]['id']);l.autograd(p[0]['id']);l.finish(p[0]['id'],error='synthetic_failure')
        with self.assertRaises(GateError):l.reserve(p[1]['id'])
        snap=ledger_snapshot(d.LEDGER);self.assertEqual(snap['forward_starts'],1);self.assertEqual(snap['autograd_calls'],1)
        with self.assertRaises(GateError):l.reserve(p[0]['id'])

    def test_second_forward_is_blocked_before_body(self):
        l=self.ledger();cid=d.plan()[0]['id'];l.reserve(cid);l.before_forward(cid)
        with self.assertRaises(GateError):l.before_forward(cid)
        s=ledger_snapshot(d.LEDGER);self.assertEqual(s['forward_starts'],1);self.assertEqual(s['blocked_forward_attempts'],1)

    def test_32_cap_and_duplicate_not_reset(self):
        l=self.ledger()
        for c in d.plan():l.reserve(c['id']);l.before_forward(c['id']);l.finish(c['id'])
        with self.assertRaises(GateError):l.reserve(d.plan()[0]['id'])
        self.assertEqual(ledger_snapshot(d.LEDGER)['forward_starts'],32)

    def test_four_gradient_call_cap(self):
        l=self.ledger();cid=d.plan()[0]['id'];l.reserve(cid);l.before_forward(cid)
        for _ in range(4):l.autograd(cid)
        with self.assertRaises(GateError):l.autograd(cid)
        self.assertEqual(ledger_snapshot(d.LEDGER)['autograd_calls'],4)

    def test_optimizer_step_rejected_and_recorded(self):
        import torch
        l=self.ledger();cid=d.plan()[0]['id'];l.reserve(cid);l.before_forward(cid)
        p=torch.nn.Parameter(torch.tensor(2.));opt=torch.optim.AdamW([p]);before=p.clone()
        with d.block_optimizers(torch,l,{'id':cid}):
            with self.assertRaises(GateError):opt.step()
        self.assertTrue(torch.equal(before,p));self.assertEqual(ledger_snapshot(d.LEDGER)['optimizer_step_attempts'],1)

    def test_cli_training_flags_rejected(self):
        for args in [[],['--train'],['--run-gpu-diagnostics','--updates','1'],['--run-gpu-diagnostics','--gpu','1']]:
            with mock.patch('sys.argv',['diag',*args]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as c:d.main()
            self.assertEqual(c.exception.code,2)

    def test_eta0_does_not_build_normal_full_auxiliary_graph(self):
        import torch
        import retinal_m0.maples_soft_cldice as loss
        z=torch.randn(4,1,32,32,requires_grad=True);g=torch.zeros_like(z);g[:,:,16,4:28]=1;f=torch.ones_like(z)
        with mock.patch.object(loss,'soft_cldice_terms',wraps=loss.soft_cldice_terms) as spy:
            base,zero,terms,aux,total=d.objectives(z,g,f,k=3,eta=0)
            self.assertEqual(spy.call_count,0)
        self.assertTrue(torch.equal(base,total));self.assertEqual(aux.item(),0);self.assertIsNone(terms['loss'])
        self.assertTrue(torch.equal(torch.autograd.grad(base,z,retain_graph=True)[0],torch.autograd.grad(zero,z)[0]))

    def test_candidate_builds_exactly_one_full_auxiliary_graph(self):
        import torch
        import retinal_m0.maples_soft_cldice as loss
        z=torch.randn(4,1,32,32,requires_grad=True);g=torch.zeros_like(z);g[:,:,16,4:28]=1;f=torch.ones_like(z)
        with mock.patch.object(loss,'soft_cldice_terms',wraps=loss.soft_cldice_terms) as spy:
            _,_,terms,_,total=d.objectives(z,g,f,k=3,eta=.15115714003619224)
            self.assertEqual(spy.call_count,1)
        self.assertTrue(torch.isfinite(torch.autograd.grad(total,z)[0]).all())

    def test_acknowledged_failure_cannot_be_replayed_or_erased(self):
        l=d.CandidateLedger(d.LEDGER,canonical_sha(self.binding),[x['id'] for x in d.plan()])
        with self.assertRaises(GateError):l.reserve(d.ACKNOWLEDGED_OOM)
        cid=d.plan()[1]['id'];l.reserve(cid);l.before_forward(cid);l.finish(cid,error='new_failure')
        with self.assertRaises(GateError):l.reserve(d.plan()[2]['id'])
        s=ledger_snapshot(d.LEDGER);self.assertEqual(s['forward_starts'],2);self.assertEqual(s['combinations'][d.ACKNOWLEDGED_OOM],d.read_json(d.RETAINED)['combinations'][d.ACKNOWLEDGED_OOM])

if __name__=='__main__':unittest.main(verbosity=2)
