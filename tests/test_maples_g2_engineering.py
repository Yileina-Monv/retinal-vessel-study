import copy, tempfile, unittest, random
import numpy as np
from pathlib import Path
from unittest.mock import patch
import torch
from retinal_m0.maples_g2_engineering import *
from retinal_m0.maples_branch_bce import branch_bce_loss, normalized_positive_bce
from retinal_m0 import maples_g2_runtime as rt

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"ledger.json"
        self.sha="ab"*32;self.b=Budget(self.path,self.sha);self.b.initialize_isolated()
    def tearDown(self):self.tmp.cleanup()
    def test_caps_sum_and_write_ahead_failure_charge(self):
        self.assertEqual(sum(TRACK_CAPS.values()),400)
        self.b.reserve("R0_LAMBDA0",1,1)
        s=self.b.snapshot();self.assertEqual(s["charged_events"],0)
        self.assertEqual(s["tracks"]["R0_LAMBDA0"]["pending"]["outcome"],"UNKNOWN")
        self.b.resolve("R0_LAMBDA0","failed")
        s=self.b.snapshot();self.assertEqual(s["charged_events"],1);self.assertEqual(s["failed_events"],1)
        self.assertEqual(s["status"],"FAILED_STOP")
        with self.assertRaises(GateError):self.b.reserve("R0_LAMBDA0",2,1)
    def test_amp_overflow_does_not_charge_and_forces_same_draw_retry(self):
        self.b.reserve("B1_CONT",1,1);self.b.resolve("B1_CONT","amp_overflow",amp_retry=True)
        with self.assertRaisesRegex(GateError,"identical planned draw"):self.b.reserve("B1_CONT",2,2)
        self.b.reserve("B1_CONT",1,2);self.b.resolve("B1_CONT","success")
        s=self.b.snapshot();self.assertEqual(s["charged_events"],1);self.assertEqual(s["amp_retries"],1)
        self.assertEqual(s["verified_successes"],1);self.assertEqual(s["attempts"],2)
    def test_ninth_attempt_stops_and_uncertain_is_charged(self):
        for attempt in range(1,9):
            self.b.reserve("B2_CONT",1,attempt);self.b.resolve("B2_CONT","amp_overflow",amp_retry=True)
        self.b.reserve("B2_CONT",1,9)
        self.b.resolve("B2_CONT","amp_overflow",amp_retry=True)
        s=self.b.snapshot();self.assertEqual(s["uncertain_events"],0);self.assertEqual(s["charged_events"],0)
        self.assertEqual(s["amp_retries"],9);self.assertEqual(s["status"],"FAILED_STOP")
        with self.assertRaisesRegex(GateError,"not runnable"):self.b.reserve("B2_CONT",1,1)
    def test_missing_ledger_never_resets(self):
        self.path.unlink()
        with self.assertRaisesRegex(GateError,"missing"):self.b.snapshot()
        self.assertFalse(self.path.exists())

    def test_full_fixed_allocation_caps_and_global_ceiling(self):
        for track,cap in TRACK_CAPS.items():
            for step in range(1,cap+1):
                self.b.reserve(track,step,1);self.b.resolve(track,"success")
        s=self.b.snapshot()
        self.assertEqual(s["charged_events"],400);self.assertEqual(s["verified_successes"],400)
        with self.assertRaisesRegex(GateError,"exhausted"):self.b.reserve("R0_LAMBDA0",201,1)
    def test_crash_pending_is_not_recovered_or_refunded(self):
        self.b.reserve("B1_SPLIT",1,1)
        second=Budget(self.path,self.sha)
        self.assertIsNotNone(second.snapshot()["tracks"]["B1_SPLIT"]["pending"])
        with self.assertRaisesRegex(GateError,"unresolved"):second.reserve("B1_SPLIT",1,1)
        second.mark_crash_uncertain("B1_SPLIT")
        s=second.snapshot();self.assertEqual(s["charged_events"],1);self.assertEqual(s["uncertain_events"],1)

class DiagnosticTests(unittest.TestCase):
    def test_same_forward_audit_preserves_grad_rng_scaler_and_model(self):
        torch.manual_seed(91)
        model=torch.nn.Conv2d(1,1,1,bias=True)
        x=torch.randn(2,1,4,4);z=model(x)
        target=torch.zeros_like(z);target[:,:,1,1]=1
        base=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
        aux=.1*(-torch.nn.functional.logsigmoid(z[:,:,1,1])).mean()
        opt=torch.optim.AdamW(model.parameters())
        for i,p in enumerate(model.parameters()):p.grad=torch.full_like(p,float(i+1))
        saved=[p.grad.clone() for p in model.parameters()]
        scaler=torch.amp.GradScaler("cpu",init_scale=128)
        rng=torch.get_rng_state().clone();digest=_tensor_digest(model) if "_tensor_digest" in globals() else None
        r=same_forward_gradient_audit(model,z,base,aux,scaler=scaler,audit_step=1)
        self.assertEqual(r["autograd_calls"],2);self.assertTrue(r["unscaled"])
        self.assertEqual(r["pause_ratio_gt_2"],r["aux_to_base_l2"]>2)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),saved)))
        self.assertEqual(scaler.get_scale(),128)
    def test_ratio_over_two_and_zero_base_positive_aux_trigger_pause(self):
        model=torch.nn.Linear(1,1,bias=False)
        x=torch.ones(1,1);z=model(x)
        scaler=torch.amp.GradScaler("cpu",init_scale=64)
        audit=same_forward_gradient_audit(model,z,z.sum()*.001,z.sum()*1.0,scaler=scaler,audit_step=5)
        self.assertGreater(audit["aux_to_base_l2"],2);self.assertTrue(audit["pause_ratio_gt_2"])
        zero=same_forward_gradient_audit(model,z,z.sum()*0,z.sum(),scaler=scaler,audit_step=10)
        self.assertTrue(zero["zero_base_positive_aux"]);self.assertTrue(zero["pause_ratio_gt_2"])
    def test_audit_rejects_unscheduled_step(self):
        m=torch.nn.Linear(2,1);x=torch.ones(1,2);z=m(x);loss=z.square().mean()
        with self.assertRaisesRegex(GateError,"outside frozen"):same_forward_gradient_audit(m,z,loss,loss,scaler=torch.amp.GradScaler("cpu"),audit_step=2)

class RecoveryTests(unittest.TestCase):
    def test_continuous_matches_independent_25_plus_25_restore_with_private_loader(self):
        torch.manual_seed(7)
        initial=torch.nn.Linear(3,1)
        a=copy.deepcopy(initial);b=copy.deepcopy(initial);c=copy.deepcopy(initial)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3);ob=torch.optim.AdamW(b.parameters(),lr=1e-3);oc=torch.optim.AdamW(c.parameters(),lr=1e-3)
        sa=torch.amp.GradScaler("cpu",init_scale=128);sb=torch.amp.GradScaler("cpu",init_scale=128);sc=torch.amp.GradScaler("cpu",init_scale=128)
        ga=torch.Generator().manual_seed(2026);gb=torch.Generator().manual_seed(2026);gc=torch.Generator().manual_seed(5)
        log_a=[];log_b=[];log_c=[]
        def step(model,opt,scaler,g,n,log):
            x=torch.arange(6,dtype=torch.float32).reshape(2,3)/7+torch.rand(1,generator=g)
            loss=model(x).square().mean()
            opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();opt.step()
            log.append({"step":n,"input":x.tolist(),"loss":float(loss.detach())})
        for n in range(1,26):step(a,oa,sa,ga,n,log_a);step(b,ob,sb,gb,n,log_b)
        self.assertEqual(log_a,log_b)
        bundle=checkpoint_bundle(b,ob,sb,track="B1_SPLIT",step=25,next_draw=25,loader_generator=gb)
        py_state=bundle["python_rng"];np_state=bundle["numpy_rng"]
        for n in range(26,51):step(a,oa,sa,ga,n,log_a)
        self.assertEqual(restore_bundle(bundle,c,oc,sc,track="B1_SPLIT",expected_step=25,loader_generator=gc),25)
        for n in range(26,51):step(c,oc,sc,gc,n,log_c)
        self.assertEqual(log_a[25:],log_c)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),c.state_dict().values())))
        def same(x,y):
            if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
            if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
            if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
            return x==y
        self.assertTrue(same(oa.state_dict(),oc.state_dict()))
        self.assertEqual(sa.state_dict(),sc.state_dict())
        self.assertEqual(ga.get_state().tolist(),gc.get_state().tolist())
        self.assertEqual(torch.get_rng_state().tolist(),bundle["torch_cpu_rng"].tolist())
        self.assertEqual(random.getstate(),py_state)
        self.assertEqual(np.random.get_state()[0],np_state[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],np_state[1]))
    def test_missing_private_loader_rng_refuses_restore(self):
        m=torch.nn.Linear(1,1);o=torch.optim.AdamW(m.parameters());s=torch.amp.GradScaler("cpu")
        bad={"track":"B1_SPLIT","step":25,"next_draw":100,"python_rng":None,"numpy_rng":None,"torch_cpu_rng":torch.get_rng_state()}
        with self.assertRaisesRegex(GateError,"loader RNG"):restore_bundle(bad,m,o,s,track="B1_SPLIT",expected_step=25,loader_generator=torch.Generator())


class OptimizerEventTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"l.json";self.sha="cd"*32
        self.b=Budget(self.path,self.sha);self.b.initialize_isolated()
    def tearDown(self):self.tmp.cleanup()
    def test_real_post_step_event_is_counted_and_amp_skip_retries_same_draw(self):
        model=torch.nn.Linear(1,1);opt=torch.optim.SGD(model.parameters(),lr=.1)
        class SkipScaler:
            def step(self,optimizer):pass
            def update(self):pass
        x=torch.ones(1,1);loss=model(x).square().mean();loss.backward()
        before=[p.detach().clone() for p in model.parameters()]
        skipped=optimizer_update(self.b,"B1_CONT",1,1,opt,SkipScaler(),amp_enabled=True)
        self.assertFalse(skipped["draw_advance"]);self.assertTrue(skipped["retry_same_draw"])
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(model.parameters(),before)))
        self.assertEqual(self.b.snapshot()["charged_events"],0)
        result=optimizer_update(self.b,"B1_CONT",1,2,opt,None,amp_enabled=False)
        self.assertTrue(result["draw_advance"])
        s=self.b.snapshot();self.assertEqual(s["charged_events"],1);self.assertEqual(s["verified_successes"],1)
        self.assertEqual(s["amp_retries"],1);self.assertEqual(s["optimizer_events"],1)

class LossTests(unittest.TestCase):
    def test_lambda_zero_exact_bypass_and_empty_sparse_batch_mean(self):
        z=torch.tensor([[[[0.,1.],[2.,-1.]]],[[[.5,-.5],[1.,-2.]]]],requires_grad=True)
        y=torch.zeros_like(z);y[1,0,0,1]=1
        f=torch.ones_like(z);sel=torch.zeros_like(z);sel[1,0,0,1]=1
        base=branch_bce_loss(z,y,f,sel,lam=0.0)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y.float(),reduction="none")
        # ?=0 returns the original A0 implementation's exact tensor; separate arithmetic gradients match too.
        g0=torch.autograd.grad(base,z,retain_graph=True)[0]
        from retinal_m0.maples_core import weighted_masked_loss
        original=weighted_masked_loss(z,y,f,f,epsilon=1e-6)
        g1=torch.autograd.grad(original,z,retain_graph=True)[0]
        self.assertTrue(torch.equal(base,original));self.assertTrue(torch.equal(g0,g1))
        aux=normalized_positive_bce(z,y,f,sel)
        manual=(torch.nn.functional.softplus(-z[1,0,0,1])+z[0].sum()*0)/2
        self.assertTrue(torch.allclose(aux,manual))
        self.assertEqual(float(aux.detach()),float(manual.detach()))

class EntryTests(unittest.TestCase):
    def test_no_args_and_explicit_run_refuses_without_sol_release(self):
        for args in ([],["--cpu-preflight","--run-g2"]):
            with self.assertRaises(SystemExit):rt.main(args)
        with patch.object(rt,"G2_RELEASE",Path("/definitely/missing")):
            with self.assertRaisesRegex(GateError,"Sol-issued"):rt.run_g2()
    def test_wrong_phase_and_policy_hash_are_rejected(self):
        good='<!-- P1_G2_RELEASE_CURRENT -->\n{"policy_id":"P1_B1_B2_G2_ENGINEERING_EXCEPTION_01_20261021","policy_sha256":"d083de28136d82b1c0cdd762afd995efff1ef644eeeff304c4ccce492d731aae","state":"HOLD_CPU_PREFLIGHT","GPU_released":false,"phase":"G2_ONLY","user_approved":true,"GPU_execution_conditionally_authorized":true,"charged_update_ceiling":400,"scientific_authorized":false}\n<!-- /P1_G2_RELEASE_CURRENT -->'
        self.assertEqual(rt.current_g2_state(good)["state"],"HOLD_CPU_PREFLIGHT")
        bad=good.replace("HOLD_CPU_PREFLIGHT","RELEASED")
        with self.assertRaisesRegex(GateError,"not CPU-only"):rt._require_current_hold(bad)
        bad_hash=good.replace("d083de28136d82b1c0cdd762afd995efff1ef644eeeff304c4ccce492d731aae","00"*32)
        with self.assertRaisesRegex(GateError,"policy/status binding mismatch"):rt.current_g2_state(bad_hash)
    def test_cpu_preflight_code_has_no_cuda_or_model_execution(self):
        source=Path(rt.__file__).read_text()
        body=source.split("def cpu_preflight():",1)[1].split("def run_g2():",1)[0]
        self.assertNotIn("torch.load(",source);self.assertNotIn("UNet(",source)
        self.assertNotIn("torch.cuda.init",body);self.assertNotIn("torch.cuda.set_device",body)

if __name__=="__main__":unittest.main()

