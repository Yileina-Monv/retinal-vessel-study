import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from paths import *
import ast, copy, json, math, tempfile, traceback
import torch
from diagnostic import bounded_gradients, DiagnosticFailure
from fast_observer import digest, rng_pack, Journal, AmpLimits, StepEvidence
from common import atomic, sha
from recovery import recovery_manifest

def main():
    torch.set_num_threads(1)
    def forbidden(*args, **kw): raise AssertionError("CUDA prohibited in CPU tests")
    torch.cuda._lazy_init = forbidden
    results=[]
    def test(name, fn):
        fn();results.append(name)
    def diagnose(value, scale, component="base"):
        rows=[]
        grads, meta=bounded_gradients(value[0], value[1], scale, component, rows.append)
        return grads, meta, rows
    def finite():
        p=torch.nn.Parameter(torch.tensor(2.))
        rng=digest(rng_pack())
        loss=p.square()
        a,meta,rows=diagnose((loss,[("p",p)]),2097152.)
        assert a[0].item()==4 and meta["tries"]==1 and p.grad is None
        assert digest(rng_pack())==rng
    test("finite_gradient_same_as_autograd_no_RNG_or_grad_accumulation",finite)
    def overflow():
        p=torch.nn.Parameter(torch.tensor(.5))
        loss=p.to(torch.float16).float()*.04
        before=p.detach().clone()
        grads,meta,rows=diagnose((loss,[("p",p)]),2097152.)
        assert len(rows)==2 and not rows[0]["finite"] and rows[1]["finite"]
        assert meta["accepted_scale"]==1048576. and abs(grads[0].item()-.04)<1e-4
        assert torch.equal(p,before) and p.grad is None
        loss.backward()
        assert abs(p.grad.item()-.04)<1e-4 # same graph retained for actual training backward
    test("real_FP16_CPU_overflow_recovered_at_half_scale_same_graph",overflow)
    def final_retry():
        p=torch.nn.Parameter(torch.tensor(.5))
        loss=p.to(torch.float16).float()*.2
        _,meta,rows=diagnose((loss,[("p",p)]),2097152.)
        assert meta["tries"]==4 and meta["accepted_scale"]==262144.
    test("fourth_diagnostic_scale_recovers_and_limit_is_bounded",final_retry)
    def persistent():
        p=torch.nn.Parameter(torch.tensor(0.))
        rows=[]
        try:bounded_gradients(torch.sqrt(p),[("p",p)],2097152.,"base",rows.append)
        except DiagnosticFailure:pass
        else:raise AssertionError("persistent Inf accepted")
        assert len(rows)==4 and all(not r["finite"] for r in rows) and p.grad is None
    test("persistent_true_nonfinite_gradient_stops_after_four",persistent)
    def empty():
        p=torch.nn.Parameter(torch.tensor([1.,2.]))
        g,m,r=diagnose((p.sum()*0,[("p",p)]),128.,"aux")
        assert torch.equal(g[0],torch.zeros_like(p)) and m["tries"]==1
        q=torch.nn.Parameter(torch.tensor(3.))
        grads,_,_=diagnose((p.sum(),[("p",p),("unused",q)]),65536.)
        assert grads[1] is None
    test("graph_connected_empty_aux_and_unused_parameter",empty)
    def refuse():
        p=torch.nn.Parameter(torch.tensor(1.))
        for scale in [0.,float("nan"),float("inf"),-1.]:
            try:diagnose((p.square(),[("p",p)]),scale)
            except DiagnosticFailure:pass
            else:raise AssertionError("bad scale allowed")
        p.grad=torch.tensor(2.)
        try:diagnose((p.square(),[("p",p)]),128.)
        except DiagnosticFailure:pass
        else:raise AssertionError("existing grad allowed")
    test("invalid_scale_and_existing_gradient_rejected",refuse)
    def nanloss():
        p=torch.nn.Parameter(torch.tensor(1.));rows=[]
        try:bounded_gradients(p*float("nan"),[("p",p)],128.,"base",rows.append)
        except DiagnosticFailure:pass
        else:raise AssertionError("NaN loss accepted")
        assert not rows
    test("nonfinite_forward_loss_is_not_retried",nanloss)
    def actual_amp():
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1,momentum=.9)
        scaler=torch.amp.GradScaler("cpu",init_scale=2097152.)
        opt.zero_grad(set_to_none=True);loss=p.square()
        initial=copy.deepcopy(scaler.state_dict());g,_,_=diagnose((loss,[("p",p)]),scaler.get_scale())
        assert scaler.state_dict()==initial and p.grad is None
        scaler.scale(loss).backward();p.grad.fill_(float("inf"))
        before=digest((p.detach(),opt.state_dict()))
        scaler.unscale_(opt);scaler.step(opt);scaler.update()
        assert digest((p.detach(),opt.state_dict()))==before and scaler.get_scale()==1048576.
    test("real_CPU_GradScaler_native_skip_and_backoff_unchanged",actual_amp)
    def journal():
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            j=Journal(Path(tmp)/"budget.json","CPU_TEST",1,250,50,AmpLimits(5,25));j.initialize()
            j.start("train",0);j.compute("forward");j.stop("interruption")
            assert j.read()["attempts"]==1 and j.read()["unknown"]==1
            try:j.start("train",0)
            except RuntimeError:pass
            else:raise AssertionError("stopped ledger bypass")
            try:j.initialize()
            except RuntimeError:pass
            else:raise AssertionError("ledger reset")
    test("original_stopped_ledger_no_refund_reset_or_auto_resume",journal)
    def source():
        original=ast.parse((PARENT/"nl_observer.py").read_text())
        revised=ast.parse((HERE/"nl_observer_repaired.py").read_text())
        def method(tree,name):
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef))
            return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
        a=method(original,"native_step");b=method(revised,"native_step")
        # Only logged diagnostic scale fields change, never scientific computation.
        row_a=next(n for n in a.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="row" for t in n.targets))
        row_b=next(n for n in b.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="row" for t in n.targets))
        a.body.remove(row_a);b.body.remove(row_b)
        assert ast.dump(a)==ast.dump(b)
        for name in ["backward","optimizer","train_step","monitor"]:
            assert ast.dump(method(original,name))==ast.dump(method(revised,name))
        try:
            import importlib.util
            spec=importlib.util.spec_from_file_location("repair_adapter",HERE/"nl_observer_repaired.py")
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            mod.NLObserver(None)
        except RuntimeError as e:assert "independent repair training release" in str(e)
        else:raise AssertionError("old release admitted new training adapter")
    test("native_training_and_ratio2_guard_unchanged_new_release_required",source)
    def skip_policy():
        from revised_policy import classify_step
        e=StepEvidence(1.,float("inf"),2097152.,1048576.,0,True,True)
        assert classify_step(e)[0]=="AMP_SKIP"
        assert classify_step(StepEvidence(float("nan"),float("inf"),2097152.,1048576.,0,True,True))[0]=="FATAL"
        assert classify_step(StepEvidence(1.,float("inf"),2097152.,2097152.,0,True,True))[0]=="FATAL"
    test("original_AMP_skip_vs_fatal_rules_unchanged",skip_policy)
    def adapter_diagnostic():
        import importlib.util
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location("adapter_integration",HERE/"nl_observer_repaired.py")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        network=torch.nn.Linear(1,1,bias=False)
        network.weight.data.fill_(.5)
        obj=mod.NLObserver.__new__(mod.NLObserver)
        obj.t=SimpleNamespace(network=network,grad_scaler=torch.amp.GradScaler("cpu",init_scale=2097152.))
        obj.j=SimpleNamespace(read=lambda:{"pending":{"optimizer":0}})
        obj.active=10028;obj.authorize=lambda:None
        rows=[];mod.log=lambda path,row:rows.append(row)
        before=digest((network.state_dict(),obj.t.grad_scaler.state_dict(),rng_pack()))
        value=network.weight.to(torch.float16).float().sum()*.04
        grads=obj.diagnostic_grad(value,"base")
        assert len(rows)==2 and obj.diagnostic_meta["base"]["accepted_scale"]==1048576.
        assert abs(grads[0].item()-.04)<1e-4
        assert obj.diag_mode is False and obj.component is None
        assert digest((network.state_dict(),obj.t.grad_scaler.state_dict(),rng_pack()))==before
        assert all(p.grad is None for p in network.parameters())
        try:obj.diagnostic_grad(value*float("nan"),"base")
        except DiagnosticFailure:pass
        else:raise AssertionError("adapter admitted NaN")
        assert obj.diag_mode is False and obj.component is None
    test("actual_repaired_observer_method_integrates_logs_and_preserves_state",adapter_diagnostic)
    def source_restore():
        tree=ast.parse((HERE/"recovery.py").read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="step" for n in ast.walk(tree))
    test("restore_path_never_steps_optimizer_or_scheduler",source_restore)
    assert not torch.cuda.is_initialized()
    atomic(OUT/"CPU_TESTS.json",{"status":"PASS","cases":results,"case_count":len(results),
        "CUDA_initialized":False,"scientific_model_forwards":0,"scientific_updates":0,
        "GPU_equivalence_proved":False},True)
    print("CPU_TESTS_PASS",len(results),flush=True)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        atomic(OUT/"CPU_TEST_FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc()},True)
        raise
