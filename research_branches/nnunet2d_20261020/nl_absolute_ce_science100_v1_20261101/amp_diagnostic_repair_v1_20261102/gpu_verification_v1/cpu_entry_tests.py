import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from support import *
import ast,copy,tempfile
def main():
    cases=[]
    def check(name,fn):fn();cases.append(name)
    def refused(fn):
        try:fn()
        except RuntimeError:return
        raise AssertionError("denial missing")
    check("missing_release_refused_without_GPU",lambda:refused(validate))
    def ledger():
        s={"status":"RUNNING","reserved":0,"actual_forwards":0,"complete":0,"optimizer_updates":0,"pending":None}
        for index in range(4):
            reserve_forward(s,index)
            refused(lambda:reserve_forward(copy.deepcopy(s),index+1))
            mark_forward(s,index)
            refused(lambda:mark_forward(copy.deepcopy(s),index))
            refused(lambda:complete_case(copy.deepcopy(s),index+1))
            complete_case(s,index)
        assert s["reserved"]==s["actual_forwards"]==s["complete"]==4
        refused(lambda:reserve_forward(s,4))
    check("four_cumulative_forwards_no_duplicates_or_extra_cases",ledger)
    def interrupted():
        s={"status":"RUNNING","reserved":0,"actual_forwards":0,"complete":0,"optimizer_updates":0,"pending":None}
        reserve_forward(s,0);s["status"]="STOPPED"
        with tempfile.TemporaryDirectory(dir=WORK) as p:
            atomic(Path(p)/"ledger.json",s)
            reloaded=read(Path(p)/"ledger.json")
            assert reloaded["reserved"]==1 and reloaded["complete"]==0
            refused(lambda:reserve_forward(reloaded,0))
            refused(lambda:mark_forward(reloaded,0))
    check("persistent_failed_reservation_is_not_refunded",interrupted)
    def ratios():
        assert ratio_guard(1,2,1)==2 and ratio_guard(0,0,.1)==0
        refused(lambda:ratio_guard(1,2.001,1))
        refused(lambda:ratio_guard(0,1,.1))
        refused(lambda:ratio_guard(float("nan"),1,.1))
    check("unchanged_ratio_guard_and_zero_norm",ratios)
    def source():
        for p in HERE.glob("*.py"):ast.parse(p.read_text())
        worker=(HERE/"worker.py").read_text();control=(HERE/"controller.py").read_text()
        assert worker.count("output=t.network(data)")==1
        assert "register_step_pre_hook(forbidden_step)" in worker
        assert "t.grad_scaler.step(t.optimizer)" in worker and "if not math.isfinite(norm):" in worker
        assert "if meta[\"tries\"]>=4:" in worker
        assert "os.killpg(child.pid,signal.SIGTERM)" in control and ">1800" in control
        assert 'a14_gpu1.lock' in control and 'start_new_session=True' in control
        assert 'if (WORK/"budget.json").exists()' in control
        assert '10240*1048576' in worker
    check("GPU_entry_resource_scope_and_no_training_entry",source)
    import torch
    torch.set_num_threads(1)
    def hook():
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1);hit=[]
        def reject(*a,**k):hit.append(True);raise RuntimeError("no optimizer update")
        h=opt.register_step_pre_hook(reject);p.grad=torch.tensor(1.)
        refused(opt.step);assert p.item()==1 and len(hit)==1
        h.remove()
    check("real_optimizer_prehook_prevents_parameter_update",hook)
    assert not torch.cuda.is_initialized()
    atomic(WORK/"CPU_ENTRY_TESTS.json",{"status":"PASS","cases":cases,"GPU_calls":0,"science_updates":0},True)
    print("GPU_ENTRY_CPU_PASS",len(cases))
if __name__=="__main__":main()
