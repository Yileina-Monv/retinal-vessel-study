import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from support import *
import ast,copy,tempfile,traceback
import torch
from journal import ResumeJournal
from fast_observer import StepEvidence,digest
from resume_loop import run_remainder
from unittest.mock import patch
from types import SimpleNamespace

def main():
 torch.set_num_threads(1)
 results=[]
 from torch.optim.optimizer import register_optimizer_step_post_hook
 toy_calls=[0]
 def count_step(*a,**kw):toy_calls[0]+=1
 toy_handle=register_optimizer_step_post_hook(count_step)
 def test(name,fn):fn();results.append(name)
 def refused(fn):
  try:fn()
  except RuntimeError:return
  raise AssertionError("expected rejection")
 def journal_fixture(folder,seed):
  root=Path(folder)/str(seed);root.mkdir();atomic(root/"budget.json",read(parent_stage(seed)/"budget.json"))
  tail=root/"tail";tail.mkdir()
  j=ResumeJournal(tail/"budget.json","CPU")
  j.migrate(read(root/"budget.json"),sha(root/"budget.json"));return j
 def migration():
  with tempfile.TemporaryDirectory(dir=WORK) as tmp:
   for seed in [2027,2028]:
    j=journal_fixture(tmp,seed);old=read(parent_stage(seed)/"budget.json");s=j.read()
    for key in ["attempts","success","skip","unknown","monitor_complete","epoch","iteration"]:
     assert s[key]==old[key]
    logical=old["success"]+old["skip"]+1
    assert j.start("train",old["epoch"])==logical
    assert j.read()["attempts"]==old["attempts"]+1
    assert j.read()["pending"]["number"]==old["attempts"]+1
    assert j.read()["pending"]["logical_number"]==logical
    j.compute("forward");j.compute("backward");j.compute("optimizer")
    kind,_=j.resolve_train(StepEvidence(1.,1.,2097152.,2097152.,1,True,True),True)
    assert kind=="SUCCESS" and j.read()["success"]==old["success"]+1 and j.read()["unknown"]==1
    refused(j.initialize)
    refused(lambda:j.migrate(old,sha(j.parent_path)))
 test("both_actual_parent_budgets_migrate_without_refund_logical_and_physical_separate",migration)
 def skip_and_failure():
  with tempfile.TemporaryDirectory(dir=WORK) as tmp:
   j=journal_fixture(tmp,2028);old=j.read();number=j.start("train",old["epoch"])
   j.compute("forward");j.compute("backward")
   kind,_=j.resolve_train(StepEvidence(1.,float("inf"),2097152.,1048576.,0,True,True),False)
   assert kind=="AMP_SKIP" and j.read()["success"]==old["success"] and j.read()["skip"]==1
   assert j.next_logical()==number+1 and j.read()["attempts"]==old["attempts"]+1
   j.start("train",old["epoch"]);j.compute("forward");j.stop("simulated failure")
   assert j.read()["unknown"]==2 and j.read()["attempts"]==old["attempts"]+2
   refused(lambda:j.start("train",old["epoch"]))
 test("native_AMP_skip_advances_logical_slot_new_failure_remains_charged",skip_and_failure)
 def budget_limits():
  with tempfile.TemporaryDirectory(dir=WORK) as tmp:
   j=journal_fixture(tmp,2027);s=j.read()
   s.update(epoch=99,iteration=249,phase="train",attempts=25000,success=24999,skip=0,fatal=0,unknown=1,
            train_forwards=25000,backwards=24999,optimizer_calls=24999,monitor_attempts=4950,
            monitor_complete=4950,monitor_forwards=4950,monitor_iteration=0,pending=None)
   j.save(s);assert j.start("train",99)==25000
   j.compute("forward");j.compute("backward");j.compute("optimizer")
   j.resolve_train(StepEvidence(1.,1.,65536.,65536.,1,True,True),True)
   refused(lambda:j.start("train",99))
   j.finish_train()
   for _ in range(50):j.start("monitor",99);j.compute("forward");j.resolve_monitor()
   j.finish_monitor();j.finish_epoch(100)
   final=j.read()
   assert final["phase"]=="complete" and final["attempts"]==25001 and final["success"]==25000 and final["unknown"]==1 and final["monitor_complete"]==5000
   refused(lambda:j.start("train",100))
 test("terminal_25000_logical_25001_physical_no_extra_epoch_or_forward",budget_limits)
 def floors():
  with tempfile.TemporaryDirectory(dir=WORK) as tmp:
   j=journal_fixture(tmp,2027);s=j.read();bad=copy.deepcopy(s)
   bad["attempts"]-=1;bad["success"]-=1;bad["iteration"]-=1
   atomic(j.path,bad);refused(j.read)
   atomic(j.path,s);j.start("train",s["epoch"]);j.compute("forward")
   refused(lambda:j.compute("forward"));refused(lambda:j.compute("optimizer"))
 test("cumulative_parent_floors_and_computation_order_enforced",floors)
 def actual_logical_selection():
  import observer as module
  with tempfile.TemporaryDirectory(dir=WORK) as tmp:
   j=journal_fixture(tmp,2027);obj=module.ResumeObserver.__new__(module.ResumeObserver)
   obj.seed=2027;obj.j=j;obj.t=SimpleNamespace(current_epoch=43,optimizer=SimpleNamespace(param_groups=[{"lr":.1}]))
   obj.history=iter([json.dumps({"attempt":10895})]);obj.paired_train=10894
   rows=[];seen=[]
   batch={"keys":["a","b"],"nl_candidates":[{"image":"a","candidates":[],"counts":{}},{"image":"b","candidates":[],"counts":{}}]}
   with patch.object(module,"compare_train",lambda b,h,n,e,l,d:seen.append(n)),patch.object(module,"log",lambda p,row:rows.append(row)),patch.object(module.FastObserver,"train_step",lambda self,b:{"loss":0.}):
    obj.train_step(batch)
   assert seen==[10895] and rows[0]["physical_attempt"]==10896 and rows[0]["logical_input"]==10895
 test("actual_resume_observer_selects_targets_using_original_logical_input",actual_logical_selection)
 def partial_loop():
  class Toy:
   def __init__(self,state):
    self.network=torch.nn.Linear(1,1,bias=False);self.network.load_state_dict(state)
    self.optimizer=torch.optim.SGD(self.network.parameters(),lr=.1,momentum=.9)
    self.num_epochs=2;self.num_iterations_per_epoch=3;self.num_val_iterations_per_epoch=1
    self.current_epoch=0;self.cursor={"epoch":0,"iteration":0,"phase":"train","pending":None}
    self.partial_train=[];self.partial_monitor=[];self.lr_calls=[];self.started=[];self.ended=0
    self.dataloader_val=iter([0,0])
   def read(self):return self.cursor.copy()
   def on_epoch_start(self):self.started.append(self.current_epoch)
   def on_train_epoch_start(self):
    self.lr_calls.append(self.current_epoch);self.optimizer.param_groups[0]["lr"]=.1/(self.current_epoch+1)
    self.partial_train=[];self.partial_monitor=[];self.network.train()
   def train_step(self,b):
    self.optimizer.zero_grad(set_to_none=True)
    loss=(self.network(torch.tensor([[float(b+1)]]))-.25).square().mean()
    loss.backward();self.optimizer.step();row={"loss":loss.detach().numpy()}
    self.partial_train.append(copy.deepcopy(row));self.cursor["iteration"]+=1;return row
   def on_train_epoch_end(self,outputs):
    assert len(outputs)==3 and digest(outputs)==digest(self.partial_train)
    self.cursor["phase"]="monitor"
   def on_validation_epoch_start(self):self.network.eval()
   def validation_step(self,b):self.partial_monitor.append(b);return b
   def on_validation_epoch_end(self,outputs):assert len(outputs)==1;self.cursor["phase"]="finalize"
   def on_epoch_end(self):
    self.current_epoch+=1;self.cursor.update(epoch=self.current_epoch,iteration=0,phase="train" if self.current_epoch<2 else "complete")
    self.partial_train=[];self.partial_monitor=[]
   def on_train_end(self):self.ended+=1
  state={"weight":torch.tensor([[.2]])}
  full=Toy(state);full.dataloader_train=iter(range(6))
  run_remainder(full,full,full,next(full.dataloader_train))
  prefix=Toy(state);prefix.on_epoch_start();prefix.on_train_epoch_start();prefix.train_step(0);prefix.train_step(1)
  resumed=Toy(prefix.network.state_dict());resumed.optimizer.load_state_dict(copy.deepcopy(prefix.optimizer.state_dict()))
  resumed.cursor=copy.deepcopy(prefix.cursor);resumed.partial_train=copy.deepcopy(prefix.partial_train)
  resumed.dataloader_train=iter(range(2,6))
  run_remainder(resumed,resumed,resumed,next(resumed.dataloader_train))
  assert digest(full.network.state_dict())==digest(resumed.network.state_dict())
  assert digest(full.optimizer.state_dict())==digest(resumed.optimizer.state_dict())
  assert resumed.lr_calls==[1] and resumed.started==[1] and resumed.ended==1
 test("real_CPU_toy_optimizer_continuous_vs_partial_resume_exact_no_repeated_LR",partial_loop)
 def admission():
  refused(validate)
  for p in HERE.glob("*.py"):ast.parse(p.read_text())
  source=(HERE/"worker.py").read_text()
  assert "set_all_seeds(seed)" in source and "restore_state(t,bundle,include_cuda_rng=True)" in source
  assert source.index("t.on_train_start()")<source.index("restore_state(t,bundle,include_cuda_rng=True)")<source.index("run_remainder(t,observer,journal,batch)")
  assert "epoch=(number-1)//250" in source and "counts[\"attempts\"]!=25001" in source
  assert "deny_forward" in source and "deny_update" in source
  controller=(HERE/"controller.py").read_text()
  assert "a14_gpu0.lock" in controller and "a14_gpu1.lock" in controller and ">=180000" in controller and ">=360000" in controller
  assert "os.killpg(p.pid,signal.SIGTERM)" in controller and "start_new_session=True" in controller
 test("release_absent_rejected_source_syntax_lifecycle_resources_shared_stop",admission)
 toy_handle.remove()
 assert toy_calls[0]==12
 assert not torch.cuda.is_initialized()
 atomic(WORK/"CPU_REVIEW.json",{"status":"PASS","cases":results,"case_count":len(results),
       "GPU_calls":0,"scientific_parameter_updates":0,"CPU_toy_optimizer_updates":toy_calls[0],
       "reused_evidence":["13 repair CPU cases","6 GPU diagnostic entry CPU cases","4/4 actual zero-update GPU cases","full Train stream CPU reconstructions","actual checkpoint CPU restore"],
       "pending_runtime_checks":["live pinned native loader reconstruction","actual restored parameter updates under cumulative budget"]},True)
 print("RESUME_CPU_REVIEW_PASS",len(results),flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as exc:
  atomic(WORK/"CPU_FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc()},True)
  raise
