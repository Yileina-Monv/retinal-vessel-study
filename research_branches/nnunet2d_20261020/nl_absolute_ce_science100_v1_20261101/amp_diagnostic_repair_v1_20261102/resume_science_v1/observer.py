import sys
from support import *
import nl_observer_repaired as repaired
from fast_observer import FastObserver,digest,rng_pack
from selection import choose_groups
from common import compare_train
class ResumeObserver(repaired.NLObserver):
 def __init__(self,trainer,journal,seed,authorize,on_event):
  authorize()
  self.seed=seed;self.diag_mode=False;self.component=None;self.groups=None
  self.coefficient=read(GROUP/"protocol.json")["method"]["lambda"]
  self.history=(BASELINE/"runs"/("seed_"+str(seed))/"attempts.jsonl").open()
  self.monitor_history=(BASELINE/"runs"/("seed_"+str(seed))/"monitor.jsonl").open()
  s=journal.read();self.paired_train=s["success"]+s["skip"];self.paired_monitor=s["monitor_complete"]
  for _ in range(self.paired_train):next(self.history)
  for _ in range(self.paired_monitor):next(self.monitor_history)
  repaired.STAGE=stage(seed)
  trainer.train_step=self.native_step
  names=set(read(BASELINE/"runs"/("seed_"+str(seed))/"protocol.json")["train_ids"])
  FastObserver.__init__(self,trainer,journal,stage(seed),names,authorize,on_event=on_event)
 def train_step(self,batch):
  number=self.j.next_logical();h=json.loads(next(self.history))
  compare_train(batch,h,number,self.t.current_epoch,float(self.t.optimizer.param_groups[0]["lr"]),digest)
  records=batch.get("nl_candidates")
  if records is None or len(records)!=2 or [r["image"] for r in records]!=list(batch["keys"]):raise RuntimeError("missing/misaligned sidecar")
  before=digest(rng_pack())
  self.groups=[choose_groups(r["candidates"],self.seed,number,i) for i,r in enumerate(records)]
  if before!=digest(rng_pack()):raise RuntimeError("private selection changed RNG")
  self.paired_train+=1
  log(stage(self.seed)/"auxiliary_targets.jsonl",{"attempt":number,"logical_input":number,
       "physical_attempt":self.j.read()["attempts"]+1,"ids":list(batch["keys"]),
       "counts":[r["counts"] for r in records],"group_counts":[len(g) for g in self.groups],
       "selected_groups_sha256":digest(self.groups),"pairing":"PASS"})
  return FastObserver.train_step(self,batch)
