"""Cumulative physical budget, separate from the original 25,000 logical input slots."""
import copy
from fast_observer import Journal,AmpLimits,atomic,RevisionStop
from common import read,sha
class ResumeJournal(Journal):
 def __init__(self,path,binding):
  super().__init__(path,binding,100,250,50,AmpLimits(5,25))
  self.parent_path=self.path.parent.parent/"budget.json"
  self.parent_counts=read(self.parent_path)
 def initialize(self):raise RevisionStop("fresh/reset budgets forbidden in resume")
 def migrate(self,old,parent_sha):
  if old!=self.parent_counts or sha(self.parent_path)!=parent_sha:raise RevisionStop("parent budget provenance mismatch")
  if self.path.exists():raise RevisionStop("resume ledger exists; never reset")
  if old["status"]!="STOPPED" or old["pending"] is not None or old["unknown"]!=1 or old["fatal"] or old["attempts"]!=old["success"]+old["skip"]+1:raise RevisionStop("unexpected parent ledger")
  if old["schedule"]!=[100,250,50] or old["limits"]!={"consecutive":5,"total":25}:raise RevisionStop("parent protocol mismatch")
  s=copy.deepcopy(old)
  s.update(schema=3,binding=self.binding,status="READY",physical_cap=25001,inherited_unknown=1,
           parent_budget_sha256=parent_sha,parent_binding=old["binding"],
           parent_stop_reason=old.get("reason"),migration_success=old["success"],migration_attempts=old["attempts"])
  s.pop("reason",None);atomic(self.path,s);return self.read()
 def read(self):
  s=read(self.path)
  if s["binding"]!=self.binding or s["schedule"]!=[100,250,50] or s["limits"]!={"consecutive":5,"total":25} or s["physical_cap"]!=25001 or s["inherited_unknown"]!=1 or s["unknown"]<1:raise RevisionStop("resume ledger identity drift")
  if sha(self.parent_path)!=s["parent_budget_sha256"]:raise RevisionStop("parent ledger changed")
  for k in ["attempts","success","skip","fatal","unknown","train_forwards","monitor_forwards","backwards","optimizer_calls","monitor_attempts","monitor_complete"]:
   if s[k]<self.parent_counts[k]:raise RevisionStop("cumulative counts were refunded "+k)
  pending=int(s["pending"] is not None and s["pending"]["kind"]=="train")
  resolved=s["success"]+s["skip"]+s["fatal"]
  if s["attempts"]!=resolved+s["unknown"]+pending:raise RevisionStop("physical attempts lost or invented")
  if resolved!=s["epoch"]*250+s["iteration"]:raise RevisionStop("logical input cursor mismatch")
  if not 0<=s["attempts"]<=25001 or not 0<=resolved<=25000 or not 0<=s["monitor_attempts"]<=5000:raise RevisionStop("cumulative budget cap")
  if s["train_forwards"]>s["attempts"] or s["backwards"]>s["attempts"] or s["optimizer_calls"]>s["attempts"] or s["monitor_forwards"]>s["monitor_attempts"]:raise RevisionStop("extra model computation")
  if pending and s["pending"]["logical_number"]!=resolved+1:raise RevisionStop("pending logical input mismatch")
  return s
 def next_logical(self):
  s=self.read();return s["epoch"]*250+s["iteration"]+1
 def start(self,kind,epoch):
  s=self.read()
  if s["status"]!="READY" or s["pending"] is not None or s["epoch"]!=epoch or s["phase"]!=kind:raise RevisionStop("stopped/unresolved/wrong phase")
  key="attempts" if kind=="train" else "monitor_attempts"
  cap=25001 if kind=="train" else 5000
  index=s["iteration" if kind=="train" else "monitor_iteration"]
  if s[key]>=cap or index>=(250 if kind=="train" else 50):raise RevisionStop("fixed cumulative budget exhausted")
  logical=self.next_logical() if kind=="train" else s["monitor_attempts"]+1
  if kind=="train" and logical>25000:raise RevisionStop("logical training budget exhausted")
  s[key]+=1;s["pending"]={"kind":kind,"number":s[key],"logical_number":logical,"forward":0,"backward":0,"optimizer":0}
  self.save(s);return logical
