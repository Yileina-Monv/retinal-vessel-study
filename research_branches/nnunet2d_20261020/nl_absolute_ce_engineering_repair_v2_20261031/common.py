from pathlib import Path
import os,json,hashlib,time
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';BASE=R/'experiments/nnunet2d_branch_20261020'
O=BASE/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031/engineering_v1/repair_v2'
GPU='${GPU_UUID}'
ORDER=['diag','base2027','zero2027','base2028','zero2028','cont','split_a','split_b']
LIMITS={'diag':38,'base2027':16,'zero2027':16,'base2028':16,'zero2028':16,'cont':32,'split_a':16,'split_b':16}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d,new=False):
 p=Path(p)
 if new and p.exists():raise RuntimeError('Refuse overwrite '+str(p))
 q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
 with q.open('x') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def log(n,d):
 with (O/n).open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def validate():
 rel=read(O/'GPU_RELEASE.json')
 if rel['scope']!='ENGINEERING_ONLY' or rel['GPU']!=GPU or rel['science_allowed'] or not rel['CPU_pass']:
  raise RuntimeError('No valid separate GPU authorization')
 for p,h in rel['bound_files'].items():
  if sha(p)!=h:raise RuntimeError('Binding changed '+p)
 return sha(O/'GPU_RELEASE.json')
class Ledger:
 def __init__(self,path,key):self.path=Path(path);self.key=key
 def init(self):atomic(self.path,dict(binding=self.key,status='READY',diag_reserved=2,update_reserved=0,forwards=2,optimizer_calls=0,success=0,skips=0,consecutive_skips=0,failed=0,unknown=0,pending=None,completed=[],role=None,role_attempts=0),True)
 def get(self):
  x=read(self.path)
  assert x['binding']==self.key and x['diag_reserved']<=40 and x['update_reserved']<=128
  assert x['update_reserved']==x['success']+x['skips']+x['failed']+x['unknown']+int(x['pending'] is not None and x['pending']['kind']=='update')
  return x
 def start(self,role):
  x=self.get();assert x['status']=='READY' and x['pending'] is None and x['role'] is None
  assert ORDER[len(x['completed'])]==role
  x.update(role=role,role_attempts=0);atomic(self.path,x)
 def reserve(self):
  x=self.get();assert x['status']=='READY' and x['pending'] is None and x['role_attempts']<LIMITS[x['role']]
  kind='diag' if x['role']=='diag' else 'update';field=kind+'_reserved'
  assert x[field]<(40 if kind=='diag' else 128)
  x[field]+=1;x['role_attempts']+=1;x['pending']={'kind':kind,'forward':False,'optimizer':False};atomic(self.path,x)
 def forward(self):
  x=self.get();assert x['pending'] and not x['pending']['forward'];x['pending']['forward']=True;x['forwards']+=1;atomic(self.path,x)
 def optimizer(self):
  x=self.get();assert x['pending'] and x['pending']['kind']=='update' and x['pending']['forward'] and not x['pending']['optimizer']
  x['pending']['optimizer']=True;x['optimizer_calls']+=1;atomic(self.path,x)
 def resolve(self,skip=False):
  x=self.get();p=x['pending'];assert p and p['forward']
  if p['kind']=='diag':assert not p['optimizer']
  elif skip:
   assert not p['optimizer'];x['skips']+=1;x['consecutive_skips']+=1
  else:assert p['optimizer'];x['success']+=1;x['consecutive_skips']=0
  x['pending']=None;atomic(self.path,x)
  if x['consecutive_skips']>=5 or x['skips']>=25:raise RuntimeError('AMP cumulative/consecutive stop')
 def finish(self):
  x=self.get();assert x['pending'] is None and x['role_attempts']==LIMITS[x['role']]
  x['completed'].append(x['role']);x['role']=None
  if len(x['completed'])==len(ORDER):x['status']='COMPLETE_PENDING_REVIEW'
  atomic(self.path,x)
 def stop(self,reason):
  x=self.get()
  if x['pending'] and x['pending']['kind']=='update':
   x['unknown' if x['pending']['optimizer'] else 'failed']+=1
  x['pending']=None;x['status']='STOPPED';x['reason']=str(reason);atomic(self.path,x)
