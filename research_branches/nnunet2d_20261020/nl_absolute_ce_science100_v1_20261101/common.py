"""Scientific admission and authority; import does not initialize a model or CUDA."""
import os,sys,json,hashlib,time,shutil,fcntl,datetime
from pathlib import Path
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study';DATA_BASE=ROOT/'experiments/nnunet2d_branch_20261020';BRANCH=Path(__file__).resolve().parent
P=DATA_BASE/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031';GROUP=P/'science100_v1';BASELINE=DATA_BASE/'replication_seed2027_2028_100epochs_v1_20261026'
GPU='${GPU_UUID}';MARKER='NL_SCIENCE100_RELEASED_20261101'
SEED=int(os.environ.get('RETINA_NL_SEED','2027'));STAGE=GROUP/'runs'/f'seed_{SEED}'
sys.path.extend([str(CODE),str(BRANCH.parent/'no_wall_limit_restart_02'),str(BRANCH.parent/'scientific_resume_v5_20261025'),str(BRANCH.parent/'amp_engineering32_20261025'),str(BRANCH.parent/'replication_seeds_2027_2028_v1_20261026')])
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x,new=False):
 p=Path(p)
 if new and p.exists():raise RuntimeError('one-shot artifact already exists '+str(p))
 tmp=p.with_name(p.name+'.pending')
 with tmp.open('x') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def log(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def card():return read(GROUP/'protocol.json')
def protocol():return read(BASELINE/'runs'/f'seed_{SEED}'/'protocol.json')
def validate_document(q):
 if not q['authorization']['science_training_approved'] or q['authorization']['Test_allowed'] or q['training']['seeds']!=[2027,2028] or q['training']['epochs']!=100 or q['training']['train_attempts_per_seed']!=25000 or q['training']['monitor_batches_per_seed']!=5000 or q['method']['lambda']!=0.0018152656553044105 or q['runtime']['device']!=GPU or q['gradients']['ratio_stop_strictly_greater_than']!=2:raise RuntimeError('scientific scope/approval drift')
 if q['training']['batch']!=2 or q['training']['patch']!=[1024,1536] or q['method']['max_positive_groups_per_crop'] is not None:raise RuntimeError('config drift')
 if q['runtime']['train_hours_cap_per_seed']!=50 or q['runtime']['total_train_hours_cap']!=100 or q['inference']['hours_cap_per_seed']!=.5:raise RuntimeError('resource budget drift')
_cache={'time':0}
def validate(full=False):
 q=card();validate_document(q)
 r=read(GROUP/'SCIENCE_RELEASE.json');b=read(GROUP/'SOURCE_BINDING.json')
 if r['scope']!='NL_TWO_SEED_SCIENCE_ONLY' or r['protocol_sha256']!=sha(GROUP/'protocol.json') or r['binding_sha256']!=sha(GROUP/'SOURCE_BINDING.json') or r['cpu_review_sha256']!=sha(GROUP/'CPU_REVIEW.json') or read(GROUP/'CPU_REVIEW.json')['status']!='PASS':raise RuntimeError('release stage/digest/CPU binding mismatch')
 if MARKER not in (CODE/'STATUS.md').read_text()[:16000]:raise RuntimeError('STATUS does not carry independent release')
 if full or time.monotonic()-_cache['time']>30:
  for p,h in b['runtime_sources'].items():
   if sha(p)!=h:raise RuntimeError('runtime source changed '+p)
  _cache['time']=time.monotonic()
 if full:
  for p,h in b['immutable_sources'].items():
   if sha(p)!=h:raise RuntimeError('source changed '+p)
 return sha(GROUP/'SCIENCE_RELEASE.json')
def runtime_authorizer(key,role):
 from retinal_m0.maples_g1_safety import process_identity
 p=STAGE/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json')
 for _ in range(100):
  if p.exists():break
  time.sleep(.05)
 s=read(p);h=sha(p)
 def check():
  if sha(p)!=h or process_identity()!=s['worker'] or process_identity(os.getppid())!=s['controller'] or os.environ.get('NNUNET100_NONCE')!=s['nonce'] or s['seed']!=SEED or s['role']!=role:raise RuntimeError('live session binding failed')
  job=read(GROUP/'job_state.json')
  if job.get('active_seed')!=SEED or job.get('role')!=role or job.get('status')!='RUNNING' or validate()!=key:raise RuntimeError('wrong role/stage or revoked release')
 check();return check

def compare_train(batch,history,number,epoch,lr,digest):
 if history['attempt']!=number or history['epoch']!=epoch or history['iteration']!=(number-1)%250 or list(batch['keys'])!=history['ids'] or digest(batch['data'])!=history['input_sha256'] or digest(batch['target'])!=history['target_sha256'] or abs(lr-history['lr'])>1e-15:raise RuntimeError('N0 input/GT/ID/LR pairing mismatch before forward')
def update_status(**changes):atomic(STAGE/'progress.json',dict(changes,time=time.time()))
def authority_entry(title,body):
 with (CODE/'.deep_supervision_status_write.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);bk=ROOT/'docs/.continuity_backups'/('nl_science_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));bk.mkdir(parents=True)
  for n in ['STATUS.md','项目全局状态与研究决策.md']:
   p=CODE/n;old=p.read_text();shutil.copy2(p,bk/n);head,tail=old.split('\n',1);tmp=p.with_name(p.name+'.nl-science');tmp.write_text(head+'\n\n### '+title+'\n\n'+body+'\n\n'+tail);os.replace(tmp,p)
