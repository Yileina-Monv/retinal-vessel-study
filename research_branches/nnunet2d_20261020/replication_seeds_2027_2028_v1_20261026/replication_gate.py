"""One authoritative two-seed scope; individual bounded ledgers and one GPU0 serial controller."""
import os,sys,json,hashlib,subprocess,shutil
from pathlib import Path
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study';DATA_BASE=ROOT/'experiments/nnunet2d_branch_20261020'
GROUP=DATA_BASE/'replication_seed2027_2028_100epochs_v1_20261026';BRANCH=CODE/'research_branches/nnunet2d_20261020/replication_seeds_2027_2028_v1_20261026'
OLD=BRANCH.parent/'no_wall_limit_restart_02';ENG=BRANCH.parent/'amp_engineering32_20261025';OBS=BRANCH.parent/'scientific_resume_v5_20261025'
sys.path.extend([str(CODE),str(OLD),str(ENG),str(OBS)])
from retinal_m0.maples_g1_safety import process_identity,atomic_json,canonical_sha,read_json
PHASE='NNUNET_REPLICATION_2027_2028_V1';GROUP_PROTOCOL_SHA='07d3038812dd7afd2216d71f46ec884ab0c7a37370e387e58013be011d73d43e';GPU='${GPU_UUID}'
def set_context(seed):
    if type(seed)is not int or seed not in [2027,2028]:raise RuntimeError('only frozen seeds2027,2028 allowed')
    global SEED,STAGE
    SEED=seed;STAGE=GROUP/'runs'/('seed_'+str(seed))
set_context(int(os.environ.get('RETINA_NN_SEED','2027')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def group_protocol():
    q=read_json(GROUP/'protocol.json')
    if sha(GROUP/'protocol.json')!=GROUP_PROTOCOL_SHA or q['phase']!=PHASE or q['seeds']!=[2027,2028] or q['device']!=GPU or q['total_train_attempt_cap']!=50000 or q['official_Test_allowed']:raise RuntimeError('replication protocol scope drift')
    if sha(GROUP/'SCIENTIFIC_PROTOCOL.md')!=q['normative_sha256']:raise RuntimeError('replication normative text changed')
    return q
def protocol():
    q=group_protocol();p=STAGE/'protocol.json'
    if sha(p)!=q['seed_protocol_sha256'][str(SEED)]:raise RuntimeError('seed-specific protocol changed')
    v=read_json(p)
    if v['scope']['seed']!=SEED or v['scope']['epochs']!=100 or v['scope']['planned_optimizer_attempts']!=25000 or v['scope']['internal_monitor_total']!=5000 or v['data']['fold']!='all' or v['scope']['test_access'] or v['budget']['GPU1_allowed']:raise RuntimeError('seed data/budget scope drift')
    return v
def authority():
    t=(CODE/'STATUS.md').read_text();a='<!-- NNUNET_SCIENCE100_CURRENT -->';z='<!-- /NNUNET_SCIENCE100_CURRENT -->'
    if t.count(a)!=1 or t.count(z)!=1:raise RuntimeError('unique authority required')
    v=json.loads(t.split(a)[1].split(z)[0])
    if v.get('phase')!=PHASE or v.get('protocol_sha256')!=GROUP_PROTOCOL_SHA or v.get('user_approved_seeds')!=[2027,2028]:raise RuntimeError('two-seed user authority missing')
    return v
def update_group(**changes):
    p=CODE/'STATUS.md'
    for _ in range(5):
        t=p.read_text();a='<!-- NNUNET_SCIENCE100_CURRENT -->';z='<!-- /NNUNET_SCIENCE100_CURRENT -->';lo=t.index(a)+len(a);hi=t.index(z);v=json.loads(t[lo:hi]);v.update(changes);tmp=p.with_name(p.name+'.nnrep.'+str(os.getpid())+'.pending')
        with tmp.open('x',encoding='utf-8') as f:f.write(t[:lo]+'\n'+json.dumps(v,ensure_ascii=False)+'\n'+t[hi:]);f.flush();os.fsync(f.fileno())
        if p.read_text()!=t:tmp.unlink();continue
        os.replace(tmp,p);return
    raise RuntimeError('concurrent STATUS updates')
def update_status(**changes):
    v=authority();runs=v['runs'];runs[str(SEED)].update(changes);update_group(runs=runs)
def binding_key(seed=None):
    seed=SEED if seed is None else seed;b=read_json(GROUP/'source_binding.json')
    return canonical_sha({'group_binding_sha256':b['binding_sha256'],'seed':seed,'seed_protocol_sha256':group_protocol()['seed_protocol_sha256'][str(seed)]})
def expected_release():
    return {'phase':PHASE,'group_protocol_sha256':GROUP_PROTOCOL_SHA,'group_binding_sha256':read_json(GROUP/'source_binding.json')['binding_sha256'],'CPU_review_sha256':sha(GROUP/'CPU_REVIEW.json'),'seed':SEED,'train_worker_seed':SEED,'monitor_worker_seed':SEED+1,'GPU':GPU,'epochs':100,'attempt_cap':25000,'monitor_cap':5000,'group_attempt_cap':50000,'official_Test_allowed':False,'automatic_retry':False,'scientific_outcome_does_not_cancel_next_seed':True}
def validate(full=False):
    protocol();v=authority();b=read_json(GROUP/'source_binding.json');r=read_json(GROUP/'CPU_REVIEW.json');key=canonical_sha({k:x for k,x in b.items() if k!='binding_sha256'})
    if key!=b['binding_sha256'] or key!=r['binding_sha256'] or r['status']!='CPU_ACCEPTED':raise RuntimeError('stale replication CPU binding')
    if v.get('active_seed')!=SEED or not v.get('GPU_released') or v['runs'][str(SEED)]['state']!='RELEASED':raise RuntimeError('seed GPU HOLD')
    p=STAGE/'GPU_RELEASE.json'
    if not p.exists() or read_json(p)!=expected_release() or v['runs'][str(SEED)]['release_sha256']!=sha(p):raise RuntimeError('independent seed release missing/stale')
    for p,h in b['all_sources' if full else 'runtime_sources'].items():
        if sha(p)!=h:raise RuntimeError('replication source changed: '+p)
    if full:
        for p,h in b['Train_sources'].items():
            if sha(p)!=h:raise RuntimeError('Train input changed: '+p)
    return binding_key()
def runtime_authorizer(key,role):
    p=STAGE/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json');s=read_json(p);h=sha(p)
    def check():
        if sha(p)!=h or process_identity()!=s['worker'] or process_identity(os.getppid())!=s['controller'] or os.environ.get('NNUNET100_NONCE')!=s['nonce'] or s['seed']!=SEED:raise RuntimeError('invalid live seed worker session')
        if validate()!=key:raise RuntimeError('lost seed authority')
    check();return check
def resources():
    a=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')];p=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if a[0]!=GPU or int(a[1])>=512 or int(a[2])>=10 or p.strip():raise RuntimeError('GPU0 busy; never evict or switch GPU')
    if shutil.disk_usage(GROUP).free<50*2**30:raise RuntimeError('free disk below50GiB')
    return {'GPU':GPU,'used_MiB':int(a[1]),'total_MiB':int(a[3])}
