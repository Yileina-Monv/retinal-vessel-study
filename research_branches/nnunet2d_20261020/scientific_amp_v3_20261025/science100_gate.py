"""Source-bound one-trajectory scientific release, GPU0 only."""
import json,os,hashlib,subprocess,shutil,sys
from pathlib import Path
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study'
STAGE=ROOT/'experiments/nnunet2d_branch_20261020/scientific_seed2026_100epochs_v3_amp_20261025'
BRANCH=CODE/'research_branches/nnunet2d_20261020/scientific_amp_v3_20261025'
PHASE='NNUNET_SCIENTIFIC100_AMP_V3_20261025';GPU='${GPU_UUID}';PROTOCOL_SHA='d73e43eeb4a2a101e5b6df680433c157ca38353d97b2b98e54e25afd353b7755'
sys.path.insert(0,str(CODE))
from retinal_m0.maples_g1_safety import process_identity
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def protocol():
    p=STAGE/'protocol.json';q=json.loads(p.read_text())
    if sha(p)!=PROTOCOL_SHA or q['protocol_id']!=PHASE or (q['scope']['epochs'],q['scope']['planned_optimizer_attempts'],q['scope']['internal_monitor_total'],q['scope']['seed'])!=(100,25000,5000,2026):raise RuntimeError('scientific protocol digest/scope drift')
    if q['AMP']['consecutive_pause']!=5 or q['AMP']['total_pause']!=25 or q['data']['fold']!='all' or q['inference']['checkpoint']!='checkpoint_final.pth' or q['scope']['test_access'] or q['budget']['GPU1_allowed']:raise RuntimeError('frozen science definitions drift')
    return q
def authority():
    text=(CODE/'STATUS.md').read_text();a='<!-- NNUNET_SCIENCE100_CURRENT -->';b='<!-- /NNUNET_SCIENCE100_CURRENT -->'
    if text.count(a)!=1 or text.count(b)!=1:raise RuntimeError('unique science authorization missing')
    v=json.loads(text.split(a)[1].split(b)[0])
    if v.get('phase')!=PHASE or v.get('protocol_sha256')!=PROTOCOL_SHA or v.get('user_approved_science100') is not True:raise RuntimeError('wrong science authority')
    return v
def validate(full=False):
    protocol();v=authority();b=json.loads((STAGE/'source_binding.json').read_text());r=json.loads((STAGE/'CPU_REVIEW.json').read_text())
    key=canonical({k:z for k,z in b.items() if k!='binding_sha256'})
    if key!=b['binding_sha256'] or key!=r['binding_sha256'] or r['status']!='CPU_ACCEPTED':raise RuntimeError('CPU binding stale')
    relpath=STAGE/'SCIENCE_RELEASE.json'
    if v.get('state')!='RELEASED' or v.get('GPU_released') is not True or not relpath.exists():raise RuntimeError('science GPU HOLD: no independent release')
    release=json.loads(relpath.read_text());expected={'phase':PHASE,'protocol_sha256':PROTOCOL_SHA,'binding_sha256':key,'CPU_review_sha256':sha(STAGE/'CPU_REVIEW.json'),'GPU':GPU,'seed':2026,'epochs':100,'attempt_cap':25000,'monitor_cap':5000,'automatic_restart':False,'official_Test_allowed':False}
    if release!=expected or v.get('binding_sha256')!=key or v.get('release_sha256')!=sha(relpath):raise RuntimeError('release scope or summary mismatch')
    for p,h in b['all_code_sources' if full else 'runtime_code_sources'].items():
        if sha(p)!=h:raise RuntimeError('code/reference drift: '+p)
    if full:
        for p,h in b['Train_sources'].items():
            if sha(p)!=h:raise RuntimeError('Train source drift: '+p)
    return key
def runtime_authorizer(key,role):
    p=STAGE/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json');session=json.loads(p.read_text());session_sha=sha(p)
    def check():
        if sha(p)!=session_sha or process_identity()!=session['worker'] or process_identity(os.getppid())!=session['controller'] or os.environ.get('NNUNET100_NONCE')!=session['nonce']:raise RuntimeError('lost live scientific child session')
        if validate()!=key:raise RuntimeError('runtime science authority changed')
    check();return check
def resources():
    a=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if a[0].strip()!=GPU or int(a[1])>=512 or int(a[2])>=10:raise RuntimeError('GPU0 busy; never evict or switch card')
    p=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if p.strip():raise RuntimeError('GPU0 compute process present')
    free=shutil.disk_usage(STAGE).free
    if free<50*2**30:raise RuntimeError('less than50GiB free disk')
    return {'GPU':GPU,'used_MiB':int(a[1]),'total_MiB':int(a[3]),'free_disk_bytes':free}
def update_status(**changes):
    p=CODE/'STATUS.md';text=p.read_text();a='<!-- NNUNET_SCIENCE100_CURRENT -->';b='<!-- /NNUNET_SCIENCE100_CURRENT -->';lo=text.index(a)+len(a);hi=text.index(b);v=json.loads(text[lo:hi]);v.update(changes);new=text[:lo]+'\n'+json.dumps(v,ensure_ascii=False)+'\n'+text[hi:];tmp=p.with_name('STATUS.md.nnunet100.pending')
    with tmp.open('x',encoding='utf-8') as f:f.write(new);f.flush();os.fsync(f.fileno())
    if p.read_text()!=text:raise RuntimeError('STATUS concurrent edit')
    os.replace(tmp,p)
