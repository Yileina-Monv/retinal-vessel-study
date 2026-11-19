"""Phase, source, owner and GPU0 gates; no implicit GPU initialization."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study'
STAGE=ROOT/'experiments/nnunet2d_branch_20261020/amp_engineering32_20261025'
BRANCH=CODE/'research_branches/nnunet2d_20261020/amp_engineering32_20261025'
PHASE='NNUNET_AMP_ENGINEERING32_20261025';GPU='${GPU_UUID}'
PROTOCOL_SHA='8e65525ec31a16a384b92a15603baf12562f21002242834fbb59445203d5f661'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1048576),b''):h.update(v)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def protocol():
    q=json.loads((STAGE/'protocol.json').read_text())
    if sha(STAGE/'protocol.json')!=PROTOCOL_SHA or q['phase']!=PHASE or q['train_attempt_cap']!=32 or q['Train_monitor_cap']!=5 or q['wall_cap_seconds']!=900 or q['injected_overflow_attempt']!=31 or q['science_authorized'] or q['GPU1_allowed'] or q['Dev_allowed'] or q['Test_allowed']:raise RuntimeError('engineering protocol scope/digest drift')
    return q
def authority():
    text=(CODE/'STATUS.md').read_text();a='<!-- NNUNET_AMP32_CURRENT -->';b='<!-- /NNUNET_AMP32_CURRENT -->'
    if text.count(a)!=1 or text.count(b)!=1:raise RuntimeError('unique engineering authority missing')
    v=json.loads(text.split(a)[1].split(b)[0])
    if v.get('phase')!=PHASE or v.get('protocol_sha256')!=PROTOCOL_SHA or v.get('user_approved_engineering') is not True:raise RuntimeError('current engineering authorization drift')
    return v
def validate(full=False):
    protocol();v=authority();b=json.loads((STAGE/'cpu_binding.json').read_text());r=json.loads((STAGE/'CPU_REVIEW.json').read_text());release=STAGE/'ENGINEERING_RELEASE.json'
    key=canonical({k:z for k,z in b.items() if k!='binding_sha256'})
    if b['binding_sha256']!=key or r['binding_sha256']!=key or r['status']!='CPU_ACCEPTED':raise RuntimeError('CPU evidence binding stale')
    if v.get('state')!='RELEASED' or v.get('GPU_released') is not True or not release.exists():raise RuntimeError('engineering GPU HOLD: no independent release')
    rel=json.loads(release.read_text())
    if rel!={'phase':PHASE,'binding_sha256':key,'protocol_sha256':PROTOCOL_SHA,'CPU_review_sha256':sha(STAGE/'CPU_REVIEW.json'),'GPU':GPU,'attempt_cap':32,'monitor_cap':5,'wall_seconds':900,'science_authorized':False}:raise RuntimeError('release scope/binding drift')
    if v.get('release_sha256')!=sha(release) or v.get('binding_sha256')!=key:raise RuntimeError('STATUS release mismatch')
    for path,want in b['code_sources' if full else 'runtime_code_sources'].items():
        if sha(path)!=want:raise RuntimeError('source drift: '+path)
    if full:
        for path,want in b['data_sources'].items():
            if sha(path)!=want:raise RuntimeError('Train content drift: '+path)
    return key
def runtime_authorizer(key):
    import sys
    sys.path.insert(0,str(CODE))
    from retinal_m0.maples_g1_safety import process_identity
    session=json.loads((STAGE/'RUN_SESSION.json').read_text());sid=sha(STAGE/'RUN_SESSION.json')
    def check():
        if sha(STAGE/'RUN_SESSION.json')!=sid or session['worker']!=process_identity() or session['controller']!=process_identity(os.getppid()) or session['nonce']!=os.environ.get('NNUNET_AMP32_NONCE'):raise RuntimeError('lost authenticated direct-child session')
        if validate()!=key:raise RuntimeError('changed CPU binding')
    check();return check
def resources():
    import shutil
    v=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    if v[0].strip()!=GPU or int(v[1])>=512 or int(v[2])>=10:raise RuntimeError('GPU0 busy; no process stopped')
    procs=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if procs.strip():raise RuntimeError('GPU0 has compute process')
    free=shutil.disk_usage(STAGE).free
    if free<50*2**30:raise RuntimeError('need50GiB free disk')
    return {'GPU':GPU,'used_MiB':int(v[1]),'total_MiB':int(v[3]),'free_disk_bytes':free}
def update_status(**kwargs):
    p=CODE/'STATUS.md';old=p.read_text();a='<!-- NNUNET_AMP32_CURRENT -->';b='<!-- /NNUNET_AMP32_CURRENT -->';lo=old.index(a)+len(a);hi=old.index(b);v=json.loads(old[lo:hi]);v.update(kwargs);new=old[:lo]+'\n'+json.dumps(v,ensure_ascii=False)+'\n'+old[hi:];tmp=p.with_name('STATUS.md.nnunet_amp32.pending')
    with tmp.open('x',encoding='utf-8') as f:f.write(new);f.flush();os.fsync(f.fileno())
    if p.read_text()!=old:raise RuntimeError('STATUS changed concurrently')
    os.replace(tmp,p)
