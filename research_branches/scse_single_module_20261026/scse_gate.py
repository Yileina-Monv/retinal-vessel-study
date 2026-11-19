"""Bound, single-GPU1 authorization for one module; never change nnU-Net GPU0 authority."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study'
STAGE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scse_single_module_2000_v1_20261026'
BRANCH=CODE/'research_branches/scse_single_module_20261026';OUT=STAGE/'runs/seed_2026/SCSE'
GPU='${GPU_UUID}';PHASE='SCSE_SINGLE_MODULE_2000_V1';PROTOCOL_SHA='8d0501f699d685b78a38065320c51711460a0dedaff62adec568081eb49cc613'
sys.path.insert(0,str(CODE))
from retinal_m0.maples_g1_safety import GateError,atomic_json,canonical_sha,read_json,process_identity
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def protocol():
    q=read_json(STAGE/'protocol.json')
    if sha(STAGE/'protocol.json')!=PROTOCOL_SHA or q['phase']!=PHASE or q['updates']!=2000 or q['seed']!=2026 or q['GPU1_UUID']!=GPU or q['Test_allowed'] or q['GPU0_allowed'] or sha(STAGE/'SCIENTIFIC_PROTOCOL.md')!=q['normative_text_sha256']:raise GateError('SCSE frozen scope drift')
    return q
def authority():
    text=(CODE/'STATUS.md').read_text();a='<!-- SCSE_CURRENT -->';z='<!-- /SCSE_CURRENT -->'
    if text.count(a)!=1 or text.count(z)!=1:raise GateError('unique SCSE authority required')
    v=json.loads(text.split(a)[1].split(z)[0])
    if v['phase']!=PHASE or v['protocol_sha256']!=PROTOCOL_SHA or v['user_approved_science2000_conditional_on_engineering'] is not True:raise GateError('SCSE user authority missing')
    return v
def update_status(**changes):
    p=CODE/'STATUS.md'
    for _ in range(5):
        text=p.read_text();a='<!-- SCSE_CURRENT -->';z='<!-- /SCSE_CURRENT -->';lo=text.index(a)+len(a);hi=text.index(z);v=json.loads(text[lo:hi]);v.update(changes);tmp=p.with_name(p.name+'.scse.'+str(os.getpid())+'.pending')
        with tmp.open('x',encoding='utf-8') as f:f.write(text[:lo]+'\n'+json.dumps(v,ensure_ascii=False)+'\n'+text[hi:]);f.flush();os.fsync(f.fileno())
        if p.read_text()!=text:tmp.unlink();continue
        os.replace(tmp,p);return
    raise GateError('SCSE concurrent STATUS modification')
def validate_release(role='science',full_sources=False):
    protocol();v=authority();b=read_json(STAGE/'source_binding.json');r=read_json(STAGE/'CPU_REVIEW.json');key=canonical_sha({k:x for k,x in b.items() if k!='binding_sha256'})
    if key!=b['binding_sha256'] or key!=r['binding_sha256'] or r['status']!='CPU_ACCEPTED':raise GateError('stale module CPU binding')
    release_path=STAGE/('ENG_RELEASE.json' if role=='engineering' else 'SCIENCE_RELEASE.json')
    if v['state']!=('ENGINEERING_RELEASED' if role=='engineering' else 'SCIENCE_RELEASED') or not v['GPU_released'] or not release_path.exists():raise GateError('SCSE GPU HOLD')
    rel=read_json(release_path)
    expected={'phase':PHASE,'role':role,'protocol_sha256':PROTOCOL_SHA,'binding_sha256':key,'CPU_review_sha256':sha(STAGE/'CPU_REVIEW.json'),'GPU':GPU,'updates':32 if role=='engineering' else 2000,'Test_allowed':False,'auto_restart':False}
    if role=='science':expected['engineering_review_sha256']=sha(STAGE/'SOL_ENGINEERING_REVIEW.json')
    if rel!=expected or v['release_sha256']!=sha(release_path) or v['binding_sha256']!=key:raise GateError('SCSE release mismatch')
    for p,h in b['sources' if full_sources else 'runtime_sources'].items():
        if sha(p)!=h:raise GateError('SCSE source changed: '+p)
    return key
def runtime_authorizer(key):
    p=STAGE/'CURRENT_CHILD.json';session=read_json(p);h=sha(p);role=session['role']
    def check():
        if sha(p)!=h or process_identity()!=session['worker'] or process_identity(os.getppid())!=session['controller'] or os.environ.get('SCSE_NONCE')!=session['nonce']:raise GateError('invalid SCSE child identity')
        if validate_release(role)!=key:raise GateError('SCSE authority drift')
    check();return check
def resources(idle=False):
    s=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
    pids=[int(x) for x in subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()]
    if s[0]!=GPU or (idle and (pids or int(s[2])>=512 or int(s[3])>=10)):raise GateError('GPU1 not idle/UUID mismatch; never kill others or switch cards')
    if not idle and any(x!=os.getpid() for x in pids):raise GateError('unexpected GPU1 compute process')
    if shutil.disk_usage(STAGE).free<20*2**30:raise GateError('free disk below20GiB')
    return {'GPU':GPU,'total_MiB':int(s[1]),'used_MiB':int(s[2]),'utilization':int(s[3]),'compute_pids':pids}
def dataset_guard():
    q=protocol();paths=read_json(STAGE/'source_binding.json')['data_content_paths']
    from retinal_m0.maples_ufl_engineering import data_guard
    data_guard(q['baseline_config']['data_root'],paths)
