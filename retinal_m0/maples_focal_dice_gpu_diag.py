"""D-only: 16 GPU0 forwards / 96 gradients, zero parameter updates.

Import and queue polling use no CUDA. No update/scientific-training modes exist.
"""
from __future__ import annotations
import argparse, contextlib, json, math, os, random, signal, subprocess, sys, time, shutil
from pathlib import Path
from retinal_m0.maples_g1_safety import (
    GateError, atomic_json, canonical_sha, exclusive_lock, file_sha,
    finite_tree, read_json, safe_evidence, process_identity, owner_alive, _unique_pairs,
)

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
CPU=BASE/'focal_dice_cpu_v1_20261024'
OUT=BASE/'focal_dice_gpu_diag_v1_20261024'
C_STAGE=BASE/'soft_cldice_scientific_2000_v1_20261023'
T0=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
FIXED=BASE/'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
BINDING=OUT/'execution_binding.json'
RELEASE=OUT/'DIAGNOSTIC_RELEASE.json'
RECEIPT=OUT/'CPU_PREFLIGHT_REVIEW.json'
LEDGER_PATH=OUT/'diagnostic_ledger.json'
PHASE='D_FOCAL_DICE_GPU_ZERO_UPDATE_V1'
GPU_UUID='${GPU_UUID}'
START='<!-- D_FOCAL_DICE_GPU_DIAG_CURRENT -->'
END='<!-- /D_FOCAL_DICE_GPU_DIAG_CURRENT -->'
MEMORY_CAP=8*1024**3


def write_new(path,obj):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())


def plan():
    return [{'id':f'b{i:02d}_{precision}','batch_index':i,'precision':precision}
            for i in range(8) for precision in ['FP32','AMP']]


def scope():
    return {'phase':PHASE,'gamma':1,'gamma_zero_control':0,'class_weights':None,
            'forward_cap':16,'parameter_autograd_cap':96,'autograd_per_forward':6,
            'parameter_updates':0,'GPU_uuid':GPU_UUID,'GPU1_query_or_use':False,
            'memory_cap_bytes':MEMORY_CAP,'memory_fraction_cap':.8,
            'queue_wall_seconds':21600,'GPU_wall_seconds':1800,
            'official_Test':False,'Dev_inference':False,'training':False}


def rules():
    return {'delta_to_A0_pause_gt':2.,'AMP_loss_relative_max':.01,
            'near_zero_loss_absolute_max':1e-5,'AMP_total_gradient_relative_L2_max':.1,
            'AMP_total_gradient_cosine_min':.99,'near_zero_gradient_norm':1e-12,
            'near_zero_gradient_absolute_L2_max':1e-12,
            'FP32_composite_relative_max':.001,'AMP_composite_relative_max':.05,
            'negative_cosine':'record, not a standalone divergence claim'}


def current_status(text):
    if text.count(START)!=1 or text.count(END)!=1:raise GateError('exactly one D release block required')
    a=text.index(START)+len(START);b=text.index(END)
    if b<=a:raise GateError('malformed D release block')
    return json.loads(text[a:b],object_pairs_hook=_unique_pairs)


class DiagnosticLedger:
    def __init__(self,path,binding):self.path=Path(path);self.binding=binding

    def snapshot(self):
        s=read_json(self.path)
        for key in ['forward_starts','autograd_calls','optimizer_step_attempts','parameter_updates','blocked_forward_attempts']:
            if type(s.get(key)) is not int or s[key]<0:raise GateError('invalid persistent counter:'+key)
        if s.get('phase')!=PHASE or s.get('binding_sha256')!=self.binding or s.get('forward_cap')!=16 or s.get('gradient_cap')!=96:
            raise GateError('ledger phase/binding/cap mismatch; no reset allowed')
        entries=s.get('entries',{})
        if s.get('status') not in ['READY','RUNNING','FAILED_STOP','COMPLETE']:
            raise GateError('invalid persistent stage state')
        if not set(entries).issubset({c['id'] for c in plan()}) or len(entries)>16:raise GateError('unknown or exhausted ledger IDs')
        for e in entries.values():
            if e['status'] not in ['RESERVED','STARTED','COMPLETE','FAILED'] or type(e['forwards']) is not int or type(e['gradients']) is not int or e['forwards'] not in [0,1] or e['gradients'] not in range(7):
                raise GateError('invalid per-combination accounting')
        if s['forward_starts']!=sum(x['forwards'] for x in entries.values()) or s['autograd_calls']!=sum(x['gradients'] for x in entries.values()):
            raise GateError('persistent counter mismatch')
        if s['forward_starts']>16 or s['autograd_calls']>96 or s.get('parameter_updates')!=0:
            raise GateError('cumulative budget exceeded')
        return s

    def change(self,fn):
        with exclusive_lock(str(self.path)+'.lock'):
            state=self.snapshot();fn(state);atomic_json(self.path,state)

    def reserve(self,combo):
        def apply(s):
            if s['status'] not in ['READY','RUNNING'] or combo not in {x['id'] for x in plan()} or combo in s['entries'] or len(s['entries'])>=16 or any(e['status']!='COMPLETE' for e in s['entries'].values()):
                raise GateError('duplicate/unknown/failed/exhausted combination; no retry')
            s['entries'][combo]={'status':'RESERVED','forwards':0,'gradients':0,'owner':process_identity()}
            s['status']='RUNNING'
        self.change(apply)

    def forward(self,combo):
        def apply(s):
            e=s['entries'].get(combo)
            if s['status']!='RUNNING' or not e or e['status']!='RESERVED' or e['forwards'] or s['forward_starts']>=16:
                s['blocked_forward_attempts']+=1;return
            e['status']='STARTED';e['forwards']=1;s['forward_starts']+=1
        before=self.snapshot()['blocked_forward_attempts'];self.change(apply)
        if self.snapshot()['blocked_forward_attempts']!=before:raise GateError('unbudgeted model forward blocked')

    def gradient(self,combo):
        def apply(s):
            e=s['entries'][combo]
            if s['status']!='RUNNING' or e['status']!='STARTED' or e['gradients']>=6 or s['autograd_calls']>=96:raise GateError('gradient cap exceeded')
            e['gradients']+=1;s['autograd_calls']+=1
        self.change(apply)

    def finish(self,combo):
        def apply(s):
            e=s['entries'][combo]
            if s['status']!='RUNNING' or e['status']!='STARTED' or e['forwards']!=1 or e['gradients']!=6:raise GateError('incomplete diagnostic combination')
            e['status']='COMPLETE'
        self.change(apply)

    def fail(self,combo,error):
        def apply(s):
            if combo in s['entries'] and s['entries'][combo]['status']!='COMPLETE':s['entries'][combo]['status']='FAILED'
            s['status']='FAILED_STOP';s['failure']=safe_evidence(error)
        self.change(apply)

    def reject_optimizer(self):
        self.change(lambda s:s.update(optimizer_step_attempts=s['optimizer_step_attempts']+1,status='FAILED_STOP'))
        raise GateError('optimizer.step forbidden: zero-update diagnostic only')


def empty_ledger(binding):
    return {'phase':PHASE,'binding_sha256':binding,'forward_cap':16,'gradient_cap':96,
            'status':'READY','entries':{},'forward_starts':0,'autograd_calls':0,
            'optimizer_step_attempts':0,'parameter_updates':0,'blocked_forward_attempts':0}


def verify_authority(check_all_sources=True):
    state=current_status((CODE/'STATUS.md').read_text())
    if state.get('state')!='RELEASED' or state.get('phase')!=PHASE:raise GateError('D GPU diagnostic not released')
    for path in [BINDING,RELEASE,RECEIPT,LEDGER_PATH]:
        if not path.is_file() or path.is_symlink():raise GateError('missing/symlink diagnostic authority')
    b,r,receipt=read_json(BINDING),read_json(RELEASE),read_json(RECEIPT);digest=canonical_sha(b)
    if b.get('scope')!=scope() or b.get('plan')!=plan() or b.get('rules')!=rules():raise GateError('D scope/plan/numeric drift')
    expected={'phase':PHASE,'authorized':True,'scope':scope(),'binding_sha256':digest,
              'binding_file_sha256':file_sha(BINDING),'CPU_review_sha256':file_sha(RECEIPT),
              'ledger_path':str(LEDGER_PATH)}
    if any(r.get(k)!=v or type(r.get(k)) is not type(v) for k,v in expected.items()):raise GateError('release binding/scope mismatch')
    if receipt.get('status')!='CPU_PREFLIGHT_ACCEPTED_GPU_NOT_EXECUTED' or receipt.get('binding_sha256')!=digest or receipt.get('all_tests_passed') is not True:
        raise GateError('CPU preflight review missing/stale')
    for path,h in receipt['evidence_sha256'].items():
        if file_sha(path)!=h:raise GateError('CPU evidence changed')
    for k,v in {'release_id':r['release_id'],'binding_sha256':digest,'release_file_sha256':file_sha(RELEASE)}.items():
        if state.get(k)!=v:raise GateError('STATUS release binding mismatch')
    paths=b['source_sha256'] if check_all_sources else b['runtime_guard_paths']
    for path in paths:
        if file_sha(path)!=b['source_sha256'][path]:raise GateError('source drift:'+path)
    ledger=DiagnosticLedger(LEDGER_PATH,digest).snapshot()
    if ledger['status'] not in ['READY','RUNNING'] or ledger['optimizer_step_attempts']:raise GateError('diagnostic ledger consumed or failed')
    return b


def get_batches():
    import torch
    from torch.utils.data._utils.collate import default_collate
    from retinal_m0.maples_core import CorePatches
    config=read_json(T0/'config.json');specs=read_json(FIXED)['batches'];assert len(specs)==8
    dataset=CorePatches(config,'T0');prefix=Path(config['data_root']).resolve()
    allow={str(prefix/n) for n in ['data_manifest.csv','split_record.json']}
    for row in dataset.native.rows:
        for key in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
            allow.add(str((prefix/row[key]).resolve()))
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            if p.startswith(str(prefix)+'/') and p not in allow:raise GateError('non-Train content forbidden:'+p)
    sys.addaudithook(audit)
    prior=read_json(CPU/'attempt_01/source_manifest.json')
    selected={r['filename'] for s in specs for r in s['draws']};sources={}
    for row in dataset.native.rows:
        if row['filename'] not in selected:continue
        for key in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
            path=str(prefix/row[key])
            if prior.get(path)!=file_sha(path):raise GateError('Train source differs from CPU acceptance')
            sources[path]=prior[path]
    batches=[]
    for spec in specs:
        items=[dataset[int(row['draw'])] for row in spec['draws']]
        for item,ref in zip(items,spec['draws']):
            if item['filename']!=ref['filename'] or item['geometry'].tolist()!=ref['geometry'] or item['input_sha256']!=ref['input_sha256_original_T0']:
                raise GateError('original T0 input/draw/geometry changed')
        batches.append(default_collate(items))
    dataset.cache.clear()
    return batches,sources


def preflight():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    if torch.cuda.is_initialized():raise GateError('CPU preflight cannot initialize GPU')
    auth=read_json(OUT/'USER_AUTHORIZATION.json');review=read_json(CPU/'SOL_CPU_REVIEW.json')
    if auth.get('user_approved') is not True or auth.get('phase')!=PHASE or auth.get('max_forwards')!=16 or auth.get('max_parameter_autograd_calls')!=96 or auth.get('parameter_updates')!=0:
        raise GateError('user diagnostic scope mismatch')
    if review['status']!='CPU_INDEPENDENTLY_ACCEPTED_GPU_NOT_AUTHORIZED' or review['gamma_zero_real_parameter_gradient_exact_batches']!=8:raise GateError('D CPU admission missing')
    for p,h in review['evidence_sha256'].items():
        if file_sha(p)!=h:raise GateError('accepted CPU evidence changed')
    batches,data=get_batches();sources=dict(data)
    files=[OUT/'USER_AUTHORIZATION.json',OUT/'GPU_DIAGNOSTIC_SPEC.md',CPU/'SOL_CPU_REVIEW.json',CPU/'attempt_01/source_manifest.json',CPU/'candidate_design.json',FIXED,T0/'config.json',CODE/'configs/development_v1.json']
    files += [CODE/'retinal_m0'/n for n in ['maples_focal_dice.py','maples_focal_dice_gpu_diag.py','maples_core.py','model.py','state.py','maples_g1_safety.py']]
    files += [CODE/'retinal_data'/n for n in ['maples_native.py','maples_fov.py','fives.py']]
    files += [CODE/'tests/test_maples_focal_dice_gpu_diag.py']
    cfg=read_json(T0/'config.json');m0=Path(cfg['source_m0'])
    if file_sha(m0)!=cfg['source_m0_sha256']:raise GateError('original M0 provenance mismatch')
    files.append(m0);sources.update({str(q):file_sha(q) for q in files})
    session=read_json(C_STAGE/'RUN_SESSION.json')
    b={'scope':scope(),'plan':plan(),'rules':rules(),'source_sha256':sources,
       'runtime_guard_paths':[str(q) for q in files if q!=m0],
       'M0_path':str(m0),'M0_sha256':cfg['source_m0_sha256'],
       'batch_input_sha256':[list(b['input_sha256']) for b in batches],
       'torch_version':str(torch.__version__),'CPU_threads':2,
       'C_dependency_binding_sha256':session['binding_sha256']}
    write_new(BINDING,b)
    write_new(OUT/'cpu_preflight.json',{'status':'PASSED_CPU_ONLY','binding_sha256':canonical_sha(b),
              'verified_batches':8,'verified_draws':32,'M0_loaded':False,'M0_forwards':0,
              'GPU_initialized':False,'parameter_updates':0})
    print('D_CPU_PREFLIGHT_PASSED',canonical_sha(b),flush=True)


def dependency_ready(binding):
    completion=C_STAGE/'completion.json';heartbeat=C_STAGE/'heartbeat.json';session=C_STAGE/'RUN_SESSION.json'
    hb=read_json(heartbeat)
    if hb.get('status') in ['FAILED','FAILED_STOP','STOPPED','ERROR'] or (C_STAGE/'failure.json').exists():
        raise GateError('C failed/stopped: stop D queue, leave both stages untouched')
    ss=read_json(session)
    if ss['binding_sha256']!=binding['C_dependency_binding_sha256']:raise GateError('C dependency changed')
    owner={'pid':ss['controller_pid'],'start_time':ss['controller_token'].split(':')[1]}
    alive=owner_alive(owner)
    if not completion.exists():
        if not alive:raise GateError('C controller disappeared before normal completion')
        return False,hb
    done=read_json(completion)
    if done.get('status')!='C2000_TRAINED_AND_EVALUATED' or done.get('updates')!=2000 or done.get('binding_sha256')!=binding['C_dependency_binding_sha256']:
        raise GateError('C normal completion evidence mismatch')
    return not alive,hb


def resources(require_idle=True):
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    x=[v.strip() for v in raw]
    if len(x)!=5 or x[0]!='0' or x[1]!=GPU_UUID:raise GateError('physical GPU0 UUID mismatch')
    rawp=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    pids=[int(v.strip()) for v in rawp.splitlines() if v.strip()]
    value={'index':0,'uuid':x[1],'total_MiB':int(x[2]),'used_MiB':int(x[3]),'util_percent':int(x[4]),'compute_pids':pids,'GPU1_queried':False,'time':time.time()}
    if shutil.disk_usage(OUT).free<1024**3:raise GateError('less than1GiB diagnostic evidence space')
    if require_idle and (value['used_MiB']>=512 or value['util_percent']>=10 or pids):return None
    if not require_idle and any(pid!=os.getpid() for pid in pids):raise GateError('another GPU0 process appeared; no process stopped')
    return value


def norm(gs):return math.sqrt(sum(float(x.detach().double().square().sum().cpu()) for x in gs))
def dot(a,b):return sum(float((x.detach().double()*y.detach().double()).sum().cpu()) for x,y in zip(a,b))


def gradient_comparison(reference,current):
    a,b=norm(reference),norm(current);delta=norm(tuple(x-y for x,y in zip(reference,current)))
    return {'reference_L2':a,'current_L2':b,'difference_L2':delta,
            'relative_L2':delta/max(a,1e-30),'cosine':dot(reference,current)/(a*b) if a and b else None}


def validate_amp(comparison):
    r=rules();g=comparison['total_gradient']
    loss_ok=(comparison['loss_absolute']<=r['near_zero_loss_absolute_max'] if abs(comparison['reference_loss'])<=r['near_zero_loss_absolute_max'] else comparison['loss_relative']<=r['AMP_loss_relative_max'])
    if g['reference_L2']<=1e-12 and g['current_L2']<=1e-12:
        gradient_ok=g['difference_L2']<=1e-12
    else:gradient_ok=g['relative_L2']<=.1 and g['cosine'] is not None and g['cosine']>=.99
    if not loss_ok or not gradient_ok:raise GateError('AMP vs same-batch FP32 loss/gradient tolerance exceeded')


@contextlib.contextmanager
def block_optimizers(torch,ledger):
    patches=[]
    def reject(*args,**kwargs):ledger.reject_optimizer()
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer) and 'step' in cls.__dict__:
            patches.append((cls,cls.__dict__['step']));cls.step=reject
    try:yield
    finally:
        for cls,method in patches:cls.step=method


def update_status(**changes):
    path=CODE/'STATUS.md';text=path.read_text();state=current_status(text);state.update(changes)
    a=text.index(START)+len(START);b=text.index(END)
    new=text[:a]+'\n'+json.dumps(state,ensure_ascii=False,separators=(',',':'))+'\n'+text[b:]
    if path.read_text()!=text:raise GateError('STATUS concurrent edit; no overwrite')
    tmp=path.with_name('STATUS.md.D_diagnostic.pending');tmp.write_text(new,encoding='utf8');os.replace(tmp,path)


def run_gpu(binding,resource_before):
    # Called only by this same process while BOTH runtime and shared GPU locks are held.
    os.environ['CUDA_VISIBLE_DEVICES']=GPU_UUID
    import numpy as np
    import torch
    from retinal_m0.maples_focal_dice import focal_dice_loss,focal_dice_terms
    from retinal_m0.maples_core import weighted_masked_loss
    from retinal_m0.model import UNet
    from retinal_m0.state import state_digest
    if torch.cuda.is_initialized():raise GateError('CUDA initialized before resource/authority admission')
    if str(torch.__version__)!=binding['torch_version']:raise GateError('torch version drift')
    batches,_=get_batches();verify_authority()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if torch.cuda.device_count()!=1:raise GateError('only authorized GPU0 must be visible')
    torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.8,MEMORY_CAP/total),0)
    atomic_json(OUT/'queue_state.json',{'status':'RUNNING_GPU_DIAGNOSTICS','owner':process_identity(),'GPU_initialized':True,'parameter_updates':0})
    arch=read_json(CODE/'configs/development_v1.json')
    model=UNet(arch['widths'],arch['group_norm_groups']).cuda().train()
    cp=torch.load(binding['M0_path'],map_location='cpu',weights_only=True)
    if cp['step']!=10000:raise GateError('wrong M0 checkpoint step')
    model.load_state_dict(cp['model'],strict=True)
    if state_digest(model.state_dict())!=state_digest(cp['model']):raise GateError('loaded M0 tensors differ')
    del cp
    ps=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
    ledger=DiagnosticLedger(LEDGER_PATH,canonical_sha(binding));active={'id':None}
    hook=model.register_forward_pre_hook(lambda *_:ledger.forward(active['id']))
    initial_model=state_digest(model.state_dict())
    def rng_hash():
        n=np.random.get_state()
        return state_digest({'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch_CPU':torch.get_rng_state(),'torch_GPU0':torch.cuda.get_rng_state(0)})
    initial_rng=rng_hash();out=OUT/'attempt_01';out.mkdir(exist_ok=False);rows=[];reference=None;started=time.monotonic()
    try:
        with block_optimizers(torch,ledger):
            for combo in binding['plan']:
                verify_authority(False);active['id']=combo['id'];ledger.reserve(combo['id']);resource_current=resources(False)
                batch=batches[combo['batch_index']]
                if list(batch['input_sha256'])!=binding['batch_input_sha256'][combo['batch_index']]:raise GateError('staged input differs from preflight')
                x,y,f=[batch[k].cuda() for k in ['image','mask','fov']]
                torch.cuda.reset_peak_memory_stats(0);stamp=time.monotonic();amp=combo['precision']=='AMP'
                scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024.)
                scale_before=scaler.state_dict();scale=float(scaler.get_scale())
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):z=model(x)
                if z.dtype!=(torch.float16 if amp else torch.float32):raise GateError('model precision mismatch')
                base=weighted_masked_loss(z,y,f,f);zero=focal_dice_loss(z,y,f,gamma=0)
                terms=focal_dice_terms(z,y,f,gamma=1);new=terms['loss'];delta=new-base
                empty=focal_dice_terms(z,torch.zeros_like(y),f,gamma=1)
                repeated=torch.stack([focal_dice_loss(z[i:i+1],y[i:i+1],f[i:i+1],gamma=1)
                                      for i in range(len(y))]).mean()
                def grad(loss):
                    ledger.gradient(combo['id'])
                    values=tuple(v.detach().float()/scale for v in torch.autograd.grad(scaler.scale(loss),ps+(z,),retain_graph=True,allow_unused=False))
                    if any(not torch.isfinite(v).all() for v in values):raise GateError('nonfinite real unscaled gradient')
                    return values[:-1],values[-1]
                g0,z0=grad(base);gz,zz=grad(zero);gn,zn=grad(new);gd,zd=grad(delta);ge,ze=grad(empty['loss']);gr,zr=grad(repeated)
                exact=torch.equal(base,zero) and torch.equal(z0,zz) and all(torch.equal(a,b) for a,b in zip(g0,gz))
                if not exact:raise GateError('gamma0 A0 value/gradient not exact')
                empty_ok=bool(torch.all(empty['per_crop_focal']>0) and torch.all(ze[f.bool()]>=0) and torch.any(ze[f.bool()]>0) and torch.all(ze[~f.bool()]==0))
                if not empty_ok:raise GateError('empty GT background penalty/gradient incorrect')
                try:focal_dice_loss(z,y,torch.zeros_like(f))
                except ValueError as e:
                    if 'empty_fov' not in str(e):raise
                else:raise GateError('empty FOV not rejected')
                n0,nn,nd=norm(g0),norm(gn),norm(gd)
                if n0<=0 or nn<=0:raise GateError('missing nonzero A0/candidate gradient')
                ratio=nd/n0;cosine=dot(g0,gn)/(n0*nn)
                composite=norm(tuple(n-a-d for n,a,d in zip(gn,g0,gd)))/max(n0+nn,1e-30)
                repeated_error=norm(tuple(a-b for a,b in zip(gn,gr)))/nn
                comparison=None
                if amp:
                    if reference is None or reference['batch']!=combo['batch_index']:raise GateError('matching FP32 reference absent')
                    gcmp=gradient_comparison(reference['g'],tuple(v.cpu() for v in gn))
                    diff=abs(float(new.detach())-reference['loss'])
                    comparison={'reference_loss':reference['loss'],'loss_absolute':diff,'loss_relative':diff/max(abs(reference['loss']),1e-30),'total_gradient':gcmp,
                                'A0_gradient_descriptive':gradient_comparison(reference['g0'],tuple(v.cpu() for v in g0)),
                                'delta_gradient_descriptive':gradient_comparison(reference['gd'],tuple(v.cpu() for v in gd))}
                else:reference={'batch':combo['batch_index'],'g':tuple(v.cpu() for v in gn),
                                'g0':tuple(v.cpu() for v in g0),'gd':tuple(v.cpu() for v in gd),'loss':float(new.detach())}
                layers={name:{'A0_L2':norm((a,)),'candidate_L2':norm((n,)),'delta_L2':norm((d,)),'A0_dot_delta':dot((a,),(d,))} for name,a,n,d in zip(names,g0,gn,gd)}
                torch.cuda.synchronize(0)
                row={**combo,'gamma':1,'input_sha256':list(batch['input_sha256']),'draws':batch['draw'].tolist(),
                     'filenames':list(batch['filename']),'geometry':batch['geometry'].tolist(),
                     'GT_pixels':terms['GT_count'].detach().cpu().tolist(),'FOV_pixels':terms['FOV_count'].detach().cpu().tolist(),
                     'per_crop_losses':{k:terms[k].detach().cpu().tolist() for k in ['per_crop_bce','per_crop_focal','per_crop_dice']},
                     'loss_A0':float(base.detach()),'loss_candidate':float(new.detach()),
                     'gamma_zero_value_and_real_parameter_gradient_exact':exact,
                     'empty_GT_background_loss_and_logit_gradient_passed':empty_ok,'empty_FOV_rejected':True,
                     'model_logit_dtype':str(z.dtype),'loss_dtype':str(new.dtype),'scale':scale,'unscaled':True,'preclip':True,
                     'A0_parameter_L2':n0,'candidate_parameter_L2':nn,'delta_parameter_L2':nd,
                     'delta_to_A0':ratio,'candidate_to_A0':nn/n0,'cosine_A0_candidate':cosine,
                     'cosine_A0_delta':dot(g0,gd)/(n0*nd) if nd else None,
                     'difference_additivity_error':composite,'per_crop_reduction_gradient_error':repeated_error,
                     'per_layer':layers,'AMP_vs_FP32':comparison,
                     'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated(0)/1048576,
                     'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved(0)/1048576,
                     'elapsed_seconds':time.monotonic()-stamp,'parameter_updates':0,'resource_before':resource_current}
                finite_tree(row);write_new(out/(combo['id']+'.json'),row)
                if ratio>2:raise GateError('delta/A0 gradient ratio>2 pause')
                limit=rules()['AMP_composite_relative_max' if amp else 'FP32_composite_relative_max']
                if composite>limit or repeated_error>limit:raise GateError('same-forward gradient aggregation mismatch')
                if comparison:validate_amp(comparison)
                if row['GPU_peak_reserved_MiB']*1048576>MEMORY_CAP:raise GateError('8GiB memory cap exceeded')
                if scaler.state_dict()!=scale_before or any(p.grad is not None for p in ps):raise GateError('scaler/accumulated-grad side effect')
                if state_digest(model.state_dict())!=initial_model or rng_hash()!=initial_rng:raise GateError('model/RNG changed in zero-update diagnostic')
                ledger.finish(combo['id']);rows.append(row)
                print('DIAGNOSTIC_COMPLETE',len(rows),'/16',combo['id'],'delta/A0',ratio,flush=True)
                if amp:reference=None
                del x,y,f,z,base,zero,terms,new,delta,empty,repeated,g0,z0,gz,zz,gn,zn,gd,zd,ge,ze,gr,zr
            verify_authority();resources(False);s=ledger.snapshot()
            if s['forward_starts']!=16 or s['autograd_calls']!=96 or len(rows)!=16 or s['optimizer_step_attempts']:raise GateError('final16/96/zero-update counters mismatch')
            summary={'status':'D_GPU16_NUMERICAL_CHECKS_PASSED_PENDING_SOL_REVIEW','phase':PHASE,
                     'completed':16,'forwards':16,'parameter_autograd_calls':96,'optimizer_step_attempts':0,'parameter_updates':0,
                     'gamma':1,'max_delta_to_A0':max(x['delta_to_A0'] for x in rows),
                     'max_reserved_MiB':max(x['GPU_peak_reserved_MiB'] for x in rows),
                     'max_AMP_total_gradient_relative_L2':max(x['AMP_vs_FP32']['total_gradient']['relative_L2'] for x in rows if x['AMP_vs_FP32']),
                     'min_AMP_total_gradient_cosine':min(x['AMP_vs_FP32']['total_gradient']['cosine'] for x in rows if x['AMP_vs_FP32']),
                     'model_and_RNG_unchanged':True,'elapsed_seconds':time.monotonic()-started,
                     'resource_before':resource_before,'GPU1_queried':False,'Dev_or_Test_images_read':False,
                     'engineering_or_scientific_training_authorized':False}
            write_new(OUT/'completion.json',summary)
            ledger.change(lambda s:s.update(status='COMPLETE'))
            update_status(state='CONSUMED',run_state='COMPLETE_PENDING_SOL_REVIEW',completed=16,forwards=16,gradients=96,parameter_updates=0)
            print(json.dumps(summary),flush=True)
    except BaseException as exc:
        error=safe_evidence({'exception':repr(exc),'time':time.time()})
        ledger.fail(active['id'],error)
        write_new(OUT/'failure.json',{'error':error,'ledger':ledger.snapshot(),'no_retry':True,'parameter_updates':0})
        update_status(state='HOLD_FAILED',run_state='FAILED_STOP')
        raise
    finally:hook.remove()


def run_queued():
    binding=verify_authority();start=time.monotonic();identity=process_identity()
    gpu_attempted=False
    def timeout(*_):raise TimeoutError('D diagnostic bounded wait/runtime expired')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(21600)
    with exclusive_lock(OUT/'RUN.lock'):
        if DiagnosticLedger(LEDGER_PATH,canonical_sha(binding)).snapshot()['entries']:
            raise GateError('ledger already consumed work; no retry/restart')
        try:
            while True:
                verify_authority(False);ready,hb=dependency_ready(binding)
                atomic_json(OUT/'queue_state.json',{'status':'WAITING_FOR_C_OR_GPU0','owner':identity,'C_step':hb.get('step'),'elapsed_seconds':time.monotonic()-start,'GPU_initialized':False,'forwards':0,'parameter_updates':0})
                if ready:
                    with contextlib.ExitStack() as stack:
                        try:stack.enter_context(exclusive_lock(ROOT/'experiments/a14_gpu0.lock'))
                        except GateError:
                            time.sleep(15);continue
                        resource=resources(True)
                        if resource is not None:
                            verify_authority();signal.alarm(1800)
                            atomic_json(OUT/'queue_state.json',{'status':'RUNNING_GPU_DIAGNOSTICS','owner':identity,'C_step':2000,'GPU_initialized':False,'parameter_updates':0})
                            update_status(run_state='RUNNING_DIAGNOSTICS',active_pid=os.getpid())
                            gpu_attempted=True
                            run_gpu(binding,resource)
                            atomic_json(OUT/'queue_state.json',{'status':'COMPLETE_PENDING_SOL_REVIEW','owner':identity,'parameter_updates':0})
                            break
                time.sleep(15)
        except BaseException as exc:
            if not (OUT/'failure.json').exists():
                DiagnosticLedger(LEDGER_PATH,canonical_sha(binding)).fail(None,repr(exc))
                write_new(OUT/'queue_failure.json',{'error':repr(exc),'owner':identity,'GPU_work_attempted':gpu_attempted,'no_auto_retry':True,'parameter_updates':0})
            atomic_json(OUT/'queue_state.json',{'status':'FAILED_STOP','owner':identity,'error':repr(exc),'parameter_updates':0})
            try:update_status(state='HOLD_FAILED',run_state='FAILED_STOP')
            except Exception as status_error:print('STATUS_UPDATE_FAILED',repr(status_error),flush=True)
            raise
        finally:signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--run-when-c-completes',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight()
    else:run_queued()


if __name__=='__main__':main()
