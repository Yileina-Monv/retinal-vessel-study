"""UFL-only: 16 GPU0 forwards / 96 gradients, zero parameter updates.

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
STAGE=BASE/'ufl_asymmetric_v1_20261024'
OUT=STAGE/'diagnostics'
C_STAGE=BASE/'soft_cldice_scientific_2000_v1_20261023'
T0=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
FIXED=BASE/'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
BINDING=OUT/'execution_binding.json'
RELEASE=OUT/'DIAGNOSTIC_RELEASE.json'
RECEIPT=OUT/'CPU_PREFLIGHT_REVIEW.json'
LEDGER_PATH=OUT/'diagnostic_ledger.json'
PHASE='UFL_ASYMMETRIC_GPU_ZERO_UPDATE_V1'
GPU_UUID='${GPU_UUID}'
START='<!-- UFL_GPU_DIAG_CURRENT -->'
END='<!-- /UFL_GPU_DIAG_CURRENT -->'
MEMORY_CAP=8*1024**3


def write_new(path,obj):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())


def plan():
    return [{'id':f'b{i:02d}_{precision}','batch_index':i,'precision':precision}
            for i in range(8) for precision in ['FP32','AMP']]


def scope():
    return {'phase':PHASE,'gamma':.5,'delta':.6,'lambda_code':.5,'epsilon':1e-7,
            'forward_cap':16,'parameter_autograd_cap':96,'autograd_per_forward':6,
            'parameter_updates':0,'GPU_uuid':GPU_UUID,'GPU1_query_or_use':False,
            'memory_cap_bytes':MEMORY_CAP,'memory_fraction_cap':.8,
            'queue_wall_seconds':21600,'GPU_wall_seconds':1800,
            'official_Test':False,'Dev_inference':False,'training':False}


def rules():
    return {'delta_to_A0_pause_gt':2.,'AMP_loss_relative_max':.01,
            'near_zero_loss_absolute_max':1e-7,'AMP_total_gradient_relative_L2_max':.05,
            'AMP_total_gradient_cosine_min':.99,'near_zero_gradient_norm':1e-7,
            'near_zero_gradient_absolute_L2_max':1e-7,
            'FP32_composite_relative_max':.001,'AMP_composite_relative_max':.05,
            'negative_cosine':'record, not a standalone divergence claim'}


def current_status(text):
    if text.count(START)!=1 or text.count(END)!=1:raise GateError('exactly one UFL release block required')
    a=text.index(START)+len(START);b=text.index(END)
    if b<=a:raise GateError('malformed UFL release block')
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
    if state.get('state')!='RELEASED' or state.get('phase')!=PHASE:raise GateError('UFL GPU diagnostic not released')
    for path in [BINDING,RELEASE,RECEIPT,LEDGER_PATH]:
        if not path.is_file() or path.is_symlink():raise GateError('missing/symlink diagnostic authority')
    b,r,receipt=read_json(BINDING),read_json(RELEASE),read_json(RECEIPT);digest=canonical_sha(b)
    if b.get('scope')!=scope() or b.get('plan')!=plan() or b.get('rules')!=rules():raise GateError('UFL scope/plan/numeric drift')
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
    # Reuse only the immutable, already verified input reconstruction helper.
    # It performs no model construction/forward and denies non-Train content.
    from retinal_m0.maples_focal_dice_gpu_diag import get_batches as original_batches
    return original_batches()


def preflight():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    if torch.cuda.is_initialized(): raise GateError('CPU preflight initialized CUDA')
    review=read_json(STAGE/'CPU_REVIEW.json')
    if review.get('accepted') is not True or review.get('GPU_executed') is not False:
        raise GateError('UFL independent CPU review missing')
    for p,h in review['evidence_sha256'].items():
        if file_sha(p)!=h: raise GateError('UFL accepted CPU evidence drift')
    batches,data=get_batches(); sources=dict(data)
    cfg=read_json(T0/'config.json'); m0=Path(cfg['source_m0'])
    if file_sha(m0)!=cfg['source_m0_sha256']: raise GateError('original M0 provenance mismatch')
    files=[STAGE/'protocol.json',STAGE/'UFL_IMPLEMENTATION_SPEC.md',STAGE/'CPU_REVIEW.json',
           STAGE/'reference/source_manifest.json',STAGE/'reference/author_original.py',
           STAGE/'reference/author_docstring_indent_fixed.py',FIXED,T0/'config.json',
           CODE/'configs/development_v1.json',m0]
    files += [CODE/'retinal_m0'/n for n in ['maples_ufl.py','maples_ufl_gpu_diag.py',
              'maples_focal_dice_gpu_diag.py','maples_core.py','model.py','state.py','maples_g1_safety.py']]
    files += [CODE/'retinal_data'/n for n in ['maples_native.py','maples_fov.py','fives.py']]
    files += [CODE/'tests'/n for n in ['test_maples_ufl.py','test_maples_ufl_gpu_diag.py']]
    sources.update({str(p):file_sha(p) for p in files})
    b={'scope':scope(),'plan':plan(),'rules':rules(),'source_sha256':sources,
       'runtime_guard_paths':[str(p) for p in files if p!=m0],
       'M0_path':str(m0),'M0_sha256':cfg['source_m0_sha256'],
       'batch_input_sha256':[list(x['input_sha256']) for x in batches],
       'torch_version':str(torch.__version__),'CPU_threads':2}
    write_new(BINDING,b)
    write_new(OUT/'cpu_preflight.json',{'status':'PASSED_CPU_ONLY','binding_sha256':canonical_sha(b),
              'verified_batches':8,'verified_draws':32,'M0_loaded':False,'M0_forwards':0,
              'GPU_initialized':False,'parameter_updates':0})
    print('UFL_CPU_PREFLIGHT_PASSED',canonical_sha(b),flush=True)


def dependency_ready(binding):
    # GPU idle/shared-lock admission is mandatory; no previous phase is run here.
    return True, {}


def resources(require_idle=True):
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    x=[v.strip() for v in raw]
    if len(x)!=5 or x[0]!='0' or x[1]!=GPU_UUID:raise GateError('physical GPU0 UUID mismatch')
    rawp=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    pids=[int(v.strip()) for v in rawp.splitlines() if v.strip()]
    value={'index':0,'uuid':x[1],'total_MiB':int(x[2]),'used_MiB':int(x[3]),'util_percent':int(x[4]),'compute_pids':pids,'GPU1_queried':False,'time':time.time()}
    if shutil.disk_usage(OUT).free<20*1024**3:raise GateError('less than1GiB diagnostic evidence space')
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
    r=rules(); g=comparison['total_gradient']
    loss_ok=(comparison['loss_absolute']<=1e-7 if abs(comparison['reference_loss'])<=1e-7 else comparison['loss_relative']<=.01)
    if max(g['reference_L2'],g['current_L2'])<=1e-7:
        gradient_ok=g['difference_L2']<=1e-7
    else:
        gradient_ok=g['relative_L2']<=.05 and g['cosine'] is not None and g['cosine']>=.99
    if not loss_ok or not gradient_ok: raise GateError('frozen UFL AMP/FP32 tolerance exceeded')


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
    tmp=path.with_name('STATUS.md.UFL_diagnostic.pending');tmp.write_text(new,encoding='utf8');os.replace(tmp,path)


def run_gpu(binding,resource_before):
    os.environ['CUDA_VISIBLE_DEVICES']=GPU_UUID
    import numpy as np
    import torch
    from retinal_m0.maples_ufl import ufl_terms
    from retinal_m0.maples_core import weighted_masked_loss
    from retinal_m0.model import UNet
    from retinal_m0.state import state_digest
    if torch.cuda.is_initialized(): raise GateError('early CUDA initialization')
    if str(torch.__version__)!=binding['torch_version']: raise GateError('torch drift')
    batches,_=get_batches();verify_authority()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if torch.cuda.device_count()!=1: raise GateError('only physical GPU0 allowed')
    torch.cuda.set_device(0);total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.8,MEMORY_CAP/total),0)
    arch=read_json(CODE/'configs/development_v1.json')
    model=UNet(arch['widths'],arch['group_norm_groups']).cuda().train()
    cp=torch.load(binding['M0_path'],map_location='cpu',weights_only=True)
    if cp['step']!=10000: raise GateError('wrong M0 step')
    model.load_state_dict(cp['model'],strict=True)
    if state_digest(model.state_dict())!=state_digest(cp['model']): raise GateError('M0 tensors mismatch')
    del cp
    ps=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
    ledger=DiagnosticLedger(LEDGER_PATH,canonical_sha(binding));active={'id':None}
    hook=model.register_forward_pre_hook(lambda *_:ledger.forward(active['id']))
    initial_model=state_digest(model.state_dict())
    def rng_hash():
        n=np.random.get_state()
        return state_digest({'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch_CPU':torch.get_rng_state(),'torch_GPU0':torch.cuda.get_rng_state(0)})
    initial_rng=rng_hash();out=OUT/'attempt_01';out.mkdir(exist_ok=False)
    rows=[];reference=None;started=time.monotonic()
    try:
        with block_optimizers(torch,ledger):
            for combo in binding['plan']:
                verify_authority(False);active['id']=combo['id'];ledger.reserve(combo['id'])
                resource_current=resources(False);batch=batches[combo['batch_index']]
                if list(batch['input_sha256'])!=binding['batch_input_sha256'][combo['batch_index']]: raise GateError('input binding drift')
                x,y,f=[batch[k].cuda() for k in ['image','mask','fov']]
                torch.cuda.reset_peak_memory_stats(0);stamp=time.monotonic();amp=combo['precision']=='AMP'
                scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024.)
                scale_before=scaler.state_dict();scale=float(scaler.get_scale())
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):z=model(x)
                if z.dtype!=(torch.float16 if amp else torch.float32): raise GateError('precision mismatch')
                base=weighted_masked_loss(z,y,f,f);terms=ufl_terms(z,y,f)
                new=terms['loss'];delta=new-base
                losses=[base,terms['per_crop_weighted_afl'].mean(),terms['per_crop_weighted_aft'].mean(),new,delta]
                gradients=[];logit_gradients=[]
                # Stream each gradient to CPU and free the temporary CUDA tuple.
                # Five diagnostic VJPs plus one actual scaled backward; no new forward.
                for loss in losses:
                    ledger.gradient(combo['id'])
                    raw=torch.autograd.grad(scaler.scale(loss),ps+(z,),retain_graph=True,allow_unused=False)
                    values=tuple(v.detach().float().div(scale).cpu() for v in raw)
                    del raw
                    if any(not torch.isfinite(v).all() for v in values): raise GateError('nonfinite real unscaled diagnostic gradient')
                    gradients.append(values[:-1]);logit_gradients.append(values[-1])
                g0,ga,gt,gn,gd=gradients
                if any(p.grad is not None for p in ps): raise GateError('diagnostic VJP changed p.grad')
                ledger.gradient(combo['id']);scaler.scale(new).backward()
                actual=tuple(p.grad.detach().float().div(scale).cpu() for p in ps)
                if any(not torch.isfinite(v).all() for v in actual): raise GateError('nonfinite actual backward gradient')
                backward_error=norm(tuple(a-b for a,b in zip(actual,gn)))/max(norm(gn),1e-30)
                for p in ps:p.grad=None
                mask=f.cpu().bool()
                outside_zero=all(bool(torch.all(v[~mask]==0)) for v in logit_gradients)
                if not outside_zero: raise GateError('FOV-exterior gradient nonzero')
                n0,nn,nd=norm(g0),norm(gn),norm(gd)
                if not n0 or not nn: raise GateError('missing A0/UFL gradient')
                ratio=nd/n0
                composite=norm(tuple(n-a-t for n,a,t in zip(gn,ga,gt)))/max(nn,1e-30)
                delta_error=norm(tuple(n-a-d for n,a,d in zip(gn,g0,gd)))/max(n0+nn,1e-30)
                comparison=None
                if amp:
                    if reference is None or reference['batch']!=combo['batch_index']: raise GateError('matching FP32 reference missing')
                    diff=abs(float(new.detach())-reference['loss'])
                    comparison={'reference_loss':reference['loss'],'loss_absolute':diff,'loss_relative':diff/max(abs(reference['loss']),1e-30),'total_gradient':gradient_comparison(reference['g'],gn)}
                else:reference={'batch':combo['batch_index'],'g':gn,'loss':float(new.detach())}
                layers={name:{'A0_L2':norm((a,)),'UFL_L2':norm((n,)),'weighted_aFL_L2':norm((fa,)),'weighted_aFT_L2':norm((ft,)),'delta_L2':norm((d,)),'A0_dot_delta':dot((a,),(d,))} for name,a,n,fa,ft,d in zip(names,g0,gn,ga,gt,gd)}
                torch.cuda.synchronize(0)
                row={**combo,'parameters':{'lambda_code':.5,'delta':.6,'gamma':.5,'epsilon':1e-7},
                     'input_sha256':list(batch['input_sha256']),'draws':batch['draw'].tolist(),
                     'filenames':list(batch['filename']),'geometry':batch['geometry'].tolist(),
                     'GT_pixels':terms['GT_count'].detach().cpu().tolist(),'FOV_pixels':terms['FOV_count'].detach().cpu().tolist(),
                     'per_crop_losses':{k:terms[k].detach().cpu().tolist() for k in ['per_crop_afl','per_crop_aft','per_crop_weighted_afl','per_crop_weighted_aft']},
                     'loss_A0':float(base.detach()),'loss_candidate':float(new.detach()),
                     'FOV_outside_gradient_zero':outside_zero,'scaled_backward_relative_error':backward_error,
                     'model_logit_dtype':str(z.dtype),'loss_dtype':str(new.dtype),'scale':scale,'unscaled':True,'preclip':True,
                     'A0_parameter_L2':n0,'candidate_parameter_L2':nn,'delta_parameter_L2':nd,
                     'delta_to_A0':ratio,'candidate_to_A0':nn/n0,'cosine_A0_candidate':dot(g0,gn)/(n0*nn),
                     'cosine_A0_delta':dot(g0,gd)/(n0*nd) if nd else None,
                     'component_additivity_error':composite,'difference_additivity_error':delta_error,
                     'per_layer':layers,'AMP_vs_FP32':comparison,
                     'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated(0)/1048576,
                     'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved(0)/1048576,
                     'elapsed_seconds':time.monotonic()-stamp,'parameter_updates':0,'resource_before':resource_current}
                finite_tree(row);write_new(out/(combo['id']+'.json'),row)
                if ratio>2:raise GateError('delta/A0 gradient ratio>2 pause')
                limit=rules()['AMP_composite_relative_max' if amp else 'FP32_composite_relative_max']
                if max(composite,delta_error,backward_error)>limit:raise GateError('same-forward aggregation/backward mismatch')
                if comparison:validate_amp(comparison)
                if row['GPU_peak_reserved_MiB']*1048576>MEMORY_CAP:raise GateError('8GiB allocator cap')
                if scaler.state_dict()!=scale_before or any(p.grad is not None for p in ps):raise GateError('scaler/grad side effect')
                if state_digest(model.state_dict())!=initial_model or rng_hash()!=initial_rng:raise GateError('model/RNG changed in zero-update diagnostic')
                ledger.finish(combo['id']);rows.append(row)
                print('DIAGNOSTIC_COMPLETE',len(rows),'/16',combo['id'],'delta/A0',ratio,flush=True)
                if amp:reference=None
                del x,y,f,z,base,terms,new,delta,losses,loss,gradients,logit_gradients,values,actual,g0,ga,gt,gn,gd
            verify_authority();resources(False);s=ledger.snapshot()
            if s['forward_starts']!=16 or s['autograd_calls']!=96 or len(rows)!=16 or s['optimizer_step_attempts']:raise GateError('final16/96/zero-update counters mismatch')
            summary={'status':'UFL_GPU16_NUMERICAL_CHECKS_PASSED_PENDING_SOL_REVIEW','phase':PHASE,
                     'completed':16,'forwards':16,'parameter_autograd_calls':96,'optimizer_step_attempts':0,'parameter_updates':0,
                     'max_delta_to_A0':max(x['delta_to_A0'] for x in rows),
                     'max_reserved_MiB':max(x['GPU_peak_reserved_MiB'] for x in rows),
                     'max_AMP_total_gradient_relative_L2':max(x['AMP_vs_FP32']['total_gradient']['relative_L2'] for x in rows if x['AMP_vs_FP32']),
                     'min_AMP_total_gradient_cosine':min(x['AMP_vs_FP32']['total_gradient']['cosine'] for x in rows if x['AMP_vs_FP32']),
                     'model_and_RNG_unchanged':True,'elapsed_seconds':time.monotonic()-started,
                     'resource_before':resource_before,'GPU1_queried':False,'Dev_or_Test_images_read':False,
                     'scientific_training_authorized':False}
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
    binding=verify_authority(); identity=process_identity()
    def timeout(*_): raise TimeoutError('UFL diagnostic30min cap')
    signal.signal(signal.SIGALRM,timeout)
    with exclusive_lock(OUT/'RUN.lock'), exclusive_lock(ROOT/'experiments/a14_gpu0.lock'):
        if DiagnosticLedger(LEDGER_PATH,canonical_sha(binding)).snapshot()['entries']:
            raise GateError('already consumed diagnostic work; no restart')
        try:
            resource=resources(True)
            if resource is None: raise GateError('GPU0 busy; stop without interference')
            verify_authority(); signal.alarm(1800)
            atomic_json(OUT/'queue_state.json',{'status':'RUNNING','owner':identity,'GPU_initialized':False})
            update_status(run_state='RUNNING_DIAGNOSTICS',active_pid=os.getpid())
            run_gpu(binding,resource)
            atomic_json(OUT/'queue_state.json',{'status':'COMPLETE_PENDING_SOL_REVIEW','owner':identity,'parameter_updates':0})
        except BaseException as exc:
            if not (OUT/'failure.json').exists():
                DiagnosticLedger(LEDGER_PATH,canonical_sha(binding)).fail(None,repr(exc))
                write_new(OUT/'queue_failure.json',{'error':repr(exc),'owner':identity,'no_auto_retry':True,'parameter_updates':0})
            update_status(state='HOLD_FAILED',run_state='FAILED_STOP')
            raise
        finally: signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight()
    else:run_queued()


if __name__=='__main__':main()
