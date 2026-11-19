"""Candidate C only: 32 fixed GPU0 diagnostic forwards, zero optimizer updates.

Import/preflight never initializes CUDA. This entry cannot start G2 or training.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, random, signal, subprocess, sys, time, shutil
from pathlib import Path
from retinal_m0.maples_g1_safety import (GateError, Ledger, atomic_json, canonical_sha,
    exclusive_lock, file_sha, finite_tree, read_json, safe_evidence, _unique_pairs)

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
CPU=BASE/'soft_cldice_cpu_v1_20261023'
OUT=BASE/'soft_cldice_gpu_diag_v1_20261023'
BATCHES=BASE/'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
T0=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
BINDING=OUT/'execution_binding_v2.json'
RELEASE=OUT/'DIAGNOSTIC_RELEASE_02.json'
RECEIPT=OUT/'CPU_PREFLIGHT_REVIEW_v2.json'
LEDGER=OUT/'diagnostic_ledger.json'
RETAINED=OUT/'retained_OOM_ledger_before_v2.json'
ACKNOWLEDGED_OOM='b00_FP32_eta0'
GPU_UUID='${GPU_UUID}'
PHASE='C_SOFT_CLDICE_GPU_ZERO_UPDATE_V1'
START='<!-- C_SOFT_CLDICE_GPU_DIAG_RELEASE_CURRENT -->'
END='<!-- /C_SOFT_CLDICE_GPU_DIAG_RELEASE_CURRENT -->'
CAP=32
MEMORY_CAP=8*1024**3
WALL_SECONDS=1800

def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def plan():
    return [{'id':f'b{i:02d}_{p}_{mode}','batch_index':i,'precision':p,'mode':mode}
            for i in range(8) for p in ['FP32','AMP'] for mode in ['eta0','candidate']]

def precision_rules():
    return {'FP32_composite_gradient_relative_limit':.001,'AMP_composite_gradient_relative_limit':.05,
            'AMP_vs_FP32_loss_relative_limit':.01,'AMP_vs_FP32_total_gradient_relative_limit':.05,
            'AMP_vs_FP32_total_gradient_cosine_min':.99,'AMP_vs_FP32_aux_gradient_relative_limit':.10,
            'AMP_vs_FP32_aux_gradient_cosine_min':.98,'pause_aux_to_A0_gt':2}

def status_release(text):
    if text.count(START)!=1 or text.count(END)!=1:raise GateError('one current candidate-C release block required')
    a=text.index(START)+len(START);b=text.index(END)
    if b<=a:raise GateError('malformed candidate-C release block')
    return json.loads(text[a:b].strip(),object_pairs_hook=_unique_pairs)

def verify_authority(*,check_sources=True):
    state=status_release((CODE/'STATUS.md').read_text())
    if state.get('state')!='RELEASED' or state.get('phase')!=PHASE:raise GateError('candidate-C diagnostic is not RELEASED')
    for p in [RELEASE,BINDING,RECEIPT,LEDGER]:
        if p.is_symlink() or not p.is_file():raise GateError('missing or symlink diagnostic evidence:'+str(p))
    b=read_json(BINDING);r=read_json(RELEASE);receipt=read_json(RECEIPT)
    if b.get('phase')!=PHASE or b.get('plan')!=plan() or b.get('execution_plan')!=[c for c in plan() if c['id']!=ACKNOWLEDGED_OOM] or b.get('parameter_updates')!=0 or b.get('max_forwards')!=32 or b.get('gpu_uuid')!=GPU_UUID:
        raise GateError('immutable diagnostic scope drift')
    cfg=read_json(CPU/'candidate_C_engineering_config.json')
    if b.get('eta')!=cfg['eta'] or b.get('k')!=cfg['k'] or b.get('precision_rules')!=precision_rules() or b.get('allocator_memory_cap_bytes')!=MEMORY_CAP or b.get('wall_seconds_cap')!=WALL_SECONDS:
        raise GateError('accepted eta/k or numerical/resource policy drift')
    digest=canonical_sha(b)
    expected={'phase':PHASE,'authorized':True,'max_forwards':32,'parameter_updates':0,
              'gpu_uuid':GPU_UUID,'binding_sha256':digest,'binding_file_sha256':file_sha(BINDING),
              'cpu_preflight_review_sha256':file_sha(RECEIPT),'ledger_path':str(LEDGER)}
    for k,v in expected.items():
        if type(r.get(k)) is not type(v) or r[k]!=v:raise GateError('candidate release mismatch:'+k)
    if receipt.get('status')!='CPU_PREFLIGHT_ACCEPTED_GPU_NOT_YET_EXECUTED' or receipt.get('binding_sha256')!=digest or receipt.get('tests_passed') is not True:
        raise GateError('CPU preflight review mismatch')
    if state.get('release_id')!=r.get('release_id') or state.get('binding_sha256')!=digest or state.get('release_file_sha256')!=file_sha(RELEASE):
        raise GateError('STATUS/release binding mismatch')
    if r.get('release_id')!='sol_C_cldice_gpu_diag_20261023_02':raise GateError('wrong candidate release ID')
    retained=read_json(RETAINED);live=read_json(LEDGER)
    if retained.get('forward_starts')!=1 or retained.get('autograd_calls')!=3 or retained.get('optimizer_step_attempts')!=0 or list(retained['combinations'])!=[ACKNOWLEDGED_OOM]:raise GateError('invalid retained first OOM evidence')
    if live['forward_starts']<1 or live['combinations'].get(ACKNOWLEDGED_OOM)!=retained['combinations'][ACKNOWLEDGED_OOM]:raise GateError('original failure/counters were erased or altered')
    for p,h in b['sources_sha256'].items():
        if (check_sources or p in b['runtime_guard_paths']) and file_sha(p)!=h:raise GateError('reviewed source drift:'+p)
    for p,h in receipt['test_evidence_sha256'].items():
        if file_sha(p)!=h:raise GateError('preflight test evidence changed:'+p)
    return b

def resources(require_idle=False):
    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    x=[v.strip() for v in raw.strip().split(',')]
    if len(x)!=5 or x[0]!='0' or x[1]!=GPU_UUID:raise GateError('physical GPU0 identity mismatch')
    apps=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    pids=[int(v.strip()) for v in apps.splitlines() if v.strip()]
    result={'index':0,'uuid':x[1],'total_MiB':int(x[2]),'used_MiB':int(x[3]),'util_percent':int(x[4]),'compute_pids':pids,'GPU1_queried':False,'time':time.time()}
    result['free_disk_bytes']=shutil.disk_usage(OUT).free
    if result['free_disk_bytes']<1024**3:raise GateError('insufficient diagnostic log space')
    if require_idle and (result['used_MiB']>=512 or result['util_percent']>=10 or pids):raise GateError('GPU0 busy; leave other processes untouched')
    if not require_idle and any(p!=os.getpid() for p in pids):raise GateError('another GPU0 process appeared')
    return result

def get_batches():
    import torch
    from torch.utils.data._utils.collate import default_collate
    from retinal_m0.maples_core import CorePatches
    config=read_json(T0/'config.json');fixed=read_json(BATCHES)['batches']
    if len(fixed)!=8:raise GateError('exact original8 fixed batches required')
    ds=CorePatches(config,'T0');batches=[]
    selected={r['filename'] for s in fixed for r in s['draws']}
    prior=read_json(CPU/'attempt_03/source_manifest.json');sources={}
    for row in ds.native.rows:
        if row['filename'] not in selected:continue
        for k in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
            p=str((Path(config['data_root'])/row[k]).resolve())
            if p not in prior or file_sha(p)!=prior[p]:raise GateError('Train data differs from CPU accepted evidence:'+p)
            sources[p]=prior[p]
    for spec in fixed:
        items=[ds[int(r['draw'])] for r in spec['draws']]
        for item,ref in zip(items,spec['draws']):
            if item['filename']!=ref['filename'] or item['geometry'].tolist()!=ref['geometry'] or item['input_sha256']!=ref['input_sha256_original_T0']:
                raise GateError('original T0 draw/geometry/input drift')
        batches.append(default_collate(items))
    ds.cache.clear()
    return batches,sources

def preflight():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    if torch.cuda.is_initialized():raise GateError('preflight must remain CPU-only')
    review=read_json(CPU/'SOL_CPU_REVIEW.json');cfg=read_json(CPU/'candidate_C_engineering_config.json')
    if review['status']!='CPU_PREPARATION_ACCEPTED_GPU_HOLD_PENDING_USER' or cfg['eta']!=review['eta'] or cfg['k']!=26:
        raise GateError('candidate CPU acceptance mismatch')
    if file_sha(CODE/'retinal_m0/maples_soft_cldice.py')!=cfg['loss_source_sha256']:raise GateError('accepted loss changed')
    batches,data=get_batches();sources=dict(data)
    files=[CPU/'SOL_CPU_REVIEW.json',CPU/'candidate_C_engineering_config.json',CPU/'attempt_03/calibration.json',BATCHES,T0/'config.json',CODE/'configs/development_v1.json',RETAINED]
    files += [CODE/'retinal_m0'/n for n in ['maples_soft_cldice.py','maples_soft_cldice_gpu_diag.py','maples_core.py','model.py','state.py','maples_g1_safety.py']]
    files += [CODE/'retinal_data'/n for n in ['fives.py','maples_native.py','maples_fov.py']]
    files += [CODE/'tests/test_maples_soft_cldice_gpu_diag.py']
    config=read_json(T0/'config.json');m0=Path(config['source_m0'])
    if file_sha(m0)!=config['source_m0_sha256']:raise GateError('M0 source changed')
    files.append(m0)
    sources.update({str(p):file_sha(p) for p in files})
    formal=read_json(ROOT/'experiments/t0_tw_core_20261016/execution_binding.json')
    for rel in ['retinal_m0/maples_core.py','retinal_m0/model.py','retinal_data/fives.py','retinal_data/maples_native.py','retinal_data/maples_fov.py','configs/development_v1.json']:
        if file_sha(CODE/rel)!=formal['sources'][rel]:raise GateError('original baseline code drift:'+rel)
    binding={'phase':PHASE,'max_forwards':32,'parameter_updates':0,'autograd_per_combination_cap':4,
             'gpu_uuid':GPU_UUID,'plan':plan(),'eta':cfg['eta'],'k':cfg['k'],
             'execution_plan':[c for c in plan() if c['id']!=ACKNOWLEDGED_OOM],
             'retained_failed_ids':[ACKNOWLEDGED_OOM],'repair_scope':'skip failed combination, run remaining31 only; cumulative32 unchanged; no failed forward replay',
             'allocator_memory_cap_bytes':MEMORY_CAP,'wall_seconds_cap':WALL_SECONDS,
             'sources_sha256':sources,'runtime_guard_paths':[str(p) for p in files if p!=m0],
             'M0_path':str(m0),'M0_sha256':config['source_m0_sha256'],
             'precision_rules':precision_rules(),
             'batch_input_sha256':[list(x['input_sha256']) for x in batches],
             'CPU_review_only_does_not_authorize_GPU':True,'torch_version':str(torch.__version__)}
    OUT.mkdir(parents=True,exist_ok=True)
    write_new(BINDING,binding)
    write_new(OUT/'cpu_preflight_v2.json',{'status':'PASSED_CPU_ONLY','binding_sha256':canonical_sha(binding),'batches':8,'draws':32,'no_M0_loaded':True,'GPU_initialized':False,'model_forwards':0,'remaining_GPU_forwards':31,'prior_failed_GPU_forwards':1})
    print('CPU_PREFLIGHT_PASSED',canonical_sha(binding),flush=True)

def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in gs))
def dot(a,b):return sum(float((x.detach().double()*y.detach().double()).sum().cpu()) for x,y in zip(a,b))

def precision_difference(a,b):
    na,nb=norm(a),norm(b)
    return {'relative_L2':norm(tuple(x-y for x,y in zip(a,b)))/max(na,1e-30),
            'cosine':dot(a,b)/(na*nb) if na and nb else None}

class CandidateLedger(Ledger):
    """Keep the original failed entry; only the explicitly unrun31 may start."""
    def reserve(self,combo):
        def action(s):
            if combo==ACKNOWLEDGED_OOM or combo not in self.ids or combo in s['combinations'] or len(s['combinations'])>=32:raise GateError('failed/unknown/duplicate/exhausted combination')
            if any(k!=ACKNOWLEDGED_OOM and v['status']=='failed' for k,v in s['combinations'].items()):raise GateError('new failure; no automatic continuation')
            if ACKNOWLEDGED_OOM not in s['combinations'] or s['combinations'][ACKNOWLEDGED_OOM]['status']!='failed':raise GateError('retained failure missing')
            s['combinations'][combo]={'status':'reserved','owner':self.owner,'forward_starts':0,'autograd_calls':0,'reserved_at':time.time(),'failure':None}
        self._change(action)

def objectives(z,y,f,*,k,eta):
    """No full-size auxiliary graph for eta0. Empty-GT is a small loss-only control."""
    import torch
    from retinal_m0.maples_core import weighted_masked_loss
    from retinal_m0.maples_soft_cldice import candidate_loss,soft_cldice_terms
    base=weighted_masked_loss(z,y,f,f);zero=candidate_loss(z,y,f,k=k,eta=0)
    if eta==0:
        aux=z.float().sum()*0.
        terms={'GT_count':(y*f).sum((1,2,3)),'GT_skeleton_mass':None,'loss':None}
        return base,zero,terms,aux,zero
    terms=soft_cldice_terms(z,y,f,k=k);aux=eta*terms['loss']
    return base,zero,terms,aux,base+aux

@contextlib.contextmanager
def block_optimizers(torch,ledger,active):
    patches=[]
    def prohibited(*a,**kw):ledger.prohibit_optimizer(active['id'])
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer) and 'step' in cls.__dict__:
            patches.append((cls,cls.__dict__['step']));cls.step=prohibited
    try:yield
    finally:
        for cls,original in patches:cls.step=original

def run_gpu():
    binding=verify_authority()
    with exclusive_lock(OUT/'RUN.lock'),exclusive_lock(ROOT/'experiments/a14_gpu0.lock'):
        before=resources(require_idle=True)
        os.environ['CUDA_VISIBLE_DEVICES']=GPU_UUID
        import numpy as np
        import torch
        from retinal_m0.maples_core import weighted_masked_loss
        from retinal_m0.maples_soft_cldice import candidate_loss,soft_cldice_terms
        from retinal_m0.model import UNet
        from retinal_m0.state import state_digest
        if torch.cuda.is_initialized():raise GateError('authority/resource checks must precede CUDA initialization')
        batches,_=get_batches();verify_authority()
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
        if torch.cuda.device_count()!=1:raise GateError('exactly one permitted visible GPU required')
        torch.cuda.set_device(0)
        total=torch.cuda.get_device_properties(0).total_memory
        fraction=min(.8,MEMORY_CAP/total);torch.cuda.set_per_process_memory_fraction(fraction,0)
        arch=read_json(CODE/'configs/development_v1.json')
        model=UNet(arch['widths'],arch['group_norm_groups']).cuda().train()
        cp=torch.load(binding['M0_path'],map_location='cpu',weights_only=True)
        if cp['step']!=10000:raise GateError('wrong M0 step')
        model.load_state_dict(cp['model'],strict=True)
        if state_digest(model.state_dict())!=state_digest(cp['model']):raise GateError('M0 loaded tensor mismatch')
        del cp
        ps=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
        initial_hash=state_digest(model.state_dict())
        def rng_digest():
            n=np.random.get_state()
            return state_digest({'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch':torch.get_rng_state(),'cuda0':torch.cuda.get_rng_state(0)})
        initial_rng=rng_digest();ledger=CandidateLedger(LEDGER,canonical_sha(binding),[x['id'] for x in plan()])
        out=OUT/'attempt_02';out.mkdir(exist_ok=False)
        active={'id':None};reference={};results=[];started=time.monotonic()
        def pre_hook(*_):
            verify_authority(check_sources=False);ledger.before_forward(active['id'])
        hook=model.register_forward_pre_hook(pre_hook)
        def timeout(*_):raise TimeoutError('candidate diagnostic wall-clock cap reached')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(WALL_SECONDS)
        error=None
        try:
            with block_optimizers(torch,ledger,active):
                for combo in binding['execution_plan']:
                    active['id']=combo['id'];verify_authority(check_sources=False);ledger.reserve(combo['id'])
                    rsrc=resources()
                    batch=batches[combo['batch_index']]
                    if list(batch['input_sha256'])!=binding['batch_input_sha256'][combo['batch_index']]:raise GateError('staged CPU input identity drift')
                    x,y,f=[batch[k].cuda() for k in ['image','mask','fov']]
                    torch.cuda.reset_peak_memory_stats(0);t=time.monotonic();amp=combo['precision']=='AMP'
                    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024.)
                    scale_before=scaler.state_dict();scale=float(scaler.get_scale())
                    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):z=model(x)
                    if z.dtype!=(torch.float16 if amp else torch.float32):raise GateError('requested model precision not used')
                    eta=0. if combo['mode']=='eta0' else binding['eta']
                    base,zero,terms,aux,total_loss=objectives(z,y,f,k=binding['k'],eta=eta)
                    if base.dtype!=torch.float32 or (terms['loss'] is not None and terms['loss'].dtype!=torch.float32):raise GateError('loss arithmetic not FP32')
                    def grad(loss):
                        ledger.autograd(combo['id'])
                        gg=tuple(v.detach().float()/scale for v in torch.autograd.grad(scaler.scale(loss),ps+(z,),retain_graph=True,allow_unused=False))
                        if not all(torch.isfinite(g).all() for g in gg):raise GateError('nonfinite actual unscaled gradient')
                        return gg[:-1],gg[-1]
                    gb,zb=grad(base);gz,zz=grad(zero);ga,za=grad(aux)
                    if eta==0:
                        zz_control=z[...,:64,:64]
                        empty=soft_cldice_terms(zz_control,torch.zeros_like(zz_control),torch.ones_like(zz_control),k=binding['k'])['loss']
                        ge,ze=grad(empty)
                        empty_ok=empty.item()==0 and not any(bool(torch.count_nonzero(g)) for g in ge) and not bool(torch.count_nonzero(ze))
                        if not empty_ok:raise GateError('empty GT auxiliary not zero')
                        gc,zc=gz,zz
                    else:gc,zc=grad(total_loss);empty_ok=None
                    zero_ok=torch.equal(base,zero) and torch.equal(zb,zz) and all(torch.equal(a,b) for a,b in zip(gb,gz))
                    if not zero_ok:raise GateError('same-precision eta0 not exact')
                    try:candidate_loss(z,y,torch.zeros_like(f),k=binding['k'],eta=eta)
                    except ValueError as e:
                        if 'empty_fov' not in str(e):raise
                    else:raise GateError('empty FOV was not rejected')
                    n0,n1,nc=norm(gb),norm(ga),norm(gc)
                    ratio=n1/n0 if n0 else None;cos=dot(gb,ga)/(n0*n1) if n0 and n1 else None
                    composite=norm(tuple(c-a-b for c,a,b in zip(gc,gb,ga)))/max(n0+n1,1e-30)
                    limit=binding['precision_rules']['AMP_composite_gradient_relative_limit' if amp else 'FP32_composite_gradient_relative_limit']
                    layers={n:{'A0_L2':norm((a,)),'weighted_aux_L2':norm((b,)),'combined_L2':norm((c,))} for n,a,b,c in zip(names,gb,ga,gc)}
                    comparison=None;key=(combo['batch_index'],combo['mode'])
                    if not amp:
                        reference[key]={'gb':tuple(g.cpu() for g in gb),'ga':tuple(g.cpu() for g in ga),'gc':tuple(g.cpu() for g in gc),'loss':float(total_loss.detach()),'origin':'matching_FP32_mode'}
                        if combo['batch_index']==0 and combo['mode']=='candidate':
                            # eta does not enter the model forward. A0 is measured from this same logits graph.
                            reference[(0,'eta0')]={'gb':tuple(g.cpu() for g in gb),'ga':tuple(torch.zeros_like(g.cpu()) for g in ga),'gc':tuple(g.cpu() for g in gb),'loss':float(base.detach()),'origin':'same_batch_FP32_candidate_measured_A0_gradient; not a replay or PASS of failed first combination'}
                    else:
                        ref=reference[key];comparison={'total':precision_difference(ref['gc'],tuple(g.cpu() for g in gc)),
                            'A0':precision_difference(ref['gb'],tuple(g.cpu() for g in gb)),
                            'aux':precision_difference(ref['ga'],tuple(g.cpu() for g in ga)) if eta else None,
                            'loss_relative':abs(float(total_loss.detach())-ref['loss'])/max(abs(ref['loss']),1e-12),'reference_origin':ref['origin']}
                    torch.cuda.synchronize(0)
                    row={**combo,'eta':eta,'k':binding['k'],'input_sha256':list(batch['input_sha256']),
                        'loss_A0':float(base.detach()),'loss_aux_raw':float(terms['loss'].detach()) if terms['loss'] is not None else None,'loss_aux_weighted':float(aux.detach()),'loss_combined':float(total_loss.detach()),
                        'GT_count':terms['GT_count'].detach().cpu().int().tolist(),'GT_skeleton_mass':terms['GT_skeleton_mass'].detach().cpu().tolist() if terms['GT_skeleton_mass'] is not None else None,
                        'raw_aux_NA_reason':'eta0 exact bypass, no full-sized auxiliary graph' if eta==0 else None,'empty_GT_control_shape':[4,1,64,64] if eta==0 else None,
                        'zero_eta_value_and_gradient_exact':zero_ok,'empty_GT_zero':empty_ok,'empty_FOV_rejected':True,
                        'actual_z_dtype':str(z.dtype),'loss_dtype':str(base.dtype),'loss_scale':scale,'unscaled':True,'preclip':True,
                        'A0_parameter_L2':n0,'weighted_aux_parameter_L2':n1,'combined_parameter_L2':nc,'aux_to_A0_L2':ratio,'cosine':cos,
                        'composite_relative_L2_error':composite,'per_layer':layers,'AMP_vs_FP32':comparison,
                        'GPU_allocated_peak_MiB':torch.cuda.max_memory_allocated(0)/1048576,'GPU_reserved_peak_MiB':torch.cuda.max_memory_reserved(0)/1048576,
                        'elapsed_seconds':time.monotonic()-t,'resource_before':rsrc,'parameter_updates':0}
                    finite_tree(row);write_new(out/(combo['id']+'.json'),row)
                    if (n0==0 and n1>0) or (ratio is not None and ratio>2):raise GateError('aux/A0 ratio>2 pause')
                    if composite>limit:raise GateError('composite gradient numeric tolerance exceeded')
                    if comparison:
                        rules=binding['precision_rules']
                        if comparison['loss_relative']>rules['AMP_vs_FP32_loss_relative_limit'] or comparison['total']['relative_L2']>rules['AMP_vs_FP32_total_gradient_relative_limit'] or comparison['total']['cosine'] is None or comparison['total']['cosine']<rules['AMP_vs_FP32_total_gradient_cosine_min']:raise GateError('AMP vs FP32 total/loss numeric difference exceeded')
                        if eta and (comparison['aux']['relative_L2']>rules['AMP_vs_FP32_aux_gradient_relative_limit'] or comparison['aux']['cosine'] is None or comparison['aux']['cosine']<rules['AMP_vs_FP32_aux_gradient_cosine_min']):raise GateError('AMP auxiliary gradient difference exceeded')
                    if torch.cuda.max_memory_reserved(0)>MEMORY_CAP:raise GateError('allocator exceeded 8GiB cap')
                    if scaler.state_dict()!=scale_before or any(p.grad is not None for p in ps):raise GateError('scaler/parameter.grad side effect')
                    if state_digest(model.state_dict())!=initial_hash or rng_digest()!=initial_rng:raise GateError('model/RNG mutated during zero-update diagnosis')
                    ledger.finish(combo['id']);results.append(row)
                    print('COMPLETED',len(results),'/31_remaining',combo['id'],'ratio',ratio,'cos',cos,'MiB',row['GPU_reserved_peak_MiB'],flush=True)
                    del z,base,zero,terms,aux,total_loss,gb,gz,ga,gc,zb,zz,za,zc,x,y,f
                    if eta==0:del empty,ge,ze,zz_control
            verify_authority();resources()
            snapshot=read_json(LEDGER)
            if snapshot['forward_starts']!=32 or snapshot['optimizer_step_attempts']!=0 or any(x['status']!='completed' for k,x in snapshot['combinations'].items() if k!=ACKNOWLEDGED_OOM):raise GateError('final cumulative accounting mismatch')
            summary={'status':'REMAINING_31_COMPLETE_WITH_ONE_RETAINED_OOM_NOT_32_PASSES','phase':PHASE,'completed':len(results),'retained_failed':1,'all32_passed':False,'forwards':32,'autograd_calls':snapshot['autograd_calls'],'parameter_updates':0,
                     'eta':binding['eta'],'k':binding['k'],'max_aux_to_A0':max(r['aux_to_A0_L2'] or 0 for r in results),'negative_cosine_count':sum(r['cosine'] is not None and r['cosine']<0 for r in results),
                     'max_reserved_MiB':max(r['GPU_reserved_peak_MiB'] for r in results),'elapsed_seconds':time.monotonic()-started,'model_and_RNG_unchanged':True,
                     'resource_before':before,'GPU1_queried':False,'Dev_or_Test_content_accessed':False,'new_training_authorized':False,'next':'Sol review; short update/resume engineering and 2000-step science still separately approved'}
            write_new(out/'completion.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException as exc:
            error=safe_evidence({'exception':repr(exc),'time':time.time()})
            current=read_json(LEDGER)['combinations'].get(active['id'])
            if current and current['status'] in ['reserved','started']:ledger.finish(active['id'],error=error)
            write_new(out/'failure.json',{'error':error,'ledger':read_json(LEDGER),'parameter_updates':0,'no_auto_retry':True})
            raise
        finally:
            signal.alarm(0);hook.remove()

def main():
    p=argparse.ArgumentParser(description='Candidate C GPU0 zero-update diagnostics ONLY')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--preflight',action='store_true');mode.add_argument('--run-gpu-diagnostics',action='store_true')
    args=p.parse_args()
    if args.preflight:preflight()
    else:run_gpu()

if __name__=='__main__':main()
