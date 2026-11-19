"""Candidate-C bounded engineering controller: 50 versus 25+25, not science."""
from __future__ import annotations
import argparse,json,os,random,subprocess,sys,time,shutil
from pathlib import Path
from retinal_m0.maples_g1_safety import (GateError,atomic_json,canonical_sha,exclusive_lock,file_sha,read_json,process_identity,_unique_pairs)
from retinal_m0.maples_soft_cldice_engineering_core import Budget,empty_budget

ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
CPU=BASE/'soft_cldice_cpu_v1_20261023';DIAG=BASE/'soft_cldice_gpu_diag_v1_20261023'
OUT=BASE/'soft_cldice_engineering_100_v1_20261023'
BINDING=OUT/'execution_binding.json';RELEASE=OUT/'ENGINEERING_RELEASE.json';RECEIPT=OUT/'CPU_PREFLIGHT_REVIEW.json';LEDGER=OUT/'engineering_ledger.json'
SESSION=OUT/'RUN_SESSION.json';SEGMENT=OUT/'CURRENT_SEGMENT.json'
T0=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
GPU_UUID='${GPU_UUID}'
PHASE='C_SOFT_CLDICE_ENGINEERING_100_V1'
START='<!-- C_SOFT_CLDICE_ENG100_RELEASE_CURRENT -->';END='<!-- /C_SOFT_CLDICE_ENG100_RELEASE_CURRENT -->'
SEGMENTS={'CONT':('CONT',0,50),'SPLIT_FIRST':('SPLIT',0,25),'SPLIT_RESUME':('SPLIT',25,50)}

def write_new(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def resources(idle=False):
    v=[x.strip() for x in subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')]
    if len(v)!=5 or v[0]!='0' or v[1]!=GPU_UUID:raise GateError('GPU0 identity mismatch')
    pids=[int(x) for x in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines() if x.strip()]
    d={'index':0,'uuid':v[1],'total_MiB':int(v[2]),'used_MiB':int(v[3]),'util_percent':int(v[4]),'compute_pids':pids,'GPU1_queried':False}
    if idle and (pids or d['used_MiB']>=512 or d['util_percent']>=10):raise GateError('GPU0 busy; other processes left untouched')
    if not idle and any(p!=os.getpid() for p in pids):raise GateError('another GPU0 compute process appeared')
    if shutil.disk_usage(OUT).free<5*1024**3:raise GateError('insufficient engineering checkpoint space')
    return d

def current_status(text):
    if text.count(START)!=1 or text.count(END)!=1:raise GateError('exactly one current100-step engineering block required')
    a=text.index(START)+len(START);b=text.index(END)
    if b<=a:raise GateError('malformed engineering status block')
    return json.loads(text[a:b],object_pairs_hook=_unique_pairs)

def wait_idle_after_own_worker():
    """Do not mistake the just-finished worker's utilization sample for a live job."""
    until=time.monotonic()+10
    while True:
        snap=resources()
        if not snap['compute_pids'] and snap['used_MiB']<512 and snap['util_percent']<10:return snap
        if time.monotonic()>=until:raise GateError('GPU0 did not become idle after own worker')
        time.sleep(.5)

def authority(*,full_sources=False):
    state=current_status((CODE/'STATUS.md').read_text())
    if state.get('state')!='RELEASED' or state.get('phase')!=PHASE:raise GateError('engineering100 not RELEASED')
    for p in [BINDING,RELEASE,RECEIPT,LEDGER]:
        if not p.is_file() or p.is_symlink():raise GateError('missing/symlink engineering evidence')
    b=read_json(BINDING);r=read_json(RELEASE);receipt=read_json(RECEIPT)
    cfg=read_json(CPU/'candidate_C_engineering_config.json')
    if b.get('phase')!=PHASE or b.get('global_charged_cap')!=100 or b.get('track_caps')!={'CONT':50,'SPLIT':50} or b.get('segments')!=[['CONT',0,50],['SPLIT',0,25],['SPLIT',25,50]] or b.get('eta')!=cfg['eta'] or b.get('k')!=cfg['k'] or b.get('pause_ratio_gt')!=2 or b.get('GPU0_uuid')!=GPU_UUID:raise GateError('engineering scope/loss/budget drift')
    digest=canonical_sha(b)
    expected={'phase':PHASE,'authorized':True,'release_id':'sol_C_engineering100_20261023_01','global_charged_cap':100,'gpu_uuid':GPU_UUID,'binding_sha256':digest,'binding_file_sha256':file_sha(BINDING),'cpu_review_sha256':file_sha(RECEIPT),'ledger_path':str(LEDGER)}
    for k,v in expected.items():
        if type(r.get(k))is not type(v) or r[k]!=v:raise GateError('engineering release mismatch:'+k)
    if state.get('release_id')!=r['release_id'] or state.get('binding_sha256')!=digest or state.get('release_file_sha256')!=file_sha(RELEASE):raise GateError('STATUS/release execution binding mismatch')
    if receipt.get('status')!='CPU_ENGINEERING_TESTS_ACCEPTED_GPU_HELD' or receipt.get('binding_sha256')!=digest or receipt.get('tests_passed') is not True:raise GateError('CPU engineering receipt mismatch')
    for p,h in receipt['evidence_sha256'].items():
        if file_sha(p)!=h:raise GateError('CPU evidence drift:'+p)
    for p,h in b['sources_sha256'].items():
        if (full_sources or p in b['runtime_guard_paths']) and file_sha(p)!=h:raise GateError('engineering source/data drift:'+p)
    return b

def worker_authority(track,start,end):
    b=authority();session=read_json(SESSION);active=read_json(SEGMENT)
    if session.get('binding_sha256')!=canonical_sha(b) or session.get('status')!='RUNNING':raise GateError('no active engineering parent session')
    parent=session['controller']
    if parent['pid']!=os.getppid() or process_identity(parent['pid'])!=parent:raise GateError('worker not launched by current live lock-owning controller')
    if active.get('binding_sha256')!=canonical_sha(b) or active.get('segment')!=[track,start,end]:raise GateError('worker segment outside active approval')
    return b

def prepare():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from retinal_m0.maples_core import CorePatches
    from torch.utils.data._utils.collate import default_collate
    from retinal_m0.maples_soft_cldice_engineering_core import pair_record
    cfg=read_json(CPU/'candidate_C_engineering_config.json');diag=read_json(DIAG/'SOL_GPU_DIAGNOSTIC_REVIEW.json')
    if cfg['eta']!=.15115714003619224 or cfg['k']!=26 or diag['completed']!=31 or diag['retained_failed']!=1 or diag['actual_cumulative_model_forwards']!=32:raise GateError('CPU/diagnostic basis drift')
    config=read_json(T0/'config.json');pairs=[json.loads(x) for x in (T0/'pairing.jsonl').read_text().splitlines()][:50]
    ds=CorePatches(config,'T0');actual=[]
    for step in range(1,51):
        batch=default_collate([ds[j] for j in range((step-1)*4,step*4)]);p=pair_record(batch,step)
        if p!=pairs[step-1]:raise GateError('first200 native inputs disagree with original T0')
        actual.append(p)
    accepted=read_json(CPU/'attempt_03/source_manifest.json');sources={}
    if file_sha(T0/'config.json')!=accepted[str(T0/'config.json')]:raise GateError('original T0 configuration differs from accepted CPU source')
    formal=read_json(ROOT/'experiments/t0_tw_core_20261016/execution_binding.json')
    for rel in ['retinal_m0/maples_core.py','retinal_m0/model.py','retinal_data/fives.py','retinal_data/maples_native.py','retinal_data/maples_fov.py','configs/development_v1.json']:
        if file_sha(CODE/rel)!=formal['sources'][rel]:raise GateError('original formal source changed:'+rel)
    for row in ds.native.rows:
        for k in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
            p=str((Path(config['data_root'])/row[k]).resolve())
            if p not in accepted or file_sha(p)!=accepted[p]:raise GateError('Train source differs from accepted CPU stage')
            sources[p]=accepted[p]
    files=[T0/'config.json',T0/'pairing.jsonl',Path(config['source_m0']),CPU/'candidate_C_engineering_config.json',DIAG/'SOL_GPU_DIAGNOSTIC_REVIEW.json',CODE/'configs/development_v1.json']
    files += [CODE/'retinal_m0'/n for n in ['maples_soft_cldice.py','maples_soft_cldice_engineering.py','maples_soft_cldice_engineering_core.py','maples_core.py','model.py','state.py','maples_g1_safety.py','maples_g2_engineering.py']]
    files += [CODE/'retinal_data'/n for n in ['fives.py','maples_native.py','maples_fov.py']]
    files += [CODE/'tests/test_maples_soft_cldice_engineering.py']
    sources.update({str(p):file_sha(p) for p in files})
    if file_sha(config['source_m0'])!=config['source_m0_sha256'] or file_sha(CODE/'retinal_m0/maples_soft_cldice.py')!=cfg['loss_source_sha256']:raise GateError('M0/accepted loss drift')
    b={'phase':PHASE,'global_charged_cap':100,'track_caps':{'CONT':50,'SPLIT':50},'segments':[['CONT',0,50],['SPLIT',0,25],['SPLIT',25,50]],'eta':cfg['eta'],'k':26,'pause_ratio_gt':2,
       'GPU0_uuid':GPU_UUID,'memory_allocator_cap_bytes':8*1024**3,'forward_cap':100,'gradient_cap':300,'audit_every_attempt':True,'AMP_overflow':'stop_without_retry','worker_timeout_seconds':1800,
       'sources_sha256':sources,'runtime_guard_paths':[str(p) for p in files if str(p)!=config['source_m0']],
       'M0_path':config['source_m0'],'M0_sha256':config['source_m0_sha256'],'config_path':str(T0/'config.json'),'expected_pairs':actual,
       'Dev_Test_inference_or_scientific_training_allowed':False,'retained_diagnostic_OOM_accepted_by_current_user_approval':True}
    OUT.mkdir(parents=True,exist_ok=True);write_new(BINDING,b)
    write_new(OUT/'cpu_preflight.json',{'status':'PASSED_CPU_ONLY','binding_sha256':canonical_sha(b),'original_T0_steps':50,'paired_draws':200,'new_M0_forwards':0,'GPU_initialized':False})
    print('PREPARED',canonical_sha(b),flush=True)

def worker(name):
    track,start,end=SEGMENTS[name];b=worker_authority(track,start,end)
    worker_authority(track,start,end);resources(idle=True)
    os.environ['CUDA_VISIBLE_DEVICES']=GPU_UUID
    import torch,numpy as np
    from retinal_m0.model import UNet
    from retinal_m0.maples_core import CorePatches
    from retinal_m0.state import state_digest
    from retinal_m0.maples_soft_cldice_engineering_core import run_segment
    if torch.cuda.is_initialized():raise GateError('worker CUDA must initialize after authority/resources')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    if torch.cuda.device_count()!=1:raise GateError('only GPU0 may be visible')
    torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(min(.8,8*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
    config=read_json(b['config_path']);arch=read_json(CODE/'configs/development_v1.json')
    model=UNet(arch['widths'],arch['group_norm_groups']).cuda()
    if start==0:
        cp=torch.load(b['M0_path'],map_location='cpu',weights_only=True)
        if cp['step']!=10000:raise GateError('wrong M0 step')
        model.load_state_dict(cp['model'],strict=True)
        if state_digest(model.state_dict())!=state_digest(cp['model']):raise GateError('M0 loaded tensor mismatch')
        del cp
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['eps'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=config['amp_initial_scale'])
    ds=CorePatches(config,'T0');budget=Budget(LEDGER,canonical_sha(b))
    result=run_segment(model=model,optimizer=optimizer,scaler=scaler,dataset=ds,track=track,start=start,end=end,out=OUT/'runs'/track,budget=budget,binding_sha=canonical_sha(b),expected_pairs=b['expected_pairs'],device='cuda:0',eta=b['eta'],k=b['k'])
    write_new(OUT/(name+'_completion.json'),result)

def controller():
    b=authority(full_sources=True)
    with exclusive_lock(OUT/'RUN.lock'),exclusive_lock(ROOT/'experiments/a14_gpu0.lock'):
        if read_json(LEDGER)!=empty_budget(canonical_sha(b)):raise GateError('controller requires initial zero ledger; no implicit restart')
        write_new(SESSION,{'status':'RUNNING','controller':process_identity(),'binding_sha256':canonical_sha(b),'segments':list(SEGMENTS)})
        budget=Budget(LEDGER,canonical_sha(b));started=time.monotonic()
        try:
            for name,segment in SEGMENTS.items():
                authority();wait_idle_after_own_worker()
                atomic_json(SEGMENT,{'binding_sha256':canonical_sha(b),'segment':list(segment),'name':name})
                with (OUT/(name+'.log')).open('x') as log:
                    p=subprocess.run([sys.executable,'-B','-m','retinal_m0.maples_soft_cldice_engineering','--worker',name],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                if p.returncode!=0:raise GateError('worker stopped:'+name)
                print('SEGMENT_COMPLETE',name,flush=True)
            import torch
            from retinal_m0.state import state_digest
            cont=[json.loads(x) for x in (OUT/'runs/CONT/train.jsonl').read_text().splitlines()]
            split=[json.loads(x) for x in (OUT/'runs/SPLIT/train.jsonl').read_text().splitlines()]
            if len(cont)!=50 or len(split)!=50:raise GateError('incomplete paired engineering logs')
            compared=['pair','A0_loss','raw_aux_loss','weighted_aux_loss','loss','audit','loss_scale_before','loss_scale_after','preclip_total_L2','state_digests']
            for a,c in zip(cont,split):
                for field in compared:
                    if a[field]!=c[field]:raise GateError(f'paired exact mismatch step{a["step"]}: {field}')
            ca=torch.load(OUT/'runs/CONT/last.pt',map_location='cpu',weights_only=True);cb=torch.load(OUT/'runs/SPLIT/last.pt',map_location='cpu',weights_only=True)
            for k in ['model','optimizer','scaler','rng','loader_generator']:
                if state_digest(ca[k])!=state_digest(cb[k]):raise GateError('terminal checkpoint mismatch:'+k)
            restore=read_json(OUT/'runs/SPLIT/restore_025.json')
            if restore['new_pid']==restore['checkpoint_pid']:raise GateError('resume did not use new process')
            s=budget.snapshot()
            if s['verified_successes']!=100 or s['charged_events']!=100 or s['optimizer_call_starts']!=100 or s['failed_events'] or s['uncertain_events'] or s['failed_compute'] or s['compute_pending']:raise GateError('100-update accounting not clean')
            authority(full_sources=True);wait_idle_after_own_worker()
            summary={'status':'ENGINEERING_100_COMPLETE_EXACT_RESTORE_PASSED','charged_updates':100,'verified_successful_updates':100,'forwards':s['forward_calls'],'gradient_calls':s['gradient_calls'],'failed_updates':0,'uncertain_updates':0,'paired_steps_exact':50,'new_process_restore_exact':True,'eta':b['eta'],'k':b['k'],'max_aux_to_A0':max(r['audit']['aux_to_A0'] or 0 for r in cont+split),'negative_cosine_steps':sum(r['audit']['cosine'] is not None and r['audit']['cosine']<0 for r in cont+split),'GPU_reserved_peak_MiB':max(r['GPU_peak_reserved_MiB'] for r in cont+split),'elapsed_seconds':time.monotonic()-started,'Dev_Test_content_used':False,'GPU1_queried_or_used':False,'scientific_training_authorized':False}
            write_new(OUT/'completion.json',summary)
            atomic_json(SESSION,{**read_json(SESSION),'status':'COMPLETE'})
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        except BaseException as e:
            s=budget.snapshot()
            for track,t in s['tracks'].items():
                if t.get('pending') is not None:budget.resolve(track,'uncertain')
            budget.stop(repr(e));atomic_json(SESSION,{**read_json(SESSION),'status':'FAILED_STOP'})
            write_new(OUT/'failure.json',{'error':repr(e),'budget':budget.snapshot(),'no_auto_restart':True,'scientific_training_authorized':False})
            raise

def main():
    p=argparse.ArgumentParser(description='Candidate C engineering100 only: 50 versus25+25, no Dev/Test/science')
    a=p.add_mutually_exclusive_group(required=True);a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true');a.add_argument('--worker',choices=list(SEGMENTS))
    args=p.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker(args.worker)
    else:controller()

if __name__=='__main__':main()
