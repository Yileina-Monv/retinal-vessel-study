"""Authenticated engineering segment or unique fresh-M0 scientific worker."""
import os,sys,json,time,ctypes,signal,math
from pathlib import Path
from scse_gate import STAGE as S,CODE,OUT,GPU,protocol,validate_release,runtime_authorizer,resources,dataset_guard,sha,update_status
from retinal_m0.maples_g1_safety import atomic_json,read_json,GateError

def setup(seed):
    import random,numpy as np,torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def build(device,*,cpu=False):
    import torch
    from scse_model import SCSEUNet,load_m0
    from retinal_m0.maples_core import CorePatches
    from scse_core import rng_state,state_digest
    q=protocol();cfg=q['baseline_config'];saved=torch.load(cfg['source_m0'],map_location='cpu',weights_only=True)
    if saved['step']!=10000:raise GateError('M0 is not step10000')
    setup(2026);model=SCSEUNet().to(device);load_m0(model,saved['model']);del saved
    if sum(p.numel() for p in model.scse.parameters())!=1105:raise GateError('unexpected module parameters')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'],enabled=not cpu)
    ds=CorePatches(cfg,'T0');ds.trajectory_sha256=q['expected_pairs_sha256'];pairs=[json.loads(x) for x in Path(q['expected_pairs_path']).read_text().splitlines()]
    return model,opt,scaler,ds,pairs

def diagnostic(model,ds,authorize):
    import torch
    from retinal_m0.model import UNet
    from scse_core import rng_state,restore_rng,state_digest
    from torch.utils.data import default_collate
    path=S/'engineering_diagnostics.json'
    if path.exists():raise GateError('GPU diagnostics already attempted')
    b=default_collate([ds[i] for i in range(4)]);before=rng_state('cuda');was=model.training;model.eval();base=UNet().cuda().eval();base.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith('scse.')});x=b['image'].cuda();record={'forward_cap':6,'forwards':0,'updates':0,'status':'RUNNING','results':[]};atomic_json(path,record)
    try:
        for amp in [False,True]:
            outputs=[]
            for mode in ['A0','bypass','identity_scse']:
                authorize();record['forwards']+=1
                if record['forwards']>6:raise GateError('diagnostic cap exceeded')
                atomic_json(path,record);model.bypass=mode=='bypass'
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):z=(base if mode=='A0' else model)(x)
                if not torch.isfinite(z).all():raise GateError('nonfinite diagnostic output')
                outputs.append(z.detach().cpu())
            equal=[torch.equal(outputs[0],y) for y in outputs[1:]]
            record['results'].append({'amp':amp,'bypass_exact':equal[0],'identity_exact':equal[1],'max_abs':[float((outputs[0]-y).abs().max()) for y in outputs[1:]]});atomic_json(path,record)
            if not all(equal):raise GateError('initial same-precision module output is not identity')
        record['status']='PASS';atomic_json(path,record)
    finally:model.bypass=False;model.train(was);restore_rng(before,'cuda');del base,x

def execute(role,track,start,end,parent,nonce):
    if os.getppid()!=parent or os.environ.get('SCSE_NONCE')!=nonce or len(nonce)<32:raise GateError('not authenticated direct child')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:raise GateError('parent death guard failed')
    key=validate_release(role,full_sources=True);session=read_json(S/'CURRENT_CHILD.json')
    if session['args']!={'role':role,'track':track,'start':start,'end':end}:raise GateError('CLI/session scope mismatch')
    if role=='engineering' and [track,start,end] not in protocol()['engineering']['segments']:raise GateError('unapproved engineering segment')
    if role=='science' and (track,start,end)!=('SCSE',0,2000):raise GateError('unapproved scientific segment')
    dataset_guard();authorize=runtime_authorizer(key)
    import torch
    from scse_core import Budget,run_updates,verify_checkpoint,rng_state,state_digest
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or torch.cuda.device_count()!=1:raise GateError('only released GPU1 may be visible')
    snap=resources();torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(8589934592/(snap['total_MiB']*1048576),0)
    model,opt,scaler,ds,pairs=build('cuda')
    if role=='engineering' and track=='CONT':diagnostic(model,ds,authorize)
    q=protocol();out=S/'engineering'/track if role=='engineering' else OUT
    ledger=Budget(S/('engineering_budget.json' if role=='engineering' else 'science_budget.json'),key,global_cap=32 if role=='engineering' else 2000,track_caps={'CONT':16,'SPLIT':16} if role=='engineering' else {'SCSE':2000},forward_cap=32 if role=='engineering' else 2000,gradient_cap=32 if role=='engineering' else 2000)
    origin={k:state_digest(v) for k,v in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state('cuda'))]}
    atomic_json(S/(f'initial_{track}_{start:04d}.json'),{'origin':origin,'source_m0_sha256':q['baseline_config']['source_m0_sha256'],'M0_shared_weights_verified':True,'new_parameters':1105,'engineering_weights_used_for_science':False,'role':role,'pid':os.getpid()})
    def callback(step,payload):
        if role=='science' and step<=16:
            old=read_json(S/'engineering_acceptance_rows.json')[step-1];row=json.loads((out/'train.jsonl').read_text().splitlines()[-1])
            for k in ['pair','loss','preclip_total_L2','amp_scale_before','amp_scale_after','state_digests','module_gradient_L2']:
                if row[k]!=old[k]:raise GateError('same-GPU initial scientific/engineering mismatch '+k)
        atomic_json(S/('heartbeat_engineering.json' if role=='engineering' else 'heartbeat.json'),{'role':role,'track':track,'step':step,'time':time.time(),'counts':payload['budget'],'elapsed_seconds':payload['elapsed_seconds'],'peak_reserved_MiB':torch.cuda.max_memory_reserved(0)/1048576})
        if step==1 or step%25==0:print('SCSE_UPDATE',role,track,step,end,'seconds',round(payload['elapsed_seconds'],2),flush=True)
        if role=='science' and (step==1 or step%200==0):update_status(completed_steps=step,charged_updates=payload['budget']['charged_events'],execution_started=True)
        if step%25==0:resources()
        if role=='science' and step%200==0:
            from scse_evaluation import evaluate_checkpoint
            validate_release('science',full_sources=True);evaluate_checkpoint(model,step,key)
        if role=='engineering' and time.monotonic()-session['engineering_wall_start']>=900:raise GateError('engineering cumulative15minute cap reached')
    result=run_updates(model=model,optimizer=opt,scaler=scaler,dataset=ds,track=track,start=start,end=end,out=out,budget=ledger,binding_sha=key,expected_pairs=pairs,device='cuda',authorize=authorize,on_update=callback)
    saved=verify_checkpoint(out/'last.pt',expected_step=end,binding_sha=key,run_dir=out,ledger=ledger);del saved
    if role=='science':
        evaluation=read_json(S/'evaluation_budget.json');images=[r for s in evaluation['slots'].values() for r in s['images'].values()]
        if len(evaluation['slots'])!=10 or len(images)!=280 or any(x['status']!='COMPLETE' or x['attempts']!=1 for x in images) or sum(x['forward_calls'] for x in images)!=1900 or sum(x['tiles'] for x in images)!=7450:raise GateError('fixed evaluation budget incomplete')
        atomic_json(S/'training_completion.json',{'status':'TRAINED2000_FIXED_TEN_EVALUATIONS','binding_sha256':key,'result':result,'counts':ledger.snapshot(),'Test_accessed':False})
    atomic_json(S/(f'result_{role}_{track}_{start:04d}.json'),result)

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['engineering','science'],required=True);p.add_argument('--track',choices=['CONT','SPLIT','SCSE'],required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);p.add_argument('--parent-pid',type=int,required=True);p.add_argument('--nonce',required=True);a=p.parse_args()
    try:execute(a.role,a.track,a.start,a.end,a.parent_pid,a.nonce)
    except BaseException as e:
        print('SCSE_WORKER_STOP',repr(e),traceback.format_exc(),flush=True);raise
