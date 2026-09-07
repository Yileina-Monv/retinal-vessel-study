"""Ticketed M0/M1 development runner. No ticket is created or run by preflight."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import math
import random
import subprocess
import time
import numpy as np
import torch
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save,rng_state,restore_rng,state_digest
from retinal_data.fives import EpochDrawSampler
from retinal_release.common import ROOT,read,write,digest,machine
from retinal_m1.verify import verify
from retinal_m1.data import SkeletonDataset as DevelopmentDataset, fingerprint
from retinal_m1.loss import m1_loss
from retinal_release.exchange import check_ticket

def main():
    p=argparse.ArgumentParser();p.add_argument('ticket');p.add_argument('--max-seconds',type=int,default=3000);p.add_argument('--stop-after-step',type=int);args=p.parse_args()
    ticket=read(args.ticket);check_ticket(ticket)
    if ticket['target_machine']!=machine():raise ValueError('Job assigned to another machine; no cross-machine resume')
    if ticket['kind']!='M0_M1_development_only' or ticket['microbatch']!=4 or ticket['accumulation']!=1:raise ValueError('Unsupported job profile')
    verified=verify()
    if verified['parent_release_id']!=ticket['release_id'] or verified['addon_id']!=ticket.get('addon_id'):raise ValueError('Job uses a different release')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one real CUDA GPU required')
    gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if gpu_uuid!=ticket['gpu_uuid']:raise ValueError('GPU changed')
    if (ticket['method']=='M0' and ticket['coefficient']!=0) or (ticket['method']=='M1' and ticket['coefficient'] not in (.1,1.)) or ticket['method'] not in ('M0','M1'):raise ValueError('Invalid method/coefficient')
    c=read(ROOT/'configs/development_v1.json')
    folder=ROOT/'runs/m1_pretrial_v1'/ticket['run_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'ticket.json').exists() and read(folder/'ticket.json')!=ticket:raise ValueError('Run directory belongs to another ticket')
    fd=os.open(folder/'running.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        write(folder/'ticket.json',ticket)
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        seed=ticket['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=UNet(c['widths'],c['group_norm_groups']);initial_digest=state_digest(model.state_dict());model=model.cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'],foreach=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        step=0;history=[];signatures=[]
        if (folder/'last.pt').exists():
            progress=read(folder/'progress.json')
            if digest(folder/'last.pt')!=progress['checkpoint_sha256']:raise ValueError('Checkpoint checksum mismatch or partial synchronization')
            saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            if saved['ticket_id']!=ticket['ticket_id'] or saved['machine']!=machine() or saved['gpu_uuid']!=gpu_uuid or saved['release_id']!=ticket['release_id']:raise ValueError('Resume binding mismatch')
            if saved['step']!=progress['step'] or saved['consumed']!=saved['step']*4 or len(saved['history'])!=saved['step'] or len(saved['signatures'])!=saved['consumed']:raise ValueError('Resume progress/history mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng'])
            step=saved['step'];history=saved['history'];signatures=saved['signatures']
            if saved['initial_digest']!=initial_digest:raise ValueError('Initialization changed')
            del saved
        ds=DevelopmentDataset('train',seed=seed,cache_size=8)
        plan=[];epoch=0
        while len(plan)<ticket['steps']*4:
            plan.extend(list(EpochDrawSampler(ds,epoch=epoch,shuffle=True)));epoch+=1
        deadline=time.perf_counter()+args.max_seconds
        def save():
            atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
                step=step,consumed=step*4,history=history,signatures=signatures,ticket_id=ticket['ticket_id'],release_id=ticket['release_id'],
                machine=machine(),gpu_uuid=gpu_uuid,initial_digest=initial_digest),folder/'last.pt')
            write(folder/'progress.json',dict(step=step,status='completed' if step==ticket['steps'] else 'paused',checkpoint_sha256=digest(folder/'last.pt'),
                note='Training budget completion only; no final evaluation or formal-method claim'))
        while step<ticket['steps']:
            if time.perf_counter()>=deadline or (args.stop_after_step is not None and step>=args.stop_after_step):break
            items=[ds[i] for i in plan[step*4:(step+1)*4]]
            x,y,f,tube=[torch.stack([r[k] for r in items]).cuda() for k in ('image','mask','fov','tube')]
            lr=c['minimum_learning_rate']+.5*(c['learning_rate']-c['minimum_learning_rate'])*(1+math.cos(math.pi*step/ticket['steps']))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss,parts=m1_loss(model(x),y,f,tube,coefficient=ticket['coefficient'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm'],error_if_nonfinite=True)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale:raise ValueError('AMP skipped update')
            step+=1;signatures.extend(fingerprint(i) for i in items)
            history.append(dict(step=step,loss=float(loss.detach()),pixel_loss=float(parts['pixel']),skeleton_loss=float(parts['skeleton']),empty_skeleton_crops=parts['empty_skeleton_crops'],lr=lr,gradient_norm=float(norm)))
            if step%64==0:save();print(f'checkpoint step={step}/{ticket["steps"]}',flush=True)
        save();print(f'Saved step={step}/{ticket["steps"]}; export only after process exit.',flush=True)
    finally:(folder/'running.lock').unlink()

if __name__=='__main__':main()
