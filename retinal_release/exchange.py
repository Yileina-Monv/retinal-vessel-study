"""Offline exchange with one non-replicated authority. Never merge mutable run folders."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile
import torch
from retinal_release.common import ROOT,read,write,digest,identity,safe,machine

STATE=ROOT/'.runtime/coordination'

@contextmanager
def authority(state=STATE):
    state=Path(state).resolve();state.mkdir(parents=True,exist_ok=True)
    lock=state/'authority.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        path=state/'registry.json'
        if path.exists(): data=read(path)
        else: data=dict(machine=machine(),location=str(state),revision=0,jobs={})
        if data['machine']!=machine() or data['location']!=str(state): raise ValueError('Authority was copied or relocated; refuse split authority')
        yield data
        data['revision']+=1;write(path,data)
    finally:
        lock.unlink()

def job_id(value):
    if not value or not value.replace('-','').replace('_','').isalnum(): raise ValueError('Simple job ID required')
    return value

def issue(run_id,target,seed,steps,gpu_uuid,*,state=STATE):
    job_id(run_id)
    if steps<1 or not target or not gpu_uuid: raise ValueError('Explicit target machine/GPU and positive budget required')
    release=read(ROOT/'releases/development_v1/release.json')['release_id']
    with authority(state) as data:
        if run_id in data['jobs']: raise ValueError('Job ID already exists; never silently reassign')
        payload=dict(run_id=run_id,release_id=release,target_machine=target,gpu_uuid=gpu_uuid,seed=seed,steps=steps,
            kind='M0_development_only',microbatch=4,accumulation=1,authority_revision=data['revision']+1)
        ticket=dict(ticket_id=identity(payload),**payload)
        data['jobs'][run_id]=dict(ticket=ticket,accepted_step=-1,accepted_snapshot=None,status='issued_no_results_received')
        write(Path(state)/'tickets'/f'{run_id}.json',ticket)
    return ticket

def check_ticket(ticket):
    if identity({k:v for k,v in ticket.items() if k!='ticket_id'})!=ticket['ticket_id']: raise ValueError('Ticket modified')
    job_id(ticket['run_id'])

def _pack_locked(run_dir,output):
    run_dir=Path(run_dir);output=Path(output)
    if output.exists(): raise ValueError('Never overwrite a transfer package')
    ticket=read(run_dir/'ticket.json');check_ticket(ticket)
    saved=torch.load(run_dir/'last.pt',map_location='cpu',weights_only=True)
    if saved['ticket_id']!=ticket['ticket_id'] or saved['machine']!=ticket['target_machine']: raise ValueError('Checkpoint owner mismatch')
    for key in ('model','optimizer','scaler','rng','step','consumed','history','signatures','gpu_uuid','release_id'):
        if key not in saved: raise ValueError('Incomplete recovery state: '+key)
    if saved['gpu_uuid']!=ticket['gpu_uuid'] or saved['release_id']!=ticket['release_id']: raise ValueError('GPU/release mismatch')
    files={p.name:digest(p) for p in [run_dir/'last.pt',run_dir/'ticket.json',run_dir/'progress.json']}
    progress=read(run_dir/'progress.json')
    if progress['step']!=saved['step'] or progress['checkpoint_sha256']!=files['last.pt']: raise ValueError('Progress/checkpoint mismatch')
    manifest=dict(ticket=ticket,step=saved['step'],status=progress['status'],files=files)
    manifest['snapshot_id']=identity(manifest)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+'.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_STORED) as z:
        for name in files:z.write(run_dir/name,name)
        from retinal_release.common import canonical
        z.writestr('snapshot.json',canonical(manifest))
    os.replace(temporary,output)
    return manifest

def pack(run_dir,output):
    lock=Path(run_dir)/'running.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:return _pack_locked(run_dir,output)
    finally:lock.unlink()

def ingest(bundle,*,state=STATE):
    with zipfile.ZipFile(bundle) as z:
        if len(z.namelist())!=len(set(z.namelist())): raise ValueError('Duplicate archive names')
        import json
        meta=json.loads(z.read('snapshot.json'));check_ticket(meta['ticket'])
        if identity({k:v for k,v in meta.items() if k!='snapshot_id'})!=meta['snapshot_id']: raise ValueError('Snapshot identity mismatch')
        if set(z.namelist())!=set(meta['files'])|{'snapshot.json'}:raise ValueError('Unlisted archive members')
        run_id=job_id(meta['ticket']['run_id'])
        with authority(state) as data:
            record=data['jobs'].get(run_id)
            if not record or record['ticket']!=meta['ticket']:raise ValueError('Unknown or superseded ticket')
            if meta['step']<=record['accepted_step']:
                if meta['snapshot_id']==record['accepted_snapshot']:return dict(status='already_accepted')
                raise ValueError('Stale or conflicting result; no rollback or overwrite')
            final_destination=Path(state)/'received'/run_id/meta['snapshot_id']
            destination=Path(state)/'quarantine'/uuid.uuid4().hex
            destination.mkdir(parents=True,exist_ok=False)
            for name,expected in meta['files'].items():
                target=safe(destination,name)
                with z.open(name) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
                if digest(target)!=expected:raise ValueError('Transfer checksum mismatch')
            saved=torch.load(destination/'last.pt',map_location='cpu',weights_only=True)
            ticket=meta['ticket']
            if saved['step']!=meta['step'] or not 0<saved['step']<=ticket['steps']:raise ValueError('Invalid progress')
            if saved['consumed']!=saved['step']*4 or len(saved['history'])!=saved['step']:raise ValueError('Inconsistent cursor')
            if len(saved['signatures'])!=saved['consumed']:raise ValueError('Missing sample history')
            if saved['ticket_id']!=ticket['ticket_id'] or saved['release_id']!=ticket['release_id'] or saved['machine']!=ticket['target_machine'] or saved['gpu_uuid']!=ticket['gpu_uuid']:raise ValueError('Checkpoint identity mismatch')
            for key in ('model','optimizer','scaler','rng'):
                if key not in saved:raise ValueError('Incomplete checkpoint')
            if read(destination/'ticket.json')!=ticket:raise ValueError('Ticket mismatch')
            progress=read(destination/'progress.json')
            expected_status='completed' if saved['step']==ticket['steps'] else 'paused'
            if meta['status']!=expected_status or progress['status']!=expected_status or progress['step']!=saved['step'] or progress['checkpoint_sha256']!=meta['files']['last.pt']:raise ValueError('False completion/progress')
            write(destination/'snapshot.json',meta)
            final_destination.parent.mkdir(parents=True,exist_ok=True)
            os.replace(destination,final_destination)
            record.update(accepted_step=meta['step'],accepted_snapshot=meta['snapshot_id'],status=expected_status,last_received_unix=time.time())
    return dict(status='accepted',run_id=run_id,step=meta['step'])

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    issue_p=s.add_parser('issue');issue_p.add_argument('--run-id',required=True);issue_p.add_argument('--target-machine',required=True);issue_p.add_argument('--gpu-uuid',required=True);issue_p.add_argument('--seed',type=int,required=True);issue_p.add_argument('--steps',type=int,required=True)
    pack_p=s.add_parser('pack');pack_p.add_argument('run_dir');pack_p.add_argument('output')
    ingest_p=s.add_parser('ingest');ingest_p.add_argument('bundle')
    s.add_parser('status')
    args=p.parse_args()
    if args.cmd=='issue':print(issue(args.run_id,args.target_machine,args.seed,args.steps,args.gpu_uuid))
    elif args.cmd=='pack':print(pack(args.run_dir,args.output))
    elif args.cmd=='ingest':print(ingest(args.bundle))
    else:
        with authority() as data:
            print({k:{'status':v['status'],'accepted_step':v['accepted_step'],'latest_live_progress':'unknown_until_new_result_received'} for k,v in data['jobs'].items()})

if __name__=='__main__':main()
