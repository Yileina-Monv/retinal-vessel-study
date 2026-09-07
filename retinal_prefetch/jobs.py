import argparse
from pathlib import Path
from retinal_release.common import identity,write
from retinal_release.exchange import authority,STATE,job_id
from retinal_prefetch.release import verify

def issue(run_id,target,seed,steps,gpu_uuid,method,coefficient,workers,*,state=STATE):
    job_id(run_id)
    if steps<1 or not target or not gpu_uuid or workers not in (1,2,4):raise ValueError('Invalid target, budget or bounded workers')
    if (method=='M0' and coefficient!=0) or (method=='M1' and coefficient not in (.1,1.)) or method not in ('M0','M1'):raise ValueError('Invalid method')
    release=verify()
    with authority(state) as data:
        if run_id in data['jobs']:raise ValueError('Never reassign an existing job')
        payload=dict(run_id=run_id,release_id=release['parent_release_id'],addon_id=release['addon_id'],execution_id=release['execution_id'],
            target_machine=target,gpu_uuid=gpu_uuid,seed=seed,steps=steps,method=method,coefficient=coefficient,
            workers=workers,depth=workers*2,kind='M0_M1_development_only',microbatch=4,accumulation=1,authority_revision=data['revision']+1)
        ticket=dict(ticket_id=identity(payload),**payload)
        data['jobs'][run_id]=dict(ticket=ticket,accepted_step=-1,accepted_snapshot=None,status='issued_no_results_received')
        write(Path(state)/'tickets'/f'{run_id}.json',ticket)
    return ticket

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--target-machine',required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,required=True)
    p.add_argument('--method',choices=['M0','M1'],required=True);p.add_argument('--coefficient',type=float,required=True)
    p.add_argument('--workers',type=int,choices=[1,2,4],default=4);a=p.parse_args()
    print(issue(a.run_id,a.target_machine,a.seed,a.steps,a.gpu_uuid,a.method,a.coefficient,a.workers))
