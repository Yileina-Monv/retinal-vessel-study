"""Pre-seal runner integration fixture; production authority is never used."""
import argparse
import subprocess
import sys
from unittest.mock import patch
from retinal_release.common import ROOT,read,write,identity,machine
from retinal_m1.verify import verify as parent_verify

def main():
    p=argparse.ArgumentParser();p.add_argument('name',choices=['reference','resume']);p.add_argument('--stop',type=int,default=16);a=p.parse_args()
    parent=parent_verify();parent['execution_id']='preseal_integration_fixture_only'
    root=ROOT/'.runtime/prefetch_round1/runner_fixture'
    write(root/'configs/development_v1.json',read(ROOT/'configs/development_v1.json'))
    payload=dict(run_id=a.name,release_id=parent['parent_release_id'],addon_id=parent['addon_id'],execution_id=parent['execution_id'],
        target_machine=machine(),gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
        seed=2026,steps=16,method='M1',coefficient=.1,workers=4,depth=8,kind='M0_M1_development_only',microbatch=4,accumulation=1)
    ticket=dict(ticket_id=identity(payload),**payload);path=root/(a.name+'_ticket.json');write(path,ticket)
    import retinal_prefetch.run as run
    # Only pre-seal identity gate and output root are substituted. The complete
    # frozen source release is verified above; reader, GPU, optimizer and saves are real.
    with patch.object(run,'ROOT',root),patch.object(run,'verify',return_value=parent),patch.object(sys,'argv',
        ['runner',str(path),'--max-seconds','300','--stop-after-step',str(a.stop)]):run.main()

if __name__=='__main__':main()
