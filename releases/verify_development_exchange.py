"""Actual local acceptance of final release; never claim a second GPU was tested."""
import subprocess
import sys
import torch
from retinal_release.common import ROOT,read,write,machine,digest
from retinal_release.exchange import issue,pack,ingest
from retinal_m0.state import state_digest

def command(*args,expected_failure=None):
    r=subprocess.run([sys.executable,'-m','retinal_release.run',*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    if expected_failure:
        assert r.returncode!=0 and expected_failure in r.stderr,(r.returncode,r.stderr)
        print('Expected rejection: '+expected_failure,flush=True)
    else:
        if r.returncode:raise RuntimeError(r.stderr)
        print(r.stdout.strip(),flush=True)

def main():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    names=['acceptance_resume_r2','acceptance_reference_r2']
    for name in names:issue(name,machine(),2026,16,gpu)
    ticket=str(ROOT/f'.runtime/coordination/tickets/{names[0]}.json')
    command(ticket,'--stop-after-step','8','--max-seconds','300')
    folder=ROOT/f'runs/development_v1/{names[0]}'
    pack(folder,ROOT/'.runtime/transfer/r2_step8.zip');ingest(ROOT/'.runtime/transfer/r2_step8.zip')
    checkpoint=folder/'last.pt';original=checkpoint.read_bytes()
    try:
        checkpoint.write_bytes(original+b'corruption_test')
        command(ticket,'--max-seconds','300',expected_failure='Checkpoint checksum mismatch')
    finally:checkpoint.write_bytes(original)
    command(ticket,'--max-seconds','300')
    pack(folder,ROOT/'.runtime/transfer/r2_step16.zip');ingest(ROOT/'.runtime/transfer/r2_step16.zip')
    try:ingest(ROOT/'.runtime/transfer/r2_step8.zip');raise AssertionError('Stale accepted')
    except ValueError as e:assert 'Stale' in str(e)
    command(str(ROOT/f'.runtime/coordination/tickets/{names[1]}.json'),'--max-seconds','300')
    a=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    b=torch.load(ROOT/f'runs/development_v1/{names[1]}/last.pt',map_location='cpu',weights_only=True)
    keys=['model','optimizer','scaler','rng','history','signatures','initial_digest','step','consumed']
    checks={k:state_digest(a[k])==state_digest(b[k]) for k in keys}
    assert all(checks.values()),checks
    ref=ROOT/f'runs/development_v1/{names[1]}'
    pack(ref,ROOT/'.runtime/transfer/r2_reference16.zip');ingest(ROOT/'.runtime/transfer/r2_reference16.zip')
    result=dict(release_id=read(ROOT/'releases/development_v1/release.json')['release_id'],status='passed',
        same_gpu_resume_vs_uninterrupted=checks,checksum_corruption_refused_before_resume=True,stale_step8_refused_after_step16=True,
        final_checkpoint_sha256=digest(checkpoint),real_gpu='RTX 5080 only',training_scope='16 updates x 2 executions, one paused at 8; engineering only')
    write(ROOT/'releases/development_v1_acceptance.json',result);print(result,flush=True)

if __name__=='__main__':main()
