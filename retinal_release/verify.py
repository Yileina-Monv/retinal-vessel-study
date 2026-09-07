import argparse
import csv
import importlib.metadata
import platform
import subprocess
import torch
from retinal_release.common import ROOT,read,write,digest,identity,safe,machine
from retinal_release.data import DevelopmentDataset,fingerprint
from retinal_m0.model import UNet
from retinal_m0.state import state_digest

def runtime():
    packages={}
    for name in ('requirements-runtime.lock.txt','requirements-torch.lock.txt'):
        for line in (ROOT/'environment'/name).read_text().splitlines():
            if line and not line.startswith('#'):
                key,version=line.split('==');packages[key]=importlib.metadata.version(key)
                if packages[key]!=version: raise ValueError(f'Package mismatch: {key}, require {version}, found {packages[key]}')
    return dict(python=platform.python_version(),packages=packages,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())

def verify(root=ROOT,*,check_runtime=True,check_samples=True):
    manifest=read(safe(root,'releases/development_v1/release.json'))
    payload={k:v for k,v in manifest.items() if k!='release_id'}
    if identity(payload)!=manifest['release_id']: raise ValueError('Release identity mismatch')
    for path,expected in manifest['files'].items():
        if digest(safe(root,path))!=expected: raise ValueError('Frozen file mismatch: '+path)
    if check_runtime and runtime()!=manifest['runtime']: raise ValueError('Runtime version mismatch')
    if check_samples:
        golden=read(safe(root,'releases/development_v1/golden_samples.json'))
        for seed,items in golden.items():
            ds=DevelopmentDataset('train',root=root,seed=int(seed))
            for row in items:
                if fingerprint(ds[(row['epoch'],row['draw'])])!=row: raise ValueError('Actual input differs')
        c=read(safe(root,'configs/development_v1.json'))
        for seed,expected in read(safe(root,'releases/development_v1/golden_initializations.json')).items():
            torch.manual_seed(int(seed))
            if state_digest(UNet(c['widths'],c['group_norm_groups']).state_dict())!=expected:raise ValueError('Initialization differs')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip().splitlines() if torch.cuda.is_available() else []
    return dict(status='passed',release_id=manifest['release_id'],verified_files=len(manifest['files']),devices=devices,
        runtime_checked=check_runtime,actual_samples_checked=check_samples,machine_id=machine(),
        cross_gpu_equivalence='not_established',latest_results_status='unknown_without_authority_registry')

def seal():
    destination=ROOT/'releases/development_v1';path=destination/'release.json'
    if path.exists(): raise ValueError('Release already sealed')
    if not (destination/'QA_APPROVED.json').exists(): raise ValueError('Missing actual QA')
    golden={}
    for seed in (2026,2027,2028):
        ds=DevelopmentDataset('train',seed=seed)
        golden[str(seed)]=[fingerprint(ds[(epoch,draw)]) for epoch,draw in [(0,0),(0,7),(0,128),(0,len(ds)-1),(1,0),(1,127)]]
    write(destination/'golden_samples.json',golden)
    initial={};c=read(ROOT/'configs/development_v1.json')
    for seed in (2026,2027,2028):
        torch.manual_seed(seed);initial[str(seed)]=state_digest(UNet(c['widths'],c['group_norm_groups']).state_dict())
    write(destination/'golden_initializations.json',initial)
    paths=[]
    for directory in ('retinal_release','retinal_data','retinal_m0'):
        paths.extend((ROOT/directory).glob('*.py'))
    paths.extend((ROOT/'environment').glob('*.lock.txt'))
    paths += [ROOT/'.gitattributes',ROOT/'environment/setup.ps1',ROOT/'environment/verify_environment.py',ROOT/'configs/development_v1.json',ROOT/'configs/data_trial_v1.json']
    paths += [p for p in destination.rglob('*') if p.is_file()]
    with (destination/'all.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f): paths += [safe(ROOT,r['image_path']),safe(ROOT,r['label_path'])]
    paths += [ROOT/'data_preparation/manifests/all_train.csv',ROOT/'data_preparation/selection_summary.json']
    paths += [ROOT/'tests/test_release.py']
    payload=dict(schema=1,scope='development_data_preprocessing_and_exchange',files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))},
        runtime=runtime(),formal_training_ready=False,cross_gpu_resume_ready=False)
    write(path,dict(release_id=identity(payload),**payload))
    print('Sealed '+identity(payload),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');args=p.parse_args()
    if args.seal: seal()
    else:
        result=verify();write(ROOT/'.runtime/release_verification.json',result);print(result)

if __name__=='__main__': main()
