import argparse
from retinal_release.common import ROOT,read,write,digest,identity,safe
from retinal_release.verify import verify as verify_parent
from retinal_m1.data import SkeletonDataset,fingerprint
from retinal_m1.maps import DEST

def verify(root=ROOT,*,samples=True):
    parent=verify_parent(root,check_samples=samples)
    manifest=read(root/DEST/'release.json')
    if identity({k:v for k,v in manifest.items() if k!='addon_id'})!=manifest['addon_id']:
        raise ValueError('Addon identity mismatch')
    if parent['release_id']!=manifest['parent_release_id']:raise ValueError('Wrong parent release')
    for path,expected in manifest['files'].items():
        if digest(safe(root,path))!=expected:raise ValueError('Addon checksum mismatch: '+path)
    if samples:
        golden=read(root/DEST/'golden_samples.json')
        for seed,records in golden.items():
            ds=SkeletonDataset('train',root=root,seed=int(seed))
            for r in records:
                if fingerprint(ds[(r['epoch'],r['draw'])])!=r:raise ValueError('Auxiliary geometry or sample differs')
    return dict(status='passed',parent_release_id=parent['release_id'],addon_id=manifest['addon_id'],
        addon_files=len(manifest['files']),actual_samples_checked=samples,cross_gpu_equivalence='not_established')

def seal():
    folder=ROOT/DEST
    if (folder/'release.json').exists():raise ValueError('Addon already sealed')
    parent=verify_parent()
    accepted=read(folder/'acceptance.json')
    if accepted['status']!='passed' or accepted['optimizer_steps']!=0:raise ValueError('Pretrial acceptance missing')
    paths=list((ROOT/'retinal_m1').glob('*.py'))+[ROOT/'configs/m1_pretrial_v1.json',ROOT/'tests/test_m1_pretrial.py']
    for path in paths:
        if digest(path)!=accepted['implementation_sha256'][path.relative_to(ROOT).as_posix()]:
            raise ValueError('Code changed after acceptance: '+str(path))
    golden={}
    for seed in (2026,2027,2028):
        ds=SkeletonDataset('train',seed=seed)
        golden[str(seed)]=[fingerprint(ds[index]) for index in ((0,0),(0,7),(0,128),(1,0))]
    write(folder/'golden_samples.json',golden)
    paths += [p for p in folder.rglob('*') if p.is_file()]
    payload=dict(parent_release_id=parent['release_id'],scope='pretrial_engineering_implementation',
        formal_training_ready=False,cross_gpu_resume_ready=False,
        files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))})
    manifest=dict(addon_id=identity(payload),**payload);write(folder/'release.json',manifest)
    return verify()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args()
    print(seal() if a.seal else verify())
