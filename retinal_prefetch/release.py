"""Execution overlay identity; immutable parent data and M1 implementation stay intact."""
from retinal_release.common import ROOT,read,write,digest,identity,safe
from retinal_m1.verify import verify as parent_verify

DEST='releases/prefetch_round1'

def verify(root=ROOT):
    parent=parent_verify(root);manifest=read(root/DEST/'release.json')
    if identity({k:v for k,v in manifest.items() if k!='execution_id'})!=manifest['execution_id']:raise ValueError('Execution identity mismatch')
    if parent['addon_id']!=manifest['addon_id']:raise ValueError('Wrong parent addon')
    for path,expected in manifest['files'].items():
        if digest(safe(root,path))!=expected:raise ValueError('Execution overlay changed: '+path)
    return dict(**parent,execution_id=manifest['execution_id'])

def seal():
    folder=ROOT/DEST
    if (folder/'release.json').exists():raise ValueError('Execution overlay already sealed')
    parent=parent_verify();qa=read(folder/'acceptance.json')
    if qa['status']!='passed':raise ValueError('Missing acceptance')
    paths=list((ROOT/'retinal_prefetch').glob('*.py'))+[ROOT/'tests/test_prefetch.py']
    for path in paths:
        if digest(path)!=qa['implementation_sha256'][path.relative_to(ROOT).as_posix()]:raise ValueError('Code changed after acceptance')
    paths += [p for p in folder.rglob('*') if p.is_file()]
    payload=dict(addon_id=parent['addon_id'],scope='prefetch_execution_only',
        files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(paths))},ram_limit_bytes=12_000_000_000,
        vram_limit_bytes=12_000_000_000,recommended_workers_5080=4,recommended_workers_4070_pending_measurement=2)
    write(folder/'release.json',dict(execution_id=identity(payload),**payload));return verify()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();print(seal() if a.seal else verify())
