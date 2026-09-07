"""Small, no-overwrite overlay. Requires the unchanged full M1 parent release."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from retinal_release.common import ROOT,read,digest,identity,safe
from retinal_prefetch.release import DEST,verify

def build(output):
    verify();output=Path(output)
    manifest=read(ROOT/DEST/'release.json');files=dict(manifest['files'])
    files[DEST+'/release.json']=digest(ROOT/DEST/'release.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path,expected in files.items():
            if digest(ROOT/path)!=expected:raise ValueError('Changed overlay file')
            z.write(ROOT/path,path)
    return dict(sha256=digest(output),bytes=output.stat().st_size,execution_id=manifest['execution_id'])

def install(bundle,root):
    from retinal_m1.verify import verify as parent_verify
    root=Path(root).resolve();parent=parent_verify(root)
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist();manifest=json.loads(z.read(DEST+'/release.json'))
        if identity({k:v for k,v in manifest.items() if k!='execution_id'})!=manifest['execution_id']:raise ValueError('Invalid overlay identity')
        if parent['addon_id']!=manifest['addon_id']:raise ValueError('Wrong parent addon')
        if len(names)!=len(set(names)) or set(names)!=set(manifest['files'])|{DEST+'/release.json'}:raise ValueError('Invalid archive members')
        for name in names:
            target=safe(root,name)
            if target.exists():raise ValueError('Overlay cannot overwrite: '+name)
            if name in manifest['files'] and hashlib.sha256(z.read(name)).hexdigest()!=manifest['files'][name]:raise ValueError('Corrupt overlay')
        for name in names:
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as handle:handle.write(z.read(name))
    return verify(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','install']);p.add_argument('bundle');p.add_argument('--root');a=p.parse_args()
    if a.action=='install' and not a.root:raise ValueError('Explicit project root required')
    print(build(a.bundle) if a.action=='build' else install(a.bundle,a.root))
