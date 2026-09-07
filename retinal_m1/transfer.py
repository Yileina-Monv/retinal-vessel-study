"""Self-contained parent + addon bundle; never transfers mutable run/authority state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
from retinal_release.common import ROOT,read,digest,identity,safe,canonical
from retinal_m1.maps import DEST

MANIFEST='bundle_manifest.json'

def build(output):
    from retinal_m1.verify import verify
    verified=verify();output=Path(output)
    if output.exists():raise ValueError('Do not overwrite bundle')
    parent=read(ROOT/'releases/development_v1/release.json');addon=read(ROOT/DEST/'release.json')
    paths=dict(parent['files'])
    if paths.keys() & addon['files'].keys():raise ValueError('Addon would overwrite frozen parent')
    paths.update(addon['files'])
    for name in ('releases/development_v1/release.json',DEST+'/release.json'):
        paths[name]=digest(ROOT/name)
    meta=dict(parent_release_id=parent['release_id'],addon_id=addon['addon_id'],files=paths)
    meta['bundle_id']=identity(meta)
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.partial')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for name,expected in paths.items():
            if digest(ROOT/name)!=expected:raise ValueError('File changed during packing')
            z.write(ROOT/name,name)
        z.writestr(MANIFEST,canonical(meta))
    os.replace(temporary,output);check(output)
    return dict(path=output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output),
        sha256=digest(output),bytes=output.stat().st_size,**verified)

def check(bundle,destination=None):
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist();meta=json.loads(z.read(MANIFEST))
        if len(names)!=len(set(names)) or set(names)!=set(meta['files'])|{MANIFEST}:raise ValueError('Unexpected archive members')
        if identity({k:v for k,v in meta.items() if k!='bundle_id'})!=meta['bundle_id']:raise ValueError('Bundle identity mismatch')
        for name in names:safe(destination or ROOT,name)
        if destination is not None:
            destination=Path(destination)
            if destination.exists():raise ValueError('Use a new directory only')
            destination.mkdir(parents=True)
        for name,expected in meta['files'].items():
            h=hashlib.sha256();out=None
            try:
                if destination is not None:
                    target=safe(destination,name);target.parent.mkdir(parents=True,exist_ok=True);out=target.open('xb')
                with z.open(name) as src:
                    for chunk in iter(lambda:src.read(4*1024*1024),b''):
                        h.update(chunk)
                        if out:out.write(chunk)
                if h.hexdigest()!=expected:raise ValueError('Corrupt archive member: '+name)
            finally:
                if out:out.close()
        return dict(status='all_members_verified',files=len(meta['files']),addon_id=meta['addon_id'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','check','extract']);p.add_argument('bundle');p.add_argument('--destination');a=p.parse_args()
    if a.action=='extract' and not a.destination:raise ValueError('New destination required')
    print(build(Path(a.bundle).resolve()) if a.action=='build' else check(a.bundle,a.destination if a.action=='extract' else None))
