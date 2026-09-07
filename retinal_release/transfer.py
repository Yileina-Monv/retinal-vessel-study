"""Copy sealed release bytes, never an environment or mutable authority registry."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
import hashlib
from retinal_release.common import ROOT,read,digest,safe,identity

def build(output):
    output=Path(output)
    if output.exists():raise ValueError('Do not overwrite a release package')
    manifest=read(ROOT/'releases/development_v1/release.json')
    paths=dict(manifest['files'])
    paths['releases/development_v1/release.json']=digest(ROOT/'releases/development_v1/release.json')
    temporary=output.with_name(output.name+'.partial');output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,(name,expected) in enumerate(paths.items()):
            source=safe(ROOT,name)
            if digest(source)!=expected:raise ValueError('Release modified: '+name)
            z.write(source,name)
            if (i+1)%200==0:print(f'Packed {i+1}/{len(paths)} files',flush=True)
    os.replace(temporary,output)
    check(output)
    return dict(path=str(output),bytes=output.stat().st_size,sha256=digest(output),release_id=manifest['release_id'])

def check(bundle,destination=None):
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate member')
        manifest=json.loads(z.read('releases/development_v1/release.json'))
        if identity({k:v for k,v in manifest.items() if k!='release_id'})!=manifest['release_id']:raise ValueError('Invalid release identity')
        if set(names)!=set(manifest['files'])|{'releases/development_v1/release.json'}:raise ValueError('Unexpected archive content')
        for name in names:safe(destination or ROOT,name)
        if destination is not None:
            destination=Path(destination)
            if destination.exists():raise ValueError('Extract to a new directory only')
            destination.mkdir(parents=True)
        for i,name in enumerate(names):
            h=hashlib.sha256();out=None
            try:
                if destination:
                    path=safe(destination,name);path.parent.mkdir(parents=True,exist_ok=True);out=path.open('xb')
                with z.open(name) as src:
                    for chunk in iter(lambda:src.read(4*1024*1024),b''):
                        h.update(chunk)
                        if out:out.write(chunk)
                if name in manifest['files'] and h.hexdigest()!=manifest['files'][name]:raise ValueError('Corrupt archive member: '+name)
            finally:
                if out:out.close()
        return dict(status='all_members_verified',files=len(names),release_id=manifest['release_id'])

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','check','extract']);p.add_argument('bundle');p.add_argument('--destination');args=p.parse_args()
    if args.action=='build':print(build(args.bundle))
    elif args.action=='extract':
        if not args.destination:raise ValueError('New destination required')
        print(check(args.bundle,args.destination))
    else:print(check(args.bundle))

if __name__=='__main__':main()
