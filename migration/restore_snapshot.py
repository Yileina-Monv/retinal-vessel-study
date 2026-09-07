"""Restore and verify GitHub snapshot with Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import zipfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe(root,name):
    p=PurePosixPath(name)
    if p.is_absolute() or ':' in name or '\\' in name or any(x in ('.','..') for x in p.parts):raise ValueError('Unsafe snapshot path')
    result=root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):raise ValueError('Path escape')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--root',required=True)
    p.add_argument('--include-old-environment',action='store_true',help='Archive recovery only; old venv is not a portable environment')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    assets=Path(a.assets).resolve();root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    index=json.loads((assets/'PAYLOAD_INDEX.json').read_text(encoding='utf-8'))
    if sha(assets/'FILE_MANIFEST.json')!=index['manifest_sha256']:raise ValueError('Manifest checksum mismatch')
    manifest=json.loads((assets/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    for archive in index['archives']:
        if sha(assets/archive['name'])!=archive['sha256']:raise ValueError('Archive checksum mismatch: '+archive['name'])
    grouped={};skipped=[];mapping=[]
    for row in manifest['files']:
        if row['category']=='environment_archive' and not a.include_old_environment:
            skipped.append(row['path']);continue
        name=row['path']
        if row['category']=='authority_archive':
            name='.runtime/migration_authority_archive/'+manifest['snapshot']+'/'+name.removeprefix('.runtime/coordination/')
        target=safe(root,name);mapping.append(dict(source=row['path'],restored=name,sha256=row['sha256']))
        if target.exists() and sha(target)==row['sha256']:continue
        if a.verify_only:raise ValueError('Missing or different restored file: '+name)
        grouped.setdefault(index['blob_archives'][row['sha256']],[]).append((row,target))
    for filename,items in grouped.items():
        with zipfile.ZipFile(assets/filename) as z:
            for row,target in items:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.migration-partial')
                if temporary.exists():raise ValueError('Existing partial: '+str(temporary))
                with z.open('blobs/'+row['sha256']) as src,temporary.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                if sha(temporary)!=row['sha256']:raise ValueError('Content checksum mismatch: '+row['path'])
                if target.exists():
                    # Git may normalize historical CRLF. Repair only newline-equivalent source text.
                    if target.stat().st_size>20000000 or temporary.stat().st_size>20000000:
                        raise ValueError('Refuse overwrite of changed file: '+str(target))
                    before=target.read_bytes();after=temporary.read_bytes()
                    if b'\x00' in before or before.replace(b'\r\n',b'\n')!=after.replace(b'\r\n',b'\n'):
                        raise ValueError('Refuse overwrite of local edits: '+str(target))
                temporary.replace(target)
        print('Restored '+filename,flush=True)
    for name in manifest['directories']:
        if name.startswith('.venv') and not a.include_old_environment:continue
        if name=='.runtime/coordination' or name.startswith('.runtime/coordination/'):continue
        safe(root,name).mkdir(parents=True,exist_ok=True)
    for item in mapping:
        if sha(safe(root,item['restored']))!=item['sha256']:raise ValueError('Final verification failed')
    # Preserve byte-exact frozen source across future Git operations, without modifying sealed .gitattributes.
    if not a.verify_only:
        probe=subprocess.run(['git','-C',str(root),'rev-parse','--show-toplevel'],capture_output=True,text=True,encoding='utf-8')
        if probe.returncode==0:
            top=Path(probe.stdout.strip());info=subprocess.check_output(['git','-C',str(root),'rev-parse','--absolute-git-dir'],text=True,encoding='utf-8').strip()
            prefix=root.relative_to(top).as_posix();rule=('/'+prefix+'/** -text') if prefix!='.' else '* -text'
            attr=Path(info)/'info/attributes';attr.parent.mkdir(parents=True,exist_ok=True)
            old=attr.read_text(encoding='utf-8') if attr.exists() else ''
            if rule not in old.splitlines():attr.write_text(old+'\n'+rule+'\n',encoding='utf-8')
    result=dict(status='verified',files_verified=len(mapping),environment_files_archived_not_activated=len(skipped),
        authority='restored_as_read_only_history_not_active_registry',snapshot=manifest['snapshot'])
    (root/'MIGRATION_RESTORE_RESULT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
