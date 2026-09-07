"""Create a byte-exact, deduplicated project snapshot; no credentials or remote operations."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    excluded='.runtime/github_migration_build'
    empty_verified={'.runtime/tmp/tmp5xvga3fw','.runtime/tmpd_4rj9ll','.runtime/tmpoakseg3m'}
    rows=[];dirs=[];blobs={};errors=[];start=time.time()
    def fail(e):
        rel=Path(e.filename).relative_to(root).as_posix()
        if rel not in empty_verified:errors.append({'path':rel,'error':str(e)})
    for folder,sub,files in os.walk(root,followlinks=False,onerror=fail):
        sub.sort();files.sort()
        for name in list(sub):
            path=Path(folder)/name;rel=path.relative_to(root).as_posix()
            if rel==excluded or name=='.git':sub.remove(name);continue
            if path.is_symlink() or path.is_junction():raise ValueError('Unexpected link: '+rel)
            dirs.append(rel)
        for name in files:
            path=Path(folder)/name;rel=path.relative_to(root).as_posix()
            before=path.stat();digest=sha(path);after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Changed during read: '+rel)
            category='environment_archive' if rel.startswith('.venv/') else ('authority_archive' if rel.startswith('.runtime/coordination/') else 'project')
            rows.append(dict(path=rel,bytes=before.st_size,sha256=digest,category=category))
            blobs.setdefault(digest,path)
            if len(rows)%5000==0:print(json.dumps({'phase':'hashing','files':len(rows)}),flush=True)
    if errors:raise ValueError(json.dumps(errors))
    inventory=dict(schema=1,snapshot='retinal-migration-20260906-v1',created_unix=start,
        files=rows,directories=sorted(set(dirs)),verified_empty_restricted_directories=sorted(empty_verified),
        excluded_generated_workspace=excluded,source_bytes=sum(r['bytes'] for r in rows),
        unique_bytes=sum(p.stat().st_size for p in blobs.values()),source_files=len(rows),unique_blobs=len(blobs))
    (out/'FILE_MANIFEST.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in inventory.items() if k not in ['files','directories']}),flush=True)
    archives=[];blob_index={};z=None;used=0
    def close():
        if z is not None:
            z.close();archives.append(dict(name=current.name,bytes=current.stat().st_size,sha256=sha(current)))
            print(json.dumps({'phase':'archive_completed',**archives[-1]}),flush=True)
    for digest,path in sorted(blobs.items()):
        size=path.stat().st_size
        if z is None or used+size>1800000000:
            close();current=out/f'payload-{len(archives)+1:03d}.zip'
            z=zipfile.ZipFile(current,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True);used=0
        z.write(path,'blobs/'+digest);used+=size;blob_index[digest]=current.name
    close()
    for item in archives:
        with zipfile.ZipFile(out/item['name']) as archive:
            if archive.testzip() is not None:raise ValueError('Corrupt archive')
    payload=dict(schema=1,snapshot=inventory['snapshot'],archives=archives,blob_archives=blob_index,
        manifest_sha256=sha(out/'FILE_MANIFEST.json'),seconds=time.time()-start)
    (out/'PAYLOAD_INDEX.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'status':'built_and_crc_verified','archives':len(archives),'upload_bytes':sum(x['bytes'] for x in archives),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
