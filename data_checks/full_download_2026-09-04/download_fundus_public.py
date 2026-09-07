"""Read a fresh public Figshare download URL from stdin; never retain its temporary query."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import sys
import time
import requests

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'archives'/'FundusAVSeg.zip'
DEST.parent.mkdir(parents=True,exist_ok=True)
url=sys.stdin.read().strip()
u=urlsplit(url)
if u.scheme!='https' or u.hostname not in {'figshare.com','ndownloader.figshare.com','s3-eu-west-1.amazonaws.com','www.kaggle.com'}:
    raise SystemExit('Unexpected download origin')
if u.hostname=='www.kaggle.com':
    if u.path!='/api/v1/datasets/download/darksoul007fedsdfds/fundusdataset':
        raise SystemExit('Unexpected mirror dataset')
    DEST=ROOT/'archives'/'fundus_avseg_kaggle_v1.zip'
record={'official_record':'https://figshare.com/articles/dataset/Fundus-AVSeg/27938034',
        'official_download':'https://figshare.com/ndownloader/files/54093641',
        'retrieval_origin_path':u.scheme+'://'+u.netloc+u.path,
        'temporary_query_retained':False,'proxy_configuration_changed':False,
        'local_path':str(DEST.relative_to(ROOT))}
try:
    began=time.monotonic()
    with requests.get(url,stream=True,timeout=(15,45)) as response:
        record['http_status']=response.status_code
        record['server']=response.headers.get('server')
        record['content_type']=response.headers.get('content-type')
        p=urlsplit(response.url)
        record['final_origin_path']=p.scheme+'://'+p.netloc+p.path
        if response.status_code!=200:
            record['error_body']=response.text[:1500]
            raise ValueError('Download HTTP status is not 200')
        length=int(response.headers.get('content-length') or 0)
        record['advertised_bytes']=length
        record['etag']=response.headers.get('etag')
        if length>400*1024**2:
            raise ValueError('Unexpected archive size')
        sha=hashlib.sha256();md5=hashlib.md5();total=0;next_progress=16*1024**2
        partial=DEST.with_suffix('.zip.part')
        with partial.open('wb') as out:
            for chunk in response.iter_content(1024**2):
                total+=len(chunk)
                if total>400*1024**2:raise ValueError('Unexpected archive size')
                if total==len(chunk) and not chunk.startswith(b'PK\x03\x04'):
                    raise ValueError('Response is not a ZIP archive')
                out.write(chunk);sha.update(chunk);md5.update(chunk)
                if total>=next_progress:
                    print(f'FundusAVSeg.zip {total/1024**2:.1f} MiB',flush=True)
                    next_progress+=16*1024**2
        if length and total!=length:raise ValueError('Incomplete response')
        if DEST.exists():raise ValueError('Destination already exists; verify instead of overwriting')
        partial.replace(DEST)
        record.update(status='DOWNLOADED_PENDING_ZIP_AUDIT',bytes=total,sha256=sha.hexdigest(),md5=md5.hexdigest(),elapsed_seconds=round(time.monotonic()-began,3))
        if u.hostname=='www.kaggle.com':
            record['retrieval_type']='THIRD_PARTY_KAGGLE_MIRROR_VERSION_1'
except Exception as exc:
    record.update(status='FAIL',error_type=type(exc).__name__)
(ROOT/'fundus_download_evidence.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='error_body'},ensure_ascii=False),flush=True)
if record['status']=='FAIL':raise SystemExit(2)
