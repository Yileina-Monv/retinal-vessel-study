"""Download a pinned public FIVES mirror with bounded resume and transport checks."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import time
import requests

ROOT=Path(__file__).resolve().parent
REF='nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation'
VERSION=6
URL=f'https://www.kaggle.com/api/v1/datasets/download/{REF}?datasetVersionNumber={VERSION}'
DEST=ROOT/'archives'/'fives_kaggle_v6.zip'
PART=DEST.with_suffix('.zip.part')
DEST.parent.mkdir(parents=True,exist_ok=True)
RECORD=ROOT/'download_evidence.json'
record={'source_url':URL,'mirror_ref':REF,'mirror_version':VERSION,'source_type':'PUBLIC_THIRD_PARTY_KAGGLE_MIRROR',
        'official_doi':'https://doi.org/10.6084/m9.figshare.19688169',
        'official_api_observation':{'url':'https://api.figshare.com/v2/articles/19688169','status':403,'server':'awselb/2.0'},
        'local_path':str(DEST.relative_to(ROOT)),'proxy_configuration_changed':False,'attempts':[]}
began=time.monotonic()
if DEST.exists():
    raise SystemExit('Completed archive already exists. Verify instead of downloading again.')
for attempt in range(1,4):
    offset=PART.stat().st_size if PART.exists() else 0
    info={'attempt':attempt,'resume_offset':offset}
    try:
        headers={'Range':f'bytes={offset}-'} if offset else {}
        with requests.get(URL,headers=headers,stream=True,timeout=(15,45)) as response:
            info['http_status']=response.status_code
            response.raise_for_status()
            p=urlsplit(response.url)
            record['final_origin_path']=p.scheme+'://'+p.netloc+p.path
            record['content_type']=response.headers.get('content-type')
            etag=response.headers.get('etag','').strip('"')
            if record.get('etag') and record['etag']!=etag:raise ValueError('Resource ETag changed')
            record['etag']=etag
            length=int(response.headers.get('content-length') or 0)
            if response.status_code==206:
                content_range=response.headers.get('content-range','')
                if not content_range.startswith(f'bytes {offset}-'):raise ValueError('Wrong resume range')
                expected=int(content_range.rsplit('/',1)[1])
            elif response.status_code==200:
                offset=0
                expected=length
            else:raise ValueError('Unexpected download status')
            if not 1024**3<expected<3*1024**3:raise ValueError('Unexpected FIVES archive size')
            record['advertised_bytes']=expected
            total=offset
            next_progress=(total//(64*1024**2)+1)*64*1024**2
            print(f'Attempt {attempt}, start {total/1024**2:.1f} MiB of {expected/1024**2:.1f} MiB',flush=True)
            with PART.open('ab' if offset else 'wb') as out:
                for chunk in response.iter_content(1024**2):
                    if total==0 and not chunk.startswith(b'PK\x03\x04'):raise ValueError('Not a ZIP response')
                    total+=len(chunk)
                    if total>expected:raise ValueError('Response exceeds advertised size')
                    out.write(chunk)
                    if total>=next_progress:
                        print(f'FIVES {total/1024**2:.0f}/{expected/1024**2:.0f} MiB',flush=True)
                        next_progress+=64*1024**2
            if total!=expected:raise ValueError('Incomplete body')
        with PART.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
        with PART.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if len(etag)==32 and all(c in '0123456789abcdef' for c in etag.lower()):
            if md5!=etag.lower():raise ValueError('Transport ETag/MD5 mismatch')
            record['etag_matches_local_md5']=True
        PART.replace(DEST)
        record.update(status='DOWNLOADED_PENDING_FULL_AUDIT',bytes=DEST.stat().st_size,sha256=sha,md5=md5,
                      elapsed_seconds=round(time.monotonic()-began,3))
        info['status']='PASS'
    except Exception as exc:
        info.update(status='FAIL',error_type=type(exc).__name__)
        record['status']='RETRYABLE_DOWNLOAD_FAILURE'
        print(f'Attempt {attempt}: {type(exc).__name__}; partial bytes preserved',flush=True)
    record['attempts'].append(info)
    RECORD.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if record['status']=='DOWNLOADED_PENDING_FULL_AUDIT':break
print(json.dumps(record,ensure_ascii=False,indent=2))
if record['status']!='DOWNLOADED_PENDING_FULL_AUDIT':raise SystemExit(2)
