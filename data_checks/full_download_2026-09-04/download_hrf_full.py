"""Retrieve the remaining six official HRF segmentation archives; reuse the first three."""
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import hashlib
import json
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
ARCHIVES = ROOT / 'archives' / 'hrf'
ARCHIVES.mkdir(parents=True, exist_ok=True)
PREVIOUS = ROOT.parent / 'minimal_download_2026-09-04' / 'archives'
BASE = 'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/'
records = []

def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            p = PurePosixPath(member.filename.replace('\\', '/'))
            if p.is_absolute() or '..' in p.parts or ':' in member.filename:
                raise ValueError('Unsafe ZIP path')
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('ZIP symlink')
        if sum(m.file_size for m in members) > 1024**3:
            raise ValueError('Unexpected expanded size')
        bad = archive.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: ' + bad)
        return [{'name':m.filename,'bytes':m.file_size} for m in members if not m.is_dir()]

for cohort in ['healthy', 'glaucoma', 'diabetic_retinopathy']:
    for role, suffix in [('images',''), ('labels','_manualsegm'), ('fov','_fovmask')]:
        name = cohort + suffix + '.zip'
        path = PREVIOUS / name if cohort == 'healthy' else ARCHIVES / name
        r = {'cohort':cohort,'role':role,'source_url':BASE+name,
             'local_path':str(path.relative_to(ROOT.parent)),
             'reused_previous_download':cohort=='healthy'}
        started = time.monotonic()
        try:
            if not path.exists():
                with requests.get(BASE+name, stream=True, timeout=(15,45)) as response:
                    r['http_status'] = response.status_code
                    response.raise_for_status()
                    length = int(response.headers.get('content-length') or 0)
                    r['advertised_bytes'] = length
                    if length > 200*1024**2:
                        raise ValueError('Unexpected download size')
                    total = 0
                    partial = path.with_suffix('.zip.part')
                    with partial.open('wb') as output:
                        for chunk in response.iter_content(1024**2):
                            total += len(chunk)
                            if total > 200*1024**2:
                                raise ValueError('Unexpected download size')
                            output.write(chunk)
                    if length and total != length:
                        raise ValueError('Incomplete response')
                inspect_zip(partial)
                partial.replace(path)
            r['bytes'] = path.stat().st_size
            r['sha256'] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
            r['members'] = inspect_zip(path)
            r['zip_crc_ok'] = True
            r['status'] = 'PASS'
            print(name, 'PASS', r['bytes'], 'bytes', flush=True)
        except Exception as exc:
            r['status'] = 'FAIL'
            r['error_type'] = type(exc).__name__
            print(name, 'FAIL', type(exc).__name__, flush=True)
        r['elapsed_seconds'] = round(time.monotonic()-started,3)
        records.append(r)
        (ROOT/'hrf_download_evidence.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if any(r['status']!='PASS' for r in records):
    raise SystemExit(2)
