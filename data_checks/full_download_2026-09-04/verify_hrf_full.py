"""Decode and audit all 45 HRF segmentation pairs and FOV masks, then extract."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'datasets'/'HRF'
downloads=json.loads((ROOT/'hrf_download_evidence.json').read_text(encoding='utf-8'))
assert len(downloads)==9 and all(r['status']=='PASS' for r in downloads)
rows=[]

def decode(data, mask=False):
    with Image.open(BytesIO(data)) as im:
        im.load()
        arr=np.asarray(im)
        mode=im.mode
        size=im.size
    if mask:
        if arr.ndim==3:
            if arr.shape[2]!=3 or not np.all(arr==arr[:,:,:1]):
                raise ValueError('Mask has differing channels')
            arr=arr[:,:,0]
        if arr.ndim!=2 or arr.dtype!=np.uint8:
            raise ValueError('Unexpected mask dimensions or data type')
    elif mode!='RGB':
        raise ValueError('Unexpected image mode')
    return arr,mode,size

def save_original(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(data).digest():
            raise ValueError('Existing file differs: '+str(path))
    else:
        path.write_bytes(data)

for cohort in ['healthy','glaucoma','diabetic_retinopathy']:
    packs={r['role']:ZipFile(ROOT.parent/r['local_path']) for r in downloads if r['cohort']==cohort}
    try:
        maps={role:{Path(n).stem.removesuffix('_mask') if role=='fov' else Path(n).stem:n
                    for n in z.namelist() if not n.endswith('/')} for role,z in packs.items()}
        assert len(maps['images'])==15
        assert set(maps['images'])==set(maps['labels'])==set(maps['fov'])
        for identifier in sorted(maps['images']):
            row={'id':identifier,'cohort':cohort}
            decoded={}
            for role,z in packs.items():
                member=maps[role][identifier]
                data=z.read(member)
                arr,mode,size=decode(data,mask=role!='images')
                decoded[role]=arr
                dest=DEST/role/Path(member).name
                save_original(dest,data)
                row[role+'_path']=str(dest.relative_to(ROOT))
                row[role+'_sha256']=hashlib.sha256(data).hexdigest()
                row[role+'_mode']=mode
                row[role+'_size_wh']=size
            assert row['images_size_wh']==row['labels_size_wh']==row['fov_size_wh']
            y,f=decoded['labels'],decoded['fov']
            assert set(np.unique(f).tolist()).issubset({0,255})
            row['label_intermediate_value_pixels']=int(np.count_nonzero((y>0)&(y<255)))
            row['label_foreground_pixels']=int(np.count_nonzero(y))
            row['label_pixels_ge_128']=int(np.count_nonzero(y>=128))
            row['label_pixels_eq_255']=int(np.count_nonzero(y==255))
            row['fov_pixels']=int(np.count_nonzero(f))
            row['label_foreground_outside_fov']=int(np.count_nonzero((y>0)&(f==0)))
            assert 0<row['label_foreground_pixels']<row['fov_pixels']
            rows.append(row)
        print(cohort, '15 complete image/label/FOV decodes PASS',flush=True)
    finally:
        for z in packs.values():z.close()

hash_counts=Counter(r['images_sha256'] for r in rows)
duplicates=[[r['id'] for r in rows if r['images_sha256']==h] for h,n in hash_counts.items() if n>1]
report={'status':'PASS_FILE_INTEGRITY_WITH_LABEL_ENCODING_WARNINGS','paired_count':len(rows),
        'cohorts':dict(Counter(r['cohort'] for r in rows)),
        'dimensions':dict(Counter(str(r['images_size_wh']) for r in rows)),
        'decoded_files':len(rows)*3,'exact_byte_duplicate_image_groups':duplicates,
        'outside_fov_pair_count':sum(r['label_foreground_outside_fov']>0 for r in rows),
        'outside_fov_pixel_count':sum(r['label_foreground_outside_fov'] for r in rows),
        'nonbinary_label_ids':[r['id'] for r in rows if r['label_intermediate_value_pixels']>0],
        'intermediate_value_pixels':sum(r['label_intermediate_value_pixels'] for r in rows),
        'label_handling':'Original labels preserved. Counts at >0, >=128, ==255 are audit-only sensitivity information; no modeling threshold selected.',
        'fov_mask_values':[0,255],'fov_handling':'Original RGB FOV preserved. Equal channels checked; first channel used only in audit memory.',
        'scope_limit':'File integrity, pairing, decoding and mask encoding only; no clinical label validation, identity verification or model evaluation.',
        'rows':rows}
(ROOT/'hrf_format_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'hrf_manifest.csv').open('w',encoding='utf-8-sig',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False,indent=2))
