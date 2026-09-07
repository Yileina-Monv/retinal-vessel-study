"""Audit every original-resolution FIVES image/label pair, preserve all original bytes."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO
from collections import Counter,defaultdict
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
ZIP=ROOT/'archives/fives_kaggle_v6.zip'
DEST=ROOT/'datasets/FIVES'
QUALITY=json.loads((ROOT/'quality_evidence.json').read_text(encoding='utf-8'))
quality={r['key']:r for r in QUALITY['records']}
rows=[]
file_records=[]

def save(data,member):
    target=DEST/PurePosixPath(member)
    target.parent.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(data).hexdigest()
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==sha,'Existing data differs'
    else:target.write_bytes(data)
    file_records.append({'path':str(target.relative_to(ROOT)),'bytes':len(data),'sha256':sha})
    return target,sha

with ZipFile(ZIP) as z:
    members=z.infolist()
    names=[m.filename for m in members if not m.is_dir()]
    assert len({n.casefold() for n in names})==len(names),'Case-colliding filenames'
    assert sum(m.file_size for m in members)<4*1024**3
    for m in members:
        p=PurePosixPath(m.filename.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts or ':' in m.filename or ((m.external_attr>>16)&0o170000)==0o120000:
            raise ValueError('Unsafe archive member')
    ancillary=[n for n in names if not n.lower().endswith('.png')]
    assert set(ancillary)=={'Quality Assessment.xlsx','train/Original/Thumbs.db'},ancillary
    ignored_bytes=z.read('train/Original/Thumbs.db')
    ignored_files=[{'name':'train/Original/Thumbs.db','bytes':len(ignored_bytes),'sha256':hashlib.sha256(ignored_bytes).hexdigest(),
                    'handling':'CRC checked; retained in original ZIP only; Windows thumbnail cache excluded from image pairing.'}]
    for split,count in [('train',600),('test',200)]:
        images={Path(n).name:n for n in names if n.startswith(split+'/Original/') and n.lower().endswith('.png')}
        labels={Path(n).name:n for n in names if n.startswith(split+'/Ground truth/') and n.lower().endswith('.png')}
        assert len(images)==len(labels)==count
        assert set(images)==set(labels)
        expected={r['filename'] for r in quality.values() if r['split']==split}
        assert set(images)==expected
        for i,name in enumerate(sorted(images),1):
            key=split+'/'+name
            row=dict(quality[key])
            row['image_alpha_values']=None
            row['label_alpha_values']=None
            for role,member in [('image',images[name]),('label',labels[name])]:
                data=z.read(member)  # zipfile validates this member's CRC while fully reading it.
                with Image.open(BytesIO(data)) as im:
                    im.load();arr=np.asarray(im);mode=im.mode;size=im.size
                assert size==(2048,2048),f'Unexpected {role} dimensions for {key}'
                if role=='image':
                    assert mode in {'RGB','RGBA'}
                    if mode=='RGBA':
                        row['image_alpha_values']=np.unique(arr[:,:,3]).tolist()
                        assert row['image_alpha_values']==[255],'Non-opaque image alpha'
                    row['pixel_sha256']=hashlib.sha256(arr[:,:,:3].tobytes()).hexdigest()
                else:
                    if arr.ndim==3:
                        assert arr.shape[2] in {3,4} and np.all(arr[:,:,:3]==arr[:,:,:1]),f'Label channel mismatch: {key}'
                        if arr.shape[2]==4:
                            row['label_alpha_values']=np.unique(arr[:,:,3]).tolist()
                            assert row['label_alpha_values']==[255],'Non-opaque label alpha'
                        arr=arr[:,:,0]
                    assert arr.ndim==2 and arr.dtype==np.uint8
                    row['label_nonbinary_pixels']=int(np.count_nonzero((arr!=0)&(arr!=255)))
                    row['label_foreground_pixels_gt_0']=int(np.count_nonzero(arr))
                    row['label_unique_values']=np.unique(arr).tolist()
                    row['label_is_empty']=row['label_foreground_pixels_gt_0']==0
                    row['label_is_full']=row['label_foreground_pixels_gt_0']==arr.size
                target,sha=save(data,member)
                row[role+'_path']=str(target.relative_to(ROOT))
                row[role+'_sha256']=sha
                row[role+'_mode']=mode
                row[role+'_size_wh']=size
            rows.append(row)
            if i%100==0:print(f'FIVES {split}: {i}/{count} paired images fully decoded and extracted',flush=True)
    workbook=z.read('Quality Assessment.xlsx')
    assert workbook==(ROOT/'archives/quality_primary.xlsx').read_bytes(),'Archive workbook differs from pinned individual download'
    save(workbook,'Quality Assessment.xlsx')
    assert len(file_records)==1601 and len(names)==1602
    central=[{'name':m.filename,'bytes':m.file_size,'crc32':f'{m.CRC:08x}'} for m in members if not m.is_dir()]

by_hash=defaultdict(list)
by_pixels=defaultdict(list)
for row in rows:
    by_hash[row['image_sha256']].append(row['key'])
    by_pixels[row['pixel_sha256']].append(row['key'])
byte_duplicates=[ids for ids in by_hash.values() if len(ids)>1]
pixel_duplicates=[ids for ids in by_pixels.values() if len(ids)>1]
cross_split_pixel_duplicates=[ids for ids in pixel_duplicates if len({i.split('/')[0] for i in ids})>1]
lookup={r['key']:r for r in rows}
duplicate_label_checks=[]
for ids in pixel_duplicates:
    group=[lookup[key] for key in ids]
    masks=[]
    for row in group:
        with Image.open(ROOT/row['label_path']) as im:
            a=np.array(im);masks.append(a[:,:,0] if a.ndim==3 else a)
    duplicate_label_checks.append({'image_keys':ids,'label_bytes_identical':len({r['label_sha256'] for r in group})==1,
                                   'label_pixels_identical':all(np.array_equal(masks[0],m) for m in masks[1:]),
                                   'label_disagreement_pixels':[int(np.count_nonzero(masks[0]!=m)) for m in masks[1:]],
                                   'quality_records_identical':len({tuple(r[k] for k in ['IC','Blur','LC']) for r in group})==1})
train_names={r['filename'] for r in rows if r['split']=='train'}
test_names={r['filename'] for r in rows if r['split']=='test'}
external={}
for path,field in [(ROOT.parent/'full_download_2026-09-04/hrf_format_evidence.json','images_sha256'),(ROOT.parent/'full_download_2026-09-04/fundus_format_evidence.json','images_sha256')]:
    if path.exists():
        report=json.loads(path.read_text(encoding='utf-8'))
        external[path.name]=[{'external_id':r['id'],'fives_keys':by_hash[r[field]]} for r in report['rows'] if r[field] in by_hash]
result={'status':'PASS_FULL_MIRROR_FILE_AUDIT','paired_count':800,'decoded_png_files':1600,'extracted_files':1601,
        'archive_files':len(names),'ignored_non_dataset_files':ignored_files,
        'all_zip_members_read_with_crc_validation':True,'dimensions_wh':[2048,2048],
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'disease_counts':dict(Counter(r['disease'] for r in rows)),
        'split_disease_counts':dict(Counter(r['split']+'/'+r['disease'] for r in rows)),
        'mask_unique_values':sorted({v for r in rows for v in r['label_unique_values']}),
        'nonbinary_label_images':[r['key'] for r in rows if r['label_nonbinary_pixels']>0],
        'empty_label_images':[r['key'] for r in rows if r['label_is_empty']],
        'full_foreground_label_images':[r['key'] for r in rows if r['label_is_full']],
        'image_modes':dict(Counter(r['image_mode'] for r in rows)),'label_modes':dict(Counter(r['label_mode'] for r in rows)),
        'alpha_handling':'RGBA accepted only after checking alpha is uniformly 255; original file bytes preserved. RGB used only for in-memory audit.',
        'quality_rows_matched':len(rows),'quality_workbook_matches_separate_download':True,
        'shared_basenames_across_source_splits':sorted(train_names&test_names),
        'id_rule':'Use split/filename as key; basename alone is not a globally unique identifier.',
        'exact_byte_duplicate_image_groups':byte_duplicates,'exact_pixel_duplicate_image_groups':pixel_duplicates,
        'exact_pixel_duplicate_groups_crossing_train_test':cross_split_pixel_duplicates,
        'duplicate_group_label_checks':duplicate_label_checks,
        'unique_image_pixel_patterns':len(rows)-sum(len(group)-1 for group in pixel_duplicates),
        'exact_byte_matches_with_downloaded_external_images':external,
        'provided_fov_files':[],'patient_or_laterality_mapping_files':[],
        'source_type':'Third-party public Kaggle mirror version 6; official Figshare archive byte identity NOT verified.',
        'scope_limit':'File/encoding/metadata audit only. No model evaluation, patient independence proof, or clinical annotation-correctness claim.',
        'rows':rows,'files':file_records}
if result['empty_label_images'] or result['full_foreground_label_images'] or result['nonbinary_label_images']:
    result['status']='PASS_FILE_INTEGRITY_WITH_LABEL_CONTENT_WARNINGS'
(ROOT/'format_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'archive_members.json').write_text(json.dumps(central,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'fives_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
p=ROOT/'download_evidence.json';d=json.loads(p.read_text(encoding='utf-8'))
d.update(status='PASS_DOWNLOAD_AND_FULL_MIRROR_FILE_AUDIT',file_audit_report='format_evidence.json',official_archive_identity='NOT_VERIFIED')
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'rows','files','shared_basenames_across_source_splits'}},ensure_ascii=False,indent=2))
print('shared basename count:',len(train_names&test_names))
