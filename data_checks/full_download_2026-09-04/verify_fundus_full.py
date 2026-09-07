"""Audit all bytes and PNG pairs in a public Kaggle Fundus-AVSeg mirror."""
from pathlib import Path, PurePosixPath
from io import BytesIO
from zipfile import ZipFile
from collections import Counter
import csv
import hashlib
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
ZIP=ROOT/'archives'/'fundus_avseg_kaggle_v1.zip'
DEST=ROOT/'datasets'
META=json.loads((ROOT/'fundus_metadata_evidence.json').read_text(encoding='utf-8'))
meta={r['image name']:r for r in META['records']}
REFERENCE_SHA='6db5ff43c4e9c25aa93093aa295c67b10fa0c089ac650df6665c7a6bbae9539f'
REFERENCE_URL='https://github.com/constantinpape/torch-em/blob/2c01ef9c0f8c274b217f1708e71b0a2732613ac5/torch_em/data/datasets/medical/fundus_avseg.py'
COLORS={0:'background',0xff0000:'artery',0x0000ff:'vein',0x00ff00:'crossing',0xffffff:'uncertain_vessel'}
rows=[]
file_records=[]

with ZIP.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
with ZIP.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
with ZipFile(ZIP) as z:
    members=z.infolist()
    assert len({m.filename.casefold() for m in members})==len(members),'Duplicate/case-colliding ZIP entries'
    assert sum(m.file_size for m in members)<1024**3
    for member in members:
        p=PurePosixPath(member.filename.replace('\\','/'))
        if p.is_absolute() or '..' in p.parts or ':' in member.filename or ((member.external_attr>>16)&0o170000)==0o120000:
            raise ValueError('Unsafe archive member')
        assert p.parts[0]=='Fundus-AVSeg'
        if not member.is_dir():assert p.suffix.lower() in {'.png','.xlsx','.txt'}
    assert z.testzip() is None,'Archive CRC mismatch'
    images={Path(n).name:n for n in z.namelist() if n.startswith('Fundus-AVSeg/images/') and n.endswith('.png')}
    labels={Path(n).name:n for n in z.namelist() if n.startswith('Fundus-AVSeg/annotation/') and n.endswith('.png')}
    assert len(images)==len(labels)==100
    assert set(images)==set(labels)==set(meta)
    splits={s:[v.strip() for v in z.read('Fundus-AVSeg/'+s+'.txt').decode('utf-8-sig').splitlines() if v.strip()] for s in ['training','testing']}
    assert len(splits['training'])==len(set(splits['training']))==80
    assert len(splits['testing'])==len(set(splits['testing']))==20
    assert not set(splits['training'])&set(splits['testing'])
    assert set(splits['training'])|set(splits['testing'])==set(images)
    for i,name in enumerate(sorted(images),1):
        arrays={}
        row={'id':Path(name).stem,'source_split':'training' if name in splits['training'] else 'testing',
             'disease':meta[name]['disease type'],'quality':meta[name]['image quality'],'eye_laterality':meta[name]['eye id']}
        assert {'G':'Glaucoma','D':'DR','A':'AMD','N':'Normal'}[Path(name).stem[-1]]==row['disease']
        for role,mapping in [('images',images),('annotation',labels)]:
            member=mapping[name];data=z.read(member)
            with Image.open(BytesIO(data)) as im:
                im.load()
                assert im.mode=='RGB',f'Unexpected {role} mode: {im.mode}'
                arrays[role]=np.asarray(im)
                row[role+'_size_wh']=im.size
                row[role+'_mode']=im.mode
            row[role+'_path']=str((DEST/member).relative_to(ROOT))
            row[role+'_sha256']=hashlib.sha256(data).hexdigest()
        assert row['images_size_wh']==row['annotation_size_wh']
        arr=arrays['annotation'].astype(np.uint32)
        codes=(arr[:,:,0]<<16)|(arr[:,:,1]<<8)|arr[:,:,2]
        values,counts=np.unique(codes,return_counts=True)
        color_counts={COLORS.get(int(v),f'unknown_rgb_hex_{v:06x}'):int(n) for v,n in zip(values,counts)}
        row['unknown_palette_pixels']=sum(n for v,n in color_counts.items() if v.startswith('unknown'))
        row['palette_counts']=color_counts
        row['vessel_pixels_nonblack']=int(np.count_nonzero(codes))
        assert 0<row['vessel_pixels_nonblack']<codes.size
        rows.append(row)
        if i%20==0:print(f'Fundus-AVSeg {i}/100 full pair decodes verified',flush=True)
    # Extract only after all pairing, metadata, decoding and CRC checks succeed.
    for m in members:
        if m.is_dir():continue
        data=z.read(m.filename)
        target=DEST/PurePosixPath(m.filename)
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256(data).hexdigest()
        if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==h,'Existing file differs'
        else:target.write_bytes(data)
        file_records.append({'path':str(target.relative_to(ROOT)),'bytes':len(data),'zip_crc32':f'{m.CRC:08x}','sha256':h})

hashes=Counter(r['images_sha256'] for r in rows)
duplicates=[[r['id'] for r in rows if r['images_sha256']==h] for h,n in hashes.items() if n>1]
colors=Counter()
for row in rows:colors.update(row['palette_counts'])
report={'status':'PASS_MIRROR_FILE_AUDIT_WITH_PROVENANCE_LIMIT','archive_bytes':ZIP.stat().st_size,
        'archive_sha256':sha,'archive_md5':md5,'zip_crc_ok':True,'extracted_files':len(file_records),
        'paired_count':len(rows),'decoded_png_files':2*len(rows),
        'dimensions':dict(Counter(str(r['images_size_wh']) for r in rows)),
        'disease_counts':dict(Counter(r['disease'] for r in rows)),
        'quality_counts':dict(Counter(r['quality'] for r in rows)),
        'source_split_counts':{k:len(v) for k,v in splits.items()},'source_split_disjoint_and_complete':True,
        'palette_pixel_counts':dict(colors),'unknown_palette_pixels':sum(r['unknown_palette_pixels'] for r in rows),
        'exact_byte_duplicate_image_groups':duplicates,
        'metadata_identity_limit':META['identity_limit'],
        'source':'https://www.kaggle.com/datasets/darksoul007fedsdfds/fundusdataset/versions/1',
        'reference_archive_sha256':REFERENCE_SHA,'reference_checksum_source':REFERENCE_URL,
        'archive_matches_reference':sha==REFERENCE_SHA,
        'provenance_limit':'This is a third-party mirror archive. Its ZIP hash differs from the torch-em reference; different packaging may explain this but unmodified identity with the official archive is NOT established.',
        'scope_limit':'No model evaluation, no clinical annotation correctness claim, and no independently verified patient identity. Source split audited for completeness only, not adopted for project use.',
        'rows':rows,'files':file_records}
(ROOT/'fundus_format_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
download_report=ROOT/'fundus_download_evidence.json'
if download_report.exists():
    download=json.loads(download_report.read_text(encoding='utf-8'))
    assert download['sha256']==sha
    download.update(status='PASS_DOWNLOAD_AND_MIRROR_FILE_AUDIT',file_audit_report='fundus_format_evidence.json',official_archive_identity='NOT_VERIFIED')
    download_report.write_text(json.dumps(download,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'fundus_manifest.csv').open('w',encoding='utf-8-sig',newline='') as output:
    writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k not in {'rows','files'}},ensure_ascii=False,indent=2))
