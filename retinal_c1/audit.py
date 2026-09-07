"""Freeze selection and thresholds before model scores; audit all 470 training labels."""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
from PIL import Image
from retinal_release.common import ROOT, read, write, digest, identity
from retinal_data.fives import read_manifest, decode_label
from retinal_data.prepare import stable_seed
from retinal_c1.geometry import build, select

OUT=ROOT/'.runtime/c1_local_v1'


def prepare():
    train=read_manifest(ROOT/'releases/development_v1/train.csv')
    val=read_manifest(ROOT/'releases/development_v1/validation.csv')
    small=read_manifest(ROOT/'data_preparation/manifests/smoke_train.csv')
    assert len(train)==470 and len(val)==118 and len(small)==32
    assert {r['key'] for r in small}<={r['key'] for r in train}
    selected=[r for d in ('A','D','G','N') for r in sorted([r for r in val if r['disease']==d],key=lambda r:r['key'])[:6]]
    payload=dict(schema=1,scope='C1_feasibility_then_small_sample_development_not_formal_study',seed=2026,
                 validation_24=[r['key'] for r in selected],training_32=[r['key'] for r in small],
                 selection='validation: first six IDs per disease; training: reuse original 32-image smoke train; no model score selection',
                 geometry=dict(path_points=32,node_end_margin=8,patch_margin=8,max_pairs=8,bg_distance=[2,6],bg_points=[16,128],
                               chain_partition='disjoint consecutive 32-point blocks, deterministic chain endpoint order; tail excluded'),
                 gates=dict(training_images_with_two_pairs_in_four_original_epoch0_draws=.70,
                            macro_full_geometry_thin_skeleton_coverage=.20,
                            macro_validation_path_rank_conflict_fraction=.10,
                            per_path_rank_conflict_threshold=.05),
                 gates_note='Decision assumptions fixed before outcomes; full-geometry coverage and actual crop exposure are reported separately',
                 pilot=dict(methods=['M0','M1','PAIR','SHUFFLE','POOL','SCNP'],steps=256,patch=512,microbatch=4,accumulation=1,
                            pair_lambda=.1,margin=1.,tail=.2,validation_images=24,seed=2026,
                            interpretation='same-device small-sample implementation and directional screening only; no 10000-step or novelty claim'),
                 source_files={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'configs/development_v1.json',ROOT/'releases/development_v1/train.csv',ROOT/'releases/development_v1/validation.csv',ROOT/'releases/m1_pretrial_v1/maps.json']})
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==dict(protocol_id=identity(payload),**payload)
    else:write(path,dict(protocol_id=identity(payload),**payload))
    write(OUT/'selected_rows.json',dict(train=train,validation=selected,small_train=small))
    return train,selected


def one(row, map_records):
    path=OUT/'geometry'/ (Path(row['filename']).stem+'.npz')
    record_path=path.with_suffix('.json')
    if record_path.exists():
        record=read(record_path);assert digest(path)==record['geometry_sha256'];return record
    started=time.perf_counter()
    for role in ('label','fov'):assert digest(ROOT/row[role+'_path'])==row[role+'_sha256']
    label=decode_label((ROOT/row['label_path']).read_bytes()).astype(bool)
    with Image.open(ROOT/row['fov_path']) as im:fov=np.asarray(im)>0
    old=map_records[row['key']];assert digest(ROOT/old['path'])==old['sha256']
    with np.load(ROOT/old['path'],allow_pickle=False) as a:
        skeleton=a['skeleton'];thin=skeleton&(a['skeleton_radius']<=4.)
    geo,stats=build(label,fov,skeleton,thin)
    crop_counts=[];covered=np.zeros(label.shape,bool)
    for draw in range(4):
        # Exactly the parent's stateless crop RNG, for this image at epoch zero.
        global_index=row['_train_index']*4+draw if '_train_index' in row else draw
        rng=np.random.default_rng(stable_seed(2026,0,global_index,row['key']))
        top=int(rng.integers(0,label.shape[0]-512+1));left=int(rng.integers(0,label.shape[1]-512+1))
        pairs=select(geo,[top,left,512,512,0,0,0],stable_seed('C1',2026,0,global_index,row['key']))
        crop_counts.append(len(pairs))
        if len(pairs)>=2:
            for pair in pairs:
                q=pair['pos']+[top,left];covered[tuple(q.T)]=True
    record=dict(key=row['key'],disease=row['disease'],split=row['trial_split'],quality=row['quality_class'],
                label_sha256=row['label_sha256'],fov_sha256=row['fov_sha256'],parent_map_sha256=old['sha256'],
                **stats,crop_pairs=crop_counts,usable_in_four_crops=any(n>=2 for n in crop_counts),
                four_crop_thin_exposure=float((covered&thin).sum()/thin.sum()) if thin.any() else None,
                seconds=time.perf_counter()-started)
    with path.open('xb') as f:np.savez_compressed(f,**geo)
    record['geometry_sha256']=digest(path);write(record_path,record)
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'geometry').mkdir(exist_ok=True)
    train,val=prepare();maps={r['key']:r for r in read(ROOT/'releases/m1_pretrial_v1/maps.json')['records']}
    assert read(ROOT/'releases/m1_pretrial_v1/calibration.json')['tau']==4.
    rows=[dict(r,_train_index=i) for i,r in enumerate(train)]+val
    started=time.perf_counter();records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(lambda r:one(r,maps),rows):
            records.append(record)
            if len(records)%20==0:print(f'geometry {len(records)}/{len(rows)} seconds={time.perf_counter()-started:.1f}',flush=True)
    tr=[r for r in records if r['split']=='train']
    usable=float(np.mean([r['usable_in_four_crops'] for r in tr]));coverage=float(np.mean([r['thin_coverage'] for r in tr if r['thin_coverage'] is not None]))
    summary=dict(status='PASS_GEOMETRY' if usable>=.70 and coverage>=.20 else 'HOLD_GEOMETRY',
                 training_images=len(tr),validation_images=len(val),usable_training_image_fraction=usable,
                 macro_full_geometry_thin_coverage=coverage,macro_four_crop_thin_exposure=float(np.mean([r['four_crop_thin_exposure'] for r in tr if r['four_crop_thin_exposure'] is not None])),
                 thresholds=read(OUT/'protocol.json')['gates'],seconds=time.perf_counter()-started,
                 source_test_images_opened=0,records=records)
    write(OUT/'geometry_audit.json',summary)
    fields=[k for k in records[0] if k!='crop_pairs']
    with (OUT/'geometry_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in records)
    print({k:v for k,v in summary.items() if k!='records'},flush=True)


if __name__=='__main__':main()
