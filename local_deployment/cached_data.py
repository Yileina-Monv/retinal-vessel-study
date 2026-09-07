"""Verified disk-backed decoded arrays; retain the parent's exact crop and tensor math."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from retinal_release.common import ROOT,read,write,digest,identity
from retinal_data.fives import FivesDataset,read_pair,read_manifest
from retinal_m1.data import SkeletonDataset,transform
from retinal_prefetch.loader import OrderedBatches,batch

OUT=ROOT/'.runtime/performance_4070_v1'
CACHE=OUT/'decoded'


def stat(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns]


def prepare_one(row,maps):
    dest=CACHE/Path(row['key']).stem;record_path=dest/'record.json'
    source={role:dict(path=row[role+'_path'],sha256=row[role+'_sha256']) for role in ('image','label','fov')}
    source['map']={k:maps[row['key']][k] for k in ('path','sha256')}
    if record_path.exists():
        old=read(record_path)
        if old['source']!=source:raise ValueError('Decoded cache source changed')
        return old
    dest.mkdir(parents=True,exist_ok=True)
    image,label=read_pair(row,ROOT,True)
    if digest(ROOT/row['fov_path'])!=row['fov_sha256']:raise ValueError('FOV checksum mismatch')
    with Image.open(ROOT/row['fov_path']) as im:fov=np.asarray(im)>0
    mp=ROOT/source['map']['path']
    if digest(mp)!=source['map']['sha256']:raise ValueError('Map checksum mismatch')
    with np.load(mp,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    arrays.update(image=image,mask=label,fov=fov)
    files={}
    for name,value in arrays.items():
        path=dest/(name+'.npy')
        if path.exists():raise ValueError('Incomplete cache directory; preserve and inspect before rebuilding')
        with path.open('xb') as f:np.save(f,value,allow_pickle=False)
        files[name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),shape=list(value.shape),dtype=str(value.dtype))
    record=dict(key=row['key'],source=source,files=files)
    write(record_path,record);return record


def prepare():
    rows=read_manifest(ROOT/'releases/development_v1/train.csv')
    rows+=read(ROOT/'.runtime/c1_local_v1/selected_rows.json')['validation']
    maps={r['key']:r for r in read(ROOT/'releases/m1_pretrial_v1/maps.json')['records']}
    CACHE.mkdir(parents=True,exist_ok=True);start=time.perf_counter();records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(lambda row:prepare_one(row,maps),rows):
            records.append(r)
            if len(records)%20==0:print(f'decoded cache {len(records)}/{len(rows)} seconds={time.perf_counter()-start:.1f}',flush=True)
    payload=dict(schema=1,records=records,normalization_sha256=digest(ROOT/'releases/development_v1/normalization.json'),
                 calibration_sha256=digest(ROOT/'releases/m1_pretrial_v1/calibration.json'))
    write(CACHE/'manifest.json',dict(cache_id=identity(payload),**payload))
    write(OUT/'cache_build.json',dict(seconds=time.perf_counter()-start,images=len(rows),bytes=sum((ROOT/v['path']).stat().st_size for r in records for v in r['files'].values())))


class Snapshot:
    def __init__(self,keys):
        start=time.perf_counter();manifest=read(CACHE/'manifest.json')
        if identity({k:v for k,v in manifest.items() if k!='cache_id'})!=manifest['cache_id']:raise ValueError('Cache manifest mismatch')
        if digest(ROOT/'releases/development_v1/normalization.json')!=manifest['normalization_sha256']:raise ValueError('Normalization changed')
        if digest(ROOT/'releases/m1_pretrial_v1/calibration.json')!=manifest['calibration_sha256']:raise ValueError('Calibration changed')
        records={r['key']:r for r in manifest['records']};self.records={k:records[k] for k in sorted(set(keys))};self.stats={}
        # Verify all bytes once before training; hot reads use the immutable verified snapshot.
        for r in self.records.values():
            for v in r['files'].values():
                path=ROOT/v['path']
                if digest(path)!=v['sha256']:raise ValueError('Decoded array checksum mismatch')
                self.stats[v['path']]=stat(path)
        self.cache_id=manifest['cache_id'];self.verification_seconds=time.perf_counter()-start

    def open(self,key):
        result={}
        for name,v in self.records[key]['files'].items():
            path=ROOT/v['path']
            if stat(path)!=self.stats[v['path']]:raise ValueError('Verified cache file changed during run')
            result[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        return result


class CachedDataset(SkeletonDataset):
    def __init__(self,split,*,snapshot,**kwargs):
        super().__init__(split,**kwargs);self.snapshot=snapshot;self.buffers=OrderedDict()
    def arrays(self,key):
        if key not in self.buffers:self.buffers[key]=self.snapshot.open(key)
        self.buffers.move_to_end(key)
        while len(self.buffers)>2:self.buffers.popitem(last=False)
        return self.buffers[key]
    def _arrays(self,row):
        a=self.arrays(row['key']);source=self.snapshot.records[row['key']]['source']
        for role in ('image','label','fov'):
            if source[role]['sha256']!=row[role+'_sha256']:raise ValueError('Cache / frozen row binding mismatch')
        if source['map']['sha256']!=self.map_records[row['key']]['sha256']:raise ValueError('Cache / map binding mismatch')
        # The parent wraps full masks with torch.from_numpy before casting;
        # supply writable storage there while retaining disk-backed patch reads.
        mask=np.array(a['mask'],copy=True) if self.view=='full' else a['mask']
        return a['image'],mask
    def __getitem__(self,index):
        item=FivesDataset.__getitem__(self,index);a=self.arrays(item['key']);g=item['geometry'].tolist()
        def tensor(value):
            value=transform(value,g) if self.view=='patch' else np.array(value,copy=True)
            return torch.from_numpy(np.ascontiguousarray(value[None])).float()
        fov=tensor(a['fov']);mean=torch.tensor(self.stats['mean'])[:,None,None];std=torch.tensor(self.stats['std'])[:,None,None]
        item['image']=((item['image']-mean)/std)*fov;item['fov']=fov
        for name in ('skeleton','tube','skeleton_radius','tube_radius'):item[name]=tensor(a[name])
        item['thin']=(item['skeleton'].bool()&(item['skeleton_radius']<=self.tau)).float()
        return item


class CachedBatches(OrderedBatches):
    snapshot=None
    def _make(self,indices):
        if self.guard:self.guard.check()
        if not hasattr(self.local,'dataset'):
            self.local.dataset=CachedDataset('train',snapshot=self.snapshot,seed=self.seed,cache_size=0)
        return batch(self.local.dataset,indices)
