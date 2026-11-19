"""Immutable reviewed crop origins with original T0 preprocessing/augmentation."""
from pathlib import Path
import csv,hashlib
import numpy as np
import torch
from retinal_m0.maples_core import CorePatches,paired_weighted_crop

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

class StratifiedPatches(CorePatches):
    def __init__(self, config, trajectory_path, expected_sha):
        super().__init__(config,'T0')
        if file_sha(trajectory_path)!=expected_sha:raise ValueError('immutable crop trajectory digest mismatch')
        with Path(trajectory_path).open(encoding='utf-8',newline='') as f:self.trajectory=list(csv.DictReader(f))
        if len(self.trajectory)!=8000:raise ValueError('exactly8000 reviewed crop origins required')
        for draw,row in enumerate(self.trajectory):
            if int(row['draw'])!=draw or row['filename']!=self.names[draw%110] or int(row['step'])!=draw//4+1 or int(row['slot'])!=draw%4:raise ValueError('crop identity/order mismatch')
        self.trajectory_sha256=expected_sha

    def __getitem__(self,draw):
        if type(draw) is not int or not 0<=draw<8000:raise IndexError(draw)
        index=draw%len(self.names);row=self.trajectory[draw]
        raw,mask,fov,weights=self._full_arrays(index)
        top,left,hf,vf,turns=[int(row[k]) for k in ['top','left','hflip','vflip','ccw_rot90_quarters']]
        if not(0<=top<=raw.shape[0]-512 and 0<=left<=raw.shape[1]-512 and hf in (0,1) and vf in (0,1) and turns in range(4)):raise ValueError('invalid frozen crop/augmentation')
        item=paired_weighted_crop(raw,mask,fov,weights,top=top,left=left,size=512,hflip=bool(hf),vflip=bool(vf),quarter_turns=turns)
        image=(item['image'].astype(np.float32)/255.0-self.mean)/self.std
        image*=item['fov'][...,None]
        tensors={'image':torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),
                 'mask':torch.from_numpy(np.ascontiguousarray(item['mask'][None])).float(),
                 'fov':torch.from_numpy(np.ascontiguousarray(item['fov'][None])).float(),
                 'weight':torch.from_numpy(np.ascontiguousarray(item['weight'][None])).float(),
                 'geometry':torch.tensor([top,left,hf,vf,turns],dtype=torch.int64)}
        h=hashlib.sha256()
        for key in ('image','mask','fov','geometry'):
            value=tensors[key].contiguous().numpy();h.update(f'{key}:{value.dtype}:{value.shape}'.encode());h.update(value.tobytes())
        tensors.update(filename=self.names[index],draw=draw,input_sha256=h.hexdigest())
        return tensors
