"""Frozen FOV-aware development samples; no implicit regeneration of derived data."""
import hashlib
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from retinal_data.fives import FivesDataset
from retinal_release.common import ROOT,read,digest,safe

class DevelopmentDataset(FivesDataset):
    def __init__(self,split,*,root=ROOT,seed=2026,full=False,cache_size=8):
        if split not in ('train','validation'): raise ValueError('Development splits only')
        self.project_root=root
        self.stats=read(safe(root,'releases/development_v1/normalization.json'))
        super().__init__(safe(root,f'releases/development_v1/{split}.csv'),root=root,
            view='full' if full else 'patch',patch_size=512,repeats=1 if full else 4,
            augment=split=='train' and not full,seed=seed,cache_size=cache_size)
    def __getitem__(self,index):
        item=super().__getitem__(index)
        draw=index[1] if isinstance(index,tuple) else index
        row=self.rows[draw//self.repeats]
        path=safe(self.project_root,row['fov_path'])
        if digest(path)!=row['fov_sha256']: raise ValueError('FOV hash mismatch')
        with Image.open(path) as im: fov=np.array(im)>0
        if self.view=='patch':
            top,left,height,width,hflip,vflip,turns=item['geometry'].tolist()
            fov=fov[top:top+height,left:left+width]
            if hflip: fov=fov[:,::-1]
            if vflip: fov=fov[::-1]
            if turns: fov=np.rot90(fov,turns)
        fov=torch.from_numpy(np.ascontiguousarray(fov[None])).float()
        mean=torch.tensor(self.stats['mean'])[:,None,None]
        std=torch.tensor(self.stats['std'])[:,None,None]
        item['image']=((item['image']-mean)/std)*fov
        item['fov']=fov
        return item

def masked_loss(logits,target,fov,epsilon=1e-6):
    if logits.shape!=target.shape or target.shape!=fov.shape or logits.ndim!=4: raise ValueError('Matching NCHW required')
    logits,target,fov=logits.float(),target.float(),fov.float()
    axes=(1,2,3);count=fov.sum(axes)
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*fov).sum(axes)/count.clamp_min(1)
    probability=logits.sigmoid()
    dice=1-(2*(probability*target*fov).sum(axes)+epsilon)/((probability*fov).sum(axes)+(target*fov).sum(axes)+epsilon)
    return (bce+dice).mean()

def fingerprint(item):
    h=hashlib.sha256()
    for k in ('image','mask','fov','geometry'):
        value=item[k].contiguous().numpy()
        h.update(f'{k}:{value.dtype}:{value.shape}'.encode());h.update(value.tobytes())
    return dict(key=item['key'],epoch=item['epoch'],draw=item['draw'],sha256=h.hexdigest())
