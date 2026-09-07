from collections import OrderedDict
import numpy as np
import torch
from retinal_release.common import ROOT,read,safe,digest
from retinal_release.data import DevelopmentDataset, fingerprint as parent_fingerprint
from retinal_m1.maps import DEST

def transform(array, geometry):
    top,left,height,width,hflip,vflip,turns=geometry
    array=array[top:top+height,left:left+width]
    if hflip: array=array[:,::-1]
    if vflip: array=array[::-1]
    if turns: array=np.rot90(array,turns)
    return np.ascontiguousarray(array)

class SkeletonDataset(DevelopmentDataset):
    def __init__(self,split,*,root=ROOT,seed=2026,full=False,cache_size=8):
        super().__init__(split,root=root,seed=seed,full=full,cache_size=cache_size)
        self.map_records={r['key']:r for r in read(root/DEST/'maps.json')['records']}
        self.tau=read(root/DEST/'calibration.json')['tau']
        self._map_cache=OrderedDict()
    def __getstate__(self):
        state=super().__getstate__();state['_map_cache']=OrderedDict();return state
    def __getitem__(self,index):
        item=super().__getitem__(index);record=self.map_records[item['key']]
        row=self.rows[item['draw']//self.repeats]
        if any(record[k]!=row[k] for k in ('label_sha256','fov_sha256')):
            raise ValueError('Auxiliary map/source binding mismatch')
        key=item['key']
        if key not in self._map_cache:
            path=safe(self.project_root,record['path'])
            if digest(path)!=record['sha256']:raise ValueError('Auxiliary map checksum mismatch')
            with np.load(path,allow_pickle=False) as a:
                arrays={k:a[k] for k in a.files}
            arrays['thin']=arrays['skeleton'] & (arrays['skeleton_radius']<=self.tau)
            self._map_cache[key]=arrays
        arrays=self._map_cache[key];self._map_cache.move_to_end(key)
        for name,array in arrays.items():
            result=transform(array,item['geometry'].tolist()) if self.view=='patch' else array
            item[name]=torch.from_numpy(np.ascontiguousarray(result[None])).float()
        while len(self._map_cache)>self.cache_size:self._map_cache.popitem(last=False)
        return item

def fingerprint(item):
    import hashlib
    result=parent_fingerprint(item);h=hashlib.sha256()
    for k in ('skeleton','tube','skeleton_radius','tube_radius','thin'):
        a=item[k].contiguous().numpy();h.update(f'{k}:{a.dtype}:{a.shape}'.encode());h.update(a.tobytes())
    result['auxiliary_sha256']=h.hexdigest()
    return result
