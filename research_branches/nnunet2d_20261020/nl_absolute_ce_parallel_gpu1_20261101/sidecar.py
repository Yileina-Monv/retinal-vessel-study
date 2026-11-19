"""Attach candidate metadata to native batches without changing native image/target/RNG."""
import copy,random
import numpy as np
import torch
from geometry import process_crop

def install():
 import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as mod
 from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
 from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
 Original=mod.nnUNetDataLoader
 class WithCandidates(Original):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.records=[];self.nl_active=False
   transform=self.transforms
   if transform is None:return
   spatial=[t for t in transform.transforms if isinstance(t,SpatialTransform)]
   if not spatial:return # internal Train monitor unchanged
   mirrors=[t for t in transform.transforms if isinstance(t,MirrorTransform)]
   if len(spatial)!=1 or len(mirrors)!=1:raise RuntimeError('unexpected geometry')
   self.nl_active=True;ctx={};load=self._data.load_case;bbox=self.get_bbox
   def load_case(name):
    data,seg,prev,props=load(name)
    if prev is not None or tuple(data.shape[1:])!=tuple(props['shape_before_cropping']) or props['bbox_used_for_cropping']!=[[0,s] for s in props['shape_before_cropping']]:raise RuntimeError('native geometry source mismatch')
    ctx.clear();ctx.update(image=name,native_shape=list(seg.shape[-2:]));return data,seg,prev,props
   def get_bbox(*aa,**kk):
    lo,hi=bbox(*aa,**kk);ctx.update(bbox_lbs=list(map(int,lo)),bbox_ubs=list(map(int,hi)));return lo,hi
   self._data.load_case=load_case;self.get_bbox=get_bbox
   st,mt=spatial[0],mirrors[0];gp,mp=st.get_parameters,mt.get_parameters
   def spatial_parameters(**d):
    par=gp(**d)
    if par['elastic_offsets'] is not None:raise RuntimeError('unexpected elastic')
    ctx['spatial']=par;ctx['input_shape']=list(d['image'].shape[-2:]);return par
   def mirror_parameters(**d):
    par=mp(**d);ctx['mirror']=list(par['axes']);return par
   st.get_parameters=spatial_parameters;mt.get_parameters=mirror_parameters
   def traced(**data):
    out=transform(**data)
    before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
    c={k:v for k,v in ctx.items() if k!='spatial'};par=ctx['spatial'];c.update(center=list(par['center_location_in_pixels']),affine=None if par['affine'] is None else par['affine'].tolist(),axes=c.pop('mirror'),output_shape=list(out['image'].shape[-2:]))
    self.records.append(process_crop(c,par['grid'],out,None,None,None))
    assert before[0]==random.getstate() and all(np.array_equal(x,y) for x,y in zip(before[1],np.random.get_state())) and torch.equal(before[2],torch.random.get_rng_state()),'sidecar changed native RNG'
    return out
   self.transforms=traced
  def generate_train_batch(self):
   self.records.clear();batch=super().generate_train_batch()
   if self.nl_active:
    if len(self.records)!=2 or [r['image'] for r in self.records]!=list(batch['keys']):raise RuntimeError('sidecar batch alignment')
    batch['nl_candidates']=copy.deepcopy(self.records)
   return batch
 mod.nnUNetDataLoader=WithCandidates
 return Original
