"""Protocol primitives for the approved MAPLES weight ablation."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from retinal_m0.maples_core import build_training_regions
ALPHAS=(1.0,1.5,2.0,3.0); GROUPS=("A_V1","A_V0","O_V1","O_V0")
def build_ablation_weights(fov,hemorrhages,uncertain,*,image_width,alpha):
 if float(alpha) not in ALPHAS: raise ValueError("alpha must be 1, 1.5, 2, or 3")
 r=build_training_regions(fov,hemorrhages,uncertain,image_width=image_width); f=np.asarray(fov,bool); a=r["rin"]|r["rnear"]
 w=np.zeros(f.shape,np.float32);w[f]=1.;w[a]=np.float32(alpha)
 if np.any(w[r["omega"]&~a]!=1.) or np.any(w[~f]!=0):raise RuntimeError("Weight invariant failed")
 return {**r,"area":a,"weights":w}
def reweight_base_sample(sample,alpha):
 if float(alpha) not in ALPHAS:raise ValueError("Invalid alpha")
 old=sample["weight"];f=sample["fov"].bool();a=old>1.
 if torch.any(old[~f]!=0) or torch.any(old[f&~a]!=1.):raise RuntimeError("Base TW crop no longer encodes FOV/A")
 w=torch.zeros_like(old,dtype=torch.float32);w[f]=1.;w[a]=float(alpha)
 out=dict(sample);out["weight"]=w;return out
def weighted_loss_with_contributions(logits,target,fov,weights,alpha,epsilon=1e-6):
 if logits.shape!=target.shape or target.shape!=fov.shape or fov.shape!=weights.shape or logits.ndim!=4:raise ValueError("Matching NCHW tensors required")
 if float(alpha) not in ALPHAS:raise ValueError("Invalid alpha")
 z,y,f,w=logits.float(),target.float(),fov.float(),weights.float()
 if not all(torch.isfinite(t).all() for t in (z,y,f,w)):raise ValueError("Nonfinite loss input")
 if not torch.all((y==0)|(y==1)) or not torch.all((f==0)|(f==1)):raise ValueError("Target/FOV must be binary")
 if torch.any(w<0) or torch.any(w[f==0]!=0):raise ValueError("Invalid weight outside FOV")
 area=(w>1.)&(f>0)
 if float(alpha)==1. and torch.any(area):raise ValueError("alpha=1 requires unit weights")
 if float(alpha)!=1. and torch.any(w[area]!=float(alpha)):raise ValueError("A weights differ from alpha")
 axes=(1,2,3); nF=f.sum(axes); nA=area.float().sum(axes); D=(f*w).sum(axes)
 if torch.any(D<=0):raise ValueError("empty_fov: no resampling is permitted")
 bm=F.binary_cross_entropy_with_logits(z,y,reduction="none")
 masks={"A_V1":area.float()*y,"A_V0":area.float()*(1-y),"O_V1":f*(~area).float()*y,"O_V0":f*(~area).float()*(1-y)}
 parts={}
 for name,g in masks.items():
  n=g.sum(axes);S=(g*bm).sum(axes);W=S*(float(alpha) if name.startswith("A_") else 1.)
  parts[name]={"n":n.detach(),"S":S.detach(),"W":W.detach(),"C":(W/D).detach()}
 bce=(bm*f*w).sum(axes)/D;p=z.sigmoid()
 dice=1-(2*(p*y*f).sum(axes)+epsilon)/((p*f).sum(axes)+(y*f).sum(axes)+epsilon)
 return (bce+dice).mean(),bce,dice,{"nF":nF.detach(),"nA":nA.detach(),"D":D.detach(),"groups":parts}
def contribution_records(parts,filenames,seed,alpha,step,global_draws,geometries,input_hashes,retries,bce,dice):
 rows=[]
 for i,name in enumerate(filenames):
  groups={}
  for g in GROUPS:
   p=parts["groups"][g];n=int(p["n"][i].item());s=float(p["S"][i].item())
   groups[g]={"n":n,"S":s,"W":float(p["W"][i].item()),"C":float(p["C"][i].item()),"mean_S":s/n if n else None}
  rows.append({"seed":seed,"alpha":float(alpha),"step":step,"global_draw":int(global_draws[i]),"batch_slot":i,"filename":name,"geometry":geometries[i],"input_sha256":input_hashes[i],"nF":int(parts["nF"][i].item()),"nA":int(parts["nA"][i].item()),"D":float(parts["D"][i].item()),"groups":groups,"BCE_image":float(bce[i].item()),"Dice_image":float(dice[i].item()),"total_image":float((bce[i]+dice[i]).item()),"amp_retries":int(retries)})
 return rows
def align_contribution_log(path,checkpoint_step,next_global_draw):
 path=Path(path)
 if not path.exists():
  if checkpoint_step or next_global_draw:raise RuntimeError("Missing contribution log for checkpoint")
  return {"kept_rows":0,"backup":None}
 lines=path.read_text(encoding="utf-8").splitlines();grouped={}
 for line in lines:
  row=json.loads(line);grouped.setdefault(int(row["step"]),[]).append((row,line))
 keep=[]
 for step in range(1,checkpoint_step+1):
  items=grouped.get(step,[])
  if len(items)!=4:raise RuntimeError(f"Step {step} must have four rows")
  if sorted(int(x[0]["global_draw"]) for x in items)!=list(range((step-1)*4,step*4)):raise RuntimeError(f"Step {step} draw discontinuity")
  keep.extend(line for row,line in sorted(items,key=lambda x:int(x[0]["batch_slot"])))
 if next_global_draw!=checkpoint_step*4:raise RuntimeError("Checkpoint step/global_draw invalid")
 if keep!=lines:
  stamp=time.strftime("%Y%m%dT%H%M%S",time.gmtime());backup=path.with_name(f"{path.name}.tail_after_step_{checkpoint_step:06d}_draw_{next_global_draw:06d}_{stamp}.bak")
  if backup.exists():raise RuntimeError("Refusing to overwrite backup")
  backup.write_text("\n".join(lines[len(keep):])+("\n" if lines[len(keep):] else ""),encoding="utf-8")
  tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text("\n".join(keep)+("\n" if keep else ""),encoding="utf-8");tmp.replace(path)
 else:backup=None
 return {"kept_rows":len(keep),"backup":str(backup) if backup else None}
