import torch
from torch.nn.functional import pad
from retinal_m0.inference import tiled_probability

def probability(model,image,fov,*,patch=512,stride=256,batch_size=8,device='cuda',amp=True):
    if tuple(fov.shape)!=tuple(image.shape[-2:]):raise ValueError('FOV/image mismatch')
    height,width=image.shape[-2:]
    # Constant zero denotes the already-normalized outside field, never resize.
    bottom=max(0,patch-height);right=max(0,patch-width)
    padded=pad(image,(0,right,0,bottom),value=0)
    result,tiles=tiled_probability(model,padded,patch=patch,stride=stride,batch_size=batch_size,device=device,amp=amp)
    return result[:height,:width]*torch.as_tensor(fov,dtype=torch.float32,device='cpu'),tiles
