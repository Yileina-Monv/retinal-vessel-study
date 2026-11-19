"""Same native network, loss wrapper and target pyramid; normalized final-only weights."""
import random, hashlib, json, copy, os
from pathlib import Path
import numpy as np
import torch

def legacy_digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def seed_cpu():
    random.seed(2027);np.random.seed(2027);torch.manual_seed(2027)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def make_trainer(mode,device,output,data_base):
    if mode!='final':raise ValueError('only final-only engineering candidate')
    data_base=Path(data_base)
    os.environ.update(nnUNet_raw=str(data_base/'raw'),nnUNet_preprocessed=str(data_base/'preprocessed'),nnUNet_results=str(output),nnUNet_compile='false',nnUNet_n_proc_DA='1')
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    plans=json.loads((data_base/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json').read_text());plans['continue_training']=False
    data=json.loads((data_base/'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    t=nnUNetTrainer_100epochs(plans,'2d','all',data,device=device);t.initialize()
    if list(t.loss.weight_factors)!=[64/127,32/127,16/127,8/127,4/127,2/127,1/127,0]:raise RuntimeError('native scale weights changed')
    t.loss=DeepSupervisionWrapper(t.loss.loss,[1,0,0,0,0,0,0,0])
    if sum(p.numel() for p in t.network.parameters())!=59177872 or not t.network.decoder.deep_supervision:raise RuntimeError('architecture drift')
    if (t.num_epochs,t.num_iterations_per_epoch,t.num_val_iterations_per_epoch,t.batch_size,list(t.configuration_manager.patch_size),t.initial_lr,t.weight_decay)!=(100,250,50,2,[1024,1536],.01,3e-5):raise RuntimeError('recipe drift')
    t.plot_network_architecture=lambda:None
    return t
