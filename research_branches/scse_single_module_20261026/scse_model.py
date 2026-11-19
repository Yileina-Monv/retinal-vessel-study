"""Single final-decoder paper-sum scSE, identity initialized; original UNet is immutable."""
import torch
from torch import nn
from retinal_m0.model import UNet

class SCSE(nn.Module):
    def __init__(self,channels=32):
        super().__init__()
        # CPU-only isolated RNG. Do not reseed CUDA or disturb the original M0 stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261026)
            self.fc1=nn.Linear(channels,channels//2)
            self.fc2=nn.Linear(channels//2,channels)
            self.spatial=nn.Conv2d(channels,1,1)
            nn.init.zeros_(self.fc2.weight);nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.spatial.weight);nn.init.zeros_(self.spatial.bias)
    def forward(self,x):
        c=torch.sigmoid(self.fc2(torch.relu(self.fc1(x.mean((2,3))))))[:,:,None,None]
        s=torch.sigmoid(self.spatial(x))
        # Algebraic paper sum (c+s)*x; centered form preserves initial x exactly.
        return x+(c-.5)*x+(s-.5)*x

class SCSEUNet(UNet):
    def __init__(self,widths=(32,64,128,256,512),groups=8):
        super().__init__(widths,groups);self.scse=SCSE(widths[0]);self.bypass=False
    def forward(self,image):
        factor=2**(len(self.encoders)-1)
        if image.ndim!=4 or image.shape[1]!=3 or any(s%factor for s in image.shape[-2:]):raise ValueError('Expected NCHW RGB divisible by encoder stride')
        skips=[]
        for index,encoder in enumerate(self.encoders):
            image=encoder(image)
            if index<len(self.encoders)-1:
                skips.append(image);image=nn.functional.max_pool2d(image,2)
        for up,decoder,skip in zip(self.ups,self.decoders,reversed(skips)):
            image=decoder(torch.cat((up(image),skip),dim=1))
        return self.head(image if self.bypass else self.scse(image))

def load_m0(model,state):
    original=set(model.state_dict())-{k for k in model.state_dict() if k.startswith('scse.')}
    if set(state)!=original:raise RuntimeError('M0 original keys mismatch')
    result=model.load_state_dict(state,strict=False)
    if result.unexpected_keys or set(result.missing_keys)!={k for k in model.state_dict() if k.startswith('scse.')}:raise RuntimeError('unexpected missing or extra M0 keys')
    if any(not torch.equal(state[k].cpu(),model.state_dict()[k].cpu()) for k in state):raise RuntimeError('M0 shared parameters changed')
