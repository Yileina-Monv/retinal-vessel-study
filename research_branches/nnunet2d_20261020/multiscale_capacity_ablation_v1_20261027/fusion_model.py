"""Branch-local final-decoder variants; official backbone/loss are unchanged."""
import hashlib
import random
import numpy as np
import torch
from torch import nn

SEED = 2027
MODULE_SEED = 92027
MODES = ('local', 'multi')
DETAIL_PREFIX = 'decoder.stages.7.detail.'


class DetailFusion(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('only frozen local/multi variants')
        self.mode = mode
        def block(cin, cout, kernel, dilation=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel, padding=(kernel // 2) * dilation,
                          dilation=dilation, bias=False),
                nn.InstanceNorm2d(cout, eps=1e-5, affine=True, track_running_stats=False),
                nn.LeakyReLU(.01, inplace=False))
        self.reduce = block(32, 8, 1)
        self.branches = nn.ModuleList([block(8, 8, 1), block(8, 8, 3),
                                      block(8, 8, 3, 1 if mode == 'local' else 2)])
        self.project = nn.Conv2d(24, 32, 1, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=.01)
        nn.init.zeros_(self.project.weight)
        if sum(p.numel() for p in self.parameters()) != 2304:
            raise RuntimeError('frozen parameter count changed')

    def forward(self, x):
        z = self.reduce(x)
        return x + self.project(torch.cat([branch(z) for branch in self.branches], dim=1))


class FinalDecoderStage(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.base = original
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(MODULE_SEED)
            self.detail = DetailFusion(mode)

    def forward(self, x):
        return self.detail(self.base(x))


def original_state(network):
    result = {}
    for key, value in network.state_dict().items():
        if key.startswith(DETAIL_PREFIX):
            continue
        key = key.replace('decoder.stages.7.base.', 'decoder.stages.7.')
        result[key] = value
    return result


def legacy_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def attach_variant(network, mode):
    if mode not in MODES or len(network.decoder.stages) != 8:
        raise ValueError('wrong variant/decoder')
    if network.decoder.seg_layers[-1].in_channels != 32:
        raise RuntimeError('expected final32-channel decoder')
    before = legacy_digest(network.state_dict())
    network.decoder.stages[-1] = FinalDecoderStage(network.decoder.stages[-1], mode)
    if legacy_digest(original_state(network)) != before:
        raise RuntimeError('module attachment changed original backbone')
    return network


def trainer_type():
    # Import only AFTER setting the branch-local native output environment.
    # Keep the inherited constructor signature: nnU-Net introspects it against base locals.
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
    class nnUNetTrainer_DetailFusion100epochs(nnUNetTrainer_100epochs):
        def build_network_architecture(self, plans_manager, configuration_manager,
                                       num_input_channels, num_output_channels,
                                       enable_deep_supervision=True):
            network = nnUNetTrainer.build_network_architecture(
                plans_manager, configuration_manager, num_input_channels,
                num_output_channels, enable_deep_supervision)
            return attach_variant(network, self.detail_mode)
    return nnUNetTrainer_DetailFusion100epochs


def seed_cpu():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_trainer(mode, device, output, data_base):
    import copy, json, os
    from pathlib import Path
    data_base = Path(data_base)
    os.environ.update(nnUNet_raw=str(data_base / 'raw'),
                      nnUNet_preprocessed=str(data_base / 'preprocessed'),
                      nnUNet_results=str(output), nnUNet_compile='false', nnUNet_n_proc_DA='1')
    plans = json.loads((data_base / 'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json').read_text())
    data = json.loads((data_base / 'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    plans = copy.deepcopy(plans); plans['continue_training'] = False
    if mode not in MODES: raise ValueError('unknown mode')
    t = trainer_type()(plans, '2d', 'all', data, device=device)
    t.detail_mode = mode
    t.initialize()
    if (t.num_epochs, t.num_iterations_per_epoch, t.num_val_iterations_per_epoch,
        t.batch_size, list(t.configuration_manager.patch_size), t.initial_lr, t.weight_decay) != (
            100, 250, 50, 2, [1024, 1536], .01, 3e-5):
        raise RuntimeError('frozen training recipe drift')
    t.plot_network_architecture = lambda: None
    return t
