"""Compare only genuinely executed, artifact-bound device runs."""
import argparse
import json
from pathlib import Path
import torch
from retinal_data.prepare import ROOT
from .run import differences, dump, read
from .summarize import exact_signflip_p, holm, mean_interval


def compare(first, second, output):
    for key in ('config.json', 'binding.json'):
        if read(first/key) != read(second/key):
            raise ValueError(f'Cross-GPU {key} differs; do not combine')
    a_hw, b_hw = read(first/'hardware.json'), read(second/'hardware.json')
    if a_hw['gpu'] == b_hw['gpu']:
        raise ValueError('Expected two distinct GPU models')
    runtime_keys = ('python','torch','cuda','cudnn','numpy','pillow','scipy')
    mismatch = {k:[a_hw[k],b_hw[k]] for k in runtime_keys if a_hw[k] != b_hw[k]}
    if mismatch:
        raise ValueError(f'Runtime mismatch prevents hardware-only interpretation: {mismatch}')
    config = read(first/'config.json')
    numeric = {}
    for path in (first/'numerical').glob('*.pt'):
        target = second/'numerical'/path.name
        if not target.exists():
            raise ValueError(f'Missing second-device numerical artifact {path.name}')
        a, b = torch.load(path, weights_only=True), torch.load(target, weights_only=True)
        if path.name == 'fixed_inputs.pt':
            if not torch.equal(a['inputs'], b['inputs']) or not torch.equal(a['targets'], b['targets']):
                raise ValueError('Cross-device inputs are not bitwise equal')
        else:
            numeric[path.stem] = {k:differences(b[k],a[k]) for k in a}
    contrasts = []
    for arm in config['arms']:
        deltas = []
        for seed in config['seeds']:
            relative = Path(f'seed_{seed}')/arm['id']/'result.json'
            if not (first/relative).exists() or not (second/relative).exists():
                raise ValueError('Both full GPU matrices must be completed before final comparison')
            a, b = read(first/relative), read(second/relative)
            for key in ('sample_sequence_digest','initial_digest','draws','updates'):
                if a[key] != b[key]:
                    raise ValueError(f'Cross-GPU {key} differs')
            deltas.append(b['final']['macro_dice']-a['final']['macro_dice'])
        contrasts.append({'arm':arm['id'],'seed_differences_second_minus_first':deltas,
                          'mean_difference':sum(deltas)/len(deltas),'exact_p':exact_signflip_p(deltas),
                          'simultaneous_ci_t':mean_interval(deltas,.05/len(config['arms']))})
    for row, p in zip(contrasts,holm([r['exact_p'] for r in contrasts])):
        row['holm_p'] = p
    dump(output, {'status':'two_real_gpu_matrices_compared','first':a_hw,'second':b_hw,'numeric':numeric,'contrasts':contrasts,
                  'scope':'Fixed M0 development data and budget only; six GPU comparisons form a separate family'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('first',type=Path)
    parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'batch_study/cross_gpu_comparison.json')
    args=parser.parse_args()
    compare(args.first,args.second,args.output)
