"""Prespecified difference-in-differences after both real GPU matrices exist."""
import argparse
from pathlib import Path
from retinal_batchstudy.compare_gpu import compare
from retinal_batchstudy.run import read,dump
from retinal_batchstudy.summarize import exact_signflip_p,holm,mean_interval


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('first',type=Path)
    parser.add_argument('second',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    # Reject missing/unequal runtime, data, initial weights or code before inference.
    compare(args.first,args.second,args.output_dir/'gpu_main_effects.json')
    config=read(args.first/'config.json')
    contrasts=[]
    for arm in config['arms']:
        if arm['id']==config['baseline']: continue
        differences=[]
        for seed in config['seeds']:
            effects=[]
            for root in (args.first,args.second):
                candidate=read(root/f'seed_{seed}'/arm['id']/'result.json')['final']['macro_dice']
                reference=read(root/f'seed_{seed}'/config['baseline']/'result.json')['final']['macro_dice']
                effects.append(candidate-reference)
            differences.append(effects[1]-effects[0])
        contrasts.append({'arm':arm['id'],'seed_interactions_second_minus_first':differences,
                          'mean_interaction':sum(differences)/len(differences),
                          'exact_p':exact_signflip_p(differences),
                          'simultaneous_ci_t':mean_interval(differences,.05/(len(config['arms'])-1))})
    for item,pvalue in zip(contrasts,holm([r['exact_p'] for r in contrasts])):
        item['holm_p']=pvalue
    dump(args.output_dir/'gpu_batch_interactions.json',{'contrasts':contrasts,
         'scope':'Five prespecified paired-seed difference-in-differences; separate Holm family',
         'margin_absolute_dice':config['practical_margin_absolute_dice']})


if __name__=='__main__': main()
