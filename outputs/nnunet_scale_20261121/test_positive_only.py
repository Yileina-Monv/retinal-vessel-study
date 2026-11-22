"""Ablate auxiliary ring negatives at unchanged positive weight and fixed sampling."""
import torch
from train_thin_abc import fit,attach,OUT
from eval_thin_abc import evaluate
from compare_addback_thresholds import load
import paired

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    for seed in (2027,2028,2029):
        for fold in range(3):
            attach(cases,fold)
            fit(cases,fold,seed,("D",))
            p=OUT/f"fold{fold}"/str(seed)
            m=load(p/"D/train.json"); c=load(p/"C/train.json")
            assert m["trace"]==c["trace"] and m["parent_sha"]==c["parent_sha"]
            evaluate(cases,fold,seed,("D",),(("C","D"),("B","D"),("A","D")))
