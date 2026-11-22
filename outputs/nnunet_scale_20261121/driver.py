"""Run paired folds serially on one GPU; completed endpoints are reusable."""
import subprocess
import sys
import time
from prepare import ROOT,stage


def main():
    for seed in (2027,2028,2029):
        for fold in range(3):
            ready=ROOT/"preprocessed"/stage(fold)/"ready.json"
            for _ in range(180):
                if ready.is_file(): break
                time.sleep(2)
            else: raise RuntimeError(f"Preparation incomplete for {fold}")
            for task in ("train","evaluate"):
                log=ROOT/f"{task}_{fold}_{seed}_current.log"
                print(task,fold,seed,flush=True)
                with log.open("ab") as f:
                    subprocess.run([sys.executable,"-B","-u",str(ROOT/(task+".py")),"--fold",str(fold),
                                    "--seed",str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)


if __name__=="__main__": main()
