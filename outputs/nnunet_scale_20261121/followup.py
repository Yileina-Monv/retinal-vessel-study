"""Finish the full matrix, audit it, then run the predeclared continuation comparisons."""
import subprocess
import sys
from prepare import ROOT


def run(script,*args,log):
    with (ROOT/log).open("ab") as f:
        subprocess.run([sys.executable,"-B","-u",str(ROOT/script),*map(str,args)],
                       stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    for s in (2027,2028,2029):
        for fold in range(3):
            d=ROOT/"eval_paired"/f"fold{fold}"/str(s)/"mix"/"dev.json"
            if not d.is_file():
                run("evaluate.py","--fold",fold,"--seed",s,log=f"replay_{fold}_{s}.log")
    run("compare.py",log="primary_comparison.log")
    run("audit.py",log="audit_primary.log")
    for s in (2027,2028,2029):
        for fold in range(3):
            print("Continuation",fold,s,flush=True)
            run("focus.py","--fold",fold,"--seed",s,log=f"focus_{fold}_{s}.log")
    run("compare.py","--source","eval_focus","--dest","results_focus",log="focus_comparison.log")
    run("audit.py","--focus",log="audit_focus.log")
    run("verify_rgb.py",log="audit_rgb.log")


if __name__=="__main__": main()
