"""Run final predeclared pairs serially after the exploration is verified."""
from followup import run


def main():
    run("spatial.py",log="spatial_focus.log")
    for seed in (2027,2028,2029):
        for fold in range(3):
            print("Correction pair",fold,seed,flush=True)
            run("correction.py","--fold",fold,"--seed",seed,log=f"correction_{fold}_{seed}.log")
    run("compare.py","--source","eval_teacher","--dest","results_teacher",log="correction_comparison.log")
    run("audit.py","--teacher",log="audit_teacher.log")
    run("replay.py",log="inference_replay.log")


if __name__=="__main__": main()
