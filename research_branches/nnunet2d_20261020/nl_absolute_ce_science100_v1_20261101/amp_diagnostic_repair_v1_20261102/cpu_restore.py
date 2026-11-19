"""Load both real interrupted snapshots into the real CPU architecture, without forward/update."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", nnUNet_compile="false")
from paths import *
from common import DATA_BASE, read, sha, atomic
from fast_observer import digest, rng_pack
from recovery import recovery_manifest, restore_state
import torch, gc, time, traceback

def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def forbidden(*a, **kw): raise RuntimeError("CPU restore forbids forward/update/CUDA/scheduler.step")
    torch.cuda._lazy_init=forbidden
    from nnunet2d_scientific_constructor import build_fresh_trainer
    results=[];start=time.monotonic()
    for seed in [2027,2028]:
        stage=GROUP/"runs"/("seed_"+str(seed))
        path=stage/"state_interrupted.pt"
        before=sha(path);budget=read(stage/"budget.json")
        state=torch.load(path,map_location="cpu",weights_only=False)
        manifest=recovery_manifest(state,budget)
        os.environ.update(nnUNet_raw=str(DATA_BASE/"raw"),nnUNet_preprocessed=str(DATA_BASE/"preprocessed"),
                          nnUNet_results=str(OUT/("cpu_restore_native_"+str(seed))),
                          MPLCONFIGDIR=str(OUT/"matplotlib"))
        t=build_fresh_trainer(read(DATA_BASE/"preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json"),
                              read(DATA_BASE/"preprocessed/Dataset501_MAPLES/dataset.json"),torch.device("cpu"))
        t.initialize()
        assert not any(True for _ in t.network.buffers())
        assert all(isinstance(m,torch.nn.Identity) or getattr(m,"p",0)==0 for m in t.network.modules()
                   if isinstance(m,torch.nn.modules.dropout._DropoutNd))
        t.network.register_forward_pre_hook(forbidden)
        t.optimizer.register_step_pre_hook(forbidden)
        # CPU GradScaler supports the same state schema; this is not CUDA behavior validation.
        t.grad_scaler=torch.amp.GradScaler("cpu")
        restored=restore_state(t,state,include_cuda_rng=False)
        expected_rng={k:v for k,v in state["rng"].items() if k!="cuda0"}
        assert digest(rng_pack())==digest(expected_rng)
        assert digest(restored["partial_train"])==state["digests"]["partial_train"]
        assert t.current_epoch==budget["epoch"]
        assert t.optimizer.param_groups[0]["lr"]==.01*(1-budget["epoch"]/100)**.9
        assert digest(t.grad_scaler.state_dict())==state["digests"]["scaler"]
        assert sha(path)==before and read(stage/"budget.json")==budget
        entry={"seed":seed,"checkpoint":str(path),"checkpoint_sha256":before,**manifest,
               "actual_CPU_architecture_loaded":True,"model_optimizer_scheduler_scaler_logger_digest_match":True,
               "CPU_RNG_restored":True,"CUDA_RNG_restored":False,
               "CUDA_RNG_preserved_sha256":digest(state["rng"]["cuda0"]),
               "partial_epoch_statistics_restored":True,"extra_scheduler_steps":0,
               "model_forwards":0,"optimizer_updates":0}
        results.append(entry);print("CPU_RESTORE_PASS",seed,flush=True)
        del t,state,restored;gc.collect()
    assert not torch.cuda.is_initialized()
    atomic(OUT/"CPU_RESTORE.json",{"status":"PASS","seeds":results,"seconds":time.monotonic()-start,
          "science_ledgers_changed":False,"GPU_calls":0,"remaining_gate":"CPU replay and separately released GPU verification"},True)

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        atomic(OUT/"CPU_RESTORE_FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc()},True)
        raise
