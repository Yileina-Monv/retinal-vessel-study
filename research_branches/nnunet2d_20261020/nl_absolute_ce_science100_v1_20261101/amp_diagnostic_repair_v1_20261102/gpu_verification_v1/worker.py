import os, sys, time, ctypes, signal, math, gc, traceback
from support import *

def execute(parent,nonce):
    if os.getppid()!=parent or os.environ.get("NL_REPAIR_NONCE")!=nonce or len(nonce)<32:
        raise RuntimeError("unauthorized diagnostic child")
    key=validate()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU:raise RuntimeError("GPU UUID mismatch")
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError("parent death guard")
    def authorized():
        if os.getppid()!=parent or sha(WORK/"GPU_RELEASE.json")!=key:
            raise RuntimeError("diagnostic session revoked")
        session=read(WORK/"SESSION.json")
        if session["worker_pid"]!=os.getpid() or session["controller_pid"]!=parent or session["nonce"]!=nonce:
            raise RuntimeError("diagnostic session mismatch")
        if time.time()-session["started"]>1800:raise RuntimeError("diagnostic deadline")
    for _ in range(100):
        if (WORK/"SESSION.json").exists():break
        time.sleep(.05)
    authorized()
    import torch, numpy as np
    from fast_observer import digest,rng_pack,states_finite
    from recovery import restore_state
    from diagnostic import bounded_gradients
    from aux_loss import sparse_aux
    from nnunet2d_scientific_constructor import build_fresh_trainer
    from seeded_runtime import set_all_seeds
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError("CUDA initialized before admission")
    if torch.cuda.device_count()!=1:raise RuntimeError("exactly GPU1 visible")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(10240*1048576/torch.cuda.get_device_properties(0).total_memory)
    os.environ.update(nnUNet_raw=str(DATA_BASE/"raw"),nnUNet_preprocessed=str(DATA_BASE/"preprocessed"),
                      nnUNet_results=str(WORK/"native_constructor"),nnUNet_compile="false",
                      MPLCONFIGDIR=str(WORK/"matplotlib"))
    coeff=read(GROUP/"protocol.json")["method"]["lambda"]
    if coeff!=0.0018152656553044105:raise RuntimeError("lambda changed")
    summaries=[]
    for index,case in enumerate(case_list()):
        authorized()
        checkpoint=case["checkpoint"]
        bundle=torch.load(checkpoint,map_location="cpu",weights_only=False)
        batch=torch.load(case["batch"]["path"],map_location="cpu",weights_only=False)
        if sha(case["batch"]["path"])!=case["batch"]["sha256"]:raise RuntimeError("batch artifact changed")
        if digest(batch["data"])!=case["batch"]["input_sha256"] or digest(batch["target"])!=case["batch"]["target_sha256"] or digest(batch["groups"])!=case["batch"]["groups_sha256"]:
            raise RuntimeError("batch content changed")
        train_ids=set(read(BASELINE/"runs"/("seed_"+str(case["seed"]))/"protocol.json")["train_ids"])
        if not set(batch["ids"])<=train_ids:raise RuntimeError("non-Train batch")
        set_all_seeds(case["seed"])
        t=build_fresh_trainer(read(DATA_BASE/"preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json"),
                              read(DATA_BASE/"preprocessed/Dataset501_MAPLES/dataset.json"),torch.device("cuda:0"))
        t.initialize()
        if any(True for _ in t.network.buffers()):raise RuntimeError("mutable model buffers")
        restore_state(t,bundle,include_cuda_rng=True)
        if digest(rng_pack())!=digest(bundle["rng"]):raise RuntimeError("CPU/CUDA RNG restoration mismatch")
        if list(t.loss.weight_factors)!=[64/127,32/127,16/127,8/127,4/127,2/127,1/127,0]:raise RuntimeError("Deep Supervision changed")
        t.network.train()
        optimizer_calls=[0]
        def forbidden_step(*args,**kw):
            optimizer_calls[0]+=1
            raise RuntimeError("optimizer.step prohibited in zero-update diagnostics")
        opt_hook=t.optimizer.register_step_pre_hook(forbidden_step)
        def forward_guard(*args):
            authorized()
            s=mark_forward(read(WORK/"budget.json"),index);atomic(WORK/"budget.json",s)
        fw_hook=t.network.register_forward_pre_hook(forward_guard)
        t.optimizer.zero_grad(set_to_none=True)
        original_model=digest(t.network.state_dict());original_opt=digest(t.optimizer.state_dict())
        original_scaler=t.grad_scaler.state_dict().copy();rng_before=digest(rng_pack())
        scale=float(t.grad_scaler.get_scale())
        s=reserve_forward(read(WORK/"budget.json"),index);atomic(WORK/"budget.json",s)
        data=batch["data"].to(t.device);target=[v.to(t.device) for v in batch["target"]]
        with torch.autocast("cuda",enabled=True):
            output=t.network(data);base=t.loss(output,target)
        if output[0].shape!=(2,2,1024,1536):raise RuntimeError("output shape")
        aux=sparse_aux(output[0],batch["groups"]);total=base+coeff*aux
        if not states_finite(base,aux,total):raise RuntimeError("nonfinite loss")
        records=[]
        def record(row):
            authorized();row=dict(row,case_index=index,seed=case["seed"])
            records.append(row);log(WORK/"diagnostic_scaling.jsonl",row)
        params=tuple(t.network.named_parameters())
        g0,meta=bounded_gradients(base,params,scale,"base",record)
        if meta["tries"]>=4:raise RuntimeError("only one finite scale within four-scale cap; stability unresolved")
        def reference_record(row):record(dict(row,component="base_scale_reference"))
        reference,reference_meta=bounded_gradients(base,params,meta["accepted_scale"]/2,"aux",reference_record)
        aa=diff=rr=dot=0.
        for x,y in zip(g0,reference):
            if (x is None)!=(y is None):raise RuntimeError("gradient support mismatch")
            if x is not None:
                x=x.double();y=y.double();aa+=float(x.square().sum());rr+=float(y.square().sum())
                diff+=float((x-y).square().sum());dot+=float((x*y).sum())
        relative=math.sqrt(diff/aa) if aa else (0. if diff==0 else math.inf)
        cosine=dot/math.sqrt(aa*rr) if aa*rr else (1. if aa==rr==0 else 0.)
        if relative>.01 or cosine<.999:raise RuntimeError("finite diagnostic scales disagree")
        del reference
        ga,aux_meta=bounded_gradients(aux,params,128.,"aux",record)
        bb=sum(float(x.double().square().sum()) for x in ga if x is not None)
        ab=sum(float(x.double().mul(y.double()).sum()) for x,y in zip(g0,ga) if x is not None and y is not None)
        ratio=ratio_guard(math.sqrt(aa),math.sqrt(bb),coeff)
        del g0,ga
        if any(p.grad is not None for _,p in params):raise RuntimeError("diagnostic accumulated grad")
        if digest(rng_pack())!=rng_before or t.grad_scaler.state_dict()!=original_scaler:
            raise RuntimeError("diagnostic changed RNG/scaler")
        # Probe native total backward without allowing any parameter update.
        t.grad_scaler.scale(total).backward()
        t.grad_scaler.unscale_(t.optimizer)
        norm=float(torch.nn.utils.clip_grad_norm_(t.network.parameters(),12).detach().cpu())
        native="FINITE_NO_OPTIMIZER_STEP"
        if not math.isfinite(norm):
            t.grad_scaler.step(t.optimizer) # native found_inf must skip: prehook rejects any actual step
            t.grad_scaler.update()
            if t.grad_scaler.get_scale()>=scale:raise RuntimeError("native AMP failed to lower scale")
            native="NATIVE_AMP_SKIP_AND_BACKOFF"
        if optimizer_calls[0] or digest(t.network.state_dict())!=original_model or digest(t.optimizer.state_dict())!=original_opt:
            raise RuntimeError("parameters or optimizer changed")
        if digest(rng_pack())!=rng_before:raise RuntimeError("native backward changed RNG")
        row={"case_index":index,"seed":case["seed"],"case":case["name"],"logical_input":batch["logical_input"],
             "base_loss":float(base.detach()),"aux_loss":float(aux.detach()),"groups":[len(g) for g in batch["groups"]],
             "original_scale":scale,"diagnostic":meta,"scale_agreement_relative_L2":relative,
             "scale_agreement_cosine":cosine,"base_L2":math.sqrt(aa),"weighted_aux_L2":coeff*math.sqrt(bb),
             "weighted_ratio":ratio,"base_aux_cosine":ab/math.sqrt(aa*bb) if aa*bb else None,
             "native_total_preclip_L2":norm if math.isfinite(norm) else str(norm),
             "native_probe":native,"native_scale_after":float(t.grad_scaler.get_scale()),
             "optimizer_calls":optimizer_calls[0],"model_unchanged":True,"optimizer_unchanged":True,"RNG_unchanged":True,
             "GPU_peak_reserved_MiB":torch.cuda.max_memory_reserved()/1048576}
        if row["GPU_peak_reserved_MiB"]>10240:raise RuntimeError("memory cap")
        log(WORK/"cases.jsonl",row);summaries.append(row)
        t.optimizer.zero_grad(set_to_none=True);t.grad_scaler.load_state_dict(original_scaler)
        fw_hook.remove();opt_hook.remove()
        s=complete_case(read(WORK/"budget.json"),index);atomic(WORK/"budget.json",s)
        print("GPU_ZERO_CASE_PASS",index,case["seed"],case["name"],native,flush=True)
        del t,bundle,batch,output,base,aux,total,data,target,params
        gc.collect();torch.cuda.empty_cache()
    atomic(WORK/"WORKER_RESULT.json",{"status":"PASS","cases":summaries,"forwards":4,"optimizer_updates":0,"training_released":False},True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--parent",type=int,required=True);p.add_argument("--nonce",required=True)
    args=p.parse_args()
    try:execute(args.parent,args.nonce)
    except BaseException as exc:
        if (WORK/"budget.json").exists():
            s=read(WORK/"budget.json");s["status"]="STOPPED";s["reason"]=repr(exc);atomic(WORK/"budget.json",s)
        atomic(WORK/"WORKER_FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc(),"automatic_retry":False},True)
        raise
