import os,sys,time,ctypes,signal,traceback,copy,shutil
from support import *

def execute(seed,parent,nonce):
 if os.getppid()!=parent or os.environ.get("NL_RESUME_NONCE")!=nonce or len(nonce)<32:raise RuntimeError("not authenticated resume child")
 key=validate(full=True);authorize=runtime_authorizer(seed,parent,nonce,key)
 if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError("parent death guard")
 if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPUS[seed] or os.environ.get("PYTHONHASHSEED")!=str(seed):raise RuntimeError("GPU/seed drift")
 import torch,numpy as np
 from fast_observer import digest,rng_pack
 from nnunet2d_scientific_constructor import build_fresh_trainer
 from seeded_runtime import set_all_seeds,install_seeded_worker_adapter
 from amp32_runner import bind_train_datasets
 from sidecar import install
 from selection import choose_groups
 from common import compare_train
 from recovery import recovery_manifest,restore_state
 from journal import ResumeJournal
 from observer import ResumeObserver
 from resume_loop import run_remainder
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if torch.cuda.is_initialized():raise RuntimeError("CUDA before admission")
 if torch.cuda.device_count()!=1:raise RuntimeError("one released GPU required")
 torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(10240*1048576/torch.cuda.get_device_properties(0).total_memory)
 set_all_seeds(seed)
 s=stage(seed);old=parent_stage(seed)
 bundle=torch.load(old/"state_interrupted.pt",map_location="cpu",weights_only=False)
 old_budget=read(old/"budget.json");manifest=recovery_manifest(bundle,old_budget)
 logical=manifest["logical_completed"]
 os.environ.update(nnUNet_raw=str(DATA_BASE/"raw"),nnUNet_preprocessed=str(DATA_BASE/"preprocessed"),
                   nnUNet_results=str(s/"nnunet_results"),nnUNet_compile="false",nnUNet_n_proc_DA="1",
                   MPLCONFIGDIR=str(s/"matplotlib"))
 t=build_fresh_trainer(read(DATA_BASE/"preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json"),
                       read(DATA_BASE/"preprocessed/Dataset501_MAPLES/dataset.json"),torch.device("cuda:0"))
 pools=install_seeded_worker_adapter(seed);t.initialize()
 baseline_init=read(BASELINE/"runs"/("seed_"+str(seed))/"initialization.json")
 for field,value in [("model",t.network.state_dict()),("optimizer",t.optimizer.state_dict()),("scaler",t.grad_scaler.state_dict()),("RNG",rng_pack())]:
  if digest(value)!=baseline_init[field]:raise RuntimeError("constructor seed/model state mismatch "+field)
 if any(True for _ in t.network.buffers()):raise RuntimeError("mutable model buffers")
 if list(t.loss.weight_factors)!=[64/127,32/127,16/127,8/127,4/127,2/127,1/127,0]:raise RuntimeError("DS changed")
 def forbidden(*a,**kw):raise RuntimeError("model forward/update forbidden during loader reconstruction")
 deny_forward=t.network.register_forward_pre_hook(forbidden)
 deny_update=t.optimizer.register_step_pre_hook(forbidden)
 install();names=set(read(BASELINE/"runs"/("seed_"+str(seed))/"protocol.json")["train_ids"])
 bind_train_datasets(t,names)
 t.plot_network_architecture=lambda:None
 journal=None;observer=None
 def stop_signal(signum,frame):raise RuntimeError("external stop "+str(signum)+"; no automatic resume")
 signal.signal(signal.SIGTERM,stop_signal)
 try:
  progress(seed,phase="REBUILDING_NATIVE_INPUT_STREAM",matched=0,total=logical+1,science_updates=0)
  t.on_train_start() # native GPU/pinned augmentation setup, metadata; no model forward
  history=[json.loads(x) for x in (BASELINE/"runs"/("seed_"+str(seed))/"attempts.jsonl").read_text().splitlines()]
  expected_targets=[json.loads(x) for x in (old/"auxiliary_targets.jsonl").read_text().splitlines()]
  old_monitors=[json.loads(x) for x in (old/"monitor.jsonl").read_text().splitlines()]
  reconstructed_monitors=[json.loads(x) for x in (OUT/("replay_seed_"+str(seed))/"matched_monitors.jsonl").read_text().splitlines()]
  if len(expected_targets)!=logical+1:raise RuntimeError("parent target prefix length")
  monitor_count=0
  for number in range(1,logical+2):
   batch=next(t.dataloader_train);epoch=(number-1)//250
   compare_train(batch,history[number-1],number,epoch,.01*(1-epoch/100)**.9,digest)
   if not set(batch["keys"])<=names:raise RuntimeError("non-Train input")
   groups=[choose_groups(r["candidates"],seed,number,i) for i,r in enumerate(batch["nl_candidates"])]
   e=expected_targets[number-1]
   if digest(groups)!=e["selected_groups_sha256"] or list(batch["keys"])!=e["ids"] or [r["counts"] for r in batch["nl_candidates"]]!=e["counts"]:
    raise RuntimeError("live auxiliary target reconstruction mismatch")
   if number%250==0 and number<=logical:
    for _ in range(50):
     m=next(t.dataloader_val);e=old_monitors[monitor_count];ref=reconstructed_monitors[monitor_count]
     if list(m["keys"])!=e["ids"] or not set(m["keys"])<=names or "nl_candidates" in m:raise RuntimeError("monitor source/ID mismatch")
     if digest(m["data"])!=ref["reconstructed_input_sha256"] or digest(m["target"])!=ref["reconstructed_target_sha256"]:
      raise RuntimeError("monitor differs from verified CPU reconstruction")
     monitor_count+=1
   if number%100==0 or number==logical+1:
    authorize();progress(seed,phase="REBUILDING_NATIVE_INPUT_STREAM",matched=number,total=logical+1,monitor_IDs=monitor_count,science_updates=0)
    print("RESUME_CPU_INPUT",seed,number,logical+1,flush=True)
  if monitor_count!=old_budget["monitor_complete"]:raise RuntimeError("monitor cursor mismatch")
  if pools!=[{"pool":"train","worker_count":1,"seeds":[seed]},{"pool":"internal_train_monitor","worker_count":1,"seeds":[seed+1]}]:raise RuntimeError("worker seed pool drift")
  if digest(batch["data"])!=read(OUT/("replay_seed_"+str(seed))/"RESULT.json")["selected_diagnostics"][-1]["input_sha256"]:
   raise RuntimeError("interrupted batch not reproduced")
  restored=restore_state(t,bundle,include_cuda_rng=True)
  if digest(rng_pack())!=digest(bundle["rng"]):raise RuntimeError("main RNG restore mismatch")
  if t.optimizer.param_groups[0]["lr"]!=.01*(1-t.current_epoch/100)**.9:raise RuntimeError("restored LR mismatch")
  journal=ResumeJournal(s/"budget.json",key);journal.migrate(old_budget,sha(old/"budget.json"))
  def event(kind,counts,o):
   resolved=counts["success"]+counts["skip"]+counts["fatal"]
   if kind=="train":
    log(s/"logical_physical_index.jsonl",{"logical_input":resolved,"physical_attempt":counts["attempts"],"success":counts["success"],"AMP_skips":counts["skip"],"inherited_unknown":counts["unknown"]})
   if (kind=="train" and (resolved==logical+1 or resolved%250==0)) or kind=="epoch":
    o.save_bundle(s/"state_last.pt",key)
    if kind=="epoch" and counts["epoch"] in [60,80,100]:
     shutil.copy2(s/"state_last.pt",s/("state_epoch_%03d.pt"%counts["epoch"]))
   if kind in ["train","epoch"]:
    progress(seed,phase="TRAINING",completed_epochs=counts["epoch"],iteration=counts["iteration"],
             logical_completed=resolved,charged_attempts=counts["attempts"],successful_updates=counts["success"],AMP_skips=counts["skip"],
             science_updates_since_resume=counts["success"]-old_budget["success"])
   if kind=="train" and (resolved==logical+1 or resolved%100==0):
    print("RESUME_UPDATE",seed,resolved,counts["attempts"],counts["success"],counts["skip"],flush=True)
   if shutil.disk_usage(s).free<50*2**30:raise RuntimeError("free disk below 50GiB")
  observer=ResumeObserver(t,journal,seed,authorize,event)
  observer.partial_train=restored["partial_train"];observer.partial_monitor=restored["partial_monitor"]
  if len(observer.partial_train)!=old_budget["iteration"]:raise RuntimeError("partial epoch restore failed")
  atomic(s/"RESTORE_VERIFIED.json",{"status":"PASS","parent_checkpoint_sha256":sha(old/"state_interrupted.pt"),
        "parent_budget_sha256":sha(old/"budget.json"),"logical_resume_input":logical+1,
        "physical_resume_attempt":journal.read()["attempts"]+1,"live_prefix_batches":logical+1,
        "monitor_IDs":monitor_count,"monitor_matches_CPU_reconstruction":True,
        "historical_monitor_tensor_hash_available":False,"model_optimizer_scaler_scheduler_logger_RNG_match":True,
        "saved_partial_epoch_outputs":len(observer.partial_train),"native_live_workers_preserved":True,
        "extra_LR_steps_at_resume":0,"science_updates_so_far":0},True)
  deny_forward.remove();deny_update.remove()
  del bundle,restored,history,expected_targets,old_monitors,reconstructed_monitors
  progress(seed,phase="TRAINING",logical_completed=logical,charged_attempts=old_budget["attempts"],successful_updates=old_budget["success"],science_updates_since_resume=0)
  run_remainder(t,observer,journal,batch)
  counts=journal.read();final=Path(t.output_folder)/"checkpoint_final.pth"
  if counts["attempts"]!=25001 or counts["success"]+counts["skip"]!=25000 or counts["unknown"]!=1 or counts["fatal"] or counts["status"]!="READY" or counts["phase"]!="complete" or counts["monitor_complete"]!=5000 or t.current_epoch!=100 or not final.exists():
   raise RuntimeError("resumed terminal budget criteria failed")
  if observer.paired_train!=25000 or observer.paired_monitor!=5000:raise RuntimeError("resumed N0 pairing incomplete")
  observer.save_bundle(s/"state_last.pt",key)
  atomic(s/"PAIRING_COMPLETE.json",{"status":"PASS","logical_train_pairs":25000,"physical_attempts":25001,"monitor_ID_pairs":5000,
        "parent_training_pairs":logical,"tail_training_pairs":25000-logical,"monitor_tensor_hash_historical_proof":False},True)
  atomic(s/"training_completion.json",{"status":"TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING","seed":seed,"epochs":100,
        "counts":counts,"checkpoint_final":str(final),"checkpoint_final_sha256":sha(final),
        "parent_checkpoint":str(old/"state_interrupted.pt"),"binding_sha256":key,"Dev_Test_calls":0},True)
  progress(seed,phase="TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING",logical_completed=25000,
           charged_attempts=25001,successful_updates=counts["success"],AMP_skips=counts["skip"])
 except BaseException as exc:
  if journal is not None:
   journal.stop(repr(exc))
   if observer is not None:
    try:observer.save_bundle(s/"state_interrupted.pt",key)
    except BaseException as se:atomic(s/"snapshot_failure.json",{"error":repr(se)})
  atomic(s/"training_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),
       "counts":journal.read() if journal is not None else None,"automatic_restart":False},True)
  raise
 finally:
  if observer is not None:observer.close()
  for loader in [getattr(t,"dataloader_train",None),getattr(t,"dataloader_val",None)]:
   if hasattr(loader,"_finish"):
    try:loader._finish()
    except BaseException:pass

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,choices=[2027,2028],required=True)
 p.add_argument("--parent",type=int,required=True);p.add_argument("--nonce",required=True)
 a=p.parse_args()
 try:execute(a.seed,a.parent,a.nonce)
 except BaseException as exc:
  f=stage(a.seed)/"training_failure.json"
  if not f.exists():atomic(f,{"error":repr(exc),"traceback":traceback.format_exc(),"phase":"STARTUP_OR_REBUILD","automatic_restart":False},True)
  raise
