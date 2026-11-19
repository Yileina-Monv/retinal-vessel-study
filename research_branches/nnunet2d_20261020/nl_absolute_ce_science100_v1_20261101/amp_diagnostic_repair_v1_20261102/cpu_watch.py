"""CPU-only completion reconciler; cannot grant GPU or training authorization."""
import os, sys, time, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from paths import *
from common import read, sha, atomic, authority_entry

def alive(job):
    p=Path("/proc")/str(job["pid"])
    try:
        raw=(p/"stat").read_text().rsplit(")",1)[1].split()
        return raw[0]!="Z" and raw[19]==job["process_start_ticks"]
    except FileNotFoundError:
        return False

def finish(status, details):
    value={"status":status,"details":details,"GPU_released":False,"training_released":False,
           "extra_training_attempts_approved":False,"official_Test_accessed":False,
           "scientific_model_updates":0,"completed_at_unix":time.time()}
    atomic(OUT/"CPU_PREPARATION_RESULT.json",value,True)
    if status=="PASS_CPU_PREPARATION":
        body=("NL_SCIENCE100_RELEASED_20261101。自动CPU事实核验完成：诊断修复13项测试通过；"
              "两份真实中断状态已在CPU原架构恢复并核对模型/optimizer/scaler/scheduler/logger/CPU RNG；"
              "Train回放2027共10903批、2028共10036批，覆盖所有已执行输入、辅助目标、两中断批及各8批后续输入，"
              "与N0完整input/GT/ID/LR核对；Train monitor分别2150/2000批只证明历史ID一致，历史缺少tensor摘要的局限保留。"
              "旧代码、协议、快照、账本及冻结来源SHA复核未变。原20923物理尝试、20921成功、2unknown保留。"
              "证据原NL science100_v1/amp_diagnostic_repair_v1_20261102/CPU_PREPARATION_RESULT.json。"
              "这是自动CPU验收事实，不是GPU或科学训练放行；仍待Sol审查并向用户提交最多4前向/零更新/30分钟GPU复核。"
              "重新处理两中断输入需另行批准合计2次物理尝试，逻辑100epochs终点不变。CUDA RNG和GPU数值尚未实测，"
              "未保存原异步队列；不得把CPU匹配称为整个训练精确接续已证实。新恢复训练入口尚未放行，不自动续训。")
    else:
        body=("NL_SCIENCE100_RELEASED_20261101。CPU恢复准备未通过，保持暂停。"
              "详情见原NL science100_v1/amp_diagnostic_repair_v1_20261102/CPU_PREPARATION_RESULT.json；"
              "原故障、预算、权重保留。自动记录不授权GPU/重跑/Test，不改变科学规则。")
    authority_entry("AMP诊断修复CPU准备"+("完成，GPU仍待独立审查" if status=="PASS_CPU_PREPARATION" else "暂停")+"（自动事实记录）",body)

def main():
    jobs=read(OUT/"CPU_REPLAY_JOBS.json")
    binding=read(OUT/"REPAIR_SOURCE_BINDING.json")
    started=time.monotonic()
    while True:
        states=[];all_done=True
        for job in jobs:
            root=OUT/("replay_seed_"+str(job["seed"]))
            result=root/"RESULT.json";failure=root/"FAILURE.json"
            if failure.exists():
                finish("HOLD_CPU_REPLAY_FAILURE",{"seed":job["seed"],"failure":read(failure)});return
            if result.exists():
                value=read(result)
                assert value["status"]=="PASS"
                assert value["matched_train"]==({2027:10903,2028:10036}[job["seed"]])
                assert value["next_inputs_verified"]==8 and value["GPU_calls"]==value["model_forwards"]==value["optimizer_updates"]==0
                states.append(value)
            else:
                all_done=False
                if not alive(job):
                    finish("HOLD_CPU_WORKER_EXIT_WITHOUT_RESULT",{"seed":job["seed"],"pid":job["pid"]});return
                progress=root/"progress.json"
                states.append(read(progress) if progress.exists() else {"seed":job["seed"],"state":"STARTING"})
        atomic(OUT/"cpu_preparation_progress.json",{"state":"CHECKING" if all_done else "RUNNING",
               "seeds":states,"GPU_released":False,"training_released":False,"updated":time.time()})
        if all_done:break
        if time.monotonic()-started>21660:
            finish("HOLD_CPU_TIMEOUT",{"seeds":states});return
        time.sleep(30)
    checks=0
    for p,h in binding["sources"].items():
        if sha(p)!=h:raise RuntimeError("repair source drift "+p)
        checks+=1
    for p,h in read(OUT/"PARENT_MANIFEST.json")["files"].items():
        if sha(p)!=h:raise RuntimeError("parent source/state/ledger changed "+p)
        checks+=1
    # Validate the original scientific binding, including its immutable input sources.
    for binding_path in [GROUP/"SOURCE_BINDING.json",GROUP/"parallel_gpu1_v1/SOURCE_BINDING.json"]:
        original=read(binding_path)
        for field in ["runtime_sources","immutable_sources"]:
            for p,h in original.get(field,{}).items():
                if sha(p)!=h:raise RuntimeError("frozen source mismatch "+p)
                checks+=1
    tests=read(OUT/"CPU_TESTS.json");restore=read(OUT/"CPU_RESTORE.json")
    assert tests["status"]==restore["status"]=="PASS" and tests["case_count"]==13
    assert len(restore["seeds"])==2 and restore["GPU_calls"]==0 and not restore["science_ledgers_changed"]
    for s in states:
        for item in s["selected_diagnostics"]:
            assert sha(item["path"])==item["sha256"]
            checks+=1
    finish("PASS_CPU_PREPARATION",{"sha_checks":checks,"replay_results":states,
           "tests":str(OUT/"CPU_TESTS.json"),"state_restore":str(OUT/"CPU_RESTORE.json"),
           "source_binding_sha256":sha(OUT/"REPAIR_SOURCE_BINDING.json")})

if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if not (OUT/"CPU_PREPARATION_RESULT.json").exists():
            finish("HOLD_CPU_REVIEW_EXCEPTION",{"error":repr(exc),"traceback":traceback.format_exc()})
        raise
