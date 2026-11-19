"""Record approved continuation; never edit frozen V4 or refund its1000 updates."""
from pathlib import Path
import json,hashlib,copy,shutil,os
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';BASE=R/'experiments/nnunet2d_branch_20261020';OLD=BASE/'scientific_seed2026_100epochs_v4_prefixfix_20261025';S=BASE/'scientific_seed2026_100epochs_v5_resume1000_20261025';B=Path(__file__).resolve().parent;PHASE='NNUNET_SCIENTIFIC100_RESUME_V5_20261025'
if S.exists():raise RuntimeError('resume stage exists; inspect rather than overwrite')
S.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=copy.deepcopy(json.loads((OLD/'protocol.json').read_text()));oldbudget=json.loads((OLD/'budget.json').read_text())
assert (oldbudget['attempts'],oldbudget['success'],oldbudget['monitor_complete'],oldbudget['epoch'],oldbudget['iteration'],oldbudget['monitor_iteration'],oldbudget['phase'])==(1000,1000,152,3,250,2,'monitor')
q.update(protocol_id=PHASE,version=5,status='FROZEN_USER_APPROVED_CONTROLLED_CONTINUATION_CPU_AND_SOL_RELEASE_REQUIRED')
q['authority']={'user_approval':'你别停啊','scope':'continue saved1000 updates, repair false watchdog; remaining24000 train attempts and4848 Train monitor batches; no restart from zero or extra GPU engineering','executor_reviewer':'Sol directly implementing/reviewing','classification':'single-seed repeated-Dev whole-pipeline comparison with disclosed interrupted/rebuilt augmentation queues','module_work':'CPU proposal only, no module training'}
q['paths'].update(scientific_root=str(S),nnUNet_results=str(S/'nnunet_results'),branch_code=str(B),freeze_directory=str(S),predictions=str(S/'dev_predictions'),evaluation=str(S/'analysis'))
q['scope']['replay_exception']='Approved continuation1000+24000=25000, no replay or refund. Previous independent V3 and engineering costs remain recorded.'
q['scope']['new_scientific_trajectories']=0
native=list((OLD/'nnunet_results').glob('**/checkpoint_best.pth'));assert len(native)==1
q['continuation']={'parent_stage':str(OLD),'parent_protocol_sha256':sha(OLD/'protocol.json'),'checkpoint':str(OLD/'state_interrupted.pt'),'checkpoint_sha256':sha(OLD/'state_interrupted.pt'),'old_budget_sha256':sha(OLD/'budget.json'),'native_logger_checkpoint':str(native[0]),'native_logger_checkpoint_sha256':sha(native[0]),'retained_training_attempts':1000,'retained_successful_updates':1000,'remaining_train_attempts':24000,'retained_monitor_batches':152,'remaining_monitor_batches':4848,'first_action':'finish remaining48 monitor batches of epoch3, then finalize epoch4; no repeated SGD','model_optimizer_scaler_scheduler_main_rng':'exact available saved-state restoration required before any new forward','native_augmentation_queue':'not serializable; rebuilt one train worker2026 and one monitor worker2027; NOT bitwise uninterrupted continuation','logger':'past3 completed epochs from native checkpoint; exact Train250 outputs from custom state; epoch start time approximated from actual console and not used for speed claims','source_false_stop_review':str(OLD/'watchdog_cpu_review_20261025/REVIEW.json')}
q['watchdog']={'progress_source':'persistent actual phase/forward/backward/optimizer/monitor counters','clock':'time.monotonic','timeout_seconds':1800,'action':'warning only; never terminate worker for heartbeat/NAS/clock/read failures','numerical_source_budget_failures':'unchanged hard stops','automatic_restart':False}
q['prefix_contract']['hard_exact']=['source binding','restored1000-update model/optimizer/scaler/scheduler/main RNG','copied exact1000 Train +152 monitor log prefixes','Train110 source and IDs; rebuilt worker seeds2026/2027']
q['stops']=[x for x in q['stops'] if x!='unknown sampler/queue recovery or interrupted scientific run']+['undisclosed continuation or mismatch of available saved state before first new forward']
norm=(OLD/'SCIENTIFIC_PROTOCOL.md').read_text()+'\n\n## V5 用户批准的1000步接续修订（仅本次，不改旧V4）\n\n用户“你别停啊”批准保留1000次成功更新，先补48 Train监控，再继续剩24000训练尝试/4848监控，总预算仍25000/5000。原V4文件及所有历史费用保持。模型、优化器、AMP、LR、主RNG按已保存状态精确恢复；原生异步增强队列不可恢复，重新建立2026/2027各一worker，明确非逐位精确接续。过去3 epoch日志来自原生checkpoint，本epoch训练loss来自250份原始输出；epoch起点用控制台时间近似重建，不用于速度判断。控制器仅以真实计数和monotonic报警，心跳超时/NAS读数不再杀进程；数值、源、身份、累计预算和5/25 AMP停止线保持。唯一final才Dev28/.5，Test/GPU1禁用，不自动重启或加组。此段仅在接续/监控恢复上取代旧文字中fresh-only/no-resume要求；其他科学定义不变。\n'
(S/'SCIENTIFIC_PROTOCOL.md').write_text(norm);q['normative_text_sha256']=sha(S/'SCIENTIFIC_PROTOCOL.md');(S/'protocol.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
g=B/'science100_gate.py';text=g.read_text();assert 'PROTOCOL_HASH_PENDING' in text;g.write_text(text.replace('PROTOCOL_HASH_PENDING',sha(S/'protocol.json')))
for n,count in [('attempts.jsonl',1000),('monitor.jsonl',152)]:
    assert len((OLD/n).read_text().splitlines())==count;shutil.copy2(OLD/n,S/n);assert sha(S/n)==sha(OLD/n)
shutil.copy2(OLD/'prior_usage_accounting.json',S/'prior_usage_accounting.json')
(S/'continuation_authorization.json').write_text(json.dumps({'user_message':'你别停啊','scope':q['authority'],'parent_consumed_counts':oldbudget,'parent_checkpoint_sha256':q['continuation']['checkpoint_sha256'],'remaining':{'train_attempts':24000,'monitor':4848},'no_refund_no_replay':True},ensure_ascii=False,indent=2)+'\n')
backup=S/'management_before_resume_authorization';backup.mkdir();paths=[C/'STATUS.md',C/'项目全局状态与研究决策.md'];previous={p:p.read_bytes() for p in paths}
for p in paths:(backup/p.name).write_bytes(previous[p])
record={'phase':PHASE,'protocol_sha256':sha(S/'protocol.json'),'state':'CPU_PREPARATION','user_approved_science100':True,'user_approved_resume1000':True,'GPU_released':False,'execution_started':False,'seed':2026,'epochs':100,'attempt_cap':25000,'monitor_cap':5000,'stage_path':str(S),'executor_reviewer':'Sol directly implementing/reviewing','charged_attempts':1000,'successful_updates':1000,'monitor_complete':152,'remaining_training_attempts':24000,'native_queue_resume_supported':False,'next':'CPU resume/logger/migration review then independent Sol release and GPU0 continuation; no extra engineering; module work CPU proposal only'}
header='### nnU-Net1000步接续已获批准，CPU修复与独立放行准备中\n\n用户“你别停啊”批准保留1000成功，不从零重跑；先补当前48个Train监控，再运行剩24000训练尝试，总25000/5000不扩充。监控改真实计数+monotonic，仅告警不因心跳杀训练；数值/AMP/预算停止线不变。可用状态精确恢复，增强队列重建须披露，不宣称逐位精确接续。旧V4及所有费用保留。CPU审查与Sol独立release前GPU HOLD；Test/GPU1封存，模块仍仅CPU提案。证据 '+str(S)+'。\n\n'
for p in paths:
    text=previous[p].decode()
    if p.name=='STATUS.md':
        a='<!-- NNUNET_SCIENCE100_CURRENT -->';z='<!-- /NNUNET_SCIENCE100_CURRENT -->';lo=text.index(a)+len(a);hi=text.index(z);text=text[:lo]+'\n'+json.dumps(record,ensure_ascii=False)+'\n'+text[hi:]
    first,_,rest=text.partition('\n');tmp=p.with_name(p.name+'.nnunet_v5_auth.pending')
    with tmp.open('x',encoding='utf-8') as f:f.write(first+'\n\n'+header+rest.lstrip('\n'));f.flush();os.fsync(f.fileno())
    assert p.read_bytes()==previous[p];os.replace(tmp,p)
print('V5_CONTINUATION_CPU_HOLD',sha(S/'protocol.json'),flush=True)
