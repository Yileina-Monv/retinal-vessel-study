"""Freeze newly approved100epochs, preserving every older frozen package."""
from pathlib import Path
import json,hashlib,copy,math,shutil,os,datetime
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';BASE=R/'experiments/nnunet2d_branch_20261020';S=BASE/'scientific_seed2026_100epochs_v3_amp_20261025';B=C/'research_branches/nnunet2d_20261020/scientific_amp_v3_20261025'
if S.exists():raise RuntimeError('science stage already exists; never overwrite')
S.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02/protocol.json';q=copy.deepcopy(json.loads(old.read_text()));phase='NNUNET_SCIENTIFIC100_AMP_V3_20261025'
eng=BASE/'amp_engineering32_20261025';accepted=json.loads((eng/'SOL_ENGINEERING_REVIEW.json').read_text());assert accepted['status']=='SOL_ENGINEERING32_ACCEPTED'
P1=R/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1';fixture=P1/'spatial_stratified_scientific_2000_v1_20261025/cpu_preparation/A0_endpoint_fixture/analysis/endpoint_2000.json';train=R/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json';dev=R/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json';d=json.loads(dev.read_text());names=[r['filename'] for r in json.loads(train.read_text())['rows']]
q.update(protocol_id=phase,status='FROZEN_USER_APPROVED_CONDITIONAL_ON_CPU_AND_SOL_RELEASE',date='2026-10-25')
q['authority']={'user_approval':'可以','scope':'one fresh seed2026 standard2D nnU-Net100epochs after accepted32 engineering; CPU-efficient observer checks then independent Sol release; final-only Dev28','executor_reviewer':'Sol directly implements/reviews; not Luna','classification':'single-seed repeated-Dev whole-pipeline comparison, not single-variable ablation or independent validation','P1':'previous P1 experiments and negative conclusions preserved; no new P1 trajectory authorized'}
q['scope'].pop('successful_updates_required_for_complete',None);q['scope'].update(successful_plus_handled_skips_equals=25000,extra_engineering_optimizer_updates=0,no_replay_budget=True,replay_exception='This is one separately approved new fresh trajectory; all previous interrupted scientific and engineering costs preserved; no automatic retries')
q['train_ids']=names;q['Dev_ids']=[r['filename'] for r in d['images']]
assert len(names)==110 and len(set(names))==110 and len(q['Dev_ids'])==28 and not set(names)&set(q['Dev_ids'])
q['paths'].update(scientific_root=str(S),nnUNet_results=str(S/'nnunet_results'),branch_code=str(B),freeze_directory=str(S),predictions=str(S/'dev_predictions'),evaluation=str(S/'analysis'))
q['recipe']['initialization']='fresh official random initialization, match original seed2026 model digest; never load any previous or engineering weights'
q['AMP']={'handled_overflow':'finite forward Loss/model/optimizer; decreasing GradScaler; no optimizer call and exact unchanged parameter/optimizer state; record skip separately','consecutive_pause':5,'total_pause':25,'replace_skips':False,'fatal':'nonfinite forward/model/optimizer, unhandled overflow, OOM, inconsistent counts/source/resource/worker failure; stop without retry','counting':'reserve attempted work persistently before forward; unknown and failed work remain charged'}
q['monitoring'].update(logging='Per-attempt real optimizer calls, true unscaled preclip gradient L2, AMP scales, input/target fingerprints, LR, outcome, persistent budget and heartbeat; no full-state CPU copies at every finite step',checkpoints='Atomic complete last bundle at first30/200, every250attempts and every epoch end; retain first30/200 and epoch20/40/60/80/100 snapshots; original native periodic/best/final preserved but only final evaluated; include model/optimizer/scaler/scheduler/main RNG/cursor/budget/log prefixes/partial epoch outputs',completion='exact100epochs/25000attempts/5000 Train monitors; successes+handled skips25000, fatal/unknown0, no stop and skips below5consecutive/25total; report actual successes')
q['randomness']['failure_resume']='No automatic restart/resume; full available state saved, native augmentation queues unavailable, never claim strict native-stream continuation; interruption requires separate user decision.'
q['budget'].update(device='Only physical GPU0 UUID ${GPU_UUID}; exclusive shared project lock; GPU1 not queried/used',GPU1_allowed=False,cumulative_gpu_wall_cap_seconds=None,estimated_gpu_hours={'old_standard_observed_estimate':10.5563,'new_efficient_observer':'unknown until real first scientific segment; report measured ETA, no normal-progress hard deadline'},stop_at_cap='25000attempts and5000 Train monitors; no supplemental attempts/epochs, no automatic replay')
q['inference'].update(checkpoint='checkpoint_final.pth',when='Only after complete100epochs/25000attempts/5000Train monitors and accepted counters; exactly one Dev28 attempt per case',no_automatic_oom_fallback=True,outputs=28)
q['inference']['expected_tiles_by_image']={r['filename']:math.prod(math.ceil(max(n-p,0)/(p*.5))+1 for n,p in zip(r['native_shape'],[1024,1536])) for r in d['images']}
q['evaluation']={'primary':'ordinary Dice, frozen image-macro on Dev28 threshold0.5, paired NNUNET-A0','success_signal':'mean>0 and whole-image paired bootstrap95% lower>0; image bootstrap does not show seed stability','protection':'original precision/farRecall nondegradation and FP-FOV nonincrease strict intervals; also report mean Recall/coarse/hard-clDice nonadverse; Dice signal and guard costs separate','secondary':'original neighborhood9valid/77positions/19NA, full S3, existing verified candidate paths and structural diagnostics; no metric replacement','A0':'reuse original seed2026/2000 probabilities, no new training or inference','validity':'exact existing GT-valid image IDs from accepted A0 CPU fixture, never prediction-based NA removal','no_selection':'fixed final only, no Dev curves/threshold/best/seed search'}
q['references']={'accepted_A0_fixture':str(fixture),'accepted_A0_fixture_sha256':sha(fixture),'science_reference_manifest':str(P1/'scientific_b1_b2_2000_v1_20261022/references/dev/manifest.json'),'A0_reuse_binding':str(P1/'scientific_b1_b2_2000_v1_20261022/a0_reuse_binding.json'),'Dev_RGB_manifest':str(dev),'Train_manifest':str(train),'accepted_engineering_review':str(eng/'SOL_ENGINEERING_REVIEW.json'),'accepted_engineering_review_sha256':sha(eng/'SOL_ENGINEERING_REVIEW.json'),'parent_protocol':str(old),'parent_protocol_sha256':sha(old),'engineering_source_binding':str(eng/'cpu_binding.json')}
q.pop('operational_amendment',None)
norm='''# nnU-Net100epochs科学运行 v3：本轮已批准，CPU与独立release前GPU HOLD

仅一条fresh随机初始化seed2026、Train110/fold=all标准2D nnU-Net；原Batch2/Patch1024×1536/PlainConvUNet/CE+Dice+deep supervision/SGD/Poly100/RGB归一化/采样增强保持。不是M0微调，不与A0声称单变量公平；不同支持域和训练预算明示为流程比较。

100epochs×250=25000次尝试，5000 Train内部监控批次。成功与受控AMP跳步分别记账，不补步；连续5或累计25跳步暂停。非有限前向Loss/参数/optimizer、未拦截异常更新、OOM、来源/计数/资源异常即停；失败/未知不退款，不自动重启。此前所有中断及工程消耗保留，不从工程或旧权重续训。不追加GPU预演。

GPU0独占，GPU1不查询/使用；原11GiB设备Batch/Patch保持，不硬压8GiB。无正常进度10小时墙钟截断；仅30分钟无进度心跳的技术挂起和资源异常触发停止。旧标准耗时约10–12小时是历史参考，新低开销观察器先CPU等价检查，正式首段更新实测ETA。可用检查点包含真实状态/主RNG/游标/预算/日志前缀/部分epoch汇总；原生增强队列不可序列化，不宣称严格原生流恢复。

标准官方run_training/train_step保留，只加分支观察器；正常步骤用设备内参数副本证明实际更新，聚合有限性检查；全量摘要集中检查点，安全跳步时精确核验模型/optimizer未变。关闭非训练所需的architecture graph随机debug前向，以避免额外GPU工作；不影响训练数据或Loss。真实科学前30步按实际输入/Loss/LR/梯度/AMP与已验收工程前30有限步骤核对，不重放、不另加GPU预算。

只在100epochs/25000尝试/5000Train监控完成，零致命/未知、跳步未达暂停线后，使用唯一checkpoint_final.pth对Dev28一次原生概率推理，阈值0.5、无TTA/集成/后处理/重试。使用原官方预处理与逆变换；CPU三尺寸通道/坐标/背景复原核验。每图预先冻结tile数量，失败即止，不启用OOM自动重试。

普通Dice为唯一主目标，原GT/FOV/Omega/图像宏平均/近远处/d1/NA保持；hard-clDice按既有full-FOV定义（含U）诊断，原S3和已验证细分支指标作辅助。复用原A0概率，固定有效集合与整图PCG64 seed20261016/1000重采样/linear分位数。Dice均值>0且95%下限>0才有单seed探索信号；原Precision/远处Recall/FP-FOV严格保护及全图Recall/粗血管Recall/hard-clDice平均保护分别报告，不以单项改善掩盖代价。无论正负只要完整、合规、可复算即工程交付；单seed反复Dev不算独立验证，不自动扩种子、阈值或预算。

训练、终点推理、CPU归档由绑定的后台controller依次执行，失败保留现场并停止。自动CPU判读明确“待Sol科研复盘”，不冒称未来人工已审核。官方Test60封存，原正式T0/TW负面结论、S3 H1和人工0/26状态保持。
'''
(S/'SCIENTIFIC_PROTOCOL.md').write_text(norm,encoding='utf-8');q['normative_text_sha256']=sha(S/'SCIENTIFIC_PROTOCOL.md')
with (S/'protocol.json').open('x',encoding='utf-8') as f:json.dump(q,f,ensure_ascii=False,indent=2);f.write('\n')
gate=B/'science100_gate.py';text=gate.read_text();assert text.count('TO_BE_FROZEN')==1;gate.write_text(text.replace('TO_BE_FROZEN',sha(S/'protocol.json')),encoding='utf-8')
backup=S/'management_before_authorization';backup.mkdir();paths=[C/'STATUS.md',C/'项目全局状态与研究决策.md'];previous={p:p.read_bytes() for p in paths}
for p in paths:shutil.copy2(p,backup/p.name)
record={'phase':phase,'protocol_sha256':sha(S/'protocol.json'),'state':'CPU_PREPARATION','user_approved_science100':True,'GPU_released':False,'execution_started':False,'seed':2026,'epochs':100,'attempt_cap':25000,'monitor_cap':5000,'stage_path':str(S),'executor_reviewer':'Sol directly implementing/reviewing','next':'CPU efficient observer/inference/endpoint admission then independent Sol science release; exactly one fresh100epochs trajectory, no extra GPU engineering/restart'}
header='### nnU-Net完整100epochs已获用户批准，CPU接线与独立release前GPU HOLD\n\n本轮“可以”承接已验收AMP32工程后的唯一fresh seed2026/100epochs方案，Sol直接实施。仅25000尝试+5000 Train内部监控，成功/安全跳步分别记账、不补步，连续5/累计25跳步暂停；标准nnU-Net配方和原评价保护保持。先对低开销观察器做CPU等价/恢复/部分epoch/评价接线检查，再单独release；不追加GPU预演或复用工程权重。GPU0独占/GPU1禁用，正常进度不硬限10小时；新入口ETA需首段实测。final模型才一次Dev28/.5，Test60封存，单seed流程比较、非独立验证。证据 '+str(S)+'。\n\n'
for p in paths:
 first,_,rest=previous[p].decode().partition('\n');note=header
 if p.name=='STATUS.md':note+='<!-- NNUNET_SCIENCE100_CURRENT -->\n'+json.dumps(record,ensure_ascii=False)+'\n<!-- /NNUNET_SCIENCE100_CURRENT -->\n\n'
 tmp=p.with_name(p.name+'.nnunet100_auth.pending')
 with tmp.open('x',encoding='utf-8') as f:f.write(first+'\n\n'+note+rest.lstrip('\n'));f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==previous[p];os.replace(tmp,p)
print('SCIENCE100_PROTOCOL_FROZEN',sha(S/'protocol.json'),flush=True)
