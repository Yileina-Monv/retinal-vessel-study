"""Apply user-approved prefix-gate amendment to a fresh, traceable replacement."""
from pathlib import Path
import json,hashlib,copy,shutil,os,datetime
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';BASE=R/'experiments/nnunet2d_branch_20261020';OLD=BASE/'scientific_seed2026_100epochs_v3_amp_20261025';S=BASE/'scientific_seed2026_100epochs_v4_prefixfix_20261025';B=C/'research_branches/nnunet2d_20261020/scientific_prefix_v4_20261025';PHASE='NNUNET_SCIENTIFIC100_PREFIX_V4_20261025'
if S.exists():raise RuntimeError('replacement stage exists; inspect instead of overwrite')
S.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=copy.deepcopy(json.loads((OLD/'protocol.json').read_text()))
repair=OLD/'prefix_gate_review_20261025/CPU_REPAIR_REVIEW.json';review=json.loads(repair.read_text());assert review['verified_successes']==2 and review['new_candidate_CPU_tests']==6
q.update(protocol_id=PHASE,status='FROZEN_USER_APPROVED_FRESH_REPLACEMENT_CPU_AND_SOL_RELEASE_REQUIRED',version=4)
q['authority']={'user_approval':'先把nnUnet跑完吧，让他先跑，我们再来研究模块方案。','scope':'adopt prepared CPU prefix amendment and one fresh seed2026/100epochs replacement; old2 updates and40.12s retained; no extra engineering; original data/recipe/final evaluation unchanged','executor_reviewer':'Sol directly implementing/reviewing','classification':'single-seed repeated-Dev whole-pipeline comparison','module_work':'CPU scientific proposal only while nnU-Net runs; no module training authorized'}
q['paths'].update(scientific_root=str(S),nnUNet_results=str(S/'nnunet_results'),branch_code=str(B),freeze_directory=str(S),predictions=str(S/'dev_predictions'),evaluation=str(S/'analysis'))
q['prefix_contract']={'hard_exact':['source binding','fresh original initialization','Train input/target/ID/LR prefix'],'hard_intrinsic':['finite Loss/model/optimizer','finite unscaled gradient for successful update','one real parameter-changing optimizer call or protected unchanged AMP skip','AMP5consecutive/25total pause','persistent attempt/update/monitor counters'],'numeric_cross_GPU_comparison':'Loss/gradient/scaler/parameter-state differences recorded diagnostically; no bitwise identity requirement, no Dev-selected tolerance'}
q['amendment_from_v3']={'old_protocol':str(OLD/'protocol.json'),'old_protocol_sha256':sha(OLD/'protocol.json'),'prepared_repair_review':str(repair),'prepared_repair_review_sha256':sha(repair),'old_consumed_attempts':2,'old_successes':2,'old_gpu_process_seconds':review['worker_elapsed_seconds'],'new_attempt_cap':25000,'interrupted_plus_replacement_total_attempts_if_complete':25002,'old_trajectory_preserved':True,'no_auto_retry_after_this_replacement':True}
norm=(OLD/'SCIENTIFIC_PROTOCOL.md').read_text()
norm=norm.replace('nnU-Net100epochs科学运行 v3','nnU-Net100epochs科学运行 v4')
old='真实科学前30步按实际输入/Loss/LR/梯度/AMP与已验收工程前30有限步骤核对，不重放、不另加GPU预算。'
new='真实科学前30步精确核对Train输入/标签/ID/LR与原全新初始化；Loss/梯度/AMP/参数状态的跨GPU数值差异只作诊断，不要求逐位相同，不按Dev挑容差。有限性/真实更新/安全跳步/5及25暂停线及所有预算仍为硬规则。不重放、不另加GPU工程预算。'
assert old in norm;norm=norm.replace(old,new)
norm+='\n\n## 本轮重启授权与版本追溯\n用户明确要求先完成nnU-Net后台运行，接受已准备的前缀门槛修订。旧V3的2次成功更新和约40.12秒全部保留，不能清零；新轨迹从原seed2026随机初始化完整25000次尝试，这一对中断及替代轨迹合计最多25002次。旧协议/代码/账本/权重不覆盖。V4不再自动重试，Test封存，模块仅CPU方案研究。\n\n推理资源沿用CPU审查澄清：expected_tiles_by_image为原生尺寸的固定最大前向数，允许官方nonzero裁剪减少实际数量；固定28图每图一次，完整逆变换和形状/概率核验不变，不扩大任何数值上限。\n'
(S/'SCIENTIFIC_PROTOCOL.md').write_text(norm,encoding='utf-8');q['normative_text_sha256']=sha(S/'SCIENTIFIC_PROTOCOL.md')
with (S/'protocol.json').open('x',encoding='utf-8') as f:json.dump(q,f,ensure_ascii=False,indent=2);f.write('\n')
gate=B/'science100_gate.py';text=gate.read_text();assert text.count('TO_BE_FROZEN')==1;gate.write_text(text.replace('TO_BE_FROZEN',sha(S/'protocol.json')),encoding='utf-8')
backup=S/'management_before_restart_authorization';backup.mkdir();paths=[C/'STATUS.md',C/'项目全局状态与研究决策.md'];previous={p:p.read_bytes() for p in paths}
for p in paths:shutil.copy2(p,backup/p.name)
oldtext=previous[C/'STATUS.md'].decode();a='<!-- NNUNET_SCIENCE100_CURRENT -->';b='<!-- /NNUNET_SCIENCE100_CURRENT -->';lo=oldtext.index(a)+len(a);hi=oldtext.index(b);oldrecord=json.loads(oldtext[lo:hi]);assert oldrecord['state']=='FAILED_STOP' and oldrecord['charged_attempts']==2
(S/'prior_V3_authority_record.json').write_text(json.dumps(oldrecord,ensure_ascii=False,indent=2))
record={'phase':PHASE,'protocol_sha256':sha(S/'protocol.json'),'state':'CPU_PREPARATION','user_approved_science100':True,'GPU_released':False,'execution_started':False,'seed':2026,'epochs':100,'attempt_cap':25000,'monitor_cap':5000,'prior_V3_attempts_preserved':2,'stage_path':str(S),'executor_reviewer':'Sol directly implementing/reviewing','next':'CPU prefix amendment review and independent release, then one fresh100epochs job; old2 attempts retained; module research CPU-only; no extra GPU engineering'}
header='### 用户批准先完成nnU-Net：V4前缀门槛修订与fresh100epochs替代轨迹准备\n\n本轮“先把nnUnet跑完吧，让他先跑，我们再来研究模块方案”批准采用已准备修订并全新启动一次100epochs。旧V3保留2成功/40.12秒，新单条25000尝试，两条合计25002，未退款或隐去历史。输入/标签/ID/LR/初始模型/源绑定仍精确；跨GPU Loss/梯度/AMP/参数摘要微差仅作诊断，不按Dev定容差；有限性、真实更新、安全跳步及5/25暂停线不变。CPU审查和Sol独立release前GPU HOLD，复用原32工程与10CPU证据，不加GPU预演。新轨迹GPU0独占/仅final才Dev28/.5，Test/GPU1封存。模块方案可CPU并行研究，未授权模块训练。证据 '+str(S)+'。\n\n'
for p in paths:
 text=previous[p].decode()
 if p.name=='STATUS.md':text=text[:lo]+'\n'+json.dumps(record,ensure_ascii=False)+'\n'+text[hi:]
 first,_,rest=text.partition('\n');tmp=p.with_name(p.name+'.nnunet_v4_auth.pending')
 with tmp.open('x',encoding='utf-8') as f:f.write(first+'\n\n'+header+rest.lstrip('\n'));f.flush();os.fsync(f.fileno())
 assert p.read_bytes()==previous[p];os.replace(tmp,p)
print('V4_RESTART_AUTHORIZED_CPU_HOLD',sha(S/'protocol.json'),flush=True)
