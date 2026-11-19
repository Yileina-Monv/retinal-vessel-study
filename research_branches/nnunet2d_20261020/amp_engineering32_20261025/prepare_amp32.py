"""Record current bounded authorization and freeze only the engineering scope."""
from pathlib import Path
import json,hashlib,shutil,os,datetime
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';B=C/'research_branches/nnunet2d_20261020/amp_engineering32_20261025';S=R/'experiments/nnunet2d_branch_20261020/amp_engineering32_20261025';D=R/'docs/nnunet2d_seed2026_100epochs_20261020/amp_recovery_revision_20261021'
if S.exists():raise RuntimeError('new phase already exists; inspect rather than overwrite')
S.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proposal=D/'next_complete_run_proposal_20261025.json';q=json.loads(proposal.read_text());train=json.loads((R/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json').read_text())['rows'];ids=[x['filename'] for x in train]
assert len(ids)==len(set(ids))==110
p={'phase':'NNUNET_AMP_ENGINEERING32_20261025','user_approval':'可以','approved_scope':'CPU wiring plus up to32 engineering attempts/5 Train monitor batches/900seconds onGPU0; no100epochs','proposal_path':str(proposal),'proposal_sha256':sha(proposal),'train_attempt_cap':32,'Train_monitor_cap':5,'wall_cap_seconds':900,'injected_overflow_attempt':31,'expected_success':31,'expected_skips':1,'skip_limits':{'consecutive':5,'total':25},'science_authorized':False,'GPU1_allowed':False,'Dev_allowed':False,'Test_allowed':False,'train_ids':ids,'preserve_legacy_trajectory':True,'native_queue_resume_supported':False,'production_changes':'branch-local observer only; original installed train_step/recipe unchanged','engineering_schedule':{'epochs':1,'attempts_per_epoch':32,'monitor_per_epoch':5},'scientific_future_schedule_not_authorized':{'epochs':100,'attempts_per_epoch':250,'monitor_per_epoch':50},'plans_sha256':'26507b98404e6f6855708567d224675805f3d2fd7680df8abd8caf74c7eb1fe9'}
with (S/'protocol.json').open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2);f.write('\n')
gate=B/'amp32_gate.py';text=gate.read_text();assert text.count('TO_BE_FROZEN')==1;gate.write_text(text.replace('TO_BE_FROZEN',sha(S/'protocol.json')),encoding='utf-8')
backup=S/'management_before_authorization';backup.mkdir();paths=[C/'STATUS.md',C/'项目全局状态与研究决策.md'];old={p:p.read_bytes() for p in paths}
for f in paths:shutil.copy2(f,backup/f.name)
record={'phase':p['phase'],'protocol_sha256':sha(S/'protocol.json'),'state':'CPU_PREPARATION','user_approved_engineering':True,'GPU_released':False,'scientific100epochs_authorized':False,'attempt_cap':32,'monitor_cap':5,'wall_seconds':900,'stage_path':str(S),'executor_reviewer':'Sol directly implementing and reviewing','next':'CPU wiring review, independent engineering release, exactly bounded GPU0 run, then stop to review; no scientific launch'}
note='### nnU-Net AMP接线与32尝试工程已获用户批准，CPU审查前GPU HOLD\n\n本轮“可以”仅批准此前明确的CPU接线及最多32尝试/5 Train监控/15分钟GPU0工程。固定30普通+1工程注入受控溢出+1正常恢复，预期31成功/1跳步；验证原官方步骤/源绑定/真实参数更新/持久计数/异常停止。仅新分支入口与证据，不改旧协议、代码、结果或数据；旧16来源与5修订文件保护。阶段独立限额，不退款/自动重启/加步，100epochs仍未批准，Dev/Test推理0，GPU1禁用。执行与实际结果见 '+str(S)+'。\n\n'
for f in paths:
 text=old[f].decode();first,_,rest=text.partition('\n');extra=note
 if f.name=='STATUS.md':extra+='<!-- NNUNET_AMP32_CURRENT -->\n'+json.dumps(record,ensure_ascii=False)+'\n<!-- /NNUNET_AMP32_CURRENT -->\n\n'
 value=first+'\n\n'+extra+rest.lstrip('\n');tmp=f.with_name(f.name+'.amp32_auth.pending')
 with tmp.open('x',encoding='utf-8') as out:out.write(value);out.flush();os.fsync(out.fileno())
 assert f.read_bytes()==old[f];os.replace(tmp,f)
print('ENGINEERING_SCOPE_FROZEN',sha(S/'protocol.json'),flush=True)
