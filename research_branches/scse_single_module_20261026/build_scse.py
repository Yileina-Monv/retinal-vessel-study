"""Create one isolated approved module experiment, deriving unchanged existing mechanisms."""
from pathlib import Path
import json,hashlib,ast,shutil,os
R=Path('${RETINA_PROJECT_ROOT}');C=R/'code/retinal-vessel-study';BASE=R/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1';S=BASE/'scse_single_module_2000_v1_20261026';B=Path(__file__).resolve().parent
if S.exists():raise RuntimeError('stage exists; do not overwrite')
S.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();a0=R/'experiments/t0_tw_core_20261016/runs/seed_2026/T0';cfg=json.loads((a0/'config.json').read_text());oldq=json.loads((BASE/'focal_dice_scientific_2000_v1_20261024/protocol.json').read_text())
core=(C/'retinal_m0/maples_stratified_science_core.py').read_text().replace('SPATIAL_STRATIFIED_SCIENTIFIC_2000_V1','SCSE_SINGLE_MODULE_2000_V1').replace("TRACK_CAPS = {'SPATIAL': 2000}","TRACK_CAPS = {'SCSE': 2000, 'CONT': 16, 'SPLIT': 16}").replace("'method': 'SPATIAL'","'method': 'SCSE'").replace("saved.get('track')!='SPATIAL'","saved.get('track') not in TRACK_CAPS")
core=core.replace("track_caps or TRACK_CAPS","track_caps or {'SCSE':2000}")
core=core.replace("b.get('charged_events')!=expected_step or b.get('verified_successes')!=expected_step","b['tracks'][saved['track']]['charged_events']!=expected_step or b['tracks'][saved['track']]['verified_successes']!=expected_step")
core=core.replace("b.get('forward_calls')!=expected_step or b.get('gradient_calls')!=expected_step or b.get('optimizer_call_starts')!=expected_step","b.get('forward_calls')!=b['charged_events'] or b.get('gradient_calls')!=b['charged_events'] or b.get('optimizer_call_starts')!=b['charged_events']")
needle='                total=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)'
assert core.count(needle)==1
core=core.replace(needle,"                module_grads={n:float(p.grad.detach().float().norm().cpu()) if p.grad is not None else None for n,p in model.named_parameters() if n.startswith('scse.')}\n"+needle)
core=core.replace("'preclip_total_L2':float(total)","'module_gradient_L2':module_grads,'preclip_total_L2':float(total)")
(B/'scse_core.py').write_text(core)
entry=(C/'retinal_m0/maples_focal_dice_science.py').read_text();tree=ast.parse(entry);needed=['_load_arrays','hard_scalar_details','validate_probability_records','reserve_evaluation_forward','evaluate_checkpoint']
parts=[ast.get_source_segment(entry,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed];assert len(parts)==5
ev='\n\n'.join(parts).replace("'D'","'SCSE'").replace("'D:'","'SCSE:'").replace('from retinal_m0.maples_focal_dice_science_core import verify_checkpoint','from scse_core import verify_checkpoint')
header='from pathlib import Path\nimport os,json\nfrom scse_gate import STAGE,CODE,ROOT,OUT,protocol as check_protocol,sha,validate_release,runtime_authorizer\nfrom retinal_m0.maples_g1_safety import read_json,atomic_json,GateError\n'
(B/'scse_evaluation.py').write_text(header+'\n'+ev+'\n')
nnsource=C/'research_branches/nnunet2d_20261020/scientific_resume_v5_20261025/science100_analysis.py';analysis=nnsource.read_text().replace('from science100_gate import STAGE,CODE,protocol,sha','from scse_gate import STAGE,CODE,OUT,protocol,sha').replace("STAGE/'analysis'","OUT/'analysis'").replace("STAGE/'dev_predictions'","OUT/'evaluation/step_002000/probabilities'").replace('NNUNET','SCSE').replace('endpoint_100epochs.json','endpoint_2000.json').replace("'standard2D nnU-Net seed2026 total100epochs; V4 first1000 + V5 remaining24000 with disclosed rebuilt augmentation queues'","'original M0 U-Net + one final-decoder sum-scSE; seed2026 fixed2000'").replace("'checkpoint_final.pth'","'step_2000.pt'").replace('whole-pipeline comparison','one module comparison; shared original M0 weights').replace('nnU-Net固定100epochs终点','U-Net单scSE固定2000步终点').replace('流程比较；固定final及0.5','单模块探索；固定2000及0.5')
analysis='\n'.join(line for line in analysis.splitlines() if "lines+=['','接续局限：" not in line)+'\n';(B/'scse_analysis.py').write_text(analysis)
norm='''# 单模块scSE seed2026固定2000步探索：用户批准版v1

用户“可以，跑，不要等我批准了，我要睡觉了”批准CPU实现/检查、GPU1最多32次真实参数更新的工程验证，以及通过后自动从原M0全新启动唯一seed2026、2000成功更新的单模块探索，无需再次确认。Sol直接实施与审查。本次明确允许物理GPU1 UUID ${GPU_UUID}，仅用于此任务；取代历史提案中本任务的GPU1禁用条款，不修改GPU0上的nnU-Net授权、源码或运行。

原UNet不改：五层32/64/128/256/512，GroupNorm8。只在最后32通道解码层和原1×1预测头之间增加一个sum-scSE，压缩比2、有bias，总新增1105参数。c=sigmoid(W2 ReLU(W1 GAP(F)))，s=sigmoid(Conv1×1(F))；冻结实现F+(c−.5)F+(s−.5)F，代数等价论文cF+sF。W2及空间头及其bias为零，W1固定隔离CPU随机流20261026，保持M0初始功能及原主RNG。非作者当前max融合，不是全网络scSE复现。首步W1梯度零可预期，后续必须检查有效梯度。

保留原M0、Train110/Dev28、原T0全部2000批/8000输入、随机原裁块与增强、Batch4/512Crop、原归一化、AdamW LR1e-4/WD1e-4/betas(.9,.999)/eps1e-8/foreachFalse、无scheduler、clip1、AMP float16初始1024。Loss严格复用原weighted_masked_loss(z,y,f,f,epsilon1e-6)，不加骨架、Focal或其他Loss。不用工程权重续科学；新参数纳入同一个AdamW与全局clip。原A0不重训，跨GPU细小浮点差异不得作为效果或逐位拒绝依据；同GPU工程分段恢复须精确。

CPU：公式/梯度、M0共用键与权重、RNG隔离、初始恒等/旁路、真实参数梯度、首步零梯度解释、空GT/局部FOV、保存/恢复、计数/越额拒绝、原评价接线。原8000输入已经过既有CPU实测，本次复用并在每次真实更新前精确核对，不重复全套旧审计。全Train/Dev允许来源SHA重新核对，官方Test仅名单元数据，不读内容。

GPU工程：6次零更新前向（同一原始Train批的A0/旁路/初始scSE，各FP32和AMP一次），同精度初始输出应一致；不要求FP32=AMP。最多32更新：连续16、独立8+新进程恢复8，共用不可清零32账本；总训练前向32、backward32，另6诊断前向，无Dev推理。检查真实未缩放梯度、有限性、真实参数改变、新门控梯度及32中至少后续W1学习、完整模型/optimizer/scaler/主RNG/loader/采样游标恢复。工程最多15分钟GPU子进程累计，不自动重试。通过后专用科学release才启动2000。

科学：唯一新M0 seed2026，2000成功/占额2000，失败/未知/AMP跳步停止且不清零、不补步。每次更新完整last，每200固定checkpoint和Dev28评价，共10次/1900前向/7450tiles，FP32原512/256 Gaussian tiling与batch4；训练RNG、模型和模式评价后恢复。唯一2000终点/阈值.5，不用中途挑选模型、阈值、模块位置或压缩比。正常进度无墙钟硬停止，异常停止不自动重启；心跳只报告，不能因NAS mtime误判杀进程。

普通Dice唯一主指标；固定Dev28配对图像宏平均、1000次PCG64(20261016) bootstrap，mean>0且95%下界>0才称Dev探索增益信号。Precision/Recall/FPFOV/远处/粗血管/hard-clDice完整保护与原严格Precision/远处/FP区间规则保留，原病灶9图77位置/19NA和细分支路径、全部逐图分母/NA保留。Dice增益和保护通过分开报告，不满足则负面/不确定；即使通过仍单seed反复使用Dev、非独立验证，不自动扩大seed或实验。

GPU1独立共享锁，不持GPU0锁、不迁移nnU-Net；只允许新GPU1直接子进程、UUID绑定、8GiB分配器上限、单CPU线程有限BLAS、空闲启动/不得驱逐他人，别人占用时停止相关新任务。共享磁盘最低20GiB，完整日志/检查点/数据/源绑定与异常证据。完成后CPU复核全部28概率和终点指标、更新同一STATUS/研究决定，明确自动统计待Sol科研复盘，不冒称人工医学审核。原正式负面结论及人工0/26不变，Test60封存，禁止其他Loss/模块/seed/预算扩展。
'''
(S/'SCIENTIFIC_PROTOCOL.md').write_text(norm)
refs={'accepted_A0_fixture':str(BASE/'spatial_stratified_scientific_2000_v1_20261025/cpu_preparation/A0_endpoint_fixture/analysis/endpoint_2000.json'),'science_reference_manifest':str(BASE/'scientific_b1_b2_2000_v1_20261022/references/dev/manifest.json'),'A0_reuse_binding':str(BASE/'scientific_b1_b2_2000_v1_20261022/a0_reuse_binding.json')}
q={'phase':'SCSE_SINGLE_MODULE_2000_V1','seed':2026,'updates':2000,'GPU1_UUID':'${GPU_UUID}','user_approval':'可以，跑，不要等我批准了，我要睡觉了','normative_text_sha256':sha(S/'SCIENTIFIC_PROTOCOL.md'),'module':{'position':'last decoder32 before original head','fusion':'centered equivalent paper sum','reduction':2,'new_parameters':1105,'initialization':'zero gate output heads; isolated CPU default_generator20261026'},'baseline_config':cfg,'A0_path':str(a0),'A0_config_sha256':sha(a0/'config.json'),'expected_pairs_path':str(a0/'pairing.jsonl'),'expected_pairs_sha256':sha(a0/'pairing.jsonl'),'references':refs,'reference':{'science_reference_manifest':refs['science_reference_manifest']},'runtime':{k:oldq['runtime'][k] for k in ['evaluation_forward_cap','evaluation_tile_cap','evaluation_image_profile']},'engineering':{'updates':32,'segments':[['CONT',0,16],['SPLIT',0,8],['SPLIT',8,16]],'diagnostic_forward_cap':6,'train_forward_cap':32,'gradient_cap':32,'GPU_child_cumulative_wall_seconds':900},'science':{'updates':2000,'train_forward_cap':2000,'gradient_cap':2000,'evaluation_steps':list(range(200,2001,200)),'threshold':.5,'no_normal_progress_wall_cap':True},'Test_allowed':False,'GPU0_allowed':False,'old_negative_results_preserved':True,'module_proposal_path':str(R/'docs/unet_single_module_direction_review_20261025.md')}
(S/'protocol.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
g=B/'scse_gate.py';raw=g.read_text();assert 'PROTOCOL_HASH_PENDING' in raw;g.write_text(raw.replace('PROTOCOL_HASH_PENDING',sha(S/'protocol.json')))
backup=S/'management_before_authorization';backup.mkdir();header='### 用户批准GPU1单scSE受控探索；GPU0 nnU-Net继续\n\n本轮明确批准CPU准备、最多32更新工程及通过后自动从原M0启动唯一seed2026/2000步，无需再次确认。仅原U-Net末端一个sum-scSE（1105参数、恒等初始化），BCE+Dice及原采样增强/训练设置不变；固定2000/.5普通Dice及全部保护。GPU1仅本任务可用，当前空闲不等于自动改用其他实验；GPU0 nnU-Net不改、不抢锁。CPU/工程验证与独立release前对应GPU工作HOLD；历史S3/医学0/26/负面结论及Test封存保持。证据 '+str(S)+'。\n\n'
record={'phase':q['phase'],'state':'CPU_PREPARATION','GPU_released':False,'user_approved_CPU':True,'user_approved_engineering32':True,'user_approved_science2000_conditional_on_engineering':True,'GPU1_UUID':q['GPU1_UUID'],'protocol_sha256':sha(S/'protocol.json'),'stage_path':str(S),'executor_reviewer':'Sol directly implementing/reviewing','next':'CPU and bound32 engineering; if accepted release one freshM0 SCSE2000; no extra user approval required within this scope','official_Test_allowed':False}
for p in [C/'STATUS.md',C/'项目全局状态与研究决策.md']:
    for _ in range(5):
        old=p.read_bytes();(backup/p.name).write_bytes(old);text=old.decode();first,_,rest=text.partition('\n');new=first+'\n\n'+header+rest.lstrip('\n')
        if p.name=='STATUS.md':new+='\n<!-- SCSE_CURRENT -->\n'+json.dumps(record,ensure_ascii=False)+'\n<!-- /SCSE_CURRENT -->\n'
        tmp=p.with_name(p.name+'.scse_auth.'+str(os.getpid())+'.pending');tmp.write_text(new)
        if p.read_bytes()!=old:tmp.unlink();continue
        os.replace(tmp,p);break
    else:raise RuntimeError('concurrent management edits')
print('SCSE_APPROVAL_FROZEN_CPU_HOLD',sha(S/'protocol.json'),flush=True)
