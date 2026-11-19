"""Implement approved fixed2027/2028 serial scope without touching any frozen prior experiment."""
from pathlib import Path
import json,hashlib,copy,os,shutil
ROOT=Path('${RETINA_PROJECT_ROOT}');CODE=ROOT/'code/retinal-vessel-study';BASE=ROOT/'experiments/nnunet2d_branch_20261020';B=Path(__file__).resolve().parent;GROUP=BASE/'replication_seed2027_2028_100epochs_v1_20261026';OLD_B=B.parent/'scientific_prefix_v4_20261025';OLD_S=BASE/'scientific_seed2026_100epochs_v4_prefixfix_20261025'
if GROUP.exists():raise RuntimeError('replication stage already exists; do not overwrite')
GROUP.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=(OLD_B/'science100_train.py').read_text().replace('from science100_gate import STAGE,CODE,GPU,protocol,validate,runtime_authorizer,sha,update_status','from replication_gate import STAGE,CODE,GPU,protocol,validate,runtime_authorizer,sha,update_status,SEED,DATA_BASE').replace('from prefix_contract import validate_prefix\n','').replace('from nnunet2d_scientific_runtime import set_all_seeds,install_seeded_worker_adapter','from seeded_runtime import set_all_seeds,install_seeded_worker_adapter').replace("os.environ.get('PYTHONHASHSEED')!='2026'","os.environ.get('PYTHONHASHSEED')!=str(SEED)").replace('base=STAGE.parent;','base=DATA_BASE;').replace('set_all_seeds(2026)','set_all_seeds(SEED)').replace('seeds=install_seeded_worker_adapter()','seeds=install_seeded_worker_adapter(SEED)')
a=source.index('    prior=json.loads(');z=source.index("    atomic(STAGE/'initialization.json'",a);source=source[:a]+source[z:]
source=source.replace("{'fresh_random_official':True","{'seed':SEED,'fresh_random_official':True")
a=source.index('    engrows=');z=source.index('    def event',a);source=source[:a]+source[z:]
source=source.replace("'seeds':[2026]","'seeds':[SEED]").replace("'seeds':[2027]","'seeds':[SEED+1]")
a=source.index("        if kind=='train' and counts['attempts']<=30:");z=source.index('        save=',a);source=source[:a]+source[z:]
source=source.replace("{'status':'TRAINING_COMPLETE'","{'seed':SEED,'status':'TRAINING_COMPLETE'")
(B/'replication_train.py').write_text(source)
source=(OLD_B/'science100_infer.py').read_text().replace('from science100_gate import','from replication_gate import');(B/'replication_infer.py').write_text(source)
source=(OLD_B/'science100_analysis.py').read_text().replace('from science100_gate import STAGE,CODE,protocol,sha','from replication_gate import STAGE,CODE,protocol,sha,SEED').replace("'model':'standard2D nnU-Net seed2026 fresh100epochs'","'model':f'standard2D nnU-Net seed{SEED} fresh continuous100epochs'")
(B/'replication_analysis.py').write_text(source)
norm='''# nnU-Net固定2027/2028连续100epochs复现协议v1

用户已选择固定2027和2028两条、GPU0串行，并以Implement the plan批准实施。Sol直接实施和审查，不新增模型/Loss/模块或额外GPU工程。两条按2027→2028依次运行，效果好坏均完整交付；技术异常暂停排程，不自动补步、清零或重试。每条新随机初始化，不用M0或任何旧实验权重；旧2026含1000步后增强队列重建，只作历史探索参照，不混称严格同执行协议三种子验证。

原2D PlainConvUNet/现有plans/数据集/标准CE+Dice深监督/所有原Loss参数、Train110/Dev28、fold=all均保持。Batch2、Patch1024×1536、SGD初始LR.01/momentum.99/Nesterov/weight decay3e−5、原Poly100 exponent.9、clip12、原生AMP及初始GradScaler不改。每seed独立100×250=25000训练尝试与100×50=5000 Train-only监控，总50000/10000。成功+安全AMP跳步=25000，不补步；连续5/累计25安全跳步停止。成功须真实有限未缩放梯度和一次实际参数更新；安全跳步须有限前向及模型/optimizer、GradScaler下降、无optimizer调用且模型及optimizer不变。致命/未知/来源变化/OOM停止，已消耗计数完整保存。

seed2027的主/Train增强/监控增强种子为2027/2027/2028；seed2028为2028/2028/2029，每池1worker。保留原producer的NumPy及Python播种语义，用真实Augmenter子类保持isinstance清理正常。只更改新入口，不改已安装包/历史源码。正常进度无墙钟截断，心跳只告警；GPU0物理UUID ${GPU_UUID}，独占共享锁，保留约10GB实际需求，不套用scSE8GiB上限；GPU1不纳入本范围。GPU忙时等待不驱逐不切卡；磁盘最低50GiB，不清理旧结果。CPU线程2/有限BLAS。每250尝试保存完整可用状态、30/200启动快照、20/40/60/80/100不可覆盖快照及原生final。异步队列不宣称可精确恢复，中断后保存状态并审查，不自动续训或重开。

CPU检查包含新种子映射/重复性/分池数/禁止额外池、原生on_train_end清理与stdout恢复、预算和CLI拒绝、种子对应A0来源和31指标接线，复用既有官方循环10CPU/AMP32工程及三尺度导出证据，不新增GPU预演。真实科学首段检查全部计入25000，不将seed27/28与seed26输入或数值强行匹配。

每条只用checkpoint_final.pth在100epochs完成后一次Dev28，固定.5、原native预处理/逆变换/tile.5/Gaussian，无TTA/集成/后处理/中途Dev或best选择。每case一次，native几何67总前向上限；两条最多56cases/134前向。各seed复用其原A0同seed固定2000终点28概率，无新增A0推理；CPU核验各自summary/配置/run binding/对应文件SHA及last与milestone实际状态，文件字节SHA可因序列化不同而不同。原GT/FOV/Omega/宽度/d1/有效图及9图77位置19NA不改。

普通Dice为主，每seed按同一28图配对NN−对应A0宏平均，PCG64 seed20261016/1000次/linear95%区间。两新seed均Dice均值正且各自区间下界>0才称两seed复现信号。完整保护仍为Precision/Recall/远处/粗血管/hardclDice均值不降、FP/FOV不升，且Precision/远处95%下界≥0、FP95%上界≤0；两条分别判读，不把区间跨零描述为已证实成本。细分支/原邻域等31指标、全部逐图分母/NA均报告。跨新两seed报告均值和ddof1样本SD(n=2)，不是56张独立图片、不用图像CI替代种子稳定性；旧2026及scSE单列为上下文。Dice复现与保护通过分开表述；工程完整可复算即交付，不以好成绩为完成条件。结果决定是否申请后续研究，不自动扩seed、重新跑2026、加模块、改阈值或访问Test60。

资料在现有nnU-Net目录下隔离，各seed各自绑定/预算/权重/日志/28概率/报告；共用原STATUS和研究决定，不建第二进度体系。完成或停止后台自动CPU核对并回写，明确待Sol科研复盘。旧正式T0/TW负面结论与所有失败/费用、人工0/26保持。
'''
(GROUP/'SCIENTIFIC_PROTOCOL.md').write_text(norm)
original=json.loads((OLD_S/'protocol.json').read_text());spec={}
for seed in [2027,2028]:
    s=GROUP/'runs'/('seed_'+str(seed));(s/'cpu_reference').mkdir(parents=True)
    q=copy.deepcopy(original);q.update(protocol_id='NNUNET_REPLICATION_2027_2028_V1_SEED_'+str(seed),version=1,status='USER_APPROVED_CPU_AND_INDEPENDENT_RELEASE_REQUIRED')
    q.pop('amendment_from_v3',None);q.pop('prefix_contract',None)
    q['scope'].update(seed=seed,new_scientific_trajectories=1,replay_exception='new fixed replication seed; no replay of old trajectories or GPU engineering')
    q['authority']={'user_approval':'Implement the plan. Fixed2027/2028, GPU0 serial.','classification':'prospectively fixed two-seed repeated-Dev workflow replication, not a single-variable ablation','old2026':'separate exploratory context with interrupted/rebuilt queues'}
    q['paths'].update(scientific_root=str(s),nnUNet_results=str(s/'nnunet_results'),branch_code=str(B),freeze_directory=str(GROUP),predictions=str(s/'dev_predictions'),evaluation=str(s/'analysis'))
    q['references'].update(accepted_A0_fixture=str(s/'cpu_reference/A0_baseline.json'),A0_reuse_binding=str(s/'cpu_reference/A0_reuse_binding.json'))
    q['references'].pop('accepted_A0_fixture_sha256',None);q['references']['seed_specific_A0_CPU_binding_required']=True
    q['worker_seeds']={'main':seed,'train':seed,'internal_monitor':seed+1,'workers_each':1};q['normative_text_sha256']=sha(GROUP/'SCIENTIFIC_PROTOCOL.md')
    (s/'protocol.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');spec[str(seed)]=sha(s/'protocol.json')
group={'phase':'NNUNET_REPLICATION_2027_2028_V1','seeds':[2027,2028],'serial_order':[2027,2028],'device':'${GPU_UUID}','total_train_attempt_cap':50000,'total_monitor_cap':10000,'official_Test_allowed':False,'GPU1_allowed':False,'seed_protocol_sha256':spec,'normative_sha256':sha(GROUP/'SCIENTIFIC_PROTOCOL.md'),'user_approval':'Implement the plan.','old_source_binding':str(OLD_S/'source_binding.json'),'accepted_observer_branch':str(B.parent/'scientific_resume_v5_20261025'),'original_2026_result':str(BASE/'scientific_seed2026_100epochs_v5_resume1000_20261025/analysis/endpoint_100epochs.json'),'original_A0_GT_validity_fixture':original['references']['accepted_A0_fixture'],'no_new_GPU_engineering':True,'estimated_GPU_hours':24,'no_normal_progress_wall_cap':True}
(GROUP/'protocol.json').write_text(json.dumps(group,ensure_ascii=False,indent=2)+'\n');p=B/'replication_gate.py';raw=p.read_text();assert 'GROUP_PROTOCOL_HASH_PENDING' in raw;p.write_text(raw.replace('GROUP_PROTOCOL_HASH_PENDING',sha(GROUP/'protocol.json')))
backup=GROUP/'management_before_authorization';backup.mkdir();header='### 用户批准nnU-Net固定2027/2028复现：GPU0串行，CPU准备前不启动\n\n两新seed各完整连续100epochs/25000尝试/5000 Train监控，依次2027→2028，总50000/10000，约24 GPU小时，不按第一条效果决定是否执行第二条。配置/标准Loss/plans/Train110/Dev28/foldall保持，仅新seed变化；分别复用对应A0 seed2027/2028已有终点，不重训或推理A0。旧2026队列重建探索单列，不包装同执行协议正式三seed。修复种子适配器类型清理，CPU/来源/基线接线通过后独立release，不加GPU预演；GPU1不纳入、Test封存、旧结论和费用保留。证据 '+str(GROUP)+'。\n\n'
record={'phase':group['phase'],'protocol_sha256':sha(GROUP/'protocol.json'),'state':'CPU_PREPARATION','user_approved_seeds':[2027,2028],'GPU_released':False,'active_seed':None,'stage_path':str(GROUP),'total_attempt_cap':50000,'total_monitor_cap':10000,'executor_reviewer':'Sol directly implementing/reviewing','runs':{str(seed):{'state':'CPU_HOLD','stage_path':str(GROUP/'runs'/('seed_'+str(seed))),'attempt_cap':25000,'monitor_cap':5000} for seed in [2027,2028]},'next':'CPU seed/cleanup/ledger and per-seedA0 evaluation binding; independent Sol release then serial2027/2028 regardless scientific outcome; no extra experiments/Test'}
for p in [CODE/'STATUS.md',CODE/'项目全局状态与研究决策.md']:
    for _ in range(5):
        old=p.read_bytes();(backup/p.name).write_bytes(old);text=old.decode()
        if p.name=='STATUS.md':
            a='<!-- NNUNET_SCIENCE100_CURRENT -->';z='<!-- /NNUNET_SCIENCE100_CURRENT -->';lo=text.index(a)+len(a);hi=text.index(z);text=text[:lo]+'\n'+json.dumps(record,ensure_ascii=False)+'\n'+text[hi:]
        first,_,rest=text.partition('\n');tmp=p.with_name(p.name+'.nnrep_auth.'+str(os.getpid())+'.pending');tmp.write_text(first+'\n\n'+header+rest.lstrip('\n'))
        if p.read_bytes()!=old:tmp.unlink();continue
        os.replace(tmp,p);break
    else:raise RuntimeError('concurrent record changes')
print('NN_REPLICATION_FROZEN_CPU_HOLD',sha(GROUP/'protocol.json'),flush=True)
