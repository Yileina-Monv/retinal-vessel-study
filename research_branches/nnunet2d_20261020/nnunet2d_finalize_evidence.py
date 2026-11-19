"""Create the P2 execution binding and concise handoff report after verification."""
from pathlib import Path
import json,hashlib,os,shutil
from datetime import datetime,timezone
ROOT=Path('${RETINA_PROJECT_ROOT}'); EXP=ROOT/'experiments/nnunet2d_branch_20261020'
ENG=EXP/'engineering'; BRANCH=ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=json.loads((ENG/'verification.json').read_text(encoding='utf-8'))
cpu=json.loads((ENG/'cpu/cpu_gate.json').read_text(encoding='utf-8'))
env=json.loads((ENG/'environment_lock.json').read_text(encoding='utf-8'))
assert v['status']=='PASS' and cpu['status']=='CPU_GATE_PASS' and v['optimizer_attempts']==100
assert v['gpu_window_seconds']<1800 and v['amp_skips']==0
code_paths=sorted(BRANCH.glob('*.py'))
binding={
 'task':'P2 approved nnU-Net 2D CPU preparation and capped GPU engineering only',
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'authorization_source':'A7 STATUS.md current task handoff, P2 current execution ticket (2026-10-20); no 100-epoch authorization',
 'feasibility_note':str(ROOT/'docs/nnunet2d_feasibility_20261020.md'),
 'feasibility_note_sha256':sha(ROOT/'docs/nnunet2d_feasibility_20261020.md'),
 'dataset_id':'Dataset501_MAPLES','train_manifest':str(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'),
 'train_manifest_sha256':sha(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'),
 'dev_reference_manifest_sha256':sha(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json'),
 'train_export_manifest_sha256':sha(EXP/'raw/train_export_manifest.csv'),
 'raw_dataset_json_sha256':sha(EXP/'raw/Dataset501_MAPLES/dataset.json'),
 'plans_sha256':sha(EXP/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'),
 'fingerprint_sha256':sha(EXP/'preprocessed/Dataset501_MAPLES/dataset_fingerprint.json'),
 'environment_lock_sha256':sha(ENG/'environment_lock.json'),
 'environment_freeze_sha256':sha(ENG/'environment_freeze.txt'),
 'environment_lock':env,
 'code_sha256s':{p.name:sha(p) for p in code_paths},
 'cpu_gate_sha256':sha(ENG/'cpu/cpu_gate.json'),
 'verification_sha256':sha(ENG/'verification.json'),
 'budget_ledger_sha256':sha(ENG/'budget_ledger.json'),
 'approved_caps':{'cumulative_gpu_process_window_seconds':1800,'optimizer_attempts':100},
 'actual_usage':{'optimizer_attempts':v['optimizer_attempts'],'successful_updates':v['successful_updates'],
   'gpu_window_seconds':v['gpu_window_seconds'],'scientific_training':False,
   'run_training_called':False,'dev_pixels_or_labels_read':False,'dev_model_calls':0,'official_test_accessed':False},
 'future_training_requires_separate_approval_and_fresh_random_initialization':True
}
bind_path=ENG/'execution_binding.json'
bind_path.write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
cp=v['native_train_size_case_coverage']; tr=v['throughput']; probe=v['gpu_validation_probe']
restore_all=all(x.get('equal') for x in v['restore_comparison'].values())
report=f'''# nnU-Net 2D P2 CPU准备与限额GPU工程验证

日期：2026-10-20。状态：工程验证完成，交 Sol 审查。仅限流程工程检查；没有启动100 epochs科学训练。

## 执行范围与数据边界

- 按 `${RETINA_PROJECT_ROOT}/docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json` 白名单转换和规划110张Train。原始尺寸计数为960×1440：58张，1488×2240：33张，1536×2304：19张；RGB TIFF无损导出为三通道RGB PNG，GT从mode 1保留0/1值导出，110/110图像与标签尺寸、源SHA和PNG往返均核对通过。
- Dev仅读取28个名单用于交集核验；没有读取Dev图像/标签、调用Dev模型或复制Test。Train/Dev名单交集0，Test未访问。nnU-Net标准背景监督范围与原FOV内T0不同，这是已批准的流程比较差异，不是单变量消融。
- 代码、环境、raw/preprocessed、规划、检查点和报告均留在A7/NAS旁支路径；未改主线源码、主环境、冻结数据/协议/旧权重，也未上传GitHub。

## CPU准备

- 建立新前缀 `${RETINA_PROJECT_ROOT}/envs/nnunet2d_v2`；pip check通过，未安装/升级包。环境以只读方式继承 `retina_a11` site-packages，锁定106条pip freeze及运行时Torch {env['torch_runtime_version']} / CUDA {env['torch_runtime_cuda_build']} / nnU-Net {env['versions']['nnunetv2']}。该前缀隔离路径和后续写入；底层依赖文件共享只读，这是环境隔离的已记录限制。
- nnU-Net数据完整性与fingerprint通过。仅Train110规划和2D预处理齐备（110 data、110 segmentation、110 properties）。使用 `ExperimentPlanner`、`nnUNetPlans_8GBtarget`、8GB规划目标；PlainConvUNet、RGB逐通道Z-score、单位面内spacing，patch 1024×1536、batch 2。旧默认planner给出推荐更新提示；本任务按已批标准PlainConvUNet方案保留。
- CPU门槛记录：`cpu/cpu_gate.json`，转换明细：`../raw/cpu_integrity.json` 与 `../raw/train_export_manifest.csv`，实际plans及fingerprint位于 `../preprocessed/Dataset501_MAPLES/`。

## GPU工程结果

- GPU0启动前核查空闲。四段工程执行分别为：50次标准Train dataloader更新；同一固定Train batch连续25次；从相同50步检查点独立运行12次并跨进程恢复13次。总计100/100 optimizer尝试、100次成功更新、0次AMP skip；累计GPU进程窗口 {v['gpu_window_seconds']:.1f}秒，低于1800秒硬上限。GPU1未使用。
- 第50步输入覆盖三个原生尺寸，实际采样唯一Train图像数 {v['unique_train_cases_sampled']}；尺寸分布：{cp}。输入为2×3×1024×1536，八级deep-supervision输出/目标空间形状一致，所有loss及裁剪前梯度范数有限。未发生OOM。
- 恢复对比：固定变换Train输入下，25次连续与12+13跨进程结果的网络权重、optimizer、AMP scaler、scheduler、RNG、epoch元数据逐项完全相等。该夹具来自真实Train batch。标准多进程增强使用`seeds=None`且队列不进入nnU-Net checkpoint；所以此恢复证明覆盖固定输入下模型训练状态，不证明原生随机增强队列逐批重放。
- 5个fold=all内部Train验证批次完成无更新前向，loss有限；模型与optimizer哈希不变，0次optimizer调用。中位验证耗时 {probe['median_validation_batch_seconds']:.3f}秒/批，Trainer配置为每epoch 50批；用于规划的100 epochs约5,000批、测得验证耗时约{probe['measured_validation_gpu_hours_estimate']:.3f} GPU小时。Train更新中位间隔 {tr['train50_update_interval_median_seconds']:.3f}秒，按25,000次标准更新估算约{tr['projected_25000_update_hours_update_only']:.2f}小时；加内部Train验证约{tr['projected_updates_plus_internal_validation_hours']:.2f}小时，尚未含检查点写入、启动、epoch账务和波动。此成本估计不授权科学训练。
- `nvidia-smi`执行中点采样曾显示GPU0占用10,275 MiB/98%，属瞬时样本而非峰值水位；验证前向测得peak allocated {probe['peak_memory_allocated_bytes']/1024**3:.2f} GiB、reserved {probe['peak_memory_reserved_bytes']/1024**3:.2f} GiB。训练反向峰值未被逐步仪表化，记录为测量局限。

## 关键文件

- 总核验：`verification.json`
- 累计硬预算账本：`budget_ledger.json`
- 绑定：`execution_binding.json`
- CPU门槛与Train-only转换/规划：`cpu/`、`../raw/`、`../preprocessed/`
- 每段optimizer调用/成功记录、检查点及恢复状态：`attempts/`；固定Train夹具：`fixtures/`
- 内部Train验证开销：`train_internal_val_probe.json`
- 新实现：`${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/research_branches/nnunet2d_20261020/`

## 未解决问题与停止点

未解决的工程限制是底层Python依赖来自只读共享环境，且反向峰值只保留了nvidia-smi点采样。nnU-Net标准随机多进程增强的队列状态不在恢复比较范围。所有结果均为工程检查，没有模型效果解释。

P2本次已完成，停止并交Sol审查。100 epochs / 25,000迭代、Dev推理评价、额外seed、配置或自定义loss均未获本次授权；任何后续科学训练需要单独批准并重新随机初始化，不能从工程检查点续训。P1仍优先推进，不由本旁支改变其授权或安排。
'''
(ENG/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'binding':str(bind_path),'binding_sha256':sha(bind_path),'report':str(ENG/'report.md'),
 'report_sha256':sha(ENG/'report.md'),'verification_status':v['status'],'restore_all_exact':restore_all,
 'gpu_seconds':v['gpu_window_seconds'],'attempts':v['optimizer_attempts'],'estimated_hours':tr['projected_updates_plus_internal_validation_hours']},indent=2))
