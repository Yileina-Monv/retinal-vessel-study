# Loss研究与既有概率诊断

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/loss_function_review_20261024/`。

其中方法建议为当时研究记录，部分已被后续实验取代；不构成当前待执行任务。

## 文件

- [LOSS_RESEARCH_REPORT.md](LOSS_RESEARCH_REPORT.md)
- [sources_manifest.json](sources_manifest.json)
- [source_archive/diagnose_existing_probabilities.py.txt](source_archive/diagnose_existing_probabilities.py.txt)
- [diagnostic_spec.json](diagnostic_spec.json)
- [threshold_per_image.csv](threshold_per_image.csv)
- [input_manifest.json](input_manifest.json)
- [diagnostic_summary.json](diagnostic_summary.json)
- [artifact_manifest.json](artifact_manifest.json)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
