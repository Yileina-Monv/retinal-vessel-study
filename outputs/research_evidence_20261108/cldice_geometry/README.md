# clDice与误报几何诊断

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/literature_mechanism_review_20261023/`。

其中方法建议为当时研究记录，部分已被后续实验取代；不构成当前待执行任务。

## 文件

- [source_archive/diagnose_cldice.py.txt](source_archive/diagnose_cldice.py.txt)
- [diagnostic_spec.json](diagnostic_spec.json)
- [cldice_per_image.csv](cldice_per_image.csv)
- [FP_geometric_partition.csv](FP_geometric_partition.csv)
- [common_hit_local_margin.csv](common_hit_local_margin.csv)
- [dice_count_decomposition.csv](dice_count_decomposition.csv)
- [cldice_summary.json](cldice_summary.json)
- [input_manifest.json](input_manifest.json)
- [diagnostic_validation.json](diagnostic_validation.json)
- [report.md](report.md)
- [source_snapshot_manifest.json](source_snapshot_manifest.json)
- [final_review.json](final_review.json)
- [source_archive/focused_geometry_v1.py.txt](source_archive/focused_geometry_v1.py.txt)
- [focused_geometry_spec_v1.json](focused_geometry_spec_v1.json)
- [focused_geometry_per_image_v1.csv](focused_geometry_per_image_v1.csv)
- [focused_geometry_summary_v1.json](focused_geometry_summary_v1.json)
- [source_archive/envelope_support_v1.py.txt](source_archive/envelope_support_v1.py.txt)
- [envelope_support_per_image_v1.csv](envelope_support_per_image_v1.csv)
- [envelope_support_summary_v1.json](envelope_support_summary_v1.json)
- [NEXT_METHOD_PROPOSAL_v1.md](NEXT_METHOD_PROPOSAL_v1.md)
- [NEXT_METHOD_REVIEW_RECEIPT_v1.json](NEXT_METHOD_REVIEW_RECEIPT_v1.json)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
