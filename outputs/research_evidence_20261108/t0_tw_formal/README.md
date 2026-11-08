# T0/TW正式实验

这是已有实验/诊断的补充归档，不是本次新增实验。源阶段：`experiments/t0_tw_core_20261016/analysis/`。

## 文件

- [source_archive/analyze.py.txt](source_archive/analyze.py.txt)
- [completion_verification.json](completion_verification.json)
- [fixed_0p5_per_image.csv](fixed_0p5_per_image.csv)
- [paired_seed_macro.csv](paired_seed_macro.csv)
- [paired_per_image.csv](paired_per_image.csv)
- [bootstrap_indices_estimates.csv](bootstrap_indices_estimates.csv)
- [same_fp_full_grid.csv](same_fp_full_grid.csv)
- [same_fp_per_image.csv](same_fp_per_image.csv)
- [same_fp_paired_per_image.csv](same_fp_paired_per_image.csv)
- [analysis_summary.json](analysis_summary.json)
- [analysis_source_hashes.json](analysis_source_hashes.json)
- [analysis_config.json](analysis_config.json)
- [analysis_validation.json](analysis_validation.json)
- [same_fp_seed_macro.csv](same_fp_seed_macro.csv)
- [z0_comparison_seed_macro.csv](z0_comparison_seed_macro.csv)
- [z0_comparison_summary.csv](z0_comparison_summary.csv)

## 阅读与复核边界

数值表保留原分母、NA和全部病例。报告为历史文本，不替代最新结论；AI视觉观察不等于医学人工核验。`source_archive/*.py.txt`是供审查的历史CPU脚本，路径已泛化，不自动运行；依赖A7私有输入的部分不声称公开克隆即可复现。完整原件SHA与公开副本SHA分别见上级source_manifest.json。
