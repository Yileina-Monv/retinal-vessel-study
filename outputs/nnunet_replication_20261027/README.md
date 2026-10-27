# nnU-Net 100 epochs 可复核成果包

先读[中文科研复盘](../../docs/research/nnunet_replication_review_20261027.md)。关联 [Issue #9](https://github.com/Yileina-Monv/retinal-vessel-study/issues/9)。仅包含数值与统计 SVG，不包含患者像素、标签、模型、概率数组或完整日志。

|文件|内容|
|---|---|
|[配置](training_config_summary.csv)、[设计与限制](protocol_design_summary.json)|6 行方法/种子配置，实际计数、版本、初始化、成本口径和完整流程差异|
|[逐尝试Loss](training_loss_steps.csv)、[逐epoch汇总](training_loss_epochs.csv)、[曲线](training_loss_curve.svg)|75000 次尝试和300个epoch；保留 AMP 跳步，监控只使用Train|
|[逐图指标](fixed_endpoint_per_image_metrics.csv)、[绝对汇总](absolute_metric_summary.csv)|5208 行含288 NA；186个方法/种子/指标汇总|
|[逐图差值](paired_image_deltas.csv)、[配对区间](paired_comparison_summary.csv)、[Dice图](dice_paired_intervals.svg)|2460有效配对、93组原图像配对区间|
|[两个新增种子汇总](seed_replication_summary.csv)|仅2027/2028，n=2、ddof=1；不混入2026|
|[bootstrap索引](bootstrap_indices_by_n.json)|n=9/18/28各1000次；配对图片顺序由逐图差值pair_index指定|
|[来源核验](source_audit.json)、[文件清单](export_manifest.json)、[验证结果](verification.json)|来源哈希、私有原件CPU核验范围、公开文件哈希和可重算结果|

## 字段与复算

联合键为 `cohort, seed, method, image, metric`；`historical_interrupted` 只用于2026，`prospective_replication` 用于2027/2028。A0 是对应 seed 的原2000步终点，不是统一复用2026。`pair_index` 必须保持原有效名单顺序。比例指标使用0–1，报告中的百分点是差值×100；长度等非比例指标保留原单位。

空 `value` 表示原 NA，`na_reason` 保留源原因，禁止补零。原记录有 numerator/denominator 时直接保留；`precomputed_scalar` 表示源记录只提供宏平均或结构分数，空分母不代表丢失值。各指标有效图数为28、18或9，不能替代病灶邻域77个评价位置的计数。

Loss 的 epoch 为1–100，iteration为0–249；cumulative_successful_updates不把AMP_SKIP算作更新。逐epoch同时给出全部尝试和仅成功更新的训练Loss均值。Train-only监控不是Dev验证集；原生负Dice项允许总Loss为负。2026时间仅为V5接续进程，不含原1000次和历史失败费用；缺少的历史推理时间留空，A0总训练费用本次未重新审计。

在仓库根目录运行 `python -B scripts/verify_nnunet_results_export.py`，仅需Python标准库。文件清单覆盖本包及报告/脚本/本次README与STATUS入口，排除清单自身和由验证程序生成的verification.json，避免自引用。

本包证明公开数表可复算，不能独立重跑全部训练或复原医学像素指标；原始资产不公开。私有项目相对路径只是来源定位，不是可下载链接。真实人工医学审核未完成、Test未访问，完整失败和历史结论保留。
