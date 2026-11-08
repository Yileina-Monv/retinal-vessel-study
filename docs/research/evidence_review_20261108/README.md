# 实验结果与图册补齐：组长阅读入口

本次只归档已有实验和CPU诊断，不含PPT、演讲稿，不启动训练/推理，不访问Test，也不修改历史协议。

## 先看结论

- 正式T0/TW三seed：原出血邻域细骨架召回增量−0.600个百分点，原成功条件未通过。
- Plan B：77位置×3seed共231次配对，恢复0、失去2；11条件下27/77始终未命中。医学人工审核仍为0/105。
- B1/B2：全FOV细分支召回提高，但Precision降低、FP增加、Dice下降；不是原出血邻域正式主指标成功。
- 标准nnU-Net：共同2027/2028，Dice提高约2.02个百分点，补充邻域细分支覆盖提高约18.84个百分点，邻域FP密度约增加0.43个百分点；完整流程比较、重复Dev、保护局限保留。
- 多尺度模块没有可靠额外Dice收益；NL固定终点也未达到原预设成功标准。

## 已经上传，尚待审查合并

|PR|内容|
|---|---|
|[#19](https://github.com/Yileina-Monv/retinal-vessel-study/pull/19)|归档NL终点结果与评价覆盖复核：nnU-Net局部收益及误报代价|
|[#18](https://github.com/Yileina-Monv/retinal-vessel-study/pull/18)|研究提案：nnU-Net 出血邻域细血管辅助监督（N0 vs NL，待审）|
|[#16](https://github.com/Yileina-Monv/retinal-vessel-study/pull/16)|归档 D0 与 Train 配对检查：V1/V2 覆盖不足，停止进入训练|
|[#14](https://github.com/Yileina-Monv/retinal-vessel-study/pull/14)|发布 nnU-Net 消融结果：模块未获增益，保留标准深监督|
|[#12](https://github.com/Yileina-Monv/retinal-vessel-study/pull/12)|归档出血邻域细血管调研、局部监督原则与后续实验建议|
|[#10](https://github.com/Yileina-Monv/retinal-vessel-study/pull/10)|发布 nnU-Net 双种子复现成果：Dice 提升约 2pp，严格保护未通过|
|[#8](https://github.com/Yileina-Monv/retinal-vessel-study/pull/8)|整理 TW 两轮独立分析、证据修正与未采纳实验提案|
|[#7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7)|发布 MAPLES-DR 正式结果与原 U-Net 阶段探索报告（截至 2026-10-26）|
|[#4](https://github.com/Yileina-Monv/retinal-vessel-study/pull/4)|同步 MAPLES 本机验证与阶段二研究进展（截至 2026-10-19）|

以上成果不是缺失文件，不重复提交。main仍主要展示早期FIVES内容；这份索引连接后续待审成果，不表示这些PR已合并。

## 本次新增的详细证据

|阶段|补充材料|
|---|---|
|T0/TW正式实验|[详细文件与说明](../../../outputs/research_evidence_20261108/t0_tw_formal/README.md)|
|区域权重消融|[详细文件与说明](../../../outputs/research_evidence_20261108/region_weights/README.md)|
|血管定向2/1探索|[详细文件与说明](../../../outputs/research_evidence_20261108/class_2_1/README.md)|
|2000→4000步预算探索|[详细文件与说明](../../../outputs/research_evidence_20261108/budget_4000/README.md)|
|Plan B错误分析|[详细文件与说明](../../../outputs/research_evidence_20261108/plan_b/README.md)|
|B1/B2终点科研审查|[详细文件与说明](../../../outputs/research_evidence_20261108/b1_b2/README.md)|
|B1/B2新增误报分析|[详细文件与说明](../../../outputs/research_evidence_20261108/b1_b2_fp/README.md)|
|clDice与误报几何诊断|[详细文件与说明](../../../outputs/research_evidence_20261108/cldice_geometry/README.md)|
|Loss研究与既有概率诊断|[详细文件与说明](../../../outputs/research_evidence_20261108/loss_review/README.md)|
|A0/D/B1/B2概率与阈值诊断|[详细文件与说明](../../../outputs/research_evidence_20261108/threshold_diagnosis/README.md)|
|Train110裁块曝光审计|[详细文件与说明](../../../outputs/research_evidence_20261108/train_crop_coverage/README.md)|
|空间分层CPU候选检查|[详细文件与说明](../../../outputs/research_evidence_20261108/spatial_candidate/README.md)|
|D0原77点诊断|[详细文件与说明](../../../outputs/research_evidence_20261108/d0_diagnostic/README.md)|
|D0低置信度17点复核|[详细文件与说明](../../../outputs/research_evidence_20261108/d0_target17/README.md)|
|NL Train-only CPU检查|[详细文件与说明](../../../outputs/research_evidence_20261108/nl_train_cpu/README.md)|
|NL终点与补充评价审查|[详细文件与说明](../../../outputs/research_evidence_20261108/nl_terminal_audit/README.md)|

## 图册怎样看

[公开图册总入口](../../../outputs/research_evidence_20261108/ATLAS.md)：B1/B2覆盖28图、87视图，Plan B覆盖全部77位置与11条件，Train裁块与空间分层各110图。公开误差图没有RGB底图，不可据此判定医学真伪；所有自动标签相对统计保持原定义。

完整含RGB图册仍在A7，具体位置、案例分母及查看命令见图册入口。MESSIDOR原图再分发限制见[官方条款](https://www.adcis.net/en/third-party/messidor/)，MAPLES-DR标注来源见[官方文档](https://liv4d.github.io/MAPLES-DR/en/)。

## 局限与复核

原9图77点与补充18图174路径的评价目标、汇总方式不同；补充评价不重判旧结果。两个seed不等于56个独立病例；Dev反复使用。人工医学核验和独立Test验证没有被本次发布补齐。历史非主指标bootstrap来源追溯的局限继续保留，本次验证不声称重新生成所有历史区间。

[公开包复核说明](../../../outputs/research_evidence_20261108/README.md)。权重、原图、原始GT、概率缓存和完整运行档案留A7；公开CPU脚本与表格的可复核性不等于完整训练重现。
