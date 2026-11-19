# 研究方案、实际代码与结果：组长阅读入口

采集日期：2026-11-19。A7是实际运行记录来源；本页把已有工作连起来，不提出新实验，也不改变历史结论。材料在不同待审PR中，尚未全部进入main。

## 建议阅读顺序

1. 先看下表，确认每阶段研究的问题及结论。
2. 查看[详细实验数据与图册（PR #20）](https://github.com/Yileina-Monv/retinal-vessel-study/blob/62c55b9deba6a03483d77265ac49d32f2d88d244/docs/research/evidence_review_20261108/README.md)。该固定版本包含逐图结果、负结果、分母、NA和无RGB图册；不重复复制。
3. 在本次新增的[协议档案](a7_protocol_archive_20261119/README.md)查看原方案、门槛调整与历史决定，再按下表找实际代码。
4. 阅读[复核方法与运行限制](a7_protocol_archive_20261119/REPRODUCIBILITY.md)。公开副本支持审查，不能直接视为已授权的完整训练镜像。

## 研究问题—对照—结果—代价—结论—证据

|阶段与问题|对照及评价|结果与代价|结论|代码入口|结果证据|
|---|---|---|---|---|---|
|预训练模型迁移到MAPLES是否有帮助|Z0直接应用M0 vs T0普通微调；同Dev28|Dice 75.892%→83.593%；FP/FOV 1.163%→2.025%|适应目标数据有收益，也有误报代价；FIVES上的分数不与MAPLES直接排名|[普通微调](../../retinal_m0/maples_t0.py)、[评价](../../retinal_m0/maples_evaluate.py)|[PR #7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7)|
|出血区域BCE加权是否有额外帮助|T0/TW三seed、2000成功更新；原9图77点|主指标TW−T0为−0.600pp；精确率/误报保护通过，主效应/远处召回保护未通过|当前冻结配置未达成功条件；不推广到所有加权方法|[核心损失](../../retinal_m0/maples_core.py)、[训练](../../retinal_m0/maples_core_experiment.py)|[PR #7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7)、[详细证据PR #20](https://github.com/Yileina-Monv/retinal-vessel-study/pull/20)|
|权重及预算是否解释负结果|1.5/3.0、2/1、seed2026的2000→4000，各按原设计|未获得可靠且受保护的TW增量；延长训练仍有质量取舍|保留负面/不确定结论；4000步不是独立验证|[权重](../../retinal_m0/maples_weight_ablation_experiment.py)、[2/1](../../retinal_m0/maples_class_split_experiment.py)、[续训](../../retinal_m0/maples_budget_extension.py)|[PR #20各阶段](https://github.com/Yileina-Monv/retinal-vessel-study/blob/62c55b9deba6a03483d77265ac49d32f2d88d244/outputs/research_evidence_20261108/README.md)|
|Plan B：具体在哪里失分|77位置×3seed配对；11条件重复遗漏诊断|恢复0次、失去2次；27/77位置在11条件均未命中|找到失分与持续遗漏位置；未证明低对比/遮挡等医学原因，人工0/105|[分析](../../retinal_m0/plan_b_error_analysis.py)、[汇总](../../retinal_m0/plan_b_finalize.py)|[PR #20](https://github.com/Yileina-Monv/retinal-vessel-study/pull/20)|
|骨架监督范围是否有帮助|A0=原T0 seed2026；B1局部点/B2完整候选路径，λ=.03，各2000步|全FOV细分支召回提高，Precision下降、FP增加、Dice下降|出现召回与误报交换，未达到可靠综合改善；不是原邻域主指标成功|[科学训练](../../retinal_m0/maples_branch_science_worker.py)、[评价](../../retinal_m0/maples_branch_science_analysis.py)|[PR #7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7)、[图册PR #20](https://github.com/Yileina-Monv/retinal-vessel-study/pull/20)|
|Loss/采样/注意力能否改善|soft-clDice、Focal、UFL、空间分层、单scSE，独立单seed探索|未获可靠整体Dice增益；不同方案存在召回/误报取舍|不能把小点估计当稳定收益，亦不能概括所有Loss无效|[soft-clDice](../../retinal_m0/maples_soft_cldice.py)、[Focal](../../retinal_m0/maples_focal_dice.py)、[UFL](../../retinal_m0/maples_ufl.py)、[采样](../../retinal_m0/maples_stratified_science.py)、[scSE](../../research_branches/scse_single_module_20261026)|[PR #7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7)、[诊断PR #20](https://github.com/Yileina-Monv/retinal-vessel-study/pull/20)|
|标准nnU-Net流程能否成为更强基线|共同seed2027/2028，固定100轮终点；与T0同划分同评价|Dice约+2.02pp；补充邻域细分支召回约+18.84pp，邻域FP密度约+.43pp|流程收益；训练预算等多项不同，非结构单变量，严格保护未全通过|[双seed训练](../../research_branches/nnunet2d_20261020/replication_seeds_2027_2028_v1_20261026/replication_train.py)|[PR #10](https://github.com/Yileina-Monv/retinal-vessel-study/pull/10)、[统一重评PR #19](https://github.com/Yileina-Monv/retinal-vessel-study/pull/19)|
|nnU-Net组件是否有额外帮助|N-Local/N-Multi容量对照；另行去除深监督|多尺度无可靠额外Dice增益；去除深监督无改善依据|单seed消融与双seed流程比较分开；局部发现不改写Dice结论|[模块](../../research_branches/nnunet2d_20261020/multiscale_capacity_ablation_v1_20261027)、[深监督](../../research_branches/nnunet2d_20261020/deep_supervision_ablation_20261028/science100_v1)|[PR #14](https://github.com/Yileina-Monv/retinal-vessel-study/pull/14)|
|局部监督是否可行/有效|D0诊断、NL/NR V1/V2配对；后续N0/NL两seed|配对覆盖13/30、23/30未达原80%门槛；后续NL未达预设成功条件|失败方案未追认通过，NL无可靠额外局部收益|[NL训练及正负例辅助项](../../research_branches/nnunet2d_20261020/nl_absolute_ce_science100_v1_20261101)|[PR #16](https://github.com/Yileina-Monv/retinal-vessel-study/pull/16)、[提案#18](https://github.com/Yileina-Monv/retinal-vessel-study/pull/18)、[结果#19](https://github.com/Yileina-Monv/retinal-vessel-study/pull/19)|

## 必须一起阅读的限制

原正式评价按宽度≤3原生像素逐点筛选，9有效图77点；补充评价先选择FOV相对宽度中位≤6、参考长度≥20的候选路径，再评价原邻域内部分，18有效图174路径。阈值0.5和原生d1保持，但目标和平均权重不同，不能跨口径直接减分；补充分析不重判正式结果。

两seed共享Dev28，不是56独立病例；Dev反复用于开发，官方Test60未访问。GT相对错误图不等于医学审核或真实连通性证明。T0/TW三seed正式结果、B1/B2单seed、nnU-Net双seed和组件单seed不得混为同一级证据。

## 原方案及未采用提案

历史TW独立审查和未采纳提案见[PR #8](https://github.com/Yileina-Monv/retinal-vessel-study/pull/8)，病灶主线文献及原则见[PR #12](https://github.com/Yileina-Monv/retinal-vessel-study/pull/12)，早期MAPLES记录见[PR #4](https://github.com/Yileina-Monv/retinal-vessel-study/pull/4)。记录存在不代表方案已执行。学习率/预算迁移到原U-Net等近期讨论仅是建议，本次没有新训练许可。
