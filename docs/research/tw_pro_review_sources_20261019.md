# 两轮 TW 分析：来源、版本与验证边界

2026-10-19；对应[综合报告](tw_pro_review_20261019.md)、[完整提案](tw_next_experiment_proposal_20261019.md)和[S1卡](tw_s1_experiment_card_20261019.md)。这是团队可直接阅读的研究整理，不依赖私人聊天链接。没有提交私人会话、文件服务ID、原图、标签、权重、概率、原始运行资产或第三方课件。

## 1. 输入材料身份与处理方式

四份生成的研究交付均已全文读取。第一轮为630行报告与33行卡；第二轮为20页报告与1页卡。第二轮正文抽取中公式是图片，公式交接另经页面视觉核对，并与首轮文本公式交叉核对。本 PR 不声称读取了第二轮聊天正文，也不把分析作者的复算自动算成本 PR 的复算。

| 输入交付 | 版本/创建时间（UTC） | 身份与本次使用 |
|---|---|---|
| `retinal_TW_independent_review_20261019.md`，题名《MAPLES-DR 病灶邻域细血管召回：TW 独立分析与下一轮实验》 | 2026-10-19 04:57:00；固定旧SHA | 62,337字节；交接原字节SHA256 `7f54c075588779db14ed7c3214593c21540db012992c5b7a8de2899e23dba11e`；保留机制推理、历史边界和具体实验设计 |
| `retinal_first_experiment_card_20261019.md` | 2026-10-19 04:57:08；固定旧SHA | 3,920字节、33行；由第二轮卡补严对照/操纵/机制规则，未按旧状态继续执行 |
| `retinal_PR7_independent_review_20261019.pdf`，题名《MAPLES-DR 病灶邻域细血管召回：PR7 最新快照的独立分析与实验设计》 | 2026-10-19 07:05:11；固定新SHA | 472,606字节、20页；交接PDF SHA256 `19a583a452837d3c398e4954387143f1d4b3bb7f3caf4b45f8cac8f2b80f1497`；作为当前修正与二次分析来源 |
| `retinal_PR7_first_experiment_card_20261019.pdf` | 2026-10-19 07:05:11；固定新SHA | 108,328字节、1页；交接PDF SHA256 `d870d4c75882d1fe9ab56799d687c4292f663fe65ab90a02151675f37a524018`；转为本仓库可读未采纳卡 |

上表时间/大小来自材料元数据；材料服务没有提供可用版本号，未虚构版本号。原字节哈希由完整材料交接提供，不等同本 PR 重新下载并校验PDF原始字节；仓库公开输入的逐文件hash则由下述脚本实际计算。公开正文是保留有效结论和具体细节的中文整合，不是四个原件的逐字副本。

两轮还引用《阶段二视网膜血管分割研究进展与实验结果分析》（2026-10-19，21页），原件4,304,490字节、SHA256 `c01221eb0237519269f74903928a84fd893a35163619ed6fa75818f14a0f66b4`。其身份见旧快照source.json；协议/同FP等未新导出的内容在本 PR 仍按报告级引用，不声称本 PR 又逐页审核了该21页原件。定位：训练P4，区域/loss/指标P5–7，正式结果P9–10，曝光/类别P11–12，预算P13–14，同FPP15，历史Plan BP17，附录P19–21。

## 2. 固定仓库入口与独立分支基线

| 角色 | SHA / 状态（2026-10-19核查） | 入口 |
|---|---|---|
| 文档PR基线main | `0ede2fc9a29e3cc808478a523c834259ff647357` | [固定main](https://github.com/Yileina-Monv/retinal-vessel-study/tree/0ede2fc9a29e3cc808478a523c834259ff647357) |
| 第一轮历史快照 | `33ad1cc693eafbbf8a5c7f60438aee7541fb8118`，PR #4 open、未合并 | [PR #4](https://github.com/Yileina-Monv/retinal-vessel-study/pull/4) / [旧tree](https://github.com/Yileina-Monv/retinal-vessel-study/tree/33ad1cc693eafbbf8a5c7f60438aee7541fb8118) |
| 第二轮当前快照 | `28a281fa2be1167533bcd0a54eca544a6e1f565d`，PR #7 open、未合并 | [PR #7](https://github.com/Yileina-Monv/retinal-vessel-study/pull/7) / [新tree](https://github.com/Yileina-Monv/retinal-vessel-study/tree/28a281fa2be1167533bcd0a54eca544a6e1f565d) |

PR #4分支为`docs/progress-20260915`，PR #7为`results/maples-stage2-20261019`；二者从main分叉，前者ahead2、后者ahead1，不是简单先后续接。新文档PR从main单独起步，只加本次分析、提案、来源与数表检查，README/STATUS仅加独立入口；不合并/关闭既有PR，不复制两批结果文件。PR #4/#7的重叠由维护者协调，本PR不替团队决定结果包合并路径。

已读[协作约定](../../CONTRIBUTING.md)、[公开边界](../../PUBLICATION.md)与PR模板；相关公开Issue [#3](https://github.com/Yileina-Monv/retinal-vessel-study/issues/3)（课程/正式实验边界）和[#6](https://github.com/Yileina-Monv/retinal-vessel-study/issues/6)（结果发布），仅引用，不宣称已领取其他人的运行任务，不自动关闭Issue。现有本地工作树中的未跟踪资料保持原样，本次使用独立工作副本。

## 3. 团队可读的证据索引

所有“新”入口固定为PR #7 SHA；所有“旧”入口固定为PR #4 SHA，不能把旧路径当成当前head已有文件。

| 编号 | 固定文件 | 用途 |
|---|---|---|
| N1 | [新STATUS](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/STATUS.md) | 当时状态 |
| N2 | [新研究页](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/docs/research/maples_training_results_20261019.md) | 正式摘要、Plan B报告、CI/预算/分档措辞差异；L17/L33/L35/L39–41 |
| N3 | [source_audit.json](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/source_audit.json) | 来源状态、协议hash、人工0、未公开AI视觉、Test声明；不是原执行验证 |
| N4 | [training_config_summary.csv](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/training_config_summary.csv) | 17配置、none调度、起止步与成功更新 |
| N5 | [loss窗口CSV](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/loss_curve_200_update_windows.csv) | 190窗口与重复前缀；窗口内SD |
| N6 | [endpoint计数](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/fixed_endpoint_per_image_metrics.csv) | 分子/分母、NA、同图同seed终点；两个主差单元位于L624/631及L1114/1121 |
| N7 | [paired逐图差](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/paired_image_deltas.csv) | 2131行配对与逐seed差；非逐位置联合矩阵 |
| N8 | [summary CSV](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/outputs/maples_training_results_20261019/paired_comparison_summary.csv) | 87汇总；formal precision/fp/far为含表头L4/L6/L8 |
| N9 | [原公开校验脚本](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/scripts/verify_maples_training_export.py) | 检查行数、比例/NA、配对均值与完成标记；不检验CI/真实模型/位置几何 |
| N10 | [新阶段页](https://github.com/Yileina-Monv/retinal-vessel-study/blob/28a281fa2be1167533bcd0a54eca544a6e1f565d/docs/research/stage2_progress_20261019.md) | 历史转录与当前公开包的边界 |
| O1 | [旧阶段页](https://github.com/Yileina-Monv/retinal-vessel-study/blob/33ad1cc693eafbbf8a5c7f60438aee7541fb8118/docs/research/stage2_progress_20261019.md) / [PDF身份](https://github.com/Yileina-Monv/retinal-vessel-study/blob/33ad1cc693eafbbf8a5c7f60438aee7541fb8118/docs/research/stage2_progress_20261019_source.json) | 第一轮报告级证据与截点 |
| O2 | [本机验证README](https://github.com/Yileina-Monv/retinal-vessel-study/blob/33ad1cc693eafbbf8a5c7f60438aee7541fb8118/outputs/maples_local_validation_20261011/README.md) / [manifest](https://github.com/Yileina-Monv/retinal-vessel-study/blob/33ad1cc693eafbbf8a5c7f60438aee7541fb8118/outputs/maples_local_validation_20261011/source_manifest.json) | 13统计文件、历史radius分档、身份与缺失资产 |
| O3 | [本机同FP CSV](https://github.com/Yileina-Monv/retinal-vessel-study/blob/33ad1cc693eafbbf8a5c7f60438aee7541fb8118/outputs/maples_local_validation_20261011/matched_fp/per_image.csv) | 历史step200示例，不等于TW |

历史代码均固定旧tree：`retinal_m0/model.py`、`retinal_data/fives.py`、`retinal_release/data.py`、`retinal_m1/maps.py/loss.py/metrics.py`、`retinal_m0/inference.py`及`retinal_c1`。两轮分析将其用作UNet/同步几何/基础loss/soft-recall/距离/分链线索；本PR未将其当作已核验正式A7版本。旧C1的48点最短链与端点裁去8点过滤不能无条件用于本任务77个短细目标。

## 4. 本次实际验证及其边界

脚本：[verify_tw_pro_review_20261019.py](../../scripts/verify_tw_pro_review_20261019.py)；机器可读记录：[checks.json](tw_pro_review_20261019_checks.json)。它只读固定Git对象，不下载数据、不加载模型、不生成概率、不访问Test。输入CSV/JSON的SHA256与Git blob SHA1写入记录，结果可重跑：

```powershell
git fetch origin 28a281fa2be1167533bcd0a54eca544a6e1f565d
python scripts/verify_tw_pro_review_20261019.py
```

需可用Python与NumPy；脚本不联网，不改写来源。`--output`为显式可选JSON输出。CI使用NumPy PCG64与linear分位数，数值重建规则在脚本与综合报告中明示。

本次还将N9原始校验脚本与其六个公开输入从固定Git对象取至临时隔离目录执行，原脚本通过（17配置、190窗口、3836逐图、2131配对、87汇总）。这不加载模型。新增范围的UTF-8、Python语法、JSON、相对链接、固定SHA文件存在性、敏感模式与差异空白检查均通过。

验证层级分开：

1. **公开数表层**：17/190/3836/2131/87行；3103有效、733严格NA，476条件×图键无重复；全图TP/预测/参考/FP/Dice代数一致。17配置对应15 fresh+2续训、34000更新；170唯一窗口+20相同前缀。
2. **配对链**：1619条paired记录（含NA核查）、3173个数值seed单元可追到endpoint；63/87比较完整endpoint→paired→summary，另24额外宽度比较缺部分基线endpoint，仅paired→summary。87/87均值与40个多seed数值SD可重建。不能将24项写成原始分母层全部验证。
3. **CI数值重建**：按显式候选规则核对87/87区间，不证明历史实现采用该规则。旧正文精确区间与差异解释仍按第二轮报告引用；实际当前CSV区间以脚本输出为准。
4. **未验证层**：没有逐点联合矩阵、真实概率、RGB/标签、完整源码/运行绑定、人工结构复核和患者映射。Plan B联合集合、可见性、偏移、假连接、历史访问记录均未独立验证。

来源`official_test_accessed=false`及完成标记只是声明，可核对字段而不能独立证明后台访问历史。本次任务自己没有启动训练/推理/图像评价或Test访问。文档与数表检查通过不代表完整训练可复现、模型有效或临床有效。

## 5. 文献线索的使用范围

两轮分析引用[MAPLES-DR](https://www.nature.com/articles/s41597-024-03739-6)、[clDice](https://arxiv.org/abs/2003.07311)、[Skeleton Recall](https://arxiv.org/abs/2404.03010)、[clCE](https://papers.miccai.org/miccai-2024/paper/1081_paper.pdf)、[SkelCon摘要](https://pubmed.ncbi.nlm.nih.gov/35320091/)及其官方实现，用于机制适配，不把别的任务效果当成本任务结果。本PR未新增一轮文献全文复核。

保留的设计含义：clDice须有短断裂/拓扑证据且仍保护假连接；Skeleton Recall和历史M1的正参考奖励不自动控制预测结构精度；clCE须区分论文与代码归一化、不可照搬λ；SkelCon只启发局部正负机制，不移植完整多因素系统。OCTA或三维器官效果不直接外推到彩色眼底出血邻域。
