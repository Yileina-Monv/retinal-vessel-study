> 历史材料快照，采集于2026-11-19。文中“当前/待批准/授权”仅适用于原阶段；不构成新的执行许可。路径与设备标识已泛化，原始SHA见来源清单。

# 2D nnU-Net v2 独立旁支可行性评估
日期：2026-10-20；Sol。状态：待用户确认的提案，不是冻结协议或执行许可。

## 建议与授权边界
有条件可行，值得作为模型流程比较开展一条有限预算探索，不替代U-Net+B1/B2主线。原S3 H1不放行、B1/B2目标未冻结、现有负面结论及Test封存均不变。
本轮仅SSH只读检查环境、Train格式及既有记录，查阅官方代码；未安装/升级软件、未创建旁支代码/数据/结果目录，未实例化模型、前向或参数更新。只将评估状态记回既有记录。
建议先批准独立环境/CPU准备及工程小验证；计划、吞吐和风险确认后再批准科学训练。本轮不交Luna实施。

## A7实际检查
- 两张GTX1080Ti，各11264MiB，compute capability6.1，驱动580.178.04；本次建议只用GPU0且不与主线并行，不把两卡合成22GB。
- CPU20逻辑线程、内存约62.7GiB，NAS余量约2197GiB；显存/吞吐尚未实测。
- 当前retina_a11：Python3.10.21，Torch2.4.0+cu121，torchvision0.19.0，nnunetv2 2.8.1，dynamic_network_architectures0.4.4；pip check无依赖破损，Trainer只导入成功。
- 三个核心nnU-Net源文件与安装RECORD一致。版本号/导入成功不等于GPU运行成功，不能据此放行。现有环境不改，另建独立环境并锁定已验版本与包摘要，禁止pip自动升级Torch破坏Pascal兼容性。
- Train110图均RGB；三尺寸960×1440(58)、1488×2240(33)、1536×2304(19)。GT均PIL mode1，读取值0/1；全110尺寸对齐。每尺寸RGB解码抽样均uint8、3通道、范围0..255。Train/Dev名单110/28无交集。
- GT不能按255除法转成全零；导出新uint8 PNG时保留0/1。RGB由TIFF无损转PNG，不缩放、不改原始标注。Test未读取。

## 首选划分与预处理
规划数据集只含Train110：imagesTr/labelsTr/numTraining=110。全部数据转换、指纹、数据集统计和自动规划只接受Train白名单，禁止从大数据根目录通配扫入Dev/Test。
明确fold=all训练全部110，不运行默认fold0..4，不以fold编号当seed。标准fold=all的内部val也来自Train，只是训练监控，不是Dev或独立验证，不能拿内部EMA Dice作科研结果。
Dev28放独立dev_inputs，仅终点应用冻结预处理并输出原生概率；Dev标签只交原冻结评价器。不将138例一起规划后再靠splits_final.json挽救。无imagesTs/Test副本。
Native NaturalImage2DIO支持单文件RGB id_0000.png；channel_names按官方R/G/B例，默认逐图逐通道Z-score，冻结最终plans。不是M0的FIVES均值/标准差。
NaturalImage2DIO给的是单位面内spacing，默认不会替本项目校正三种图像尺寸的相对血管口径。无物理spacing时不假造毫米数，不自行把所有图缩到固定分辨率；非零区裁剪必须可逆回原图。

## 标准配方与明确差异
首条建议为classic PlainConvUNet的2D ExperimentPlanner（生成nnUNetPlans）+ 官方nnUNetTrainer_100epochs；不用ResEnc/Primus、不用额外预训练、不加骨架/病灶Loss。
- 初始化：随机He风格初始化、seed=2026；不加载FIVES M0，也不做不完整权重映射。
- 原生二类V标签，标准softmax CE+Dice、deep supervision。T0是单logit BCE+Dice，无新增辅助监督；不能把二者写成只换网络。
- 采用标准前景过采样（参数0.33，实际批内比例受取整影响）、标准几何/强度/模糊等增强，不伪称与T08000draw配对。
- 标准SGD/Nesterov，初始LR0.01、momentum0.99、weight_decay3e-5；Poly以100epochs为周期。不是默认1000epochs配方截取前100epochs。
- 显存规划目标8GB，patch/batch由Train-only plans确定，当前不编造数值。关闭torch.compile；FP16/GradScaler按实际兼容门槛验收。若OOM/非有限，停止而不擅自改变batch、分辨率、精度或Loss。
- 为保持首条配方清晰，提案使用原二值V与nnU-Net默认裁剪/背景/padding监督，不加入自定义FOV掩膜Loss。它与T0“仅原FOV内Loss”的支持域不同，必须作为流程差异列报；原FOV只读且用于所有外部评价。若用户要求训练支持域也严格一致，应另批原生ignore-label及padding适配，不能把它当本次默认已经一致。此选择尚待用户确认。

## 单种子固定预算
拟100epochs×250训练迭代=25000次计划训练迭代；同时有100×50内部Train监控批次。官方默认是1000epochs×250=250000，首轮不运行默认完整预算。
记录真正成功optimizer更新及AMP跳过，不把epoch/迭代数混作成功更新。预期无跳过时为25000次成功更新；发生跳过/数值不稳即保存停止报告，不自行补步延长。
采用同一固定终点checkpoint_final；checkpoint_best可留作软件产物但不用来比较、推理或选模。终点只对Dev28做一次预先固定评价，不根据Dev曲线决定延长/选Checkpoint。
同一个seed不等于同输入序列或完全可重放：标准多进程增强使用seeds=None；现成Checkpoint也不能自动证明随机流/队列状态恢复。未来工程需明确主RNG、worker seed及恢复边界并记录；无法满足约定可复现性则不启动科学轨迹。
T0只读复用seed2026/2000步终点，另附原三seed结果作背景，不重训T0，不将两边不同预算称为公平单变量比较；报告训练更新、batch×patch像素预算、GPU时间。

## 预算估计，不是测速结果或授权
已有T0日志seed2027在相邻有效训练步的中位间隔约0.529秒（排除评价/保存和重置等间隔）；不可直接当作nnU-Net速度。
按候选0.8..2.5秒/训练迭代的情景估算，25000迭代约5.6..17.4小时，加内部监控、保存、末次推理，暂按6..20 GPU小时规划，置信度低。最终以获批工程测得的plans和端到端吞吐重估，不用“已安装”冒充已测速。
建议：
1. CPU：隔离环境、白名单转换、完整性、指纹/plans、原生坐标往返；预留1..3小时人工/CPU窗口，不保证安装下载耗时。
2. GPU工程：独立可丢弃验证，至多100次更新且GPU占用至多30分钟，覆盖三个尺寸Train样本、目标/概率形状、AMP/显存、种子与保存加载检查及稳定吞吐；不看Dev。尚未授权。
3. 科学训练：上述门槛和计划由Sol审查、用户确认预算后，才执行单seed100epochs。若批准建议设24 GPU小时硬上限；不足以完成则记未完成，不自动超时续跑或下调配置。预留50GiB空间，CPU增强进程建议先4个并记录。

## 评价与判读
- 建议旁支主指标为整体Dice；全域原S3 d0/d1为关键次要指标。Precision、FP/FOV、全图Recall、原远处/病灶邻域及NA均强制报告，不通过选有利指标宣布成功。
- 固定0.5，取vessel softmax概率，逆变换还原原生H×W后交原评价器，不直接拿nnU-Net硬mask/内部patch Dice作替代。原FOV/U/d1/FP分母及原邻域9图77点19NA不改。
- 单模型、无TTA、无自动连通域删除、无阈值扫描、无五折集成、无find_best_configuration。切窗融合采用冻结nnU-Net流程，报告它与T0的差别。
- 原零退让保护不放宽；Dice/细骨架任何方向均报告。如果Dice升而细骨架不升，只能称整体流程信号；Precision/FP等保护失败不能写成无代价改善。
- 完整细分支定义未核验，不提前采用当前6/20试案；将来获批后用共同GT定义复算所有方法，并披露是后续辅助分析，不根据新模型有利区域改定义。覆盖不等于真实连通恢复。
- 随机起点、不同规划/归一化/采样/增强/Loss/训练预算共同变化，结论只是模型流程比较；阴性不能证明完整默认nnU-Net无效，阳性不能归因某一组件。
- 单seed、反复使用Dev均非独立验证；不接触Test。完整、可复算的负结果也可工程交付。

## 拟定隔离位置（均尚未创建）
代码：${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/research_branches/nnunet2d_20261020/
环境：${RETINA_PROJECT_ROOT}/envs/nnunet2d_v2/
数据/规划/结果：${RETINA_PROJECT_ROOT}/experiments/nnunet2d_branch_20261020/ 下raw、preprocessed、results、dev_inputs、evaluation、engineering。
只在旁支进程设置nnUNet_raw/preprocessed/results，禁止改全局环境或主线目录。旧源/模型/预测不覆盖；GPU0主线优先、不并行、不抢占他人。唯一STATUS仍在原仓库。

## 依据
- [官方数据格式与RGB例外](http${LOCAL_PATH}
- [官方100epoch变体](http${LOCAL_PATH}
- [官方fold=all与数据加载实现](http${LOCAL_PATH}
- [官方2D规划器](http${LOCAL_PATH}
- A7实际site-packages源码/依赖元数据、既有Train/Dev清单、T0 seed2027 train.jsonl；安装版本与网页master不混同，以未来冻结包和实际源摘要为准。

