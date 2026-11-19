> 历史材料快照，采集于2026-11-19。文中“当前/待批准/授权”仅适用于原阶段；不构成新的执行许可。路径与设备标识已泛化，原始SHA见来源清单。

# nnU-Net 2D 单种子100 epochs探索：冻结阅读版
2026-10-20，Sol。用户已明确批准；科研规则已冻结，Luna通过本轮启动前门槛后可执行唯一一条轨迹。不是新增三seed正式实验。

## 范围与固定配方
- seed2026，fresh随机初始化；不载入M0/T0/工程Checkpoint，不接续100次工程更新。
- 原Train110与Dev28不变，fold=all。标准PlainConvUNet、CE+Dice/deep supervision；已验Train-only 8GB plans，patch1024×1536、batch2，SGD/Nesterov、LR0.01、Poly周期100epochs、其余以protocol.json与plans为准。
- 用户接受标准二值背景/padding与T0 FOV掩膜训练域不同；连同随机初始化、预处理/采样/Loss/预算差异，只能解释为模型流程比较。
- 原T0 seed2026/step2000只读复用。B1/B2继续CPU研究，旧S3 H1不放行，不继承任何旁支GPU授权。

## 预算、随机性与停止
固定100epochs×250=25000次计划optimizer尝试，无AMP跳步时恰25000次成功更新；每epoch50批内部Train监控，共5000批，不是Dev验证。工程100次已经用完，不能再跑额外工程更新。
本次新科学运行加终点推理累计GPU占用最多36000秒（10小时），不是保证完成时间；实测基础7.24小时，含其他开销暂估8..10小时。GPU0空闲才开始，P1优先；不抢占/杀进程/换卡。时间不足、OOM、非有限或AMP跳步保存停止，不自动缩batch/改精度/补步/延长。
主seed2026，训练/内部监控增强各1worker，明确worker seeds为[2026]/[2027]并设置实际worker RNG，CPU门槛验证其生效。cudnn deterministic=true、benchmark=false。这里只固定可记录的随机源，不承诺跨硬件或异步队列的逐步完全重放；科学中断不自动resume/restart，保存状态交Sol/用户决定，预算不清零。

## Sol工程交付审查与门槛
已独立核对工程源/计划/环境摘要、四段日志100次成功及四Checkpoint哈希、原T0终点28概率摘要。存在一项元数据不一致：旧engineering/execution_binding.json记录的budget_ledger哈希为旧版本；现账本已补入5批Train监控前向，累计246.77秒仍低于1800秒。不能改写旧绑定“追认无异常”；Luna必须在新科学preflight追加账本版本/实际日志核对及当前哈希，保留原件，无需且不得重跑工程。
其他局限保留：环境只读共享底层依赖；恢复仅证明固定输入；训练反向峰值尚非完整仪表化；坐标原检查只是最近邻identity标记。启动前需CPU合成验证概率通道/逆变换、累计预算守护、100epoch LR、worker种子、源码/数据/包摘要，并绑定新实现。存在影响定义/数值的异常则停，不自行更改科学方案。

## 唯一终点评价
仅checkpoint_final.pth、单模型、显式fold=all、无TTA/集成/自动后处理；不调用训练CLI附带的Train110实际全图验证。
仅终点Dev28预测：nnU-Net冻结预处理与tile_step0.5/Gaussian流程，导出class1 vessel softmax概率，逆变换回原生H×W、float32、固定>=0.5交原评价器。T0推理与nnU-Net不同必须披露，不强行称相同流程。
主为整体Dice；必报Precision、FP/FOV、全图Recall、远处召回、原邻域像素/细骨架召回与全域原S3 d0/d1。全域28图7291点；原邻域9图77点、19NA不变。完整细分支指标尚未批准，不使用当前B1/B2试案选有利口径。
相对T0精确率/远处召回不降、FP/FOV不升，点差零退让，逐图恶化和区间全部报告，不把点差保护通过称严格统计不劣。整体Dice升而细骨架不升只能称整体流程信号。覆盖不等于连续性恢复。
单seed按图配对1000次PCG64(20261016)，每种有效n重新初始化、同n共享索引；保存分子分母/NA/全部重抽样。只用固定终点，不按Dev曲线选模/延长、调阈值或扫描口径。正负均为复用Dev的探索证据，不是独立验证，不改原正式结论。

## 交接与产物
冻结包为protocol.json、source_manifest.json、sol_engineering_review.json及本阅读版；实际执行绑定由Luna在新代码CPU门槛完成后生成，不反写冻结协议。
新科学输出位于原旁支根目录下scientific_seed2026_100epochs_v1，GPU/更新账本、源绑定、启动核验、日志、final与28概率、逐图表/统计/报告、completion_verification齐备后回写原STATUS并停止交Sol。失败/未完成也保留证据，不要求结果为正。
此授权不包含其他seed、配置、Loss、额外步数、基线重训、S3/B1/B2 GPU或官方Test。唯一总进度仍为原STATUS。
