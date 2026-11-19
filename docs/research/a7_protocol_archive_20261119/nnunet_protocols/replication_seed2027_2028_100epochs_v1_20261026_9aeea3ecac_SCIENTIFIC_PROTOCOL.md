> 历史材料快照，采集于2026-11-19。文中“当前/待批准/授权”仅适用于原阶段；不构成新的执行许可。路径与设备标识已泛化，原始SHA见来源清单。

# nnU-Net固定2027/2028连续100epochs复现协议v1

用户已选择固定2027和2028两条、GPU0串行，并以Implement the plan批准实施。Sol直接实施和审查，不新增模型/Loss/模块或额外GPU工程。两条按2027→2028依次运行，效果好坏均完整交付；技术异常暂停排程，不自动补步、清零或重试。每条新随机初始化，不用M0或任何旧实验权重；旧2026含1000步后增强队列重建，只作历史探索参照，不混称严格同执行协议三种子验证。

原2D PlainConvUNet/现有plans/数据集/标准CE+Dice深监督/所有原Loss参数、Train110/Dev28、fold=all均保持。Batch2、Patch1024×1536、SGD初始LR.01/momentum.99/Nesterov/weight decay3e−5、原Poly100 exponent.9、clip12、原生AMP及初始GradScaler不改。每seed独立100×250=25000训练尝试与100×50=5000 Train-only监控，总50000/10000。成功+安全AMP跳步=25000，不补步；连续5/累计25安全跳步停止。成功须真实有限未缩放梯度和一次实际参数更新；安全跳步须有限前向及模型/optimizer、GradScaler下降、无optimizer调用且模型及optimizer不变。致命/未知/来源变化/OOM停止，已消耗计数完整保存。

seed2027的主/Train增强/监控增强种子为2027/2027/2028；seed2028为2028/2028/2029，每池1worker。保留原producer的NumPy及Python播种语义，用真实Augmenter子类保持isinstance清理正常。只更改新入口，不改已安装包/历史源码。正常进度无墙钟截断，心跳只告警；GPU0物理UUID ${GPU_UUID}，独占共享锁，保留约10GB实际需求，不套用scSE8GiB上限；GPU1不纳入本范围。GPU忙时等待不驱逐不切卡；磁盘最低50GiB，不清理旧结果。CPU线程2/有限BLAS。每250尝试保存完整可用状态、30/200启动快照、20/40/60/80/100不可覆盖快照及原生final。异步队列不宣称可精确恢复，中断后保存状态并审查，不自动续训或重开。

CPU检查包含新种子映射/重复性/分池数/禁止额外池、原生on_train_end清理与stdout恢复、预算和CLI拒绝、种子对应A0来源和31指标接线，复用既有官方循环10CPU/AMP32工程及三尺度导出证据，不新增GPU预演。真实科学首段检查全部计入25000，不将seed27/28与seed26输入或数值强行匹配。

每条只用checkpoint_final.pth在100epochs完成后一次Dev28，固定.5、原native预处理/逆变换/tile.5/Gaussian，无TTA/集成/后处理/中途Dev或best选择。每case一次，native几何67总前向上限；两条最多56cases/134前向。各seed复用其原A0同seed固定2000终点28概率，无新增A0推理；CPU核验各自summary/配置/run binding/对应文件SHA及last与milestone实际状态，文件字节SHA可因序列化不同而不同。原GT/FOV/Omega/宽度/d1/有效图及9图77位置19NA不改。

普通Dice为主，每seed按同一28图配对NN−对应A0宏平均，PCG64 seed20261016/1000次/linear95%区间。两新seed均Dice均值正且各自区间下界>0才称两seed复现信号。完整保护仍为Precision/Recall/远处/粗血管/hardclDice均值不降、FP/FOV不升，且Precision/远处95%下界≥0、FP95%上界≤0；两条分别判读，不把区间跨零描述为已证实成本。细分支/原邻域等31指标、全部逐图分母/NA均报告。跨新两seed报告均值和ddof1样本SD(n=2)，不是56张独立图片、不用图像CI替代种子稳定性；旧2026及scSE单列为上下文。Dice复现与保护通过分开表述；工程完整可复算即交付，不以好成绩为完成条件。结果决定是否申请后续研究，不自动扩seed、重新跑2026、加模块、改阈值或访问Test60。

资料在现有nnU-Net目录下隔离，各seed各自绑定/预算/权重/日志/28概率/报告；共用原STATUS和研究决定，不建第二进度体系。完成或停止后台自动CPU核对并回写，明确待Sol科研复盘。旧正式T0/TW负面结论与所有失败/费用、人工0/26保持。
