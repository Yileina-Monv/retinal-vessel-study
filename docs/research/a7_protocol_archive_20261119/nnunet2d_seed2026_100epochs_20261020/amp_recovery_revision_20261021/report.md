> 历史材料快照，采集于2026-11-19。文中“当前/待批准/授权”仅适用于原阶段；不构成新的执行许可。路径与设备标识已泛化，原始SHA见来源清单。

# nnU-Net AMP、计数与中途恢复修订
日期：2026-10-21；Sol直接执行。状态：CPU修订验证完成，科学规则修订稿；不是GPU放行。

## 事实与本次授权
用户“可以，先修订”仅批准本次修订及CPU验证。本次未启动新训练、视网膜模型前向、Dev推理或Test访问。父冻结协议、旧代码、旧权重及失败记录不改写。

原轨迹6527次尝试、6526次成功、1次AMP跳步；scale由524288降到262144。原保护把此事件判为整轮终止。CPU检查中断权重和优化器状态均有限。stop_record把行数误当成功数，多计1；旧原件保留，审计更正另存。旧轨迹仍是INCOMPLETE，不转写成新规则下已完成。

## 修订的事件与计数
通过optimizer.step前置钩子观察真正的调用，不增加forward/backward/step或消耗随机流：
- 有限Loss、梯度/状态安全且有1次更新：成功。
- Loss/模型/optimizer状态有限、scaler降低且实际更新0次：受控AMP跳步，单次不立即终止。
- 非有限Loss、参数/optimizer状态损坏、未被scaler阻止的异常更新或调用计数矛盾：停止。
- 持续/累计跳步仍须暂停；候选建议连续达到5次或累计达到25次暂停，属于待确认工程门槛，不是按Dev表现选择。代码要求显式配置，没有隐含默认。
- 分别记录尝试、成功、跳步、致命尝试、实际optimizer调用；异常尝试也记账，不把日志行数当成功数。

## 预算修订建议，待确认后才可冻结
建议保持100epochs×250=25000次计划尝试，照实报告实际成功次数和AMP跳步，不自动补步、不加epochs、不恢复小时截断。允许跳步后，不能同时保证“25000尝试且25000成功”；若用户选择成功数预算，需另行明确额外尝试上限，当前不授权。
结构、Batch/Patch、标准CE+Dice、SGD/Poly100、采样增强及评价不改变。只使用GPU0，GPU1属于别人。

## 恢复修订与硬边界
旧日志定位：完成26个epoch，当前零基epoch26；已消费27次本epoch尝试，下次零基iteration27。当前epoch剩223次，总固定尝试预算剩18473次，内部Train监控尚余3700批。
原native checkpoint无条件存current_epoch+1，故标为27，不能按原CLI直接接下一epoch。候选状态显式保存train/monitor/epoch_finalize阶段、epoch和iteration、成功/尝试/跳步计数、主RNG、scheduler及部分epoch汇总；不能因保存动作自动把epoch加1。候选bundle故意不是vanilla checkpoint，须revision-aware训练循环处理。
历史主RNG和原生worker队列并不存在，不能追认补齐。主RNG保存不等于增强队列恢复。原生队列严格恢复仍不支持，严格恢复入口明确拒绝；新完整GPU循环和GPU恢复验证尚未实施。是否接受披露随机流重建的非精确接续，或另选fresh轨迹，仍需用户决定。本次没有导出或改写训练权重。

## 实际验证与局限
37项CPU回归通过，包括真实CPU GradScaler溢出跳步后成功更新、模型不因该跳步改变、观察钩子不增加更新/随机消耗、错误停止、预算不补步、epoch各阶段边界、主RNG回放、旧断点位置和成功数核对。
12项检查通过，原16个受保护代码/协议/日志/中断checkpoint前后摘要一致。CPU标量探针不是视网膜训练，不证明GPU端恢复或标准worker队列可逐位续接。没有视网膜模型实例化/前向，没有CUDA初始化和GPU更新。

证据：
- ${RETINA_PROJECT_ROOT}/experiments/nnunet2d_branch_20261020/scientific_seed2026_100epochs_v2_no_wall_limit/repair_amp_cpu_20261021/attempt_01/verification.json
- 同目录unit_tests.log、legacy_checkpoint_audit.json
- 候选实现：research_branches/nnunet2d_20261020/amp_recovery_revision_20261021/

下一步：用户确认预算/暂停门槛和可接受恢复类型，再实施完整循环与新的启动前核验；GPU启动仍另行确认。B1/B2未放行、官方Test封存、原正式结论保持。
