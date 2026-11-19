> 历史材料快照，采集于2026-11-19。文中“当前/待批准/授权”仅适用于原阶段；不构成新的执行许可。路径与设备标识已泛化，原始SHA见来源清单。

# nnU-Net运行保护补充：无墙钟限时重启02

本补充经用户明确批准，由Sol直接执行；不改写父冻结包。取消正常进度下9.5h/10h硬截断，保留100epochs/25000次尝试与全部异常保护。旧916次更新及累计1441.203855秒另列保留，新轨迹独立输出，不从中断或工程checkpoint续接。

GPU0按UUID专用，GPU1留给未来另批主线；GPU资源检查按卡隔离。不改变batch2、patch1024x1536、标准CE+Dice、SGD/Poly100、seed/worker增强及Train110/Dev28规则。固定final-only评价，无官方Test。B1/B2仍未授权训练。

protocol.json为父协议的有授权运行修订副本；source_manifest.json原样引用父包受保护资料。prior_trajectory_accounting.json解释旧账本继承字段与实际日志。freeze_receipt.json绑定本补充。CPU预检与回归通过后，authorization.json一次消费，失败不自动再跑。

保留局限：nnU-Net与T0属于流程比较而非单变量消融；单seed、反复使用Dev，非独立验证；增强队列不保证跨进程恢复，因此本轮从头开始。最终指标及NA/保护全部沿用，不因结果正负扩预算。
