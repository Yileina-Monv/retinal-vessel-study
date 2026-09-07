# FIVES 数据准备入口

**2026-09-05 更新：**正式开发划分与 FOV 已形成独立冻结版本（470 训练 / 118 验证 / 12 边界案例），见[冻结与跨电脑验收](../releases/FREEZE_STATUS_2026-09-05.md)。本目录以下内容保留为原 32/8 小规模试跑的历史说明，原文件和配置不改写。

本目录对应用户授权的步骤 1、2：试跑规则、样本清单、数据加载、同步裁块与增强、读取核验。规则见 [试跑规则 v1](试跑规则_v1.md)，机器配置见 [data_trial_v1.json](../configs/data_trial_v1.json)。

**当前状态：步骤 1、2 完成，自动检查和样例目视检查通过。** 已形成 32 张试训、8 张验证、4 张拟合检查和 20 张异常检查清单。原始训练分区全部 600 对均通过新读取入口；128 个裁块在单/双进程下的 RGB、标注和几何参数完全一致；5 项契约测试通过，覆盖 16 种几何组合及非法输入。

首次在受限执行模式下运行时，Windows 双进程通信被权限限制阻止；改用获批的普通用户执行方式后通过。原始失败记录保存在 `verification-restricted-attempt.*`，成功记录为 `verification.json` 和 `verification.log`。没有修改系统 ACL 或进程通信设置。

## 文件入口

- [样本选择摘要](selection_summary.json)：数量、分层组成、来源和清单哈希。
- [完整训练元数据](manifests/all_train.csv)、[试训 32 张](manifests/smoke_train.csv)、[验证 8 张](manifests/smoke_val.csv)。
- [拟合检查 4 张](manifests/overfit.csv)、[异常检查 20 张](manifests/edge_cases.csv)。
- [读取验收](verification.json)、[验收运行日志](verification.log)、[目视核验](visual_review.md)。
- [裁块与对齐图](artifacts/paired_crops.png)、[空标注及 RGBA 图](artifacts/edge_cases.png)。图像是本地生成物，已排除在 Git 之外。

## 重建清单与运行验收

在项目目录使用既有独立环境：

```powershell
& '.\.venv\Scripts\python.exe' -m retinal_data.prepare
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m retinal_data.verify
```

准备命令只重建派生清单，不修改源数据。第三条命令重新读取 600 对训练图，再检查裁块、双进程和可视化；不会建立模型或调用 GPU 训练。重新生成图片后，自动报告会将目视复核状态设为待检查，需重新确认图像。

## 后续训练如何调用

```python
from retinal_data.fives import trial_dataset, make_loader

def main():
    train = trial_dataset("smoke_train")
    for batch in make_loader(train, batch_size=2, workers=2, epoch=0, shuffle=True):
        images = batch["image"]   # [B, 3, 512, 512], float32, 0..1
        masks = batch["mask"]     # [B, 1, 512, 512], float32, 0/1
        quality = batch["quality"]  # [B, 3], IC / Blur / LC; no training weights
        break

if __name__ == "__main__":  # Windows spawn requires an import-safe entry point.
    main()
```

下一轮传入新的 `epoch`。要精确重放一个数据样本，可以调用 `dataset[(epoch, draw_index)]`；这只重放数据，不等于恢复完整模型、优化器和训练状态。

拟合检查清单有 4 张图，默认每图仍对应 4 个裁块索引。如果下一步要固定每张图的一个裁块，可使用索引 `0、4、8、12` 并固定 `epoch=0`；训练代码应明确记录实际使用的索引。

`trial_dataset("smoke_val")` 默认返回 2048×2048 整图；`trial_dataset("overfit")` 默认裁块但不增强。返回值中的 `geometry` 顺序为 `[top, left, height, width, horizontal_flip, vertical_flip, quarter_turns]`，左上角坐标相对于原图，旋转发生在裁块和翻转之后。

配置或清单被修改后，默认入口会检查指纹并报错，要求重建和重新验收。清单选择根据已经审计的质量和空标注信息，不根据模型结果调整。当前代码只接受原始训练分区；测试和外部评价应在后续明确的评价入口中接入。

## 当前边界

FOV、完整训练预算、正式划分和重复标注的正式汇总方式尚未冻结。当前质量比例用于覆盖检查，不能当作总体分布；临时排除重复图和整图空标注仅适用于本次小规模主试跑。原始 600 对均保留并可读取。
