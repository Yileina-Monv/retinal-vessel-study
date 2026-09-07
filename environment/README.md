# 眼底血管分割项目环境

准备日期：2026-09-04。范围：Windows 本地训练依赖和合成数据环境验收。

**当前状态：准备完成，验收通过。** 本机依赖一致性检查通过，54 个依赖包已记录精确版本；`.venv` 文件合计约 3.38 GiB。原有基础 Python 环境中仍未安装 torch，本项目安装与其隔离。

| 本机验收项 | 结果 |
| --- | --- |
| RTX 5080、原生 SM 12.0、CUDA 13.0 | 通过 |
| 合成 U-Net，输入 1×3×512×512，约 776 万参数 | 三次前向、反向和 AdamW 更新通过 |
| FP16 混合精度训练算子、BF16 推理 | 通过 |
| 权重保存读取一致性 | 通过 |
| 双进程数据加载与传入 GPU | 通过 |
| 图像、骨架、距离变换、表格、绘图及日志 | 通过 |
| pip 依赖冲突检查 | 无冲突 |

合成 GPU 验收期间，PyTorch 记录的峰值已分配显存为 0.912 GiB、峰值保留显存为 0.941 GiB；这些值不包含其他进程占用，也不代表全部驱动/库开销或正式训练显存预算。实际输入批量、数据增强、损失和整图验证仍需在真实流程中测量。详细机器结果见 [verification.json](verification.json)。

## 环境选择

- Python 3.13.11，项目内 `.venv`，基于本机现有 Miniconda Python 创建，不继承其第三方包。
- PyTorch 2.12.1 + CUDA 13.0、torchvision 0.27.1 + CUDA 13.0。
- 采用官方发布的补丁版组合，不以跟随最新框架版本作为本次目标。
- 图像、骨架、距离变换、统计、表格、绘图和日志依赖由精确版本清单记录。
- 原有 NVIDIA 驱动用于运行安装包自带的 CUDA 库；本次不安装系统 CUDA Toolkit。

选择依据：[PyTorch 官方版本安装组合](https://pytorch.org/get-started/previous-versions/)；[官方 CUDA 架构支持矩阵](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)。该矩阵列出的 CUDA 13.0 Windows 构建支持 Blackwell SM 12.0。最终以本机实际算子验收为准。

## 使用

在项目目录打开 PowerShell，直接调用独立解释器，不必切换全局环境：

```powershell
& '.\.venv\Scripts\python.exe' --version
& '.\.venv\Scripts\python.exe' '.\environment\verify_environment.py'
```

在 VS Code 中打开 `AI临床应用` 文件夹时，项目设置提供 `.venv/Scripts/python.exe` 为默认解释器。若 VS Code 已为该文件夹记住别的解释器，手动选择本项目 `.venv` 即可。设置文件不保证覆盖已有的解释器选择。

## 验收内容

`verify_environment.py` 只使用合成数据，检查：

1. CPU 图像处理、骨架化、距离变换、NumPy/Torch 数组互转。
2. 中文路径 PNG 与 Excel 读写、绘图、日志和基本统计依赖。
3. CUDA 矩阵运算、torchvision CUDA 算子。
4. 512×512 输入、32/64/128/256/512 通道的 U-Net 形状探针；三次 FP16 前向、反向及 AdamW 更新，检查梯度与权重更新。
5. BF16 推理、模型权重保存及读取一致性。
6. Windows 两个 spawn 工作进程的数据加载、固定页内存及传入 GPU。

这个网络是环境探针，尚未冻结为正式基线。验收不是模型效果、完整训练耗时或恢复完整优化器状态的证明。短测时间包含首次初始化与检查，不用于外推正式训练总时长。

当前验收结果见 `verification.json`，控制台记录见 `verification.log`。合成图片、权重等输出在 `artifacts/`，已排除在 Git 之外。

## 复建与依赖记录

- `requirements-direct.txt`：依赖用途清单，允许版本解析，仅用于阅读和维护。
- `requirements-runtime.lock.txt`：PyPI 依赖的实际安装版本，包含间接依赖和 pip。
- `requirements-torch.lock.txt`：PyTorch 官方 CUDA 安装包版本。
- `setup.ps1`：先从 PyPI 安装锁定运行依赖，再从 PyTorch 官方 CUDA 源安装锁定框架，最后运行验收。
- `install-*.json`：安装来源及安装包哈希，由 pip 记录。
- `install-*.log`：安装日志。

在具有 NVIDIA GPU、兼容驱动和 Python 3.13 的 Windows 机器上，于项目目录执行：

```powershell
& '.\environment\setup.ps1' -BasePython 'python'
```

脚本保留已有网络代理，使用本项目临时目录，结束时恢复本次进程的临时目录环境变量。初次安装在受限工具中遇到过 Windows 临时目录权限问题，已使用获批的普通用户执行方式完成相应安装步骤，没有修改全局 ACL。

`.venv` 依赖创建它的本机 Python，不能直接复制给组员作为可移植环境；组员应使用版本清单复建。两份锁定清单限定本次 Windows/Python/CUDA 组合，不承诺 Linux 或其他 Python 版本直接通用。重建入口已提供，另一台机器的独立复建尚待执行。

## 项目边界

本次不读取真实数据进行训练，不冻结训练集划分、损失定义、训练预算或正式模型；不启动长期任务。正式资源测量应在数据加载和整图验证流程就绪后执行。
