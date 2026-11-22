"""Render the full, source-linked comparison once all job outputs have been audited."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(rel):
    return json.loads((ROOT/rel).read_text())

def main():
    lines=["# 出血边缘细血管召回探索记录","",
        "主要终点为出血外缘 11 像素内窄骨架点 d1 召回率，按图等权。三折、三种子，共 51 张有效图、3,632 个点。邻域误报涵盖 53 图，全图指标涵盖 54 图。潜在双眼关系未知。下列均为开发集证据。","",
        "## 同一批图的完整对照","",
        "低、高两档分别沿用已有普通 M0 和标准 nnU-Net 的训练折邻域误报目标。每个版本各自只用训练图定阈值；表中列留出图实际误报。除每千像素误报外，各项以 % 为单位。后四列保护和次要指标为较低误报档。","",
        "|训练与推理|低档细召回|低档邻域误报/千|高档细召回|高档邻域误报/千|低档细分支覆盖|低档全图 Dice|低档 Precision|低档远处召回|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sources=[
        ("U-Net M0 专项，原有双尺度","m0_contrast_transfer","base"),
        ("U-Net M0 专项，增强双尺度","m0_contrast_transfer","candidate"),
        ("2D nnU-Net 既有纠错训练，原有双尺度","contrast_cycle","base"),
        ("同权重，增强双尺度","contrast_cycle","candidate"),
        ("同权重，只增强两倍视图","hybrid_scale_cycle","candidate"),
        ("同架构三种子权重等权集成，原有双尺度","seed_agreement_cycle","candidate"),
        ("U-Net 与 nnU-Net 各一份权重，增强双尺度后等权融合","cross_family_cycle/nnunet","candidate")]
    for m,desc in zip("ABCD",("普通采样与损失","困难采样","困难采样及正负例辅助项","困难采样及仅正例辅助项")):
        for view,method in (("native","原图"),("dual","双尺度")):
            ref="B_vs_A" if m in "AB" else (m+"_vs_A")
            sources.append((f"2D nnU-Net {m} {desc}，{method}","thin_training/"+ref+"_"+view,"base" if m=="A" else "candidate"))
    for label,sub,side in sources:
        data=read(sub+"/summary.json");lo=data["m0matched"];hi=data["matched"]
        nums=[st[k][side] for st,k in ((lo,"thin_d1"),(lo,"near_fp_density"),(hi,"thin_d1"),(hi,"near_fp_density"),
            (lo,"path_d1"),(lo,"dice"),(lo,"precision"),(lo,"far_recall"))]
        lines.append("|"+f"[{label}]({sub}/summary.json)|"+"|".join(f"{n:.2f}" for n in nums)+"|")
    lines+=["","三种子集成使用同折三份权重，单权重对照也重新校准到同一个折内平均误报目标。跨架构融合使用同折同种子的一份 M0 与一份 nnU-Net。两者均在概率上取均值，再按训练误报目标定一个阈值。","",
        "## 配对得失","",
        "下表均为较低误报档。差值单位为百分点；区间按唯一图像配对重抽样 10,000 次，先在每图内对三个种子平均。集成相对各种子的差值共享一份集成输出。","",
        "|变化|细召回差及 95% 区间|三种子差值|提高/持平/下降图数|获益且邻域误报未增图数|邻域误报差/千|",
        "|---|---:|---|---:|---:|---:|"]
    comparisons=[("M0 增强输入","m0_contrast_transfer"),("nnU-Net 增强输入","contrast_cycle"),
        ("仅两倍视图增强","hybrid_scale_cycle"),("三种子集成相对共同校准的单权重","seed_agreement_cycle"),
        ("跨架构融合相对增强 M0","cross_family_cycle/m0"),("跨架构融合相对增强 nnU-Net","cross_family_cycle/nnunet")]
    comparisons += [(x.replace("_vs_"," 相对 ").replace("_dual","，双尺度"),"thin_training/"+x) for x in
                    ("B_vs_A_dual","C_vs_B_dual","D_vs_C_dual","D_vs_B_dual","B_vs_correction_dual","C_vs_correction_dual","D_vs_correction_dual")]
    for name,sub in comparisons:
        s=read(sub+"/summary.json")["m0matched"];t=s["thin_d1"];fp=s["near_fp_density"]
        wins=sum(v>1e-8 and fp["by_image"][k]<=1e-8 for k,v in t["by_image"].items())
        seeds=" / ".join(f"{t['by_seed'][str(i)]:+.2f}" for i in (2027,2028,2029))
        lines.append(f"|[{name}]({sub}/summary.json)|{t['delta']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]|{seeds}|"
                     f"{t['up']} / {t['images']-t['up']-t['down']} / {t['down']}|{wins}|{fp['delta']:+.2f}|")
    lines+=["","## 实现与解释边界","",
        "对比增强在原生 RGB 上完成。绿色通道减去其局部高斯均值（标准差 5 像素），差值乘 0.5、限幅至 ±8 灰度级，然后同量加回三个通道。背景支持从 RGB 计算。增强双尺度将增强后的原图及两倍图分别预测、回到原网格后等权平均。受试版本没有利用留出病灶标注进行推理。","",
        "A–D 均由对应折、种子相同的普通 800 步 nnU-Net 续训 400 步。三份挖掘器各在互不重叠的 18 张图上从同一 FIVES M0 起点普通微调 240 步，各自校准后只用于未见图。每份外层模型只使用本折 36 张训练图所构成的两份交叉难例清单；不使用另外 18 张留出图。图块保留原生上下文。A 使用随机与普通血管采样；B 仅替换部分中心为距挖掘器预测超过 5 像素的窄骨架片段；C 与 B 逐批共享图块，在基础损失上加权 0.05 的困难正例和环内负例平均 BCE；D 保留 C 的正例系数，移除辅助负例，基础分割损失保留。D 在看到部分 C 组开发结果后追加。","",
        "图像配对区间未校正多种探索方法的筛选；同一开发队列已被反复使用，各种子及各折不能当成外部验证。897 个既有 nnU-Net 远漏点是按历史输出选出的诊断子集，不能用该子集给不同架构排名或宣布临床改善。官方测试集封存。","",
        "[训练权重与校验](thin_training/verification.json) · [nnU-Net 的 RGB 重放](contrast_cycle/rgb_replay.json) · [U-Net 的 RGB 重放](m0_contrast_transfer/rgb_replay.json) · [预测入口](predict_contrast.py) · [困难片段图册](far_fragments/index.html)"]
    timing=read("cross_family_cycle/timing.json")
    med=timing["median_ms"]
    lines+=["","## 如何使用这些结果","",
        "在较低误报档，两种单模型在不增加网络前向次数的情况下都从温和对比增强中受益。两者都对远处同宽细点有改善，项目可以报告病灶邻域真实受益，空间特异性仍缺少证据。跨架构等权融合的细召回在两档均数值最高；相对增强 M0，低误报档 22 图提高、10 图持平、19 图下降，其中 18 张获益图的局部误报增加。融合在旧 nnU-Net 的 897 个远漏点上每种子命中 33、20、21 个，而增强 M0 命中 74、62、65 个。二者的取舍应同时考虑总体召回和局部故障类型。","",
        "A–D 同预算试验中，困难采样的增量很小，窄骨架正负辅助项及移除辅助负例的变体均未超过既有纠错权重的总体双尺度细召回。本轮可复现的净收益来自推理改动。","",
        f"在 {timing['gpu']} 上用第 0 折种子 2027 的 18 张 1024×1024 留出图，预热后交错测量两遍。从解码后的 RGB 到 CPU 二值掩膜，增强 M0、增强 nnU-Net 和跨架构融合的单张中位数为 {med['m0']:.0f}、{med['nnunet']:.0f}、{med['pair']:.0f} 毫秒。实际部署硬件需另测。[逐图计时及复算检查](cross_family_cycle/timing.json)","",
        "项目根目录下使用下列命令可复用对应折的研究模型；输入为 1024×1024 的 RGB 文件。用 m0 或 nnunet 可切换单模型，用 matched 可切换较高误报工作点。","","```powershell",
        "& 'outputs/nnunet_scale_20261121/env/Scripts/python.exe' -B 'outputs/nnunet_scale_20261121/predict_contrast.py' input.png output.png --fold 0 --seed 2027 --family pair --point m0matched",
        "```"]
    (ROOT/"细召回迭代对照.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
