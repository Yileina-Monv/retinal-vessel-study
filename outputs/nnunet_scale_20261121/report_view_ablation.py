"""Paired display of both FP budgets, seed counts, small denominators and reference misses."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def spatial_report(diag, counts):
    root=ROOT/"local_scale_gate"
    audits=[r for fold in range(3) for r in read(root/f"gate{fold}"/"audit.json")["rows"]]
    assert len({r["image"] for r in audits})==len(audits)==54
    coverage={k:100*np.mean([r[k] for r in audits if r[k] is not None]) for k in ("fov","near","thin","rotated_thin")}
    regional=read(root/"regional/summary.json")["m0matched"]
    rotated=read(root/"rotated/summary.json")["m0matched"]
    contrasts={}
    for metric in ("thin_d1","near_fp_density","far_recall","dice"):
        a=regional[metric]["by_image"];b=rotated[metric]["by_image"]
        assert set(a)==set(b)
        delta=np.array([a[k]-b[k] for k in sorted(a)])
        boots=np.random.default_rng(20261122).choice(delta,(10000,len(a))).mean(1)
        contrasts[metric]={"delta":delta.mean(),"ci":np.percentile(boots,[2.5,97.5]).tolist()}
    diag["regional_check"]={"coverage":coverage,"regional_minus_rotated":contrasts}
    c=contrasts["thin_d1"]
    lines=["","## 把尺度切换限制在疑似出血区域","",
           "每折用 36 张训练图的出血标注及外环训练一个小型颜色和纹理分类器。在留出图上，它从 RGB 输入生成平滑权重。疑似病灶内逐渐切换到 M0 仅两倍、nnU-Net 双尺度，其他位置保留原双尺度融合概率。每个方案仍在训练图上单独定阈值，因此区域外的最终二值结果也可能变化。旋转 180° 的区域保留原形状与面积，用作位置对照。人工标签区域只用于定位诊断。",
           "",f"按图平均，RGB 区域权重占视野 {coverage['fov']:.1f}%，出血外环内平均为 {coverage['near']:.1f}%，目标细点处为 {coverage['thin']:.1f}%；旋转后目标细点处为 {coverage['rotated_thin']:.1f}%。",
           "",f"低误报档直接比较 RGB 区域和旋转区域，细召回差为 {c['delta']:+.2f} 个百分点，逐图配对 95% 区间 [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]。"]
    by=regional["thin_d1"]["by_image"]
    for k in (max(by,key=by.get),min(by,key=by.get)):
        lines.append(f"- {k} 有 {counts[k]} 个目标点，相对原双尺度融合的细召回变化为 {by[k]:+.2f} 个百分点，邻域每千像素误报变化为 {regional['near_fp_density']['by_image'][k]:+.2f}。")
    best=sorted(by,key=by.get,reverse=True)
    remainder=[np.mean([v for k,v in by.items() if k not in best[:n]]) for n in (1,2)]
    lines+=["",f"在按图等权的平均差中，{best[0]} 占净增量的 {100*by[best[0]]/sum(by.values()):.1f}%。去掉这张图，剩余图的增益为 {remainder[0]:+.2f} 个百分点；再去掉 {best[1]}，剩余图为 {remainder[1]:+.2f} 个百分点。"]
    return lines

def timing_report():
    stats=read(ROOT/"appearance_fusion_cycle/timing.json")
    times=stats["median_ms"]
    lines=["","## 单图推理实测","",
           f"使用 {stats['gpu']}，折 0、种子 2027，18 张原生 1024 像素方图，预热后随机交错测量两轮。计时从内存中的 RGB 到输出二值掩膜，包含增强、尺度变换、滑窗和融合。模型加载与读写文件在计时区间外。",
           "","|推理组合|中位毫秒/图|相对四视图用时变化|","|---|---:|---:|"]
    names={"full":"增强双尺度四视图","eight":"未增强和增强八视图","m0_supported":"M0 双尺度共同支持","regional":"RGB 区域切换尺度","large_dual":"M0 两倍、nnU-Net 双尺度","large_native":"M0 两倍、nnU-Net 原尺度"}
    for k,v in times.items():
        lines.append(f"|{names[k]}|{v:.0f}|{100*(v/times['full']-1):+.1f}%|")
    lines+=["",f"对照归档的逐图评分，共复核 {len(stats['rows'])} 个计时输出，出现任一评分差异的次数为 {stats['comparisons_with_difference']}。[计时逐图数据](appearance_fusion_cycle/timing.json)"]
    return lines

def at_eighteen(group):
    # Descriptive randomized mixture. Thresholds are unchanged; the mixing probability
    # is estimated on held-out images and is not a deployment calibration.
    values=defaultdict(list)
    for f in sorted(group.glob("fold*.json")):
        for r in read(f)["rows"]:
            for arm in ("base","candidate"):
                rec=r[arm]
                values[(r["point"],arm,r["image"])].append(
                    [np.nan if rec["thin_d1"] is None else 100*rec["thin_d1"],
                     np.nan if rec["near_fp_density"] is None else 1000*rec["near_fp_density"]])
    names=sorted({k[2] for k in values})
    # image, budget, arm, metric
    arr=np.array([[[np.mean(values[(pt,arm,name)],axis=0) for arm in ("base","candidate")]
                   for pt in ("m0matched","matched")] for name in names])
    def mix(mean):
        low,high=mean[...,0,:,:],mean[...,1,:,:]
        w=(18-low[...,1])/(high[...,1]-low[...,1])
        scores=low[...,0]+w*(high[...,0]-low[...,0])
        return scores,w
    scores,w=mix(np.nanmean(arr,axis=0));assert ((w>=0)&(w<=1)).all()
    draws=np.random.default_rng(20261122).integers(len(names),size=(10000,len(names)))
    sims,weights=mix(np.nanmean(arr[draws],axis=1))
    valid=((weights>=0)&(weights<=1)).all(1)
    ci=np.quantile((sims[:,1]-sims[:,0])[valid],[.025,.975])
    return dict(base=scores[0],candidate=scores[1],delta=scores[1]-scores[0],ci=ci.tolist(),
                valid_bootstraps=int(valid.sum()),weights=w.tolist())

def main():
    counts={r["image"]:r["thin_points"] for r in read(ROOT/"thin_recall_audit.json")}
    families={"native":"原图","large":"仅两倍图","dual":"双尺度"}
    specs=[("既有增强双尺度，等权融合","appearance_fusion_cycle/eight","base")]
    for k,label in [("raw","两架构原有双尺度，未增强"),("eight","未增强与增强都保留，八组等权"),
                    ("m0_two_thirds","增强双尺度，M0 占三分之二"),("m0_supported","增强双尺度，仅在 M0 两尺度共同支持时增加其权重")]:
        specs.append((label,"appearance_fusion_cycle/"+k,"candidate"))
    specs.extend([("增强双尺度，等权平均对数几率","evidence_scale_cycle/odds","candidate"),
                  ("增强双尺度，训练误报阈值对齐后平均对数几率","evidence_scale_cycle/aligned_odds","candidate")])
    for m,ms in families.items():
        for n,ns in families.items():
            if m!=n or m!="dual":
                specs.append((f"M0 {ms}，nnU-Net {ns}","family_scale_ablation/"+m+"_"+n,"candidate"))
    for k,label in (("regional","RGB 疑似病灶内 M0 改用两倍"),("rotated","相同权重图旋转 180°（位置对照）"),("annotation_diagnostic","人工出血标注及外环内 M0 改用两倍（诊断用）")):
        specs.append((label,"local_scale_gate/"+k,"candidate"))
    lines=["# 出血边缘细召回：输入外观、模型权重与尺度消融","",
        "在 54 张 Refined IDRiD 开发图上沿用 3 折、3 种子。51 张图、3,632 个窄骨架点进入主指标。出血外缘 11 像素环内、宽度代理 ≤3 像素的骨架点，只要预测覆盖该点或上下左右一格即算命中。先算每图比例，再按图等权。53 张图可算邻域误报，分母是全体邻域像素。54 张图可算全图指标。","",
        "两种分割网络的权重固定。U-Net M0 从 FIVES 预训练权重专项微调 240 步；2D nnU-Net 使用既有 1,200 步纠错监督版本。每折每种子只用训练图校准阈值，区域分类器也只在训练图拟合。参照为两种网络各看增强后的原生及两倍图，在原生坐标上平均四张概率图。增强把绿通道与局部平滑值之差的一半限于 ±8 个灰度级，加回 RGB。不同尺度消融下两个模型仍各占一半，同一模型的可用尺度在模型内等权。[既有模型训练明细](模型训练与推理详细对照.md)","",
        "|方案|低档细召回 %|低档邻域误报/千|高档细召回 %|高档邻域误报/千|低档细分支覆盖 %|Dice %|Precision %|远处召回 %|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    detail=[]
    geometry=[]
    matched=[]
    diag={}
    for label,sub,side in specs:
        group=ROOT/sub
        summ=read(group/"summary.json");lo=summ["m0matched"];hi=summ["matched"]
        cols=[s[k][side] for s,k in [(lo,"thin_d1"),(lo,"near_fp_density"),(hi,"thin_d1"),(hi,"near_fp_density"),
                                     (lo,"path_d1"),(lo,"dice"),(lo,"precision"),(lo,"far_recall")]]
        lines.append(f"|[{label}]({sub}/summary.json)|"+"|".join(f"{v:.2f}" for v in cols)+"|")
        if side=="base": continue
        diag[sub]={}
        common=at_eighteen(group)
        diag[sub]["matched_18_diagnostic"]=common
        matched.append(f"|{label}|{common['candidate']:.2f}|{common['delta']:+.2f} [{common['ci'][0]:+.2f}, {common['ci'][1]:+.2f}]|")
        def interval(per_image):
            vec=np.array(list(per_image))
            sim=np.random.default_rng(20261122).choice(vec,(10000,len(vec))).mean(1)
            return [float(np.mean(vec)),*np.percentile(sim,[2.5,97.5]).tolist()]
        d0=lo["thin_d0"]["by_image"];d1=lo["thin_d1"]["by_image"];far=lo["far_thin_d1"]["by_image"]
        assert set(d0)==set(d1)==set(far)
        shell=interval([d1[k]-d0[k] for k in d1])
        specific=interval([d1[k]-far[k] for k in d1])
        diag[sub]["geometry"]={"neighbor_only_delta":shell,"near_minus_far_thin_delta":specific}
        geometry.append(f"|{label}|{lo['thin_d0']['delta']:+.2f}|{shell[0]:+.2f}|{lo['thin_d1']['delta']:+.2f}|{lo['far_thin_d1']['delta']:+.2f}|{specific[0]:+.2f} [{specific[1]:+.2f}, {specific[2]:+.2f}]|")
        for key in ("m0matched","matched"):
            t=summ[key]["thin_d1"]; fp=summ[key]["near_fp_density"]["by_image"]
            diffs=t["by_image"]
            wins_no_fp=sum(v>1e-8 and fp[k]<=1e-8 for k,v in diffs.items())
            v=np.array(list(diffs.values()))
            loo=(v.sum()-v)/(len(v)-1)
            seed_counts=[]
            for seed in (2027,2028,2029):
                rs=[r for f in group.glob(f"fold*_{seed}.json") for r in read(f)["rows"] if r["point"]==key]
                assert len(rs)==54
                seed_counts.append({k:sum(r[k] for r in rs) for k in ("m0_only","m0_rescued","hard_base","hard_cand","hard_gained","hard_lost","gains","losses")})
            diag[sub][key]=dict(seed_counts=seed_counts,wins_no_added_fp=wins_no_fp,
                leave_one_out_range=[float(loo.min()),float(loo.max())],
                delta_at_least_ten_points=float(np.mean([d for k,d in diffs.items() if counts[k]>=10])))
            if key=="m0matched":
                seed_str=" / ".join(f"{t['by_seed'][str(k)]:+.2f}" for k in (2027,2028,2029))
                hard_str=" / ".join(str(r["hard_cand"]) for r in seed_counts)
                rescue_str=" / ".join(str(r["m0_rescued"]) for r in seed_counts)
                detail.append(f"|{label}|{t['delta']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]|{seed_str}|"
                    f"{t['up']}/{t['images']-t['up']-t['down']}/{t['down']}|{wins_no_fp}|{hard_str}|{rescue_str}|")
    lines+=["","## 较低误报档的逐图与逐点变化","",
        "括号内为同图配对 95% 重抽样区间，先在每图内平均三个种子。困难点是此前按 nnU-Net 筛出的 897 点；同位置在不同种子重复评分，不能当成独立样本。M0 独有点指增强 M0 命中且既有融合漏掉、同时属于这 897 点的位置。","",
        "|方法|细召回差，百分点|种子差值|提高/持平/下降图数|获益且局部误报未增加图数|困难点命中数（分种子）|找回 M0 独有点（分种子）|",
        "|---|---:|---|---:|---:|---|---|",*detail,"",
        "M0 双尺度共同支持规则使用两个 M0 尺度概率的较小值。该值超过 nnU-Net 双尺度概率时，在原融合概率上增加二者差值的一半。规则覆盖全图，无需病灶输入。各组分别校准到同一训练误报目标，留出误报允许出现偏差。","",
        "各方案重新评分时均对照既有 54.02% 流程的归档逐图指标。八组指两种输入外观 × 两个尺度 × 两个模型；它的网络前向工作量为既有方案的两倍。去掉一个尺度所节省的时间取决于该尺度的窗口数，不能用视图数直接换算延迟。此队列已反复用于开发，探索性比较与多个区间尚不能代替独立患者验证。","",
        "[逐方案敏感性、逐种子救回和丢失](appearance_fusion_cycle/diagnostics.json)"]
    lines+=["","## 相同期望误报的敏感性分析","",
        "固定已有低、高两档阈值，考虑按固定概率随机选择两档中的一档，使该开发队列的期望邻域误报为每千像素 18 个。每种方法单独求混合概率，以下为期望召回；没有测量一个新的确定性阈值。在图像配对重抽样中重算混合概率，区间仅使用两臂均覆盖 18 的样本。部署比较仍以上述训练折校准结果为准。","",
        "|方法|18 个误报/千时的期望细召回 %|相对既有融合，百分点及区间|","|---|---:|---:|",*matched]
    lines+=spatial_report(diag,counts)
    lines+=["","## 放大后，中心线命中发生在哪里","",
            "d0 要求覆盖参考点本身。d1 另允许覆盖四邻域。两者使用相同分母，因而 d1 的变化可以拆成直接覆盖变化和仅靠一格容差命中的净变化。远处同宽细点沿用同一宽度定义与容差，仅取也有邻域目标的同批 51 图。下表均为较低误报档相对原融合的百分点变化。","",
            "|方法|直接覆盖 d0|仅邻格命中|邻域 d1|远处同宽 d1|邻域增量减远处同宽增量及 95% 区间|","|---|---:|---:|---:|---:|---:|",*geometry]
    zoom=read(ROOT/"family_scale_ablation/large_dual/summary.json")["m0matched"]
    lines+=["","M0 仅看两倍、nnU-Net 保留双尺度时的逐图得失。以下列出至少 20 个目标点的图中增益及损失最大者，另列一张召回提高、误报减少的图。数值为三种子的图内平均。","",
            "|图像|目标点|d1 变化，百分点|d0 变化，百分点|邻域误报变化/千|","|---|---:|---:|---:|---:|"]
    for k in ("IDRiD_18","IDRiD_05","IDRiD_54"):
        lines.append(f"|{k}|{counts[k]}|{zoom['thin_d1']['by_image'][k]:+.2f}|{zoom['thin_d0']['by_image'][k]:+.2f}|{zoom['near_fp_density']['by_image'][k]:+.2f}|")
    lines+=["","[三张图的原始局部与逐点得失](family_scale_ablation/large_dual/paired_crops.png)。每个圈标出在至少两个种子中同向改变命中状态的参考点。框选 96×96 像素窗口，正例与反例分别取新增命中或新增漏点最集中的区域。图展示变化位置，不能估计全队列发生频率。[全图与框内实际点数及坐标](family_scale_ablation/large_dual/paired_crops.json)"]
    lines+=timing_report()
    lines+=["","## 当前判断","",
            "相近邻域误报下，两种模型都只看原生尺度时细召回 41.12%，原生加两倍达到 54.02%。在原有四组增强预测上再平均四组未增强预测，主指标降至 52.65%。两倍视图贡献明确，继续平均更多外观没有显示收益。",
            "",
            "M0 全图改为仅两倍后，897 个历史困难点中的命中分别从 33/20/21 增至 84/85/63；全体目标直接覆盖 d0 从 41.50% 降到 39.49%，远处所有血管像素召回从 77.63% 降到 71.79%。空间限制把远处召回保留在 76.52%，同时三个种子的困难点命中仍达到 55/44/32。RGB 区域候选的 d1 为 55.00%，比基准高 0.98 个百分点，图像配对区间为 −0.43 至 +2.62，种子差值为 −0.33、+1.71、+1.58。高误报档两者仅差 0.13 个百分点。按开发图的两档工作点插值至相同期望误报时，候选的差值为 +0.46 个百分点。",
            "",
            "下一轮优先检验原融合漏掉、两倍 M0 仍能给出证据的局部细点。保留两组已训练分割权重，用训练图的折外预测标记两类位置：切换尺度能补回的点，以及切换会丢点或把背景误报成血管的位置。只训练一个读取局部预测与 RGB 的选择器，和现有四视图、简单 RGB 区域规则同时比较。主结果继续报告 d1、邻域误报与逐图得失，同时核对 d0 和远处召回。现有开发图已被反复使用；独立证据需要先冻结方法及阈值选择流程，再在未参与开发的患者上评估。"]
    (ROOT/"appearance_fusion_cycle/diagnostics.json").write_text(json.dumps(diag,indent=2))
    (ROOT/"双视图与融合消融对照.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__": main()
