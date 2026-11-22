"""Image-level cost of preserving all reference detections and a matched confidence control."""
import json
import numpy as np
from union_budget_cycle import OUT,ROOT
from report_view_ablation import read,at_eighteen

def main():
    groups={k:read(OUT/k/"summary.json") for k in ("union_vs_previous","lower_vs_previous","union_vs_lower")}
    complete=groups["union_vs_previous"];lower=groups["lower_vs_previous"];direct=groups["union_vs_lower"]
    base_anchors=read(OUT.parent/"audit.json")["per_image"]
    den={r["image"]:r["thin"] for r in base_anchors}
    lines=["# 保留原有命中时，新增的误报换到了什么","",
           "原四视图与 M0 仅两倍方案各沿用训练折确定的二值阈值。并集方案保留两张掩膜的全部命中。对照仅调整原融合的一个全图阈值，使其在训练图上达到并集的邻域误报密度；再分别评估同一批留出图。两者使用相同的 M0 专项 U-Net 与 1,200 步纠错 2D nnU-Net，每个模型只推理原尺寸和两倍各一次。","",
           "|原误报档与方案|细召回 %|邻域每千像素误报|直接命中 d0 %|细分支覆盖 %|Dice %|Precision %|全视野误报/千|远处召回 %|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pt,label in (("m0matched","较低档"),("matched","较高档")):
        for name,s,side in (("原四视图",complete,"base"),("仅放宽原阈值",lower,"candidate"),("两掩膜并集",complete,"candidate")):
            vals=[s[pt][k][side] for k in ("thin_d1","near_fp_density","thin_d0","path_d1","dice","precision","fp_fov","far_recall")]
            lines.append(f"|{label} · {name}|"+"|".join(f"{v:.2f}" for v in vals)+"|")
    extra={}
    lines+=["","|对照|细召回差及图像配对 95% 区间，百分点|种子差值|提高/持平/下降图数|邻域误报差/千|","|---|---:|---:|---:|---:|"]
    for pt,label in (("m0matched","较低档"),("matched","较高档")):
        for k,name in (("union_vs_previous","并集减原融合"),("lower_vs_previous","降阈值减原融合"),("union_vs_lower","并集减降阈值")):
            s=groups[k][pt];t=s["thin_d1"];f=s["near_fp_density"]
            seed="/".join(f"{t['by_seed'][str(i)]:+.2f}" for i in (2027,2028,2029))
            lines.append(f"|{label} · {name}|{t['delta']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]|{seed}|{t['up']}/{t['images']-t['up']-t['down']}/{t['down']}|{f['delta']:+.2f}|")
            x=t["by_image"];y=f["by_image"];vec=np.array(list(y.values()))
            best=max(x,key=x.get);worst=min(x,key=x.get)
            extra[pt+"_"+k]=dict(fp_quantiles=np.quantile(vec,[0,.25,.5,.75,1]).tolist(),best=best,worst=worst,
                 without_best=float(np.mean([v for i,v in x.items() if i!=best])),
                 improve_no_added_fp=sum(v>1e-8 and y[i]<=1e-8 for i,v in x.items()),
                 improve_with_added_fp=sum(v>1e-8 and y[i]>1e-8 for i,v in x.items()))
    s=complete["m0matched"];alt=direct["m0matched"];t=s["thin_d1"];fp=s["near_fp_density"]
    d=extra["m0matched_union_vs_previous"];q=d["fp_quantiles"]
    lines+=["",f"相对原低档，逐图新增邻域误报的中位数为每千像素 {q[2]:.2f} 个，四分位范围 {q[1]:.2f}–{q[3]:.2f}，最大 {q[4]:.2f}。召回提高的图中 {d['improve_with_added_fp']} 张局部误报增加，{d['improve_no_added_fp']} 张局部误报持平。两张掩膜取并集后，无一原有细点丢失。","",
            "|逐图示例（至少 20 点）|目标点|并集相对原融合的细召回差|邻域误报差/千|并集相对放宽阈值的细召回差|相对放宽阈值的误报差/千|","|---|---:|---:|---:|---:|---:|"]
    eligible=[k for k in t["by_image"] if den[k]>=20]
    for k in dict.fromkeys((max(eligible,key=t["by_image"].get),max(eligible,key=alt["thin_d1"]["by_image"].get),min(eligible,key=alt["thin_d1"]["by_image"].get))):
        lines.append(f"|{k}|{den[k]}|{t['by_image'][k]:+.2f}|{fp['by_image'][k]:+.2f}|{alt['thin_d1']['by_image'][k]:+.2f}|{alt['near_fp_density']['by_image'][k]:+.2f}|")
    if all(direct["m0matched"]["near_fp_density"][side]<18<direct["matched"]["near_fp_density"][side] for side in ("base","candidate")):
        mix=at_eighteen(OUT/"union_vs_lower")
        extra["expected_18"]=mix
        lines+=["",f"把各自已有的两档阈值作开发集随机混合，在相同期望 18 个邻域误报/千时，推算并集和放宽阈值的细召回为 {mix['candidate']:.2f}% 与 {mix['base']:.2f}%，差 {mix['delta']:+.2f} 个百分点 [{mix['ci'][0]:+.2f}, {mix['ci'][1]:+.2f}]。这项敏感性分析使用开发图求混合概率。"]
    lines+=["","[并集相对原融合的逐图数据](union_vs_previous/summary.json) · [放宽阈值相对原融合](lower_vs_previous/summary.json) · [两者直接配对](union_vs_lower/summary.json)"]
    (OUT/"配对误报对照.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (OUT/"cost_by_image.json").write_text(json.dumps(extra,indent=2))

if __name__=="__main__":main()
