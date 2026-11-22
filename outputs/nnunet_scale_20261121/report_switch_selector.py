"""Verify the nested holdouts and quantify image-level and pointwise switch effects."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from prepare import ROOT,split
import paired
from report_view_ablation import at_eighteen,read

OUT=ROOT/"switch_supervision"
DIRS={"base":OUT/"learned_add","regional":ROOT/"local_scale_gate/regional",**{k:OUT/k for k in
       ("learned_add","shuffled_add","all_add","learned_signed","shuffled_signed","zoom_control")}}
NAMES={"base":"原四视图融合","regional":"上一轮 RGB 区域切换","learned_add":"学习局部得失，仅添加",
       "shuffled_add":"打乱监督，仅添加","all_add":"无条件添加全部正向差","learned_signed":"学习局部得失，可增减",
       "shuffled_signed":"打乱监督，可增减","zoom_control":"全图 M0 仅两倍"}
SEEDS=(2027,2028,2029)

def paired_contrast(a,b):
    assert set(a)==set(b)
    changes={k:a[k]-b[k] for k in a}
    d=np.array(list(changes.values()))
    boots=np.random.default_rng(20261122).choice(d,(10000,len(d))).mean(1)
    best=max(changes,key=changes.get)
    return dict(delta=float(d.mean()),ci=np.percentile(boots,[2.5,97.5]).tolist(),
        up=int((d>1e-8).sum()),tie=int((np.abs(d)<=1e-8).sum()),down=int((d< -1e-8).sum()),
        best=best,without_best=float((d.sum()-changes[best])/(len(d)-1)))

def case_audit():
    items=paired.dataset()[0];stats=[];pointmaps={}
    for f in range(3):
        training=read(OUT/f"selector{f}/training.json");held=set(split(f)["dev"])
        assert set(training["scored"])==held
        for ex in training["examples"]:
            train=set(read(OUT/f"inner{ex['expert_group']}/training.json")["train"])
            assert not train&held and ex["image"] not in train|held
        for k in sorted(held):
            d=items[k];target=d["n"]&d["s"]
            hits={m:[] for m in ("base","zoom_control","learned_add","all_add","shuffled_add")}
            distances=[];perfect=[];union=[];seen_paths=set();high_base=[];high_union=[]
            for seed in SEEDS:
                with np.load(OUT/"heldout_masks"/f"fold{f}_{seed}"/(k+".npz")) as z:
                    reference_masks={}
                    for m in hits:
                        mask=np.unpackbits(z["m0matched_"+m]).reshape(target.shape).astype(bool)
                        if m in ("base","zoom_control"):reference_masks[m]=mask
                        hitmap=ndi.binary_dilation(mask,structure=paired.cross)
                        h=hitmap[target]
                        hits[m].append(h)
                        if m=="base":
                            distances.append(ndi.distance_transform_edt(~mask)[target])
                            seen_paths.update(np.unique(d["i"][hitmap&(d["i"]>0)]).tolist())
                    base_mask=reference_masks["base"];zoom_mask=reference_masks["zoom_control"]
                    wide=[np.unpackbits(z["matched_"+mode]).reshape(target.shape).astype(bool) for mode in ("base","zoom_control")]
                    high_base.append(ndi.binary_dilation(wide[0],structure=paired.cross)[target])
                    high_union.append(ndi.binary_dilation(wide[0]|wide[1],structure=paired.cross)[target])
                    for group,mask in ((perfect,base_mask|(zoom_mask&d["v"])),(union,base_mask|zoom_mask)):
                        hit=ndi.binary_dilation(mask,structure=paired.cross)[target]
                        added_fp=int((mask&~base_mask&~d["v"]&d["n"]).sum())
                        group.append(dict(hit=hit,new_fp=added_fp))
            hits={m:np.stack(h) for m,h in hits.items()}
            dist=np.stack(distances);base=hits["base"];zoom=hits["zoom_control"]
            current=hits["learned_add"]
            robust=~base.any(0)
            close=robust&(dist.max(0)<=2);far=robust&(dist.min(0)>5)
            recoverable=(zoom&~base).sum(0)>=2
            # The same seeds must show both the original recoverability and the new recovery.
            gained=(current&~base).sum(0)>=2;lost=(base&~current).sum(0)>=2
            def image_mask(pt):
                a=np.zeros_like(target);a[target]=pt;return a
            pointmaps[k]=(image_mask(gained),image_mask(lost))
            strata={label:dict(points=int(mask.sum()),rescued=int((mask&(current.sum(0)>=2)).sum()),
                    union_rescued=int((mask&(zoom.sum(0)>=2)).sum()),
                    higher_base_rescued=int((mask&(np.stack(high_base).sum(0)>=2)).sum()),
                    higher_union_rescued=int((mask&(np.stack(high_union).sum(0)>=2)).sum()),
                    true_rescued=int((mask&(np.stack([v["hit"] for v in perfect]).sum(0)>=2)).sum()))
                    for label,mask in (("close",close),("far",far),("other",robust&~(close|far)))}
            path_ids=d["i"][target];valid=far&(path_ids>0)
            far_image=image_mask(far)
            cc,_=ndi.label(far_image,structure=np.ones((3,3),bool))
            sizes=np.bincount(cc.ravel())[1:]
            geometry=dict(on_qualified_path=int(valid.sum()),on_zero_hit_path=int((valid&~np.isin(path_ids,list(seen_paths))).sum()),
                connected_ge_five=int(sizes[sizes>=5].sum()),clusters_ge_five=int((sizes>=5).sum()),
                singleton=int((sizes==1).sum()))
            rec={m:int(((hits[m]&zoom&~base).sum(0)>=2).sum()) for m in ("learned_add","all_add","shuffled_add")}
            at_risk=(base&~zoom).sum(0)>=2
            preserved={m:int(((hits[m]&base&~zoom).sum(0)>=2).sum()) for m in ("learned_add","all_add","shuffled_add")}
            support={}
            for key,group in (("true_vessel_additions",perfect),("union",union)):
                added=np.stack([v["hit"] for v in group])&~base
                support[key]=dict(delta=None if not target.any() else float(100*added.mean()),
                    robust_gains=int((added.sum(0)>=2).sum()),
                    fp_delta=None if not d["n"].any() else float(1000*np.mean([v["new_fp"] for v in group])/d["n"].sum()))
            assert all(v["new_fp"]==0 for v in perfect)
            stats.append(dict(image=k,thin=int(target.sum()),gain=int(gained.sum()),loss=int(lost.sum()),support=support,
                    recoverable=int(recoverable.sum()),recovered=rec,at_risk=int(at_risk.sum()),preserved=preserved,
                    baseline_consistent_missed=int(robust.sum()),strata=strata,far_geometry=geometry))
    assert len(stats)==54 and sum(r["thin"] for r in stats)==3632
    return stats,items,pointmaps

def gallery(stats,items,maps):
    font=ImageFont.truetype("msyh.ttc",17)
    canvas=Image.new("RGB",(630,602),"white");draw=ImageDraw.Draw(canvas)
    draw.text((8,4),"左：原始 RGB 局部。右：青圈新增命中，橙圈新增漏点。",font=font,fill="black")
    draw.text((8,29),"至少两个种子同向变化；所示区域按变化点集中程度选取。",font=font,fill="black")
    used=[]
    for idx,key in enumerate(("gain","loss")):
        row=max((r for r in stats if r["image"] not in used),key=lambda r:r[key]);k=row["image"]
        if not row[key]:continue
        used.append(k);masks=maps[k];mode=masks[idx];size=96
        integral=np.pad(mode.astype(int),((1,0),(1,0))).cumsum(0).cumsum(1)
        density=integral[size:,size:]-integral[:-size,size:]-integral[size:,:-size]+integral[:-size,:-size]
        peaks=np.argwhere(density==density.max());y,x=peaks[len(peaks)//2]
        pts=np.argwhere(mode[y:y+size,x:x+size])+[y,x]
        y,x=np.clip(np.rint(pts.mean(0)-(size-1)/2),np.maximum(pts.max(0)-size+1,0),np.minimum(pts.min(0),np.array(mode.shape)-size)).astype(int)
        crop=Image.fromarray(items[k]["rgb"][y:y+size,x:x+size]).resize((236,236),Image.Resampling.NEAREST)
        overlay=crop.copy();pen=ImageDraw.Draw(overlay)
        for arr,color in zip(masks,("#31ffff","#ff9f22")):
            for yy,xx in np.argwhere(arr[y:y+size,x:x+size]):
                u,v=(xx+.5)*236/size,(yy+.5)*236/size
                pen.ellipse((u-2,v-2,u+2,v+2),outline=color)
        yy=54+270*idx
        draw.text((8,yy),f"{k}　全图目标 {row['thin']}，稳定新增 {row['gain']}，稳定丢失 {row['loss']}",font=font,fill="black")
        canvas.paste(crop,(8,yy+27));canvas.paste(overlay,(256,yy+27))
    canvas.save(OUT/"paired_examples.png")

def main():
    summaries={k:read(v/"summary.json") for k,v in DIRS.items()}
    stats,items,maps=case_audit()
    gallery(stats,items,maps)
    lines=["# 在出血邻域学习何时保留两倍图的细血管证据","",
           "Refined IDRiD 开发集 54 图，3 折 × 3 个外层种子，图内先平均，再跨图等权。主指标为出血外缘 11 像素内、宽度代理 ≤3 像素的细骨架点 d1 召回，51 图、3,632 个点。d1 允许上下左右一格，d0 要求覆盖原点。邻域误报在 53 图上按全体邻域像素计，全图保护指标使用 54 图。","",
           "外层固定专项微调 240 步的 U-Net M0 和错点监督训练 1,200 步的 2D nnU-Net。各在增强后的原尺寸图和两倍图上推理，等权形成基准。所有新规则只改变如何融合已有的四张概率图。另在每个 18 图内层组重训一对分割器，在外层训练集另一半未见图上提取切换后救回、丢掉和误报的标签。外层选择器各用 36 图训练，按训练折误报目标定阈值，再在留出的 18 图测量。内层只训练一个种子，外层使用三个既有种子。公开测试集未用于本轮实验，患者层面双眼关系未知。[训练协议及对照](PROTOCOL.md)","",
           "|规则|细召回 %|邻域误报/千|高档细召回 %|高档误报/千|d0 %|细分支覆盖 %|Dice %|Precision %|全视野误报/千|远处召回 %|",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    import os
    for k,s in summaries.items():
        side="base" if k=="base" else "candidate";lo=s["m0matched"];hi=s["matched"]
        vals=[lo["thin_d1"],lo["near_fp_density"],hi["thin_d1"],hi["near_fp_density"],lo["thin_d0"],
              lo["path_d1"],lo["dice"],lo["precision"],lo["fp_fov"],lo["far_recall"]]
        link=Path(os.path.relpath(DIRS[k]/"summary.json",OUT)).as_posix()
        lines.append(f"|[{NAMES[k]}]({link})|"+"|".join(f"{v[side]:.2f}" for v in vals)+"|")
    lines+=["","把增强 U-Net 的原尺寸与两倍概率记为 $u_1,u_2$，nnU-Net 的为 $n_1,n_2$。原融合 $B=(u_1+u_2+n_1+n_2)/4$。保留 nnU-Net 双尺度而让 M0 仅用两倍，相当于给 $B$ 加上 $D=(u_2-u_1)/4$。学习添加输出 $B+g\\max(D,0)$，学习增减输出 $B+gD$；$g$ 是由 RGB 与局部预测估计的权重，取值在零到一之间。无条件添加取 $g=1$ 并只接受正差值。每个方案最终都用自己的训练折阈值二值化。"]
    contrasts={}
    lines+=["","## 直接配对","",
            "差值单位为百分点，邻域误报另按每千像素计算。区间按图配对重抽样 10,000 次，包含同图三个种子的相关性。开发集已重复使用。","",
            "|较低误报档对照|细召回差及 95% 区间|三种子差值|提高/持平/下降图数|去掉最大获益图后的差值|邻域误报差/千|",
            "|---|---:|---:|---:|---:|---:|"]
    for a,b in (("learned_add","base"),("learned_add","regional"),("learned_add","shuffled_add"),("learned_add","all_add"),
                ("all_add","base"),("learned_signed","base"),("learned_signed","shuffled_signed")):
        both={}
        for pt in ("m0matched","matched"):
            aa=summaries[a][pt];bb=summaries[b][pt]
            side="base" if b=="base" else "candidate"
            ya=aa["thin_d1"]["by_image"];yb={k:0. for k in ya} if b=="base" else bb["thin_d1"]["by_image"]
            c=paired_contrast(ya,yb)
            c["fp_delta"]=aa["near_fp_density"]["candidate"]-bb["near_fp_density"][side]
            c["seeds"]=[aa["thin_d1"]["by_seed"][str(i)]-(0 if b=="base" else bb["thin_d1"]["by_seed"][str(i)]) for i in SEEDS]
            both[pt]=c
            if pt=="m0matched":
                lines.append(f"|{NAMES[a]} 减 {NAMES[b]}|{c['delta']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]|"
                        +"/".join(f"{v:+.2f}" for v in c["seeds"])+f"|{c['up']}/{c['tie']}/{c['down']}|{c['without_best']:+.2f}（{c['best']}）|{c['fp_delta']:+.2f}|")
        contrasts[a+"__"+b]=both
    mixed={k:at_eighteen(p) for k,p in DIRS.items() if k!="base"}
    lines+=["","## 相同期望误报的敏感性分析","",
            "沿用两个预先校准阈值，在开发图上求随机混合比例，使期望邻域误报为每千像素 18 个。表中为该混合的期望值，用于识别工作点差异；部署时仍使用训练折确定的阈值。","",
            "|方法|期望细召回 %|相对原融合差及区间，百分点|","|---|---:|---:|"]
    for k,r in mixed.items():lines.append(f"|{NAMES[k]}|{r['candidate']:.2f}|{r['delta']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]|")
    lines+=["","同样的宽度与 d1 容差下，以下比较出血外缘和远处的变化，只取两处都能评分的同一批图。","",
            "|方案相对原四视图|邻域窄骨架变化|远处窄骨架变化|两处变化之差及配对区间，百分点|","|---|---:|---:|---:|"]
    for k in ("learned_add","learned_signed","all_add","regional"):
        lo=summaries[k]["m0matched"];n=lo["thin_d1"];f=lo["far_thin_d1"]
        co=paired_contrast(n["by_image"],f["by_image"])
        lines.append(f"|{NAMES[k]}|{n['delta']:+.2f}|{f['delta']:+.2f}|{co['delta']:+.2f} [{co['ci'][0]:+.2f}, {co['ci'][1]:+.2f}]|")
    total=lambda key:sum(r[key] for r in stats)
    lines+=["","## 实际点数与持续漏检","",
            f"对同一物理位置按种子匹配。相对原融合，至少两个种子同向变化的结果为新增 {total('gain')} 个点、丢掉 {total('loss')} 个点。基准漏掉、M0 全图仅两倍方案能命中的点，在至少两个种子重复出现的共有 {total('recoverable')} 个。这些点中，要求相同种子也被候选救回，学习添加命中 {sum(r['recovered']['learned_add'] for r in stats)} 个，无条件添加命中 {sum(r['recovered']['all_add'] for r in stats)} 个，打乱监督后添加命中 {sum(r['recovered']['shuffled_add'] for r in stats)} 个。点数统计与按图等权主指标分母不同。","",
            "下表仅含四视图基准在三个种子均漏掉的目标点。距离在原生网格上按欧氏距离测量，救回要求至少两个种子命中。","",
            "|基准预测离标注点的距离|点数|学习添加救回|二值并集救回|并集只加入真实血管像素时救回（诊断）|","|---|---:|---:|---:|---:|"]
    for k,label in (("close","三个种子均 ≤2 像素"),("other","其余距离或种子间距离不同"),("far","三个种子均 >5 像素")):
        lines.append(f"|{label}|"+"|".join(str(sum(r['strata'][k][field] for r in stats)) for field in ("points","rescued","union_rescued","true_rescued"))+"|")
    geometry={k:sum(r["far_geometry"][k] for r in stats) for k in stats[0]["far_geometry"]}
    lines+=["",f"当前基准的三种子远漏点中，有 {geometry['connected_ge_five']} 个落在至少 5 个相邻窄骨架点组成的连续片段，共 {geometry['clusters_ge_five']} 个这样的八邻接片段；另有 {geometry['singleton']} 个孤立远漏点。有 {geometry['on_qualified_path']} 个远漏点落在已登记的合格分支上，其中 {geometry['on_zero_hit_path']} 个点所属的环内分支在三个种子里都没有一个 d1 命中。这些数量描述标注中心线与模型预测的几何关系。"]
    lines+=["",f"放宽到已有较高误报档，在这同一批远漏点上，原融合有 {sum(r['strata']['far']['higher_base_rescued'] for r in stats)} 个点在至少两个种子中恢复 d1 命中；较高档两掩膜并集有 {sum(r['strata']['far']['higher_union_rescued'] for r in stats)} 个。[连续远漏、有机会救回与孤立点的原图局部](far_examples.png)"]
    contrast=read(OUT/"local_contrast.json")["summary"]
    lines+=["",f"原始绿通道的局部对比以周围 2–5 像素环带内背景中位数减中心像素值衡量。给远漏点匹配同图、同一出血灶、相同宽度代理、距病灶边缘相差至多 2 像素、空间相距至多 64 像素的三种子稳定命中点。每个成功对照仅用一次，先最大化匹配数再最小化距离。{contrast['total_far']} 个远漏点中，{contrast['pairs']} 个在 {contrast['images']} 张图有对照。按图等权，远漏点平均对比为 {contrast['miss']:.2f} 灰度级，对照为 {contrast['hit']:.2f}，差值 {contrast['delta']:+.2f}，按图重抽样 95% 区间 [{contrast['ci'][0]:+.2f}, {contrast['ci'][1]:+.2f}]。可配对的子集支持低局部对比与漏检相关。其余远漏点的定义、可辨程度和漏检原因仍需分别核查。[一对一匹配点与逐图统计](local_contrast.json)"]
    lines+=["",f"反向核对，原融合能命中、全图 M0 仅两倍会丢掉，而且至少两个种子一致的目标有 {total('at_risk')} 个。要求在同样的种子仍命中，学习添加保住 {sum(r['preserved']['learned_add'] for r in stats)} 个，无条件添加保住 {sum(r['preserved']['all_add'] for r in stats)} 个，打乱监督后保住 {sum(r['preserved']['shuffled_add'] for r in stats)} 个。","",
            "添加规则逐像素取原融合概率加上一项非负增量。在原阈值下，它能保留全部基准命中。表格使用每个方案为匹配误报重新校准的阈值，因此添加规则仍会丢掉原有细点。","",
            "## 现有预测里还有多少可用信号","",
            "固定两个已按低档误报校准的二值掩膜，完整保留基准，再添加 M0 仅两倍方案提出的像素。第一行用人工血管标签从新增像素中挑出全部真实血管，是这两个固定掩膜中只添加真实血管像素可达到的上限。第二行将两张掩膜直接取并集。两行均未重新调阈值，和在概率图上融合后再做误报校准的无条件添加有区别。","",
            "|从 M0 仅两倍掩膜中添加|主指标增量，百分点|邻域每千像素新增误报|重复救回点数|","|---|---:|---:|---:|"]
    for key,label in (("true_vessel_additions","仅真实血管，使用标注的诊断"),("union","全部新增像素，直接并集")):
        sub=[r["support"][key] for r in stats]
        lines.append(f"|{label}|{np.mean([r['delta'] for r in sub if r['delta'] is not None]):+.2f}|{np.mean([r['fp_delta'] for r in sub if r['fp_delta'] is not None]):+.2f}|{sum(r['robust_gains'] for r in sub)}|")
    lines+=["",
            "## 推理复核",""]
    timing=read(OUT/"timing.json")
    lines+=[f"{timing['gpu']} 在折 0、种子 2027 的 18 张图上，预热后随机交错重复两遍。从解码后的 RGB 到阈值掩膜，单图中位数为原融合 {timing['median_ms']['base']:.0f} 毫秒、RGB 区域规则 {timing['median_ms']['regional']:.0f} 毫秒、学习添加 {timing['median_ms']['learned_add']:.0f} 毫秒、无条件添加 {timing['median_ms']['all_add']:.0f} 毫秒。{len(timing['rows'])} 次输出分别对照归档，在视野内逐像素复核原融合和两种添加规则，RGB 区域规则复核全部归档评分。[逐图时间](timing.json) · [RGB 推理入口](../predict_switch.py)"]
    lines+=["","[局部得失图](paired_examples.png) · [逐图计数、直接配对与相同期望误报结果](audit.json)"]
    lines+=["","## 本轮判断","",
            "真实点位监督的添加规则为 54.56%，打乱监督后的添加规则为 54.64%。允许选择器增减概率时两者均为 55.16%。当前结果无法把收益归因于学会辨别该救或该丢的局部位置。若接受更多邻域误报，应比较同等预算下的普通降阈值与两张掩膜并集。[相近误报预算的三折三种子直接对照](union_budget/配对误报对照.md)","",
            "下一种训练目标值得聚焦到连续且稳定的远漏细段。原融合在三个种子均离标注超过 5 像素的点上，即使允许较高误报并合并两张掩膜，能重复救回的仍很少。可先在原图上复核连续段的可辨性，再以完整片段及同病灶可见细段组织训练和评估，并保留逐图邻域误报对照。"]
    (OUT/"audit.json").write_text(json.dumps(dict(per_image=stats,contrasts=contrasts,matched_18=mixed),indent=2))
    (OUT/"局部尺度选择结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":main()
