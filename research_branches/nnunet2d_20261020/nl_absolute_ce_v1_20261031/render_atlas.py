"""Display accepted CPU replay crops and targets; engineering evidence, never medical review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,html,csv,hashlib,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('${RETINA_PROJECT_ROOT}/experiments/nnunet2d_branch_20261020')
STAGE=BASE/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v1_20261031'
OUT=STAGE/'results'

def main():
    summary=json.loads((OUT/'cpu_summary.json').read_text())
    selected=json.loads((OUT/'atlas_selection.json').read_text())
    figs=OUT/'figures';figs.mkdir(exist_ok=False)
    entries=[]
    for case in selected['cases']:
        with np.load(case['path'],allow_pickle=False) as a:
            rgb=a['image'].transpose(1,2,0);gt=a['GT']
        # Per-channel display stretch only, never fed back into data or supervision.
        low=np.percentile(rgb,1,axis=(0,1));high=np.percentile(rgb,99,axis=(0,1))
        display=np.clip((rgb-low)/np.maximum(high-low,1e-6),0,1)
        fig,axes=plt.subplots(1,3,figsize=(18,5.8))
        axes[0].imshow(display);axes[0].set_title('Actual augmented RGB (display stretch only)')
        axes[1].imshow(gt,cmap='gray',vmin=0,vmax=1);axes[1].set_title('Actual highest-resolution training GT')
        axes[2].imshow(display);axes[2].set_title('Selected positive targets + / negative targets x')
        groups=case['selected']
        for j,g in enumerate(groups):
            y,x=g['positive'];axes[2].scatter([x],[y],s=120,c='#00ff66',marker='+',linewidths=1.7)
            axes[2].annotate(str(j+1),(x,y),color='#00ff66',fontsize=9)
            for ny,nx in g['negatives']:axes[2].scatter([nx],[ny],s=50,c='#ff44ff',marker='x',linewidths=1.1)
        for ax in axes:ax.set_axis_off()
        fig.suptitle(f"{case['category']} | seed {case['seed']}, attempt {case['attempt']}, crop {case['crop']}\n"
                     f"{case['image']} | visible {case['counts']['visible']}, selected {case['counts']['selected']} | Engineering evidence only")
        fig.tight_layout();path=figs/(case['category']+'.png');fig.savefig(path,dpi=130);plt.close(fig)
        # Additional zoom makes individual one-pixel targets legible while retaining full-crop context.
        zooms=[]
        if groups:
            g=groups[0];y,x=g['positive'];rad=40
            x0,x1=max(0,x-rad),min(gt.shape[1],x+rad+1);y0,y1=max(0,y-rad),min(gt.shape[0],y+rad+1)
            f,ax=plt.subplots(1,3,figsize=(12,4))
            ax[0].imshow(display[y0:y1,x0:x1]);ax[0].set_title('RGB detail: first selected identity')
            ax[1].imshow(gt[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1);ax[1].set_title('GT detail')
            ax[2].imshow(gt[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=1)
            ax[2].scatter([x-x0],[y-y0],s=200,c='#00ff66',marker='+')
            for ny,nx in g['negatives']:ax[2].scatter([nx-x0],[ny-y0],s=100,c='#ff44ff',marker='x')
            ax[2].set_title('Green + positive / magenta x negative')
            for a in ax:a.set_axis_off()
            f.tight_layout();zpath=figs/(case['category']+'_zoom.png');f.savefig(zpath,dpi=150);plt.close(f);zooms=[zpath.name]
        entries.append({**case,'figure':path.name,'zooms':zooms})
    existing={e['category'] for e in entries};missing=[x for x in selected['all_categories'] if x not in existing]
    out=['<!doctype html><meta charset="utf-8"><title>NL CPU 工程图册</title>',
         '<style>body{font:16px sans-serif;max-width:1600px;margin:30px auto;background:#f3f4f6;color:#18202a}img{max-width:100%;background:white}section{background:white;padding:20px;margin:25px 0}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>',
         '<h1>NL 绝对分类监督：CPU 工程审核图册</h1>',
         '<p>这些图来自真实 Train CPU 增强重放，不含模型预测。左：实际增强 RGB（仅显示拉伸）；中：实际训练 GT；右：绿色 + 为选中正例，紫色 × 为选中负例。细节图显示第一选中目标身份，按固定规则生成。</p>',
         '<p><b>仅供几何与实现核查，不是医学人工验证。</b>不要求用户确认标注医学正确。原人工记录 0/105、0/26 不变。</p>',
         '<p>选例：每种预定类型的首次合格 crop，最多 12 例；不存在的类型保留 NA。没有按训练效果挑图。</p>',
         '<p>CPU结论：'+html.escape(summary['status'])+'；GPU 工程与正式训练保持 HOLD。</p>',
         '<p>缺失类型（NA）：'+html.escape(', '.join(missing) or '无')+'</p>',
         '<p>核对问题：目标是否与实际 GT 同步？负例是否落在 GT 背景？边界与空间变换是否一致？缺失／冲突／cap 是否按规则记录？</p>']
    for e in entries:
        out.append('<section><h2>'+html.escape(e['category'])+'</h2><p>'+html.escape(f"seed {e['seed']} / attempt {e['attempt']} / crop {e['crop']} / {e['image']}")+'</p>')
        out.append('<img src="figures/'+e['figure']+'">')
        for z in e['zooms']:out.append('<img src="figures/'+z+'">')
        out.append('</section>')
    (OUT/'atlas.html').write_text('\n'.join(out),encoding='utf-8')
    (OUT/'atlas_manifest.json').write_text(json.dumps({'selection_rule':selected['rule'],'cases':entries,'NA_categories':missing},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# NL 绝对分类辅助监督：CPU 一页结论','',f"实际结论：{summary['status']}。GPU 工程与科学训练均未放行。",'',
           '| seed | 已核对尝试 | 非空辅助批 | 保留/可见实例 | 曝光 Neff | 结论 |','|---|---:|---:|---:|---:|---|']
    for s in summary['seeds']:
        lines.append(f"| {s['seed']} | {s['attempts']} | {s['nonempty_batches']} | {s['selected']}/{s['visible']} ({s['retained_fraction']:.2%}) | {s['Neff']:.3f} | {s['status']} |")
    lines+=['',f"合成公式/梯度/几何测试 {summary['formula_tests']} 项；独立核验结果见 independent_review.json。",'',
            '保留全部110图、351原生目标（逐seed记录）；空图、NA与不利结果不删除。',
            'Neff表示Loss系数曝光，不是实际参数梯度贡献。RGB图仅用于几何工程核查。',
            '只检查固定前缀，未验证完整科学轨迹；真实网络梯度、AMP、lambda定标尚未执行。',
            '该候选无NR，不支持病灶位置特异性；不证明标注医学正确、完整分支或临床连接恢复。',
            '新增模型/GPU前向、参数更新及Test访问均0；CPU交付后停止，交Sol独立审查。']
    (OUT/'CPU_一页结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'atlas_cases':len(entries),'NA_categories':missing,'figures':len(list(figs.glob('*.png')))},ensure_ascii=False))

if __name__=='__main__':main()
