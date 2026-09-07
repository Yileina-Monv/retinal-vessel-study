"""Audit and describe the completed exploratory follow-up; no confirmatory p-values."""
import csv
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from retinal_batchstudy.run import read,dump,schedule_lr
from retinal_data.prepare import ROOT,sha256

NEW=ROOT/'runs/batch_supplement_v1/rtx5080'
OLD=ROOT/'runs/batch_study_v1/rtx5080'
DEST=ROOT/'batch_study/results/supplement_rtx5080'

def main():
    c=read(NEW/'config.json'); binding=read(NEW/'binding.json')
    for path,digest in binding.items():
        if path=='supplement_config':
            assert digest==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        else: assert sha256(ROOT/path)==digest,path
    results={}; trajectories=[]; checks=[]
    for seed in c['seeds']:
        signatures=None
        for arm in c['arms']:
            folder=NEW/f'seed_{seed}'/arm['id']
            result=read(folder/'result.json'); saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
            assert saved['binding']==binding and saved['config']==c
            assert saved['consumed']==8192 and saved['step']==8192//(arm['microbatch']*arm['accumulation'])
            assert result['draws']==saved['consumed'] and result['updates']==saved['step']
            assert result['initial_digest']==saved['initial_digest']
            assert len(saved['history'])==saved['step'] and len(saved['signatures'])==8192
            assert all(torch.isfinite(v).all() for v in saved['model'].values())
            effective=arm['microbatch']*arm['accumulation']
            assert all(h['lr']==schedule_lr(c,i*effective) for i,h in enumerate(saved['history']))
            digest=hashlib.sha256(json.dumps(saved['signatures'],sort_keys=True).encode()).hexdigest()
            assert result['sample_sequence_digest']==digest
            if signatures is None: signatures=saved['signatures']
            else: assert signatures==saved['signatures']
            old=torch.load(OLD/f'seed_{seed}/m2_a2_e4/last.pt',map_location='cpu',weights_only=True)
            assert old['initial_digest']==saved['initial_digest']
            assert old['signatures']==saved['signatures'][:2048]
            if effective==16:
                assert [h['lr'] for h in old['history']]==[h['lr'] for h in saved['history']]
            for draw in c['evaluation_draws']:
                evaluation=read(NEW/f'initial_eval_{seed}.json') if draw==0 else read(folder/f'eval_draw_{draw}.json')
                assert len(evaluation['rows'])==8
                if draw==8192: assert evaluation==result['final']
                trajectories.append(dict(seed=seed,arm=arm['id'],draw=draw,dice=evaluation['macro_dice']))
            checks.append(dict(seed=seed,arm=arm['id'],checkpoint_sha256=sha256(folder/'last.pt'),verified=True))
            results[(seed,arm['id'])]=result
            del saved,old
    summaries=[]
    for arm in c['arms']:
        entries=[results[(s,arm['id'])] for s in c['seeds']]
        dice=[e['final']['macro_dice'] for e in entries]
        differences=[results[(s,arm['id'])]['final']['macro_dice']-results[(s,'m2_a2_e4')]['final']['macro_dice'] for s in c['seeds']]
        summaries.append(dict(arm=arm['id'],dice=dice,mean_dice=float(np.mean(dice)),
            paired_deltas=differences,mean_delta=float(np.mean(differences)),
            training_seconds=sum(e['updates_seconds'] for e in entries),
            crops_per_second=sum(e['draws'] for e in entries)/sum(e['updates_seconds'] for e in entries)))
    equal_updates=[]
    for seed in c['seeds']:
        old=read(OLD/f'seed_{seed}/m2_a2_e4/result.json')
        new=results[(seed,'m2_a8_e16')]
        equal_updates.append(dict(seed=seed,old_e4_dice=old['final']['macro_dice'],new_e16_dice=new['final']['macro_dice'],
            delta=new['final']['macro_dice']-old['final']['macro_dice']))
    inference=read(NEW/'inference.json')
    assert len(inference['rows'])==3*8*5
    for seed in c['seeds']:
        assert inference['checkpoints'][str(seed)]==sha256(OLD/f'seed_{seed}/m2_a2_e4/last.pt')
        reference=read(OLD/f'seed_{seed}/m2_a2_e4/result.json')['final']
        observed=[r for r in inference['rows'] if r['seed']==seed and r['batch']==4]
        by_key={r['key']:r for r in reference['rows']}
        assert all(r['dice']==by_key[r['key']]['dice'] for r in observed)
    DEST.mkdir(parents=True,exist_ok=True)
    dump(DEST/'summary.json',dict(training=summaries,equal_updates=equal_updates,inference=inference['summary'],
        audit=checks,claim='Exploratory three paired training seeds, eight fixed development images; no significance or equivalence claims.'))
    with (DEST/'learning_curves.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['seed','arm','draw','dice']); writer.writeheader();writer.writerows(trajectories)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in c['arms']:
        draws=c['evaluation_draws']; mean=[]
        for draw in draws: mean.append(np.mean([r['dice'] for r in trajectories if r['arm']==arm['id'] and r['draw']==draw]))
        axes[0].plot(draws,mean,marker='o',label=arm['id'])
    axes[0].set(xlabel='Training crop draws',ylabel='Development macro Dice',title='Fresh long-budget training (mean of 3 seeds)');axes[0].legend()
    for i,row in enumerate(summaries):
        axes[1].scatter([i]*3,np.array(row['paired_deltas'])*100,s=55)
        axes[1].plot([i-.18,i+.18],[row['mean_delta']*100]*2,color='black')
    axes[1].axhline(0,color='gray',linestyle='--');axes[1].set_xticks(range(3),[r['arm'] for r in summaries])
    axes[1].set(ylabel='Dice difference vs e4 baseline (percentage points)',title='8192 draws: each dot is one paired seed')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(DEST/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# 5080 补充试验结果','',
        '完成推理 batch 对照及 3 组 × 3 种子长预算训练。仍是 32 张训练图、8 张开发图的探索；未使用官方测试集。',
        '', '## 同为 8192 样本的训练结果','',
        '| 布局 | 平均 Dice | 相对基准差/百分点 | 三个配对差/百分点 |', '|---|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['arm']} | {r['mean_dice']:.5f} | {r['mean_delta']*100:+.3f} | {', '.join(f'{x*100:+.3f}' for x in r['paired_deltas'])} |")
    lines+=['','## 同为 512 次参数更新的辅助对照','',
        '新 effective 16 使用 8192 样本；旧 effective 4 使用 2048 样本。初始化和逐更新学习率相同已核验，实际输入的共有前缀一致。样本暴露量不同，且实验先后执行，不是排除全部混杂的因果估计。','']
    for r in equal_updates: lines.append(f"- 种子 {r['seed']}：旧 e4={r['old_e4_dice']:.5f}，新 e16={r['new_e16_dice']:.5f}，差 {r['delta']*100:+.3f} 个百分点。")
    lines+=['','## 固定权重的推理 batch 对照','','计时为预热后的单次描述性测量；三种子模型 × 八张图，不视为 24 个独立患者。以原 batch 4 为参照。','',
        '| batch | 总秒数 | 峰值显存/GiB | 平均 Dice 差/百分点 | 最大概率差 | 二值翻转比例 |','|---|---:|---:|---:|---:|---:|']
    for r in inference['summary']:
        lines.append(f"| {r['batch']} | {r['seconds']:.2f} | {r['peak_gib']:.2f} | {r['mean_dice_delta']*100:+.6f} | {r['max_probability_delta']:.8f} | {r['flipped_fraction']:.8%} |")
    lines+=['','## 限制与核验','',
        '- 三个种子仅作探索，不宣称显著、等效或临床意义，不自动修改正式配置。',
        '- 9 个最终检查点、配置和代码摘要、初始化、跨组完整样本序列、评价结果及逐更新学习率均已核验。',
        '- 长短预算使用各自完整余弦计划；比较终点时同时改变了预算和学习率轨迹。',
        '- 4070 未参与，不能给出跨 GPU 结论；FP32 训练及 FOV 对照未在本轮执行。',
        '', '![学习曲线](learning_curves.png)']
    (DEST/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(training=summaries,equal_updates=equal_updates,inference=inference['summary']),indent=2))

if __name__=='__main__': main()
