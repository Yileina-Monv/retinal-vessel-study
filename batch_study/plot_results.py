"""Static scientific figure from completed paired-seed results only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'batch_study/results/rtx5080/summary.json'
report=json.loads(path.read_text(encoding='utf-8'))
if report['status']!='complete_single_gpu':
    raise ValueError('Plot requires the complete prespecified matrix')
arms=report['arms']
contrasts=report['contrasts']
labels=[a['arm'].replace('_',' / ') for a in arms]
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
colors=['#273c75','#168a79','#cf7641','#7954a1','#3f91b2','#a55b73']
for i,arm in enumerate(arms):
    values=list(arm['seed_dice'].values())
    axes[0].scatter(np.linspace(i-.12,i+.12,len(values)),values,color=colors[i],s=28,alpha=.8)
    axes[0].plot([i-.2,i+.2],[np.mean(values)]*2,color='black',lw=2)
axes[0].set_xticks(range(len(arms)),labels,rotation=45,ha='right')
axes[0].set_ylabel('Development image-macro Dice')
axes[0].set_title('Eight matched training seeds per arm')
for i,row in enumerate(contrasts):
    mean=row['mean_difference']*100
    low,high=np.array(row['simultaneous_ci_t'])*100
    axes[1].errorbar(mean,i,xerr=[[max(0,mean-low)],[max(0,high-mean)]],fmt='o',color=colors[i+1],capsize=4)
axes[1].axvspan(-1,1,color='#168a79',alpha=.10)
axes[1].axvline(0,color='gray',lw=1)
axes[1].axvline(-1,color='#168a79',ls='--',lw=1)
axes[1].axvline(1,color='#168a79',ls='--',lw=1)
axes[1].set_yticks(range(len(contrasts)),[r['arm'] for r in contrasts])
axes[1].set_xlabel('Dice difference vs m2_a2_e4 (percentage points)')
axes[1].set_title('Bonferroni simultaneous t intervals')
axes[2].barh(range(len(arms)),[a['mean_draws_per_second'] for a in arms],color=colors)
axes[2].set_yticks(range(len(arms)),labels)
axes[2].set_xlabel('Training crops / second, including input audit')
axes[2].set_title('Observed RTX 5080 throughput')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('M0 batch sensitivity | 32 train / 8 development | 2,048 crops per run | not convergence evidence',fontsize=12)
fig.savefig(path.parent/'batch_sensitivity.png',dpi=180)
plt.close(fig)
print(path.parent/'batch_sensitivity.png')
