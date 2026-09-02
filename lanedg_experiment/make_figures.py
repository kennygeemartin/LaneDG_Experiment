import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).parent; O=R/'figures'; O.mkdir(exist_ok=True); A=json.loads((R/'results/experiment_results.json').read_text())
D=('tusimple','culane','bdd100k'); L=('TuSimple','CULane','BDD100K'); M=('precision','recall','f1','iou','map50','map50_95'); ML=('Precision','Recall','F1','IoU','mAP@0.5','mAP@0.5:0.95'); C=('#2878b5','#43a047','#ef8a17')
def mat(m): return np.array([[A['matrix'][s][t][m] for t in D] for s in D])
def save(f,n): f.tight_layout(); f.savefig(O/n,dpi=220,bbox_inches='tight'); f.savefig(O/(Path(n).stem+'.pdf'),bbox_inches='tight'); plt.close(f)
def heat(ax,a,title):
 im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis'); ax.set_xticks(range(3),L,rotation=20); ax.set_yticks(range(3),L); ax.set(title=title,xlabel='Target dataset',ylabel='Training source')
 for i in range(3):
  for j in range(3): ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',weight='bold',color='white' if a[i,j]<.55 else 'black')
 return im
f,axs=plt.subplots(2,3,figsize=(13,8))
for m,l,ax in zip(M,ML,axs.flat): f.colorbar(heat(ax,mat(m),l),ax=ax,fraction=.046,pad=.04)
save(f,'metric_matrix_overview.png')
for m,l in zip(M,ML):
 f,ax=plt.subplots(figsize=(5.5,4.5)); f.colorbar(heat(ax,mat(m),f'{l}: frozen source-to-target evaluation'),ax=ax); save(f,f'heatmap_{m}.png')
ind=np.array([np.diag(mat(m)).mean() for m in M]); zero=np.array([mat(m)[~np.eye(3,dtype=bool)].mean() for m in M]); x=np.arange(6)
f,ax=plt.subplots(figsize=(10,5)); ax.bar(x-.18,ind,.36,label='In-domain (diagonal)',color=C[0]); ax.bar(x+.18,zero,.36,label='Zero-shot (off-diagonal)',color=C[2]); ax.set_xticks(x,ML,rotation=15); ax.set(ylim=(0,1),ylabel='Score',title='Dataset-shift effect: in-domain versus zero-shot'); ax.legend(); ax.grid(axis='y',alpha=.25); save(f,'in_domain_vs_zero_shot.png')
gap=ind-zero; f,ax=plt.subplots(figsize=(9,4.7)); b=ax.bar(ML,gap,color=['#d1495b' if v>=0 else C[1] for v in gap]); ax.axhline(0,color='black',lw=.8); ax.set(ylabel='In-domain âˆ’ zero-shot',title='Generalization gap by metric'); ax.tick_params(axis='x',rotation=15); ax.bar_label(b,fmt='%.3f'); ax.grid(axis='y',alpha=.25); save(f,'generalization_gap.png')
TM=('f1','iou','map50','map50_95'); TL=('F1','IoU','mAP@0.5','mAP@0.5:0.95'); trans=np.array([[np.mean([A['matrix'][s][t][m] for t in D if t!=s]) for m in TM] for s in D]); x=np.arange(4)
f,ax=plt.subplots(figsize=(9,5));
for i,s in enumerate(D): ax.bar(x+(i-1)*.24,trans[i],.24,label=L[i],color=C[i])
ax.set_xticks(x,TL); ax.set(ylim=(0,1),ylabel='Mean zero-shot score',title='Source-domain transferability'); ax.legend(title='Training source'); ax.grid(axis='y',alpha=.25); save(f,'source_transferability.png')
incoming=np.array([[np.mean([A['matrix'][s][t][m] for s in D if s!=t]) for m in ('f1','iou','map50')] for t in D]); f,ax=plt.subplots(figsize=(8.5,5)); x=np.arange(3)
for j,l in enumerate(('F1','IoU','mAP@0.5')): ax.bar(x+(j-1)*.24,incoming[:,j],.24,label=l)
ax.set_xticks(x,L); ax.set(ylim=(0,1),ylabel='Mean incoming zero-shot score',title='Target-domain difficulty under dataset shift'); ax.legend(); ax.grid(axis='y',alpha=.25); save(f,'target_domain_difficulty.png')
pairs=[(s,t) for s in D for t in D if s!=t]; f,ax=plt.subplots(figsize=(11,5)); x=np.arange(6); ax.bar(x-.18,[A['matrix'][s][t]['f1'] for s,t in pairs],.36,label='F1'); ax.bar(x+.18,[A['matrix'][s][t]['iou'] for s,t in pairs],.36,label='IoU'); ax.set_xticks(x,[f'{L[D.index(s)]}â†’{L[D.index(t)]}' for s,t in pairs],rotation=25,ha='right'); ax.set(ylim=(0,1),ylabel='Score',title='All six frozen zero-shot transfers'); ax.legend(); ax.grid(axis='y',alpha=.25); save(f,'all_zero_shot_pairs.png')
f,axs=plt.subplots(1,2,figsize=(12,4.5))
for i,d in enumerate(D):
 h=A['histories'][d]; e=[r['epoch'] for r in h]; axs[0].plot(e,[r['train_loss'] for r in h],marker='o',color=C[i],label=L[i]); axs[1].plot(e,[r['val_loss'] for r in h],marker='o',color=C[i],label=L[i])
for ax,title in zip(axs,('Training loss','Validation loss')): ax.set(title=title,xlabel='Epoch',ylabel='Composite loss'); ax.grid(alpha=.25); ax.legend()
save(f,'training_convergence.png')
f,axs=plt.subplots(1,2,figsize=(10,4)); b=axs[0].bar(L,[A['wall_times_seconds'][d] for d in D],color=C); axs[0].bar_label(b,fmt='%.1f s'); axs[0].set(title='Training wall time',ylabel='Seconds'); b=axs[1].bar(L,[A['thresholds'][d] for d in D],color=C); axs[1].bar_label(b,fmt='%.2f'); axs[1].set(title='Validation-tuned confidence',ylabel='Threshold',ylim=(0,1)); save(f,'training_runtime_and_thresholds.png')
f,ax=plt.subplots(figsize=(12,5)); ax.axis('off')
for i,l in enumerate(L):
 y=.82-i*.31; ax.text(.07,y,f'{l}\ntraining split',ha='center',va='center',bbox=dict(boxstyle='round',fc=C[i]),color='white',weight='bold'); ax.annotate('',(.27,y),(.14,y),arrowprops=dict(arrowstyle='->')); ax.text(.36,y,f'LaneDG trained on\n{l} only',ha='center',va='center',bbox=dict(boxstyle='round',fc='#eceff1')); ax.annotate('',(.58,y),(.47,y),arrowprops=dict(arrowstyle='->')); ax.text(.75,y,'Frozen evaluation on\nTuSimple | CULane | BDD100K',ha='center',va='center',bbox=dict(boxstyle='round',fc='#fff3e0'))
ax.text(.5,.02,'No target fine-tuning â€¢ No domain adaptation â€¢ Identical training configuration',ha='center',weight='bold'); ax.set_title('Cross-dataset experimental design',weight='bold',size=15); save(f,'methodology_experimental_design.png')
with (R/'results/metric_matrix.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['source','target','evaluation_type',*M]); [w.writerow([s,t,'in-domain' if s==t else 'zero-shot',*[A['matrix'][s][t][m] for m in M]]) for s in D for t in D]
summary={'in_domain_mean':dict(zip(M,ind.tolist())),'zero_shot_mean':dict(zip(M,zero.tolist())),'generalization_gap':dict(zip(M,gap.tolist())),'source_zero_shot_mean':{s:dict(zip(TM,trans[i].tolist())) for i,s in enumerate(D)},'target_incoming_zero_shot_mean':{t:dict(zip(('f1','iou','map50'),incoming[i].tolist())) for i,t in enumerate(D)}}; (R/'results/methodology_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

