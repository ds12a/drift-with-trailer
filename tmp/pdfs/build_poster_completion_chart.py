import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out=Path('/home/dshen/Code/drift-with-trailer/tmp/pdfs/completion-poster.png')
labels=['Asphalt','Snow','Slope','Snow + Slope']
models=['Learned MPPI','Fiala prior','LQR-PID','IPOPT MPC']
counts=np.array([[27,27,25,19],[20,19,23,15],[20,18,21,15],[14,11,16,13]])
colors=['#1296d9','#ef7e03','#3ea000','#e0012c']
x=np.arange(4);width=.19
fig,ax=plt.subplots(figsize=(14.7,7.8),dpi=170)
for i,model in enumerate(models):
 bars=ax.bar(x+(i-1.5)*width,counts[i],width,label=model,color=colors[i])
 ax.bar_label(bars,padding=3,fontsize=17)
ax.set_ylim(0,31);ax.set_yticks([0,9,18,27]);ax.set_ylabel('Completed runs',fontsize=22)
ax.set_xticks(x,labels);ax.set_xlabel('Environment',fontsize=22,labelpad=9)
ax.tick_params(labelsize=20)
ax.grid(axis='y',alpha=.24);ax.set_axisbelow(True)
for s in ['top','right']:ax.spines[s].set_visible(False)
ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.14),frameon=False,fontsize=16,columnspacing=1.1,handlelength=1.25)
fig.subplots_adjust(left=.09,right=.985,top=.80,bottom=.18)
fig.savefig(out,facecolor='white',bbox_inches='tight',pad_inches=.12)
print(out)
