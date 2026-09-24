from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('/home/dshen/Code/drift-with-trailer/output/figures')
out.mkdir(parents=True, exist_ok=True)
environments = ['Asphalt', 'Snow', 'Slope', 'Snow + Slope']
controllers = ['Learned MPPI', 'Fiala-prior MPPI', 'LQR-PID', 'IPOPT MPC', 'MPCC']
velocities = ['Reverse −30 km/h', 'Reverse −50 km/h', 'Forward +80 km/h']
# Table II: [velocity, controller, environment] completed runs, nine trials per cell.
counts = np.array([
    [[9,9,8,6], [5,5,8,4], [8,6,6,3], [9,5,7,4], [0,0,1,0]],
    [[9,9,8,4], [6,5,6,2], [6,6,6,3], [0,0,3,3], [0,0,1,0]],
    [[9,9,9,9], [9,9,9,9], [6,6,9,9], [5,6,6,6], [4,4,3,4]],
])
assert counts.shape == (3,5,4)
assert counts.sum(axis=(0,2)).tolist() == [98,77,74,54,17]
fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), dpi=220, sharex=True, sharey=True)
x = np.arange(4)
width = 0.16
for vi, ax in enumerate(axes):
    for ci, controller in enumerate(controllers):
        pos = x + (ci-2)*width
        ax.bar(pos, counts[vi,ci], width=width, label=controller, zorder=3)
    ax.set_ylim(0, 10.3)
    ax.set_yticks([0,3,6,9])
    ax.set_title(velocities[vi], loc='left', fontsize=12, fontweight='bold', pad=5)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.tick_params(axis='both', length=0, labelsize=10, pad=6)
    for side in ['top','right','left']:
        ax.spines[side].set_visible(False)
    if vi < 2:
        ax.spines['bottom'].set_visible(False)
axes[-1].set_xticks(x, environments)
axes[-1].tick_params(axis='x', labelbottom=True)
axes[-1].set_xlabel('Environment', fontsize=12, labelpad=8)
fig.supylabel('Completed runs', fontsize=12, x=0.035)
fig.suptitle('Completed runs by environment and target speed', fontsize=17, y=0.977)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, 0.938), fontsize=9.2, columnspacing=1.3, handlelength=1.4)
fig.text(0.5, 0.025, 'Nine runs per controller, environment, and speed · completion = 500 control steps',
         ha='center', fontsize=9.4)
fig.subplots_adjust(left=0.09, right=0.985, top=0.865, bottom=0.135, hspace=0.37)
for ext in ('png','svg'):
    fig.savefig(out/f'closed-loop-by-velocity.{ext}', dpi=220,
                bbox_inches='tight', pad_inches=0.15)
plt.close(fig)
print('Saved velocity-stratified PNG and SVG')
