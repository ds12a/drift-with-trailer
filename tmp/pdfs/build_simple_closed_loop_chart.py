from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path('/home/dshen/Code/drift-with-trailer/output/figures')
out.mkdir(parents=True, exist_ok=True)
controllers = ['Learned MPPI', 'Fiala-prior MPPI', 'LQR-PID', 'IPOPT MPC', 'MPCC']
# Completed-run counts from Table II: four environments × three target speeds × nine trials.
completed_by_environment = np.array([
    [27, 27, 25, 19],
    [20, 19, 23, 15],
    [20, 18, 21, 15],
    [14, 11, 16, 13],
    [4, 4, 5, 4],
])
counts = completed_by_environment.sum(axis=1)
assert counts.tolist() == [98, 77, 74, 54, 17]
total = 4 * 3 * 9
rates = 100 * counts / total

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 13,
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
})
fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=220)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
y = np.arange(len(controllers))
colors = ['#008C88'] + ['#94A8B7'] * 4
ax.barh(y, rates, height=0.58, color=colors, edgecolor='none', zorder=3)
ax.set_yticks(y, controllers)
ax.invert_yaxis()
ax.tick_params(axis='y', length=0, pad=10)
ax.tick_params(axis='x', colors='#536676', labelsize=11, length=0)
ax.set_xlim(0, 112)
ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(['0','25','50','75','100%'])
ax.grid(axis='x', color='#E2E8EC', linewidth=0.8, zorder=0)
ax.spines['bottom'].set_color('#B4C3CC')
for yi, (r, n) in enumerate(zip(rates, counts)):
    ax.text(r+1.5, yi, f'{n}/{total}', va='center', fontsize=12,
            fontweight='bold' if yi == 0 else 'normal', color='#18364A')
ax.set_title('Completed runs across all test conditions', loc='left',
             fontsize=19, fontweight='bold', color='#00274C', pad=22)
fig.text(0.21, 0.04,
         'Four environments · three target speeds · nine runs each. Completion = 500 control steps.',
         fontsize=10.5, color='#536676')
fig.subplots_adjust(left=0.27, right=0.95, top=0.82, bottom=0.19)
for ext in ('png','svg'):
    fig.savefig(out / f'closed-loop-baseline-comparison.{ext}',
                dpi=220, facecolor='white', bbox_inches='tight', pad_inches=0.18)
plt.close(fig)
print('Saved PNG and SVG')
