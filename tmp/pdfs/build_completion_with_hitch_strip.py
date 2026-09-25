"""Single-panel poster chart with reverse hitch-angle advantage as a text strip."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path('/home/dshen/Code/drift-with-trailer/output/figures')
out.mkdir(parents=True, exist_ok=True)
labels = ['Asphalt', 'Snow', 'Slope', 'Snow + Slope']
models = ['Learned MPPI', 'Fiala prior', 'LQR-PID', 'IPOPT MPC', 'MPCC']
counts = np.array([
    [27, 27, 25, 19], [20, 19, 23, 15], [20, 18, 21, 15],
    [14, 11, 16, 13], [4, 4, 5, 4],
])
# Table II: equal-sized nine-run groups at -30 and -50 km/h, all runs included.
learned_hitch = np.array([[1.14, 1.75], [1.18, 1.82], [1.03, 3.06], [10.82, 15.41]]).mean(axis=1)
fiala_hitch = np.array([[5.25, 6.70], [7.82, 8.49], [2.86, 6.48], [11.47, 15.86]]).mean(axis=1)
reduction = fiala_hitch - learned_hitch

fig, ax = plt.subplots(figsize=(12.8, 6.6), dpi=220)
x = np.arange(4)
width = .16
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
for i, (model, color) in enumerate(zip(models, colors)):
    bars = ax.bar(x + (i-2)*width, counts[i], width, label=model, color=color, zorder=3)
    ax.bar_label(bars, padding=2, fontsize=10)

ax.set_ylim(0, 31)
ax.set_yticks([0, 9, 18, 27])
ax.set_ylabel('Completed runs', fontsize=14)
ax.set_xlim(-.55, 3.55)
ax.set_xticks(x, labels)
ax.tick_params(axis='x', labelsize=12, pad=8)
ax.tick_params(axis='y', labelsize=11)
ax.grid(axis='y', alpha=.24, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.18),
          fontsize=10.5, columnspacing=1.3, handlelength=1.25)

# Text annotations are aligned with the bar groups; they do not create a second plot.
fig.text(.095, .145, 'Reverse hitch RMS improvement vs Fiala prior',
         fontsize=11, weight='bold', color='0.28')
for xi, delta in zip(x, reduction):
    px = ax.transData.transform((xi, 0))[0]
    fx = fig.transFigure.inverted().transform((px, 0))[0]
    fig.text(fx, .087, f'{delta + 1e-8:.1f}° lower', ha='center',
             fontsize=12, weight='bold', color=colors[0])
fig.text(.095, .025,
         'Completion: 27 trials per environment, all speeds. Hitch: mean over 18 reverse trials (-30/-50 km/h), including failures.',
         fontsize=9.5, color='0.4')
fig.subplots_adjust(left=.095, right=.99, top=.83, bottom=.30)
for ext in ('png', 'svg'):
    fig.savefig(out / f'closed-loop-with-hitch-strip.{ext}', bbox_inches='tight', pad_inches=.15)
plt.close(fig)
print(out / 'closed-loop-with-hitch-strip.png')
