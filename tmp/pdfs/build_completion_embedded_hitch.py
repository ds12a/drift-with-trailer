"""Single grouped bar chart with hitch-angle values embedded in comparable MPPI bars."""
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
# Table II: nine runs at each reverse target speed, including failures.
reverse_hitch = np.array([
    [[1.14, 1.75], [1.18, 1.82], [1.03, 3.06], [10.82, 15.41]],
    [[5.25, 6.70], [7.82, 8.49], [2.86, 6.48], [11.47, 15.86]],
]).mean(axis=2)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
fig, ax = plt.subplots(figsize=(12.8, 5.75), dpi=220)
x = np.arange(4)
width = .16
for i, (model, color) in enumerate(zip(models, colors)):
    positions = x + (i-2)*width
    bars = ax.bar(positions, counts[i], width, label=model, color=color, zorder=3)
    ax.bar_label(bars, padding=2, fontsize=11)
    if i < 2:
        for j, xi in enumerate(positions):
            # Hitch values sit in the bar but do not use the completion y-scale.
            ax.text(xi, 2.6, f'{reverse_hitch[i,j]:.1f}°', ha='center', va='center',
                    color='white', fontsize=10, weight='bold', zorder=4)

ax.set_ylim(0, 31)
ax.set_yticks([0, 9, 18, 27])
ax.set_ylabel('Completed runs', fontsize=14)
ax.set_xlim(-.55, 3.55)
ax.set_xticks(x, labels)
ax.set_xlabel('Environment', fontsize=14, labelpad=8)
ax.tick_params(labelsize=11)
ax.grid(axis='y', alpha=.24, zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.21),
          fontsize=10.6, columnspacing=1.3, handlelength=1.25)
fig.text(.095, .036,
         'Bar height: completed runs (27 trials/environment, all speeds). '
         'Numbers inside blue/orange bars: mean reverse RMS hitch angle in degrees '
         '(18 trials; lower is better).',
         fontsize=9.8, color='0.35')
fig.subplots_adjust(left=.095, right=.99, top=.78, bottom=.22)
for ext in ('png', 'svg'):
    fig.savefig(out / f'closed-loop-embedded-hitch.{ext}', bbox_inches='tight', pad_inches=.15)
plt.close(fig)
print(out / 'closed-loop-embedded-hitch.png')
