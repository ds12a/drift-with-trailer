from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('/home/dshen/Code/drift-with-trailer/output/figures')
out.mkdir(parents=True, exist_ok=True)
environments = ['Asphalt', 'Snow', 'Slope', 'Snow + Slope']
controllers = ['Learned MPPI', 'Fiala-prior MPPI', 'LQR-PID', 'IPOPT MPC', 'MPCC']
# Table II: completed runs pooled across -30, -50, and +80 km/h (9 runs per speed).
counts = np.array([
    [27, 27, 25, 19],
    [20, 19, 23, 15],
    [20, 18, 21, 15],
    [14, 11, 16, 13],
    [4, 4, 5, 4],
])
assert counts.shape == (5, 4)
assert counts.sum(axis=1).tolist() == [98, 77, 74, 54, 17]

fig, ax = plt.subplots(figsize=(10.5, 5.1), dpi=220)
x = np.arange(len(environments))
width = 0.16
for i, controller in enumerate(controllers):
    positions = x + (i - 2) * width
    bars = ax.bar(positions, counts[i], width=width, label=controller, zorder=3)
    ax.bar_label(bars, padding=3, fontsize=9)

ax.set_ylim(0, 30.5)
ax.set_yticks([0, 9, 18, 27])
ax.set_ylabel('Completed runs', fontsize=12, labelpad=10)
ax.set_xticks(x, environments)
ax.set_xlabel('Environment', fontsize=12, labelpad=10)
ax.tick_params(axis='both', labelsize=11, length=0, pad=7)
ax.grid(axis='y', alpha=0.25, zorder=0)
for side in ['top','right','left']:
    ax.spines[side].set_visible(False)
ax.set_title('Completed runs by environment', loc='left', fontsize=18,
             fontweight='bold', pad=34)
ax.legend(ncol=5, frameon=False, loc='lower center',
          bbox_to_anchor=(0.5, 1.025), borderaxespad=0,
          fontsize=9.2, columnspacing=1.3, handlelength=1.4)
fig.text(0.105, 0.045,
         '27 trials per controller and environment · 3 target speeds × 9 runs · completion = 500 steps',
         fontsize=9.4)
fig.subplots_adjust(left=0.10, right=0.985, top=0.77, bottom=0.20)
for ext in ('png', 'svg'):
    fig.savefig(out / f'closed-loop-by-environment.{ext}', dpi=220,
                bbox_inches='tight', pad_inches=0.15)
plt.close(fig)
print('Saved default-style grouped bar chart')
