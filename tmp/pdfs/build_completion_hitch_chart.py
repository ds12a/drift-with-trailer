"""Poster chart from Table II of Learning Dynamics for High-Speed Tractor-Trailer Control."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path('/home/dshen/Code/drift-with-trailer/output/figures')
out.mkdir(parents=True, exist_ok=True)

environments = ['Asphalt', 'Snow', 'Slope', 'Snow + Slope']
controllers = ['Learned MPPI', 'Fiala prior', 'LQR-PID', 'IPOPT MPC', 'MPCC']
# Completed runs across -30, -50, and +80 km/h; 27 trials per controller/environment.
completed = np.array([
    [27, 27, 25, 19],
    [20, 19, 23, 15],
    [20, 18, 21, 15],
    [14, 11, 16, 13],
    [4, 4, 5, 4],
])
# Table II mean per-run RMS hitch angles (degrees), for -30 and -50 km/h.
# Each cell is based on nine runs, so the mean of the two speeds pools 18 runs.
hitch_reverse_by_speed = np.array([
    [[1.14, 1.75], [1.18, 1.82], [1.03, 3.06], [10.82, 15.41]],
    [[5.25, 6.70], [7.82, 8.49], [2.86, 6.48], [11.47, 15.86]],
    [[4.80, 14.41], [5.36, 8.97], [10.02, 10.58], [16.56, 16.91]],
    [[3.11, 54.93], [20.22, 39.96], [16.32, 28.09], [20.79, 30.59]],
    [[26.57, 14.74], [39.16, 24.95], [30.46, 17.98], [31.79, 25.63]],
])
hitch_reverse = hitch_reverse_by_speed.mean(axis=2)
assert completed.sum(axis=1).tolist() == [98, 77, 74, 54, 17]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, figsize=(12.8, 8.7), dpi=220, sharex=True,
    gridspec_kw={'height_ratios': [1.25, 1], 'hspace': .28},
)
x = np.arange(4)
width = .16
for i, (name, color) in enumerate(zip(controllers, colors)):
    xi = x + (i - 2) * width
    bars = ax_top.bar(xi, completed[i], width, label=name, color=color, zorder=3)
    ax_top.bar_label(bars, padding=2, fontsize=10)
    ax_bottom.scatter(xi, hitch_reverse[i], s=95, color=color,
                      edgecolors='white', linewidths=.7, zorder=4)

ax_top.set_ylim(0, 31)
ax_top.set_yticks([0, 9, 18, 27])
ax_top.set_ylabel('Completed runs', fontsize=14)
ax_top.tick_params(axis='x', bottom=False, labelbottom=False)
ax_top.legend(ncol=5, frameon=False, loc='upper center',
              bbox_to_anchor=(.5, 1.19), fontsize=10.6,
              columnspacing=1.3, handlelength=1.25)

ax_bottom.set_ylim(0, 36)
ax_bottom.set_yticks([0, 10, 20, 30])
ax_bottom.set_ylabel('Reverse RMS hitch angle (deg)', fontsize=14)
ax_bottom.set_xticks(x, environments)
ax_bottom.set_xlabel('Environment', fontsize=14, labelpad=9)
ax_bottom.text(.995, .97, 'Lower is better', transform=ax_bottom.transAxes,
               ha='right', va='top', fontsize=10.5, color='0.35')

for ax in (ax_top, ax_bottom):
    ax.set_xlim(-.55, 3.55)
    ax.grid(axis='y', alpha=.24, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    ax.spines[['top', 'right']].set_visible(False)

fig.text(.095, .012,
         'Top: 27 trials per controller and environment, all three target speeds. '
         'Bottom: mean over 18 reverse trials (-30 and -50 km/h); all runs included, including failures.',
         fontsize=9.5, color='0.35')
fig.subplots_adjust(left=.095, right=.99, top=.88, bottom=.13)
for ext in ('png', 'svg'):
    fig.savefig(out / f'closed-loop-completion-and-hitch.{ext}',
                bbox_inches='tight', pad_inches=.15)
plt.close(fig)
print(out / 'closed-loop-completion-and-hitch.png')
