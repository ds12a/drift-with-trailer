"""Plot one matched seed of measured BeamNG drives and summarize all trials."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import numpy as np

from experiments.exp_008_beamng import plot_model_divergence as background


MODELS = (
    ('h1_fixed_best', 'H=1', '#0072B2', '-'),
    ('h2_mixed_best', 'H=2', '#D55E00', '-'),
    ('h4_mixed_rollout10', 'H=4', '#009E73', '-'),
    ('h8_mixed_best', 'H=8', '#AA3377', '-'),
    ('fiala', 'Analytic', '#222222', '-'),
)
LINE_WIDTHS = {'#0072B2': 5.0, '#D55E00': 4.2, '#009E73': 3.4, '#AA3377': 2.6, '#222222': 1.8}


def line_style(color, style):
    # Draw wide paths first and narrow paths last, so coincident colors remain
    # visible without moving any measured coordinates.
    return dict(color=color, linestyle=style, linewidth=LINE_WIDTHS[color])


def load_runs(root, protocol):
    runs = {}
    for path in sorted((root / 'raw').glob('*.npz')):
        with np.load(path) as data:
            run = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
        n = run['iters']
        if run['states'].shape != (n + 1, 10) or run['velocities'].shape != (n, 2):
            raise ValueError(f'Incorrect measured state dimensions: {path}')
        for field in ('states', 'velocities', 'hitches', 'cum_dist', 'us', 'measured_friction'):
            if not np.isfinite(run[field]).all():
                raise ValueError(f'Nonfinite {field}: {path}')
        np.testing.assert_array_equal(run['states'][1:, 4:6], run['velocities'])
        np.testing.assert_allclose(run['time_s'], np.arange(n + 1) * protocol['dt'])
        runs[run['candidate'], run['direction'], run['seed']] = run
    return runs


def save(fig, output, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot(root, seed=0):
    protocol = json.loads((root / 'protocol.json').read_text())
    if seed not in protocol['seeds']:
        raise ValueError(f'Seed {seed} is not in the recorded seeds {protocol["seeds"]}')
    runs = load_runs(root, protocol)
    if not runs:
        raise ValueError('No measured BeamNG drives have been saved yet')
    output = root / 'figures'
    output.mkdir(exist_ok=True)
    from experiments.exp_008_beamng.experiment_paths import terrain_file, resolve_saved_path
    terrain = background.load_terrain(terrain_file(protocol))
    centerline = np.genfromtxt(resolve_saved_path(protocol['scenario_config']['track']['csv']), delimiter=',', names=True)
    track = np.column_stack((centerline['x'], centerline['y']))
    track = np.vstack((track, track[0]))
    handles = [Line2D([], [], color=color, linestyle=style, lw=LINE_WIDTHS[color], label=label)
               for _, label, color, style in MODELS]
    summary = []
    vehicle = protocol['scenario_config']['vehicle']
    for direction, start in protocol['starts'].items():
        direction_runs = {(name, run_seed): run for (name, d, run_seed), run in runs.items() if d == direction}
        selected = {(name, run_seed): run for (name, run_seed), run in direction_runs.items() if run_seed == seed}
        if not selected:
            continue
        points = np.concatenate([r['states'][:, :2] for r in selected.values()] +
                                [background.endpoint_rig_vertices(r['states'][-1, :4], vehicle)
                                 for r in selected.values()])
        rotate = np.ptp(points[:, 1]) > np.ptp(points[:, 0])
        rotation = np.array([[0., -1.], [1., 0.]]) if rotate else np.eye(2)
        view_terrain = tuple(np.rot90(array, -1) for array in terrain) if rotate else terrain
        xy = points @ rotation.T
        lo, hi = xy.min(0), xy.max(0)
        padding = np.maximum(2., (hi - lo) * .05)
        lo, hi = lo - padding, hi + padding
        # Keep both figures landscape while preserving metric geometry.
        extra = max(0., 1.6 * (hi[1] - lo[1]) - (hi[0] - lo[0])) / 2
        lo[0] -= extra
        hi[0] += extra
        aspect = (hi[0] - lo[0]) / (hi[1] - lo[1])
        axis_width = 15.3
        axis_height = axis_width / aspect
        figure_width = 16.0
        has_render = background.load_rendered_background('both', direction) is not None
        legend_height = 1.35 if has_render else 2.0
        fig_height = axis_height + legend_height + .85
        fig = plt.figure(figsize=(figure_width, fig_height))
        ax = fig.add_axes((.35 / figure_width, legend_height / fig_height,
                           axis_width / figure_width, axis_height / fig_height))
        terrain_handles = []
        transform = Affine2D().rotate_deg(90 if rotate else 0) + ax.transData
        if not background.add_rendered_background(ax, 'both', direction, transform):
            terrain_handles = background.add_environment_background(
                ax, 'both', view_terrain, (lo[0], hi[0]), (lo[1], hi[1]))
            for image in ax.images:
                image.set_alpha(.18)
            for contour in ax.collections:
                contour.set_alpha(.25)
            view_track = track @ rotation.T
            ax.add_collection(PolyCollection(background.track_ribbon(
                view_track[:-1], protocol['scenario_config']['track']['width']),
                facecolors='#78838f', edgecolors='none', alpha=.25, zorder=4))
        for name, _, color, style in MODELS:
            for (candidate, _), run in selected.items():
                if candidate != name:
                    continue
                positions = run['states'][:, :2] @ rotation.T
                ax.plot(positions[:, 0], positions[:, 1],
                        **line_style(color, style), zorder=6)
                background.draw_endpoint_rig(ax, run['states'][-1, :4], vehicle, color, transform)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        reference = selected['h4_mixed_rollout10', seed]
        reference_xy = reference['states'][:, :2] @ rotation.T
        start_xy = reference_xy[0]
        ax.scatter(*start_xy, s=100, color='black', edgecolors='white', linewidths=1.2, zorder=20)
        title = 'Forward' if direction == 'forward' else 'Reverse'
        ax.set_title(f'Snow + slope | {title} | Seed {seed}', pad=18)
        fig.legend(handles=handles, loc='lower center', ncol=5, handlelength=3.3,
                   bbox_to_anchor=(.5, (.2 if has_render else .95) / fig_height), frameon=False)
        if terrain_handles:
            fig.legend(handles=terrain_handles, loc='lower center', ncol=2,
                       bbox_to_anchor=(.5, .15 / fig_height), frameon=False)
        save(fig, output, f'beamng_spatial_{direction}')

        for name, label, _, _ in MODELS:
            group = [r for (candidate, _), r in direction_runs.items() if candidate == name]
            if not group:
                continue
            outcomes = Counter(r['outcome'] for r in group)
            speed_rmse, hitches = [], []
            for r in group:
                cutoff = 100 if r['iters'] > 100 else 0
                speed_rmse.append(np.mean((r['velocities'][cutoff:, 0] * 3.6 - start['target_kph']) ** 2))
                hitches.append(np.mean(np.degrees(r['hitches'][cutoff:]) ** 2))
            summary.append(dict(direction=direction, candidate=name, model=label, n=len(group),
                                finished=outcomes['completed'], stalled=outcomes['stalled'],
                                jackknife=outcomes['jackknife'], offtrack=outcomes['offtrack'],
                                speed_rmse_kph=float(np.sqrt(np.mean(speed_rmse))),
                                mean_progress_m=float(np.mean([np.sign(start['target_kph']) *
                                                               r['cum_dist'][-1] for r in group])),
                                hitch_rms_deg=float(np.sqrt(np.mean(hitches)))))

    hitch_limit = protocol['stopping_conditions']['hitch_limit_deg']
    for direction in ('forward', 'backward'):
        if direction not in protocol['starts']:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
        target = protocol['starts'][direction]['target_kph']
        axes[0].axhline(target, color='.45', linestyle='--', lw=1.8)
        axes[1].axhline(hitch_limit, color='.45', linestyle='--', lw=1.8)
        for name, _, color, style in MODELS:
            r = runs.get((name, direction, seed))
            if r is None:
                continue
            states = r['states']
            hitch = np.abs(np.degrees(background.wrap_angle(states[:, 3] - states[:, 2])))
            for ax, value in zip(axes, (states[:, 4] * 3.6, hitch)):
                ax.plot(r['time_s'], value, **line_style(color, style))
        for ax, label in zip(axes, ('Longitudinal speed\n(km/h)', 'Hitch angle (deg)')):
            ax.set_xlim(0, protocol['max_steps'] * protocol['dt'])
            ax.grid(alpha=.28)
            ax.set_ylabel(label, labelpad=8)
        axes[1].set_ylim(bottom=0)
        axes[1].set_xlabel('Time (s)')
        title = 'Forward' if direction == 'forward' else 'Reverse'
        fig.suptitle(f'Snow + slope | {title} | Seed {seed}',
                     fontsize=12 * background.FONT_SCALE, y=.98)
        fig.legend(handles=handles, loc='lower center', ncol=5, handlelength=3.3,
                   bbox_to_anchor=(.5, .075), frameon=False)
        reference = Line2D([], [], color='.45', linestyle='--', lw=1.8, label='Target / limit')
        fig.legend(handles=[reference], loc='lower center', ncol=1,
                   bbox_to_anchor=(.5, .005), frameon=False)
        fig.subplots_adjust(left=.13, right=.98, bottom=.26, top=.90, hspace=.18)
        save(fig, output, f'beamng_control_performance_{direction}')
    (output / 'plot_config.json').write_text(json.dumps(dict(
        plotted_seed=seed, centerline=False,
        start_marker=True, start_label=False, spatial_detail=False,
        endpoint_vehicles='schematic, metric scale, final tractor position and both headings; trailer XY reconstructed from hitch geometry',
        spatial_background='BeamNG orthographic render' if background.load_rendered_background('both', 'forward') else 'terrain height/material map',
        performance_rows=['longitudinal_speed', 'hitch_angle'],
        performance_separate_directions=True, time_axis_label='Time (s)',
        models=[dict(candidate=name, label=label, color=color, linestyle=style, marker=None,
                     linewidth=LINE_WIDTHS[color])
                for name, label, color, style in MODELS],
        summary_seeds=protocol['seeds']), indent=2) + '\n')
    with (root / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report = ['# Measured BeamNG drives: selected history models', '',
              f"**Collected {len(runs)}/{protocol['episodes']} episodes.** Every colored path is "
              'a measured BeamNG trajectory under that model\'s MPPI controller. '
              'There are no model-generated trajectories in these figures.', '',
              'H1 fixed, H2 mixed, original rollout-refined H4, H8 mixed, and Analytic '
              '(mu=1) use the same controller settings, physical starts, targets and '
              'eight measured warmup steps. Networks retain their own last H rows. '
              f"Figures plot **seed {seed} only**, consistently for every model and direction. "
              f"The table and summary CSV still include all recorded seeds {protocol['seeds']}. "
              f"Episodes last up to {protocol['max_steps'] * protocol['dt']:g} seconds. "
              'Starts are from rest at the snow-covered turn waypoints; these differ '
              'from the moving initial states used in the five-second prediction test. '
              'The custom pickup configuration and loaded vehicle parts are verified in every episode. '
              'The terrain collision object must be active and the tractor must report actual '
              'SNOW contacts at controller takeover; per-trial evidence is in `surface_checks/`. '
              'BeamNG graphics stay enabled because null-graphics mode omitted the terrain.', '',
              '![Measured forward paths](figures/beamng_spatial_forward.png)', '',
              '![Measured reverse paths](figures/beamng_spatial_backward.png)', '',
              '![Forward control performance](figures/beamng_control_performance_forward.png)', '',
              '![Reverse control performance](figures/beamng_control_performance_backward.png)', '',
              'Solid colored lines identify each model. Wider paths are drawn first and '
              'narrower paths last so overlapping colors remain visible, using the exact '
              'measured coordinates. Each spatial figure marks the common start with an '
              'unlabeled black dot. Spatial backgrounds use georeferenced orthographic '
              'BeamNG captures when available; the capture metadata are in '
              '`experiments/exp_008_beamng/rendered_backgrounds/`. '
              'Color-matched tractor/trailer schematics mark the final pose of each run '
              'at metric scale. Tractor position and both headings are recorded; trailer '
              'placement is reconstructed from the saved hitch geometry. '
              'There is no magnified panel. '
              'The track centerline and repeated line markers are omitted. Lines end at the actual stopping time; '
              'early stopping outcomes are recorded in the episode table. Gray dashed lines in the time plots '
              'show requested longitudinal speed and the hitch-angle limit. Hitch angles '
              'are plotted as absolute magnitudes. Spatial scenes '
              'are rotated if needed for landscape presentation, with equal distance scales.', '',
              '## Results across all recorded seeds', '',
              '| Direction | Model | Finished | Speed RMSE (km/h) | Progress (m) | Hitch RMS (deg) |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary:
        report.append(f"| {r['direction']} | {r['model']} | {r['finished']}/{r['n']} | "
                      f"{r['speed_rmse_kph']:.2f} | {r['mean_progress_m']:.1f} | {r['hitch_rms_deg']:.2f} |")
    report += ['', 'Finished means reaching the full requested duration without a stopping '
               'condition; it does not require reaching target speed. Speed and hitch '
               'statistics exclude the first 100 control steps when available, following '
               'run_sweep. Reported RMSE averages episode mean-square errors equally. '
               'Progress is signed track arc in the requested travel direction. '
               'Different early stopping times are visible in the plots and outcomes.', '',
               '[Summary CSV](summary.csv) · [Episode outcomes](episodes.csv) · '
               '[Exact checkpoints and protocol](protocol.json)', '',
               '[Forward PDF](figures/beamng_spatial_forward.pdf) · '
               '[Reverse PDF](figures/beamng_spatial_backward.pdf) · '
               '[Forward performance PDF](figures/beamng_control_performance_forward.pdf) · '
               '[Reverse performance PDF](figures/beamng_control_performance_backward.pdf)', '',
               'Reproduce with a fresh output directory:', '', '```bash',
               'HISTORY_JAX_PLUGIN=cuda13 MPLBACKEND=Agg .venv/bin/python -m '
               'experiments.exp_008_beamng.run_selected_beamng_control --output OUTPUT',
               f'MPLBACKEND=Agg .venv/bin/python -m experiments.exp_008_beamng.plot_beamng_control OUTPUT --seed {seed}',
               '```', '']
    (root / 'README.md').write_text('\n'.join(report))
    print(f'Saved measured control figures for {len(runs)}/{protocol["episodes"]} episodes to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--seed', type=int, default=0, help='single recorded seed to plot (default: 0)')
    args = parser.parse_args()
    plot(args.root, args.seed)
