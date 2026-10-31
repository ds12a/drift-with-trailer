"""Plot four history models and the analytic baseline on the snow+slope turn.

    MPLBACKEND=Agg .venv/bin/python -m experiments.exp_008_beamng.plot_history_divergence
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import numpy as np

from experiments.exp_008_beamng import plot_model_divergence as divergence
from experiments.exp_008_beamng.experiment_paths import DATA_ROOT


MODELS = (
    ('h1_fixed_best', 'H=1 (fixed)', 'tab:blue', '-'),
    ('h2_mixed_best', 'H=2 (mixed)', 'tab:orange', '--'),
    ('h4_mixed_rollout10', 'Original (H=4)', 'tab:green', '-.'),
    ('h8_mixed_best', 'H=8 (mixed)', 'tab:purple', ':'),
    ('fiala', 'Analytic', 'tab:red', '-'),
)
DIRECTIONS = ('forward', 'backward')
ERROR_NAMES = ('position_error_m', 'hitch_error_deg', 'velocity_error_mps')


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rollouts(root):
    """Require identical reference states, commands, channels and sample times."""
    runs, rows, scores = {}, [], []
    for direction in DIRECTIONS:
        reference = None
        for name, label, _, _ in MODELS:
            candidate = 'h4_mixed_rollout10' if name == 'fiala' else name
            run = root / candidate / 'both4_div' / direction
            manifest = json.loads((run / 'manifest.json').read_text())
            with np.load(run / f'prediction_expert-learned_h4_model-{name}.npz') as data:
                actual = data['expert_states'].astype(np.float64)
                predicted = data['predicted_states'].astype(np.float64)
                controls = data['controls'].copy()
                channels = data['state_channels'].copy()
            dt = float(manifest['dt'])
            if actual.shape != predicted.shape or not np.isfinite(predicted).all():
                raise ValueError(f'Invalid predicted states: {run}')
            if not np.isfinite(actual).all() or not np.isfinite(controls).all():
                raise ValueError(f'Invalid recorded states or commands: {run}')
            if len(actual) != 101 or len(controls) != 100 or not np.isclose(dt, 0.05):
                raise ValueError(f'Expected the saved five-second rollout: {run}')
            if reference is None:
                reference = actual, controls, channels, dt
            else:
                for array, expected in zip((actual, controls, channels), reference[:3]):
                    np.testing.assert_array_equal(array, expected)
                if dt != reference[3]:
                    raise ValueError(f'Sample times disagree: {run}')
            np.testing.assert_array_equal(predicted[0], actual[0])
            indices = divergence.channel_indices({'state_channels': channels})
            xy = [indices['x'], indices['y']]
            velocity = [indices['vx'], indices['vy']]
            delta = predicted - actual
            hitch = divergence.wrap_angle(
                delta[:, indices['yaw_tractor']] - delta[:, indices['yaw_trailer']]
            )
            errors = np.column_stack((
                np.linalg.norm(delta[:, xy], axis=1),
                np.abs(np.degrees(hitch)),
                np.linalg.norm(delta[:, velocity], axis=1),
            ))
            time = np.arange(len(actual)) * dt
            runs[name, direction] = dict(actual=actual, predicted=predicted,
                                         indices=indices, time=time, errors=errors,
                                         manifest=manifest, directory=run)
            for step, time_s in enumerate(time):
                rows.append(dict(candidate=name, model=label, direction=direction,
                                 time_s=float(time_s), **dict(zip(ERROR_NAMES, errors[step]))))
            # Exclude the shared initial state, matching the existing figures.
            rmse = np.sqrt(np.mean(errors[1:] ** 2, axis=0))
            scores.append(dict(candidate=name, model=label, direction=direction,
                               position_rmse_m=rmse[0], hitch_angle_rmse_deg=rmse[1],
                               velocity_rmse_mps=rmse[2]))
    return runs, rows, scores


def save_figure(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot(root, output, levels_dir):
    runs, rows, scores = load_rollouts(root)
    output.mkdir(parents=True, exist_ok=True)
    terrain = divergence.load_terrain(levels_dir / divergence.TERRAIN_PATHS['both'])
    handles = [Line2D([], [], color=color, linestyle=style, lw=3.5, label=label)
               for _, label, color, style in MODELS]

    for direction, title in zip(DIRECTIONS, ('Forward', 'Reverse')):
        first = runs[MODELS[0][0], direction]
        indices, actual = first['indices'], first['actual']
        x, y = indices['x'], indices['y']
        states = [actual] + [runs[name, direction]['predicted'] for name, *_ in MODELS]
        vehicle = first['manifest']['scenario_config']['vehicle']
        pose_indices = [x, y, indices['yaw_tractor'], indices['yaw_trailer']]
        world_positions = np.concatenate([np.concatenate(states)[:, [x, y]]] +
            [divergence.endpoint_rig_vertices(state[-1, pose_indices], vehicle) for state in states])
        rotate = np.ptp(world_positions[:, 1]) > np.ptp(world_positions[:, 0])
        # Rotate tall views 90 degrees counterclockwise, including the terrain
        # and track. Orthogonal rotation preserves all distances and angles.
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]]) if rotate else np.eye(2)
        view_terrain = tuple(np.rot90(array, -1) for array in terrain) if rotate else terrain
        # Fit a rectangular view around every rollout instead of extending both
        # dimensions to the same span with the combined plot's 20 m margin.
        positions = world_positions @ rotation.T
        lower, upper = positions.min(axis=0), positions.max(axis=0)
        padding = np.maximum(2.0, 0.08 * (upper - lower))
        xlim, ylim = np.column_stack((lower - padding, upper + padding))
        aspect = np.diff(xlim).item() / np.diff(ylim).item()
        figure_width = 16.0
        axis_width = figure_width - 0.7
        axis_height = axis_width / aspect
        legend_height = 2.0
        figure_height = axis_height + legend_height + 0.85
        fig = plt.figure(figsize=(figure_width, figure_height))
        ax = fig.add_axes((0.35 / figure_width, legend_height / figure_height,
                           axis_width / figure_width, axis_height / figure_height))
        terrain_handles = []
        transform = Affine2D().rotate_deg(90 if rotate else 0) + ax.transData
        if not divergence.add_rendered_background(ax, 'both', direction, transform):
            terrain_handles = divergence.add_environment_background(ax, 'both', view_terrain, xlim, ylim)
            track = divergence.load_track(first['manifest'], first['directory']) @ rotation.T
            width = float(first['manifest']['scenario_config']['track'].get('width', 15.0))
            ax.add_collection(PolyCollection(
                divergence.track_ribbon(track, width), facecolors='#78838f',
                edgecolors='none', alpha=0.18, zorder=4,
            ))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        actual_xy = actual[:, [x, y]] @ rotation.T
        ax.plot(actual_xy[:, 0], actual_xy[:, 1], color='black', lw=4, zorder=7)
        ax.scatter(*actual_xy[0], color='black', s=35, zorder=8)
        divergence.draw_endpoint_rig(ax, actual[-1, pose_indices], vehicle, 'black', transform)
        for name, _, color, style in MODELS:
            predicted_xy = runs[name, direction]['predicted'][:, [x, y]] @ rotation.T
            ax.plot(predicted_xy[:, 0], predicted_xy[:, 1], color=color,
                    linestyle=style, lw=3.5, zorder=6)
            divergence.draw_endpoint_rig(ax, runs[name, direction]['predicted'][-1, pose_indices],
                                         vehicle, color, transform)
        ax.set_title(f'Snow + slope | {title}', pad=18)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('0.4')
        fig.legend(handles=[Line2D([], [], color='black', lw=4, label='Actual')]
                   + handles + terrain_handles, loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.5, 0.15 / figure_height), frameon=False)
        save_figure(fig, output, f'spatial_rollouts_{direction}')

    fig, axes = plt.subplots(2, 2, figsize=(22, 13), sharex=True)
    for column, (direction, title) in enumerate(zip(DIRECTIONS, ('Forward', 'Reverse'))):
        axes[0, column].set_title(title, pad=14)
        for name, _, color, style in MODELS:
            run = runs[name, direction]
            for row, error_column in enumerate((1, 2)):
                axes[row, column].plot(run['time'], run['errors'][:, error_column],
                                       color=color, linestyle=style, lw=3.5)
        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(0, 5)
            ax.set_ylim(bottom=0)
            ax.set_xticks(np.arange(6))
            ax.grid(alpha=0.28)
        axes[1, column].set_xlabel('Rollout time (s)')
    axes[0, 0].set_ylabel('Hitch Angle Error (deg)', labelpad=8)
    axes[1, 0].set_ylabel('Velocity Error (m/s)', labelpad=8)
    fig.suptitle('Snow + slope | Rollout errors', fontsize=12 * divergence.FONT_SCALE, y=0.97)
    fig.legend(handles=handles, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.15, top=0.87,
                        hspace=0.16, wspace=0.22)
    save_figure(fig, output, 'hitch_velocity_errors')

    write_csv(output / 'errors_over_time.csv', rows)
    write_csv(output / 'rmse.csv', scores)
    cases = {case['name']: case for case in json.loads((root / 'cases.json').read_text())}
    selection = [dict(cases[name], plot_label=label)
                 for name, label, _, _ in MODELS if name in cases]
    analytic_manifest = runs['fiala', 'forward']['manifest']
    selection.append(dict(
        name='fiala', plot_label='Analytic', model_type='analytic',
        friction=analytic_manifest['prior_friction']['fiala'],
        parameters=analytic_manifest['scenario_config']['vehicle'],
        prediction_archives={
            direction: str(runs['fiala', direction]['directory']
                           / 'prediction_expert-learned_h4_model-fiala.npz')
            for direction in DIRECTIONS
        },
    ))
    (output / 'models.json').write_text(json.dumps(selection, indent=2) + '\n')
    report = [
        '# Selected history models: snow+slope divergence', '',
        'The corrected snow-covered `turn_snow_div` turn, snow+slope (`both4_div`) only. '
        'Spatial rollouts are separate landscape figures for forward and reverse, '
        'with bounds fitted to all trajectories and equal distance scales. '
        'The forward view, including its terrain and track, is rotated 90 degrees '
        'counterclockwise to place the rollout horizontally. '
        'Spatial backgrounds use georeferenced orthographic BeamNG renders when '
        'available in `experiments/exp_008_beamng/rendered_backgrounds/`. '
        'Color-matched, metric-scale tractor/trailer schematics show each final pose. '
        'Trailer placement follows the saved hitch geometry; both headings come from '
        'the corresponding final state. '
        'The error figure shows both directions. All five models use identical '
        'recorded initial states and future controls over five seconds (100 steps at 0.05 s).', '',
        '| Plot label | Saved candidate | Selection |',
        '| --- | --- | --- |',
        '| H=1 (fixed) | `h1_fixed_best` | Best validation |',
        '| H=2 (mixed) | `h2_mixed_best` | Best validation |',
        '| Original (H=4) | `h4_mixed_rollout10` | Original `beamng-l4-128-test10-new_rollout10` |',
        '| H=8 (mixed) | `h8_mixed_best` | Best validation |',
        '| Analytic | `fiala` | Analytic model with fixed friction mu=1 |', '',
        'The original H=4 retains its rollout refinement; the other three use one-step '
        'training. The analytic baseline uses the saved `fiala` predictions for the '
        'same reference and commands. Model paths, analytic parameters, and the '
        'bundle\'s training-friction labels are preserved '
        'in [models.json](models.json).', '',
        '![Forward spatial rollouts](spatial_rollouts_forward.png)', '',
        '![Reverse spatial rollouts](spatial_rollouts_backward.png)', '',
        '![Hitch-angle and velocity errors](hitch_velocity_errors.png)', '',
        'Hitch-angle error is the absolute wrapped angular difference in degrees. '
        'Velocity error is the magnitude of the velocity-vector difference in m/s. '
        'Curves show error at each instant. The [RMSE table](rmse.csv) excludes the '
        'shared initial state; [errors_over_time.csv](errors_over_time.csv) contains every sample.', '',
        '[Forward spatial PDF](spatial_rollouts_forward.pdf) · '
        '[Reverse spatial PDF](spatial_rollouts_backward.pdf) · '
        '[Error PDF](hitch_velocity_errors.pdf)', '',
        'Reproduce from the repository root:', '', '```bash',
        'MPLBACKEND=Agg .venv/bin/python -m experiments.exp_008_beamng.plot_history_divergence',
        '```', '',
    ]
    (output / 'README.md').write_text('\n'.join(report))
    print(f'Saved separate spatial figures, the error figure and data to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path,
                        default=DATA_ROOT / 'turn_snow_div_history_friction_20261029')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--levels-dir', type=Path, default=divergence.DEFAULT_LEVELS_DIR)
    args = parser.parse_args()
    plot(args.root, args.output or args.root / 'rollout_comparison' / 'both4_div' / 'selected_models',
         args.levels_dir)
