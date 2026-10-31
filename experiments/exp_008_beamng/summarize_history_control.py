"""Summarize the matched closed-loop history-length benchmark, including partial runs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    original_h4 = protocol['models']['model_h4']['rollout_refinement']
    episodes = list(csv.DictReader((root / 'sweep/episodes.csv').open()))
    raw_files = {p.name: p for p in (root / 'sweep').glob('**/raw_*.npz')}
    assert len(raw_files) == len(episodes)
    enriched = []
    for row in episodes:
        name = f"raw_{row['controller']}_loc{row['loc']}_v{row['v_kph']}_s{row['seed']}.npz"
        with np.load(raw_files[name]) as data:
            count = int(data['iters'])
            assert len(data['velocities']) == len(data['hitches']) == len(data['us']) == count
            assert np.isfinite(data['velocities']).all() and np.isfinite(data['hitches']).all()
            cutoff = 100 if count > 100 else 0
            speed_error = data['velocities'][cutoff:, 0] * 3.6 - int(row['v_kph'])
            tracking = float(np.sqrt(np.mean(speed_error**2))) if count else abs(int(row['v_kph']))
            times = data['solve_ms'][1:]
            enriched.append(dict(row, tracking_rmse=tracking, steady_times=times))
    summaries = []
    for velocity in protocol['velocities_kph']:
        for kind in ('model_h1', 'model_h4'):
            group = [r for r in enriched if r['controller'] == kind and int(r['v_kph']) == velocity]
            if not group:
                continue
            outcomes = Counter(r['outcome'] for r in group)
            times = np.concatenate([r['steady_times'] for r in group])
            summaries.append(dict(
                controller=kind, target_kph=velocity, n=len(group),
                completed=outcomes['completed'], stalled=outcomes['stalled'],
                jackknife=outcomes['jackknife'], offtrack=outcomes['offtrack'], nan=outcomes['nan'],
                completion_rate=outcomes['completed'] / len(group),
                mean_signed_vx_kph=float(np.mean([float(r['avg_vx_kph']) for r in group])),
                speed_tracking_rmse_kph=float(np.sqrt(np.mean([r['tracking_rmse']**2 for r in group]))),
                mean_distance_m=float(np.mean([float(r['distance_survived_m']) for r in group])),
                mean_hitch_rms_deg=float(np.mean([float(r['hitch_rms_deg']) for r in group])),
                mean_abs_lateral_error_m=float(np.mean([float(r['mean_lat_err']) for r in group])),
                steady_solve_mean_ms=float(np.mean(times)) if times.size else float('nan'),
                steady_solve_p95_ms=float(np.percentile(times, 95)) if times.size else float('nan'),
            ))
    lookup = {(r['controller'], int(r['loc']), int(r['v_kph']), int(r['seed'])): r for r in enriched}
    pairs = []
    for location in protocol['locations']:
        for velocity in protocol['velocities_kph']:
            for seed in protocol['seeds']:
                a, b = (lookup.get((kind, location, velocity, seed)) for kind in ('model_h1', 'model_h4'))
                if a is None or b is None:
                    continue
                pairs.append(dict(location=location, target_kph=velocity, seed=seed,
                                  h1_outcome=a['outcome'], h4_outcome=b['outcome'],
                                  h1_tracking_rmse=a['tracking_rmse'], h4_tracking_rmse=b['tracking_rmse'],
                                  h1_distance_m=a['distance_survived_m'], h4_distance_m=b['distance_survived_m'],
                                  h1_hitch_rms_deg=a['hitch_rms_deg'], h4_hitch_rms_deg=b['hitch_rms_deg']))
    for filename, records in [('summary.csv', summaries), ('paired.csv', pairs)]:
        if records:
            with (root / filename).open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    expected = protocol['episodes']
    complete = len(episodes) == expected
    model_description = (
        'H=1 uses the mixed-friction best one-step checkpoint `beamng-l4-h1_best`. '
        'H=4 uses the original mixed-friction rollout-refined checkpoint '
        '`beamng-l4-128-test10-new_rollout10`. This compares the two trained models, '
        'including their different training objectives, rather than isolating history length alone. '
        if original_h4 else
        'Both models use mixed-friction training, best-validation checkpoints and the '
        'one-step derivative objective. Neither is rollout-refined. '
    )
    report = ['# Closed-loop H=1 versus ' + ('original refined H=4' if original_h4 else 'H=4'), '',
              f"**{'Complete' if complete else 'In progress'}: {len(episodes)}/{expected} episodes; "
              f"{len(pairs)}/{expected // 2} complete matched pairs.**", '',
              model_description + 'The original production '
              'MPPI dynamics, controller settings, episode loop and metrics come from run_sweep. '
              'H=1 retains the final measured warmup row; H=4 retains all four. Both receive '
              'the same four-step physical warmup, starts, target speeds and sampling seeds. '
              'A fresh BeamNG process runs each episode on the staged snow+slope map.', '',
              f"Settings: {protocol['locations']} start indices; "
              f"{protocol['velocities_kph']} km/h targets; seeds {protocol['seeds']}; "
              f"{protocol['max_steps']} control steps at dt=0.05 s; "
              f"K={protocol['fwd_mppi']['K']} forward / {protocol['rev_mppi']['K']} reverse samples; "
              f"horizons {protocol['fwd_mppi']['T']} / {protocol['rev_mppi']['T']} steps.", '',
              '**Completed means surviving the full episode without a stopping condition. '
              'It does not require reaching the requested speed.** Speed tracking and distance '
              'therefore matter alongside completion.', '',
              '| Target (km/h) | Model | Finished / trials | Mean vx (km/h) | Speed RMSE (km/h) | Distance (m) | Hitch RMS (deg) | Lateral error (m) |',
              '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    if 'vehicle_config_sha256' in protocol:
        report.insert(8, 'The custom `vehicles/pickup/hitch.pc` is staged with the terrain. '
                      'Every trial verifies the loaded engine, transmission, tires, hitch, '
                      'steering and ESC selections before the episode. See `vehicle_checks/`.\n')
    for row in summaries:
        report.append(f"| {row['target_kph']} | {row['controller'].replace('model_', '').upper()} | "
                      f"{row['completed']} / {row['n']} | {row['mean_signed_vx_kph']:.2f} | "
                      f"{row['speed_tracking_rmse_kph']:.2f} | {row['mean_distance_m']:.1f} | "
                      f"{row['mean_hitch_rms_deg']:.2f} | {row['mean_abs_lateral_error_m']:.2f} |")
    report += ['', '## Results by starting location', '',
               '| Target (km/h) | Start index | H=1 finished | H=4 finished | H=1 mean vx (km/h) | H=4 mean vx (km/h) |',
               '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for velocity in protocol['velocities_kph']:
        for location in protocol['locations']:
            cells, speeds = [], []
            for kind in ('model_h1', 'model_h4'):
                group = [r for r in enriched if r['controller'] == kind
                         and int(r['v_kph']) == velocity and int(r['loc']) == location]
                if group:
                    cells.append(f"{sum(r['outcome'] == 'completed' for r in group)} / {len(group)}")
                    speeds.append(f"{np.mean([float(r['avg_vx_kph']) for r in group]):.2f}")
                else:
                    cells.append('pending')
                    speeds.append('pending')
            report.append(f"| {velocity} | {location} | " + ' | '.join(cells + speeds) + ' |')
    report += ['', '## Outcomes', '',
               '| Target (km/h) | Model | Completed | Stalled | Jackknife | Offtrack | Nonfinite control |',
               '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summaries:
        report.append(f"| {row['target_kph']} | {row['controller'].replace('model_', '').upper()} | "
                      + ' | '.join(str(row[key]) for key in ('completed', 'stalled', 'jackknife', 'offtrack', 'nan')) + ' |')
    report += ['', 'Speed and hitch statistics use the post-100-step tail when available, '
               'following run_sweep. Lateral error and distance include the full recorded episode. '
               'Speed RMSE pools per-episode squared RMSE equally. Early failures and surviving '
               'episodes have different lengths, so inspect outcomes and matched pairs alongside '
               'averages. Timing in summary.csv excludes the first controller call of each episode.', '',
               'The MPPI seeds repeat controller sampling, not network training. There is one '
               'trained network per history length; input-layer size, available history windows '
               'and normalization statistics vary with history length.', '',
               '[Protocol and exact checkpoint paths](protocol.json) · [Summary CSV](summary.csv) · '
               '[Matched pairs](paired.csv) · [Episode outcomes](sweep/episodes.csv)', '']
    report += ['## Reproduce', '',
               'From the repository root, choose a fresh output directory:', '', '```bash',
               'HISTORY_JAX_PLUGIN=cuda13 MPLBACKEND=Agg .venv/bin/python -m '
               'experiments.exp_008_beamng.run_history_comparison \\',
               ('  --h4-selection original \\' if original_h4 else '  --h4-selection best \\'),
               '  --output experiments/exp_008_beamng/history_control_out/repeat_comparison',
               '```', '',
               'CUDA plugin selection is specific to this environment, which has both CUDA 12 '
               'and CUDA 13 plugins installed. The simulator uses a separate user directory '
               'under the experiment output. The runner supports `--resume` with an existing '
               'output directory and refuses changes to the matched experimental settings.', '']
    if summaries:
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        x = np.arange(len(protocol['velocities_kph']))
        for offset, kind, label in [(-0.18, 'model_h1', 'H=1'), (0.18, 'model_h4', 'Original H=4' if original_h4 else 'H=4')]:
            indexed = {r['target_kph']: r for r in summaries if r['controller'] == kind}
            for ax, metric in zip(axes, ['completion_rate', 'mean_signed_vx_kph', 'speed_tracking_rmse_kph']):
                values = [indexed[v][metric] if v in indexed else np.nan for v in protocol['velocities_kph']]
                ax.bar(x + offset, values, width=0.36, label=label)
        axes[0].set_ylabel('Fraction finishing full episode')
        axes[0].set_ylim(0, 1.05)
        axes[1].set_ylabel('Mean signed vx (km/h)')
        axes[1].plot(x, protocol['velocities_kph'], 'kx', markersize=10, label='Target')
        axes[2].set_ylabel('Speed tracking RMSE (km/h)')
        for ax in axes:
            ax.set_xticks(x, [str(v) for v in protocol['velocities_kph']])
            ax.set_xlabel('Target speed (km/h)')
            ax.grid(axis='y', alpha=0.25)
            ax.legend()
        fig.suptitle(f"Closed-loop comparison: {len(episodes)}/{expected} episodes")
        fig.tight_layout()
        fig.savefig(root / 'comparison.png', dpi=180)
        fig.savefig(root / 'comparison.pdf')
        plt.close(fig)
        report.insert(4, '![Performance comparison](comparison.png)\n')
    (root / 'README.md').write_text('\n'.join(report))
    print('\n'.join(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path,
                        default=Path('experiments/exp_008_beamng/history_control_out/20261029_mixed_best'))
    summarize(parser.parse_args().root)
