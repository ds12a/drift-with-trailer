"""Collect actual BeamNG drives for the five selected divergence-plot models.

HISTORY_JAX_PLUGIN=cuda13 MPLBACKEND=Agg .venv/bin/python -m \
    experiments.exp_008_beamng.run_selected_beamng_control --output OUTPUT
"""

import argparse
import csv
from dataclasses import asdict, astuple
import hashlib
import json
from pathlib import Path
import shutil
import traceback

# Retain the established CUDA selection, checkpoint restoration and rig checks.
from experiments.exp_008_beamng import run_history_comparison as common
from experiments.exp_008_beamng.plot_history_divergence import MODELS
from experiments.exp_008_beamng.experiment_paths import MODEL_BUNDLE, TERRAIN_LEVEL, VEHICLE_CONFIG, TURN_SNOW

import jax
import jax.numpy as jnp
import numpy as np

sweep = common.sweep
STATE_CHANNELS = ('x', 'y', 'yaw_tractor', 'yaw_trailer', 'vx', 'vy',
                  'yaw_rate_tractor', 'yaw_rate_trailer', 'steer_realized', 'accel_realized')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_terrain(source, destination):
    if not destination.exists():
        shutil.copytree(source, destination)
    # Level groups name their child directories; preserve that spelling on Linux.
    pending = [destination / 'main']
    while pending:
        directory = pending.pop()
        file = directory / 'items.level.json'
        if not file.exists():
            continue
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('class') != 'SimGroup' or not obj.get('name'):
                continue
            name = obj['name']
            child = next((p for p in directory.iterdir()
                          if p.is_dir() and p.name.casefold() == name.casefold()), None)
            if child is not None:
                target = directory / name
                if child != target:
                    child.rename(target)
                pending.append(target)


def check_snow_contacts(snapshot):
    contacts = snapshot['tractor']['contacts']
    if not any(c['material'].upper() == 'SNOW' and c['wheel_count'] > 0 for c in contacts):
        raise ValueError(f'Expected actual SNOW contacts at controller takeover, got {contacts}')


def collect_episode(env, planner, target, seed, kind, location):
    """Instrument real simulator steps while retaining the production episode loop."""
    measurements, frictions, initial_contacts = [], [], []
    step = env.step

    def measured_step(action):
        result = step(action)
        measurements.append(astuple(env.unwrapped._state))
        surface = env.unwrapped.query_surface_friction()
        frictions.append(surface.dynamics_mu)
        if len(measurements) == 1 + sweep.WARMUP_STEPS:
            snapshot = surface.as_dict()
            check_snow_contacts(snapshot)
            initial_contacts.append(snapshot)
        return result

    env.step = measured_step
    try:
        episode = sweep.run_episode(env, planner, target, seed, kind, location)
    finally:
        env.step = step
    count = episode['iters']
    expected = 1 + sweep.WARMUP_STEPS + count
    if len(measurements) != expected:
        raise ValueError(f'Unexpected simulator measurement count: {len(measurements)} != {expected}')
    states = np.asarray(measurements, dtype=float)[-(count + 1):]
    mu = np.asarray(frictions, dtype=float)[-(count + 1):]
    np.testing.assert_allclose(states[1:, 4:6], episode['velocities'], rtol=0, atol=0)
    np.testing.assert_allclose(sweep.wrap_angle(states[1:, 3] - states[1:, 2]), episode['hitches'])
    if not np.isfinite(states).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite measured BeamNG state or friction')
    return dict(episode, states=states, measured_friction=mu,
                initial_contacts_json=json.dumps(initial_contacts[0]),
                state_channels=np.asarray(STATE_CHANNELS),
                time_s=np.arange(count + 1) * sweep.DT)


def write_summaries(output):
    rows = []
    for path in sorted((output / 'raw').glob('*.npz')):
        with np.load(path) as archive:
            episode = {key: (archive[key].item() if archive[key].ndim == 0 else archive[key])
                       for key in archive.files}
        rows.append(dict(candidate=episode['candidate'], direction=episode['direction'],
                         **sweep.summarize(episode)))
    if rows:
        with (output / 'episodes.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=MODEL_BUNDLE)
    parser.add_argument('--source', type=Path, default=TURN_SNOW / 'both4_div')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--samples', type=int, default=1500)
    parser.add_argument('--directions', choices=['forward', 'backward'], nargs='+',
                        default=['forward', 'backward'])
    parser.add_argument('--beamng-home', type=Path, default=Path.home() / 'BeamNG.tech.v0.38.5.0')
    parser.add_argument('--vehicle-config', type=Path, default=VEHICLE_CONFIG)
    parser.add_argument('--terrain', type=Path, default=TERRAIN_LEVEL)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.samples) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('steps/samples must be positive and seeds unique')
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'Choose a fresh output directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output / 'raw').mkdir(exist_ok=True)
    user = args.output.resolve() / 'beamng_user'
    level = user / 'current/levels/tech_ground'
    stage_terrain(args.terrain, level)
    vehicle, vehicle_path = common.stage_vehicle_config(args.vehicle_config, user)
    cases = {case['name']: case for case in json.loads((args.bundle / 'bundled_cases.json').read_text())}
    configs, provenance = {}, {}
    for name, label, _, _ in MODELS:
        if name == 'fiala':
            provenance[name] = dict(label=label, controller='prior', model_type='analytic', friction=1.0)
        else:
            case = cases[name]
            stats, checkpoint = args.bundle / case['stats'], args.bundle / case['checkpoint']
            configs[name] = (case['H'], stats, checkpoint)
            provenance[name] = dict(case, label=label, controller=name,
                                    checkpoint=str(checkpoint.resolve()), stats=str(stats.resolve()),
                                    stats_sha256=sha256(stats))
    sweep.MODEL_CONFIGS.clear()
    sweep.MODEL_CONFIGS.update(configs)
    sweep.load_learned_model.cache_clear()
    sweep.restore_model = common.replay.restore_checked
    sweep.WARMUP_STEPS = max(case[0] for case in configs.values())
    sweep.MAX_STEPS = args.steps
    sweep.FWD_MPPI['K'] = sweep.REV_MPPI['K'] = args.samples
    # The Analytic baseline assumes mu=1; the physical BeamNG surface still
    # supplies the actual snow+slope friction because use_custom_mu=False.
    sweep.scenario.track.friction_csv = None
    sweep.scenario.track.mu = 1.0
    centerline = np.genfromtxt(sweep.scenario.track.csv, delimiter=',', names=True)
    xy = np.column_stack((centerline['x'], centerline['y']))
    starts = {}
    for direction in args.directions:
        manifest = json.loads((args.source / direction / 'manifest.json').read_text())
        query = np.asarray(manifest['waypoint_selection']['query_xy'])
        index = int(np.argmin(np.linalg.norm(xy - query, axis=1)))
        starts[direction] = dict(track_index=index, xy=xy[index].tolist(), query_xy=query.tolist(),
                                 target_kph=int(manifest['target_kph']))
    protocol = dict(
        purpose='Actual measured BeamNG closed-loop controller trajectories; no model-generated paths',
        models=provenance, starts=starts, seeds=args.seeds, max_steps=args.steps, dt=sweep.DT,
        warmup_steps=sweep.WARMUP_STEPS,
        initialization='Same from-rest spawn and eight physical warmup steps; each network retains its last H measured rows',
        episodes=len(MODELS) * len(args.directions) * len(args.seeds),
        terrain_source=str(args.terrain.resolve()), terrain_sha256=sha256(level / 'terrain8.ter'),
        staged_level=str(level), nogpu=False,
        surface_validation='Active theTerrain object plus measured tractor SNOW contacts at takeover in every trial',
        vehicle_config_sha256=sha256(vehicle_path),
        vehicle_config_source=str(args.vehicle_config.resolve()),
        scenario_config=asdict(sweep.scenario),
        fwd_mppi={k: np.asarray(v).tolist() for k, v in sweep.FWD_MPPI.items()},
        rev_mppi={k: np.asarray(v).tolist() for k, v in sweep.REV_MPPI.items()},
        fwd_weights=sweep.FWD_WEIGHTS, rev_weights=sweep.REV_WEIGHTS,
        stopping_conditions=dict(hitch_limit_deg=float(np.degrees(sweep.scenario.vehicle.max_hitch)),
                                 track_width_m=sweep.scenario.track.width,
                                 stall_speed_mps=sweep.STALL_VEL, stall_window_steps=sweep.STALL_WINDOW,
                                 stall_progress_m=sweep.STALL_ARC, stall_grace_steps=sweep.STALL_GRACE),
    )
    if args.resume:
        previous = json.loads((args.output / 'protocol.json').read_text())
        if previous != protocol:
            raise ValueError('Cannot resume with a changed experimental protocol')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    for path, name in ((Path(__file__), 'runner_source.py'), (Path(sweep.__file__), 'sweep_source.py'),
                       (Path(common.__file__), 'rig_checks_source.py')):
        shutil.copy2(path, args.output / name)
    readiness = []
    if not args.resume:
        for direction, start in starts.items():
            expert, _ = common.replay.measured_segment(args.source / direction, sweep.WARMUP_STEPS)
            for name, model in provenance.items():
                planner = sweep.make_mpc(start['target_kph'], model['controller'])
                history = configs[name][0] if name in configs else None
                state = (expert['initial_history'][-history:].reshape(-1) if history else
                         np.r_[expert['states'][0, :8], 1.0, expert['states'][0, -1]])
                control = np.asarray(planner.run_mpc(jnp.asarray(state)))
                if control.shape != (2,) or not np.isfinite(control).all():
                    raise ValueError(f'Invalid readiness control for {name}/{direction}')
                readiness.append(dict(candidate=name, direction=direction, control=control.tolist(),
                                      history=history, samples=planner.K, horizon=planner.T))
                print(f'Ready {name}/{direction}: H={history}, u={control}', flush=True)
                del planner
                jax.clear_caches()
        (args.output / 'controller_checks.json').write_text(json.dumps(readiness, indent=2))
    if args.prepare_only:
        return

    checks_dir = args.output / 'vehicle_checks'
    checks_dir.mkdir(exist_ok=True)
    surface_dir = args.output / 'surface_checks'
    surface_dir.mkdir(exist_ok=True)
    for direction, start in starts.items():
        planners = {name: sweep.make_mpc(start['target_kph'], model['controller'])
                    for name, model in provenance.items()}
        for seed_index, seed in enumerate(args.seeds):
            order = list(provenance)
            order = order[seed_index:] + order[:seed_index]
            for name in order:
                path = args.output / 'raw' / f'{name}_{direction}_s{seed}.npz'
                if path.exists():
                    print(f'Reusing {path.name}', flush=True)
                    continue
                env = None
                try:
                    env = sweep.BeamNGTrailerEnv(config=sweep.scenario, use_custom_mu=False,
                                                headless=True, nogpu=False, spidx=start['track_index'],
                                                beamng_home_dir=args.beamng_home, beamng_user_dir=user)
                    selected = common.check_vehicle_parts(env.tractor.get_part_config(), vehicle['parts'])
                    (checks_dir / f'{name}_{direction}_s{seed}.json').write_text(json.dumps(dict(
                        config_sha256=protocol['vehicle_config_sha256'], selected_parts=selected), indent=2))
                    terrain_info = json.loads(env.bng.control.queue_lua_command(
                        "local t=scenetree.findObject('theTerrain'); assert(t, 'Required terrain is missing'); "
                        "return jsonEncode({file=t.terrainFile, maxHeight=t.maxHeight, "
                        "realPath=FS:getFileRealPath('/levels/tech_ground/terrain8.ter')})", response=True))
                    terrain_info['height_at_start_m'] = env._query_terrain_height(*start['xy'])
                    episode = collect_episode(env, planners[name], start['target_kph'], seed,
                                              provenance[name]['controller'], start['track_index'])
                    (surface_dir / f'{name}_{direction}_s{seed}.json').write_text(json.dumps(dict(
                        terrain=terrain_info, initial_contacts=json.loads(episode['initial_contacts_json'])), indent=2))
                except Exception:
                    (args.output / 'last_error.txt').write_text(traceback.format_exc())
                    raise
                finally:
                    if env is not None:
                        env.bng.close()
                episode.update(candidate=name, direction=direction)
                np.savez_compressed(path, **episode)
                rows = write_summaries(args.output)
                summary = sweep.summarize(episode)
                print(f"SAVED {len(rows)}/{protocol['episodes']}: {name}/{direction}/seed{seed}: "
                      f"{summary['outcome']}, vx={summary['avg_vx_kph']:.2f} km/h, "
                      f"distance={summary['distance_survived_m']:.1f} m", flush=True)
                (args.output / 'status.json').write_text(json.dumps(dict(
                    completed=len(rows), expected=protocol['episodes'], last_case=path.stem), indent=2))
        del planners
        jax.clear_caches()
    print('All measured BeamNG drives collected.', flush=True)


if __name__ == '__main__':
    main()
