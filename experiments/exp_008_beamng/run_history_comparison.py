"""Matched H=1/H=4 closed-loop experiment using run_sweep's MPPI and metrics.

    MPLBACKEND=Agg .venv/bin/python -m experiments.exp_008_beamng.run_history_comparison

By default both networks use mixed-friction best one-step validation weights.
Use --h4-selection original to compare the original rollout-refined H=4.
The terrain and required custom pickup configuration are staged together.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import os
import json
from pathlib import Path
import shutil
import socket
import traceback

# Select one installed CUDA plugin explicitly when both CUDA 12 and 13 are
# present. Leave normal discovery unchanged unless the experiment requests it.
if os.environ.get('HISTORY_JAX_PLUGIN'):
    from importlib import import_module
    from jax._src import xla_bridge
    plugin = os.environ['HISTORY_JAX_PLUGIN']
    if plugin not in ('cuda12', 'cuda13'):
        raise ValueError('HISTORY_JAX_PLUGIN must be cuda12 or cuda13')
    xla_bridge.discover_pjrt_plugins = import_module(f'jax_plugins.xla_{plugin}').initialize

from experiments.exp_008_beamng import replay_history_friction_divergence as replay
from experiments.exp_008_beamng import run_sweep as sweep
from experiments.exp_008_beamng.experiment_paths import MODEL_BUNDLE, TERRAIN_LEVEL, VEHICLE_CONFIG, TURN_SNOW

from beamngpy import BeamNGpy
import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=MODEL_BUNDLE)
    parser.add_argument('--output', type=Path, default=Path('experiments/exp_008_beamng/history_control_out/20261029_mixed_best'))
    parser.add_argument('--terrain', type=Path, default=TERRAIN_LEVEL)
    parser.add_argument('--beamng-home', type=Path, default=Path.home() / 'BeamNG.tech.v0.38.5.0')
    parser.add_argument('--vehicle-config', type=Path, default=VEHICLE_CONFIG)
    parser.add_argument('--h4-selection', choices=['best', 'original'], default='best')
    parser.add_argument('--velocities', type=int, nargs='+', default=[-30, -50, 80])
    parser.add_argument('--locations', type=int, nargs='+', default=[0, 800, 1500])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--samples', type=int, help='override MPPI samples equally for both models')
    parser.add_argument('--resume', action='store_true', help='continue matching saved episodes after an infrastructure failure')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='restore both models and check controller calls without launching BeamNG')
    mode.add_argument('--preflight-only', action='store_true', help='check both controllers and simulator startup, then stop')
    return parser.parse_args()


def stage_vehicle_config(source, user):
    """A missing custom pickup config otherwise silently loads another truck."""
    config = json.loads(source.read_text())
    if config.get('model') != 'pickup' or not config.get('parts'):
        raise ValueError(f'invalid pickup configuration: {source}')
    destination = user / 'current/vehicles/pickup/hitch.pc'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return config, destination


def check_vehicle_parts(tree, expected):
    chosen = {}
    def visit(value):
        if isinstance(value, dict):
            if 'id' in value and 'chosenPartName' in value:
                chosen[value['id']] = value['chosenPartName'] or ''
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(tree)
    critical = ('pickup_engine', 'pickup_transmission', 'pickup_towhitch',
                'pickup_receiver_attachment', 'pickup_ESC', 'pickup_steering',
                'tire_F_16x7_alt', 'tire_R_16x7_alt')
    mismatches = {key: dict(expected=expected[key], actual=chosen.get(key))
                  for key in critical if chosen.get(key, '') != expected[key]}
    if mismatches:
        raise ValueError(f'loaded pickup configuration differs: {mismatches}')
    if not chosen:
        raise ValueError('simulator returned no selected vehicle parts')
    return {key: chosen.get(key, '') for key in critical}


def run_paired_episodes(args, configs, make_environment):
    """Reuse the production episode and scoring functions with paired ordering."""
    output = args.output / 'sweep'
    output.mkdir(exist_ok=True)
    summaries = {}
    for path in sorted(output.glob('**/raw_*.npz')):
        with np.load(path) as data:
            episode = {key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data.files}
        summary = sweep.summarize(episode)
        key = (summary['controller'], summary['loc'], summary['v_kph'], summary['seed'])
        if key in summaries:
            raise ValueError(f'duplicate episode: {key}')
        summaries[key] = summary

    def flush():
        rows = list(summaries.values())
        sweep.write_csv(output / 'episodes.csv', sweep.EP_COLS, rows)
        groups = defaultdict(list)
        for row in rows:
            if row['outcome'] != 'interrupted':
                groups[row['controller'], row['loc'], row['v_kph']].append(row)
        sweep.write_csv(output / 'agg.csv', sweep.AGG_COLS,
                        [sweep.aggregate(group) for group in groups.values()])

    flush()
    for velocity in args.velocities:
        planners = {kind: sweep.make_mpc(velocity, kind) for kind in configs}
        for loc_index, location in enumerate(args.locations):
            for seed_index, seed in enumerate(args.seeds):
                # Alternate which history runs first within the matched pair.
                order = list(configs)[::(-1 if seed_index % 2 else 1)]
                for kind in order:
                    key = (kind, location, velocity, seed)
                    if key in summaries:
                        print(f'Reusing completed episode {key}', flush=True)
                        continue
                    env = None
                    try:
                        env = make_environment(config=sweep.scenario, use_custom_mu=False, headless=True)
                        episode = sweep.run_episode(env, planners[kind], velocity, seed, kind, location)
                    finally:
                        if env is not None:
                            env.close()
                    episode['run_index'] = loc_index * len(args.seeds) + seed_index
                    summary = sweep.summarize(episode)
                    np.savez_compressed(output / f'raw_{kind}_loc{location}_v{velocity}_s{seed}.npz',
                                        **episode, distance_survived_m=summary['distance_survived_m'])
                    summaries[key] = summary
                    flush()
                    print(f"Saved {len(summaries)}/{2 * len(args.locations) * len(args.velocities) * len(args.seeds)}: "
                          f"{key}, {summary['outcome']}, speed={summary['avg_vx_kph']:.2f} km/h, "
                          f"hitch RMS={summary['hitch_rms_deg']:.2f} deg, "
                          f"distance={summary['distance_survived_m']:.1f} m", flush=True)
        del planners
        jax.clear_caches()


def main():
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f'choose a fresh --output directory: {args.output}')
    if args.steps < 1 or args.samples is not None and args.samples < 1:
        raise ValueError('steps and samples must be positive')
    if not (args.terrain / 'terrain8.ter').is_file():
        raise FileNotFoundError(args.terrain / 'terrain8.ter')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    user = args.output.resolve() / 'beamng_user'
    level = user / 'current/levels/tech_ground'
    if not level.exists():
        shutil.copytree(args.terrain, level)
    vehicle_config, staged_vehicle_config = stage_vehicle_config(args.vehicle_config, user)
    cases = json.loads((args.bundle / 'bundled_cases.json').read_text())
    configs = {}
    provenance = {}
    for history in (1, 4):
        name = ('h4_mixed_rollout10' if history == 4 and args.h4_selection == 'original'
                else f'h{history}_mixed_best')
        case = next(c for c in cases if c['name'] == name)
        checkpoint = (args.bundle / case['checkpoint']).resolve()
        stats = (args.bundle / case['stats']).resolve()
        configs[f'model_h{history}'] = (history, stats, checkpoint)
        provenance[f'model_h{history}'] = dict(
            history=history, training_friction='mixed', checkpoint_selection=case['selection'],
            objective=('one-step training plus 10-step rollout refinement' if case['refined']
                       else 'one-step normalized derivative MSE'), rollout_refinement=case['refined'],
            checkpoint=str(checkpoint), stats=str(stats),
            stats_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),
        )
    sweep.MODEL_CONFIGS.clear()
    sweep.MODEL_CONFIGS.update(configs)
    sweep.load_learned_model.cache_clear()
    sweep.restore_model = replay.restore_checked
    sweep.VELS_KPH, sweep.LOCS, sweep.SEEDS = args.velocities, args.locations, args.seeds
    sweep.MAX_STEPS = args.steps
    sweep.WARMUP_STEPS = 4
    sweep.HEADLESS = True
    sweep.OUT_ROOT = args.output.resolve() / 'sweep'
    if args.samples is not None:
        sweep.FWD_MPPI['K'] = sweep.REV_MPPI['K'] = args.samples
    protocol = dict(
        purpose=('Original rollout-refined H=4 versus best one-step H=1 in closed-loop MPPI'
                 if args.h4_selection == 'original' else
                 'Effect of history length on matched closed-loop MPPI performance'),
        models=provenance, environment='snow+slope',
        terrain_source=str(args.terrain.resolve()), staged_level=str(level),
        terrain_sha256=hashlib.sha256((level / 'terrain8.ter').read_bytes()).hexdigest(),
        simulator_user=str(user), nogpu=False,
        vehicle_config_source=str(args.vehicle_config.resolve()),
        staged_vehicle_config=str(staged_vehicle_config),
        vehicle_config_sha256=hashlib.sha256(staged_vehicle_config.read_bytes()).hexdigest(),
        locations=args.locations, velocities_kph=args.velocities, seeds=args.seeds,
        max_steps=args.steps, warmup_steps=sweep.WARMUP_STEPS,
        episodes=2 * len(args.locations) * len(args.velocities) * len(args.seeds),
        fwd_mppi={k: np.asarray(v).tolist() for k, v in sweep.FWD_MPPI.items()},
        rev_mppi={k: np.asarray(v).tolist() for k, v in sweep.REV_MPPI.items()},
        fwd_weights=sweep.FWD_WEIGHTS, rev_weights=sweep.REV_WEIGHTS,
        scenario_config=asdict(sweep.scenario),
        jax_devices=[str(d) for d in jax.devices()],
        pairing='Identical physical warmup, starts, targets and random keys; H=1 retains the last measured row and H=4 all four rows',
        simulator_lifecycle='Fresh process for each episode to avoid scenario-reload crashes',
    )
    if args.resume:
        previous = json.loads((args.output / 'protocol.json').read_text())
        for key in ('models', 'terrain_sha256', 'vehicle_config_sha256', 'nogpu', 'locations', 'velocities_kph', 'seeds', 'max_steps',
                    'warmup_steps', 'fwd_mppi', 'rev_mppi', 'fwd_weights', 'rev_weights', 'scenario_config'):
            if previous[key] != protocol[key]:
                raise ValueError(f'cannot resume with changed {key}')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(__file__, args.output / 'runner_source.py')
    shutil.copy2(sweep.__file__, args.output / 'sweep_source.py')
    checks = []
    # Validate production controller construction and actual MPPI calls using
    # measured states. This is a readiness check, not simulator performance.
    for kind, (history, _, _) in ([] if args.resume else configs.items()):
        for velocity in args.velocities:
            direction = 'forward' if velocity > 0 else 'backward'
            with np.load(TURN_SNOW / 'both4_div' / direction / 'expert_learned_h4.npz') as archive:
                context = archive['initial_history'][-history:]
            planner = sweep.make_mpc(velocity, kind)
            planner.key = jax.random.key(args.seeds[0])
            control = np.asarray(planner.run_mpc(jnp.asarray(context.reshape(-1))))
            if control.shape != (2,) or not np.isfinite(control).all():
                raise ValueError(f'invalid readiness control: {kind}/{velocity}: {control}')
            checks.append(dict(controller=kind, velocity_kph=velocity, history=planner.history,
                               samples=planner.K, horizon=planner.T, control=control.tolist()))
            print(f'Controller ready: {kind}, target={velocity} km/h, H={planner.history}, u={control}', flush=True)
            del planner
            jax.clear_caches()
    if not args.resume:
        (args.output / 'controller_checks.json').write_text(json.dumps(checks, indent=2))
    if args.prepare_only:
        return

    status = {'status': 'starting'}
    (args.output / 'simulator_preflight.json').write_text(json.dumps(status, indent=2))
    simulator = None
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
        simulator = BeamNGpy('localhost', 25252, home=args.beamng_home, user=user,
                            headless=True, nogpu=False)
        simulator.open(launch=True)
        paths = simulator.system.get_environment_paths()
        status = dict(status='connected', paths=paths, info=simulator.system.get_info())
        (args.output / 'simulator_preflight.json').write_text(json.dumps(status, indent=2, default=str))
        print('Simulator connected:', paths, flush=True)
    except Exception as error:
        status = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
        (args.output / 'simulator_preflight.json').write_text(json.dumps(status, indent=2))
        raise
    finally:
        if simulator is not None:
            simulator.close()
    original_env = sweep.BeamNGTrailerEnv
    owned_simulators = []
    def make_environment(**kwargs):
        env = original_env(**kwargs, beamng_home_dir=args.beamng_home,
                           beamng_user_dir=user, nogpu=False)
        if env.bng.process is not None:
            owned_simulators.append(env.bng)
        env.close = env.bng.close
        tree = env.tractor.get_part_config()
        try:
            # A null graphics device can omit TerrainBlock while still allowing
            # the vehicle to drive on a flat default surface. Require the level.
            env.bng.control.queue_lua_command(
                "assert(scenetree.findObject('theTerrain'), 'Required terrain collision object is missing'); return true",
                response=True,
            )
            selected = check_vehicle_parts(tree, vehicle_config['parts'])
        except Exception:
            env.close()
            raise
        checks = args.output / 'vehicle_checks'
        checks.mkdir(exist_ok=True)
        (checks / f'loaded_{len(owned_simulators):03d}.json').write_text(json.dumps(
            dict(config_sha256=protocol['vehicle_config_sha256'], selected_parts=selected), indent=2))
        return env
    try:
        if args.preflight_only:
            env = make_environment(config=sweep.scenario, use_custom_mu=False, headless=True)
            env.close()
        else:
            run_paired_episodes(args, configs, make_environment)
    finally:
        for owned in owned_simulators:
            if owned.process is not None and owned.process.poll() is None:
                owned.close()


if __name__ == '__main__':
    main()
