# Divergence collection

Run from the repository root with the project environment:

```bash
.venv/bin/python -m experiments.exp_008_beamng.run_model_for_div \
  --environment-label dry \
  --output experiments/exp_008_beamng/model_divergence_out/dry_run1
```

The default batch runs **four controllers × two directions × 1,000 steps**:

| Name | Configuration |
| --- | --- |
| `learned_h1` | `beamng-l4-h1`, `data_proc_h1_stats.json`, H=1 |
| `learned_h4` | `beamng-l4-128-test10-new_rollout10`, `data_proc_test10-new_stats.json`, H=4 |
| `fiala_surface` | Fiala prior with independently measured tractor/trailer friction |
| `fiala` | Same Fiala equations with tractor/trailer friction fixed at 1.0 |

Each learned model gets its own feature spec and MPPI history size. Stats must match the expected history length; checkpoint restoration checks parameter shapes. The global H=1 training spec is unchanged. `learned` remains an alias for `learned_h4`.

Defaults: forward target +80 km/h, backward target -30 km/h, MPPI K=1000, planning horizon T=80, dt=0.05 s, spawn index 800, track width 15 m. Each run has four warmup steps when H=4 is selected, followed by 1,000 recorded controller steps (50 simulated seconds). Track/hitch termination flags are recorded in `termination_steps` but do not shorten collection. Nonfinite controls stop that expert with `status=nonfinite_control`; check status and actual array lengths.

After collecting four experts in each direction, all four models replay each expert's commands open loop: **16 predictions per direction, 32 total**. Only the surface-aware prior receives the recorded friction schedule during replay. Physical states are not corrected after initialization. The common stored friction channels describe the observed environment even for predictors that do not consume friction.

## Options

- `--target-kph -30`: run just one direction, directly in the output directory.
- `--forward-kph 60 --backward-kph -20`: change targets for a two-direction batch.
- `--steps 1000 --samples 1000 --horizon 80`: collection length and planner settings.
- `--models learned_h1,learned_h4,fiala_surface,fiala`: select models.
- `--h1-checkpoint PATH --h1-stats PATH` and corresponding `--h4-*`: choose different trained checkpoints and their matching normalization stats.
- `--track-csv PATH --track-width 15 --spawn-index 800`: configure the track and spawn point.
- `--environment-label NAME --output PATH`: record the environment identity and keep each batch separate. Output paths must not already exist.

The environment label is metadata; it does not change simulator surfaces. The BeamNG environment currently builds its scenario on `tech_ground` in `src/simulation/beamng_trailer_env.py`. Configure your desired environment there before each batch. Collection uses `use_custom_mu=False` so it queries actual simulator contact friction instead of applying a global override. Changing `--track-csv` changes the generated track, not the BeamNG level.

## Saved data

Each `forward/` and `backward/` directory contains:

- `manifest.json`: models, checkpoint/stats paths, histories, environment label, scenario config, targets, seed and run settings.
- Four `expert_NAME.npz` files: states `(1001,13)`, BeamNG commands `(1000,2)`, initial history, solve times, completion status, termination step indices and channel names.
- Sixteen `prediction_expert-NAME_model-NAME.npz` files: expert/predicted states, commands, wrapped yaw errors and channel names.
- `summary.csv`: finite fractions and position/hitch/velocity/yaw-rate divergence metrics.

Experts are saved after each completed controller run; predictions are saved individually after collection. Interrupting mid-controller does not save that controller's in-memory partial trajectory. No plotting is performed.
