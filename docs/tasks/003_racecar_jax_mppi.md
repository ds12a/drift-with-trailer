# Task 003: Racecar Track Following With JAX MPPI and Smooth MPPI

## Purpose

Task 1 introduced nonlinear MPC on cartpole. Task 2 introduced MPPI and Smooth MPPI, including JAX implementations. Task 3 moves from cartpole to a racecar environment.

The goal is to initialize the `uncertain-racecar-gym` submodule and use its nominal Gymnasium racecar environment to drive around a track with our JAX MPPI and JAX Smooth MPPI controllers.

For this task, do not use the uncertainty models. Also do not spend time on Blender rendering. Use the normal Gymnasium `reset` / `step` interface and, when you render, use the PyBullet renderer only.

## Learning Goals

By the end of this task, you should be able to:

- Add and work with a Git submodule.
- Use an external Gymnasium environment without copying its source into this repository.
- Explain the bicycle model used by the racecar simulator.
- Adapt the existing JAX MPPI and Smooth MPPI controllers to a higher-dimensional vehicle system.
- Design a track-following cost function based on the MPPI papers and the racecar submodule baseline.
- Evaluate controller performance over randomized trials.
- Compare forward driving and backward driving behavior.
- Explain why high-speed vehicle control is harder than low-speed vehicle control.

## Reading and References

Use these references:

- Racecar environment: the `uncertain-racecar-gym` submodule
- MPPI paper in `docs/reading-list/papers.md`
- Smooth MPPI paper in `docs/reading-list/papers.md`

The racecar repository already includes JAX MPPI and Smooth MPPI baselines. You may read those files to understand the environment and cost design, but the controller used in this task should be our own JAX MPPI / Smooth MPPI implementation from this repository, adapted for the racecar.

## Repository Locations

Initialize the existing racecar environment submodule at the repository root:

```text
uncertain-racecar-gym/
```

Use this command:

```bash
git submodule update --init --recursive
```

Put reusable controller or adapter code here:

```text
src/controllers/mpc/
```

Suggested files:

```text
src/controllers/mpc/racecar_mppi_jax.py
src/controllers/mpc/racecar_smppi_jax.py
src/controllers/mpc/racecar_costs.py
```

Put experiment-specific code here:

```text
experiments/exp_003_racecar_mppi/
```

Put runnable entrypoints here:

```text
examples/
```

Suggested runnable files:

```text
examples/run_racecar_mppi_jax.py
examples/run_racecar_smppi_jax.py
examples/run_racecar_speed_sweep.py
examples/run_racecar_backward_sweep.py
```

## Environment Setup Notes

After initializing the submodule, install it through `uv` in editable mode:

```bash
uv add --editable uncertain-racecar-gym
uv add jax jaxlib pybullet matplotlib pandas pyyaml scipy imageio
```

Check the dependency versions carefully. At the time this task was written, `uncertain-racecar-gym` requires `gymnasium>=0.29,<1.0`, while this repository may already have a newer Gymnasium version from the cartpole tasks. Resolve the dependency conflict in `pyproject.toml` so both the cartpole code and racecar code run.

Use a nominal environment:

```python
import gymnasium as gym
import uncertain_racecar_gym

env = gym.make(
    "UncertainRacecar-v0",
    scenario="package://scenarios/ks_barcelona_layout_gp_dallara_f317_rl_long.yaml",
    uncertainty=None,
    renderer="pybullet",
    render_mode="rgb_array_follow",
)
```

The action is:

```text
[steer_cmd, throttle_cmd, brake_cmd]
```

with approximate bounds:

```text
steer_cmd in [-1, 1]
throttle_cmd in [0, 1]
brake_cmd in [0, 1]
```

The observation includes compact track-following features such as progress, lateral error, heading error, longitudinal velocity, lateral velocity, yaw rate, current curvature, lookahead curvature, and recent action history.

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 003: Racecar Track Following With JAX MPPI and Smooth MPPI
```

2. In the issue, briefly describe your plan:

- How you will add and install the submodule.
- Which scenario/track you will use.
- How you will adapt the existing JAX MPPI and Smooth MPPI controller interface.
- What cost terms you will use.
- What target speeds you will evaluate.
- How you will measure success, speed, slip angle, and yaw rate.

3. Create a branch from `main`. Use a branch name like:

```text
task-003-racecar-jax-mppi
```

4. Make small commits with clear messages. Example:

```text
Add uncertain racecar gym submodule
Add racecar JAX MPPI adapter
Add racecar Smooth MPPI adapter
Add racecar speed sweep benchmark
Document task 003 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Added `uncertain-racecar-gym` as a Git submodule.
- Interfaced nominal racecar environment with our JAX MPPI controller.
- Interfaced nominal racecar environment with our JAX Smooth MPPI controller.
- Added forward and backward driving experiments.

## Dynamics Summary

Briefly describe the state, input, output, and equations of motion using LaTeX.

## Forward Driving Results

- Track/scenario:
- Random seeds:
- Target speeds tested:
- MPPI success rate:
- Smooth MPPI success rate:
- MPPI average speed:
- Smooth MPPI average speed:
- MPPI average slip angle:
- Smooth MPPI average slip angle:
- MPPI average yaw rate:
- Smooth MPPI average yaw rate:

## Backward Driving Results

- How backward driving was defined:
- Target speeds tested:
- MPPI success rate:
- Smooth MPPI success rate:
- Main failure modes:

## Tuning Notes

- Tuning method, for example manual tuning, Optuna, W&B, or a combination:
- Tuning target speed:
- Main parameters tuned:
- Hyperparameter tuning range for each tuned parameter:
- Parameters kept fixed across the speed sweep:
- Notes for reproducing the tuning:

## Answers

### Why is the vehicle harder to control at higher target speeds?

### What is the role of slip penalty in the cost function in this racing task?

### Which driving mode, forward or backward, is harder to control? What can be the reasons?

### What do you think about the fidelity of the bicycle model used here? Is it accurate enough to incorporate road friction? If not, what should we consider in the dynamics?

## Checklist

- [ ] Submodule is added under `uncertain-racecar-gym/`
- [ ] `git submodule update --init --recursive` works
- [ ] Code runs with `uv run ...`
- [ ] Uncertainty models are disabled for this task
- [ ] Blender rendering is not required
- [ ] PyBullet rendering is used only if rendering is needed
- [ ] JAX MPPI is interfaced with the racecar environment
- [ ] JAX Smooth MPPI is interfaced with the racecar environment
- [ ] Forward speed sweep results are reported
- [ ] Backward driving results are reported
- [ ] Dynamics equations are written in the report
- [ ] Written question answers are included
```

## Technical Requirements

Your implementation should include:

- `uncertain-racecar-gym` as a Git submodule.
- A nominal racecar experiment using `uncertainty=None`.
- An adapter between the racecar state/action representation and our JAX MPPI controller.
- An adapter between the racecar state/action representation and our JAX Smooth MPPI controller.
- A racecar rollout function suitable for JAX MPPI sampling.
- A track-following cost function.
- A forward-driving speed sweep.
- A backward-driving experiment.
- A report with equations, metrics, plots or tables, and answers to the written questions.

Use the submodule environment's `step` function for real evaluation. For MPPI internal rollouts, use a JAX-compatible nominal dynamics function. The submodule's `uncertain_racecar_gym.jax_env.step_nominal` is a useful reference for how the nominal bicycle model is written in JAX.

Do not use the submodule's uncertainty model in this task. The purpose is to first make our controller work on the nominal racecar system.

## Vehicle Dynamics To Report

In your report, write the bicycle model in LaTeX. Use the simulator source as the reference.

At minimum, include:

- State definition.
- Input definition.
- Equations of motion.
- A short note explaining how the global pose is projected to track progress, lateral error, and heading error.

## Suggested Cost Function

Refer to the MPPI and Smooth MPPI papers for the cost-function design. Also look at the racecar submodule's MPPI implementation to see how track progress, lateral error, heading error, speed tracking, action smoothness, and off-track penalties are applied.

You do not need to write a long derivation in the task report. Clearly list the cost terms you used and explain why they are needed for racing.

## Forward Driving Experiment

Tune hyperparameters at one target speed only. Use `100 km/h` as the recommended tuning speed if the controller can reach it. You may consider using Optuna for hyperparameter search and W&B for tracking runs, plots, and metrics. Keep the tuning process simple enough that another student can reproduce it.

Convert speeds to meters per second in code:

```text
target_speed_mps = target_speed_kmh / 3.6
```

After tuning, keep the hyperparameters fixed. Then evaluate a speed sweep:

```text
30 km/h, 40 km/h, 50 km/h, ...
```

Increase the target speed by `10 km/h` until the controller fails often enough that the method is clearly no longer reliable.

For each target speed and controller, run several randomized trials. Use random initial progress, small lateral error, small heading error, and reasonable initial speed. Report at least:

- Success rate: fraction of trials that complete one track/lap without collision or off-track termination.
- Average speed.
- Average slip angle magnitude.
- Average yaw-rate magnitude.

You may also report:

- Lap time.
- Mean lateral error.
- Maximum lateral error.
- Collision/off-track count.
- Average control-update time.
- Rendered PyBullet GIF or video.

## Backward Driving Experiment

Run the same style of evaluation for backward driving.

Be careful: the current racecar environment action and dynamics are designed mainly for forward driving. The action has throttle and brake, but no explicit reverse gear, and the nominal dynamics clamp longitudinal velocity to be nonnegative. Therefore, true reverse-gear driving may not be possible without changing the simulator.

For this task, do not redesign the simulator. Instead:

- Define clearly what you mean by backward driving.
- Try a reverse-track objective, reversed heading objective, or negative-progress objective using the existing environment.
- Report what actually happens.
- If the model prevents true backward driving, say that explicitly and explain which modeling assumptions caused it.

Use the same hyperparameters that you tuned for forward driving unless a small change is absolutely necessary. If you change anything, report it.

## Written Question 1

### Why is the vehicle harder to control at higher target speeds?

Attach the answer to the PR.

Discuss tire slip, reduced time to react, stronger coupling between lateral and longitudinal dynamics, higher yaw-rate sensitivity, longer stopping distance, and the fact that small steering errors produce larger path deviations at high speed.

## Written Question 2

### What is the role of slip penalty in the cost function in this racing task?

Attach the answer to the PR.

Discuss how slip penalty discourages sideways motion and unstable yaw behavior, but can also make the controller too conservative if the weight is too high.

## Written Question 3

### Which driving mode, forward or backward, is harder to control? What can be the reasons?

Attach the answer to the PR.

Discuss the simulator's forward-driving assumptions, the lack of reverse gear, the nonnegative longitudinal-speed clamp, track-heading conventions, tire-force behavior, and whether the controller cost was designed around forward progress.

## Written Question 4

### What do you think about the fidelity of the bicycle model used here? Is it accurate enough to incorporate road friction? If not, what should we consider in the dynamics?

Attach the answer to the PR.

Discuss what the current bicycle model captures, what it simplifies, and what may be missing for friction-aware racing. Consider tire saturation, nonlinear tire models, load transfer, combined longitudinal/lateral tire forces, varying road friction, actuator limits, and whether the controller would need online friction estimation or adaptation.

## What To Submit In The PR

Your PR should include:

- The Git submodule entry.
- Source code for the racecar JAX MPPI interface.
- Source code for the racecar JAX Smooth MPPI interface.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- A short experiment README explaining how to run your code.
- A report with vehicle dynamics equations in LaTeX.
- Forward-driving speed sweep results.
- Backward-driving results.
- Answers to the written questions.

The most important thing is to produce a clean interface and an honest report. It is okay if high-speed or backward driving fails. The point is to understand why, measure it clearly, and explain the controller and model limitations.
