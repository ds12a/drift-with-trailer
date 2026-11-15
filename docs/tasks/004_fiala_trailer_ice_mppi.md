# Task 004: Fiala Racecar, Trailer, and Ice Patches

## Purpose

Task 3 tested JAX MPPI and Smooth MPPI on the racecar setting and asked you to think about the fidelity of the bicycle model. Task 4 is the next step: implement a better dynamics model, add a trailer, and then test how the controller behaves when friction changes unexpectedly.

This task has four subtasks:

1. Implement a dynamic bicycle model with Fiala tire forces.
2. Extend that model with a realistic trailer.
3. Add black-ice patches, but keep the MPPI rollout model at nominal friction.
4. Add black-ice patches again, but let the MPPI rollout model update friction only after the robot has actually stepped onto ice.

For all four subtasks, run MPPI and Smooth MPPI in forward and backward driving, report results in tables, and include GIFs for representative success and failure cases.

## Learning Goals

By the end of this task, you should be able to:

- Implement a JAX vehicle dynamics model with Fiala tire forces.
- Separate actual simulation dynamics from controller rollout dynamics.
- Add a trailer state and trailer tire forces while keeping the same control input.
- Design dynamics classes so the bicycle-only and bicycle-with-trailer models are swappable.
- Use PyBullet rendering support without depending on the old Gym environment for dynamics.
- Add spatially varying friction patches and verify that friction changes affect the actual dynamics.
- Compare MPPI and Smooth MPPI under nominal, trailer, and low-friction conditions.
- Explain why trailers and low-friction surfaces make high-speed control harder.

## Reading and References

Use these references:

- PL-CBF paper that explains this dynamics: <https://arxiv.org/pdf/2605.16588v1>
- Simple acceleration-controlled tractor-trailer reference (note that this only gives an example of a kinematic trailer): <https://github.com/cps-atlas/safe-mpd/blob/main/mbd/robots/acc_tt2d.py>
- Semitrailer dynamics references in `docs/reading-list/papers.md`
- MPPI and Smooth MPPI papers in `docs/reading-list/papers.md`

The external references are starting points. Your implementation for this task should live in this repository.

## Important Scope Change From Task 3

Do not use `uncertain-racecar-gym` as the simulator for this task and later.

You may reuse (bring) only the essential PyBullet rendering ideas or code needed to visualize the car, trailer, track, and ice patches. Do not bring in the uncertainty models, Gym environment, track-following environment logic, Blender assets, or unrelated utilities. If you copy or adapt rendering code, preserve attribution and any license notes.

The dynamics, step function, rollout functions, friction map, and experiments should be implemented in this repository.

## Repository Locations

Put reusable dynamics code here:

```text
src/dynamics/vehicle/
src/dynamics/trailer/
```

Suggested files:

```text
src/dynamics/vehicle/dynamic_bicycle_fiala.py
src/dynamics/trailer/dynamic_bicycle_trailer_fiala.py
src/dynamics/vehicle/friction_map.py
```

Put reusable controller or cost code here:

```text
src/controllers/mpc/
```

Suggested files:

```text
src/controllers/mpc/racecar_fiala_costs.py
src/controllers/mpc/racecar_fiala_mppi_jax.py
src/controllers/mpc/racecar_fiala_smppi_jax.py
```

Put rendering utilities here:

```text
src/simulation/
```

Suggested files:

```text
src/simulation/bullet_racecar_renderer.py
src/simulation/track_geometry.py
```

Put experiment-specific code here:

```text
experiments/exp_004_fiala_trailer_ice/
```

Put runnable entrypoints here:

```text
examples/
```

Suggested runnable files:

```text
examples/run_fiala_bicycle_mppi.py
examples/run_fiala_trailer_mppi.py
examples/run_fiala_trailer_ice_nominal_rollout.py
examples/run_fiala_trailer_ice_adaptive_rollout.py
```

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 004: Fiala Racecar, Trailer, and Ice Patches
```

2. In the issue, briefly describe your plan:

- Which dynamics files you will add.
- How you will make bicycle-only and bicycle-with-trailer dynamics swappable.
- How you will adapt MPPI and Smooth MPPI rollout dynamics.
- What track and target speeds you will test.
- How you will implement and visualize ice patches.
- What results tables and GIFs you plan to include.

3. Create a branch from `main`. Use a branch name like:

```text
task-004-fiala-trailer-ice
```

4. Make small commits with clear messages. Example:

```text
Add Fiala bicycle dynamics
Add trailer dynamics model
Add black ice friction map
Add MPPI ice patch benchmarks
Document task 004 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Implemented dynamic bicycle model with Fiala tire forces.
- Implemented trailer dynamics with Fiala-style tire forces.
- Added black-ice patches and PyBullet visualization.
- Evaluated MPPI and Smooth MPPI in forward and backward driving.

## Dynamics Summary

- Bicycle state:
- Bicycle input:
- Trailer state extension:
- Control input:
- How the trailer dynamics interface with the bicycle dynamics:
- How the dynamics classes are swapped in experiments:

## Experiments

- Track:
- Target speeds:
- Random seeds:
- Controllers tested:
- Tuning method:
- Hyperparameter tuning ranges:

## Results

### Subtask 1: Fiala Bicycle

Include a results table for forward and backward driving.

### Subtask 2: Fiala Bicycle With Trailer

Include a results table for forward and backward driving.

### Subtask 3: Ice Patches, Nominal Rollout Friction

Include a results table for forward and backward driving. For each target speed, report which ice patch caused failure, if any.

### Subtask 4: Ice Patches, Online Rollout Friction Update

Include a results table for forward and backward driving. For each target speed, report which ice patch caused failure, if any.

## Media

- Success GIF:
- Failure GIF:
- Ice patch visualization:

## Answers

### Why is the car harder to control with a trailer?

### Why is the car harder to control at lower friction levels?

### What can we do to get better control in the setting where the actual surface friction changes but the rollout dynamics do not know the change ahead of time?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] Fiala bicycle dynamics are implemented in this repository
- [ ] Trailer dynamics are implemented in this repository
- [ ] Bicycle-only and trailer dynamics are swappable
- [ ] Control dimension stays the same after adding the trailer
- [ ] Only essential PyBullet rendering support is reused
- [ ] Black-ice patches are visualized in PyBullet
- [ ] Actual simulation dynamics use the true local friction
- [ ] Subtask 3 keeps rollout friction nominal
- [ ] Subtask 4 updates rollout friction only after contact with ice
- [ ] MPPI and Smooth MPPI are tested
- [ ] Forward and backward results are reported
- [ ] Success and failure GIFs are included
- [ ] Written question answers are included
```

## Technical Requirements

Your implementation should include:

- A JAX dynamic bicycle model with Fiala tire forces.
- A step function for actual simulation.
- A JAX rollout function for MPPI and Smooth MPPI.
- A trailer dynamics extension.
- A black-ice friction map.
- A PyBullet visualization showing the vehicle, trailer, track, and ice patches (note that the ice patches are visualized only for visual purposes, the vehicle has no information a priori).
- Results tables for all four subtasks.
- GIFs for representative success and failure cases.
- A short report with dynamics explanation, experiment setup, metrics, and written answers.

## Subtask 1: Fiala Dynamic Bicycle

Implement a dynamic bicycle model with Fiala tire forces in this repository. Use the PL-CBF drift-car dynamics code and paper as references. Since the vehicle now drift, you can consider to impose two stages of cost function on the slip angle. (1) a soft penalty for the slip angle, and (2) a large impulse penalty for the state where the slip angle is larger than some threshold.

At minimum, define:

- State.
- Control input.
- Tire slip variables.
- Longitudinal and lateral tire force model.
- Friction coefficient parameter.
- Step function.
- JAX-compatible rollout function.

The model should support changing friction coefficient `mu`. This matters later for black-ice patches.

Use the same style of evaluation as Task 3:

- Forward driving speed sweep.
- Backward driving experiment.
- MPPI and Smooth MPPI.
- Proper tuning at one selected target speed.
- Results table with success rate, average speed, average slip angle, average yaw rate, and any other useful metric.

## Subtask 2: Add Trailer Dynamics

Add a trailer behind the Fiala bicycle model.

The simple kinematic tractor-trailer code is only a starting reference. It assumes low-speed behavior, so it is not enough for this task by itself. For this assignment, implement a trailer model that can represent lateral dynamics and tire-force effects. Use the semitrailer references in the reading list for guidance.

Autonomous driving with the trailer has a very unique challenge. The trailer might hit the car, which is called jackknifing. [Jackknifing](https://youtu.be/hudKbI6ZXps?list=PL86soRB32vYORlmMpcmkWtQ1mX6pbQP_J)
This can be prevented by maintaining the hitch angle to be lower than some threshold. So the MPPI can impulsely penalize the hitch angle if it is higher than some threshold.

Design expectations:

- The trailer dynamics should extend or wrap the dynamic bicycle model.
- The control input should remain the same as the car-only model.
- The state dimension should increase to include trailer pose, articulation angle, trailer yaw rate or lateral velocity as needed, and trailer tire states if you use them.
- The trailer should have its own geometry and center of mass.
- The trailer should have its own tire-force calculation, preferably Fiala-style or at least nonlinear/saturating.
- The step function should update both the car and trailer.
- The dynamics object should be swappable with the bicycle-only dynamics in the same MPPI experiment code.

Run the same tests as Subtask 1 and report the results in a table.

## Subtask 3: Black Ice With Nominal Rollout Friction

Add black-ice patches to the track.

The default surface friction should be:

```text
mu = 1.0
```

Place ice patches at corners with decreasing friction levels:

```text
0.8, 0.7, 0.6, 0.5, 0.4, 0.3
```

For visualization, show each patch as a blue circle or blue transparent disk in the PyBullet renderer.

The actual simulated dynamics must use the true local friction. If the car COM or trailer COM is inside an ice patch, the friction for that body should drop accordingly. If both are on different surfaces, handle that explicitly and document your choice.

Very important: in this subtask, the MPPI rollout dynamics should not know about the ice. The rollout dynamics should keep using nominal friction, such as `mu = 1.0`, even when the actual simulated vehicle is on ice.

For each target speed and controller, report:

- Success rate.
- Average speed.
- Average slip angle.
- Average yaw rate.
- Which ice patch caused failure, if failure occurred.
- Whether failure started from the car, the trailer, or both.

Run forward and backward driving.

## Subtask 4: Black Ice With Online Rollout Friction Update

Repeat Subtask 3, but now the robot updates the friction used by the MPPI rollout dynamics only after it actually steps onto ice.

The robot should not know about the lower-friction patch before contact. This is black ice: the surface change is not anticipated visually or from the map.

Experiment rules:

- Actual simulation dynamics always use the true local friction.
- Before contact, rollout dynamics use nominal friction.
- After the car or trailer COM steps onto an ice patch, the rollout dynamics may update the corresponding friction value.
- Do not preview upcoming patches.
- Do not update friction in the rollout just because an ice patch is visible in the renderer.

JAX warning: make friction a runtime value, not a hard-coded compiled constant. If `mu` is captured as a static argument or constant inside a jitted function, changing `mu` may not actually change the dynamics you are rolling out. Test this explicitly by rolling out the same state/action sequence with two different friction values and confirming the trajectory changes.

Run forward and backward driving, and report the same table as Subtask 3.

## Tuning Guidance

Tune hyperparameters at one chosen target speed, then keep those hyperparameters fixed for speed sweeps and comparisons. You may use manual tuning, Optuna, W&B, or a combination.

In the PR, report:

- Tuning method.
- Tuning target speed.
- Main parameters tuned.
- Hyperparameter tuning range for each tuned parameter.
- Parameters kept fixed across the speed sweep.
- Notes for reproducing the tuning.

If you used W&B and optuna to tune the parameters, also attach the W&D dashboard screenshot.

## Result Tables

Create tables that make it easy to compare:

- Bicycle vs trailer.
- No ice vs ice.
- Nominal rollout friction vs online updated rollout friction.
- MPPI vs Smooth MPPI.
- Forward vs backward driving.

At minimum, include:

- Controller.
- Dynamics model.
- Driving direction.
- Target speed.
- Number of trials.
- Success rate.
- Average speed.
- Average slip angle.
- Average yaw rate.
- First failed ice patch, if applicable.
- Notes on failure mode.

## Written Question 1

### Why is the car harder to control with a trailer?

Attach the answer to the PR.

Discuss articulation dynamics, trailer yaw, delayed response, additional inertia, jackknife risk, off-tracking, and how trailer tire saturation can destabilize the combined system.

## Written Question 2

### Why is the car harder to control at lower friction levels?

Attach the answer to the PR.

Discuss reduced tire force limits, larger slip angles, lower available lateral acceleration, longer braking distance, higher chance of drift, and controller-model mismatch.

## Written Question 3

### What can we do to get better control when the actual surface friction changes but the rollout dynamics do not know the change ahead of time?

Attach the answer to the PR.

Discuss online friction estimation, adaptive MPPI, robust cost design, uncertainty-aware sampling, fallback policies, lower target speeds near suspected low-friction regions, and using observed slip/yaw behavior to update the rollout model.

## What To Submit In The PR

Your PR should include:

- Source code for the Fiala bicycle dynamics.
- Source code for the trailer dynamics.
- Source code for the ice patch friction map.
- Source code for the PyBullet visualization support.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- Results tables for all four subtasks.
- GIFs for success and failure cases.
- A short experiment README explaining how to run everything.
- Answers to the written questions.

This is a larger assignment than the previous tasks. Keep the implementation modular, document assumptions, and make the comparison honest. It is okay if some high-speed or low-friction cases fail; the important part is to identify where and why they fail.
