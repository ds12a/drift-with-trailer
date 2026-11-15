# Task 005: Trailer MPPI Under Model Mismatch

## Purpose

Task 4 introduced a higher-fidelity Fiala tire model, a trailer, black-ice patches, and the distinction between actual simulation dynamics and MPPI rollout dynamics. Task 5 focuses on that distinction more carefully.

In Task 4, one setting allowed the controller to update its internal rollout friction as soon as the vehicle stepped onto ice. This is useful, but it is still optimistic. In a real system, the controller may not immediately know the correct friction, and the model used inside MPPI may be much simpler than the true system.

For this task, use the trailer setting from Task 4. Car-only experiments can be skipped. The goal is to compare controller performance when the actual simulator remains high fidelity, but the model used inside MPPI is wrong in two different ways:

1. The rollout dynamics keep nominal friction even after the actual vehicle drives onto ice.
2. The rollout dynamics use a reduced-order kinematic tractor-trailer model instead of the Fiala trailer model.

The important idea is:

```text
actual simulator step function != MPPI rollout dynamics
```

The actual simulator must always use the true local friction and the higher-fidelity trailer dynamics. Only the model used by MPPI for sampled rollouts changes between experiments.

## Learning Goals

By the end of this task, you should be able to:

- Separate actual simulation dynamics from controller rollout dynamics.
- Evaluate how model mismatch affects MPPI and Smooth MPPI.
- Compare high-fidelity trailer rollouts against reduced-order kinematic rollouts.
- Explain why unknown friction and simplified dynamics degrade closed-loop control.
- Design experiments that isolate one modeling assumption at a time.
- Report success rates and failure modes across target speeds and driving directions.
- Connect model mismatch to future residual dynamics learning and adaptation.

## Reading and References

Use these references:

- Task 4 specification: `docs/tasks/004_fiala_trailer_ice_mppi.md`
- MPPI and Smooth MPPI papers in `docs/reading-list/papers.md`
- Simple acceleration-controlled tractor-trailer reference from Task 4: <https://github.com/cps-atlas/safe-mpd/blob/main/mbd/robots/acc_tt2d.py>

Task 5 should build on your Task 4 implementation. Do not reintroduce `uncertain-racecar-gym` as the simulator.

## Repository Locations

Put reusable dynamics code here:

```text
src/dynamics/trailer/
src/dynamics/vehicle/
```

Suggested files:

```text
src/dynamics/trailer/kinematic_trailer.py
src/dynamics/trailer/trailer_rollout_models.py
```

Put reusable controller or cost helpers here:

```text
src/controllers/mpc/
```

Suggested files:

```text
src/controllers/mpc/trailer_model_mismatch_costs.py
```

Put experiment-specific code here:

```text
experiments/exp_005_trailer_model_mismatch/
```

Put runnable entrypoints here:

```text
examples/
```

Suggested runnable files:

```text
examples/run_trailer_unknown_friction_mppi.py
examples/run_trailer_kinematic_rollout_mppi.py
examples/run_trailer_model_mismatch_sweep.py
```

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 005: Trailer MPPI Under Model Mismatch
```

2. In the issue, briefly describe your plan:

- Which Task 4 trailer simulator code you will reuse.
- How you will keep actual simulation dynamics separate from MPPI rollout dynamics.
- How you will implement the nominal-friction rollout experiment.
- How you will implement the kinematic tractor-trailer rollout model.
- What target speeds and driving directions you will test.
- What metrics and result tables you will report.

3. Create a branch from `main`. Use a branch name like:

```text
task-005-trailer-model-mismatch
```

4. Make small commits with clear messages. Example:

```text
Add nominal-friction trailer rollout mode
Add kinematic tractor-trailer rollout model
Add trailer model mismatch speed sweep
Document task 005 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Added trailer MPPI experiments with unknown friction in the rollout model.
- Added reduced-order kinematic tractor-trailer rollout dynamics.
- Compared MPPI and Smooth MPPI against the Task 4 online-friction baseline.
- Evaluated forward and backward driving across target speeds.

## Dynamics Separation

- Actual simulator dynamics:
- Actual simulator friction update rule:
- MPPI rollout dynamics for unknown-friction experiment:
- MPPI rollout dynamics for kinematic experiment:
- How these dynamics are selected in code:

## Kinematic Rollout Model

- State:
- Input:
- Equations of motion:
- Assumptions:
- What effects are ignored compared with the Fiala trailer model:

## Experiments

- Track:
- Ice patch configuration:
- Target speeds:
- Driving directions:
- Controllers tested:
- Random seeds:
- Task 4 controller parameters reused:
- Any small parameter changes, if absolutely needed:

## Results

### Baseline: Task 4 Online Rollout Friction Update

Summarize or link the Task 4 baseline used for comparison.

### Subtask 1: Fiala Trailer Rollout With Unknown Friction

Include a results table for forward and backward driving.

### Subtask 2: Kinematic Trailer Rollout

Include a results table for forward and backward driving.

## Media

- Success case link:
- Failure case link:
- Notes on failure mode:

Do not commit large GIF files directly to git. Upload GIFs to a GitHub issue or PR comment, then paste the uploaded media links here.

## Answers

### How much performance degradation do you observe when the rollout friction is never updated?

### Why can a kinematic rollout model still sometimes work for MPPI?

### When does the kinematic rollout model fail compared with the Fiala trailer rollout model?

### What data or residual model would you want to learn next to reduce this mismatch?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] Actual simulation dynamics always use true local friction
- [ ] MPPI rollout dynamics are separate from the simulator step dynamics
- [ ] Unknown-friction rollout keeps nominal friction after entering ice
- [ ] Kinematic tractor-trailer rollout model is implemented
- [ ] Kinematic rollout uses the same control interface as the Fiala trailer controller
- [ ] MPPI and Smooth MPPI are tested
- [ ] Forward and backward driving are tested
- [ ] Target-speed sweep is reported
- [ ] Results are compared against the Task 4 online-friction baseline
- [ ] Controller parameters reuse the best Task 4 settings unless a small change is justified
- [ ] Large media files are not committed to git
- [ ] Written question answers are included
```

## Technical Requirements

Your implementation should include:

- A trailer-only experiment setup based on Task 4.
- A clear way to choose the rollout model used by MPPI and Smooth MPPI.
- An actual simulator step function that always uses the true local friction.
- An unknown-friction rollout mode where MPPI always uses nominal friction, such as `mu = 1.0`.
- A reduced-order kinematic tractor-trailer rollout model.
- Forward and backward speed sweeps.
- Results tables comparing MPPI, Smooth MPPI, rollout model, direction, and target speed.
- A short experiment README explaining how to run the task.
- Answers to the written questions.

## Subtask 1: Unknown Friction in the Rollout Model

Repeat the Task 4 trailer ice experiment, but do not update the friction inside MPPI rollouts.

The actual simulator should still behave realistically:

- If the car COM is on ice, the car tire friction should drop.
- If the trailer COM is on ice, the trailer tire friction should drop.
- If the car and trailer are on different surfaces, handle that case explicitly and document the choice.

The controller should not know this friction change:

- MPPI rollout dynamics should keep nominal friction.
- Smooth MPPI rollout dynamics should keep nominal friction.
- Do not update rollout friction when the vehicle enters ice.
- Do not preview ice patches from the renderer or map.

Compare this setting against the Task 4 setting where the rollout friction was updated after contact.

Run:

- Forward driving.
- Backward driving.
- Multiple target speeds.
- MPPI.
- Smooth MPPI.

For each target speed, report whether performance degrades compared with the Task 4 online-friction baseline.

## Subtask 2: Kinematic Tractor-Trailer Rollout Model

Now keep the same high-fidelity actual simulator, but replace the MPPI rollout dynamics with a simple kinematic tractor-trailer model.

Use acceleration and steering as the control input, keeping the same control interface used by the Task 4 trailer controller.

The rollout state should be:

```text
x, y, theta_1, theta_2, v
```

where:

- `x, y` are the tractor position.
- `theta_1` is the tractor yaw.
- `theta_2` is the trailer yaw.
- `v` is the longitudinal speed.

The input should be:

```text
steering, acceleration
```

The kinematic rollout model should not include:

- Tire forces.
- Fiala tire saturation.
- Lateral slip angle.
- Yaw-rate dynamics.
- Explicit friction coefficient.

This gives a reduced-order model inside the controller while the actual simulator remains high fidelity. The purpose is to measure how much control performance degrades when MPPI plans with a simplified model.

Run the same forward and backward speed sweeps as Subtask 1.

## Controller Parameter Reuse

This task should not require a new controller tuning effort.

Use the best MPPI and Smooth MPPI parameters you found in Task 4 as the default controller settings. The main point of Task 5 is to isolate model mismatch, not to re-optimize the controller for every mismatch case.

In the PR, report:

- Which Task 4 run or configuration your parameters came from.
- The reused horizon, sample count, temperature, noise covariance, smoothness settings, cost weights, and control limits.
- Whether any small parameter change was necessary to make the experiment run.
- Why you made that change, if any.

Do not run a new tuning sweep for this task. The required comparison should use the Task 4 parameters as much as possible.

## Result Tables

Create tables that make the comparison easy to read.

At minimum, include:

- Controller.
- Rollout model.
- Actual simulator model.
- Driving direction.
- Target speed.
- Number of trials.
- Success rate.
- Average speed.
- Average slip angle.
- Average yaw rate.
- Average hitch angle.
- First failed ice patch, if applicable.
- Notes on failure mode.

Recommended rollout model labels:

```text
Fiala trailer, online friction update
Fiala trailer, nominal friction only
Kinematic trailer
```

## Written Question 1

### How much performance degradation do you observe when the rollout friction is never updated?

Attach the answer to the PR.

Discuss success rate, target speed, slip angle, yaw rate, hitch angle, and failure location.

## Written Question 2

### Why can a kinematic rollout model still sometimes work for MPPI?

Attach the answer to the PR.

Discuss local planning, receding-horizon feedback, low-speed behavior, and why an approximate model can be enough when the system stays near the model assumptions.

## Written Question 3

### When does the kinematic rollout model fail compared with the Fiala trailer rollout model?

Attach the answer to the PR.

Discuss high speed, low friction, lateral slip, tire saturation, trailer articulation, and delayed trailer response.

## Written Question 4

### What data or residual model would you want to learn next to reduce this mismatch?

Attach the answer to the PR.

Discuss what the reduced-order model is missing and which signals would help learn a correction, such as slip angle, yaw rate, lateral velocity, hitch angle, local friction, and tracking error.

## What To Submit In The PR

Your PR should include:

- Source code for the rollout model selection.
- Source code for the kinematic tractor-trailer rollout model.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- A short experiment README explaining how to run everything.
- Results tables comparing Task 4 baseline, unknown-friction rollout, and kinematic rollout.
- Links to representative success and failure GIFs, if useful.
- Answers to the written questions.

Large GIFs or videos should not be committed directly to git. Upload media to GitHub comments or issues and link them from the README or PR.
