# Task 002: MPPI and Smooth MPPI CartPole Swing-Up

## Purpose

Task 1 used nonlinear MPC with IPOPT for cartpole swing-up. Task 2 moves to sampling-based MPC. The goal is to implement Model Predictive Path Integral control (MPPI) and Smooth MPPI for the same cartpole swing-up task, using the same environment and controller interface from Task 1.

This task has two implementation phases:

1. Implement MPPI and Smooth MPPI in PyTorch.
2. Re-implement the same controllers in JAX, using `vmap`, `lax.scan`, and `jit` where appropriate.

After both implementations work, compare MPPI vs Smooth MPPI, compare sampling-based MPC vs nonlinear MPC with a nonlinear solver, and compare PyTorch vs JAX runtime.

## Learning Goals

By the end of this task, you should be able to:

- Explain the basic MPPI update rule from the information-theoretic MPC paper.
- Implement a sampling-based MPC controller for cartpole swing-up.
- Implement a smoother action-sequence version of MPPI.
- Use PyTorch tensor operations for batched trajectory rollouts.
- Use JAX `vmap` to parallelize independent sampled trajectories.
- Use JAX `lax.scan` to roll out trajectories efficiently through time.
- Use JAX `jit` to compile rollout and control-update functions.
- Benchmark controller runtime in a fair and repeatable way.
- Discuss the practical tradeoffs between nonlinear-solver MPC, MPPI, Smooth MPPI, PyTorch, and JAX.

## Reading and References

Start from the MPPI and Smooth MPPI papers in the reading list:

```text
docs/reading-list/papers.md
```

Your implementation should fit this repository's controller interface and experiment structure.

## Repository Locations

Put reusable controller code here:

```text
src/controllers/mpc/
```

Suggested controller files:

```text
src/controllers/mpc/mppi_torch.py
src/controllers/mpc/smooth_mppi_torch.py
src/controllers/mpc/mppi_jax.py
src/controllers/mpc/smooth_mppi_jax.py
```

Put experiment-specific code here:

```text
experiments/exp_002_mppi_smooth_mppi/
```

Put runnable entrypoints here:

```text
examples/
```

Good examples of runnable entrypoints:

```text
examples/run_cartpole_mppi_torch.py
examples/run_cartpole_smooth_mppi_torch.py
examples/run_cartpole_mppi_jax.py
examples/run_cartpole_smooth_mppi_jax.py
```

Use the same cartpole environment and control interface as Task 1. Do not create a second incompatible cartpole environment unless the Task 1 environment has not been merged yet; if you need a temporary environment, keep it small and document why.

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 002: MPPI and Smooth MPPI CartPole Swing-Up
```

2. In the issue, briefly describe your plan:

- Which Task 1 cartpole environment/interface you will use.
- Which controller files you expect to add.
- Which PyTorch implementation you will finish first.
- How you plan to structure the JAX implementation.
- What metrics you will use for success and runtime comparison.

3. Create a branch from `main`. Use a branch name like:

```text
task-002-mppi-smooth-mppi
```

4. Make small commits with clear messages. Example:

```text
Add PyTorch MPPI controller
Add PyTorch Smooth MPPI controller
Add JAX MPPI rollout with vmap and scan
Add cartpole MPPI benchmarking experiment
Document task 002 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Implemented MPPI and Smooth MPPI for cartpole swing-up.
- Added PyTorch controller implementations.
- Added JAX controller implementations using `vmap`, `lax.scan`, and `jit`.
- Added benchmark and result summary under `experiments/exp_002_mppi_smooth_mppi/`.

## Results

- Cartpole initial condition:
- MPPI success/failure:
- Smooth MPPI success/failure:
- Best PyTorch MPPI runtime:
- Best PyTorch Smooth MPPI runtime:
- Best JAX MPPI runtime:
- Best JAX Smooth MPPI runtime:
- Hardware used:

## Tuning Notes

- Horizon length:
- Number of samples:
- Temperature / lambda:
- Noise covariance:
- Control limits:
- Cost terms:
- Smoothness penalty or smoothing strategy:

## Answers

### What are the pros and cons of MPPI compared with Smooth MPPI?

### What are the pros and cons of sampling-based MPC compared with nonlinear MPC using nonlinear solvers?

### What are the runtime differences between the PyTorch and JAX implementations?

### When would you prefer PyTorch for this controller, and when would you prefer JAX?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] PyTorch MPPI controller is in `src/controllers/mpc/`
- [ ] PyTorch Smooth MPPI controller is in `src/controllers/mpc/`
- [ ] JAX MPPI controller is in `src/controllers/mpc/`
- [ ] JAX Smooth MPPI controller is in `src/controllers/mpc/`
- [ ] Experiment code is in `experiments/exp_002_mppi_smooth_mppi/`
- [ ] Runnable entrypoints are in `examples/`
- [ ] Runtime comparison is included
- [ ] Written comparison answers are included
```

## Technical Requirements

Implement MPPI and Smooth MPPI for the cartpole swing-up task.

Your implementation should include:

- A reusable MPPI controller.
- A reusable Smooth MPPI controller.
- PyTorch versions of both controllers.
- JAX versions of both controllers.
- Batched rollout of sampled action sequences.
- A cost function that encourages swing-up, upright stabilization, bounded cart position, and reasonable control effort.
- Control limits matching the cartpole environment.
- Receding-horizon execution: sample trajectories, update the nominal action sequence, apply the first action, shift the sequence, and repeat.
- A plot, GIF animation, or clear numerical summary showing whether swing-up succeeded.
- A runtime benchmark that reports average control-update time.

Adding Python packages to the `uv` environment, for example:

```bash
uv add torch jax jaxlib matplotlib pyyaml
```

If GPU setup is difficult, CPU results are acceptable. Clearly report what hardware you used.

## Suggested MPPI Design

You are expected to tune these values. The numbers below are starting points, not final answers.

- Time step: match the Task 1 cartpole environment.
- Horizon length: start with `40` to `100` steps.
- Number of samples: start with `512` to `4096`; use more if runtime allows.
- Temperature `lambda`: start around `1.0` to `20.0`.
- Control noise standard deviation: tune based on force limits.
- Force limit: match the cartpole environment.
- Use angle wrapping in the cost.
- Penalize cart position, cart velocity, pole angle error, pole angular velocity, and control effort.
- Warm-start by shifting the optimized action sequence forward by one step.

For MPPI, sample noise around the current nominal action sequence and compute trajectory costs for all sampled sequences.

For Smooth MPPI, encourage temporally smooth controls. Reasonable choices include:

- Sample action differences instead of raw actions.
- Add a penalty on action rate, such as `(u[t] - u[t-1]) ** 2`.
- Follow the Smooth MPPI reference implementation's idea of producing smoother action sequences without relying only on a post-processing filter.

## Suggested JAX Design

The JAX implementation should be more than a line-by-line PyTorch translation. Use JAX features where they naturally fit:

- Use `jax.vmap` across sampled trajectories.
- Use `jax.lax.scan` across the time dimension for rollout.
- Use `jax.jit` around the controller update and rollout functions.
- Use `jax.random.PRNGKey` and key splitting explicitly.
- Keep arrays immutable and return updated controller state instead of modifying global state in-place.

Make sure your benchmark separates compile time from runtime. The first JAX call includes compilation and should not be reported as the steady-state control-update time.

## Benchmarking Requirements

Compare these four controllers:

- PyTorch MPPI.
- PyTorch Smooth MPPI.
- JAX MPPI.
- JAX Smooth MPPI.

Use the same cartpole initial condition, horizon, sample count, cost function, and force limits when possible. If you must tune each controller differently, explain why.

Report at least:

- Average control-update time.
- Standard deviation or min/max control-update time.
- Total episode time.
- Whether swing-up succeeded.
- Maximum cart position magnitude.
- Maximum force magnitude.
- A short note on hardware, such as CPU model or GPU model.

## Written Question 1

### What are the pros and cons of MPPI compared with Smooth MPPI?

Attach the answer to the PR.

Discuss action smoothness, exploration, sensitivity to noise covariance, runtime, and whether each method produces control commands that look realistic for a physical system.

## Written Question 2

### What are the pros and cons of sampling-based MPC compared with nonlinear MPC using nonlinear solvers?

Attach the answer to the PR.

Discuss local minima, gradient requirements, constraints, runtime predictability, ease of parallelization, tuning difficulty, and how each approach might scale toward vehicle/trailer dynamics.

## Written Question 3

### What are the runtime differences between the PyTorch and JAX implementations?

Attach the answer to the PR.

Discuss CPU/GPU behavior, JAX compile time, steady-state runtime, vectorization, and whether `vmap`, `lax.scan`, and `jit` helped.

## What To Submit In The PR

Your PR should include:

- Source code for the reusable PyTorch controllers.
- Source code for the reusable JAX controllers.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- A short experiment README explaining how to run your code.
- A plot, GIF animation, or clear numerical summary of swing-up behavior.
- Runtime benchmark results.
- Answers to the written comparison questions.

The most important thing is to build a clean comparison. Good engineering matters here: use the same task setup, report the benchmark honestly, and explain tradeoffs clearly.
