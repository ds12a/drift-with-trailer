# Task 006: Online Learned CartPole Dynamics for MPPI

## Purpose

Task 6 returns to cartpole and introduces a first machine-learning loop for model-based control.

The goal is to learn a one-step neural network dynamics model from data and use that learned model inside MPPI. The true cartpole simulator is still used for the actual environment step. The neural network is only used by MPPI to roll out sampled trajectories.

Start with a randomly initialized neural network dynamics model. The model should take the current state and input, then predict the next state:

```text
(x_t, u_t) -> x_{t+1}
```

or equivalently predict the state change:

```text
(x_t, u_t) -> x_{t+1} - x_t
```

Run MPPI using the current learned model. As the controller interacts with the true cartpole environment, collect transition data. Every `N` environment steps, train or fine-tune the neural network on the accumulated dataset. Then continue control using the updated model. The dataset should keep growing, and the learned dynamics should keep improving.

This is intentionally a quick introduction to learned dynamics. The main objective is to complete the cartpole control task with an online learned model. Keep the implementation simple and understandable.

If there is no inverted pendulum environment already in this repository, skip inverted pendulum and use cartpole only.

## Learning Goals

By the end of this task, you should be able to:

- Create a simple neural network dynamics model.
- Collect transition data from a control task.
- Train a one-step dynamics model with supervised regression.
- Use a learned model inside MPPI rollouts.
- Repeat data collection and training online.
- Compare control behavior before and after the model improves.
- Explain how model error affects sampling-based MPC.

## Reading and References

Use these references:

- Task 1 nonlinear MPC cartpole specification: `docs/tasks/001_nonlinear_mpc_cartpole_swingup.md`
- Task 2 MPPI and Smooth MPPI specification: `docs/tasks/002_mppi_smooth_mppi_cartpole.md`
- MPPI and Smooth MPPI papers in `docs/reading-list/papers.md`

Use JAX to collect data, train a neural dynamics model, and use it for MPPI control within this repository.

For the controller, reuse the cartpole setup and MPPI hyperparameters from Task 1 and Task 2 as much as possible. This task is about learned dynamics, so avoid spending most of the effort on controller retuning.

## Repository Locations

Put reusable learned-dynamics code here:

```text
src/learning/models/
```

Suggested files:

```text
src/learning/models/cartpole_nn_dynamics.py
src/learning/models/cartpole_dataset.py
```

Put reusable MPPI helpers or learned-model rollout adapters here:

```text
src/controllers/mpc/
```

Suggested files:

```text
src/controllers/mpc/cartpole_learned_dynamics_mppi.py
```

Put experiment-specific code here:

```text
experiments/exp_006_learned_cartpole_dynamics/
```

Put runnable entrypoints here:

```text
examples/
```

Suggested runnable files:

```text
examples/run_cartpole_learned_mppi.py
examples/run_cartpole_learned_mppi_training_loop.py
```

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 006: Online Learned CartPole Dynamics for MPPI
```

2. In the issue, briefly describe your plan:

- Which cartpole environment you will use.
- What neural network model you will implement.
- Whether the model predicts next state or state difference.
- How often you will train the model.
- How you will use the learned model inside MPPI.
- Which Task 1 or Task 2 controller parameters you plan to reuse.
- What result will count as success.

3. Create a branch from `main`. Use a branch name like:

```text
task-006-learned-cartpole-mppi
```

4. Make small commits with clear messages. Example:

```text
Add cartpole transition dataset
Add JAX neural network dynamics model
Add learned-dynamics MPPI rollout
Add online training loop for cartpole
Document task 006 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Added a JAX neural network dynamics model for cartpole.
- Added a dataset buffer for cartpole transition data.
- Added an online collect/train/control loop.
- Used the learned dynamics model inside MPPI rollouts.

## Model

- Input:
- Output:
- Predicts next state or state difference:
- Network architecture:
- Loss function:
- Optimizer:
- Data normalization:

## Online Training Loop

- Initial data collection strategy:
- Training interval `N`:
- Training steps per update:
- Dataset size:
- Train/validation split:
- How the MPPI rollout uses the learned model:

## Controller Parameters

- Task 1 or Task 2 controller configuration reused:
- Horizon:
- Number of samples:
- Temperature:
- Noise covariance:
- Cost weights:
- Control limits:
- Any parameter changes, if needed:

## Results

- Initial condition:
- Number of episodes:
- Whether swing-up/control succeeded:
- Final cart position:
- Final pole angle:
- Model loss before training:
- Model loss after training:
- Control performance before learning:
- Control performance after learning:

## Media

- Plot or GIF link:

Do not commit large GIF files directly to git. Upload media to a GitHub issue or PR comment, then paste the uploaded media link here.

## Answers

### What happens when MPPI uses a randomly initialized dynamics model?

### Why does accumulating the dataset help?

### How does learned-model error affect MPPI control?

### What would you improve if this learned model had to control a real robot?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] Neural network dynamics model is implemented in JAX
- [ ] Dataset collects `(state, action, next_state)` transitions
- [ ] Model is trained by supervised regression
- [ ] MPPI uses the learned model for rollouts
- [ ] MPPI controller parameters mostly reuse Task 1 or Task 2 settings
- [ ] Actual environment step uses the true cartpole simulator
- [ ] Dataset keeps accumulating over time
- [ ] Model is retrained or fine-tuned every `N` steps
- [ ] Results before and after learning are reported
- [ ] Large media files are not committed to git
- [ ] Written question answers are included
```

## Technical Requirements

Your implementation should include:

- A true cartpole environment used for actual simulation.
- A randomly initialized neural network dynamics model.
- A dataset buffer storing `(state, action, next_state)` transitions.
- A supervised training loop for one-step dynamics prediction.
- A JAX MPPI rollout that uses the learned dynamics model.
- An online loop that alternates between control, data collection, and training.
- A short experiment README explaining how to run the task.
- Results showing whether the learned model can complete the cartpole control task.
- Answers to the written questions.

Recommended libraries:

```bash
uv add flax optax
```

You may also implement the neural network in pure JAX if you prefer. If you use Flax and Optax, keep the model small and the training loop easy to read.

## Controller Parameter Reuse

Start from the best cartpole controller settings from Task 1 or Task 2.

Good values to reuse include:

- Time step.
- Horizon length.
- Number of MPPI samples.
- Temperature.
- Control noise covariance.
- Force limits.
- Cart position and angle cost weights.
- Terminal cost, if used.

Only make small changes if the learned dynamics rollout makes the original settings unusable. If you change a controller parameter, report what changed and why. The main variable in this task should be the learned dynamics model, not a fully retuned controller.

## Suggested Model Design

Start with a small multilayer perceptron.

Input:

```text
[cart_position, cart_velocity, pole_angle, pole_angular_velocity, force]
```

Output option 1:

```text
[next_cart_position, next_cart_velocity, next_pole_angle, next_pole_angular_velocity]
```

Output option 2:

```text
[delta_cart_position, delta_cart_velocity, delta_pole_angle, delta_pole_angular_velocity]
```

Predicting the state difference is often easier, because the model only has to learn the local change over one time step.

Suggested architecture:

- 2 hidden layers.
- 64 or 128 hidden units per layer.
- `tanh`, `relu`, or `silu` activation.
- Mean-squared error loss.
- Adam optimizer.
- Input and output normalization if needed.

Keep the time step the same as the cartpole environment.

## Suggested Online Loop

Use a simple loop like this:

```text
initialize neural network dynamics model randomly
initialize empty dataset
reset true cartpole environment

for each environment step:
    run MPPI using current learned dynamics
    apply first MPPI action to true cartpole environment
    store (state, action, next_state) in dataset

    if step % N == 0 and dataset is large enough:
        train or fine-tune neural network on accumulated dataset
        continue MPPI with updated learned dynamics
```

Because the initial random model may be very poor, it is acceptable to use a short warm-start phase:

- Random actions.
- Small stabilizing actions.
- A simple existing cartpole controller.
- MPPI with the true dynamics for only a short data-collection warm start.

If you use a warm start, document it clearly. The main learned-control experiment should use the neural network model inside MPPI.

## MPPI Rollout Requirements

Inside MPPI, the learned model should be used for sampled trajectory rollouts.

Use JAX patterns where they help:

- Use `jax.vmap` across sampled trajectories.
- Use `jax.lax.scan` across the time dimension.
- Use `jax.jit` for rollout and training steps when appropriate.
- Keep model parameters explicit in the rollout function.

Do not use the true cartpole equations inside the MPPI rollout for the main learned-model result. The true simulator is only for the actual environment transition and for collecting labels.

## Result Requirements

Report at least:

- Dataset size over time.
- Training loss over time.
- Validation loss if you use a validation split.
- Whether cartpole swing-up/control succeeded.
- Control performance before learning.
- Control performance after learning.
- Average MPPI update time.
- Main failure modes.

Useful plots:

- Training loss vs update number.
- Pole angle over time.
- Cart position over time.
- Control force over time.
- Model prediction error over time.

## Written Question 1

### What happens when MPPI uses a randomly initialized dynamics model?

Attach the answer to the PR.

Discuss sampled rollouts, action selection, model error, and why the controller may initially behave poorly.

## Written Question 2

### Why does accumulating the dataset help?

Attach the answer to the PR.

Discuss coverage of visited states, supervised learning, reducing one-step prediction error, and why online data helps the model learn the parts of state space the controller actually reaches.

## Written Question 3

### How does learned-model error affect MPPI control?

Attach the answer to the PR.

Discuss rollout compounding error, wrong cost estimates, optimistic trajectories, and why receding-horizon feedback can partially compensate.

## Written Question 4

### What would you improve if this learned model had to control a real robot?

Attach the answer to the PR.

Discuss safety constraints, uncertainty estimation, robust MPPI, data quality, action limits, online validation, and fallback controllers.

## What To Submit In The PR

Your PR should include:

- Source code for the JAX neural network dynamics model.
- Source code for the transition dataset buffer.
- Source code for the online training loop.
- MPPI code or adapters that use learned dynamics for rollouts.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- A short experiment README explaining how to run everything.
- Plots, tables, or linked media showing the result.
- Answers to the written questions.

This task is a first pass at learned dynamics, so keep it simple. The important result is a clear data collection, training, and control loop that students can understand and extend later for vehicle and trailer residual learning.
