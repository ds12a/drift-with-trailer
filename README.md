# Drift With Trailer

Research code for building from cartpole MPC baselines toward vehicle/trailer dynamics, friction variation, data collection, learned dynamics, and RMA-style adaptation.

## Repository Layout

- `docs/reading-list/`: papers, notes, and background material.
- `docs/meeting-notes/`: weekly advising and project meeting notes.
- `docs/tasks/`: student-facing assignments and check-in tasks.
- `src/`: reusable project code.
- `configs/`: vehicle, controller, and experiment configuration files.
- `examples/`: runnable examples and command-line entrypoints.
- `experiments/`: experiment-specific code and results organization.
- `tests/`: automated tests.
- `results/`: shared figures and tables.

## Python Setup With uv

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the local environment:

```bash
uv sync
```

Run the cartpole smoke simulation:

```bash
uv run python examples/cartpole_smoke.py
```

The smoke simulation is intentionally simple. It only checks that the Python environment, dependency installation, and a basic cartpole dynamics example are working.

## Student Tasks

- [Task 001: Nonlinear MPC CartPole Swing-Up](docs/tasks/001_nonlinear_mpc_cartpole_swingup.md)
- [Task 002: MPPI and Smooth MPPI CartPole Swing-Up](docs/tasks/002_mppi_smooth_mppi_cartpole.md)
- [Task 003: Racecar Track Following With JAX MPPI and Smooth MPPI](docs/tasks/003_racecar_jax_mppi.md)
- [Task 004: Fiala Racecar, Trailer, and Ice Patches](docs/tasks/004_fiala_trailer_ice_mppi.md)
- [Task 005: Trailer MPPI Under Model Mismatch](docs/tasks/005_trailer_model_mismatch_mppi.md)
- [Task 006: Online Learned CartPole Dynamics for MPPI](docs/tasks/006_online_learned_cartpole_mppi.md)
- [Task 007: Vehicle Residual Dynamics Learning for MPPI](docs/tasks/007_vehicle_residual_dynamics_mppi.md)
