# Experiment 007: Vehicle Residual Dynamics Learning

This folder is reserved for Task 007:

```text
docs/tasks/007_vehicle_residual_dynamics_mppi.md
```

Data gathering from the simulator is performed in `trailer_collect.py`, which feeds a outputs `.npz` file with trajectory data. Because rollout dynamics are the same as the simulator, we collect data directly off the rollouts.

> All custom dataset processing/loading can be found in `src/learning/datasets/trailer_data.py`. We implement custom loading, blocked + strided train/test split, etc.

Training is done in `train.py`. Running is done in `run_model_dynamics.py`. The residual file is left over from earlier when the kinematic model was being considered.
