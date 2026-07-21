import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp

from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.trailer_spec import KIN_FS, kin
# from src.learning.models.trailer_spec_nores import RAW_FS, kin_zeros
from src.learning.datasets.trailer_data import DataLoader
# from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs, TrackProjection
from src.dynamics.trailer.residual_dynamics import gen_util_funs
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from src.utils.track import TrackModel

spec = KIN_FS
kin_fn = kin

HISTORY = spec.H

# below are currently in dyn function, make sure no desync
D_STATE_DIM = 7
D_U_DIM = 2
D_EXTRA_DIM = 1
K_STATE_DIM = 7
M_STATE_DIM = 6

scenario = TrailerBicycleEnvConfig(
    ".", TrackConfig(mu=0.6, width=20), VehicleConfig(), SimulationConfig()
)


loader = DataLoader.load(Path("./experiments/exp_007_vehicle_residual_dynamics/data_proc1.npz"), spec)
x_mean, x_std = jnp.asarray(loader.x_mean), jnp.asarray(loader.x_std)
y_mean, y_std = jnp.asarray(loader.y_mean), jnp.asarray(loader.y_std)

model = TrailerModel(48, 4)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(model, ckpt.restore(Path.cwd() / "src/learning/models/trained/trailer-nokin-best", state))

def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}

dynamics, cost, bound, _ = gen_util_funs(
    scenario,
    spec,
    kin_fn,
    model,
    loader,
    reverse=False,
    v_target=25,
    p_weight=1e2,
    p_slow_weight=1e0,
    # s_weight=2e1,
    c_weight=2e2,
    a_weight=1e2,
)

env = TrailerBicycleEnv(
    renderer="pybullet",
    render_mode="rgb_array_birds_eye",
    render_width=450,
    render_height=300,
    scenario=scenario,
)

mpc = MPPI_Jax_Debug(
    (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM),
    2,
    dynamics,
    None,
    cost,
    bound,
    jnp.diag(jnp.array([3e-3, 0.2])),
    inverse_temp=1,
    K=500,
    step=0.05,
    T=50,
    alpha=0.05,
    history=HISTORY
)

env.reset()
observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

history = jnp.zeros(HISTORY * (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM))

i = 0
try:
    # Necessary, the model panics when seeing 0/default windoww
    for _ in range(HISTORY):
        state = env.unwrapped._state
        arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        curr = jnp.concatenate([jnp.array([*astuple(state)[:6], env.unwrapped.track.mu]),
                                jnp.zeros(2), jnp.array([arclen])])
        history = jnp.concatenate([history[10:], curr])
        env.step(np.zeros(2))
    while True:
        state: VehicleState = env.unwrapped._state

        arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        main_slice = jnp.array(
            [
                *astuple(state)[:6],
                env.unwrapped.track.mu,
                # env.unwrapped.track._arc_samples[env.unwrapped._last_index],
            ]
        ) # env state is x, y, phi1, phi2, vx, vy, phi1dot, phi2dot, steer, accel
        curr = jnp.concatenate([main_slice, jnp.zeros(2), jnp.array([arclen])]) # Control is filled out inside

        history = jnp.concatenate([history[(D_STATE_DIM + D_U_DIM + D_EXTRA_DIM):], curr])

        start = time.perf_counter()
        u, xhist, vhist = mpc.run_mpc(history)
        u.block_until_ready()
        # print(u)
        action = np.array(u)

        print(i, time.perf_counter() - start, action)

        observation, reward, terminated, truncated, info = env.step(action)
        i += 1

        n_viz = 50
        env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz)

        if i % 2 == 0:
            frame = env.render()
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)

        if terminated:
            env.reset()

        
finally:
    env.close()
