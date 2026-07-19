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
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.trailer_spec import KIN_FS
from src.learning.datasets.trailer_data import DataLoader
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)

HISTORY = 4

scenario = TrailerBicycleEnvConfig(
    ".", TrackConfig(mu=0.6, width=20), VehicleConfig(), SimulationConfig()
)

kin_dynamics, cost, bound, _ = gen_util_funs(
    scenario,
    reverse=False,
    v_target=25,
    p_weight=1e2,
    p_slow_weight=1e0,
    s_weight=2e1,
    c_weight=2e2,
    a_weight=1e2,
)

loader = DataLoader.load(
    Path("./experiments/exp_007_vehicle_residual_dynamics/data_proc1.npz"), KIN_FS
)
# Why is this not fixed? surely the data should be normalized wrt some fixed mean/std???
x_mean, x_std = jnp.asarray(loader.x_mean), jnp.asarray(loader.x_std)
y_mean, y_std = jnp.asarray(loader.y_mean), jnp.asarray(loader.y_std)

model = TrailerModel(24, 4)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(model, ckpt.restore(Path.cwd() / "src/learning/models/trained/trailer", state))

def dynamics(x, u):

    @jax.vmap
    def compute_feat(window):
        hitch = window[2] - window[3]
        
        return jnp.stack([
            jnp.sin(hitch), 
            jnp.cos(hitch), 
            window[4], 
            window[5], 
            window[6], 
            window[7]
        ])

    x_windows = jax.lax.sliding_window_view(jnp.concatenate([x, u]), window_shape=(9,))[::6] 
    feats = compute_feat(x_windows).flatten()[None]

    feats = (feats - x_mean) / x_std

    res = model(feats)[0] * y_std + y_mean

    dx = kin_dynamics(x[-7:], u)
    dx = dx.at[4:6].add(res[:2])
    dx = dx.at[2:4].add(res[2:])

    return dx

env = TrailerBicycleEnv(
    renderer="pybullet",
    render_mode="rgb_array_birds_eye",
    render_width=150,
    render_height=100,
    scenario=scenario,
)

mpc = MPPI_Jax(
    6,
    2,
    dynamics,
    None,
    cost,
    bound,
    jnp.diag(jnp.array([3e-3, 0.2])),
    inverse_temp=1,
    K=500,
    step=0.05,
    T=80,
    alpha=0.05,
    history=HISTORY
)

env.reset()
observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

history = jnp.zeros(HISTORY * 7)

i = 0
try:
    while True:
        state: VehicleState = env.unwrapped._state
        mpc_state = jnp.array(
            [
                *astuple(state)[:-2],
                env.unwrapped.track.mu,
                env.unwrapped.track._arc_samples[env.unwrapped._last_index],
            ]
        )

        history = jnp.concatenate([history[-9:], mpc_state])

        start = time.perf_counter()
        u = mpc.run_mpc(mpc_state)
        u.block_until_ready()
        action = np.array(u[0])

        print(i, time.perf_counter() - start, action)

        observation, reward, terminated, truncated, info = env.step(action)
        i += 1

        if i % 2 == 0:
            frame = env.render()
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)

        if terminated:
            env.reset()

        
finally:
    env.close()
