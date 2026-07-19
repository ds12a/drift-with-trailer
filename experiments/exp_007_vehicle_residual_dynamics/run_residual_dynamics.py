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
from src.learning.models.trailer_spec import KIN_FS, kin
from src.learning.models.trailer_spec_nores import RAW_FS, kin_zeros
from src.learning.datasets.trailer_data import DataLoader
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)

spec = KIN_FS
kin_fn = kin_zeros

HISTORY = spec.H

scenario = TrailerBicycleEnvConfig(
    ".", TrackConfig(mu=0.6, width=20), VehicleConfig(), SimulationConfig()
)

_, cost, bound, _ = gen_util_funs(
    scenario,
    reverse=False,
    v_target=25,
    p_weight=1e2,
    p_slow_weight=1e0,
    s_weight=2e1,
    c_weight=2e2,
    a_weight=1e2,
)

loader = DataLoader.load(Path("./experiments/exp_007_vehicle_residual_dynamics/data_proc1.npz"), spec)
x_mean, x_std = jnp.asarray(loader.x_mean), jnp.asarray(loader.x_std)
y_mean, y_std = jnp.asarray(loader.y_mean), jnp.asarray(loader.y_std)

model = TrailerModel(24, 4)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(model, ckpt.restore(Path.cwd() / "src/learning/models/trained/trailer", state))

# Dynamics state: [x, y, phi1, phi2, vx, vy, mu, delta, a] + ctrl [delta, a]
# Needs to keep for u history
# In model state: [sin(hitch), cos(hitch), vx, vy, mu, delta, a], mu is only for prior tanh
# pred [ax, ay, phi1dot, phi2dot]

D_STATE_DIM = 7
D_U_DIM = 2
K_STATE_DIM = 7
M_STATE_DIM = 6
H = spec.H
dt = 0.05 # TODO dont hardcode
def dynamics(x, u):  # passed as windows
    x_windows = x.reshape(H, D_STATE_DIM + D_U_DIM)
    x_windows[-1][-1], x_windows[-1][2] = u[0], u[1]  # Control

    def slice_kin(window):
        hitch = window[2] - window[3]
        
        return jnp.stack([
            jnp.sin(hitch),  # sh
            jnp.cos(hitch),  # ch
            window[4],  # vx
            window[5],  # vy
            window[6],  # mu
            window[7],  # delta
            window[8],  # a
        ])
    @jax.vmap
    def slice_mod(window):
        hitch = window[2] - window[3]
        
        return jnp.stack([
            jnp.sin(hitch),  # sh
            jnp.cos(hitch),  # ch
            window[4],  # vx
            window[5],  # vy
            window[7],  # delta
            window[8],  # a
        ])
    
    kin_in = slice_kin(x_windows[-1]).flatten()
    model_in = slice_mod(x_windows).flatten()[None, ...]

    _, _, phi1, phi2, vx, vy, *_ = x_windows[-1]
    pred = kin_fn(kin_in)
    pred += model(model_in)[0] * y_std + y_mean

    ax, ay, phi1dot, phi2dot = pred
    xdot = vx * jnp.cos(phi1) - vy * jnp.sin(phi1)
    ydot = vx * jnp.sin(phi1) + vy * jnp.cos(phi1)


    dx = jnp.array([
        xdot, ydot, phi1dot, phi2dot, ax, ay, 0
    ])
    dx_history = (x_windows[1:] - x_windows[:-1]) / dt
    dx_window = jnp.concatenate([dx_history, dx[None, :]], axis=0)
    return dx_window.flatten()

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
