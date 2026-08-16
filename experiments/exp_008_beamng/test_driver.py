import logging
import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import absl.logging
import orbax.checkpoint as ocp  # this breaks beamng logging

from src.simulation.beamng_trailer_env import BeamNGTrailerEnv, VehicleState, bng_pickup_trailer_cfg
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from gymnasium.wrappers import RecordVideo
import json

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False

# Reverse/fwd configs should be automated
V_TARGET = -40 / 3.6

config = BeamNGTrailerEnvConfig(
    ".", TrackConfig(mu=0.5, width=30), bng_pickup_trailer_cfg, SimulationConfig()
)

# config.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}

env = BeamNGTrailerEnv(
    config=config,
)


bng_log = logging.getLogger("beamngpy")
bng_log.propagate = False                      # stop feeding orbax's root handler
h = logging.StreamHandler()
h.setLevel(logging.WARNING)
bng_log.addHandler(h)

fwd_weights = {
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "c_weight": 1e0,
    "a_weight": 7e2,
    "v_target": V_TARGET,
    "reverse": False,
}
rev_weights = {
    "p_weight": 2e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 2e2,
    "v_target": V_TARGET,
    "reverse": False,
}


if V_TARGET > 0:
    dynamics, cost, bound, _ = gen_util_funs(
        config,
        s_weight=0,
        **fwd_weights,
    )
    mpc = MPPI_Jax_Debug(
        6,
        2,
        dynamics,
        None,
        cost,
        bound,
        # bound_der,
        jnp.diag(jnp.array([3e-3, 0.2])),
        # jnp.diag(jnp.array([1e-2, 1e-1])),
        inverse_temp=0.5,
        K=500,
        step=0.05,
        T=80,
        alpha=0.05,
    )

else:
    dynamics, cost, bound, _ = gen_util_funs(
        config,
        s_weight=0,
        **rev_weights,
    )
    mpc = MPPI_Jax_Debug(
        6,
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([1e-2, 0.2])),
        # inverse_temp=5e2,
        inverse_temp=0.5,
        K=2000,
        step=0.05,
        T=55,
        alpha=0.01,
    )

# fname = "rl-video" # if record_file_name is None else record_file_name
# env = RecordVideo(env, video_folder="gym_videos", episode_trigger=lambda x: True, disable_logger=True, name_prefix=fname)

env.reset()
observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

# history = jnp.zeros(HISTORY * (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM))
speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
i = 0
try:

    for i in range(2000):
        if terminated:
            break

        start = time.perf_counter()

        state: VehicleState = env.unwrapped._state

        mpc_state = jnp.array(
            [
                *astuple(state)[:-2],
                env.unwrapped.track.find_mu(state.x, state.y),
                env.unwrapped.track._arc_samples[env.unwrapped._last_index],
            ]
        )

        xhist = None
        u, xhist, *_ = mpc.run_mpc(mpc_state)
        
        u.block_until_ready()

        elapsed = time.perf_counter() - start
        
        speeds.append(jnp.hypot(state.vx, state.vy))
        yaw_rates.append(state.yaw_truck_rate)

        vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
        # steer_angle = state.steer * env.unwrapped.config.vehicle.max_steer_rad
        # alpha_f = steer_angle - jnp.arctan2(
        #     state.vy + env.unwrapped.config.vehicle.lf * state.yaw_truck_rate, vx_safe
        # )
        # alpha_r = -jnp.arctan2(
        #     state.vy - env.unwrapped.config.vehicle.lr * state.yaw_truck_rate, vx_safe
        # )

        # slip_angles_f.append(alpha_f)
        # slip_angles_r.append(alpha_r)

        # print(
        #     f"Step: {i:<5d} | "
        #     f"State: {observation}"
        # )
        i += 1

        action = jnp.array([-u[0], u[1]])

        n_viz = 10    
        env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz)

        observation, reward, terminated, truncated, info = env.step(action)
    cutoff = 100
    print(
        f"Iters: {i}, "
        f"Reverse: {V_TARGET > 0}, "
        f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
        f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
        f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
        f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
    )
finally:
    env.close()
