from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.controllers.mpc.debug.smppi_jax_debug import SMPPI_Jax_Debug

from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
import time
import cv2
import numpy as np
from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
from dataclasses import astuple
import warnings
import itertools

import jax.numpy as jnp

from src.learning.datasets.trailer_data import Data
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
import pickle

from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig, 
    VehicleConfig, 
    TrackConfig, 
    SimulationConfig,
)


def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}


# Controller generalized mpc to support other predefined actions
def run_controller(
    controller,
    max_steps=None,
    headless=False,
):

    warnings.filterwarnings("ignore", module="gymnasium")

    states = [[]]
    dynamics = [[]]

    data = Data()

    if env_kwargs is None:
        env_kwargs = {
            "renderer": "pybullet",
            "render_mode": "rgb_array_birds_eye",
            "render_width": 600,
            "render_height": 400,
        }
    env = TrailerBicycleEnv(**env_kwargs)

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    loop = range(max_steps) if max_steps is not None else itertools.count()

    i = 0
    nn_state = None

    try:
        for i in loop:
            if terminated:
                env.reset()
                states.append([])
                dynamics.append([])
                nn_state = None

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.mu,
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            u = controller.run_mpc(mpc_state)
            i += 1

            action = jnp.array([u[0], u[1]])

            observation, reward, terminated, truncated, info = env.step(action)

            if not headless:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)

            next_nn_state = [
                state.yaw_truck - state.yaw_trailer,  # alpha
                *astuple(state)[4:-2],  # vx, vy, yaw_truck_rate, yaw_trailer_rate
                env.unwrapped.track.mu,
                u[0],
                u[1],
            ]

            if nn_state:
                dynamics[-1].append(
                    [
                        next_state - curr_state
                        for next_state, curr_state in zip(next_nn_state, nn_state)[:-3]
                    ]
                )

            nn_state = next_nn_state

            state[-1].append(nn_state)

    except KeyboardInterrupt:
        pass

    env.close()

    return state, dynamics

def build_controller(config):
    scenario = TrailerBicycleEnvConfig(
        "scenario", TrackConfig(), VehicleConfig(), SimulationConfig()
    )

    dynamics, cost, bound, _ = gen_util_funs(scenario, **cost_kwargs)
    ctl_args, ctl_kwargs, cost_kwargs = config
    ctl_args = (6, 2, dynamics, None, cost, bound, *ctl_args)

    return MPPI_Jax_Debug(*ctl_args, **ctl_kwargs)

mppi_cfg_fwd = (
    (
       jnp.diag(jnp.array([3e-3, 0.2])),
    ),
    {
        "inverse_temp": 1,
        "K": 500,
        "step": 0.05,
        "T": 80,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": 25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "s_weight": 2e2,
        "c_weight": 1e0,
        "a_weight": 7e2,
    },
)

mppi_cfg_rev = (
    (
        jnp.diag(jnp.array([3e-3, 0.2])),
    ),
    {
        "inverse_temp": 0.5,
        "K": 750,
        "step": 0.05,
        "T": 55,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": -25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "s_weight": 1e2,
        "c_weight": 1e-2,
        "a_weight": 1e2,
    },
)

data = Data(-12252023, jnp.zeros(8), jnp.ones(8), jnp.zeros(5), jnp.ones(5))

runs = [
    (build_controller(mppi_cfg_fwd), 0, jnp.sin(jnp.)) # controller, gaussian noise mag, sin freq, sin amp
    (build_controller(mppi_cfg_fwd), 0.01)
    (build_controller(mppi_cfg_fwd), 0.1)
]

for c in controllers:
    state, dynamics = run_controller(c, 5000, False)

    for s, d in zip(state, dynamics):
        data.add(s, d)

pickle.dump(data, "experiments/exp_007_vehicle_residual_dynamics/data.pkl")