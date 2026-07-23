from pathlib import Path
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.controllers.mpc.debug.smppi_jax_debug import SMPPI_Jax_Debug

from src.learning.models.trailer_spec import spec

from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
import time
import cv2
import numpy as np
from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
from dataclasses import astuple
import warnings
import itertools

import jax.numpy as jnp

from src.learning.datasets.trailer_data import DataCollector, DataStore
from experiments.exp_007_vehicle_residual_dynamics.util_fns import gen_util_funs
import pickle
import jax

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


# state output : [sin(hitch), cos(hitch), vx, vy, truck_yaw_rate, yaw_trailer_rate, mu, delta, brake/accel]
# dynamics output: [...]
# x, y, phi_1, phi_2, v_1x, v_1y, phi_1_dot, phi_2_dot, mu, arc_len = state input
@jax.jit
def convert(xhist, uhist, dt):

    # Trimming as otherwise there is too much data
    xhist = xhist[:3, ...]
    uhist = uhist[:3, ...]

    B = xhist.shape[0]
    T = xhist.shape[1]

    hitch = xhist[..., 2] - xhist[..., 3]
    sh = jnp.sin(hitch)
    ch = jnp.cos(hitch)

    in_state_clean = jnp.concatenate([sh[..., None], ch[..., None], xhist[..., 4:9]], axis=-1)
    states = jnp.concatenate([in_state_clean[:, :-1, :], uhist[:, :-1, :]], axis=-1)
    # dynamics = (in_state_clean[:, 1:, :] - in_state_clean[:, :-1, :])[..., :-1] / dt

    return states


# Controller generalized mpc to support other predefined actions
def run_controller(
    controller,
    data,
    env: TrailerBicycleEnv,
    env_i,
    ctl_i,  # for metadata
    max_steps=None,
    headless=False,
):

    warnings.filterwarnings("ignore", module="gymnasium")

    env.reset()

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    loop = range(max_steps) if max_steps is not None else itertools.count()

    i = 0
    # nn_state = None
    t = 0

    try:
        for i in loop:
            if terminated:
                t += 1
                env.reset()

            state: VehicleState = env.unwrapped._state

            # print(*astuple(state)[:-2], env.unwrapped.track.mu, env.unwrapped.track._arc_samples[env.unwrapped._last_index])
            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.mu,
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            u, xhist, vhist = controller.run_mpc(mpc_state)

            for state_i in convert(xhist, vhist, env.scenario.simulation.dt):
                data.add(np.array(state_i), env_i, ctl_i, i)

            action = np.array([u[0], u[1]])

            observation, reward, terminated, truncated, info = env.step(action)

            n_viz = 50
            env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz)

            print(
                f"\rIter: {i}/{max_steps}, terminated #: {t}. Env {env_i}, Controller {ctl_i}",
                end="",
            )

            if not headless and i % 2 == 0:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    env.close()

    # return state, dynamics


def build_controller(config):
    scenario = TrailerBicycleEnvConfig(
        "scenario", TrackConfig(), VehicleConfig(), SimulationConfig()
    )

    ctl_args, ctl_kwargs, cost_kwargs = config
    dynamics, cost, bound, _ = gen_util_funs(scenario, **cost_kwargs)
    ctl_args = (6, 2, dynamics, None, cost, bound, *ctl_args)

    return MPPI_Jax_Debug(*ctl_args, **ctl_kwargs)


mppi_cfg_fwd = [
    [
        jnp.diag(jnp.array([3e-3, 0.2])),
    ],
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
        "s_weight": 2e1,
        "c_weight": 2e2,
        "a_weight": 1e2,
    },
]

mppi_cfg_rev = [
    [
        jnp.diag(jnp.array([3e-3, 0.2])),
    ],
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
        "s_weight": 1e1,
        "c_weight": 1e0,
        "a_weight": 1e2,
    },
]

# State is:
# [sin(hitch), cos(hitch), vx, vy, truck_yaw_rate, yaw_trailer_rate, mu, delta, brake/accel]
#
# Dynamics are dState/dt
# [hitch_rate, ...]

data = DataCollector(9, 0.05)

# runs = [
#     (build_controller(mppi_cfg_fwd), 0, jnp.sin(jnp.)), # controller, gaussian noise mag, sin freq, sin amp
#     (build_controller(mppi_cfg_fwd), 0.01),
#     (build_controller(mppi_cfg_fwd), 0.1),
# ]

env_kwargs = {
    "renderer": "pybullet",
    "render_mode": "rgb_array_birds_eye",
    "render_width": 150,
    "render_height": 100,
}

envs = [
    TrailerBicycleEnv(**env_kwargs, scenario=TrailerBicycleEnvConfig(".", TrackConfig(mu=0.2, width=20), VehicleConfig(), SimulationConfig())),
    TrailerBicycleEnv(**env_kwargs, scenario=TrailerBicycleEnvConfig(".", TrackConfig(mu=0.4, width=20), VehicleConfig(), SimulationConfig())),
    TrailerBicycleEnv(**env_kwargs, scenario=TrailerBicycleEnvConfig(".", TrackConfig(mu=0.6, width=20), VehicleConfig(), SimulationConfig())),
    TrailerBicycleEnv(**env_kwargs, scenario=TrailerBicycleEnvConfig(".", TrackConfig(mu=0.8, width=20), VehicleConfig(), SimulationConfig())),
    TrailerBicycleEnv(**env_kwargs, scenario=TrailerBicycleEnvConfig(".", TrackConfig(mu=1.0, width=20), VehicleConfig(), SimulationConfig())),
]

controllers = []
for v in [25, 15, 5]:
    mppi_cfg_fwd[2]["v_target"] = v
    mppi_cfg_rev[2]["v_target"] = -v
    controllers.extend([build_controller(mppi_cfg_fwd), build_controller(mppi_cfg_rev)])

# controllers = [build_controller(mppi_cfg_fwd), build_controller(mppi_cfg_rev)]

for e_i, e in enumerate(envs):
    for c_i, c in enumerate(controllers):
        print()
        run_controller(c, data, e, e_i, c_i, 2000, True)

d = data.store(spec.data_version, verbose=True)
d.save(Path("./experiments/exp_007_vehicle_residual_dynamics/data_raw.npz"))
