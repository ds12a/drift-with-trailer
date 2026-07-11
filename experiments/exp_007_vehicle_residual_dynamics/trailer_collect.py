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

def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None
    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)      # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])             # (n, T, 2), small transfer
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
    try:
        for i in loop:
            if terminated:
                env.reset()
                states.append([])
                dynamics.append([])

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
    
            states[-1].append()

    except KeyboardInterrupt:
        pass

    env.close()

