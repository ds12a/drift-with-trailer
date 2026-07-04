"""
Unified MPPI + SMPPI driver for trailer dynamics
"""

from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.controllers.mpc.debug.smppi_jax_debug import SMPPI_Jax_Debug

from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
import time
import cv2
import numpy as np
from gymnasium.wrappers import RecordVideo
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


def run_mpc(
    controller,
    ctl_args,
    ctl_kwargs,
    cost_kwargs,
    record=False,
    debug=False,
    quiet=False,
    benchmark=False,
    headless=False,
    env_kwargs=None,
    max_steps=None,
    print_name=None,
    record_file_name=None,
):
    """
    ctl_args for MPPI is only covariance. For SMPPI is covariance and omega.
    """

    warnings.filterwarnings("ignore", module="gymnasium")

    # Benchmarking
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
    
    if env_kwargs is None:
        env_kwargs = {
            "renderer": "pybullet",
            "render_mode": "rgb_array_birds_eye",
            "render_width": 600,
            "render_height": 400,
        }

    env = TrailerBicycleEnv(**env_kwargs)

    fname = "rl-video" if record_file_name is None else record_file_name

    if record:
        env = RecordVideo(env, video_folder="gym_videos", episode_trigger=lambda x: True, disable_logger=True, name_prefix=fname)

    env.reset()

    dynamics, cost, bound, bound_der = gen_util_funs(
        env.unwrapped.scenario, 
        **cost_kwargs
    )

    if controller == "MPPI":
        ctl_args = (10, 2, dynamics, None, cost, bound, *ctl_args)
        
        if debug:
            # print("Using MPPI")
            mpc = MPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = MPPI_Jax(*ctl_args, **ctl_kwargs)
    else:
        # Assume SMPPI
        ctl_args = (10, 2, dynamics, None, cost, bound, bound_der, *ctl_args)

        if debug:
            # print("Using SMPPI")
            mpc = SMPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = SMPPI_Jax(*ctl_args, **ctl_kwargs)
    

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    loop = range(max_steps) if max_steps is not None else itertools.count()

    i = 0
    try:
        for i in loop:
            if terminated:
                break

            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.mu,
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            xhist = None
            if debug:
                u, xhist = mpc.run_mpc(mpc_state)
            else:
                u = mpc.run_mpc(mpc_state)
            
            u.block_until_ready()

            elapsed = time.perf_counter() - start
            
            if benchmark:
                speeds.append(jnp.hypot(state.vx, state.vy))
                yaw_rates.append(state.yaw_truck_rate)

                vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
                steer_angle = state.steer * env.unwrapped.scenario.vehicle.max_steer_rad
                alpha_f = steer_angle - jnp.arctan2(
                    state.vy + env.unwrapped.scenario.vehicle.lf * state.yaw_truck, vx_safe
                )
                alpha_r = -jnp.arctan2(
                    state.vy - env.unwrapped.scenario.vehicle.lr * state.yaw_truck, vx_safe
                )

                slip_angles_f.append(alpha_f)
                slip_angles_r.append(alpha_r)

            if not quiet:
                print(
                    f"Step: {i:<5d} | "
                    f"Time: {elapsed:<7.3f} | "
                    f"u: {u[0]:<7.3f} {u[1]:<7.3f} | "
                    # f"Prog: {state.progress:<6.3f} | "
                    f"vx: {state.vx:<7.3f} | "
                    f"vy: {state.vy:<7.3f} | "
                    f"|v|: {jnp.hypot(state.vx, state.vy):<7.3f} | "
                    f"mu: {env.unwrapped.track.find_mu(state.x, state.y):<7.3f} | "
                )
            i += 1

            action = jnp.array([u[0], u[1]])

            n_viz = 50    
            env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz) if debug else None

            observation, reward, terminated, truncated, info = env.step(action)

            if not headless:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass

    env.close()

    if benchmark:
        cutoff = 100

        if print_name is not None:
            print(print_name)

        print(
            f"Iters: {i}, "
            f"Reverse: {cost_kwargs['reverse']}, "
            f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
            f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
            f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
            f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
    )

