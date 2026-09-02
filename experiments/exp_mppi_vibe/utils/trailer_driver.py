"""
Unified MPPI / SMPPI / RBR / BSMC driver for trailer dynamics.

Copied from exp_004_fiala_trailer_ice/utils and extended with the two
particle controllers, a seed passthrough, and per-control-step diagnostics.
exp_004 is deliberately left untouched.
"""

from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.rbr_jax import RBR_Jax
from src.controllers.mpc.bsmc_jax import BSMC_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.controllers.mpc.debug.smppi_jax_debug import SMPPI_Jax_Debug
from src.controllers.mpc.debug.rbr_jax_debug import RBR_Jax_Debug
from src.controllers.mpc.debug.bsmc_jax_debug import BSMC_Jax_Debug

from experiments.exp_mppi_vibe.utils.particle_diag import ParticleDiag

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

def build_planner_debug(all_samples, n_vis, weights=None, islands=None):
    """
    Subsample candidate rollouts for the overlay.

    Extends the exp_004 version with two optional channels the renderer uses to
    colour the lines: per-trajectory MPPI weight (brightness) and island id
    (hue). Both are subsampled with the same index set so they stay aligned, and
    everything is reduced on device before the single host transfer.
    """
    if all_samples is None:
        return None
    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)      # even spread across samples
    out = {"candidate_xy": np.asarray(all_samples[idx, :, :2])}   # (n, T, 2)

    if weights is not None:
        w = jnp.asarray(weights)[idx]
        out["candidate_w"] = np.asarray(w / jnp.maximum(w.max(), 1e-30))
    if islands is not None:
        out["candidate_island"] = np.asarray(jnp.asarray(islands)[idx])
    return out


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
    seed=0,
    diag_file=None,
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

    dynamics, cost, bound, bound_der, violation = gen_util_funs(
        env.unwrapped.scenario,
        with_violation=True,
        **cost_kwargs
    )

    # RBR and BSMC take a seed so paired-seed sweeps are actually paired; SMPPI
    # does not have the kwarg, so it is only injected where it is supported.
    ctl_kwargs = dict(ctl_kwargs)
    if controller in ("MPPI", "RBR", "BSMC"):
        ctl_kwargs.setdefault("seed", seed)

    if controller == "MPPI":
        ctl_args = (6, 2, dynamics, None, cost, bound, *ctl_args)

        if debug:
            mpc = MPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = MPPI_Jax(*ctl_args, **ctl_kwargs)
    elif controller == "RBR":
        # violation_func goes before cv, mirroring where SMPPI puts bound_der
        ctl_args = (6, 2, dynamics, None, cost, bound, violation, *ctl_args)

        if debug:
            mpc = RBR_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = RBR_Jax(*ctl_args, **ctl_kwargs)
    elif controller == "BSMC":
        ctl_args = (6, 2, dynamics, None, cost, bound, *ctl_args)

        if debug:
            mpc = BSMC_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = BSMC_Jax(*ctl_args, **ctl_kwargs)
    else:
        # Assume SMPPI
        ctl_args = (6, 2, dynamics, None, cost, bound, bound_der, *ctl_args)

        if debug:
            mpc = SMPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = SMPPI_Jax(*ctl_args, **ctl_kwargs)

    diag_log = ParticleDiag(controller, diag_file) if diag_file is not None else None
    

    island_id = None
    if getattr(mpc, "islands", None) is not None:
        island_id = np.concatenate(
            [np.full(n, m, dtype=np.int32) for m, n in enumerate(mpc.islands)]
        )

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
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            xhist, diag = None, None
            if debug:
                out = mpc.run_mpc(mpc_state)
                u, xhist = out[0], out[1]
                if len(out) > 3:
                    diag = out[3]
            else:
                u = mpc.run_mpc(mpc_state)
            
            u.block_until_ready()

            elapsed = time.perf_counter() - start
            
            if benchmark:
                speeds.append(jnp.hypot(state.vx, state.vy))
                yaw_rates.append(state.yaw_truck)

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
            env.unwrapped.planner_debug = (
                build_planner_debug(xhist, n_viz, islands=island_id) if debug else None
            )

            if diag_log is not None and diag is not None:
                diag_log.add(
                    i,
                    float(env.unwrapped.track._arc_samples[env.unwrapped._last_index]),
                    float(state.vx),
                    diag,
                )

            observation, reward, terminated, truncated, info = env.step(action)

            if not headless:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass

    env.close()

    if diag_log is not None:
        diag_log.save()

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