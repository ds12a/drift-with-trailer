import gymnasium as gym
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug

from src.dynamics.vehicle.bicycle_fiala import gen_util_funs
import time
import cv2
import numpy as np
from gymnasium.wrappers import RecordVideo
from src.simulation.bicycle_env import BicycleEnv, VehicleState
from src.simulation.config.bicycle_config import BicycleEnvConfig

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
    scenario, 
    reverse=False, 
    record=False,
    debug=False,
):
    
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
    
    env = BicycleEnv(
        renderer="pybullet",
        render_mode="rgb_array_birds_eye",
        render_width=450,
        render_height=300,
    )

    if record:
        env = RecordVideo(env, video_folder="gym_videos", episode_trigger=lambda x: True)

    env.reset()

    dynamics, cost, bound, _ = gen_util_funs(
        env.unwrapped.scenario, 
        reverse=reverse, 
        v_target=30,
        p_weight = 1e2,
        p_slow_weight = 1e0,
        s_weight = 808.4087416389013,
        c_weight = 0.012476149239058718,
    )

    # For hand-tuning

    args = (
        6, 
        2, 
        dynamics, 
        None, 
        cost, 
        bound, 
        jnp.diag(jnp.array([0.013178974044529336, 0.06115174214186951]))
    )
    kwargs = {
        "inverse_temp": 0.01155091515764931,
        "K": 750,
        "step": 0.05,
        "T": 85,
        "alpha": 0.851356016887989,
    }

    if debug:
        mpc = MPPI_Jax_Debug(*args, **kwargs)
    else:
        mpc = MPPI_Jax(*args, **kwargs)
    

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    i = 0
    try:
        while True:
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(
                [
                    state.x,
                    state.y,
                    state.yaw,
                    state.vx,
                    state.vy,
                    state.yaw_rate,
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )

            xhist = None
            if debug:
                u, xhist, *_ = mpc.run_mpc(mpc_state)
            else:
                u = mpc.run_mpc(mpc_state)
            
            u.block_until_ready()

            elapsed = time.perf_counter() - start
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

            observation, reward, terminated, truncated, info = env.step(action)

            n_viz = 50

            planner_debug = build_planner_debug(xhist, n_viz) if debug else None

            frame = env.render(planner_debug=planner_debug)
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass

    env.close()

    cutoff = 100
    print(
        f"Reverse: {reverse}, Avg speed: {jnp.mean(jnp.array(speeds[cutoff:]))}, Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
    )


if __name__ == "__main__":
    scenario = "ks_barcelona_layout_gp_dallara_f317_rl_long.yaml"
    # scenario = "sample_oval.yaml"

    run_mpc(
        scenario, 
        reverse=False,
        record=False,
        debug=True,
    )