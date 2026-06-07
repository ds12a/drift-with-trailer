from uncertain_racecar_gym.jax_env import build_nominal_jax_params
from src.simulation.trailer_bicycle_env import VehicleState
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.dynamics.vehicle.trailer_bicycle_fiala import gen_util_funs
import time
import cv2
from gymnasium.wrappers import RecordVideo
from src.simulation.trailer_bicycle_env import TrailerBicycleEnv
from dataclasses import astuple 
import jax.numpy as jnp

from jax import config

config.update("jax_debug_nans", True)


"""
TODO:
- get mu and gamma from track (both env dynamics and mppi dynamics)
- tune
- benchmark
"""



def run_mpc(scenario, reverse=False):
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
    env = TrailerBicycleEnv(
        renderer="pybullet",
        render_mode="rgb_array_birds_eye",
        render_width=300,
        render_height=200,
    )

    env.reset()

    params = build_nominal_jax_params(
        scenario=f"package://scenarios/{scenario}",
    )

    dynamics, cost, bound, _ = gen_util_funs(env.scenario, reverse=reverse, v_target=None)

    mpc = MPPI_Jax(
        8,
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([0.125, 1])),
        inverse_temp=10,
        K=320,
        gamma=0.1,
        step=0.05,
        T=55,
    )

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    i = 0
    try:
        while True:
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(astuple(state)[:-2])

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
                f"mu: {env.track.find_mu(state.x, state.y)}"
            )

            # Benchmarking
            speeds.append(jnp.hypot(state.vx, state.vy))
            yaw_rates.append(state.yaw_truck_rate)

            vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
            steer_angle = state.steer * params[1].vehicle.max_steer_rad
            alpha_f = steer_angle - jnp.arctan2(
                state.vy + params[1].vehicle.lf * state.yaw_truck_rate, vx_safe
            )
            alpha_r = -jnp.arctan2(state.vy - params[1].vehicle.lr * state.yaw_truck_rate, vx_safe)

            slip_angles_f.append(alpha_f)
            slip_angles_r.append(alpha_r)

            action = jnp.array([u[0], u[1]])

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated:  # or truncated:
                break

            i += 1
            frame = env.render()
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

    run_mpc(scenario, reverse=False)
