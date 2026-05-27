from uncertain_racecar_gym.jax_env import build_nominal_jax_params
from uncertain_racecar_gym.env import VehicleState
import gymnasium as gym
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from experiments.exp_003_racecar_mppi.dynamics import gen_util_funs

import time
import cv2
from gymnasium.wrappers import RecordVideo

import optuna
import jax.numpy as jnp


def run_mpc(scenario, trial, reverse=False, max_steps=None, render=True):
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []

    make = gym.make(
        "UncertainRacecar-v0",
        scenario=f"package://scenarios/{scenario}",
        uncertainty=None,
        renderer="pybullet",
        render_mode="rgb_array_birds_eye",
        render_width=300,
        render_height=200,
    )
    env = (
        RecordVideo(make, video_folder="gym_videos", episode_trigger=lambda x: True)
        if render
        else make
    )
    env.reset()

    params = build_nominal_jax_params(
        scenario=f"package://scenarios/{scenario}",
    )
    dynamics, cost, bound = gen_util_funs(params[0], reverse=reverse)

    mpc = SMPPI_Jax(
        6,
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(
            jnp.array(
                [
                    trial.suggest_float("cov_steer", 0.05, 1.5),
                    trial.suggest_float("cov_accel", 0.05, 1.5),
                ]
            )
        ),
        jnp.diag(
            jnp.array(
                [0.01, 0.01]
                # [trial.suggest_float("omega_steer", 0.0, 1.0), trial.suggest_float("omega_accel", 0, 1)]
            )
        ),
        inverse_temp=trial.suggest_float("inverse_temp", 0.01, 8),
        K=350,
        gamma=0.1, # trial.suggest_float("gamma", 0.01, 0.3),
        step=0.05,
        T=45,
    )

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    total_cost = 0

    try:
        for i in range(max_steps + 1):
            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array([state.x, state.y, state.yaw, state.vx, state.vy, state.yaw_rate])

            u = mpc.run_mpc(mpc_state)
            u.block_until_ready()

            elapsed = time.perf_counter() - start

            step_cost = cost(mpc_state, u, 0)

            print(
                f"Step: {i:<5d} | "
                f"Time: {elapsed:<7.3f} | "
                f"u: {u[0]:<7.3f} {u[1]:<7.3f} | "
                f"Prog: {state.progress:<6.3f} | "
                f"vx: {state.vx:<7.3f} | "
                f"vy: {state.vy:<7.3f} | "
                f"|v|: {jnp.hypot(state.vx, state.vy):<7.3f}"
                f"cost: {step_cost:<7.3f}"
            )

            action = jnp.array([u[0], jnp.maximum(u[1], 0), -jnp.minimum(u[1], 0)])
            observation, reward, terminated, truncated, info = env.step(action)

            total_cost += step_cost

            if i % 500 == 0:
                trial.report(total_cost, step=i)
                
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

            if terminated:
                break

            if render:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    env.close()

    return total_cost


def objective(trial):
    return run_mpc(
        "ks_barcelona_layout_gp_dallara_f317_rl_long.yaml",
        reverse=False,
        max_steps=5000,
        trial=trial,
        render=False,
    )


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="smppi",
        storage="sqlite:///optuna_smppi.db",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=500,    
            max_resource=5000,
            reduction_factor=2,
        )
    )
    study.optimize(objective, n_trials=600)

    print("\nBest value:", study.best_value)

    for k, v in study.best_params.items():
        print(f"{k}: {v}")
