import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import logging
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

from src.learning.datasets.trailer_data import DataCollector, DataStore
from src.simulation.beamng_trailer_env import BeamNGTrailerEnv, VehicleState, bng_pickup_trailer_cfg
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from experiments.exp_008_beamng.util_fns import gen_util_funs as sin_fn
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as straight_fn
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from gymnasium.wrappers import RecordVideo
import json
from src.learning.models.beamng_trailer_spec import STATE_FS

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False


def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}


def run_mpc(env: BeamNGTrailerEnv, mpc: MPPI_Jax | MPPI_Jax_Debug, data: DataCollector, env_i, ctl_i, run_i, noise_stdev = 0.1, steps=2000):
    env.reset()
    observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

    # history = jnp.zeros(HISTORY * (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM))
    speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
    i = 0
    t = 0

    rng = np.random.default_rng()

    try:
        traj = []

        for i in range(steps):
            # print(terminated)
            if terminated:
                # print(np.array(traj).shape)
                if len(traj) > 0:  # Should impl larger cutoff
                    data.add(np.array(traj), env_i, ctl_i, 0)
                traj = []
                t += 1
                run_i += 1
                # if t > 2:
                #     break
                env.reset()
                mpc.reset()
                observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

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
            #     f"Time: {elapsed:<7.3f} | "
            #     f"u: {u[0]:<7.3f} {u[1]:<7.3f} | "
            #     # f"Prog: {state.progress:<6.3f} | "
            #     f"vx: {state.vx:<7.3f} | "
            #     f"vy: {state.vy:<7.3f} | "
            #     f"|v|: {jnp.hypot(state.vx, state.vy):<7.3f} | "
            #     f"mu: {env.unwrapped.track.find_mu(state.x, state.y):<7.3f} | "
            # )
            i += 1

            action = jnp.array([-u[0], u[1]])
            noise = rng.standard_normal(size=2) * noise_stdev
            action += noise
            action = jnp.clip(action, -1.0, 1.0)
            

            n_viz = 10    
            # env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz)
            # self._state.x,
            # self._state.y,
            # self._state.yaw_truck,
            # self._state.yaw_trailer,
            # self._state.vx,
            # self._state.vy,
            # self._state.yaw_truck_rate,
            # self._state.yaw_trailer_rate,
            # self._state.steer,
            # self._state.accel,

            print(
                f"\rIter: {i}/{steps}, terimnated: {t}, (env, controller, run #): ( {env_i}, {ctl_i}, {run_i})"
                f"commanded: [{action[0]:6.3f}, {action[1]:6.3f}], "
                f"actual: [{observation[8]:6.3f}, {observation[9]:6.3f}]",
                end="",
            )

            if jnp.any(jnp.isnan(action)):
                terminated = True
                continue

            traj.append(
                np.array([
                    np.sin(observation[2] - observation[3]),
                    np.cos(observation[2] - observation[3]),
                    observation[4],
                    observation[5],
                    observation[6],
                    observation[7],
                    env.unwrapped.track.find_mu(state.x, state.y),
                    observation[8],
                    observation[9],
                    action[0],
                    action[1],
                ])
            )
            observation, reward, terminated, truncated, info = env.step(action)
        # cutoff = 100
        # print(
        #     f"Iters: {i}, "
        #     f"Reverse: {V_TARGET > 0}, "
        #     f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
        #     f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
        #     f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
        #     f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
        # )
    finally:
        if traj != []:
            data.add(np.array(traj), env_i, ctl_i, run_i)
            run_i += 1

        env.close()
        return run_i


# Reverse/fwd configs should be automated

config = BeamNGTrailerEnvConfig(
    ".", TrackConfig(mu=1.0, width=30), bng_pickup_trailer_cfg, SimulationConfig()
)

# config.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

env = BeamNGTrailerEnv(
    config=config,
)

d = DataCollector(11, 0.05)

vels = []
# for v in range(25, 125, 10):
#     vels.append(v)
#     vels.append(-v)
vels = [-20, -30, -40, -50, -60, -70, -80, -90, -100]

controllers = []

# Maybe k=20 will be more noisy
for v in vels:
    v /= 3.6  # to m/s
    for gen_util_funs in [straight_fn]:
        if v > 0:
            dynamics, cost, bound, _ = gen_util_funs(
                config,
                reverse=False,
                v_target=v,
                p_weight=1e2,
                p_slow_weight=1e0,
                c_weight=2e1,
                s_weight=1e-1,
                a_weight=7e2,
            )
            mpc = MPPI_Jax_Debug(
                6,
                2,
                dynamics,
                None,
                cost,
                bound,
                jnp.diag(jnp.array([3e-3, 0.2])),
                inverse_temp=0.5,
                K=20,
                step=0.05,
                T=80,
                alpha=0.05,
            )

    else:
        dynamics, cost, bound, _ = gen_util_funs(
            config,
            reverse=False,
            v_target=v,
            p_weight=2e2,
            p_slow_weight=1e0,
            s_weight=0,
            c_weight=1e1,
            a_weight=3e2,
        )
        mpc = MPPI_Jax_Debug(
            6,
            2,
            dynamics,
            None,
            cost,
            bound,
            jnp.diag(jnp.array([2e-3, 0.2])),
            inverse_temp=0.5,
            K=500,
            step=0.05,
            T=55,
            alpha=0.05,
        )
        controllers.append(mpc)

# Prelim run with full friction

for i, c in enumerate(controllers):
    run_i = 0
    # for j in range(4):
    if vels[i] > 0:
        run_i = run_mpc(env, c, d, 0, i, run_i, noise_stdev=0.3, steps=4000)  # run_i in case several trials of the same
    else:
        run_i = run_mpc(env, c, d, 0, i, run_i, noise_stdev=0.0, steps=1000)  # run_i in case several trials of the same

# ds = d.store(STATE_FS.data_version, verbose=True)

load = DataStore.load(Path("experiments/exp_008_beamng/data_trial2.npz"))
print(load.data.shape)
load.ingest(d)
print(load.data.shape)
load.save("experiments/exp_008_beamng/data_trial2_aug.npz")

# ds.save(Path("./experiments/exp_008_beamng/data_trial2.npz"))