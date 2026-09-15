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


from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import STATE_FS, fiala_dyn, IN_COLS
from src.learning.models.beamng_model_spec import STATE_FS
from src.dynamics.trailer.beamng_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as prior_util

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False

STALL_VEL = 0.5
STALL_WINDOW = 40
STALL_ARC = 1.0
STALL_GRACE = 100


def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}



spec = STATE_FS
kin_fn = fiala_dyn
HISTORY = 4
scenario = BeamNGTrailerEnvConfig   (
    ".", TrackConfig(mu=1.0, width=15), bng_pickup_trailer_cfg, SimulationConfig(dt=0.05)
)

NPZ_SAVE_HEAD = "data_proc_test9-new"
JSON_PTH = f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"

with open(Path(JSON_PTH), "r") as f:
    norm_stats = json.load(f)

scenario.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

model = TrailerModel(4 * len(IN_COLS), 6)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(
    model,
    ckpt.restore(
        Path.cwd() / "src/learning/models/trained/beamng-l4-128-test9-new_best",
        state,
    ),
)

def run_mpc(env: BeamNGTrailerEnv, mpc: MPPI_Jax | MPPI_Jax_Debug, data: DataCollector, env_i, ctl_i, run_i, noise_stdev = 0.1, steps=2000, mirror=False):
    env.reset()
    observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

    # history = jnp.zeros(HISTORY * (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM))
    history = jnp.zeros(HISTORY * 13)
    speeds, arc_lengths = [], []
    slip_angles_f, slip_angles_r, yaw_rates = [], [], []
    i = 0
    t = 0

    rng = np.random.default_rng()

    try:
        traj = []


        for _ in range(HISTORY):
            u = jnp.array([0.0, -0.35])
            action = np.array(u)
            observation, reward, terminated, truncated, info = env.step(action)

            state = env.unwrapped._state
            arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]

            # TODO reduce code dupe
            curr = jnp.concatenate(
                [jnp.array([*astuple(state)[:10]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
            )
            history = jnp.concatenate([history[13:], curr])


            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )
            
        for i in range(steps):
            # print(terminated)
            if terminated:
                # print(np.array(traj).shape)
                if len(traj) > 0:  # Should impl larger cutoff
                    data.add(np.array(traj), env_i, ctl_i, 0)

                    if mirror:
                        mirrored = np.array(traj)
                        mirrored[:, [0, 3, 4, 5, 7, 9]] *= -1   
                        data.add(mirrored, env_i, ctl_i, 0)


                traj = []
                t += 1
                run_i += 1
                # if t > 2:
                #     break
                env.reset()
                mpc.reset()
                break
                mpc_u_traj = None
                observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

                # TODO reducue code dupe
                for _ in range(HISTORY):
                    # print("here") 
                    u = jnp.array([0.0, -0.35])
                    action = np.array(u)
                    observation, reward, terminated, truncated, info = env.step(action)
        
                    state = env.unwrapped._state
                    arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        
                    # TODO reduce code dupe
                    curr = jnp.concatenate(
                        [jnp.array([*astuple(state)[:10]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
                    )
                    history = jnp.concatenate([history[13:], curr])
        
        
                    mpc_state = jnp.array(
                        [
                            *astuple(state)[:-2],
                            env.unwrapped.track.find_mu(state.x, state.y),
                            env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                        ]
                    )

            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state

            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )
            
            u, xhist, vhist = mpc.run_mpc(history)
            u.block_until_ready()

            elapsed = time.perf_counter() - start
            i += 1

            action = jnp.array([u[0], u[1]])
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

            current_state = env.unwrapped._state
            curr = jnp.concatenate(
                [jnp.array([*astuple(current_state)[:10]]), jnp.array([action[0], action[1]]), jnp.array([env.unwrapped.track._arc_samples[env.unwrapped._last_index]])]
            )
            history = jnp.concatenate([history[13:], curr])

            speeds.append(float(np.hypot(current_state.vx, current_state.vy)))
            arc_lengths.append(float(env.unwrapped.track._arc_samples[env.unwrapped._last_index]))
            if i >= STALL_GRACE and len(speeds) > STALL_WINDOW:
                slow = np.mean(speeds[-STALL_WINDOW:]) < STALL_VEL
                arc_delta = arc_lengths[-1] - arc_lengths[-STALL_WINDOW]
                track_length = env.unwrapped.track.length
                arc_delta = (arc_delta + track_length / 2) % track_length - track_length / 2
                if slow and abs(arc_delta) < STALL_ARC:
                    print("\nStall detected; ending collection run.")
                    break

            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.find_mu(state.x, state.y),
                    env.unwrapped.track._arc_samples[env.unwrapped._last_index],
                ]
            )
        # cutoff = 100
        # print(
        #     f"Iters: {i}, "
        #     f"Reverse: {V_TARGET > 0}, "
        #     f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
        #     f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
        #     f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
        #     f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
        # )
    except KeyboardInterrupt:
        pass
    except:
        raise
    finally:
        # print("woomp")
        if traj != []:
            data.add(np.array(traj), env_i, ctl_i, run_i)
            run_i += 1

        env.close()
    return run_i


# Reverse/fwd configs should be automated

d = DataCollector(11, 0.05)

vels = []
# for v in range(25, 125, 10):
#     vels.append(v)
#     vels.append(-v)
vels = [-40, -50,-60]

fwd_weights = {
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 7e2,
    "reverse": False,
}
rev_weights = {
    "p_weight": 5e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 2e2,
    "reverse": False,
}



mus = [0.4, 0.6, 0.8, 1.0]
for env_i, m in enumerate(mus):

    controllers = []

    config = BeamNGTrailerEnvConfig(
        ".", TrackConfig(mu=m, width=15), bng_pickup_trailer_cfg, SimulationConfig()
    )

    # config.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

    for V_TARGET in vels:
        V_TARGET /= 3.6  # to m/s
        # for gen_util_funs in [straight_fn]:

        # if V_TARGET > 0:
        #     gen_util_funs = sin_fn
        # else:
        #     gen_util_funs = straight_fn

            
        if V_TARGET > 0:
            
            dynamics, cost, bound, bound_der = res_util(
                scenario,
                spec,
                kin_fn,
                model,
                norm_stats,
                v_target=V_TARGET,
                **fwd_weights,
            )
            mpc = MPPI_Jax_Debug(
                13,
                2,
                dynamics,
                None,
                cost,
                bound,
                # bound_der,
                jnp.diag(jnp.array([7e-2, 0.2])),
                inverse_temp=150,
                # inverse_temp=10,
                K=500,
                step=0.05,
                T=80,
                alpha=0.01,
                gamma=0.0,
                history=HISTORY,
            )

        else:
            dynamics, cost, bound, bound_der = res_util(
                scenario,
                spec,
                kin_fn,
                model,
                norm_stats,
                v_target=V_TARGET,
                **rev_weights,
            )
            mpc = MPPI_Jax_Debug(
                13,
                2,
                dynamics,
                None,
                cost,
                bound,
                jnp.diag(jnp.array([1e-2, 0.2])),
                inverse_temp=500,
                # inverse_temp=10,
                K=500,
                step=0.05,
                T=75,
                alpha=0.01,
                gamma=0.0,
                history=HISTORY,
            )
        controllers.append(mpc)

    for i, c in enumerate(controllers):

        run_i = 0
        env = BeamNGTrailerEnv(config=config)
        # for j in range(4):
        if vels[i] > 0:
            run_i = run_mpc(env, c, d, env_i, i, run_i, noise_stdev=0.3, steps=1000)  # run_i in case several trials of the same
        else:
            run_i = run_mpc(env, c, d, env_i, i, run_i, noise_stdev=0.0, steps=1500, mirror=True)  # run_i in case several trials of the same

# ds = d.store(STATE_FS.data_version, verbose=True)

load = DataStore.load(Path("experiments/exp_008_beamng/data_trial3_aug1v3.npz"))
print(load.data.shape)
load.ingest(d)
print(load.data.shape)
load.save("experiments/exp_008_beamng/data_trial3_aug1v4.npz")

# ds.save(Path("./experiments/exp_008_beamng/data_trial2.npz"))
