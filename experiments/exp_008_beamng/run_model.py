import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp
import absl.logging

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False

from src.simulation.beamng_trailer_env import BeamNGTrailerEnv, VehicleState, bng_pickup_trailer_cfg

from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug

from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.debug.smppi_jax_debug import SMPPI_Jax_Debug

from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import STATE_FS, fiala_dyn, IN_COLS
from src.dynamics.trailer.beamng_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from gymnasium.wrappers import RecordVideo
import json

# Reverse/fwd configs should be automated
V_TARGET = -50 / 3.6

spec = STATE_FS
kin_fn = fiala_dyn
HISTORY = spec.H
scenario = BeamNGTrailerEnvConfig   (
    ".", TrackConfig(mu=1.0, width=15), bng_pickup_trailer_cfg, SimulationConfig(dt=0.05)
)

NPZ_SAVE_HEAD = "data_proc_test4"
JSON_PTH = f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"

with open(Path(JSON_PTH), "r") as f:
    norm_stats = json.load(f)

scenario.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

model = TrailerModel(spec.H * len(IN_COLS), 6)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(
    model,
    ckpt.restore(
        Path.cwd() / "src/learning/models/trained/beamng-l4-128-test4_best",
        state,
    ),
)


def build_planner_debug(all_samples, n_vis):
    if all_samples is None:
        return None

    K = all_samples.shape[0]
    n = int(min(n_vis, K))
    idx = jnp.linspace(0, K - 1, n).astype(jnp.int32)  # even spread across samples
    cand = np.asarray(all_samples[idx, :, :2])  # (n, T, 2), small transfer
    return {"candidate_xy": cand}

env = BeamNGTrailerEnv(
    config=scenario,
)

if V_TARGET > 0:
    dynamics, cost, bound, bound_der = res_util(
        scenario,
        spec,
        kin_fn,
        model,
        norm_stats,
        reverse=False,
        v_target=V_TARGET,
        p_weight=1e2,
        p_slow_weight=1e0,
        c_weight=1e0,
        a_weight=7e2,
    )
    mpc = MPPI_Jax_Debug(
        13,
        2,
        dynamics,
        None,
        cost,
        bound,
        # bound_der,
        jnp.diag(jnp.array([3e-3, 0.2])),
        # jnp.diag(jnp.array([1e-2, 1e-1])),
        inverse_temp=0.5,
        K=500,
        step=0.05,
        T=80,
        alpha=0.05,
        history=HISTORY,
    )

else:
    dynamics, cost, bound, _ = res_util(
        scenario,
        spec,
        kin_fn,
        model,
        norm_stats,
        reverse=False,
        v_target=V_TARGET,
        p_weight=1e2,
        p_slow_weight=1e0,
        # s_weight=2e1,
        c_weight=5e1,
        a_weight=2e2,
    )
    mpc = MPPI_Jax_Debug(
        13,
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([2e-2, 0.2])),
        # inverse_temp=5e2,
        inverse_temp=10,
        K=1500,
        step=0.05,
        T=55,
        alpha=0.005,
        history=HISTORY,
    )

# fname = "rl-video" # if record_file_name is None else record_file_name
# env = RecordVideo(env, video_folder="gym_videos", episode_trigger=lambda x: True, disable_logger=True, name_prefix=fname)

env.reset()
observation, reward, terminated, truncated, info = env.step(jnp.zeros(2))

history = jnp.zeros(HISTORY * 13)
speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
i = 0
try:
    # cv2.namedWindow("sim", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    # Necessary, the model panics when seeing 0/default windoww
    for _ in range(HISTORY + 20):
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
    for i in range(2000):
        start = time.perf_counter()
        u, xhist, vhist = mpc.run_mpc(history)
        u.block_until_ready()
        elapsed = time.perf_counter() - start
        action = np.array(u)
        observation, reward, terminated, truncated, info = env.step(action)
        # break
        state: VehicleState = env.unwrapped._state
        arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        curr = jnp.concatenate(
            [jnp.array([*astuple(state)[:10]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
        )
        history = jnp.concatenate([history[13:], curr])
        i += 1

        print(i, elapsed, action)

        n_viz = 10
        # print(xhist.shape)
        env.unwrapped.planner_debug = build_planner_debug(
            xhist[..., -13:], n_viz
        )

        # speeds.append(jnp.hypot(state.vx, state.vy))
        # yaw_rates.append(state.yaw_truck_rate)

        # vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
        # steer_angle = state.steer * env.unwrapped.scenario.vehicle.max_steer_rad
        # alpha_f = steer_angle - jnp.arctan2(
        #     state.vy + env.unwrapped.scenario.vehicle.lf * state.yaw_truck_rate, vx_safe
        # )
        # alpha_r = -jnp.arctan2(
        #     state.vy - env.unwrapped.scenario.vehicle.lr * state.yaw_truck_rate, vx_safe
        # )

        # slip_angles_f.append(alpha_f)
        # slip_angles_r.append(alpha_r)

        # if i % 2 == 0:
        #     frame = env.render()
        #     cv2.imshow("sim", frame[..., ::-1])
        #     cv2.waitKey(1)

        if terminated:
            break
    cutoff = 100
    # print(
    # f"Iters: {i}, "
    # f"Reverse: {V_TARGET > 0}, "
    # f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
    # f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
    # f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
    # f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
    # )
finally:
    env.close()
