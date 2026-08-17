import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp

from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.trailer_spec_nores import RAW_FS, kin_zeros
from src.dynamics.trailer.model_acceleration_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs as kin_util
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from gymnasium.wrappers import RecordVideo
import json

# Reverse/fwd configs should be automated
V_TARGET = -60 / 3.6

spec = RAW_FS
kin_fn = kin_zeros
HISTORY = spec.H
scenario = TrailerBicycleEnvConfig(
    ".", TrackConfig(mu=1.0, width=10), VehicleConfig(), SimulationConfig()
)

NPZ_SAVE_HEAD = "data_proc2"
JSON_PTH = f"./experiments/exp_007_vehicle_residual_dynamics/{NPZ_SAVE_HEAD}_stats.json"

with open(Path(JSON_PTH), "r") as f:
    norm_stats = json.load(f)

scenario.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

model = TrailerModel(32, 4)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(
    model,
    ckpt.restore(
        Path.cwd() / "src/learning/models/trained/trailer-h4-128-4l-pruned-accel-augsplit2_best",
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


env = TrailerBicycleEnv(
    renderer="pybullet",
    render_mode="rgb_array_birds_eye",
    render_width=450,
    render_height=300,
    scenario=scenario,
)

if V_TARGET > 0:
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
        c_weight=1e0,
        a_weight=7e2,
    )
    mpc = MPPI_Jax_Debug(
        (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM),
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([3e-3, 0.2])),
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
        c_weight=1e-2,
        a_weight=1e2,
    )
    mpc = MPPI_Jax_Debug(
        (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM),
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([3e-3, 0.2])),
        inverse_temp=0.5,
        K=500,
        step=0.05,
        T=55,
        alpha=0.05,
        history=HISTORY,
    )


fname = "rl-video"  # if record_file_name is None else record_file_name
env = RecordVideo(
    env,
    video_folder="gym_videos",
    episode_trigger=lambda x: True,
    disable_logger=True,
    name_prefix=fname,
)

env.reset()
observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

history = jnp.zeros(HISTORY * (D_STATE_DIM + D_U_DIM + D_EXTRA_DIM))
speeds, slip_angles_f, slip_angles_r, yaw_rates = [], [], [], []
i = 0
try:
    cv2.namedWindow("sim", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    # Necessary, the model panics when seeing 0/default windoww
    for _ in range(HISTORY + 1):
        u = jnp.array([0.0, -0.01])
        action = np.array(u)
        observation, reward, terminated, truncated, info = env.step(action)

        state = env.unwrapped._state
        arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        curr = jnp.concatenate(
            [jnp.array([*astuple(state)[:8]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
        )
        history = jnp.concatenate([history[11:], curr])
    for i in range(2000):
        start = time.perf_counter()
        u, xhist, vhist = mpc.run_mpc(history)
        u.block_until_ready()
        elapsed = time.perf_counter() - start
        action = np.array(u)
        observation, reward, terminated, truncated, info = env.step(action)

        state: VehicleState = env.unwrapped._state
        arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
        curr = jnp.concatenate(
            [jnp.array([*astuple(state)[:8]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
        )
        history = jnp.concatenate([history[(D_STATE_DIM + D_U_DIM + D_EXTRA_DIM) :], curr])
        i += 1

        print(i, elapsed, action)

        n_viz = 50
        # print(xhist.shape)
        env.unwrapped.planner_debug = build_planner_debug(
            xhist[..., -(D_STATE_DIM + D_U_DIM + D_EXTRA_DIM) :], n_viz
        )

        speeds.append(jnp.hypot(state.vx, state.vy))
        yaw_rates.append(state.yaw_truck_rate)

        vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
        steer_angle = state.steer * env.unwrapped.scenario.vehicle.max_steer_rad
        alpha_f = steer_angle - jnp.arctan2(
            state.vy + env.unwrapped.scenario.vehicle.lf * state.yaw_truck_rate, vx_safe
        )
        alpha_r = -jnp.arctan2(
            state.vy - env.unwrapped.scenario.vehicle.lr * state.yaw_truck_rate, vx_safe
        )

        slip_angles_f.append(alpha_f)
        slip_angles_r.append(alpha_r)

        if i % 2 == 0:
            frame = env.render()
            cv2.imshow("sim", frame[..., ::-1])
            cv2.waitKey(1)

        if terminated:
            break
    cutoff = 100
    print(
        f"Iters: {i}, "
        f"Reverse: {V_TARGET > 0}, "
        f"Avg speed: {jnp.mean(jnp.array(speeds[cutoff:])) * 3.6}, "
        f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:]))}, "
        f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:]))}, "
        f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:]))}"
    )
finally:
    env.close()
