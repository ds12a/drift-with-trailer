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

spec = RAW_FS
kin_fn = kin_zeros
HISTORY = spec.H
STATE_DIM = D_STATE_DIM + D_U_DIM + D_EXTRA_DIM  # 11

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


def build_mpc(v_target):
    """Build the dynamics/cost/bound + MPPI controller for a given target speed (m/s)."""
    if v_target > 0:
        dynamics, cost, bound, _ = res_util(
            scenario,
            spec,
            kin_fn,
            model,
            norm_stats,
            reverse=False,
            v_target=v_target,
            p_weight=1e2,
            p_slow_weight=1e0,
            c_weight=1e0,
            a_weight=7e2,
        )
    else:
        dynamics, cost, bound, _ = res_util(
            scenario,
            spec,
            kin_fn,
            model,
            norm_stats,
            reverse=False,
            v_target=v_target,
            p_weight=1e2,
            p_slow_weight=1e0,
            c_weight=1e-2,
            a_weight=1e2,
        )

    # NOTE: original script always overwrote the branch-specific controller
    # with a final T=45 build (dynamics/cost/bound reused). Preserved here.
    mpc = MPPI_Jax_Debug(
        STATE_DIM,
        2,
        dynamics,
        None,
        cost,
        bound,
        jnp.diag(jnp.array([3e-3, 0.2])),
        inverse_temp=0.5,
        K=500,
        step=0.05,
        T=45,
        alpha=0.05,
        history=HISTORY,
    )
    return mpc


def run_episode(v_target, max_iters=2000, render=True, seed=None):
    """Run one closed-loop rollout at a fixed target speed (m/s).

    Returns a dict with iters survived, avg speed (km/h, post-warmup),
    success flag, and net distance traveled along the track (signed).
    """
    mpc = build_mpc(v_target)

    env = TrailerBicycleEnv(
        renderer="pybullet",
        render_mode="rgb_array_birds_eye",
        render_width=450,
        render_height=300,
        scenario=scenario,
    )
    if render:
        env = RecordVideo(
            env,
            video_folder="gym_videos",
            episode_trigger=lambda x: True,
            disable_logger=True,
            name_prefix="rl-video",
        )
        cv2.namedWindow("sim", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    env.step(jnp.zeros(3))

    history = jnp.zeros(HISTORY * STATE_DIM)
    speeds = []
    terminated, truncated = False, False
    start_arclen, arclen = None, None
    i = 0

    try:
        for _ in range(HISTORY + 1):
            u = jnp.array([0.0, -0.01])
            action = np.array(u)
            observation, reward, terminated, truncated, info = env.step(action)

            state = env.unwrapped._state
            arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
            if start_arclen is None:
                start_arclen = arclen
            curr = jnp.concatenate(
                [jnp.array([*astuple(state)[:8]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
            )
            history = jnp.concatenate([history[11:], curr])

        for i in range(1, max_iters + 1):
            u, xhist, vhist = mpc.run_mpc(history)
            u.block_until_ready()
            action = np.array(u)
            observation, reward, terminated, truncated, info = env.step(action)

            state: VehicleState = env.unwrapped._state
            arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
            curr = jnp.concatenate(
                [jnp.array([*astuple(state)[:8]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
            )
            history = jnp.concatenate([history[STATE_DIM:], curr])

            speeds.append(float(jnp.hypot(state.vx, state.vy)))

            if render:
                n_viz = 50
                env.unwrapped.planner_debug = build_planner_debug(
                    xhist[..., -STATE_DIM:], n_viz
                )
                if i % 2 == 0:
                    frame = env.render()
                    cv2.imshow("sim", frame[..., ::-1])
                    cv2.waitKey(1)

            if terminated or truncated:
                break
    finally:
        env.close()
        if render:
            cv2.destroyAllWindows()

    cutoff = min(100, len(speeds))
    avg_v = float(np.mean(speeds[cutoff:])) * 3.6 if len(speeds) > cutoff else float("nan")
    distance = (
        float(arclen - start_arclen) if (arclen is not None and start_arclen is not None) else float("nan")
    )
    # Success := ran the full iteration budget without the env terminating/truncating.
    # Swap in your own criterion here if you have a better definition (finish line, etc).
    success = (not terminated) and (not truncated) and (i >= max_iters)

    return {
        "v_target": v_target * 3.6,  # km/h, for table readability
        "iters": i,
        "avg_v": avg_v,
        "success": success,
        "distance": distance,
    }


def run_sweep(v_targets_kmh, n_seeds=1, max_iters=2000):
    results = []
    for v_kmh in v_targets_kmh:
        v_target = v_kmh / 3.6
        runs = [
            run_episode(v_target, max_iters=max_iters, seed=s if n_seeds > 1 else None)
            for s in range(n_seeds)
        ]
        iters = np.array([r["iters"] for r in runs])
        avg_vs = np.array([r["avg_v"] for r in runs])
        dists = np.array([r["distance"] for r in runs])
        succ = np.array([r["success"] for r in runs])
        results.append(
            {
                "v_target": v_kmh,
                "avg_it": iters.mean(),
                "avg_v": np.nanmean(avg_vs),
                "succ": succ.mean() * 100,
                "dist": dists.mean(),
            }
        )
    return results


def print_table(results, mu, n_seeds):
    width = 60
    print(f" SWEEP mu={mu} n={n_seeds} seeds ".center(width, "="))
    print(f"{'v_tgt':>7} {'avg_v':>8} {'avg_it':>8} {'succ':>6} {'dist':>10}")
    for r in results:
        print(
            f"{r['v_target']:>7.0f} {r['avg_v']:>8.1f} {r['avg_it']:>8.0f} "
            f"{r['succ']:>5.0f}% {r['dist']:>10.1f}"
        )
    print("=" * width)


if __name__ == "__main__":
    V_TARGETS_KMH = [-30, -50, -70, 40, 60, 80]
    N_SEEDS = 1
    MAX_ITERS = 2000

    results = run_sweep(V_TARGETS_KMH, n_seeds=N_SEEDS, max_iters=MAX_ITERS)
    print_table(results, mu=scenario.track.mu, n_seeds=N_SEEDS)