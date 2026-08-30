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

    Returns a summary dict regardless of `quiet`/`benchmark`:
        {v_target, iters, avg_v, success, distance}
    v_target is cost_kwargs['v_target'] converted to km/h, used as-is (i.e.
    already signed in your convention -- negative means reverse); None if
    cost_kwargs['v_target'] is None.
    success := ran the full `max_steps` budget without the env terminating
    (only meaningful when max_steps is given; otherwise always False).
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
        ctl_args = (6, 2, dynamics, None, cost, bound, *ctl_args)
        
        if debug:
            # print("Using MPPI")
            mpc = MPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = MPPI_Jax(*ctl_args, **ctl_kwargs)
    else:
        # Assume SMPPI
        ctl_args = (6, 2, dynamics, None, cost, bound, bound_der, *ctl_args)

        if debug:
            # print("Using SMPPI")
            mpc = SMPPI_Jax_Debug(*ctl_args, **ctl_kwargs)
        else:
            mpc = SMPPI_Jax(*ctl_args, **ctl_kwargs)
    

    observation, reward, terminated, truncated, info = env.step(jnp.zeros(3))

    start_arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]
    last_arclen = start_arclen

    loop = range(max_steps) if max_steps is not None else itertools.count()

    i = 0
    try:
        for i in loop:
            if terminated:
                break

            start = time.perf_counter()

            state: VehicleState = env.unwrapped._state
            last_arclen = env.unwrapped.track._arc_samples[env.unwrapped._last_index]

            mpc_state = jnp.array(
                [
                    *astuple(state)[:-2],
                    env.unwrapped.track.find_mu(state.x, state.y),
                    last_arclen,
                ]
            )

            xhist = None
            if debug:
                u, xhist, *_ = mpc.run_mpc(mpc_state)
            else:
                u = mpc.run_mpc(mpc_state)
            
            u.block_until_ready()

            elapsed = time.perf_counter() - start

            speeds.append(float(jnp.hypot(state.vx, state.vy)))

            if benchmark:
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
            env.unwrapped.planner_debug = build_planner_debug(xhist, n_viz) if debug else None

            observation, reward, terminated, truncated, info = env.step(action)

            if not headless:
                frame = env.render()
                cv2.imshow("sim", frame[..., ::-1])
                cv2.waitKey(1)
    
    except KeyboardInterrupt:
        pass

    env.close()

    cutoff = min(100, len(speeds))
    avg_v = float(np.mean(speeds[cutoff:])) * 3.6 if len(speeds) > cutoff else float("nan")
    distance = float(last_arclen - start_arclen)
    success = bool((not terminated) and (max_steps is not None) and (i >= max_steps))

    raw_v_target = cost_kwargs.get("v_target")
    v_target_kmh = raw_v_target * 3.6 if raw_v_target is not None else float("nan")

    if benchmark:
        if print_name is not None:
            print(print_name)

        print(
            f"Iters: {i}, "
            f"Reverse: {cost_kwargs['reverse']}, "
            f"Avg speed: {avg_v}, "
            f"Avg alpha_f: {jnp.mean(jnp.array(slip_angles_f[cutoff:])) if slip_angles_f else float('nan')}, "
            f"Avg alpha_r: {jnp.mean(jnp.array(slip_angles_r[cutoff:])) if slip_angles_r else float('nan')}, "
            f"Avg yaw_rate: {jnp.mean(jnp.array(yaw_rates[cutoff:])) if yaw_rates else float('nan')}"
        )

    return {
        "v_target": v_target_kmh,
        "iters": i,
        "avg_v": avg_v,
        "success": success,
        "distance": distance,
    }


def run_sweep(controller, ctl_args, ctl_kwargs, cost_kwargs_list, n_seeds=1, max_steps=2000, **run_mpc_kwargs):
    """Call run_mpc once per dict in cost_kwargs_list (x n_seeds), quietly and
    headless (no printing, no window, no recording), and collect summary rows.

    cost_kwargs_list: list of cost_kwargs dicts, each typically varying at least
    'v_target' (and 'reverse') -- everything else (mu, weights, etc.) can be
    baked into each dict as needed.

    Any extra kwargs (e.g. env_kwargs) are forwarded to every run_mpc call.
    """
    results = []
    for cost_kwargs in cost_kwargs_list:
        runs = []
        for _ in range(n_seeds):
            r = run_mpc(
                controller,
                ctl_args,
                ctl_kwargs,
                cost_kwargs,
                record=False,
                debug=False,
                quiet=True,
                benchmark=False,
                headless=True,
                max_steps=max_steps,
                **run_mpc_kwargs,
            )
            runs.append(r)

        iters = np.array([r["iters"] for r in runs])
        avg_vs = np.array([r["avg_v"] for r in runs])
        dists = np.array([r["distance"] for r in runs])
        succ = np.array([r["success"] for r in runs])

        results.append(
            {
                "v_target": runs[0]["v_target"],
                "avg_it": iters.mean(),
                "avg_v": np.nanmean(avg_vs),
                "succ": succ.mean() * 100,
                "dist": dists.mean(),
            }
        )
    return results


def print_table(results, n_seeds=1, mu=None):
    # If rows carry a 'label' (e.g. controller name), show it as a leading column
    # so rows sharing the same v_target are still distinguishable.
    has_label = bool(results) and "label" in results[0]
    width = 70 if has_label else 60

    header = f" SWEEP n={n_seeds} seeds " if mu is None else f" SWEEP mu={mu} n={n_seeds} seeds "
    print(header.center(width, "="))

    if has_label:
        print(f"{'ctl':>8} {'v_tgt':>7} {'avg_v':>8} {'avg_it':>8} {'succ':>6} {'dist':>10}")
    else:
        print(f"{'v_tgt':>7} {'avg_v':>8} {'avg_it':>8} {'succ':>6} {'dist':>10}")

    for r in results:
        row = f"{r['v_target']:>7.0f} {r['avg_v']:>8.1f} {r['avg_it']:>8.0f} {r['succ']:>5.0f}% {r['dist']:>10.1f}"
        if has_label:
            row = f"{r.get('label', ''):>8} " + row
        print(row)
    print("=" * width)


import jax.numpy as jnp

# Configs to be used -- MPPI only, forward and reverse

mppi_cfg_fwd = (
    (
       jnp.diag(jnp.array([3e-3, 0.2])),
    ),
    {
        "inverse_temp": 1,
        "K": 500,
        "step": 0.05,
        "T": 80,
        "alpha": 0.05,
    },
    {
        "reverse": False,
        "v_target": 25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "s_weight": 2e2,
        "c_weight": 1e0,
        "a_weight": 7e2,
    },
)

mppi_cfg_rev = (
    (
        jnp.diag(jnp.array([3e-3, 0.2])),
    ),
    {
        "inverse_temp": 0.5,
        "K": 750,
        "step": 0.05,
        "T": 55,
        "alpha": 0.05,
    },
    {
        "reverse": False,
        "v_target": -25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "s_weight": 1e2,
        "c_weight": 1e-2,
        "a_weight": 1e2,
    },
)


if __name__ == "__main__":

    # (print_name, v_t in km/h; negative = reverse)
    trials = [
        ("MPPI For.; v_t = 40", 40),
        ("MPPI For.; v_t = 60", 60),
        ("MPPI For.; v_t = 90", 90),
        ("MPPI Rev.; v_t = 30", -30),
        ("MPPI Rev.; v_t = 50", -50),
        ("MPPI Rev.; v_t = 70", -70),
    ]

    table_rows = []

    for name, v_t in trials:

        v_tt = None if v_t is None else v_t / 3.6
        is_reverse = v_t is not None and v_t < 0

        base_ctl_args, base_ctl_kwargs, base_cost_kwargs = mppi_cfg_rev if is_reverse else mppi_cfg_fwd
        # copy so trials don't mutate the shared base config dict
        cost_kwargs = dict(base_cost_kwargs)
        cost_kwargs["v_target"] = v_tt

        result = run_mpc(
            "MPPI",
            base_ctl_args,
            base_ctl_kwargs,
            cost_kwargs,
            record=True,
            benchmark=True,
            quiet=True,
            debug=True,
            max_steps=2000,
            print_name=name,
            record_file_name=f"MPPI_v={v_t}",
        )

        # run_mpc returns per-call field names (iters/success/distance);
        # print_table expects the aggregated names (avg_it/succ/dist), so a
        # single call's result maps 1:1 as its own one-row "average".
        table_rows.append(
            {
                "label": "MPPI",
                "v_target": result["v_target"],
                "avg_v": result["avg_v"],
                "avg_it": result["iters"],
                "succ": 100.0 if result["success"] else 0.0,
                "dist": result["distance"],
            }
        )

    print_table(table_rows, n_seeds=1)