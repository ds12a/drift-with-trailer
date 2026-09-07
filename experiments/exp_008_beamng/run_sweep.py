"""
BeamNG sweep over controllers, velocities, start indices and seeds.
python -m experiments.exp_008_beamng.run_sweep

CSV summaries and raw NPZ samples are saved under sweep_out.
Timing includes first-call compilation; aggregate p95 pools calls per cell.
Distance survived is net progress in the requested direction after warmup.
Ctrl-C skips an episode; twice within DOUBLE_TAP_S stops the sweep.
"""

import os

from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import csv
import json
import logging
import time
from dataclasses import astuple
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import absl.logging
import random

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False

from src.simulation.beamng_trailer_env import (
    BeamNGTrailerEnv,
    VehicleState,
    bng_pickup_trailer_cfg,
)
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    TrackConfig,
    SimulationConfig,
)
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.utils.track import TrackModel
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import fiala_dyn
from src.learning.models.beamng_model_spec import STATE_FS, IN_COLS
from src.dynamics.trailer.beamng_dynamics import gen_util_funs as res_util
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as prior_util
from experiments.exp_008_beamng.run_ipopt import (
    HORIZON as IPOPT_HORIZON,
    IPOPT_SETTINGS,
    make_mpc as make_ipopt_mpc,
)
from experiments.exp_008_beamng.run_pure_pursuit import (
    PURE_PURSUIT_SETTINGS,
    make_mpc as make_pure_pursuit,
)
from experiments.exp_008_beamng.run_lqr_pid import (
    LQR_PID_SETTINGS,
    make_mpc as make_lqr_pid,
)

jnp.set_printoptions(precision=2, suppress=True)


# Config

CONTROLLERS = [ "ipopt", "lqr_pid"]
VELS_KPH = [-40, -60, 100]
START_INDICES = [0, 800, 1600]  # centerline indices
SEEDS = [random.randint(0, 100000000) for _ in range(3)]

print("seeds", SEEDS)

MAX_STEPS = 500
CUTOFF = 100  # discard initial steps for velocity/hitch averages

# Stall requires both low speed and low track progress.
STALL_VEL = 0.5  # m/s
STALL_WINDOW = 40  # steps (2.0 s at dt=0.05)
STALL_ARC = 1.0  # m of track progress over the window
STALL_GRACE = 100  # steps

DOUBLE_TAP_S = 2.0

MU = 1.0
TRACK_WIDTH = 15
FRICTION_CSV = None #"src/simulation/assets/tracks/barcelona_ice.csv"  # None to leave unset
DT = 0.05

NPZ_SAVE_HEAD = "data_proc_test8"
JSON_PTH = f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"
CKPT_PTH = "src/learning/models/trained/beamng-l4-128-test8_best"

OUT_ROOT = Path("./experiments/exp_008_beamng/sweep_out")

FWD_WEIGHTS = {
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 7e2,
    "reverse": False,
}
REV_WEIGHTS = {
    "p_weight": 2e1,
    "p_slow_weight": 1e0,
    "c_weight": 3e1,
    "a_weight": 2e2,
    "reverse": False,
}


# Forward/reverse MPPI settings
FWD_MPPI = {
    "cv": jnp.diag(jnp.array([3e-2, 0.2])),
    "inverse_temp": 150,
    "K": 500,
    "step": 0.05,
    "T": 80,
    "alpha": 0.01,
    "gamma": 0.0,
}
REV_MPPI = {
    "cv": jnp.diag(jnp.array([2e-2, 0.2])),
    "inverse_temp": 100,
    "K": 500,
    "step": 0.05,
    "T": 65,
    "alpha": 0.01,
    "gamma": 0.0,
}

# Override order: FWD/REV_MPPI, PER_VEL_OVERRIDE, PRIOR_OVERRIDE.
PER_VEL_OVERRIDE: dict[int, dict] = {}

# Empty uses the learned controller's sampling settings.
PRIOR_OVERRIDE: dict = {}


# Setup

spec = STATE_FS
HISTORY = spec.H
ROW_W = 13  # [x, y, phi1, phi2, vx, vy, phi1dot, phi2dot, delta_s, accel_s | d_cmd, a_cmd | arclen]

with open(Path(JSON_PTH), "r") as f:
    norm_stats = json.load(f)

scenario = BeamNGTrailerEnvConfig(
    ".", TrackConfig(mu=MU, width=TRACK_WIDTH), bng_pickup_trailer_cfg, SimulationConfig(dt=DT)
)
if FRICTION_CSV is not None:
    scenario.track.friction_csv = FRICTION_CSV

model = None


def load_model():
    """Load the checkpoint once, when needed."""
    global model
    if model is None:
        model = TrailerModel(spec.H * len(IN_COLS), 6)
        _, model_state = nnx.split(model)
        checkpoint = ocp.StandardCheckpointer()
        nnx.update(model, checkpoint.restore(Path.cwd() / CKPT_PTH, model_state))
    return model

# beamng_dynamics currently ignores the prior term.
KIN_FN = fiala_dyn


def log(msg=""):
    print(msg, flush=True)


def progress(msg):
    print(f"\r{msg:<124}", end="", flush=True)


def clear_progress():
    print("\r" + " " * 124 + "\r", end="", flush=True)


# Controllers


def make_mpc(v_kph, kind):
    """Reuse each controller across starts/seeds to avoid JAX retracing."""
    v_target = v_kph / 3.6
    if kind == "pure_pursuit":
        return make_pure_pursuit(scenario, v_target, PURE_PURSUIT_SETTINGS)
    if kind == "lqr_pid":
        return make_lqr_pid(scenario, v_target, LQR_PID_SETTINGS)
    if kind not in ("model", "prior", "ipopt"):
        raise ValueError(f"Unknown controller: {kind}")
    weights = dict(FWD_WEIGHTS if v_target > 0 else REV_WEIGHTS)
    weights["v_target"] = v_target

    if kind == "ipopt":
        return make_ipopt_mpc(scenario, v_target, IPOPT_HORIZON, weights)

    cfg = dict(FWD_MPPI if v_target > 0 else REV_MPPI)
    cfg.update(PER_VEL_OVERRIDE.get(v_kph, {}))
    if kind == "prior":
        cfg.update(PRIOR_OVERRIDE)
    cv = cfg.pop("cv")

    if kind == "model":
        dynamics, cost, bound, _ = res_util(
            scenario, spec, KIN_FN, load_model(), norm_stats, **weights
        )
        x_d, hist = ROW_W, HISTORY
    else:
        dynamics, cost, bound, _ = prior_util(scenario, s_weight=0, **weights)
        x_d, hist = 6, None

    return MPPI_Jax_Debug(x_d, 2, dynamics, None, cost, bound, cv, history=hist, **cfg)


# Episode

_last_interrupt = [0.0]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_episode(env, mpc, v_kph, seed, kind, start_index=0):
    """Return episode samples and outcome. Double Ctrl-C raises KeyboardInterrupt."""
    v_target = v_kph / 3.6
    track = env.unwrapped.track
    L = track.length
    max_hitch = env.unwrapped.config.vehicle.max_hitch

    def arc_delta(prev, curr):
        d = curr - prev
        if d < -L / 2:
            d += L
        elif d > L / 2:
            d -= L
        return d

    def mpc_state():
        """State + mu + arc length."""
        s = env.unwrapped._state
        return jnp.array(
            [
                *astuple(s)[:-2],
                track.find_mu(s.x, s.y),
                track._arc_samples[env.unwrapped._last_index],
            ]
        )

    env.reset(seed=seed, options={"start_index": start_index})
    if kind != "ipopt":
        mpc.reset()
    if kind in ("model", "prior"):
        mpc.key = jax.random.key(seed)

    env.step(jnp.zeros(2))

    # Fill the history window; same open-loop warmup for all controllers.
    history = jnp.zeros(HISTORY * ROW_W)
    warm_a = 0.35 if v_target > 0 else -0.35
    for _ in range(HISTORY):
        u = jnp.array([0.0, warm_a])
        env.step(np.array(u))
        state = env.unwrapped._state
        arclen = track._arc_samples[env.unwrapped._last_index]
        curr = jnp.concatenate(
            [jnp.array([*astuple(state)[:10]]), jnp.array([u[0], u[1]]), jnp.array([arclen])]
        )
        history = jnp.concatenate([history[ROW_W:], curr])

    speeds, velocities, hitches, hitch_rates = [], [], [], []
    lat_errs, cum_dist, solve_ms, us = [], [], [], []
    dist = 0.0
    initial_state = env.unwrapped._state
    initial_proj, _ = track.project(initial_state.x, initial_state.y, env.unwrapped._last_index)
    prev_arc = initial_proj.arc_length
    outcome = "completed"
    iters = 0

    try:
        for i in range(MAX_STEPS):
            t0 = time.perf_counter()
            if kind == "ipopt":
                try:
                    u = mpc.run_mpc(mpc_state(), verbose=False, warm_start=False)
                except RuntimeError:
                    solve_ms.append((time.perf_counter() - t0) * 1e3)
                    outcome = "solver_failed"
                    break
            elif kind in ("pure_pursuit", "lqr_pid"):
                u = mpc.run_mpc(mpc_state())
            else:
                u, *_ = mpc.run_mpc(history if kind == "model" else mpc_state())
                u.block_until_ready()
            solve_ms.append((time.perf_counter() - t0) * 1e3)

            if bool(jnp.any(jnp.isnan(u))):
                outcome = "nan"
                break

            # The learned model uses BeamNG commands; analytic steer is inverted.
            if kind in ("ipopt", "pure_pursuit", "lqr_pid"):
                action = jnp.clip(jnp.array([-u[0], u[1]]), -1.0, 1.0)
            elif kind == "prior":
                action = jnp.array([-u[0], u[1]])
            else:
                action = u

            env.step(np.array(action))
            iters = i + 1

            state: VehicleState = env.unwrapped._state
            arclen = track._arc_samples[env.unwrapped._last_index]
            curr = jnp.concatenate(
                [
                    jnp.array([*astuple(state)[:10]]),
                    jnp.array([action[0], action[1]]),
                    jnp.array([arclen]),
                ]
            )
            history = jnp.concatenate([history[ROW_W:], curr])

            # Projected arc length for metrics, not the quantized training value.
            proj, _ = track.project(state.x, state.y, env.unwrapped._last_index)
            if prev_arc is not None:
                dist += arc_delta(prev_arc, proj.arc_length)
            prev_arc = proj.arc_length

            speed = float(np.hypot(state.vx, state.vy))
            hitch = float(wrap_angle(state.yaw_trailer - state.yaw_truck))
            hitch_rate = float(state.yaw_trailer_rate - state.yaw_truck_rate)
            speeds.append(speed)
            velocities.append([float(state.vx), float(state.vy)])
            hitches.append(hitch)
            hitch_rates.append(hitch_rate)
            lat_errs.append(float(proj.lateral_error))
            cum_dist.append(dist)
            us.append([float(action[0]), float(action[1])])

            progress(
                f"[{kind:<5} idx={start_index} v={v_kph:>4} kph seed {seed}] step {iters:>4}/{MAX_STEPS} "
                f"|v| {speed * 3.6:>6.1f} kph  hitch {hitch * 180.0 / np.pi:>6.2f}\u00b0  "
                f"lat {proj.lateral_error:>6.2f}  dist {dist:>7.1f} m"
            )

            # Separate jackknife and offtrack outcomes.
            if abs(hitch) >= max_hitch:
                outcome = "jackknife"
                break
            if track.out_of_bounds(proj.lateral_error):
                outcome = "offtrack"
                break

            if i >= STALL_GRACE and len(speeds) > STALL_WINDOW:
                slow = np.mean(speeds[-STALL_WINDOW:]) < STALL_VEL
                stuck = abs(cum_dist[-1] - cum_dist[-STALL_WINDOW]) < STALL_ARC
                if slow and stuck:
                    outcome = "stalled"
                    break

    except KeyboardInterrupt:
        now = time.time()
        if now - _last_interrupt[0] < DOUBLE_TAP_S:
            raise
        _last_interrupt[0] = now
        outcome = "interrupted"

    clear_progress()

    return {
        "controller": kind,
        "v_kph": v_kph,
        "seed": seed,
        "start_index": start_index,
        "outcome": outcome,
        "iters": iters,
        "speeds": np.asarray(speeds),
        "velocities": np.asarray(velocities).reshape(-1, 2),
        "hitches": np.asarray(hitches),
        "hitch_rates": np.asarray(hitch_rates),
        "lat_errs": np.asarray(lat_errs),
        "cum_dist": np.asarray(cum_dist),
        "solve_ms": np.asarray(solve_ms),
        "us": np.asarray(us).reshape(-1, 2),
    }


DEG = 180.0 / np.pi


def summarize(ep):
    """Velocity/hitch averages use the post-CUTOFF tail; max hitch uses the full run."""
    sp = ep["speeds"]
    h = np.abs(ep["hitches"])
    hd = ep["hitch_rates"]
    tail = sp[CUTOFF:] if sp.size > CUTOFF else sp
    h_t = h[CUTOFF:] if h.size > CUTOFF else h
    hd_t = hd[CUTOFF:] if hd.size > CUTOFF else hd
    vx = ep["velocities"][:, 0]
    vx_t = vx[CUTOFF:] if vx.size > CUTOFF else vx
    distance = float(ep["cum_dist"][-1]) if ep["cum_dist"].size else 0.0
    return {
        "controller": ep["controller"],
        "v_kph": ep["v_kph"],
        "seed": ep["seed"],
        "start_index": ep["start_index"],
        "run_index": ep.get("run_index", 0),
        "outcome": ep["outcome"],
        "iters": ep["iters"],
        "avg_v_kph": float(np.mean(tail) * 3.6) if tail.size else 0.0,
        "avg_vx_kph": float(np.mean(vx_t) * 3.6) if vx_t.size else 0.0,
        "distance_m": distance,
        "distance_survived_m": max(0.0, np.sign(ep["v_kph"]) * distance),
        "frac_below_v": float(np.mean(tail < STALL_VEL)) if tail.size else 1.0,
        "hitch_rms_deg": float(np.sqrt(np.mean(h_t**2)) * DEG) if h_t.size else 0.0,
        "hitch_mean_deg": float(np.mean(h_t) * DEG) if h_t.size else 0.0,
        "hitch_p95_deg": float(np.percentile(h_t, 95) * DEG) if h_t.size else 0.0,
        "hitch_max_deg": float(np.max(h) * DEG) if h.size else 0.0,
        "hitch_final_deg": float(h[-1] * DEG) if h.size else 0.0,
        "hitch_rate_rms": float(np.sqrt(np.mean(hd_t**2))) if hd_t.size else 0.0,
        "mean_lat_err": float(np.mean(np.abs(ep["lat_errs"]))) if ep["lat_errs"].size else 0.0,
        "solve_ms": float(np.mean(ep["solve_ms"])) if ep["solve_ms"].size else 0.0,
        "solve_p95_ms": float(np.percentile(ep["solve_ms"], 95)) if ep["solve_ms"].size else 0.0,
        # Raw timings for aggregation, omitted from CSV.
        "_solve_samples": ep["solve_ms"],
    }


def aggregate(rows):
    """Aggregate non-interrupted runs for one controller/index/velocity."""
    n = len(rows)
    if n == 0:
        return None
    outc = [r["outcome"] for r in rows]
    times = np.concatenate([r["_solve_samples"] for r in rows])
    return {
        "controller": rows[0]["controller"],
        "v_kph": rows[0]["v_kph"],
        "start_index": rows[0]["start_index"],
        "n": n,
        "avg_v_kph": float(np.mean([r["avg_v_kph"] for r in rows])),
        "std_v_kph": float(np.std([r["avg_v_kph"] for r in rows])),
        "avg_vx_kph": float(np.mean([r["avg_vx_kph"] for r in rows])),
        "avg_iters": float(np.mean([r["iters"] for r in rows])),
        "std_iters": float(np.std([r["iters"] for r in rows])),
        "success_rate": outc.count("completed") / n,
        "stall_rate": outc.count("stalled") / n,
        "fail_rate": sum(
            o in ("jackknife", "offtrack", "nan", "solver_failed") for o in outc
        )
        / n,
        "avg_distance_m": float(np.mean([r["distance_m"] for r in rows])),
        "avg_distance_survived_m": float(np.mean([r["distance_survived_m"] for r in rows])),
        "avg_frac_below_v": float(np.mean([r["frac_below_v"] for r in rows])),
        "avg_hitch_rms_deg": float(np.mean([r["hitch_rms_deg"] for r in rows])),
        "std_hitch_rms_deg": float(np.std([r["hitch_rms_deg"] for r in rows])),
        "avg_hitch_mean_deg": float(np.mean([r["hitch_mean_deg"] for r in rows])),
        "avg_hitch_p95_deg": float(np.mean([r["hitch_p95_deg"] for r in rows])),
        "avg_hitch_max_deg": float(np.mean([r["hitch_max_deg"] for r in rows])),
        "avg_hitch_final_deg": float(np.mean([r["hitch_final_deg"] for r in rows])),
        "avg_hitch_rate_rms": float(np.mean([r["hitch_rate_rms"] for r in rows])),
        "avg_solve_ms": float(np.mean(times)) if times.size else 0.0,
        "p95_solve_ms": float(np.percentile(times, 95)) if times.size else 0.0,
    }


# Output

EP_COLS = [
    "controller", "v_kph", "seed", "outcome", "iters", "avg_v_kph", "distance_m",
    "frac_below_v", "hitch_rms_deg", "hitch_mean_deg", "hitch_p95_deg",
    "hitch_max_deg", "hitch_final_deg", "hitch_rate_rms", "mean_lat_err", "solve_ms",
    "start_index", "run_index", "avg_vx_kph", "distance_survived_m", "solve_p95_ms",
]
AGG_COLS = [
    "controller", "v_kph", "n", "avg_v_kph", "std_v_kph", "avg_iters", "std_iters",
    "success_rate", "stall_rate", "fail_rate", "avg_distance_m", "avg_frac_below_v",
    "avg_hitch_rms_deg", "std_hitch_rms_deg", "avg_hitch_mean_deg", "avg_hitch_p95_deg",
    "avg_hitch_max_deg", "avg_hitch_final_deg", "avg_hitch_rate_rms", "avg_solve_ms",
    "start_index", "avg_vx_kph", "avg_distance_survived_m", "p95_solve_ms",
]


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def agg_header():
    return (
        f"  {'ctl':<6} {'index':>6} {'v_tgt':>6}  {'avg_v':>7}  {'avg_it':>7}  "
        f"{'succ':>6}  {'stall':>6}  {'fail':>6}  {'dist':>8}  {'<v_lo':>6}  "
        f"{'h_rms':>7}  {'h_p95':>7}  {'h_max':>7}  {'ms_mean':>8} {'ms_p95':>8}"
    )


def agg_line(a):
    return (
        f"  {a['controller']:<6} {a['start_index']:>6} {a['v_kph']:>6}  {a['avg_v_kph']:>7.1f}  {a['avg_iters']:>7.0f}  "
        f"{a['success_rate']:>6.0%}  {a['stall_rate']:>6.0%}  {a['fail_rate']:>6.0%}  "
        f"{a['avg_distance_survived_m']:>8.1f}  {a['avg_frac_below_v']:>6.1%}  "
        f"{a['avg_hitch_rms_deg']:>6.2f}\u00b0  {a['avg_hitch_p95_deg']:>6.2f}\u00b0  "
        f"{a['avg_hitch_max_deg']:>6.2f}\u00b0  {a['avg_solve_ms']:>8.2f} {a['p95_solve_ms']:>8.2f}"
    )


# Sweep


def main():
    track = TrackModel.from_config(scenario.track)
    if not START_INDICES or any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < len(track.centerline)
        for index in START_INDICES
    ):
        raise ValueError(f"START_INDICES must contain indices in [0, {len(track.centerline) - 1}]")
    if len(set(START_INDICES)) != len(START_INDICES):
        raise ValueError("START_INDICES must be distinct")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "controllers": CONTROLLERS,
                "vels_kph": VELS_KPH,
                "start_indices": START_INDICES,
                "seeds": SEEDS,
                "max_steps": MAX_STEPS,
                "cutoff": CUTOFF,
                "metric_definitions": {
                    "solve_ms": "All controller calls, including first-call JIT compilation and failed solves; excludes construction and env.step",
                    "aggregate_timing": "Mean and p95 pooled over calls in non-interrupted episodes for each controller/start_index/velocity",
                    "distance_survived_m": "Nonnegative net track progress in the requested direction, from controller takeover after warmup",
                    "velocities": "Raw signed truck-frame [vx, vy] in m/s; speeds is magnitude in m/s",
                    "hitches": "Raw signed trailer yaw minus truck yaw, wrapped, in radians; summary magnitudes in degrees",
                },
                "stall": {
                    "vel": STALL_VEL,
                    "window": STALL_WINDOW,
                    "arc": STALL_ARC,
                    "grace": STALL_GRACE,
                },
                "mu": MU,
                "track_width": TRACK_WIDTH,
                "friction_csv": FRICTION_CSV,
                "dt": DT,
                "checkpoint": CKPT_PTH,
                "norm_stats": JSON_PTH,
                "data_version": spec.data_version,
                "H": HISTORY,
                "fwd_weights": FWD_WEIGHTS,
                "rev_weights": REV_WEIGHTS,
                "fwd_mppi": {k: str(v) for k, v in FWD_MPPI.items()},
                "rev_mppi": {k: str(v) for k, v in REV_MPPI.items()},
                "per_vel_override": {str(k): str(v) for k, v in PER_VEL_OVERRIDE.items()},
                "prior_override": {k: str(v) for k, v in PRIOR_OVERRIDE.items()},
                "ipopt_horizon": IPOPT_HORIZON,
                "ipopt_settings": IPOPT_SETTINGS,
                "pure_pursuit_settings": PURE_PURSUIT_SETTINGS,
                "lqr_pid_settings": LQR_PID_SETTINGS,
                "lqr_pid_model": "fiala_linearized",
            },
            f,
            indent=2,
        )

    env = BeamNGTrailerEnv(config=scenario)

    ep_rows, agg_rows = [], []
    aborted = False

    log(f"sweep -> {out_dir}")
    log(
        f"{len(CONTROLLERS)} controllers x {len(START_INDICES)} locations x {len(VELS_KPH)} velocities x "
        f"{len(SEEDS)} seeds, max {MAX_STEPS} steps"
    )
    log()

    try:
        for kind in CONTROLLERS:
            for v_kph in VELS_KPH:
                progress(f"[{kind:<5} v={v_kph:>4} kph]  compiling...")
                mpc = make_mpc(v_kph, kind)
                clear_progress()

                per_start = {index: [] for index in START_INDICES}
                for run_index, (start_index, seed) in enumerate(product(START_INDICES, SEEDS)):
                    ep = run_episode(env, mpc, v_kph, seed, kind, start_index)
                    ep["run_index"] = run_index
                    s = summarize(ep)
                    ep_rows.append(s)
                    if s["outcome"] != "interrupted":
                        per_start[start_index].append(s)

                    np.savez_compressed(
                        out_dir / f"raw_{kind}_idx{start_index}_v{v_kph}_s{seed}_r{run_index}.npz",
                        controller=kind,
                        start_index=start_index,
                        v_kph=v_kph,
                        seed=seed,
                        run_index=run_index,
                        outcome=ep["outcome"],
                        iters=ep["iters"],
                        speeds=ep["speeds"],
                        velocities=ep["velocities"],
                        distance_survived_m=s["distance_survived_m"],
                        hitches=ep["hitches"],
                        hitch_rates=ep["hitch_rates"],
                        lat_errs=ep["lat_errs"],
                        cum_dist=ep["cum_dist"],
                        solve_ms=ep["solve_ms"],
                        us=ep["us"],
                    )

                    log(
                        f"  {kind:<5} idx={start_index} v={v_kph:>4} seed {seed} | {s['outcome']:<11} "
                        f"iters {s['iters']:>4}  |v| {s['avg_v_kph']:>6.1f} kph  "
                        f"dist {s['distance_survived_m']:>7.1f} m  "
                        f"hitch rms {s['hitch_rms_deg']:>5.2f}\u00b0 max {s['hitch_max_deg']:>5.2f}\u00b0 "
                        f"fin {s['hitch_final_deg']:>5.2f}\u00b0  "
                        f"below_v {s['frac_below_v']:>5.1%}  solve mean {s['solve_ms']:>5.1f} p95 {s['solve_p95_ms']:>5.1f} ms"
                    )

                    # Save after each episode.
                    partial = [aggregate(rows) for rows in per_start.values() if rows]
                    write_csv(out_dir / "episodes.csv", EP_COLS, ep_rows)
                    write_csv(
                        out_dir / "agg.csv",
                        AGG_COLS,
                        agg_rows + partial,
                    )

                for rows in per_start.values():
                    a = aggregate(rows)
                    if a is not None:
                        agg_rows.append(a)
                        log(agg_header())
                        log(agg_line(a))
                log()
                write_csv(out_dir / "agg.csv", AGG_COLS, agg_rows)

    except KeyboardInterrupt:
        aborted = True
        clear_progress()
        log("\n  sweep aborted")
    finally:
        env.close()
        # Include the partial cell when interrupted.
        groups = {}
        for row in ep_rows:
            if row["outcome"] != "interrupted":
                key = (row["controller"], row["start_index"], row["v_kph"])
                groups.setdefault(key, []).append(row)
        agg_rows = [aggregate(rows) for rows in groups.values()]
        write_csv(out_dir / "episodes.csv", EP_COLS, ep_rows)
        write_csv(out_dir / "agg.csv", AGG_COLS, agg_rows)

    log()
    log("=" * 88)
    log(f"  {'SWEEP' if not aborted else 'SWEEP (partial)'}   mu={MU}  n={len(SEEDS)} seeds")
    log("=" * 88)
    log(agg_header())
    last = None
    for a in agg_rows:
        if last is not None and a["controller"] != last:
            log()
        log(agg_line(a))
        last = a["controller"]
    log("=" * 88)
    log(f"  {out_dir}")


if __name__ == "__main__":
    main()
