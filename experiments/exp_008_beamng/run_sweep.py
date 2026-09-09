"""
Velocity sweep benchmark for the BeamNG MPPI stack.

Sweeps start locations x velocity targets x seeds for two rollout models:
    "model"  learned dynamics, 13-wide history window   (run_model.py)
    "prior"  Fiala bicycle, 6-wide instantaneous state  (test_driver.py)

Both run the same cost weights and the same MPPI hyperparameters, so the only
difference between them is the rollout model. Start locations exercise different
track geometry; seeds vary the MPPI sampling key at each location.

Outcomes are mutually exclusive and partition the runs:
    completed  reached MAX_STEPS without stalling      -> success_rate
    stalled    sat below STALL_VEL with no progress    -> stall_rate
    jackknife  |hitch| >= max_hitch                    -> fail_rate
    offtrack   |lateral_error| > width/2               -> fail_rate
    nan        control went NaN                        -> fail_rate

Ctrl-C once  : abort the current episode, mark it `interrupted`, continue.
Ctrl-C twice : (within DOUBLE_TAP_S) abort the whole sweep cleanly.
Interrupted episodes are excluded from that cell's aggregate.

Place at experiments/exp_008_beamng/run_sweep.py and run from the repo root:
    python -m experiments.exp_008_beamng.run_sweep
"""

import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import csv
import json
import logging
import time
from dataclasses import astuple
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import absl.logging

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
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import fiala_dyn
from src.learning.models.beamng_model_spec import STATE_FS, IN_COLS
from src.dynamics.trailer.beamng_dynamics import gen_util_funs as res_util
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as prior_util

jnp.set_printoptions(precision=2, suppress=True)


# ----------------------------------------------------------------------------
# Config -- the only region you should need to edit
# ----------------------------------------------------------------------------

CONTROLLERS = ["model"]
VELS_KPH = [-30, -50, 80]

LOCS = [0, 800, 1500]
SEEDS = [i for i in range(3)]

MAX_STEPS = 1500
CUTOFF = 100  # steps discarded from the head before averaging, as in the other benchmarks

# Stall detector. Fires only when BOTH conditions hold over the trailing window:
# speed alone false-positives on in-place oscillation, arc alone false-positives
# on a genuine low-speed direction change.
STALL_VEL = 0.5  # m/s -- matches the vx_safe clamp in fiala_dyn
STALL_WINDOW = 40  # steps (2.0 s at dt=0.05)
STALL_ARC = 1.0  # m of track progress over the window
STALL_GRACE = 100  # steps before the detector arms (every run starts at rest)

DOUBLE_TAP_S = 2.0

MU = 1.0
TRACK_WIDTH = 15
FRICTION_CSV = None
DT = 0.05

NPZ_SAVE_HEAD = "data_proc_test9-new"
JSON_PTH = f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"
CKPT_PTH = "src/learning/models/trained/beamng-l4-128-test9-new_best"

OUT_ROOT = Path("./experiments/exp_008_beamng/sweep_out")

# Cost weights, verbatim from run_model.py. Shared by both controllers -- this is
# the controlled comparison, the cost must be identical or the rollout model is
# not the only thing that differs.
FWD_WEIGHTS = {
    "p_weight": 2e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 1e2,
    "reverse": False,
}
REV_WEIGHTS = {
    "p_weight": 5e1,
    "p_slow_weight": 1e0,
    "c_weight": 5e1,
    "a_weight": 2e2,
    "reverse": False,
}


# MPPI hyperparameters, keyed by sign(v_target). Verbatim from run_model.py,
# shared by both controllers.
FWD_MPPI = {
    "cv": jnp.diag(jnp.array([7e-2, 0.2])),
    "inverse_temp": 100,
    "K": 500,
    "step": 0.05,
    "T": 80,
    "alpha": 0.01,
    "gamma": 0.0,
}
REV_MPPI = {
    "cv": jnp.diag(jnp.array([1e-2, 0.2])),
    "inverse_temp": 500,
    "K": 500,
    "step": 0.05,
    "T": 75,
    "alpha": 0.01,
    "gamma": 0.0,
}

# Merged on top of the sign-keyed dicts above, in this order:
#   sign-keyed  <-  PER_VEL_OVERRIDE[kph]  <-  PRIOR_OVERRIDE (prior only)
PER_VEL_OVERRIDE: dict[int, dict] = {}

# Empty = the prior runs the model's tuning, per the controlled-comparison intent.
# test_driver.py's own tuning, if you want to fall back to it:
#   fwd: cv=diag([3e-3, 0.2]), inverse_temp=0.5, K=500,  T=80, alpha=0.05
#   rev: cv=diag([1e-2, 0.2]), inverse_temp=0.5, K=2000, T=55, alpha=0.01
# Note inverse_temp is a temperature (exp(-(S-S_min)/lambda)) and the prior's cost
# landscape has a different spread, so 150/100 may be far off for it -- if the
# prior collapses to argmin or to nominal, this is the first knob.
PRIOR_OVERRIDE: dict = {}


# ----------------------------------------------------------------------------
# One-time setup
# ----------------------------------------------------------------------------

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

model = TrailerModel(spec.H * len(IN_COLS), 6)
_, _mstate = nnx.split(model)
_ckpt = ocp.StandardCheckpointer()
nnx.update(model, _ckpt.restore(Path.cwd() / CKPT_PTH, _mstate))

# NOTE: `pred += prior(...)` is commented out in beamng_dynamics, so the learned
# model is pure-learned and fiala_dyn is inert there. Passed only for the signature.
KIN_FN = fiala_dyn


def log(msg=""):
    print(msg, flush=True)


def progress(msg):
    print(f"\r{msg:<124}", end="", flush=True)


def clear_progress():
    print("\r" + " " * 124 + "\r", end="", flush=True)


# ----------------------------------------------------------------------------
# Controller construction
# ----------------------------------------------------------------------------


def make_mpc(v_kph, kind):
    """
    One controller per (kind, velocity), reused across seeds. Rebuilding per seed
    would hand JAX a fresh cost closure -- a static arg -- and retrace every episode.
    """
    v_target = v_kph / 3.6
    weights = dict(FWD_WEIGHTS if v_target > 0 else REV_WEIGHTS)
    weights["v_target"] = v_target

    cfg = dict(FWD_MPPI if v_target > 0 else REV_MPPI)
    cfg.update(PER_VEL_OVERRIDE.get(v_kph, {}))
    if kind == "prior":
        cfg.update(PRIOR_OVERRIDE)
    cv = cfg.pop("cv")

    if kind == "model":
        dynamics, cost, bound, _ = res_util(scenario, spec, KIN_FN, model, norm_stats, **weights)
        x_d, hist = ROW_W, HISTORY
    else:
        dynamics, cost, bound, _ = prior_util(scenario, s_weight=0, **weights)
        x_d, hist = 6, None

    # The sweep never consumes candidate rollout histories, so use the lean MPPI
    # implementation. The debug implementation materializes K x T state/control
    # histories and creates substantial avoidable GPU pressure.
    gamma = cfg["gamma"]
    mpc = MPPI_Jax(x_d, 2, dynamics, None, cost, bound, cv, history=hist, **cfg)
    # MPPI_Jax currently derives gamma in __init__; preserve the explicit sweep
    # setting so this remains behaviorally matched to run_model's debug controller.
    mpc.gamma = gamma
    return mpc


# ----------------------------------------------------------------------------
# Episode
# ----------------------------------------------------------------------------

_last_interrupt = [0.0]


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_episode(mpc, v_kph, seed, kind, loc):
    """
    One episode. Returns per-step arrays plus the outcome label. Raises
    KeyboardInterrupt only on a double tap; a single tap aborts the episode
    and returns outcome="interrupted".
    """

    env = BeamNGTrailerEnv(config=scenario, use_custom_mu=False, spidx=loc)


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
        """Prior input: instantaneous state + mu + arclen (test_driver.py)."""
        s = env.unwrapped._state
        return jnp.array(
            [
                *astuple(s)[:-2],
                track.find_mu(s.x, s.y),
                track._arc_samples[env.unwrapped._last_index],
            ]
        )

    env.reset()
    mpc.reset()  # clears last_trajectory, else seeds chain off each other's warm start
    mpc.key = jax.random.key(seed)  # MPPI_Jax hardcodes key(0); reset() does not touch it

    env.step(jnp.zeros(2))

    # Warmup. The learned model panics on a zero/default window, so drive H steps
    # open-loop. The prior does not need it, but gets the same H steps so both
    # controllers take over from the same physical state.
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

    speeds, hitches, hitch_rates = [], [], []
    lat_errs, cum_dist, solve_ms, us = [], [], [], []
    dist = 0.0
    prev_arc = None
    outcome = "completed"
    iters = 0

    try:
        for i in range(MAX_STEPS):
            t0 = time.perf_counter()
            u = mpc.run_mpc(history if kind == "model" else mpc_state())
            u.block_until_ready()
            solve_ms.append((time.perf_counter() - t0) * 1e3)

            if bool(jnp.any(jnp.isnan(u))):
                outcome = "nan"
                break

            # Frame: the Fiala paths work in the internal steering frame, BeamNG's
            # command frame is inverted. run_model.py passes u straight through
            # because the learned model was trained on stored BeamNG commands;
            # test_driver.py negates. Do not remove.
            action = jnp.array([-u[0], u[1]]) if kind == "prior" else u

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

            # Independent projection for metrics -- do NOT reuse the quantised
            # _arc_samples value above, that one exists to match the training row.
            proj, _ = track.project(state.x, state.y, env.unwrapped._last_index)
            if prev_arc is not None:
                dist += arc_delta(prev_arc, proj.arc_length)
            prev_arc = proj.arc_length

            speed = float(np.hypot(state.vx, state.vy))
            hitch = float(wrap_angle(state.yaw_trailer - state.yaw_truck))
            hitch_rate = float(state.yaw_trailer_rate - state.yaw_truck_rate)
            speeds.append(speed)
            hitches.append(hitch)
            hitch_rates.append(hitch_rate)
            lat_errs.append(float(proj.lateral_error))
            cum_dist.append(dist)
            us.append([float(action[0]), float(action[1])])

            progress(
                f"[{kind:<5} loc={loc:>4} v={v_kph:>4} kph  seed {seed}]  step {iters:>4}/{MAX_STEPS}  "
                f"|v| {speed * 3.6:>6.1f} kph  hitch {hitch * 180.0 / np.pi:>6.2f}\u00b0  "
                f"lat {proj.lateral_error:>6.2f}  dist {dist:>7.1f} m"
            )

            # Attribute the termination ourselves; env.step collapses both into one bool.
            if abs(hitch) >= max_hitch:
                outcome = "jackknife"
                break
            if track.out_of_bounds(proj.lateral_error):
                outcome = "offtrack"
                break

            if i >= STALL_GRACE and len(speeds) > STALL_WINDOW:
                slow = np.mean(speeds[-STALL_WINDOW:]) < STALL_VEL
                stuck = (cum_dist[-1] - cum_dist[-STALL_WINDOW]) < STALL_ARC
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
    env.close()

    return {
        "controller": kind,
        "loc": loc,
        "v_kph": v_kph,
        "seed": seed,
        "outcome": outcome,
        "iters": iters,
        "speeds": np.asarray(speeds),
        "hitches": np.asarray(hitches),
        "hitch_rates": np.asarray(hitch_rates),
        "lat_errs": np.asarray(lat_errs),
        "cum_dist": np.asarray(cum_dist),
        "solve_ms": np.asarray(solve_ms),
        "us": np.asarray(us),
    }


DEG = 180.0 / np.pi


def summarize(ep):
    """
    Hitch stats are reported in degrees. `hitch_max_deg` is over the whole run
    (worst excursion); everything else is over the post-CUTOFF tail, matching
    avg_v_kph. Note hitch_max saturates at max_hitch for any jackknife run by
    definition, so it only discriminates among runs that survived -- the rms and
    p95 numbers are the ones to compare across cells.
    """
    sp = ep["speeds"]
    h = np.abs(ep["hitches"])
    hd = ep["hitch_rates"]
    tail = sp[CUTOFF:] if sp.size > CUTOFF else sp
    h_t = h[CUTOFF:] if h.size > CUTOFF else h
    hd_t = hd[CUTOFF:] if hd.size > CUTOFF else hd
    return {
        "controller": ep["controller"],
        "loc": ep["loc"],
        "v_kph": ep["v_kph"],
        "seed": ep["seed"],
        "outcome": ep["outcome"],
        "iters": ep["iters"],
        "avg_v_kph": float(np.mean(tail) * 3.6) if tail.size else 0.0,
        "distance_m": float(ep["cum_dist"][-1]) if ep["cum_dist"].size else 0.0,
        "frac_below_v": float(np.mean(tail < STALL_VEL)) if tail.size else 1.0,
        "hitch_rms_deg": float(np.sqrt(np.mean(h_t**2)) * DEG) if h_t.size else 0.0,
        "hitch_mean_deg": float(np.mean(h_t) * DEG) if h_t.size else 0.0,
        "hitch_p95_deg": float(np.percentile(h_t, 95) * DEG) if h_t.size else 0.0,
        "hitch_max_deg": float(np.max(h) * DEG) if h.size else 0.0,
        "hitch_final_deg": float(h[-1] * DEG) if h.size else 0.0,
        "hitch_rate_rms": float(np.sqrt(np.mean(hd_t**2))) if hd_t.size else 0.0,
        "mean_lat_err": float(np.mean(np.abs(ep["lat_errs"]))) if ep["lat_errs"].size else 0.0,
        "solve_ms": float(np.mean(ep["solve_ms"])) if ep["solve_ms"].size else 0.0,
    }


def aggregate(rows):
    """Per-seed summaries for one (controller, location, velocity) cell."""
    n = len(rows)
    if n == 0:
        return None
    outc = [r["outcome"] for r in rows]
    return {
        "controller": rows[0]["controller"],
        "loc": rows[0]["loc"],
        "v_kph": rows[0]["v_kph"],
        "n": n,
        "avg_v_kph": float(np.mean([r["avg_v_kph"] for r in rows])),
        "std_v_kph": float(np.std([r["avg_v_kph"] for r in rows])),
        "avg_iters": float(np.mean([r["iters"] for r in rows])),
        "std_iters": float(np.std([r["iters"] for r in rows])),
        "success_rate": outc.count("completed") / n,
        "stall_rate": outc.count("stalled") / n,
        "fail_rate": sum(o in ("jackknife", "offtrack", "nan") for o in outc) / n,
        "avg_distance_m": float(np.mean([r["distance_m"] for r in rows])),
        "avg_frac_below_v": float(np.mean([r["frac_below_v"] for r in rows])),
        "avg_hitch_rms_deg": float(np.mean([r["hitch_rms_deg"] for r in rows])),
        "std_hitch_rms_deg": float(np.std([r["hitch_rms_deg"] for r in rows])),
        "avg_hitch_mean_deg": float(np.mean([r["hitch_mean_deg"] for r in rows])),
        "avg_hitch_p95_deg": float(np.mean([r["hitch_p95_deg"] for r in rows])),
        "avg_hitch_max_deg": float(np.mean([r["hitch_max_deg"] for r in rows])),
        "avg_hitch_final_deg": float(np.mean([r["hitch_final_deg"] for r in rows])),
        "avg_hitch_rate_rms": float(np.mean([r["hitch_rate_rms"] for r in rows])),
        "avg_solve_ms": float(np.mean([r["solve_ms"] for r in rows])),
    }


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

EP_COLS = [
    "controller", "loc", "v_kph", "seed", "outcome", "iters", "avg_v_kph", "distance_m",
    "frac_below_v", "hitch_rms_deg", "hitch_mean_deg", "hitch_p95_deg",
    "hitch_max_deg", "hitch_final_deg", "hitch_rate_rms", "mean_lat_err", "solve_ms",
]
AGG_COLS = [
    "controller", "loc", "v_kph", "n", "avg_v_kph", "std_v_kph", "avg_iters", "std_iters",
    "success_rate", "stall_rate", "fail_rate", "avg_distance_m", "avg_frac_below_v",
    "avg_hitch_rms_deg", "std_hitch_rms_deg", "avg_hitch_mean_deg", "avg_hitch_p95_deg",
    "avg_hitch_max_deg", "avg_hitch_final_deg", "avg_hitch_rate_rms", "avg_solve_ms",
]


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def agg_header():
    return (
        f"  {'ctl':<6} {'loc':>5} {'v_tgt':>6}  {'avg_v':>7}  {'avg_it':>7}  "
        f"{'succ':>6}  {'stall':>6}  {'fail':>6}  {'dist':>8}  {'<v_lo':>6}  "
        f"{'h_rms':>7}  {'h_p95':>7}  {'h_max':>7}"
    )


def agg_line(a):
    return (
        f"  {a['controller']:<6} {a['loc']:>5} {a['v_kph']:>6}  {a['avg_v_kph']:>7.1f}  {a['avg_iters']:>7.0f}  "
        f"{a['success_rate']:>6.0%}  {a['stall_rate']:>6.0%}  {a['fail_rate']:>6.0%}  "
        f"{a['avg_distance_m']:>8.1f}  {a['avg_frac_below_v']:>6.1%}  "
        f"{a['avg_hitch_rms_deg']:>6.2f}\u00b0  {a['avg_hitch_p95_deg']:>6.2f}\u00b0  "
        f"{a['avg_hitch_max_deg']:>6.2f}\u00b0"
    )


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "controllers": CONTROLLERS,
                "vels_kph": VELS_KPH,
                "locs": LOCS,
                "seeds": SEEDS,
                "max_steps": MAX_STEPS,
                "cutoff": CUTOFF,
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
            },
            f,
            indent=2,
        )

    # env = BeamNGTrailerEnv(config=scenario, use_custom_mu=False)

    ep_rows, agg_rows = [], []
    aborted = False

    log(f"sweep -> {out_dir}")
    log(
        f"{len(CONTROLLERS)} controllers x {len(LOCS)} locations x {len(VELS_KPH)} velocities x "
        f"{len(SEEDS)} seeds, max {MAX_STEPS} steps"
    )
    log()

    try:
        for kind in CONTROLLERS:
            for v_kph in VELS_KPH:
                progress(f"[{kind:<5} v={v_kph:>4} kph]  compiling...")
                mpc = make_mpc(v_kph, kind)
                clear_progress()

                # Location changes only the BeamNG initial condition; reuse the
                # same controller closure and compiled executable at every start.
                for idx in LOCS:
                    per_v = []
                    for seed in SEEDS:
                        ep = run_episode(mpc, v_kph, seed, kind, idx)
                        s = summarize(ep)
                        ep_rows.append(s)
                        if s["outcome"] != "interrupted":
                            per_v.append(s)

                        np.savez_compressed(
                            out_dir / f"raw_{kind}_loc{idx}_v{v_kph}_s{seed}.npz",
                            loc=ep["loc"],
                            outcome=ep["outcome"],
                            iters=ep["iters"],
                            speeds=ep["speeds"],
                            hitches=ep["hitches"],
                            hitch_rates=ep["hitch_rates"],
                            lat_errs=ep["lat_errs"],
                            cum_dist=ep["cum_dist"],
                            solve_ms=ep["solve_ms"],
                            us=ep["us"],
                        )

                        log(
                            f"  {kind:<5} loc={idx:>4}  v={v_kph:>4}  seed {seed}  |  {s['outcome']:<11}  "
                            f"iters {s['iters']:>4}  |v| {s['avg_v_kph']:>6.1f} kph  "
                            f"dist {s['distance_m']:>7.1f} m  "
                            f"hitch rms {s['hitch_rms_deg']:>5.2f}\u00b0 max {s['hitch_max_deg']:>5.2f}\u00b0 "
                            f"fin {s['hitch_final_deg']:>5.2f}\u00b0  "
                            f"below_v {s['frac_below_v']:>5.1%}  solve {s['solve_ms']:>5.1f} ms"
                        )

                        # Flush after every episode -- a hard kill loses one run at most.
                        partial = aggregate(per_v)
                        write_csv(out_dir / "episodes.csv", EP_COLS, ep_rows)
                        write_csv(
                            out_dir / "agg.csv",
                            AGG_COLS,
                            agg_rows + ([partial] if partial else []),
                        )

                    a = aggregate(per_v)
                    if a is not None:
                        agg_rows.append(a)
                        log(agg_header())
                        log(agg_line(a))
                    log()
                    write_csv(out_dir / "agg.csv", AGG_COLS, agg_rows)

                # Each velocity creates distinct static dynamics/cost closures.
                # Release their compiled executables before constructing the next
                # cell instead of accumulating all variants in device memory.
                del mpc
                jax.clear_caches()

    except KeyboardInterrupt:
        aborted = True
        clear_progress()
        log("\n  sweep aborted")
    finally:
        # env.close()
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
