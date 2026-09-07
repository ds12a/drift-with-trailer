"""
IPOPT BeamNG test.
python -m experiments.exp_008_beamng.run_ipopt
"""

import logging
import time
from dataclasses import astuple

import numpy as np

from src.controllers.mpc.ipopt_cartpole import MPC
from src.dynamics.trailer.trailer_bicycle_fiala_casadi import gen_util_funs
from src.simulation.beamng_trailer_env import (
    BeamNGTrailerEnv,
    VehicleState,
    bng_pickup_trailer_cfg,
)
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    SimulationConfig,
    TrackConfig,
)


V_TARGET = -60 / 3.6
MAX_STEPS = 1000
# 3 s horizon
HORIZON = 60
DT = 0.05

scenario = BeamNGTrailerEnvConfig(
    ".",
    TrackConfig(mu=1.0, width=15),
    bng_pickup_trailer_cfg,
    SimulationConfig(dt=DT),
)
scenario.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

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

IPOPT_SETTINGS = {
    "record_time": True,
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
    # Use the current iterate if the iteration limit is reached.
    "ipopt.max_iter": 150,
    "detect_simple_bounds": True,
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.nlp_scaling_method": "gradient-based",
    # Relax active bounds; clip commands before env.step.
    "ipopt.bound_relax_factor": 1e-4,
    # Quasi-Newton Hessian for piecewise Fiala saturation.
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.tol": 1e-2,
    "ipopt.acceptable_tol": 25.0,
    "ipopt.acceptable_iter": 1,
    "ipopt.derivative_test": "none",
}


def make_mpc(config=scenario, v_target=V_TARGET, horizon=HORIZON, weights=None):
    if weights is None:
        weights = FWD_WEIGHTS if v_target > 0 else REV_WEIGHTS
    weights = dict(weights)
    weights["v_target"] = v_target
    dynamics, cost, constraints = gen_util_funs(config, s_weight=0, **weights)
    mpc = MPC(
        10,
        2,
        dynamics,
        None,
        constraints,
        None,
        cost,
        IPOPT_SETTINGS,
        step=config.simulation.dt,
        n=horizon,
    )
    accel_guess = 0.35 if v_target > 0 else -0.35
    mpc.opti.set_initial(mpc.u_sym, np.tile([0.0, accel_guess], (horizon, 1)))
    return mpc


def mpc_state(env):
    state = env.unwrapped._state
    return np.array(
        [
            *astuple(state)[:-2],
            env.unwrapped.track.find_mu(state.x, state.y),
            env.unwrapped.track._arc_samples[env.unwrapped._last_index],
        ],
        dtype=float,
    )


def solve_or_use_iterate(mpc, state_vector):
    """Use the current finite iterate on Maximum_Iterations_Exceeded."""
    try:
        return mpc.run_mpc(state_vector, verbose=False, warm_start=True), False
    except RuntimeError:
        stats = mpc.opti.stats()
        if stats.get("return_status") != "Maximum_Iterations_Exceeded":
            raise

        trajectory = np.asarray(mpc.opti.debug.value(mpc.u_sym), dtype=float)
        if trajectory.shape != (mpc.n, mpc.u_d) or not np.all(np.isfinite(trajectory)):
            raise

        # Primal warm start for the next solve.
        mpc.last_trajectory = trajectory
        return trajectory[0], True


def run_mpc():
    logging.getLogger("beamngpy").setLevel(logging.WARNING)
    logging.getLogger("beamngpy").propagate = False

    env = BeamNGTrailerEnv(config=scenario)
    mpc = make_mpc()
    speeds, solve_times = [], []

    env.reset()
    _, _, terminated, _, _ = env.step(np.zeros(2))

    try:
        for i in range(MAX_STEPS):
            if terminated:
                break

            state: VehicleState = env.unwrapped._state
            state_vector = mpc_state(env)
            start = time.perf_counter()
            u, used_iterate = solve_or_use_iterate(mpc, state_vector)
            elapsed = time.perf_counter() - start

            # BeamNG steering sign
            action = np.clip(np.array([-u[0], u[1]]), -1.0, 1.0)
            _, _, terminated, _, _ = env.step(action)

            speeds.append(np.hypot(state.vx, state.vy))
            solve_times.append(elapsed)
            print(
                f"Step: {i:<5d} | "
                f"Time: {elapsed:<7.3f} | "
                f"u: {u[0]:<7.3f} {u[1]:<7.3f} | "
                f"vx: {state.vx:<7.3f} | "
                f"vy: {state.vy:<7.3f} | "
                f"|v|: {speeds[-1]:<7.3f} | "
                f"mu: {state_vector[-2]:<7.3f} | "
                f"status: {'iter-cap' if used_iterate else 'solved'}"
            )
    finally:
        env.close()

    cutoff = min(100, len(speeds))
    avg_speed = np.mean(speeds[cutoff:]) * 3.6 if len(speeds) > cutoff else float("nan")
    avg_solve_ms = np.mean(solve_times) * 1e3 if solve_times else float("nan")
    print(
        f"Iters: {len(speeds)}, "
        f"Target speed: {V_TARGET * 3.6}, "
        f"Avg speed: {avg_speed}, "
        f"Avg solve: {avg_solve_ms} ms"
    )


if __name__ == "__main__":
    run_mpc()
