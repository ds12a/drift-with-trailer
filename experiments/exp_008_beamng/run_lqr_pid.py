"""
LQR/PID BeamNG test.
python -m experiments.exp_008_beamng.run_lqr_pid
"""

import logging
import time
from dataclasses import astuple

import numpy as np

from src.controllers.lqr_pid import LQR_PID
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


V_TARGET = -40 / 3.6
MAX_STEPS = 1000
DT = 0.05

scenario = BeamNGTrailerEnvConfig(
    ".",
    TrackConfig(mu=1.0, width=15),
    bng_pickup_trailer_cfg,
    SimulationConfig(dt=DT),
)
scenario.track.friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

LQR_PID_SETTINGS = {
    "q": [0.1, 1.0, 5.0, 0.1, 1.0, 1.0],  # lateral, heading, hitch, vy, r_truck, r_trailer
    "r": 10.0,  # steering weight, rad
    "design_mu": 1.0,
    "min_design_speed": 0.5,  # m/s
    "curvature_spacing": 3.0,  # half-window, m
    "heading_smoothing": 5.0,  # half-window, m
    "kp": 1.5,
    "ki": 0.3,
    "kd": 0.1,
    "integral_limit": 10.0,  # m
    "derivative_tau": 0.2,  # s
}


def make_mpc(config=scenario, v_target=V_TARGET, settings=None):
    if settings is None:
        settings = LQR_PID_SETTINGS
    return LQR_PID(config, v_target, **settings)


def run_mpc():
    logging.getLogger("beamngpy").setLevel(logging.WARNING)
    logging.getLogger("beamngpy").propagate = False

    mpc = make_mpc()
    env = BeamNGTrailerEnv(config=scenario)
    speeds, solve_times = [], []

    try:
        env.reset()
        _, _, terminated, _, _ = env.step(np.zeros(2))

        for i in range(MAX_STEPS):
            if terminated:
                break

            state: VehicleState = env.unwrapped._state
            state_vector = np.array(astuple(state)[:-2], dtype=float)
            start = time.perf_counter()
            u = mpc.run_mpc(state_vector)
            elapsed = time.perf_counter() - start

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
                f"|v|: {speeds[-1]:<7.3f}"
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
