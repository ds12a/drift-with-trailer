"""LQR/PID controller construction for the BeamNG sweep."""

from src.controllers.lqr_pid import LQR_PID


LQR_PID_SETTINGS = {
    "q": [0.1, 1.0, 5.0, 0.1, 1.0, 1.0],
    "r": 10.0,
    "design_mu": 1.0,
    "min_design_speed": 0.5,
    "gain_speed_step": 1.0,
    "kp": 1.5,
    "ki": 0.3,
    "kd": 0.1,
    "integral_limit": 10.0,
    "derivative_tau": 0.2,
}


def make_mpc(config, v_target, settings=None):
    return LQR_PID(config, v_target, **(settings or LQR_PID_SETTINGS))
