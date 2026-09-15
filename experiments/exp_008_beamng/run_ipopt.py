"""IPOPT controller construction and finite-iterate recovery for BeamNG."""

import numpy as np

from src.controllers.mpc.ipopt_cartpole import MPC
from src.dynamics.trailer.trailer_bicycle_fiala_casadi import gen_util_funs


HORIZON = 60
IPOPT_SETTINGS = {
    "record_time": True,
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "yes",
    "ipopt.max_iter": 150,
    "detect_simple_bounds": True,
    "ipopt.linear_solver": "mumps",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.nlp_scaling_method": "gradient-based",
    "ipopt.bound_relax_factor": 1e-4,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.tol": 1e-2,
    "ipopt.acceptable_tol": 25.0,
    "ipopt.acceptable_iter": 1,
    "ipopt.derivative_test": "none",
}


def make_mpc(config, v_target, horizon=HORIZON, weights=None):
    weights = dict(weights or {})
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


def solve_or_use_iterate(mpc, state_vector):
    """Use a finite bounded iterate when IPOPT stops before convergence."""
    previous = None
    if mpc.last_trajectory is not None:
        previous = np.vstack([mpc.last_trajectory[1:], mpc.last_trajectory[-1]])
    try:
        control = mpc.run_mpc(state_vector, verbose=False, warm_start=True)
        mpc.last_solver_status = mpc.opti.stats().get("return_status")
        return control, False
    except RuntimeError as solver_error:
        try:
            mpc.last_solver_status = mpc.opti.stats().get("return_status")
        except RuntimeError:
            mpc.last_solver_status = "unavailable"
        try:
            trajectory = np.asarray(mpc.opti.debug.value(mpc.u_sym), dtype=float)
        except RuntimeError:
            trajectory = previous
        if trajectory is None or trajectory.shape != (mpc.n, mpc.u_d):
            raise solver_error
        if not np.all(np.isfinite(trajectory)):
            trajectory = previous
        if trajectory is None or not np.all(np.isfinite(trajectory)):
            raise solver_error
        trajectory = np.clip(trajectory, -1.0, 1.0)
        mpc.last_trajectory = trajectory
        mpc.lt_lam_g = None
        return trajectory[0], True
