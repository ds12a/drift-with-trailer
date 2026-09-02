"""
Closed-loop RBR on the analytic trailer plant.

    python3 -m experiments.exp_mppi_vibe.run_trailer_rbr

The feasibility indicator comes from `gen_util_funs(..., with_violation=True)`,
so RBR prunes on the exact same constraint the cost penalises: track half-width
excursion plus hitch beyond max_hitch. No CBF is trained or approximated, which
makes this the apples-to-apples baseline for cost-directed selection rather than
a reimplementation of the paper's learned certificate.

The number to watch in the diag CSV is `n_all_unsafe`: horizon steps at which
every particle violated and RBR had nothing to rewire onto. Where that is
nonzero the mechanism is inert, and that is exactly the regime soft weighting is
supposed to cover.
"""

import jax.numpy as jnp

from experiments.exp_mppi_vibe.utils.trailer_driver import run_mpc

# Fwd Args
# ctl_args = (
#     jnp.diag(jnp.array([3e-3, 0.2])),
# )
# ctl_kwargs = {
#     "inverse_temp": 1,
#     "K": 500,
#     "step": 0.05,
#     "T": 80,
#     "alpha": 0.05,
#     "gamma": 0.0,
# }
# cost_kwargs = {
#     "reverse": False,
#     "v_target": 25,
#     "p_weight": 1e2,
#     "p_slow_weight": 1e0,
#     "s_weight": 2e2,
#     "c_weight": 1e0,
#     "a_weight": 7e2,
#     "viol_weight": 1e12,
# }

# Rev Args
ctl_args = (
    jnp.diag(jnp.array([3e-3, 0.2])),
)
ctl_kwargs = {
    # See run_trailer_bsmc.py for the lambda = sigma_S / 20 argument; 0.5 against
    # per-step costs of order 1e3 is deep in the hard-zero regime.
    "inverse_temp": 0.5,
    "K": 500,
    "step": 0.05,
    "T": 55,
    "alpha": 0.01,
    # "gamma": 0.0,
}
cost_kwargs = {
    "reverse": False,
    "v_target": -25,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 1e2,
    "c_weight": 1e-2,
    "a_weight": 1e2,
    "viol_weight": 1e12,
}

if __name__ == "__main__":

    run_mpc(
        "RBR",
        ctl_args,
        ctl_kwargs,
        cost_kwargs,
        record=False,
        debug=True,
        quiet=False,
        benchmark=False,
        headless=False,
        env_kwargs=None,
        seed=0,
        diag_file="experiments/exp_mppi_vibe/diag/rbr_rev.csv",
    )