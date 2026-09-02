"""
Designed as a drop-in replacement for RBR_Jax but with extra visualization and
diagnostic tools for debugging purposes.

As with bsmc_jax_debug.py, the solver body is not duplicated -- `rbr_jax.build_solver`
takes a `debug` flag that only adds returns, so the numerical path cannot drift
between the two classes.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.controllers.mpc.rbr_jax import RBR_Jax


class RBR_Jax_Debug(RBR_Jax):
    """
    JAX RBR, with debug returns.

    `run_mpc` returns `(u, xhist, vhist, diag)` instead of `u`, where

        xhist (K, T, x_d)   pre-rewire simulated states, so the drawn candidate
                            lines are true dynamics output rather than teleports
        vhist (K, T, u_d)   per-particle controls (never gathered, by design)
        diag  dict          see below

    diag keys
    ---------
    n_unsafe        violating particles per horizon step, (T,)
    n_all_unsafe    steps at which every particle violated and the mechanism had
                    nothing to rewire onto. This is the regime the whole
                    soft-weighting argument turns on -- if it is nonzero in
                    high-speed reverse, RBR is inert exactly where it matters
    frac_unsafe     mean violating fraction over the horizon
    sigma_S         cost spread across the population
    ess_final       ESS of the final MPPI weights
    """

    def __init__(self, *args, **kwargs):
        kwargs["_debug"] = True
        super().__init__(*args, **kwargs)

    def run_mpc(self, x: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, dict]:
        """
        Runs a single MPC solve.

        Args:
            x (ArrayLike): State (x_d)

        Returns:
            tuple: (control output, xhist, vhist, diag)
        """
        u, self.key, xhist, vhist, diag = self._solve(
            jnp.asarray(x), self._shift(), self.key
        )
        self.last_trajectory = u
        return u[0], xhist, vhist, diag