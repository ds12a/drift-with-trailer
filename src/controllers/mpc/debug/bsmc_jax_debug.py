"""
Designed as a drop-in replacement for BSMC_Jax but with extra visualization and
diagnostic tools for debugging purposes.

Unlike mppi_jax_debug.py / smppi_jax_debug.py this does not duplicate the solver
body. The solver is built by `bsmc_jax.build_solver`, which takes a `debug` flag
that only adds returns -- the numerical path is one piece of code. That is
deliberate: the MPPI_Jax / MPPI_Jax_Debug `gamma` desync (one class silently
recomputing a value the other honoured) came from exactly this duplication, and
the block/evolve solver is large enough that a second copy would be a matter of
time before it drifted.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.controllers.mpc.bsmc_jax import BSMC_Jax


class BSMC_Jax_Debug(BSMC_Jax):
    """
    Block-SMC MPPI with per-block particle evolution, with debug returns.

    `run_mpc` returns `(u, xhist, vhist, diag)` instead of `u`, where

        xhist (K, T, x_d)   per-particle simulated states over the horizon
        vhist (K, T, u_d)   per-particle applied controls, post-splice
        diag  dict          per-checkpoint diagnostics, see below

    diag keys
    ---------
    ess_pre / ess_post   ESS before and after selection at each checkpoint
    sigma_S              per-block cost spread; tells you whether the block is
                         longer than the trajectory-separation timescale
    accept               evolution acceptance rate per checkpoint. The primary
                         number: if this collapses, the mechanism is inert
    d_cost               mean window cost improvement on accepted moves
    unique_anc0          ancestral particle indices surviving to the end,
                         padded with -1. `(unique_anc0 >= 0).sum()` is the
                         make-or-break diversity number -- if selection fixes
                         ESS while this collapses to a handful, degeneracy was
                         relocated rather than solved
    ess_final / w_max    final population weighting
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