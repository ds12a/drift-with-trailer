"""
Resampling-Based Rollouts (RBR) MPPI.

Implementation of Yin, So, Yu, Fan, Tsiotras, "Safe Beyond the Horizon:
Efficient Sampling-based MPC with Neural Control Barrier Functions"
(arXiv 2502.15006v2), with the paper's learned DPNCBF replaced by the exact
analytic constraint indicator this testbed already has (track half-width
excursion plus hitch beyond max_hitch, from `gen_util_funs(..., with_violation=True)`).
That substitution removes their approximation caveat and makes this the
apples-to-apples baseline for cost-directed selection.

Algorithm, per rollout step:
    1. advance every particle with its own control
    2. safe = violation(x) <= 0                                  (their Eq. 31)
    3. if any particle is safe, each violator is teleported onto a particle
       drawn *uniformly* from the safe set (their Fig. 3 / Theorem 3 -- the
       conditional-prior identity requires uniform, not cost-proportional,
       replacement); safe particles are left alone
    4. if every particle violates, no resample; the run falls through to the
       cost term (their Eq. 18), which here is the 1e12 * violation penalty
    5. only the state is gathered. Controls are never gathered: a rewired
       particle keeps its own sequence, applied from the new state

Cost accumulates along the spliced state path, and the final update is plain
MPPI SNIS over the original noise. Two properties worth recording rather than
"fixing": the accumulated S follows a path no single control sequence realises
(inherent to RBR -- their theorem is about the state distribution, not the
path), and because nothing but `x` is gathered there is no genealogy
bookkeeping at all, so per-step resampling is cheap.

Islands are deliberately not offered: RBR's replacement must be uniform over the
whole safe set for Theorem 3 to hold.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.controllers.mpc.particle_common import (
    bulk_std,
    cost_gaps,
    ess,
    make_step_fn,
    uniform_among,
)


def build_solver(
    dynamics,
    term_cost,
    cost,
    bound_control,
    violation,
    cv,
    inv_cv,
    inverse_temp,
    alpha,
    gamma,
    K,
    T,
    step,
    history,
    u_d,
    debug=False,
):
    """Build the jitted per-control-step solver (closure factory, see bsmc_jax)."""
    step_all = make_step_fn(dynamics, cost, step, history, gamma, inv_cv)
    viol_all = jax.vmap(violation)
    sqrt_diag = jnp.sqrt(jnp.diag(cv))
    n_alpha = K - round(K * (1 - alpha))

    def solve(x0, u, key):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (K, T, u_d)) * sqrt_diag

        v = u[None] + noise
        prev = K - n_alpha
        v = v.at[prev:].set(noise[prev:])
        v = bound_control(v)
        eps = v - u[None]

        x = jnp.repeat(jnp.expand_dims(jnp.asarray(x0), 0), K, axis=0)

        def body(carry, xs):
            x, S, key = carry
            v_t, u_t, i = xs

            new_x, c = step_all(x, v_t, u_t, i)
            S = S + c

            safe = viol_all(new_x) <= 0.0
            any_safe = jnp.any(safe)

            key, subkey = jax.random.split(key)
            anc = jnp.where(safe, jnp.arange(K), uniform_among(subkey, safe))
            anc = jnp.where(any_safe, anc, jnp.arange(K))

            return (new_x[anc], S, key), (new_x, jnp.sum(~safe), any_safe)

        (x, S, key), (xhist, n_unsafe, any_safe) = jax.lax.scan(
            body,
            (x, jnp.zeros(K), key),
            (jnp.swapaxes(v, 0, 1), u, jnp.arange(T)),
        )

        if term_cost is not None:
            S = S + jax.vmap(term_cost, in_axes=(0, None))(x, u[-1])

        w = jax.nn.softmax(-(S - jnp.min(S)) / inverse_temp)
        u_new = u + jnp.sum(w.reshape(-1, 1, 1) * eps, axis=0)

        if not debug:
            return u_new, key

        diag = {
            "n_unsafe": n_unsafe,
            "n_all_unsafe": jnp.sum(~any_safe),
            "frac_unsafe": jnp.mean(n_unsafe / K),
            "sigma_S": jnp.std(S),
            "sigma_S_bulk": bulk_std(S),
            "dS_p10": cost_gaps(S)[0],
            "dS_p50": cost_gaps(S)[1],
            "w_max": jnp.max(w),
            "ess_final": ess(w),
        }
        # xhist is the pre-rewire simulated state at each step, so the drawn
        # candidate lines are true dynamics output; the teleports are not shown.
        return u_new, key, jnp.swapaxes(xhist, 0, 1), v, diag

    return jax.jit(solve)


class RBR_Jax:
    """
    JAX RBR (feasibility-resampled MPPI).
    """

    def __init__(
        self,
        x_d: int,
        u_d: int,
        dynamics_func,
        term_cost_func,
        cost_func,
        bound_control_func,
        violation_func,
        cv,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.0,
        K=20000,
        step=0.02,
        T=70,
        history=None,
        seed=0,
        device="mps",
        _debug=False,
    ):
        """
        Args:
            violation_func (Callable): (x_d,) -> scalar, > 0 iff the state is
                infeasible. Obtain from
                `gen_util_funs(..., with_violation=True)`; it is the same
                expression the cost function penalises, so the indicator and the
                penalty can never disagree.
            seed (int): PRNG seed, for paired-seed sweeps.

        Remaining arguments match MPPI_Jax.
        """
        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
        self.violation = violation_func
        self.alpha = alpha
        self.inverse_temp = inverse_temp
        self.gamma = gamma
        self.K = K
        self.device = device

        self.x_d = x_d
        self.u_d = u_d
        self.T = T
        self.history = history
        self.step = step
        self.cv = cv
        self.inv_cv = jnp.linalg.inv(self.cv)

        self.seed = seed
        self.key = jax.random.key(seed)

        self._solve = build_solver(
            dynamics=dynamics_func,
            term_cost=term_cost_func,
            cost=cost_func,
            bound_control=bound_control_func,
            violation=violation_func,
            cv=self.cv,
            inv_cv=self.inv_cv,
            inverse_temp=inverse_temp,
            alpha=alpha,
            gamma=gamma,
            K=K,
            T=T,
            step=step,
            history=history,
            u_d=u_d,
            debug=_debug,
        )

    def _shift(self):
        if self.last_trajectory is None:
            return jnp.zeros((self.T, self.u_d))
        u = jnp.roll(self.last_trajectory, -1, axis=0)
        return u.at[-1].set(0)

    def run_mpc(self, x: ArrayLike):
        """
        Runs a single MPC solve.

        Args:
            x (ArrayLike): State (x_d)

        Returns:
            ArrayLike: Control output
        """
        u, self.key = self._solve(jnp.asarray(x), self._shift(), self.key)
        self.last_trajectory = u
        return u[0]

    def reset(self):
        """
        Clean resets the controller (i.e. when the env is reset and there is NaN
        pollution in history).
        """
        self.last_trajectory = None
        self.key = jax.random.key(self.seed)