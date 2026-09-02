"""
Block-SMC MPPI with per-block particle evolution.

The horizon is partitioned into blocks. At each block boundary the population
gets one evolution attempt over a sliding window of the last `evolve_window`
completed blocks: propose a perturbed noise sequence over that window,
re-simulate the window from the cached block-start state, accept or reject per
particle. Selection (within-island resampling on soft accumulated cost) then
runs as a secondary normaliser.

Relationship to plain MPPI
--------------------------
`blocks=(T,)`, `evolve_rule="none"`, `resample=None` reproduces `MPPI_Jax_Debug`
to floating-point tolerance. The loop order differs (scan-over-T with vmap
inside, rather than vmap-over-K wrapping scan-over-T) because both selection and
evolution couple particles at block boundaries, so exact bit equality is not
claimed; `tests` assert allclose.

Weighting
---------
At checkpoint b the log-weight increment uses the newest block only,

    logw += -(S_b - min_k S_b) / lambda

with a *global* minimum so islands stay on a common scale. Evolution of older
blocks is a move that leaves the target invariant and therefore does not
re-weight -- its effect propagates physically through the committed state.
Because logw is reset after every resample, per-island log-normalisers are
carried so a good island is not silently equalised with a bad one:

    logZ_m += log( mean_{k in m} exp(logw_k) )
    w       = softmax( logw + logZ_{island(k)} )

Cost
----
`evolve_window=W` costs W*T extra dynamics evaluations per particle, i.e.
(1+W)x vanilla MPPI. The matched-budget baseline is vanilla MPPI at (1+W)*K
samples. The W re-simulations are fully parallel across K, but the B checkpoints
are sequentially dependent, so wall-clock is B round-trips rather than one.
"""

import functools

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from src.controllers.mpc.particle_common import (
    RESAMPLE_SCHEMES,
    bulk_std,
    cost_gaps,
    ess,
    gather_state,
    gen_evolve_fun,
    island_arrays,
    island_logZ_increment,
    make_islands,
    make_roll_fn,
    make_step_fn,
    resample_islands,
    splice,
)


def build_solver(
    dynamics,
    term_cost,
    cost,
    bound_control,
    cv,
    inv_cv,
    inverse_temp,
    alpha,
    gamma,
    K,
    T,
    step,
    blocks,
    islands,
    evolve_window,
    evolve_rule,
    evolve_kwargs,
    accept,
    prior_ratio,
    resample,
    history,
    u_d,
    debug=False,
):
    """
    Build the jitted per-control-step solver.

    A closure factory rather than a module-level `mpc_step` with
    `static_argnames`, because the block/island structure and the evolve rule are
    Python-level structure that cannot be passed as hashable static arguments
    without pushing them through every call.
    """
    step_all = make_step_fn(dynamics, cost, step, history, gamma, inv_cv)
    roll = make_roll_fn(step_all)

    B = len(blocks)
    bounds = []
    t = 0
    for n in blocks:
        bounds.append((t, t + n))
        t += n

    n_alpha = K - round(K * (1 - alpha))
    iid, istart, isize = island_arrays(islands)
    n_isl = len(islands)

    resample_fn = RESAMPLE_SCHEMES[resample] if resample is not None else None

    init_state, propose, evolve_update = gen_evolve_fun(
        rule=evolve_rule, u_d=u_d, cv=cv, lam=inverse_temp, **(evolve_kwargs or {})
    )
    evolving = evolve_rule != "none" and evolve_window > 0

    sqrt_diag = jnp.sqrt(jnp.diag(cv))

    def _term(x, u_last):
        return jax.vmap(term_cost, in_axes=(0, None))(x, u_last)

    def _logp0(eps):
        """log N(eps; 0, cv) up to an eps-independent constant, per particle."""
        return -0.5 * jnp.einsum("kln,nm,klm->k", eps, inv_cv, eps)

    def solve(x0, u, key):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (K, T, u_d)) * sqrt_diag

        v = u[None] + noise
        prev = K - n_alpha
        v = v.at[prev:].set(noise[prev:])
        v = bound_control(v)

        x = jnp.repeat(jnp.expand_dims(jnp.asarray(x0), 0), K, axis=0)
        x_ck = jnp.zeros((B + 1,) + x.shape).at[0].set(x)
        S_blk = jnp.zeros((B, K))
        logw = jnp.zeros(K)
        logZ = jnp.zeros(n_isl)
        anc0 = jnp.arange(K)
        state = init_state(K, T)

        xhist_blocks = []
        d_ess_pre, d_ess_post, d_sigma_S, d_accept, d_dS = [], [], [], [], []
        d_sigma_S_bulk, d_gaps = [], []
        d_aux: dict[str, list] = {}

        for b in range(B):
            t0, t1 = bounds[b]

            x, S_b, xh = roll(x, v[:, t0:t1], u[t0:t1], t0)
            if term_cost is not None and b == B - 1:
                S_b = S_b + _term(x, u[-1])
            S_blk = S_blk.at[b].set(S_b)
            x_ck = x_ck.at[b + 1].set(x)
            if debug:
                xhist_blocks.append(xh)

            # ---------------- evolution over the sliding window ----------------
            if evolving:
                a = max(0, b + 1 - evolve_window)
                t_a = bounds[a][0]

                eps_win = v[:, t_a:t1] - u[t_a:t1][None]
                S_win = jnp.sum(S_blk[a : b + 1], axis=0)

                key, subkey = jax.random.split(key)
                eps_prop, aux = propose(
                    subkey, state, eps_win, S_win, iid, istart, isize, n_isl, t_a
                )
                v_prop = bound_control(u[t_a:t1][None] + eps_prop)
                eps_prop = v_prop - u[t_a:t1][None]

                xs = x_ck[a]
                S_prop_blocks, x_prop_ck = [], []
                for j in range(a, b + 1):
                    j0, j1 = bounds[j]
                    xs, S_j, _ = roll(xs, v_prop[:, j0 - t_a : j1 - t_a], u[j0:j1], j0)
                    if term_cost is not None and j == B - 1:
                        S_j = S_j + _term(xs, u[-1])
                    S_prop_blocks.append(S_j)
                    x_prop_ck.append(xs)
                S_prop = sum(S_prop_blocks)

                log_ratio = -(S_prop - S_win) / inverse_temp
                if prior_ratio:
                    log_ratio = log_ratio + _logp0(eps_prop) - _logp0(eps_win)

                if accept == "greedy":
                    accepted = S_prop < S_win
                else:
                    key, subkey = jax.random.split(key)
                    lu = jnp.log(jax.random.uniform(subkey, (K,), minval=1e-30, maxval=1.0))
                    accepted = lu < log_ratio

                acc3 = accepted[:, None, None]
                acc2 = accepted[:, None]

                v = v.at[:, t_a:t1].set(jnp.where(acc3, v_prop, v[:, t_a:t1]))
                x = jnp.where(acc2, x_prop_ck[-1], x)
                for j in range(a, b + 1):
                    S_blk = S_blk.at[j].set(
                        jnp.where(accepted, S_prop_blocks[j - a], S_blk[j])
                    )
                    x_ck = x_ck.at[j + 1].set(
                        jnp.where(acc2, x_prop_ck[j - a], x_ck[j + 1])
                    )
                state = evolve_update(
                    state, eps_win, eps_prop, S_win, S_prop, accepted, t_a
                )

                if debug:
                    d_accept.append(jnp.mean(accepted.astype(jnp.float32)))
                    d_dS.append(jnp.mean(jnp.where(accepted, S_prop - S_win, 0.0)))
                    for name, value in aux.items():
                        d_aux.setdefault(name, []).append(value)

            # ---------------- weighting and selection ----------------
            S_b = S_blk[b]
            logw = logw - (S_b - jnp.min(S_b)) / inverse_temp
            if debug:
                d_sigma_S.append(jnp.std(S_b))
                d_sigma_S_bulk.append(bulk_std(S_b))
                d_gaps.append(cost_gaps(S_b))
                d_ess_pre.append(ess(jax.nn.softmax(logw)))

            if b < B - 1 and resample_fn is not None:
                logZ = logZ + island_logZ_increment(logw, islands)

                key, subkey = jax.random.split(key)
                anc = resample_islands(subkey, logw, islands, resample_fn)

                x = x[anc]
                x_ck = x_ck[:, anc]
                S_blk = S_blk[:, anc]
                v = splice(v, anc, t1)
                anc0 = anc0[anc]
                state = gather_state(state, anc)
                logw = jnp.zeros(K)

            if debug:
                d_ess_post.append(ess(jax.nn.softmax(logw + logZ[iid])))

        w = jax.nn.softmax(logw + logZ[iid])
        eps_full = v - u[None]
        u_new = u + jnp.sum(w.reshape(-1, 1, 1) * eps_full, axis=0)

        if not debug:
            return u_new, key

        diag = {
            "ess_pre": jnp.stack(d_ess_pre),
            "ess_post": jnp.stack(d_ess_post),
            "sigma_S": jnp.stack(d_sigma_S),
            "sigma_S_bulk": jnp.stack(d_sigma_S_bulk),
            "dS_p10": jnp.stack(d_gaps)[:, 0],
            "dS_p50": jnp.stack(d_gaps)[:, 1],
            "unique_anc0": jnp.unique(anc0, size=K, fill_value=-1),
            "accept": jnp.stack(d_accept) if d_accept else jnp.zeros(0),
            "d_cost": jnp.stack(d_dS) if d_dS else jnp.zeros(0),
            "w_max": jnp.max(w),
            "ess_final": ess(w),
        }
        diag.update({f"evolve_{k}": jnp.stack(v) for k, v in d_aux.items()})
        xhist = jnp.swapaxes(jnp.concatenate(xhist_blocks, axis=0), 0, 1)
        return u_new, key, xhist, v, diag

    return jax.jit(solve)


class BSMC_Jax:
    """
    Block-SMC MPPI with per-block particle evolution.
    """

    def __init__(
        self,
        x_d: int,
        u_d: int,
        dynamics_func,
        term_cost_func,
        cost_func,
        bound_control_func,
        cv,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.0,
        K=20000,
        step=0.02,
        T=70,
        blocks=None,
        evolve_window=1,
        evolve_rule="de",
        evolve_kwargs=None,
        accept="mh",
        prior_ratio=True,
        resample="systematic",
        island_size=100,
        history=None,
        seed=0,
        device="mps",
        _debug=False,
    ):
        """
        Args:
            blocks (tuple[int], optional): Block lengths, must sum to T. Defaults
                to (T,), which is the vanilla-MPPI regression configuration.
            evolve_window (int): W, how many completed blocks back the evolution
                window reaches. 0 disables evolution. Costs (1+W)x vanilla.
            evolve_rule (str): "none" | "mh" | "de" | "de_best" | "soft" |
                "soft_de" | "pso". The informed rules ("soft", "soft_de") pull
                particles toward softmax-weighted good regions of their island
                rather than toward the single best, at a temperature of
                inverse_temp * evolve_kwargs["temp_scale"].
            evolve_kwargs (dict, optional): Rule parameters, see gen_evolve_fun.
            accept (str): "mh" | "greedy". Only "mh" with evolve_rule="mh" is a
                valid Metropolis-Hastings move; the others are optimisers.
            prior_ratio (bool): Include log p0(eps') - log p0(eps) in the
                acceptance ratio. Required for a correct MH move under a
                symmetric random-walk proposal. Note the alpha particles carry
                eps = noise - u, which is not prior-distributed, so the ratio is
                approximate for that island.
            resample (str, optional): "systematic" (default) | "stratified" |
                "multinomial" | None. Runs unconditionally at every checkpoint
                except the last -- there is no ESS threshold, because systematic
                resampling is the identity at uniform weights and therefore
                self-limiting.
            island_size (int, optional): Target island size; the partition is
                derived automatically and always puts the alpha particles in
                their own island. None gives a single main island.
            seed (int): PRNG seed, for paired-seed sweeps.
        """
        if blocks is None:
            blocks = (T,)
        blocks = tuple(int(b) for b in blocks)
        if sum(blocks) != T:
            raise ValueError(f"blocks {blocks} sum to {sum(blocks)}, expected T={T}")
        if any(b <= 0 for b in blocks):
            raise ValueError(f"blocks {blocks} must all be positive")
        if accept not in ("mh", "greedy"):
            raise ValueError(f"accept must be 'mh' or 'greedy', got {accept!r}")
        if resample is not None and resample not in RESAMPLE_SCHEMES:
            raise ValueError(f"unknown resample scheme {resample!r}")

        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
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

        self.blocks = blocks
        self.evolve_window = evolve_window
        self.evolve_rule = evolve_rule
        self.accept = accept
        self.resample = resample

        n_alpha = K - round(K * (1 - alpha))
        self.islands = make_islands(K, n_alpha, island_size)

        self.seed = seed
        self.key = jax.random.key(seed)

        self._solve = build_solver(
            dynamics=dynamics_func,
            term_cost=term_cost_func,
            cost=cost_func,
            bound_control=bound_control_func,
            cv=self.cv,
            inv_cv=self.inv_cv,
            inverse_temp=inverse_temp,
            alpha=alpha,
            gamma=gamma,
            K=K,
            T=T,
            step=step,
            blocks=self.blocks,
            islands=self.islands,
            evolve_window=evolve_window,
            evolve_rule=evolve_rule,
            evolve_kwargs=evolve_kwargs,
            accept=accept,
            prior_ratio=prior_ratio,
            resample=resample,
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