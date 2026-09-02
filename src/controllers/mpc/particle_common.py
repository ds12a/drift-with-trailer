"""
Shared primitives for the particle-based MPPI variants (RBR_Jax, BSMC_Jax).

These are pure functions with no controller-specific behaviour, so they live in
one module rather than being duplicated per controller. The `debug/` split still
applies to the controllers themselves (`rbr_jax.py` / `debug/rbr_jax_debug.py`,
`bsmc_jax.py` / `debug/bsmc_jax_debug.py`), mirroring
`mppi_jax.py` / `debug/mppi_jax_debug.py`.

Contents
--------
ess / resampling schemes            selection operators, all cumsum+searchsorted
uniform_among                       RBR's uniform draw over the safe set
make_islands / island_arrays        automatic island partition (alpha block last)
resample_islands                    within-island selection, contiguous slices
make_step_fn                        one vmapped population step (matches
                                    mppi_jax.rollout's step_dynamics exactly)
gen_evolve_fun                      particle update rules, in the same
                                    generator style as gen_util_funs
"""

import functools

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


# --------------------------------------------------------------------------- #
# Weights / resampling
# --------------------------------------------------------------------------- #


def ess(w: ArrayLike) -> jax.Array:
    """Effective sample size of a normalised weight vector."""
    return 1.0 / jnp.sum(w**2)


def bulk_std(S: ArrayLike, q: float = 0.5) -> jax.Array:
    """
    Cost spread over the lowest `q` quantile of the population.

    The raw std is dominated by rollouts carrying the violation penalty (1e12
    here, 1e5 in BeamNG), which correctly get zero weight and therefore
    contribute nothing to selection. Only the non-violating bulk is
    discriminative. Note that when violations are *common* rather than outliers
    -- reverse on a narrow track, where most rollouts leave the boundary -- even
    a high quantile is contaminated, so read `cost_gaps` instead.
    """
    S = jnp.asarray(S)
    cut = jnp.quantile(S, q)
    keep = S <= cut
    n = jnp.maximum(jnp.sum(keep), 1.0)
    mean = jnp.sum(jnp.where(keep, S, 0.0)) / n
    var = jnp.sum(jnp.where(keep, (S - mean) ** 2, 0.0)) / n
    return jnp.sqrt(var)


def cost_gaps(S: ArrayLike) -> jax.Array:
    """
    (S_p10 - S_min, S_p50 - S_min) -- the temperature-setting numbers.

    Weight degeneracy is governed by how far the *competitive* particles sit
    above the best one, not by the population std, which any violating tail
    swamps. Setting lambda ~ dS_p10 / 2 puts roughly the top decile at
    non-negligible weight; lambda below dS_p10 / 87 hard-zeros everything but
    the argmin in float32.
    """
    S = jnp.asarray(S)
    lo = jnp.min(S)
    return jnp.stack([jnp.quantile(S, 0.1) - lo, jnp.quantile(S, 0.5) - lo])


def multinomial_resample(key, w: ArrayLike) -> jax.Array:
    """K iid draws from the categorical(w). Highest variance; baseline only."""
    K = w.shape[0]
    cdf = jnp.cumsum(w)
    r = jax.random.uniform(key, (K,)) * cdf[-1]
    return jnp.searchsorted(cdf, r, side="right")


def systematic_resample(key, w: ArrayLike) -> jax.Array:
    """
    Single stratified offset, K evenly spaced comb points.

    This already gives the residual guarantee: particle k receives either
    floor(K w_k) or ceil(K w_k) offspring, so a separate residual pass is
    redundant. It is also the *identity* at uniform weights, which is what makes
    unconditional resampling (no ESS threshold) safe -- a non-degenerate
    population is left alone.

    The known caveat is that offspring assignment is correlated with particle
    ordering. Within an island the ordering is arbitrary (iid noise), so this is
    benign here; use `stratified_resample` if that assumption is ever broken.
    """
    K = w.shape[0]
    cdf = jnp.cumsum(w)
    u0 = jax.random.uniform(key, ()) / K
    pts = (u0 + jnp.arange(K) / K) * cdf[-1]
    return jnp.searchsorted(cdf, pts, side="right")


def stratified_resample(key, w: ArrayLike) -> jax.Array:
    """Independent uniform per stratum. Breaks systematic's ordering coupling."""
    K = w.shape[0]
    cdf = jnp.cumsum(w)
    u = jax.random.uniform(key, (K,)) / K
    pts = (u + jnp.arange(K) / K) * cdf[-1]
    return jnp.searchsorted(cdf, pts, side="right")


RESAMPLE_SCHEMES = {
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "multinomial": multinomial_resample,
}


def uniform_among(key, mask: ArrayLike) -> jax.Array:
    """
    For every particle, an index drawn uniformly from the entries where `mask`
    is True. RBR Theorem 3 requires the replacement to be uniform (not
    cost-proportional), so this is deliberately weight-free.

    Undefined when `mask` is all False; callers must guard with that branch.
    """
    K = mask.shape[0]
    cdf = jnp.cumsum(mask.astype(jnp.float32))
    n_safe = cdf[-1]
    r = jax.random.uniform(key, (K,)) * n_safe
    return jnp.clip(jnp.searchsorted(cdf, r, side="right"), 0, K - 1)


# --------------------------------------------------------------------------- #
# Islands
# --------------------------------------------------------------------------- #


def make_islands(K: int, n_alpha: int, island_size: int | None = 100) -> tuple[int, ...]:
    """
    Automatic island partition.

    `n_alpha` must be `K - round(K*(1-alpha))`, i.e. the size of the trailing
    slice that `_forward_sim` strips the nominal from -- taken as an int rather
    than recomputed from `alpha` so the island boundary and the alpha slice can
    never disagree by a rounding step. Those particles are already contiguous at
    the end of the population, so they become the final island with no
    reordering. Keeping them separate stops nominal-tracking lineages from
    overwriting the only nominal-free samples in the population.

    `island_size` is a resolution choice, not a derived quantity: below ~100
    particles a per-island softmax over accumulated block cost stops
    discriminating. `island_size=None` (or >= n_main) gives a single main island,
    i.e. the no-islands ablation.
    """
    n_main = K - n_alpha
    if n_main <= 0:
        raise ValueError(f"n_alpha={n_alpha} leaves no main particles at K={K}")

    if island_size is None or island_size >= n_main:
        main = (n_main,)
    else:
        M = max(1, n_main // int(island_size))
        base, rem = divmod(n_main, M)
        main = tuple(base + (1 if i < rem else 0) for i in range(M))

    islands = main + ((n_alpha,) if n_alpha > 0 else ())
    assert sum(islands) == K
    return islands


def island_arrays(islands: tuple[int, ...]):
    """(island_id, island_start, island_size) as per-particle (K,) int32 arrays."""
    ids, starts, sizes = [], [], []
    off = 0
    for m, n in enumerate(islands):
        ids.append(jnp.full((n,), m, dtype=jnp.int32))
        starts.append(jnp.full((n,), off, dtype=jnp.int32))
        sizes.append(jnp.full((n,), n, dtype=jnp.int32))
        off += n
    return jnp.concatenate(ids), jnp.concatenate(starts), jnp.concatenate(sizes)


def resample_islands(key, logw: ArrayLike, islands: tuple[int, ...], scheme) -> jax.Array:
    """
    Ancestor indices from within-island selection. Islands are contiguous slices,
    so this is a static Python loop over at most a handful of segments.

    Weights are renormalised inside each island; cross-island mass is carried
    separately by the log-normaliser (see BSMC_Jax), because logw is reset to
    zero after every resample and would otherwise equalise the islands.
    """
    out = []
    off = 0
    for n in islands:
        li = logw[off : off + n]
        wi = jax.nn.softmax(li)
        key, subkey = jax.random.split(key)
        out.append(scheme(subkey, wi) + off)
        off += n
    return jnp.concatenate(out)


def island_logZ_increment(logw: ArrayLike, islands: tuple[int, ...]) -> jax.Array:
    """
    log( mean_{k in m} exp(logw_k) ) per island.

    `logw` must have been accumulated against a *global* per-block minimum so
    that islands remain on a common scale.
    """
    out = []
    off = 0
    for n in islands:
        li = logw[off : off + n]
        out.append(jax.scipy.special.logsumexp(li) - jnp.log(n))
        off += n
    return jnp.stack(out)


# --------------------------------------------------------------------------- #
# Population step
# --------------------------------------------------------------------------- #


def make_step_fn(dynamics, cost, step, history, gamma, inv_cv):
    """
    One population step, vmapped over the K particles.

    Body is copied verbatim from `mppi_jax.rollout.step_dynamics` -- including
    the `history` branch and the gamma control-effort term -- so that a
    single-block, no-resample, no-evolve configuration reproduces MPPI exactly.

    Note the `history` branch needs no separate window carry: the flat H-row
    window *is* `x`, so gathering `x` by ancestor gathers the history for free.

    Args:
        dynamics: (x_d,) x (u_d,) -> (x_d,)
        cost:     (x_d,) x (u_d,) x scalar -> scalar
        step:     integration timestep
        history:  number of stacked rows in x, or None
        gamma:    control-effort weight (0.0 in every current config)
        inv_cv:   (u_d, u_d)

    Returns:
        step_all(x, u_t, v_t, i) -> (new_x, c) with x (K, x_d), v_t (K, u_d),
        u_t (u_d,) the shared nominal, i the *global* horizon index.
    """

    def _one(x, v, u_t, i):
        if history is not None:
            step_dim = x.shape[0] // history
            curr_x = x[-step_dim:]
            dx = dynamics(x, v)
            new_curr_x = curr_x + dx * step
            new_x = jnp.concatenate([x[step_dim:], new_curr_x])
        else:
            new_x = x + dynamics(x, v) * step

        c = cost(new_x, v, i) + gamma * jnp.einsum("n,nm,m->", u_t, inv_cv, v - u_t)
        return new_x, c

    return jax.vmap(_one, in_axes=(0, 0, None, None))


def make_roll_fn(step_all):
    """
    Scan a population over a contiguous span of the horizon.

    roll(x, v_span, u_span, t0) -> (x_end, S_span, xhist)
        v_span (K, L, u_d), u_span (L, u_d), t0 the global index of the first
        step. `xhist` is (L, K, x_d); transpose at the call site if a (K, L, .)
        layout is wanted.
    """

    def roll(x, v_span, u_span, t0):
        L = u_span.shape[0]

        def body(carry, xs):
            x, S = carry
            v_t, u_t, i = xs
            new_x, c = step_all(x, v_t, u_t, i)
            return (new_x, S + c), new_x

        (x, S), xhist = jax.lax.scan(
            body,
            (x, jnp.zeros(x.shape[0])),
            (jnp.swapaxes(v_span, 0, 1), u_span, t0 + jnp.arange(L)),
        )
        return x, S, xhist

    return roll


def splice(v_full: ArrayLike, anc: ArrayLike, t_b: int) -> jax.Array:
    """
    Gather the *applied* prefix [0, t_b) by ancestor; leave the tail alone.

    The asymmetry is the whole mechanism. Duplicated lineages share a state and
    a past but carry different futures, so they separate immediately. Gathering
    the tail as well would leave duplicates bit-identical for the rest of the
    horizon -- ESS reports as repaired while the effective population has
    shrunk to the number of distinct ancestors. This function is written so that
    cannot happen.
    """
    T = v_full.shape[1]
    keep = (jnp.arange(T) < t_b)[None, :, None]
    return jnp.where(keep, v_full[anc], v_full)


# --------------------------------------------------------------------------- #
# Particle update rules
# --------------------------------------------------------------------------- #


def _island_softmax(S, iid, n_isl, lam):
    """
    Per-island softmax over window cost, min-subtracted inside each island so
    islands stay independent and float32 never sees a large positive exponent.
    """
    smin = jax.ops.segment_min(S, iid, num_segments=n_isl)
    e = jnp.exp(-(S - smin[iid]) / lam)
    Z = jax.ops.segment_sum(e, iid, num_segments=n_isl)
    return e / jnp.maximum(Z[iid], 1e-30)


def _island_ess(w, iid, n_isl):
    """Mean ESS of the per-island weight vectors, as a fraction-free count."""
    sq = jax.ops.segment_sum(w**2, iid, num_segments=n_isl)
    return jnp.mean(1.0 / jnp.maximum(sq, 1e-30))


def _island_categorical(key, w, iid, istart, isize, n_isl):
    """
    One partner index per particle, drawn from its own island's softmax.

    Islands are contiguous, so the global cumsum is monotone across them and a
    single searchsorted lands inside the right island once the draw is offset by
    that island's exclusive prefix mass. Weights are already normalised per
    island, so the within-island mass is 1.
    """
    K = w.shape[0]
    csum = jnp.cumsum(w)
    excl = jnp.where(istart > 0, csum[jnp.maximum(istart - 1, 0)], 0.0)
    r = jax.random.uniform(key, (K,))
    idx = jnp.searchsorted(csum, excl + r, side="right")
    return jnp.clip(idx, istart, istart + isize - 1)


def _island_best(S, iid, n_isl):
    K = S.shape[0]
    best_val = jax.ops.segment_min(S, iid, num_segments=n_isl)
    is_best = S <= best_val[iid]
    best_idx = jax.ops.segment_max(
        jnp.where(is_best, jnp.arange(K), -1), iid, num_segments=n_isl
    )
    return jnp.maximum(best_idx, 0)


def gen_evolve_fun(
    rule: str = "de",
    u_d: int = 2,
    cv: ArrayLike | None = None,
    lam: float = 1.0,
    temp_scale: float = 1.0,
    eta: float = 0.5,
    sigma_scale: float = 0.5,
    F: float = 0.6,
    CR: float = 0.9,
    w_inertia: float = 0.6,
    c1: float = 1.2,
    c2: float = 1.2,
):
    """
    Build a particle update rule, in the same generator style as `gen_util_funs`.

    Returns `(init_state, propose, update)`:

        init_state(K, T)                      -> pytree carried across checkpoints
        propose(key, state, eps_win, S_win,
                island_id, island_start,
                island_size, n_islands, t_a)  -> (eps_prop (K, L, u_d), aux dict)
        update(state, eps_win, eps_prop,
               S_win, S_prop, accepted, t_a)  -> pytree

    `aux` carries scalars for the debug diagnostics and may be empty.

    All proposals act on the *noise* eps = v - u over the window, never on v, so
    the shared nominal cancels out of difference vectors and attractors. The
    caller re-bounds `u + eps_prop` before simulating it.

    Temperature
    -----------
    The informed rules weight the population by `exp(-(S_win - min)/lam_e)` with
    `lam_e = lam * temp_scale`, `lam` defaulting to the controller's own
    `inverse_temp`. This inherits MPPI's own pathology in miniature: too sharp
    and the attractor is the single best particle, too soft and it is the island
    mean, which is the nominal. `temp_scale` is the knob and
    `aux["attract_ess"]` is its health check -- the mean per-island ESS of the
    attraction weights. Somewhere in the tens is the useful range at K/island
    around 100.

    Rules
    -----
    "none"     identity; disables evolution (resample-only ablation)
    "mh"       Gaussian random walk. Uninformed, and the only rule for which
               accept="mh" is a valid Metropolis-Hastings move on the path
               measure -- the others are optimisers and change the target.
    "de"       DE/rand/1/bin within island. Difference vector is scale-adaptive
               for free: wide when the island is spread, narrow as it
               concentrates.
    "de_best"  DE/current-to-best/1/bin. Single attractor, fastest collapse.
    "soft"     Path-integral attraction. Each particle moves a fraction `eta`
               toward its island's softmax-weighted mean noise sequence, plus
               exploration noise:

                   mu_m  = sum_k w_k eps_k          (w from the island softmax)
                   eps' = eps + eta (mu_m - eps) + sigma xi

               The attractor is exactly MPPI's own update restricted to one
               block and one island, so this is the path-integral estimate used
               as a search direction rather than as a final answer. Note
               eta=1, sigma=0 makes every particle in an island identical, which
               is total collapse -- eta < 1 and sigma > 0 are what keep the
               population alive. Closest relatives are CEM/EDA mean-shift
               (Estimation of Distribution Algorithms) and, if a repulsion term
               were added, Stein Variational MPC.
    "soft_de"  DE/current-to-soft-best/1/bin. Each particle draws its *own*
               attractor from the island softmax, so good particles pull without
               everyone collapsing onto one of them, and adds a difference
               vector for scale adaptation:

                   eps' = eps + eta (eps_a - eps) + F (eps_b - eps_c)

               `a ~ softmax(-S/lam_e)` within island, `b, c` uniform. This is
               JADE's current-to-pbest/1 (Zhang & Sanderson 2009) with a
               softmax replacing the top-p truncation, which is the standard
               answer to "attract toward good, not toward best". Default rule
               for the informed setting.
    "pso"      Velocity + personal best carried on the full (K, T, u_d) buffer
               and sliced by the window. An approximation here: with one
               proposal per checkpoint and a sliding window, personal best is
               only meaningful across checkpoints, so `pbest` stores the best
               committed noise sequence so far.
    """
    valid = ("none", "mh", "de", "de_best", "soft", "soft_de", "pso")
    if rule not in valid:
        raise ValueError(f"unknown evolve rule {rule!r}, expected one of {valid}")

    sigma = None
    if cv is not None:
        sigma = sigma_scale * jnp.sqrt(jnp.diag(jnp.asarray(cv)))

    lam_e = lam * temp_scale

    def _no_state(K, T):
        return {}

    def _no_update(state, eps_win, eps_prop, S_win, S_prop, accepted, t_a):
        return state

    # ---------------- none ----------------
    if rule == "none":

        def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            return eps_win, {}

        return _no_state, propose, _no_update

    # ---------------- mh ----------------
    if rule == "mh":
        if sigma is None:
            raise ValueError("rule='mh' needs cv for the proposal scale")

        def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            return eps_win + jax.random.normal(key, eps_win.shape) * sigma, {}

        return _no_state, propose, _no_update

    # ---------------- de / de_best ----------------
    if rule in ("de", "de_best"):

        def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            K = eps_win.shape[0]
            k1, k2, k3 = jax.random.split(key, 3)
            a = istart + jax.random.randint(k1, (K,), 0, isize)
            b = istart + jax.random.randint(k2, (K,), 0, isize)
            diff = eps_win[a] - eps_win[b]

            if rule == "de":
                mutant = eps_win + F * diff
            else:
                best = _island_best(S_win, iid, n_isl)[iid]
                mutant = eps_win + F * (eps_win[best] - eps_win) + F * diff

            cross = jax.random.uniform(k3, eps_win.shape) < CR
            return jnp.where(cross, mutant, eps_win), {}

        return _no_state, propose, _no_update

    # ---------------- soft ----------------
    if rule == "soft":
        if sigma is None:
            raise ValueError("rule='soft' needs cv for the exploration scale")

        def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            w = _island_softmax(S_win, iid, n_isl, lam_e)
            mu = jax.ops.segment_sum(
                w[:, None, None] * eps_win, iid, num_segments=n_isl
            )
            xi = jax.random.normal(key, eps_win.shape) * sigma
            prop = eps_win + eta * (mu[iid] - eps_win) + xi
            return prop, {"attract_ess": _island_ess(w, iid, n_isl)}

        return _no_state, propose, _no_update

    # ---------------- soft_de ----------------
    if rule == "soft_de":

        def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            K = eps_win.shape[0]
            k1, k2, k3, k4 = jax.random.split(key, 4)

            w = _island_softmax(S_win, iid, n_isl, lam_e)
            a = _island_categorical(k1, w, iid, istart, isize, n_isl)
            b = istart + jax.random.randint(k2, (K,), 0, isize)
            c = istart + jax.random.randint(k3, (K,), 0, isize)

            mutant = (
                eps_win
                + eta * (eps_win[a] - eps_win)
                + F * (eps_win[b] - eps_win[c])
            )
            cross = jax.random.uniform(k4, eps_win.shape) < CR
            prop = jnp.where(cross, mutant, eps_win)

            aux = {
                "attract_ess": _island_ess(w, iid, n_isl),
                "attract_uniq": jnp.sum(
                    jnp.bincount(a, length=K) > 0
                ).astype(jnp.float32),
            }
            return prop, aux

        return _no_state, propose, _no_update

    # ---------------- pso ----------------
    def init_state(K, T):
        return {
            "vel": jnp.zeros((K, T, u_d)),
            "pbest": jnp.zeros((K, T, u_d)),
            "pbest_S": jnp.full((K,), jnp.inf),
            "pbest_set": jnp.zeros((K,), dtype=bool),
        }

    def propose(key, state, eps_win, S_win, iid, istart, isize, n_isl, t_a):
            K, L = eps_win.shape[0], eps_win.shape[1]
            k1, k2, k3, k4 = jax.random.split(key, 4)

            vel = jax.lax.dynamic_slice(state["vel"], (0, t_a, 0), (K, L, u_d))
            pbest = jax.lax.dynamic_slice(state["pbest"], (0, t_a, 0), (K, L, u_d))
            pbest = jnp.where(state["pbest_set"][:, None, None], pbest, eps_win)

            # 1. FAST O(K) Softmax Weighting
            w = _island_softmax(S_win, iid, n_isl, lam_e)

            # 2. FAST O(K) Center of Mass (Weighted by Fitness)
            mu = jax.ops.segment_sum(w[:, None, None] * eps_win, iid, num_segments=n_isl)
            island_mean = mu[iid]

            # 3. FAST O(K) Empirical Non-Gaussian Spread (DE-style)
            # Pick two random partners from the same island
            b = istart + jax.random.randint(k3, (K,), 0, isize)
            c = istart + jax.random.randint(k4, (K,), 0, isize)
            
            # Calculate the vector between them
            empirical_spread = F * (eps_win[b] - eps_win[c])

            # 4. Standard PSO Uniform Random Factors (Uniform is non-Gaussian)
            r1 = jax.random.uniform(k1, eps_win.shape)
            r2 = jax.random.uniform(k2, eps_win.shape)

            # Velocity Update: No Gaussians. Driven purely by empirical population geometry.
            new_vel = (
                w_inertia * vel * 0
                + c1 * r1 * (pbest - eps_win) 
                + c2 * r2 * (island_mean - eps_win)
                + empirical_spread
            )
            
            return eps_win + new_vel, {"attract_ess": _island_ess(w, iid, n_isl)}

    def update(state, eps_win, eps_prop, S_win, S_prop, accepted, t_a):
        K, L = eps_win.shape[0], eps_win.shape[1]
        acc = accepted[:, None, None]

        committed = jnp.where(acc, eps_prop, eps_win)
        new_vel = jnp.where(acc, eps_prop - eps_win, jnp.zeros_like(eps_win))
        S_committed = jnp.where(accepted, S_prop, S_win)

        improved = S_committed < state["pbest_S"]
        pbest_slice = jax.lax.dynamic_slice(state["pbest"], (0, t_a, 0), (K, L, u_d))
        new_pbest = jnp.where(improved[:, None, None], committed, pbest_slice)

        return {
            "vel": jax.lax.dynamic_update_slice(state["vel"], new_vel, (0, t_a, 0)),
            "pbest": jax.lax.dynamic_update_slice(state["pbest"], new_pbest, (0, t_a, 0)),
            "pbest_S": jnp.minimum(state["pbest_S"], S_committed),
            "pbest_set": state["pbest_set"] | improved,
        }

    return init_state, propose, update


def gather_state(state, anc):
    """Gather every leading-K leaf of an evolve state pytree by ancestor index."""
    K = anc.shape[0]
    return jax.tree.map(lambda a: a[anc] if a.ndim >= 1 and a.shape[0] == K else a, state)