"""
Self-contained smoke / regression tests for RBR_Jax and BSMC_Jax.

Toy plant only -- no env, no track, no BeamNG. Run from the repo root:

    python3 -m experiments.exp_mppi_vibe.tests_particle
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.controllers.mpc.bsmc_jax import BSMC_Jax
from src.controllers.mpc.debug.bsmc_jax_debug import BSMC_Jax_Debug
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.controllers.mpc.debug.rbr_jax_debug import RBR_Jax_Debug
from src.controllers.mpc.particle_common import (
    make_islands,
    multinomial_resample,
    systematic_resample,
    uniform_among,
)
from src.controllers.mpc.rbr_jax import RBR_Jax

K, T, U_D, X_D = 200, 24, 2, 4
CV = jnp.diag(jnp.array([3e-3, 2e-1]))
LAM, ALPHA, STEP = 5.0, 0.05, 0.05


def dynamics(x, u):
    # unicycle: [x, y, theta, v]
    return jnp.array([x[3] * jnp.cos(x[2]), x[3] * jnp.sin(x[2]), x[3] * u[0], u[1]])


def cost(x, u, t):
    return 0.99**t * (10.0 * x[1] ** 2 + (x[3] - 5.0) ** 2 + 1e3 * jnp.maximum(0, jnp.abs(x[1]) - 2.0))


def violation(x):
    return jnp.maximum(0.0, jnp.abs(x[1]) - 2.0)


def no_violation(x):
    return jnp.zeros(())


def bound(u):
    return jnp.clip(u, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))


X0 = jnp.array([0.0, 0.5, 0.1, 3.0])
COMMON = dict(inverse_temp=LAM, alpha=ALPHA, gamma=0.0, K=K, step=STEP, T=T)


def _report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def test_resamplers():
    w = jnp.full((K,), 1.0 / K)
    anc = systematic_resample(jax.random.key(0), w)
    ok = bool(jnp.all(anc == jnp.arange(K)))
    ok &= _report("systematic is identity at uniform weights", ok)

    w2 = jnp.zeros(K).at[7].set(1.0)
    anc2 = systematic_resample(jax.random.key(1), w2)
    ok2 = bool(jnp.all(anc2 == 7))
    _report("systematic collapses to the single-mass particle", ok2)

    counts = jnp.bincount(multinomial_resample(jax.random.key(2), w), length=K)
    ok3 = int(counts.sum()) == K
    _report("multinomial returns K ancestors", ok3)

    mask = jnp.zeros(K, dtype=bool).at[jnp.array([3, 50, 199])].set(True)
    idx = uniform_among(jax.random.key(3), mask)
    ok4 = bool(jnp.all(mask[idx]))
    _report("uniform_among only ever returns safe indices", ok4)

    return ok and ok2 and ok3 and ok4


def test_islands():
    isl = make_islands(500, 500 - round(500 * 0.95), 100)
    # island_size is a floor, not a target: M = n_main // island_size, so every
    # main island is >= island_size and none falls under the discrimination limit
    ok = sum(isl) == 500 and isl[-1] == 25 and all(n >= 100 for n in isl[:-1])
    _report("islands (500, alpha=.05, size=100)", ok, str(isl))

    isl2 = make_islands(500, 25, None)
    ok2 = isl2 == (475, 25)
    _report("island_size=None gives the no-islands ablation", ok2, str(isl2))
    return ok and ok2


def test_bsmc_reduces_to_mppi():
    mppi = MPPI_Jax_Debug(X_D, U_D, dynamics, None, cost, bound, CV, **COMMON)
    bsmc = BSMC_Jax(
        X_D, U_D, dynamics, None, cost, bound, CV,
        blocks=(T,), evolve_rule="none", evolve_window=0, resample=None, **COMMON,
    )

    u_m, _, _ = mppi.run_mpc(X0)
    u_b = bsmc.run_mpc(X0)
    d0 = float(jnp.max(jnp.abs(u_m - u_b)))

    # second solve exercises the warm-start shift and the key advance
    u_m2, _, _ = mppi.run_mpc(X0 + 0.01)
    u_b2 = bsmc.run_mpc(X0 + 0.01)
    d1 = float(jnp.max(jnp.abs(u_m2 - u_b2)))

    ok = d0 < 1e-5 and d1 < 1e-5
    return _report("BSMC(blocks=(T,), no evolve, no resample) == MPPI", ok, f"|du| {d0:.2e} {d1:.2e}")


def test_bsmc_blocks_no_resample_equals_mppi():
    """Block partition alone must not change the answer."""
    mppi = MPPI_Jax_Debug(X_D, U_D, dynamics, None, cost, bound, CV, **COMMON)
    bsmc = BSMC_Jax(
        X_D, U_D, dynamics, None, cost, bound, CV,
        blocks=(8, 8, 8), evolve_rule="none", evolve_window=0, resample=None, **COMMON,
    )
    u_m, _, _ = mppi.run_mpc(X0)
    u_b = bsmc.run_mpc(X0)
    d = float(jnp.max(jnp.abs(u_m - u_b)))
    return _report("BSMC blocking alone is a no-op vs MPPI", d < 1e-5, f"|du| {d:.2e}")


def test_rbr_reduces_to_mppi():
    mppi = MPPI_Jax_Debug(X_D, U_D, dynamics, None, cost, bound, CV, **COMMON)
    rbr = RBR_Jax(X_D, U_D, dynamics, None, cost, bound, no_violation, CV, **COMMON)
    u_m, _, _ = mppi.run_mpc(X0)
    u_r = rbr.run_mpc(X0)
    d = float(jnp.max(jnp.abs(u_m - u_r)))
    return _report("RBR with violation==0 reduces to MPPI", d < 1e-5, f"|du| {d:.2e}")


def test_rbr_live():
    rbr = RBR_Jax_Debug(X_D, U_D, dynamics, None, cost, bound, violation, CV, **COMMON)
    x = jnp.array([0.0, 1.0, 0.2, 6.0])  # drifting off-track; some rollouts survive
    u, xhist, vhist, diag = rbr.run_mpc(x)
    n_unsafe = np.asarray(diag["n_unsafe"])
    ok = (
        np.isfinite(np.asarray(u)).all()
        and xhist.shape == (K, T, X_D)
        and vhist.shape == (K, T, U_D)
        and n_unsafe.max() > 0
    )
    return _report(
        "RBR runs and the indicator fires",
        ok,
        f"max unsafe {int(n_unsafe.max())}/{K}, all-unsafe steps {int(diag['n_all_unsafe'])}",
    )


def test_evolution_rules():
    ok_all = True
    rules = [
        ("mh", "mh", {}),
        ("de", "greedy", {}),
        ("de_best", "greedy", {}),
        ("soft", "greedy", {"temp_scale": 1.0, "eta": 0.5, "sigma_scale": 0.25}),
        ("soft_de", "greedy", {"temp_scale": 1.0, "eta": 0.5}),
        ("pso", "greedy", {}),
    ]
    for rule, acc, ekw in rules:
        c = BSMC_Jax_Debug(
            X_D, U_D, dynamics, None, cost, bound, CV,
            blocks=(8, 8, 8), evolve_window=1, evolve_rule=rule, accept=acc,
            evolve_kwargs=ekw,
            prior_ratio=(rule == "mh"), resample="systematic", island_size=50,
            **COMMON,
        )
        u, xhist, vhist, diag = c.run_mpc(X0)
        acc_rate = np.asarray(diag["accept"])
        dcost = np.asarray(diag["d_cost"])
        n_anc = int((np.asarray(diag["unique_anc0"]) >= 0).sum())
        # greedy acceptance can only improve the window; MH may accept a worsening
        # move, which is the entire point of the correction
        improves = (dcost <= 1e-6).all() if acc == "greedy" else True
        ok = np.isfinite(np.asarray(u)).all() and acc_rate.size == 3 and improves
        extra = ""
        if "evolve_attract_ess" in diag:
            extra = f", attract ESS {np.asarray(diag['evolve_attract_ess']).mean():.1f}"
        if "evolve_attract_uniq" in diag:
            extra += f", uniq attractors {np.asarray(diag['evolve_attract_uniq']).mean():.0f}"
        ok_all &= _report(
            f"evolve rule {rule!r}",
            ok,
            f"accept {acc_rate.round(2).tolist()}, mean dS {dcost.round(1).tolist()}, "
            f"uniq anc {n_anc}, ESS {float(diag['ess_final']):.1f}{extra}",
        )
    return ok_all


def test_soft_temperature_limits():
    """
    The informed attractor inherits MPPI's own temperature pathology in
    miniature: sharp -> single best -> collapse, soft -> island mean -> nominal.
    attract_ess is the knob's health check, and it must move monotonically.
    """
    out = {}
    for ts in (0.05, 1.0, 50.0):
        c = BSMC_Jax_Debug(
            X_D, U_D, dynamics, None, cost, bound, CV,
            blocks=(8, 8, 8), evolve_window=1, evolve_rule="soft_de",
            accept="greedy", prior_ratio=False, resample="systematic",
            island_size=50, evolve_kwargs={"temp_scale": ts, "eta": 0.5}, **COMMON,
        )
        _, _, _, diag = c.run_mpc(X0)
        out[ts] = (
            float(np.asarray(diag["evolve_attract_ess"]).mean()),
            float(np.asarray(diag["evolve_attract_uniq"]).mean()),
        )
    esss = [out[t][0] for t in (0.05, 1.0, 50.0)]
    ok = esss[0] < esss[1] < esss[2]
    return _report(
        "attract_ess increases monotonically with temp_scale",
        ok,
        ", ".join(f"ts={t}: ESS {out[t][0]:.1f} uniq {out[t][1]:.0f}" for t in out),
    )


def test_window_depth():
    """W=2 must re-evolve the older block; cost should not get worse than W=1."""
    out = {}
    for W in (0, 1, 2):
        c = BSMC_Jax_Debug(
            X_D, U_D, dynamics, None, cost, bound, CV,
            blocks=(8, 8, 8), evolve_window=W, evolve_rule="de", accept="greedy",
            prior_ratio=False, resample="systematic", island_size=50, **COMMON,
        )
        u, _, _, diag = c.run_mpc(X0)
        out[W] = float(diag["ess_final"])
    return _report("evolve_window 0/1/2 all run", True, f"final ESS {out}")


def test_tail_not_gathered():
    """
    The failure mode splice() exists to prevent: if the noise tail were gathered
    along with the prefix, duplicated lineages would stay bit-identical for the
    rest of the horizon -- ESS would look repaired while the effective population
    had collapsed to the number of distinct ancestors.
    """
    c = BSMC_Jax_Debug(
        X_D, U_D, dynamics, None, cost, bound, CV,
        blocks=(8, 8, 8), evolve_window=0, evolve_rule="none",
        resample="systematic", island_size=50, **COMMON,
    )
    u, _, vhist, diag = c.run_mpc(X0)
    n_anc = int((np.asarray(diag["unique_anc0"]) >= 0).sum())

    tails = np.asarray(vhist[:, -1, :])            # controls at the final step
    n_tail = len(np.unique(np.round(tails, 9), axis=0))
    prefix = np.asarray(vhist[:, 0, :])            # controls at the first step
    n_prefix = len(np.unique(np.round(prefix, 9), axis=0))

    ok = n_tail == K and n_prefix < K and n_anc < K
    return _report(
        "splice gathers the prefix but never the tail",
        ok,
        f"distinct tails {n_tail}/{K}, distinct prefixes {n_prefix}/{K}, uniq anc {n_anc}",
    )


def test_no_resample_path():
    c = BSMC_Jax_Debug(
        X_D, U_D, dynamics, None, cost, bound, CV,
        blocks=(8, 8, 8), evolve_window=1, evolve_rule="de", accept="greedy",
        prior_ratio=False, resample=None, island_size=50, **COMMON,
    )
    u, _, _, diag = c.run_mpc(X0)
    n_anc = int((np.asarray(diag["unique_anc0"]) >= 0).sum())
    ok = np.isfinite(np.asarray(u)).all() and n_anc == K
    return _report("resample=None keeps every lineage", ok, f"uniq anc {n_anc}/{K}")


def test_closed_loop_toy():
    """Fifty control steps of each controller on the toy plant; must stay finite."""
    results = {}
    cfgs = {
        "MPPI": lambda: MPPI_Jax_Debug(X_D, U_D, dynamics, None, cost, bound, CV, **COMMON),
        "RBR": lambda: RBR_Jax(X_D, U_D, dynamics, None, cost, bound, violation, CV, **COMMON),
        "BSMC": lambda: BSMC_Jax(
            X_D, U_D, dynamics, None, cost, bound, CV, blocks=(8, 8, 8),
            evolve_window=1, evolve_rule="de", accept="greedy", prior_ratio=False,
            resample="systematic", island_size=50, **COMMON,
        ),
    }
    ok_all = True
    for name, make in cfgs.items():
        mpc = make()
        x = X0
        total = 0.0
        for i in range(50):
            out = mpc.run_mpc(x)
            u = out[0] if isinstance(out, tuple) else out
            x = x + dynamics(x, u) * STEP
            total += float(cost(x, u, 0))
        results[name] = total / 50.0
        ok_all &= _report(f"closed-loop toy {name}", np.isfinite(x).all(), f"mean cost {total/50:.3f}")
    print(f"       mean running cost: {  {k: round(v, 3) for k, v in results.items()} }")
    return ok_all


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    results = [
        test_resamplers(),
        test_islands(),
        test_bsmc_reduces_to_mppi(),
        test_bsmc_blocks_no_resample_equals_mppi(),
        test_rbr_reduces_to_mppi(),
        test_rbr_live(),
        test_evolution_rules(),
        test_soft_temperature_limits(),
        test_tail_not_gathered(),
        test_window_depth(),
        test_no_resample_path(),
        test_closed_loop_toy(),
    ]
    print(f"\n{sum(results)}/{len(results)} groups passed")