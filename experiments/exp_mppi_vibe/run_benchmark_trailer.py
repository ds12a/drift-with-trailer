"""
Matched-budget, paired-seed sweep across MPPI / RBR / BSMC.

    python3 -m experiments.exp_mppi_vibe.run_benchmark_trailer

Budget accounting
-----------------
BSMC with `evolve_window=W` costs (1+W)*K*T dynamics evaluations, so the honest
baseline is vanilla MPPI at (1+W)*K samples -- K_MPPI below is set from K_BSMC
and W rather than typed in, so the two cannot drift apart. RBR costs the same
K*T as vanilla plus one constraint query per step.

Seeds are threaded through `run_sweep` (exp_004's version looped against a
hardcoded key(0), so every "seed" was the same trajectory). The plant is
unstable, so the run-to-run spread is itself a result: `dist_sd` is reported
alongside the mean and widening spread with speed is the finding, not noise to
be averaged away.
"""

import jax.numpy as jnp

from experiments.exp_mppi_vibe.utils.run_eval import print_table, run_sweep

K_BSMC = 500
W = 1
K_MPPI = K_BSMC * (1 + W)          # matched dynamics-evaluation budget

T_FWD, T_REV = 80, 55
N_SEEDS = 20
MAX_STEPS = 2000

CV = jnp.diag(jnp.array([3e-3, 0.2]))

# Held identical across controllers -- differences must come from the sampler.
LAMBDA_FWD, LAMBDA_REV = 1.0, 0.5
VIOL_WEIGHT = 1e12

cost_fwd = {
    "reverse": False,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 2e2,
    "c_weight": 1e0,
    "a_weight": 7e2,
    "viol_weight": VIOL_WEIGHT,
}
cost_rev = {
    "reverse": False,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 1e2,
    "c_weight": 1e-2,
    "a_weight": 1e2,
    "viol_weight": VIOL_WEIGHT,
}


def _base(K, T, lam):
    return {"inverse_temp": lam, "K": K, "step": 0.05, "T": T, "alpha": 0.05, "gamma": 0.0}


def _blocks(T, n):
    """n blocks over T steps, remainder spread over the leading blocks."""
    base, rem = divmod(T, n)
    return tuple(base + (1 if i < rem else 0) for i in range(n))


def configs(T, lam):
    """(label, controller, ctl_kwargs) at matched budget."""
    return [
        ("MPPI", "MPPI", _base(K_MPPI, T, lam)),
        ("RBR", "RBR", _base(K_MPPI, T, lam)),
        (
            "BSMC",
            "BSMC",
            {
                **_base(K_BSMC, T, lam),
                "blocks": _blocks(T, 5),
                "evolve_window": W,
                "evolve_rule": "soft_de",
                "evolve_kwargs": {"eta": 0.5, "temp_scale": 0.1},
                "accept": "greedy",
                "prior_ratio": False,
                "resample": "systematic",
                "island_size": 100,
            },
        ),
        # ablations: uninformed update, islands off, evolution off, selection off
        (
            "BSMC-de",
            "BSMC",
            {
                **_base(K_BSMC, T, lam),
                "blocks": _blocks(T, 5),
                "evolve_window": W,
                "evolve_rule": "de",
                "accept": "greedy",
                "prior_ratio": False,
                "resample": "systematic",
                "island_size": 100,
            },
        ),
        (
            "BSMC-1i",
            "BSMC",
            {
                **_base(K_BSMC, T, lam),
                "blocks": _blocks(T, 5),
                "evolve_window": W,
                "evolve_rule": "soft_de",
                "evolve_kwargs": {"eta": 0.5, "temp_scale": 0.1},
                "accept": "greedy",
                "prior_ratio": False,
                "resample": "systematic",
                "island_size": None,
            },
        ),
        (
            "BSMC-noev",
            "BSMC",
            {
                **_base(K_MPPI, T, lam),
                "blocks": _blocks(T, 5),
                "evolve_window": 0,
                "evolve_rule": "none",
                "resample": "systematic",
                "island_size": 100,
            },
        ),
        (
            "BSMC-nors",
            "BSMC",
            {
                **_base(K_BSMC, T, lam),
                "blocks": _blocks(T, 5),
                "evolve_window": W,
                "evolve_rule": "soft_de",
                "evolve_kwargs": {"eta": 0.5, "temp_scale": 0.1},
                "accept": "greedy",
                "prior_ratio": False,
                "resample": None,
                "island_size": 100,
            },
        ),
    ]


if __name__ == "__main__":

    trials = [
        ("fwd", [40, 60, 90], T_FWD, LAMBDA_FWD, cost_fwd),
        ("rev", [-30, -50, -70], T_REV, LAMBDA_REV, cost_rev),
    ]

    rows = []
    for tag, targets, T, lam, cost_base in trials:
        cost_list = [{**cost_base, "v_target": v / 3.6} for v in targets]

        for label, controller, ctl_kwargs in configs(T, lam):
            results = run_sweep(
                controller,
                (CV,),
                ctl_kwargs,
                cost_list,
                n_seeds=N_SEEDS,
                max_steps=MAX_STEPS,
            )
            for r in results:
                rows.append({"label": label, **r})

    print_table(rows, n_seeds=N_SEEDS)