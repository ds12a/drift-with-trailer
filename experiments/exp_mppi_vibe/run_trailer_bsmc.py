"""
Closed-loop Block-SMC MPPI with per-block particle evolution.

    python3 -m experiments.exp_mppi_vibe.run_trailer_bsmc

Regression / step-0 probe
-------------------------
`blocks=(T,)`, `evolve_rule="none"`, `evolve_window=0`, `resample=None` is
numerically MPPI, and in that configuration the debug diag still reports
`sigma_S`, `ess_final` and `w_max` per control step. That is the rung-0
instrument at zero extra code: run it forward and reverse, plot ESS against
track position on a successful and a failing run, and if ESS is flat across both
then weight degeneracy is not the binding constraint and the direction stops.

Temperature
-----------
lambda is dimensionful -- it lives in cost units -- so the BeamNG value does not
port directly. Set it from the measured cost landscape instead. Run the probe
config below and read `dS_p10` from the diag: that is how far the tenth
percentile of the population sits above the best particle, and it is the number
that governs weight degeneracy. The population std is *not* usable here, because
the violation tail swamps it.

    inverse_temp ~ dS_p10 / 2        top decile at non-negligible weight
    inverse_temp < dS_p10 / 87       float32 hard zeros, i.e. pure argmin

Measured at the track-start reverse IC (K=500, T=55, v_target=-25/3.6, the
weights below): dS_p10 = 269, so lambda ~ 134 -- against exp_004's 0.5, which is
two orders of magnitude into the argmin regime. Confirmed by sweeping: at
lambda=0.5 per-block ESS is 1.0 at every checkpoint and only 5 ancestral
lineages survive to the end (one per island, i.e. degeneracy relocated rather
than solved, kill condition 3); at lambda=134 per-block ESS runs 50-270 and 53
lineages survive. Re-measure per cost configuration -- the number moves with
the weights, the speed, and the track width.

Attraction temperature
----------------------
The informed rules weight the island by exp(-(S_win - min)/(inverse_temp *
temp_scale)), which inherits MPPI's own pathology in miniature: sharp collapses
onto the single best particle, soft averages to the island mean, which is the
nominal. `evolve_attract_ess` in the diag is the health check. Measured on the
toy plant at K/island = 50: temp_scale 0.05 -> attract ESS 1.6 with 10 distinct
attractors, 1.0 -> 33 with 104, 50.0 -> 50 (uniform) with 124.

On the real reverse IC at lambda = 134, K/island = 119, W = 1, all rules
accepting greedily:

    rule          accept   mean dS    attrESS  attrUniq  uniq_anc  final ESS
    none            --        --         --       --        38       101
    de             0.49    -6.9e7        --       --        57       175
    de_best        0.60    -5.8e7        --       --        84       161
    soft           0.68    -3.0e6       32.9      --        80       177
    soft_de        0.56    -5.5e7       31.8      188        59        86
    soft_de@0.1    0.59    -6.6e8        2.9       27        82       135

The dS column is dominated by violation-penalty repair (1e12 scale), i.e. the
move is annealing infeasible particles back inside the track -- which is the
feasibility-repair behaviour the design is for, showing up as a measurement
rather than an assumption.

viol_weight
-----------
exp_004 uses 1e12 against BeamNG's 1e5. At a sane lambda a 1e12 penalty flushes
to hard zero instantly, so any block containing a violation is pruned
absolutely and the soft-weighting claim is vacuous in exactly the blocks where
it should matter. Scale it toward 87*lambda once lambda is set, and hold it
identical across every controller in the comparison.
"""

import jax.numpy as jnp

from experiments.exp_mppi_vibe.utils.trailer_driver import run_mpc

# Rev Args -- T = 55 = 5 blocks of 11
ctl_args = (
    jnp.diag(jnp.array([1e-2, 0.2])),
)
ctl_kwargs = {
    "inverse_temp": 150,
    "K": 500,
    "step": 0.05,
    "T": 80,
    "alpha": 0.05,
    "gamma": 0.0,
    # Block lengths must sum to T. Lower bound on a block: long enough that
    # within-block cost spread exceeds cost noise (trajectory separation, ~1-2 s
    # at trailer speeds, i.e. 20-40 steps). Upper bound: short enough that
    # accumulated dS stays under 87*lambda. Both scale with 1/lambda_plant, so
    # uniform spacing is right at constant speed.
    "blocks": (40, 20, 20),
    "evolve_window": 1,          # W blocks back; costs (1+W)x vanilla MPPI
    # none | mh | de | de_best | soft | soft_de | pso.  soft_de is the informed
    # default: each particle draws its own attractor from the island softmax
    # over window cost, so it moves toward good regions rather than toward the
    # single best. temp_scale multiplies inverse_temp for that attraction only.
    "evolve_rule": "pso",
    "evolve_kwargs": {"eta": 0.5, "F": 0.6, "CR": 0.9, "temp_scale": 10},
    "accept": "greedy",          # "mh" only means MH under evolve_rule="mh"
    "prior_ratio": False,
    "resample": "systematic",    # unconditional; identity at uniform weights
    "island_size": 100,
}
cost_kwargs = {
    "reverse": False,
    "v_target": 40,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 0,
    "c_weight": 1e2,
    "a_weight": 1e2,
    "viol_weight": 1e8,
}

# Step-0 probe / bit-regression against MPPI: swap this in wholesale.
probe_ctl_kwargs = {
    **ctl_kwargs,
    "blocks": (55,),
    "evolve_window": 0,
    "evolve_rule": "none",
    "resample": None,
}

if __name__ == "__main__":

    run_mpc(
        "BSMC",
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
        diag_file="experiments/exp_mppi_vibe/diag/bsmc_rev.csv",
    )