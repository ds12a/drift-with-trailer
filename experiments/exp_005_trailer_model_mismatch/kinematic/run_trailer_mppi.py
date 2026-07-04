from experiments.exp_005_trailer_model_mismatch.kinematic.trailer_driver import run_mpc
import jax.numpy as jnp
# Fwd Args (max 25)
# ctl_args = (
#     jnp.diag(jnp.array([3e-3, 0.2])),
# )
# ctl_kwargs = {
#     "inverse_temp": 1,
#     "K": 500,
#     "step": 0.05,
#     "T": 80,
#     "alpha": 0.05,
# }
# cost_kwargs = {
#     "reverse": False, 
#     "v_target": 25,
#     "p_weight": 1e2,
#     "p_slow_weight": 1e0,
#     "c_weight": 1e-2,
#     "a_weight": 1e2,
# }


# Rev Args
ctl_args = (
    jnp.diag(jnp.array([3e-3, 0.2])),
)
ctl_kwargs = {
    "inverse_temp": 0.5,
    "K": 750,
    "step": 0.05,
    "T": 55,
    "alpha": 0.05,
}
cost_kwargs = {
    "reverse": False, 
    "v_target": -25,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "c_weight": 1e-2,
    "a_weight": 1e2,
}


if __name__ == "__main__":
        
    run_mpc(
        "MPPI",
        ctl_args,
        ctl_kwargs,
        cost_kwargs,
        record=True,
        debug=True,
        quiet=False,
        benchmark=False,
        headless=False,
        env_kwargs=None,
    )