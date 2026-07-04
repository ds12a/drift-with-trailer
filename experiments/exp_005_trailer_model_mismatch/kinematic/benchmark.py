from experiments.exp_005_trailer_model_mismatch.kinematic.trailer_driver import run_mpc
import jax.numpy as jnp

from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig, 
    VehicleConfig, 
    TrackConfig, 
    SimulationConfig,
)



scenario = TrailerBicycleEnvConfig(
    "scenario", TrackConfig(), VehicleConfig(), SimulationConfig()
)
scenario.track.friction_csv = "src/simulation/assets/tracks/oval_ice.csv"

env_kwargs = {
    "renderer": "pybullet",
    "render_mode": "rgb_array_birds_eye",
    "render_width": 600,
    "render_height": 400,
    "scenario": scenario
}

# Configs to be used

# Fwd
mppi_cfg = (
    (
       jnp.diag(jnp.array([3e-3, 0.2])),
    ),
    {
        "inverse_temp": 1,
        "K": 500,
        "step": 0.05,
        "T": 80,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": 25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "c_weight": 1e0,
        "a_weight": 7e2,
    },
)

smppi_cfg = (
    (
        jnp.diag(jnp.array([4e-2, 0.2])),
        jnp.diag(jnp.array([1e-2, 1e-1])),
    ),
    {
        "inverse_temp": 0.25,
        "K": 500,
        "step": 0.05,
        "T": 85,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": 25,
        "p_weight": 1e2,
        "p_slow_weight": 1e0,
        "c_weight": 1e0,
        "a_weight": 7e2,
    },
)

# Rev
# mppi_cfg = (
#     (
#         jnp.diag(jnp.array([3e-3, 0.2])),
#     ),
#     {
#         "inverse_temp": 0.5,
#         "K": 750,
#         "step": 0.05,
#         "T": 55,
#         "alpha": 0.05,
#     },
#     {
#         "reverse": False, 
#         "v_target": -25,
#         "p_weight": 1e2,
#         "p_slow_weight": 1e0,
#         "c_weight": 1e-2,
#         "a_weight": 1e2,
#     },
# )

# smppi_cfg = (
#     (
#         jnp.diag(jnp.array([3e-2, 0.2])),
#         jnp.diag(jnp.array([1e-2, 1e-1])),
#     ),
#     {
#         "inverse_temp": 0.5,
#         "K": 750,
#         "step": 0.05,
#         "T": 55,
#         "alpha": 0.05,
#     },
#     {
#         "reverse": False, 
#         "v_target": -25,
#         "p_weight": 1e2,
#         "p_slow_weight": 1e0,
#         "c_weight": 1e1,
#         "a_weight": 1e2,
#     },
# )

cfg = [smppi_cfg, mppi_cfg]


if __name__ == "__main__":


    trials = [
        # ("MPPI For.; no v bound", "SMPPI For.;no v bound", False, None),
        # ("MPPI Rev.; no v bound", "SMPPI Rev.;no v bound", True, None),
        ("SMPPI For.;  v_t = 60 ", "MPPI For.; v_t = 60 ", False, 60),
        ("SMPPI For.;  v_t = 90", "MPPI For.; v_t = 90", False, 90),
        # ("SMPPI Rev.; v_t = 60 ", "MPPI Rev.;  v_t = 60 ", False, -60),
        # ("SMPPI Rev.; v_t = 90 ", "MPPI Rev.;  v_t = 90 ", False, -90),
    ]

    for config_mppi, config_smppi, reverse, v_t in trials:
        
        name = [config_mppi, config_smppi]

        for i in range(2):

            v_tt = None if v_t is None else v_t / 3.6
            
            ctl_args, ctl_kwargs, cost_kwargs = cfg[i]
            cost_kwargs["reverse"] = reverse
            cost_kwargs["v_target"] = v_tt

            mode = "MPPI" if i == 1 else "SMPPI"

            run_mpc(
                mode,
                ctl_args,
                ctl_kwargs,
                cost_kwargs,
                record=True,
                benchmark=True,
                quiet=True,
                debug=True,
                max_steps=3000,
                print_name=name[i],
                record_file_name=f"{mode}_v={str(v_t)}_r={reverse}",
                env_kwargs=env_kwargs
            )