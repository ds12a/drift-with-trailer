# import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from src.learning.datasets.trailer_data import FeatureSpec
from src.simulation.config.trailer_beamng_config import VehicleConfig
import jax
import jax.numpy as jnp

from src.simulation.beamng_trailer_env import bng_pickup_trailer_cfg


"""
Spec(s) for the BeamNG integration. Main architectural point is the realized controls
are in state and distinguished from commanded control.

Denote 
 - vehicle state as x_s = [sin(hitch), cos(hitch), vx, vy, phi1dot, phi2dot]
 - state controls as x_u = [d, a]
 - input controls as u = [d_cmd, a_cmd]

Pipeline:
 - [x_s, x_u, u] -> d/dt [x_s, x_u]

Collected state: [sh, ch, vx, vy, phi1dot, phi2dot, mu, d_state, a_state, d_cmd, a_cmd]
"""

# Note: the old control-shift bug is no longer acceptable as u_t+1 is actually needed now

V = bng_pickup_trailer_cfg

XS_COLS = jnp.array([0, 1, 2, 3, 4, 5])  # sh, ch, vx, vy, phi1dot, phi2dot
XU_COLS = jnp.array([7, 8])
U_COLS = jnp.array([9, 10])

X_COLS = jnp.array([0, 1, 2, 3, 4, 5, 7, 8])  # sh, ch, vx, vy, phi1dot, phi2dot, delta, accel
IN_COLS = jnp.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])

FD_COLS = jnp.array([2, 3, 4, 5, 7, 8])

fzr = V.mass * 9.8 * V.lf / (V.lf + V.lr) + V.trailer_mass * 9.8 * V.l2r * (
    V.lf + V.hitch_offset
) / ((V.lf + V.lr) * (V.l2f + V.l2r))

def compute_fy(alpha, cc, fz, fx, mu, gamma):
    fy_max = jnp.sqrt(jnp.maximum((mu * fz) ** 2 - gamma * fx**2, 0))

    alpha_sl = jnp.arctan2(3 * fy_max, cc)

    return jnp.where(
        jnp.abs(alpha) < alpha_sl,
        (
            -cc * jnp.tan(alpha)
            + (cc**2 / (3 * fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
            - (cc**3 / (27 * fy_max**2)) * jnp.tan(alpha) ** 3
        ),
        -fy_max * jnp.sign(alpha),
    )

def make_main_spec(H=4, dt=0.05, train_frac=0.7, split_seed=137, tag="fiala"):
    F = 1

    @jax.jit
    @jax.vmap
    def in_fn(w):                            # (H+F, 9) -> (H*8,)
        return (w[:H][:, IN_COLS]).reshape(-1)
        # return win_proc.at[-4].set(-win_proc[-4])

    @jax.jit
    @jax.vmap
    def out_fn(w):  # (H+F, 9) -> (4,)
        k, kp = w[H - 1], w[H]
        # k = k.at[-4].set(-k[-4])
        # kp = kp.at[-4].set(-kp[-4])

        # prior = fiala_dyn(k[X_COLS])

        return (kp[FD_COLS] - k[FD_COLS]) / dt # - jnp.concatenate([prior, jnp.array([0, 0])])

    return FeatureSpec(in_fn, out_fn, H, F, train_frac, split_seed, f"v2-{tag}-H{H}-dt{dt}")


H = 4
STATE_FS = make_main_spec(H=H)