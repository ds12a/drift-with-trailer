from src.learning.datasets.trailer_data import FeatureSpec
from src.simulation.config.trailer_bicycle_config import VehicleConfig
import jax
import jax.numpy as jnp

"""
Collected : [sin(hitch), cos(hitch), vx, vy, phi1dot, phi2dot, mu, delta, brake/accel] for timestep

Input to network: [sin(hitch), cos(hitch), vx, vy, delta, a] (at time t-H+1 to t)
Output of network:  [ax, ay, phi1dot, phi2dot]
Kinematics gives [ax, 0, phi1dot, phi2dot]
"""


V = VehicleConfig()
IN_COLS  = jnp.array([0, 1, 2, 3, 7, 8])   # sh, ch, vx, vy, delta, accel
KIN_COLS = jnp.array([0, 1, 2, 3, 6, 7, 8])  # mu for kin prior
VEL_COLS = jnp.array([2, 3])               # vx, vy -> FD -> ax, ay
YAW_COLS = jnp.array([4, 5])               # w1, w2

fzr = V.mass * 9.8 * V.lf / (
            V.lf + V.lr
        ) + V.trailer_mass * 9.8 * V.l2r * (V.lf + V.hitch_offset) / (
            (V.lf + V.lr) * (V.l2f + V.l2r)
        )


# def kin(r):
#     """
#     r: (..., 7) = [sh, ch, vx, vy, mu, delta, a] -> (..., 4) = [ax, ay, w1, w2]
#     """
#     sh, ch, vx, vy = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
#     mu = r[..., 4]
#     delta = jnp.clip(r[..., 5], -1, 1) * V.max_steer_rad
#     a = r[..., 6]
#     L1, L2 = V.lf + V.lr, V.l2f + V.l2r
#     w1 = (vx / L1) * jnp.tan(delta)
#     w2 = (vx * sh + (vy - V.hitch_offset * w1) * ch) / L2
    
#     cmd = jnp.maximum(a, 0) * V.max_accel + jnp.minimum(a, 0) * V.max_brake
#     fxr = mu * fzr * jnp.tanh(V.mass * cmd / (fzr * mu))
#     ax  = fxr / (V.mass + V.trailer_mass)  # prior is allowed to use correct v curve (in this case)
#     return jnp.stack([ax, jnp.zeros_like(ax), w1, w2], -1)

def kin_zeros(r):
    sh, ch, vx, vy = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    mu = r[..., 4]
    delta = jnp.clip(r[..., 5], -1, 1) * V.max_steer_rad
    a = r[..., 6]
    return jnp.zeros((*r.shape[:-1], 4))


def make_spec(H=4, dt=0.05, train_frac=0.7, split_seed=137, tag="kin-vy"):
    F = 1

    @jax.jit
    @jax.vmap
    def in_fn(w):                       # (H+F, 9) -> (H*6,)
        return w[:H][:, IN_COLS].reshape(-1)

    @jax.jit
    @jax.vmap
    def out_fn(w):                      # (H+F, 9) -> (4,)
        k, kp = w[H - 1], w[H]
        acc = (kp[VEL_COLS] - k[VEL_COLS]) / dt
        return jnp.concatenate([acc, k[YAW_COLS]]) # - kin(k[KIN_COLS])

    return FeatureSpec(in_fn, out_fn, H, F, train_frac, split_seed,
                       f"v2-{tag}-H{H}-dt{dt}")

RAW_FS = make_spec(H=16)