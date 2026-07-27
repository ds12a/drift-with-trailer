from src.learning.datasets.trailer_data import FeatureSpec
from src.simulation.config.trailer_bicycle_config import VehicleConfig
import jax
import jax.numpy as jnp

"""
Collected : [sin(hitch), cos(hitch), vx, vy, phi1dot, phi2dot, mu, delta, brake/accel] per timestep

Network input : [sh, ch, vx, vy, phi1dot, phi2dot] (rows t-H+1..t) + [delta, a] (rows t-H+2..t+1)
Network output: [ax, ay, alpha1, alpha2] 
Kinematics gives [ax, 0, 0, 0] 
"""

V = VehicleConfig()

IN_COLS  = jnp.array([0, 1, 2, 3, 4, 5, 7, 8])  # sh, ch, vx, vy, w1, w2, delta, accel
VEL_COLS = jnp.array([2, 3, 4, 5])              # FD of vx, vy, w1, w2  -> ax, ay, alpha1, alpha2

S_COLS = jnp.array([0, 1, 2, 3, 4, 5])          # state features incl. yaw rates
C_COLS = jnp.array([7, 8])                       # delta, acc

def kin_zeros(r):
    return jnp.zeros((*r.shape[:-1], 4))

def make_spec(H=4, dt=0.05, train_frac=0.7, split_seed=137, tag="res-accel"):
    F = 1

    @jax.jit
    @jax.vmap
    def in_fn(w):                            # (H+F, 9) -> (H*8,)
        s = w[:H][:, S_COLS]                 # states, rows 0..H-1
        c = w[1:H + 1][:, C_COLS]            # controls, rows 1..H  (driving control)
        return jnp.concatenate([s, c], axis=-1).reshape(-1)

    @jax.jit
    @jax.vmap
    def out_fn(w):                          
        k, kp = w[H - 1], w[H]
        return (kp[VEL_COLS] - k[VEL_COLS]) / dt # [ax, ay, alpha1, alpha2]
    
    return FeatureSpec(
        in_fn, out_fn, H, F, train_frac, split_seed,
        f"v3-{tag}-in{len(IN_COLS)}-H{H}-dt{dt}",
    )

RAW_FS = make_spec(H=4)