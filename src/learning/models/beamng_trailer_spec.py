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

def fiala_dyn(r):
    def slip_angle(v_lon, v_lat, eps=0.5):
        return -jnp.arctan2(v_lat, jnp.maximum(jnp.abs(v_lon), eps))
    mu = 1.0  # Assumption for prior

    sh, ch, v_1x, v_1y, phi_1_dot, phi_2_dot, steer_cmd, accel_cmd = r
    vehicle = V
    steer_cmd = -jnp.clip(steer_cmd, -1.0, 1.0)  # BeamNG convention is opposite
    accel_cmd = jnp.clip(accel_cmd, -1.0, 1.0)

    throttle = jnp.maximum(accel_cmd, 0.0)
    brake = -jnp.minimum(accel_cmd, 0.0)
    
    # Steer
    delta = steer_cmd * vehicle.max_steer_rad
    cd = jnp.cos(delta)
    sd = jnp.sin(delta)

    # Hitch
    sa = sh
    ca = ch

    v_2x = v_1x * ca - (v_1y - phi_1_dot * vehicle.hitch_offset) * sa
    v_2y = v_1x * sa + (v_1y - phi_1_dot * vehicle.hitch_offset) * ca - vehicle.l2f * phi_2_dot

    # Original formulas do not work when the trailer drives backwards
    v_yf = v_1y + vehicle.lf * phi_1_dot
    v_yr = v_1y - vehicle.lr * phi_1_dot
    v_2y_wheel = v_2y - vehicle.l2r * phi_2_dot
    alpha_f = slip_angle(v_1x * cd + v_yf * sd, -v_1x * sd + v_yf * cd)
    alpha_r = slip_angle(v_1x, v_yr)
    alpha_t = slip_angle(v_2x, v_2y_wheel)

    fzf = vehicle.mass * 9.8 * vehicle.lr / (
        vehicle.lf + vehicle.lr
    ) + vehicle.trailer_mass * 9.8 * vehicle.l2r * (vehicle.lr - vehicle.hitch_offset) / (
        (vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r)
    )

    F_1yf = -compute_fy(
        alpha_f,
        vehicle.cornering_stiffness_front,
        fzf,
        0,
        mu,
        vehicle.gamma,
    )

    fzr = vehicle.mass * 9.8 * vehicle.lf / (
        vehicle.lf + vehicle.lr
    ) + vehicle.trailer_mass * 9.8 * vehicle.l2r * (vehicle.lf + vehicle.hitch_offset) / (
        (vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r)
    )
    commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake
    fxr = mu * fzr * jnp.tanh(vehicle.mass * commanded / (fzr * mu))

    F_1yr = -compute_fy(alpha_r, vehicle.cornering_stiffness_rear, fzr, fxr, mu, vehicle.gamma)


    fzr_trailer = vehicle.trailer_mass * 9.8 * vehicle.l2f / (vehicle.l2f + vehicle.l2r)
    F_2yr = -compute_fy(
        alpha_t, vehicle.cornering_stiffness_trailer, fzr_trailer, 0, mu, vehicle.gamma
    )

    total_mass = vehicle.mass + vehicle.trailer_mass
    alpha_dot = phi_1_dot - phi_2_dot

    A = jnp.array(
        [
            [total_mass, 0, 0, -vehicle.trailer_mass * vehicle.l2f * sa],
            [
                0,
                total_mass,
                -vehicle.trailer_mass * vehicle.hitch_offset,
                -vehicle.trailer_mass * vehicle.l2f * ca,
            ],
            [
                0,
                -vehicle.trailer_mass * vehicle.hitch_offset,
                vehicle.inertia_z + vehicle.trailer_mass * vehicle.hitch_offset**2,
                vehicle.trailer_mass * vehicle.l2f * vehicle.hitch_offset * ca,
            ],
            [
                -vehicle.trailer_mass * vehicle.l2f * sa,
                -vehicle.trailer_mass * vehicle.l2f * ca,
                vehicle.trailer_mass * vehicle.l2f * vehicle.hitch_offset * ca,
                vehicle.trailer_inertia_z + vehicle.trailer_mass * vehicle.l2f**2,
            ],
        ]
    )

    b = jnp.array(
        [
            fxr
            - F_1yf * sd
            + F_2yr * sa
            + vehicle.mass * v_1y * phi_1_dot
            + vehicle.trailer_mass * phi_1_dot * (v_2y * ca - v_2x * sa)
            + vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * ca,
            F_1yr
            + F_1yf * cd
            + F_2yr * ca
            - vehicle.mass * v_1x * phi_1_dot
            - vehicle.trailer_mass * phi_1_dot * (v_2x * ca + v_2y * sa)
            - vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * sa,
            -F_1yr * vehicle.lr
            + F_1yf * cd * vehicle.lf
            - vehicle.hitch_offset * F_2yr * ca
            + vehicle.trailer_mass * vehicle.hitch_offset * phi_1_dot * (v_2x * ca + v_2y * sa)
            + vehicle.trailer_mass
            * vehicle.hitch_offset
            * vehicle.l2f
            * alpha_dot
            * phi_2_dot
            * sa,
            -(vehicle.l2f + vehicle.l2r) * F_2yr
            + vehicle.trailer_mass * vehicle.l2f * v_2x * phi_1_dot,
        ]
    )

    v_1x_dot, v_1y_dot, phi_1_ddot, phi_2_ddot = jnp.linalg.solve(A, b)

    # Track v for efficiency
    return jnp.stack([v_1x_dot, v_1y_dot, phi_1_ddot, phi_2_ddot], -1)


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

        prior = fiala_dyn(k[X_COLS])

        return (kp[FD_COLS] - k[FD_COLS]) / dt - jnp.concatenate([prior, jnp.array([0, 0])])

    return FeatureSpec(in_fn, out_fn, H, F, train_frac, split_seed, f"v2-{tag}-H{H}-dt{dt}")


H = 8
STATE_FS = make_main_spec(H=H)