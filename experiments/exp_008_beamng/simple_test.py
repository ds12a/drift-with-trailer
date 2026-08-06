import json
import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp

# from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
# from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.learning.models.trailer_nn import TrailerModel

# from src.learning.models.trailer_spec import KIN_FS, kin
# from src.learning.models.trailer_spec_nores import RAW_FS, kin_zeros
from src.learning.datasets.trailer_data import DataLoader, FeatureSpec
from src.dynamics.trailer.trailer_bicycle_kinematic import TrackProjection
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from src.utils.track import TrackModel
from src.learning.models.beamng_trailer_spec import (
    X_COLS,
    XS_COLS,
    XU_COLS,
    U_COLS,
    FD_COLS,
)


from src.simulation.beamng_trailer_env import BeamNGTrailerEnv, VehicleState, bng_pickup_trailer_cfg
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import STATE_FS, fiala_dyn, IN_COLS
from src.dynamics.trailer.beamng_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)

from pathlib import Path
from dataclasses import astuple
from flax import nnx
import orbax.checkpoint as ocp
import absl.logging

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False

jnp.set_printoptions(precision=2, suppress=True)

spec = STATE_FS
kin_fn = fiala_dyn
prior_fn = fiala_dyn
dt = 0.05
H = spec.H
HISTORY = H
NPZ_SAVE_HEAD = "data_proc_test2"
JSON_PTH = f"./experiments/exp_008_beamng/{NPZ_SAVE_HEAD}_stats.json"

with open(Path(JSON_PTH), "r") as f:
    norm_stats = json.load(f)
x_mean, x_std = jnp.asarray(norm_stats["x_mean"]), jnp.asarray(norm_stats["x_std"])
y_mean, y_std = jnp.asarray(norm_stats["y_mean"]), jnp.asarray(norm_stats["y_std"])

print("x_mean\n", x_mean.reshape(4, 10))
print("x_std\n", x_std.reshape(4, 10))
print("y_mean", y_mean)
print("y_std", y_std)

model = TrailerModel(spec.H * len(IN_COLS), 6)
_, state = nnx.split(model)
ckpt = ocp.StandardCheckpointer()
nnx.update(
    model,
    ckpt.restore(
        Path.cwd() / "src/learning/models/trained/beamng-l4-128-test2_best",
        state,
    ),
)

ds = np.load("experiments/exp_008_beamng/data_trial2.npz")
phi1dot_col = 4  # in FD_COLS
targets = (ds["data"][1:, FD_COLS] - ds["data"][:-1, FD_COLS]) / 0.05
print("phi1ddot mean:", targets[:, 2].mean())  # should be near zero
print("phi1ddot std:", targets[:, 2].std())
print("phi1ddot |max|:", np.abs(targets[:, 2]).max())

phi1ddot = targets[:, 2]
for thresh in [5, 10, 20, 30, 40]:
    n = (np.abs(phi1ddot) > thresh).sum()
    print(f"|phi1ddot| > {thresh}: {n} rows ({100*n/len(phi1ddot):.3f}%)")

# Reduced state  [x, y, phi1, phi2, vx, vy, phi1dot, phi2dot, delta_s, accel_s, delta_u, accel_u] for testing
def dynamics(x, u):  # passed as windows
    x_windows = x.reshape(H, 12)
    old_u = x_windows[-1][-2:]

    def row(w):
        """
        Transform from integrator/dynamics state to model state
        """
        hitch = w[2] - w[3]
        return jnp.stack(
            [
                jnp.sin(hitch),
                jnp.cos(hitch),
                w[4],  # vx
                w[5],  # vy
                w[6],  # phi1dot
                w[7],  # phi2dot
                # jnp.clip(w[8], -1, 1),  # delta_s
                # jnp.clip(w[9], -1, 1),  # accel_s
                w[8],
                w[9],
                w[10], # delta_u
                w[11], # accel_u
            ]
        )

    def prior(w):
        """
        New interface should be the prior fn gets fed the whole window and slices itself
        """
        return jnp.concatenate([prior_fn(row(w)[:-2]), jnp.array([0, 0])])

    def proc(window):
        """
        Transforms into network input form [x_s, x_u, u] * H
        """
        rows = jax.vmap(row)(window)
        return (rows.reshape(-1) - x_mean) / x_std

    # x_windows = x_windows.at[:, -3:-1].set(
    #     jnp.concatenate([x_windows[:, -3:-1], u[None, :]], axis=0)
    # )
    # TODO I'm not sure what the thing above was for
    x_windows = x_windows.at[-1, -2:].set(u)

    model_in = proc(x_windows).flatten()[None, ...]

    xpos, ypos, phi1, phi2, vx, vy, phi1dot, phi2dot, delta_s, accel_s, *_ = x_windows[-1]
    arc_len = x_windows[-1][-1]
    # pred = kin_fn(kin_in)
    pred = model(model_in)[0] * y_std + y_mean
    # pred = pred.at[:4].set(jnp.zeros(4))
    # jax.debug.print("pred: {}", pred)
    pred += prior(x_windows[-1])
    # print("prior: ", prior(x_windows[-1]))
    # jax.debug.print("pred after prior: {}", pred)

    ax, ay, phi1ddot, phi2ddot, ddelta_s, daccel_s = pred

    def clip_deriv(x, dx, dt):
        x_next = x + dx * dt
        x_next_clipped = jnp.clip(x_next, -1.0, 1.0)
        return (x_next_clipped - x) / dt

    ddelta_s = clip_deriv(delta_s, ddelta_s, dt)
    daccel_s = clip_deriv(accel_s, daccel_s, dt)
    # print(ddelta_s * dt + delta_s, daccel_s * dt + accel_s)

    # TODO maybe keep trapezoidal consistency? doesnt really matter because its beamng
    next_vx = vx + ax * dt
    next_vy = vy + ay * dt
    next_phi1dot = phi1dot + phi1ddot * dt
    next_phi2dot = phi2dot + phi2ddot * dt

    avg_vx = 0.5 * (vx + next_vx)
    avg_vy = 0.5 * (vy + next_vy)
    # avg_phi_1_dot = 0.5 * (phi1dot + next_phi1dot)
    # avg_phi_2_dot = 0.5 * (phi2dot + next_phi2dot)

    xdot = avg_vx * jnp.cos(phi1) - avg_vy * jnp.sin(phi1)
    ydot = avg_vx * jnp.sin(phi1) + avg_vy * jnp.cos(phi1)

    # print("u comp", u, old_u)
    du = (u - old_u) / dt  # Goofy

    dx = jnp.array(
        [
            xdot,
            ydot,
            next_phi1dot,
            next_phi2dot,
            ax,
            ay,
            phi1ddot,
            phi2ddot,
            ddelta_s,
            daccel_s,
            du[0],
            du[1],
        ]
    )
    # print("dx_pred:    ", dx * dt + x_windows[-1])
    # dx_history = (x_windows[1:] - x_windows[:-1]) / dt
    # dx_window = jnp.concatenate([dx_history, dx[None, :]], axis=0)
    # return dx_window.flatten()
    return dx


# Dim is 12
state = jnp.zeros(12 * 4)
state = state.at[jnp.array([4, 4+12, 4+24, 4+36])].set(3.0)
print(state[-12:])

rng = np.random.default_rng()
controls = rng.standard_normal(size=(30, 2)) * 3e-2

for i in range(30):
    
    curr_x = state[-12:]
    hitch = state[2] - state[3]
    # print("in:", jnp.array([jnp.sin(hitch), jnp.cos(hitch), *state[4:10]]))
    # print("test: ", fiala_dyn(jnp.array([jnp.sin(hitch), jnp.cos(hitch), *state[4:10]])))
    dx = dynamics(state, jnp.array(controls[i]))
    new_x = curr_x + dx * dt
    # print(state[12:].shape, new_x.shape, new_x)
    state = jnp.concatenate([state[12:], new_x])


    print("state_next: ", state[-12:])