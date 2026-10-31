import os

# JAX is stupid
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import logging
import cv2
import numpy as np
import time
import jax

import jax.numpy as jnp
from pathlib import Path
from dataclasses import astuple
from flax import nnx
import absl.logging
import orbax.checkpoint as ocp  # this breaks beamng logging

from src.learning.datasets.trailer_data import DataCollector, DataStore
from src.simulation.beamng_trailer_env import BeamNGTrailerEnv, VehicleState, bng_pickup_trailer_cfg
from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from experiments.exp_008_beamng.util_fns import gen_util_funs as sin_fn
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as straight_fn
from src.simulation.config.trailer_beamng_config import (
    BeamNGTrailerEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
from gymnasium.wrappers import RecordVideo
import json


from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import STATE_FS, fiala_dyn, IN_COLS
from src.learning.models.beamng_model_spec import STATE_FS
from src.dynamics.trailer.beamng_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)
from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs as prior_util

import logging

# Orbax is stupid
absl.logging.set_verbosity(absl.logging.WARNING)
logging.getLogger("beamngpy").setLevel(logging.WARNING)
logging.getLogger("beamngpy").propagate = False



load1 = DataStore.load(Path("experiments/exp_008_beamng/data_trial3_aug1.npz"))
load2 = DataStore.load(Path("experiments/exp_008_beamng/data_trial2_aug3.npz"))
print(load1.data.shape)
print(load2.data.shape)
load1.ingest_ds(load2)
print(load1.data.shape)
load1.save("experiments/exp_008_beamng/data_trial23.npz")