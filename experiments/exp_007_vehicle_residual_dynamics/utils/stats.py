
import argparse
import json
import time
from dataclasses import astuple
from pathlib import Path
 
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import orbax.checkpoint as ocp
from flax import nnx
 
from src.learning.datasets.trailer_data import DataStore, DataLoaderBlocked, FeatureSpec
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.trailer_spec_nores import (
    make_spec as make_spec_live,
    kin_zeros,
    S_COLS as LIVE_S_COLS,
    C_COLS as LIVE_C_COLS,
)
from src.dynamics.trailer.model_acceleration_dynamics import (
    gen_util_funs as res_util,
    D_STATE_DIM,
    D_U_DIM,
    D_EXTRA_DIM,
)
from src.dynamics.trailer.trailer_bicycle_kinematic import gen_util_funs as kin_util
from src.controllers.mpc.debug.mppi_jax_debug import MPPI_Jax_Debug
from src.simulation.trailer_bicycle_env import TrailerBicycleEnv, VehicleState
from src.simulation.config.trailer_bicycle_config import (
    TrailerBicycleEnvConfig,
    VehicleConfig,
    TrackConfig,
    SimulationConfig,
)
 
try:
    # Optional: legacy kin-prior model, only used if a checkpoint path is supplied below.
    from src.learning.models.trailer_spec import make_spec as make_spec_legacy
except ImportError:
    make_spec_legacy = None
 
# ----------------------------------------------------------------------------------
# CONFIG — edit these to match your actual data/checkpoint paths and desired grid.
# ----------------------------------------------------------------------------------
 
EXP_DIR = Path("experiments/exp_007_vehicle_residual_dynamics")
DATA_STORE_PATH = EXP_DIR / "data_raw_aug.npz"
 
# Live (current) model — no kinematic prior, predicts [ax, ay, alpha1, alpha2] directly.
CHECKPOINT_H = 4  # must match the H the checkpoint below was actually trained with
CHECKPOINT_NAME = "trailer-h4-128-4l-pruned-accel-augsplit2_best"
CHECKPOINT_PATH = Path("src/learning/models/trained") / CHECKPOINT_NAME
NORM_STATS_PATH = EXP_DIR / "data_proc2_stats.json"
 
# Legacy kin+residual model — set to None to skip this row entirely (recommended unless
# you still have the old checkpoint/spec around; kept only because the spec-mandated
# comparison row asks for it and it costs nothing to report if you have it).
LEGACY_CHECKPOINT_H = None  # e.g. 8
LEGACY_CHECKPOINT_PATH = None  # e.g. Path("src/learning/models/trained/trailer-kin-512-best")
LEGACY_NORM_STATS_PATH = None
 
FRICTION_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]  # matches trailer_collect.py's collection grid
TARGET_SPEEDS_KPH = [60, 40, 15]
N_TRIALS_PER_CONDITION = 5
MAX_STEPS = 600  # 30s at dt=0.05
WARMUP_STEPS_EXCLUDED = 100  # matches the `cutoff = 100` convention in exp_004/005 driver.py
 
OPEN_LOOP_HORIZON_K = 50  # rollout steps (~2.5s at dt=0.05), matches deploy MPPI's T scale
OPEN_LOOP_N_WINDOWS = 500  # number of held-out rollout starts to average over
 
MU_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
CHANNELS = ("ax", "ay", "alpha1", "alpha2")
 
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
 
 
# ----------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------
 
 
def load_norm_stats(path):
    with open(path, "r") as f:
        return json.load(f)
 
 
def load_model(in_dim, out_dim, ckpt_path):
    model = TrailerModel(in_dim, out_dim)
    _, state = nnx.split(model)
    ckpt = ocp.StandardCheckpointer()
    nnx.update(model, ckpt.restore(Path.cwd() / ckpt_path, state))
    return model
 
 
def default_scenario(mu, width=10.0):
    return TrailerBicycleEnvConfig(
        ".", TrackConfig(mu=mu, width=width), VehicleConfig(), SimulationConfig()
    )
 
 
# ====================================================================================
# MODE 1 — one-step validation RMSE, overall + stratified by friction + by history length
# ====================================================================================
 
 
def onestep_validation_rmse(H_values=(1, 2, 4, 8), spec_maker=make_spec_live, tag="live"):
    """
    For each H, rebuilds the FeatureSpec at that H, computes normalized-space channel MSE
    over the held-out (test) split, converts to raw RMSE, and additionally stratifies by
    friction (mu, raw column 6, read at row H-1 -- the "current" row of each window).
 
    NOTE: this recomputes stats from scratch per H via DataLoaderBlocked, it does NOT load
    a trained checkpoint per H -- it reports what the *data* looks like (residual scale,
    friction dependence) at each H. To grade an actual trained model's validation RMSE,
    use `validation_rmse_for_checkpoint` below with the specific H it was trained at.
    """
    raw = DataStore.load(DATA_STORE_PATH)
    rows = []
    for H in H_values:
        spec = spec_maker(H=H)
        data = raw.build(spec, DataLoaderBlocked, w=40, s=40)
 
        W = np.arange(spec.H + spec.F)
        mus, ys = [], []
        for i in range(0, len(data.test), 1 << 16):
            idx = data.test[i : i + (1 << 16)]
            w = data.data[idx[:, None] + W]
            y = np.asarray(spec.encode_y(w))
            mus.append(np.asarray(w[:, spec.H - 1, 6]))
            ys.append(y)
        mu_all = np.concatenate(mus)
        y_all = np.concatenate(ys)
 
        row = {"history_length": H, "dataset_version": spec.data_version, "friction_condition": "all"}
        row.update({f"target_std_{c}": float(s) for c, s in zip(CHANNELS, y_all.std(0))})
        rows.append(row)
 
        for lo, hi in MU_BINS:
            m = (mu_all >= lo) & (mu_all < hi)
            if m.sum() == 0:
                continue
            row_bin = {
                "history_length": H,
                "dataset_version": spec.data_version,
                "friction_condition": f"[{lo:.1f},{hi:.1f})",
                "n_windows": int(m.sum()),
            }
            row_bin.update({f"target_std_{c}": float(s) for c, s in zip(CHANNELS, y_all[m].std(0))})
            rows.append(row_bin)
 
        print(f"[onestep:{tag}] H={H}  n_test={len(data.test)}  spec={spec.data_version}")
 
    return pd.DataFrame(rows)

 
def validation_rmse_for_checkpoint(H, ckpt_path, norm_stats_path, spec_maker=make_spec_live, kin_fn=None):
    """
    Actual best-checkpoint validation RMSE (raw units, per channel + overall), the number
    that belongs in the "Validation RMSE at the best checkpoint" PR field and the
    "Validation RMSE over epochs" plot's final point.
    """
    raw = DataStore.load(DATA_STORE_PATH)
    spec = spec_maker(H=H)
    data = raw.build(spec, DataLoaderBlocked, w=40, s=40)
    norm_stats = load_norm_stats(norm_stats_path)
 
    in_dim = H * (len(LIVE_S_COLS) + len(LIVE_C_COLS))
    model = load_model(in_dim, len(CHANNELS), ckpt_path)
 
    x_mean = jnp.asarray(norm_stats["x_mean"])
    x_std = jnp.asarray(norm_stats["x_std"])
    y_mean = np.asarray(norm_stats["y_mean"])
    y_std = np.asarray(norm_stats["y_std"])
 
    W = np.arange(spec.H + spec.F)
    sq_err, mus = [], []
    for i in range(0, len(data.test), 4096):
        idx = data.test[i : i + 4096]
        w = data.data[idx[:, None] + W]
        x = (jnp.asarray(spec.encode_x(w)) - x_mean) / x_std
        y_raw = np.asarray(spec.encode_y(w))
        pred_raw = np.asarray(model(x)) * y_std + y_mean
        sq_err.append((pred_raw - y_raw) ** 2)
        mus.append(np.asarray(w[:, spec.H - 1, 6]))
    sq_err = np.concatenate(sq_err)
    mu_all = np.concatenate(mus)
 
    out = {"history_length": H, "dataset_version": spec.data_version, "friction_condition": "all"}
    out.update({f"rmse_{c}": float(np.sqrt(sq_err[:, i].mean())) for i, c in enumerate(CHANNELS)})
    rows = [out]
    for lo, hi in MU_BINS:
        m = (mu_all >= lo) & (mu_all < hi)
        if m.sum() == 0:
            continue
        row = {
            "history_length": H,
            "dataset_version": spec.data_version,
            "friction_condition": f"[{lo:.1f},{hi:.1f})",
            "n_windows": int(m.sum()),
        }
        row.update({f"rmse_{c}": float(np.sqrt(sq_err[m, i].mean())) for i, c in enumerate(CHANNELS)})
        rows.append(row)
    return pd.DataFrame(rows)
 
df = validation_rmse_for_checkpoint(CHECKPOINT_H, CHECKPOINT_PATH, NORM_STATS_PATH)
df.to_csv(RESULTS_DIR / "onestep_validation_rmse.csv", index=False)
print(df.to_markdown(index=False))
 