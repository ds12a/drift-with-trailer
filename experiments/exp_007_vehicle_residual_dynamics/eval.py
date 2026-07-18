"""
Standalone eval for the trailer residual model. No wandb.
    python -m experiments.exp_007_vehicle_residual_dynamics.eval
Prints per-channel raw RMSE (physical units) and prior explained-variance.
"""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from src.learning.datasets.trailer_data import DataStore
from src.learning.models.trailer_spec import KIN_FS, kin, IN_COLS, VEL_COLS, YAW_COLS, KIN_COLS, fzr, V
from src.learning.models.trailer_nn import TrailerModel

CHANNELS = ("ax", "ay", "w1", "w2")               # out_fn order
UNITS = ("m/s^2", "m/s^2", "rad/s", "rad/s")

DATA = Path("./experiments/exp_007_vehicle_residual_dynamics/data_raw.npz")
CKPT = Path("src/learning/models/trained/trailer-best")


def load_model(in_dim, out_dim, path):
    model = TrailerModel(in_dim, out_dim)
    _, state = nnx.split(model)
    state = ocp.StandardCheckpointer().restore(Path.cwd() / path, state)
    nnx.update(model, state)
    return model


def main(ckpt=CKPT, data=DATA, spec=KIN_FS, batch_size=4096):
    dl = DataStore.load(data).build(spec)
    ys = np.asarray(dl.y_std)                                   # raw residual std
    ym = np.asarray(dl.y_mean)

    model = load_model(spec.H * len(IN_COLS), len(CHANNELS), ckpt)

    # --- accumulate normalized SE on the test set, batched ---
    W = np.arange(spec.H + spec.F)
    se = np.zeros(len(CHANNELS))
    n = 0
    for i in range(0, len(dl.test), batch_size):
        idx = dl.test[i : i + batch_size]
        w = dl.data[idx[:, None] + W]
        x = (spec.encode_x(w) - dl.x_mean) / dl.x_std
        y = (spec.encode_y(w) - dl.y_mean) / dl.y_std
        se += np.asarray(((model(x) - y) ** 2).sum(0))
        n += len(idx)
    norm_mse = se / n                                          # per channel, normalized

    raw_rmse = np.sqrt(norm_mse) * ys                         # undo normalization
    resid_r2 = 1 - norm_mse                                   # R^2 vs residual (y is zero-mean-ish)

    # --- prior explained variance: how much did kin() buy per channel ---
    # residual std ys vs raw target std (residual + prior). recompute raw target std once.
    tgt_std = _raw_target_std(dl, spec)
    prior_ev = 1 - (ys / tgt_std) ** 2

    w9 = max(len(u) for u in UNITS)
    print(f"\n{'chan':<6}{'raw_RMSE':>12}{'unit':>{w9+2}}{'resid_R2':>11}{'prior_EV':>11}")
    for i, c in enumerate(CHANNELS):
        print(f"{c:<6}{raw_rmse[i]:>12.4f}{UNITS[i]:>{w9+2}}"
              f"{resid_r2[i]:>11.4f}{prior_ev[i]:>11.4f}")
    print(f"\ntest windows: {n}   mean raw RMSE: {raw_rmse.mean():.4f}")
    print("prior_EV < 0  => kinematic prior HURTS that channel (net must undo it)")

    W = np.arange(spec.H + spec.F)
    idx = dl.test
    w = dl.data[idx[:, None] + W]
    k, kp = w[:, spec.H-1], w[:, spec.H]
    tgt = np.concatenate([np.asarray((kp[:, VEL_COLS]-k[:, VEL_COLS])/dl.dt),
                        np.asarray(k[:, YAW_COLS])], -1)     # raw [ax,ay,w1,w2]
    pri = np.asarray(kin(k[:, KIN_COLS]))                       # [ax,ay,w1,w2]
    res = tgt - pri
    ev = 1 - res.var(0) / tgt.var(0)
    print("prior EV (test, correct):", dict(zip(CHANNELS, ev.round(3))))
    print("prior mean vs tgt mean:", pri.mean(0).round(3), tgt.mean(0).round(3))
    print(np.corrcoef(pri[:,3], tgt[:,3])[0,1])   # w2: prior vs target

    k = dl.data[dl.test[:4096, None] + np.arange(spec.H+spec.F)][:, spec.H-1]
    r = np.asarray(k[:, KIN_COLS]); a = r[:,6]; mu = r[:,4]
    cmd = np.maximum(a,0)*V.max_accel + np.minimum(a,0)*V.max_brake
    print("cmd", cmd.min(), cmd.max(), "| tanh_arg", (V.mass*cmd/(fzr*mu)).min(), (V.mass*cmd/(fzr*mu)).max())
    print("fxr/tot", (mu*fzr*np.tanh(V.mass*cmd/(fzr*mu))/(V.mass+V.trailer_mass)).mean())


def _raw_target_std(dl, spec):
    """std of the pre-residual target [ax, ay, w1, w2], for prior EV."""
    W = np.arange(spec.H + spec.F)
    acc = np.zeros(4); acc2 = np.zeros(4); n = 0
    for i in range(0, len(dl.test), 1 << 19):
        idx = dl.test[i : i + (1 << 19)]
        w = dl.data[idx[:, None] + W]
        k, kp = w[:, spec.H - 1], w[:, spec.H]
        acc_ = (kp[:, VEL_COLS] - k[:, VEL_COLS]) / dl.dt
        tgt = np.concatenate([np.asarray(acc_), np.asarray(k[:, YAW_COLS])], -1)
        acc += tgt.sum(0); acc2 += (tgt ** 2).sum(0); n += len(idx)
    mu = acc / n
    return np.sqrt(np.maximum(acc2 / n - mu ** 2, 1e-12))


if __name__ == "__main__":
    main()