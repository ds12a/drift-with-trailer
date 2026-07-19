"""
Eval + diagnostics for the trailer residual model, all w.r.t. the kinematic prior.

    python -m experiments.exp_007_vehicle_residual_dynamics.eval               # everything
    python -m experiments.exp_007_vehicle_residual_dynamics.eval --no-probe    # skip training
    python -m experiments.exp_007_vehicle_residual_dynamics.eval --no-model    # prior-only

Reports, per channel [ax, ay, w1, w2]:
  prior block  : raw target std, prior mean vs target mean, corr, prior EV, w2-with-true-w1
  model block  : raw RMSE (physical units), residual R^2
  probe block  : traj-split vs group-split RMSE inflation (leakage test)
"""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from src.learning.datasets.trailer_data import DataStore
from src.learning.models.trailer_spec import (
    spec, kin, V, IN_COLS, KIN_COLS, VEL_COLS, YAW_COLS,
)

CHANNELS = ("ax", "ay", "w1", "w2")
UNITS = ("m/s^2", "m/s^2", "rad/s", "rad/s")

DATA = Path("./experiments/exp_007_vehicle_residual_dynamics/data_raw.npz")
CKPT = Path("src/learning/models/trained/trailer-best")

BIG = 1 << 19


# ----------------------------------------------------------------------------- windows

def _test_windows(dl, spec):
    """Gather all test windows once. Returns (k, kp): raw rows at t and t+1."""
    W = np.arange(spec.H + spec.F)
    k = np.empty((len(dl.test), dl.data.shape[1]), np.float32)
    kp = np.empty_like(k)
    for i in range(0, len(dl.test), BIG):
        idx = dl.test[i : i + BIG]
        w = dl.data[idx[:, None] + W]
        k[i : i + BIG] = w[:, spec.H - 1]
        kp[i : i + BIG] = w[:, spec.H]
    return k, kp


def _raw_target(k, kp, dt):
    """[ax, ay, w1, w2] pre-residual, from raw rows."""
    acc = (kp[:, VEL_COLS] - k[:, VEL_COLS]) / dt
    return np.concatenate([acc, k[:, YAW_COLS]], -1)


# ----------------------------------------------------------------------------- prior

def prior_block(dl, spec):
    k, kp = _test_windows(dl, spec)
    tgt = _raw_target(k, kp, dl.dt)                       # (N, 4)
    pri = np.asarray(kin(k[:, KIN_COLS]))                 # (N, 4)
    res = tgt - pri

    tgt_std = tgt.std(0)
    ev = 1 - res.var(0) / np.maximum(tgt.var(0), 1e-12)
    corr = np.array([np.corrcoef(pri[:, i], tgt[:, i])[0, 1] for i in range(4)])

    # w2 discriminator: does feeding TRUE w1 into w2's formula fix it?
    sh, ch, vx, vy = k[:, 0], k[:, 1], k[:, 2], k[:, 3]
    L2 = V.l2f + V.l2r
    w1_true = k[:, YAW_COLS[0]]
    w2_true_w1 = (vx * sh + (vy - V.hitch_offset * w1_true) * ch) / L2
    w2_ev_truew1 = 1 - ((tgt[:, 3] - w2_true_w1).var() / max(tgt[:, 3].var(), 1e-12))

    print("\n===== PRIOR (test split, correct) =====")
    print(f"{'chan':<6}{'tgt_std':>10}{'pri_mean':>10}{'tgt_mean':>10}"
          f"{'corr':>8}{'prior_EV':>10}")
    for i, c in enumerate(CHANNELS):
        print(f"{c:<6}{tgt_std[i]:>10.4f}{pri.mean(0)[i]:>10.4f}{tgt.mean(0)[i]:>10.4f}"
              f"{corr[i]:>8.3f}{ev[i]:>10.4f}")
    print(f"\nw2 EV with TRUE w1: {w2_ev_truew1:>7.4f}   "
          f"(vs prior EV {ev[3]:.4f}: jump => w1-propagation error, not formula)")
    print("prior_EV < 0 => prior HURTS; net must undo it")


# ----------------------------------------------------------------------------- model

def _load_model(in_dim, out_dim, path):
    from flax import nnx
    import orbax.checkpoint as ocp
    from src.learning.models.trailer_nn import TrailerModel
    model = TrailerModel(in_dim, out_dim)
    _, state = nnx.split(model)
    state = ocp.StandardCheckpointer().restore(Path.cwd() / path, state)
    nnx.update(model, state)
    return model


def model_block(dl, spec, ckpt):
    model = _load_model(spec.H * len(IN_COLS), len(CHANNELS), ckpt)
    ys = np.asarray(dl.y_std)

    W = np.arange(spec.H + spec.F)
    se = np.zeros(len(CHANNELS))
    n = 0
    for i in range(0, len(dl.test), 4096):
        idx = dl.test[i : i + 4096]
        w = dl.data[idx[:, None] + W]
        x = (spec.encode_x(w) - dl.x_mean) / dl.x_std
        y = (spec.encode_y(w) - dl.y_mean) / dl.y_std
        se += np.asarray(((model(x) - y) ** 2).sum(0))
        n += len(idx)

    norm_mse = se / n
    raw_rmse = np.sqrt(norm_mse) * ys
    resid_r2 = 1 - norm_mse

    print("\n===== MODEL (test split) =====")
    w = max(len(u) for u in UNITS)
    print(f"{'chan':<6}{'raw_RMSE':>11}{'unit':>{w+2}}{'resid_R2':>11}")
    for i, c in enumerate(CHANNELS):
        print(f"{c:<6}{raw_rmse[i]:>11.4f}{UNITS[i]:>{w+2}}{resid_r2[i]:>11.4f}")
    print(f"\ntest windows: {n}   mean raw RMSE: {raw_rmse.mean():.4f}")


# ----------------------------------------------------------------------------- probe

def _regroup(dl, spec, group):
    """Rebuild train/test + stats. group=True keeps same-(run,ctrl,step) windows together."""
    valid, tid = dl._windows(dl.traj_len)
    if group:
        _, ginv = np.unique(dl.meta, axis=0, return_inverse=True)   # same-IC triplets
    else:
        ginv = np.arange(len(dl.traj_len))                          # per-trajectory
    gmap = ginv[tid]
    ng = ginv.max() + 1
    keep = np.zeros(ng, bool)
    keep[np.random.default_rng(spec.split_seed)
         .permutation(ng)[: int(ng * spec.train_frac)]] = True
    m = keep[gmap]
    object.__setattr__(dl, "train", valid[m])
    object.__setattr__(dl, "test", valid[~m])
    for kk, vv in zip(("x_mean", "x_std", "y_mean", "y_std"), dl.compute_stats()):
        object.__setattr__(dl, kk, vv)


def _quick_train(dl, spec, epochs, bs):
    from src.learning.models.trailer_nn import TrailerModel
    from experiments.exp_007_vehicle_residual_dynamics.train import LearnedDynamics, train_step, loss_fn, col_loss
    ld = LearnedDynamics(TrailerModel(spec.H * len(IN_COLS), len(CHANNELS)), dl, batch_size=bs)
    for e in range(epochs):
        tr, _ = dl.get_data(bs, jax.random.fold_in(jax.random.PRNGKey(0), e))
        for b in tr:
            train_step(ld.model, ld.optimizer, ld.metrics, b)
        ld.metrics.reset()
    _, te = dl.get_data(bs, jax.random.PRNGKey(1))
    for b in te:
        ld.metrics.update(loss=loss_fn(ld.model, b), channel_losses=col_loss(ld.model, b))
    c = np.asarray(ld.metrics.compute()["channel_losses"])
    return np.sqrt(c) * np.asarray(dl.y_std)


def probe_block(spec, data, epochs, bs=4096):
    out = {}
    for name, group in (("traj", False), ("group", True)):
        dl = DataStore.load(data).build(spec)
        _regroup(dl, spec, group)
        out[name] = (_quick_train(dl, spec, epochs, bs), len(dl.train), len(dl.test))

    (traj, ntr, nte), (grp, gtr, gte) = out["traj"], out["group"]
    print(f"\n===== LEAKAGE PROBE ({epochs} epochs) =====")
    print(f"traj-split : {ntr:>9} train {nte:>9} test")
    print(f"group-split: {gtr:>9} train {gte:>9} test")
    print(f"\n{'chan':<6}{'traj':>10}{'group':>10}{'inflation':>11}")
    for i, c in enumerate(CHANNELS):
        print(f"{c:<6}{traj[i]:>10.4f}{grp[i]:>10.4f}{grp[i] / traj[i]:>10.2f}x")
    print("\ninflation ~1x  => residual fit is real")
    print("inflation >>1x => channel was echoing an input in the window (leak)")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    dl = DataStore.load(args.data).build(spec)
    prior_block(dl, spec)
    if not args.no_model:
        model_block(dl, spec, args.ckpt)
    if not args.no_probe:
        probe_block(spec, args.data, args.epochs)


if __name__ == "__main__":
    main()