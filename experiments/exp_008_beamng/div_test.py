"""
Open-loop divergence of the learned model against ground truth, plus the
sign-convention and wrap diagnostics that gate interpreting it.

Read-only: touches no spec, no checkpoint, no dataset. Prior variants are
implemented by pre-negating the steering column of the prior's input rather
than editing `fiala_dyn`, so this can run against HEAD unmodified.

Rollout mirrors `beamng_dynamics.dynamics` exactly:
    v      <- v      + a   * dt          (forward Euler; matches FD targets)
    phidot <- phidot + phiddot * dt
    h      <- h      + (phi1dot' - phi2dot') * dt   (NEXT rates -- see note)
    d_s    <- clip(d_s + ddelta_s * dt, -1, 1)
    a_s    <- clip(a_s + daccel_s * dt, -1, 1)
Commands (cols 9,10) are always ground truth; only the state feeds back.

Note on the hitch: `_poll_state` defines phidot_t = (phi_t - phi_{t-1})/dt, so
phi_{t+1} = phi_t + phidot_{t+1}*dt is FD-exact, which is what deploy does
(dx[2] = next_phi1dot). Do not "fix" this to trapezoidal.

Usage:
    python -m experiments.exp_008_beamng.divergence            # everything
    python -m experiments.exp_008_beamng.divergence --checks    # diagnostics only
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

from src.learning.datasets.trailer_data import DataStore
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models.beamng_trailer_spec import (
    STATE_FS,
    fiala_dyn,
    IN_COLS,
    X_COLS,
    FD_COLS,
    V,
)

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DATA = Path("experiments/exp_008_beamng/data_trial2_aug.npz")
STATS = Path("experiments/exp_008_beamng/data_proc_test5_stats.json")
CKPT = Path.cwd() / "src/learning/models/trained/beamng-l4-128-test5_best"
OUT = Path("experiments/exp_008_beamng/divergence_out")

K = 55  # rollout steps; match the deploy horizon T
DT = 0.05
STRIDE = 25  # spacing between rollout start points, in rows
MAX_STARTS = 20000  # cap for runtime; sampled uniformly if exceeded
CHUNK = 1024  # rollout starts per vmapped call

# Speed buckets in kph, on mean |vx| over the IC window.
SPEED_EDGES_KPH = np.array([0, 20, 40, 60, 80, 200])

# Which prior the rolled model uses. MUST match what the checkpoint was
# trained against, or the residual is being added to a different function than
# it was fit to. "current" = HEAD. Only use "unflipped" with a checkpoint
# retrained under the unflipped prior.
PRIOR_MODE = "current"  # {"current", "unflipped", "zero"}

STATE_NAMES = ("h", "vx", "vy", "phi1dot", "phi2dot", "delta_s", "accel_s")
PRIOR_CHANNELS = ("ax", "ay", "phi1ddot", "phi2ddot")

# raw-column indices (11-wide stored row)
C_SH, C_CH, C_VX, C_VY, C_P1D, C_P2D, C_MU, C_DS, C_AS, C_DCMD, C_ACMD = range(11)


# ----------------------------------------------------------------------------
# prior variants
# ----------------------------------------------------------------------------


def make_prior(mode):
    """r is in X_COLS order: [sh, ch, vx, vy, phi1dot, phi2dot, delta_s, accel_s]."""
    if mode == "zero":
        return lambda r: jnp.zeros(4)
    if mode == "current":
        return fiala_dyn  # HEAD: fiala_dyn negates internally
    if mode == "unflipped":
        # cancel fiala_dyn's internal negation without editing the spec
        return lambda r: fiala_dyn(r.at[6].set(-r[6]))
    raise ValueError(mode)


def wrap(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ----------------------------------------------------------------------------
# window enumeration
# ----------------------------------------------------------------------------


def rollout_starts(traj_len, H, k, stride):
    """Start rows s such that [s, s+H+k) lies inside a single trajectory."""
    starts = np.concatenate([[0], np.cumsum(traj_len)[:-1]])
    out = []
    for s0, L in zip(starts, traj_len):
        n = L - (H + k) + 1
        if n > 0:
            out.append(s0 + np.arange(0, n, stride))
    if not out:
        raise RuntimeError(
            f"no trajectory is long enough for H={H} + K={k}; "
            f"longest is {int(traj_len.max())} rows"
        )
    return np.concatenate(out).astype(np.int64)


# ----------------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------------


def make_rollout(model, prior_fn, x_mean, x_std, y_mean, y_std, H, k, dt):
    in_cols = np.asarray(IN_COLS)

    def one(seg):
        """seg: (H+k, 11) ground-truth rows. -> (k, 7) signed errors."""
        buf0 = seg[:H][:, in_cols]  # (H, 10)
        h0 = jnp.arctan2(seg[H - 1, C_SH], seg[H - 1, C_CH])

        def step(carry, j):
            buf, h = carry

            x = ((buf.reshape(-1) - x_mean) / x_std)[None, :]
            pred = model(x)[0] * y_std + y_mean
            # buf[:, :8] is exactly X_COLS order by construction of IN_COLS
            pred = pred + jnp.concatenate([prior_fn(buf[-1, :8]), jnp.zeros(2)])
            ax, ay, a1, a2, dds, das = pred

            vx, vy = buf[-1, 2], buf[-1, 3]
            p1d, p2d = buf[-1, 4], buf[-1, 5]
            ds, as_ = buf[-1, 6], buf[-1, 7]

            vx_n = vx + ax * dt
            vy_n = vy + ay * dt
            p1d_n = p1d + a1 * dt
            p2d_n = p2d + a2 * dt
            ds_n = jnp.clip(ds + dds * dt, -1.0, 1.0)  # == deploy's clip_deriv
            as_n = jnp.clip(as_ + das * dt, -1.0, 1.0)
            h_n = h + (p1d_n - p2d_n) * dt

            true = seg[H + j]
            cmd = jax.lax.dynamic_slice(true, (C_DCMD,), (2,))

            row = jnp.concatenate(
                [
                    jnp.array([jnp.sin(h_n), jnp.cos(h_n), vx_n, vy_n, p1d_n, p2d_n, ds_n, as_n]),
                    cmd,  # ground-truth commands
                ]
            )
            buf_n = jnp.concatenate([buf[1:], row[None, :]], axis=0)

            h_true = jnp.arctan2(true[C_SH], true[C_CH])
            err = jnp.array(
                [
                    wrap(h_n - h_true),
                    vx_n - true[C_VX],
                    vy_n - true[C_VY],
                    p1d_n - true[C_P1D],
                    p2d_n - true[C_P2D],
                    ds_n - true[C_DS],
                    as_n - true[C_AS],
                ]
            )
            return (buf_n, h_n), err

        (_, _), errs = jax.lax.scan(step, (buf0, h0), jnp.arange(k))
        return errs

    return nnx.jit(nnx.vmap(one, in_axes=0))


# ----------------------------------------------------------------------------
# diagnostics (checkpoint-free)
# ----------------------------------------------------------------------------


def valid_pairs(traj_len, n):
    """Boolean mask over rows: True where row i and i+1 are in the same traj."""
    m = np.ones(n, bool)
    m[np.cumsum(traj_len) - 1] = False
    return m


def check_steer_convention(d, traj_len, sample=400000):
    """Which frame is column 7 (electrics readback) in?

    corr(col7, col9) ~ -1  => readback opposite to command => col7 is INTERNAL
                              => fiala_dyn's leading minus is WRONG on col 7.
    corr(col7, col9) ~ +1  => readback agrees with command => the minus is right.

    The regression is the frame-agnostic version: for a kinematic bicycle,
    phi1dot = (vx / L) * tan(delta * max_steer), so a slope of +1 means the
    column you fed is in the same frame as the dynamics model.
    """
    print("\n=== steering frame ===")
    n = len(d)
    idx = np.random.default_rng(0).choice(n, size=min(sample, n), replace=False)
    s = d[idx]

    r = np.corrcoef(s[:, C_DS], s[:, C_DCMD])[0, 1]
    print(f"  corr(col7 readback, col9 command) = {r:+.4f}")
    print(
        "    => col 7 is "
        + ("INTERNAL frame; fiala_dyn's leading minus is a BUG" if r < 0 else "COMMAND frame; the minus is correct")
    )

    L = V.lf + V.lr
    for name, col in (("col7 (readback)", C_DS), ("col9 (command)", C_DCMD)):
        pred = (s[:, C_VX] / L) * np.tan(s[:, col] * V.max_steer_rad)
        keep = np.abs(s[:, C_VX]) > 2.0
        if keep.sum() < 100:
            print(f"  {name}: too few rows with |vx|>2")
            continue
        a, b = np.polyfit(pred[keep], s[keep, C_P1D], 1)
        rho = np.corrcoef(pred[keep], s[keep, C_P1D])[0, 1]
        print(f"  phi1dot ~ (vx/L)tan(d*max_steer) using {name}: slope={a:+.3f} rho={rho:+.3f}")
    print("    slope ~ +1 => that column matches the dynamics-model frame")


def check_prior_sign(d, traj_len, sample=200000):
    """Which prior variant predicts the FD truth? No checkpoint involved."""
    print("\n=== prior sign (prior vs finite-difference truth) ===")
    ok = np.flatnonzero(valid_pairs(traj_len, len(d)))
    idx = np.random.default_rng(1).choice(ok, size=min(sample, len(ok)), replace=False)
    k = jnp.asarray(d[idx])
    kp = jnp.asarray(d[idx + 1])
    truth = np.asarray((kp[:, FD_COLS] - k[:, FD_COLS]) / DT)[:, :4]
    vx = np.asarray(k[:, C_VX])

    for mode in ("current", "unflipped"):
        pf = jax.jit(jax.vmap(make_prior(mode)))
        p = np.asarray(pf(k[:, X_COLS]))
        print(f"  --- prior_mode={mode}")
        for sgn, lbl in ((vx > 0.5, "fwd"), (vx < -0.5, "rev")):
            if sgn.sum() < 100:
                continue
            cells = []
            for c, name in enumerate(PRIOR_CHANNELS):
                t, q = truth[sgn, c], p[sgn, c]
                good = np.isfinite(t) & np.isfinite(q)
                if good.sum() < 100:
                    cells.append(f"{name}: --")
                    continue
                rho = np.corrcoef(q[good], t[good])[0, 1]
                slope = np.polyfit(q[good], t[good], 1)[0]
                cells.append(f"{name}: rho={rho:+.3f} slope={slope:+.2f}")
            print(f"      {lbl} (n={int(sgn.sum())}): " + "  ".join(cells))
    print("    higher rho and slope nearer +1 wins; sign flip shows as rho changing sign")


def check_yaw_wrap(d, traj_len, thresh=20.0):
    """Unwrapped finite differences in _poll_state inject ~2pi/dt spikes."""
    print("\n=== yaw-rate wrap spikes ===")
    for c, name in ((C_P1D, "phi1dot"), (C_P2D, "phi2dot")):
        v = d[:, c]
        bad = np.abs(v) > thresh
        print(
            f"  {name}: |.|>{thresh} -> {bad.sum()} rows ({100*bad.mean():.4f}%)  "
            f"max|.|={np.abs(v).max():.1f}  std={v.std():.3f}"
        )
    print(f"    2*pi/dt = {2*np.pi/DT:.1f} rad/s is the signature of an unwrapped FD")

    ok = valid_pairs(traj_len, len(d))
    h = np.arctan2(d[:, C_SH], d[:, C_CH])
    hdot_fd = np.zeros(len(d))
    hdot_fd[:-1] = wrap_np(h[1:] - h[:-1]) / DT
    hdot_state = d[:, C_P1D] - d[:, C_P2D]
    m = ok & (np.abs(hdot_state) < thresh)
    slope = np.polyfit(hdot_fd[m], hdot_state[m], 1)[0]
    print(f"  hitch-rate consistency slope (want ~1.0, 2.5 => dt mismatch): {slope:.3f}")


def check_excitation(d, traj_len, meta):
    """Row counts and trajectory lengths per controller -- how much of the set
    carries real control excitation, and how long those runs survive."""
    print("\n=== excitation / trajectory survival by controller ===")
    starts = np.concatenate([[0], np.cumsum(traj_len)[:-1]])
    for c in np.unique(meta[:, 1]):
        sel = meta[:, 1] == c
        lens = traj_len[sel]
        rows = np.concatenate(
            [np.arange(s, s + L) for s, L in zip(starts[sel], lens)]
        )
        vx = d[rows, C_VX]
        # residual of the command about a local mean is a crude excitation proxy
        dc = d[rows, C_DCMD]
        print(
            f"  ctrl {int(c):2d}: {int(sel.sum()):4d} traj  {len(rows):8d} rows  "
            f"len med/max {int(np.median(lens)):5d}/{int(lens.max()):5d}  "
            f"mean|vx| {np.abs(vx).mean():5.2f}  std(d_cmd) {dc.std():.3f}"
        )
    print("    short runs with high std(d_cmd) = excited-but-dying; long runs")
    print("    with low std(d_cmd) = unexcited. Neither identifies df/du at speed.")


def wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def run_divergence(store, model, stats):
    d = store.data
    H = STATE_FS.H
    prior_fn = make_prior(PRIOR_MODE)

    starts = rollout_starts(store.traj_len, H, K, STRIDE)
    if len(starts) > MAX_STARTS:
        starts = np.random.default_rng(2).choice(starts, MAX_STARTS, replace=False)
    print(f"\n=== open-loop divergence: {len(starts)} starts, K={K}, prior={PRIOR_MODE} ===")

    roll = make_rollout(
        model,
        prior_fn,
        jnp.asarray(stats["x_mean"]),
        jnp.asarray(stats["x_std"]),
        jnp.asarray(stats["y_mean"]),
        jnp.asarray(stats["y_std"]),
        H,
        K,
        DT,
    )

    W = np.arange(H + K)
    errs, ic_vx = [], []
    for i in range(0, len(starts), CHUNK):
        s = starts[i : i + CHUNK]
        seg = jnp.asarray(d[s[:, None] + W])  # (B, H+K, 11)
        errs.append(np.asarray(roll(seg)))
        ic_vx.append(d[s[:, None] + np.arange(H)][:, :, C_VX].mean(1))
        print(f"\r  {min(i+CHUNK, len(starts))}/{len(starts)}", end="")
    print()

    errs = np.concatenate(errs)  # (N, K, 7)
    ic_vx = np.concatenate(ic_vx)  # (N,)

    finite = np.isfinite(errs).all(axis=(1, 2))
    if (~finite).any():
        print(f"  {(~finite).sum()} rollouts went non-finite; excluded from RMSE")
    return errs, ic_vx, finite


def report(errs, ic_vx, finite):
    speed_kph = np.abs(ic_vx) * 3.6
    print("\n--- per-step RMSE, by IC speed bucket and direction ---")
    rows = {}
    for lo, hi in zip(SPEED_EDGES_KPH[:-1], SPEED_EDGES_KPH[1:]):
        band = (speed_kph >= lo) & (speed_kph < hi)
        for sgn, lbl in ((ic_vx > 0, "fwd"), (ic_vx < 0, "rev")):
            m = band & sgn & finite
            if m.sum() < 20:
                continue
            e = errs[m]  # (n, K, 7)
            rmse = np.sqrt((e**2).mean(0))  # (K, 7)
            key = f"{lbl} {int(lo):3d}-{int(hi):3d}kph"
            rows[key] = rmse
            h = rmse[:, 0]
            # growth exponent: fit h_k ~ k^p over the middle of the horizon
            ks = np.arange(1, len(h) + 1)
            sl = slice(max(1, len(h) // 8), len(h))
            p = np.polyfit(np.log(ks[sl]), np.log(np.maximum(h[sl], 1e-12)), 1)[0]
            print(
                f"  {key}  n={int(m.sum()):5d}   "
                f"h: k=1 {h[0]:.2e}  k={len(h)//2} {h[len(h)//2]:.2e}  k={len(h)} {h[-1]:.2e}   "
                f"ratio {h[-1]/max(h[0],1e-12):8.1f}   k^p fit p={p:.2f}"
            )
    print(
        "\n  p ~ 1     : error accumulates linearly (static bias)\n"
        "  p ~ 1.5-2 : compounding bias\n"
        "  p ~ 0.5   : noise accumulation\n"
        "  ratio growing sharply with speed, reverse >> forward, or clearly\n"
        "  super-polynomial growth => the unstable hitch mode dominates and\n"
        "  no amount of one-step accuracy fixes it."
    )

    print("\n--- terminal RMSE per channel (all starts) ---")
    term = np.sqrt((errs[finite][:, -1, :] ** 2).mean(0))
    for name, v in zip(STATE_NAMES, term):
        print(f"  {name:9s} {v:.4e}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / f"divergence_{PRIOR_MODE}_K{K}.npz",
        errs=errs.astype(np.float32),
        ic_vx=ic_vx.astype(np.float32),
        finite=finite,
        channels=np.array(STATE_NAMES),
    )
    print(f"\n  saved -> {OUT / f'divergence_{PRIOR_MODE}_K{K}.npz'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for key, rmse in rows.items():
            ls = "-" if key.startswith("fwd") else "--"
            axes[0].semilogy(np.arange(1, len(rmse) + 1), rmse[:, 0], ls, label=key)
            axes[1].semilogy(np.arange(1, len(rmse) + 1), rmse[:, 1], ls, label=key)
        axes[0].set_title("hitch |h| RMSE (rad)")
        axes[1].set_title("vx RMSE (m/s)")
        for a in axes:
            a.set_xlabel("rollout step k")
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
        fig.suptitle(f"open-loop divergence, prior={PRIOR_MODE}, K={K}")
        fig.tight_layout()
        fig.savefig(OUT / f"divergence_{PRIOR_MODE}_K{K}.png", dpi=140)
        print(f"  saved -> {OUT / f'divergence_{PRIOR_MODE}_K{K}.png'}")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="diagnostics only, no checkpoint")
    ap.add_argument("--no-checks", action="store_true", help="divergence only")
    args = ap.parse_args()

    store = DataStore.load(DATA)
    d = np.asarray(store.data)
    print(f"loaded {DATA}: {d.shape}, {len(store.traj_len)} trajectories, "
          f"version={store.version}, spec={STATE_FS.data_version}")
    if store.version != STATE_FS.data_version:
        print("  WARNING: data_version mismatch between store and spec")

    if not args.no_checks:
        check_steer_convention(d, store.traj_len)
        check_prior_sign(d, store.traj_len)
        check_yaw_wrap(d, store.traj_len)
        check_excitation(d, store.traj_len, np.asarray(store.meta))

    if args.checks:
        return

    with open(STATS) as f:
        stats = json.load(f)
    y_std = np.asarray(stats["y_std"])
    print(f"\ny_std = {np.array2string(y_std, precision=3)}")
    if y_std[2] < 1.0:
        print("  WARNING: y_std[2:4] looks like rate stats, not accelerations -- "
              "stale normalisation for this spec")

    model = TrailerModel(STATE_FS.H * len(IN_COLS), 6)
    _, state = nnx.split(model)
    nnx.update(model, ocp.StandardCheckpointer().restore(CKPT, state))

    errs, ic_vx, finite = run_divergence(store, model, stats)
    report(errs, ic_vx, finite)


if __name__ == "__main__":
    main()