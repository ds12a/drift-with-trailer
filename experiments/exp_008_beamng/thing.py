"""
Open-loop divergence for BOTH rollout models in one pass, plotted on a shared grid.

Merges `div_test_old.py` (kinematic prior + learned residual, test5) and
`div_test.py` (pure learned model, test9). The two scripts differed in exactly
four places -- spec module, data, stats/checkpoint, and whether `prior_fn` is
added to the network output -- so those four are now the fields of `Variant`
and everything else is shared.

Read-only: touches no spec, no checkpoint, no dataset. Prior variants are
implemented by pre-negating the steering column of the prior's input rather
than editing `fiala_dyn`, so this runs against HEAD unmodified.

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

Plot: rows = channel, cols = variant, `sharey="row"` so the two models are on
identical axes; colours/linestyles are keyed off the speed bucket so a bucket
looks the same in every panel and one figure-level legend serves all four.

Usage:
    python -m experiments.exp_008_beamng.div_test_combined              # both + grid
    python -m experiments.exp_008_beamng.div_test_combined --checks     # diagnostics only
    python -m experiments.exp_008_beamng.div_test_combined --only new   # one variant
    python -m experiments.exp_008_beamng.div_test_combined --replot     # grid from saved npz
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

from src.learning.datasets.trailer_data import DataStore
from src.learning.models.trailer_nn import TrailerModel
from src.learning.models import beamng_trailer_spec as trailer_spec
from src.learning.models import beamng_model_spec as model_spec
from src.learning.models.beamng_trailer_spec import (
    fiala_dyn,
    IN_COLS,
    X_COLS,
    FD_COLS,
    V,
)

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

ROOT = Path("experiments/exp_008_beamng")
OUT = ROOT / "divergence_out"

K = 55  # rollout steps; match the deploy horizon T
DT = 0.05
N_OUT = 6  # [ax, ay, phi1ddot, phi2ddot, ddelta_s, daccel_s]
STRIDE = 25  # spacing between rollout start points, in rows
MAX_STARTS = 20000  # cap for runtime; sampled uniformly if exceeded
CHUNK = 1024  # rollout starts per vmapped call

# Speed buckets in km/h, on mean |vx| over the IC window.
SPEED_EDGES_KPH = np.array([0, 20, 40, 60, 80, 200])
MIN_BUCKET_N = 20

STATE_NAMES = ("h", "vx", "vy", "phi1dot", "phi2dot", "delta_s", "accel_s")
PRIOR_CHANNELS = ("ax", "ay", "phi1ddot", "phi2ddot")

# Which state channels get a row in the grid. 2 channels x 2 variants = the 4-grid.
PLOT_CHANNELS = ("h", "vx")
# "h" is the internal/index key (matches STATE_NAMES); display uses the paper's
# hitch-angle notation, alpha, not "h".
CHANNEL_LABEL = {"h": r"$\alpha$ (hitch)", "vx": r"$v_x$", "vy": r"$v_y$",
                 "phi1dot": r"$\dot\phi_1$", "phi2dot": r"$\dot\phi_2$",
                 "delta_s": r"$\delta_s$", "accel_s": r"$a_s$"}
CHANNEL_UNITS = {"h": "rad", "vx": "km/h", "vy": "km/h",
                 "phi1dot": "rad/s", "phi2dot": "rad/s",
                 "delta_s": "-", "accel_s": "-"}

# raw-column indices (11-wide stored row)
C_SH, C_CH, C_VX, C_VY, C_P1D, C_P2D, C_MU, C_DS, C_AS, C_DCMD, C_ACMD = range(11)


@dataclass(frozen=True)
class Variant:
    key: str            # short id, used in filenames
    label: str          # detailed label -- printed in report()/load_variant(), not the figure
    short_label: str    # terse figure column title, e.g. "Prior + Residual"
    spec: Any           # FeatureSpec (supplies H, data_version)
    in_cols: tuple      # network input columns for THIS spec
    hidden: tuple       # hidden widths; must match what the checkpoint was trained with
    data: Path
    stats: Path
    ckpt: Path
    use_prior: bool     # True  -> pred += fiala prior (residual formulation)
    prior_mode: str = "current"  # {"current", "unflipped", "zero"}; ignored if not use_prior

    def arch(self, in_dim, out_dim):
        """`TrailerModel(total=...)` layer spec: [in, *hidden, out]."""
        return [in_dim, *self.hidden, out_dim]


def _in_cols(spec_mod):
    """Each spec module carries its own IN_COLS; fall back to the trailer spec's
    if a module does not define one (they were identical when div_test.py was
    written, which is why it got away with importing only one)."""
    return tuple(np.asarray(getattr(spec_mod, "IN_COLS", IN_COLS)).tolist())


VARIANTS = (
    Variant(
        key="kintrain_v2",
        label="prior + residual (kintrain, dt-fixed, 128-128-128-128)",
        short_label="Prior + Residual",
        spec=trailer_spec.STATE_FS,
        in_cols=_in_cols(trailer_spec),
        # Verified directly against the checkpoint's tensorstore metadata
        # (array_metadatas/process_0), not assumed: kernel shapes are
        # 80->128, 128->128, 128->128, 128->128, 128->6. NOT 128-128-64-64 --
        # that was an earlier unverified guess against a different, never-
        # uploaded checkpoint. data_kintrain_test.npz's embedded version
        # string is "v2-fiala-H8-dt0.05": H=8, dt already correct in this
        # dataset's collection (the dt=0.02 bug from the session notes is
        # not present here).
        hidden=(128, 128, 128, 128),
        data=ROOT / "data_kintrain_test.npz",
        stats=ROOT / "data_kintrain_test_stats.json",
        ckpt=Path.cwd() / "src/learning/models/trained/beamng-kintrain-test_best",
        use_prior=True,
        prior_mode="current",
    ),
    Variant(
        key="new_model",
        label="pure learned model (test9, 128-128-128-128)",
        short_label="Pure Learned",
        spec=model_spec.STATE_FS,
        in_cols=_in_cols(model_spec),
        hidden=(128, 128, 128, 128),  # verified against checkpoint metadata too
        data=ROOT / "data_trial3_aug1.npz",
        stats=ROOT / "data_proc_test9_stats.json",
        ckpt=Path.cwd() / "src/learning/models/trained/beamng-l4-128-test9_best",
        use_prior=False,
    ),
)


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


def wrap_np(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


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


def make_rollout(model, prior_fn, in_cols, x_mean, x_std, y_mean, y_std, H, k, dt):
    in_cols = np.asarray(in_cols)

    def one(seg):
        """seg: (H+k, 11) ground-truth rows. -> (k, 7) signed errors."""
        buf0 = seg[:H][:, in_cols]  # (H, len(in_cols))
        h0 = jnp.arctan2(seg[H - 1, C_SH], seg[H - 1, C_CH])

        def step(carry, j):
            buf, h = carry

            x = ((buf.reshape(-1) - x_mean) / x_std)[None, :]
            pred = model(x)[0] * y_std + y_mean
            if prior_fn is not None:
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


def run_checks(store, d):
    check_steer_convention(d, store.traj_len)
    check_prior_sign(d, store.traj_len)
    check_yaw_wrap(d, store.traj_len)
    check_excitation(d, store.traj_len, np.asarray(store.meta))


# ----------------------------------------------------------------------------
# per-variant divergence
# ----------------------------------------------------------------------------


def run_variant(v: Variant, store, model, stats):
    d = store.data
    H = v.spec.H
    prior_fn = make_prior(v.prior_mode) if v.use_prior else None

    starts = rollout_starts(store.traj_len, H, K, STRIDE)
    if len(starts) > MAX_STARTS:
        starts = np.random.default_rng(2).choice(starts, MAX_STARTS, replace=False)
    print(
        f"\n=== [{v.key}] open-loop divergence: {len(starts)} starts, K={K}, "
        f"H={H}, prior={v.prior_mode if v.use_prior else 'none'} ==="
    )

    roll = make_rollout(
        model,
        prior_fn,
        v.in_cols,
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


# Plotting statistic. RMSE (mean of squared error) is dominated by a handful of
# rollouts that diverge to 1e6-1e14 -- one exploding candidate swamps the mean
# and the resulting curve tells you nothing except "something diverged". Median
# is robust to that: it reflects what most rollouts do and only moves once more
# than half the pool has blown up. The printed report() table keeps true RMSE
# (that number is a deliberate, honest record); this only affects the figure.
# Plotting statistic. Median was hiding real divergence entirely (a genuinely
# diverging majority just doesn't move a median-based curve) -- back to mean
# RMSE. Splitting low/high speed into separate panel rows (below) is what
# keeps that from swamping the axis, instead of a robust statistic.
PLOT_STAT = "rmse"  # {"rmse", "median"}
DIR_LABEL = {"fwd": "forward", "rev": "reverse"}
DIR_TITLE = {"fwd": "Forward", "rev": "Reverse"}
VARIANT_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# Coarse bins for the figure only -- report()'s per-bucket table stays at the
# finer SPEED_EDGES_KPH resolution for diagnosis. Each bin gets its own row
# of panels (see plot_grid) so a high-speed blowup can't swamp the low-speed
# axis.
PLOT_SPEED_EDGES_KPH = np.array([0, 40, 200])


def speed_band_label(lo, hi, edges=PLOT_SPEED_EDGES_KPH):
    if hi >= edges[-1]:
        return f"|v| {int(lo)}+ km/h"
    return f"|v| {int(lo)}-{int(hi)} km/h"


def report(v: Variant, errs, ic_vx, finite, save=True, report_channels=PLOT_CHANNELS):
    """-> {bucket key: (K, 7) RMSE}. Prints the same table the originals did,
    extended to one growth-fit line per channel in `report_channels` (not just
    `h` -- `vx` was silently only going into the saved/plotted `rows`, never
    printed)."""
    speed_kph = np.abs(ic_vx) * 3.6
    ch_idx = [STATE_NAMES.index(c) for c in report_channels]
    print(f"\n--- [{v.key}] per-step RMSE, by IC speed bucket and direction ---")
    rows = {}
    for lo, hi in zip(SPEED_EDGES_KPH[:-1], SPEED_EDGES_KPH[1:]):
        band = (speed_kph >= lo) & (speed_kph < hi)
        for sgn, lbl in ((ic_vx > 0, "fwd"), (ic_vx < 0, "rev")):
            m = band & sgn & finite
            if m.sum() < MIN_BUCKET_N:
                continue
            e = errs[m]  # (n, K, 7)
            rmse = np.sqrt((e**2).mean(0))  # (K, 7)
            key = f"{lbl} |v| {int(lo):3d}-{int(hi):3d}km/h"
            rows[key] = rmse
            print(f"  {key}  n={int(m.sum()):5d}")
            for cname, ci in zip(report_channels, ch_idx):
                c = rmse[:, ci]
                ks = np.arange(1, len(c) + 1)
                sl = slice(max(1, len(c) // 8), len(c))
                p = np.polyfit(np.log(ks[sl]), np.log(np.maximum(c[sl], 1e-12)), 1)[0]
                print(
                    f"    {cname:8s} k=1 {c[0]:.2e}  k={len(c)//2} {c[len(c)//2]:.2e}  "
                    f"k={len(c)} {c[-1]:.2e}   ratio {c[-1]/max(c[0],1e-12):8.1f}   k^p fit p={p:.2f}"
                )
    print(
        "\n  p ~ 1     : error accumulates linearly (static bias)\n"
        "  p ~ 1.5-2 : compounding bias\n"
        "  p ~ 0.5   : noise accumulation\n"
        "  ratio growing sharply with speed, reverse >> forward, or clearly\n"
        "  super-polynomial growth => the unstable hitch mode dominates and\n"
        "  no amount of one-step accuracy fixes it."
    )

    print(f"\n--- [{v.key}] terminal RMSE per channel (all starts) ---")
    term = np.sqrt((errs[finite][:, -1, :] ** 2).mean(0))
    for name, val in zip(STATE_NAMES, term):
        print(f"  {name:9s} {val:.4e}")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"divergence_{v.key}_K{K}.npz"
        np.savez(
            path,
            errs=errs.astype(np.float32),
            ic_vx=ic_vx.astype(np.float32),
            finite=finite,
            channels=np.array(STATE_NAMES),
        )
        print(f"\n  saved -> {path}")
    return rows


# ----------------------------------------------------------------------------
# combined grid
# ----------------------------------------------------------------------------


def _agg(errs_sub, stat):
    """errs_sub: (n, K, C) signed errors -> (K, C) aggregate magnitude."""
    sq = errs_sub**2
    if stat == "median":
        return np.sqrt(np.median(sq, axis=0))
    if stat == "rmse":
        return np.sqrt(np.mean(sq, axis=0))
    raise ValueError(stat)


def build_curves(errs, ic_vx, finite, channels, stat=PLOT_STAT, edges=PLOT_SPEED_EDGES_KPH):
    """(errs, ic_vx, finite) -> {(lo, hi, direction): (K, C)}.

    Speed bins are on |v| at the rollout IC (both fwd and rev bucket edges use
    the unsigned speed; direction is tracked separately). One curve per
    (bin, direction) -- the plotted aggregate (mean or median RMS, see
    PLOT_STAT).
    """
    ch_idx = [STATE_NAMES.index(c) for c in channels]
    speed_kph = np.abs(ic_vx) * 3.6
    curves = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        for lbl, sgn in (("fwd", ic_vx > 0), ("rev", ic_vx < 0)):
            m = sgn & finite & (speed_kph >= lo) & (speed_kph < hi)
            if m.sum() < MIN_BUCKET_N:
                continue
            curves[(lo, hi, lbl)] = _agg(errs[m][..., ch_idx], stat)
    return curves


# ----------------------------------------------------------------------------
# combined grid
# ----------------------------------------------------------------------------


def plot_grid(curves_by_variant, variants, channels=PLOT_CHANNELS, stat=PLOT_STAT,
              edges=PLOT_SPEED_EDGES_KPH, directions=("fwd", "rev")):
    """curves_by_variant: {variant key: build_curves(...) output, keyed by
    (lo, hi, direction) -> (K, C)}.

    rows = direction (forward, reverse), cols = (channel, speed bin) --
    horizontal/wide layout for a two-column IEEE figure. Each panel overlays
    every variant for one (direction, channel, bin) combo, colour = variant.

    y-limits are shared across a whole CHANNEL GROUP (all its speed bins and
    both direction rows), not per column: same channel means same units, so
    one scale per group makes the low-vs-high-speed comparison meaningful and
    lets every column but the group's first drop its tick labels -- which is
    what was previously colliding into the neighbouring panel. Channel name
    and units appear once per group as the y-label of the group's first
    column; per-column titles carry only the speed bin. Direction is labelled
    once per row on the far left. No figure title -- that goes in the caption.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ch_idx = [STATE_NAMES.index(c) for c in channels]
    nbin = len(list(zip(edges[:-1], edges[1:])))
    bins = list(zip(edges[:-1], edges[1:]))
    col_specs = [(cname, ci, lo, hi) for cname, ci in zip(channels, ch_idx) for lo, hi in bins]
    nr, nc = len(directions), len(col_specs)
    variant_color = {v.key: VARIANT_PALETTE[i % len(VARIANT_PALETTE)] for i, v in enumerate(variants)}
    stat_name = "median RMS" if stat == "median" else "RMSE"

    # Narrow spacer column between channel groups so the grouping reads
    # visually instead of nc undifferentiated columns.
    COL_GROUP_GAP = 0.22  # relative to a normal column's width of 1.0
    width_ratios, col_to_grid = [], {}
    grid_c = 0
    for orig_c in range(nc):
        if orig_c > 0 and orig_c % nbin == 0:
            width_ratios.append(COL_GROUP_GAP)
            grid_c += 1
        width_ratios.append(1.0)
        col_to_grid[orig_c] = grid_c
        grid_c += 1
    total_grid_cols = grid_c

    fig = plt.figure(figsize=(2.85 * nc + 0.22 * (len(channels) - 1), 2.5 * nr))
    gs = fig.add_gridspec(nr, total_grid_cols, width_ratios=width_ratios, hspace=0.12, wspace=0.16)

    # y-limits per channel group, over every bin/direction/variant in it
    group_ylim = {}
    for g, (cname, ci) in enumerate(zip(channels, ch_idx)):
        vals = []
        for lo, hi in bins:
            for v in variants:
                cv = curves_by_variant.get(v.key, {})
                for direction in directions:
                    curve = cv.get((lo, hi, direction))
                    if curve is not None:
                        vals.append(curve[:, ci])
        vals = np.concatenate(vals) if vals else np.array([1.0])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            group_ylim[g] = (max(np.percentile(vals, 1) * 0.7, vals.min()),
                             np.percentile(vals, 99) * 1.5)
        else:
            group_ylim[g] = (1e-3, 1.0)

    # one shared-y anchor per channel group, one shared-x anchor overall
    axes = [[None] * nc for _ in range(nr)]
    x_anchor = None
    group_anchor = {}
    for c in range(nc):
        g = c // nbin
        for r in range(nr):
            ax = fig.add_subplot(
                gs[r, col_to_grid[c]], sharex=x_anchor, sharey=group_anchor.get(g)
            )
            if x_anchor is None:
                x_anchor = ax
            group_anchor.setdefault(g, ax)
            axes[r][c] = ax

    for c, (cname, ci, lo, hi) in enumerate(col_specs):
        g = c // nbin
        first_in_group = (c % nbin == 0)
        for r, direction in enumerate(directions):
            ax = axes[r][c]
            for v in variants:
                curve = curves_by_variant.get(v.key, {}).get((lo, hi, direction))
                if curve is None:
                    continue
                k_ax = np.arange(1, len(curve) + 1)
                ax.semilogy(
                    k_ax, curve[:, ci] * (3.6 if cname == "vx" else 1),  # to km/h
                    color=variant_color[v.key], lw=1.6, label=v.short_label,
                )
            ax.set_ylim(*group_ylim[g])
            ax.grid(alpha=0.3, which="major")
            ax.grid(False, which="minor")
            ax.tick_params(labelsize=8)

            if r == 0:
                ax.set_title(speed_band_label(lo, hi, edges), fontsize=9)
            if r == nr - 1:
                ax.set_xlabel("rollout step $i$", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)

            # tick labels only on the first column of each group (y is shared
            # within the group, so the rest are duplicates that collide)
            if first_in_group:
                ax.set_ylabel(
                    f"{CHANNEL_LABEL.get(cname, cname)} {stat_name}"
                    f" ({CHANNEL_UNITS.get(cname, '-')})",
                    fontsize=9,
                )
            else:
                ax.tick_params(labelleft=False)

    # direction label once per row, on the far left
    for r, direction in enumerate(directions):
        axes[r][0].annotate(
            DIR_TITLE.get(direction, direction),
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-58, 0), textcoords="offset points",
            rotation=90, ha="center", va="center", fontsize=11,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:  # first panel may have plotted nothing (e.g. only rev)
        for r in range(nr):
            for c in range(nc):
                handles, labels = axes[r][c].get_legend_handles_labels()
                if handles:
                    break
            if handles:
                break
    by_label = dict(zip(labels, handles))
    # Explicit margins rather than tight_layout: the spacer-column GridSpec is
    # "not compatible with tight_layout", which fails with only a warning and
    # silently ignores the reserved rect -- that is what kept dropping the
    # legend on top of the x-labels.
    gs.update(left=0.095, right=0.995, top=0.93, bottom=0.16)
    if by_label:
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            ncol=len(by_label), fontsize=9, frameon=False,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"divergence_grid_K{K}.pdf"
    fig.savefig(path)
    print(f"\n  saved -> {path}")
    return path


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------


def verify_ckpt_arch(ckpt: Path, arch):
    """Cross-check `arch` (the [in, *hidden, out] list passed to TrailerModel)
    against the checkpoint's own tensorstore metadata -- plain JSON, no JAX or
    orbax needed. Catches an architecture mismatch (wrong Variant.hidden, or a
    config pointed at the wrong checkpoint entirely) before the much less
    legible orbax shape-mismatch error, or worse, before a *silent* wrong
    restore. This is exactly the class of bug that produced meaningless
    divergence plots earlier in this project: Variant.hidden was guessed
    against a checkpoint that had never actually been inspected.
    """
    meta_path = ckpt / "array_metadatas" / "process_0"
    if not meta_path.exists():
        print(f"  (no array_metadatas at {meta_path}; skipping offline arch check)")
        return
    meta = json.loads(meta_path.read_text())
    shapes = {}
    for a in meta["array_metadatas"]:
        m = a["array_metadata"]
        shapes[m["param_name"]] = tuple(m["write_shape"])
    kernels = sorted(
        (k for k in shapes if k.endswith("kernel.value")),
        key=lambda k: int(k.split(".")[2]),
    )
    if not kernels:
        print("  (no kernel params found in checkpoint metadata; skipping arch check)")
        return
    ckpt_arch = [shapes[kernels[0]][0]] + [shapes[k][1] for k in kernels]
    if ckpt_arch != list(arch):
        raise RuntimeError(
            f"configured arch {list(arch)} does not match {ckpt.name}'s actual "
            f"layer shapes {ckpt_arch} (read from array_metadatas/process_0). "
            "Fix Variant.hidden -- do not guess it."
        )
    print(f"  arch verified against checkpoint metadata: {'-'.join(str(a) for a in ckpt_arch)}")


def restore_state(ckpt: Path, model):
    """Restore into `model` regardless of whether the checkpoint was written as
    an `nnx.State` or as a pure dict (`state.to_pure_dict()`).

    Orbax matches the on-disk tree structure against the template you pass, and
    the two forms are not interchangeable -- a State template against a dict
    tree raises "tree structures do not match" before any shape is checked.
    Older checkpoints in this repo are pure dicts; newer ones are States.
    """
    ckptr = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    try:
        nnx.update(model, ckptr.restore(ckpt, state))
        return "state"
    except ValueError as e:
        if "structures do not match" not in str(e):
            raise
    pure = state.to_pure_dict()
    state.replace_by_pure_dict(ckptr.restore(ckpt, pure))
    nnx.update(model, state)
    return "pure_dict"


def load_variant(v: Variant, checks=False):
    store = DataStore.load(v.data)
    d = np.asarray(store.data)
    print(
        f"\n[{v.key}] loaded {v.data}: {d.shape}, {len(store.traj_len)} trajectories, "
        f"version={store.version}, spec={v.spec.data_version}"
    )
    if store.version != v.spec.data_version:
        print("  WARNING: data_version mismatch between store and spec")

    if checks:
        run_checks(store, d)

    with open(v.stats) as f:
        stats = json.load(f)
    y_std = np.asarray(stats["y_std"])
    print(f"  y_std = {np.array2string(y_std, precision=3)}")
    if y_std[2] < 1.0:
        print("  WARNING: y_std[2:4] looks like rate stats, not accelerations -- "
              "stale normalisation for this spec")

    # the (spec, stats, checkpoint) triple has desynced before; this catches it
    in_dim = v.spec.H * len(v.in_cols)
    if len(stats["x_mean"]) != in_dim:
        raise RuntimeError(
            f"[{v.key}] stats x_mean has {len(stats['x_mean'])} entries but spec gives "
            f"H={v.spec.H} x {len(v.in_cols)} cols = {in_dim}. "
            "Wrong stats file for this spec, or IN_COLS changed since it was written."
        )
    if v.use_prior and list(np.asarray(v.in_cols)[:8]) != list(np.asarray(X_COLS)):
        raise RuntimeError(
            f"[{v.key}] use_prior=True requires IN_COLS[:8] == X_COLS "
            "(the rollout feeds buf[-1, :8] straight to fiala_dyn)."
        )

    arch = v.arch(in_dim, N_OUT)
    print(f"  arch (configured) = {'-'.join(str(a) for a in arch)}")
    verify_ckpt_arch(v.ckpt, arch)
    model = TrailerModel(in_dim, N_OUT, total=arch)
    fmt = restore_state(v.ckpt, model)
    print(f"  restored {v.ckpt.name} (on-disk tree: {fmt})")
    return store, model, stats


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", action="store_true", help="diagnostics only, no checkpoint")
    ap.add_argument("--no-checks", action="store_true", help="divergence only")
    ap.add_argument("--only", choices=[v.key for v in VARIANTS], help="run one variant")
    ap.add_argument("--replot", action="store_true", help="rebuild the grid from saved npz")
    ap.add_argument(
        "--stat", choices=["median", "rmse"], default=PLOT_STAT,
        help="plotted aggregate (report() table always prints true RMSE)",
    )
    args = ap.parse_args()

    variants = [v for v in VARIANTS if args.only in (None, v.key)]

    if args.replot:
        curves = {}
        for v in variants:
            path = OUT / f"divergence_{v.key}_K{K}.npz"
            if not path.exists():
                print(f"  [{v.key}] no saved run at {path}; skipping")
                continue
            z = np.load(path)
            report(v, z["errs"], z["ic_vx"], z["finite"], save=False)
            curves[v.key] = build_curves(z["errs"], z["ic_vx"], z["finite"], PLOT_CHANNELS, args.stat)
        plot_grid(curves, [v for v in variants if v.key in curves], stat=args.stat)
        return

    if args.checks:
        for v in variants:
            store = DataStore.load(v.data)
            print(f"\n########## [{v.key}] {v.data} ##########")
            run_checks(store, np.asarray(store.data))
        return

    datasets = {v.data for v in variants}
    if len(datasets) > 1:
        print(
            "\nNOTE: the variants are evaluated on DIFFERENT datasets "
            f"({', '.join(p.name for p in sorted(datasets, key=str))}). "
            "Bucket-wise curves are still comparable in shape and exponent, but the\n"
            "      absolute levels also reflect different ICs. Point both `data` fields\n"
            "      at one store if you want a like-for-like level comparison."
        )

    curves = {}
    for v in variants:
        store, model, stats = load_variant(v, checks=not args.no_checks)
        errs, ic_vx, finite = run_variant(v, store, model, stats)
        report(v, errs, ic_vx, finite)
        curves[v.key] = build_curves(errs, ic_vx, finite, PLOT_CHANNELS, args.stat)
        del store, model  # free device memory before the next checkpoint loads

    plot_grid(curves, variants, stat=args.stat)


if __name__ == "__main__":
    main()