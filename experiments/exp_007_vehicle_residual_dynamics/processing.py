
# Utils for rewriting the pickle
# python -m experiments.exp_007_vehicle_residual_dynamics.renormalize

import argparse
import pickle
import shutil
from pathlib import Path

import jax
import numpy as np

STATE_NAMES = ["sin_h", "cos_h", "vx", "vy", "w1", "w2", "mu", "delta", "accel"]
DYN_NAMES = ["d_sin_h", "d_cos_h", "ax", "ay", "alpha1", "alpha2"]

DEFAULT_PKL = Path("experiments/exp_007_vehicle_residual_dynamics/data.pkl")


class _Acc:
    def __init__(self, d):
        self.s = np.zeros(d, np.float64)
        self.s2 = np.zeros(d, np.float64)
        self.lo = np.full(d, np.inf)
        self.hi = np.full(d, -np.inf)
        self.n = 0

    def push(self, x):
        self.s += x.sum(0)
        self.s2 += (x * x).sum(0)
        self.lo = np.minimum(self.lo, x.min(0))
        self.hi = np.maximum(self.hi, x.max(0))
        self.n += x.shape[0]

    def out(self):
        m = self.s / self.n
        v = np.maximum(self.s2 / self.n - m * m, 0.0)
        return m, np.sqrt(v), self.lo, self.hi


def _report(names, m_o, s_o, m_n, s_n, lo, hi, tag):
    print(f"\n=== {tag} ===")
    print(f"{'chan':<9}{'old_mean':>10}{'old_std':>10}{'new_mean':>10}{'new_std':>10}"
          f"{'std_x':>8}{'z_min':>9}{'z_max':>9}")
    for j, nm in enumerate(names):
        zlo, zhi = (lo[j] - m_n[j]) / s_n[j], (hi[j] - m_n[j]) / s_n[j]
        ratio = s_o[j] / s_n[j]
        flag = "  <-- " if (ratio > 3 or ratio < 1 / 3) else ""
        print(f"{nm:<9}{m_o[j]:>10.4f}{s_o[j]:>10.4f}{m_n[j]:>10.4f}{s_n[j]:>10.4f}"
              f"{ratio:>8.2f}{zlo:>9.2f}{zhi:>9.2f}{flag}")


def renormalize(data, prior_fn=None, chunk=1 << 20, std_floor=1e-8,
                apply=True, verbose=True):
    """
    Measure per-channel stats from `data`, report, and (if apply) rescale in place.

    Sets state_mean/state_std, dynamics_mean/dynamics_std to measured values and
    adds res_mean/res_std (the std the loss should actually divide by).
    """
    if not getattr(data, "_is_compiled", False):
        data._compile_dataset()

    fs, fd = data.flat_states, data.flat_dynamics
    N = fs.shape[0]
    assert fd.shape[0] == N, f"state/dyn row mismatch: {fs.shape} vs {fd.shape}"

    ms_o = np.asarray(data.state_mean, np.float64)
    ss_o = np.asarray(data.state_std, np.float64)
    md_o = np.asarray(data.dynamics_mean, np.float64)
    sd_o = np.asarray(data.dynamics_std, np.float64)

    a_s, a_d = _Acc(fs.shape[1]), _Acc(fd.shape[1])
    a_r = _Acc(fd.shape[1]) if prior_fn is not None else None
    pred = jax.jit(jax.vmap(prior_fn)) if prior_fn is not None else None

    for i in range(0, N, chunk):
        xs = fs[i : i + chunk].astype(np.float64) * ss_o + ms_o
        yd = fd[i : i + chunk].astype(np.float64) * sd_o + md_o
        a_s.push(xs)
        a_d.push(yd)
        if a_r is not None:
            a_r.push(yd - np.asarray(pred(xs.astype(np.float32)), np.float64))

    ms_n, ss_n, s_lo, s_hi = a_s.out()
    md_n, sd_n, d_lo, d_hi = a_d.out()

    degenerate = ss_n < std_floor
    if degenerate.any():
        print(f"!! near-constant state channels: "
              f"{[STATE_NAMES[i] for i in np.where(degenerate)[0]]}")
    ss_n = np.maximum(ss_n, std_floor)
    sd_n = np.maximum(sd_n, std_floor)

    if verbose:
        _report(STATE_NAMES, ms_o, ss_o, ms_n, ss_n, s_lo, s_hi, "STATE")
        _report(DYN_NAMES, md_o, sd_o, md_n, sd_n, d_lo, d_hi, "DYNAMICS (raw targets)")

    if a_r is not None:
        mr_n, sr_n, r_lo, r_hi = a_r.out()
        sr_n = np.maximum(sr_n, std_floor)
        if verbose:
            _report(DYN_NAMES, md_o, sd_o, mr_n, sr_n, r_lo, r_hi, "RESIDUAL (actual target)")
            print("\nprior explained-variance per channel (1 - Var[r]/Var[y]):")
            for j, nm in enumerate(DYN_NAMES):
                print(f"  {nm:<9}{1 - (sr_n[j] / sd_n[j]) ** 2:>8.4f}")

    if not apply:
        print("\n[dry-run] no mutation, no save")
        return data

    # single in-place affine: z_new = z_old*(s_old/s_new) + (m_old-m_new)/s_new
    for flat, m_o, s_o, m_n, s_n in ((fs, ms_o, ss_o, ms_n, ss_n),
                                     (fd, md_o, sd_o, md_n, sd_n)):
        a = (s_o / s_n).astype(np.float32)
        b = ((m_o - m_n) / s_n).astype(np.float32)
        for i in range(0, flat.shape[0], chunk):
            flat[i : i + chunk] *= a
            flat[i : i + chunk] += b

    data.state_mean, data.state_std = ms_n, ss_n
    data.dynamics_mean, data.dynamics_std = md_n, sd_n
    if a_r is not None:
        data.res_mean, data.res_std = mr_n, sr_n

    bounds = np.cumsum(data.traj_len)[:-1]
    data.states = list(np.split(data.flat_states, bounds))      # views, no copy
    data.dynamics = list(np.split(data.flat_dynamics, bounds))

    if verbose:
        chk = data.flat_states[: min(N, 1 << 20)]
        print(f"\nsanity: z-mean {chk.mean(0).round(3)}\n        z-std  {chk.std(0).round(3)}")
    return data

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--src", type=Path, default=DEFAULT_PKL)
#     ap.add_argument("--dst", type=Path, default=None, help="default: in-place")
#     ap.add_argument("--no-prior", action="store_true",
#                     help="skip residual stats (don't import train.py)")
#     ap.add_argument("--dry-run", action="store_true")
#     ap.add_argument("--no-backup", action="store_true")
#     args = ap.parse_args()

#     dst = args.dst or args.src

#     prior_fn = None
#     if not args.no_prior:
#         from experiments.exp_007_vehicle_residual_dynamics.train import dynamics as prior_fn

#     print(f"loading {args.src}")
#     with open(args.src, "rb") as f:
#         data = pickle.load(f)
#     print(f"{len(data.states)} trajectories, {sum(data.traj_len)} rows")

#     renormalize(data, prior_fn=prior_fn, apply=not args.dry_run)
#     if args.dry_run:
#         return

#     if dst == args.src and not args.no_backup:
#         bak = args.src.with_suffix(".pkl.bak")
#         if not bak.exists():
#             shutil.copy2(args.src, bak)
#             print(f"backup -> {bak}")

#     assert data._is_compiled, "expected compiled cache before save"
#     tmp = dst.with_suffix(".pkl.tmp")
#     with open(tmp, "wb") as f:
#         pickle.dump(data, f, protocol=5)
#     tmp.replace(dst)
#     print(f"saved -> {dst} ({dst.stat().st_size / 1e6:.0f} MB, cache included)")

STATE_KEEP = [0, 1, 2, 3, 7, 8]     # sh, ch, vx, vy, delta, accel
KIN_STATE_NAMES = ["sin_h", "cos_h", "vx", "vy", "delta", "accel"]
KIN_DYN_NAMES = ["w1", "w2", "ax", "ay"]

def to_kinematic_targets(data, chunk=1 << 20, std_floor=1e-8):
    """
    Reindex a normalized Data into the kinematic-output framing.
    Not idempotent: run once, on data already passed through renormalize().
    """
    if not getattr(data, "_is_compiled", False):
        data._compile_dataset()

    ms, ss = np.asarray(data.state_mean), np.asarray(data.state_std)
    md, sd = np.asarray(data.dynamics_mean), np.asarray(data.dynamics_std)
    N = data.flat_states.shape[0]

    new_s = np.empty((N, len(STATE_KEEP)), np.float32)
    new_d = np.empty((N, 4), np.float32)
    for i in range(0, N, chunk):
        rs = data.flat_states[i : i + chunk] * ss + ms
        rd = data.flat_dynamics[i : i + chunk] * sd + md
        new_s[i : i + chunk] = rs[:, STATE_KEEP]
        new_d[i : i + chunk] = np.concatenate([rs[:, 4:6], rd[:, 2:4]], 1)

    ms_n, ss_n = new_s.mean(0, dtype=np.float64), new_s.std(0, dtype=np.float64)
    md_n, sd_n = new_d.mean(0, dtype=np.float64), new_d.std(0, dtype=np.float64)
    ss_n[0:2], ms_n[0:2] = 1.0, 0.0          # sin/cos already O(1); preserve sh^2+ch^2=1
    ss_n = np.maximum(ss_n, std_floor)
    sd_n = np.maximum(sd_n, std_floor)

    for j, nm in enumerate(KIN_STATE_NAMES):
        print(f"state  {nm:<8}{ms_n[j]:>9.4f}{ss_n[j]:>9.4f}")
    for j, nm in enumerate(KIN_DYN_NAMES):
        print(f"target {nm:<8}{md_n[j]:>9.4f}{sd_n[j]:>9.4f}")

    data.flat_states = ((new_s - ms_n) / ss_n).astype(np.float32)
    data.flat_dynamics = ((new_d - md_n) / sd_n).astype(np.float32)
    data.state_mean, data.state_std = ms_n, ss_n
    data.dynamics_mean, data.dynamics_std = md_n, sd_n
    for a in ("res_mean", "res_std"):
        data.__dict__.pop(a, None)           # stale; rerun renormalize with the new prior
    data._relist_views()
    return data

KIN_PKL = Path("experiments/exp_007_vehicle_residual_dynamics/data_kin.pkl")

if __name__ == "__main__":
    with open(DEFAULT_PKL, 'rb') as f:
        data = pickle.load(f)
    
    data = to_kinematic_targets(data)

    with open(KIN_PKL, 'wb') as f:
        pickle.dump(data, f)

