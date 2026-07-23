from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import jax
import json

"""
Refactored after the old class gave great difficulty.

To avoid shennanigans, collection and distribtion/loading are separated

DataCollector is purely for data collection

DataStore is purely for storing raw unprocessed data (in case we need to modify the 
data feed to the network)

DataLoader is purely for data processing + feeding training loop
"""


class DataCollector:
    """
    Serves purely as a collector class with a very large cache.
    """

    def __init__(self, data_size, dt):
        self.dt = dt
        self.data_size = data_size

        self.raw_data = []  # (n_traj, n_time, D) - All info per step
        self.traj_len = []  # (n_traj)
        self.meta = []  # (n_traj, 3)

        self.n = 0

    def add(self, data: np.ndarray, run, ctrl, step):
        """
        Assuming 2D input (a whole trajectory history)
        """

        assert data.ndim == 2 and data.shape[1] == self.data_size
        self.raw_data.append(data)
        self.traj_len.append(len(data))
        self.meta.append([run, ctrl, step])

    def store(self, version, verbose=False):
        """
        Stores data in a DataStore
        """
        data = np.concatenate(self.raw_data, axis=0).astype(np.float32, copy=False)
        traj_len = np.asarray(self.traj_len)
        meta = np.asarray(self.meta)

        if verbose:
            print(f"Generating DataStore, data is of size {data.shape}")

        return DataStore(data=data, traj_len=traj_len, meta=meta, dt=self.dt, version=version)


@dataclass(frozen=True)
class FeatureSpec:
    """
    Everything needed to rebuild a DataLoader from an archive.
    """

    encode_x: Callable  # NN input, encodes horizon into network input
    encode_y: Callable  # NN output, encodes horizon into network output
    H: int = 4  # Pre-horizon
    F: int = 1  # Post-horizon
    train_frac: float = 0.7
    split_seed: int = 137
    data_version: str = "v1"


@dataclass(frozen=True)
class DataStore:
    """
    Stores/loads trajectories in a processed numpy state.
    """

    data: np.ndarray  # (N, D)
    traj_len: np.ndarray
    meta: np.ndarray  # (n_traj, 3): run, ctrl, step
    dt: float
    version: str

    _KEYS = ("data", "traj_len", "meta", "dt", "version")

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self._KEYS})

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            d = {k: f[k] for k in cls._KEYS}
        return cls(**{**d, "dt": float(d["dt"]), "version": str(d["version"])})

    def build(self, spec: FeatureSpec, verbose=False):
        """
        Builds masks for training, computes statistics for normalization.
        Actual normalization happens during training
        """

        if verbose:
            print(f"Building DataLoader, data {self.data.shape}")
        return DataLoader(self.data, self.traj_len, self.meta, self.dt, spec)


@dataclass(frozen=True)
class DataLoader:
    """
    DataLoader internal data is intended to be immutable. If augmentation
    of dataset is desired, keep/regenerate the DataCollector instead
    """

    data: np.ndarray
    traj_len: np.ndarray
    meta: np.ndarray
    dt: float
    spec: FeatureSpec
    x_mean: Optional[np.ndarray] = None  # None -> computed in __post_init__
    x_std: Optional[np.ndarray] = None
    y_mean: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None

    _KEYS = ("data", "traj_len", "meta", "dt", "x_mean", "x_std", "y_mean", "y_std")

    def __post_init__(self):
        object.__setattr__(self, "version", self.spec.data_version)
        train, test = self._split()  # Lazy cache
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)

        if self.x_mean is None or self.y_mean is None or self.x_std is None or self.y_std is None:
            for k, v in zip(("x_mean", "x_std", "y_mean", "y_std"), self.compute_stats()):
                object.__setattr__(self, k, v)

    def get_data(self, batch_size, key=None):
        key = jax.random.PRNGKey(137) if key is None else key
        k1, k2 = jax.random.split(key)
        return self._batch(self.train, batch_size, k1), self._batch(self.test, batch_size, k2)

    def _batch(self, idx, B, key):
        spec = self.spec
        W = np.arange(self.spec.H + self.spec.F)

        n = (len(idx) // B) * B
        p = jax.random.permutation(key, len(idx))[:n].reshape(-1, B)

        for b in range(len(p)):
            # pb_idx = idx[p[b]]
            w = self.data[idx[p[b]][:, None] + W]  # (B, H+F, D)
            yield (
                (spec.encode_x(w) - self.x_mean) / self.x_std,
                (spec.encode_y(w) - self.y_mean) / self.y_std,
            )

    def _split(self):
        """
        (train, test) window starts
        """
        spec = self.spec
        valid, tid = self._windows(self.traj_len)
        t_n = len(self.traj_len)
        keep = np.zeros(t_n, bool)
        keep[
            np.random.default_rng(spec.split_seed).permutation(t_n)[: int(t_n * spec.train_frac)]
        ] = True
        m = keep[tid]
        return valid[m], valid[~m]

    def _windows(self, traj_len):
        """
        Recomputes window
        """
        spec = self.spec
        traj_start = np.concatenate([[0], np.cumsum(traj_len)[:-1]])
        n_win = np.maximum(traj_len - spec.H - spec.F + 1, 0)
        base = np.repeat(traj_start, n_win)
        offs = np.arange(int(n_win.sum())) - np.repeat(np.cumsum(n_win) - n_win, n_win)
        return (base + offs).astype(np.int32), np.repeat(np.arange(len(traj_len)), n_win)

    def compute_stats(self, chunk=1 << 19):
        """
        Train-only stats over encode_x/encode_y outputs
        """
        data = self.data
        spec = self.spec
        train = self.train

        W = np.arange(spec.H + spec.F)
        acc = {}
        for i in range(0, len(train), chunk):
            w = data[train[i : i + chunk, None] + W]
            for k, v in (("x", spec.encode_x(w)), ("y", spec.encode_y(w))):
                v = np.asarray(v, np.float64)
                a = acc.setdefault(k, [np.zeros(v.shape[1]), np.zeros(v.shape[1]), 0])
                a[0] += v.sum(0)
                a[1] += (v * v).sum(0)
                a[2] += len(v)

        def stat(k):
            s, s2, n = acc[k]
            mu = s / n
            sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
            return mu.astype(np.float32), np.maximum(sd, 1e-8).astype(np.float32)

        (xm, xs), (ym, ys) = stat("x"), stat("y")
        return xm, xs, ym, ys

    # def update(self, spec: FeatureSpec):
    #     """
    #     Goofy, should not be used under normal conditions, only if somehow there is
    #     no more memory
    #     """
    #     object.__setattr__(self, "spec", spec)
    #     object.__setattr__(self, "version", self.spec.data_version)
    #     train, test = self._split(self.traj_len)  # Lazy cache
    #     object.__setattr__(self, "train", train)
    #     object.__setattr__(self, "test", test)
    #     xm, xs, ym, ys = self.compute_stats(train, spec)
    #     object.__setattr__(self, "x_mean", xm)
    #     object.__setattr__(self, "x_std", xs)
    #     object.__setattr__(self, "y_mean", ym)
    #     object.__setattr__(self, "y_std", ys)

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self._KEYS}, version=self.spec.data_version)

    @classmethod
    def load(cls, path, spec: FeatureSpec):
        with np.load(path) as f:
            # assert f["version"].item() == spec.data_version
            d = {k: f[k] for k in cls._KEYS}
        return cls(**{**d, "dt": float(d["dt"])}, spec=spec)
