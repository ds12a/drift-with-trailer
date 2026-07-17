from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List
import numpy as np
import jax
import json

"""
Refactored after the old class gave great difficulty.

To avoid shennanigans, collection and distribtion/loading are separated
"""


class DataCollector:
    """
    Serves purely as a collector class with a very large cache.
    """

    raw_data: List = []  # (n_traj, n_time, D) - All info per step
    traj_len: List = []  # (n_traj)
    meta: List = []  # (n_traj, 3)

    def __init__(self, data_size, dt):
        self.dt = dt
        self.data_size = data_size

        self.n = 0

    def add(self, data: np.ndarray, run, ctrl, step):
        """
        Assuming 2D input (a whole trajectory history)
        """

        assert(data.ndim == 2 and data.shape[1] == self.data_size)
        self.raw_data.append(data)
        self.traj_len.append(len(data))
        self.meta.append([run, ctrl, step])

    def store(self, version):
        """
        Stores data in a DataStore
        """
        data = np.concatenate(self.raw_data, axis=0).astype(np.float32, copy=False)
        traj_len = np.asarray(self.traj_len)
        meta = np.asarray(meta)
        
        return DataStore(
            data=data,
            traj_len=traj_len,
            meta=meta,
            dt=self.dt,
            version=version
        )
    
@dataclass(frozen=True)
class FeatureSpec:
    """
    Everything needed to rebuild a DataLoader from an archive.
    """
    encode_x: Callable  # NN input, encodes horizon into network input
    target_fn: Callable  # NN output, encodes horizon into network output
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

    data: np.ndarray  # (N, 9)
    traj_len: np.ndarray
    meta: np.ndarray  # (n_traj, 3): run, ctrl, step
    dt: float
    version: str

    _KEYS = ("data", "traj_len", "meta", "dt", "version")

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self._KEYS})
    
    @classmethod
    def load(path):
        with np.load(path) as f:
            return DataStore(**{k: f[k] for k in DataStore._KEYS})

    def build(self, spec: FeatureSpec):

        # Building valid_starts
        traj_start = np.concatenate([[0], np.cumsum(self.traj_len)[:-1]])
        n_win = np.maximum(self.traj_len - spec.H - spec.F + 1, 0)
        total = int(n_win.sum())
        base = np.repeat(traj_start, n_win)
        offs = np.arange(total) - np.repeat(np.cumsum(n_win) - n_win, n_win)
        valid_starts = (base + offs).astype(np.int32)

        # Train/test split
        runs = np.unique(self.meta[:, 0])


@dataclass(frozen=True)
class DataLoader:
    """
    DataLoader internal data is intended to be immutable. If augmentation
    of dataset is desired, keep/regenerate the DataCollector instead
    """

    data: np.ndarray
    valid: np.ndarray  # Contains valid idx (given horizon considerations)
    is_train: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    spec: FeatureSpec

    # def __init__(
    #         self, 
    #         data: np.ndarray,  # (n, d), where n is total number of datapoints
    #         valid_idx: np.ndarray,  # valid first indices for horizon
    #         traj_len: np.ndarray,  # from datacollector, in case horizon needs changing
    #         batch_size: int,
    #         horizon_len: int,
    #         ):
    #     """
    #     This constructor should not be used externally
    #     """
    #     self.data = data
    #     self.valid_idx = valid_idx
    #     self.traj_len = traj_len
    #     self.batch_size = batch_size
    #     self.horizon_len = horizon_len
        
    #     self.norm_state = False # Do not pass normalized data!
        
    # @classmethod
    # def generate(c: DataCollector, horizon_len=4, batch_size=64):
    #     """
    #     datacollector internal contents are kept undisturbed, could change for less memory impact
    #     """

    #     data = np.concatenate(c.raw_data, axis=0).astype(np.float32, copy=False)
    #     traj_len = np.asarray(c.traj_len)

    #     # valid_start creation
    #     traj_start = np.concatenate([[0], np.cumsum(traj_len)[:-1]])
    #     n_win = np.maximum(traj_len - horizon_len + 1, 0)
    #     total = int(n_win.sum())
    #     base = np.repeat(traj_start, n_win)
    #     offs = np.arange(total) - np.repeat(np.cumsum(n_win) - n_win, n_win)
    #     valid_starts = (base + offs).astype(np.int32)

    #     l = DataLoader(data, valid_starts, traj_len, batch_size)
    #     return l
    
    # def normalize(self, proc_fn=None, prior_fn_dim=-1, verbose=False):
    #     """
    #     Processor function takes state and calculates another value i.e. residual, etc.
    #     Because of dimension change (potentially), normalization is irreversible
    #     """

    #     N, D = self.data.shape

    #     CHUNK = 1 << 20  # Nice alignment in RAM
    #     a = _Acc(D if proc_fn is None else prior_fn_dim)

    #     if proc is not None:
    #         proc = jax.jit(jax.vmap(proc_fn))
    #     data_new = []
        
    #     for i in range(0, N, CHUNK):
    #         if proc_fn is not None:
    #             data_new.append(proc(self.data[i:i+CHUNK]))
    #         else:
    #             data_new.append(self.data[i:i+CHUNK])
    #         a.push(data_new[-1])
        
    #     m, std, lo, hi = a.out()
    #     self.m = m
    #     self.std = std
    #     self.data = np.concatenate(data_new, axis=0).astype(np.float32, copy=False) # TODO This is a bit wasteful
    #     self.data = (self.data - m) / std
    #     self.norm_state = True

    # def statistics(self):
    #     raise NotImplementedError
    
    # @classmethod
    # def load(path):
    #     path = Path(path)
    #     assert(path.suffix.lower() == ".npz")
    #     with open(path, 'rb') as f:
    #         file = np.load(f)
        
    #     return DataLoader(
    #         file["data"], 
    #         file["valid_idx"],
    #         file["traj_len"],
    #         file["batch_size"],
    #         file["horizon_len"],
    #     )

    # def save(self, path):
    #     assert(not self.norm_state)
    #     path = Path(path)

    #     self_dict = {
    #         "data": self.data,
    #         "valid_idx": self.valid_idx,
    #         "traj_len": self.traj_len,
    #         "batch_size": self.batch_size,
    #         "horizon_len": self.horizon_len,
    #     }

    #     with open(path, 'wb') as f:
    #         file = np.savez(f, **self_dict)


        
        