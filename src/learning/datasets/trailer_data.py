import numpy as np
import jax
import json


class Data:

    def __init__(
        self,
        batch_size,
        state_mean,
        state_std,
        dynamics_mean,
        dynamics_std,
        horizon_len=4,
    ):

        # State, dynamics 3D array size (run idx, timestep, data)
        self.states = []
        self.dynamics = []
        self.traj_len = []

        self.n = 0

        self.state_std = state_std
        self.state_mean = state_mean
        self.dynamics_std = dynamics_std
        self.dynamics_mean = dynamics_mean
        self.batch_size = batch_size
        self.horizon_len = horizon_len
        self.key = jax.random.PRNGKey(1)

    def __len__(self):

        true_traj_len = [max(i + 1 - self.horizon_len, 0) for i in self.traj_len]
        self.true_traj_len_buffer = true_traj_len  # Maybe unecessary

        return sum(true_traj_len)

    def add(self, state, dynamics):
        """
        Assuming 2D input (a whole trajectory history)
        """
        n_old = len(self)

        state, dynamics = self._normalize(state, dynamics)

        self.states.append(state)
        self.dynamics.append(dynamics)
        self.traj_len.append(len(state))
        self.n += len(state)

    def get_data(self, train_test=0.7):

        if not getattr(self, '_is_compiled', False):
            self._compile_dataset()

        n = len(self)  # refresh true_traj_len_buffer

        fixed_key = jax.random.PRNGKey(137)
        all = np.array(jax.random.permutation(fixed_key, n))

        split_idx = int(n * train_test)
        train_indices = all[:split_idx]
        test_indices = all[split_idx:]

        self.key, k1, k2 = jax.random.split(self.key, 3)
        train = np.array(jax.random.permutation(k1, train_indices))
        test = np.array(jax.random.permutation(k2, test_indices))

        return self._batch(train), self._batch(test)
    
    def _compile_dataset(self):
        """
        Flattens ragged trajectories into contiguous memory
        """
        print("Flattening dataset")
        
        self.flat_states = np.concatenate(self.states, axis=0).astype(np.float32)
        self.flat_dynamics = np.concatenate(self.dynamics, axis=0).astype(np.float32)
        valid_starts = []
        current_idx = 0
        for length in self.traj_len:
            if length >= self.horizon_len:
                starts = np.arange(current_idx, current_idx + length - self.horizon_len + 1)
                valid_starts.append(starts)
            current_idx += length
        
        self.valid_starts = np.concatenate(valid_starts)
        self._is_compiled = True
        print(f"Compilation complete. {len(self.valid_starts)} valid windows available.")

    def _batch(self, perm):
        s = len(self.states[0][0])
        d = len(self.dynamics[0][0])
        h = self.horizon_len

        leftover = len(perm) % self.batch_size
        perm = perm[leftover:]

        B = len(perm) // self.batch_size
        batch_perm = perm.reshape((B, self.batch_size))

        # traj_len_prefix = np.cumsum(np.insert(self.true_traj_len_buffer, 0, 0))
        # traj_indices = np.searchsorted(traj_len_prefix[1:], batch_perm, side="right")
        # offset = batch_perm - traj_len_prefix[traj_indices]

        window_offsets = np.arange(h)

        # batched_states = np.zeros((B, self.batch_size, h * s))
        # batched_dynamics = np.zeros((B, self.batch_size, h * d))

        for b in range(B):

            starts = self.valid_starts[batch_perm[b]]
            window_indices = starts[:, None] + window_offsets
            batch_states = self.flat_states[window_indices].reshape(self.batch_size, h * s)
            batch_dynamics = self.flat_dynamics[starts + h - 1]

            # batch_states = np.zeros((self.batch_size, h * s), dtype=np.float32)
            # batch_dynamics = np.zeros((self.batch_size, d), dtype=np.float32)

            # for i in range(self.batch_size):
            #     t_idx = traj_indices[b, i]
            #     o_idx = offset[b, i]
                
            #     st_window = self.states[t_idx][o_idx : o_idx + h]
            #     batch_dynamics[i] = self.dynamics[t_idx][o_idx + h - 1]

            #     batch_states[i] = st_window.ravel()
            #     # batch_dynamics[i] = dy_window.ravel()

            # So my gpu does not die
            yield batch_states, batch_dynamics

    def _normalize(self, states, dynamics):
        states = (states - self.state_mean) / self.state_std
        dynamics = (dynamics - self.dynamics_mean) / self.dynamics_std

        return states, dynamics

    def dump_json(self, filepath: str):

        # Convert internal arrays to list
        data_dict = {
            "batch_size": self.batch_size,
            "horizon_len": self.horizon_len,
            "n": self.n,
            "traj_len": self.traj_len,
            "state_mean": np.asarray(self.state_mean).tolist(),
            "state_std": np.asarray(self.state_std).tolist(),
            "dynamics_mean": np.asarray(self.dynamics_mean).tolist(),
            "dynamics_std": np.asarray(self.dynamics_std).tolist(),
            "states": [np.asarray(s).tolist() for s in self.states],
            "dynamics": [np.asarray(d).tolist() for d in self.dynamics],
            # Convert JAX PRNGKey to a list of integers
            "key": np.asarray(self.key).tolist(),
        }

        with open(filepath, "w") as f:
            json.dump(data_dict, f)

    @classmethod
    def load_json(cls, filepath: str):
        """Loads a Data object from a JSON file."""
        import json

        with open(filepath, "r") as f:
            data_dict = json.load(f)

        obj = cls(
            batch_size=data_dict["batch_size"],
            state_mean=np.array(data_dict["state_mean"]),
            state_std=np.array(data_dict["state_std"]),
            dynamics_mean=np.array(data_dict["dynamics_mean"]),
            dynamics_std=np.array(data_dict["dynamics_std"]),
            horizon_len=data_dict["horizon_len"],
        )

        obj.n = data_dict["n"]
        obj.traj_len = data_dict["traj_len"]
        obj.states = [np.array(s) for s in data_dict["states"]]
        obj.dynamics = [np.array(d) for d in data_dict["dynamics"]]
        obj.key = jax.numpy.array(data_dict["key"])

        return obj
