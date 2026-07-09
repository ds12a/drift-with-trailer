import numpy as np
import jax

class Data:

    def __init__(self, batch_size, state_mean, state_std, dynamics_mean, dynamics_std, horizon_len=4):
        
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

        state, dynamics = self._normalize(state, dynamics)

        self.states.append(state)
        self.dynamics.append(dynamics)
        self.traj_len.append(len(state))
        self.n += len(state)

    def get_data(self):
        """
        Must generate horizion accumulation before returning
        """

        self.key, subkey = jax.random.split(self.key)

        s = len(self.states[0][0])
        d = len(self.dynamics[0][0])
        h = self.horizon_len

        n = len(self)

        leftover = n % self.batch_size
        perm = np.array(jax.random.permutation(subkey, n)[leftover:])

        B = len(perm) // self.batch_size
        batch_perm = perm.reshape((B, self.batch_size))

        traj_len_prefix = np.cumsum(np.insert(self.true_traj_len_buffer, 0, 0))
        traj_indices = np.searchsorted(traj_len_prefix[1:], batch_perm, side='right')
        offset = batch_perm - traj_len_prefix[traj_indices]

        batched_states = np.zeros((B, self.batch_size, h * s))
        batched_dynamics = np.zeros((B, self.batch_size, h * d))

        for b in range(B):
            for i in range(self.batch_size):
                t_idx = traj_indices[b, i]
                o_idx = offset[b, i]
                st_window = self.states[t_idx][o_idx : o_idx + h]
                dy_window = self.dynamics[t_idx][o_idx : o_idx + h]

                batched_states[b, i] = st_window.ravel()
                batched_dynamics[b, i] = dy_window.ravel()

        return batched_states, batched_dynamics

    def _normalize(self, states, dynamics):
        states = (states - self.state_mean) / self.state_std
        dynamics = (dynamics - self.dynamics_mean) / self.dynamics_std

        return states, dynamics
    
if __name__ == "__main__":
    data = Data(4, np.ones(4), np.ones(4), np.ones(5), np.ones(5), 4)

    N = [10, 50, 1, 3, 5]
    for n in N:
        data.add(np.zeros((n, 4)), np.zeros((n, 5)))

    print(data.get_data()[0].shape)