"""
Designed as a drop-in replacement for MPPI_Jax but with extra visualization tools
for debugging purposes
"""

from scipy.signal import savgol_filter
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
import numpy as np

# import torch
import functools


# @jax.jit
@functools.partial(
    jax.jit, static_argnames=["cost", "term_cost", "bound_control", "dynamics", "history"]
)
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None))
def rollout(
    x: ArrayLike,
    u: ArrayLike,
    noise: ArrayLike,
    gamma,
    inv_cv,
    cost,
    term_cost,
    bound_control,
    dynamics,
    step,
    history=None,
) -> float:
    """
    Uses Euler's method to integrate the dynamics

    Args:
        x (ArrayLike): State (x_d)
        u (ArrayLike): Control (T, u_d)
        noise (ArrayLike): Noise (T, u_d)

    Returns:
        float: Cost of rollout
    """

    v = u + noise
    v = bound_control(v)
    bounded_noise = v - u

    def step_dynamics(carry, control):
        x, S, i = carry

        if history is not None:
            x_dim = (x.shape[0] + u.shape[0]) / history
            historical_x = x[:-x.shape[0]]
            x = x[-x.shape[0]:]

        u, v, bounded_noise = control

        new_x = x + dynamics(x, v) * step
        new_S = S + cost(new_x, v, i) + gamma * jnp.einsum("n,nm,m->", u, inv_cv, bounded_noise)
        new_i = i + 1

        new_carry = (
            jnp.concatenate([historical_x[:x_dim], new_x]) if history is not None else new_x,
            new_S,
            new_i,
        )

        return new_carry, (new_x, new_S, v)

    (x, S, _), (xhist, _, vhist) = jax.lax.scan(step_dynamics, (x, 0, 0), (u, v, bounded_noise))

    if term_cost:
        S += term_cost(x, u[-1])
    return S, xhist, vhist


@functools.partial(
    jax.jit,
    static_argnames=["K", "T", "u_d", "forward_sim"],
)
def mpc_step(x, last_trajectory, u_d, key, K, T, cv, inverse_temp, forward_sim):
    if last_trajectory is None:
        u = jnp.zeros((T, u_d))
    else:
        u = jnp.roll(last_trajectory, -1, axis=0)
        u = u.at[-1].set(0)

    x = jnp.asarray(x)

    x_batch = jnp.repeat(jnp.expand_dims(x, 0), K, axis=0)
    u_batch = jnp.repeat(jnp.expand_dims(u, 0), K, axis=0)

    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, u_batch.shape) * jnp.sqrt(jnp.diag(cv))

    costs, bounded_noise, xhist, uhist = forward_sim(x_batch, u_batch, noise)

    weights = jnp.exp(-(costs - costs.min()) / inverse_temp)
    weights = weights / weights.sum()

    weighted_noise = jnp.sum(weights.reshape(-1, 1, 1) * bounded_noise, axis=0)
    u = u + weighted_noise

    return u, key, xhist, uhist


class MPPI_Jax_Debug:
    """
    JAX MPPI
    """

    def __init__(
        self,
        x_d: int,
        u_d: int,
        dynamics_func,
        term_cost_func,
        cost_func,
        bound_control_func,
        cv,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.01,  # TODO purge the repo of this term, it is calculated
        K=20000,
        step=0.02,
        T=70,
        history=None,
        device="mps",
    ):
        """
        Args:
            x_d (int): State dimension
            u_d (int): Control dimension
            dynamics_func (Callable): Dynamics function
            term_cost_func (Callable): Terminal cost function
            cost_func (Callable): Cost function
            bound_control_func (Callable): Function that bounds controls
            inverse_temp (int, optional): Actually the temperature. Defaults to 1.
            alpha (float, optional): Proportion of samples set to just noise. Defaults to 0.01.
            gamma (float, optional): Cost weight for pertubations. Defaults to 0.1.
            K (int, optional): Samples. Defaults to 5000.
            step (float, optional): Time step. Defaults to 0.02.
            T (int, optional): Time horizon in steps. Defaults to 50.
        """
        self.last_trajectory = None
        self.u_history = jnp.zeros((T, u_d))
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
        self.alpha = alpha
        self.inverse_temp = inverse_temp
        self.gamma = (1 - alpha) * inverse_temp
        self.K = K
        self.device = device

        self.x_d = x_d
        self.u_d = u_d
        self.T = T
        self.history = history

        self.step = step
        self.cv = cv

        self.inv_cv = jnp.linalg.inv(self.cv)

        self.key = jax.random.key(0)

    def _forward_sim(self, x: ArrayLike, u: ArrayLike, noise: ArrayLike) -> jax.Array:
        """
        Uses Euler's method to integrate the dynamics

        Args:
            x (ArrayLike): State (K, x_d)
            u (ArrayLike): Control (K, T, u_d)
            noise (ArrayLike): Noise (K, T, u_d)

        Returns:
            jax.Array: Cost per sample (K)
        """

        v = u + noise
        prev = round(self.K * (1 - self.alpha))

        # v[:, prev:] = noise[:, prev:]
        v = v.at[prev:].set(noise[prev:])

        v = self.bound_control(v)
        noise = v - u

        S, xhist, vhist = rollout(
            x,
            u,
            noise,
            self.gamma,
            self.inv_cv,
            self.cost,
            self.term_cost,
            self.bound_control,
            self.dynamics,
            self.step,
            self.history
        )

        return S, noise, xhist, vhist

    def run_mpc(self, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Runs a single MPC solve.

        Args:
            x (torch.Tensor): State (x_d)

        Returns:
            torch.Tensor: Control output
        """

        u, self.key, xhist, vhist = mpc_step(
            x,
            self.last_trajectory,
            self.u_d,
            self.key,
            self.K,
            self.T,
            self.cv,
            self.inverse_temp,
            self._forward_sim,
        )

        # u_padded = jnp.concatenate([self.u_history, u])
        # u_smoothed = jnp.array(savgol_filter(np.array(u_padded), 5, 3, axis=0)[-self.T :])

        # self.u_history = jnp.roll(self.u_history, -1, axis=0)
        # self.u_history = self.u_history.at[-1].set(u[0])
        self.last_trajectory = u

        return u[0], xhist, vhist
