from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.smppi_torch import SMPPI_Torch
from src.controllers.mpc.mppi_torch import MPPI_Torch
from src.controllers.mpc.smppi_jax import SMPPI_Jax
from src.controllers.mpc.mppi_jax import MPPI_Jax

import numpy as np
import torch
import time
import math
import jax
import jax.numpy as jnp

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN

# Torch methods

def gen_dynamics_torch(x, u):
    u = torch.clamp(u, -FORCE, FORCE)

    x_c = x[:, 0].unsqueeze(1)
    d_x_c = x[:, 1].unsqueeze(1)
    theta = x[:, 2].unsqueeze(1)
    theta_dot = x[:, 3].unsqueeze(1)

    # x_c, d_x_c, theta, theta_dot = torch.transpose(x, 0, 1)

    # x, x_dot, theta, theta_dot = self.state
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (
        u + POLE_LEN * (theta_dot * theta_dot) * sintheta
    ) / TOTAL_MASS

    thetaacc = (g * sintheta - costheta * temp) / (
        POLE_LEN
        * (4.0 / 3.0 - POLE_MASS * (costheta * costheta) / TOTAL_MASS)
    )

    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    dx = torch.hstack((d_x_c, xacc, theta_dot, thetaacc))

    return dx

def constraints_torch(x, u):
    x_threshold = 2.4
    return torch.logical_not(
        torch.logical_and(-x_threshold <= x[:, 0].unsqueeze(1), x[:, 0].unsqueeze(1) <= x_threshold),
    )

def cost_torch(x, u, t):
    return x[:, 2] ** 2  + x[:, 0] ** 2  + 1**t * 10_00 * constraints_torch(x, u).squeeze() #+ x[:, 3]**2 /10

def term_cost_torch(x, u):
    return x[:, 3]**2 + x[:, 1]**2

def bound_control_torch(u):
    return torch.clamp(u, -FORCE, FORCE)

@jax.jit
def gen_dynamics_jax(x, u):
    u = jnp.clip(u, -FORCE, FORCE)

    x_c = x[0, None]
    d_x_c = x[1, None]
    theta = x[2, None]
    theta_dot = x[3, None]

    # x_c, d_x_c, theta, theta_dot = torch.transpose(x, 0, 1)

    # x, x_dot, theta, theta_dot = self.state
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    temp = (
        u + POLE_LEN * (theta_dot * theta_dot) * sintheta
    ) / TOTAL_MASS

    thetaacc = (g * sintheta - costheta * temp) / (
        POLE_LEN
        * (4.0 / 3.0 - POLE_MASS * (costheta * costheta) / TOTAL_MASS)
    )

    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    dx = jnp.hstack((d_x_c, xacc, theta_dot, thetaacc))

    return dx


@jax.jit
def constraints_jax(x, u):
    x_threshold = 2.4

    return ~((-x_threshold <= x[0]) & (x[0] <= x_threshold))
    

@jax.jit
def cost_jax(x, u, t):
    return x[2] ** 2  + x[0] ** 2  + 1**t * 10_000 * constraints_jax(x, u).astype(jnp.int64) #+ x[:, 3]**2 /10

@jax.jit
def term_cost_jax(x, u):
    return x[3]**2 + x[1]**2 # torch.sum(x**2, axis=1)

@jax.jit
def bound_control_jax(u):
    return jnp.clip(u, -FORCE, FORCE)


def bench():
    env = CartPoleEnv(render_mode="human")

    env.reset()

    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    torch_mppi = MPPI_Torch(4, 1, gen_dynamics_torch, term_cost_torch, cost_torch, bound_control_torch, torch.eye(1) * 3, device=device)
    torch_smppi = SMPPI_Torch(4, 1, gen_dynamics_torch, term_cost_torch, cost_torch, bound_control_torch, torch.eye(1) * 0.7, torch.eye(1) * 2e-2, device=device)
    jax_mppi = MPPI_Jax(4, 1, gen_dynamics_jax, term_cost_jax, cost_jax, bound_control_jax, jnp.eye(1) * 3)
    jax_smppi = SMPPI_Jax(4, 1, gen_dynamics_jax, term_cost_jax, cost_jax, bound_control_jax, jnp.eye(1) * 0.7, jnp.identity(1) * 2e-2)

    ctls = [torch_mppi, torch_smppi, jax_mppi, jax_smppi]
    ctl_n = ["PyTorch MPPI", "PyTorch SMPPI", "Jax MPPI", "Jax SMPPI"]

    for i, ctl in enumerate(ctls):
        env.reset()
        observation, reward, terminated, truncated, info = env.step(0)

        t = np.zeros(500)
        x = np.zeros((500, 4))
        u = np.zeros(500)
        
        for j in range(500): # 10 sec

            start = time.perf_counter()
            action = np.clip(float(ctl.run_mpc(observation)[0]), -FORCE, FORCE)
            t[j] = time.perf_counter() - start
            x[j, :] = observation
            u[j] = action
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated: 
                break
        if terminated: 
            break
        
        print(ctl_n[i])
        print(f"Total t: {sum(t)}")
        print(f"Avg t: {np.average(t)}")
        print(f"Min t: {min(t)}")
        print(f"Max t: {max(t)}")
        print(f"Stdev t: {np.std(t)}")
        print(f"Max x: {np.max(np.abs(x[:, 0]))}")
        print(f"Avg |x|: {np.average(np.abs(x[:, 0]))}")
        print(f"Avg y: {np.average(np.cos(x[:, 2]))}")
        print(f"Max u: {np.max(np.abs(u))}")
        print(f"Avg |u|: {np.average(np.abs(u))}")
        print(f"Avg |du|: {np.average(np.abs(np.roll(u, 1)[:-1] - u[:-1]))}")

        # TODO if interested coudl save off t, u, x


if __name__ == "__main__":
    with torch.no_grad():
        bench()