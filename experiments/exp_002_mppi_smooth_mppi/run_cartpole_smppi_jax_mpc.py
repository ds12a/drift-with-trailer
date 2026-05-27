from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.smppi_jax import SMPPI_Jax

import numpy as np
import time

import jax.numpy as jnp
import jax

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN

@jax.jit
def gen_dynamics(x, u):
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
def constraints(x, u):
    x_threshold = 2.4

    return ~((-x_threshold <= x[0]) & (x[0] <= x_threshold))
    

@jax.jit
def cost(x, u, t):
    return x[2] ** 2  + x[0] ** 2  + 1**t * 10_000 * constraints(x, u).astype(jnp.int64) #+ x[:, 3]**2 /10

@jax.jit
def term_cost(x, u):
    return x[3]**2 + x[1]**2 # torch.sum(x**2, axis=1)

@jax.jit
def bound_control(u):
    return jnp.clip(u, -FORCE, FORCE)

# Placeholder for cartpole
def der_bound_control(u):
    return u



def run_mpc():
    env = CartPoleEnv(render_mode="human")

    env.reset()


    mpc = SMPPI_Jax(4, 1, gen_dynamics, term_cost, cost, bound_control, der_bound_control, jnp.eye(1) * 0.7)
    
    observation, reward, terminated, truncated, info = env.step(0)
    
    i = 0

    while True:
        start = time.perf_counter()
        u = mpc.run_mpc(observation)
        action = np.clip(u[0], -FORCE, FORCE)
        print(i, time.perf_counter() - start, action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated: 
            break

        i += 1


    env.close()

if __name__ == "__main__":
    run_mpc()