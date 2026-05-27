from src.simulation.cartpole_env import CartPoleEnv
from src.controllers.mpc.mppi_torch import MPPI_Torch

import numpy as np
import torch
import time

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN

def gen_dynamics(x, u):
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

def constraints(x, u):
    x_threshold = 2.4

    return torch.logical_not(
        # torch.logical_and(
        torch.logical_and(-x_threshold <= x[:, 0].unsqueeze(1), x[:, 0].unsqueeze(1) <= x_threshold),
        # torch.logical_and(-FORCE <= u, u <= FORCE))
    
)

def cost(x, u, t):
    return x[:, 2] ** 2  + x[:, 0] ** 2  + 1**t * 10_00 * constraints(x, u).squeeze() #+ x[:, 3]**2 /10

def term_cost(x, u):
    return x[:, 3]**2 + x[:, 1]**2 # torch.sum(x**2, axis=1)

def bound_control(u):
    return torch.clamp(u, -FORCE, FORCE)



def run_mpc():
    env = CartPoleEnv(render_mode="human")

    env.reset()

    device = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    mpc = MPPI_Torch(4, 1, gen_dynamics, term_cost, cost, bound_control, torch.eye(1) * 3, device=device)
    
    observation, reward, terminated, truncated, info = env.step(0)

    i = 0

    while True:
        start = time.perf_counter()
        u = mpc.run_mpc(observation)
        action = np.clip(float(u[0]), -FORCE, FORCE)
        print(i, time.perf_counter() - start, action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated: 
            break
        i += 1


    env.close()

if __name__ == "__main__":
    with torch.no_grad():
        run_mpc()