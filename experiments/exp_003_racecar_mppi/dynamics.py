from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple

from uncertain_racecar_gym.jax_env import (
    JaxTrackProjection,
    JaxTrackData,
    JaxVehicleParams,
    JaxSimulationParams,
    JaxRewardParams,
    NominalJaxEnvParams,
    JaxRacecarState,
    JaxResetOutput,
    JaxStepOutput,
    _wrap_angle,
    _project_to_track,
    _observation,
)

Array = jax.Array

params = NominalJaxEnvParams()

def step_nominal(
    state: JaxRacecarState,
    action: Array,
) -> JaxStepOutput:
    action = jnp.asarray(action, dtype=jnp.float32)
    steer_cmd = jnp.clip(action[0], -1.0, 1.0)
    throttle_cmd = jnp.clip(action[1], 0.0, 1.0)
    brake_cmd = jnp.clip(action[2], 0.0, 1.0)
    dt = params.simulation.dt
    vehicle = params.vehicle

    steer = state.steer + (steer_cmd - state.steer) * jnp.minimum(1.0, dt * 8.0)
    throttle = throttle_cmd
    brake = brake_cmd

    vx_safe = jnp.maximum(jnp.abs(state.vx), 0.5)
    steer_angle = steer * vehicle.max_steer_rad
    alpha_f = steer_angle - jnp.arctan2(state.vy + vehicle.lf * state.yaw_rate, vx_safe)
    alpha_r = -jnp.arctan2(state.vy - vehicle.lr * state.yaw_rate, vx_safe)

    fyf = vehicle.cornering_stiffness_front * alpha_f
    fyr = vehicle.cornering_stiffness_rear * alpha_r
    longitudinal_acc = (
        throttle * vehicle.max_accel
        - brake * vehicle.max_brake
        - vehicle.drag_coefficient
        * state.vx
        * jnp.abs(state.vx)
        / jnp.maximum(vehicle.mass, 1.0)
    )

    vx_dot = longitudinal_acc + state.vy * state.yaw_rate
    vy_dot = (
        fyf * jnp.cos(steer_angle) + fyr
    ) / vehicle.mass - state.vx * state.yaw_rate
    yaw_rate_dot = (
        vehicle.lf * fyf * jnp.cos(steer_angle) - vehicle.lr * fyr
    ) / vehicle.inertia_z

    next_vx = jnp.maximum(0.0, state.vx + vx_dot * dt)
    next_vy = state.vy + vy_dot * dt
    next_yaw_rate = state.yaw_rate + yaw_rate_dot * dt

    # Trapezoidal (avg) approximations
    avg_vx = 0.5 * (state.vx + next_vx)
    avg_vy = 0.5 * (state.vy + next_vy)
    avg_yaw_rate = 0.5 * (state.yaw_rate + next_yaw_rate)

    # Change of frame
    xdot = avg_vx * jnp.cos(state.yaw) - avg_vy * jnp.sin(state.yaw)
    ydot = avg_vx * jnp.sin(state.yaw) + avg_vy * jnp.cos(state.yaw)

    return jnp.asarray([xdot, ydot, avg_yaw_rate, vx_dot, vy_dot, yaw_rate_dot ])
    