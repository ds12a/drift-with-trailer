from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple

from uncertain_racecar_gym.jax_env import NominalJaxEnvParams, JaxTrackProjection
from uncertain_racecar_gym.track import TrackProjection, TrackModel
from uncertain_racecar_gym.scenario import Scenario
Array = jax.Array



def gen_util_funs(scenario: Scenario, reverse=False, v_target=None):
    step = scenario.simulation.dt
    dt = scenario.simulation.dt # redundant, but fine
    vehicle = scenario.vehicle
    track = TrackModel.from_config(scenario.track)
    
    reverse = 1 if reverse else -1


    def project(x, y):
        point = jnp.asarray([x, y], dtype=float)

        delta_from_start = point - track.centerline
        
        
        safe_denominator = jnp.where(track._segment_valid, track._segment_length_sq, 1.0)
        t = jnp.where(
            track._segment_valid,
            jnp.divide(jnp.einsum("ij,ij->i", delta_from_start, track._segments), safe_denominator),
            jnp.zeros_like(track._segment_lengths)
        )
        t = jnp.clip(t, 0.0, 1.0)
        projected = jnp.asarray(track.centerline + track._segments * t[:, None])
        delta = point - projected
        distance_sq = jnp.einsum("ij,ij->i", delta, delta)
        distance_sq = jnp.where(track._segment_valid, distance_sq, jnp.inf)
        index = jnp.argmin(distance_sq)

        signed_offset = jnp.dot(point - projected[index], jnp.asarray(track._segment_normals)[index])
        arc = jnp.asarray(track._cumulative)[index] + jnp.asarray(t)[index] * jnp.asarray(track._segment_lengths)[index]
        
        return JaxTrackProjection(
            progress=(arc % track.length) / track.length,
            arc_length=arc,
            x=projected[index, 0],
            y=projected[index, 1],
            heading=jnp.asarray(track._segment_headings)[index],
            lateral_error=signed_offset,
            curvature=jnp.interp(arc, track._arc_samples, track._curvature_samples, period=track.length),
        )

    @jax.jit
    def dynamics(
        state: Array,
        action: Array,
    ) -> Array:
        state_x, state_y, state_yaw, state_xdot, state_ydot, state_yaw_dot = jnp.unstack(state)

        action = jnp.asarray(action, dtype=jnp.float32)
        steer_cmd = jnp.clip(action[0], -1.0, 1.0)

        throttle_cmd = jnp.maximum(action[1], 0.0)
        brake_cmd = -jnp.minimum(action[1], 0.0)

        # steer = state.steer + (steer_cmd - state.steer) * jnp.minimum(1.0, dt * 8.0)
        steer = steer_cmd
        throttle = throttle_cmd
        brake = brake_cmd

        vx_safe = jnp.maximum(jnp.abs(state_xdot), 0.5)
        steer_angle = steer * vehicle.max_steer_rad
        alpha_f = steer_angle - jnp.arctan2(state_ydot + vehicle.lf * state_yaw_dot, vx_safe)
        alpha_r = -jnp.arctan2(state_ydot - vehicle.lr * state_yaw_dot, vx_safe)

        fyf = vehicle.cornering_stiffness_front * alpha_f
        fyr = vehicle.cornering_stiffness_rear * alpha_r
        longitudinal_acc = (
            throttle * vehicle.max_accel
            - brake * vehicle.max_brake
            - vehicle.drag_coefficient
            * state_xdot
            * jnp.abs(state_xdot)
            / jnp.maximum(vehicle.mass, 1.0)
        )

        vx_dot = longitudinal_acc + state_ydot * state_yaw_dot
        vy_dot = (fyf * jnp.cos(steer_angle) + fyr) / vehicle.mass - state_xdot * state_yaw_dot
        yaw_rate_dot = (
            vehicle.lf * fyf * jnp.cos(steer_angle) - vehicle.lr * fyr
        ) / vehicle.inertia_z

        next_vx = state_xdot + vx_dot * dt
        next_vy = state_ydot + vy_dot * dt
        next_yaw_rate = state_yaw_dot + yaw_rate_dot * dt

        # Trapezoidal (avg) approximations
        avg_vx = 0.5 * (state_xdot + next_vx)
        avg_vy = 0.5 * (state_ydot + next_vy)
        avg_yaw_rate = 0.5 * (state_yaw_dot + next_yaw_rate)

        # Change of frame
        xdot = avg_vx * jnp.cos(state_yaw) - avg_vy * jnp.sin(state_yaw)
        ydot = avg_vx * jnp.sin(state_yaw) + avg_vy * jnp.cos(state_yaw)

        return jnp.asarray([xdot, ydot, avg_yaw_rate, vx_dot, vy_dot, yaw_rate_dot])

    @jax.jit
    def cost(x, u, t):
        p_weight = 1e2

        yaw = x[2]
        gvx = x[3] * jnp.cos(yaw) - x[4] * jnp.sin(yaw)
        gvy = x[3] * jnp.sin(yaw) + x[4] * jnp.cos(yaw)

        projection_curr = project(x[0], x[1])
        projection_next = project(x[0] + step * gvx, x[1] + step * gvy)
        # track_vel = (projection_next.arc_length - projection_curr.arc_length) / step

        # progress_gain = projection_next.progress - projection_curr.progress

        # crossed = progress_gain < -0.5
        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (
            raw_diff - track.length * jnp.round(raw_diff / track.length)
        ) / step
        # track_vel = jnp.where(crossed, track_vel + params.track.length / step, track_vel)
        # progress_gain = jnp.where(crossed, progress_gain + 1, progress_gain)

        violation = jnp.maximum(
            0, jnp.abs(projection_curr.lateral_error) - track.width / 2 + 0.1
        )

        if v_target is None:
            v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[3])
        else:
            v_term = p_weight * jnp.abs(track_vel - v_target)

        return 0.9**t * (10_000_000 * violation) + v_term

    @jax.jit
    def bound(u):
        return jnp.clip(
            u,
            jnp.array([-1, -1]),
            jnp.array([1, 1]),
        )

    return dynamics, cost, bound
