from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple

Array = jax.Array


class JaxTrackProjection(NamedTuple):
    progress: Array
    arc_length: Array
    x: Array
    y: Array
    heading: Array
    lateral_error: Array
    curvature: Array


class JaxTrackData(NamedTuple):
    centerline: Array
    segments: Array
    segment_lengths: Array
    cumulative: Array
    arc_samples: Array
    curvature_samples: Array
    curvature_interp_arc: Array
    curvature_interp_values: Array
    width: Array
    road_half_width: Array
    length: Array


class JaxVehicleParams(NamedTuple):
    wheelbase: Array
    lf: Array
    lr: Array
    mass: Array
    inertia_z: Array
    cornering_stiffness_front: Array
    cornering_stiffness_rear: Array
    max_steer_rad: Array
    max_accel: Array
    max_brake: Array
    drag_coefficient: Array
    wheel_radius: Array


class JaxSimulationParams(NamedTuple):
    dt: Array
    max_steps: Array
    lookahead_offsets: Array
    history_template: Array


class JaxRewardParams(NamedTuple):
    progress_coef: Array
    speed_coef: Array
    lateral_error_coef: Array
    heading_error_coef: Array


class NominalJaxEnvParams(NamedTuple):
    track: JaxTrackData
    vehicle: JaxVehicleParams
    simulation: JaxSimulationParams
    reward: JaxRewardParams


class JaxRacecarState(NamedTuple):
    x: Array
    y: Array
    yaw: Array
    progress: Array
    lateral_error: Array
    heading_error: Array
    vx: Array
    vy: Array
    yaw_rate: Array
    steer: Array
    throttle: Array
    brake: Array
    wheel_rotation: Array
    lap_count: Array
    step_count: Array
    action_history: Array


class JaxResetOutput(NamedTuple):
    state: JaxRacecarState
    observation: Array


class JaxStepOutput(NamedTuple):
    state: JaxRacecarState
    observation: Array
    reward: Array
    terminated: Array
    truncated: Array


def gen_util_funs(params: NominalJaxEnvParams, reverse=False, v_target=None):
    step = params.simulation.dt
    reverse = 1 if reverse else -1

    def compute_fy(alpha, cc, fz, fx, mu):
        gamma = 0.7  # no idea if good, put in vehicle params later

        fy_max = jnp.sqrt((mu * fz) ** 2 + gamma * fx**2)

        alpha_sl = jnp.arctan2(3 * fy_max, cc)

        return jnp.where(
            jnp.abs(alpha) < alpha_sl,
            (
                -cc * jnp.tan(alpha)
                + (cc**2 / (3 * fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
                - (cc**fy_max / (27 * fy_max**2)) * jnp.tan(alpha) ** 3
            ),
            -fy_max * jnp.sign(alpha),
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
        dt = params.simulation.dt
        vehicle = params.vehicle

        # steer = state.steer + (steer_cmd - state.steer) * jnp.minimum(1.0, dt * 8.0)
        steer = steer_cmd
        throttle = throttle_cmd
        brake = brake_cmd

        vx_safe = jnp.maximum(jnp.abs(state_xdot), 0.5)
        steer_angle = steer * vehicle.max_steer_rad
        alpha_f = steer_angle - jnp.arctan2(state_ydot + vehicle.lf * state_yaw_dot, vx_safe)
        alpha_r = -jnp.arctan2(state_ydot - vehicle.lr * state_yaw_dot, vx_safe)

        # somehow get mu from track
        mu = 1

        fyf = compute_fy(
            alpha_f,
            vehicle.cornering_stiffness_front,
            vehicle.mass * 9.8 * vehicle.lf / (vehicle.lf + vehicle.rf),
            0,
            mu,
        )

        fzr = vehicle.mass * 9.8 * vehicle.rf / (vehicle.lf + vehicle.rf)
        fyr = compute_fy(
            alpha_r,
            vehicle.cornering_stiffness_rear,
            fzr,
            mu * fzr * jnp.tanh(action[1] / (fzr * mu)),
            mu,
        )

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

    # @jax.jit
    # def cost(x, u, t):
    #     p_weight = 1e2

    #     yaw = x[2]
    #     gvx = x[3] * jnp.cos(yaw) - x[4] * jnp.sin(yaw)
    #     gvy = x[3] * jnp.sin(yaw) + x[4] * jnp.cos(yaw)

    #     projection_curr = _project_to_track(params.track, x[0], x[1])
    #     projection_next = _project_to_track(params.track, x[0] + step * gvx, x[1] + step * gvy)
    #     # track_vel = (projection_next.arc_length - projection_curr.arc_length) / step

    #     # progress_gain = projection_next.progress - projection_curr.progress

    #     # crossed = progress_gain < -0.5
    #     raw_diff = projection_next.arc_length - projection_curr.arc_length
    #     track_vel = (
    #         raw_diff - params.track.length * jnp.round(raw_diff / params.track.length)
    #     ) / step
    #     # track_vel = jnp.where(crossed, track_vel + params.track.length / step, track_vel)
    #     # progress_gain = jnp.where(crossed, progress_gain + 1, progress_gain)

    #     violation = jnp.maximum(
    #         0, jnp.abs(projection_curr.lateral_error) - params.track.road_half_width + 0.1
    #     )

    #     if v_target is None:
    #         v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[3])
    #     else:
    #         v_term = p_weight * jnp.abs(track_vel - v_target)

    #     return 0.9**t * (1e9 * violation) + v_term

    @jax.jit
    def bound(u):
        return jnp.clip(
            u,
            jnp.array([-1, -1]),
            jnp.array([1, 1]),
        )

    # @jax.jit
    # def bound_der(u):
    #     return jnp.clip(
    #         u,
    #         jnp.array([-1.5, -1]),
    #         jnp.array([1.5, 1]),
    #     )

    return dynamics, bound
