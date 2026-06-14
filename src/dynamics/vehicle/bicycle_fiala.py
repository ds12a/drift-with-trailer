from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple
from src.simulation.config.bicycle_config import BicycleEnvConfig
from src.utils.track import TrackModel


Array = jax.Array

# from jax import config

# config.update("jax_debug_nans", True)


class TrackProjection(NamedTuple):
    progress: Array
    arc_length: Array
    x: Array
    y: Array
    heading: Array
    lateral_error: Array
    curvature: Array


def gen_util_funs(params: BicycleEnvConfig, reverse=False, v_target=None):
    reverse = 1 if reverse else -1
    step = params.simulation.dt

    track = TrackModel.from_config(params.track)

    def compute_fy(alpha, cc, fz, fx, mu):
        gamma = params.vehicle.gamma

        fy_max = jnp.sqrt(jnp.maximum((mu * fz) ** 2 - gamma * fx**2, 1e-9))


        alpha_sl = jnp.arctan2(3 * fy_max, cc)

        return jnp.where(
            jnp.abs(alpha) < alpha_sl,
            (
                -cc * jnp.tan(alpha)
                + (cc**2 / (3 * fy_max)) * jnp.abs(jnp.tan(alpha)) * jnp.tan(alpha)
                - (cc**3 / (27 * fy_max**2)) * jnp.tan(alpha) ** 3
            ),
            -fy_max * jnp.sign(alpha),
        )

    @jax.jit
    def dynamics(
        state: Array,
        action: Array,
    ) -> Array:
        state_x, state_y, state_yaw, state_xdot, state_ydot, state_yaw_dot, mu, index = jnp.unstack(state)
        # jax.debug.print("{mu}", mu=mu)

        action = jnp.asarray(action, dtype=jnp.float32)
        steer_cmd = jnp.clip(action[0], -1.0, 1.0)

        throttle_cmd = jnp.maximum(action[1], 0.0)
        brake_cmd = -jnp.minimum(action[1], 0.0)
        dt = params.simulation.dt
        vehicle = params.vehicle

        steer = steer_cmd
        throttle = throttle_cmd
        brake = brake_cmd

        vx_safe = jnp.maximum(jnp.abs(state_xdot), 0.5)
        steer_angle = steer * vehicle.max_steer_rad
        alpha_f = steer_angle - jnp.arctan2(state_ydot + vehicle.lf * state_yaw_dot, vx_safe)
        alpha_r = -jnp.arctan2(state_ydot - vehicle.lr * state_yaw_dot, vx_safe)

        fyf = -compute_fy(
            alpha_f,
            vehicle.cornering_stiffness_front,
            vehicle.mass * 9.8 * vehicle.lr / (vehicle.lf + vehicle.lr),
            0,
            mu,
        )

        fzr = vehicle.mass * 9.8 * vehicle.lf / (vehicle.lf + vehicle.lr)

        commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake
        fxr = mu * fzr * jnp.tanh(vehicle.mass * commanded / (fzr * mu))

        fyr = -compute_fy(
            alpha_r,
            vehicle.cornering_stiffness_rear,
            fzr,
            fxr,
            mu,
        )

        longitudinal_acc = (
            fxr - vehicle.drag_coefficient * state_xdot * jnp.abs(state_xdot)
        ) / jnp.maximum(vehicle.mass, 1.0)

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

        return jnp.asarray([xdot, ydot, avg_yaw_rate, vx_dot, vy_dot, yaw_rate_dot, 0, 0])

    @jax.jit
    def cost(x, u, t):

        # Tunable values

        p_weight = 1e3
        p_slow_weight = 1e0
        s_weight = 1e3
        c_weight = 1e-2

        ####### Helpers #######

        def tire_traction_penalty(alpha, cc, fz, fx, mu):
            gamma = params.vehicle.gamma
            fy_max = jnp.sqrt(jnp.maximum((mu * fz) ** 2 - gamma * fx**2, 1e-9))
            alpha_sl = jnp.arctan2(3 * fy_max, cc)

            return jnp.maximum(0, jnp.abs(alpha) - alpha_sl)

        def combined_traction_penalty(state, action):
            state_x, state_y, state_yaw, state_xdot, state_ydot, state_yaw_dot, mu, index = jnp.unstack(state)
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

            fzr = vehicle.mass * 9.8 * vehicle.lf / (vehicle.lf + vehicle.lr)

            commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake
            fxr = mu * fzr * jnp.tanh(vehicle.mass * commanded / (fzr * mu))

            pen_f = tire_traction_penalty(
                alpha_f,
                vehicle.cornering_stiffness_front,
                vehicle.mass * 9.8 * vehicle.lr / (vehicle.lf + vehicle.lr),
                0,
                mu,
            )
            pen_r = tire_traction_penalty(
                alpha_r,
                vehicle.cornering_stiffness_rear,
                fzr,
                fxr,
                mu,
            )

            return pen_f ** 2 + pen_r ** 2

        def _project_to_track(x, y, guess) -> tuple[TrackProjection, jax.Array]:
            """
            From Uncertain Racecar Gym, adapted
            """
            WINDOW = 10
            if guess is not None:
                window = (guess.astype(jnp.int32) - 5) + jnp.arange(WINDOW)
            else:
                window = jnp.arange(len(track.centerline))

            segments_window = jnp.take(track._segments, window, mode="wrap", axis=0)
            segments_len_window = jnp.take(track._segment_lengths, window, mode="wrap")
            segments_sq_window = jnp.take(track._segment_length_sq, window, mode="wrap")
            centerline_window = jnp.take(track.centerline, window, mode="wrap", axis=0)
            segments_normal_window = jnp.take(track._segment_normals, window, mode="wrap", axis=0)
            segments_heading_window = jnp.take(track._segment_headings, window, mode="wrap")
            valid_window = jnp.take(track._segment_valid, window, mode="wrap")
            cumulative_window = jnp.take(track._cumulative, window, mode="wrap")

            point = jnp.stack([x, y])
            delta_from_start = point - centerline_window

            denom = jnp.where(valid_window, segments_sq_window, 1.0)
            t = jnp.where(
                valid_window,
                jnp.einsum("ij,ij->i", delta_from_start, segments_window) / denom,
                0.0,
            )
            t = jnp.clip(t, 0.0, 1.0)
            projected = centerline_window + segments_window * t[:, None]
            delta = point - projected
            distance_sq = jnp.einsum("ij,ij->i", delta, delta)
            distance_sq = jnp.where(valid_window, distance_sq, jnp.inf)
            index = jnp.argmin(distance_sq)  # stays traced; dynamic indexing -> gather

            signed_offset = jnp.dot(point - projected[index], segments_normal_window[index])
            arc = cumulative_window[index] + t[index] * segments_len_window[index]
            return (
                TrackProjection(
                    progress=track.arc_to_progress(arc),
                    arc_length=arc,
                    x=projected[index, 0],
                    y=projected[index, 1],
                    heading=segments_heading_window[index],
                    lateral_error=signed_offset,
                    curvature=jnp.interp(
                        arc, track._arc_samples, track._curvature_samples, period=track.length
                    ),
                ),
                window[index],
            )   

        ####### End Helpers #######

        yaw = x[2]
        gvx = x[3] * jnp.cos(yaw) - x[4] * jnp.sin(yaw)
        gvy = x[3] * jnp.sin(yaw) + x[4] * jnp.cos(yaw)

        nominal_v = jnp.sqrt(x[3] ** 2 + x[4] ** 2)

        projection_curr, _ = _project_to_track(x[0], x[1], x[7])
        projection_next, _ = _project_to_track(x[0] + step * gvx, x[1] + step * gvy, x[7])

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / step

        violation = jnp.maximum(
            0, jnp.abs(projection_curr.lateral_error) - (params.track.width * 0.5) * 0.9 + 0.1
        )

        # safety_thresh = 1.0
        # max_safe_v = (
        #     jnp.sqrt(1.0 * safety_thresh * 9.8 / (projection_next.curvature + 1e-5))
        # )

        if v_target is None:
            v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[3])
        else:
            v_term = p_weight * jnp.abs(v_target + reverse * jnp.abs(track_vel) * jnp.sign(x[3]))

            # v_baseline = jnp.minimum(max_safe_v, v_target)
            # # If v is above threshold use actual car velocity instead of track velocity to stop cheating
            # v_car = jnp.where(nominal_v > max_safe_v, nominal_v, track_vel)

            # v_term = p_weight * jnp.maximum(
            #     0, v_car - v_baseline
            # ) + p_weight * p_slow_weight * jnp.maximum(0, v_baseline - v_car)

        return 1**t * (
            1e12 * violation
            + v_term
            + combined_traction_penalty(x, u) * s_weight
            + projection_curr.lateral_error**2 * c_weight
        )

    @jax.jit
    def bound(u):
        return jnp.clip(
            u,
            jnp.array([-1, -1]),
            jnp.array([1, 1]),
        )

    @jax.jit
    def bound_der(u):
        return u
        # return jnp.clip(
        #     u,
        #     jnp.array([-1.5, -1]),
        #     jnp.array([1.5, 1]),
        # )

    return dynamics, cost, bound, bound_der
