from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
from typing import NamedTuple
from src.simulation.config.trailer_bicycle_config import TrailerBicycleEnvConfig
from src.utils.track import TrackModel


Array = jax.Array


class TrackProjection(NamedTuple):
    progress: Array
    arc_length: Array
    x: Array
    y: Array
    heading: Array
    lateral_error: Array
    curvature: Array


def gen_util_funs(params: TrailerBicycleEnvConfig, reverse=False, v_target=None):
    reverse = 1 if reverse else -1
    step = params.simulation.dt

    track = TrackModel.from_config(params.track)

    def compute_fy(alpha, cc, fz, fx, mu, gamma):
        fy_max = jnp.sqrt(jnp.maximum((mu * fz) ** 2 - gamma * fx**2, 0))

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
    def dynamics(state, u):
        x, y, phi_1, phi_2, v_1x, v_1y, phi_1_dot, phi_2_dot = state

        vehicle = params.vehicle
        steer_cmd = jnp.clip(u[0], -1.0, 1.0)
        accel_cmd = jnp.clip(u[1], -1.0, 1.0)
        dt = params.simulation.dt

        throttle = jnp.maximum(accel_cmd, 0.0)
        brake = -jnp.minimum(accel_cmd, 0.0)

        vx_safe = jnp.maximum(jnp.abs(v_1x), 0.5)
        delta = steer_cmd * vehicle.max_steer_rad
        alpha_f = delta - jnp.arctan2(v_1y + vehicle.lf * phi_1_dot, vx_safe)
        alpha_r = -jnp.arctan2(v_1y - vehicle.lr * phi_1_dot, vx_safe)

        mu = track.find_mu(x, y)

        fzf = vehicle.mass * 9.8 * vehicle.lr / (
            vehicle.lf + vehicle.lr
        ) + vehicle.trailer_mass * 9.8 * vehicle.l2r * (vehicle.lr - vehicle.hitch_offset) / (
            (vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r)
        )

        F_1yf = -compute_fy(
            alpha_f,
            vehicle.cornering_stiffness_front,
            fzf,
            0,
            mu,
            vehicle.gamma,
        )

        fzr = vehicle.mass * 9.8 * vehicle.lf / (
            vehicle.lf + vehicle.lr
        ) + vehicle.trailer_mass * 9.8 * vehicle.l2r * (vehicle.lf + vehicle.hitch_offset) / (
            (vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r)
        )
        commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake
        fxr = mu * fzr * jnp.tanh(vehicle.mass * commanded / (fzr * mu))

        F_1yr = -compute_fy(alpha_r, vehicle.cornering_stiffness_rear, fzr, fxr, mu, vehicle.gamma)

        alpha = phi_1 - phi_2
        sa = jnp.sin(alpha)
        ca = jnp.cos(alpha)

        v_2x = v_1x * ca - (v_1y - phi_1_dot * vehicle.hitch_offset) * sa
        v_2y = v_1x * sa + (v_1y - phi_1_dot * vehicle.hitch_offset) * ca - vehicle.l2f * phi_2_dot

        v2x_safe = jnp.maximum(jnp.abs(v_2x), 0.5)
        alpha_t = -jnp.arctan2(v_2y - vehicle.l2r * phi_2_dot, v2x_safe)

        fzr_trailer = vehicle.trailer_mass * 9.8 * vehicle.l2f / (vehicle.l2f + vehicle.l2r)
        F_2yr = -compute_fy(
            alpha_t, vehicle.cornering_stiffness_trailer, fzr_trailer, 0, mu, vehicle.gamma
        )

        total_mass = vehicle.mass + vehicle.trailer_mass
        cd = jnp.cos(delta)
        sd = jnp.sin(delta)
        alpha_dot = phi_1_dot - phi_2_dot

        A = jnp.array(
            [
                [total_mass, 0, 0, -vehicle.trailer_mass * vehicle.l2f * sa],
                [
                    0,
                    total_mass,
                    -vehicle.trailer_mass * vehicle.hitch_offset,
                    -vehicle.trailer_mass * vehicle.l2f * ca,
                ],
                [
                    0,
                    -vehicle.trailer_mass * vehicle.hitch_offset,
                    vehicle.inertia_z + vehicle.trailer_mass * vehicle.hitch_offset**2,
                    vehicle.trailer_mass * vehicle.l2f * vehicle.hitch_offset * ca,
                ],
                [
                    -vehicle.trailer_mass * vehicle.l2f * sa,
                    -vehicle.trailer_mass * vehicle.l2f * ca,
                    vehicle.trailer_mass * vehicle.l2f * vehicle.hitch_offset * ca,
                    vehicle.trailer_inertia_z + vehicle.trailer_mass * vehicle.l2f**2,
                ],
            ]
        )

        b = jnp.array(
            [
                fxr
                - F_1yf * sd
                + F_2yr * sa
                + vehicle.mass * v_1y * phi_1_dot
                + vehicle.trailer_mass * phi_1_dot * (v_2y * ca - v_2x * sa)
                + vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * ca,
                F_1yr
                + F_1yf * cd
                + F_2yr * ca
                - vehicle.mass * v_1x * phi_1_dot
                - vehicle.trailer_mass * phi_1_dot * (v_2x * ca + v_2y * sa)
                - vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * sa,
                -F_1yr * vehicle.lr
                + F_1yf * cd * vehicle.lf
                - vehicle.hitch_offset * F_2yr * ca
                + vehicle.trailer_mass * vehicle.hitch_offset * phi_1_dot * (v_2x * ca + v_2y * sa)
                + vehicle.trailer_mass
                * vehicle.hitch_offset
                * vehicle.l2f
                * alpha_dot
                * phi_2_dot
                * sa,
                -(vehicle.l2f + vehicle.l2r) * F_2yr
                + vehicle.trailer_mass * vehicle.l2f * v_2x * phi_1_dot,
            ]
        )

        v_1x_dot, v_1y_dot, phi_1_ddot, phi_2_ddot = jnp.linalg.solve(A, b)

        next_vx = v_1x + v_1x_dot * dt
        next_vy = v_1y + v_1y_dot * dt
        next_phi_1_dot = phi_1_dot + phi_1_ddot * dt
        next_phi_2_dot = phi_2_dot + phi_2_ddot * dt

        # Trapezoidal (avg) approximations
        avg_vx = 0.5 * (v_1x + next_vx)
        avg_vy = 0.5 * (v_1y + next_vy)
        avg_phi_1_dot = 0.5 * (phi_1_dot + next_phi_1_dot)
        avg_phi_2_dot = 0.5 * (phi_2_dot + next_phi_2_dot)

        # Change of frame
        xdot = avg_vx * jnp.cos(phi_1) - avg_vy * jnp.sin(phi_1)
        ydot = avg_vx * jnp.sin(phi_1) + avg_vy * jnp.cos(phi_1)

        return jnp.array(
            [xdot, ydot, phi_1_dot, phi_2_dot, v_1x_dot, v_1y_dot, phi_1_ddot, phi_2_ddot]
        )

    @jax.jit
    def cost(x, u, t):

        # Tunable values

        p_weight = 1e2
        p_slow_weight = 1e0
        s_weight = 1e4
        c_weight = 1e-2

        ####### Helpers #######

        def tire_traction_penalty(alpha, cc, fz, fx, mu):
            gamma = params.vehicle.gamma
            fy_max = jnp.sqrt(jnp.maximum((mu * fz) ** 2 - gamma * fx**2, 1e-9))
            alpha_sl = jnp.arctan2(3 * fy_max, cc)

            return jnp.maximum(0, jnp.abs(alpha) - alpha_sl) / jnp.maximum(alpha_sl, 1e-9)

        def combined_traction_penalty(state, action):
            (
                state_x,
                state_y,
                state_yaw,
                state_yaw_trailer,
                state_xdot,
                state_ydot,
                state_yaw_dot,
                state_yaw_trailer_dot,
            ) = jnp.unstack(state)

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

            mu = track.find_mu(state_x, state_y)

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
                mu
                * fzr
                * jnp.tanh(
                    vehicle.mass
                    * (throttle * vehicle.max_accel - brake * vehicle.max_brake)
                    / (fzr * mu)
                ),
                mu,
            )

            return (pen_f + 0) ** 2  # if we include pen_r how will we drift??

        def _project_to_track(x, y) -> TrackProjection:
            """
            From Uncertain Racecar Gym, adapted
            """
            lookahead_thresh = 50  # should be meters, TODO tune
            lookahead_num = 5

            point = jnp.stack([x, y])
            p0 = jnp.array(track.centerline)
            segments = jnp.array(track._segments)
            seg_len = jnp.array(track._segment_lengths)

            denom = jnp.maximum(seg_len * seg_len, 1e-9)
            t = jnp.clip(jnp.sum((point - p0) * segments, axis=1) / denom, 0.0, 1.0)
            projected = p0 + segments * t[:, None]
            delta = point[None, :] - projected
            distance_sq = jnp.sum(delta * delta, axis=1)
            valid_distance_sq = jnp.where(seg_len > 1e-9, distance_sq, jnp.inf)
            index = jnp.argmin(valid_distance_sq)

            chosen_segment = segments[index]
            chosen_length = seg_len[index]
            tangent = chosen_segment / jnp.maximum(chosen_length, 1e-9)
            normal = jnp.stack([-tangent[1], tangent[0]])
            chosen_projected = projected[index]
            signed_offset = jnp.sum((point - chosen_projected) * normal)

            arc = jnp.array(track._cumulative)[index] + t[index] * chosen_length

            lookahead_pts = jnp.linspace(arc, arc + lookahead_thresh, lookahead_num)

            # curvature = jnp.interp(arc, track._arc_samples, track._curvature_samples)
            curvature = jnp.mean(
                jnp.abs(jnp.interp(lookahead_pts, track._arc_samples, track._curvature_samples))
            )  # TODO split off curvature for efficiency?

            heading = jnp.arctan2(chosen_segment[1], chosen_segment[0])
            return TrackProjection(
                progress=jnp.mod(arc / track.length, 1.0),
                arc_length=arc,
                x=chosen_projected[0],
                y=chosen_projected[1],
                heading=heading,
                lateral_error=signed_offset,
                curvature=curvature,
            )

        ####### End Helpers #######

        yaw = x[2]
        gvx = x[4] * jnp.cos(yaw) - x[5] * jnp.sin(yaw)
        gvy = x[4] * jnp.sin(yaw) + x[5] * jnp.cos(yaw)

        nominal_v = jnp.sqrt(x[4] ** 2 + x[5] ** 2)


        projection_curr = _project_to_track(x[0], x[1])
        projection_next = _project_to_track(x[0] + step * gvx, x[1] + step * gvy)

        raw_diff = projection_next.arc_length - projection_curr.arc_length
        track_vel = (raw_diff - track.length * jnp.round(raw_diff / track.length)) / step

        violation = jnp.maximum(
            0, jnp.abs(projection_curr.lateral_error) - (params.track.width * 0.5) * 0.9 + 0.1
        )

        max_safe_v = (
            jnp.sqrt(1.0 * 1.5 * 9.8 / (projection_next.curvature + 1e-5)) + 1e7
        )  # currently disabled

        if v_target is None:
            v_term = reverse * p_weight * jnp.abs(track_vel) * jnp.sign(x[4])

        else:
            v_baseline = jnp.minimum(max_safe_v, v_target)
            # If v is above threshold use actual car velocity instead of track velocity to stop cheating
            v_car = jnp.where(nominal_v > max_safe_v, nominal_v, track_vel)

            v_term = p_weight * jnp.maximum(
                0, v_car - v_baseline
            ) + p_weight * p_slow_weight * jnp.maximum(0, v_baseline - v_car)

        c = (
            0.99**t * (1e9 * violation)
            + v_term
            # + combined_traction_penalty(x, u) * s_weight
            # + projection_curr.lateral_error**2 * c_weight
        )

        # jax.debug.print("cost {c}", c=c)
        return c

    def bound(u):
        return jnp.clip(u, jnp.array([-1, -1]), jnp.array([1, 1]))

    def bound_der(u):
        return u

    return dynamics, cost, bound, bound_der
