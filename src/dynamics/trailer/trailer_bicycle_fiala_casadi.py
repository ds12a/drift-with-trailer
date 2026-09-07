"""
CasADi Fiala dynamics with spline track projection.
"""

import casadi as ca
import numpy as np

from src.utils.track import TrackModel


def gen_util_funs(
    params,
    reverse=False,
    v_target=None,
    p_weight=1e4,
    p_slow_weight=1e0,
    s_weight=1e4,
    c_weight=1e-2,
    a_weight=1e5,
    viol_weight=1e7,
):
    reverse = 1 if reverse else -1
    track = TrackModel.from_config(params.track)
    vehicle = params.vehicle
    step = params.simulation.dt

    # Extend the track across the lap boundary.
    arc = track._arc_samples
    arc_ext = np.concatenate([arc - track.length, arc, arc + track.length])
    x_ext = np.tile(track.centerline[:, 0], 3)
    y_ext = np.tile(track.centerline[:, 1], 3)
    track_x = ca.interpolant("fiala_track_x", "bspline", [arc_ext], x_ext)
    track_y = ca.interpolant("fiala_track_y", "bspline", [arc_ext], y_ext)

    arc_sym = ca.SX.sym("arc")
    x_ref = track_x(arc_sym)
    y_ref = track_y(arc_sym)
    dx_ds = ca.jacobian(x_ref, arc_sym)
    dy_ds = ca.jacobian(y_ref, arc_sym)
    tangent_norm = ca.sqrt(dx_ds**2 + dy_ds**2 + 1e-9)
    track_frame = ca.Function(
        "fiala_track_frame",
        [arc_sym],
        [x_ref, y_ref, dx_ds / tangent_norm, dy_ds / tangent_norm],
    )

    def slip_angle(v_lon, v_lat, eps=0.5):
        v_lon_abs = ca.sqrt(v_lon**2 + 1e-6)
        above_eps = v_lon_abs - eps
        v_lon_safe = eps + 0.5 * (above_eps + ca.sqrt(above_eps**2 + 1e-6))
        return -ca.atan2(v_lat, v_lon_safe)

    def compute_fy(alpha, cc, fz, fx, mu, gamma):
        fy_max = ca.sqrt(ca.fmax((mu * fz) ** 2 - gamma * fx**2, 1e-9))
        tan_alpha = ca.tan(alpha)
        alpha_sl = ca.atan2(3 * fy_max, cc)
        elastic = (
            -cc * tan_alpha
            + (cc**2 / (3 * fy_max)) * ca.fabs(tan_alpha) * tan_alpha
            - (cc**3 / (27 * fy_max**2)) * tan_alpha**3
        )
        return ca.if_else(
            ca.fabs(alpha) < alpha_sl,
            elastic,
            -fy_max * ca.sign(alpha),
        )

    def track_terms(state):
        x, y, phi_1, _, v_1x, v_1y, _, _, _, arc_len = ca.vertsplit(state)
        ref_x, ref_y, tx, ty = track_frame(arc_len)
        lateral_error = -ty * (x - ref_x) + tx * (y - ref_y)
        global_vx = v_1x * ca.cos(phi_1) - v_1y * ca.sin(phi_1)
        global_vy = v_1x * ca.sin(phi_1) + v_1y * ca.cos(phi_1)
        track_vel = tx * global_vx + ty * global_vy
        return lateral_error, track_vel

    def dynamics():
        state = ca.SX.sym("x", 10)
        u = ca.SX.sym("u", 2)
        x, y, phi_1, phi_2, v_1x, v_1y, phi_1_dot, phi_2_dot, mu, _ = ca.vertsplit(
            state
        )

        # Control bounds are imposed by Opti.
        steer_cmd = u[0]
        accel_cmd = u[1]

        delta = steer_cmd * vehicle.max_steer_rad
        cd = ca.cos(delta)
        sd = ca.sin(delta)

        alpha = phi_1 - phi_2
        sa = ca.sin(alpha)
        cos_alpha = ca.cos(alpha)

        v_2x = v_1x * cos_alpha - (v_1y - phi_1_dot * vehicle.hitch_offset) * sa
        v_2y = (
            v_1x * sa
            + (v_1y - phi_1_dot * vehicle.hitch_offset) * cos_alpha
            - vehicle.l2f * phi_2_dot
        )

        v_yf = v_1y + vehicle.lf * phi_1_dot
        v_yr = v_1y - vehicle.lr * phi_1_dot
        v_2y_wheel = v_2y - vehicle.l2r * phi_2_dot
        alpha_f = slip_angle(v_1x * cd + v_yf * sd, -v_1x * sd + v_yf * cd)
        alpha_r = slip_angle(v_1x, v_yr)
        alpha_t = slip_angle(v_2x, v_2y_wheel)

        fzf = (
            vehicle.mass * 9.8 * vehicle.lr / (vehicle.lf + vehicle.lr)
            + vehicle.trailer_mass
            * 9.8
            * vehicle.l2r
            * (vehicle.lr - vehicle.hitch_offset)
            / ((vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r))
        )
        f_1yf = -compute_fy(
            alpha_f,
            vehicle.cornering_stiffness_front,
            fzf,
            0,
            mu,
            vehicle.gamma,
        )

        fzr = (
            vehicle.mass * 9.8 * vehicle.lf / (vehicle.lf + vehicle.lr)
            + vehicle.trailer_mass
            * 9.8
            * vehicle.l2r
            * (vehicle.lf + vehicle.hitch_offset)
            / ((vehicle.lf + vehicle.lr) * (vehicle.l2f + vehicle.l2r))
        )
        throttle_blend = 0.5 * (1 + ca.tanh(accel_cmd / 0.02))
        accel_limit = (
            throttle_blend * vehicle.max_accel
            + (1 - throttle_blend) * vehicle.max_brake
        )
        commanded = accel_cmd * accel_limit
        fxr = mu * fzr * ca.tanh(vehicle.mass * commanded / (fzr * mu))
        f_1yr = -compute_fy(
            alpha_r,
            vehicle.cornering_stiffness_rear,
            fzr,
            fxr,
            mu,
            vehicle.gamma,
        )

        fzr_trailer = vehicle.trailer_mass * 9.8 * vehicle.l2f / (
            vehicle.l2f + vehicle.l2r
        )
        f_2yr = -compute_fy(
            alpha_t,
            vehicle.cornering_stiffness_trailer,
            fzr_trailer,
            0,
            mu,
            vehicle.gamma,
        )

        total_mass = vehicle.mass + vehicle.trailer_mass
        alpha_dot = phi_1_dot - phi_2_dot
        A = ca.vertcat(
            ca.horzcat(total_mass, 0, 0, -vehicle.trailer_mass * vehicle.l2f * sa),
            ca.horzcat(
                0,
                total_mass,
                -vehicle.trailer_mass * vehicle.hitch_offset,
                -vehicle.trailer_mass * vehicle.l2f * cos_alpha,
            ),
            ca.horzcat(
                0,
                -vehicle.trailer_mass * vehicle.hitch_offset,
                vehicle.inertia_z + vehicle.trailer_mass * vehicle.hitch_offset**2,
                vehicle.trailer_mass
                * vehicle.l2f
                * vehicle.hitch_offset
                * cos_alpha,
            ),
            ca.horzcat(
                -vehicle.trailer_mass * vehicle.l2f * sa,
                -vehicle.trailer_mass * vehicle.l2f * cos_alpha,
                vehicle.trailer_mass
                * vehicle.l2f
                * vehicle.hitch_offset
                * cos_alpha,
                vehicle.trailer_inertia_z + vehicle.trailer_mass * vehicle.l2f**2,
            ),
        )
        b = ca.vertcat(
            fxr
            - f_1yf * sd
            + f_2yr * sa
            + vehicle.mass * v_1y * phi_1_dot
            + vehicle.trailer_mass * phi_1_dot * (v_2y * cos_alpha - v_2x * sa)
            + vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * cos_alpha,
            f_1yr
            + f_1yf * cd
            + f_2yr * cos_alpha
            - vehicle.mass * v_1x * phi_1_dot
            - vehicle.trailer_mass * phi_1_dot * (v_2x * cos_alpha + v_2y * sa)
            - vehicle.trailer_mass * vehicle.l2f * alpha_dot * phi_2_dot * sa,
            -f_1yr * vehicle.lr
            + f_1yf * cd * vehicle.lf
            - vehicle.hitch_offset * f_2yr * cos_alpha
            + vehicle.trailer_mass
            * vehicle.hitch_offset
            * phi_1_dot
            * (v_2x * cos_alpha + v_2y * sa)
            + vehicle.trailer_mass
            * vehicle.hitch_offset
            * vehicle.l2f
            * alpha_dot
            * phi_2_dot
            * sa,
            -(vehicle.l2f + vehicle.l2r) * f_2yr
            + vehicle.trailer_mass * vehicle.l2f * v_2x * phi_1_dot,
        )
        v_1x_dot, v_1y_dot, phi_1_ddot, phi_2_ddot = ca.vertsplit(ca.solve(A, b))

        next_vx = v_1x + v_1x_dot * step
        next_vy = v_1y + v_1y_dot * step
        next_phi_1_dot = phi_1_dot + phi_1_ddot * step
        next_phi_2_dot = phi_2_dot + phi_2_ddot * step
        avg_vx = 0.5 * (v_1x + next_vx)
        avg_vy = 0.5 * (v_1y + next_vy)
        avg_phi_1_dot = 0.5 * (phi_1_dot + next_phi_1_dot)
        avg_phi_2_dot = 0.5 * (phi_2_dot + next_phi_2_dot)
        xdot = avg_vx * ca.cos(phi_1) - avg_vy * ca.sin(phi_1)
        ydot = avg_vx * ca.sin(phi_1) + avg_vy * ca.cos(phi_1)

        _, track_vel = track_terms(state)
        dx = ca.vertcat(
            xdot,
            ydot,
            avg_phi_1_dot,
            avg_phi_2_dot,
            v_1x_dot,
            v_1y_dot,
            phi_1_ddot,
            phi_2_ddot,
            0,
            track_vel,
        )
        return ca.Function("fiala_dynamics", [state, u], [dx])

    def cost(state, u):
        lateral_error, track_vel = track_terms(state)
        hitch_angle = ca.atan2(ca.sin(state[2] - state[3]), ca.cos(state[2] - state[3]))
        track_limit = params.track.width * 0.5 * 0.9 - 0.1

        def smooth_abs(value):
            return ca.sqrt(value**2 + 1e-8)

        def smooth_positive(value):
            return 0.5 * (value + ca.sqrt(value**2 + 1e-8))

        violation = smooth_positive(smooth_abs(lateral_error) - track_limit)
        violation += smooth_positive(smooth_abs(hitch_angle) - vehicle.max_hitch)

        if v_target is None:
            v_term = reverse * p_weight * track_vel
        else:
            v_term = p_weight * (track_vel - v_target) ** 2

        # p_slow_weight and s_weight are unused (shared JAX signature).
        return (
            v_term
            + viol_weight * violation**2
            + lateral_error**2 * c_weight
            + hitch_angle**2 * a_weight
            + u[0] ** 2
            + 1e-2 * u[1] ** 2
        )

    def constraints(opti, state, u):
        opti.subject_to(opti.bounded(-1.0, u, 1.0))

    return dynamics, cost, constraints
