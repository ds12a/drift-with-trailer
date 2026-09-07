"""Fiala LQR steering and PID speed control."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.dynamics.trailer.trailer_bicycle_fiala import gen_util_funs
from src.utils.track import TrackModel, wrap_angle


class LQR_PID:
    def __init__(
        self,
        config,
        v_target,
        q=(0.1, 1.0, 5.0, 0.1, 1.0, 1.0),
        r=10.0,
        design_mu=1.0,
        min_design_speed=0.5,
        curvature_spacing=3.0,
        heading_smoothing=5.0,
        kp=1.5,
        ki=0.3,
        kd=0.1,
        integral_limit=10.0,
        derivative_tau=0.2,
    ):
        self.track = TrackModel.from_config(config.track)
        self.vehicle = config.vehicle
        self.dt = config.simulation.dt
        if not config.track.closed:
            raise ValueError("LQR_PID currently requires a closed track.")
        q = np.asarray(q, dtype=float)
        settings = [
            r, design_mu, min_design_speed, curvature_spacing, heading_smoothing, kp, ki, kd,
            integral_limit, derivative_tau, self.dt, v_target,
        ]
        if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError("q must contain six positive finite state weights.")
        if not np.all(np.isfinite(settings)) or min(
            r, design_mu, min_design_speed, curvature_spacing, heading_smoothing, self.dt
        ) <= 0:
            raise ValueError("Invalid LQR settings or timestep.")
        if min(kp, ki, kd, integral_limit, derivative_tau) < 0:
            raise ValueError("PID gains and limits must be nonnegative.")

        self.v_target = v_target
        self.min_design_speed = min_design_speed
        self.curvature_spacing = curvature_spacing
        self.heading_smoothing = heading_smoothing
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.derivative_tau = derivative_tau
        self.design_mu = design_mu
        dynamics, *_ = gen_util_funs(config, v_target=v_target, s_weight=0)

        def lateral_dynamics(z, delta, velocity):
            # z = [rear-axle lateral error, heading error, hitch, vy, r1, r2]
            e, heading, beta, vy, r1, r2 = z
            state = jnp.array([
                self.vehicle.lr * jnp.cos(heading),
                e + self.vehicle.lr * jnp.sin(heading),
                heading, heading + beta, velocity, vy, r1, r2, design_mu, 0.0,
            ])
            dx = dynamics(state, jnp.array([delta / self.vehicle.max_steer_rad, 0.0]))
            # Use instantaneous pose rates; dynamics() averages them over dt.
            return jnp.array([
                velocity * jnp.sin(heading) + (vy - self.vehicle.lr * r1) * jnp.cos(heading),
                r1, r2 - r1, dx[5], dx[6], dx[7],
            ])

        self.lateral_dynamics = jax.jit(lateral_dynamics)
        self._jacobian = jax.jit(jax.jacfwd(lateral_dynamics, argnums=(0, 1)))

        # Forward/reverse gains at the target speed. DARE is singular at v=0.
        speed = max(abs(v_target), min_design_speed)
        self.gains = {}
        self.references = {}
        for direction in (-1, 1):
            a, b = self.linear_model(direction * speed)
            p = solve_discrete_are(a, b, np.diag(q), np.array([[r]]))
            self.gains[direction] = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)[0]
            self.references[direction] = self.curvature_reference(direction * speed)
        self.reset()

    def continuous_model(self, velocity):
        """Straight-motion Jacobians at fixed vx/mu, accel=0. Input is steer in rad."""
        a, b = self._jacobian(jnp.zeros(6), 0.0, float(velocity))
        return np.asarray(a, dtype=float), np.asarray(b, dtype=float)[:, None]

    def linear_model(self, velocity):
        """Zero-order-hold discretization of A, B."""
        a, b = self.continuous_model(velocity)
        augmented = np.zeros((7, 7))
        augmented[:6, :6] = a
        augmented[:6, 6:] = b
        discrete = expm(augmented * self.dt)
        return discrete[:6, :6], discrete[:6, 6:]

    def curvature_reference(self, velocity):
        """Steady-turn state and steer per unit curvature, with r1=r2=v*kappa."""
        a, b = self.continuous_model(velocity)
        matrix = np.column_stack([a[3:, 2], a[3:, 3], b[3:, 0]])
        beta, vy, delta = np.linalg.solve(matrix, -a[3:, 4:6] @ np.array([velocity, velocity]))
        reference = np.array([0, self.vehicle.lr - vy / velocity, beta, vy, velocity, velocity])
        return reference, delta

    def reset(self):
        self._last_index = None
        self._integral = 0.0
        self._previous_speed = None
        self._speed_derivative = 0.0

    def path_heading(self, arc):
        """Chord tangent for smoothed heading and curvature."""
        before = self.track.sample((arc - self.heading_smoothing) / self.track.length)
        after = self.track.sample((arc + self.heading_smoothing) / self.track.length)
        return np.arctan2(after.y - before.y, after.x - before.x)

    def speed_control(self, vx):
        error = self.v_target - vx
        derivative = 0.0 if self._previous_speed is None else (vx - self._previous_speed) / self.dt
        alpha = self.dt / (self.derivative_tau + self.dt)
        self._speed_derivative += alpha * (derivative - self._speed_derivative)
        self._previous_speed = vx

        integral = np.clip(
            self._integral + error * self.dt, -self.integral_limit, self.integral_limit
        )
        pd = self.kp * error - self.kd * self._speed_derivative
        requested = pd + self.ki * integral
        # Anti-windup
        if not (
            (requested > self.vehicle.max_accel and error > 0)
            or (requested < -self.vehicle.max_brake and error < 0)
        ):
            self._integral = integral
        accel = pd + self.ki * self._integral
        scale = self.vehicle.max_accel if accel >= 0 else self.vehicle.max_brake
        return np.clip(accel / scale, -1.0, 1.0)

    def run_mpc(self, state):
        """Return normalized [steer, accel]. Negate steer for BeamNG."""
        x, y, yaw, trailer_yaw, vx, vy, r1, r2 = np.asarray(state, dtype=float)[:8]
        rear_x = x - self.vehicle.lr * np.cos(yaw)
        rear_y = y - self.vehicle.lr * np.sin(yaw)
        projection, self._last_index = self.track.project(rear_x, rear_y, self._last_index)
        # Heading and curvature use the same smoothed path tangent.
        spacing = self.curvature_spacing
        heading = self.path_heading(projection.arc_length)
        before = self.path_heading(projection.arc_length - spacing)
        after = self.path_heading(projection.arc_length + spacing)
        curvature = wrap_angle(after - before) / (2 * spacing)
        lateral_state = np.array([
            projection.lateral_error,
            wrap_angle(yaw - heading),
            wrap_angle(trailer_yaw - yaw), vy, r1, r2,
        ])
        direction_speed = vx if abs(vx) >= self.min_design_speed else self.v_target
        direction = -1 if direction_speed < 0 else 1
        reference, feedforward = self.references[direction]
        error = lateral_state - reference * curvature
        error[1:3] = wrap_angle(error[1:3])
        delta = feedforward * curvature - self.gains[direction] @ error
        steer = np.clip(delta / self.vehicle.max_steer_rad, -1.0, 1.0)
        return np.array([steer, self.speed_control(vx)])
