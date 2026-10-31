"""Causal Fiala LQR steering and fixed-target PID speed control."""

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
        kp=1.5,
        ki=0.3,
        kd=0.1,
        integral_limit=10.0,
        derivative_tau=0.2,
        gain_speed_step=1.0,
    ):
        self.track = TrackModel.from_config(config.track)
        self.vehicle = config.vehicle
        self.dt = config.simulation.dt
        if not config.track.closed:
            raise ValueError("LQR_PID currently requires a closed track.")

        q = np.asarray(q, dtype=float)
        settings = [
            r,
            design_mu,
            min_design_speed,
            kp,
            ki,
            kd,
            integral_limit,
            derivative_tau,
            gain_speed_step,
            self.dt,
            v_target,
        ]
        if q.shape != (6,) or not np.all(np.isfinite(q)) or np.any(q <= 0):
            raise ValueError("q must contain six positive finite state weights.")
        if not np.all(np.isfinite(settings)) or min(
            r, design_mu, min_design_speed, gain_speed_step, self.dt
        ) <= 0:
            raise ValueError("Invalid LQR settings or timestep.")
        if min(kp, ki, kd, integral_limit, derivative_tau) < 0:
            raise ValueError("PID gains and limits must be nonnegative.")

        self.v_target = v_target
        self.q = q
        self.r = r
        self.min_design_speed = min_design_speed
        self.gain_speed_step = gain_speed_step
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.derivative_tau = derivative_tau
        self.design_mu = design_mu

        dynamics, *_ = gen_util_funs(config, v_target=v_target, s_weight=0)

        def lateral_dynamics(z, delta, velocity):
            e, heading, beta, vy, r1, r2 = z
            state = jnp.array(
                [
                    self.vehicle.lr * jnp.cos(heading),
                    e + self.vehicle.lr * jnp.sin(heading),
                    heading,
                    heading + beta,
                    velocity,
                    vy,
                    r1,
                    r2,
                    design_mu,
                    0.0,
                ]
            )
            dx = dynamics(
                state, jnp.array([delta / self.vehicle.max_steer_rad, 0.0])
            )
            return jnp.array(
                [
                    velocity * jnp.sin(heading)
                    + (vy - self.vehicle.lr * r1) * jnp.cos(heading),
                    r1,
                    r2 - r1,
                    dx[5],
                    dx[6],
                    dx[7],
                ]
            )

        self.lateral_dynamics = jax.jit(lateral_dynamics)
        self._jacobian = jax.jit(jax.jacfwd(lateral_dynamics, argnums=(0, 1)))

        speed = max(abs(v_target), min_design_speed)
        self.gains = {
            direction: self._design_controller(direction * speed)
            for direction in (-1, 1)
        }
        target_bin = max(
            self.min_design_speed,
            self.gain_speed_step * round(speed / self.gain_speed_step),
        )
        self._controller_cache = {
            (direction, target_bin): gain for direction, gain in self.gains.items()
        }
        self.reset()

    def _design_controller(self, velocity):
        """Return the straight-line regulation gain for one measured speed."""
        a, b = self.linear_model(velocity)
        p = solve_discrete_are(a, b, np.diag(self.q), np.array([[self.r]]))
        return np.linalg.solve(self.r + b.T @ p @ b, b.T @ p @ a)[0]

    def controller_at_speed(self, velocity):
        """Return a cached controller at the nearest configured speed bin."""
        direction = -1 if velocity < 0 else 1
        speed = min(
            max(abs(velocity), self.min_design_speed),
            max(abs(self.v_target), self.min_design_speed),
        )
        speed = max(
            self.min_design_speed,
            self.gain_speed_step * round(speed / self.gain_speed_step),
        )
        key = (direction, speed)
        if key not in self._controller_cache:
            self._controller_cache[key] = self._design_controller(direction * speed)
        return self._controller_cache[key]

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

    def reset(self):
        self._last_index = None
        self._integral = 0.0
        self._previous_speed = None
        self._speed_derivative = 0.0

    def speed_control(self, vx):
        error = self.v_target - vx
        derivative = (
            0.0
            if self._previous_speed is None
            else (vx - self._previous_speed) / self.dt
        )
        alpha = self.dt / (self.derivative_tau + self.dt)
        self._speed_derivative += alpha * (derivative - self._speed_derivative)
        self._previous_speed = vx

        integral = np.clip(
            self._integral + error * self.dt,
            -self.integral_limit,
            self.integral_limit,
        )
        pd = self.kp * error - self.kd * self._speed_derivative
        requested = pd + self.ki * integral
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
        x, y, yaw, trailer_yaw, vx, vy, r1, r2 = np.asarray(
            state, dtype=float
        )[:8]
        rear_x = x - self.vehicle.lr * np.cos(yaw)
        rear_y = y - self.vehicle.lr * np.sin(yaw)
        projection, self._last_index = self.track.project(
            rear_x, rear_y, self._last_index
        )
        lateral_state = np.array(
            [
                projection.lateral_error,
                wrap_angle(yaw - projection.heading),
                wrap_angle(trailer_yaw - yaw),
                vy,
                r1,
                r2,
            ]
        )
        direction_speed = vx if abs(vx) >= self.min_design_speed else self.v_target
        delta = -self.controller_at_speed(direction_speed) @ lateral_state
        steer = np.clip(delta / self.vehicle.max_steer_rad, -1.0, 1.0)
        return np.array([steer, self.speed_control(vx)])
