from __future__ import annotations
from typing import Any

from src.simulation.config.bicycle_config import (
    TrackConfig,
    VehicleConfig,
    SimulationConfig,
    BicycleEnvConfig,
)
from src.utils.track import TrackModel, TrackProjection
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd


from src.simulation.rendering import PyBulletMirrorRenderer


from dataclasses import dataclass, replace


def wrap_angle(angle: float) -> float:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    # progress: float
    # lateral_error: float
    # heading_error: float
    vx: float
    vy: float
    yaw_rate: float
    steer: float
    accel: float


def compute_fy(alpha, cc, fz, fx, mu, gamma):

    fy_max = jnp.sqrt((mu * fz) ** 2 - gamma * fx**2)

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


class DynamicBicycleModel:
    def __init__(self, config: BicycleEnvConfig):
        self.config = config

    def initial_state(
        self,
        track: TrackModel,
        progress: float = 0.0,
        lateral_error: float = 0.0,
        heading_error: float = 0.0,
        speed: float = 6.0,
    ) -> VehicleState:
        x, y, yaw = track.spawn_pose(
            progress, lateral_error=lateral_error, heading_error=heading_error
        )
        # projection = track.project(x, y)
        return VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            # progress=projection.progress,
            # lateral_error=projection.lateral_error,
            # heading_error=wrap_angle(yaw - projection.heading),
            vx=speed,
            vy=0.0,
            yaw_rate=0.0,
            steer=0.0,
            accel=0.0,
        )

    def step(self, state: VehicleState, action: jnp.ndarray, track: TrackModel) -> VehicleState:
        state_yaw = state.yaw
        state_xdot = state.vx
        state_ydot = state.vy
        state_yaw_dot = state.yaw_rate

        action = jnp.asarray(action, dtype=jnp.float32)
        steer_cmd = jnp.clip(action[0], -1.0, 1.0)
        accel_cmd = jnp.clip(action[1], -1.0, 1.0)
        dt = self.config.simulation.dt
        vehicle = self.config.vehicle

        # steer = state.steer + (steer_cmd - state.steer) * jnp.minimum(1.0, dt * 8.0)
        steer = steer_cmd
        accel = accel_cmd
        throttle = jnp.maximum(accel, 0.0)
        brake = -jnp.minimum(accel, 0.0)

        vx_safe = jnp.maximum(jnp.abs(state_xdot), 0.5)
        steer_angle = steer * vehicle.max_steer_rad
        alpha_f = steer_angle - jnp.arctan2(state_ydot + vehicle.lf * state_yaw_dot, vx_safe)
        alpha_r = -jnp.arctan2(state_ydot - vehicle.lr * state_yaw_dot, vx_safe)

        mu = track.find_mu(state.x, state.y)

        fyf = -compute_fy(
            alpha_f,
            vehicle.cornering_stiffness_front,
            vehicle.mass * 9.8 * vehicle.lr / (vehicle.lf + vehicle.lr),
            0,
            mu,
            vehicle.gamma,
        )

        fzr = vehicle.mass * 9.8 * vehicle.lf / (vehicle.lf + vehicle.lr)
        commanded = throttle * vehicle.max_accel - brake * vehicle.max_brake
        fxr = mu * fzr * jnp.tanh(vehicle.mass * commanded / (fzr * mu))

        fyr = -compute_fy(alpha_r, vehicle.cornering_stiffness_rear, fzr, fxr, mu, vehicle.gamma)

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

        longitudinal_acc, vx_dot, brake * vehicle.max_brake

        # print(
        #     f"Long: {longitudinal_acc:>7.3f} "
        #     f"ax: {vx_dot:>7.3f} "
        #     f"F_b: {brake * vehicle.max_brake:>7.3f} "
        #     f"b_sig: {brake:>7.3f} "
        #     f"accel: {accel:>7.3f}"
        # )

        # Trapezoidal (avg) approximations
        avg_vx = 0.5 * (state_xdot + next_vx)
        avg_vy = 0.5 * (state_ydot + next_vy)
        avg_yaw_rate = 0.5 * (state_yaw_dot + next_yaw_rate)

        # Change of frame
        xdot = avg_vx * jnp.cos(state_yaw) - avg_vy * jnp.sin(state_yaw)
        ydot = avg_vx * jnp.sin(state_yaw) + avg_vy * jnp.cos(state_yaw)

        next_x = state.x + xdot * dt
        next_y = state.y + ydot * dt
        next_yaw = state.yaw + avg_yaw_rate * dt

        # projection = track.project(next_x, next_y)

        return VehicleState(
            x=next_x,
            y=next_y,
            yaw=next_yaw,
            vx=next_vx,
            vy=next_vy,
            yaw_rate=next_yaw_rate,
            steer=steer,
            accel=accel,
            # progress=projection.progress,
            # lateral_error=projection.lateral_error,
            # heading_error=wrap_angle(next_yaw - projection.heading),
        )


class BicycleEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        renderer: str | None = None,
        render_mode: str | None = None,
        render_width: int = 1280,
        render_height: int = 720,
    ) -> None:
        super().__init__()

        self.scenario = BicycleEnvConfig(
            "aach aach aach", TrackConfig(), VehicleConfig(), SimulationConfig()
        )
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicBicycleModel(self.scenario)  # TODO

        self.renderer_kind = renderer
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render_width and render_height must be positive integers.")
        self.renderer = None

        # self.runtime_track_id, self.runtime_car_id = self._resolve_runtime_ids()

        self._state: VehicleState | None = None

        obs_dim = 7

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _initial_state(
        self,
        initial_progress: float | None = None,
        initial_lateral_error: float | None = None,
        initial_heading_error: float | None = None,
        initial_speed: float | None = None,
    ) -> VehicleState:
        if any(
            value is not None
            for value in (
                initial_progress,
                initial_lateral_error,
                initial_heading_error,
                initial_speed,
            )
        ):
            progress = float(initial_progress if initial_progress is not None else 0.0) % 1.0
            lateral_error = float(
                initial_lateral_error if initial_lateral_error is not None else 0.0
            )
            heading_error = float(
                initial_heading_error if initial_heading_error is not None else 0.0
            )
            speed = float(initial_speed if initial_speed is not None else 8.0)
            return self.dynamics.initial_state(
                self.track,
                progress=progress,
                lateral_error=lateral_error,
                heading_error=heading_error,
                speed=speed,
            )

        return self.dynamics.initial_state(self.track, progress=0.0, speed=8.0)

    def _observation(self) -> np.ndarray:
        assert self._state is not None

        obs = np.concatenate(
            [
                np.array(
                    [
                        # self._state.progress,
                        # self._state.lateral_error,
                        # self._state.heading_error,
                        self._state.vx,
                        self._state.vy,
                        self._state.yaw_rate,
                        # self.track.sample(self._state.progress).curvature,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def _render_state(self) -> dict[str, Any]:
        assert self._state is not None
        return {
            "x": self._state.x,
            "y": self._state.y,
            "yaw": self._state.yaw,
            "steering_angle": self._state.steer * self.scenario.vehicle.max_steer_rad,
            "speed": self._state.vx,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_mode = options.get("start_mode", options.get("mode", "grid"))

        self._state = self._initial_state(
            initial_progress=options.get("initial_progress"),
            initial_lateral_error=options.get("initial_lateral_error"),
            initial_heading_error=options.get("initial_heading_error"),
            initial_speed=options.get("initial_speed"),
        )

        self._previous_feature_state = None
        self._step_count = 0
        self._lap_count = 0

        obs = self._observation()
        info = {
            "state": self._render_state(),
            "render_state": self._render_state(),
            "reset": {
                "start_mode": start_mode,
                "initial_progress": (
                    None
                    if options.get("initial_progress") is None
                    else float(options["initial_progress"])
                ),
                "initial_lateral_error": (
                    None
                    if options.get("initial_lateral_error") is None
                    else float(options["initial_lateral_error"])
                ),
                "initial_heading_error": (
                    None
                    if options.get("initial_heading_error") is None
                    else float(options["initial_heading_error"])
                ),
                "initial_speed": (
                    None
                    if options.get("initial_speed") is None
                    else float(options["initial_speed"])
                ),
            },
        }
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        # previous_progress = self._state.progress
        # previous_state_for_features = self._state

        self._state = self.dynamics.step(self._state, action, self.track)
        # self._previous_feature_state = previous_state_for_features

        self._step_count += 1
        # if self._state.progress < previous_progress - 0.5:
        #     self._lap_count += 1

        # reward = self._reward(previous_progress, self._state)
        # terminated = self.track.out_of_bounds(self._state.lateral_error)
        # truncated = self._step_count >= self.scenario.simulation.max_steps

        render_state = self._render_state()
        info = {
            "state": render_state,
            "render_state": render_state,
            "lap_count": self._lap_count,
        }
        terminated = self.track.out_of_bounds(
            self.track.project(self._state.x, self._state.y).lateral_error
        )
        return self._observation(), 0, terminated, False, info

    def render(self):
        if self.render_mode is None or self.renderer_kind != "pybullet":
            return None
        if self.renderer is None:
            self.renderer = PyBulletMirrorRenderer(
                self.scenario,
                self.track,
                self.render_mode,
                width=self.render_width,
                height=self.render_height,
            )
        return self.renderer.render(self._render_state())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
