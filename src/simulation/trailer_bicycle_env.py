from __future__ import annotations
from typing import Any

from src.simulation.config.trailer_bicycle_config import (
    TrackConfig,
    VehicleConfig,
    SimulationConfig,
    TrailerBicycleEnvConfig,
)
from src.utils.track import TrackModel, TrackProjection
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd


from src.simulation.rendering import PyBulletMirrorRenderer


from dataclasses import dataclass, astuple


def wrap_angle(angle: float) -> float:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    yaw_truck: float
    yaw_trailer: float
    vx: float  # Truck frame
    vy: float  # Truck frame
    yaw_truck_rate: float
    yaw_trailer_rate: float
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


class DynamicTrailerBicycleModel:
    def __init__(self, config: TrailerBicycleEnvConfig):
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

        return VehicleState(
            x=x,
            y=y,
            yaw_truck=yaw,
            yaw_trailer=yaw + 0.5,
            vx=speed,
            vy=0.0,
            yaw_truck_rate=0.0,
            yaw_trailer_rate=0,
            steer=0.0,
            accel=0.0,
        )

    def step(self, state: VehicleState, action: jnp.ndarray, track: TrackModel) -> VehicleState:
        x, y, phi_1, phi_2, v_1x, v_1y, phi_1_dot, phi_2_dot, _, _ = astuple(state)

        """
        $$\begin{bmatrix}
        m_1+m_2 & 0 & 0 & -m_2 L_{2f}\sin\alpha\\
        0 & m_1+m_2 & -m_2 h & -m_2 L_{2f}\cos\alpha\\
        0 & -m_2 h & I_{z1}+m_2 h^2 & m_2 L_{2f} h\cos\alpha\\
        -m_2 L_{2f}\sin\alpha & -m_2 L_{2f}\cos\alpha & m_2 L_{2f} h\cos\alpha & I_{z2}+m_2 L_{2f}^2
        \end{bmatrix}
        \begin{bmatrix}\dot v_{1x}\\ \dot v_{1y}\\ \ddot\phi_1\\ \ddot\phi_2\end{bmatrix}
        =
        \begin{bmatrix}
        F_{1xr}+F_{1xf}\cos\delta_{1f}-F_{1yf}\sin\delta_{1f}+F_{2xr}\cos\alpha+F_{2yr}\sin\alpha +m_1 v_{1y}\dot\phi_1+m_2\dot\phi_1\left(v_{2y}\cos\alpha-v_{2x}\sin\alpha\right)+m_2 L_{2f}\dot\alpha\dot\phi_2\cos\alpha \\ 
        F_{1yr}+F_{1yf}\cos\delta_{1f}+F_{1xf}\sin\delta_{1f}-F_{2xr}\sin\alpha+F_{2yr}\cos\alpha -m_1 v_{1x}\dot\phi_1-m_2\dot\phi_1\left(v_{2x}\cos\alpha+v_{2y}\sin\alpha\right)-m_2 L_{2f}\dot\alpha\dot\phi_2\sin\alpha  \\ 
         -F_{1yr}L_{1r}+\left(F_{1yf}\cos\delta_{1f}+F_{1xf}\sin\delta_{1f}\right)L_{1f}+h F_{2xr}\sin\alpha-h F_{2yr}\cos\alpha +m_2 h\dot\phi_1\left(v_{2x}\cos\alpha+v_{2y}\sin\alpha\right)+m_2 h L_{2f}\dot\alpha\dot\phi_2\sin\alpha\\ 
         -(L_{2f}+L_{2r})F_{2yr}+m_2 L_{2f} v_{2x}\dot\phi_1
        \end{bmatrix}$$
        with

        $$v_{2x} = v_{1x} \cos \alpha - (v_{1y} - \dot{\phi}_{1}h)\sin \alpha$$
        $$v_{2y} = v_{1x} \sin \alpha + (v_{1}y - \dot{\phi}_{1}h) \cos \alpha - L_{2f}\dot{\phi}_{2}$$

        """

        vehicle = self.config.vehicle
        steer_cmd = jnp.clip(action[0], -1.0, 1.0)
        accel_cmd = jnp.clip(action[1], -1.0, 1.0)
        dt = self.config.simulation.dt

        throttle = jnp.maximum(accel_cmd, 0.0)
        brake = -jnp.minimum(accel_cmd, 0.0)

        vx_safe = jnp.maximum(jnp.abs(v_1x), 0.5)
        delta = steer_cmd * vehicle.max_steer_rad
        alpha_f = delta - jnp.arctan2(v_1y + vehicle.lf * phi_1_dot, vx_safe)
        alpha_r = -jnp.arctan2(v_1y - vehicle.lr * phi_1_dot, vx_safe)

        mu = track.find_mu(state.x, state.y)

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
            alpha_t,
            vehicle.cornering_stiffness_trailer,
            fzr_trailer,
            0,
            mu,
            vehicle.gamma,
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

        next_x = x + xdot * dt
        next_y = y + ydot * dt
        next_phi_1 = phi_1 + avg_phi_1_dot * dt
        next_phi_2 = phi_2 + avg_phi_2_dot * dt

        return VehicleState(
            next_x,
            next_y,
            next_phi_1,
            next_phi_2,
            next_vx,
            next_vy,
            next_phi_1_dot,
            next_phi_2_dot,
            steer_cmd,
            accel_cmd,
        )


class TrailerBicycleEnv(gym.Env):
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

        self.scenario = TrailerBicycleEnvConfig(
            "aach aach aach", TrackConfig(), VehicleConfig(), SimulationConfig()
        )
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicTrailerBicycleModel(self.scenario)

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
                        self._state.yaw_truck_rate,
                        self._state.yaw_trailer_rate,
                        self._state.yaw_truck,
                        self._state.yaw_trailer,
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
            "yaw": self._state.yaw_truck,
            "trailer_yaw": self._state.yaw_trailer,
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
        self._last_index = None

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
        proj, self._last_index = self.track.project(
            self._state.x, self._state.y, self._last_index
        )

        previous_progress = proj.progress

        self._state = self.dynamics.step(self._state, action, self.track)

        self._step_count += 1
        projection, self._last_index = self.track.project(
            self._state.x, self._state.y, self._last_index
        )
        if projection.progress < previous_progress - 0.5:
            self._lap_count += 1

        render_state = self._render_state()
        info = {
            "state": render_state,
            "render_state": render_state,
            "lap_count": self._lap_count,
        }
        terminated = (
            self.track.out_of_bounds(projection.lateral_error)
            or np.abs(self._state.yaw_trailer - self._state.yaw_truck)
            >= self.scenario.vehicle.max_hitch
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
