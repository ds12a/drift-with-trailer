from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd

# import DynamicBicycleModel, VehicleState
# from uncertain_racecar_gym.features import build_feature_vector_from_state
from rendering import PyBulletMirrorRenderer
from uncertain_racecar_gym.scenario import DEFAULT_SCENARIO, Scenario, load_scenario
from uncertain_racecar_gym.track import TrackModel

from dataclasses import dataclass
@dataclass(slots=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    progress: float
    lateral_error: float
    heading_error: float
    vx: float
    vy: float
    yaw_rate: float
    steer: float
    throttle: float
    brake: float
    wheel_rotation: float
    lap_count: int = 0
    step_count: int = 0


class UncertainRacecarEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        scenario: str | Path | None = None,
        apply_mean_correction: bool = False,
        gaussian_noise_mean: Sequence[float] | None = None,
        gaussian_noise_std: Sequence[float] | None = None,
        renderer: str | None = None,
        render_mode: str | None = None,
        render_width: int = 1280,
        render_height: int = 720,
    ) -> None:
        super().__init__()
        self.scenario: Scenario = load_scenario(scenario or DEFAULT_SCENARIO)
        self.track = TrackModel.from_config(self.scenario.track)
        self.dynamics = DynamicBicycleModel(self.scenario.vehicle)

        self.renderer_kind = renderer
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render_width and render_height must be positive integers.")
        self.renderer = None
        self.reset_rows = None
        self.apply_mean_correction = bool(apply_mean_correction)
        self.gaussian_noise_mean = self._normalize_gaussian_vector(gaussian_noise_mean, default=np.zeros(4, dtype=float))
        self.gaussian_noise_std = self._normalize_gaussian_vector(
            gaussian_noise_std,
            default=np.array([0.12, 0.08, 0.05, 0.015], dtype=float),
            nonnegative=True,
        )
      
        self.runtime_track_id, self.runtime_car_id = self._resolve_runtime_ids()


        self._state: VehicleState | None = None

        obs_dim = 8

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, 0.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32)

    def _initial_state(
        self,
        initial_progress: float | None = None,
        initial_lateral_error: float | None = None,
        initial_heading_error: float | None = None,
        initial_speed: float | None = None,
    ) -> VehicleState:
        if any(value is not None for value in (initial_progress, initial_lateral_error, initial_heading_error, initial_speed)):
            progress = float(initial_progress if initial_progress is not None else 0.0) % 1.0
            lateral_error = float(initial_lateral_error if initial_lateral_error is not None else 0.0)
            heading_error = float(initial_heading_error if initial_heading_error is not None else 0.0)
            speed = float(initial_speed if initial_speed is not None else 8.0)
            return self.dynamics.initial_state(
                self.track,
                progress=progress,
                lateral_error=lateral_error,
                heading_error=heading_error,
                speed=speed,
            )
        # if start_mode == "dataset_match" and self.reset_rows is not None and len(self.reset_rows):
        #     row = self.reset_rows.iloc[int(self.np_random.integers(0, len(self.reset_rows)))]
        #     return self.dynamics.state_from_canonical_row(row)
        # if start_mode == "random":
        #     progress = float(self.np_random.uniform(0.0, 1.0))
        #     lateral_error = float(self.np_random.uniform(-0.2, 0.2))
        #     heading_error = float(self.np_random.uniform(-0.08, 0.08))
        #     speed = float(self.np_random.uniform(7.0, 12.0))
        #     return self.dynamics.initial_state(self.track, progress=progress, lateral_error=lateral_error, heading_error=heading_error, speed=speed)
        return self.dynamics.initial_state(self.track, progress=0.0, speed=8.0)


    def _observation(self) -> np.ndarray:
        assert self._state is not None
        lookahead_curvature = self.track.lookahead_curvatures(
            self._state.progress,
            count=self.scenario.simulation.lookahead_points,
            spacing_m=self.scenario.simulation.lookahead_spacing_m,
        )
      
      
        obs = np.concatenate(
            [
                np.array(
                    [
                        self._state.progress,
                        self._state.lateral_error,
                        self._state.heading_error,
                        self._state.vx,
                        self._state.vy,
                        self._state.yaw_rate,
                        self.track.sample(self._state.progress).curvature,
                    ],
                    dtype=np.float32,
                ),
                lookahead_curvature.astype(np.float32),
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
            "wheel_rotation": self._state.wheel_rotation,
            "progress": self._state.progress,
            "frame_index": self._state.step_count,
            "speed": self._state.vx,
        }

    def _reward(self, previous_progress: float, state: VehicleState) -> float:
        delta = state.progress - previous_progress
        if delta < -0.5:
            delta += 1.0
        penalty = (
            self.scenario.reward.lateral_error_coef * abs(state.lateral_error)
            + self.scenario.reward.heading_error_coef * abs(state.heading_error)
        )
        return float(
            delta * self.scenario.reward.progress_coef
            + self.scenario.reward.speed_coef * state.vx
            - penalty
        )



    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        start_mode = options.get("start_mode", options.get("mode", "grid"))

        self._state = self._initial_state(
            start_mode,
            initial_progress=options.get("initial_progress"),
            initial_lateral_error=options.get("initial_lateral_error"),
            initial_heading_error=options.get("initial_heading_error"),
            initial_speed=options.get("initial_speed"),
        )

        self._previous_feature_state = None
        

        obs = self._observation()
        info = {
            "state": self._render_state(),
            "render_state": self._render_state(),
            "reset": {
                "start_mode": start_mode,
                "initial_progress": None if options.get("initial_progress") is None else float(options["initial_progress"]),
                "initial_lateral_error": None if options.get("initial_lateral_error") is None else float(options["initial_lateral_error"]),
                "initial_heading_error": None if options.get("initial_heading_error") is None else float(options["initial_heading_error"]),
                "initial_speed": None if options.get("initial_speed") is None else float(options["initial_speed"]),
            },
        }
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        previous_progress = self._state.progress
        previous_state_for_features = self._state
        

        residual = np.zeros(4, dtype=float)
        calibration_info = None
       
        self._state = self.dynamics.step(self._state, action, self.track, self.scenario.simulation.dt, residual=residual)
        self._previous_feature_state = previous_state_for_features

        reward = self._reward(previous_progress, self._state)
        terminated = self.track.out_of_bounds(self._state.lateral_error)
        truncated = self._state.step_count >= self.scenario.simulation.max_steps

        render_state = self._render_state()
        info = {
            "state": render_state,
            "render_state": render_state,
            "calibration": calibration_info,
            "lap_count": self._state.lap_count,
        }
        return self._observation(), reward, terminated, truncated, info

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
