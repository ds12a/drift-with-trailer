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
from beamngpy import BeamNGpy, Scenario, Vehicle
from pathlib import Path


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



class BeamngTrailerEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        beamng_home_dir = None,
        beamng_user_dir = None,
        renderer: str | None = None,
        v_init = 6.0,
    ) -> None:

        super().__init__()
        self.v_init = v_init
        
        csv = "src/simulation/assets/tracks/ks_barcelona_layout_gp_centerline.csv"
        width = 10

        frame = pd.read_csv(Path(csv))
        if {"x", "y"}.issubset(frame.columns):
            centerline = frame[["x", "y"]].to_numpy(dtype=float)
        else:
            centerline = frame.iloc[:, :2].to_numpy(dtype=float)

        print(centerline.shape)
        centerline = np.hstack(
            [
                centerline,
                np.zeros((centerline.shape[0], 1)),
                np.ones((centerline.shape[0], 1)) * width,
            ]
        )


        def decimate_by_arclength(xy, min_spacing=5.0):
            keep = [0]
            for i in range(1, len(xy)):
                if np.linalg.norm(xy[i] - xy[keep[-1]]) >= min_spacing:
                    keep.append(i)
            return np.array(keep)


        idx = decimate_by_arclength(centerline[:, :2], min_spacing=10.0)
        centerline = centerline[idx]
        print(centerline.shape)

        if beamng_home_dir is None:
            beamng_home_dir = Path.home() / "BeamNG.tech.v0.38.5.0"
        if beamng_user_dir is None:
            beamng_user_dir = Path.home() / ".local/share/BeamNG/BeamNG.tech/current"
        print(beamng_home_dir)
        bng = BeamNGpy("localhost", 25252, home=beamng_home_dir, user=beamng_user_dir)
        bng.open(launch=True)
        bng.settings.set_deterministic(steps_per_second=50)

        scenario = Scenario("tech_ground", "Barcelona")

        road = Road("track_editor_C_center", rid="test_road", looped=False)
        material = "road_asphalt_2lane"
        CHUNK_SIZE = 30
        roads = []
        for k, start in enumerate(range(0, len(centerline), CHUNK_SIZE)):
            seg = centerline[start : start + CHUNK_SIZE + 1]
            if len(seg) < 2:
                continue
            r = Road(material, rid=f"segment_{k}", looped=False)
            r.add_nodes(*(seg.tolist()))
            roads.append(r)
            scenario.add_road(r)
        tail = np.vstack([centerline[-1], centerline[0]])
        r = Road(material, rid=f"segment_close", looped=False)
        r.add_nodes(*(tail.tolist()))
        roads.append(r)
        scenario.add_road(r)

        tractor = Vehicle("car", model="scintilla", part_config="vehicles/scintilla/hitch.pc")
        dx, dy = -(centerline[1] - centerline[0])[:2]
        yaw = np.arctan2(dx, dy)
        yaw_quat = np.array([0.0, 0.0, float(np.sin(yaw * 0.5)), float(np.cos(yaw * 0.5))])
        scenario.add_vehicle(
            tractor,
            pos=centerline[0, :3],
            rot_quat=yaw_quat,
        )

        trailer = Vehicle("trailer", model="cargotrailer")
        l = 7.0
        tangent = np.array([dx, dy, 0]) / np.linalg.norm(np.array([dx, dy, 0]))
        trailer_pos = centerline[0, :3].reshape(3) + tangent * l
        print(trailer_pos)
        scenario.add_vehicle(
            trailer,
            pos=trailer_pos,
            rot_quat=yaw_quat,
        )

        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        tractor.couplers.attach()

        self._state: VehicleState | None = None

        obs_dim = 6

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
            speed = float(initial_speed if initial_speed is not None else self.v_init)
            return self.dynamics.initial_state(
                self.track,
                progress=progress,
                lateral_error=lateral_error,
                heading_error=heading_error,
                speed=speed,
            )

        return self.dynamics.initial_state(self.track, progress=0.0, speed=self.v_init)

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

        _, self._last_index = self.track.project(self._state.x, self._state.y, None)

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
        proj, self._last_index = self.track.project(self._state.x, self._state.y, self._last_index)

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

        def wrap_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        terminated = (
            self.track.out_of_bounds(projection.lateral_error)
            or np.abs(wrap_angle(self._state.yaw_trailer - self._state.yaw_truck))
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
        return self.renderer.render(self._render_state(), planner_debug=self.planner_debug)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
