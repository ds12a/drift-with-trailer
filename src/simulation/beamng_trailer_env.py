from __future__ import annotations
from typing import Any

from src.simulation.config.trailer_bicycle_config import (
    TrackConfig,
    VehicleConfig,
    SimulationConfig,
    TrailerBicycleEnvConfig,
)
from src.utils.track import TrackModel, TrackProjection
from src.simulation.config.trailer_beamng_config import BeamNGTrailerEnvConfig
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd
from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import AdvancedIMU, Electrics
from pathlib import Path


from src.simulation.rendering import PyBulletMirrorRenderer


from dataclasses import asdict, dataclass, astuple


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


class BeamNGTrailerEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        config: BeamNGTrailerEnvConfig=None,
        beamng_home_dir=None,
        beamng_user_dir=None,
        headless=False,
    ) -> None:

        if config is None:
            config = BeamNGTrailerEnvConfig()

        super().__init__()

        self.config = config

        if beamng_home_dir is None:
            beamng_home_dir = Path.home() / "BeamNG.tech.v0.38.5.0"
        if beamng_user_dir is None:
            beamng_user_dir = Path.home() / ".local/share/BeamNG/BeamNG.tech/current"

        self.track = TrackModel.from_config(config.track)
        centerline = np.hstack(
            [
                self.track.centerline,
                np.zeros((self.track.centerline.shape[0], 1)),
                np.ones((self.track.centerline.shape[0], 1)) * self.track.width,
            ]
        )

        # Process to make it lighter on BeamNG engine
        def decimate_by_arclength(xy, min_spacing=5.0):
            keep = [0]
            for i in range(1, len(xy)):
                if np.linalg.norm(xy[i] - xy[keep[-1]]) >= min_spacing:
                    keep.append(i)
            return np.array(keep)

        idx = decimate_by_arclength(centerline[:, :2], min_spacing=10.0)
        centerline = centerline[idx]

        bng = BeamNGpy("localhost", 25252, home=beamng_home_dir, user=beamng_user_dir)
        bng.open(launch=True)
        bng.settings.set_deterministic(steps_per_second=20)
        self.bng = bng  # For convenience

        scenario = Scenario("tech_ground", "Barcelona")
        self.scenario = scenario
        material = "road_asphalt_2lane"
        CHUNK_SIZE = 30
        roads = []
        self.roads = roads
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

        tractor_xyz, trailer_xyz, yaw = self._initial_beamng_state()
        yaw_quat = BeamNGTrailerEnv.yaw_to_quat(yaw)
        tractor = Vehicle(
            "car", model="scintilla", part_config="vehicles/scintilla/hitch.pc"
        )
        self.tractor = tractor
        scenario.add_vehicle(
            tractor,
            pos=tractor_xyz,
            rot_quat=yaw_quat,
        )
        trailer = Vehicle("trailer", model="cargotrailer")
        self.trailer = trailer
        scenario.add_vehicle(
            trailer,
            pos=trailer_xyz,
            rot_quat=yaw_quat,
        )


        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        tractor.couplers.attach()
        bng.control.pause()

        self.imu1 = AdvancedIMU("imu1", bng, self.tractor)
        self.imu2 = AdvancedIMU("imu2", bng, self.trailer)

        e = Electrics()
        tractor.attach_sensor("e1", e)

        self._state: VehicleState | None = None

        obs_dim = 6

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    @classmethod
    def yaw_to_quat(cls, yaw):
        """
        In BeamNG convention, its really weird
        """
        yaw -= np.pi / 2
        return (0.0, 0.0, np.cos(yaw / 2), np.sin(yaw / 2))  # terrible

    def _initial_beamng_state(self):
        centerline = self.track.centerline
        tractor_xyz = np.concat([centerline[0], np.array([0.0])], axis=0)
        dx, dy = (centerline[1] - centerline[0])[:2]
        tangent = np.array([dx, dy, 0]) / np.linalg.norm(np.array([dx, dy, 0]))
        l = 7.0
        trailer_xyz = tractor_xyz - tangent * l
        yaw = np.arctan2(dy, dx)
        return tractor_xyz, trailer_xyz, yaw

    def _initial_env_state(self) -> VehicleState:
        tractor_xyz, _, yaw = self._initial_beamng_state()
        return VehicleState(
            *tractor_xyz[:2],
            yaw,
            yaw,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _poll_state(self) -> VehicleState:
        self.tractor.poll_sensors()
        self.trailer.poll_sensors()

        x, y, z = self.tractor.state["pos"]
        vx, vy, vz = self.tractor.state["vel"]
        dir1 = np.zeros(2)
        dir2 = np.zeros(2)

        dir1[0], dir1[1], _ = self.tractor.state["dir"]
        dir2[0], dir2[1], _ = self.trailer.state["dir"]

        phi1 = np.arctan2(*(dir1[::-1]))
        phi2 = np.arctan2(*(dir2[::-1]))

        # print(self.imu1.poll())
        imu1data = self.imu1.poll()
        if imu1data != []:
            lastimu1 = max(imu1data.keys())
            phi1dot = imu1data[lastimu1]["angVelSmooth"][2]
        else:
            phi1dot = self._state.yaw_truck_rate

        imu2data = self.imu2.poll()
        if imu2data != []:
            lastimu2 = max(imu2data.keys())
            phi2dot = imu2data[lastimu2]["angVelSmooth"][2]
        else:
            phi2dot = self._state.yaw_trailer_rate

        delta = self.tractor.sensors["e1"]["steering"]
        throt = self.tractor.sensors["e1"]["throttle"]

        return VehicleState(x, y, phi1, phi2, vx, vy, phi1dot, phi2dot, delta, throt)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        obs = np.concatenate(
            [
                np.array(
                    [
                        # self._state.progress,
                        # self._state.lateral_error,
                        # self._state.heading_error,
                        self._state.x,
                        self._state.y,
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._state = self._initial_env_state()
        tractor_xyz, trailer_xyz, yaw = self._initial_beamng_state()
        yaw_quat = BeamNGTrailerEnv.yaw_to_quat(yaw)
        self.tractor.teleport(pos=tuple(tractor_xyz), rot_quat=yaw_quat)
        self.trailer.teleport(pos=tuple(trailer_xyz), rot_quat=yaw_quat)

        self._previous_feature_state = None
        self._step_count = 0
        self._lap_count = 0

        _, self._last_index = self.track.project(self._state.x, self._state.y, None)

        self.bng.step(10)

        obs = self._observation()
        info = {
            "state": asdict(self._state),
            "reset": {},
        }
        return obs, info

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=float)
        proj, self._last_index = self.track.project(
            self._state.x, self._state.y, self._last_index
        )

        previous_progress = proj.progress

        steer, accel = action
        throttle = max(accel, 0.0)
        brake = min(accel, 0.0)
        self.tractor.control(steering=steer, throttle=throttle, brake=brake)
        self.bng.step(1)

        self._state = self._poll_state() # Note: There is control lag so throttle, brake are independent from MPPI control

        self._step_count += 1
        projection, self._last_index = self.track.project(
            self._state.x, self._state.y, self._last_index
        )
        if projection.progress < previous_progress - 0.5:
            self._lap_count += 1

        # render_state = self._render_state()
        render_state = None
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
            >= self.config.vehicle.max_hitch
        )

        return self._observation(), 0, terminated, False, info

    def render(self):
        # TODO impl MPPI viz
        # if self.render_mode is None or self.renderer_kind != "pybullet":
        #     return None
        # if self.renderer is None:
        #     self.renderer = PyBulletMirrorRenderer(
        #         self.scenario,
        #         self.track,
        #         self.render_mode,
        #         width=self.render_width,
        #         height=self.render_height,
        #     )
        # return self.renderer.render(
        #     self._render_state(), planner_debug=self.planner_debug
        # )
        pass

    def close(self):
        # if self.renderer is not None:
        #     self.renderer.close()
        #     self.renderer = None
        pass

if __name__ == "__main__":
    env = BeamNGTrailerEnv()
    env.reset()
    obs, reward, term, *_ = env.step(np.array([0, 0]))
    print(obs, term)
    while True:
        obs, reward, term, *_ = env.step(np.array([0, 1]))
        print(obs, term)