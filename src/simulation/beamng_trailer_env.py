from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any

from src.simulation.config.trailer_beamng_config import (
    TrackConfig,
    VehicleConfig,
    SimulationConfig,
)
from src.utils.track import TrackModel, TrackProjection, BadGuessException
from src.simulation.config.trailer_beamng_config import BeamNGTrailerEnvConfig
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd
from beamngpy import BeamNGpy, Scenario, Vehicle, Road, set_up_simple_logging
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


# One of the following cfgs MUST be used or there will be big divergence in prior
bng_pickup_trailer_cfg = VehicleConfig(
    # Tractor (Gavril D-Series / "pickup")
    wheelbase=3.0,
    lf=1.35,                          # ~45% front, 55% rear (RWD truck, engine forward of front axle but heavy rear)
    lr=1.65,
    mass=2000.0,                      # ~4400 lbs
    inertia_z=4500.0,                 # box-on-frame pickup, ~mass * 0.27 * wheelbase^2
    cornering_stiffness_front=75000.0,   # tall sidewall truck tires, softer than sports car
    cornering_stiffness_rear=85000.0,    # solid rear axle, slightly stiffer
    max_steer_rad=0.55,               # ~31 deg road-wheel, typical pickup rack
    max_accel=8.0,                    # 230 hp / 2000 kg, realistic
    max_brake=12.0,                   # drums rear, discs front, no sport brakes
    drag_coefficient=0.75,            # boxy body
    wheel_radius=0.39,               # 245/75R16
    chassis_size=[3.2, 1.5, 0.35],
    gamma=1,

    # Trailer (cargotrailer, tandem axle modeled as single effective axle)
    trailer_mass=830,
    trailer_inertia_z=650,            # enclosed box ~4.5m long
    l2f=2.8,                         # tongue to effective axle (~long enclosed body)
    l2r=0.3,                         # effective tandem midpoint to rear
    cornering_stiffness_trailer=60000.0,  # small 15" trailer tires, but 4 of them (tandem)
    hitch_offset=2.0,                # CG to hitch ball at rear bumper (~lr + 0.35m)
    max_hitch=np.deg2rad(80),
)


class BeamNGTrailerEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array_follow", "rgb_array_birds_eye"],
        "render_fps": 20,
    }

    def __init__(
        self,
        config: BeamNGTrailerEnvConfig = None,
        beamng_home_dir=None,
        beamng_user_dir=None,
        headless=False,
    ) -> None:

        if config is None:
            config = BeamNGTrailerEnvConfig()

        super().__init__()

        self.config = config
        self.planner_debug = None
        self._debug_line_ids = []

        if beamng_home_dir is None:
            beamng_home_dir = Path.home() / "Executables/BeamNG" # "BeamNG.tech.v0.38.5.0"
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

        set_up_simple_logging(level=logging.WARNING)
        bng = BeamNGpy("localhost", 25252, home=beamng_home_dir, user=beamng_user_dir)
        bng.open(launch=True)
        logging.getLogger("beamngpy").handlers.clear()
        logging.getLogger("beamngpy").setLevel(logging.WARNING)

        # Stupid
        bng.settings.set_deterministic(speed_factor=1)
        bng.control.queue_lua_command("settings.setValue('fpsLimitEnabled', false)")
        bng.control.queue_lua_command(
            "settings.setValue('fpsLimitBackgroundEnabled', false)"
        )
        bng.set_steps_per_second(1 / self.config.simulation.dt)
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
            r = Road(material, rid=f"segment_{k}", looped=False, interpolate=False)
            r.add_nodes(*(seg.tolist()))
            roads.append(r)
            scenario.add_road(r)
        tail = np.vstack([centerline[-1], centerline[0]])
        r = Road(material, rid=f"segment_close", looped=False, interpolate=False)
        r.add_nodes(*(tail.tolist()))
        roads.append(r)
        scenario.add_road(r)

        tractor_xyz, trailer_xyz, yaw = self._initial_beamng_state()
        yaw_quat = BeamNGTrailerEnv.yaw_to_quat(yaw)
        # tractor = Vehicle(
        #     "car", model="scintilla", part_config="vehicles/scintilla/hitch.pc"
        # )
        tractor = Vehicle("car", model="pickup", part_config="vehicles/pickup/hitch.pc")
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

        # self.imu1 = AdvancedIMU("imu1", bng, self.tractor)
        # self.imu2 = AdvancedIMU("imu2", bng, self.trailer)

        e = Electrics()
        tractor.attach_sensor("e1", e)

        self._state: VehicleState | None = None

        obs_dim = 10
        self._pool = ThreadPoolExecutor(max_workers=2)

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
        l = 7.5
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

        # t1 = time.perf_counter()
        self.tractor.poll_sensors() # Costs a frame, unavoidable
        self.trailer.poll_sensors()

        # f1 = self._pool.submit(self.tractor.poll_sensors())
        # f2 = self._pool.submit(self.trailer.poll_sensors())
        # f1.result(); f2.result()
        # print(time.perf_counter() - t1)

        x, y, z = self.tractor.state["pos"]
        vx_w, vy_w, vz_w = self.tractor.state["vel"]
        dir1 = np.zeros(2)
        dir2 = np.zeros(2)

        dir1[0], dir1[1], _ = self.tractor.state["dir"]
        dir2[0], dir2[1], _ = self.trailer.state["dir"]

        phi1 = np.arctan2(*(dir1[::-1]))
        phi2 = np.arctan2(*(dir2[::-1]))

        c, s = np.cos(phi1), np.sin(phi1)
        vx =  c * vx_w + s * vy_w
        vy = -s * vx_w + c * vy_w

        # Actual IMU polling is slow
        phi1dot = wrap_angle(phi1 - self._state.yaw_truck) / self.config.simulation.dt
        phi2dot = wrap_angle(phi2 - self._state.yaw_trailer) / self.config.simulation.dt

        # print("steer: ", self.tractor.sensors["e1"]["steering"])

        delta = self.tractor.sensors["e1"]["steering"] / 720.0 # Normalization
        throt = self.tractor.sensors["e1"]["throttle"]
        brake = self.tractor.sensors["e1"]["brake"]
        accel_realized = throt - brake
        # print(delta, throt)

        return VehicleState(x, y, phi1, phi2, vx, vy, phi1dot, phi2dot, delta, accel_realized)

    def _observation(self) -> np.ndarray:
        assert self._state is not None
        obs = np.array(
            [
                # self._state.progress,
                # self._state.lateral_error,
                # self._state.heading_error,
                self._state.x,
                self._state.y,
                self._state.yaw_truck,
                self._state.yaw_trailer,
                self._state.vx,
                self._state.vy,
                self._state.yaw_truck_rate,
                self._state.yaw_trailer_rate,
                self._state.steer,
                self._state.accel,
                # self.track.sample(self._state.progress).curvature,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._state = self._initial_env_state()
        tractor_xyz, trailer_xyz, yaw = self._initial_beamng_state()
        yaw_quat = BeamNGTrailerEnv.yaw_to_quat(yaw)
        self.tractor.teleport(pos=tuple(tractor_xyz), rot_quat=yaw_quat)
        self.trailer.teleport(pos=tuple(trailer_xyz), rot_quat=yaw_quat)

        self._step_count = 0
        self._lap_count = 0

        _, self._last_index = self.track.project(self._state.x, self._state.y, None)

        self.tractor.couplers.attach()
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
        brake = -min(accel, 0.0)
        self.tractor.control(steering=steer, throttle=throttle, brake=brake)
        self.render()
        self.bng.step(1)

        self._state = (
            self._poll_state()
        )  # Note: There is control lag so throttle, brake are independent from MPPI control

        self._step_count += 1

        try:
            projection, self._last_index = self.track.project(
                self._state.x, self._state.y, self._last_index
            )
        except BadGuessException:
            # Occurs when sim is restarted by pressing r, invalidating the guess
            projection, self._last_index = self.track.project(
                self._state.x, self._state.y, None
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

        terminated = (
            self.track.out_of_bounds(projection.lateral_error)
            or np.abs(wrap_angle(self._state.yaw_trailer - self._state.yaw_truck))
            >= self.config.vehicle.max_hitch
        )

        return self._observation(), 0, terminated, False, info

    def render(self, candidate_rgba=None):
        if self.planner_debug is None:
            return

        if hasattr(self, "_debug_line_ids") and self._debug_line_ids:
            data = {"type": "RemoveDebugObjects", "objType": "polylines",
                    "objIDs": self._debug_line_ids}
            self.bng._send(data).ack("DebugObjectsRemoved")
        self._debug_line_ids = []

        cand_xy = self.planner_debug.get("candidate_xy")
        if cand_xy is None:
            return

        z = self.tractor.state["pos"][2] + 0.3
        n_vis, T, _ = cand_xy.shape

        stride = max(1, T // 15)
        t_idx = list(range(0, T, stride))
        if t_idx[-1] != T - 1:
            t_idx.append(T - 1)

        default_color = (0.0, 1.0, 0.0, 0.5)

        for i in range(n_vis):
            coords = [(float(cand_xy[i, t, 0]), float(cand_xy[i, t, 1]), z)
                    for t in t_idx]
            color = candidate_rgba[i] if candidate_rgba is not None else default_color
            lid = self.bng.debug.add_polyline(coords, rgba_color=color, cling=True, offset=0.3)
            self._debug_line_ids.append(lid)

    def close(self):
        self.bng.close()


if __name__ == "__main__":
    env = BeamNGTrailerEnv()
    env.reset()
    obs, reward, term, *_ = env.step(np.array([0, 0]))
    print(obs, term)
    while True:
        obs, reward, term, *_ = env.step(np.array([1, 0]))
        if term:
            break
        print(obs, term)
