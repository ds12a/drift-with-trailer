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
from beamngpy import BeamNGpy, Scenario, Vehicle, Road, angle_to_quat
from pathlib import Path
import pandas as pd

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

beamng_home_dir = None
beamng_user_dir = None

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

def yaw_to_quat(yaw):
    yaw -= np.pi / 2
    return (0.0, 0.0, np.cos(yaw/2), np.sin(yaw/2))  # terrible

tractor = Vehicle("car", model="scintilla", part_config="vehicles/scintilla/hitch.pc")
dx, dy = (centerline[1] - centerline[0])[:2]
yaw = np.arctan2(dy, dx)
# yaw_quat = np.array([0.0, 0.0, -float(np.sin(yaw * 0.5)), float(np.cos(yaw * 0.5))])
yaw_quat = yaw_to_quat(yaw)
scenario.add_vehicle(
    tractor,
    pos=centerline[0, :3],
    rot_quat=yaw_quat,
)

trailer = Vehicle("trailer", model="cargotrailer")
l = 7.0
tangent = np.array([dx, dy, 0]) / np.linalg.norm(np.array([dx, dy, 0]))
trailer_pos = centerline[0, :3].reshape(3) - tangent * l
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