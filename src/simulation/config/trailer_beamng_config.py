from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class TrackConfig:
    csv: str = "src/simulation/assets/tracks/ks_barcelona_layout_gp_centerline.csv"
    # friction_csv = "src/simulation/assets/tracks/barcelona_ice.csv"

    # csv = "src/simulation/assets/tracks/sample_oval_centerline.csv"
    friction_csv: str = "src/simulation/assets/tracks/oval_ice.csv"
    mu: float = 1
    width: float = 8.0
    closed: bool = True
    # progress bins seems uncecessary, check


@dataclass(slots=True)
class VehicleConfig:
    wheelbase = 3.05
    lf = 1.45
    lr = 1.6
    mass = 2400.0
    inertia_z = 6500.0
    cornering_stiffness_front = 90000.0
    cornering_stiffness_rear = 98000.0
    max_steer_rad = 0.32
    max_accel = 12.0
    max_brake = 18.0
    drag_coefficient = 0.85
    wheel_radius = 0.33
    chassis_size = [3.2, 1.4, 0.32]
    gamma = 1

    # Trailer
    trailer_mass = 1225
    trailer_inertia_z = 850
    l2f = 2.05
    l2r = 0.4
    cornering_stiffness_trailer = 80000.0
    hitch_offset = 2.3  # tractor CG to hitch (positive behind)
    max_hitch = np.deg2rad(90)


@dataclass(slots=True)
class SimulationConfig:
    dt = 0.05
    lookahead_points = 6
    lookahead_spacing_m = 10.0


@dataclass(slots=True)
class TrailerBicycleEnvConfig:
    name: str
    track: TrackConfig
    vehicle: VehicleConfig
    simulation: SimulationConfig
