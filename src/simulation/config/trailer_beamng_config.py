from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class TrackConfig:
    csv: str = "src/simulation/assets/tracks/ks_barcelona_layout_gp_centerline.csv"
    width: float = 8.0
    friction_csv: str = None
    closed: bool = True
    mu: float = 1.0 # Not used, do not use do not do not


@dataclass(slots=True)
class SimulationConfig:
    dt = 0.02
    lookahead_points = 6
    lookahead_spacing_m = 10.0


@dataclass(slots=True)
class VehicleConfig:
    wheelbase: float = 3.05
    lf: float = 1.45
    lr: float = 1.6
    mass: float = 2400.0
    inertia_z: float = 6500.0
    cornering_stiffness_front: float = 90000.0
    cornering_stiffness_rear: float = 98000.0
    max_steer_rad: float = 0.32
    max_accel: float = 12.0
    max_brake: float = 18.0
    drag_coefficient: float = 0.85
    wheel_radius: float = 0.33
    
    # Use default_factory for mutable structures (like lists or dicts)
    chassis_size: list[float] = field(default_factory=lambda: [3.2, 1.4, 0.32])
    gamma: int = 1

    # Trailer
    trailer_mass: float = 1225.0
    trailer_inertia_z: float = 850.0
    l2f: float = 2.05
    l2r: float = 0.4
    cornering_stiffness_trailer: float = 80000.0
    hitch_offset: float = 2.3  # tractor CG to hitch (positive behind)
    max_hitch: float = np.deg2rad(80)

@dataclass(slots=True)
class BeamNGTrailerEnvConfig:
    name: str = "."
    track: TrackConfig = field(default_factory=TrackConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
