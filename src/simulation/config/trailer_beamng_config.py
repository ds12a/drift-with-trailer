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

class VehicleConfig:
    max_hitch: float = np.deg2rad(80)

@dataclass(slots=True)
class BeamNGTrailerEnvConfig:
    name: str = "."
    track: TrackConfig = field(default_factory=TrackConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
