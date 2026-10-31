"""Locate consolidated inputs without depending on old simulator user folders."""
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (PROJECT_ROOT if (PROJECT_ROOT / 'bundled_cases.json').is_file()
             else PROJECT_ROOT / 'experiment_data_20261031')
MODEL_BUNDLE = DATA_ROOT / 'history_friction_ablation_20261029'
LEVELS_DIR = DATA_ROOT / 'assets/levels'
TERRAIN_LEVEL = LEVELS_DIR / 'tech_ground_final/tech_ground'
VEHICLE_CONFIG = DATA_ROOT / 'assets/vehicles/pickup/hitch.pc'
TRACK_CSV = DATA_ROOT / 'src/simulation/assets/tracks/ks_barcelona_layout_gp_centerline.csv'
TURN_SNOW = DATA_ROOT / 'turn_snow_div'
RENDERED_BACKGROUNDS = DATA_ROOT / 'experiments/exp_008_beamng/rendered_backgrounds'


def resolve_saved_path(value):
    """Resolve an original repository path against the preserved data copies.

    Saved manifests keep their historical paths for provenance. Resolve the
    same relative file, without substituting a different checkpoint or dataset.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            parts = path.parts
            if 'drift-with-trailer' not in parts:
                if path.exists():
                    return path
                raise FileNotFoundError(path)
            relative = Path(*parts[parts.index('drift-with-trailer') + 1:])
    else:
        relative = path
    # Prefer the canonical data over leftover original runtime directories.
    for candidate in (DATA_ROOT / relative, MODEL_BUNDLE / relative, path,
                      PROJECT_ROOT / relative):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Preserved input not found for {value}')


def terrain_file(protocol):
    """The verified local terrain replaces the old staged-level runtime path."""
    original = Path(protocol['staged_level']) / 'terrain8.ter'
    preserved = TERRAIN_LEVEL / 'terrain8.ter'
    expected = protocol.get('terrain_sha256')
    candidates = (preserved, original) if expected else (original, preserved)
    for candidate in candidates:
        if candidate.is_file() and (not expected or hashlib.sha256(candidate.read_bytes()).hexdigest() == expected):
            return candidate
    raise FileNotFoundError(f'No terrain matching recorded SHA-256 {expected}: {preserved}, {original}')
