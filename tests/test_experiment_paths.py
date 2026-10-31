import hashlib
from pathlib import Path

import pytest

from experiments.exp_008_beamng import experiment_paths as paths


def test_historical_recording_resolves_into_consolidated_data(tmp_path, monkeypatch):
    project, data = tmp_path / 'project', tmp_path / 'preserved'
    recording = data / 'both4_div/forward/expert_learned_h4.npz'
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b'recorded states')
    monkeypatch.setattr(paths, 'PROJECT_ROOT', project)
    monkeypatch.setattr(paths, 'DATA_ROOT', data)
    monkeypatch.setattr(paths, 'MODEL_BUNDLE', data / 'history_friction_ablation_20261029')
    historical = Path('/home/previous-machine/drift-with-trailer/both4_div/forward/expert_learned_h4.npz')
    assert paths.resolve_saved_path(historical) == recording


def test_terrain_reproduction_does_not_require_old_runtime(tmp_path, monkeypatch):
    level = tmp_path / 'preserved-level'
    level.mkdir()
    terrain = level / 'terrain8.ter'
    terrain.write_bytes(b'exact recorded terrain')
    monkeypatch.setattr(paths, 'TERRAIN_LEVEL', level)
    protocol = dict(staged_level=str(tmp_path / 'deleted-runtime'),
                    terrain_sha256=hashlib.sha256(terrain.read_bytes()).hexdigest())
    assert paths.terrain_file(protocol) == terrain


def test_different_terrain_cannot_silently_replace_recorded_environment(tmp_path, monkeypatch):
    preserved, custom = tmp_path / 'preserved', tmp_path / 'custom'
    preserved.mkdir()
    custom.mkdir()
    (preserved / 'terrain8.ter').write_bytes(b'snow+slope')
    terrain = custom / 'terrain8.ter'
    terrain.write_bytes(b'different experiment')
    monkeypatch.setattr(paths, 'TERRAIN_LEVEL', preserved)
    protocol = dict(staged_level=str(custom), terrain_sha256=hashlib.sha256(terrain.read_bytes()).hexdigest())
    assert paths.terrain_file(protocol) == terrain
    terrain.unlink()
    with pytest.raises(FileNotFoundError, match='SHA-256'):
        paths.terrain_file(protocol)
