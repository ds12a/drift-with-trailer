from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.exp_008_beamng import run_sweep as sweep
from src.learning.models import beamng_model_spec
from src.simulation.beamng_trailer_env import VehicleState


@pytest.mark.parametrize("kind,history", [("model_h1", 1), ("model_h4", 4), ("model", 4)])
def test_learned_controller_uses_checkpoint_history(kind, history):
    training_history = beamng_model_spec.STATE_FS.H
    planner = sweep.make_mpc(80, kind)
    assert planner.history == history
    spec, model, stats = sweep.load_learned_model(kind)
    assert spec.H == history
    assert len(stats["x_mean"]) == history * 10
    assert model(jnp.zeros((1, history * 10))).shape == (1, 6)
    assert beamng_model_spec.STATE_FS.H == training_history


def test_prior_does_not_load_learned_checkpoint(monkeypatch):
    def unexpected_load(kind):
        raise AssertionError("prior should not load a checkpoint")
    monkeypatch.setattr(sweep, "load_learned_model", unexpected_load)
    for kind in ("prior", "prior_surface"):
        assert sweep.make_mpc(-30, kind).history is None


def test_surface_prior_receives_both_measured_coefficients():
    state = VehicleState(*range(10))
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        _state=state, _last_index=0,
        track=SimpleNamespace(_arc_samples=[123]),
        query_surface_friction=lambda: SimpleNamespace(dynamics_mu=(0.2, 0.7)),
    ))
    np.testing.assert_allclose(sweep.surface_prior_state(env), [*range(8), 0.2, 0.7, 123])


def test_cli_accepts_requested_additions(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sweep", "--controllers", "model_h1", "prior_surface"])
    assert sweep.parse_args().controllers == ["model_h1", "prior_surface"]
