from types import SimpleNamespace

import jax.numpy as jnp
import jax
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


@pytest.mark.parametrize("kind,history", [("model_h1", 1), ("model_h4", 4), ("prior_surface", None)])
def test_episode_restarts_sampling_from_requested_seed(monkeypatch, kind, history):
    monkeypatch.setattr(sweep, "MAX_STEPS", 1)
    monkeypatch.setattr(sweep, "progress", lambda *args: None)
    monkeypatch.setattr(sweep, "clear_progress", lambda: None)
    captured_keys = []
    planner = SimpleNamespace(history=history, key=jax.random.key(99), reset=lambda: None)
    def control(state):
        captured_keys.append(np.asarray(jax.random.key_data(planner.key)))
        return jnp.zeros(2)
    planner.run_mpc = control
    track = SimpleNamespace(
        length=100.0, _arc_samples=np.array([0.0]),
        project=lambda *args: (SimpleNamespace(arc_length=0.0, lateral_error=0.0), 0),
        out_of_bounds=lambda error: False,
    )
    env = SimpleNamespace(
        track=track, _state=VehicleState(*([0.0] * 10)), _last_index=0,
        config=SimpleNamespace(vehicle=SimpleNamespace(max_hitch=1.4)),
        reset=lambda: None, step=lambda action: None,
        query_surface_friction=lambda: SimpleNamespace(dynamics_mu=(1.0, 1.0)),
    )
    env.unwrapped = env
    sweep.run_episode(env, planner, -30, 7, kind, 0)
    np.testing.assert_array_equal(captured_keys[0], jax.random.key_data(jax.random.key(7)))


@pytest.mark.parametrize("kind", ["ipopt", "lqr-pid", "mpcc"])
@pytest.mark.parametrize("velocity", [80, -30])
def test_baseline_factory_uses_existing_controller_settings(monkeypatch, kind, velocity):
    calls, compiled = [], []
    controller = SimpleNamespace(compile=lambda: compiled.append(True))

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return controller

    monkeypatch.setattr(sweep, "make_ipopt_mpc", build)
    monkeypatch.setattr(sweep, "make_lqr_pid", build)
    monkeypatch.setattr(sweep, "MPCC", build)
    monkeypatch.setattr(sweep, "load_learned_model", lambda _: pytest.fail("Baseline loaded a learned model"))
    assert sweep.make_mpc(velocity, kind) is controller
    args, kwargs = calls[0]
    assert args[0] is sweep.scenario
    assert args[1] == velocity / 3.6
    if kind == "ipopt":
        assert args[2] == sweep.IPOPT_HORIZON
        assert args[3]['a_weight'] == (sweep.FWD_WEIGHTS if velocity > 0 else sweep.REV_WEIGHTS)['a_weight']
    elif kind == "lqr-pid":
        assert args[2] is sweep.LQR_PID_SETTINGS
    else:
        assert kwargs['horizon'] == sweep.MPCC_HORIZON
        assert kwargs['weights'].hitch == (sweep.FWD_MPCC_WEIGHTS if velocity > 0 else sweep.REV_MPCC_WEIGHTS)['hitch']
    assert compiled == ([True] if kind == "mpcc" else [])


def test_default_controllers_are_also_accepted_explicitly(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_sweep"])
    assert sweep.parse_args().controllers == sweep.CONTROLLERS
    monkeypatch.setattr("sys.argv", ["run_sweep", "--controllers", *sweep.CONTROLLERS])
    assert sweep.parse_args().controllers == sweep.CONTROLLERS


@pytest.mark.parametrize("kind", ["ipopt", "lqr-pid", "mpcc"])
@pytest.mark.parametrize("solver_fails", [False, True])
def test_baseline_episode_handles_numpy_controls_and_solver_failure(monkeypatch, kind, solver_fails):
    monkeypatch.setattr(sweep, "MAX_STEPS", 1)
    monkeypatch.setattr(sweep, "progress", lambda *args: None)
    monkeypatch.setattr(sweep, "clear_progress", lambda: None)
    calls, resets, initializations, actions = [], [], [], []

    def control(state):
        calls.append(np.asarray(state))
        if solver_fails:
            raise RuntimeError("no finite solver iterate")
        return np.array([0.2, 0.3])

    planner = SimpleNamespace(
        reset=lambda **kwargs: resets.append(kwargs), run_mpc=control,
        last_trajectory=object(), lt_lam_g=object(), n=2, u_sym=object(),
        opti=SimpleNamespace(set_initial=lambda *args: initializations.append(args)),
    )
    # IPOPT's adapter may recover a finite iterate; dispatch through it rather
    # than calling the generic run_mpc interface or waiting on a NumPy array.
    monkeypatch.setattr(sweep, "solve_or_use_iterate", lambda mpc, state: (control(state), True))
    track = SimpleNamespace(
        length=100.0, _arc_samples=np.array([0.0]), find_mu=lambda *args: 1.0,
        project=lambda *args: (SimpleNamespace(arc_length=0.0, lateral_error=0.0), 0),
        out_of_bounds=lambda error: False,
    )
    env = SimpleNamespace(
        track=track, _state=VehicleState(*([0.0] * 10)), _last_index=0,
        config=SimpleNamespace(vehicle=SimpleNamespace(max_hitch=1.4)),
        reset=lambda: None, step=lambda action: actions.append(np.asarray(action)),
    )
    env.unwrapped = env
    result = sweep.run_episode(env, planner, -30, 7, kind, 0)
    assert calls[0].shape == (10,)
    assert result['outcome'] == ('solver_failed' if solver_fails else 'completed')
    assert result['iters'] == (0 if solver_fails else 1)
    assert result['solve_ms'].shape == (1,)
    assert len(actions) == 1 + sweep.WARMUP_STEPS + result['iters']
    if not solver_fails:
        np.testing.assert_allclose(actions[-1], [-0.2, 0.3])
    if kind == 'ipopt':
        assert planner.last_trajectory is None and planner.lt_lam_g is None
        assert len(initializations) == 1
    elif kind == 'mpcc':
        np.testing.assert_allclose(resets[-1]['previous_vehicle_control'], [0.0, -0.35])
