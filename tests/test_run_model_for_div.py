import numpy as np

from experiments.exp_008_beamng.run_model_for_div import compute_metrics


def test_metrics_wrap_yaw_and_report_position_error():
    expert = np.zeros((3, 13))
    predicted = expert.copy()
    predicted[1:, 0] = [3.0, 6.0]
    predicted[1:, 1] = [4.0, 8.0]
    predicted[1:, 2] = 2 * np.pi - 0.1
    expert[1:, 2] = -0.1

    metrics = compute_metrics(expert, predicted)

    np.testing.assert_allclose(metrics["position_rmse_m"], np.sqrt((25 + 100) / 2))
    np.testing.assert_allclose(metrics["hitch_rmse_deg"], 0.0, atol=1e-10)
    assert metrics["finite_fraction"] == 1.0


def test_prior_without_friction_ignores_measurements():
    import jax.numpy as jnp
    from experiments.exp_008_beamng.run_model_for_div import DynamicsRunner

    # Make acceleration depend directly on the two friction inputs.
    def dynamics(x, u):
        return jnp.zeros(11).at[4:6].set(x[8:10])

    common = np.zeros((3, 13))
    common[:, 10:12] = [[0.2, 0.4], [0.3, 0.5], [0.6, 0.8]]
    controls = np.zeros((2, 2))
    fixed = DynamicsRunner("fiala", dynamics, None, None, None, 11, -1)
    surface = DynamicsRunner("fiala_surface", dynamics, None, None, None, 11, -1)
    np.testing.assert_allclose(fixed.planner_state(common[0], [])[8:10], [1, 1])
    np.testing.assert_allclose(surface.planner_state(common[0], [])[8:10], [0.2, 0.4])
    np.testing.assert_allclose(fixed.open_loop(common, controls, np.empty((0, 13)), 1)[-1, 4:6], [2, 2])
    np.testing.assert_allclose(surface.open_loop(common, controls, np.empty((0, 13)), 1)[-1, 4:6], [0.5, 0.9])


def test_learned_models_restore_with_independent_history():
    from experiments.exp_008_beamng.run_model_for_div import (
        build_runners, DEFAULT_MODELS, BeamNGTrailerEnvConfig,
        TrackConfig, SimulationConfig, bng_pickup_trailer_cfg,
    )
    from src.learning.models import beamng_model_spec

    training_history = beamng_model_spec.STATE_FS.H
    scenario = BeamNGTrailerEnvConfig(".", TrackConfig(mu=1, width=15),
                                    bng_pickup_trailer_cfg, SimulationConfig(dt=0.05))
    runners, history, config = build_runners(DEFAULT_MODELS, scenario, 80)
    assert history == 4
    assert runners["learned_h1"].history == 1
    assert runners["learned_h4"].history == 4
    assert beamng_model_spec.STATE_FS.H == training_history
    common = np.zeros(13)
    common[4] = 2
    common[10:12] = 1
    rows = [np.zeros(13) for _ in range(4)]
    for name in ("learned_h1", "learned_h4"):
        runner = runners[name]
        state = runner.planner_state(common, rows)
        assert state.shape == (13 * runner.history,)
        result = runner.dynamics(state, np.zeros(2))
        assert result.shape == (13,)
        assert np.isfinite(result).all()


def test_default_batch_runs_both_directions(monkeypatch, tmp_path):
    from argparse import Namespace
    from experiments.exp_008_beamng import run_model_for_div as module
    args = Namespace(steps=1000, samples=1000, horizon=80, target_kph=None,
                     output=tmp_path / "batch", forward_kph=80.0, backward_kph=-30.0)
    calls = []
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "run_direction", calls.append)
    module.main()
    assert [call.target_kph for call in calls] == [80.0, -30.0]
    assert [call.output.name for call in calls] == ["forward", "backward"]
    assert all(call.steps == 1000 for call in calls)
