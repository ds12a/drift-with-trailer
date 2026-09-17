import jax.numpy as jnp
import numpy as np

from src.dynamics.trailer.trailer_bicycle_fiala_surface import gen_util_funs
from src.simulation.config.trailer_beamng_config import BeamNGTrailerEnvConfig


def test_surface_fiala_has_independent_constant_friction_states():
    dynamics, *_ = gen_util_funs(BeamNGTrailerEnvConfig())
    state = jnp.array(
        [0.0, 0.0, 0.0, 0.05, 8.0, 1.0, 0.1, 0.15, 0.9, 0.2, 0.0]
    )

    derivative = np.asarray(dynamics(state, jnp.array([0.1, 0.4])))

    assert derivative.shape == (11,)
    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(derivative[8:10], np.zeros(2))


def test_each_friction_input_changes_its_associated_dynamics():
    dynamics, *_ = gen_util_funs(BeamNGTrailerEnvConfig())
    state = jnp.array(
        [0.0, 0.0, 0.0, 0.1, 8.0, 1.0, 0.1, 0.2, 1.0, 1.0, 0.0]
    )
    action = jnp.array([0.2, 0.8])

    nominal = np.asarray(dynamics(state, action))
    low_tractor = np.asarray(dynamics(state.at[8].set(0.2), action))
    low_trailer = np.asarray(dynamics(state.at[9].set(0.2), action))

    assert not np.allclose(nominal[4:8], low_tractor[4:8])
    assert not np.allclose(nominal[4:8], low_trailer[4:8])
