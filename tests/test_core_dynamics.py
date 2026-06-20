import jax.numpy as jnp

from oscnet.core import (
    create_coupling_matrix,
    create_hierarchical_coupling,
    create_log_periodic_coupling,
    create_power_law_coupling,
    harmonic_oscillator_update,
    nonlinear_harmonic_oscillator_update,
    van_der_pol_update,
)


def test_harmonic_update_matches_symplectic_euler_equations():
    x = jnp.array([1.0, -1.0])
    v = jnp.array([0.5, 0.25])
    inputs = jnp.array([0.1, -0.2])
    omega_squared = jnp.array([4.0, 9.0])
    gamma_factor = jnp.array([0.2, 0.4])
    dt = 0.1

    x_new, v_new = harmonic_oscillator_update(
        x, v, inputs, omega_squared, gamma_factor, dt
    )

    expected_x = x + dt * v
    expected_v = v + dt * (inputs - omega_squared * expected_x - gamma_factor * v)
    assert jnp.allclose(x_new, expected_x)
    assert jnp.allclose(v_new, expected_v)


def test_nonlinear_harmonic_update_uses_tanh_forcing():
    x = jnp.zeros(2)
    v = jnp.zeros(2)
    inputs = jnp.array([0.0, 2.0])

    _, v_new = nonlinear_harmonic_oscillator_update(
        x,
        v,
        inputs,
        alpha=0.5,
        omega_squared=jnp.ones(2),
        gamma_factor=jnp.zeros(2),
        dt=0.2,
    )

    assert jnp.allclose(v_new, 0.2 * 0.5 * jnp.tanh(inputs))


def test_van_der_pol_update_shape_and_finiteness():
    x_new, v_new = van_der_pol_update(
        jnp.ones(4),
        jnp.zeros(4),
        jnp.ones(4) * 0.1,
        mu=1.0,
        dt=0.05,
    )

    assert x_new.shape == (4,)
    assert v_new.shape == (4,)
    assert jnp.all(jnp.isfinite(x_new))
    assert jnp.all(jnp.isfinite(v_new))


def test_coupling_matrices_are_finite_normalized_and_dispatchable():
    hierarchical = create_hierarchical_coupling(8, depth=2)
    power_law = create_power_law_coupling(8, exponent=-1.5)
    log_periodic = create_log_periodic_coupling(8, period=2.0)
    dispatched = create_coupling_matrix(8, coupling_type="hierarchical", depth=2)

    for matrix in (hierarchical, power_law, log_periodic, dispatched):
        assert matrix.shape == (8, 8)
        assert jnp.all(jnp.isfinite(matrix))
        assert jnp.max(jnp.abs(matrix)) <= 1.0 + 1e-6

    assert jnp.allclose(hierarchical, dispatched)
    assert jnp.allclose(hierarchical, hierarchical.T)
    assert jnp.all(power_law > 0)
