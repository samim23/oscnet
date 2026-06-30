import jax.numpy as jnp

from oscnet.core.coupling import (
    coupling_profile_from_name,
    distance_decay_coupling_profile,
    local_radius_coupling_profile,
    normalize_coupling_profile,
    oscillator_grid_coordinates,
    rectangular_coupling_profile_from_name,
    row_laplacian,
)


def test_oscillator_grid_coordinates_are_near_square():
    coords = oscillator_grid_coordinates(10)

    assert coords.shape == (10, 2)
    assert float(jnp.min(coords)) >= -1.0
    assert float(jnp.max(coords)) <= 1.0


def test_distance_decay_profile_can_preserve_legacy_unnormalized_scale():
    profile = distance_decay_coupling_profile(
        num_oscillators=9,
        length_scale=0.6,
        floor=0.05,
        normalization="none",
    )

    assert profile.shape == (9, 9)
    assert jnp.allclose(jnp.diag(profile), 0.0)
    assert float(jnp.max(profile)) < 1.0
    assert float(jnp.max(profile)) > 0.05


def test_row_sum_normalization_preserves_sparsity_and_sets_gain():
    raw = local_radius_coupling_profile(
        num_oscillators=16,
        radius=0.7,
        normalization="none",
    )
    normalized = normalize_coupling_profile(
        raw,
        mode="row_sum",
        target_row_sum=16.0,
    )

    assert jnp.allclose(jnp.diag(normalized), 0.0)
    assert jnp.array_equal(raw > 0.0, normalized > 0.0)
    assert jnp.allclose(jnp.sum(normalized, axis=-1), 16.0, atol=1e-5)


def test_rectangular_profile_normalizes_target_rows_to_source_count():
    profile = rectangular_coupling_profile_from_name(
        name="local_radius",
        num_targets=16,
        num_sources=4,
        length_scale=1.0,
        normalization="row_sum",
        target_row_sum=4.0,
    )

    assert profile.shape == (16, 4)
    assert jnp.allclose(jnp.sum(profile, axis=-1), 4.0, atol=1e-5)
    assert float(jnp.count_nonzero(profile)) < float(profile.size)


def test_named_normalized_distance_decay_profile_has_expected_row_gain():
    profile = coupling_profile_from_name(
        name="distance_decay",
        num_oscillators=9,
        length_scale=0.6,
        normalization="row_sum",
        target_row_sum=9.0,
    )

    assert profile.shape == (9, 9)
    assert jnp.allclose(jnp.diag(profile), 0.0)
    assert jnp.allclose(jnp.sum(profile, axis=-1), 9.0, atol=1e-5)


def test_row_laplacian_reports_degree_and_zero_row_sum():
    profile = coupling_profile_from_name(
        name="local_radius",
        num_oscillators=9,
        length_scale=1.0,
        normalization="row_sum",
        target_row_sum=9.0,
    )
    laplacian, degree = row_laplacian(profile)

    assert laplacian.shape == (9, 9)
    assert degree.shape == (9,)
    assert jnp.allclose(degree, 9.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(laplacian, axis=-1), 0.0, atol=1e-5)
