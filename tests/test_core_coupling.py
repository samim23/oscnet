import jax.numpy as jnp

from oscnet.core.coupling import (
    coupling_profile_from_name,
    distance_decay_coupling_profile,
    hierarchical_coupling_profile,
    local_radius_coupling_profile,
    normalize_coupling_profile,
    oscillator_grid_coordinates,
    rectangular_coupling_profile_from_name,
    row_laplacian,
)
from oscnet.core.layered import (
    OscillatorLayerSpec,
    adjacent_inter_layer_specs,
    inter_layer_profile,
    intra_layer_profile,
    validate_layer_specs,
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


def test_fractal_profile_is_nonlocal_with_discrete_scales():
    fractal = hierarchical_coupling_profile(
        num_oscillators=64,
        inter_block_strength=0.5,
        normalization="none",
    )
    local = local_radius_coupling_profile(
        num_oscillators=64,
        radius=0.0,
        normalization="none",
    )

    assert fractal.shape == (64, 64)
    assert jnp.allclose(jnp.diag(fractal), 0.0)
    # Symmetric ultrametric kernel.
    assert jnp.allclose(fractal, fractal.T, atol=1e-6)
    # Unlike the sparse local profile, the fractal profile keeps direct
    # long-range links: the far corners of the grid remain coupled.
    assert float(fractal[0, -1]) > 0.0
    assert float(local[0, -1]) == 0.0
    # Discrete self-similar scales: only a handful of distinct nonzero
    # coupling magnitudes rather than a smooth continuum.
    magnitudes = jnp.unique(jnp.round(fractal[fractal > 0.0], 5))
    assert int(magnitudes.shape[0]) <= 8


def test_named_fractal_profile_matches_row_gain_and_uses_strength():
    profile = coupling_profile_from_name(
        name="fractal",
        num_oscillators=64,
        length_scale=0.5,
        normalization="row_sum",
        target_row_sum=64.0,
    )

    assert profile.shape == (64, 64)
    assert jnp.allclose(jnp.diag(profile), 0.0)
    assert jnp.allclose(jnp.sum(profile, axis=-1), 64.0, atol=1e-4)
    # Every site reaches every other after row-sum normalization (dense
    # support), in contrast to the sparse local profile.
    assert float(jnp.count_nonzero(profile)) == float(64 * 64 - 64)


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


def test_layered_oscillator_specs_build_horizontal_and_vertical_profiles():
    layers = validate_layer_specs(
        (
            OscillatorLayerSpec(
                name="coarse",
                num_oscillators=4,
                frequency_scale=0.5,
                coupling_profile="dense",
            ),
            OscillatorLayerSpec(
                name="fine",
                num_oscillators=16,
                frequency_scale=1.0,
                coupling_profile="local_radius",
                coupling_length_scale=0.7,
            ),
        )
    )
    vertical_specs = adjacent_inter_layer_specs(
        num_layers=len(layers),
        forward_strength=0.25,
        feedback_strength=0.05,
        profile="distance_decay",
        length_scale=0.8,
    )

    coarse_profile = intra_layer_profile(layers[0])
    fine_profile = intra_layer_profile(layers[1])
    down_profile = inter_layer_profile(vertical_specs[0], layers)
    up_profile = inter_layer_profile(vertical_specs[1], layers)

    assert coarse_profile.shape == (4, 4)
    assert fine_profile.shape == (16, 16)
    assert down_profile.shape == (16, 4)
    assert up_profile.shape == (4, 16)
    assert jnp.allclose(jnp.sum(down_profile, axis=-1), 4.0, atol=1e-5)
    assert jnp.allclose(jnp.sum(up_profile, axis=-1), 16.0, atol=1e-5)
    assert vertical_specs[0].source_layer == 0
    assert vertical_specs[0].target_layer == 1
    assert vertical_specs[1].source_layer == 1
    assert vertical_specs[1].target_layer == 0
