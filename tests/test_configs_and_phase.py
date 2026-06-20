import jax
import jax.numpy as jnp

from oscnet.models import (
    OscillatoryAutoencoderConfig,
    PatchOscillatoryAutoencoderConfig,
    WaveletAutoencoderConfig,
    WinfreePhaseAutoencoder,
    WinfreePhaseAutoencoderConfig,
    WinfreePhaseOscillatorCell,
)


def test_model_configs_build_expected_shapes():
    sequence = jnp.ones((3, 2, 4))

    sequence_model = OscillatoryAutoencoderConfig(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
        sequence_length=3,
    ).build(jax.random.PRNGKey(0))
    assert sequence_model(sequence).shape == sequence.shape

    patch_model = PatchOscillatoryAutoencoderConfig(
        hidden_dim=8,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
    ).build(jax.random.PRNGKey(1))
    assert patch_model(jnp.ones((2, 64))).shape == (2, 64)

    wavelet_model = WaveletAutoencoderConfig(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
    ).build(jax.random.PRNGKey(2))
    assert wavelet_model(sequence).shape == sequence.shape


def test_winfree_phase_cell_wraps_phase_and_outputs_finite_values():
    cell = WinfreePhaseOscillatorCell(
        input_dim=4,
        hidden_dim=6,
        output_dim=5,
        key=jax.random.PRNGKey(3),
    )
    output, phases = cell(jnp.ones((2, 4)))

    assert output.shape == (2, 5)
    assert phases.shape == (2, 6)
    assert jnp.all(phases <= jnp.pi + 1e-6)
    assert jnp.all(phases >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(output))


def test_winfree_phase_autoencoder_and_config_shapes():
    inputs = jnp.ones((4, 2, 5))
    model = WinfreePhaseAutoencoder(
        input_dim=5,
        hidden_dim=7,
        latent_dim=3,
        sequence_length=4,
        key=jax.random.PRNGKey(4),
    )
    config_model = WinfreePhaseAutoencoderConfig(
        input_dim=5,
        hidden_dim=7,
        latent_dim=3,
        sequence_length=4,
    ).build(jax.random.PRNGKey(5))

    assert model(inputs).shape == inputs.shape
    assert config_model(inputs).shape == inputs.shape
    assert jnp.all(jnp.isfinite(model(inputs)))
