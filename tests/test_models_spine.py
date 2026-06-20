import jax
import jax.numpy as jnp

from oscnet.models import (
    AutoregressiveOscillatoryDecoder,
    OscillatoryAutoencoder,
    PatchOscillatoryAutoencoder,
    PositionalLatentOscillatoryDecoder,
    WinfreeFieldLayer,
    WinfreePatchAutoencoder,
)


def test_sequence_autoencoder_repeat_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=8,
        hidden_dim=16,
        latent_dim=4,
        sequence_length=5,
        decoder_mode="repeat",
        key=jax.random.PRNGKey(0),
    )
    inputs = jnp.ones((5, 3, 8))

    reconstruction, latent = model(inputs, return_latent=True)

    assert reconstruction.shape == inputs.shape
    assert latent.shape == (3, 4)
    assert jnp.all(jnp.isfinite(reconstruction))
    assert jnp.all(jnp.isfinite(latent))


def test_sequence_autoencoder_autoregressive_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=6,
        hidden_dim=12,
        latent_dim=5,
        sequence_length=4,
        decoder_mode="autoregressive",
        key=jax.random.PRNGKey(1),
    )
    inputs = jnp.ones((4, 2, 6))

    reconstruction = model(inputs)

    assert reconstruction.shape == inputs.shape
    assert jnp.all(jnp.isfinite(reconstruction))
    assert isinstance(model.decoder, AutoregressiveOscillatoryDecoder)


def test_sequence_autoencoder_positional_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=6,
        hidden_dim=12,
        latent_dim=5,
        sequence_length=4,
        decoder_mode="positional",
        latent_conditioning_strength=2.0,
        key=jax.random.PRNGKey(3),
    )
    inputs = jnp.ones((4, 2, 6))

    reconstruction, latent = model(inputs, return_latent=True)

    assert reconstruction.shape == inputs.shape
    assert latent.shape == (2, 5)
    assert isinstance(model.decoder, PositionalLatentOscillatoryDecoder)
    assert model.decoder.positional_inputs.shape == (4, 12)
    assert jnp.all(jnp.isfinite(reconstruction))


def test_patch_autoencoder_maps_flat_images_back_to_flat_images():
    model = PatchOscillatoryAutoencoder(
        hidden_dim=8,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        key=jax.random.PRNGKey(2),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert jnp.all(jnp.isfinite(reconstruction))


def test_winfree_field_layer_records_phase_and_energy_trace():
    layer = WinfreeFieldLayer(
        num_positions=4,
        channels=6,
        steps=3,
        gamma=0.1,
        key=jax.random.PRNGKey(4),
    )
    theta = jnp.zeros((2, 4, 6))
    omega = jnp.ones((2, 4, 6)) * 0.05

    trace = layer(theta, omega, return_trajectory=True)

    assert trace["final_theta"].shape == theta.shape
    assert trace["thetas"].shape == (3, 2, 4, 6)
    assert trace["energies"].shape == (3, 2)
    assert jnp.all(trace["thetas"] <= jnp.pi + 1e-6)
    assert jnp.all(trace["thetas"] >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(trace["energies"]))


def test_winfree_patch_autoencoder_maps_flat_images_back_to_flat_images():
    model = WinfreePatchAutoencoder(
        hidden_dim=6,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        key=jax.random.PRNGKey(5),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 4)
    assert trace["encoder_thetas"].shape[:3] == (2, 2, 4)
    assert trace["decoder_thetas"].shape[:3] == (2, 2, 4)
    assert jnp.all(jnp.isfinite(reconstruction))


def test_winfree_field_layer_supports_learned_grouped_interactions():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=5,
        grid_shape=(4, 4),
        group_size=2,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(6),
    )
    theta = jnp.zeros((3, 16, 5))
    omega = jnp.ones((3, 16, 5)) * 0.05

    trace = layer(theta, omega, return_trajectory=True)

    assert layer.coupling.shape == (4, 4)
    assert trace["final_theta"].shape == theta.shape
    assert trace["thetas"].shape == (2, 3, 16, 5)
    assert trace["energies"].shape == (2, 3)
    assert jnp.all(trace["thetas"] <= jnp.pi + 1e-6)
    assert jnp.all(trace["thetas"] >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(trace["energies"]))
