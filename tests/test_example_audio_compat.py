import jax
import jax.numpy as jnp

from examples.audio_wavelet_oscillatory_autoencoder import (
    ProductionWaveletAutoencoder,
    WaveletOscillatorCell,
)
from oscnet.models import WaveletOscillatoryAutoencoder


def test_audio_example_uses_reusable_wavelet_model_spine():
    model = ProductionWaveletAutoencoder(
        input_dim=8,
        hidden_dim=12,
        latent_dim=5,
        omega_bounds=(0.2, 2.0),
        gamma_bounds=(0.01, 0.1),
        key=jax.random.PRNGKey(0),
    )
    sequence = jnp.ones((4, 2, 8))

    latent = model.encode(sequence)
    reconstruction = model(sequence)
    decoded = model.decode(latent, target_length=4)
    decoded_from_zero = model.decode(jnp.zeros_like(latent), target_length=4)
    decoded_from_one = model.decode(jnp.ones_like(latent), target_length=4)

    assert isinstance(model, WaveletOscillatoryAutoencoder)
    assert latent.shape == (2, 5)
    assert reconstruction.shape == sequence.shape
    assert decoded.shape == sequence.shape
    assert not jnp.allclose(decoded_from_zero, decoded_from_one)
    assert model.encoder_cell.hidden_dim == 12
    assert model.decoder_cell.hidden_dim == 12
    assert model.encoder_cell.oscillator.omega.shape == (12,)
    assert jnp.all(jnp.isfinite(reconstruction))

    cell = WaveletOscillatorCell(8, 12, 8, key=jax.random.PRNGKey(1))
    output, state = cell(jnp.ones((2, 8)))

    assert output.shape == (2, 8)
    assert state[0].shape == (2, 12)
