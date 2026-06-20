import numpy as np
import jax
import jax.numpy as jnp

from examples.image_mnist_oscillatory_autoencoder import (
    AmplitudeVelocityAutoencoder,
    export_encoder_complex_states,
)


def test_mnist_example_uses_reusable_model_spine(tmp_path):
    model = AmplitudeVelocityAutoencoder(
        hidden_dim=8,
        latent_dim=4,
        key=jax.random.PRNGKey(0),
    )
    images = jnp.ones((2, 784))

    reconstruction = model(images)

    assert reconstruction.shape == images.shape
    assert model.encoder.rnn.cell.hidden_dim == 8
    assert model.decoder.rnn.cell.hidden_dim == 8
    assert jnp.all(jnp.isfinite(reconstruction))

    export_path = tmp_path / "encoder_states.npz"
    export_encoder_complex_states(model, images, str(export_path))
    exported = np.load(export_path)

    assert exported["amplitude"].shape == (2, 8)
    assert exported["phase"].shape == (2, 8)
