import json

import jax
import jax.numpy as jnp

from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_phase_vae import (
    MNISTPhaseVAEExperimentConfig,
    phase_vae_loss,
    run_mnist_phase_vae_experiment,
)
from oscnet.models import KuramotoPhaseVAE


def test_kuramoto_phase_vae_reconstructs_and_samples_shapes():
    model = KuramotoPhaseVAE(
        latent_dim=8,
        hidden_dim=16,
        encoder_depth=1,
        decoder_depth=1,
        steps=2,
        key=jax.random.PRNGKey(1),
    )
    images = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)

    reconstruction, latent = model(
        images,
        jax.random.PRNGKey(2),
        return_latent=True,
    )
    samples = model.sample(jax.random.PRNGKey(3), 5)
    trace = model.collect_trace(images, jax.random.PRNGKey(4))

    assert reconstruction.shape == images.shape
    assert samples.shape == (5, 28 * 28)
    assert latent["mu"].shape == (3, 8)
    assert latent["logvar"].shape == (3, 8)
    assert trace["theta_trajectory"].shape == (2, 3, 8)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_kuramoto_phase_vae_supports_relative_readout():
    model = KuramotoPhaseVAE(
        latent_dim=8,
        hidden_dim=16,
        encoder_depth=1,
        decoder_depth=1,
        steps=2,
        phase_readout_mode="mean_relative",
        key=jax.random.PRNGKey(11),
    )
    images = jnp.linspace(0.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)

    reconstruction, latent = model(
        images,
        jax.random.PRNGKey(12),
        return_latent=True,
    )
    readout_theta = model.readout_theta(latent["final_theta"])
    centered = latent["final_theta"] - jnp.mean(
        latent["final_theta"],
        axis=-1,
        keepdims=True,
    )
    expected = jnp.atan2(jnp.sin(centered), jnp.cos(centered))

    assert reconstruction.shape == images.shape
    assert readout_theta.shape == (2, 8)
    assert jnp.allclose(readout_theta, expected)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_phase_vae_loss_backprops_to_model():
    model = KuramotoPhaseVAE(
        latent_dim=6,
        hidden_dim=12,
        encoder_depth=1,
        decoder_depth=1,
        steps=1,
        key=jax.random.PRNGKey(5),
    )
    images = jnp.linspace(0.0, 1.0, 4 * 28 * 28).reshape(4, 28 * 28)

    loss, parts = phase_vae_loss(
        model,
        images,
        jax.random.PRNGKey(6),
        kl_weight=1e-3,
        reconstruction_mode="bce",
    )
    grads = jax.grad(
        lambda current_model: phase_vae_loss(
            current_model,
            images,
            jax.random.PRNGKey(7),
            kl_weight=1e-3,
            reconstruction_mode="bce",
        )[0]
    )(model)

    flat_grads = jax.tree.leaves(grads)
    grad_norm = sum(
        float(jnp.linalg.norm(grad)) for grad in flat_grads if hasattr(grad, "shape")
    )
    assert loss > 0.0
    assert parts["reconstruction_loss"] > 0.0
    assert parts["kl_loss"] >= 0.0
    assert grad_norm > 0.0


def test_mnist_phase_vae_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_phase_vae_test",
        output_dir=tmp_path / "mnist_phase_vae",
        seed=8,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTPhaseVAEExperimentConfig(
        run=run,
        model_family="phase_vae",
        latent_dim=6,
        hidden_dim=12,
        encoder_depth=1,
        decoder_depth=1,
        steps=1,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_phase_vae_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    assert (result.paths.artifacts / "samples_epoch_001.png").exists()
    assert (result.paths.artifacts / "reconstruction_epoch_001.png").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["phase_vae"]["model_family"] == "phase_vae"
    assert summary["phase_vae"]["steps"] == 1
    assert summary["final_eval_loss"] >= 0.0
    assert summary["phase_vae"]["reconstruction_mse"] >= 0.0
