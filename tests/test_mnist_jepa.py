import json

import jax
import jax.numpy as jnp
import numpy as np

from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_jepa import (
    MNISTJEPAExperimentConfig,
    compute_hidden_patch_weights,
    compute_patch_embeddings,
    dct_lowfreq_basis,
    run_mnist_jepa_experiment,
)
from oscnet.models import (
    FeedForwardPatchJEPAPredictor,
    WinfreeGlobalRatePhasePatchJEPAPredictor,
)


def test_dct_lowfreq_patch_embeddings_and_hidden_weights():
    images = jnp.linspace(0.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)
    visibility = jnp.ones_like(images)
    visibility = visibility.at[:, : 14 * 14].set(0.0)

    basis = dct_lowfreq_basis((7, 7), 4)
    embeddings = compute_patch_embeddings(
        images,
        patch_shape=(7, 7),
        embedding_dim=4,
    )
    weights = compute_hidden_patch_weights(
        visibility,
        patch_shape=(7, 7),
        embedding_dim=4,
    )

    assert basis.shape == (49, 4)
    np.testing.assert_allclose(np.asarray(basis.T @ basis), np.eye(4), atol=1e-5)
    assert embeddings.shape == (2, 16 * 4)
    assert weights.shape == embeddings.shape
    assert float(weights.sum()) > 0.0
    assert float(weights.sum()) < float(weights.size)


def test_mnist_jepa_predictors_accept_image_plus_mask_inputs():
    inputs = jnp.ones((2, 2 * 64))
    model_specs = [
        (
            FeedForwardPatchJEPAPredictor,
            {
                "latent_dim": 4,
            },
        ),
        (
            WinfreeGlobalRatePhasePatchJEPAPredictor,
            {
                "steps": 2,
            },
        ),
    ]

    for index, (model_cls, kwargs) in enumerate(model_specs):
        model = model_cls(
            input_dim=32,
            hidden_dim=6,
            embedding_dim=5,
            image_shape=(8, 8),
            patch_shape=(4, 4),
            key=jax.random.PRNGKey(20 + index),
            **kwargs,
        )
        prediction = model(inputs)
        trace = model.collect_trace(inputs)

        assert prediction.shape == (2, 4 * 5)
        assert trace["prediction_sequence"].shape == (4, 2, 5)
        assert jnp.all(jnp.isfinite(prediction))


def test_mnist_jepa_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_jepa_test",
        output_dir=tmp_path / "mnist_jepa",
        seed=3,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTJEPAExperimentConfig(
        run=run,
        model_family="winfree_global_rate_phase",
        hidden_dim=4,
        latent_dim=4,
        embedding_dim=4,
        patch_shape=(7, 7),
        winfree_steps=1,
        corruption_mode="block_occlusion",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_jepa_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    assert (result.paths.artifacts / "mnist_jepa_predictions_epoch_001.npz").exists()
    assert (result.paths.traces / "mnist_jepa_trace_epoch_001.npz").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert "jepa" in summary
    assert "zero_embedding_mse" in summary["jepa"]
    assert summary["final_eval_loss"] >= 0.0
