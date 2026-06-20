import json
from dataclasses import replace

import numpy as np

from oscnet.experiments.audio_wavelet import (
    AudioWaveletExperimentConfig,
    run_audio_wavelet_experiment,
)
from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_autoencoder import (
    MNISTAutoencoderExperimentConfig,
    run_mnist_experiment,
)


def test_mnist_reference_experiment_train_eval_artifacts(tmp_path):
    train_run = AutoencoderExperimentConfig(
        name="mnist_test",
        output_dir=tmp_path / "mnist_train",
        seed=0,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=train_run,
        hidden_dim=4,
        latent_dim=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    output_dir = result.paths.root

    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics" / "history.json").exists()
    assert (output_dir / "metrics" / "history.csv").exists()
    assert (output_dir / "metrics" / "summary.json").exists()
    assert (output_dir / "plots" / "loss_curve.png").exists()
    assert (output_dir / "plots" / "mnist_reconstructions_epoch_001.png").exists()
    assert (output_dir / "traces" / "mnist_latent_state_epoch_001.npz").exists()
    assert (output_dir / "checkpoints" / "best_model.eqx").exists()
    assert (output_dir / "checkpoints" / "best_model_metadata.json").exists()

    with open(output_dir / "metrics" / "history.json") as f:
        history = json.load(f)
    assert history["epoch"] == [1]
    assert len(history["train_loss"]) == 1
    assert len(history["eval_loss"]) == 1

    with open(output_dir / "metrics" / "summary.json") as f:
        summary = json.load(f)
    assert "baselines" in summary
    assert "pixel_mean_mse" in summary["baselines"]
    assert "beats_pixel_mean" in summary
    assert "margin_vs_pixel_mean" in summary
    assert "quality" in summary
    assert "pixel_correlation" in summary["quality"]
    assert "foreground_f1" in summary["quality"]

    trace = np.load(output_dir / "traces" / "mnist_latent_state_epoch_001.npz")
    assert trace["latent"].shape == (2, 2)
    assert trace["encoder_positions"].shape[0] == 49

    eval_run = replace(
        train_run,
        mode="eval",
        output_dir=tmp_path / "mnist_eval",
    )
    eval_config = replace(
        config,
        run=eval_run,
        checkpoint=output_dir / "checkpoints" / "best_model.eqx",
    )
    eval_result = run_mnist_experiment(eval_config)

    assert (eval_result.paths.metrics / "summary.json").exists()
    assert (eval_result.paths.plots / "mnist_reconstructions_epoch_000.png").exists()
    assert (eval_result.paths.traces / "mnist_latent_state_epoch_000.npz").exists()
    with open(eval_result.paths.metrics / "summary.json") as f:
        eval_summary = json.load(f)
    assert "baselines" in eval_summary
    assert "beats_pixel_mean" in eval_summary
    assert "quality" in eval_summary


def test_audio_wavelet_reference_experiment_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="audio_wavelet_test",
        output_dir=tmp_path / "audio_wavelet",
        seed=1,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = AudioWaveletExperimentConfig(
        run=run,
        input_dim=8,
        hidden_dim=6,
        latent_dim=3,
        sequence_length=3,
        feature_source="synthetic-features",
        train_limit=4,
        eval_limit=2,
    )

    result = run_audio_wavelet_experiment(config)
    output_dir = result.paths.root

    assert (output_dir / "config.json").exists()
    assert (output_dir / "metrics" / "history.json").exists()
    assert (output_dir / "metrics" / "summary.json").exists()
    assert (output_dir / "plots" / "loss_curve.png").exists()
    assert (
        output_dir / "plots" / "audio_wavelet_reconstruction_epoch_001.png"
    ).exists()
    assert (
        output_dir / "traces" / "audio_wavelet_latent_state_epoch_001.npz"
    ).exists()
    assert (output_dir / "checkpoints" / "best_model.eqx").exists()
    assert (output_dir / "checkpoints" / "best_model_metadata.json").exists()

    trace = np.load(output_dir / "traces" / "audio_wavelet_latent_state_epoch_001.npz")
    assert trace["latent"].shape == (2, 3)
    assert trace["inputs"].shape == (3, 2, 8)
    assert trace["encoder_positions"].shape[0] == 3
