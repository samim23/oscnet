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
    compute_corruption_loss_weights,
    corrupt_mnist_images,
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

    resume_run = replace(
        train_run,
        output_dir=tmp_path / "mnist_resume",
        epochs=1,
    )
    resume_config = replace(
        config,
        run=resume_run,
        checkpoint=output_dir / "checkpoints" / "checkpoint_epoch_001.eqx",
    )
    resume_result = run_mnist_experiment(resume_config)

    with open(resume_result.paths.metrics / "history.json") as f:
        resume_history = json.load(f)
    with open(resume_result.paths.metrics / "summary.json") as f:
        resume_summary = json.load(f)
    assert resume_history["epoch"] == [2]
    assert resume_summary["start_epoch"] == 1
    assert resume_summary["final_epoch"] == 2
    assert resume_summary["resume_from_checkpoint"] == str(
        output_dir / "checkpoints" / "checkpoint_epoch_001.eqx"
    )
    assert (resume_result.paths.checkpoints / "checkpoint_epoch_002.eqx").exists()


def test_mnist_reference_experiment_supports_latent_variance_loss(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_latent_variance_test",
        output_dir=tmp_path / "mnist_latent_variance",
        seed=3,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        latent_variance_weight=0.01,
        latent_std_floor=0.5,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["experiment"]["latent_variance_weight"] == 0.01
    assert saved_config["experiment"]["latent_std_floor"] == 0.5
    assert result.metrics["epoch"] == [1]
    assert np.isfinite(result.metrics["train_loss"][0])


def test_mnist_winfree_field_experiment_train_eval_artifacts(tmp_path):
    train_run = AutoencoderExperimentConfig(
        name="mnist_winfree_test",
        output_dir=tmp_path / "mnist_winfree_train",
        seed=2,
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
        patch_shape=(7, 7),
        model_family="winfree_field",
        winfree_steps=2,
        winfree_si_func="mlp",
        winfree_group_size=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    output_dir = result.paths.root

    assert (output_dir / "metrics" / "summary.json").exists()
    assert (output_dir / "plots" / "mnist_reconstructions_epoch_001.png").exists()
    assert (output_dir / "traces" / "mnist_latent_state_epoch_001.npz").exists()
    assert (output_dir / "checkpoints" / "best_model.eqx").exists()

    trace = np.load(output_dir / "traces" / "mnist_latent_state_epoch_001.npz")
    assert trace["latent"].shape == (2, 2)
    assert trace["encoder_thetas"].shape[:3] == (2, 2, 16)
    assert trace["encoder_energies"].shape == (2, 2)
    assert trace["decoder_thetas"].shape[:3] == (2, 2, 16)

    eval_run = replace(
        train_run,
        mode="eval",
        output_dir=tmp_path / "mnist_winfree_eval",
    )
    eval_config = replace(
        config,
        run=eval_run,
        checkpoint=output_dir / "checkpoints" / "best_model.eqx",
    )
    eval_result = run_mnist_experiment(eval_config)

    assert (eval_result.paths.metrics / "summary.json").exists()
    assert (eval_result.paths.traces / "mnist_latent_state_epoch_000.npz").exists()
    with open(eval_result.paths.metrics / "summary.json") as f:
        eval_summary = json.load(f)
    assert "baselines" in eval_summary
    assert "quality" in eval_summary


def test_mnist_feedforward_patch_experiment_train_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_feedforward_test",
        output_dir=tmp_path / "mnist_feedforward_train",
        seed=4,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="feedforward_patch",
        feedforward_output_activation="sigmoid",
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    output_dir = result.paths.root

    assert (output_dir / "metrics" / "summary.json").exists()
    assert (output_dir / "plots" / "mnist_reconstructions_epoch_001.png").exists()
    assert (output_dir / "traces" / "mnist_latent_state_epoch_001.npz").exists()
    assert (output_dir / "checkpoints" / "best_model.eqx").exists()

    trace = np.load(output_dir / "traces" / "mnist_latent_state_epoch_001.npz")
    assert trace["latent"].shape == (2, 2)
    assert trace["encoder_hidden"].shape == (2, 16, 4)
    assert trace["decoder_hidden"].shape == (2, 16, 4)

    with open(output_dir / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["model_family"] == "feedforward_patch"
    assert saved_config["task"]["feedforward_output_activation"] == "sigmoid"


def test_mnist_masked_reconstruction_uses_clean_targets(tmp_path):
    images = np.ones((2, 28 * 28), dtype=np.float32)
    masked = corrupt_mnist_images(
        images,
        mode="patch_mask",
        patch_shape=(7, 7),
        fraction=0.5,
        seed=0,
    )
    assert np.asarray(masked).shape == images.shape
    assert np.any(np.asarray(masked) != images)
    weights = compute_corruption_loss_weights(
        masked,
        images,
        visible_weight=0.25,
        changed_weight=2.0,
    )
    changed = np.asarray(masked) != images
    assert np.allclose(np.asarray(weights)[changed], 2.0)
    assert np.allclose(np.asarray(weights)[~changed], 0.25)

    run = AutoencoderExperimentConfig(
        name="mnist_masked_test",
        output_dir=tmp_path / "mnist_masked_train",
        seed=5,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="feedforward_patch",
        feedforward_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        corruption_visible_loss_weight=0.25,
        corruption_changed_loss_weight=2.0,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    artifact = np.load(
        result.paths.artifacts / "mnist_reconstructions_epoch_001.npz"
    )

    assert "inputs" in artifact.files
    assert artifact["inputs"].shape == artifact["originals"].shape
    assert np.any(artifact["inputs"] != artifact["originals"])

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["baselines"]["pixel_mean_mse"] >= 0.0
    assert "quality" in summary
    assert "changed_mse" in summary["quality"]
    assert "unchanged_mse" in summary["quality"]
    assert summary["quality"]["changed_fraction"] > 0.0

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["corruption_visible_loss_weight"] == 0.25
    assert saved_config["task"]["corruption_changed_loss_weight"] == 2.0


def test_mnist_recurrent_conv_masked_reconstruction_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_recurrent_conv_masked_test",
        output_dir=tmp_path / "mnist_recurrent_conv_masked_train",
        seed=7,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="recurrent_conv",
        recurrent_conv_steps=2,
        recurrent_conv_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 4)
    assert trace["initial_hidden"].shape == (2, 16, 4)
    assert trace["hidden_states"].shape[:3] == (2, 2, 16)
    assert (result.paths.checkpoints / "best_model.eqx").exists()


def test_mnist_conv_lstm_masked_reconstruction_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_conv_lstm_masked_test",
        output_dir=tmp_path / "mnist_conv_lstm_masked_train",
        seed=8,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="conv_lstm",
        conv_lstm_steps=2,
        conv_lstm_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 4)
    assert trace["drive"].shape == (2, 16, 4)
    assert trace["hidden_states"].shape[:3] == (2, 2, 16)
    assert trace["cell_states"].shape[:3] == (2, 2, 16)
    assert (result.paths.checkpoints / "best_model.eqx").exists()


def test_mnist_image_plus_mask_input_reconstructs_clean_targets(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_image_plus_mask_test",
        output_dir=tmp_path / "mnist_image_plus_mask_train",
        seed=13,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="winfree_rate_phase",
        winfree_steps=1,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_visibility_gate="visibility",
        winfree_visibility_drive_floor=0.0,
        winfree_missing_transport_strength=1.0,
        winfree_output_activation="sigmoid",
        corruption_mode="block_occlusion",
        corruption_fraction=0.5,
        corruption_input_mode="image_plus_mask",
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    artifact = np.load(
        result.paths.artifacts / "mnist_reconstructions_epoch_001.npz"
    )
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert artifact["inputs"].shape == (2, 28, 28)
    assert artifact["originals"].shape == (2, 28, 28)
    assert artifact["reconstructions"].shape == (2, 28, 28)
    assert trace["omega"].shape == (2, 16, 4)
    assert trace["visibility"].shape == (2, 16, 1)
    assert trace["reconstruction_sequence"].shape == (16, 2, 49)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["quality"]["changed_fraction"] > 0.0

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["corruption_input_mode"] == "image_plus_mask"
    assert saved_config["task"]["winfree_visibility_gate"] == "visibility"


def test_mnist_boundary_clamped_protocol_uses_hidden_region_loss(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_boundary_clamped_test",
        output_dir=tmp_path / "mnist_boundary_clamped_train",
        seed=14,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="feedforward_patch",
        feedforward_output_activation="sigmoid",
        corruption_mode="block_occlusion",
        corruption_fraction=0.5,
        corruption_input_mode="image_plus_mask",
        corruption_protocol="boundary_clamped",
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)

    assert saved_config["task"]["corruption_protocol"] == "boundary_clamped"
    assert summary["loss_protocol"] == "boundary_clamped"
    assert summary["primary_metric"] == "hidden_region_mse"
    assert summary["full_image_metrics_are_secondary"] is True
    assert "beats_pixel_mean" not in summary
    assert summary["quality"]["changed_fraction"] > 0.0
    assert summary["quality"]["unchanged_mse"] < 1e-8
    assert np.isclose(
        summary["final_eval_loss"],
        summary["quality"]["changed_mse"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert "raw_reconstruction" in trace.files
    assert "clamped_reconstruction" in trace.files


def test_mnist_winfree_conditional_masked_reconstruction_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_conditional_masked_test",
        output_dir=tmp_path / "mnist_conditional_masked_train",
        seed=6,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="winfree_conditional",
        winfree_steps=2,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_phase_init="rotary_2d",
        winfree_phase_init_scale=0.75,
        winfree_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 8)
    assert trace["omega"].shape == (2, 16, 4)
    assert trace["thetas"].shape[:3] == (2, 2, 16)
    assert (result.paths.checkpoints / "best_model.eqx").exists()

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["winfree_phase_init"] == "rotary_2d"
    assert saved_config["task"]["winfree_phase_init_scale"] == 0.75


def test_mnist_winfree_rate_phase_masked_reconstruction_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_rate_phase_masked_test",
        output_dir=tmp_path / "mnist_rate_phase_masked_train",
        seed=7,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="winfree_rate_phase",
        winfree_steps=2,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_rate_update_rate=0.25,
        winfree_rate_gate_strength=0.75,
        winfree_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 12)
    assert trace["omega"].shape == (2, 16, 4)
    assert trace["thetas"].shape[:3] == (2, 2, 16)
    assert trace["rate_states"].shape[:3] == (2, 2, 16)
    assert (result.paths.checkpoints / "best_model.eqx").exists()

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["model_family"] == "winfree_rate_phase"
    assert saved_config["task"]["winfree_rate_update_rate"] == 0.25
    assert saved_config["task"]["winfree_rate_gate_strength"] == 0.75


def test_mnist_winfree_global_rate_phase_masked_reconstruction_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_global_rate_phase_masked_test",
        output_dir=tmp_path / "mnist_global_rate_phase_masked_train",
        seed=8,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(7, 7),
        model_family="winfree_global_rate_phase",
        winfree_steps=2,
        winfree_global_gamma=0.03,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_rate_update_rate=0.25,
        winfree_rate_gate_strength=0.75,
        winfree_global_gate_strength=0.25,
        winfree_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 20)
    assert trace["omega"].shape == (2, 16, 4)
    assert trace["global_omega"].shape == (2, 1, 4)
    assert trace["thetas"].shape[:3] == (2, 2, 16)
    assert trace["global_thetas"].shape[:3] == (2, 2, 1)
    assert trace["rate_states"].shape[:3] == (2, 2, 16)
    assert (result.paths.checkpoints / "best_model.eqx").exists()

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["model_family"] == "winfree_global_rate_phase"
    assert saved_config["task"]["winfree_global_gamma"] == 0.03
    assert saved_config["task"]["winfree_global_gate_strength"] == 0.25


def test_mnist_winfree_coarse_global_rate_phase_masked_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_coarse_global_rate_phase_masked_test",
        output_dir=tmp_path / "mnist_coarse_global_rate_phase_masked_train",
        seed=9,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(4, 4),
        model_family="winfree_coarse_global_rate_phase",
        winfree_steps=2,
        winfree_global_gamma=0.03,
        winfree_coarse_grid_size=2,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_rate_update_rate=0.25,
        winfree_rate_gate_strength=0.75,
        winfree_global_gate_strength=0.25,
        winfree_global_phase_control="shuffle",
        winfree_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 44)
    assert trace["omega"].shape == (2, 49, 4)
    assert trace["coarse_omega"].shape == (2, 4, 4)
    assert trace["thetas"].shape[:3] == (2, 2, 49)
    assert trace["coarse_thetas"].shape[:3] == (2, 2, 4)
    assert trace["fine_to_coarse_weights"].shape == (4, 49)
    assert trace["coarse_to_fine_weights"].shape == (49, 4)
    assert (result.paths.checkpoints / "best_model.eqx").exists()

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["model_family"] == "winfree_coarse_global_rate_phase"
    assert saved_config["task"]["winfree_global_gamma"] == 0.03
    assert saved_config["task"]["winfree_coarse_grid_size"] == 2
    assert saved_config["task"]["winfree_global_gate_strength"] == 0.25
    assert saved_config["task"]["winfree_global_phase_control"] == "shuffle"


def test_mnist_winfree_coarse_rate_phase_masked_artifacts(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_coarse_rate_phase_masked_test",
        output_dir=tmp_path / "mnist_coarse_rate_phase_masked_train",
        seed=10,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTAutoencoderExperimentConfig(
        run=run,
        hidden_dim=4,
        latent_dim=2,
        patch_shape=(4, 4),
        model_family="winfree_coarse_rate_phase",
        winfree_steps=2,
        winfree_global_gamma=0.03,
        winfree_coarse_grid_size=2,
        winfree_coupling_mode="conv",
        winfree_si_func="mlp",
        winfree_rate_update_rate=0.25,
        winfree_rate_gate_strength=0.75,
        winfree_global_gate_strength=0.25,
        winfree_global_phase_control="shuffle",
        winfree_global_content_strength=0.4,
        winfree_global_content_control="shuffle",
        winfree_output_activation="sigmoid",
        corruption_mode="patch_mask",
        corruption_fraction=0.5,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_experiment(config)
    trace = np.load(result.paths.traces / "mnist_latent_state_epoch_001.npz")

    assert trace["latent"].shape == (2, 60)
    assert trace["omega"].shape == (2, 49, 4)
    assert trace["coarse_omega"].shape == (2, 4, 4)
    assert trace["thetas"].shape[:3] == (2, 2, 49)
    assert trace["coarse_thetas"].shape[:3] == (2, 2, 4)
    assert trace["coarse_rate_states"].shape[:3] == (2, 2, 4)
    assert trace["coarse_rate_to_fine"].shape == (2, 49, 4)
    assert (result.paths.checkpoints / "best_model.eqx").exists()

    with open(result.paths.root / "config.json") as f:
        saved_config = json.load(f)
    assert saved_config["task"]["model_family"] == "winfree_coarse_rate_phase"
    assert saved_config["task"]["winfree_global_gamma"] == 0.03
    assert saved_config["task"]["winfree_coarse_grid_size"] == 2
    assert saved_config["task"]["winfree_global_gate_strength"] == 0.25
    assert saved_config["task"]["winfree_global_phase_control"] == "shuffle"
    assert saved_config["task"]["winfree_global_content_strength"] == 0.4
    assert saved_config["task"]["winfree_global_content_control"] == "shuffle"


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
