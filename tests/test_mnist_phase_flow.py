import json

import jax
import jax.numpy as jnp

from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_phase_flow import (
    MNISTPhaseFlowExperimentConfig,
    closure_loss,
    decode_phase_flow_primary_channel,
    decode_phase_flow_sample_readout,
    phase_flow_target_channels,
    phase_flow_loss,
    prepare_phase_flow_targets,
    primary_phase_flow_channel,
    run_mnist_phase_flow_experiment,
    sample_phase_flow_from_chord,
    sample_phase_flow_images,
    signed_distance_targets,
    sobel_edge_targets,
)
from oscnet.models import (
    CoarseGlobalPhaseRateFlowField,
    PhaseRateFlowField,
    RecurrentConvFlowField,
)


def test_phase_rate_flow_field_predicts_and_samples_shapes():
    model = PhaseRateFlowField(
        field_channels=3,
        steps=2,
        key=jax.random.PRNGKey(1),
    )
    images = jnp.linspace(-1.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)
    labels = jnp.asarray([1, 7], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75])

    velocity, trace = model(images, t, labels, return_trace=True)
    samples = model.sample(
        jax.random.PRNGKey(2),
        4,
        labels=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        outer_steps=3,
    )

    assert velocity.shape == images.shape
    assert samples.shape == (4, 28 * 28)
    assert trace["theta_trajectory"].shape == (2, 2, 28, 28, 3)
    assert trace["rate_trajectory"].shape == (2, 2, 28, 28, 3)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_phase_rate_flow_field_supports_two_value_channels():
    model = PhaseRateFlowField(
        value_channels=2,
        field_channels=3,
        steps=2,
        key=jax.random.PRNGKey(41),
    )
    images = jnp.linspace(-1.0, 1.0, 2 * 28 * 28 * 2).reshape(2, 28 * 28 * 2)
    labels = jnp.asarray([1, 7], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75])

    velocity, trace = model(images, t, labels, return_trace=True)
    samples = model.sample(
        jax.random.PRNGKey(42),
        4,
        labels=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        outer_steps=3,
    )

    assert model.value_channels == 2
    assert model.image_dim == 28 * 28 * 2
    assert velocity.shape == images.shape
    assert samples.shape == (4, 28 * 28 * 2)
    assert trace["theta_trajectory"].shape == (2, 2, 28, 28, 3)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_phase_rate_flow_field_can_sample_without_clipping():
    model = PhaseRateFlowField(
        value_channels=2,
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(43),
    )

    samples = model.sample(
        jax.random.PRNGKey(44),
        4,
        outer_steps=2,
        clip=False,
    )

    assert samples.shape == (4, 28 * 28 * 2)
    assert jnp.any(samples < 0.0) or jnp.any(samples > 1.0)


def test_shape_guided_sampler_runs_for_two_channel_phase_flow():
    model = PhaseRateFlowField(
        value_channels=2,
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(45),
    )

    samples = sample_phase_flow_images(
        model,
        key=jax.random.PRNGKey(46),
        sample_count=4,
        sample_steps=3,
        sample_method="euler",
        labels=None,
        batch_size=2,
        clip_samples=False,
        sample_schedule="shape_guided",
    )

    assert samples.shape == (4, 28 * 28 * 2)
    assert jnp.any(samples < 0.0) or jnp.any(samples > 1.0)


def test_basin_chord_sampler_runs_from_partial_real_state():
    model = PhaseRateFlowField(
        value_channels=2,
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(47),
    )
    targets = jnp.linspace(-1.0, 1.0, 4 * 28 * 28 * 2).reshape(4, 28 * 28 * 2)
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    samples = sample_phase_flow_from_chord(
        model,
        targets,
        key=jax.random.PRNGKey(48),
        start_t=0.5,
        sample_steps=3,
        sample_method="euler",
        labels=labels,
        batch_size=2,
        clip_samples=False,
    )

    assert samples.shape == targets.shape
    assert jnp.any(samples < 0.0) or jnp.any(samples > 1.0)


def test_phase_flow_loss_backprops_to_model():
    model = PhaseRateFlowField(
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(3),
    )
    images = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    loss, parts = phase_flow_loss(
        model,
        images,
        labels,
        jax.random.PRNGKey(4),
        clean_loss_weight=0.25,
        t_min=1e-3,
        t_max=0.999,
    )
    grads = jax.grad(
        lambda current_model: phase_flow_loss(
            current_model,
            images,
            labels,
            jax.random.PRNGKey(5),
            clean_loss_weight=0.25,
            t_min=1e-3,
            t_max=0.999,
        )[0]
    )(model)

    flat_grads = jax.tree.leaves(grads)
    grad_norm = sum(
        float(jnp.linalg.norm(grad)) for grad in flat_grads if hasattr(grad, "shape")
    )
    assert loss > 0.0
    assert parts["velocity_loss"] > 0.0
    assert parts["clean_loss"] >= 0.0
    assert parts["closure_loss"] >= 0.0
    assert grad_norm > 0.0


def test_closure_loss_rewards_matching_coarse_digit_envelope():
    target = jnp.zeros((2, 28 * 28))
    target = target.at[:, 8:20].set(1.0)
    shifted = jnp.roll(target, shift=4, axis=1)

    matching_loss = closure_loss(target, target)
    shifted_loss = closure_loss(shifted, target)

    assert matching_loss == 0.0
    assert shifted_loss > matching_loss


def test_sobel_edge_targets_are_normalized_contour_maps():
    images = jnp.zeros((2, 28 * 28))
    images = images.at[:, 8:20].set(1.0)

    edges = sobel_edge_targets(images)
    pixels = prepare_phase_flow_targets(images, "pixels")
    prepared_edges = prepare_phase_flow_targets(images, "sobel_edges")

    assert pixels is images
    assert edges.shape == images.shape
    assert prepared_edges.shape == images.shape
    assert jnp.max(edges) <= 1.0
    assert jnp.min(edges) >= 0.0
    assert jnp.mean(edges) > 0.0
    assert jnp.allclose(edges, prepared_edges)


def test_signed_distance_targets_are_smooth_shape_fields():
    image_grid = jnp.zeros((1, 28, 28))
    image_grid = image_grid.at[:, 9:19, 9:19].set(1.0)
    images = image_grid.reshape(1, 28 * 28)

    targets = signed_distance_targets(images)
    prepared = prepare_phase_flow_targets(images, "signed_distance")
    target_grid = targets.reshape(1, 28, 28)

    assert targets.shape == images.shape
    assert prepared.shape == images.shape
    assert jnp.max(targets) <= 1.0
    assert jnp.min(targets) >= 0.0
    assert target_grid[0, 14, 14] > target_grid[0, 8, 14]
    assert target_grid[0, 8, 14] > target_grid[0, 0, 0]
    assert jnp.allclose(targets, prepared)


def test_pixels_signed_distance_targets_keep_pixel_channel_primary():
    images = jnp.zeros((2, 28 * 28))
    images = images.at[:, 8:20].set(1.0)

    targets = prepare_phase_flow_targets(images, "pixels_signed_distance")
    target_grid = targets.reshape(2, 28, 28, 2)

    assert phase_flow_target_channels("pixels_signed_distance") == 2
    assert targets.shape == (2, 28 * 28 * 2)
    assert jnp.allclose(target_grid[..., 0].reshape(2, 28 * 28), images)
    assert jnp.allclose(
        target_grid[..., 1].reshape(2, 28 * 28),
        signed_distance_targets(images),
    )
    assert jnp.allclose(primary_phase_flow_channel(targets, 2), images)


def test_centered_pixels_signed_distance_targets_decode_to_pixels():
    images = jnp.zeros((2, 28 * 28))
    images = images.at[:, 8:20].set(1.0)

    targets = prepare_phase_flow_targets(images, "centered_pixels_signed_distance")
    target_grid = targets.reshape(2, 28, 28, 2)
    decoded = decode_phase_flow_primary_channel(
        targets,
        value_channels=2,
        target_representation="centered_pixels_signed_distance",
    )

    assert phase_flow_target_channels("centered_pixels_signed_distance") == 2
    assert targets.shape == (2, 28 * 28 * 2)
    assert jnp.min(targets) >= -1.0
    assert jnp.max(targets) <= 1.0
    assert jnp.allclose(target_grid[..., 0].reshape(2, 28 * 28), 2.0 * images - 1.0)
    assert jnp.allclose(decoded, images)


def test_shape_gated_sample_readout_uses_auxiliary_shape_channel():
    primary = jnp.ones((1, 28 * 28))
    high_shape = jnp.ones((1, 28 * 28))
    low_shape = -jnp.ones((1, 28 * 28))
    high_samples = jnp.stack(
        [primary.reshape(1, 28, 28), high_shape.reshape(1, 28, 28)],
        axis=-1,
    ).reshape(1, 28 * 28 * 2)
    low_samples = jnp.stack(
        [primary.reshape(1, 28, 28), low_shape.reshape(1, 28, 28)],
        axis=-1,
    ).reshape(1, 28 * 28 * 2)

    high_readout = decode_phase_flow_sample_readout(
        high_samples,
        value_channels=2,
        target_representation="centered_pixels_signed_distance",
        sample_readout_mode="shape_gated",
    )
    low_readout = decode_phase_flow_sample_readout(
        low_samples,
        value_channels=2,
        target_representation="centered_pixels_signed_distance",
        sample_readout_mode="shape_gated",
    )
    primary_readout = decode_phase_flow_sample_readout(
        low_samples,
        value_channels=2,
        target_representation="centered_pixels_signed_distance",
        sample_readout_mode="primary",
    )

    assert jnp.mean(high_readout) > 0.9
    assert jnp.mean(low_readout) < 0.1
    assert jnp.mean(primary_readout) == 1.0


def test_phase_flow_loss_includes_optional_closure_term():
    model = PhaseRateFlowField(
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(7),
    )
    images = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    key = jax.random.PRNGKey(8)

    loss_without_closure, parts_without_closure = phase_flow_loss(
        model,
        images,
        labels,
        key,
        clean_loss_weight=0.25,
        closure_loss_weight=0.0,
        t_min=1e-3,
        t_max=0.999,
    )
    loss_with_closure, parts_with_closure = phase_flow_loss(
        model,
        images,
        labels,
        key,
        clean_loss_weight=0.25,
        closure_loss_weight=1.0,
        t_min=1e-3,
        t_max=0.999,
    )

    assert parts_without_closure["closure_loss"] == parts_with_closure["closure_loss"]
    assert parts_with_closure["closure_loss"] >= 0.0
    assert loss_with_closure >= loss_without_closure


def test_recurrent_conv_flow_field_predicts_and_samples_shapes():
    model = RecurrentConvFlowField(
        field_channels=3,
        steps=2,
        key=jax.random.PRNGKey(11),
    )
    images = jnp.linspace(-1.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)
    labels = jnp.asarray([1, 7], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75])

    velocity, trace = model(images, t, labels, return_trace=True)
    samples = model.sample(
        jax.random.PRNGKey(12),
        4,
        labels=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        outer_steps=3,
    )

    assert velocity.shape == images.shape
    assert samples.shape == (4, 28 * 28)
    assert trace["hidden_trajectory"].shape == (2, 2, 28, 28, 3)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_recurrent_conv_flow_loss_backprops_to_model():
    model = RecurrentConvFlowField(
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(13),
    )
    images = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    loss, parts = phase_flow_loss(
        model,
        images,
        labels,
        jax.random.PRNGKey(14),
        clean_loss_weight=0.25,
        t_min=1e-3,
        t_max=0.999,
    )
    grads = jax.grad(
        lambda current_model: phase_flow_loss(
            current_model,
            images,
            labels,
            jax.random.PRNGKey(15),
            clean_loss_weight=0.25,
            t_min=1e-3,
            t_max=0.999,
        )[0]
    )(model)

    flat_grads = jax.tree.leaves(grads)
    grad_norm = sum(
        float(jnp.linalg.norm(grad)) for grad in flat_grads if hasattr(grad, "shape")
    )
    assert loss > 0.0
    assert parts["velocity_loss"] > 0.0
    assert parts["clean_loss"] >= 0.0
    assert parts["closure_loss"] >= 0.0
    assert grad_norm > 0.0


def test_coarse_global_phase_flow_field_predicts_and_samples_shapes():
    model = CoarseGlobalPhaseRateFlowField(
        field_channels=3,
        steps=2,
        coarse_grid_size=4,
        position_features=True,
        key=jax.random.PRNGKey(21),
    )
    images = jnp.linspace(-1.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)
    labels = jnp.asarray([1, 7], dtype=jnp.int32)
    t = jnp.asarray([0.25, 0.75])

    velocity, trace = model(images, t, labels, return_trace=True)
    samples = model.sample(
        jax.random.PRNGKey(22),
        4,
        labels=jnp.asarray([0, 1, 2, 3], dtype=jnp.int32),
        outer_steps=3,
    )

    assert velocity.shape == images.shape
    assert samples.shape == (4, 28 * 28)
    assert trace["theta_trajectory"].shape == (2, 2, 28, 28, 3)
    assert trace["coarse_theta_trajectory"].shape == (2, 2, 4, 4, 3)
    assert model.position_features is True
    assert model.fine.position_features is True
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_phase_flow_heun_sampler_shapes():
    model = PhaseRateFlowField(
        field_channels=2,
        steps=1,
        key=jax.random.PRNGKey(31),
    )
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    samples = sample_phase_flow_images(
        model,
        key=jax.random.PRNGKey(32),
        sample_count=4,
        sample_steps=3,
        sample_method="heun",
        labels=labels,
        batch_size=2,
    )

    assert samples.shape == (4, 28 * 28)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_mnist_phase_flow_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_phase_flow_test",
        output_dir=tmp_path / "mnist_phase_flow",
        seed=6,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTPhaseFlowExperimentConfig(
        run=run,
        model_family="phase_flow",
        field_channels=2,
        steps=1,
        closure_loss_weight=0.5,
        target_representation="pixels_signed_distance",
        eval_sample_count=2,
        sample_steps=2,
        basin_t_values=(0.5,),
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_phase_flow_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    assert (result.paths.artifacts / "samples_epoch_001.png").exists()
    assert (result.paths.artifacts / "denoised_epoch_001.png").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["phase_flow"]["model_family"] == "phase_flow"
    assert summary["phase_flow"]["steps"] == 1
    assert summary["phase_flow"]["sample_method"] == "euler"
    assert summary["phase_flow"]["sample_schedule"] == "standard"
    assert summary["phase_flow"]["sample_readout_mode"] == "primary"
    assert summary["phase_flow"]["basin_t_values"] == [0.5]
    basin_metrics = summary["phase_flow"]["basin"]["t0_500"]
    assert basin_metrics["initial_paired_mse"] >= 0.0
    assert basin_metrics["paired_mse"] >= 0.0
    assert "paired_mse_delta" in basin_metrics
    assert "paired_mse_improvement_fraction" in basin_metrics
    assert summary["phase_flow"]["closure_loss_weight"] == 0.5
    assert summary["phase_flow"]["target_representation"] == "pixels_signed_distance"
    assert summary["phase_flow"]["target_channels"] == 2
    assert summary["final_eval_closure_loss"] >= 0.0
    assert summary["final_eval_loss"] >= 0.0
    assert summary["phase_flow"]["sample_diversity_ratio"] >= 0.0
    assert summary["phase_flow"]["sample_component_count"] >= 0.0


def test_mnist_recurrent_conv_flow_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_recurrent_conv_flow_test",
        output_dir=tmp_path / "mnist_recurrent_conv_flow",
        seed=16,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTPhaseFlowExperimentConfig(
        run=run,
        model_family="recurrent_conv_flow",
        field_channels=2,
        steps=1,
        eval_sample_count=2,
        sample_steps=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_phase_flow_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["phase_flow"]["model_family"] == "recurrent_conv_flow"
    assert summary["phase_flow"]["steps"] == 1
    assert summary["final_eval_loss"] >= 0.0
    assert summary["phase_flow"]["state_mean_abs_displacement"] >= 0.0


def test_mnist_coarse_phase_flow_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_coarse_phase_flow_test",
        output_dir=tmp_path / "mnist_coarse_phase_flow",
        seed=26,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTPhaseFlowExperimentConfig(
        run=run,
        model_family="coarse_phase_flow",
        field_channels=2,
        steps=1,
        coarse_grid_size=4,
        position_features=True,
        eval_sample_count=2,
        sample_steps=2,
        sample_method="heun",
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_phase_flow_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["phase_flow"]["model_family"] == "coarse_phase_flow"
    assert summary["phase_flow"]["steps"] == 1
    assert summary["phase_flow"]["sample_method"] == "heun"
    assert summary["phase_flow"]["coarse_grid_size"] == 4
    assert summary["phase_flow"]["position_features"] is True
    assert summary["final_eval_loss"] >= 0.0
    assert summary["phase_flow"]["sample_largest_component_fraction"] >= 0.0
