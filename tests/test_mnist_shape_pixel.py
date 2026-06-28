import json

import jax
import jax.numpy as jnp

from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_shape_pixel import (
    MNISTShapePixelExperimentConfig,
    apply_shape_pixel_readout,
    build_mnist_shape_pixel_model,
    compute_shape_condition_probe_metrics,
    compute_shape_pixel_basin_metrics,
    corrupt_shape_conditions,
    make_shape_pixel_flow_batch,
    run_mnist_shape_pixel_experiment,
    sample_shape_pixel_from_chord,
    sample_shape_pixel_images,
    shape_pixel_loss,
    split_pixel_shape_channels,
    stack_pixel_shape_channels,
)


def test_shape_pixel_channel_stack_round_trips():
    pixels = jnp.linspace(0.0, 1.0, 2 * 28 * 28).reshape(2, 28 * 28)
    shapes = 1.0 - pixels

    state = stack_pixel_shape_channels(pixels, shapes)
    restored_pixels, restored_shapes = split_pixel_shape_channels(state)

    assert state.shape == (2, 28 * 28 * 2)
    assert jnp.allclose(restored_pixels, pixels)
    assert jnp.allclose(restored_shapes, shapes)


def test_shape_pixel_shape_gated_readout_uses_shape_scaffold():
    pixels = jnp.ones((2, 28 * 28))
    dark_shape = jnp.zeros_like(pixels)
    bright_shape = jnp.ones_like(pixels)

    primary = apply_shape_pixel_readout(
        pixels,
        dark_shape,
        sample_readout_mode="primary",
    )
    dark_gated = apply_shape_pixel_readout(
        pixels,
        dark_shape,
        sample_readout_mode="shape_gated",
    )
    bright_gated = apply_shape_pixel_readout(
        pixels,
        bright_shape,
        sample_readout_mode="shape_gated",
    )

    assert jnp.allclose(primary, pixels)
    assert float(jnp.mean(dark_gated)) < 0.1
    assert float(jnp.mean(bright_gated)) > 0.9


def test_shape_pixel_flow_batch_keeps_shape_condition_static():
    pixels = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)
    shapes = 1.0 - pixels

    state, t, target_velocity, _ = make_shape_pixel_flow_batch(
        pixels,
        shapes,
        jax.random.PRNGKey(1),
        t_min=0.2,
        t_max=0.8,
    )
    _state_pixels, state_shapes = split_pixel_shape_channels(state)
    _target_pixels, target_shapes = split_pixel_shape_channels(target_velocity)

    assert state.shape == (3, 28 * 28 * 2)
    assert t.shape == (3,)
    assert jnp.allclose(state_shapes, shapes)
    assert jnp.allclose(target_shapes, jnp.zeros_like(shapes))


def test_shape_pixel_loss_backprops_to_model():
    config = MNISTShapePixelExperimentConfig(
        run=AutoencoderExperimentConfig(name="shape_pixel_loss_test"),
        model_family="phase_flow",
        field_channels=2,
        steps=1,
    )
    model = build_mnist_shape_pixel_model(config, jax.random.PRNGKey(2))
    pixels = jnp.linspace(0.0, 1.0, 3 * 28 * 28).reshape(3, 28 * 28)
    shapes = 1.0 - pixels
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    loss, parts = shape_pixel_loss(
        model,
        pixels,
        shapes,
        labels,
        jax.random.PRNGKey(3),
        clean_loss_weight=0.25,
        shape_velocity_weight=0.1,
        closure_loss_weight=0.0,
        t_min=1e-3,
        t_max=0.999,
    )
    grads = jax.grad(
        lambda current_model: shape_pixel_loss(
            current_model,
            pixels,
            shapes,
            labels,
            jax.random.PRNGKey(4),
            clean_loss_weight=0.25,
            shape_velocity_weight=0.1,
            closure_loss_weight=0.0,
            t_min=1e-3,
            t_max=0.999,
        )[0]
    )(model)

    flat_grads = jax.tree.leaves(grads)
    grad_norm = sum(
        float(jnp.linalg.norm(grad)) for grad in flat_grads if hasattr(grad, "shape")
    )
    assert loss > 0.0
    assert parts["pixel_velocity_loss"] > 0.0
    assert parts["shape_velocity_loss"] >= 0.0
    assert parts["clean_loss"] >= 0.0
    assert grad_norm > 0.0


def test_shape_pixel_sampler_generates_pixels_from_shape_condition():
    config = MNISTShapePixelExperimentConfig(
        run=AutoencoderExperimentConfig(name="shape_pixel_sample_test"),
        model_family="recurrent_conv_flow",
        field_channels=2,
        steps=1,
    )
    model = build_mnist_shape_pixel_model(config, jax.random.PRNGKey(5))
    shapes = jnp.linspace(0.0, 1.0, 4 * 28 * 28).reshape(4, 28 * 28)
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    samples = sample_shape_pixel_images(
        model,
        shapes,
        key=jax.random.PRNGKey(6),
        sample_steps=2,
        sample_method="euler",
        labels=labels,
        batch_size=2,
        clamp_shape=True,
    )

    assert samples.shape == (4, 28 * 28)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_shape_pixel_basin_probe_measures_chord_completion():
    config = MNISTShapePixelExperimentConfig(
        run=AutoencoderExperimentConfig(name="shape_pixel_basin_test"),
        model_family="recurrent_conv_flow",
        field_channels=2,
        steps=1,
    )
    model = build_mnist_shape_pixel_model(config, jax.random.PRNGKey(8))
    pixels = jnp.linspace(0.0, 1.0, 4 * 28 * 28).reshape(4, 28 * 28)
    shapes = 1.0 - pixels
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    samples = sample_shape_pixel_from_chord(
        model,
        pixels,
        shapes,
        key=jax.random.PRNGKey(9),
        start_t=0.5,
        sample_steps=2,
        sample_method="euler",
        labels=labels,
        batch_size=2,
        clamp_shape=True,
    )
    metrics = compute_shape_pixel_basin_metrics(
        model,
        pixels,
        shapes,
        labels,
        key=jax.random.PRNGKey(10),
        t_values=(0.5,),
        sample_steps=2,
        sample_method="euler",
        batch_size=2,
        clamp_shape=True,
    )

    assert samples.shape == (4, 28 * 28)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)
    assert "t0_500" in metrics
    assert metrics["t0_500"]["initial_paired_mse"] >= 0.0
    assert metrics["t0_500"]["paired_mse"] >= 0.0


def test_shape_condition_probe_measures_imperfect_scaffolds():
    config = MNISTShapePixelExperimentConfig(
        run=AutoencoderExperimentConfig(name="shape_condition_probe_test"),
        model_family="recurrent_conv_flow",
        field_channels=2,
        steps=1,
    )
    model = build_mnist_shape_pixel_model(config, jax.random.PRNGKey(11))
    pixels = jnp.linspace(0.0, 1.0, 4 * 28 * 28).reshape(4, 28 * 28)
    shapes = 1.0 - pixels
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)

    corrupted = corrupt_shape_conditions(
        shapes,
        key=jax.random.PRNGKey(12),
        start_t=0.5,
        noise_mode="uniform",
    )
    metrics = compute_shape_condition_probe_metrics(
        model,
        pixels,
        shapes,
        labels,
        key=jax.random.PRNGKey(13),
        t_values=(0.5,),
        noise_modes=("uniform",),
        sample_steps=2,
        sample_method="euler",
        batch_size=2,
        clamp_shape=True,
    )

    assert corrupted.shape == shapes.shape
    assert jnp.all(corrupted >= 0.0)
    assert jnp.all(corrupted <= 1.0)
    assert "uniform" in metrics
    assert "t0_500" in metrics["uniform"]
    assert metrics["uniform"]["t0_500"]["condition_paired_mse"] >= 0.0
    assert metrics["uniform"]["t0_500"]["paired_sample_mse"] >= 0.0


def test_mnist_shape_pixel_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_shape_pixel_test",
        output_dir=tmp_path / "mnist_shape_pixel",
        seed=7,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTShapePixelExperimentConfig(
        run=run,
        model_family="phase_flow",
        field_channels=2,
        steps=1,
        eval_sample_count=2,
        sample_steps=2,
        basin_t_values=(0.5,),
        shape_condition_t_values=(0.5,),
        shape_condition_noise_modes=("uniform",),
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_shape_pixel_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    assert (result.paths.artifacts / "samples_epoch_001.png").exists()
    assert (result.paths.artifacts / "shape_epoch_001.png").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["shape_pixel"]["model_family"] == "phase_flow"
    assert summary["shape_pixel"]["value_channels"] == 2
    assert summary["shape_pixel"]["clamp_shape"] is True
    assert summary["shape_pixel"]["sample_readout_mode"] == "primary"
    assert summary["shape_pixel"]["paired_sample_mse"] >= 0.0
    assert "t0_500" in summary["shape_pixel"]["basin"]
    assert "uniform" in summary["shape_pixel"]["shape_condition_probe"]
    assert "t0_500" in summary["shape_pixel"]["shape_condition_probe"]["uniform"]
    assert summary["final_eval_clean_loss"] >= 0.0
