import json

import jax
import jax.numpy as jnp

from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_generator import (
    MNISTDriftQueue,
    MNISTFeatureClassifier,
    MNISTGeneratorExperimentConfig,
    RECOMMENDED_GENERATOR_PRESET,
    build_arg_parser,
    build_mnist_generator_model,
    conditional_feature_drift_loss,
    conditional_pixel_drift_loss,
    config_from_args,
    compute_class_prototypes,
    compute_generator_settling_metrics,
    compute_generator_success_diagnostics,
    compute_generator_quality_metrics,
    generator_distribution_loss,
    generator_loss,
    make_projection_matrix,
    mnist_structural_features,
    parse_args,
    run_mnist_generator_experiment,
    sliced_wasserstein_loss,
    train_mnist_feature_classifier,
)


def test_sliced_wasserstein_loss_is_zero_for_identical_batches():
    images = jnp.linspace(0.0, 1.0, 4 * 16).reshape(4, 16)
    projections = make_projection_matrix(
        jax.random.PRNGKey(0),
        image_dim=16,
        num_projections=8,
    )

    loss = sliced_wasserstein_loss(images, images, projections)

    assert loss < 1e-7


def test_sparse_horn_mnist_preset_sets_current_recipe():
    args = parse_args(
        [
            "--preset",
            "sparse_horn_mnist",
            "--epochs",
            "1",
            "--train-limit",
            "8",
        ]
    )
    parsed = config_from_args(args)

    assert parsed.model_family == "horn"
    assert parsed.decoder_mode == "resize_conv"
    assert parsed.coupling_profile == "local_radius"
    assert parsed.coupling_length_scale == 0.24
    assert parsed.train_settling_steps == (8, 16, 32)
    assert parsed.run.epochs == 1
    assert parsed.train_limit == 8


def test_generator_recommended_preset_is_opt_in_default():
    generic = config_from_args(parse_args([]))

    assert generic.model_family == "kuramoto"
    assert generic.run.epochs == 10

    explicit_none = config_from_args(
        parse_args(["--preset", "none"], default_preset=RECOMMENDED_GENERATOR_PRESET)
    )
    assert explicit_none.model_family == "kuramoto"

    recommended = config_from_args(
        parse_args([], default_preset=RECOMMENDED_GENERATOR_PRESET)
    )
    assert recommended.model_family == "horn"
    assert recommended.conditioning_mode == "class_coupling"
    assert recommended.conditioning_strength == 8.0
    assert recommended.horn_damping == 0.30
    assert recommended.distributional_weight == 0.0
    assert recommended.train_settling_steps == (16, 32, 48)
    assert recommended.settling_steps == (0, 1, 8, 16, 32, 48, 64)

    model = build_mnist_generator_model(recommended, jax.random.PRNGKey(0))
    assert model.label_phase_shift is None
    assert model.label_condition_coupling is not None


def test_config_from_args_rejects_unapplied_preset_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(["--preset", "sparse_horn_mnist"])

    try:
        config_from_args(args)
    except ValueError as exc:
        assert "preset defaults were not applied" in str(exc)
    else:
        raise AssertionError("config_from_args should reject unapplied presets")


def test_sparse_horn_mnist_control_presets_share_recipe():
    expected = {
        "sparse_horn_mnist": "horn",
        "sparse_horn_mnist_frozen": "frozen_horn",
        "sparse_horn_mnist_decoder_only": "horn_decoder_only",
        "sparse_horn_mnist_state_mlp": "state_mlp",
        "sparse_horn_mnist_state_mlp_frozen": "frozen_state_mlp",
        "sparse_horn_mnist_state_mlp_decoder_only": "state_mlp_decoder_only",
        "sparse_horn_mnist_step1": "horn",
        "sparse_horn_mnist_class_oscillator": "horn",
        "sparse_horn_mnist_class_oscillator_step1": "horn",
        "sparse_horn_mnist_class_oscillator_frozen": "frozen_horn",
        "sparse_horn_mnist_class_coupling": "horn",
        "sparse_horn_mnist_class_coupling_step1": "horn",
        "sparse_horn_mnist_class_coupling_long": "horn",
        "sparse_horn_mnist_class_coupling_strong": "horn",
        "sparse_horn_mnist_class_coupling_strength4": "horn",
        "sparse_horn_mnist_class_coupling_strength8": "horn",
        "sparse_horn_mnist_class_coupling_strength8_dist001": "horn",
        "sparse_horn_mnist_class_coupling_strength8_dist0025": "horn",
        "sparse_horn_mnist_class_coupling_strength8_dist005": "horn",
        "sparse_horn_mnist_class_coupling_strength8_freq13": "horn",
        "sparse_horn_mnist_class_coupling_strength8_damp030": "horn",
        "sparse_horn_mnist_class_coupling_strength8_freq13_dist0025": "horn",
        "sparse_horn_mnist_recommended": "horn",
        "sparse_horn_mnist_strict": "horn",
        "sparse_horn_mnist_quality": "horn",
        "sparse_horn_mnist_dynamics_quality": "horn",
        "sparse_horn_mnist_class_coupling_strong_frozen": "frozen_horn",
        "sparse_horn_mnist_class_coupling_strong_decoder_only": "horn_decoder_only",
        "sparse_horn_mnist_state_mlp_class_coupling_strong": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strong_dist005": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01_class": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strong_frozen": "frozen_state_mlp",
        "sparse_horn_mnist_class_coupling_anchor": "horn",
    }

    for preset, model_family in expected.items():
        parsed = config_from_args(parse_args(["--preset", preset]))
        assert parsed.model_family == model_family
        assert parsed.decoder_mode == "resize_conv"
        assert parsed.readout_mode == "mean_relative"
        assert parsed.loss_mode == "pixel_drift"
        assert parsed.train_limit == 500

    step1 = config_from_args(parse_args(["--preset", "sparse_horn_mnist_step1"]))
    assert step1.steps == 1
    assert step1.train_settling_steps == (1,)

    strong = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_class_coupling_strong"])
    )
    assert strong.conditioning_mode == "class_coupling"
    assert strong.conditioning_strength == 2.0
    assert strong.train_settling_steps == (16, 32, 48)
    assert strong.settling_steps == (0, 1, 8, 16, 32, 48, 64)

    strength8 = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_class_coupling_strength8"])
    )
    assert strength8.conditioning_mode == "class_coupling"
    assert strength8.conditioning_strength == 8.0

    strict = config_from_args(parse_args(["--preset", "sparse_horn_mnist_strict"]))
    assert strict.conditioning_mode == "class_coupling"
    assert strict.conditioning_strength == 8.0
    assert strict.horn_damping == 0.15
    assert strict.distributional_weight == 0.0

    strength8_dist = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_mnist_class_coupling_strength8_dist0025"]
        )
    )
    assert strength8_dist.conditioning_mode == "class_coupling"
    assert strength8_dist.conditioning_strength == 8.0
    assert strength8_dist.distributional_weight == 0.025
    assert strength8_dist.train_settling_steps == (16, 32, 48)

    quality = config_from_args(parse_args(["--preset", "sparse_horn_mnist_quality"]))
    assert quality.conditioning_mode == "class_coupling"
    assert quality.conditioning_strength == 8.0
    assert quality.horn_damping == 0.15
    assert quality.distributional_weight == 0.025

    strength8_freq = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_class_coupling_strength8_freq13"])
    )
    assert strength8_freq.horn_frequency == 1.3
    assert strength8_freq.horn_damping == 0.15
    assert strength8_freq.distributional_weight == 0.0

    strength8_freq_dist = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_class_coupling_strength8_freq13_dist0025",
            ]
        )
    )
    assert strength8_freq_dist.horn_frequency == 1.3
    assert strength8_freq_dist.distributional_weight == 0.025

    dynamics_quality = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_dynamics_quality"])
    )
    recommended = config_from_args(
        parse_args(["--preset", RECOMMENDED_GENERATOR_PRESET])
    )
    for parsed in (dynamics_quality, recommended):
        assert parsed.conditioning_mode == "class_coupling"
        assert parsed.conditioning_strength == 8.0
        assert parsed.horn_damping == 0.30
        assert parsed.distributional_weight == 0.0

    state_mlp_dist = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_state_mlp_class_coupling_strong_dist01_class",
            ]
        )
    )
    assert state_mlp_dist.model_family == "state_mlp"
    assert state_mlp_dist.conditioning_mode == "class_coupling"
    assert state_mlp_dist.conditioning_strength == 2.0
    assert state_mlp_dist.distributional_weight == 0.1
    assert state_mlp_dist.class_moment_weight == 1.0

    anchor = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_class_coupling_anchor"])
    )
    assert anchor.conditioning_mode == "class_coupling"
    assert anchor.label_phase_scale == 0.5


def test_class_oscillator_preset_removes_initial_label_shift():
    parsed = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_class_oscillator",
                "--epochs",
                "1",
                "--train-limit",
                "8",
            ]
        )
    )

    assert parsed.conditioning_mode == "class_oscillator"
    assert parsed.num_condition_oscillators == 32

    model = build_mnist_generator_model(parsed, jax.random.PRNGKey(0))

    assert model.label_phase_shift is None
    assert model.label_condition_coupling is not None
    assert model.condition_omega is not None
    assert model.condition_coupling is not None


def test_class_coupling_strong_preset_keeps_no_initial_label_shift():
    parsed = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_class_coupling_strong",
                "--epochs",
                "1",
                "--train-limit",
                "8",
            ]
        )
    )

    model = build_mnist_generator_model(parsed, jax.random.PRNGKey(0))

    assert model.conditioning_mode == "class_coupling"
    assert model.conditioning_strength == 2.0
    assert model.label_phase_shift is None
    assert model.label_condition_phase is not None
    assert model.label_condition_coupling is not None


def test_state_mlp_class_coupling_strong_uses_label_drive():
    parsed = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_state_mlp_class_coupling_strong",
                "--epochs",
                "1",
                "--train-limit",
                "8",
            ]
        )
    )

    model = build_mnist_generator_model(parsed, jax.random.PRNGKey(0))
    position, velocity = model.initial_state(jax.random.PRNGKey(1), 1, None)
    next_zero = model.step_state(
        (position, velocity),
        jnp.asarray([0], dtype=jnp.int32),
    )
    next_one = model.step_state(
        (position, velocity),
        jnp.asarray([1], dtype=jnp.int32),
    )

    assert model.dynamics_family == "state_mlp"
    assert model.conditioning_mode == "class_coupling"
    assert model.conditioning_strength == 2.0
    assert model.label_phase_shift is None
    assert model.label_condition_phase is not None
    assert model.label_condition_coupling is not None
    assert float(jnp.max(jnp.abs(next_zero[1] - next_one[1]))) > 0.0


def test_quality_classifier_limits_parse_without_changing_generator_limits():
    parsed = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_class_coupling_strength8",
                "--train-limit",
                "32",
                "--eval-limit",
                "16",
                "--quality-classifier-train-limit",
                "5000",
                "--quality-classifier-eval-limit",
                "2000",
            ]
        )
    )

    assert parsed.train_limit == 32
    assert parsed.eval_limit == 16
    assert parsed.quality_classifier_train_limit == 5000
    assert parsed.quality_classifier_eval_limit == 2000


def test_conditional_generator_loss_uses_class_terms():
    images = jnp.linspace(0.0, 1.0, 6 * 16).reshape(6, 16)
    labels = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    generated = jnp.flip(images, axis=0)
    projections = make_projection_matrix(
        jax.random.PRNGKey(1),
        image_dim=16,
        num_projections=8,
    )
    prototypes = compute_class_prototypes(images, labels, num_classes=3)

    loss, parts = generator_distribution_loss(
        images,
        generated,
        projections,
        labels=labels,
        prototypes=prototypes,
        num_classes=3,
        class_moment_weight=0.5,
        prototype_weight=0.25,
    )

    assert loss > 0.0
    assert parts["class_moment_loss"] > 0.0
    assert parts["prototype_loss"] > 0.0
    assert prototypes.shape == (3, 16)


def test_conditional_pixel_drift_loss_backprops_to_generated_samples():
    real = jnp.linspace(0.0, 1.0, 12 * 16).reshape(12, 16)
    generated = jnp.flip(real, axis=0)
    labels = jnp.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=jnp.int32,
    )

    loss = conditional_pixel_drift_loss(
        real,
        generated,
        labels,
        num_classes=4,
        gamma=0.2,
    )
    grad = jax.grad(
        lambda samples: conditional_pixel_drift_loss(
            real,
            samples,
            labels,
            num_classes=4,
            gamma=0.2,
        )
    )(generated)

    assert loss > 0.0
    assert grad.shape == generated.shape
    assert jnp.linalg.norm(grad) > 0.0
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_mnist_drift_queue_draws_balanced_positive_memory():
    queue = MNISTDriftQueue.create(
        num_classes=3,
        queue_size=4,
        image_dim=5,
        seed=123,
    )
    images = jnp.arange(12 * 5, dtype=jnp.float32).reshape(12, 5)
    labels = jnp.asarray([0, 1, 2] * 4, dtype=jnp.int32)

    queue.push(images, labels)
    positives, positive_labels = queue.draw(2)

    assert positives.shape == (6, 5)
    assert positive_labels.shape == (6,)
    assert queue.ready(2)
    assert jnp.array_equal(
        jnp.bincount(positive_labels, length=3),
        jnp.asarray([2, 2, 2]),
    )


def test_conditional_pixel_drift_loss_accepts_queue_positive_pool():
    real = jnp.linspace(0.0, 1.0, 18 * 16).reshape(18, 16)
    generated = jnp.flip(real[:6], axis=0)
    generated_labels = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    positive_labels = jnp.asarray([0, 1, 2] * 6, dtype=jnp.int32)
    gamma_real = real[:9]
    gamma_labels = jnp.asarray([0, 1, 2] * 3, dtype=jnp.int32)

    loss = conditional_pixel_drift_loss(
        real,
        generated,
        generated_labels,
        num_classes=3,
        positive_labels=positive_labels,
        gamma_real=gamma_real,
        gamma_labels=gamma_labels,
        gamma=0.2,
    )
    grad = jax.grad(
        lambda samples: conditional_pixel_drift_loss(
            real,
            samples,
            generated_labels,
            num_classes=3,
            positive_labels=positive_labels,
            gamma_real=gamma_real,
            gamma_labels=gamma_labels,
            gamma=0.2,
        )
    )(generated)

    assert loss > 0.0
    assert grad.shape == generated.shape
    assert jnp.linalg.norm(grad) > 0.0
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_structural_feature_drift_backprops_to_generated_samples():
    real = jnp.linspace(0.0, 1.0, 12 * 28 * 28).reshape(12, 28 * 28)
    generated = jnp.flip(real, axis=0)
    labels = jnp.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=jnp.int32,
    )

    features = mnist_structural_features(real)
    loss = conditional_feature_drift_loss(
        real,
        generated,
        labels,
        num_classes=4,
        feature_mode="structural",
        gamma=0.2,
    )
    grad = jax.grad(
        lambda samples: conditional_feature_drift_loss(
            real,
            samples,
            labels,
            num_classes=4,
            feature_mode="structural",
            gamma=0.2,
        )
    )(generated)

    assert features.shape[0] == real.shape[0]
    assert 0 < features.shape[1] < real.shape[1]
    assert loss > 0.0
    assert grad.shape == generated.shape
    assert jnp.linalg.norm(grad) > 0.0
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_learned_feature_drift_backprops_through_frozen_classifier():
    real = jnp.linspace(0.0, 1.0, 12 * 28 * 28).reshape(12, 28 * 28)
    generated = jnp.flip(real, axis=0)
    labels = jnp.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=jnp.int32,
    )
    classifier = MNISTFeatureClassifier(
        feature_dim=16,
        depth=1,
        num_classes=4,
        key=jax.random.PRNGKey(13),
    )

    loss = conditional_feature_drift_loss(
        real,
        generated,
        labels,
        num_classes=4,
        feature_mode="learned",
        feature_model=classifier,
        gamma=0.2,
    )
    grad = jax.grad(
        lambda samples: conditional_feature_drift_loss(
            real,
            samples,
            labels,
            num_classes=4,
            feature_mode="learned",
            feature_model=classifier,
            gamma=0.2,
        )
    )(generated)

    assert classifier.features(real).shape == (12, 16)
    assert loss > 0.0
    assert grad.shape == generated.shape
    assert jnp.linalg.norm(grad) > 0.0
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_generator_loss_combines_pixel_and_feature_drift():
    real = jnp.linspace(0.0, 1.0, 12 * 28 * 28).reshape(12, 28 * 28)
    generated = jnp.flip(real, axis=0)
    labels = jnp.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        dtype=jnp.int32,
    )
    projections = make_projection_matrix(
        jax.random.PRNGKey(12),
        image_dim=28 * 28,
        num_projections=8,
    )

    loss, parts = generator_loss(
        real,
        generated,
        projections,
        labels=labels,
        num_classes=4,
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=1.0,
        distributional_weight=0.0,
    )

    expected = 0.5 * parts["pixel_drift_loss"] + parts["feature_drift_loss"]
    assert loss > 0.0
    assert jnp.allclose(loss, expected)
    assert parts["pixel_drift_loss"] > 0.0
    assert parts["feature_drift_loss"] > 0.0


def test_generator_quality_metrics_can_use_classifier_labels():
    real = jnp.linspace(0.0, 1.0, 6 * 28 * 28).reshape(6, 28 * 28)
    generated = jnp.flip(real, axis=0)
    labels = jnp.asarray([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
    classifier = MNISTFeatureClassifier(
        feature_dim=8,
        depth=1,
        num_classes=3,
        key=jax.random.PRNGKey(70),
    )

    metrics = compute_generator_quality_metrics(
        real,
        generated,
        labels=labels,
        classifier=classifier,
    )

    assert 0.0 <= metrics["classifier_label_accuracy"] <= 1.0
    assert 0.0 <= metrics["classifier_label_confidence"] <= 1.0
    assert 0.0 <= metrics["classifier_max_confidence"] <= 1.0
    assert metrics["classifier_entropy"] >= 0.0


def test_train_mnist_feature_classifier_smoke():
    images = jnp.linspace(0.0, 1.0, 12 * 28 * 28).reshape(12, 28 * 28)
    labels = jnp.asarray([0, 1, 2, 3] * 3, dtype=jnp.int32)

    classifier, history = train_mnist_feature_classifier(
        images,
        labels,
        images[:8],
        labels[:8],
        key=jax.random.PRNGKey(14),
        num_classes=4,
        feature_dim=12,
        depth=1,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    assert classifier.features(images[:2]).shape == (2, 12)
    assert history["epochs"] == 1
    assert 0.0 <= history["final_eval_accuracy"] <= 1.0
    assert history["final_eval_loss"] >= 0.0


def test_mnist_generator_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_test",
        output_dir=tmp_path / "mnist_generator",
        seed=5,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        num_oscillators=8,
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    assert (result.paths.metrics / "summary.json").exists()
    assert (
        result.paths.artifacts / "mnist_generator_samples_epoch_001.npz"
    ).exists()
    assert (result.paths.traces / "mnist_generator_trace_epoch_001.npz").exists()
    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["generator"]["distributional_not_paired_reconstruction"]
    diagnostics = summary["generator"]["success_diagnostics"]
    assert diagnostics["total_params"] > 0
    assert 0.0 <= diagnostics["decoder_param_fraction"] <= 1.0
    assert "estimated_recurrent_op_fraction" in diagnostics
    assert "phase_mean_abs_displacement" in diagnostics
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_conditional_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_conditional_test",
        output_dir=tmp_path / "mnist_generator_conditional",
        seed=6,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        conditional=True,
        num_classes=10,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        class_moment_weight=0.2,
        prototype_weight=0.1,
        num_oscillators=8,
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert summary["generator"]["conditional"]
    assert "prototype_nearest_accuracy" in summary["generator"]
    assert "success_diagnostics" in summary["generator"]
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_spatial_basis_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_spatial_basis_test",
        output_dir=tmp_path / "mnist_generator_spatial_basis",
        seed=7,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        conditional=True,
        num_classes=10,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        decoder_mode="spatial_basis",
        class_moment_weight=0.2,
        prototype_weight=0.1,
        num_oscillators=9,
        decoder_hidden_dim=12,
        decoder_depth=0,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    diagnostics = summary["generator"]["success_diagnostics"]
    assert summary["generator"]["decoder_mode"] == "spatial_basis"
    assert diagnostics["decoder_param_fraction"] < 0.5
    assert diagnostics["decoder_mode"] == "spatial_basis"
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_local_basis_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_local_basis_test",
        output_dir=tmp_path / "mnist_generator_local_basis",
        seed=8,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        conditional=True,
        num_classes=10,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        decoder_mode="local_basis",
        local_patch_size=3,
        class_moment_weight=0.2,
        prototype_weight=0.1,
        num_oscillators=9,
        decoder_hidden_dim=12,
        decoder_depth=0,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    diagnostics = summary["generator"]["success_diagnostics"]
    assert summary["generator"]["decoder_mode"] == "local_basis"
    assert diagnostics["decoder_mode"] == "local_basis"
    assert 0.0 < diagnostics["decoder_param_fraction"] < 0.75
    assert summary["final_eval_loss"] >= 0.0


def test_resize_conv_generator_decodes_spatial_phase_seed():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=98,
        image_shape=(28, 28),
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        steps=1,
        key=jax.random.PRNGKey(81),
    )
    generated = model(jax.random.PRNGKey(82), 3)
    trace = model.collect_trace(jax.random.PRNGKey(83), 3)

    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 28 * 28)
    assert model.resize_conv_seed_shape == (4, 7, 7)
    assert len(model.resize_conv_layers) == 4
    assert diagnostics["decoder_mode"] == "resize_conv"
    assert diagnostics["decoder_params"] > 0
    assert diagnostics["estimated_decoder_ops_per_sample"] > 0


def test_horn_image_generator_samples_and_traces_state():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        conditioning_mode="phase_shift",
        key=jax.random.PRNGKey(90),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(91), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(92), 3, labels)
    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 64)
    assert bool(jnp.all(jnp.isfinite(generated)))
    assert trace["initial_velocity"].shape == (3, 8)
    assert trace["velocity_trajectory"].shape == (2, 3, 8)
    assert diagnostics["dynamics_family"] == "horn"
    assert diagnostics["state_final_energy"] >= 0.0


def test_generator_settling_metrics_score_multiple_step_depths():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        key=jax.random.PRNGKey(88),
    )
    real = jnp.linspace(0.0, 1.0, 4 * 64, dtype=jnp.float32).reshape(4, 64)

    metrics = compute_generator_settling_metrics(
        model,
        key=jax.random.PRNGKey(89),
        real_images=real,
        sample_count=4,
        batch_size=2,
        settling_steps=(0, 1, 2),
    )

    assert metrics["steps"] == [0, 1, 2]
    assert "step_000" in metrics["by_step"]
    assert "step_002" in metrics["by_step"]
    assert "diversity_ratio_best_step" in metrics
    assert "nearest_real_mse_last_minus_first" in metrics


def test_horn_resize_conv_generator_decodes_spatial_state_seed():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=98,
        image_shape=(28, 28),
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        steps=1,
        key=jax.random.PRNGKey(93),
    )
    generated = model(jax.random.PRNGKey(94), 3)
    trace = model.collect_trace(jax.random.PRNGKey(95), 3)
    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 28 * 28)
    assert model.resize_conv_seed_shape == (4, 7, 7)
    assert diagnostics["dynamics_family"] == "horn"
    assert diagnostics["decoder_mode"] == "resize_conv"
    assert diagnostics["estimated_decoder_ops_per_sample"] > 0


def test_state_mlp_image_generator_is_non_oscillatory_control():
    from oscnet.models import StateMLPImageGenerator

    model = StateMLPImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        conditioning_mode="phase_shift",
        state_mlp_hidden_dim=4,
        key=jax.random.PRNGKey(96),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(97), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(98), 3, labels)
    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 64)
    assert bool(jnp.all(jnp.isfinite(generated)))
    assert model.dynamics_family == "state_mlp"
    assert model.coupling_profile == "none"
    assert diagnostics["dynamics_family"] == "state_mlp"
    assert diagnostics["coupling_density"] == 0.0
    assert diagnostics["transition_params"] > 0
    assert diagnostics["recurrent_params"] == diagnostics["transition_params"]
    assert diagnostics["state_mean_abs_velocity_displacement"] >= 0.0


def test_mnist_generator_resize_conv_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_resize_conv_test",
        output_dir=tmp_path / "mnist_generator_resize_conv",
        seed=82,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        num_oscillators=98,
        decoder_hidden_dim=12,
        decoder_depth=0,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    diagnostics = summary["generator"]["success_diagnostics"]
    assert summary["generator"]["decoder_mode"] == "resize_conv"
    assert summary["generator"]["resize_conv_seed_size"] == 7
    assert diagnostics["decoder_mode"] == "resize_conv"
    assert diagnostics["decoder_params"] > 0
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_horn_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_horn_test",
        output_dir=tmp_path / "mnist_generator_horn",
        seed=83,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="horn",
        conditional=True,
        num_classes=10,
        conditioning_mode="phase_shift",
        readout_mode="mean_relative",
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        num_oscillators=98,
        decoder_hidden_dim=12,
        decoder_depth=0,
        steps=1,
        num_projections=8,
        quality_classifier_epochs=1,
        quality_classifier_dim=8,
        quality_classifier_depth=1,
        eval_sample_count=2,
        train_settling_steps=(0, 1),
        settling_steps=(0, 1),
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    diagnostics = summary["generator"]["success_diagnostics"]
    assert summary["generator"]["dynamics_family"] == "horn"
    assert summary["generator"]["train_settling_steps"] == [0, 1]
    assert summary["generator"]["quality_classifier"]["epochs"] == 1
    assert "classifier_label_accuracy" in summary["generator"]
    assert summary["generator"]["settling"]["steps"] == [0, 1]
    assert diagnostics["dynamics_family"] == "horn"
    assert "state_final_energy" in diagnostics
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_state_mlp_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_state_mlp_test",
        output_dir=tmp_path / "mnist_generator_state_mlp",
        seed=85,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="state_mlp",
        conditional=True,
        num_classes=10,
        conditioning_mode="phase_shift",
        readout_mode="mean_relative",
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        num_oscillators=98,
        decoder_hidden_dim=12,
        decoder_depth=0,
        steps=1,
        state_mlp_hidden_dim=16,
        num_projections=8,
        quality_classifier_epochs=1,
        quality_classifier_dim=8,
        quality_classifier_depth=1,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    diagnostics = summary["generator"]["success_diagnostics"]
    assert summary["generator"]["dynamics_family"] == "state_mlp"
    assert summary["generator"]["state_mlp_hidden_dim"] == 16
    assert diagnostics["dynamics_family"] == "state_mlp"
    assert diagnostics["transition_params"] > 0
    assert diagnostics["coupling_density"] == 0.0
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_learned_feature_drift_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_learned_feature_test",
        output_dir=tmp_path / "mnist_generator_learned_feature",
        seed=84,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="kuramoto",
        conditional=True,
        num_classes=10,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        loss_mode="pixel_feature_drift",
        pixel_drift_weight=0.5,
        feature_drift_weight=1.0,
        feature_drift_mode="learned",
        learned_feature_epochs=1,
        learned_feature_dim=8,
        learned_feature_depth=1,
        num_oscillators=8,
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        data_source="synthetic",
        train_limit=4,
        eval_limit=2,
    )

    result = run_mnist_generator_experiment(config)

    with open(result.paths.metrics / "summary.json") as f:
        summary = json.load(f)
    assert (result.paths.metrics / "feature_classifier.json").exists()
    assert summary["generator"]["loss"] == "pixel_feature_drift"
    assert summary["generator"]["feature_drift_mode"] == "learned"
    assert summary["generator"]["feature_classifier"]["epochs"] == 1
    assert summary["final_eval_feature_drift_loss"] >= 0.0


def test_generator_success_diagnostics_expose_attribution_proxies():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        key=jax.random.PRNGKey(9),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    trace = model.collect_trace(jax.random.PRNGKey(10), 3, labels)

    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=3.0,
    )

    assert diagnostics["total_params"] > diagnostics["decoder_params"]
    assert diagnostics["conditioning_params"] > 0
    assert diagnostics["train_recurrent_dynamics"]
    assert diagnostics["train_conditioning_dynamics"]
    assert diagnostics["trainable_main_recurrent_params"] > 0
    assert diagnostics["trainable_conditioning_params"] > 0
    assert diagnostics["trainable_recurrent_param_fraction"] > 0.0
    assert diagnostics["samples_per_train_second"] == 4.0
    assert diagnostics["phase_mean_abs_displacement"] >= 0.0


def test_generator_success_diagnostics_count_spatial_basis_as_decoder():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=9,
        image_shape=(8, 8),
        decoder_mode="spatial_basis",
        decoder_depth=0,
        steps=2,
        num_classes=3,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        key=jax.random.PRNGKey(11),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    trace = model.collect_trace(jax.random.PRNGKey(12), 3, labels)

    diagnostics = compute_generator_success_diagnostics(model, trace=trace)

    assert diagnostics["decoder_mode"] == "spatial_basis"
    assert diagnostics["decoder_params"] == 19
    assert diagnostics["decoder_param_fraction"] < 0.5


def test_generator_success_diagnostics_count_local_basis_as_decoder():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=9,
        image_shape=(8, 8),
        decoder_mode="local_basis",
        local_patch_size=3,
        decoder_depth=0,
        steps=2,
        num_classes=3,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        key=jax.random.PRNGKey(13),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    trace = model.collect_trace(jax.random.PRNGKey(14), 3, labels)

    diagnostics = compute_generator_success_diagnostics(model, trace=trace)

    assert diagnostics["decoder_mode"] == "local_basis"
    assert diagnostics["decoder_params"] == 163
    assert diagnostics["decoder_param_fraction"] < 0.5


def test_generator_success_diagnostics_report_coupling_profile():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=9,
        image_shape=(8, 8),
        decoder_mode="local_basis",
        local_patch_size=3,
        decoder_depth=0,
        steps=2,
        coupling_profile="distance_decay",
        coupling_length_scale=0.6,
        coupling_floor=0.05,
        coupling_bias_strength=0.1,
        key=jax.random.PRNGKey(15),
    )

    diagnostics = compute_generator_success_diagnostics(model)

    assert diagnostics["coupling_profile"] == "distance_decay"
    assert diagnostics["coupling_length_scale"] == 0.6
    assert diagnostics["coupling_floor"] == 0.05
    assert diagnostics["coupling_bias_strength"] == 0.1
    assert diagnostics["coupling_profile_mean"] < 1.0
    assert diagnostics["coupling_profile_max"] < 1.0
    assert diagnostics["coupling_profile_max"] > 0.05


def test_generator_success_diagnostics_report_sparse_local_coupling_profile():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=16,
        image_shape=(8, 8),
        decoder_mode="local_basis",
        local_patch_size=3,
        decoder_depth=0,
        steps=2,
        coupling_profile="local_radius",
        coupling_length_scale=0.7,
        key=jax.random.PRNGKey(17),
    )

    diagnostics = compute_generator_success_diagnostics(model)

    assert diagnostics["coupling_profile"] == "local_radius"
    assert diagnostics["coupling_density"] < 1.0
    assert diagnostics["coupling_profile_min"] == 0.0
    assert diagnostics["coupling_profile_max"] == 1.0


def test_generator_success_diagnostics_report_split_trainability():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=9,
        image_shape=(8, 8),
        decoder_mode="local_basis",
        local_patch_size=3,
        decoder_depth=0,
        steps=2,
        num_classes=3,
        num_condition_oscillators=4,
        conditioning_mode="class_coupling",
        readout_mode="relative",
        train_recurrent_dynamics=False,
        train_conditioning_dynamics=True,
        key=jax.random.PRNGKey(16),
    )

    diagnostics = compute_generator_success_diagnostics(model)

    assert diagnostics["train_recurrent_dynamics"] is False
    assert diagnostics["train_conditioning_dynamics"] is True
    assert diagnostics["trainable_main_recurrent_params"] == 0
    assert diagnostics["trainable_conditioning_params"] == (
        diagnostics["conditioning_params"]
    )
    assert diagnostics["trainable_recurrent_params"] == (
        diagnostics["trainable_conditioning_params"]
    )
