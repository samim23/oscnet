import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oscnet.analysis.generator_state_prior import (
    fit_class_state_prior,
    generate_from_initial_states,
    sample_class_state_prior,
)
from oscnet.experiments.harness import AutoencoderExperimentConfig
from oscnet.experiments.mnist_autoencoder import load_mnist_data
from oscnet.experiments.mnist_generator import (
    ConvImageFeatureClassifier,
    CURRENT_CIFAR10_RGB_GENERATOR_PRESET,
    CURRENT_CIFAR10_RGB_HIERARCHY_PRESET,
    CURRENT_MNIST_GENERATOR_PRESET,
    MNISTDriftQueue,
    MNISTFeatureClassifier,
    MNISTGeneratorExperimentConfig,
    RECOMMENDED_GENERATOR_PRESET,
    ResidualConvImageFeatureClassifier,
    build_arg_parser,
    build_mnist_generator_model,
    conditional_feature_drift_loss,
    conditional_pixel_drift_loss,
    config_from_args,
    compute_class_prototypes,
    compute_generator_attractor_robustness,
    compute_generator_quality_metrics,
    compute_generator_settling_metrics,
    compute_generator_recovery_metrics,
    compute_generator_robustness_metrics,
    compute_generator_state_fitting_probe,
    compute_generator_state_information_probe,
    compute_generator_success_diagnostics,
    compute_generator_trace_dynamics,
    compute_generator_vertical_intervention_audit,
    coarse_auxiliary_image_loss,
    coarse_readout_consistency_loss,
    downsample_image_batch,
    frequency_statistics_loss,
    generator_distribution_loss,
    generator_loss,
    make_projection_matrix,
    mnist_structural_features,
    patch_sliced_wasserstein_loss,
    parse_args,
    run_mnist_generator_experiment,
    sample_generator_images,
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


def test_downsample_image_batch_preserves_channel_first_rgb_layout():
    red = jnp.ones((4, 4), dtype=jnp.float32)
    green = jnp.ones((4, 4), dtype=jnp.float32) * 2.0
    blue = jnp.ones((4, 4), dtype=jnp.float32) * 3.0
    image = jnp.stack([red, green, blue], axis=0).reshape(1, -1)

    downsampled = downsample_image_batch(
        image,
        image_shape=(4, 4, 3),
        target_size=2,
    )

    expected = jnp.concatenate(
        [
            jnp.ones((4,), dtype=jnp.float32),
            jnp.ones((4,), dtype=jnp.float32) * 2.0,
            jnp.ones((4,), dtype=jnp.float32) * 3.0,
        ]
    ).reshape(1, -1)
    assert jnp.allclose(downsampled, expected)


def test_frequency_statistics_loss_rewards_matching_frequency_statistics():
    checker = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    ).reshape(1, -1)
    smooth = jnp.ones((1, 16), dtype=jnp.float32) * 0.5

    matched = frequency_statistics_loss(
        checker,
        checker,
        image_shape=(4, 4),
    )
    mismatched = frequency_statistics_loss(
        checker,
        smooth,
        image_shape=(4, 4),
    )

    assert matched < 1e-7
    assert mismatched > 1.0


def test_patch_sliced_wasserstein_loss_rewards_matching_local_patches():
    checker = jnp.asarray(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ],
        dtype=jnp.float32,
    ).reshape(1, -1)
    smooth = jnp.ones((1, 16), dtype=jnp.float32) * 0.5
    projections = make_projection_matrix(
        jax.random.PRNGKey(123),
        image_dim=4,
        num_projections=8,
    )

    matched = patch_sliced_wasserstein_loss(
        checker,
        checker,
        projections,
        image_shape=(4, 4),
        patch_size=2,
        patch_sizes=(2,),
        stride=2,
        offsets=(0, 1),
        edge_weight=0.5,
    )
    mismatched = patch_sliced_wasserstein_loss(
        checker,
        smooth,
        projections,
        image_shape=(4, 4),
        patch_size=2,
        patch_sizes=(2,),
        stride=2,
        offsets=(0, 1),
        edge_weight=0.5,
    )

    assert matched < 1e-7
    assert mismatched > matched


def test_generator_quality_metrics_duplicate_rates_use_full_reference_set():
    real = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.25, 0.25],
        ],
        dtype=jnp.float32,
    )
    generated = jnp.asarray([[0.25, 0.25]], dtype=jnp.float32)

    metrics = compute_generator_quality_metrics(real, generated)

    assert metrics["nearest_real_mse_min"] < 1e-8
    assert metrics["duplicate_rate_mse_001"] == 1.0


def test_class_state_prior_samples_means_and_drives_horn_states():
    from oscnet.models import HORNImageGenerator

    labels = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    position = jnp.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [-0.1, -0.2, -0.3, -0.4],
            [-0.2, -0.3, -0.4, -0.5],
        ],
        dtype=jnp.float32,
    )
    velocity = position * 0.5
    states = jnp.concatenate([position, velocity], axis=-1)

    prior = fit_class_state_prior(
        states,
        labels,
        num_classes=2,
        rank=2,
    )
    mean_position, mean_velocity = sample_class_state_prior(
        prior,
        np.asarray([0, 1], dtype=np.int32),
        rng=np.random.default_rng(123),
        mean_only=True,
    )

    assert mean_position.shape == (2, 4)
    assert mean_velocity.shape == (2, 4)
    assert np.allclose(mean_position[0], np.asarray(position[:2]).mean(axis=0))
    assert np.allclose(mean_position[1], np.asarray(position[2:]).mean(axis=0))

    model = HORNImageGenerator(
        num_oscillators=4,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        key=jax.random.PRNGKey(715),
    )
    generated = generate_from_initial_states(
        model,
        mean_position,
        mean_velocity,
        labels=np.asarray([0, 1], dtype=np.int32),
        settle_steps=1,
        batch_size=2,
    )

    assert generated.shape == (2, 16)


def test_sample_generator_images_uses_initial_state_sampler():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=4,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        key=jax.random.PRNGKey(716),
    )
    calls = []

    def sampler(key, batch_size, labels):
        del key
        calls.append((int(batch_size), None if labels is None else labels.shape[0]))
        return (
            jnp.ones((batch_size, 4), dtype=jnp.float32) * 0.1,
            jnp.zeros((batch_size, 4), dtype=jnp.float32),
        )

    labels = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    generated = sample_generator_images(
        model,
        key=jax.random.PRNGKey(717),
        sample_count=3,
        batch_size=2,
        labels=labels,
        initial_state_sampler=sampler,
    )

    assert generated.shape == (3, 16)
    assert calls == [(2, 2), (1, 1)]


def test_generator_state_information_probe_decodes_synthetic_state_signals():
    class DummyModel:
        num_classes = 2

    labels = jnp.asarray([0, 1] * 8, dtype=jnp.int32)
    label_values = labels.astype(jnp.float32)
    alternating = jnp.asarray(
        [
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    generated = []
    for label in label_values:
        base = jnp.ones((4, 4), dtype=jnp.float32) * (0.25 + 0.5 * label)
        generated.append((base + 0.1 * label * alternating).reshape(-1))
    generated = jnp.stack(generated)
    final_theta = jnp.stack(
        [
            label_values,
            1.0 - label_values,
            2.0 * label_values - 1.0,
            jnp.linspace(-0.5, 0.5, labels.shape[0]),
        ],
        axis=-1,
    )
    trace = {
        "initial_theta": jnp.zeros_like(final_theta),
        "initial_velocity": jnp.zeros_like(final_theta),
        "final_theta": final_theta,
        "final_velocity": final_theta * 0.1,
        "generated": generated,
    }

    probe = compute_generator_state_information_probe(
        DummyModel(),
        trace,
        labels=labels,
        image_shape=(4, 4),
        target_size=2,
        ridge=1e-3,
    )

    assert probe["sample_count"] == 16
    assert probe["state_sets"]["fine_final"]["label_accuracy"] >= 0.75
    assert "generated_lowres_r2" in probe["state_sets"]["fine_final"]
    assert "generated_highpass_r2" in probe["state_sets"]["fine_final"]
    assert (
        probe["fine_final_minus_initial_label_accuracy"]
        >= 0.25
    )


def test_generator_state_fitting_probe_scores_frozen_decoder_states():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=4,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        key=jax.random.PRNGKey(390),
    )
    labels = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    real_images = model(jax.random.PRNGKey(391), 4, labels)

    probe = compute_generator_state_fitting_probe(
        model,
        real_images,
        key=jax.random.PRNGKey(392),
        labels=labels,
        image_shape=(4, 4),
        sample_count=4,
        fit_steps=3,
        learning_rate=1e-2,
        settle_steps=(0, 1),
    )

    assert probe["sample_count"] == 4
    assert probe["fit_steps"] == 3
    assert probe["final_mse"] >= 0.0
    assert probe["fit_paired_mse"] >= 0.0
    assert "settle_001_paired_mse" in probe
    assert "fresh_readout" in probe
    assert probe["fresh_readout"]["feature_dim"] == 16


def test_generator_state_fitting_probe_recovery_conditions():
    from oscnet.models import HORNImageGenerator
    from oscnet.models.generative.state_mlp import StateMLPImageGenerator

    common = dict(
        num_oscillators=4,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
    )
    labels = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    for model in (
        HORNImageGenerator(**common, key=jax.random.PRNGKey(410)),
        StateMLPImageGenerator(**common, key=jax.random.PRNGKey(411)),
    ):
        real_images = model(jax.random.PRNGKey(412), 4, labels)
        probe = compute_generator_state_fitting_probe(
            model,
            real_images,
            key=jax.random.PRNGKey(413),
            labels=labels,
            image_shape=(4, 4),
            sample_count=4,
            fit_steps=3,
            learning_rate=1e-2,
            settle_steps=(0, 1),
            recovery_noise_scales=(0.25, 0.5),
            recovery_settle_steps=(1, 2),
            occlusion_fractions=(0.25,),
            return_images=True,
        )

        for scale_index in (0, 1):
            prefix = f"recover_n{scale_index}"
            assert probe[f"{prefix}_noise_scale"] > 0.0
            assert probe[f"{prefix}_state_displacement_rms"] > 0.0
            assert f"{prefix}_settle_000_paired_mse" in probe
            for step in (1, 2):
                step_prefix = f"{prefix}_settle_{step:03d}"
                assert f"{step_prefix}_paired_mse" in probe
                assert f"{step_prefix}_repair_delta_vs_noisy" in probe
                assert f"{step_prefix}_excess_mse_vs_clean_fit" in probe
                assert probe[f"{step_prefix}_state_return_ratio"] > 0.0
                assert probe[f"{step_prefix}_decode_drift_from_fit"] >= 0.0
        assert "occlusion_unsupported" not in probe
        assert probe["occl_f0_fraction"] == 0.25
        assert probe["occl_f0_patch_sites"] == 1
        assert probe["occl_f0_settle_000_paired_mse"] >= 0.0
        assert probe["occl_f0_settle_000_occluded_region_mse"] >= 0.0
        assert probe["occl_f0_settle_000_intact_region_mse"] >= 0.0
        for step in (1, 2):
            step_prefix = f"occl_f0_settle_{step:03d}"
            assert probe[f"{step_prefix}_paired_mse"] >= 0.0
            assert f"{step_prefix}_repair_delta_vs_occluded" in probe
            assert probe[f"{step_prefix}_occluded_region_mse"] >= 0.0
            assert probe[f"{step_prefix}_intact_region_mse"] >= 0.0
        images = probe["images"]
        assert set(images) >= {
            "target",
            "fit",
            "recover_n0_settle_000",
            "recover_n0_settle_001",
            "recover_n1_settle_002",
            "occl_f0_settle_000",
            "occl_f0_settle_002",
        }
        assert images["fit"].shape == (4, 16)


def _tiny_anchor_horn(key=None):
    from oscnet.models import HORNImageGenerator

    return HORNImageGenerator(
        num_oscillators=4,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        state_anchor_encoder_enabled=True,
        state_anchor_num_spatial_sites=4,
        state_anchor_num_modes=1,
        key=key if key is not None else jax.random.PRNGKey(420),
    )


def test_occlude_image_batch_masks_expected_fraction():
    from oscnet.experiments.mnist_generator.common import occlude_image_batch

    images = jnp.ones((6, 8 * 8 * 3), dtype=jnp.float32)
    occluded, mask = occlude_image_batch(
        images,
        key=jax.random.PRNGKey(431),
        image_shape=(8, 8, 3),
        fraction=0.25,
        patches=1,
        probability=1.0,
    )
    assert occluded.shape == images.shape
    assert mask.shape == (6, 8, 8)
    per_sample = np.asarray(mask).reshape(6, -1).mean(axis=1)
    # One square patch of exactly 4x4 = 25% of an 8x8 grid.
    assert np.allclose(per_sample, 0.25)
    occluded_np = np.asarray(occluded).reshape(6, 3, 8, 8)
    assert np.all(occluded_np[np.asarray(mask)[:, None, :, :].repeat(3, 1)] == 0.0)

    _, scattered_mask = occlude_image_batch(
        images,
        key=jax.random.PRNGKey(432),
        image_shape=(8, 8, 3),
        fraction=0.25,
        patches=4,
        probability=1.0,
    )
    scattered = np.asarray(scattered_mask).reshape(6, -1).mean(axis=1)
    # Patches may overlap, so scattered coverage is bounded by the target.
    assert np.all(scattered > 0.0)
    assert np.all(scattered <= 0.25 + 1e-6)

    _, none_mask = occlude_image_batch(
        images,
        key=jax.random.PRNGKey(433),
        image_shape=(8, 8, 3),
        fraction=0.25,
        patches=1,
        probability=0.0,
    )
    assert not np.any(np.asarray(none_mask))


def test_state_anchor_loss_supports_occlusion_and_clean_term():
    from oscnet.experiments.mnist_generator.runner import _state_anchor_image_loss

    model = _tiny_anchor_horn()
    real_batch = jax.random.uniform(jax.random.PRNGKey(434), (4, 16))

    loss = _state_anchor_image_loss(
        model,
        real_batch,
        key=jax.random.PRNGKey(435),
        state_anchor_weight=1.0,
        state_anchor_steps=(1,),
        state_anchor_noise_scale=0.1,
        state_anchor_mode="settle",
        state_anchor_occlusion_fraction=0.25,
        state_anchor_occlusion_patches=2,
        state_anchor_occlusion_probability=1.0,
        state_anchor_clean_weight=0.5,
    )
    assert jnp.isfinite(loss)
    assert float(loss) > 0.0

    baseline = _state_anchor_image_loss(
        model,
        real_batch,
        key=jax.random.PRNGKey(435),
        state_anchor_weight=1.0,
        state_anchor_steps=(1,),
        state_anchor_noise_scale=0.1,
        state_anchor_mode="settle",
    )
    assert jnp.isfinite(baseline)


def test_compute_generator_recovery_metrics_scores_conditions():
    model = _tiny_anchor_horn()
    real_images = jax.random.uniform(jax.random.PRNGKey(436), (8, 16))

    metrics = compute_generator_recovery_metrics(
        model,
        real_images,
        key=jax.random.PRNGKey(437),
        image_shape=(4, 4),
        sample_count=8,
        noise_scales=(0.25,),
        occlusion_fractions=(0.25,),
        occlusion_patch_counts=(1,),
        settle_steps=(0, 1),
    )

    assert metrics["sample_count"] == 8
    for prefix in ("clean_k000", "clean_k001", "noise_s0_k001"):
        assert metrics[f"{prefix}_paired_mse"] >= 0.0
        assert np.isfinite(metrics[f"{prefix}_psnr"])
        assert -1.0 <= metrics[f"{prefix}_ssim"] <= 1.0
    assert metrics["noise_s0_scale"] == 0.25
    assert metrics["occl_f0_p1_fraction"] == 0.25
    for step in (0, 1):
        assert metrics[f"occl_f0_p1_k{step:03d}_occluded_region_mse"] >= 0.0
        assert metrics[f"occl_f0_p1_k{step:03d}_intact_region_mse"] >= 0.0


def test_compute_generator_robustness_metrics_scores_stressors():
    model = _tiny_anchor_horn()
    real_images = jax.random.uniform(jax.random.PRNGKey(438), (8, 16))

    metrics = compute_generator_robustness_metrics(
        model,
        real_images,
        key=jax.random.PRNGKey(439),
        image_shape=(4, 4),
        sample_count=8,
        settle_step=1,
        weight_noise_scales=(0.05, 0.2),
        quant_bits=(8, 4),
        ood_occlusion_fractions=(0.25, 0.5),
        weight_noise_draws=2,
    )

    assert metrics["sample_count"] == 8
    assert metrics["settle_step"] == 1
    assert metrics["baseline_occluded_region_mse"] >= 0.0
    assert np.isfinite(metrics["baseline_clean_psnr"])
    for scale_index in (0, 1):
        assert metrics[f"wnoise_s{scale_index}_occluded_region_mse"] >= 0.0
        assert np.isfinite(metrics[f"wnoise_s{scale_index}_clean_psnr"])
    for bits in (8, 4):
        assert metrics[f"quant_b{bits}_occluded_region_mse"] >= 0.0
    for frac_index in (0, 1):
        assert metrics[f"ood_occl_f{frac_index}_occluded_region_mse"] >= 0.0
    # A strong weight perturbation should not improve fill-in over baseline.
    assert (
        metrics["wnoise_s1_occluded_region_mse"]
        >= metrics["baseline_occluded_region_mse"] - 1e-6
    )


def test_robustness_weight_perturbation_changes_outputs():
    from oscnet.experiments.mnist_generator.metrics import (
        _perturb_model_weights,
        _quantize_model_weights,
    )

    model = _tiny_anchor_horn()
    real = jax.random.uniform(jax.random.PRNGKey(440), (4, 16))
    labels = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
    base = model(jax.random.PRNGKey(441), 4, labels)

    perturbed = _perturb_model_weights(
        model, key=jax.random.PRNGKey(442), noise_scale=0.3
    )
    quantized = _quantize_model_weights(model, bits=3)
    out_perturbed = perturbed(jax.random.PRNGKey(441), 4, labels)
    out_quant = quantized(jax.random.PRNGKey(441), 4, labels)

    assert base.shape == out_perturbed.shape == out_quant.shape
    assert not jnp.allclose(base, out_perturbed)
    assert not jnp.allclose(base, out_quant)


def _tiny_coarse_carrier_horn(key=None):
    from oscnet.models.generative.coarse_horn import CoarseToFineHORNImageGenerator

    return CoarseToFineHORNImageGenerator(
        num_oscillators=16,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(4, 4),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=0,
        resize_conv_min_channels=2,
        steps=3,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        num_coarse_oscillators=4,
        coarse_coupling_profile="dense",
        coarse_to_fine_strength=0.5,
        coarse_to_fine_profile="dense",
        state_anchor_encoder_enabled=True,
        state_anchor_num_spatial_sites=16,
        state_anchor_num_modes=1,
        key=key if key is not None else jax.random.PRNGKey(451),
    )


def test_coarse_carrier_evolve_state_uses_pooled_carrier():
    from oscnet.models import HORNImageGenerator

    model = _tiny_coarse_carrier_horn()
    position = jax.random.uniform(jax.random.PRNGKey(452), (3, 16)) - 0.5
    velocity = jax.random.uniform(jax.random.PRNGKey(453), (3, 16)) - 0.5

    final_position, final_velocity = model.evolve_state((position, velocity), None)
    assert final_position.shape == (3, 16)
    assert final_velocity.shape == (3, 16)
    assert jnp.all(jnp.isfinite(final_position))

    # A coarse carrier should change the settled fine state relative to a
    # fine-only evolution with the same recurrent parameters.
    fine_only = HORNImageGenerator(
        num_oscillators=16,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(4, 4),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=0,
        resize_conv_min_channels=2,
        steps=3,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        key=jax.random.PRNGKey(451),
    )
    fine_only_position, _ = fine_only.evolve_state((position, velocity), None)
    assert not jnp.allclose(final_position, fine_only_position)


def test_coarse_carrier_recovery_preset_builds_and_evolves():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_coarse_carrier_retinotopic_recovery_mixed",
                "--steps",
                "2",
                "--state-anchor-steps",
                "2",
            ]
        )
    )
    assert config.model_family == "coarse_horn"
    assert config.num_oscillators == 256
    assert config.num_coarse_oscillators == 16
    assert config.coarse_to_fine_strength == 0.5
    assert config.state_anchor_occlusion_fraction == 0.25

    model = build_mnist_generator_model(config, jax.random.PRNGKey(454))
    assert model.state_anchor_encoder is not None
    images = jnp.zeros((2, 32 * 32 * 3), dtype=jnp.float32)
    position, velocity = model.encode_image_state(images)
    settled_position, settled_velocity = model.evolve_state(
        (position, velocity), None
    )
    assert settled_position.shape == (2, 256)
    assert settled_velocity.shape == (2, 256)


def test_coarse_multimode_carrier_builds_and_evolves():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_slow_carrier_retinotopic_recovery_mixed",
                "--steps",
                "2",
                "--state-anchor-steps",
                "2",
            ]
        )
    )
    assert config.model_family == "coarse_multimode_horn"
    assert config.multimode_num_modes == 2
    assert config.coarse_frequency_scale == 0.5
    assert config.num_coarse_oscillators == 16

    model = build_mnist_generator_model(config, jax.random.PRNGKey(460))
    assert model.dynamics_family == "coarse_multimode_horn"
    assert model.num_oscillators == 512
    assert model.num_modes == 2
    assert model.coarse_frequency_scale == 0.5
    assert model.state_anchor_encoder is not None

    images = jnp.zeros((2, 32 * 32 * 3), dtype=jnp.float32)
    position, velocity = model.encode_image_state(images)
    settled_position, settled_velocity = model.evolve_state(
        (position, velocity), None
    )
    assert settled_position.shape == (2, 512)
    assert jnp.all(jnp.isfinite(settled_position))
    generated = model.decode_state(settled_position, settled_velocity)
    assert generated.shape == (2, 32 * 32 * 3)

    sampled = model(jax.random.PRNGKey(461), 2, jnp.array([0, 1]))
    assert sampled.shape == (2, 32 * 32 * 3)


def test_multimode_ablation_presets_set_frequency_scales():
    eqfreq = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_eqfreq_retinotopic_recovery_mixed",
            ]
        )
    )
    wide = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_wide_retinotopic_recovery_mixed",
            ]
        )
    )
    mm4 = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode4_retinotopic_recovery_mixed",
            ]
        )
    )
    assert eqfreq.multimode_frequency_scales == (1.0, 1.0)
    assert wide.multimode_frequency_scales == (0.5, 1.5)
    assert mm4.multimode_num_modes == 4
    assert mm4.multimode_frequency_scales == ()
    for cfg in (eqfreq, wide, mm4):
        assert cfg.model_family == "multimode_horn"
        assert cfg.state_anchor_occlusion_fraction == 0.25


def test_recovery_training_presets_configure_corruption_and_eval():
    noise_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_recovery_noise",
            ]
        )
    )
    mixed_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_recovery_mixed",
            ]
        )
    )
    state_mlp_mixed_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_state_mlp_retinotopic_recovery_mixed",
            ]
        )
    )

    assert noise_config.state_anchor_weight == 1.0
    assert noise_config.state_anchor_noise_scale == 0.25
    assert noise_config.state_anchor_clean_weight == 0.5
    assert noise_config.state_anchor_occlusion_fraction == 0.0
    assert noise_config.recovery_eval_sample_count == 256

    assert mixed_config.state_anchor_occlusion_fraction == 0.25
    assert mixed_config.state_anchor_occlusion_patches == 4
    assert mixed_config.state_anchor_occlusion_probability == 0.5
    assert mixed_config.state_anchor_clean_weight == 0.5

    assert state_mlp_mixed_config.model_family == "state_mlp"
    assert state_mlp_mixed_config.num_oscillators == 512
    assert state_mlp_mixed_config.state_anchor_occlusion_fraction == 0.25


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
    assert parsed.coupling_normalization == "none"
    assert parsed.coupling_length_scale == 0.24
    assert parsed.train_settling_steps == (8, 16, 32)
    assert parsed.run.epochs == 1
    assert parsed.train_limit == 8

    probe_args = config_from_args(
        parse_args(
            [
                "--state-probe-sample-count",
                "24",
                "--state-probe-target-size",
                "4",
                "--state-probe-ridge",
                "0.01",
            ]
        )
    )
    assert probe_args.state_probe_sample_count == 24
    assert probe_args.state_probe_target_size == 4
    assert probe_args.state_probe_ridge == 0.01


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


def test_current_generator_aliases_are_stable_winners():
    assert CURRENT_MNIST_GENERATOR_PRESET == RECOMMENDED_GENERATOR_PRESET

    cifar = config_from_args(
        parse_args(["--preset", CURRENT_CIFAR10_RGB_GENERATOR_PRESET])
    )
    assert cifar.model_family == "horn"
    assert cifar.dataset_name == "cifar10_rgb"
    assert cifar.image_shape == (32, 32, 3)
    assert cifar.coupling_normalization == "row_sum"
    assert cifar.loss_mode == "pixel_feature_drift"
    assert cifar.train_limit == 2000
    assert cifar.run.output_dir.name.endswith("cifar10_rgb_current")

    hierarchy = config_from_args(
        parse_args(["--preset", CURRENT_CIFAR10_RGB_HIERARCHY_PRESET])
    )
    assert hierarchy.model_family == "multiscale_horn"
    assert hierarchy.dataset_name == "cifar10_rgb"
    assert hierarchy.coarse_auxiliary_loss_mode == "distributional"
    assert hierarchy.multiscale_feedback_source_gate == "weighted"
    assert hierarchy.multiscale_feedback_source_mix == (0.5, 0.5)
    assert hierarchy.coarse_readout_consistency_weight == 0.0
    assert hierarchy.run.output_dir.name.endswith("cifar10_rgb_hierarchy_lead")


def test_cifar_hierarchy_state_residual_readout_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_hierarchy_state_residual005",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "4",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(123))
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(124), 4, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert generated.shape == (4, 32 * 32 * 3)
    assert model.state_residual_readout_strength == 0.05
    assert model.state_residual_readout_weight.shape[2] == 3
    assert diagnostics["state_residual_readout_params"] > 0
    assert diagnostics["state_residual_readout_strength"] == 0.05


def test_cifar_current_resonant_readout_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_resonant005",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "4",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(125))
    labels = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(126), 4, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert generated.shape == (4, 32 * 32 * 3)
    assert model.resonant_readout_strength == 0.05
    assert model.resonant_readout_weight.shape == (9, 3, 25)
    assert diagnostics["resonant_readout_params"] > 0
    assert diagnostics["resonant_readout_strength"] == 0.05


def test_cifar_current_n512_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_n512_resonant005",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(127))
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(128), 2, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert generated.shape == (2, 32 * 32 * 3)
    assert model.num_oscillators == 512
    assert model.resize_conv_seed_shape == (16, 8, 8)
    assert model.resonant_readout_strength == 0.05
    assert diagnostics["resonant_readout_params"] == 675
    assert diagnostics["recurrent_params"] > 256 * 256


def test_cifar_current_multimode_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(129))
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(130), 2, labels)
    diagnostics = compute_generator_success_diagnostics(model)
    profile = model.coupling_profile_matrix()

    assert generated.shape == (2, 32 * 32 * 3)
    assert model.dynamics_family == "multimode_horn"
    assert model.num_spatial_sites == 256
    assert model.num_modes == 2
    assert model.num_oscillators == 512
    assert model.mode_frequency_scales == (0.75, 1.35)
    assert profile.shape == (512, 512)
    assert diagnostics["dynamics_family"] == "multimode_horn"
    assert diagnostics["num_spatial_sites"] == 256
    assert diagnostics["num_modes"] == 2
    assert diagnostics["mode_coupling_strength"] == 0.25
    assert diagnostics["recurrent_params"] > 256 * 256


def _build_recovery_model(preset_name, key):
    config = config_from_args(
        parse_args(
            [
                "--preset",
                preset_name,
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    )
    return config, build_mnist_generator_model(config, key)


def test_single_fractal_coupling_is_dense_and_nonlocal():
    local_config, local_model = _build_recovery_model(
        "sparse_horn_cifar10_rgb_current_single_retinotopic_recovery_mixed",
        jax.random.PRNGKey(716),
    )
    config, model = _build_recovery_model(
        "sparse_horn_cifar10_rgb_current_single_fractal_"
        "retinotopic_recovery_mixed",
        jax.random.PRNGKey(717),
    )
    assert local_config.coupling_profile == "local_radius"
    assert config.coupling_profile == "fractal"
    profile = model.coupling_profile_matrix()
    local_profile = local_model.coupling_profile_matrix()
    n = profile.shape[0]
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(718), 2, labels)

    assert generated.shape == (2, 32 * 32 * 3)
    # Fractal coupling keeps direct long-range links between distant corner
    # sites; the sparse local profile leaves them disconnected.
    assert float(profile[0, n - 1]) > 0.0
    assert float(local_profile[0, n - 1]) == 0.0
    assert float(jnp.count_nonzero(profile)) == float(n * n - n)
    assert float(jnp.count_nonzero(profile)) > float(
        jnp.count_nonzero(local_profile)
    )


def test_multimode_dense_and_fractal_are_more_connected_than_local():
    _, local_model = _build_recovery_model(
        "sparse_horn_cifar10_rgb_current_multimode2_"
        "retinotopic_recovery_mixed",
        jax.random.PRNGKey(720),
    )
    dense_config, dense_model = _build_recovery_model(
        "sparse_horn_cifar10_rgb_current_multimode2_dense_"
        "retinotopic_recovery_mixed",
        jax.random.PRNGKey(721),
    )
    fractal_config, fractal_model = _build_recovery_model(
        "sparse_horn_cifar10_rgb_current_multimode2_fractal_"
        "retinotopic_recovery_mixed",
        jax.random.PRNGKey(722),
    )
    assert dense_config.coupling_profile == "dense"
    assert fractal_config.coupling_profile == "fractal"

    local_nonzero = float(jnp.count_nonzero(local_model.coupling_profile_matrix()))
    dense_nonzero = float(jnp.count_nonzero(dense_model.coupling_profile_matrix()))
    fractal_nonzero = float(
        jnp.count_nonzero(fractal_model.coupling_profile_matrix())
    )

    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    assert dense_model(jax.random.PRNGKey(723), 2, labels).shape == (
        2,
        32 * 32 * 3,
    )
    assert dense_nonzero > local_nonzero
    assert fractal_nonzero > local_nonzero


def test_cifar_current_multimode_retinotopic_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(131))
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(132), 2, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert generated.shape == (2, 32 * 32 * 3)
    assert model.dynamics_family == "multimode_horn"
    assert model.resize_conv_seed_layout == "retinotopic"
    assert model.resize_conv_seed_shape == (4, 16, 16)
    assert model.resize_conv_upsamples == 1
    assert diagnostics["resize_conv_seed_layout"] == "retinotopic"


def test_cifar_current_multimode_retinotopic_anchor_preset_builds_encoder():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor010",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
                "--state-anchor-steps",
                "1",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(135))
    images = jnp.zeros((2, 32 * 32 * 3), dtype=jnp.float32)
    position, velocity = model.encode_image_state(images)
    generated = model.decode_state(position, velocity)
    diagnostics = compute_generator_success_diagnostics(model)

    assert config.state_anchor_weight == 0.10
    assert config.state_anchor_mode == "settle"
    assert model.state_anchor_encoder is not None
    assert position.shape == (2, 512)
    assert velocity.shape == (2, 512)
    assert generated.shape == (2, 32 * 32 * 3)
    assert diagnostics["state_anchor_encoder_enabled"] is True
    assert diagnostics["state_anchor_encoder_params"] > 0


def test_cifar_current_state_prior_training_presets_are_explicit_opt_ins():
    global_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_global",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--epochs",
                "1",
            ]
        )
    )
    global_patch_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_global_patch005",
            ]
        )
    )
    class_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class",
                "--state-prior-rank",
                "4",
                "--state-prior-noise-scale",
                "0.5",
                "--state-prior-start-epoch",
                "3",
            ]
        )
    )
    class_patch_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class_patch005",
            ]
        )
    )
    state_mlp_config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_state_mlp_retinotopic_anchor030_prior_class_patch005",
                "--steps",
                "1",
                "--state-anchor-steps",
                "1",
            ]
        )
    )

    assert global_config.state_prior_sampling_mode == "global"
    assert global_config.state_prior_rank == 32
    assert global_config.state_prior_start_epoch == 2
    assert global_config.state_anchor_weight == 0.30
    assert global_patch_config.state_prior_sampling_mode == "global"
    assert global_patch_config.patch_objective_weight == 0.05
    assert global_patch_config.patch_objective_offsets == (0, 2)

    assert class_config.state_prior_sampling_mode == "class"
    assert class_config.state_prior_rank == 4
    assert class_config.state_prior_noise_scale == 0.5
    assert class_config.state_prior_start_epoch == 3
    assert class_patch_config.state_prior_sampling_mode == "class"
    assert class_patch_config.patch_objective_weight == 0.05
    assert class_patch_config.patch_objective_offsets == (0, 2)
    assert state_mlp_config.model_family == "state_mlp"
    assert state_mlp_config.num_oscillators == 512
    assert state_mlp_config.state_mlp_hidden_dim == 128

    model = build_mnist_generator_model(global_config, jax.random.PRNGKey(136))
    assert model.state_anchor_encoder is not None
    assert model.dynamics_family == "multimode_horn"
    state_mlp_model = build_mnist_generator_model(
        state_mlp_config,
        jax.random.PRNGKey(137),
    )
    assert state_mlp_model.state_anchor_encoder is not None
    assert state_mlp_model.dynamics_family == "state_mlp"
    assert state_mlp_model.num_spatial_sites == 256
    assert state_mlp_model.num_modes == 2


def test_cifar_single_mode_retinotopic_seed4_preset_builds_rgb_model():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_current_retinotopic_seed4_ch30",
                "--train-limit",
                "8",
                "--eval-limit",
                "4",
                "--batch-size",
                "2",
                "--steps",
                "1",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(133))
    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(134), 2, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert generated.shape == (2, 32 * 32 * 3)
    assert model.dynamics_family == "horn"
    assert model.resize_conv_seed_layout == "retinotopic"
    assert model.resize_conv_seed_min_channels == 4
    assert model.resize_conv_seed_shape == (4, 16, 16)
    assert model.resize_conv_min_channels == 30
    assert diagnostics["resize_conv_seed_layout"] == "retinotopic"
    assert diagnostics["resize_conv_seed_min_channels"] == 4


def test_generator_cli_accepts_coupling_normalization():
    parsed = config_from_args(
        parse_args(
            [
                "--model-family",
                "horn",
                "--coupling-profile",
                "distance_decay",
                "--coupling-normalization",
                "row_sum",
                "--coupling-length-scale",
                "0.24",
                "--coupling-floor",
                "0.05",
            ]
        )
    )

    assert parsed.coupling_profile == "distance_decay"
    assert parsed.coupling_normalization == "row_sum"
    assert parsed.coupling_length_scale == 0.24
    assert parsed.coupling_floor == 0.05


def test_generator_cli_accepts_coarse_horn_options():
    parsed = config_from_args(
        parse_args(
            [
                "--model-family",
                "coarse_horn",
                "--num-coarse-oscillators",
                "9",
                "--coarse-coupling-profile",
                "distance_decay",
                "--coarse-coupling-normalization",
                "row_sum",
                "--coarse-coupling-length-scale",
                "0.5",
                "--coarse-to-fine-strength",
                "1.25",
                "--coarse-to-fine-profile",
                "local_radius",
                "--coarse-to-fine-normalization",
                "row_sum",
                "--coarse-to-fine-length-scale",
                "0.5",
                "--coarse-to-fine-floor",
                "0.05",
                "--coarse-conditioning-strength",
                "0.75",
                "--output-feedback-mode",
                "image",
                "--output-feedback-strength",
                "0.5",
                "--output-feedback-init-scale",
                "0.04",
                "--output-feedback-basis-sigma",
                "0.2",
            ]
        )
    )

    assert parsed.model_family == "coarse_horn"
    assert parsed.num_coarse_oscillators == 9
    assert parsed.coarse_coupling_profile == "distance_decay"
    assert parsed.coarse_coupling_normalization == "row_sum"
    assert parsed.coarse_coupling_length_scale == 0.5
    assert parsed.coarse_to_fine_strength == 1.25
    assert parsed.coarse_to_fine_profile == "local_radius"
    assert parsed.coarse_to_fine_normalization == "row_sum"
    assert parsed.coarse_to_fine_length_scale == 0.5
    assert parsed.coarse_to_fine_floor == 0.05
    assert parsed.coarse_conditioning_strength == 0.75
    assert parsed.output_feedback_mode == "image"
    assert parsed.output_feedback_strength == 0.5
    assert parsed.output_feedback_init_scale == 0.04
    assert parsed.output_feedback_basis_sigma == 0.2


def test_generator_cli_accepts_multiscale_horn_options():
    parsed = config_from_args(
        parse_args(
            [
                "--model-family",
                "multiscale_horn",
                "--multiscale-layer-sizes",
                "4,16",
                "--multiscale-frequency-scales",
                "0.5,0.8",
                "--multiscale-coupling-profile",
                "distance_decay",
                "--multiscale-coupling-normalization",
                "row_sum",
                "--multiscale-coupling-length-scale",
                "0.6",
                "--multiscale-coupling-floor",
                "0.05",
                "--multiscale-vertical-strength",
                "0.25",
                "--multiscale-feedback-strength",
                "0.1",
                "--multiscale-vertical-profile",
                "local_radius",
                "--multiscale-vertical-normalization",
                "row_sum",
                "--multiscale-vertical-length-scale",
                "0.8",
                "--multiscale-vertical-floor",
                "0.02",
                "--multiscale-vertical-phase-lag",
                "0.3",
                "--multiscale-feedback-phase-lag",
                "-0.2",
                "--multiscale-vertical-signal-scale",
                "10.0",
                "--multiscale-feedback-signal-mode",
                "state",
                "--multiscale-feedback-source-gate",
                "conditioning",
                "--multiscale-feedback-source-mix",
                "0.75,0.25",
                "--multiscale-vertical-target-gate",
                "conditioning",
                "--multiscale-vertical-soft-gate-floor",
                "0.25",
                "--multiscale-vertical-mode",
                "dual_gain",
                "--multiscale-vertical-gain-target",
                "coupling",
                "--multiscale-vertical-gain-normalization",
                "center_rms",
                "--multiscale-vertical-gain-target-std",
                "0.015",
                "--multiscale-vertical-broad-gain-scale",
                "0.75",
                "--multiscale-vertical-selective-gain-scale",
                "1.25",
                "--multiscale-vertical-schedule",
                "linear_ramp",
                "--multiscale-vertical-onset-step",
                "3",
                "--multiscale-vertical-ramp-steps",
                "5",
                "--multiscale-conditioning-strength",
                "0.75",
                "--multiscale-auxiliary-readout-layer",
                "1",
                "--multiscale-readout-fusion-strength",
                "0.25",
                "--multiscale-readout-gate-mode",
                "seed_film",
                "--multiscale-readout-gate-strength",
                "0.10",
                "--multiscale-readout-gate-init-scale",
                "0.01",
                "--coarse-auxiliary-weight",
                "0.05",
                "--coarse-auxiliary-target-size",
                "8",
                "--coarse-auxiliary-loss-mode",
                "distributional",
                "--coarse-readout-consistency-weight",
                "0.05",
                "--coarse-readout-consistency-onset-epoch",
                "5",
                "--frequency-objective-weight",
                "0.03",
                "--frequency-objective-edge-weight",
                "0.5",
                "--patch-objective-weight",
                "0.07",
                "--patch-objective-patch-size",
                "3",
                "--patch-objective-patch-sizes",
                "3,5",
                "--patch-objective-stride",
                "2",
                "--patch-objective-offsets",
                "0,1",
                "--patch-objective-projections",
                "12",
                "--patch-objective-edge-weight",
                "0.4",
                "--vertical-audit-modes",
                "normal,zero,shuffle,flip,scale025",
                "--vertical-audit-sample-count",
                "32",
            ]
        )
    )

    assert parsed.model_family == "multiscale_horn"
    assert parsed.multiscale_layer_sizes == (4, 16)
    assert parsed.multiscale_frequency_scales == (0.5, 0.8)
    assert parsed.multiscale_coupling_profile == "distance_decay"
    assert parsed.multiscale_coupling_normalization == "row_sum"
    assert parsed.multiscale_coupling_length_scale == 0.6
    assert parsed.multiscale_coupling_floor == 0.05
    assert parsed.multiscale_vertical_strength == 0.25
    assert parsed.multiscale_feedback_strength == 0.1
    assert parsed.multiscale_vertical_profile == "local_radius"
    assert parsed.multiscale_vertical_normalization == "row_sum"
    assert parsed.multiscale_vertical_length_scale == 0.8
    assert parsed.multiscale_vertical_floor == 0.02
    assert parsed.multiscale_vertical_phase_lag == 0.3
    assert parsed.multiscale_feedback_phase_lag == -0.2
    assert parsed.multiscale_vertical_signal_scale == 10.0
    assert parsed.multiscale_feedback_signal_mode == "state"
    assert parsed.multiscale_feedback_source_gate == "conditioning"
    assert parsed.multiscale_feedback_source_mix == (0.75, 0.25)
    assert parsed.multiscale_vertical_target_gate == "conditioning"
    assert parsed.multiscale_vertical_soft_gate_floor == 0.25
    assert parsed.multiscale_vertical_mode == "dual_gain"
    assert parsed.multiscale_vertical_gain_target == "coupling"
    assert parsed.multiscale_vertical_gain_normalization == "center_rms"
    assert parsed.multiscale_vertical_gain_target_std == 0.015
    assert parsed.multiscale_vertical_broad_gain_scale == 0.75
    assert parsed.multiscale_vertical_selective_gain_scale == 1.25
    assert parsed.multiscale_vertical_schedule == "linear_ramp"
    assert parsed.multiscale_vertical_onset_step == 3
    assert parsed.multiscale_vertical_ramp_steps == 5
    assert parsed.multiscale_conditioning_strength == 0.75
    assert parsed.multiscale_auxiliary_readout_layer == 1
    assert parsed.multiscale_readout_fusion_strength == 0.25
    assert parsed.multiscale_readout_gate_mode == "seed_film"
    assert parsed.multiscale_readout_gate_strength == 0.10
    assert parsed.multiscale_readout_gate_init_scale == 0.01
    assert parsed.coarse_auxiliary_weight == 0.05
    assert parsed.coarse_auxiliary_target_size == 8
    assert parsed.coarse_auxiliary_loss_mode == "distributional"
    assert parsed.coarse_readout_consistency_weight == 0.05
    assert parsed.coarse_readout_consistency_onset_epoch == 5
    assert parsed.frequency_objective_weight == 0.03
    assert parsed.frequency_objective_edge_weight == 0.5
    assert parsed.patch_objective_weight == 0.07
    assert parsed.patch_objective_patch_size == 3
    assert parsed.patch_objective_patch_sizes == (3, 5)
    assert parsed.patch_objective_stride == 2
    assert parsed.patch_objective_offsets == (0, 1)
    assert parsed.patch_objective_projections == 12
    assert parsed.patch_objective_edge_weight == 0.4
    assert parsed.vertical_audit_modes == (
        "normal",
        "zero",
        "shuffle",
        "flip",
        "scale025",
    )
    assert parsed.vertical_audit_sample_count == 32


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
        "sparse_horn_mnist_dynamics_quality_dist001": "horn",
        "sparse_horn_mnist_dynamics_quality_dist0025": "horn",
        "sparse_horn_mnist_dynamics_quality_dist005": "horn",
        "sparse_horn_mnist_recommended_no_main_coupling": "horn",
        "sparse_horn_mnist_recommended_frozen_recurrent": "horn",
        "sparse_horn_mnist_recommended_frozen_conditioning": "horn",
        "sparse_horn_mnist_recommended_frozen": "frozen_horn",
        "sparse_horn_mnist_recommended_decoder_only": "horn_decoder_only",
        "sparse_horn_mnist_recommended_step1": "horn",
        "sparse_horn_mnist_class_coupling_strong_frozen": "frozen_horn",
        "sparse_horn_mnist_class_coupling_strong_decoder_only": "horn_decoder_only",
        "sparse_horn_mnist_state_mlp_class_coupling_strong": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strength8": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist005": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01": "state_mlp",
        "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class": "state_mlp",
        "sparse_horn_fashion_mnist_recommended": "horn",
        "sparse_horn_fashion_mnist_state_mlp_strength8": "state_mlp",
        "sparse_horn_fashion_mnist_state_mlp_strength8_dist005": "state_mlp",
        "sparse_horn_fashion_mnist_recommended_ch16": "horn",
        "sparse_horn_fashion_mnist_state_mlp_strength8_ch16": "state_mlp",
        "sparse_horn_fashion_mnist_recommended_dist0025": "horn",
        "sparse_horn_fashion_mnist_recommended_dist005": "horn",
        "sparse_horn_cifar10_gray_recommended": "horn",
        "sparse_horn_cifar10_gray_recommended_dist005": "horn",
        "sparse_horn_cifar10_gray_state_mlp_strength8": "state_mlp",
        "sparse_horn_cifar10_rgb_recommended": "horn",
        "sparse_horn_cifar10_rgb_recommended_drive025": "horn",
        "sparse_horn_cifar10_rgb_recommended_normlocal": "horn",
        "sparse_horn_cifar10_rgb_current": "horn",
        "sparse_horn_cifar10_rgb_current_resonant005": "horn",
        "sparse_horn_cifar10_rgb_current_resonant010": "horn",
        "sparse_horn_cifar10_rgb_current_n512": "horn",
        "sparse_horn_cifar10_rgb_current_n512_resonant005": "horn",
        "sparse_horn_cifar10_rgb_current_multimode2": "multimode_horn",
        "sparse_horn_cifar10_rgb_current_multimode2_weak": "multimode_horn",
        "sparse_horn_cifar10_rgb_current_retinotopic": "horn",
        "sparse_horn_cifar10_rgb_current_retinotopic_ch30": "horn",
        "sparse_horn_cifar10_rgb_current_retinotopic_seed4_ch30": "horn",
        "sparse_horn_cifar10_rgb_current_multimode2_retinotopic": (
            "multimode_horn"
        ),
        "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30": (
            "multimode_horn"
        ),
        (
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor_reconstruct010"
        ): "multimode_horn",
        (
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor010"
        ): "multimode_horn",
        (
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor030"
        ): "multimode_horn",
        (
            "sparse_horn_cifar10_rgb_current_multimode2_"
            "retinotopic_anchor_frozen010"
        ): "multimode_horn",
        "sparse_horn_cifar10_rgb_coarse16_normlocal": "coarse_horn",
        "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle": "coarse_horn",
        "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050": (
            "coarse_horn"
        ),
        "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050": (
            "coarse_horn"
        ),
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005": (
            "multiscale_horn"
        ),
        "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical": (
            "multiscale_horn"
        ),
        "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8": (
            "multiscale_horn"
        ),
        "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical_auxlow8": (
            "multiscale_horn"
        ),
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_vgate_conditioning"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_vgate_non_conditioning"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_all"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_conditioning"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_all"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "no_vertical_auxlow8_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_vgate_conditioning_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_all_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_conditioning_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_all_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_conditioning_soft025"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_soft025"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_conditioning_soft025_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_soft025_drive2"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_all_vscale10"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_all_vscale30"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale10"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale30"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_vscale30_center"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_dual_gain_conditioning_vscale10"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_dual_gain_conditioning_vscale30"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_gain_all_vscale30_normstd015"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_signed_gain_conditioning_"
            "vscale30_normstd015"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxlow8_dual_gain_conditioning_vscale30_normstd015"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix75_25_fusion010"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix75_25_consistency005"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_consistency005"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_consistency010"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_fusion010"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_multiscale16_64_"
            "local050_fb005_auxdist8_signed_gain_conditioning_vscale30_"
            "center_feedback_state_mix50_50_fusion025"
        ): "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_lead": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_gate010": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_gate025": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_freq001": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_freq003": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_patch005": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_patch010": "multiscale_horn",
        "sparse_horn_cifar10_rgb_hierarchy_patch010_overlap": "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_hierarchy_patch010_multiscale"
        ): "multiscale_horn",
        (
            "sparse_horn_cifar10_rgb_hierarchy_patch010_multiscale_overlap"
        ): "multiscale_horn",
        "sparse_horn_cifar10_rgb_recommended_drive025_spatial_grid": "horn",
        "sparse_horn_cifar10_rgb_recommended_drive025_center_block": "horn",
        "sparse_horn_cifar10_rgb_recommended_drive010": "horn",
        "sparse_horn_cifar10_rgb_recommended_dist005": "horn",
        "sparse_horn_cifar10_rgb_state_mlp_strength8": "state_mlp",
        "sparse_horn_cifar10_rgb_recommended_step1": "horn",
        "sparse_horn_cifar10_rgb_recommended_frozen_recurrent": "horn",
        "sparse_horn_cifar10_rgb_recommended_frozen_conditioning": "horn",
        "sparse_horn_cifar10_rgb_recommended_no_main_interaction": "horn",
        "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025": "horn",
        "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_spatial_grid": "horn",
        "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_center_block": "horn",
        "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive010": "horn",
        "sparse_horn_cifar10_rgb_recommended_decoder_only": "horn_decoder_only",
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
        if preset in (
            "sparse_horn_cifar10_rgb_recommended_normlocal",
            "sparse_horn_cifar10_rgb_current",
            "sparse_horn_cifar10_rgb_current_resonant005",
            "sparse_horn_cifar10_rgb_current_resonant010",
            "sparse_horn_cifar10_rgb_current_n512",
            "sparse_horn_cifar10_rgb_current_n512_resonant005",
            "sparse_horn_cifar10_rgb_current_multimode2",
            "sparse_horn_cifar10_rgb_current_multimode2_weak",
            "sparse_horn_cifar10_rgb_current_retinotopic",
            "sparse_horn_cifar10_rgb_current_retinotopic_ch30",
            "sparse_horn_cifar10_rgb_current_retinotopic_seed4_ch30",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30",
            (
                "sparse_horn_cifar10_rgb_current_multimode2_"
                "retinotopic_anchor_reconstruct010"
            ),
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor010",
            "sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030",
            (
                "sparse_horn_cifar10_rgb_current_multimode2_"
                "retinotopic_anchor_frozen010"
            ),
            "sparse_horn_cifar10_rgb_coarse16_normlocal",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050",
            "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050",
        ):
            assert parsed.loss_mode == "pixel_feature_drift"
            assert parsed.train_limit == 2000
            assert parsed.coupling_normalization == "row_sum"
            assert parsed.main_coupling_strength == 1.0
            if preset.endswith("_resonant005"):
                assert parsed.resonant_readout_strength == 0.05
                assert parsed.resonant_readout_patch_size == 5
            elif preset.endswith("_resonant010"):
                assert parsed.resonant_readout_strength == 0.10
                assert parsed.resonant_readout_patch_size == 5
            else:
                assert parsed.resonant_readout_strength == 0.0
            if "_n512" in preset:
                assert parsed.num_oscillators == 512
            else:
                assert parsed.num_oscillators == 256
            if "_multimode2" in preset:
                assert parsed.model_family == "multimode_horn"
                assert parsed.multimode_num_modes == 2
                assert parsed.multimode_frequency_scales == (0.75, 1.35)
                expected_mode_strength = (
                    0.10 if preset.endswith("_weak") else 0.25
                )
                assert parsed.multimode_mode_coupling_strength == (
                    expected_mode_strength
                )
            if "_retinotopic" in preset:
                assert parsed.resize_conv_seed_layout == "retinotopic"
                assert parsed.resize_conv_seed_size == 16
                assert parsed.resize_conv_upsamples == 1
                if preset.endswith("_ch30"):
                    assert parsed.resize_conv_min_channels == 30
                if "_anchor" in preset:
                    assert parsed.resize_conv_min_channels == 30
                    assert parsed.state_anchor_weight > 0.0
                    assert parsed.state_anchor_mode in (
                        "reconstruct",
                        "settle",
                        "frozen_dynamics",
                    )
                if "_seed4" in preset:
                    assert parsed.resize_conv_seed_min_channels == 4
            else:
                assert parsed.resize_conv_seed_layout == "flat"
            if preset in (
                "sparse_horn_cifar10_rgb_coarse16_normlocal",
                "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle",
                "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_dist050",
                "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050",
            ):
                assert parsed.model_family == "coarse_horn"
                assert parsed.num_coarse_oscillators == 16
                expected_strength = (
                    0.25
                    if "_gentle" in preset
                    else 1.0
                )
                assert parsed.coarse_to_fine_strength == expected_strength
                if preset.endswith("_dist050"):
                    assert parsed.coarse_to_fine_profile == "distance_decay"
                    assert parsed.coarse_to_fine_length_scale == 0.5
                if preset.endswith("_local050"):
                    assert parsed.coarse_to_fine_profile == "local_radius"
                    assert parsed.coarse_to_fine_length_scale == 0.5
        elif preset.startswith(
            "sparse_horn_cifar10_rgb_multiscale16_64"
        ) or preset.startswith("sparse_horn_cifar10_rgb_hierarchy_"):
            assert parsed.loss_mode == "pixel_feature_drift"
            assert parsed.train_limit == 2000
            assert parsed.model_family == "multiscale_horn"
            assert parsed.multiscale_layer_sizes == (16, 64)
            assert parsed.multiscale_coupling_normalization == "row_sum"
            assert parsed.multiscale_vertical_profile == "local_radius"
            assert parsed.multiscale_vertical_length_scale == 0.5
            is_hierarchy_alias = preset.startswith(
                "sparse_horn_cifar10_rgb_hierarchy_"
            )
            has_auxiliary_objective = (
                "_auxlow8" in preset
                or "_auxdist8" in preset
                or is_hierarchy_alias
            )
            if has_auxiliary_objective:
                assert parsed.coarse_auxiliary_weight == 0.05
                assert parsed.coarse_auxiliary_target_size == 8
                assert parsed.multiscale_auxiliary_readout_layer == 0
                expected_loss_mode = (
                    "distributional"
                    if "_auxdist8" in preset or preset.startswith(
                        "sparse_horn_cifar10_rgb_hierarchy_"
                    )
                    else "mse"
                )
                assert parsed.coarse_auxiliary_loss_mode == expected_loss_mode
            else:
                assert parsed.coarse_auxiliary_weight == 0.0
            if preset.endswith("_vgate_conditioning") or preset.endswith(
                "_vgate_conditioning_drive2"
            ):
                assert parsed.multiscale_vertical_target_gate == "conditioning"
            elif preset.endswith("_vgate_non_conditioning"):
                    assert parsed.multiscale_vertical_target_gate == "non_conditioning"
            elif "_gain_conditioning" in preset or is_hierarchy_alias:
                assert parsed.multiscale_vertical_target_gate == "conditioning"
            else:
                assert parsed.multiscale_vertical_target_gate == "all"
            if "_dual_gain_" in preset:
                assert parsed.multiscale_vertical_mode == "dual_gain"
            elif "_signed_gain_" in preset or is_hierarchy_alias:
                assert parsed.multiscale_vertical_mode == "signed_gain"
            elif "_gain_" in preset:
                assert parsed.multiscale_vertical_mode == "gain_modulation"
            else:
                assert parsed.multiscale_vertical_mode == "additive"
            if preset.endswith("_drive2"):
                assert parsed.conditioning_strength == 2.0
                assert parsed.multiscale_conditioning_strength == 0.25
            else:
                assert parsed.conditioning_strength == 8.0
            if "_soft025" in preset:
                assert parsed.multiscale_vertical_soft_gate_floor == 0.25
            else:
                assert parsed.multiscale_vertical_soft_gate_floor == 0.0
            if "_vscale10" in preset:
                assert parsed.multiscale_vertical_signal_scale == 10.0
            elif "_vscale30" in preset or is_hierarchy_alias:
                assert parsed.multiscale_vertical_signal_scale == 30.0
            else:
                assert parsed.multiscale_vertical_signal_scale == 1.0
            expected_feedback_mode = (
                "state"
                if "_feedback_state" in preset or is_hierarchy_alias
                else "position"
            )
            assert parsed.multiscale_feedback_signal_mode == expected_feedback_mode
            if "_source_conditioning" in preset:
                expected_source_gate = "conditioning"
            elif "_source_non_conditioning" in preset:
                expected_source_gate = "non_conditioning"
            elif "_mix" in preset or is_hierarchy_alias:
                expected_source_gate = "weighted"
            else:
                expected_source_gate = "all"
            assert parsed.multiscale_feedback_source_gate == expected_source_gate
            if "_mix75_25" in preset:
                assert parsed.multiscale_feedback_source_mix == (0.75, 0.25)
            elif "_mix50_50" in preset or is_hierarchy_alias:
                assert parsed.multiscale_feedback_source_mix == (0.5, 0.5)
            elif "_mix25_75" in preset:
                assert parsed.multiscale_feedback_source_mix == (0.25, 0.75)
            else:
                assert parsed.multiscale_feedback_source_mix == (1.0, 1.0)
            if "_fusion010" in preset:
                assert parsed.multiscale_readout_fusion_strength == 0.10
            elif "_fusion025" in preset:
                assert parsed.multiscale_readout_fusion_strength == 0.25
            else:
                assert parsed.multiscale_readout_fusion_strength == 0.0
            if preset.endswith("_gate010"):
                assert parsed.multiscale_readout_gate_mode == "seed_film"
                assert parsed.multiscale_readout_gate_strength == 0.10
            elif preset.endswith("_gate025"):
                assert parsed.multiscale_readout_gate_mode == "seed_film"
                assert parsed.multiscale_readout_gate_strength == 0.25
            else:
                assert parsed.multiscale_readout_gate_mode == "none"
                assert parsed.multiscale_readout_gate_strength == 0.0
            if "_consistency005" in preset:
                assert parsed.coarse_readout_consistency_weight == 0.05
                assert parsed.coarse_readout_consistency_onset_epoch == 5
            elif "_consistency010" in preset:
                assert parsed.coarse_readout_consistency_weight == 0.10
                assert parsed.coarse_readout_consistency_onset_epoch == 5
            else:
                assert parsed.coarse_readout_consistency_weight == 0.0
                assert parsed.coarse_readout_consistency_onset_epoch == 0
            if preset.endswith("_freq001"):
                assert parsed.frequency_objective_weight == 0.01
                assert parsed.frequency_objective_edge_weight == 1.0
            elif preset.endswith("_freq003"):
                assert parsed.frequency_objective_weight == 0.03
                assert parsed.frequency_objective_edge_weight == 1.0
            else:
                assert parsed.frequency_objective_weight == 0.0
                assert parsed.frequency_objective_edge_weight == 1.0
            if preset.endswith("_patch005"):
                assert parsed.patch_objective_weight == 0.05
                assert parsed.patch_objective_patch_size == 5
                assert parsed.patch_objective_patch_sizes == ()
                assert parsed.patch_objective_stride == 4
                assert parsed.patch_objective_offsets == (0,)
                assert parsed.patch_objective_projections == 32
                assert parsed.patch_objective_edge_weight == 0.25
            elif "_patch010" in preset:
                expected_patch_weight = (
                    0.07
                    if preset.endswith("_patch010_multiscale_overlap")
                    else 0.10
                )
                assert parsed.patch_objective_weight == expected_patch_weight
                assert parsed.patch_objective_patch_size == 5
                if "_multiscale" in preset:
                    assert parsed.patch_objective_patch_sizes == (3, 5, 7)
                    assert parsed.patch_objective_projections == 48
                else:
                    assert parsed.patch_objective_patch_sizes == ()
                    assert parsed.patch_objective_projections == 32
                assert parsed.patch_objective_stride == 4
                if preset.endswith("_overlap"):
                    assert parsed.patch_objective_offsets == (0, 2)
                else:
                    assert parsed.patch_objective_offsets == (0,)
                assert parsed.patch_objective_edge_weight == 0.25
            else:
                assert parsed.patch_objective_weight == 0.0
                assert parsed.patch_objective_patch_size == 5
                assert parsed.patch_objective_patch_sizes == ()
                assert parsed.patch_objective_stride == 4
                assert parsed.patch_objective_offsets == (0,)
                assert parsed.patch_objective_projections == 32
                assert parsed.patch_objective_edge_weight == 0.25
            if "_normstd015" in preset:
                assert parsed.multiscale_vertical_gain_normalization == "center_rms"
                assert parsed.multiscale_vertical_gain_target_std == 0.015
            elif "_center" in preset or is_hierarchy_alias:
                assert parsed.multiscale_vertical_gain_normalization == "center"
                assert parsed.multiscale_vertical_gain_target_std == 0.0
            else:
                assert parsed.multiscale_vertical_gain_normalization == "none"
                assert parsed.multiscale_vertical_gain_target_std == 0.0
        else:
            assert parsed.loss_mode == "pixel_drift"
        if parsed.dataset_name in ("cifar10_gray", "cifar10_rgb"):
            assert parsed.train_limit in (1000, 2000)
        else:
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
    split_coupling = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_class_coupling_strength8",
                "--main-coupling-strength",
                "0.25",
            ]
        )
    )
    assert split_coupling.coupling_strength == 1.0
    assert split_coupling.main_coupling_strength == 0.25

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

    dynamics_quality_dist = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_dynamics_quality_dist0025"])
    )
    assert dynamics_quality_dist.conditioning_strength == 8.0
    assert dynamics_quality_dist.horn_damping == 0.30
    assert dynamics_quality_dist.distributional_weight == 0.025

    no_main_coupling = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_recommended_no_main_coupling"])
    )
    assert no_main_coupling.coupling_strength == 0.0
    assert no_main_coupling.conditioning_mode == "class_coupling"
    assert no_main_coupling.conditioning_strength == 8.0

    frozen_recurrent = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_recommended_frozen_recurrent"])
    )
    assert frozen_recurrent.train_recurrent_dynamics is False
    assert frozen_recurrent.train_conditioning_dynamics is True

    frozen_conditioning = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_recommended_frozen_conditioning"])
    )
    assert frozen_conditioning.train_recurrent_dynamics is True
    assert frozen_conditioning.train_conditioning_dynamics is False

    recommended_decoder = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_recommended_decoder_only"])
    )
    assert recommended_decoder.model_family == "horn_decoder_only"
    assert recommended_decoder.train_settling_steps == ()

    recommended_step1 = config_from_args(
        parse_args(["--preset", "sparse_horn_mnist_recommended_step1"])
    )
    assert recommended_step1.steps == 1
    assert recommended_step1.train_settling_steps == (1,)

    state_mlp_strength8 = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_mnist_state_mlp_class_coupling_strength8"]
        )
    )
    assert state_mlp_strength8.model_family == "state_mlp"
    assert state_mlp_strength8.conditioning_strength == 8.0

    state_mlp_strength8_dist = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class",
            ]
        )
    )
    assert state_mlp_strength8_dist.model_family == "state_mlp"
    assert state_mlp_strength8_dist.conditioning_strength == 8.0
    assert state_mlp_strength8_dist.distributional_weight == 0.1
    assert state_mlp_strength8_dist.class_moment_weight == 1.0

    fashion = config_from_args(
        parse_args(["--preset", "sparse_horn_fashion_mnist_recommended"])
    )
    assert fashion.dataset_name == "fashion_mnist"
    assert fashion.data_source == "idx"
    assert fashion.conditioning_mode == "class_coupling"
    assert fashion.conditioning_strength == 8.0
    assert fashion.horn_damping == 0.30

    fashion_ch16 = config_from_args(
        parse_args(["--preset", "sparse_horn_fashion_mnist_recommended_ch16"])
    )
    assert fashion_ch16.dataset_name == "fashion_mnist"
    assert fashion_ch16.resize_conv_min_channels == 16

    fashion_dist = config_from_args(
        parse_args(["--preset", "sparse_horn_fashion_mnist_recommended_dist0025"])
    )
    assert fashion_dist.dataset_name == "fashion_mnist"
    assert fashion_dist.distributional_weight == 0.025

    cifar = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_gray_recommended"])
    )
    assert cifar.dataset_name == "cifar10_gray"
    assert cifar.image_shape == (32, 32)
    assert cifar.resize_conv_seed_size == 8
    assert cifar.num_oscillators == 256
    assert cifar.train_limit == 1000

    cifar_model = build_mnist_generator_model(cifar, jax.random.PRNGKey(0))
    assert cifar_model.image_shape == (32, 32)
    assert cifar_model.image_dim == 32 * 32

    cifar_rgb = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_rgb_recommended"])
    )
    assert cifar_rgb.dataset_name == "cifar10_rgb"
    assert cifar_rgb.image_shape == (32, 32, 3)
    assert cifar_rgb.resize_conv_seed_size == 8
    assert cifar_rgb.resize_conv_min_channels == 16
    assert cifar_rgb.quality_classifier_kind == "conv"
    assert cifar_rgb.train_limit == 1000

    cifar_rgb_model = build_mnist_generator_model(cifar_rgb, jax.random.PRNGKey(0))
    assert cifar_rgb_model.image_shape == (32, 32, 3)
    assert cifar_rgb_model.image_dim == 3 * 32 * 32
    assert cifar_rgb_model.resize_conv_output is not None
    assert cifar_rgb_model.resize_conv_output.out_channels == 3

    cifar_rgb_c2f = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050"]
        )
    )
    assert cifar_rgb_c2f.model_family == "coarse_horn"
    assert cifar_rgb_c2f.coarse_to_fine_profile == "local_radius"
    assert cifar_rgb_c2f.coarse_to_fine_strength == 0.25
    assert cifar_rgb_c2f.resize_conv_min_channels == 16
    assert cifar_rgb_c2f.distributional_weight == 0.0

    cifar_rgb_c2f_ch32 = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_ch32",
            ]
        )
    )
    assert cifar_rgb_c2f_ch32.model_family == "coarse_horn"
    assert cifar_rgb_c2f_ch32.coarse_to_fine_profile == "local_radius"
    assert cifar_rgb_c2f_ch32.resize_conv_min_channels == 32

    cifar_rgb_c2f_dist = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050_dist0025",
            ]
        )
    )
    assert cifar_rgb_c2f_dist.distributional_weight == 0.025
    assert cifar_rgb_c2f_dist.resize_conv_min_channels == 16

    cifar_rgb_c2f_ch32_dist = config_from_args(
        parse_args(
            [
                "--preset",
                (
                    "sparse_horn_cifar10_rgb_coarse16_normlocal_"
                    "gentle_local050_ch32_dist0025"
                ),
            ]
        )
    )
    assert cifar_rgb_c2f_ch32_dist.distributional_weight == 0.025
    assert cifar_rgb_c2f_ch32_dist.resize_conv_min_channels == 32

    cifar_rgb_multiscale = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005",
            ]
        )
    )
    assert cifar_rgb_multiscale.model_family == "multiscale_horn"
    assert cifar_rgb_multiscale.multiscale_layer_sizes == (16, 64)
    assert cifar_rgb_multiscale.multiscale_vertical_strength == 0.25
    assert cifar_rgb_multiscale.multiscale_feedback_strength == 0.05

    cifar_rgb_multiscale_no_vertical = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_multiscale16_64_no_vertical",
            ]
        )
    )
    assert cifar_rgb_multiscale_no_vertical.model_family == "multiscale_horn"
    assert cifar_rgb_multiscale_no_vertical.multiscale_vertical_strength == 0.0
    assert cifar_rgb_multiscale_no_vertical.multiscale_feedback_strength == 0.0

    cifar_rgb_multiscale_aux = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxlow8",
            ]
        )
    )
    assert cifar_rgb_multiscale_aux.coarse_auxiliary_weight == 0.05
    assert cifar_rgb_multiscale_aux.coarse_auxiliary_target_size == 8
    assert cifar_rgb_multiscale_aux.coarse_auxiliary_loss_mode == "mse"

    cifar_rgb_multiscale_auxdist = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005_auxdist8",
            ]
        )
    )
    assert cifar_rgb_multiscale_auxdist.coarse_auxiliary_weight == 0.05
    assert cifar_rgb_multiscale_auxdist.coarse_auxiliary_target_size == 8
    assert cifar_rgb_multiscale_auxdist.coarse_auxiliary_loss_mode == "distributional"

    cifar_rgb_drive025 = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_rgb_recommended_drive025"])
    )
    assert cifar_rgb_drive025.conditioning_target_fraction == 0.25

    cifar_rgb_drive025_grid = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_recommended_drive025_spatial_grid"]
        )
    )
    assert cifar_rgb_drive025_grid.conditioning_target_fraction == 0.25
    assert cifar_rgb_drive025_grid.conditioning_target_pattern == "spatial_grid"

    cifar_rgb_drive025_center = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_recommended_drive025_center_block"]
        )
    )
    assert cifar_rgb_drive025_center.conditioning_target_fraction == 0.25
    assert cifar_rgb_drive025_center.conditioning_target_pattern == "center_block"

    cifar_rgb_drive010 = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_rgb_recommended_drive010"])
    )
    assert cifar_rgb_drive010.conditioning_target_fraction == 0.10

    cifar_rgb_step1 = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_rgb_recommended_step1"])
    )
    assert cifar_rgb_step1.steps == 1
    assert cifar_rgb_step1.train_settling_steps == (1,)

    cifar_rgb_frozen_recurrent = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_recommended_frozen_recurrent"]
        )
    )
    assert cifar_rgb_frozen_recurrent.train_recurrent_dynamics is False
    assert cifar_rgb_frozen_recurrent.train_conditioning_dynamics is True

    cifar_rgb_frozen_conditioning = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_recommended_frozen_conditioning"]
        )
    )
    assert cifar_rgb_frozen_conditioning.train_recurrent_dynamics is True
    assert cifar_rgb_frozen_conditioning.train_conditioning_dynamics is False

    cifar_rgb_no_main = config_from_args(
        parse_args(
            ["--preset", "sparse_horn_cifar10_rgb_recommended_no_main_interaction"]
        )
    )
    assert cifar_rgb_no_main.coupling_init_scale == 0.0
    assert cifar_rgb_no_main.train_recurrent_dynamics is False
    assert cifar_rgb_no_main.train_conditioning_dynamics is True

    cifar_rgb_no_main_drive025 = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025",
            ]
        )
    )
    assert cifar_rgb_no_main_drive025.conditioning_target_fraction == 0.25
    assert cifar_rgb_no_main_drive025.coupling_init_scale == 0.0
    assert cifar_rgb_no_main_drive025.train_recurrent_dynamics is False

    cifar_rgb_no_main_drive025_grid = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_spatial_grid",
            ]
        )
    )
    assert cifar_rgb_no_main_drive025_grid.conditioning_target_fraction == 0.25
    assert cifar_rgb_no_main_drive025_grid.conditioning_target_pattern == "spatial_grid"
    assert cifar_rgb_no_main_drive025_grid.coupling_init_scale == 0.0

    cifar_rgb_no_main_drive025_center = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_no_main_interaction_drive025_center_block",
            ]
        )
    )
    assert cifar_rgb_no_main_drive025_center.conditioning_target_fraction == 0.25
    assert (
        cifar_rgb_no_main_drive025_center.conditioning_target_pattern
        == "center_block"
    )
    assert cifar_rgb_no_main_drive025_center.coupling_init_scale == 0.0

    cifar_rgb_decoder = config_from_args(
        parse_args(["--preset", "sparse_horn_cifar10_rgb_recommended_decoder_only"])
    )
    assert cifar_rgb_decoder.model_family == "horn_decoder_only"
    assert cifar_rgb_decoder.train_settling_steps == ()

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


def test_sparse_conditioning_target_fraction_masks_direct_class_drive():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_drive025",
                "--epochs",
                "1",
                "--train-limit",
                "8",
                "--eval-limit",
                "8",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(0))

    mask = jnp.asarray(model.conditioning_target_mask)
    assert mask.shape == (256,)
    assert int(jnp.sum(mask)) == 64
    assert jnp.all(mask[:64] == 1.0)
    assert jnp.all(mask[64:] == 0.0)

    labels = jnp.array([0, 1])
    position = jnp.ones((2, model.num_oscillators))
    drive = model._horn_static_conditioning_drive(position, labels)

    assert jnp.max(jnp.abs(drive[:, 64:])) == 0.0
    assert jnp.max(jnp.abs(drive[:, :64])) > 0.0

    diagnostics = compute_generator_success_diagnostics(model)
    assert diagnostics["conditioning_target_fraction"] == 0.25
    assert diagnostics["conditioning_target_pattern"] == "prefix"
    assert diagnostics["conditioning_target_count"] == 64
    assert diagnostics["conditioning_target_effective_fraction"] == 0.25


def test_spatial_grid_conditioning_target_pattern_distributes_drive():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_drive025_spatial_grid",
                "--epochs",
                "1",
                "--train-limit",
                "8",
                "--eval-limit",
                "8",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(0))

    mask = jnp.asarray(model.conditioning_target_mask)
    assert mask.shape == (256,)
    assert int(jnp.sum(mask)) == 64
    assert model.conditioning_target_pattern == "spatial_grid"
    assert jnp.sum(mask[:64]) < 64
    assert jnp.sum(mask[64:]) > 0

    labels = jnp.array([0, 1])
    position = jnp.ones((2, model.num_oscillators))
    drive = model._horn_static_conditioning_drive(position, labels)

    assert jnp.max(jnp.abs(drive * (1.0 - mask)[None, :])) == 0.0
    assert jnp.max(jnp.abs(drive * mask[None, :])) > 0.0

    diagnostics = compute_generator_success_diagnostics(model)
    assert diagnostics["conditioning_target_pattern"] == "spatial_grid"
    assert diagnostics["conditioning_target_count"] == 64


def test_center_block_conditioning_target_pattern_uses_center_patch():
    config = config_from_args(
        parse_args(
            [
                "--preset",
                "sparse_horn_cifar10_rgb_recommended_drive025_center_block",
                "--epochs",
                "1",
                "--train-limit",
                "8",
                "--eval-limit",
                "8",
            ]
        )
    )
    model = build_mnist_generator_model(config, jax.random.PRNGKey(0))

    mask = jnp.asarray(model.conditioning_target_mask).reshape(16, 16)
    assert mask.shape == (16, 16)
    assert int(jnp.sum(mask)) == 64
    assert model.conditioning_target_pattern == "center_block"
    assert jnp.all(mask[4:12, 4:12] == 1.0)
    assert jnp.sum(mask[:4, :]) == 0.0
    assert jnp.sum(mask[:, :4]) == 0.0

    flat_mask = mask.reshape(-1)
    labels = jnp.array([0, 1])
    position = jnp.ones((2, model.num_oscillators))
    drive = model._horn_static_conditioning_drive(position, labels)

    assert jnp.max(jnp.abs(drive * (1.0 - flat_mask)[None, :])) == 0.0
    assert jnp.max(jnp.abs(drive * flat_mask[None, :])) > 0.0

    diagnostics = compute_generator_success_diagnostics(model)
    assert diagnostics["conditioning_target_pattern"] == "center_block"
    assert diagnostics["conditioning_target_count"] == 64


def test_unknown_idx_dataset_loading_is_rejected():
    try:
        load_mnist_data(
            source="idx",
            dataset_name="not_a_dataset",
            train_limit=1,
            eval_limit=1,
        )
    except ValueError as exc:
        assert "direct dataset loading" in str(exc)
    else:
        raise AssertionError("unknown IDX dataset must not silently load MNIST data")


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
                "--quality-classifier-kind",
                "residual_conv",
                "--learned-feature-kind",
                "residual_conv",
                "--attractor-variants-per-class",
                "3",
            ]
        )
    )

    assert parsed.train_limit == 32
    assert parsed.eval_limit == 16
    assert parsed.quality_classifier_train_limit == 5000
    assert parsed.quality_classifier_eval_limit == 2000
    assert parsed.quality_classifier_kind == "residual_conv"
    assert parsed.learned_feature_kind == "residual_conv"
    assert parsed.attractor_variants_per_class == 3


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


def test_downsample_image_batch_block_averages_flat_images():
    images = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 16)
    lowres = downsample_image_batch(
        images,
        image_shape=(4, 4),
        target_size=2,
    )

    expected = jnp.asarray(
        [
            [2.5, 4.5, 10.5, 12.5],
            [18.5, 20.5, 26.5, 28.5],
        ],
        dtype=jnp.float32,
    )
    assert lowres.shape == (2, 4)
    assert bool(jnp.allclose(lowres, expected))


def test_coarse_auxiliary_image_loss_supports_distributional_mode():
    class DummyAuxiliaryModel:
        def sample_auxiliary_image(self, key, batch_size, labels):
            del key, labels
            return jnp.zeros((batch_size, 4), dtype=jnp.float32)

    real = jnp.asarray(
        [
            [0.0, 0.2, 0.7, 1.0] * 4,
            [1.0, 0.7, 0.2, 0.0] * 4,
            [0.1, 0.3, 0.6, 0.8] * 4,
        ],
        dtype=jnp.float32,
    )
    labels = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    key = jax.random.PRNGKey(999)

    mse_loss = coarse_auxiliary_image_loss(
        DummyAuxiliaryModel(),
        real,
        key=key,
        labels=labels,
        image_shape=(4, 4),
        target_size=2,
        loss_mode="mse",
        num_classes=2,
    )
    distributional_loss = coarse_auxiliary_image_loss(
        DummyAuxiliaryModel(),
        real,
        key=key,
        labels=labels,
        image_shape=(4, 4),
        target_size=2,
        loss_mode="distributional",
        num_classes=2,
    )

    assert mse_loss > 0.0
    assert distributional_loss > 0.0
    assert not bool(jnp.allclose(mse_loss, distributional_loss))
    with pytest.raises(ValueError):
        coarse_auxiliary_image_loss(
            DummyAuxiliaryModel(),
            real,
            key=key,
            labels=labels,
            image_shape=(4, 4),
            target_size=2,
            loss_mode="bogus",
            num_classes=2,
        )


def test_coarse_readout_consistency_loss_matches_same_scale_scaffold():
    generated = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0] * 4,
            [1.0, 1.0, 0.0, 0.0] * 4,
        ],
        dtype=jnp.float32,
    )
    auxiliary = downsample_image_batch(
        generated,
        image_shape=(4, 4),
        target_size=2,
    )
    shifted_auxiliary = jnp.clip(auxiliary + 0.25, 0.0, 1.0)

    aligned = coarse_readout_consistency_loss(
        generated,
        auxiliary,
        image_shape=(4, 4),
        target_size=2,
    )
    shifted = coarse_readout_consistency_loss(
        generated,
        shifted_auxiliary,
        image_shape=(4, 4),
        target_size=2,
    )

    assert bool(jnp.allclose(aligned, 0.0))
    assert shifted > aligned


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
        image_shape=(28, 28),
    )

    assert 0.0 <= metrics["classifier_label_accuracy"] <= 1.0
    assert 0.0 <= metrics["classifier_label_confidence"] <= 1.0
    assert 0.0 <= metrics["classifier_max_confidence"] <= 1.0
    assert metrics["classifier_entropy"] >= 0.0
    assert metrics["classifier_feature_mean_mse"] >= 0.0
    assert metrics["classifier_feature_std_mse"] >= 0.0
    assert metrics["classifier_feature_diversity_ratio"] >= 0.0
    assert metrics["classifier_feature_nearest_real_mse"] >= 0.0
    assert metrics["classifier_feature_real_nearest_real_mse"] >= 0.0
    assert metrics["classifier_feature_pairwise_distance_ratio"] >= 0.0
    assert metrics["classifier_feature_frechet_distance"] >= 0.0
    assert "classifier_feature_kid_mmd2" in metrics
    assert 0.0 <= metrics["classifier_feature_precision_at_real_median"] <= 1.0
    assert 0.0 <= metrics["classifier_feature_recall_at_real_median"] <= 1.0
    assert metrics["frequency_real_high_power_ratio"] >= 0.0
    assert metrics["edge_laplacian_variance_ratio"] >= 0.0


def test_generator_quality_metrics_report_frequency_diagnostics_for_rgb():
    real = jnp.full((2, 3 * 4 * 4), 0.5, dtype=jnp.float32)
    high_frequency = jnp.asarray(
        [
            [[0.0, 1.0, 0.0, 1.0] * 4] * 3,
            [[1.0, 0.0, 1.0, 0.0] * 4] * 3,
        ],
        dtype=jnp.float32,
    ).reshape(2, 3 * 4 * 4)

    metrics = compute_generator_quality_metrics(
        real,
        high_frequency,
        image_shape=(4, 4, 3),
    )

    assert metrics["frequency_generated_high_power_ratio"] > 0.0
    assert metrics["frequency_spectral_centroid_generated"] > 0.0
    assert metrics["edge_laplacian_variance_generated"] > (
        metrics["edge_laplacian_variance_real"]
    )


def test_conv_image_feature_classifier_smoke():
    images = jnp.linspace(0.0, 1.0, 5 * 3 * 32 * 32).reshape(5, 3 * 32 * 32)
    classifier = ConvImageFeatureClassifier(
        image_dim=3 * 32 * 32,
        image_shape=(32, 32, 3),
        feature_dim=12,
        depth=2,
        num_classes=4,
        key=jax.random.PRNGKey(71),
    )

    logits = classifier(images)
    features = classifier.features(images)

    assert logits.shape == (5, 4)
    assert features.shape == (5, 12)
    assert classifier.image_channels == 3
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert bool(jnp.all(jnp.isfinite(features)))


def test_residual_conv_image_feature_classifier_smoke():
    images = jnp.linspace(0.0, 1.0, 5 * 3 * 32 * 32).reshape(5, 3 * 32 * 32)
    classifier = ResidualConvImageFeatureClassifier(
        image_dim=3 * 32 * 32,
        image_shape=(32, 32, 3),
        feature_dim=12,
        depth=2,
        num_classes=4,
        key=jax.random.PRNGKey(72),
    )

    logits = classifier(images)
    features = classifier.features(images)

    assert logits.shape == (5, 4)
    assert features.shape == (5, 12)
    assert classifier.image_channels == 3
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert bool(jnp.all(jnp.isfinite(features)))


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
    assert history["classifier_kind"] == "mlp"
    assert 0.0 <= history["final_eval_accuracy"] <= 1.0
    assert history["final_eval_loss"] >= 0.0


def test_train_conv_feature_classifier_smoke():
    images = jnp.linspace(0.0, 1.0, 12 * 32 * 32).reshape(12, 32 * 32)
    labels = jnp.asarray([0, 1, 2, 3] * 3, dtype=jnp.int32)

    classifier, history = train_mnist_feature_classifier(
        images,
        labels,
        images[:8],
        labels[:8],
        key=jax.random.PRNGKey(15),
        num_classes=4,
        feature_dim=12,
        depth=2,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        classifier_kind="conv",
        image_shape=(32, 32),
    )

    assert classifier.features(images[:2]).shape == (2, 12)
    assert history["epochs"] == 1
    assert history["classifier_kind"] == "conv"
    assert 0.0 <= history["final_eval_accuracy"] <= 1.0
    assert history["final_eval_loss"] >= 0.0


def test_train_residual_conv_feature_classifier_smoke():
    images = jnp.linspace(0.0, 1.0, 12 * 3 * 16 * 16).reshape(12, 3 * 16 * 16)
    labels = jnp.asarray([0, 1, 2, 3] * 3, dtype=jnp.int32)

    classifier, history = train_mnist_feature_classifier(
        images,
        labels,
        images[:8],
        labels[:8],
        key=jax.random.PRNGKey(16),
        num_classes=4,
        feature_dim=12,
        depth=1,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        classifier_kind="residual_conv",
        image_shape=(16, 16, 3),
    )

    assert classifier.features(images[:2]).shape == (2, 12)
    assert history["epochs"] == 1
    assert history["classifier_kind"] == "residual_conv"
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
    assert diagnostics["state_energy_final"] >= 0.0
    assert diagnostics["state_velocity_rms_final"] >= 0.0
    assert diagnostics["state_update_rms_mean"] >= 0.0
    assert diagnostics["state_acceleration_rms_mean"] >= 0.0
    assert diagnostics["coupling_potential_proxy_final"] >= 0.0
    assert diagnostics["output_step_mse_mean"] >= 0.0

    trace_dynamics = compute_generator_trace_dynamics(model, trace)
    assert trace_dynamics["state_energy_final"] == diagnostics["state_energy_final"]
    assert trace_dynamics["output_step_mse_mean"] >= 0.0


def test_horn_output_feedback_changes_settled_state_and_reports_costs():
    from oscnet.models import HORNImageGenerator

    base_kwargs = dict(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        conditioning_mode="phase_shift",
    )
    disabled = HORNImageGenerator(
        **base_kwargs,
        key=jax.random.PRNGKey(191),
    )
    enabled = HORNImageGenerator(
        **base_kwargs,
        output_feedback_strength=1.0,
        output_feedback_init_scale=0.5,
        key=jax.random.PRNGKey(191),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    sample_key = jax.random.PRNGKey(192)
    generated_disabled = disabled(sample_key, 3, labels)
    generated_enabled = enabled(sample_key, 3, labels)
    trace = enabled.collect_trace(jax.random.PRNGKey(193), 3, labels)
    diagnostics = compute_generator_success_diagnostics(
        enabled,
        trace=trace,
    )

    assert not bool(jnp.allclose(generated_disabled, generated_enabled))
    assert trace["output_feedback_drive"].shape == (3, 8)
    assert trace["output_feedback_gain"].shape == (8,)
    assert diagnostics["output_feedback_strength"] == 1.0
    assert diagnostics["output_feedback_mode"] == "state_proxy"
    assert diagnostics["output_feedback_params"] == 8
    assert diagnostics["estimated_output_feedback_ops_per_sample"] > 0
    assert diagnostics["estimated_output_feedback_op_fraction"] > 0.0


def test_coarse_to_fine_horn_generator_samples_and_counts_coarse_params():
    from oscnet.models import CoarseToFineHORNImageGenerator

    model = CoarseToFineHORNImageGenerator(
        num_oscillators=8,
        num_coarse_oscillators=4,
        image_shape=(8, 8),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=2,
        resize_conv_min_channels=4,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        label_phase_scale=0.0,
        coupling_profile="local_radius",
        coupling_normalization="row_sum",
        coupling_length_scale=0.8,
        coarse_coupling_profile="dense",
        coarse_coupling_normalization="row_sum",
        coarse_to_fine_profile="local_radius",
        coarse_to_fine_length_scale=1.0,
        key=jax.random.PRNGKey(930),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(931), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(932), 3, labels)
    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 64)
    assert trace["coarse_theta_trajectory"].shape == (2, 3, 4)
    assert trace["coarse_velocity_trajectory"].shape == (2, 3, 4)
    assert trace["coarse_to_fine_coupling"].shape == (8, 4)
    assert trace["coarse_to_fine_profile"].shape == (8, 4)
    assert diagnostics["dynamics_family"] == "coarse_horn"
    assert diagnostics["num_coarse_oscillators"] == 4
    assert diagnostics["coarse_to_fine_profile"] == "local_radius"
    assert diagnostics["coarse_to_fine_profile_density"] < 1.0
    assert diagnostics["coarse_recurrent_params"] == 4 + 16 + 32
    assert diagnostics["coarse_conditioning_params"] == 3 * 4 * 3
    assert diagnostics["recurrent_params"] > diagnostics["coarse_recurrent_params"]
    assert diagnostics["estimated_recurrent_ops_per_sample"] > 2 * 8 * 8
    assert diagnostics["coarse_state_energy_final"] >= 0.0
    assert diagnostics["coarse_state_update_rms_mean"] >= 0.0
    assert diagnostics["coarse_coupling_potential_proxy_final"] >= 0.0
    assert diagnostics["coarse_to_fine_potential_proxy_final"] >= 0.0

    no_drive_model = CoarseToFineHORNImageGenerator(
        num_oscillators=8,
        num_coarse_oscillators=4,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        label_phase_scale=0.0,
        coarse_to_fine_strength=0.0,
        key=jax.random.PRNGKey(933),
    )
    no_drive_trace = no_drive_model.collect_trace(jax.random.PRNGKey(934), 3, labels)
    no_drive_diagnostics = compute_generator_success_diagnostics(
        no_drive_model,
        trace=no_drive_trace,
    )
    assert no_drive_diagnostics["coarse_to_fine_potential_proxy_final"] == 0.0
    assert no_drive_diagnostics["coarse_to_fine_potential_proxy_delta"] == 0.0


def test_multiscale_horn_generator_samples_and_counts_layered_params():
    from oscnet.models import MultiscaleHORNImageGenerator

    model = MultiscaleHORNImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=2,
        resize_conv_min_channels=4,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        label_phase_scale=0.0,
        coupling_profile="local_radius",
        coupling_normalization="row_sum",
        coupling_length_scale=0.8,
        multiscale_layer_sizes=(2, 4),
        multiscale_frequency_scales=(0.5, 0.8),
        multiscale_coupling_profile="dense",
        multiscale_coupling_normalization="row_sum",
        multiscale_vertical_strength=0.25,
        multiscale_feedback_strength=0.1,
        multiscale_vertical_profile="local_radius",
        multiscale_vertical_length_scale=1.0,
        multiscale_auxiliary_readout_size=4,
        multiscale_auxiliary_readout_layer=0,
        multiscale_readout_fusion_strength=0.25,
        key=jax.random.PRNGKey(940),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    generated = model(jax.random.PRNGKey(941), 3, labels)
    auxiliary = model.sample_auxiliary_image(jax.random.PRNGKey(943), 3, labels)
    paired_generated, paired_auxiliary = model.sample_with_auxiliary_image(
        jax.random.PRNGKey(944),
        3,
        labels,
    )
    trace = model.collect_trace(jax.random.PRNGKey(942), 3, labels)
    diagnostics = compute_generator_success_diagnostics(
        model,
        trace=trace,
        sample_count=12,
        total_train_seconds=2.0,
    )

    assert generated.shape == (3, 64)
    assert auxiliary.shape == (3, 16)
    assert paired_generated.shape == (3, 64)
    assert paired_auxiliary.shape == (3, 16)
    assert model.num_auxiliary_layers == 2
    assert model.num_vertical_couplings == 4
    assert model.multiscale_auxiliary_readout_layer == 0
    assert trace["aux_0_theta_trajectory"].shape == (2, 3, 2)
    assert trace["aux_1_theta_trajectory"].shape == (2, 3, 4)
    assert trace["auxiliary_lowres_generated"].shape == (3, 16)
    assert trace["auxiliary_upsampled_generated"].shape == (3, 64)
    assert trace["fine_generated"].shape == (3, 64)
    assert float(trace["readout_fusion_strength"]) == 0.25
    expected_fused = (
        0.75 * trace["fine_generated"]
        + 0.25 * trace["auxiliary_upsampled_generated"]
    )
    assert bool(jnp.allclose(trace["generated"], expected_fused, atol=1e-6))
    assert trace["auxiliary_readout_weight"].shape == (4, 16)
    assert trace["vertical_0_coupling"].shape == (4, 2)
    assert trace["vertical_1_coupling"].shape == (2, 4)
    assert trace["vertical_0_source_gate"].shape == (2,)
    assert int(trace["vertical_mode"]) == 0
    assert trace["vertical_gain_final"].shape == (3, 8)
    assert diagnostics["dynamics_family"] == "multiscale_horn"
    assert diagnostics["multiscale_layer_sizes"] == [2, 4]
    assert diagnostics["num_auxiliary_layers"] == 2
    assert diagnostics["num_vertical_couplings"] == 4
    assert diagnostics["auxiliary_recurrent_params"] == 2 + 4 + 4 + 16
    assert diagnostics["vertical_recurrent_params"] == 2 * 4 + 4 * 2 + 4 * 8 + 8 * 4
    assert diagnostics["multiscale_conditioning_params"] == (3 * 2 * 3 + 3 * 4 * 3)
    assert diagnostics["auxiliary_readout_params"] == 4 * 16 + 16
    assert diagnostics["multiscale_auxiliary_readout_layer"] == 0
    assert diagnostics["multiscale_auxiliary_readout_size"] == 4
    assert diagnostics["multiscale_readout_fusion_strength"] == 0.25
    assert diagnostics["multiscale_readout_gate_mode"] == "none"
    assert diagnostics["multiscale_readout_gate_params"] == 0
    assert diagnostics["multiscale_vertical_mode"] == "additive"
    assert diagnostics["multiscale_feedback_signal_mode"] == "position"
    assert diagnostics["multiscale_feedback_source_gate"] == "all"
    assert diagnostics["multiscale_feedback_source_mix"] == [1.0, 1.0]
    assert diagnostics["vertical_profile_density"] <= 1.0
    assert diagnostics["estimated_recurrent_ops_per_sample"] > 2 * 8 * 8
    assert diagnostics["aux_0_state_energy_final"] >= 0.0
    assert diagnostics["aux_1_state_update_rms_mean"] >= 0.0
    assert diagnostics["vertical_0_potential_proxy_final"] >= 0.0
    assert "vertical_potential_proxy_delta_mean" in diagnostics


def test_multiscale_horn_readout_gate_modulates_resize_conv_seed():
    from oscnet.models import MultiscaleHORNImageGenerator

    model = MultiscaleHORNImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_upsamples=2,
        resize_conv_min_channels=4,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        label_phase_scale=0.0,
        coupling_profile="local_radius",
        coupling_normalization="row_sum",
        coupling_length_scale=0.8,
        multiscale_layer_sizes=(2, 4),
        multiscale_vertical_strength=0.25,
        multiscale_feedback_strength=0.1,
        multiscale_auxiliary_readout_size=4,
        multiscale_readout_gate_mode="seed_film",
        multiscale_readout_gate_strength=0.5,
        multiscale_readout_gate_init_scale=0.0,
        key=jax.random.PRNGKey(945),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    key = jax.random.PRNGKey(946)
    ungated_output = model(key, 3, labels)

    seed_channels = int(model.resize_conv_seed_shape[0])
    gate_bias = jnp.concatenate(
        [
            jnp.full((seed_channels,), 0.5, dtype=jnp.float32),
            jnp.full((seed_channels,), -0.25, dtype=jnp.float32),
        ],
        axis=0,
    )
    object.__setattr__(model, "multiscale_readout_gate_bias", gate_bias)
    gated_output = model(key, 3, labels)
    diagnostics = compute_generator_success_diagnostics(model)

    assert jnp.max(jnp.abs(gated_output - ungated_output)) > 0.0
    assert diagnostics["multiscale_readout_gate_mode"] == "seed_film"
    assert diagnostics["multiscale_readout_gate_params"] > 0


def test_multiscale_vertical_target_gate_routes_fine_drive():
    from oscnet.models import MultiscaleHORNImageGenerator

    base_kwargs = dict(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        label_phase_scale=0.0,
        coupling_init_scale=0.2,
        coupling_profile="local_radius",
        coupling_normalization="row_sum",
        coupling_length_scale=0.8,
        multiscale_layer_sizes=(2, 4),
        multiscale_vertical_strength=1.0,
        multiscale_feedback_strength=0.1,
        multiscale_vertical_profile="local_radius",
        multiscale_vertical_length_scale=1.0,
    )
    conditioning_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_target_gate="conditioning",
        key=jax.random.PRNGKey(950),
    )
    non_conditioning_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_target_gate="non_conditioning",
        key=jax.random.PRNGKey(950),
    )
    soft_conditioning_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_target_gate="conditioning",
        multiscale_vertical_soft_gate_floor=0.25,
        key=jax.random.PRNGKey(950),
    )
    fine_layer = len(conditioning_model.layer_specs) - 1
    fine_spec_index = next(
        index
        for index, spec in enumerate(conditioning_model.vertical_specs)
        if spec.target_layer == fine_layer
    )

    mask = conditioning_model._conditioning_target_mask_array()
    conditioning_profile = conditioning_model.vertical_profile_matrix(fine_spec_index)
    non_conditioning_profile = non_conditioning_model.vertical_profile_matrix(
        fine_spec_index,
    )
    soft_conditioning_profile = soft_conditioning_model.vertical_profile_matrix(
        fine_spec_index,
    )
    conditioning_rows = jnp.sum(jnp.abs(conditioning_profile), axis=-1)
    non_conditioning_rows = jnp.sum(jnp.abs(non_conditioning_profile), axis=-1)
    soft_conditioning_rows = jnp.sum(jnp.abs(soft_conditioning_profile), axis=-1)

    assert int(jnp.sum(mask)) == 2
    assert bool(jnp.all(conditioning_rows[mask == 1.0] > 0.0))
    assert float(jnp.max(conditioning_rows[mask == 0.0])) == 0.0
    assert bool(jnp.all(non_conditioning_rows[mask == 0.0] > 0.0))
    assert float(jnp.max(non_conditioning_rows[mask == 1.0])) == 0.0
    assert bool(jnp.all(soft_conditioning_rows[mask == 1.0] > 0.0))
    assert bool(jnp.all(soft_conditioning_rows[mask == 0.0] > 0.0))
    assert float(jnp.max(soft_conditioning_rows[mask == 0.0])) < float(
        jnp.min(soft_conditioning_rows[mask == 1.0])
    )

    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    trace = conditioning_model.collect_trace(jax.random.PRNGKey(951), 2, labels)
    diagnostics = compute_generator_success_diagnostics(
        conditioning_model,
        trace=trace,
    )

    assert int(trace["vertical_target_gate"]) == 1
    assert diagnostics["multiscale_vertical_target_gate"] == "conditioning"
    assert diagnostics["multiscale_vertical_soft_gate_floor"] == 0.0
    assert diagnostics["vertical_profile_density"] < 1.0
    assert diagnostics["vertical_profile_row_sum_min"] == 0.0

    soft_trace = soft_conditioning_model.collect_trace(
        jax.random.PRNGKey(951),
        2,
        labels,
    )
    soft_diagnostics = compute_generator_success_diagnostics(
        soft_conditioning_model,
        trace=soft_trace,
    )
    assert soft_diagnostics["multiscale_vertical_soft_gate_floor"] == 0.25
    assert soft_diagnostics["vertical_profile_density"] > diagnostics[
        "vertical_profile_density"
    ]
    assert float(jnp.min(soft_conditioning_rows)) > 0.0


def test_multiscale_gain_modulation_changes_layered_dynamics():
    from oscnet.models import MultiscaleHORNImageGenerator

    base_kwargs = dict(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        num_condition_oscillators=3,
        conditioning_mode="class_coupling",
        conditioning_target_fraction=0.25,
        label_phase_scale=0.0,
        coupling_init_scale=0.2,
        coupling_profile="local_radius",
        coupling_normalization="row_sum",
        coupling_length_scale=0.8,
        multiscale_layer_sizes=(2, 4),
        multiscale_vertical_strength=1.0,
        multiscale_feedback_strength=0.1,
        multiscale_vertical_profile="local_radius",
        multiscale_vertical_length_scale=1.0,
        key=jax.random.PRNGKey(960),
    )
    additive_model = MultiscaleHORNImageGenerator(**base_kwargs)
    gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
    )
    scaled_gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
    )
    zero_gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_intervention="zero",
    )
    flipped_gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_intervention="flip",
    )
    signed_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="signed_gain",
    )
    dual_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="dual_gain",
        multiscale_vertical_target_gate="conditioning",
        multiscale_vertical_signal_scale=5.0,
    )
    normalized_gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_gain_normalization="center_rms",
        multiscale_vertical_gain_target_std=0.02,
    )
    normalized_scaled_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_gain_normalization="center_rms",
        multiscale_vertical_gain_target_std=0.02,
        multiscale_vertical_intervention_scale=0.5,
    )
    ramped_gain_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_schedule="linear_ramp",
        multiscale_vertical_onset_step=1,
        multiscale_vertical_ramp_steps=1,
    )
    coupling_target_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_gain_target="coupling",
    )
    conditioning_target_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_gain_target="conditioning",
    )
    damping_target_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_vertical_signal_scale=5.0,
        multiscale_vertical_gain_target="damping",
    )
    state_feedback_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_feedback_signal_mode="state",
    )
    state_feedback_source_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_feedback_signal_mode="state",
        multiscale_feedback_source_gate="conditioning",
    )
    weighted_feedback_source_model = MultiscaleHORNImageGenerator(
        **base_kwargs,
        multiscale_vertical_mode="gain_modulation",
        multiscale_feedback_signal_mode="state",
        multiscale_feedback_source_gate="weighted",
        multiscale_feedback_source_mix=(0.75, 0.25),
    )

    labels = jnp.asarray([0, 1], dtype=jnp.int32)
    sample_key = jax.random.PRNGKey(961)
    additive_generated = additive_model(sample_key, 2, labels)
    gain_generated = gain_model(sample_key, 2, labels)
    scaled_gain_generated = scaled_gain_model(sample_key, 2, labels)
    zero_generated = zero_gain_model(sample_key, 2, labels)
    flipped_generated = flipped_gain_model(sample_key, 2, labels)
    coupling_target_generated = coupling_target_model(sample_key, 2, labels)
    conditioning_target_generated = conditioning_target_model(sample_key, 2, labels)
    damping_target_generated = damping_target_model(sample_key, 2, labels)
    trace = gain_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    scaled_trace = scaled_gain_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    zero_trace = zero_gain_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    signed_trace = signed_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    dual_trace = dual_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    normalized_trace = normalized_gain_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    normalized_scaled_trace = normalized_scaled_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    ramped_trace = ramped_gain_model.collect_trace(jax.random.PRNGKey(962), 2, labels)
    coupling_target_trace = coupling_target_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    conditioning_target_trace = conditioning_target_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    damping_target_trace = damping_target_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    state_feedback_trace = state_feedback_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    state_feedback_source_trace = state_feedback_source_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    weighted_feedback_source_trace = weighted_feedback_source_model.collect_trace(
        jax.random.PRNGKey(962),
        2,
        labels,
    )
    diagnostics = compute_generator_success_diagnostics(gain_model, trace=trace)
    trace_diagnostics = compute_generator_trace_dynamics(gain_model, trace)
    zero_diagnostics = compute_generator_success_diagnostics(
        zero_gain_model,
        trace=zero_trace,
    )
    signed_diagnostics = compute_generator_success_diagnostics(
        signed_model,
        trace=signed_trace,
    )
    dual_diagnostics = compute_generator_success_diagnostics(
        dual_model,
        trace=dual_trace,
    )
    normalized_diagnostics = compute_generator_success_diagnostics(
        normalized_gain_model,
        trace=normalized_trace,
    )
    ramped_diagnostics = compute_generator_success_diagnostics(
        ramped_gain_model,
        trace=ramped_trace,
    )
    coupling_target_diagnostics = compute_generator_success_diagnostics(
        coupling_target_model,
        trace=coupling_target_trace,
    )
    conditioning_target_diagnostics = compute_generator_success_diagnostics(
        conditioning_target_model,
        trace=conditioning_target_trace,
    )
    damping_target_diagnostics = compute_generator_success_diagnostics(
        damping_target_model,
        trace=damping_target_trace,
    )
    state_feedback_diagnostics = compute_generator_success_diagnostics(
        state_feedback_model,
        trace=state_feedback_trace,
    )
    state_feedback_source_diagnostics = compute_generator_success_diagnostics(
        state_feedback_source_model,
        trace=state_feedback_source_trace,
    )
    weighted_feedback_source_diagnostics = compute_generator_success_diagnostics(
        weighted_feedback_source_model,
        trace=weighted_feedback_source_trace,
    )

    assert not bool(jnp.allclose(additive_generated, gain_generated))
    assert not bool(jnp.allclose(gain_generated, zero_generated))
    assert not bool(jnp.allclose(gain_generated, flipped_generated))
    assert not bool(jnp.allclose(scaled_gain_generated, coupling_target_generated))
    assert not bool(
        jnp.allclose(scaled_gain_generated, conditioning_target_generated)
    )
    assert not bool(jnp.allclose(scaled_gain_generated, damping_target_generated))
    assert int(trace["vertical_mode"]) == 1
    assert int(trace["feedback_signal_mode"]) == 0
    assert int(state_feedback_trace["feedback_signal_mode"]) == 1
    assert int(trace["feedback_source_gate"]) == 0
    assert int(state_feedback_source_trace["feedback_source_gate"]) == 1
    assert int(weighted_feedback_source_trace["feedback_source_gate"]) == 3
    assert state_feedback_source_trace["vertical_3_source_gate"].shape == (8,)
    assert float(jnp.sum(state_feedback_source_trace["vertical_3_source_gate"])) == 2.0
    weighted_source_gate = weighted_feedback_source_trace["vertical_3_source_gate"]
    assert weighted_source_gate.shape == (8,)
    assert float(jnp.sum(weighted_source_gate)) == pytest.approx(8.0)
    assert float(jnp.max(weighted_source_gate)) > float(jnp.min(weighted_source_gate))
    aux_positions, aux_velocities = gain_model.initial_auxiliary_state(
        jax.random.PRNGKey(964),
        2,
    )
    fine_position, fine_velocity = gain_model.initial_state(
        jax.random.PRNGKey(965),
        2,
        labels,
    )
    layered_positions = (*aux_positions, fine_position)
    layered_velocities = (*aux_velocities, fine_velocity)
    _, position_feedback_gain, _ = gain_model._vertical_layer_terms(
        1,
        layered_positions[1],
        layered_positions,
        layered_velocities,
        0,
    )
    _, state_feedback_gain, _ = state_feedback_model._vertical_layer_terms(
        1,
        layered_positions[1],
        layered_positions,
        layered_velocities,
        0,
    )
    _, source_gated_feedback_gain, _ = (
        state_feedback_source_model._vertical_layer_terms(
            1,
            layered_positions[1],
            layered_positions,
            layered_velocities,
            0,
        )
    )
    assert not bool(
        jnp.allclose(
            position_feedback_gain,
            state_feedback_gain,
        )
    )
    assert not bool(
        jnp.allclose(
            state_feedback_gain,
            source_gated_feedback_gain,
        )
    )
    assert int(trace["vertical_gain_target"]) == 0
    assert trace["vertical_gain_final"].shape == (2, 8)
    assert trace["vertical_gain_trajectory"].shape == (2, 2, 8)
    assert trace["vertical_modulation_final"].shape == (2, 8)
    assert bool(jnp.all(trace["vertical_gain_final"] >= 0.0))
    assert bool(jnp.all(trace["vertical_gain_final"] <= 2.0))
    assert not bool(jnp.allclose(trace["vertical_gain_final"], 1.0))
    assert diagnostics["multiscale_vertical_mode"] == "gain_modulation"
    assert diagnostics["multiscale_vertical_gain_target"] == "drive"
    assert diagnostics["multiscale_vertical_signal_scale"] == 1.0
    assert diagnostics["multiscale_feedback_signal_mode"] == "position"
    assert diagnostics["multiscale_feedback_source_gate"] == "all"
    assert state_feedback_diagnostics["multiscale_feedback_signal_mode"] == "state"
    assert (
        state_feedback_source_diagnostics["multiscale_feedback_source_gate"]
        == "conditioning"
    )
    assert (
        weighted_feedback_source_diagnostics["multiscale_feedback_source_gate"]
        == "weighted"
    )
    assert weighted_feedback_source_diagnostics["multiscale_feedback_source_mix"] == [
        0.75,
        0.25,
    ]
    scaled_diagnostics = compute_generator_success_diagnostics(
        scaled_gain_model,
        trace=scaled_trace,
    )
    assert scaled_diagnostics["multiscale_vertical_signal_scale"] == 5.0
    assert scaled_diagnostics["vertical_gain_std"] > diagnostics["vertical_gain_std"]
    assert diagnostics["multiscale_vertical_intervention"] == "normal"
    assert "vertical_gain_mean" in diagnostics
    assert "vertical_gain_target_minus_non_target_mean" in diagnostics
    assert "vertical_modulation_negative_fraction" in trace_diagnostics
    assert int(zero_trace["vertical_intervention"]) == 1
    assert bool(jnp.allclose(zero_trace["vertical_gain_final"], 1.0))
    assert zero_diagnostics["multiscale_vertical_intervention"] == "zero"
    assert int(signed_trace["vertical_mode"]) == 2
    assert signed_trace["vertical_gain_final"].shape == (2, 8)
    assert bool(jnp.all(signed_trace["vertical_gain_final"] >= -1.0))
    assert bool(jnp.all(signed_trace["vertical_gain_final"] <= 2.0))
    assert signed_diagnostics["multiscale_vertical_mode"] == "signed_gain"
    assert int(dual_trace["vertical_mode"]) == 3
    assert dual_trace["vertical_gain_final"].shape == (2, 8)
    assert bool(jnp.all(dual_trace["vertical_gain_final"] >= -1.0))
    assert bool(jnp.all(dual_trace["vertical_gain_final"] <= 2.0))
    assert dual_diagnostics["multiscale_vertical_mode"] == "dual_gain"
    assert dual_diagnostics["multiscale_vertical_signal_scale"] == 5.0
    assert int(normalized_trace["vertical_gain_normalization"]) == 2
    assert float(normalized_trace["vertical_gain_target_std"]) == pytest.approx(0.02)
    normalized_modulation = normalized_trace["vertical_modulation_final"]
    normalized_modulation_mean = jnp.mean(normalized_modulation, axis=-1)
    normalized_modulation_rms = jnp.sqrt(
        jnp.mean(normalized_modulation**2, axis=-1)
    )
    normalized_scaled_modulation_rms = jnp.sqrt(
        jnp.mean(
            normalized_scaled_trace["vertical_modulation_final"] ** 2,
            axis=-1,
        )
    )
    assert bool(jnp.allclose(normalized_modulation_mean, 0.0, atol=1e-5))
    assert bool(jnp.allclose(normalized_modulation_rms, 0.02, atol=1e-3))
    assert bool(
        jnp.allclose(normalized_scaled_modulation_rms, 0.01, atol=1e-3)
    )
    assert (
        normalized_diagnostics["multiscale_vertical_gain_normalization"]
        == "center_rms"
    )
    assert normalized_diagnostics[
        "multiscale_vertical_gain_target_std"
    ] == pytest.approx(0.02)
    assert int(ramped_trace["vertical_schedule"]) == 2
    assert int(ramped_trace["vertical_onset_step"]) == 1
    assert int(ramped_trace["vertical_ramp_steps"]) == 1
    assert bool(jnp.allclose(ramped_trace["vertical_schedule_trajectory"], jnp.asarray([0.0, 1.0])))
    assert bool(jnp.allclose(ramped_trace["vertical_gain_trajectory"][0], 1.0))
    assert not bool(jnp.allclose(ramped_trace["vertical_gain_trajectory"][-1], 1.0))
    assert ramped_diagnostics["multiscale_vertical_schedule"] == "linear_ramp"
    assert ramped_diagnostics["multiscale_vertical_onset_step"] == 1
    assert ramped_diagnostics["multiscale_vertical_ramp_steps"] == 1
    assert int(coupling_target_trace["vertical_gain_target"]) == 1
    assert int(conditioning_target_trace["vertical_gain_target"]) == 2
    assert int(damping_target_trace["vertical_gain_target"]) == 3
    assert (
        coupling_target_diagnostics["multiscale_vertical_gain_target"]
        == "coupling"
    )
    assert (
        conditioning_target_diagnostics["multiscale_vertical_gain_target"]
        == "conditioning"
    )
    assert damping_target_diagnostics["multiscale_vertical_gain_target"] == "damping"

    real = jnp.linspace(0.0, 1.0, 4 * 64, dtype=jnp.float32).reshape(4, 64)
    audit_labels = jnp.asarray([0, 1, 2, 0], dtype=jnp.int32)
    audit = compute_generator_vertical_intervention_audit(
        gain_model,
        key=jax.random.PRNGKey(963),
        real_images=real,
        sample_count=4,
        batch_size=2,
        labels=audit_labels,
        modes=("normal", "zero", "flip", "scale025"),
        trace_batch_size=2,
    )
    assert set(audit) == {"normal", "zero", "flip", "scale025"}
    assert audit["normal"]["output_mse_vs_normal"] == 0.0
    assert audit["zero"]["output_mse_vs_normal"] >= 0.0
    assert "delta_nearest_real_mse" in audit["zero"]
    assert "trace_vertical_gain_mean" in audit["normal"]


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


def test_generator_attractor_robustness_scores_same_label_variants():
    from oscnet.models import HORNImageGenerator

    model = HORNImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=1,
        num_classes=3,
        conditioning_mode="phase_shift",
        key=jax.random.PRNGKey(890),
    )
    classifier = MNISTFeatureClassifier(
        image_dim=64,
        num_classes=3,
        feature_dim=8,
        depth=1,
        key=jax.random.PRNGKey(891),
    )

    metrics = compute_generator_attractor_robustness(
        model,
        key=jax.random.PRNGKey(892),
        batch_size=2,
        variants_per_class=3,
        num_classes=3,
        classifier=classifier,
    )

    assert metrics["num_classes"] == 3.0
    assert metrics["variants_per_class"] == 3.0
    assert metrics["sample_count"] == 9.0
    assert 0.0 <= metrics["label_accuracy"] <= 1.0
    assert metrics["pixel_within_class_pairwise_mse"] >= 0.0
    assert metrics["pixel_between_class_centroid_mse"] >= 0.0
    assert metrics["pixel_attractor_diversity_score"] >= 0.0
    assert metrics["feature_within_class_pairwise_distance"] >= 0.0
    assert metrics["feature_attractor_diversity_score"] >= 0.0
    assert "feature_separation_ratio" in metrics


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
    assert model.coupling_normalization == "none"
    assert diagnostics["dynamics_family"] == "state_mlp"
    assert diagnostics["coupling_density"] == 0.0
    assert diagnostics["coupling_normalization"] == "none"
    assert diagnostics["transition_params"] > 0
    assert diagnostics["recurrent_params"] == diagnostics["transition_params"]
    assert diagnostics["state_mean_abs_velocity_displacement"] >= 0.0
    assert diagnostics["state_update_rms_mean"] >= 0.0
    assert diagnostics["output_step_mse_mean"] >= 0.0


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
        quality_classifier_kind="conv",
        quality_classifier_dim=8,
        quality_classifier_depth=1,
        eval_sample_count=2,
        attractor_variants_per_class=1,
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
    assert summary["generator"]["quality_classifier"]["classifier_kind"] == "conv"
    assert summary["generator"]["quality_classifier_kind"] == "conv"
    assert "classifier_label_accuracy" in summary["generator"]
    assert summary["generator"]["settling"]["steps"] == [0, 1]
    assert summary["generator"]["attractor_variants_per_class"] == 1
    assert "attractor_robustness" in summary["generator"]
    assert summary["generator"]["attractor_robustness"]["sample_count"] == 10.0
    assert "label_accuracy" in summary["generator"]["attractor_robustness"]
    assert diagnostics["dynamics_family"] == "horn"
    assert "state_final_energy" in diagnostics
    assert "state_update_rms_settling_ratio" in diagnostics
    assert "output_step_mse_settling_ratio" in diagnostics
    assert summary["final_eval_loss"] >= 0.0


def test_mnist_generator_multiscale_auxiliary_synthetic_training_smoke(tmp_path):
    run = AutoencoderExperimentConfig(
        name="mnist_generator_multiscale_aux_test",
        output_dir=tmp_path / "mnist_generator_multiscale_aux",
        seed=86,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
        artifact_every=1,
    )
    config = MNISTGeneratorExperimentConfig(
        run=run,
        model_family="multiscale_horn",
        conditional=True,
        num_classes=10,
        conditioning_mode="class_coupling",
        num_condition_oscillators=3,
        label_phase_scale=0.0,
        readout_mode="mean_relative",
        decoder_mode="resize_conv",
        resize_conv_min_channels=4,
        num_oscillators=98,
        multiscale_layer_sizes=(2,),
        multiscale_frequency_scales=(0.5,),
        multiscale_auxiliary_readout_layer=0,
        coarse_auxiliary_weight=0.1,
        coarse_auxiliary_target_size=7,
        coarse_auxiliary_loss_mode="distributional",
        coarse_readout_consistency_weight=0.05,
        coarse_readout_consistency_onset_epoch=1,
        frequency_objective_weight=0.01,
        frequency_objective_edge_weight=0.5,
        patch_objective_weight=0.02,
        patch_objective_patch_size=3,
        patch_objective_stride=2,
        patch_objective_projections=4,
        patch_objective_edge_weight=0.25,
        steps=1,
        num_projections=8,
        eval_sample_count=2,
        attractor_variants_per_class=1,
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
    assert summary["generator"]["dynamics_family"] == "multiscale_horn"
    assert summary["generator"]["coarse_auxiliary_weight"] == 0.1
    assert summary["generator"]["coarse_auxiliary_loss_mode"] == "distributional"
    assert summary["generator"]["coarse_readout_consistency_weight"] == 0.05
    assert summary["generator"]["coarse_readout_consistency_onset_epoch"] == 1
    assert summary["generator"]["frequency_objective_weight"] == 0.01
    assert summary["generator"]["frequency_objective_edge_weight"] == 0.5
    assert summary["generator"]["patch_objective_weight"] == 0.02
    assert summary["generator"]["patch_objective_patch_size"] == 3
    assert summary["generator"]["patch_objective_stride"] == 2
    assert summary["generator"]["patch_objective_projections"] == 4
    assert summary["generator"]["patch_objective_edge_weight"] == 0.25
    assert summary["final_train_coarse_auxiliary_loss"] >= 0.0
    assert summary["final_eval_coarse_auxiliary_loss"] >= 0.0
    assert summary["final_train_coarse_readout_consistency_loss"] >= 0.0
    assert summary["final_eval_coarse_readout_consistency_loss"] >= 0.0
    assert summary["final_train_frequency_objective_loss"] >= 0.0
    assert summary["final_eval_frequency_objective_loss"] >= 0.0
    assert summary["final_train_patch_objective_loss"] >= 0.0
    assert summary["final_eval_patch_objective_loss"] >= 0.0
    assert diagnostics["auxiliary_readout_params"] > 0


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
    assert summary["generator"]["learned_feature_kind"] == "mlp"
    assert summary["generator"]["feature_classifier"]["epochs"] == 1
    assert summary["generator"]["feature_classifier"]["classifier_kind"] == "mlp"
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
    assert diagnostics["phase_update_rms_mean"] >= 0.0
    assert diagnostics["output_step_mse_mean"] >= 0.0


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
    assert "state_final_spatial_high_power_ratio" in diagnostics
    assert "state_final_spatial_spectral_centroid" in diagnostics
    assert "state_spatial_high_power_ratio_delta" in diagnostics
    assert "velocity_final_spatial_high_power_ratio" not in diagnostics


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


def test_generator_main_coupling_strength_defaults_and_overrides():
    from oscnet.models import HORNImageGenerator

    default_model = HORNImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=1.5,
        key=jax.random.PRNGKey(14),
    )
    assert default_model.coupling_strength == 1.5
    assert default_model.main_coupling_strength == 1.5

    split_model = HORNImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=1.5,
        main_coupling_strength=0.25,
        key=jax.random.PRNGKey(15),
    )
    assert split_model.coupling_strength == 1.5
    assert split_model.main_coupling_strength == 0.25

    diagnostics = compute_generator_success_diagnostics(split_model)
    assert diagnostics["coupling_strength"] == 1.5
    assert diagnostics["main_coupling_strength"] == 0.25

    no_main = HORNImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=0.0,
        main_coupling_strength=0.0,
        coupling_init_scale=0.0,
        coupling_bias_strength=1.0,
        key=jax.random.PRNGKey(16),
    )
    main_on = HORNImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=0.0,
        main_coupling_strength=1.0,
        coupling_init_scale=0.0,
        coupling_bias_strength=1.0,
        key=jax.random.PRNGKey(16),
    )
    state = (
        jnp.asarray([[0.0, 1.0, -0.5, 0.25, -1.0, 0.5]], dtype=jnp.float32),
        jnp.zeros((1, 6), dtype=jnp.float32),
    )
    no_main_next, _ = no_main.step_state(state)
    main_on_next, _ = main_on.step_state(state)
    assert jnp.max(jnp.abs(main_on_next - no_main_next)) > 0.0


def test_generator_split_coupling_preserves_legacy_default_step():
    from oscnet.models import HORNImageGenerator, KuramotoImageGenerator

    kuramoto = KuramotoImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=1.7,
        coupling_init_scale=0.2,
        coupling_bias_strength=0.1,
        num_classes=3,
        num_condition_oscillators=2,
        conditioning_mode="class_coupling",
        conditioning_strength=0.8,
        key=jax.random.PRNGKey(17),
    )
    theta = jnp.asarray(
        [[0.0, 0.3, -0.4, 1.0, -1.2, 0.7]],
        dtype=jnp.float32,
    )
    labels = jnp.asarray([2], dtype=jnp.int32)
    omega, coupling = kuramoto._dynamics_params()
    phase_diff = theta[:, None, :] - theta[:, :, None]
    interaction = jnp.sum(coupling[None, :, :] * jnp.sin(phase_diff), axis=-1)
    condition_drive = kuramoto._conditioning_drive(theta, labels)
    legacy_velocity = omega[None, :] + kuramoto.coupling_strength * (
        interaction / float(kuramoto.num_oscillators) + condition_drive
    )
    expected_theta = jnp.angle(jnp.exp(1j * (theta + kuramoto.dt * legacy_velocity)))

    assert kuramoto.main_coupling_strength == kuramoto.coupling_strength
    assert jnp.allclose(kuramoto.step(theta, labels), expected_theta, atol=1e-6)

    horn = HORNImageGenerator(
        num_oscillators=6,
        image_shape=(8, 8),
        decoder_depth=0,
        steps=1,
        coupling_strength=1.7,
        coupling_init_scale=0.2,
        coupling_bias_strength=0.1,
        num_classes=3,
        num_condition_oscillators=2,
        conditioning_mode="class_coupling",
        conditioning_strength=0.8,
        key=jax.random.PRNGKey(18),
    )
    position = jnp.asarray(
        [[0.0, 0.3, -0.4, 1.0, -1.2, 0.7]],
        dtype=jnp.float32,
    )
    velocity = jnp.asarray(
        [[0.2, -0.1, 0.4, -0.3, 0.1, -0.2]],
        dtype=jnp.float32,
    )
    frequency, coupling = horn._horn_dynamics_params()
    displacement = position[:, None, :] - position[:, :, None]
    interaction = jnp.sum(coupling[None, :, :] * displacement, axis=-1)
    condition_drive = horn._horn_static_conditioning_drive(position, labels)
    legacy_acceleration = (
        -(frequency[None, :] ** 2) * position
        - float(horn.horn_damping) * velocity
        - float(horn.horn_nonlinearity) * (position**3)
        + horn.coupling_strength
        * (interaction / float(horn.num_oscillators) + condition_drive)
    )
    expected_velocity = horn._bound_state(velocity + horn.dt * legacy_acceleration)
    expected_position = horn._bound_state(position + horn.dt * expected_velocity)
    actual_position, actual_velocity = horn.step_state((position, velocity), labels)

    assert horn.main_coupling_strength == horn.coupling_strength
    assert jnp.allclose(actual_position, expected_position, atol=1e-6)
    assert jnp.allclose(actual_velocity, expected_velocity, atol=1e-6)


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
    assert diagnostics["coupling_normalization"] == "none"


def test_generator_normalized_distance_decay_profile_reports_row_gain():
    from oscnet.models import KuramotoImageGenerator

    model = KuramotoImageGenerator(
        num_oscillators=9,
        image_shape=(8, 8),
        decoder_mode="local_basis",
        local_patch_size=3,
        decoder_depth=0,
        steps=2,
        coupling_profile="distance_decay",
        coupling_normalization="row_sum",
        coupling_length_scale=0.6,
        coupling_floor=0.0,
        key=jax.random.PRNGKey(151),
    )
    profile = model.coupling_profile_matrix()
    diagnostics = compute_generator_success_diagnostics(model)

    assert model.coupling_normalization == "row_sum"
    assert jnp.allclose(jnp.sum(profile, axis=-1), 9.0, atol=1e-5)
    assert diagnostics["coupling_normalization"] == "row_sum"
    assert diagnostics["coupling_profile_row_sum_mean"] == pytest.approx(9.0)
    assert diagnostics["coupling_profile_row_sum_std"] < 1e-5


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


def _tiny_hybrid(key=None):
    from oscnet.models import HybridImageGenerator

    return HybridImageGenerator(
        num_oscillators=4,
        num_modes=2,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        state_anchor_encoder_enabled=True,
        state_mlp_hidden_dim=8,
        state_mlp_depth=1,
        router_hidden_dim=4,
        router_bias_init=-1.0,
        key=key if key is not None else jax.random.PRNGKey(430),
    )


def test_hybrid_generator_routes_between_paths():
    model = _tiny_hybrid()
    labels = jnp.asarray([0, 1], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(600), 2, labels)
    assert generated.shape == (2, 4 * 4)

    images = jax.random.uniform(jax.random.PRNGKey(601), (2, 4 * 4))
    position, velocity = model.encode_image_state(images)
    gate = model.router_gate(position, velocity)
    assert gate.shape == (2, model.num_spatial_sites)
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0
    # Bias init -1.0 starts the router trusting the free-form path.
    assert float(gate.mean()) < 0.5

    hybrid_next = model.step_state((position, velocity), labels)
    horn_next = super(type(model), model).step_state(
        (position, velocity), labels
    )
    assert not jnp.allclose(hybrid_next[0], horn_next[0])
    decoded = model.decode_state(*hybrid_next)
    assert decoded.shape == (2, 4 * 4)


def test_hybrid_router_modes_fixed_and_oracle():
    from oscnet.models import HybridImageGenerator

    base_kwargs = dict(
        num_oscillators=4,
        num_modes=2,
        image_shape=(4, 4),
        decoder_mode="resize_conv",
        resize_conv_seed_shape=(2, 2),
        resize_conv_seed_layout="retinotopic",
        resize_conv_upsamples=1,
        resize_conv_min_channels=2,
        steps=1,
        num_classes=2,
        conditioning_mode="class_coupling",
        num_condition_oscillators=2,
        state_anchor_encoder_enabled=True,
        state_mlp_hidden_dim=8,
        router_hidden_dim=4,
        router_bias_init=-1.0,
        key=jax.random.PRNGKey(431),
    )
    fixed = HybridImageGenerator(router_mode="fixed_statistic", **base_kwargs)
    images = jax.random.uniform(jax.random.PRNGKey(602), (2, 16))
    position, velocity = fixed.encode_image_state(images)
    gate = fixed.router_gate(position, velocity)
    assert gate.shape == (2, fixed.num_spatial_sites)

    oracle = HybridImageGenerator(router_mode="oracle", **base_kwargs)
    mask = jnp.asarray(
        [[1.0, 0.0, 0.5, 0.25], [0.25, 0.75, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    oracle = oracle.with_oracle_gate(mask)
    assert jnp.allclose(oracle.router_gate(position, velocity), mask)

    free = HybridImageGenerator(router_mode="free_form", **base_kwargs)
    assert float(free.router_gate(position, velocity).max()) == 0.0



def test_state_anchor_occlusion_curriculum_loss_is_finite():
    from oscnet.experiments.mnist_generator.runner import (
        _state_anchor_image_loss,
    )

    model = _tiny_anchor_horn()
    real = jax.random.uniform(jax.random.PRNGKey(610), (4, 16))

    loss = _state_anchor_image_loss(
        model,
        real,
        key=jax.random.PRNGKey(611),
        state_anchor_weight=1.0,
        state_anchor_steps=(1,),
        state_anchor_noise_scale=0.1,
        state_anchor_mode="settle",
        state_anchor_occlusion_fraction=0.0,
        state_anchor_occlusion_patches=1,
        state_anchor_occlusion_probability=1.0,
        state_anchor_occlusion_curriculum=(0.1, 0.4),
        state_anchor_clean_weight=0.5,
    )

    assert jnp.isfinite(loss)
    assert float(loss) > 0.0


def test_heldout_corruption_families_change_images():
    from oscnet.experiments.mnist_generator.metrics import (
        _corrupt_images_heldout,
    )

    images = jax.random.uniform(jax.random.PRNGKey(620), (3, 8 * 8 * 3))
    for family, level in (
        ("gaussian", 0.3),
        ("salt_pepper", 0.2),
        ("stripes", 0.5),
    ):
        corrupted, mask = _corrupt_images_heldout(
            images,
            key=jax.random.PRNGKey(621),
            image_shape=(8, 8, 3),
            family=family,
            level=level,
        )
        assert corrupted.shape == images.shape
        assert not jnp.allclose(corrupted, images)
        assert float(corrupted.min()) >= 0.0
        assert float(corrupted.max()) <= 1.0
        if family == "stripes":
            assert mask is not None
            assert mask.shape == (3, 8 * 8 * 3)
            fraction = float(mask.mean())
            assert 0.3 < fraction < 0.7
        else:
            assert mask is None


def test_robustness_metrics_score_heldout_battery():
    model = _tiny_anchor_horn()
    real_images = jax.random.uniform(jax.random.PRNGKey(630), (8, 16))

    metrics = compute_generator_robustness_metrics(
        model,
        real_images,
        key=jax.random.PRNGKey(631),
        image_shape=(4, 4),
        sample_count=8,
        settle_step=1,
        weight_noise_scales=(),
        quant_bits=(),
        ood_occlusion_fractions=(0.25,),
        weight_noise_draws=1,
        heldout_corruptions=("gaussian:0.3", "stripes:0.5"),
    )

    assert metrics["heldout_c0_spec"] == "gaussian:0.3"
    assert metrics["heldout_c0_level"] == 0.3
    assert metrics["heldout_c0_mse"] >= 0.0
    assert np.isfinite(metrics["heldout_c0_psnr"])
    assert "heldout_c0_region_mse" not in metrics
    assert metrics["heldout_c1_spec"] == "stripes:0.5"
    assert metrics["heldout_c1_mse"] >= 0.0
    assert metrics["heldout_c1_region_mse"] >= 0.0
