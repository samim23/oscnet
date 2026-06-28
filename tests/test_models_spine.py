import jax
import jax.numpy as jnp

from oscnet.models import (
    AutoregressiveOscillatoryDecoder,
    ConvLSTMPatchDenoiser,
    FeedForwardPatchAutoencoder,
    KuramotoImageGenerator,
    OscillatoryAutoencoder,
    PatchOscillatoryAutoencoder,
    PositionalLatentOscillatoryDecoder,
    RecurrentConvPatchDenoiser,
    RecurrentConvPriorRefinementPatchDenoiser,
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser,
    WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser,
    WinfreeCoarseRatePhaseConditionalPatchDenoiser,
    WinfreeConditionalPatchDenoiser,
    WinfreeFieldLayer,
    WinfreeGlobalRatePhaseConditionalPatchDenoiser,
    WinfreePatchAutoencoder,
    WinfreePriorRefinementPatchDenoiser,
    WinfreeRatePhaseConditionalPatchDenoiser,
)


def test_sequence_autoencoder_repeat_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=8,
        hidden_dim=16,
        latent_dim=4,
        sequence_length=5,
        decoder_mode="repeat",
        key=jax.random.PRNGKey(0),
    )
    inputs = jnp.ones((5, 3, 8))

    reconstruction, latent = model(inputs, return_latent=True)

    assert reconstruction.shape == inputs.shape
    assert latent.shape == (3, 4)
    assert jnp.all(jnp.isfinite(reconstruction))
    assert jnp.all(jnp.isfinite(latent))


def test_sequence_autoencoder_autoregressive_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=6,
        hidden_dim=12,
        latent_dim=5,
        sequence_length=4,
        decoder_mode="autoregressive",
        key=jax.random.PRNGKey(1),
    )
    inputs = jnp.ones((4, 2, 6))

    reconstruction = model(inputs)

    assert reconstruction.shape == inputs.shape
    assert jnp.all(jnp.isfinite(reconstruction))
    assert isinstance(model.decoder, AutoregressiveOscillatoryDecoder)


def test_sequence_autoencoder_positional_mode_shapes():
    model = OscillatoryAutoencoder(
        input_dim=6,
        hidden_dim=12,
        latent_dim=5,
        sequence_length=4,
        decoder_mode="positional",
        latent_conditioning_strength=2.0,
        key=jax.random.PRNGKey(3),
    )
    inputs = jnp.ones((4, 2, 6))

    reconstruction, latent = model(inputs, return_latent=True)

    assert reconstruction.shape == inputs.shape
    assert latent.shape == (2, 5)
    assert isinstance(model.decoder, PositionalLatentOscillatoryDecoder)
    assert model.decoder.positional_inputs.shape == (4, 12)
    assert jnp.all(jnp.isfinite(reconstruction))


def test_sequence_autoencoder_output_activation_bounds_reconstructions():
    model = OscillatoryAutoencoder(
        input_dim=6,
        hidden_dim=12,
        latent_dim=5,
        sequence_length=4,
        decoder_mode="positional",
        output_activation="sigmoid",
        key=jax.random.PRNGKey(7),
    )
    inputs = jnp.ones((4, 2, 6))

    reconstruction, latent = model(inputs, return_latent=True)
    decoded = model.decode(latent, sequence_length=inputs.shape[0])

    assert reconstruction.shape == inputs.shape
    assert decoded.shape == inputs.shape
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)
    assert jnp.all(decoded >= 0.0)
    assert jnp.all(decoded <= 1.0)


def test_patch_autoencoder_maps_flat_images_back_to_flat_images():
    model = PatchOscillatoryAutoencoder(
        hidden_dim=8,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        key=jax.random.PRNGKey(2),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert jnp.all(jnp.isfinite(reconstruction))


def test_feedforward_patch_autoencoder_maps_flat_images_back_to_flat_images():
    model = FeedForwardPatchAutoencoder(
        hidden_dim=8,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        output_activation="sigmoid",
        key=jax.random.PRNGKey(13),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 4)
    assert trace["encoder_hidden"].shape == (2, 4, 8)
    assert trace["decoder_hidden"].shape == (2, 4, 8)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_kuramoto_image_generator_samples_from_phase_noise():
    model = KuramotoImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(30),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(31), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(32), 3, labels)

    assert generated.shape == (3, 64)
    assert trace["initial_theta"].shape == (3, 8)
    assert trace["theta_trajectory"].shape == (2, 3, 8)
    assert trace["final_theta"].shape == (3, 8)
    assert trace["generated"].shape == (3, 64)
    assert trace["coupling"].shape == (8, 8)
    assert trace["label_phase_shift"].shape == (3, 8)
    assert trace["label_condition_phase"].shape == (0, 0)
    assert trace["label_condition_coupling"].shape == (0, 8, 0)
    assert jnp.all(generated >= 0.0)
    assert jnp.all(generated <= 1.0)


def test_kuramoto_image_generator_supports_class_coupled_conditioning():
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
        output_activation="sigmoid",
        key=jax.random.PRNGKey(33),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(34), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(35), 3, labels)

    assert generated.shape == (3, 64)
    assert trace["label_phase_shift"].shape == (0, 8)
    assert trace["label_condition_phase"].shape == (3, 4)
    assert trace["label_condition_coupling"].shape == (3, 8, 4)
    assert model.conditioning_mode == "class_coupling"
    assert model.readout_mode == "relative"
    assert jnp.all(jnp.isfinite(generated))


def test_kuramoto_image_generator_supports_condition_oscillator_drive():
    model = KuramotoImageGenerator(
        num_oscillators=8,
        image_shape=(8, 8),
        decoder_hidden_dim=12,
        decoder_depth=1,
        steps=2,
        num_classes=3,
        num_condition_oscillators=4,
        conditioning_mode="class_oscillator",
        readout_mode="mean_relative",
        output_activation="sigmoid",
        key=jax.random.PRNGKey(49),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(50), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(51), 3, labels)

    assert generated.shape == (3, 64)
    assert trace["condition_initial_theta"].shape == (3, 4)
    assert trace["condition_theta_trajectory"].shape == (2, 3, 4)
    assert trace["condition_final_theta"].shape == (3, 4)
    assert trace["condition_omega"].shape == (4,)
    assert trace["condition_coupling"].shape == (4, 4)
    assert trace["label_condition_phase"].shape == (0, 4)
    assert trace["label_condition_coupling"].shape == (3, 8, 4)
    assert model.conditioning_mode == "class_oscillator"
    assert model.readout_mode == "mean_relative"
    assert jnp.all(jnp.isfinite(generated))


def test_kuramoto_image_generator_supports_spatial_basis_readout():
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
        output_activation="sigmoid",
        key=jax.random.PRNGKey(36),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(37), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(38), 3, labels)

    assert generated.shape == (3, 64)
    assert trace["spatial_phase_weights"].shape == (9, 2)
    assert trace["spatial_output_bias"].shape == ()
    assert model.decoder_mode == "spatial_basis"
    assert jnp.all(generated >= 0.0)
    assert jnp.all(generated <= 1.0)


def test_kuramoto_image_generator_supports_local_basis_readout():
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
        output_activation="sigmoid",
        key=jax.random.PRNGKey(39),
    )
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    generated = model(jax.random.PRNGKey(40), 3, labels)
    trace = model.collect_trace(jax.random.PRNGKey(41), 3, labels)

    assert generated.shape == (3, 64)
    assert trace["local_patch_weights"].shape == (9, 2, 9)
    assert model.decoder_mode == "local_basis"
    assert jnp.all(generated >= 0.0)
    assert jnp.all(generated <= 1.0)


def test_kuramoto_image_generator_supports_distance_decay_coupling():
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
        output_activation="sigmoid",
        key=jax.random.PRNGKey(42),
    )

    generated = model(jax.random.PRNGKey(43), 2)
    trace = model.collect_trace(jax.random.PRNGKey(44), 2)
    profile = model.coupling_profile_matrix()
    effective_coupling = model._dynamics_params()[1]

    assert generated.shape == (2, 64)
    assert trace["coupling_profile"].shape == (9, 9)
    assert profile.shape == (9, 9)
    assert jnp.allclose(jnp.diag(profile), 0.0)
    assert jnp.max(profile) <= 1.0
    assert jnp.min(profile + jnp.eye(9)) >= 0.05
    assert jnp.allclose(
        effective_coupling - model.coupling * profile,
        0.1 * profile,
    )
    assert model.coupling_profile == "distance_decay"
    assert jnp.all(generated >= 0.0)
    assert jnp.all(generated <= 1.0)


def test_kuramoto_image_generator_supports_split_dynamics_trainability():
    conditioning_only = KuramotoImageGenerator(
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
        output_activation="sigmoid",
        key=jax.random.PRNGKey(45),
    )
    recurrent_only = KuramotoImageGenerator(
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
        train_recurrent_dynamics=True,
        train_conditioning_dynamics=False,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(46),
    )
    labels = jnp.asarray([0, 1], dtype=jnp.int32)

    assert conditioning_only(jax.random.PRNGKey(47), 2, labels).shape == (2, 64)
    assert recurrent_only(jax.random.PRNGKey(48), 2, labels).shape == (2, 64)
    assert conditioning_only.train_dynamics
    assert conditioning_only.train_conditioning_dynamics
    assert not conditioning_only.train_recurrent_dynamics
    assert recurrent_only.train_dynamics
    assert recurrent_only.train_recurrent_dynamics
    assert not recurrent_only.train_conditioning_dynamics


def test_core_patch_models_accept_extra_input_channel_and_output_one_image():
    images = jnp.ones((2, 2 * 64))
    model_specs = [
        (
            FeedForwardPatchAutoencoder,
            {
                "latent_dim": 4,
                "output_activation": "sigmoid",
            },
        ),
        (
            RecurrentConvPatchDenoiser,
            {
                "steps": 2,
                "output_activation": "sigmoid",
            },
        ),
        (
            ConvLSTMPatchDenoiser,
            {
                "steps": 2,
                "output_activation": "sigmoid",
            },
        ),
        (
            WinfreeRatePhaseConditionalPatchDenoiser,
            {
                "steps": 2,
                "output_activation": "sigmoid",
            },
        ),
        (
            WinfreeGlobalRatePhaseConditionalPatchDenoiser,
            {
                "steps": 2,
                "output_activation": "sigmoid",
            },
        ),
    ]

    for index, (model_cls, kwargs) in enumerate(model_specs):
        model = model_cls(
            input_dim=32,
            hidden_dim=6,
            image_shape=(8, 8),
            patch_shape=(4, 4),
            key=jax.random.PRNGKey(100 + index),
            **kwargs,
        )

        sequence = model.images_to_sequence(images)
        reconstruction = model(images)

        assert sequence.shape == (4, 2, 32)
        assert reconstruction.shape == (2, 64)
        assert jnp.all(reconstruction >= 0.0)
        assert jnp.all(reconstruction <= 1.0)


def test_recurrent_conv_patch_denoiser_maps_flat_images_back_to_flat_images():
    model = RecurrentConvPatchDenoiser(
        hidden_dim=8,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=3,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(15),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 8)
    assert trace["initial_hidden"].shape == (2, 4, 8)
    assert trace["hidden_states"].shape == (3, 2, 4, 8)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_conv_lstm_patch_denoiser_maps_flat_images_back_to_flat_images():
    model = ConvLSTMPatchDenoiser(
        hidden_dim=8,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=3,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(16),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 8)
    assert trace["drive"].shape == (2, 4, 8)
    assert trace["hidden_states"].shape == (3, 2, 4, 8)
    assert trace["cell_states"].shape == (3, 2, 4, 8)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_field_layer_records_phase_and_energy_trace():
    layer = WinfreeFieldLayer(
        num_positions=4,
        channels=6,
        steps=3,
        gamma=0.1,
        key=jax.random.PRNGKey(4),
    )
    theta = jnp.zeros((2, 4, 6))
    omega = jnp.ones((2, 4, 6)) * 0.05

    trace = layer(theta, omega, return_trajectory=True)

    assert trace["final_theta"].shape == theta.shape
    assert trace["thetas"].shape == (3, 2, 4, 6)
    assert trace["energies"].shape == (3, 2)
    assert jnp.all(trace["thetas"] <= jnp.pi + 1e-6)
    assert jnp.all(trace["thetas"] >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(trace["energies"]))


def test_winfree_patch_autoencoder_maps_flat_images_back_to_flat_images():
    model = WinfreePatchAutoencoder(
        hidden_dim=6,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        key=jax.random.PRNGKey(5),
    )
    images = jnp.ones((2, 64))

    sequence = model.images_to_sequence(images)
    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert sequence.shape == (4, 2, 16)
    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 4)
    assert trace["encoder_thetas"].shape[:3] == (2, 2, 4)
    assert trace["decoder_thetas"].shape[:3] == (2, 2, 4)
    assert jnp.all(jnp.isfinite(reconstruction))


def test_winfree_conditional_patch_denoiser_maps_flat_images_back_to_flat_images():
    model = WinfreeConditionalPatchDenoiser(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(14),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 12)
    assert trace["omega"].shape == (2, 4, 6)
    assert trace["thetas"].shape == (2, 2, 4, 6)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_conditional_patch_denoiser_supports_rotary_2d_phase_init():
    model_a = WinfreeConditionalPatchDenoiser(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        phase_init="rotary_2d",
        phase_init_scale=0.75,
        key=jax.random.PRNGKey(16),
    )
    model_b = WinfreeConditionalPatchDenoiser(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        phase_init="rotary_2d",
        phase_init_scale=0.75,
        key=jax.random.PRNGKey(17),
    )

    assert model_a.phase_init == "rotary_2d"
    assert model_a.positional_theta.shape == (4, 6)
    assert jnp.allclose(model_a.positional_theta, model_b.positional_theta)
    assert jnp.all(model_a.positional_theta <= jnp.pi + 1e-6)
    assert jnp.all(model_a.positional_theta >= -jnp.pi - 1e-6)


def test_winfree_rate_phase_conditional_denoiser_maps_flat_images_back_to_flat_images():
    model = WinfreeRatePhaseConditionalPatchDenoiser(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(18),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 18)
    assert trace["omega"].shape == (2, 4, 6)
    assert trace["thetas"].shape == (2, 2, 4, 6)
    assert trace["rate_states"].shape == (2, 2, 4, 6)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_rate_phase_visibility_gate_uses_second_input_channel():
    model = WinfreeRatePhaseConditionalPatchDenoiser(
        input_dim=32,
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        visibility_gate="visibility",
        visibility_drive_floor=0.0,
        missing_transport_strength=1.0,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(118),
    )
    image = jnp.ones((2, 64))
    visibility = jnp.concatenate([jnp.ones((2, 32)), jnp.zeros((2, 32))], axis=1)
    inputs = jnp.concatenate([image, visibility], axis=1)

    reconstruction = model(inputs)
    trace = model.collect_trace(inputs)

    assert reconstruction.shape == (2, 64)
    assert trace["visibility"].shape == (2, 4, 1)
    assert jnp.any(trace["visibility"] == 0.0)
    assert jnp.any(trace["visibility"] == 1.0)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_global_rate_phase_conditional_denoiser_traces_global_phase():
    model = WinfreeGlobalRatePhaseConditionalPatchDenoiser(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        global_gate_strength=0.25,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(19),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 30)
    assert trace["omega"].shape == (2, 4, 6)
    assert trace["global_omega"].shape == (2, 1, 6)
    assert trace["thetas"].shape == (2, 2, 4, 6)
    assert trace["global_thetas"].shape == (2, 2, 1, 6)
    assert trace["rate_states"].shape == (2, 2, 4, 6)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_coarse_global_rate_phase_denoiser_traces_coarse_phase_mesh():
    model = WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(
        hidden_dim=4,
        image_shape=(12, 12),
        patch_shape=(4, 4),
        coarse_grid_shape=(2, 2),
        steps=2,
        global_gate_strength=0.25,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(20),
    )
    images = jnp.ones((2, 144))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 44)
    assert trace["omega"].shape == (2, 9, 4)
    assert trace["coarse_omega"].shape == (2, 4, 4)
    assert trace["thetas"].shape == (2, 2, 9, 4)
    assert trace["coarse_thetas"].shape == (2, 2, 4, 4)
    assert trace["rate_states"].shape == (2, 2, 9, 4)
    assert trace["fine_to_coarse_weights"].shape == (4, 9)
    assert trace["coarse_to_fine_weights"].shape == (9, 4)
    assert jnp.allclose(trace["fine_to_coarse_weights"].sum(axis=1), 1.0)
    assert jnp.allclose(trace["coarse_to_fine_weights"].sum(axis=1), 1.0)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_coarse_global_rate_phase_supports_phase_shuffle_control():
    model = WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser(
        hidden_dim=4,
        image_shape=(12, 12),
        patch_shape=(4, 4),
        coarse_grid_shape=(2, 2),
        steps=1,
        global_phase_control="shuffle",
        key=jax.random.PRNGKey(21),
    )

    assert model.global_phase_control == "shuffle"
    assert model.coarse_phase_permutation != (0, 1, 2, 3)
    assert model(jnp.ones((2, 144))).shape == (2, 144)


def test_winfree_coarse_rate_phase_denoiser_traces_content_transport():
    model = WinfreeCoarseRatePhaseConditionalPatchDenoiser(
        hidden_dim=4,
        image_shape=(12, 12),
        patch_shape=(4, 4),
        coarse_grid_shape=(2, 2),
        steps=2,
        global_gate_strength=0.25,
        global_content_strength=0.5,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(22),
    )
    images = jnp.ones((2, 144))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert trace["latent"].shape == (2, 60)
    assert trace["omega"].shape == (2, 9, 4)
    assert trace["coarse_omega"].shape == (2, 4, 4)
    assert trace["thetas"].shape == (2, 2, 9, 4)
    assert trace["coarse_thetas"].shape == (2, 2, 4, 4)
    assert trace["rate_states"].shape == (2, 2, 9, 4)
    assert trace["coarse_rate_states"].shape == (2, 2, 4, 4)
    assert trace["coarse_rate_drive"].shape == (2, 4, 4)
    assert trace["coarse_rate_to_fine"].shape == (2, 9, 4)
    assert trace["fine_to_coarse_weights"].shape == (4, 9)
    assert trace["coarse_to_fine_weights"].shape == (9, 4)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_coarse_rate_phase_supports_content_shuffle_control():
    model = WinfreeCoarseRatePhaseConditionalPatchDenoiser(
        hidden_dim=4,
        image_shape=(12, 12),
        patch_shape=(4, 4),
        coarse_grid_shape=(2, 2),
        steps=1,
        global_content_control="shuffle",
        key=jax.random.PRNGKey(23),
    )

    assert model.global_content_control == "shuffle"
    assert model.coarse_phase_permutation != (0, 1, 2, 3)
    assert model(jnp.ones((2, 144))).shape == (2, 144)


def test_winfree_coarse_predictive_rate_phase_uses_readout_branch():
    model = WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser(
        hidden_dim=4,
        image_shape=(12, 12),
        patch_shape=(4, 4),
        coarse_grid_shape=(2, 2),
        steps=2,
        global_gate_strength=0.25,
        global_content_control="shuffle",
        coarse_readout_strength=0.35,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(24),
    )
    images = jnp.ones((2, 144))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert model.global_content_strength == 0.0
    assert model.global_content_control == "shuffle"
    assert model.coarse_readout_strength == 0.35
    assert trace["latent"].shape == (2, 60)
    assert trace["local_readout_sequence"].shape == (9, 2, 16)
    assert trace["coarse_readout_sequence"].shape == (9, 2, 16)
    assert trace["coarse_readout_to_fine"].shape == (2, 9, 16)
    assert trace["coarse_rate_states"].shape == (2, 2, 4, 4)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_prior_refinement_denoiser_combines_prior_and_residual():
    model = WinfreePriorRefinementPatchDenoiser(
        input_dim=32,
        hidden_dim=4,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=1,
        refinement_strength=0.25,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(25),
    )
    images = jnp.ones((2, 128))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == (2, 64)
    assert model.refinement_strength == 0.25
    assert trace["prior_reconstruction"].shape == (2, 64)
    assert trace["residual_reconstruction"].shape == (2, 64)
    assert trace["combined_reconstruction"].shape == (2, 64)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert trace["refiner_reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_recurrent_conv_prior_refinement_denoiser_combines_prior_and_residual():
    model = RecurrentConvPriorRefinementPatchDenoiser(
        input_dim=32,
        hidden_dim=4,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        refinement_strength=0.5,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(26),
    )
    images = jnp.ones((2, 128))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == (2, 64)
    assert model.refinement_strength == 0.5
    assert trace["prior_reconstruction"].shape == (2, 64)
    assert trace["residual_reconstruction"].shape == (2, 64)
    assert trace["combined_reconstruction"].shape == (2, 64)
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert trace["refiner_hidden_states"].shape == (2, 2, 4, 4)
    assert trace["refiner_reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_patch_autoencoder_supports_latent_phase_readout():
    model = WinfreePatchAutoencoder(
        hidden_dim=6,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        latent_readout="phase_bias",
        latent_readout_strength=1.5,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(10),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)
    trace = model.collect_trace(images)

    assert reconstruction.shape == images.shape
    assert model.decoder.latent_readout == "phase_bias"
    assert model.decoder.latent_to_readout is not None
    assert trace["reconstruction_sequence"].shape == (4, 2, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_patch_autoencoder_supports_latent_concat_readout():
    model = WinfreePatchAutoencoder(
        hidden_dim=6,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        latent_readout="concat",
        latent_readout_strength=1.5,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(11),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)

    assert reconstruction.shape == images.shape
    assert model.decoder.latent_readout == "concat"
    assert model.decoder.theta_to_output.weight.shape == (16, 16)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_patch_autoencoder_supports_latent_sequence_output_skip():
    model = WinfreePatchAutoencoder(
        hidden_dim=6,
        latent_dim=4,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        latent_output_skip="sequence",
        latent_output_skip_strength=1.5,
        output_activation="sigmoid",
        key=jax.random.PRNGKey(12),
    )
    images = jnp.ones((2, 64))

    reconstruction = model(images)

    assert reconstruction.shape == images.shape
    assert model.decoder.latent_output_skip == "sequence"
    assert model.decoder.latent_to_output_skip is not None
    assert model.decoder.latent_to_output_skip.weight.shape == (64, 4)
    assert jnp.all(reconstruction >= 0.0)
    assert jnp.all(reconstruction <= 1.0)


def test_winfree_field_layer_supports_learned_grouped_interactions():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=5,
        grid_shape=(4, 4),
        group_size=2,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(6),
    )
    theta = jnp.zeros((3, 16, 5))
    omega = jnp.ones((3, 16, 5)) * 0.05

    trace = layer(theta, omega, return_trajectory=True)

    assert layer.coupling.shape == (4, 4)
    assert trace["final_theta"].shape == theta.shape
    assert trace["thetas"].shape == (2, 3, 16, 5)
    assert trace["energies"].shape == (2, 3)
    assert jnp.all(trace["thetas"] <= jnp.pi + 1e-6)
    assert jnp.all(trace["thetas"] >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(trace["energies"]))


def test_winfree_field_layer_supports_spatial_coupling_decay():
    layer = WinfreeFieldLayer(
        num_positions=49,
        channels=3,
        grid_shape=(7, 7),
        coupling_decay_length=2.0,
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(8),
    )
    no_decay = WinfreeFieldLayer(
        num_positions=49,
        channels=3,
        grid_shape=(7, 7),
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(9),
    )

    mask = jnp.asarray(layer.coupling_decay_mask)
    no_decay_mask = jnp.asarray(no_decay.coupling_decay_mask)
    assert mask.shape == (49, 49)
    assert mask[0, 0] > mask[0, 1]
    assert mask[0, 1] > mask[0, -1]
    assert jnp.allclose(no_decay_mask, 1.0)


def test_winfree_field_layer_supports_local_conv_coupling():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=4,
        grid_shape=(4, 4),
        coupling_mode="conv",
        coupling_kernel_size=3,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(13),
    )
    theta = jnp.ones((2, 16, 4)) * 0.2
    omega = jnp.ones((2, 16, 4)) * 0.1

    output = layer(theta, omega)
    trace = layer(theta, omega, return_trajectory=True)

    assert layer.coupling_mode == "conv"
    assert layer.conv_kernel is not None
    assert layer.conv_bias is not None
    assert layer.conv_kernel.shape == (3, 3, 4, 4)
    assert layer.conv_bias.shape == (4,)
    assert output.shape == theta.shape
    assert trace["final_theta"].shape == theta.shape
    assert jnp.all(jnp.isfinite(output))


def test_winfree_field_layer_supports_adaptive_local_coupling():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=4,
        grid_shape=(4, 4),
        coupling_mode="adaptive",
        coupling_kernel_size=3,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(14),
    )
    theta = jnp.ones((2, 16, 4)) * 0.2
    omega = jnp.ones((2, 16, 4)) * 0.1

    output = layer(theta, omega)
    trace = layer(theta, omega, return_trajectory=True)
    local_mask = jnp.asarray(layer.local_coupling_mask)

    assert layer.coupling_mode == "adaptive"
    assert layer.adaptive_query is not None
    assert layer.adaptive_key is not None
    assert layer.adaptive_value is not None
    assert local_mask.shape == (16, 16)
    assert bool(local_mask[0, 0])
    assert bool(local_mask[0, 1])
    assert not bool(local_mask[0, -1])
    assert output.shape == theta.shape
    assert trace["final_theta"].shape == theta.shape
    assert jnp.all(jnp.isfinite(output))


def test_winfree_field_layer_supports_residual_conv_adaptive_coupling():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=4,
        grid_shape=(4, 4),
        coupling_mode="conv_adaptive",
        coupling_kernel_size=3,
        adaptive_coupling_strength=0.05,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(15),
    )
    theta = jnp.ones((2, 16, 4)) * 0.2
    omega = jnp.ones((2, 16, 4)) * 0.1

    output = layer(theta, omega)

    assert layer.coupling_mode == "conv_adaptive"
    assert layer.adaptive_coupling_strength == 0.05
    assert layer.conv_kernel is not None
    assert layer.conv_bias is not None
    assert layer.adaptive_query is not None
    assert layer.adaptive_key is not None
    assert layer.adaptive_value is not None
    assert output.shape == theta.shape
    assert jnp.all(jnp.isfinite(output))


def test_winfree_field_layer_supports_residual_conv_matrix_coupling():
    layer = WinfreeFieldLayer(
        num_positions=16,
        channels=4,
        grid_shape=(4, 4),
        coupling_mode="conv_matrix",
        coupling_decay_length=2.0,
        coupling_kernel_size=3,
        adaptive_coupling_strength=0.05,
        si_func="mlp",
        steps=2,
        gamma=0.1,
        key=jax.random.PRNGKey(16),
    )
    theta = jnp.ones((2, 16, 4)) * 0.2
    omega = jnp.ones((2, 16, 4)) * 0.1

    output = layer(theta, omega)

    assert layer.coupling_mode == "conv_matrix"
    assert layer.adaptive_coupling_strength == 0.05
    assert layer.conv_kernel is not None
    assert layer.conv_bias is not None
    assert jnp.asarray(layer.coupling_decay_mask).shape == (16, 16)
    assert output.shape == theta.shape
    assert jnp.all(jnp.isfinite(output))
