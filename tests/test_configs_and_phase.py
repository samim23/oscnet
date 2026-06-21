import jax
import jax.numpy as jnp

from oscnet.models import (
    ConvLSTMPatchDenoiserConfig,
    FeedForwardPatchAutoencoderConfig,
    OscillatoryAutoencoderConfig,
    PatchOscillatoryAutoencoderConfig,
    RecurrentConvPatchDenoiserConfig,
    RecurrentConvPriorRefinementPatchDenoiserConfig,
    WaveletAutoencoderConfig,
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig,
    WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiserConfig,
    WinfreeCoarseRatePhaseConditionalPatchDenoiserConfig,
    WinfreeFieldAutoencoderConfig,
    WinfreeConditionalPatchDenoiserConfig,
    WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig,
    WinfreePatchAutoencoderConfig,
    WinfreePhaseAutoencoder,
    WinfreePhaseAutoencoderConfig,
    WinfreePhaseOscillatorCell,
    WinfreePriorRefinementPatchDenoiserConfig,
    WinfreeRatePhaseConditionalPatchDenoiserConfig,
)


def test_model_configs_build_expected_shapes():
    sequence = jnp.ones((3, 2, 4))

    sequence_model = OscillatoryAutoencoderConfig(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
        sequence_length=3,
    ).build(jax.random.PRNGKey(0))
    assert sequence_model(sequence).shape == sequence.shape

    patch_model = PatchOscillatoryAutoencoderConfig(
        hidden_dim=8,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
    ).build(jax.random.PRNGKey(1))
    assert patch_model(jnp.ones((2, 64))).shape == (2, 64)

    feedforward_patch_model = FeedForwardPatchAutoencoderConfig(
        hidden_dim=8,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(8))
    assert feedforward_patch_model(jnp.ones((2, 64))).shape == (2, 64)
    assert feedforward_patch_model.latent_output_skip == "sequence"

    recurrent_conv_model = RecurrentConvPatchDenoiserConfig(
        hidden_dim=8,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(10))
    assert recurrent_conv_model(jnp.ones((2, 64))).shape == (2, 64)
    assert recurrent_conv_model.steps == 2

    recurrent_prior_refinement_model = (
        RecurrentConvPriorRefinementPatchDenoiserConfig(
            input_dim=32,
            hidden_dim=8,
            latent_dim=5,
            image_shape=(8, 8),
            patch_shape=(4, 4),
            steps=2,
            recurrent_residual_strength=0.4,
            refinement_strength=0.5,
            output_activation="sigmoid",
        ).build(jax.random.PRNGKey(16))
    )
    assert recurrent_prior_refinement_model(jnp.ones((2, 128))).shape == (2, 64)
    assert recurrent_prior_refinement_model.refiner.steps == 2
    assert recurrent_prior_refinement_model.refiner.residual_strength == 0.4
    assert recurrent_prior_refinement_model.refinement_strength == 0.5

    conv_lstm_model = ConvLSTMPatchDenoiserConfig(
        input_dim=32,
        hidden_dim=8,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(11))
    assert conv_lstm_model(jnp.ones((2, 128))).shape == (2, 64)
    assert conv_lstm_model.steps == 2

    wavelet_model = WaveletAutoencoderConfig(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
    ).build(jax.random.PRNGKey(2))
    assert wavelet_model(sequence).shape == sequence.shape

    winfree_sequence_model = WinfreeFieldAutoencoderConfig(
        input_dim=4,
        hidden_dim=6,
        latent_dim=3,
        sequence_length=3,
        steps=2,
    ).build(jax.random.PRNGKey(6))
    assert winfree_sequence_model(sequence).shape == sequence.shape

    winfree_patch_model = WinfreePatchAutoencoderConfig(
        hidden_dim=6,
        latent_dim=3,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        group_size=2,
        coupling_mode="conv",
        coupling_kernel_size=3,
        si_func="mlp",
        latent_readout="phase_bias",
        latent_readout_strength=1.5,
        latent_output_skip="sequence",
        latent_output_skip_strength=0.75,
        steps=2,
    ).build(jax.random.PRNGKey(7))
    assert winfree_patch_model(jnp.ones((2, 64))).shape == (2, 64)
    assert winfree_patch_model.decoder.latent_readout == "phase_bias"
    assert winfree_patch_model.decoder.latent_output_skip == "sequence"
    assert winfree_patch_model.decoder.layer.coupling_mode == "conv"

    winfree_conditional_model = WinfreeConditionalPatchDenoiserConfig(
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        coupling_mode="conv_adaptive",
        adaptive_coupling_strength=0.05,
        phase_init="rotary_2d",
        phase_init_scale=0.75,
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(9))
    assert winfree_conditional_model(jnp.ones((2, 64))).shape == (2, 64)
    assert winfree_conditional_model.layer.coupling_mode == "conv_adaptive"
    assert winfree_conditional_model.layer.adaptive_coupling_strength == 0.05
    assert winfree_conditional_model.phase_init == "rotary_2d"

    winfree_rate_phase_model = WinfreeRatePhaseConditionalPatchDenoiserConfig(
        input_dim=32,
        hidden_dim=6,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        coupling_mode="conv_matrix",
        coupling_decay_length=2.0,
        adaptive_coupling_strength=0.05,
        rate_kernel_size=3,
        rate_update_rate=0.25,
        rate_gate_strength=0.75,
        visibility_gate="visibility",
        visibility_drive_floor=0.1,
        missing_transport_strength=1.5,
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(10))
    assert winfree_rate_phase_model(jnp.ones((2, 128))).shape == (2, 64)
    assert winfree_rate_phase_model.layer.coupling_mode == "conv_matrix"
    assert winfree_rate_phase_model.layer.adaptive_coupling_strength == 0.05
    assert winfree_rate_phase_model.rate_update_rate == 0.25
    assert winfree_rate_phase_model.rate_gate_strength == 0.75
    assert winfree_rate_phase_model.visibility_gate == "visibility"
    assert winfree_rate_phase_model.visibility_drive_floor == 0.1
    assert winfree_rate_phase_model.missing_transport_strength == 1.5

    winfree_global_rate_phase_model = (
        WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig(
            input_dim=32,
            hidden_dim=6,
            image_shape=(8, 8),
            patch_shape=(4, 4),
            steps=2,
            global_gamma=0.03,
            global_gate_strength=0.25,
            visibility_gate="shuffle",
            visibility_drive_floor=0.2,
            missing_transport_strength=2.0,
            output_activation="sigmoid",
        ).build(jax.random.PRNGKey(11))
    )
    assert winfree_global_rate_phase_model(jnp.ones((2, 128))).shape == (2, 64)
    assert winfree_global_rate_phase_model.global_layer.gamma == 0.03
    assert winfree_global_rate_phase_model.global_gate_strength == 0.25
    assert winfree_global_rate_phase_model.local.visibility_gate == "shuffle"
    assert winfree_global_rate_phase_model.local.visibility_drive_floor == 0.2
    assert winfree_global_rate_phase_model.local.missing_transport_strength == 2.0

    winfree_coarse_global_rate_phase_model = (
        WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig(
            hidden_dim=6,
            image_shape=(12, 12),
            patch_shape=(4, 4),
            coarse_grid_shape=(2, 2),
            steps=2,
            global_gamma=0.03,
            global_gate_strength=0.25,
            global_phase_control="shuffle",
            output_activation="sigmoid",
        ).build(jax.random.PRNGKey(12))
    )
    assert winfree_coarse_global_rate_phase_model(jnp.ones((2, 144))).shape == (
        2,
        144,
    )
    assert winfree_coarse_global_rate_phase_model.coarse_layer.gamma == 0.03
    assert winfree_coarse_global_rate_phase_model.global_gate_strength == 0.25
    assert (
        winfree_coarse_global_rate_phase_model.global_phase_control
        == "shuffle"
    )

    winfree_coarse_rate_phase_model = (
        WinfreeCoarseRatePhaseConditionalPatchDenoiserConfig(
            hidden_dim=6,
            image_shape=(12, 12),
            patch_shape=(4, 4),
            coarse_grid_shape=(2, 2),
            steps=2,
            global_gamma=0.03,
            global_gate_strength=0.25,
            global_phase_control="shuffle",
            global_content_strength=0.3,
            global_content_control="shuffle",
            output_activation="sigmoid",
        ).build(jax.random.PRNGKey(13))
    )
    assert winfree_coarse_rate_phase_model(jnp.ones((2, 144))).shape == (
        2,
        144,
    )
    assert winfree_coarse_rate_phase_model.coarse_layer.gamma == 0.03
    assert winfree_coarse_rate_phase_model.global_gate_strength == 0.25
    assert winfree_coarse_rate_phase_model.global_content_strength == 0.3
    assert winfree_coarse_rate_phase_model.global_content_control == "shuffle"

    winfree_coarse_predictive_rate_phase_model = (
        WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiserConfig(
            hidden_dim=6,
            image_shape=(12, 12),
            patch_shape=(4, 4),
            coarse_grid_shape=(2, 2),
            steps=2,
            global_gamma=0.03,
            global_gate_strength=0.25,
            global_phase_control="shuffle",
            global_content_control="shuffle",
            coarse_readout_strength=0.4,
            output_activation="sigmoid",
        ).build(jax.random.PRNGKey(14))
    )
    assert winfree_coarse_predictive_rate_phase_model(
        jnp.ones((2, 144))
    ).shape == (
        2,
        144,
    )
    assert winfree_coarse_predictive_rate_phase_model.coarse_layer.gamma == 0.03
    assert (
        winfree_coarse_predictive_rate_phase_model.global_gate_strength
        == 0.25
    )
    assert (
        winfree_coarse_predictive_rate_phase_model.global_content_control
        == "shuffle"
    )
    assert (
        winfree_coarse_predictive_rate_phase_model.coarse_readout_strength
        == 0.4
    )

    winfree_prior_refinement_model = WinfreePriorRefinementPatchDenoiserConfig(
        input_dim=32,
        hidden_dim=6,
        latent_dim=5,
        image_shape=(8, 8),
        patch_shape=(4, 4),
        steps=2,
        global_gamma=0.03,
        global_gate_strength=0.25,
        refinement_strength=0.2,
        output_activation="sigmoid",
    ).build(jax.random.PRNGKey(15))
    assert winfree_prior_refinement_model(jnp.ones((2, 128))).shape == (2, 64)
    assert winfree_prior_refinement_model.refiner.global_layer.gamma == 0.03
    assert winfree_prior_refinement_model.refiner.global_gate_strength == 0.25
    assert winfree_prior_refinement_model.refinement_strength == 0.2


def test_winfree_phase_cell_wraps_phase_and_outputs_finite_values():
    cell = WinfreePhaseOscillatorCell(
        input_dim=4,
        hidden_dim=6,
        output_dim=5,
        key=jax.random.PRNGKey(3),
    )
    output, phases = cell(jnp.ones((2, 4)))

    assert output.shape == (2, 5)
    assert phases.shape == (2, 6)
    assert jnp.all(phases <= jnp.pi + 1e-6)
    assert jnp.all(phases >= -jnp.pi - 1e-6)
    assert jnp.all(jnp.isfinite(output))


def test_winfree_phase_autoencoder_and_config_shapes():
    inputs = jnp.ones((4, 2, 5))
    model = WinfreePhaseAutoencoder(
        input_dim=5,
        hidden_dim=7,
        latent_dim=3,
        sequence_length=4,
        key=jax.random.PRNGKey(4),
    )
    config_model = WinfreePhaseAutoencoderConfig(
        input_dim=5,
        hidden_dim=7,
        latent_dim=3,
        sequence_length=4,
    ).build(jax.random.PRNGKey(5))

    assert model(inputs).shape == inputs.shape
    assert config_model(inputs).shape == inputs.shape
    assert jnp.all(jnp.isfinite(model(inputs)))
