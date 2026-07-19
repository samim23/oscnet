"""Smoke tests for Stage 0a audio-digit frontend probe."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from oscnet.experiments.audio_digit.config import AudioDigitConfig
from oscnet.experiments.audio_digit.data import make_synthetic_split
from oscnet.experiments.audio_digit.frontends import build_frontend
from oscnet.experiments.audio_digit.runner import (
    build_model,
    run_audio_digit_experiment,
)


def test_synthetic_digits_are_balanced_and_finite():
    cfg = AudioDigitConfig(train_samples=50, num_classes=10, sample_rate=4000.0)
    waves, labels = make_synthetic_split(
        jax.random.PRNGKey(0),
        num_samples_total=50,
        config=cfg,
    )
    assert waves.ndim == 2
    assert waves.shape[0] == 50
    assert jnp.all(jnp.isfinite(waves))
    # Roughly balanced
    for d in range(10):
        assert int(jnp.sum(labels == d)) >= 5


def test_frontends_emit_matching_feature_dims():
    cfg = AudioDigitConfig(num_bands=8, sample_rate=4000.0, duration_sec=0.25)
    key = jax.random.PRNGKey(1)
    waves = jax.random.normal(key, (3, cfg.num_samples))
    for name in ("resonator", "resonator_equal", "mel", "stft", "raw_wide"):
        front = build_frontend(replace(cfg, frontend=name), key=key)
        feats = front(waves)
        assert feats.shape[0] == 3
        assert feats.shape[1] == front.feature_dim
        assert jnp.all(jnp.isfinite(feats))


def test_tiny_audio_digit_train_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        num_bands=6,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_hidden_dim=16,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        seed=0,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["final_eval_accuracy"] is not None
    assert 0.0 <= result["final_eval_accuracy"] <= 1.0
    assert result["trainable_params"] > 0
    assert Path(result["output_path"]).is_file()


def test_model_freeze_frontend_for_resonator():
    cfg = AudioDigitConfig(frontend="resonator", num_bands=4)
    model = build_model(cfg, key=jax.random.PRNGKey(2))
    assert model.freeze_frontend is True
    logits = model(jax.random.normal(jax.random.PRNGKey(3), (2, cfg.num_samples)))
    assert logits.shape == (2, 10)


def test_horn_head_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        head_kind="horn",
        num_bands=6,
        quality_factor=8.0,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_hidden_dim=16,
        horn_steps=4,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        seed=0,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["head_kind"] == "horn"
    assert result["final_eval_accuracy"] is not None
    assert Path(result["output_path"]).is_file()


def test_framed_rfb_horn_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        head_kind="horn",
        feature_mode="frames",
        num_frames=8,
        num_bands=6,
        sample_rate=4000.0,
        duration_sec=0.25,
        head_hidden_dim=16,
        horn_steps=8,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        seed=1,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["feature_mode"] == "frames"
    assert result["final_eval_accuracy"] is not None


def test_learnable_rfb_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator_learn",
        learnable_frontend=True,
        num_bands=6,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_hidden_dim=16,
        epochs=3,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        learning_rate=3e-3,
        seed=2,
        output_dir=tmp_path,
    )
    model = build_model(cfg, key=jax.random.PRNGKey(0))
    assert model.freeze_frontend is False
    result = run_audio_digit_experiment(cfg)
    assert result["learnable_frontend"] is True
    assert result["band_collapse"]
    assert "min_freq_ratio" in result["band_collapse"]


def test_collapse_reg_and_noise_eval_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        learnable_frontend=True,
        collapse_reg_weight=0.1,
        feature_mode="frames",
        num_frames=4,
        num_bands=6,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_kind="horn",
        head_hidden_dim=16,
        horn_steps=4,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        learning_rate=3e-3,
        eval_noise_snrs_db=(10.0, 0.0),
        seed=3,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["collapse_reg_weight"] == 0.1
    assert "eval_accuracy_snr10db" in result
    assert "eval_accuracy_snr0db" in result
    assert 0.0 <= result["eval_accuracy_snr10db"] <= 1.0


def test_phase_and_both_readout_smoke(tmp_path: Path):
    for readout, expected_mult in (("phase", 2), ("both", 3)):
        cfg = AudioDigitConfig(
            frontend="resonator",
            learnable_frontend=True,
            readout=readout,
            feature_mode="frames",
            num_frames=4,
            num_bands=6,
            sample_rate=4000.0,
            duration_sec=0.2,
            head_kind="horn",
            head_hidden_dim=16,
            horn_steps=4,
            epochs=2,
            batch_size=16,
            train_samples=64,
            eval_samples=32,
            learning_rate=3e-3,
            seed=4,
            output_dir=tmp_path / readout,
        )
        assert cfg.feature_dim == expected_mult * 6
        result = run_audio_digit_experiment(cfg)
        assert result["readout"] == readout
        assert result["final_eval_accuracy"] is not None


def test_nonlinearity_smoke(tmp_path: Path):
    for nonlin in ("drive_tanh", "envelope_soft", "agc"):
        cfg = AudioDigitConfig(
            frontend="resonator",
            learnable_frontend=nonlin != "agc",
            nonlinearity=nonlin,
            feature_mode="frames",
            num_frames=4,
            num_bands=6,
            sample_rate=4000.0,
            duration_sec=0.2,
            head_kind="horn",
            head_hidden_dim=16,
            horn_steps=4,
            epochs=2,
            batch_size=16,
            train_samples=64,
            eval_samples=32,
            learning_rate=3e-3,
            seed=5,
            output_dir=tmp_path / nonlin,
        )
        result = run_audio_digit_experiment(cfg)
        assert result["nonlinearity"] == nonlin
        assert result["final_eval_accuracy"] is not None


def test_robustness_eval_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        learnable_frontend=True,
        feature_mode="frames",
        num_frames=4,
        num_bands=6,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_kind="horn",
        head_hidden_dim=16,
        horn_steps=4,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        learning_rate=3e-3,
        train_level_aug_db=6.0,
        eval_noise_snrs_db=(10.0,),
        eval_pink_snrs_db=(10.0,),
        eval_band_snrs_db=(10.0,),
        eval_level_gains_db=(-12.0, 6.0),
        seed=6,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["train_level_aug_db"] == 6.0
    for key in (
        "eval_accuracy_snr10db",
        "eval_accuracy_pink10db",
        "eval_accuracy_band10db",
        "eval_accuracy_levelm12db",
        "eval_accuracy_levelp6db",
    ):
        assert key in result
        assert 0.0 <= result[key] <= 1.0


def test_stage5_diagnostics_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        learnable_frontend=True,
        feature_mode="frames",
        num_frames=4,
        num_bands=8,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_kind="horn",
        head_hidden_dim=16,
        horn_steps=4,
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        learning_rate=3e-3,
        run_stage5_diagnostics=True,
        seed=7,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert "stage5_diagnostics" in result
    assert "ablate_low_drop" in result
    assert "ablate_high_drop" in result
    assert "coupling_high_to_low_ratio" in result


def test_fractal_coupling_structure_is_frozen():
    import equinox as eqx

    from oscnet.core.fractal_coupling import HierarchicalCouplingLayer

    layer = HierarchicalCouplingLayer(hidden_dim=8, depth=1, key=jax.random.PRNGKey(0))
    trainable, _ = eqx.partition(layer, eqx.is_inexact_array)
    leaves = jax.tree_util.tree_leaves(trainable)
    # Only scalar strength should be trainable.
    assert len(leaves) == 1
    assert leaves[0].shape == ()


def test_dense_tonotopic_init_prefers_high_to_low():
    from oscnet.core.fractal_coupling import create_tonotopic_coupling_init

    W = np.asarray(
        create_tonotopic_coupling_init(
            16, bias_strength=0.8, noise_scale=0.0, key=jax.random.PRNGKey(0)
        )
    )
    # Lower triangle (i > j): low←high? Wait: W[i,j] influence of j on i.
    # High→low: j high, i low ⇒ upper triangle i < j.
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)  # i < j → high source j → low target i
    il = np.tril_indices(n, k=-1)
    assert float(np.mean(np.abs(W[iu]))) > float(np.mean(np.abs(W[il])))


def test_gru_and_mel_frames_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        learnable_frontend=True,
        feature_mode="frames",
        num_frames=4,
        num_bands=8,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_kind="gru",
        head_hidden_dim=16,
        epochs=1,
        batch_size=16,
        train_samples=32,
        eval_samples=16,
        learning_rate=3e-3,
        seed=1,
        output_dir=tmp_path / "gru",
    )
    result = run_audio_digit_experiment(cfg)
    assert result["head_kind"] == "gru"
    assert 0.0 <= result["final_eval_accuracy"] <= 1.0

    mel_cfg = replace(
        cfg,
        frontend="mel",
        learnable_frontend=False,
        head_kind="horn",
        horn_coupling_kind="dense",
        horn_steps=4,
        collapse_reg_weight=0.0,
        learning_rate=1e-2,
        output_dir=tmp_path / "mel_horn",
    )
    mel_result = run_audio_digit_experiment(mel_cfg)
    assert mel_result["frontend"] == "mel"
    assert mel_result["head_kind"] == "horn"


def test_dense_tonotopic_horn_smoke(tmp_path: Path):
    cfg = AudioDigitConfig(
        frontend="resonator",
        learnable_frontend=True,
        feature_mode="frames",
        num_frames=4,
        num_bands=8,
        sample_rate=4000.0,
        duration_sec=0.2,
        head_kind="horn",
        head_hidden_dim=16,
        horn_steps=4,
        horn_coupling_kind="dense_tonotopic",
        epochs=2,
        batch_size=16,
        train_samples=64,
        eval_samples=32,
        learning_rate=3e-3,
        run_stage5_diagnostics=True,
        seed=3,
        output_dir=tmp_path,
    )
    result = run_audio_digit_experiment(cfg)
    assert result["horn_coupling_kind"] == "dense_tonotopic"
    assert result.get("horn_coupling_fixed_structure") is False
    assert "coupling_high_to_low_ratio" in result


def test_band_spacing_regularizer_penalizes_duplicates():
    from oscnet.core.resonators import band_spacing_regularizer

    spread = jnp.asarray([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
    collapsed = jnp.asarray([1.0, 1.01, 1.02, 8.0], dtype=jnp.float32)
    assert float(band_spacing_regularizer(collapsed)) > float(
        band_spacing_regularizer(spread)
    )
