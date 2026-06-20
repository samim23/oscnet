"""Reference audio wavelet benchmark for the oscillatory autoencoder."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from oscnet.experiments.harness import (
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
    ExperimentPaths,
    collect_sequence_state_trace,
    run_eval_only,
    train_autoencoder,
)
from oscnet.models import ProductionWaveletAutoencoder

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

try:
    import jaxwt as jwt

    HAS_JAXWT = True
except ImportError:
    jwt = None
    HAS_JAXWT = False

Array = jnp.ndarray


@dataclass(frozen=True)
class AudioWaveletExperimentConfig:
    """Task-specific controls for the audio wavelet reference experiment."""

    run: AutoencoderExperimentConfig
    input_dim: int = 8192
    hidden_dim: int = 128
    latent_dim: int = 128
    sequence_length: int = 8
    wavelet: str = "db4"
    levels: int = 3
    sample_rate: int = 22_050
    duration: float = 1.0
    audio_dir: Path = Path("my_audio_samples")
    data_source: str = "auto"
    feature_source: str = "auto"
    train_limit: int = 8
    eval_limit: int = 2
    omega_bounds: Tuple[float, float] = (0.2, 6.0)
    gamma_bounds: Tuple[float, float] = (0.01, 0.15)
    checkpoint: Optional[Path] = None


@dataclass
class AudioExperimentData:
    train_sequences: Array
    eval_sequences: Array
    train_audio: Optional[Array]
    eval_audio: Optional[Array]
    train_coeffs: List[List[Array]]
    eval_coeffs: List[List[Array]]
    file_names: List[str]
    metadata: Dict[str, object]


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.audio_wavelet")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    duration = len(audio) / float(source_rate)
    source_t = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_n = max(1, int(round(duration * target_rate)))
    target_t = np.linspace(0.0, duration, num=target_n, endpoint=False)
    return np.interp(target_t, source_t, audio).astype(np.float32)


def _fix_audio_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if len(audio) >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio)))


def _synthetic_audio(
    n_samples: int,
    sample_rate: int,
    duration: float,
    seed: int,
) -> Tuple[Array, List[str]]:
    rng = np.random.default_rng(seed)
    n_audio_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n_audio_samples, endpoint=False, dtype=np.float32)
    waves = []
    names = []
    base_frequencies = [110, 165, 220, 330, 440, 660, 880, 1320]
    for i in range(n_samples):
        f0 = base_frequencies[i % len(base_frequencies)]
        phase = rng.uniform(0.0, 2.0 * np.pi)
        envelope = np.linspace(1.0, 0.4, n_audio_samples, dtype=np.float32)
        wave = (
            0.45 * np.sin(2.0 * np.pi * f0 * t + phase)
            + 0.18 * np.sin(2.0 * np.pi * (f0 * 2.0) * t)
            + 0.06 * rng.normal(size=n_audio_samples)
        )
        waves.append(np.clip(wave * envelope, -1.0, 1.0).astype(np.float32))
        names.append(f"synthetic_{i:02d}_{f0}hz")
    return jnp.asarray(np.stack(waves)), names


def load_audio_files(
    audio_dir: Path,
    *,
    sample_rate: int,
    duration: float,
    n_samples: int,
    seed: int,
    data_source: str = "auto",
) -> Tuple[Array, List[str], str]:
    """Load real audio files or deterministic synthetic waveforms."""

    if data_source not in {"auto", "audio-dir", "synthetic"}:
        raise ValueError("data_source must be 'auto', 'audio-dir', or 'synthetic'")

    n_audio_samples = int(sample_rate * duration)
    audio_dir = Path(audio_dir)
    suffixes = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
    files = sorted(
        path for path in audio_dir.glob("*") if path.suffix.lower() in suffixes
    )

    should_try_files = data_source in {"auto", "audio-dir"} and files
    if should_try_files:
        if not HAS_SOUNDFILE:
            if data_source == "audio-dir":
                raise ImportError("soundfile is required for data_source='audio-dir'")
        else:
            loaded = []
            names = []
            for path in files[:n_samples]:
                audio, source_rate = sf.read(path)
                audio = _resample_audio(audio, source_rate, sample_rate)
                loaded.append(_fix_audio_length(audio, n_audio_samples))
                names.append(path.name)
            if loaded:
                return jnp.asarray(np.stack(loaded)), names, "audio-dir"

    if data_source == "audio-dir":
        raise FileNotFoundError(f"No readable audio files found in {audio_dir}")

    audio, names = _synthetic_audio(n_samples, sample_rate, duration, seed)
    return audio, names, "synthetic"


def audio_to_wavelets(audio: Array, levels: int = 3, wavelet: str = "db4") -> List[Array]:
    """Convert one waveform into wavelet coefficients."""

    if not HAS_JAXWT:
        raise ImportError("jaxwt is required for wavelet feature extraction")
    return list(jwt.wavedec(audio, wavelet, mode="symmetric", level=levels))


def wavelets_to_audio(coeffs: List[Array], wavelet: str = "db4") -> Array:
    """Convert wavelet coefficients back to a waveform."""

    if not HAS_JAXWT:
        raise ImportError("jaxwt is required for inverse wavelet reconstruction")
    audio = jwt.waverec(coeffs, wavelet)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    max_val = jnp.max(jnp.abs(audio))
    return jnp.where(max_val > 1.0, audio / (max_val + 1e-8), audio)


def smart_wavelet_compression(coeffs: List[Array], target_features: int = 8192) -> Array:
    """Flatten wavelet coefficients while preserving coefficient order."""

    features = []
    for coeff in coeffs:
        flat = jnp.ravel(coeff)
        if jnp.iscomplexobj(flat):
            flat = jnp.concatenate([jnp.real(flat), jnp.imag(flat)])
        features.append(flat)
    if not features:
        return jnp.zeros((target_features,), dtype=jnp.float32)
    flat_features = jnp.concatenate(features).astype(jnp.float32)
    if flat_features.shape[0] >= target_features:
        return flat_features[:target_features]
    return jnp.pad(flat_features, (0, target_features - flat_features.shape[0]))


def smart_wavelet_decompression(
    features: Array,
    reference_coeffs: List[Array],
    target_features: int = 8192,
) -> List[Array]:
    """Map flattened features back into a reference wavelet coefficient tree."""

    reconstructed = []
    index = 0
    for ref in reference_coeffs:
        flat_size = int(ref.size)
        if jnp.iscomplexobj(ref):
            feature_count = flat_size * 2
            chunk = features[index : index + feature_count]
            index += feature_count
            if chunk.shape[0] < feature_count:
                chunk = jnp.pad(chunk, (0, feature_count - chunk.shape[0]))
            coeff = (chunk[:flat_size] + 1j * chunk[flat_size:feature_count]).reshape(
                ref.shape
            )
        else:
            chunk = features[index : index + flat_size]
            index += flat_size
            if chunk.shape[0] < flat_size:
                chunk = jnp.pad(chunk, (0, flat_size - chunk.shape[0]))
            coeff = chunk.reshape(ref.shape)
        reconstructed.append(coeff)
    return reconstructed


def create_temporal_wavelet_sequences(
    features: Array,
    seq_length: int = 8,
    *,
    seed: int = 42,
) -> Array:
    """Create time-major feature sequences for oscillatory processing."""

    key = jax.random.PRNGKey(seed)
    sequences = []
    for step in range(seq_length):
        phase = 2.0 * jnp.pi * step / seq_length
        weight = 0.75 + 0.2 * jnp.sin(phase) + 0.05 * jnp.sin(2.0 * phase)
        noise = jax.random.normal(jax.random.fold_in(key, step), features.shape) * 0.001
        sequences.append(features * weight + noise)
    return jnp.stack(sequences, axis=0)


def _synthetic_feature_sequences(
    n_samples: int,
    input_dim: int,
    seq_length: int,
    seed: int,
) -> Array:
    key = jax.random.PRNGKey(seed)
    feature_axis = jnp.linspace(0.0, 1.0, input_dim)
    samples = []
    for sample_idx in range(n_samples):
        freq = 1.0 + (sample_idx % 7)
        base = jnp.sin(2.0 * jnp.pi * freq * feature_axis)
        harmonic = 0.25 * jnp.cos(2.0 * jnp.pi * (freq + 0.5) * feature_axis)
        noise = jax.random.normal(jax.random.fold_in(key, sample_idx), (input_dim,)) * 0.01
        samples.append(base + harmonic + noise)
    features = jnp.stack(samples, axis=0).astype(jnp.float32)
    return create_temporal_wavelet_sequences(features, seq_length, seed=seed)


def prepare_audio_experiment_data(
    config: AudioWaveletExperimentConfig,
) -> AudioExperimentData:
    """Prepare time-major wavelet feature sequences for train/eval."""

    total_samples = config.train_limit + config.eval_limit
    feature_source = config.feature_source
    if feature_source == "auto":
        feature_source = "wavelet" if HAS_JAXWT else "synthetic-features"

    if feature_source == "synthetic-features":
        sequences = _synthetic_feature_sequences(
            total_samples,
            config.input_dim,
            config.sequence_length,
            config.run.seed,
        )
        return AudioExperimentData(
            train_sequences=sequences[:, : config.train_limit, :],
            eval_sequences=sequences[
                :, config.train_limit : config.train_limit + config.eval_limit, :
            ],
            train_audio=None,
            eval_audio=None,
            train_coeffs=[],
            eval_coeffs=[],
            file_names=[f"synthetic_features_{i:02d}" for i in range(total_samples)],
            metadata={
                "feature_source": "synthetic-features",
                "sample_axis": 1,
            },
        )

    if feature_source != "wavelet":
        raise ValueError("feature_source must be 'auto', 'wavelet', or 'synthetic-features'")
    if not HAS_JAXWT:
        raise ImportError("jaxwt is required for feature_source='wavelet'")

    audio, names, resolved_source = load_audio_files(
        config.audio_dir,
        sample_rate=config.sample_rate,
        duration=config.duration,
        n_samples=total_samples,
        seed=config.run.seed,
        data_source=config.data_source,
    )

    coeffs = [
        audio_to_wavelets(audio[index], levels=config.levels, wavelet=config.wavelet)
        for index in range(audio.shape[0])
    ]
    features = jnp.stack(
        [
            smart_wavelet_compression(item, target_features=config.input_dim)
            for item in coeffs
        ],
        axis=0,
    )
    sequences = create_temporal_wavelet_sequences(
        features,
        config.sequence_length,
        seed=config.run.seed,
    )

    train_end = config.train_limit
    eval_end = config.train_limit + config.eval_limit
    return AudioExperimentData(
        train_sequences=sequences[:, :train_end, :],
        eval_sequences=sequences[:, train_end:eval_end, :],
        train_audio=audio[:train_end],
        eval_audio=audio[train_end:eval_end],
        train_coeffs=coeffs[:train_end],
        eval_coeffs=coeffs[train_end:eval_end],
        file_names=names,
        metadata={
            "feature_source": "wavelet",
            "data_source": resolved_source,
            "sample_axis": 1,
            "wavelet": config.wavelet,
            "levels": config.levels,
            "sample_rate": config.sample_rate,
            "duration": config.duration,
        },
    )


def build_audio_model(
    config: AudioWaveletExperimentConfig,
    key: jax.random.PRNGKey,
) -> ProductionWaveletAutoencoder:
    return ProductionWaveletAutoencoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        omega_bounds=config.omega_bounds,
        gamma_bounds=config.gamma_bounds,
        key=key,
    )


def _checkpoint_hyperparams(config: AudioWaveletExperimentConfig) -> Dict[str, object]:
    return {
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "omega_bounds": list(config.omega_bounds),
        "gamma_bounds": list(config.gamma_bounds),
    }


def _build_audio_model_from_hyperparams(**hyperparams) -> ProductionWaveletAutoencoder:
    return ProductionWaveletAutoencoder(
        input_dim=int(hyperparams["input_dim"]),
        hidden_dim=int(hyperparams["hidden_dim"]),
        latent_dim=int(hyperparams["latent_dim"]),
        omega_bounds=tuple(hyperparams.get("omega_bounds", (0.2, 6.0))),
        gamma_bounds=tuple(hyperparams.get("gamma_bounds", (0.01, 0.15))),
        key=jax.random.PRNGKey(0),
    )


def _load_checkpoint(checkpoint_path: Path) -> ProductionWaveletAutoencoder:
    with open(checkpoint_path, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = _build_audio_model_from_hyperparams(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)


def save_audio_artifacts(
    model: ProductionWaveletAutoencoder,
    batch: Optional[Array],
    paths: ExperimentPaths,
    epoch: int,
    metrics: Dict[str, object],
    *,
    config: AudioWaveletExperimentConfig,
    data: AudioExperimentData,
) -> None:
    """Save feature reconstructions plus latent and oscillator-state traces."""

    if batch is None:
        return

    prediction = model(batch)
    latent = model.encode(batch)
    trace = collect_sequence_state_trace(model.autoencoder.encoder.sequence, batch)

    np.savez(
        paths.traces / f"audio_wavelet_latent_state_epoch_{epoch:03d}.npz",
        latent=np.asarray(latent),
        inputs=np.asarray(batch),
        reconstructions=np.asarray(prediction),
        encoder_outputs=np.asarray(trace["outputs"]),
        encoder_positions=np.asarray(trace["positions"]),
        encoder_velocities=np.asarray(trace["velocities"]),
        final_position=np.asarray(trace["final_position"]),
        final_velocity=np.asarray(trace["final_velocity"]),
    )

    first_input = np.asarray(batch[-1, 0, :])
    first_reconstruction = np.asarray(prediction[-1, 0, :])
    feature_count = min(512, first_input.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(first_input[:feature_count], label="input", linewidth=1.0)
    ax.plot(first_reconstruction[:feature_count], label="reconstruction", linewidth=1.0)
    ax.set_title(f"Audio Wavelet Feature Reconstruction - Epoch {epoch}")
    ax.set_xlabel("feature")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        paths.plots / f"audio_wavelet_reconstruction_epoch_{epoch:03d}.png",
        dpi=150,
    )
    plt.close(fig)

    if data.eval_audio is not None and data.eval_coeffs and HAS_JAXWT:
        recon_coeffs = smart_wavelet_decompression(
            prediction[-1, 0, :],
            data.eval_coeffs[0],
            target_features=config.input_dim,
        )
        reconstructed_audio = wavelets_to_audio(recon_coeffs, config.wavelet)
        original_audio = data.eval_audio[0]
        n_audio = min(int(original_audio.shape[0]), int(reconstructed_audio.shape[0]))

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(np.asarray(original_audio[:n_audio]), label="input", alpha=0.8)
        ax.plot(np.asarray(reconstructed_audio[:n_audio]), label="reconstruction", alpha=0.8)
        ax.set_title(f"Audio Waveform Reconstruction - Epoch {epoch}")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.plots / f"audio_waveform_epoch_{epoch:03d}.png", dpi=150)
        plt.close(fig)

        if HAS_SOUNDFILE:
            audio_path = paths.artifacts / f"audio_reconstruction_epoch_{epoch:03d}.wav"
            sf.write(audio_path, np.asarray(reconstructed_audio[:n_audio]), config.sample_rate)


def run_audio_wavelet_experiment(
    config: AudioWaveletExperimentConfig,
) -> AutoencoderExperimentResult:
    """Run the audio wavelet oscillator autoencoder benchmark."""

    logger = _logger()
    data = prepare_audio_experiment_data(config)
    task_config = asdict(config)
    task_config["resolved_data"] = data.metadata
    task_config["file_names"] = data.file_names

    def artifact_callback(model, batch, paths, epoch, metrics):
        save_audio_artifacts(
            model,
            batch,
            paths,
            epoch,
            metrics,
            config=config,
            data=data,
        )

    if config.run.mode == "eval":
        if config.checkpoint is not None:
            model = _load_checkpoint(config.checkpoint)
        else:
            model = build_audio_model(config, jax.random.PRNGKey(config.run.seed))
        return run_eval_only(
            model,
            data.eval_sequences,
            config.run,
            sample_axis=1,
            task_config=task_config,
            artifact_callback=artifact_callback,
        )

    model = build_audio_model(config, jax.random.PRNGKey(config.run.seed))
    logger.info(
        "audio_wavelet data=%s train_shape=%s eval_shape=%s",
        data.metadata,
        data.train_sequences.shape,
        data.eval_sequences.shape,
    )
    return train_autoencoder(
        model,
        data.train_sequences,
        data.eval_sequences,
        config.run,
        sample_axis=1,
        task_config=task_config,
        checkpoint_hyperparams=_checkpoint_hyperparams(config),
        artifact_callback=artifact_callback,
        logger=logger,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OscNet audio wavelet oscillator autoencoder benchmark."
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reference/audio_wavelet"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--artifact-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--audio-dir", type=Path, default=Path("my_audio_samples"))
    parser.add_argument("--data-source", choices=["auto", "audio-dir", "synthetic"], default="auto")
    parser.add_argument(
        "--feature-source",
        choices=["auto", "wavelet", "synthetic-features"],
        default="auto",
    )
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=2)
    return parser


def config_from_args(args: argparse.Namespace) -> AudioWaveletExperimentConfig:
    run = AutoencoderExperimentConfig(
        name="audio_wavelet_oscillator_autoencoder",
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        artifact_every=args.artifact_every,
    )
    return AudioWaveletExperimentConfig(
        run=run,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        sequence_length=args.sequence_length,
        wavelet=args.wavelet,
        levels=args.levels,
        sample_rate=args.sample_rate,
        duration=args.duration,
        audio_dir=args.audio_dir,
        data_source=args.data_source,
        feature_source=args.feature_source,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        checkpoint=args.checkpoint,
    )


def main(argv: Optional[list[str]] = None) -> AutoencoderExperimentResult:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_audio_wavelet_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
