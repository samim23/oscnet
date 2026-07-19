"""Spoken-word datasets for the audio-digit RFB→HORN experiment.

Default ``synthetic`` data needs no downloads and is CI-safe.
``speech_commands`` downloads Google's Speech Commands v0.02 tarball
(no TensorFlow required) and loads ``word_set`` subsets (digits / core10).
"""

from __future__ import annotations

import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import AudioDigitConfig

Array = jnp.ndarray

_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)

# Standard Speech Commands "core" 10-word set (non-digit transfer probe).
_CORE10_WORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


def _words_for_config(config: AudioDigitConfig) -> Tuple[str, ...]:
    if config.word_set == "digits":
        return _DIGIT_WORDS
    if config.word_set == "core10":
        return _CORE10_WORDS
    raise ValueError(
        f"unknown word_set {config.word_set!r}; choose 'digits' or 'core10'"
    )

_SPEECH_COMMANDS_URL = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
)
_NATIVE_SR = 16_000


def _formant_freqs(digit: int) -> Tuple[float, float, float]:
    """Crude digit-dependent formant centers (Hz) for synthetic speech-ish tones."""

    base = 200.0 + 35.0 * float(digit)
    return base, base * 2.3, base * 3.7


def synthesize_digit_waveform(
    key: jax.random.PRNGKey,
    digit: int,
    *,
    sample_rate: float,
    num_samples: int,
) -> Array:
    """One synthetic spoken-digit-like waveform, shape ``(num_samples,)``."""

    k_env, k_noise, k_phase = jax.random.split(key, 3)
    t = jnp.arange(num_samples, dtype=jnp.float32) / float(sample_rate)
    f1, f2, f3 = _formant_freqs(int(digit))
    phases = jax.random.uniform(k_phase, (3,), minval=0.0, maxval=2.0 * jnp.pi)
    carrier = (
        0.55 * jnp.sin(2.0 * jnp.pi * f1 * t + phases[0])
        + 0.30 * jnp.sin(2.0 * jnp.pi * f2 * t + phases[1])
        + 0.15 * jnp.sin(2.0 * jnp.pi * f3 * t + phases[2])
    )
    attack = 0.02 + 0.01 * float(digit % 3)
    env = jnp.exp(-0.5 * ((t - 0.15) / 0.12) ** 2)
    env = env * (1.0 - jnp.exp(-t / attack))
    if digit in (6, 7):
        noise_burst = jax.random.normal(k_noise, (num_samples,)) * jnp.exp(
            -t / 0.05
        )
        carrier = carrier + 0.35 * noise_burst
    else:
        carrier = carrier + 0.05 * jax.random.normal(k_noise, (num_samples,))
    wave_arr = (env * carrier).astype(jnp.float32)
    peak = jnp.maximum(jnp.max(jnp.abs(wave_arr)), 1e-3)
    return wave_arr / peak


def make_synthetic_split(
    key: jax.random.PRNGKey,
    *,
    num_samples_total: int,
    config: AudioDigitConfig,
) -> Tuple[Array, Array]:
    """Return ``(waveforms, labels)`` with balanced digits."""

    per_class = max(1, int(num_samples_total) // int(config.num_classes))
    keys = jax.random.split(key, per_class * config.num_classes)
    waves = []
    labels = []
    idx = 0
    for digit in range(config.num_classes):
        for _ in range(per_class):
            waves.append(
                synthesize_digit_waveform(
                    keys[idx],
                    digit,
                    sample_rate=config.sample_rate,
                    num_samples=config.num_samples,
                )
            )
            labels.append(digit)
            idx += 1
    waveforms = jnp.stack(waves, axis=0)
    label_arr = jnp.asarray(labels, dtype=jnp.int32)
    perm = jax.random.permutation(key, label_arr.shape[0])
    return waveforms[perm], label_arr[perm]


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "oscnet" / "speech_commands_v0.02"


def ensure_speech_commands(data_dir: Optional[str] = None) -> Path:
    """Download + extract Speech Commands v0.02 if needed; return root path."""

    root = Path(data_dir) if data_dir else _default_cache_dir()
    marker = root / ".oscnet_extracted"
    if marker.is_file() and (root / "one").is_dir():
        return root
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "speech_commands_v0.02.tar.gz"
    if not tar_path.is_file():
        print(f"Downloading Speech Commands to {tar_path} ...")
        urllib.request.urlretrieve(_SPEECH_COMMANDS_URL, tar_path)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root)
    marker.write_text("ok\n")
    return root


def _read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"unsupported sampwidth {sampwidth} in {path}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio


def _resample_linear(audio: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    if abs(src_sr - dst_sr) < 1e-3:
        return audio.astype(np.float32)
    ratio = float(dst_sr) / float(src_sr)
    new_len = max(1, int(round(audio.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _fit_length(audio: np.ndarray, target: int) -> np.ndarray:
    if audio.shape[0] >= target:
        return audio[:target]
    return np.pad(audio, (0, target - audio.shape[0]))


def _load_list_set(root: Path, name: str) -> set:
    path = root / name
    if not path.is_file():
        return set()
    return {line.strip().replace("\\", "/") for line in path.read_text().splitlines() if line.strip()}


def load_speech_commands_digits(
    config: AudioDigitConfig,
) -> Tuple[Array, Array, Array, Array]:
    """Load a Speech Commands word subset (TF-free)."""

    root = ensure_speech_commands(config.speech_commands_data_dir)
    val_set = _load_list_set(root, "validation_list.txt")
    test_set = _load_list_set(root, "testing_list.txt")
    words = _words_for_config(config)
    word_ids = {name: i for i, name in enumerate(words)}
    n_classes = len(words)

    # Collect paths first (cheap), then load a balanced subsample.
    train_paths: List[Tuple[Path, int]] = []
    eval_paths: List[Tuple[Path, int]] = []
    for word, label in word_ids.items():
        folder = root / word
        if not folder.is_dir():
            continue
        for path in folder.glob("*.wav"):
            rel = str(path.relative_to(root)).replace("\\", "/")
            if rel in val_set or rel in test_set:
                eval_paths.append((path, label))
            else:
                train_paths.append((path, label))

    if len(train_paths) < n_classes or len(eval_paths) < n_classes:
        raise RuntimeError(
            f"insufficient speech_commands {config.word_set}: "
            f"train={len(train_paths)} eval={len(eval_paths)} under {root}"
        )

    rng = np.random.default_rng(int(config.seed) + 17)

    def _balanced_take(
        items: List[Tuple[Path, int]], n: int
    ) -> List[Tuple[Path, int]]:
        by_label: Dict[int, List[Path]] = {i: [] for i in range(n_classes)}
        for path, label in items:
            by_label[label].append(path)
        per = max(1, n // n_classes)
        chosen: List[Tuple[Path, int]] = []
        for label in range(n_classes):
            paths = by_label[label]
            rng.shuffle(paths)
            for path in paths[:per]:
                chosen.append((path, label))
        rng.shuffle(chosen)
        return chosen[:n]

    def _load_pairs(pairs: List[Tuple[Path, int]]) -> Tuple[np.ndarray, np.ndarray]:
        waves = []
        labels = []
        for path, label in pairs:
            audio = _read_wav_mono(path)
            audio = _resample_linear(audio, _NATIVE_SR, config.sample_rate)
            audio = _fit_length(audio, config.num_samples)
            peak = float(np.max(np.abs(audio))) + 1e-6
            waves.append((audio / peak).astype(np.float32))
            labels.append(label)
        return np.stack(waves), np.asarray(labels, dtype=np.int32)

    train_n = min(int(config.train_samples), len(train_paths))
    eval_n = min(int(config.eval_samples), len(eval_paths))
    train_x, train_y = _load_pairs(_balanced_take(train_paths, train_n))
    eval_x, eval_y = _load_pairs(_balanced_take(eval_paths, eval_n))
    return (
        jnp.asarray(train_x),
        jnp.asarray(train_y),
        jnp.asarray(eval_x),
        jnp.asarray(eval_y),
    )


def load_audio_digit_data(
    config: AudioDigitConfig,
    *,
    key: jax.random.PRNGKey,
) -> Dict[str, Array]:
    """Load train/eval waveforms and labels for the configured source."""

    if config.data_source == "synthetic":
        k_train, k_eval = jax.random.split(key)
        train_x, train_y = make_synthetic_split(
            k_train, num_samples_total=config.train_samples, config=config
        )
        eval_x, eval_y = make_synthetic_split(
            k_eval, num_samples_total=config.eval_samples, config=config
        )
    elif config.data_source == "speech_commands":
        train_x, train_y, eval_x, eval_y = load_speech_commands_digits(config)
    else:
        raise ValueError(
            "data_source must be 'synthetic' or 'speech_commands', "
            f"got {config.data_source!r}"
        )
    return {
        "train_waveforms": train_x,
        "train_labels": train_y,
        "eval_waveforms": eval_x,
        "eval_labels": eval_y,
    }
