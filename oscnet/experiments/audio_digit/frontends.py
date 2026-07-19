"""Frontend feature extractors for the audio-digit probe.

All frontends map ``(batch, T)`` waveforms to ``(batch, feature_dim)``.
The resonator arms use the library ``ResonatorBank`` in ``oscnet.core``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.core import ResonatorBank

from .config import AudioDigitConfig

Array = jnp.ndarray


class ResonatorFrontend(eqx.Module):
    bank: ResonatorBank
    feature_mode: str = eqx.field(static=True)
    num_frames: int = eqx.field(static=True)

    def __init__(self, config: AudioDigitConfig, *, spacing: str = "log"):
        learnable = bool(config.learnable_frontend) or (
            config.frontend == "resonator_learn"
        )
        self.bank = ResonatorBank(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            f_min_hz=config.f_min_hz,
            f_max_hz=config.f_max_hz,
            quality_factor=config.quality_factor,
            spacing=spacing,
            pool=config.pool,
            transient_fraction=config.transient_fraction,
            learnable=learnable,
            unit_peak_gain=True,
            readout=config.readout,
            nonlinearity=config.nonlinearity,
            agc_strength=config.agc_strength,
        )
        self.feature_mode = str(config.feature_mode)
        self.num_frames = int(config.num_frames)

    @property
    def feature_dim(self) -> int:
        return self.bank.feature_dim

    def __call__(self, waveforms: Array) -> Array:
        # Cap filter length so 16 kHz Speech Commands stays practical.
        # Sample rate is scaled with the subsample so band centers stay
        # consistent; disclosed in the paper protocol.
        if self.feature_mode == "frames":
            return self.bank.encode_frames(
                waveforms, num_frames=self.num_frames, max_samples=4096
            )
        if self.feature_mode != "pooled":
            raise ValueError(f"unknown feature_mode {self.feature_mode!r}")
        return self.bank.encode(waveforms, max_samples=4096)


class MelFrontend(eqx.Module):
    """Mel-filterbank log-energies (pooled whole-clip or framed)."""

    num_bands: int = eqx.field(static=True)
    feature_mode: str = eqx.field(static=True)
    num_frames: int = eqx.field(static=True)
    frame_len: int = eqx.field(static=True)
    mel_filters: Array

    def __init__(self, config: AudioDigitConfig):
        self.num_bands = int(config.num_bands)
        self.feature_mode = str(config.feature_mode)
        self.num_frames = int(config.num_frames)
        n_full = int(config.num_samples)
        if self.feature_mode == "frames":
            self.frame_len = max(256, n_full // max(self.num_frames, 1))
            n_fft = self.frame_len
        else:
            self.frame_len = n_full
            n_fft = n_full
        self.mel_filters = _mel_filterbank(
            num_bands=self.num_bands,
            n_fft=n_fft,
            sample_rate=float(config.sample_rate),
            f_min=float(config.f_min_hz),
            f_max=float(
                min(config.f_max_hz, 0.5 * config.sample_rate - 1.0)
            ),
        )

    @property
    def feature_dim(self) -> int:
        return self.num_bands

    def _mel_log(self, waveforms: Array) -> Array:
        n_fft = waveforms.shape[-1]
        window = jnp.hanning(n_fft).astype(jnp.float32)
        spec = jnp.abs(jnp.fft.rfft(waveforms * window, n=n_fft, axis=-1)) ** 2
        mel = spec @ self.mel_filters.T
        return jnp.log(mel + 1e-6)

    def __call__(self, waveforms: Array) -> Array:
        if self.feature_mode == "pooled":
            return self._mel_log(waveforms)
        if self.feature_mode != "frames":
            raise ValueError(f"unknown feature_mode {self.feature_mode!r}")
        # Non-overlapping frames padded/truncated to ``num_frames``.
        t = waveforms.shape[-1]
        frame_len = int(self.frame_len)
        n_frames = int(self.num_frames)
        need = frame_len * n_frames
        if t < need:
            waveforms = jnp.pad(waveforms, ((0, 0), (0, need - t)))
        else:
            waveforms = waveforms[:, :need]
        frames = waveforms.reshape(waveforms.shape[0], n_frames, frame_len)
        return jax.vmap(self._mel_log, in_axes=1, out_axes=1)(frames)


class STFTFrontend(eqx.Module):
    """Whole-clip linear-frequency band log-energies."""

    num_bands: int = eqx.field(static=True)
    band_masks: Array

    def __init__(self, config: AudioDigitConfig):
        self.num_bands = int(config.num_bands)
        n_fft = int(config.num_samples)
        freqs = jnp.fft.rfftfreq(n_fft, d=1.0 / float(config.sample_rate))
        f_max = float(min(config.f_max_hz, 0.5 * config.sample_rate - 1.0))
        edges = jnp.linspace(
            float(config.f_min_hz),
            f_max,
            self.num_bands + 1,
            dtype=jnp.float32,
        )
        masks = []
        for i in range(self.num_bands):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            masks.append(mask.astype(jnp.float32))
        self.band_masks = jnp.stack(masks, axis=0)

    @property
    def feature_dim(self) -> int:
        return self.num_bands

    def __call__(self, waveforms: Array) -> Array:
        n_fft = waveforms.shape[-1]
        window = jnp.hanning(n_fft).astype(jnp.float32)
        power = jnp.abs(jnp.fft.rfft(waveforms * window, n=n_fft, axis=-1)) ** 2
        # (B, F) @ (F, N) via masks (N, F)
        denom = jnp.sum(self.band_masks, axis=-1) + 1e-6
        bands = (power @ self.band_masks.T) / denom
        return jnp.log(bands + 1e-6)


class RawFrontend(eqx.Module):
    target_dim: int = eqx.field(static=True)

    def __init__(self, config: AudioDigitConfig, *, wide: bool = False):
        if wide:
            self.target_dim = int(
                config.raw_wide_dim
                if config.raw_wide_dim is not None
                else config.num_bands
            )
        else:
            self.target_dim = int(config.num_samples)

    @property
    def feature_dim(self) -> int:
        return self.target_dim

    def __call__(self, waveforms: Array) -> Array:
        t = waveforms.shape[-1]
        target = self.target_dim

        def one(wave: Array) -> Array:
            idx = jnp.linspace(0, t - 1, target)
            left = jnp.floor(idx).astype(jnp.int32)
            right = jnp.minimum(left + 1, t - 1)
            w = idx - left.astype(jnp.float32)
            return (1.0 - w) * wave[left] + w * wave[right]

        return jax.vmap(one)(waveforms)


class Conv1dFrontend(eqx.Module):
    """Tiny learned 1-D conv stem → ``num_bands`` features."""

    conv: eqx.nn.Conv1d
    num_bands: int = eqx.field(static=True)

    def __init__(self, config: AudioDigitConfig, *, key: jax.random.PRNGKey):
        self.num_bands = int(config.num_bands)
        self.conv = eqx.nn.Conv1d(
            in_channels=1,
            out_channels=self.num_bands,
            kernel_size=25,
            stride=4,
            padding=12,
            key=key,
        )

    @property
    def feature_dim(self) -> int:
        return self.num_bands

    def __call__(self, waveforms: Array) -> Array:
        x = waveforms[:, None, :]
        y = jax.vmap(self.conv)(x)
        y = jax.nn.relu(y)
        return jnp.mean(y, axis=-1)


def build_frontend(
    config: AudioDigitConfig,
    *,
    key: jax.random.PRNGKey,
) -> eqx.Module:
    name = config.frontend
    if name in ("resonator", "resonator_learn"):
        return ResonatorFrontend(config, spacing="log")
    if name == "resonator_equal":
        return ResonatorFrontend(config, spacing="equal")
    if name == "mel":
        return MelFrontend(config)
    if name == "stft":
        return STFTFrontend(config)
    if name == "raw":
        return RawFrontend(config, wide=False)
    if name == "raw_wide":
        return RawFrontend(config, wide=True)
    if name == "conv1d":
        return Conv1dFrontend(config, key=key)
    raise ValueError(f"unknown frontend {name!r}")


def _hz_to_mel(hz: Array) -> Array:
    return 2595.0 * jnp.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: Array) -> Array:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    num_bands: int,
    n_fft: int,
    sample_rate: float,
    f_min: float,
    f_max: float,
) -> Array:
    n_freqs = n_fft // 2 + 1
    m_min = float(_hz_to_mel(jnp.asarray(f_min)))
    m_max = float(_hz_to_mel(jnp.asarray(f_max)))
    m_pts = jnp.linspace(m_min, m_max, num_bands + 2)
    hz_pts = _mel_to_hz(m_pts)
    bins = jnp.floor((n_fft + 1) * hz_pts / sample_rate).astype(jnp.int32)
    filters = jnp.zeros((num_bands, n_freqs), dtype=jnp.float32)
    for i in range(num_bands):
        left = int(bins[i])
        center = int(max(int(bins[i + 1]), left + 1))
        right = int(max(int(bins[i + 2]), center + 1))
        right = min(right, n_freqs - 1)
        if center <= left or right <= center:
            continue
        up = jnp.arange(left, center, dtype=jnp.float32)
        down = jnp.arange(center, right, dtype=jnp.float32)
        filters = filters.at[i, left:center].set((up - left) / (center - left))
        filters = filters.at[i, center:right].set(
            (right - down) / (right - center)
        )
    return filters
