"""Image encoders that replace / precede the conv state-anchor stem.

Two Stage-0b drives:

- ``gabor``: frozen spatial-frequency / Gabor bank (default honest image RFB)
- ``row_scan``: treat each image row as a 1-D stream into ``ResonatorBank``
  (audio analogy; diagnostic only — anisotropic pathology)

Both return ``(position, velocity)`` shaped for a retinotopic HORN grid.
"""

from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.core import GaborFilterBank, ResonatorBank

from .common import Array


def _image_hw_channels(image_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(image_shape) == 2:
        return int(image_shape[0]), int(image_shape[1]), 1
    if len(image_shape) == 3:
        return int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
    raise ValueError(f"unsupported image_shape {image_shape}")


class GaborStateEncoder(eqx.Module):
    """Frozen Gabor bank → linear map into HORN ``(position, velocity)``."""

    bank: GaborFilterBank
    proj: eqx.nn.Linear
    num_oscillators: int = eqx.field(static=True)
    image_shape: Tuple[int, ...] = eqx.field(static=True)
    freeze_bank: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_oscillators: int,
        image_shape: Tuple[int, ...],
        num_orientations: int = 4,
        num_scales: int = 4,
        kernel_size: int = 9,
        freeze_bank: bool = True,
        key: jax.random.PRNGKey,
    ):
        self.bank = GaborFilterBank(
            num_orientations=num_orientations,
            num_scales=num_scales,
            kernel_size=kernel_size,
        )
        self.num_oscillators = int(num_oscillators)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.freeze_bank = bool(freeze_bank)
        self.proj = eqx.nn.Linear(
            self.bank.num_bands, 2 * self.num_oscillators, key=key
        )

    def __call__(self, images: Array) -> Tuple[Array, Array]:
        """``images`` flat ``(B, H*W*C)`` → ``(pos, vel)`` each ``(B, N)``."""

        height, width, channels = _image_hw_channels(self.image_shape)
        batch = images.shape[0]
        imgs = images.reshape(batch, channels, height, width)
        feats = self.bank.encode(imgs)
        if self.freeze_bank:
            feats = jax.lax.stop_gradient(feats)
        state = jax.vmap(self.proj)(feats)
        position, velocity = jnp.split(jnp.tanh(state), 2, axis=-1)
        return position, velocity


class RowScanResonatorEncoder(eqx.Module):
    """Diagnostic: each row is a 1-D drive into a shared ResonatorBank."""

    bank: ResonatorBank
    proj: eqx.nn.Linear
    num_oscillators: int = eqx.field(static=True)
    image_shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_oscillators: int,
        image_shape: Tuple[int, ...],
        num_bands: int = 16,
        quality_factor: float = 4.0,
        key: jax.random.PRNGKey,
    ):
        height, width, channels = _image_hw_channels(image_shape)
        # Treat horizontal pixel index as "time"; sample_rate = width (unitless)
        self.bank = ResonatorBank(
            num_bands=num_bands,
            sample_rate=float(width),
            f_min_hz=1.0,
            f_max_hz=0.45 * float(width),
            quality_factor=quality_factor,
            spacing="log",
            pool="log_rms",
            unit_peak_gain=True,
        )
        self.num_oscillators = int(num_oscillators)
        self.image_shape = tuple(int(v) for v in image_shape)
        # Mean over rows of band features → project
        self.proj = eqx.nn.Linear(
            self.bank.num_bands, 2 * self.num_oscillators, key=key
        )

    def __call__(self, images: Array) -> Tuple[Array, Array]:
        height, width, channels = _image_hw_channels(self.image_shape)
        batch = images.shape[0]
        imgs = images.reshape(batch, channels, height, width)
        gray = jnp.mean(imgs, axis=1)  # (B, H, W)

        def one(img: Array) -> Array:
            # img (H, W): encode each row, mean over rows
            row_feats = self.bank.encode(img)  # (H, bands)
            return jnp.mean(row_feats, axis=0)

        feats = jax.vmap(one)(gray)
        feats = jax.lax.stop_gradient(feats)
        state = jax.vmap(self.proj)(feats)
        position, velocity = jnp.split(jnp.tanh(state), 2, axis=-1)
        return position, velocity


def build_image_rfb_encoder(
    *,
    drive: str,
    num_oscillators: int,
    image_shape: Tuple[int, ...],
    key: jax.random.PRNGKey,
    num_bands: int = 16,
) -> eqx.Module:
    """Factory for Stage 0b / Stage 1 image RFB encoders."""

    if drive == "gabor":
        return GaborStateEncoder(
            num_oscillators=num_oscillators,
            image_shape=image_shape,
            key=key,
        )
    if drive == "row_scan":
        return RowScanResonatorEncoder(
            num_oscillators=num_oscillators,
            image_shape=image_shape,
            num_bands=num_bands,
            key=key,
        )
    raise ValueError("drive must be 'gabor' or 'row_scan'")
