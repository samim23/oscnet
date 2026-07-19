"""Fixed Gabor / spatial-frequency filter banks for image RFB frontends.

This is the honest static-image analog of a 1-D resonator filter bank: a
frozen, structured spectral frontend (Stage 0b option b in the RFB plan).
"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def gabor_kernel(
    *,
    kernel_size: int,
    wavelength: float,
    theta: float,
    sigma: float,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> Array:
    """Single real Gabor kernel, shape ``(kernel_size, kernel_size)``."""

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    if wavelength <= 0.0 or sigma <= 0.0:
        raise ValueError("wavelength and sigma must be positive")
    half = kernel_size // 2
    y, x = jnp.mgrid[-half : half + 1, -half : half + 1]
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    x_theta = x * jnp.cos(theta) + y * jnp.sin(theta)
    y_theta = -x * jnp.sin(theta) + y * jnp.cos(theta)
    gaussian = jnp.exp(
        -0.5
        * (
            x_theta**2 / (sigma**2)
            + (gamma**2) * y_theta**2 / (sigma**2)
        )
    )
    wave = jnp.cos(2.0 * jnp.pi * x_theta / wavelength + psi)
    kernel = gaussian * wave
    kernel = kernel - jnp.mean(kernel)
    norm = jnp.sqrt(jnp.sum(kernel * kernel) + 1e-8)
    return (kernel / norm).astype(jnp.float32)


def build_gabor_bank(
    *,
    num_orientations: int = 4,
    num_scales: int = 4,
    kernel_size: int = 9,
    base_wavelength: float = 3.0,
    scale_factor: float = 1.4,
    sigma_over_wavelength: float = 0.56,
    gamma: float = 0.5,
    spacing: str = "log",
) -> Array:
    """Return kernels ``(num_bands, 1, k, k)`` for depthwise/grouped conv use.

    ``spacing="log"`` uses multi-scale wavelengths (default). ``spacing="equal"``
    collapses all scales onto the mid-band wavelength — the image analog of
    the audio equal-frequency resonator control.
    """

    if spacing not in ("log", "equal"):
        raise ValueError("spacing must be 'log' or 'equal'")
    kernels = []
    mid_scale = 0.5 * float(max(num_scales - 1, 0))
    for scale in range(num_scales):
        if spacing == "equal":
            wavelength = float(base_wavelength) * (
                float(scale_factor) ** mid_scale
            )
        else:
            wavelength = float(base_wavelength) * (float(scale_factor) ** scale)
        sigma = sigma_over_wavelength * wavelength
        for orient in range(num_orientations):
            theta = jnp.pi * float(orient) / float(num_orientations)
            kernels.append(
                gabor_kernel(
                    kernel_size=kernel_size,
                    wavelength=wavelength,
                    theta=float(theta),
                    sigma=sigma,
                    gamma=gamma,
                )
            )
    bank = jnp.stack(kernels, axis=0)[:, None, :, :]
    return bank.astype(jnp.float32)


class GaborFilterBank(eqx.Module):
    """Frozen multi-orientation / multi-scale Gabor frontend."""

    kernels: Array
    num_bands: int = eqx.field(static=True)
    pool: str = eqx.field(static=True)
    spacing: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_orientations: int = 4,
        num_scales: int = 4,
        kernel_size: int = 9,
        pool: str = "mean",
        spacing: str = "log",
    ):
        self.kernels = build_gabor_bank(
            num_orientations=num_orientations,
            num_scales=num_scales,
            kernel_size=kernel_size,
            spacing=spacing,
        )
        self.num_bands = int(self.kernels.shape[0])
        self.pool = str(pool)
        self.spacing = str(spacing)

    def filter_map(self, images: Array) -> Array:
        """Apply bank to images.

        Args:
            images: ``(batch, H, W)`` or ``(batch, C, H, W)`` (uses mean over C)

        Returns:
            ``(batch, num_bands, H', W')`` valid convolution maps.
        """

        if images.ndim == 3:
            x = images[:, None, :, :]
        elif images.ndim == 4:
            x = jnp.mean(images, axis=1, keepdims=True)
        else:
            raise ValueError("images must be (B,H,W) or (B,C,H,W)")

        k = self.kernels.shape[-1]
        pad = k // 2

        def one_image(img: Array) -> Array:
            # img: (1, H, W)
            padded = jnp.pad(img, ((0, 0), (pad, pad), (pad, pad)))

            def one_kernel(kernel: Array) -> Array:
                # kernel: (1, k, k)
                return jax.lax.conv_general_dilated(
                    padded[None, ...],
                    kernel[None, ...],
                    window_strides=(1, 1),
                    padding="VALID",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )[0, 0]

            return jax.vmap(one_kernel)(self.kernels)

        return jax.vmap(one_image)(x)

    def encode(self, images: Array) -> Array:
        """Pool filter maps to ``(batch, num_bands)`` log-energy features."""

        maps = self.filter_map(images)
        energy = jnp.mean(maps * maps, axis=(-2, -1))
        if self.pool == "mean":
            return jnp.log(energy + 1e-6)
        if self.pool == "max":
            return jnp.log(jnp.max(maps * maps, axis=(-2, -1)) + 1e-6)
        raise ValueError("pool must be 'mean' or 'max'")
