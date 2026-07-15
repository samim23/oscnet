"""Shared helpers for MNIST oscillator generator experiments."""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from oscnet.models.generative.common import _image_hw_channels

Array = jnp.ndarray


def occlude_image_batch(
    images_flat: Array,
    *,
    key: jax.random.PRNGKey,
    image_shape: Sequence[int],
    fraction: float,
    patches: int = 1,
    probability: float = 1.0,
) -> Tuple[Array, Array]:
    """Zero random square patches of a flat image batch.

    ``fraction`` is the total image-area fraction to occlude, split across
    ``patches`` square patches placed independently and uniformly at random
    per sample (patches may overlap, so the realized occluded area is a lower
    bound when ``patches > 1``). With ``probability < 1`` some samples are
    left untouched. Returns the occluded flat batch and the boolean pixel
    mask with shape ``(batch, height, width)`` (True = occluded).
    """

    height, width, channels = _image_hw_channels(
        tuple(int(size) for size in image_shape)
    )
    batch = int(images_flat.shape[0])
    patches = max(1, int(patches))
    per_patch = float(fraction) / patches
    side_h = int(min(max(round(height * math.sqrt(per_patch)), 1), height))
    side_w = int(min(max(round(width * math.sqrt(per_patch)), 1), width))

    images = images_flat.reshape(batch, channels, height, width)
    mask = jnp.zeros((batch, height, width), dtype=bool)
    apply_key, *patch_keys = jax.random.split(key, patches + 1)
    rows = jnp.arange(height)[None, :]
    cols = jnp.arange(width)[None, :]
    for patch_key in patch_keys:
        row_key, col_key = jax.random.split(patch_key)
        row0 = jax.random.randint(row_key, (batch,), 0, height - side_h + 1)
        col0 = jax.random.randint(col_key, (batch,), 0, width - side_w + 1)
        row_hit = (rows >= row0[:, None]) & (rows < (row0 + side_h)[:, None])
        col_hit = (cols >= col0[:, None]) & (cols < (col0 + side_w)[:, None])
        mask = mask | (row_hit[:, :, None] & col_hit[:, None, :])
    if probability < 1.0:
        applied = jax.random.bernoulli(apply_key, float(probability), (batch,))
        mask = mask & applied[:, None, None]
    occluded = jnp.where(mask[:, None, :, :], 0.0, images)
    return occluded.reshape(batch, -1), mask


def _logger() -> logging.Logger:
    logger = logging.getLogger("oscnet.experiments.mnist_generator")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _tree_norm(tree: Any) -> Array:
    if hasattr(optax, "tree") and hasattr(optax.tree, "norm"):
        return optax.tree.norm(tree)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if leaf is not None]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
