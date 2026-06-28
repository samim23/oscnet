"""Shared helpers for MNIST oscillator generator experiments."""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import optax

Array = jnp.ndarray


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
