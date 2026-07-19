"""Classification heads for audio RFB→HORN (MLP, HORN, GRU)."""

from __future__ import annotations

from typing import Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.models.fractal import FractalHORNCell

Array = jnp.ndarray


class DigitsClassifier(eqx.Module):
    """Frontend + MLP head. Frontend may be frozen (stop-gradient).

    If the frontend returns frames ``(B, T, F)``, they are mean-pooled over
    time before the MLP (matched whole-clip baseline).
    """

    frontend: eqx.Module
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    freeze_frontend: bool = eqx.field(static=True)

    def __init__(
        self,
        frontend: eqx.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        freeze_frontend: bool,
        key: jax.random.PRNGKey,
    ):
        k1, k2 = jax.random.split(key)
        self.frontend = frontend
        self.layer1 = eqx.nn.Linear(feature_dim, hidden_dim, key=k1)
        self.layer2 = eqx.nn.Linear(hidden_dim, num_classes, key=k2)
        self.freeze_frontend = bool(freeze_frontend)

    def features(self, waveforms: Array) -> Array:
        feats = self.frontend(waveforms)
        if self.freeze_frontend:
            feats = jax.lax.stop_gradient(feats)
        if feats.ndim == 3:
            feats = jnp.mean(feats, axis=1)
        return feats

    def __call__(self, waveforms: Array) -> Array:
        hidden = jax.nn.gelu(jax.vmap(self.layer1)(self.features(waveforms)))
        return jax.vmap(self.layer2)(hidden)


class DigitsHornClassifier(eqx.Module):
    """Frontend → FractalHORN → class logits.

    - ``pooled`` features ``(B, F)``: constant drive for ``horn_steps`` settles
    - ``frames`` features ``(B, T, F)``: one HORN step per frame (AudioPrism-like)
    """

    frontend: eqx.Module
    cell: FractalHORNCell
    freeze_frontend: bool = eqx.field(static=True)
    horn_steps: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)

    def __init__(
        self,
        frontend: eqx.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        horn_steps: int,
        coupling_depth: int,
        coupling_kind: str = "fractal_fixed",
        tonotopic_init_strength: float = 0.5,
        freeze_frontend: bool,
        key: jax.random.PRNGKey,
    ):
        self.frontend = frontend
        self.cell = FractalHORNCell(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            coupling_depth=coupling_depth,
            coupling_kind=coupling_kind,
            tonotopic_init_strength=tonotopic_init_strength,
            key=key,
        )
        self.freeze_frontend = bool(freeze_frontend)
        self.horn_steps = int(horn_steps)
        self.hidden_dim = int(hidden_dim)

    def features(self, waveforms: Array) -> Array:
        feats = self.frontend(waveforms)
        if self.freeze_frontend:
            feats = jax.lax.stop_gradient(feats)
        return feats

    def __call__(self, waveforms: Array) -> Array:
        feats = self.features(waveforms)
        if feats.ndim == 2:
            drives = jnp.repeat(feats[:, None, :], self.horn_steps, axis=1)
        elif feats.ndim == 3:
            drives = feats
        else:
            raise ValueError(f"expected features rank 2 or 3, got {feats.ndim}")

        batch = drives.shape[0]
        state = (
            jnp.zeros((batch, self.hidden_dim), dtype=drives.dtype),
            jnp.zeros((batch, self.hidden_dim), dtype=drives.dtype),
        )
        drives_t = jnp.swapaxes(drives, 0, 1)  # (T, B, F)

        def step(state, x_t):
            logits, state = self.cell(x_t, state)
            return state, logits

        _, logits_seq = jax.lax.scan(step, state, drives_t)
        return logits_seq[-1]


class DigitsGRUClassifier(eqx.Module):
    """Frontend → GRU over frames → class logits (non-oscillatory sequential control)."""

    frontend: eqx.Module
    cell: eqx.nn.GRUCell
    readout: eqx.nn.Linear
    freeze_frontend: bool = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    fall_back_steps: int = eqx.field(static=True)

    def __init__(
        self,
        frontend: eqx.Module,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        freeze_frontend: bool,
        fall_back_steps: int = 16,
        key: jax.random.PRNGKey,
    ):
        k_cell, k_out = jax.random.split(key)
        self.frontend = frontend
        self.cell = eqx.nn.GRUCell(feature_dim, hidden_dim, key=k_cell)
        self.readout = eqx.nn.Linear(hidden_dim, num_classes, key=k_out)
        self.freeze_frontend = bool(freeze_frontend)
        self.hidden_dim = int(hidden_dim)
        self.fall_back_steps = int(fall_back_steps)

    def features(self, waveforms: Array) -> Array:
        feats = self.frontend(waveforms)
        if self.freeze_frontend:
            feats = jax.lax.stop_gradient(feats)
        return feats

    def __call__(self, waveforms: Array) -> Array:
        feats = self.features(waveforms)
        if feats.ndim == 2:
            drives = jnp.repeat(feats[:, None, :], self.fall_back_steps, axis=1)
        elif feats.ndim == 3:
            drives = feats
        else:
            raise ValueError(f"expected features rank 2 or 3, got {feats.ndim}")
        batch = drives.shape[0]
        h0 = jnp.zeros((batch, self.hidden_dim), dtype=drives.dtype)
        drives_t = jnp.swapaxes(drives, 0, 1)

        def step(h, x_t):
            h_new = jax.vmap(self.cell)(x_t, h)
            return h_new, h_new

        h_final, _ = jax.lax.scan(step, h0, drives_t)
        return jax.vmap(self.readout)(h_final)


AudioDigitModel = Union[DigitsClassifier, DigitsHornClassifier, DigitsGRUClassifier]


def _param_count(module: eqx.Module) -> int:
    params, _ = eqx.partition(module, eqx.is_inexact_array)
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        total += int(leaf.size)
    return total


def count_trainable_params(model: AudioDigitModel) -> Tuple[int, int]:
    """Return ``(trainable_params, frontend_params)``."""

    frontend_n = _param_count(model.frontend)
    if isinstance(model, DigitsHornClassifier):
        head_n = _param_count(model.cell)
    elif isinstance(model, DigitsGRUClassifier):
        head_n = _param_count(model.cell) + _param_count(model.readout)
    else:
        head_n = _param_count(model.layer1) + _param_count(model.layer2)
    trainable = head_n if model.freeze_frontend else head_n + frontend_n
    return trainable, frontend_n
