"""Stage 5 diagnostics: high→low frequency routing in RFB→HORN stacks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .models import DigitsClassifier, DigitsGRUClassifier, DigitsHornClassifier

Array = jnp.ndarray


def _accuracy(logits: Array, labels: Array) -> float:
    return float(jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)))


def _mask_band_half(feats: Array, *, which: str) -> Array:
    """Zero the low / high / none half of the last feature axis (band dim)."""

    n = int(feats.shape[-1])
    mid = n // 2
    mask = jnp.ones((n,), dtype=feats.dtype)
    if which == "low":
        mask = mask.at[:mid].set(0.0)
    elif which == "high":
        mask = mask.at[mid:].set(0.0)
    elif which == "none":
        pass
    else:
        raise ValueError(f"unknown band mask {which!r}")
    return feats * mask


def band_ablation_accuracies(
    model: DigitsHornClassifier | DigitsClassifier | DigitsGRUClassifier,
    waveforms: Array,
    labels: Array,
) -> Dict[str, float]:
    """Eval accuracy with low / high band halves zeroed in frontend features."""

    feats = model.features(waveforms)

    def logits_from_feats(f: Array) -> Array:
        if isinstance(model, DigitsClassifier):
            if f.ndim == 3:
                f = jnp.mean(f, axis=1)
            hidden = jax.nn.gelu(jax.vmap(model.layer1)(f))
            return jax.vmap(model.layer2)(hidden)
        if f.ndim == 2:
            steps = (
                model.horn_steps
                if isinstance(model, DigitsHornClassifier)
                else model.fall_back_steps
            )
            drives = jnp.repeat(f[:, None, :], steps, axis=1)
        else:
            drives = f
        batch = drives.shape[0]
        drives_t = jnp.swapaxes(drives, 0, 1)
        if isinstance(model, DigitsGRUClassifier):
            h0 = jnp.zeros((batch, model.hidden_dim), dtype=drives.dtype)

            def gru_step(h, x_t):
                h_new = jax.vmap(model.cell)(x_t, h)
                return h_new, h_new

            h_final, _ = jax.lax.scan(gru_step, h0, drives_t)
            return jax.vmap(model.readout)(h_final)
        # HORN path
        state = (
            jnp.zeros((batch, model.hidden_dim), dtype=drives.dtype),
            jnp.zeros((batch, model.hidden_dim), dtype=drives.dtype),
        )

        def step(state, x_t):
            out, state = model.cell(x_t, state)
            return state, out

        _, logits_seq = jax.lax.scan(step, state, drives_t)
        return logits_seq[-1]

    out = {"ablate_none": _accuracy(logits_from_feats(feats), labels)}
    for which in ("low", "high"):
        out[f"ablate_{which}"] = _accuracy(
            logits_from_feats(_mask_band_half(feats, which=which)), labels
        )
    # Drop = how much that half mattered (higher ⇒ more necessary)
    out["ablate_low_drop"] = out["ablate_none"] - out["ablate_low"]
    out["ablate_high_drop"] = out["ablate_none"] - out["ablate_high"]
    return out


def horn_routing_metrics(model: DigitsHornClassifier) -> Dict[str, Any]:
    """Map hidden units to preferred RFB bands; score high→low coupling flow.

    Uses ``y = W @ x`` convention:
    ``W[i, j]`` = influence of unit ``j`` on unit ``i``.
    High→low score = mean |W[low_pref, high_pref]| / mean |W[high_pref, low_pref]|.
    """

    i2h_w = np.asarray(model.cell.i2h.weight)  # (hidden, input)
    preferred = np.argmax(np.abs(i2h_w), axis=1)  # (hidden,)
    mid = float(np.median(preferred))
    high_pref = preferred >= mid
    low_pref = ~high_pref
    n_high = int(np.sum(high_pref))
    n_low = int(np.sum(low_pref))

    h2h = model.cell.h2h
    if hasattr(h2h, "effective_coupling"):
        W = np.asarray(h2h.effective_coupling())
    else:
        W = np.asarray(h2h.strength * h2h.coupling_matrix)
    abs_w = np.abs(W)
    # from high-pref (cols) → low-pref (rows)
    if n_high > 0 and n_low > 0:
        high_to_low = float(np.mean(abs_w[np.ix_(low_pref, high_pref)]))
        low_to_high = float(np.mean(abs_w[np.ix_(high_pref, low_pref)]))
        ratio = high_to_low / max(low_to_high, 1e-12)
    else:
        high_to_low = 0.0
        low_to_high = 0.0
        ratio = 1.0

    return {
        "n_high_pref_units": n_high,
        "n_low_pref_units": n_low,
        "preferred_band_mean": float(np.mean(preferred)),
        "preferred_band_std": float(np.std(preferred)),
        "coupling_high_to_low": high_to_low,
        "coupling_low_to_high": low_to_high,
        "coupling_high_to_low_ratio": ratio,
        "horn_coupling_strength": float(h2h.strength),
        "horn_coupling_fixed_structure": bool(
            getattr(h2h, "fixed_structure", False)
        ),
        "horn_coupling_kind": str(
            getattr(model.cell, "coupling_kind", "fractal_fixed")
        ),
    }


def stage5_diagnostics(
    model: eqx.Module,
    waveforms: Array,
    labels: Array,
) -> Dict[str, Any]:
    """Run Stage 5 mechanistic diagnostics when applicable."""

    out: Dict[str, Any] = {}
    out.update(band_ablation_accuracies(model, waveforms, labels))
    if isinstance(model, DigitsHornClassifier):
        out.update(horn_routing_metrics(model))
    return out
