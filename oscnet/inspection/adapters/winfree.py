"""Adapters for Winfree field autoencoder / denoiser traces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..geometry import infer_grid_shape
from ..schema import GridShape, TraceBundle
from .base import Array


class WinfreeTraceAdapter:
    """Normalize Winfree field traces with ``{prefix}_thetas`` trajectories."""

    name = "winfree"

    def matches(self, arrays: Mapping[str, Array]) -> bool:
        keys = set(arrays)
        return (
            "decoder_thetas" in keys
            or "encoder_thetas" in keys
            or any(key.endswith("_thetas") for key in keys)
        )

    def adapt(
        self,
        arrays: Mapping[str, Array],
        *,
        source: Path,
        grid_shape: Optional[GridShape] = None,
        prefix: str = "decoder",
        **kwargs,
    ) -> TraceBundle:
        del kwargs
        raw = {key: np.asarray(value) for key, value in arrays.items()}
        prefix = _resolve_prefix(raw, prefix)
        thetas = raw[f"{prefix}_thetas"]
        if thetas.ndim != 4:
            raise ValueError(
                f"expected {prefix}_thetas with shape (steps, batch, positions, channels); "
                f"got {thetas.shape}"
            )
        n_sites = int(thetas.shape[-2])
        resolved = infer_grid_shape(n_sites, hint=grid_shape)
        energies = raw.get(f"{prefix}_energies")
        omega = raw.get(f"{prefix}_omega")
        readout = raw.get("reconstruction_sequence")
        if readout is None:
            readout = raw.get("latent")
        return TraceBundle(
            family=self.name,
            source=Path(source),
            arrays=raw,
            phase_trajectory=thetas,
            energies=energies,
            omega=omega,
            readout=readout,
            grid_shape=resolved,
            meta={
                "prefix": prefix,
                "n_steps": int(thetas.shape[0]),
                "batch_size": int(thetas.shape[1]),
                "n_sites": n_sites,
                "n_channels": int(thetas.shape[-1]),
                "available_prefixes": _available_prefixes(raw),
            },
        )


def _available_prefixes(raw: Mapping[str, Array]) -> list[str]:
    prefixes = []
    for key in raw:
        if key.endswith("_thetas"):
            prefixes.append(key[: -len("_thetas")])
    return sorted(prefixes)


def _resolve_prefix(raw: Mapping[str, Array], preferred: str) -> str:
    if f"{preferred}_thetas" in raw:
        return preferred
    available = _available_prefixes(raw)
    if not available:
        raise ValueError("no *_thetas arrays found in Winfree trace")
    if "decoder" in available:
        return "decoder"
    return available[0]
