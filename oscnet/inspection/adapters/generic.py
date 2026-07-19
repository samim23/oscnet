"""Fallback adapter for unrecognized but partially structured NPZ traces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..geometry import infer_grid_shape
from ..schema import GridShape, TraceBundle
from .base import Array


class GenericTraceAdapter:
    """Best-effort normalization when no specialized family matches."""

    name = "generic"

    def matches(self, arrays: Mapping[str, Array]) -> bool:
        return len(arrays) > 0

    def adapt(
        self,
        arrays: Mapping[str, Array],
        *,
        source: Path,
        grid_shape: Optional[GridShape] = None,
        **kwargs,
    ) -> TraceBundle:
        del kwargs
        raw = {key: np.asarray(value) for key, value in arrays.items()}
        phase = _first_present(
            raw,
            (
                "theta_trajectory",
                "decoder_thetas",
                "encoder_thetas",
                "thetas",
            ),
        )
        resolved = None
        if phase is not None and phase.ndim >= 3:
            n_sites = int(phase.shape[-1] if phase.ndim == 3 else phase.shape[-2])
            try:
                resolved = infer_grid_shape(n_sites, hint=grid_shape)
            except ValueError:
                resolved = None
        return TraceBundle(
            family=self.name,
            source=Path(source),
            arrays=raw,
            phase_trajectory=phase,
            velocity_trajectory=_first_present(raw, ("velocity_trajectory",)),
            rate_trajectory=_first_present(raw, ("rate_trajectory",)),
            energies=_first_present(raw, ("energies", "decoder_energies", "encoder_energies")),
            omega=_first_present(raw, ("omega", "decoder_omega", "encoder_omega")),
            coupling=_first_present(raw, ("coupling",)),
            coupling_profile=_first_present(raw, ("coupling_profile",)),
            readout=_first_present(
                raw,
                ("generated", "predicted_clean", "reconstruction_sequence", "reconstructions"),
            ),
            grid_shape=resolved,
            meta={"fallback": True, "n_arrays": len(raw)},
        )


def _first_present(raw: Mapping[str, Array], keys: tuple[str, ...]) -> Optional[Array]:
    for key in keys:
        if key in raw and np.asarray(raw[key]).size > 0:
            return np.asarray(raw[key])
    return None
