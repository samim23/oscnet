"""Adapters for HORN / Kuramoto image-generator traces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..geometry import infer_grid_shape
from ..schema import GridShape, TraceBundle
from .base import Array

_GENERATOR_MARKERS = {
    "theta_trajectory",
    "generated",
    "coupling",
}


class GeneratorTraceAdapter:
    """Normalize ``mnist_generator_trace_*.npz`` style dumps."""

    name = "generator"

    def matches(self, arrays: Mapping[str, Array]) -> bool:
        keys = set(arrays)
        if not _GENERATOR_MARKERS.issubset(keys):
            return False
        # Distinguish from phase-flow, which also has theta_trajectory but is spatial.
        theta = np.asarray(arrays["theta_trajectory"])
        return theta.ndim == 3

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
        theta = raw["theta_trajectory"]
        n_sites = int(theta.shape[-1])
        resolved = infer_grid_shape(n_sites, hint=grid_shape)
        has_velocity = "velocity_trajectory" in raw and raw["velocity_trajectory"].size > 0
        return TraceBundle(
            family=self.name,
            source=Path(source),
            arrays=raw,
            phase_trajectory=theta,
            velocity_trajectory=raw["velocity_trajectory"] if has_velocity else None,
            omega=raw.get("omega"),
            coupling=raw.get("coupling"),
            coupling_profile=raw.get("coupling_profile"),
            readout=raw.get("generated"),
            vertical_gain_trajectory=_optional_trajectory(
                raw, "fine_vertical_gain_trajectory"
            ),
            grid_shape=resolved,
            meta={
                "n_steps": int(theta.shape[0]),
                "batch_size": int(theta.shape[1]),
                "n_sites": n_sites,
                "has_velocity": has_velocity,
                "has_condition_bank": "condition_final_theta" in raw,
            },
        )


def _optional_trajectory(raw: Mapping[str, Array], key: str) -> Optional[Array]:
    if key not in raw:
        return None
    value = np.asarray(raw[key])
    if value.size == 0:
        return None
    return value
