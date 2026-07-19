"""Adapters for phase-flow / rate-phase spatial field traces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from ..schema import GridShape, TraceBundle
from .base import Array


class PhaseFlowTraceAdapter:
    """Normalize phase-flow traces with spatial ``(T, B, H, W, C)`` trajectories."""

    name = "phase_flow"

    def matches(self, arrays: Mapping[str, Array]) -> bool:
        if "theta_trajectory" not in arrays:
            return False
        theta = np.asarray(arrays["theta_trajectory"])
        return theta.ndim == 5

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
        height, width = int(theta.shape[2]), int(theta.shape[3])
        if grid_shape is not None and grid_shape != (height, width):
            raise ValueError(
                f"grid_shape {grid_shape} does not match trajectory spatial "
                f"shape {(height, width)}"
            )
        readout = raw.get("predicted_clean")
        if readout is None:
            readout = raw.get("input")
        return TraceBundle(
            family=self.name,
            source=Path(source),
            arrays=raw,
            phase_trajectory=theta,
            rate_trajectory=raw.get("rate_trajectory"),
            readout=readout,
            grid_shape=(height, width),
            meta={
                "n_steps": int(theta.shape[0]),
                "batch_size": int(theta.shape[1]),
                "n_channels": int(theta.shape[-1]),
                "has_velocity_field": "velocity" in raw,
            },
        )
