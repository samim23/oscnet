"""Normalized schema for oscillatory trace inspection.

Adapters map family-specific NPZ layouts into :class:`TraceBundle`. Views
consume only the bundle (plus optional raw extras), so new model families grow
by adding an adapter rather than rewriting plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
GridShape = Tuple[int, int]


@dataclass(frozen=True)
class ArrayRef:
    """A named array slot with light semantic tags."""

    name: str
    data: Array
    kind: str = "other"
    """Semantic tag: phase | velocity | rate | energy | omega | coupling | readout | gain | other."""

    layout: str = "array"
    """Structural hint: trajectory | batch_sites | grid | matrix | vector | scalar_series."""

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "layout": self.layout,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
        }


@dataclass
class TraceBundle:
    """Family-agnostic view of a saved oscillatory trace.

    Required identity fields always exist. Optional slots are ``None`` when the
    source family does not provide them. Raw arrays remain available under
    :attr:`arrays` for family-specific extensions.
    """

    family: str
    source: Path
    arrays: Dict[str, Array]
    phase_trajectory: Optional[Array] = None
    """Phase / position over settle steps. Leading axis is time when present."""

    velocity_trajectory: Optional[Array] = None
    rate_trajectory: Optional[Array] = None
    energies: Optional[Array] = None
    omega: Optional[Array] = None
    coupling: Optional[Array] = None
    coupling_profile: Optional[Array] = None
    readout: Optional[Array] = None
    """Decoded samples / reconstructions when present, batch-major."""

    vertical_gain_trajectory: Optional[Array] = None
    """Coarse→fine gain maps over settle steps (multiscale), if present."""

    grid_shape: Optional[GridShape] = None
    """Preferred spatial reshape for flat site axes, when known or inferred."""

    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self.arrays))

    def get(self, name: str) -> Optional[Array]:
        value = self.arrays.get(name)
        return None if value is None else np.asarray(value)

    def slots(self) -> Sequence[ArrayRef]:
        """Return filled normalized slots for overview / discovery UIs."""

        refs = []
        mapping: Mapping[str, Tuple[Optional[Array], str, str]] = {
            "phase_trajectory": (self.phase_trajectory, "phase", "trajectory"),
            "velocity_trajectory": (
                self.velocity_trajectory,
                "velocity",
                "trajectory",
            ),
            "rate_trajectory": (self.rate_trajectory, "rate", "trajectory"),
            "energies": (self.energies, "energy", "scalar_series"),
            "omega": (self.omega, "omega", "vector"),
            "coupling": (self.coupling, "coupling", "matrix"),
            "coupling_profile": (self.coupling_profile, "coupling", "matrix"),
            "readout": (self.readout, "readout", "batch_sites"),
            "vertical_gain_trajectory": (
                self.vertical_gain_trajectory,
                "gain",
                "trajectory",
            ),
        }
        for name, (data, kind, layout) in mapping.items():
            if data is None:
                continue
            refs.append(ArrayRef(name=name, data=np.asarray(data), kind=kind, layout=layout))
        return refs

    def overview(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "source": str(self.source),
            "raw_keys": list(self.keys),
            "grid_shape": list(self.grid_shape) if self.grid_shape is not None else None,
            "slots": [ref.summary() for ref in self.slots()],
            "meta": dict(self.meta),
        }
