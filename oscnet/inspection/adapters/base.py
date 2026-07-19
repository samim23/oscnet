"""Adapter protocol for mapping raw NPZ dicts into TraceBundle."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from ..schema import GridShape, TraceBundle

Array = np.ndarray
ArrayMap = Dict[str, Array]


def load_npz_arrays(path: Path | str) -> ArrayMap:
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


@runtime_checkable
class TraceAdapter(Protocol):
    """Detect and normalize one family of oscillatory traces."""

    name: str

    def matches(self, arrays: Mapping[str, Array]) -> bool:
        """Return True when this adapter should own the NPZ key set."""

    def adapt(
        self,
        arrays: Mapping[str, Array],
        *,
        source: Path,
        grid_shape: Optional[GridShape] = None,
        **kwargs,
    ) -> TraceBundle:
        """Build a normalized bundle from raw arrays."""
