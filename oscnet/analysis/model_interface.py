"""Lightweight adapter for analysis modules that inspect model parameters."""

from typing import Any, Dict

import jax.numpy as jnp

ParamDict = Dict[str, Any]

_PARAMETER_NAMES = {
    "alpha",
    "omega",
    "gamma",
    "mu",
    "dt",
    "h",
    "beta",
    "a",
    "b",
    "tau",
    "base_omega",
    "base_gamma",
}


class OscillatoryModelInterface:
    """Minimal interface consumed by analysis helpers."""

    def extract_parameters(self) -> ParamDict:
        raise NotImplementedError


class SimpleModelInterface(OscillatoryModelInterface):
    """Best-effort parameter extractor for Equinox/JAX model objects."""

    def __init__(self, model: Any):
        self.model = model

    def extract_parameters(self) -> ParamDict:
        params: ParamDict = {}
        self._collect(self.model, params, prefix="", seen=set())
        return params

    def _collect(self, obj: Any, params: ParamDict, prefix: str, seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                self._collect(item, params, f"{prefix}{idx}.", seen)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._maybe_add(str(key), value, params, prefix)
                self._collect(value, params, f"{prefix}{key}.", seen)
            return

        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if callable(value):
                continue
            self._maybe_add(name, value, params, prefix)
            if not isinstance(value, (str, bytes, int, float, bool, type(None))):
                self._collect(value, params, f"{prefix}{name}.", seen)

    def _maybe_add(self, name: str, value: Any, params: ParamDict, prefix: str) -> None:
        if name not in _PARAMETER_NAMES:
            return
        if isinstance(value, (int, float)):
            clean_value = float(value)
        else:
            try:
                clean_value = jnp.asarray(value)
            except Exception:
                clean_value = value

        params.setdefault(name, clean_value)
        if prefix:
            params[f"{prefix}{name}"] = clean_value


def adapt_model(model: Any) -> OscillatoryModelInterface:
    if isinstance(model, OscillatoryModelInterface):
        return model
    return SimpleModelInterface(model)


__all__ = [
    "ParamDict",
    "OscillatoryModelInterface",
    "SimpleModelInterface",
    "adapt_model",
]
