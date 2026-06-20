"""Shared type aliases for analysis and dynamics helpers."""

from typing import Any

import jax.numpy as jnp

Array = jnp.ndarray
StateType = Array
TimeType = Array
ArgsType = Any


__all__ = ["Array", "StateType", "TimeType", "ArgsType"]
