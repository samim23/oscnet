"""Layered oscillator topology specifications.

These small dataclasses describe multi-population oscillator fields without
committing to a specific model implementation. They are intentionally shared
between research generators and future dynamical-system modules so layered
oscillator stacks do not become one-off experiment wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .coupling import (
    CouplingNormalization,
    coupling_profile_from_name,
    rectangular_coupling_profile_from_name,
)


@dataclass(frozen=True)
class OscillatorLayerSpec:
    """Specification for one oscillator population in a layered field."""

    name: str
    num_oscillators: int
    frequency_scale: float = 1.0
    coupling_profile: str = "local_radius"
    coupling_normalization: CouplingNormalization = "row_sum"
    coupling_length_scale: float = 0.0
    coupling_floor: float = 0.0
    recurrent_strength: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if self.num_oscillators < 1:
            raise ValueError("num_oscillators must be positive")
        if self.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be positive")
        if self.coupling_profile not in ("dense", "distance_decay", "local_radius"):
            raise ValueError(
                "coupling_profile must be 'dense', 'distance_decay', or "
                "'local_radius'"
            )
        if self.coupling_normalization not in ("none", "row_sum"):
            raise ValueError("coupling_normalization must be 'none' or 'row_sum'")
        if self.coupling_floor < 0.0 or self.coupling_floor > 1.0:
            raise ValueError("coupling_floor must be in [0, 1]")
        if self.recurrent_strength < 0.0:
            raise ValueError("recurrent_strength must be non-negative")


@dataclass(frozen=True)
class InterLayerCouplingSpec:
    """Specification for a directed source-to-target oscillator projection."""

    source_layer: int
    target_layer: int
    strength: float = 1.0
    profile: str = "local_radius"
    normalization: CouplingNormalization = "row_sum"
    length_scale: float = 0.0
    floor: float = 0.0
    phase_lag: float = 0.0

    def __post_init__(self) -> None:
        if self.source_layer == self.target_layer:
            raise ValueError("source_layer and target_layer must differ")
        if self.source_layer < 0 or self.target_layer < 0:
            raise ValueError("layer indices must be non-negative")
        if self.strength < 0.0:
            raise ValueError("strength must be non-negative")
        if self.profile not in ("dense", "distance_decay", "local_radius"):
            raise ValueError(
                "profile must be 'dense', 'distance_decay', or 'local_radius'"
            )
        if self.normalization not in ("none", "row_sum"):
            raise ValueError("normalization must be 'none' or 'row_sum'")
        if self.floor < 0.0 or self.floor > 1.0:
            raise ValueError("floor must be in [0, 1]")


def validate_layer_specs(
    layers: Tuple[OscillatorLayerSpec, ...],
) -> Tuple[OscillatorLayerSpec, ...]:
    """Validate and return a tuple of layer specs."""

    if not layers:
        raise ValueError("at least one oscillator layer is required")
    names = [layer.name for layer in layers]
    if len(names) != len(set(names)):
        raise ValueError("layer names must be unique")
    return tuple(layers)


def adjacent_inter_layer_specs(
    *,
    num_layers: int,
    forward_strength: float,
    feedback_strength: float = 0.0,
    profile: str = "local_radius",
    normalization: CouplingNormalization = "row_sum",
    length_scale: float = 0.0,
    floor: float = 0.0,
    forward_phase_lag: float = 0.0,
    feedback_phase_lag: float = 0.0,
) -> Tuple[InterLayerCouplingSpec, ...]:
    """Create nearest-neighbor vertical specs for a layered oscillator stack."""

    if num_layers < 2:
        return ()
    specs = []
    for layer_index in range(num_layers - 1):
        specs.append(
            InterLayerCouplingSpec(
                source_layer=layer_index,
                target_layer=layer_index + 1,
                strength=forward_strength,
                profile=profile,
                normalization=normalization,
                length_scale=length_scale,
                floor=floor,
                phase_lag=forward_phase_lag,
            )
        )
        if feedback_strength > 0.0:
            specs.append(
                InterLayerCouplingSpec(
                    source_layer=layer_index + 1,
                    target_layer=layer_index,
                    strength=feedback_strength,
                    profile=profile,
                    normalization=normalization,
                    length_scale=length_scale,
                    floor=floor,
                    phase_lag=feedback_phase_lag,
                )
            )
    return tuple(specs)


def intra_layer_profile(layer: OscillatorLayerSpec):
    """Build the fixed recurrent profile for one layer spec."""

    return coupling_profile_from_name(
        name=layer.coupling_profile,
        num_oscillators=layer.num_oscillators,
        length_scale=layer.coupling_length_scale,
        floor=layer.coupling_floor,
        normalization=layer.coupling_normalization,
        target_row_sum=float(layer.num_oscillators),
    )


def inter_layer_profile(
    spec: InterLayerCouplingSpec,
    layers: Tuple[OscillatorLayerSpec, ...],
):
    """Build the fixed source-to-target profile for one inter-layer spec."""

    validate_layer_specs(layers)
    if spec.source_layer >= len(layers) or spec.target_layer >= len(layers):
        raise ValueError("inter-layer spec references a missing layer")
    source = layers[spec.source_layer]
    target = layers[spec.target_layer]
    return rectangular_coupling_profile_from_name(
        name=spec.profile,
        num_targets=target.num_oscillators,
        num_sources=source.num_oscillators,
        length_scale=spec.length_scale,
        floor=spec.floor,
        normalization=spec.normalization,
        target_row_sum=float(source.num_oscillators),
    )


__all__ = [
    "InterLayerCouplingSpec",
    "OscillatorLayerSpec",
    "adjacent_inter_layer_specs",
    "inter_layer_profile",
    "intra_layer_profile",
    "validate_layer_specs",
]
