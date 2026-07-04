"""Multimode HORN image generator."""

from __future__ import annotations

from oscnet.core.coupling import (
    coupling_profile_from_name,
    normalize_coupling_profile,
)

from .common import Array, Tuple, eqx, jnp
from .horn import HORNImageGenerator


class MultiModeHORNImageGenerator(HORNImageGenerator):
    """HORN generator with several frequency modes per spatial site.

    This is a structured capacity variant. Instead of increasing the number of
    spatial oscillator sites, the model creates a small bank of HORN modes at
    each site. Same-frequency modes couple across nearby spatial sites, while
    different modes couple only within the same site. That gives the field more
    local spectral degrees of freedom without treating the whole state as one
    flat 512-node spatial grid.
    """

    num_spatial_sites: int = eqx.field(static=True)
    num_modes: int = eqx.field(static=True)
    mode_frequency_scales: Tuple[float, ...] = eqx.field(static=True)
    mode_coupling_strength: float = eqx.field(static=True)
    mode_coupling_profile: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_modes: int = 2,
        mode_frequency_scales: Tuple[float, ...] = (),
        mode_coupling_strength: float = 0.25,
        mode_coupling_profile: str = "dense",
        **kwargs,
    ):
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        if mode_coupling_strength < 0.0 or mode_coupling_strength > 1.0:
            raise ValueError("mode_coupling_strength must be in [0, 1]")
        if mode_coupling_profile not in ("dense", "adjacent"):
            raise ValueError("mode_coupling_profile must be 'dense' or 'adjacent'")

        num_spatial_sites = int(kwargs.get("num_oscillators", 64))
        if num_spatial_sites < 1:
            raise ValueError("num_oscillators must be positive")
        if not mode_frequency_scales:
            if num_modes == 1:
                mode_frequency_scales = (1.0,)
            else:
                span = 1.35 - 0.75
                mode_frequency_scales = tuple(
                    0.75 + span * float(index) / float(int(num_modes) - 1)
                    for index in range(int(num_modes))
                )
        if len(mode_frequency_scales) != int(num_modes):
            raise ValueError("mode_frequency_scales must have length num_modes")
        if any(scale <= 0.0 for scale in mode_frequency_scales):
            raise ValueError("mode_frequency_scales must be positive")

        kwargs["num_oscillators"] = num_spatial_sites * int(num_modes)
        kwargs.setdefault("state_anchor_num_spatial_sites", num_spatial_sites)
        kwargs.setdefault("state_anchor_num_modes", int(num_modes))
        super().__init__(**kwargs)
        self.num_spatial_sites = num_spatial_sites
        self.num_modes = int(num_modes)
        self.mode_frequency_scales = tuple(float(v) for v in mode_frequency_scales)
        self.mode_coupling_strength = float(mode_coupling_strength)
        self.mode_coupling_profile = mode_coupling_profile
        self.dynamics_family = "multimode_horn"

    def _mode_profile_matrix(self, *, target_row_sum: float) -> Array:
        """Return fixed within-site coupling among frequency modes."""

        if (
            self.num_modes == 1
            or self.mode_coupling_strength == 0.0
            or target_row_sum <= 0.0
        ):
            return jnp.zeros((self.num_modes, self.num_modes), dtype=jnp.float32)
        mode_index = jnp.arange(self.num_modes)
        if self.mode_coupling_profile == "adjacent":
            profile = (jnp.abs(mode_index[:, None] - mode_index[None, :]) == 1).astype(
                jnp.float32
            )
        else:
            profile = jnp.ones((self.num_modes, self.num_modes), dtype=jnp.float32)
            profile = profile - jnp.eye(self.num_modes, dtype=jnp.float32)
        return normalize_coupling_profile(
            profile,
            mode="row_sum",
            target_row_sum=float(target_row_sum),
        )

    def coupling_profile_matrix(self) -> Array:
        """Return spatial same-mode coupling plus within-site mode coupling."""

        mode_row_sum = float(self.num_oscillators) * float(
            self.mode_coupling_strength
        )
        spatial_row_sum = float(self.num_oscillators) - mode_row_sum
        spatial_profile = coupling_profile_from_name(
            name=self.coupling_profile,
            num_oscillators=self.num_spatial_sites,
            length_scale=self.coupling_length_scale,
            floor=self.coupling_floor,
            normalization=self.coupling_normalization,
            target_row_sum=spatial_row_sum,
        )
        same_mode_profile = jnp.kron(
            spatial_profile,
            jnp.eye(self.num_modes, dtype=jnp.float32),
        )
        mode_profile = jnp.kron(
            jnp.eye(self.num_spatial_sites, dtype=jnp.float32),
            self._mode_profile_matrix(target_row_sum=mode_row_sum),
        )
        return same_mode_profile + mode_profile

    def _horn_frequency(self, omega: Array) -> Array:
        frequency = super()._horn_frequency(omega)
        if int(omega.shape[-1]) != int(self.num_oscillators):
            return frequency
        mode_scales = jnp.tile(
            jnp.asarray(self.mode_frequency_scales, dtype=frequency.dtype),
            self.num_spatial_sites,
        )
        return frequency * mode_scales
