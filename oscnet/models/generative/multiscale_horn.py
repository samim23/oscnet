"""Multiscale layered HORN image generator."""

from __future__ import annotations

from oscnet.core.layered import (
    InterLayerCouplingSpec,
    OscillatorLayerSpec,
    adjacent_inter_layer_specs,
    inter_layer_profile,
    intra_layer_profile,
)

from .common import (
    Array,
    Dict,
    Optional,
    Tuple,
    _activation,
    _image_hw_channels,
    eqx,
    jax,
    jnp,
    math,
)
from .horn import HORNImageGenerator


class MultiscaleHORNImageGenerator(HORNImageGenerator):
    """HORN generator with multiple coupled oscillator populations.

    Auxiliary layers evolve alongside the decoded fine layer. The decoder still
    reads only the fine HORN field, while directed vertical couplings let slower
    coarse/mid fields coordinate it during settling. This generalizes the
    earlier single coarse-to-fine generator into a reusable layered-field model.
    """

    auxiliary_omega: Tuple[Array, ...]
    auxiliary_coupling: Tuple[Array, ...]
    vertical_coupling: Tuple[Array, ...]
    auxiliary_label_condition_coupling: Tuple[Array, ...]
    auxiliary_readout_weight: Array
    auxiliary_readout_bias: Array
    layer_specs: Tuple[OscillatorLayerSpec, ...] = eqx.field(static=True)
    vertical_specs: Tuple[InterLayerCouplingSpec, ...] = eqx.field(static=True)
    multiscale_layer_sizes: Tuple[int, ...] = eqx.field(static=True)
    multiscale_frequency_scales: Tuple[float, ...] = eqx.field(static=True)
    multiscale_coupling_profile: str = eqx.field(static=True)
    multiscale_coupling_normalization: str = eqx.field(static=True)
    multiscale_coupling_length_scale: float = eqx.field(static=True)
    multiscale_coupling_floor: float = eqx.field(static=True)
    multiscale_vertical_strength: float = eqx.field(static=True)
    multiscale_feedback_strength: float = eqx.field(static=True)
    multiscale_vertical_profile: str = eqx.field(static=True)
    multiscale_vertical_normalization: str = eqx.field(static=True)
    multiscale_vertical_length_scale: float = eqx.field(static=True)
    multiscale_vertical_floor: float = eqx.field(static=True)
    multiscale_vertical_phase_lag: float = eqx.field(static=True)
    multiscale_feedback_phase_lag: float = eqx.field(static=True)
    multiscale_vertical_signal_scale: float = eqx.field(static=True)
    multiscale_vertical_target_gate: str = eqx.field(static=True)
    multiscale_vertical_soft_gate_floor: float = eqx.field(static=True)
    multiscale_vertical_mode: str = eqx.field(static=True)
    multiscale_vertical_gain_target: str = eqx.field(static=True)
    multiscale_vertical_gain_normalization: str = eqx.field(static=True)
    multiscale_vertical_gain_target_std: float = eqx.field(static=True)
    multiscale_vertical_broad_gain_scale: float = eqx.field(static=True)
    multiscale_vertical_selective_gain_scale: float = eqx.field(static=True)
    multiscale_vertical_schedule: str = eqx.field(static=True)
    multiscale_vertical_onset_step: int = eqx.field(static=True)
    multiscale_vertical_ramp_steps: int = eqx.field(static=True)
    multiscale_vertical_intervention: str = eqx.field(static=True)
    multiscale_vertical_intervention_scale: float = eqx.field(static=True)
    multiscale_conditioning_strength: float = eqx.field(static=True)
    multiscale_auxiliary_readout_size: int = eqx.field(static=True)
    multiscale_auxiliary_readout_layer: int = eqx.field(static=True)
    num_auxiliary_layers: int = eqx.field(static=True)
    num_vertical_couplings: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        multiscale_layer_sizes: Tuple[int, ...] = (16, 64),
        multiscale_frequency_scales: Tuple[float, ...] = (),
        multiscale_coupling_profile: str = "local_radius",
        multiscale_coupling_normalization: str = "row_sum",
        multiscale_coupling_length_scale: float = 0.0,
        multiscale_coupling_floor: float = 0.0,
        multiscale_vertical_strength: float = 0.25,
        multiscale_feedback_strength: float = 0.0,
        multiscale_vertical_profile: str = "local_radius",
        multiscale_vertical_normalization: str = "row_sum",
        multiscale_vertical_length_scale: float = 0.0,
        multiscale_vertical_floor: float = 0.0,
        multiscale_vertical_phase_lag: float = 0.0,
        multiscale_feedback_phase_lag: float = 0.0,
        multiscale_vertical_signal_scale: float = 1.0,
        multiscale_vertical_target_gate: str = "all",
        multiscale_vertical_soft_gate_floor: float = 0.0,
        multiscale_vertical_mode: str = "additive",
        multiscale_vertical_gain_target: str = "drive",
        multiscale_vertical_gain_normalization: str = "none",
        multiscale_vertical_gain_target_std: float = 0.0,
        multiscale_vertical_broad_gain_scale: float = 1.0,
        multiscale_vertical_selective_gain_scale: float = 1.0,
        multiscale_vertical_schedule: str = "constant",
        multiscale_vertical_onset_step: int = 0,
        multiscale_vertical_ramp_steps: int = 0,
        multiscale_vertical_intervention: str = "normal",
        multiscale_vertical_intervention_scale: float = 1.0,
        multiscale_conditioning_strength: float = 1.0,
        multiscale_auxiliary_readout_size: int = 8,
        multiscale_auxiliary_readout_layer: int = 0,
        **kwargs,
    ):
        if not multiscale_layer_sizes:
            raise ValueError("multiscale_layer_sizes must include at least one layer")
        if any(size < 1 for size in multiscale_layer_sizes):
            raise ValueError("multiscale layer sizes must be positive")
        if multiscale_frequency_scales and len(multiscale_frequency_scales) != len(
            multiscale_layer_sizes
        ):
            raise ValueError(
                "multiscale_frequency_scales must be empty or match "
                "multiscale_layer_sizes"
            )
        if not multiscale_frequency_scales:
            denom = float(max(len(multiscale_layer_sizes) - 1, 1))
            multiscale_frequency_scales = tuple(
                0.5 + 0.3 * float(index) / denom
                for index in range(len(multiscale_layer_sizes))
            )
        if any(scale <= 0.0 for scale in multiscale_frequency_scales):
            raise ValueError("multiscale_frequency_scales must be positive")
        if multiscale_auxiliary_readout_size < 1:
            raise ValueError("multiscale_auxiliary_readout_size must be positive")
        if multiscale_auxiliary_readout_layer < 0:
            multiscale_auxiliary_readout_layer = (
                len(multiscale_layer_sizes) + multiscale_auxiliary_readout_layer
            )
        if (
            multiscale_auxiliary_readout_layer < 0
            or multiscale_auxiliary_readout_layer >= len(multiscale_layer_sizes)
        ):
            raise ValueError("multiscale_auxiliary_readout_layer out of range")
        if multiscale_vertical_target_gate not in (
            "all",
            "conditioning",
            "non_conditioning",
        ):
            raise ValueError(
                "multiscale_vertical_target_gate must be 'all', "
                "'conditioning', or 'non_conditioning'"
            )
        if not 0.0 <= multiscale_vertical_soft_gate_floor <= 1.0:
            raise ValueError(
                "multiscale_vertical_soft_gate_floor must be between 0 and 1"
            )
        if multiscale_vertical_mode not in (
            "additive",
            "gain_modulation",
            "signed_gain",
            "dual_gain",
        ):
            raise ValueError(
                "multiscale_vertical_mode must be 'additive', "
                "'gain_modulation', 'signed_gain', or 'dual_gain'"
            )
        if multiscale_vertical_gain_target not in (
            "drive",
            "coupling",
            "conditioning",
            "damping",
        ):
            raise ValueError(
                "multiscale_vertical_gain_target must be 'drive', "
                "'coupling', 'conditioning', or 'damping'"
            )
        if multiscale_vertical_gain_normalization not in (
            "none",
            "center",
            "center_rms",
        ):
            raise ValueError(
                "multiscale_vertical_gain_normalization must be 'none', "
                "'center', or 'center_rms'"
            )
        if multiscale_vertical_schedule not in (
            "constant",
            "delayed",
            "linear_ramp",
        ):
            raise ValueError(
                "multiscale_vertical_schedule must be 'constant', "
                "'delayed', or 'linear_ramp'"
            )
        if multiscale_vertical_intervention not in (
            "normal",
            "zero",
            "shuffle_batch",
            "flip",
        ):
            raise ValueError(
                "multiscale_vertical_intervention must be 'normal', 'zero', "
                "'shuffle_batch', or 'flip'"
            )
        if multiscale_vertical_onset_step < 0:
            raise ValueError("multiscale_vertical_onset_step must be non-negative")
        if multiscale_vertical_ramp_steps < 0:
            raise ValueError("multiscale_vertical_ramp_steps must be non-negative")
        if multiscale_vertical_intervention_scale < 0.0:
            raise ValueError("multiscale_vertical_intervention_scale must be non-negative")
        for name, value in (
            ("multiscale_vertical_strength", multiscale_vertical_strength),
            ("multiscale_feedback_strength", multiscale_feedback_strength),
            ("multiscale_vertical_signal_scale", multiscale_vertical_signal_scale),
            (
                "multiscale_vertical_broad_gain_scale",
                multiscale_vertical_broad_gain_scale,
            ),
            (
                "multiscale_vertical_selective_gain_scale",
                multiscale_vertical_selective_gain_scale,
            ),
            (
                "multiscale_vertical_gain_target_std",
                multiscale_vertical_gain_target_std,
            ),
            ("multiscale_conditioning_strength", multiscale_conditioning_strength),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        split_count = (
            2
            + len(multiscale_layer_sizes) * 3
            + max(1, 2 * (len(multiscale_layer_sizes)))
        )
        keys = jax.random.split(key, split_count)
        kwargs["key"] = keys[0]
        key_index = 1
        omega_keys = keys[key_index : key_index + len(multiscale_layer_sizes)]
        key_index += len(multiscale_layer_sizes)
        coupling_keys = keys[key_index : key_index + len(multiscale_layer_sizes)]
        key_index += len(multiscale_layer_sizes)
        condition_keys = keys[key_index : key_index + len(multiscale_layer_sizes)]
        key_index += len(multiscale_layer_sizes)

        omega_scale = float(kwargs.get("omega_scale", 0.2))
        coupling_init_scale = float(kwargs.get("coupling_init_scale", 0.05))

        super().__init__(**kwargs)
        if self.conditioning_mode == "class_oscillator":
            raise ValueError(
                "MultiscaleHORNImageGenerator currently supports "
                "'none', 'phase_shift', and 'class_coupling' conditioning"
            )

        self.multiscale_layer_sizes = tuple(int(size) for size in multiscale_layer_sizes)
        self.multiscale_frequency_scales = tuple(
            float(scale) for scale in multiscale_frequency_scales
        )
        self.multiscale_coupling_profile = multiscale_coupling_profile
        self.multiscale_coupling_normalization = multiscale_coupling_normalization
        self.multiscale_coupling_length_scale = float(multiscale_coupling_length_scale)
        self.multiscale_coupling_floor = float(multiscale_coupling_floor)
        self.multiscale_vertical_strength = float(multiscale_vertical_strength)
        self.multiscale_feedback_strength = float(multiscale_feedback_strength)
        self.multiscale_vertical_profile = multiscale_vertical_profile
        self.multiscale_vertical_normalization = multiscale_vertical_normalization
        self.multiscale_vertical_length_scale = float(multiscale_vertical_length_scale)
        self.multiscale_vertical_floor = float(multiscale_vertical_floor)
        self.multiscale_vertical_phase_lag = float(multiscale_vertical_phase_lag)
        self.multiscale_feedback_phase_lag = float(multiscale_feedback_phase_lag)
        self.multiscale_vertical_signal_scale = float(
            multiscale_vertical_signal_scale
        )
        self.multiscale_vertical_target_gate = multiscale_vertical_target_gate
        self.multiscale_vertical_soft_gate_floor = float(
            multiscale_vertical_soft_gate_floor
        )
        self.multiscale_vertical_mode = multiscale_vertical_mode
        self.multiscale_vertical_gain_target = multiscale_vertical_gain_target
        self.multiscale_vertical_gain_normalization = (
            multiscale_vertical_gain_normalization
        )
        self.multiscale_vertical_gain_target_std = float(
            multiscale_vertical_gain_target_std
        )
        self.multiscale_vertical_broad_gain_scale = float(
            multiscale_vertical_broad_gain_scale
        )
        self.multiscale_vertical_selective_gain_scale = float(
            multiscale_vertical_selective_gain_scale
        )
        self.multiscale_vertical_schedule = multiscale_vertical_schedule
        self.multiscale_vertical_onset_step = int(multiscale_vertical_onset_step)
        self.multiscale_vertical_ramp_steps = int(multiscale_vertical_ramp_steps)
        self.multiscale_vertical_intervention = multiscale_vertical_intervention
        self.multiscale_vertical_intervention_scale = float(
            multiscale_vertical_intervention_scale
        )
        self.multiscale_conditioning_strength = float(multiscale_conditioning_strength)
        self.multiscale_auxiliary_readout_size = int(multiscale_auxiliary_readout_size)
        self.multiscale_auxiliary_readout_layer = int(multiscale_auxiliary_readout_layer)
        self.num_auxiliary_layers = len(self.multiscale_layer_sizes)
        self.dynamics_family = "multiscale_horn"

        auxiliary_specs = tuple(
            OscillatorLayerSpec(
                name=f"aux_{index}",
                num_oscillators=size,
                frequency_scale=self.multiscale_frequency_scales[index],
                coupling_profile=self.multiscale_coupling_profile,
                coupling_normalization=self.multiscale_coupling_normalization,
                coupling_length_scale=self.multiscale_coupling_length_scale,
                coupling_floor=self.multiscale_coupling_floor,
            )
            for index, size in enumerate(self.multiscale_layer_sizes)
        )
        fine_spec = OscillatorLayerSpec(
            name="fine",
            num_oscillators=self.num_oscillators,
            frequency_scale=1.0,
            coupling_profile=self.coupling_profile,
            coupling_normalization=self.coupling_normalization,
            coupling_length_scale=self.coupling_length_scale,
            coupling_floor=self.coupling_floor,
        )
        self.layer_specs = (*auxiliary_specs, fine_spec)
        self.vertical_specs = adjacent_inter_layer_specs(
            num_layers=len(self.layer_specs),
            forward_strength=self.multiscale_vertical_strength,
            feedback_strength=self.multiscale_feedback_strength,
            profile=self.multiscale_vertical_profile,
            normalization=self.multiscale_vertical_normalization,
            length_scale=self.multiscale_vertical_length_scale,
            floor=self.multiscale_vertical_floor,
            forward_phase_lag=self.multiscale_vertical_phase_lag,
            feedback_phase_lag=self.multiscale_feedback_phase_lag,
        )
        self.num_vertical_couplings = len(self.vertical_specs)

        self.auxiliary_omega = tuple(
            jax.random.normal(omega_key, (size,)) * omega_scale
            for omega_key, size in zip(omega_keys, self.multiscale_layer_sizes)
        )
        self.auxiliary_coupling = tuple(
            (
                jax.random.normal(coupling_key, (size, size))
                * coupling_init_scale
                / jnp.sqrt(float(size))
            )
            * (1.0 - jnp.eye(size, dtype=jnp.float32))
            for coupling_key, size in zip(coupling_keys, self.multiscale_layer_sizes)
        )

        vertical_keys = keys[key_index : key_index + len(self.vertical_specs)]
        self.vertical_coupling = tuple(
            jax.random.normal(
                vertical_key,
                (
                    self.layer_specs[spec.target_layer].num_oscillators,
                    self.layer_specs[spec.source_layer].num_oscillators,
                ),
            )
            * coupling_init_scale
            / jnp.sqrt(float(self.layer_specs[spec.source_layer].num_oscillators))
            for vertical_key, spec in zip(vertical_keys, self.vertical_specs)
        )
        readout_key = keys[key_index + len(self.vertical_specs)]
        _, _, channels = _image_hw_channels(self.image_shape)
        readout_layer_size = self.multiscale_layer_sizes[
            self.multiscale_auxiliary_readout_layer
        ]
        readout_features = 2 * int(readout_layer_size)
        readout_dim = (
            int(self.multiscale_auxiliary_readout_size)
            * int(self.multiscale_auxiliary_readout_size)
            * int(channels)
        )
        self.auxiliary_readout_weight = (
            jax.random.normal(readout_key, (readout_features, readout_dim))
            * coupling_init_scale
            / jnp.sqrt(float(max(readout_features, 1)))
        )
        aux_bias_value = kwargs.get("output_bias_init", None)
        if aux_bias_value is None:
            self.auxiliary_readout_bias = jnp.zeros((readout_dim,), dtype=jnp.float32)
        else:
            self.auxiliary_readout_bias = jnp.full(
                (readout_dim,),
                float(aux_bias_value),
                dtype=jnp.float32,
            )

        if (
            self.num_classes > 0
            and self.conditioning_mode == "class_coupling"
            and self.num_condition_oscillators > 0
        ):
            self.auxiliary_label_condition_coupling = tuple(
                jax.random.normal(
                    condition_key,
                    (self.num_classes, size, self.num_condition_oscillators),
                )
                * coupling_init_scale
                / jnp.sqrt(float(max(self.num_condition_oscillators, 1)))
                for condition_key, size in zip(
                    condition_keys,
                    self.multiscale_layer_sizes,
                )
            )
        else:
            self.auxiliary_label_condition_coupling = ()

    def auxiliary_state_features(self, position: Array, velocity: Array) -> Array:
        """Return bounded features for an auxiliary HORN layer."""

        if self.readout_mode in ("relative", "ref_oscillator"):
            position = position - position[:, :1]
            velocity = velocity - velocity[:, :1]
        elif self.readout_mode == "mean_relative":
            position = position - jnp.mean(position, axis=-1, keepdims=True)
            velocity = velocity - jnp.mean(velocity, axis=-1, keepdims=True)
        return jnp.concatenate([jnp.tanh(position), jnp.tanh(velocity)], axis=-1)

    def decode_auxiliary_state(self, position: Array, velocity: Array) -> Array:
        """Decode the selected auxiliary layer into a low-resolution image."""

        features = self.auxiliary_state_features(position, velocity)
        pixels = features @ self.auxiliary_readout_weight + self.auxiliary_readout_bias
        return _activation(self.output_activation)(pixels)

    def initial_auxiliary_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[Tuple[Array, ...], Tuple[Array, ...]]:
        """Sample random auxiliary HORN states."""

        keys = jax.random.split(key, 2 * len(self.multiscale_layer_sizes))
        position_keys = keys[: len(self.multiscale_layer_sizes)]
        velocity_keys = keys[len(self.multiscale_layer_sizes) :]
        positions = []
        velocities = []
        for size, position_key, velocity_key in zip(
            self.multiscale_layer_sizes,
            position_keys,
            velocity_keys,
        ):
            scale = 1.0 / math.sqrt(float(max(size, 1)))
            positions.append(
                jax.random.normal(position_key, (int(batch_size), size)) * scale
            )
            velocities.append(
                jax.random.normal(velocity_key, (int(batch_size), size)) * scale
            )
        return tuple(positions), tuple(velocities)

    def auxiliary_coupling_profile_matrix(self, layer_index: int) -> Array:
        """Return the fixed recurrent profile for an auxiliary layer."""

        if layer_index < 0 or layer_index >= len(self.multiscale_layer_sizes):
            raise ValueError("auxiliary layer index out of range")
        return intra_layer_profile(self.layer_specs[layer_index])

    def vertical_profile_matrix(
        self,
        spec_index: int,
        target_gate: Optional[str] = None,
    ) -> Array:
        """Return a fixed source-to-target profile for a vertical projection."""

        if spec_index < 0 or spec_index >= len(self.vertical_specs):
            raise ValueError("vertical spec index out of range")
        spec = self.vertical_specs[spec_index]
        profile = inter_layer_profile(spec, self.layer_specs)
        return profile * self._vertical_target_gate(spec, target_gate)[:, None]

    def _auxiliary_dynamics_params(self, layer_index: int) -> Tuple[Array, Array]:
        omega = self.auxiliary_omega[layer_index]
        coupling = self.auxiliary_coupling[layer_index]
        if not self.train_recurrent_dynamics:
            omega = jax.lax.stop_gradient(omega)
            coupling = jax.lax.stop_gradient(coupling)
        frequency = (
            self._horn_frequency(omega)
            * self.layer_specs[layer_index].frequency_scale
        )
        coupling = coupling * self.auxiliary_coupling_profile_matrix(layer_index)
        return frequency, coupling

    def _auxiliary_conditioning_drive(
        self,
        layer_index: int,
        position: Array,
        labels: Optional[Array],
    ) -> Array:
        if (
            labels is None
            or self.label_condition_phase is None
            or not self.auxiliary_label_condition_coupling
        ):
            return jnp.zeros_like(position)
        anchor = jnp.tanh(self.label_condition_phase[labels.astype(jnp.int32)])
        coupling = self.auxiliary_label_condition_coupling[layer_index][
            labels.astype(jnp.int32)
        ]
        if not self.train_conditioning_dynamics:
            anchor = jax.lax.stop_gradient(anchor)
            coupling = jax.lax.stop_gradient(coupling)
        displacement = anchor[:, None, :] - position[:, :, None]
        drive = jnp.sum(coupling * displacement, axis=-1)
        return (
            float(self.multiscale_conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def _vertical_drive(
        self,
        spec_index: int,
        target_position: Array,
        source_position: Array,
        target_gate: Optional[str] = None,
    ) -> Array:
        spec = self.vertical_specs[spec_index]
        coupling = self.vertical_coupling[spec_index] * self.vertical_profile_matrix(
            spec_index,
            target_gate,
        )
        if not self.train_recurrent_dynamics:
            coupling = jax.lax.stop_gradient(coupling)
            source_position = jax.lax.stop_gradient(source_position)
        displacement = (
            source_position[:, None, :]
            + float(spec.phase_lag)
            - target_position[:, :, None]
        )
        drive = (
            float(spec.strength)
            * float(self.multiscale_vertical_signal_scale)
            * jnp.sum(coupling[None, :, :] * displacement, axis=-1)
            / float(self.layer_specs[spec.source_layer].num_oscillators)
        )
        return drive

    def _vertical_modulation_signal(
        self,
        spec_index: int,
        source_position: Array,
        target_gate: Optional[str] = None,
    ) -> Array:
        spec = self.vertical_specs[spec_index]
        coupling = self.vertical_coupling[spec_index] * self.vertical_profile_matrix(
            spec_index,
            target_gate,
        )
        if not self.train_recurrent_dynamics:
            coupling = jax.lax.stop_gradient(coupling)
            source_position = jax.lax.stop_gradient(source_position)
        source = jnp.tanh(source_position + float(spec.phase_lag))
        signal = (
            float(spec.strength)
            * float(self.multiscale_vertical_signal_scale)
            * jnp.sum(coupling[None, :, :] * source[:, None, :], axis=-1)
            / float(self.layer_specs[spec.source_layer].num_oscillators)
        )
        return jnp.tanh(signal)

    def _normalize_vertical_modulation(self, modulation: Array) -> Array:
        """Return a homeostatic vertical modulation signal."""

        normalization = self.multiscale_vertical_gain_normalization
        if normalization == "none":
            return modulation
        centered = modulation - jnp.mean(modulation, axis=-1, keepdims=True)
        if normalization == "center":
            return centered
        rms = jnp.sqrt(jnp.mean(centered**2, axis=-1, keepdims=True) + 1e-8)
        target_std = float(self.multiscale_vertical_gain_target_std)
        if target_std <= 0.0:
            return centered
        return centered * (target_std / rms)

    def _vertical_schedule_scale(self, step_index: Optional[Array] = None) -> Array:
        """Return the top-down gain schedule multiplier for one settling step."""

        if self.multiscale_vertical_schedule == "constant":
            return jnp.asarray(1.0, dtype=jnp.float32)
        if step_index is None:
            step_value = max(self.steps - 1, 0)
        else:
            step_value = step_index
        step = jnp.asarray(step_value, dtype=jnp.float32)
        onset = float(self.multiscale_vertical_onset_step)
        if self.multiscale_vertical_schedule == "delayed":
            return jnp.where(step >= onset, 1.0, 0.0).astype(jnp.float32)
        ramp_steps = float(max(self.multiscale_vertical_ramp_steps, 1))
        return jnp.clip((step - onset + 1.0) / ramp_steps, 0.0, 1.0).astype(
            jnp.float32
        )

    def _vertical_layer_terms(
        self,
        layer_index: int,
        target_position: Array,
        positions: Tuple[Array, ...],
        step_index: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        vertical_drive = jnp.zeros_like(target_position)
        modulation = jnp.zeros_like(target_position)
        broad_modulation = jnp.zeros_like(target_position)
        selective_modulation = jnp.zeros_like(target_position)
        for spec_index, spec in enumerate(self.vertical_specs):
            if spec.target_layer != layer_index or spec.strength <= 0.0:
                continue
            if self.multiscale_vertical_mode == "additive":
                vertical_drive = vertical_drive + self._vertical_drive(
                    spec_index,
                    target_position,
                    positions[spec.source_layer],
                )
            else:
                if self.multiscale_vertical_mode == "dual_gain":
                    broad_modulation = (
                        broad_modulation
                        + float(self.multiscale_vertical_broad_gain_scale)
                        * self._vertical_modulation_signal(
                            spec_index,
                            positions[spec.source_layer],
                            target_gate="all",
                        )
                    )
                    selective_modulation = (
                        selective_modulation
                        + float(self.multiscale_vertical_selective_gain_scale)
                        * self._vertical_modulation_signal(
                            spec_index,
                            positions[spec.source_layer],
                            target_gate=self.multiscale_vertical_target_gate,
                        )
                    )
                else:
                    modulation = modulation + self._vertical_modulation_signal(
                        spec_index,
                        positions[spec.source_layer],
                    )
        schedule_scale = self._vertical_schedule_scale(step_index)
        if self.multiscale_vertical_mode == "dual_gain":
            modulation = broad_modulation + selective_modulation
            modulation = self._normalize_vertical_modulation(modulation)
            vertical_drive, modulation = self._apply_vertical_intervention(
                vertical_drive,
                modulation,
            )
            vertical_drive = vertical_drive * schedule_scale
            modulation = modulation * schedule_scale
            gain = jnp.clip(1.0 + modulation, -1.0, 2.0)
            return vertical_drive, gain, modulation
        if self.multiscale_vertical_mode != "additive":
            modulation = self._normalize_vertical_modulation(modulation)
        vertical_drive, modulation = self._apply_vertical_intervention(
            vertical_drive,
            modulation,
        )
        vertical_drive = vertical_drive * schedule_scale
        modulation = modulation * schedule_scale
        if self.multiscale_vertical_mode == "signed_gain":
            gain = jnp.clip(1.0 + modulation, -1.0, 2.0)
        else:
            gain = jnp.clip(1.0 + modulation, 0.0, 2.0)
        return vertical_drive, gain, modulation

    def _apply_vertical_intervention(
        self,
        vertical_drive: Array,
        modulation: Array,
    ) -> Tuple[Array, Array]:
        """Apply sample-time vertical intervention controls."""

        scale = float(self.multiscale_vertical_intervention_scale)
        drive = vertical_drive * scale
        mod = modulation * scale
        intervention = self.multiscale_vertical_intervention
        if intervention == "normal":
            return drive, mod
        if intervention == "zero":
            return jnp.zeros_like(drive), jnp.zeros_like(mod)
        if intervention == "flip":
            return -drive, -mod
        if intervention == "shuffle_batch":
            return jnp.roll(drive, shift=1, axis=0), jnp.roll(mod, shift=1, axis=0)
        return drive, mod

    def _vertical_target_gate(
        self,
        spec: InterLayerCouplingSpec,
        target_gate: Optional[str] = None,
    ) -> Array:
        """Return target-side gating for selective vertical projections."""

        target_size = self.layer_specs[spec.target_layer].num_oscillators
        gate_mode = target_gate or self.multiscale_vertical_target_gate
        if gate_mode == "all":
            return jnp.ones((target_size,), dtype=jnp.float32)
        is_fine_target = spec.target_layer == len(self.layer_specs) - 1
        if not is_fine_target or target_size != self.num_oscillators:
            return jnp.ones((target_size,), dtype=jnp.float32)
        mask = self._conditioning_target_mask_array()
        if gate_mode == "conditioning":
            gate = mask
        else:
            gate = 1.0 - mask
        floor = float(self.multiscale_vertical_soft_gate_floor)
        if floor <= 0.0:
            return gate
        return floor + (1.0 - floor) * gate

    def step_layered_state(
        self,
        state: Tuple[Tuple[Array, ...], Tuple[Array, ...]],
        labels: Optional[Array] = None,
        step_index: Optional[Array] = None,
    ) -> Tuple[Tuple[Array, ...], Tuple[Array, ...]]:
        """Advance all HORN layers by one Euler step."""

        positions, velocities = state
        accelerations = []
        for layer_index, (position, velocity) in enumerate(zip(positions, velocities)):
            if layer_index == len(positions) - 1:
                frequency, coupling = self._horn_dynamics_params()
                condition_drive = self._horn_static_conditioning_drive(
                    position,
                    labels,
                )
                output_feedback_drive = self._output_feedback_drive(position, velocity)
            else:
                frequency, coupling = self._auxiliary_dynamics_params(layer_index)
                condition_drive = self._auxiliary_conditioning_drive(
                    layer_index,
                    position,
                    labels,
                )
                output_feedback_drive = jnp.zeros_like(position)

            displacement = position[:, None, :] - position[:, :, None]
            interaction = jnp.sum(coupling[None, :, :] * displacement, axis=-1)
            vertical_drive, vertical_gain, _ = self._vertical_layer_terms(
                layer_index,
                position,
                positions,
                step_index,
            )
            interaction_term = (
                float(self.main_coupling_strength)
                * interaction
                / float(self.layer_specs[layer_index].num_oscillators)
            )
            conditioning_term = float(self.coupling_strength) * condition_drive
            damping_gain = jnp.ones_like(vertical_gain)
            if self.multiscale_vertical_gain_target == "drive":
                interaction_term = vertical_gain * interaction_term
                conditioning_term = vertical_gain * conditioning_term
            elif self.multiscale_vertical_gain_target == "coupling":
                interaction_term = vertical_gain * interaction_term
            elif self.multiscale_vertical_gain_target == "conditioning":
                conditioning_term = vertical_gain * conditioning_term
            else:
                damping_gain = jnp.clip(vertical_gain, 0.0, 2.0)
            accelerations.append(
                -(frequency[None, :] ** 2) * position
                - float(self.horn_damping) * damping_gain * velocity
                - float(self.horn_nonlinearity) * (position**3)
                + interaction_term
                + conditioning_term
                + vertical_drive
                + output_feedback_drive
            )

        next_velocities = tuple(
            self._bound_state(velocity + self.dt * acceleration)
            for velocity, acceleration in zip(velocities, accelerations)
        )
        next_positions = tuple(
            self._bound_state(position + self.dt * next_velocity)
            for position, next_velocity in zip(positions, next_velocities)
        )
        return next_positions, next_velocities

    def evolve_layered_state(
        self,
        state0: Tuple[Tuple[Array, ...], Tuple[Array, ...]],
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve all HORN layers for the configured settling length."""

        positions0, velocities0 = state0
        if self.steps == 0:
            empty_positions = tuple(
                jnp.zeros((0, *position.shape), dtype=position.dtype)
                for position in positions0
            )
            empty_velocities = tuple(
                jnp.zeros((0, *velocity.shape), dtype=velocity.dtype)
                for velocity in velocities0
            )
            if return_trajectory:
                return positions0, velocities0, empty_positions, empty_velocities
            return positions0, velocities0

        def scan_fn(state, step_index):
            next_state = self.step_layered_state(state, labels, step_index)
            return next_state, next_state

        (final_positions, final_velocities), (
            position_trajectories,
            velocity_trajectories,
        ) = jax.lax.scan(scan_fn, state0, jnp.arange(self.steps))
        if return_trajectory:
            return (
                final_positions,
                final_velocities,
                position_trajectories,
                velocity_trajectories,
            )
        return final_positions, final_velocities

    def sample_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
        *,
        return_initial: bool = False,
        return_trajectory: bool = False,
    ):
        """Sample layered HORN states and return the evolved fine state."""

        auxiliary_key, fine_key = jax.random.split(key)
        aux_positions0, aux_velocities0 = self.initial_auxiliary_state(
            auxiliary_key,
            batch_size,
        )
        fine_position0, fine_velocity0 = self.initial_state(fine_key, batch_size, labels)
        positions0 = (*aux_positions0, fine_position0)
        velocities0 = (*aux_velocities0, fine_velocity0)
        if return_trajectory:
            (
                final_positions,
                final_velocities,
                position_trajectories,
                velocity_trajectories,
            ) = self.evolve_layered_state(
                (positions0, velocities0),
                labels,
                return_trajectory=True,
            )
            if return_initial:
                return (
                    final_positions[-1],
                    final_velocities[-1],
                    fine_position0,
                    fine_velocity0,
                    position_trajectories[-1],
                    velocity_trajectories[-1],
                )
            return (
                final_positions[-1],
                final_velocities[-1],
                position_trajectories[-1],
                velocity_trajectories[-1],
            )

        final_positions, final_velocities = self.evolve_layered_state(
            (positions0, velocities0),
            labels,
        )
        if return_initial:
            return (
                final_positions[-1],
                final_velocities[-1],
                fine_position0,
                fine_velocity0,
            )
        return final_positions[-1], final_velocities[-1]

    def sample_auxiliary_image(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Array:
        """Sample the low-resolution auxiliary image prediction."""

        auxiliary_key, fine_key = jax.random.split(key)
        aux_positions0, aux_velocities0 = self.initial_auxiliary_state(
            auxiliary_key,
            batch_size,
        )
        fine_position0, fine_velocity0 = self.initial_state(fine_key, batch_size, labels)
        final_positions, final_velocities = self.evolve_layered_state(
            ((*aux_positions0, fine_position0), (*aux_velocities0, fine_velocity0)),
            labels,
        )
        layer_index = self.multiscale_auxiliary_readout_layer
        return self.decode_auxiliary_state(
            final_positions[layer_index],
            final_velocities[layer_index],
        )

    def collect_trace(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect layered trajectories and generated samples."""

        auxiliary_key, fine_key = jax.random.split(key)
        aux_positions0, aux_velocities0 = self.initial_auxiliary_state(
            auxiliary_key,
            batch_size,
        )
        fine_position0, fine_velocity0 = self.initial_state(fine_key, batch_size, labels)
        positions0 = (*aux_positions0, fine_position0)
        velocities0 = (*aux_velocities0, fine_velocity0)
        (
            final_positions,
            final_velocities,
            position_trajectories,
            velocity_trajectories,
        ) = self.evolve_layered_state(
            (positions0, velocities0),
            labels,
            return_trajectory=True,
        )
        generated = self.decode_state(final_positions[-1], final_velocities[-1])
        aux_readout_layer = self.multiscale_auxiliary_readout_layer
        _, fine_vertical_gain, fine_vertical_modulation = self._vertical_layer_terms(
            len(final_positions) - 1,
            final_positions[-1],
            final_positions,
            max(self.steps - 1, 0),
        )
        fine_vertical_gain_trajectory = []
        fine_vertical_modulation_trajectory = []
        for step_index in range(self.steps):
            step_positions = tuple(
                trajectory[step_index] for trajectory in position_trajectories
            )
            _, step_gain, step_modulation = self._vertical_layer_terms(
                len(step_positions) - 1,
                step_positions[-1],
                step_positions,
                step_index,
            )
            fine_vertical_gain_trajectory.append(step_gain)
            fine_vertical_modulation_trajectory.append(step_modulation)
        if fine_vertical_gain_trajectory:
            fine_vertical_gain_trajectory_array = jnp.stack(
                fine_vertical_gain_trajectory,
                axis=0,
            )
            fine_vertical_modulation_trajectory_array = jnp.stack(
                fine_vertical_modulation_trajectory,
                axis=0,
            )
        else:
            fine_vertical_gain_trajectory_array = jnp.zeros(
                (0, *fine_vertical_gain.shape),
                dtype=fine_vertical_gain.dtype,
            )
            fine_vertical_modulation_trajectory_array = jnp.zeros(
                (0, *fine_vertical_modulation.shape),
                dtype=fine_vertical_modulation.dtype,
            )
        trace: Dict[str, Array] = {
            "initial_theta": fine_position0,
            "theta_trajectory": position_trajectories[-1],
            "final_theta": final_positions[-1],
            "initial_velocity": fine_velocity0,
            "velocity_trajectory": velocity_trajectories[-1],
            "final_velocity": final_velocities[-1],
            "generated": generated,
            "auxiliary_lowres_generated": self.decode_auxiliary_state(
                final_positions[aux_readout_layer],
                final_velocities[aux_readout_layer],
            ),
            "auxiliary_readout_weight": self.auxiliary_readout_weight,
            "auxiliary_readout_bias": self.auxiliary_readout_bias,
            "auxiliary_readout_layer": jnp.asarray(
                aux_readout_layer,
                dtype=jnp.int32,
            ),
            "auxiliary_readout_size": jnp.asarray(
                self.multiscale_auxiliary_readout_size,
                dtype=jnp.int32,
            ),
            "vertical_target_gate": jnp.asarray(
                (
                    0
                    if self.multiscale_vertical_target_gate == "all"
                    else 1
                    if self.multiscale_vertical_target_gate == "conditioning"
                    else 2
                ),
                dtype=jnp.int32,
            ),
            "vertical_mode": jnp.asarray(
                {
                    "additive": 0,
                    "gain_modulation": 1,
                    "signed_gain": 2,
                    "dual_gain": 3,
                }[self.multiscale_vertical_mode],
                dtype=jnp.int32,
            ),
            "vertical_gain_target": jnp.asarray(
                {
                    "drive": 0,
                    "coupling": 1,
                    "conditioning": 2,
                    "damping": 3,
                }[self.multiscale_vertical_gain_target],
                dtype=jnp.int32,
            ),
            "vertical_gain_normalization": jnp.asarray(
                {
                    "none": 0,
                    "center": 1,
                    "center_rms": 2,
                }[self.multiscale_vertical_gain_normalization],
                dtype=jnp.int32,
            ),
            "vertical_gain_target_std": jnp.asarray(
                self.multiscale_vertical_gain_target_std,
                dtype=fine_vertical_gain.dtype,
            ),
            "vertical_schedule": jnp.asarray(
                {
                    "constant": 0,
                    "delayed": 1,
                    "linear_ramp": 2,
                }[self.multiscale_vertical_schedule],
                dtype=jnp.int32,
            ),
            "vertical_onset_step": jnp.asarray(
                self.multiscale_vertical_onset_step,
                dtype=jnp.int32,
            ),
            "vertical_ramp_steps": jnp.asarray(
                self.multiscale_vertical_ramp_steps,
                dtype=jnp.int32,
            ),
            "vertical_schedule_trajectory": jnp.asarray(
                [
                    self._vertical_schedule_scale(step_index)
                    for step_index in range(self.steps)
                ],
                dtype=fine_vertical_gain.dtype,
            ),
            "vertical_intervention": jnp.asarray(
                {
                    "normal": 0,
                    "zero": 1,
                    "shuffle_batch": 2,
                    "flip": 3,
                }[self.multiscale_vertical_intervention],
                dtype=jnp.int32,
            ),
            "vertical_intervention_scale": jnp.asarray(
                self.multiscale_vertical_intervention_scale,
                dtype=fine_vertical_gain.dtype,
            ),
            "vertical_gain_final": fine_vertical_gain,
            "vertical_gain_trajectory": fine_vertical_gain_trajectory_array,
            "vertical_modulation_final": fine_vertical_modulation,
            "vertical_modulation_trajectory": (
                fine_vertical_modulation_trajectory_array
            ),
            "output_feedback_drive": self._output_feedback_drive(
                final_positions[-1],
                final_velocities[-1],
            ),
            "output_feedback_gain": self.output_feedback_gain,
            "omega": self.omega,
            "coupling": self.coupling,
            "coupling_profile": self.coupling_profile_matrix(),
            "condition_omega": jnp.zeros((0,), dtype=self.omega.dtype),
            "condition_coupling": jnp.zeros((0, 0), dtype=self.coupling.dtype),
            "label_phase_shift": (
                jnp.zeros((0, self.num_oscillators))
                if self.label_phase_shift is None
                else self.label_phase_shift
            ),
            "label_condition_phase": (
                jnp.zeros((0, self.num_condition_oscillators))
                if self.label_condition_phase is None
                else self.label_condition_phase
            ),
            "label_condition_coupling": (
                jnp.zeros((0, self.num_oscillators, self.num_condition_oscillators))
                if self.label_condition_coupling is None
                else self.label_condition_coupling
            ),
            "conditioning_target_mask": self._conditioning_target_mask_array(),
            "spatial_phase_weights": (
                jnp.zeros((0, 2))
                if self.spatial_phase_weights is None
                else self.spatial_phase_weights
            ),
            "local_patch_weights": (
                jnp.zeros((0, 2, 0))
                if self.local_patch_weights is None
                else self.local_patch_weights
            ),
            "spatial_output_bias": (
                jnp.zeros(())
                if self.spatial_output_bias is None
                else self.spatial_output_bias
            ),
        }
        for layer_index in range(len(self.multiscale_layer_sizes)):
            trace[f"aux_{layer_index}_initial_theta"] = positions0[layer_index]
            trace[f"aux_{layer_index}_theta_trajectory"] = position_trajectories[
                layer_index
            ]
            trace[f"aux_{layer_index}_final_theta"] = final_positions[layer_index]
            trace[f"aux_{layer_index}_initial_velocity"] = velocities0[layer_index]
            trace[f"aux_{layer_index}_velocity_trajectory"] = velocity_trajectories[
                layer_index
            ]
            trace[f"aux_{layer_index}_final_velocity"] = final_velocities[layer_index]
            trace[f"aux_{layer_index}_omega"] = self.auxiliary_omega[layer_index]
            trace[f"aux_{layer_index}_coupling"] = self.auxiliary_coupling[
                layer_index
            ]
            trace[f"aux_{layer_index}_coupling_profile"] = (
                self.auxiliary_coupling_profile_matrix(layer_index)
            )
        for spec_index, spec in enumerate(self.vertical_specs):
            trace[f"vertical_{spec_index}_coupling"] = self.vertical_coupling[
                spec_index
            ]
            trace[f"vertical_{spec_index}_profile"] = self.vertical_profile_matrix(
                spec_index
            )
            trace[f"vertical_{spec_index}_source_layer"] = jnp.asarray(
                spec.source_layer,
                dtype=jnp.int32,
            )
            trace[f"vertical_{spec_index}_target_layer"] = jnp.asarray(
                spec.target_layer,
                dtype=jnp.int32,
            )
        return trace
