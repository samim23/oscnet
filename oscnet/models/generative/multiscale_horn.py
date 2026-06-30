"""Multiscale layered HORN image generator."""

from __future__ import annotations

from oscnet.core.layered import (
    InterLayerCouplingSpec,
    OscillatorLayerSpec,
    adjacent_inter_layer_specs,
    inter_layer_profile,
    intra_layer_profile,
)

from .common import Array, Dict, Optional, Tuple, eqx, jax, jnp, math
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
    multiscale_conditioning_strength: float = eqx.field(static=True)
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
        multiscale_conditioning_strength: float = 1.0,
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
        for name, value in (
            ("multiscale_vertical_strength", multiscale_vertical_strength),
            ("multiscale_feedback_strength", multiscale_feedback_strength),
            ("multiscale_conditioning_strength", multiscale_conditioning_strength),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        split_count = (
            1
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
        self.multiscale_conditioning_strength = float(multiscale_conditioning_strength)
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

    def vertical_profile_matrix(self, spec_index: int) -> Array:
        """Return a fixed source-to-target profile for a vertical projection."""

        if spec_index < 0 or spec_index >= len(self.vertical_specs):
            raise ValueError("vertical spec index out of range")
        return inter_layer_profile(self.vertical_specs[spec_index], self.layer_specs)

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
    ) -> Array:
        spec = self.vertical_specs[spec_index]
        coupling = self.vertical_coupling[spec_index] * self.vertical_profile_matrix(
            spec_index
        )
        if not self.train_recurrent_dynamics:
            coupling = jax.lax.stop_gradient(coupling)
            source_position = jax.lax.stop_gradient(source_position)
        displacement = (
            source_position[:, None, :]
            + float(spec.phase_lag)
            - target_position[:, :, None]
        )
        return (
            float(spec.strength)
            * jnp.sum(coupling[None, :, :] * displacement, axis=-1)
            / float(self.layer_specs[spec.source_layer].num_oscillators)
        )

    def step_layered_state(
        self,
        state: Tuple[Tuple[Array, ...], Tuple[Array, ...]],
        labels: Optional[Array] = None,
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
            vertical_drive = jnp.zeros_like(position)
            for spec_index, spec in enumerate(self.vertical_specs):
                if spec.target_layer != layer_index or spec.strength <= 0.0:
                    continue
                vertical_drive = vertical_drive + self._vertical_drive(
                    spec_index,
                    position,
                    positions[spec.source_layer],
                )
            accelerations.append(
                -(frequency[None, :] ** 2) * position
                - float(self.horn_damping) * velocity
                - float(self.horn_nonlinearity) * (position**3)
                + float(self.main_coupling_strength)
                * interaction
                / float(self.layer_specs[layer_index].num_oscillators)
                + float(self.coupling_strength) * condition_drive
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

        def scan_fn(state, _):
            next_state = self.step_layered_state(state, labels)
            return next_state, next_state

        (final_positions, final_velocities), (
            position_trajectories,
            velocity_trajectories,
        ) = jax.lax.scan(scan_fn, state0, xs=None, length=self.steps)
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
        trace: Dict[str, Array] = {
            "initial_theta": fine_position0,
            "theta_trajectory": position_trajectories[-1],
            "final_theta": final_positions[-1],
            "initial_velocity": fine_velocity0,
            "velocity_trajectory": velocity_trajectories[-1],
            "final_velocity": final_velocities[-1],
            "generated": generated,
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
