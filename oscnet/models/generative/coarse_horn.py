"""Coarse-to-fine HORN image generator."""

from __future__ import annotations

from oscnet.core.coupling import (
    coupling_profile_from_name,
    rectangular_coupling_profile_from_name,
)

from .common import Array, Dict, Optional, Tuple, eqx, jax, jnp, math
from .horn import HORNImageGenerator


class CoarseToFineHORNImageGenerator(HORNImageGenerator):
    """HORN generator with a slow coarse oscillator field driving the fine field.

    The fine field keeps the same decoder surface as ``HORNImageGenerator``.
    A smaller coarse HORN population evolves in parallel and sends learned
    displacement drive into each fine oscillator. This tests whether global
    structure is better carried by a low-resolution oscillatory mode instead
    of widening the fine recurrent coupling everywhere.
    """

    coarse_omega: Array
    coarse_coupling: Array
    coarse_to_fine_coupling: Array
    coarse_label_condition_coupling: Optional[Array]
    num_coarse_oscillators: int = eqx.field(static=True)
    coarse_coupling_profile: str = eqx.field(static=True)
    coarse_coupling_normalization: str = eqx.field(static=True)
    coarse_coupling_length_scale: float = eqx.field(static=True)
    coarse_to_fine_strength: float = eqx.field(static=True)
    coarse_to_fine_profile: str = eqx.field(static=True)
    coarse_to_fine_normalization: str = eqx.field(static=True)
    coarse_to_fine_length_scale: float = eqx.field(static=True)
    coarse_to_fine_floor: float = eqx.field(static=True)
    coarse_conditioning_strength: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_coarse_oscillators: int = 16,
        coarse_coupling_profile: str = "dense",
        coarse_coupling_normalization: str = "row_sum",
        coarse_coupling_length_scale: float = 0.0,
        coarse_to_fine_strength: float = 1.0,
        coarse_to_fine_profile: str = "dense",
        coarse_to_fine_normalization: str = "row_sum",
        coarse_to_fine_length_scale: float = 0.0,
        coarse_to_fine_floor: float = 0.0,
        coarse_conditioning_strength: float = 1.0,
        **kwargs,
    ):
        if num_coarse_oscillators < 1:
            raise ValueError("num_coarse_oscillators must be positive")
        if coarse_coupling_profile not in ("dense", "distance_decay", "local_radius"):
            raise ValueError(
                "coarse_coupling_profile must be 'dense', 'distance_decay', or "
                "'local_radius'"
            )
        if coarse_coupling_normalization not in ("none", "row_sum"):
            raise ValueError(
                "coarse_coupling_normalization must be 'none' or 'row_sum'"
            )
        if coarse_to_fine_strength < 0.0:
            raise ValueError("coarse_to_fine_strength must be non-negative")
        if coarse_to_fine_profile not in ("dense", "distance_decay", "local_radius"):
            raise ValueError(
                "coarse_to_fine_profile must be 'dense', 'distance_decay', or "
                "'local_radius'"
            )
        if coarse_to_fine_normalization not in ("none", "row_sum"):
            raise ValueError("coarse_to_fine_normalization must be 'none' or 'row_sum'")
        if coarse_to_fine_floor < 0.0 or coarse_to_fine_floor > 1.0:
            raise ValueError("coarse_to_fine_floor must be in [0, 1]")
        if coarse_conditioning_strength < 0.0:
            raise ValueError("coarse_conditioning_strength must be non-negative")

        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        (
            base_key,
            coarse_omega_key,
            coarse_coupling_key,
            coarse_to_fine_key,
            coarse_condition_key,
        ) = jax.random.split(key, 5)
        kwargs["key"] = base_key
        omega_scale = float(kwargs.get("omega_scale", 0.2))
        coupling_init_scale = float(kwargs.get("coupling_init_scale", 0.05))

        super().__init__(**kwargs)
        if self.conditioning_mode == "class_oscillator":
            raise ValueError(
                "CoarseToFineHORNImageGenerator currently supports "
                "'none', 'phase_shift', and 'class_coupling' conditioning"
            )

        self.num_coarse_oscillators = int(num_coarse_oscillators)
        self.coarse_coupling_profile = coarse_coupling_profile
        self.coarse_coupling_normalization = coarse_coupling_normalization
        self.coarse_coupling_length_scale = float(coarse_coupling_length_scale)
        self.coarse_to_fine_strength = float(coarse_to_fine_strength)
        self.coarse_to_fine_profile = coarse_to_fine_profile
        self.coarse_to_fine_normalization = coarse_to_fine_normalization
        self.coarse_to_fine_length_scale = float(coarse_to_fine_length_scale)
        self.coarse_to_fine_floor = float(coarse_to_fine_floor)
        self.coarse_conditioning_strength = float(coarse_conditioning_strength)
        self.dynamics_family = "coarse_horn"

        self.coarse_omega = (
            jax.random.normal(coarse_omega_key, (self.num_coarse_oscillators,))
            * omega_scale
        )
        coarse_coupling = (
            jax.random.normal(
                coarse_coupling_key,
                (self.num_coarse_oscillators, self.num_coarse_oscillators),
            )
            * coupling_init_scale
            / jnp.sqrt(float(self.num_coarse_oscillators))
        )
        self.coarse_coupling = coarse_coupling * (
            1.0 - jnp.eye(self.num_coarse_oscillators, dtype=jnp.float32)
        )
        self.coarse_to_fine_coupling = (
            jax.random.normal(
                coarse_to_fine_key,
                (self.num_oscillators, self.num_coarse_oscillators),
            )
            * coupling_init_scale
            / jnp.sqrt(float(self.num_coarse_oscillators))
        )
        self.coarse_label_condition_coupling = None
        if (
            self.num_classes > 0
            and self.conditioning_mode == "class_coupling"
            and self.num_condition_oscillators > 0
        ):
            self.coarse_label_condition_coupling = (
                jax.random.normal(
                    coarse_condition_key,
                    (
                        self.num_classes,
                        self.num_coarse_oscillators,
                        self.num_condition_oscillators,
                    ),
                )
                * coupling_init_scale
                / jnp.sqrt(float(max(self.num_condition_oscillators, 1)))
            )

    def initial_coarse_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[Array, Array]:
        """Sample random coarse HORN position and velocity state."""

        position_key, velocity_key = jax.random.split(key)
        scale = 1.0 / math.sqrt(float(max(self.num_coarse_oscillators, 1)))
        return (
            jax.random.normal(
                position_key,
                (int(batch_size), self.num_coarse_oscillators),
            )
            * scale,
            jax.random.normal(
                velocity_key,
                (int(batch_size), self.num_coarse_oscillators),
            )
            * scale,
        )

    def coarse_coupling_profile_matrix(self) -> Array:
        """Return the fixed recurrent profile for the coarse HORN field."""

        return coupling_profile_from_name(
            name=self.coarse_coupling_profile,
            num_oscillators=self.num_coarse_oscillators,
            length_scale=self.coarse_coupling_length_scale,
            floor=0.0,
            normalization=self.coarse_coupling_normalization,
            target_row_sum=float(self.num_coarse_oscillators),
        )

    def coarse_to_fine_profile_matrix(self) -> Array:
        """Return the fixed source-to-target profile for coarse drive."""

        return rectangular_coupling_profile_from_name(
            name=self.coarse_to_fine_profile,
            num_targets=self.num_oscillators,
            num_sources=self.num_coarse_oscillators,
            length_scale=self.coarse_to_fine_length_scale,
            floor=self.coarse_to_fine_floor,
            normalization=self.coarse_to_fine_normalization,
            target_row_sum=float(self.num_coarse_oscillators),
        )

    def _coarse_dynamics_params(self) -> Tuple[Array, Array]:
        omega = self.coarse_omega
        coupling = self.coarse_coupling
        if not self.train_recurrent_dynamics:
            omega = jax.lax.stop_gradient(omega)
            coupling = jax.lax.stop_gradient(coupling)
        coupling = coupling * self.coarse_coupling_profile_matrix()
        return self._horn_frequency(omega), coupling

    def _coarse_conditioning_drive(
        self,
        coarse_position: Array,
        labels: Optional[Array],
    ) -> Array:
        if (
            labels is None
            or self.label_condition_phase is None
            or self.coarse_label_condition_coupling is None
        ):
            return jnp.zeros_like(coarse_position)

        anchor = jnp.tanh(self.label_condition_phase[labels.astype(jnp.int32)])
        coupling = self.coarse_label_condition_coupling[labels.astype(jnp.int32)]
        if not self.train_conditioning_dynamics:
            anchor = jax.lax.stop_gradient(anchor)
            coupling = jax.lax.stop_gradient(coupling)
        displacement = anchor[:, None, :] - coarse_position[:, :, None]
        drive = jnp.sum(coupling * displacement, axis=-1)
        return (
            float(self.coarse_conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def _coarse_to_fine_drive(
        self,
        position: Array,
        coarse_position: Array,
    ) -> Array:
        coupling = self.coarse_to_fine_coupling * self.coarse_to_fine_profile_matrix()
        if not self.train_recurrent_dynamics:
            coupling = jax.lax.stop_gradient(coupling)
            coarse_position = jax.lax.stop_gradient(coarse_position)
        displacement = coarse_position[:, None, :] - position[:, :, None]
        return jnp.sum(coupling[None, :, :] * displacement, axis=-1) / float(
            self.num_coarse_oscillators
        )

    def step_coarse_fine_state(
        self,
        state: Tuple[Array, Array],
        coarse_state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Advance fine and coarse HORN states by one Euler step."""

        position, velocity = state
        coarse_position, coarse_velocity = coarse_state

        coarse_frequency, coarse_coupling = self._coarse_dynamics_params()
        coarse_displacement = (
            coarse_position[:, None, :] - coarse_position[:, :, None]
        )
        coarse_interaction = jnp.sum(
            coarse_coupling[None, :, :] * coarse_displacement,
            axis=-1,
        )
        coarse_condition_drive = self._coarse_conditioning_drive(
            coarse_position,
            labels,
        )
        coarse_acceleration = (
            -(coarse_frequency[None, :] ** 2) * coarse_position
            - float(self.horn_damping) * coarse_velocity
            - float(self.horn_nonlinearity) * (coarse_position**3)
            + float(self.main_coupling_strength)
            * coarse_interaction
            / float(self.num_coarse_oscillators)
            + float(self.coupling_strength) * coarse_condition_drive
        )
        next_coarse_velocity = self._bound_state(
            coarse_velocity + self.dt * coarse_acceleration
        )
        next_coarse_position = self._bound_state(
            coarse_position + self.dt * next_coarse_velocity
        )

        frequency, coupling = self._horn_dynamics_params()
        displacement = position[:, None, :] - position[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * displacement, axis=-1)
        condition_drive = self._horn_static_conditioning_drive(position, labels)
        coarse_drive = self._coarse_to_fine_drive(position, coarse_position)
        output_feedback_drive = self._output_feedback_drive(position, velocity)
        acceleration = (
            -(frequency[None, :] ** 2) * position
            - float(self.horn_damping) * velocity
            - float(self.horn_nonlinearity) * (position**3)
            + float(self.main_coupling_strength)
            * interaction
            / float(self.num_oscillators)
            + float(self.coupling_strength) * condition_drive
            + float(self.coarse_to_fine_strength) * coarse_drive
            + output_feedback_drive
        )
        next_velocity = self._bound_state(velocity + self.dt * acceleration)
        next_position = self._bound_state(position + self.dt * next_velocity)
        return (next_position, next_velocity), (
            next_coarse_position,
            next_coarse_velocity,
        )

    def evolve_coarse_fine_state(
        self,
        state0: Tuple[Array, Array],
        coarse_state0: Tuple[Array, Array],
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve fine and coarse states for the configured scan length."""

        position0, velocity0 = state0
        coarse_position0, coarse_velocity0 = coarse_state0
        if self.steps == 0:
            empty_position = jnp.zeros((0, *position0.shape), dtype=position0.dtype)
            empty_velocity = jnp.zeros((0, *velocity0.shape), dtype=velocity0.dtype)
            empty_coarse_position = jnp.zeros(
                (0, *coarse_position0.shape),
                dtype=coarse_position0.dtype,
            )
            empty_coarse_velocity = jnp.zeros_like(empty_coarse_position)
            if return_trajectory:
                return (
                    position0,
                    velocity0,
                    coarse_position0,
                    coarse_velocity0,
                    empty_position,
                    empty_velocity,
                    empty_coarse_position,
                    empty_coarse_velocity,
                )
            return position0, velocity0, coarse_position0, coarse_velocity0

        def scan_fn(carry, _):
            state, coarse_state = carry
            next_state, next_coarse_state = self.step_coarse_fine_state(
                state,
                coarse_state,
                labels,
            )
            return (next_state, next_coarse_state), (
                next_state,
                next_coarse_state,
            )

        (
            (final_position, final_velocity),
            (final_coarse_position, final_coarse_velocity),
        ), (
            (position_trajectory, velocity_trajectory),
            (coarse_position_trajectory, coarse_velocity_trajectory),
        ) = jax.lax.scan(
            scan_fn,
            (state0, coarse_state0),
            xs=None,
            length=self.steps,
        )
        if return_trajectory:
            return (
                final_position,
                final_velocity,
                final_coarse_position,
                final_coarse_velocity,
                position_trajectory,
                velocity_trajectory,
                coarse_position_trajectory,
                coarse_velocity_trajectory,
            )
        return (
            final_position,
            final_velocity,
            final_coarse_position,
            final_coarse_velocity,
        )

    def sample_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
        *,
        return_initial: bool = False,
        return_trajectory: bool = False,
    ):
        """Sample and evolve fine/coarse HORN state, returning fine state."""

        state_key, coarse_key = jax.random.split(key)
        state0 = self.initial_state(state_key, batch_size, labels)
        coarse_state0 = self.initial_coarse_state(coarse_key, batch_size)
        if return_trajectory:
            (
                final_position,
                final_velocity,
                _,
                _,
                position_trajectory,
                velocity_trajectory,
                _,
                _,
            ) = self.evolve_coarse_fine_state(
                state0,
                coarse_state0,
                labels,
                return_trajectory=True,
            )
            if return_initial:
                return (
                    final_position,
                    final_velocity,
                    state0[0],
                    state0[1],
                    position_trajectory,
                    velocity_trajectory,
                )
            return (
                final_position,
                final_velocity,
                position_trajectory,
                velocity_trajectory,
            )
        final_position, final_velocity, _, _ = self.evolve_coarse_fine_state(
            state0,
            coarse_state0,
            labels,
        )
        if return_initial:
            return final_position, final_velocity, state0[0], state0[1]
        return final_position, final_velocity

    def collect_trace(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect fine/coarse HORN trajectories and samples for diagnostics."""

        state_key, coarse_key = jax.random.split(key)
        position0, velocity0 = self.initial_state(state_key, batch_size, labels)
        coarse_position0, coarse_velocity0 = self.initial_coarse_state(
            coarse_key,
            batch_size,
        )
        (
            final_position,
            final_velocity,
            final_coarse_position,
            final_coarse_velocity,
            position_trajectory,
            velocity_trajectory,
            coarse_position_trajectory,
            coarse_velocity_trajectory,
        ) = self.evolve_coarse_fine_state(
            (position0, velocity0),
            (coarse_position0, coarse_velocity0),
            labels,
            return_trajectory=True,
        )
        generated = self.decode_state(final_position, final_velocity)
        output_feedback_drive = self._output_feedback_drive(
            final_position,
            final_velocity,
        )
        return {
            "initial_theta": position0,
            "theta_trajectory": position_trajectory,
            "final_theta": final_position,
            "initial_velocity": velocity0,
            "velocity_trajectory": velocity_trajectory,
            "final_velocity": final_velocity,
            "coarse_initial_theta": coarse_position0,
            "coarse_theta_trajectory": coarse_position_trajectory,
            "coarse_final_theta": final_coarse_position,
            "coarse_initial_velocity": coarse_velocity0,
            "coarse_velocity_trajectory": coarse_velocity_trajectory,
            "coarse_final_velocity": final_coarse_velocity,
            "condition_initial_theta": jnp.zeros(
                (int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "condition_theta_trajectory": jnp.zeros(
                (0, int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "condition_final_theta": jnp.zeros(
                (int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "condition_initial_velocity": jnp.zeros(
                (int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "condition_velocity_trajectory": jnp.zeros(
                (0, int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "condition_final_velocity": jnp.zeros(
                (int(batch_size), self.num_condition_oscillators),
                dtype=position0.dtype,
            ),
            "generated": generated,
            "output_feedback_drive": output_feedback_drive,
            "output_feedback_gain": self.output_feedback_gain,
            "omega": self.omega,
            "coupling": self.coupling,
            "coupling_profile": self.coupling_profile_matrix(),
            "coarse_omega": self.coarse_omega,
            "coarse_coupling": self.coarse_coupling,
            "coarse_coupling_profile": self.coarse_coupling_profile_matrix(),
            "coarse_to_fine_coupling": self.coarse_to_fine_coupling,
            "coarse_to_fine_profile": self.coarse_to_fine_profile_matrix(),
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
            "coarse_label_condition_coupling": (
                jnp.zeros(
                    (
                        0,
                        self.num_coarse_oscillators,
                        self.num_condition_oscillators,
                    )
                )
                if self.coarse_label_condition_coupling is None
                else self.coarse_label_condition_coupling
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
