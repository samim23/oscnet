"""HORN-style second-order implicit oscillator image generator."""

from __future__ import annotations

from .common import (
    Array,
    Dict,
    Optional,
    Tuple,
    _activation,
    _image_hw_channels,
    _local_basis_tensor,
    _oscillator_grid_coordinates,
    _softplus_inverse,
    _spatial_basis_matrix,
    eqx,
    jax,
    jnp,
    math,
)
from .kuramoto import KuramotoImageGenerator

class HORNImageGenerator(KuramotoImageGenerator):
    """Implicit image generator with second-order HORN-style dynamics.

    This keeps the same generator/readout surface as ``KuramotoImageGenerator``
    so experiments can compare phase-only Kuramoto dynamics against a
    homogeneous oscillator with explicit position and velocity state. The
    decoder sees ``tanh(position)`` and ``tanh(velocity)`` features, matching the
    two-channel sin/cos phase feature budget used by the Kuramoto generator.
    """

    horn_frequency: float = eqx.field(static=True)
    horn_damping: float = eqx.field(static=True)
    horn_nonlinearity: float = eqx.field(static=True)
    horn_state_bound: float = eqx.field(static=True)
    output_feedback_mode: str = eqx.field(static=True)
    output_feedback_strength: float = eqx.field(static=True)
    output_feedback_init_scale: float = eqx.field(static=True)
    output_feedback_basis_sigma: float = eqx.field(static=True)
    dynamics_family: str = eqx.field(static=True)
    output_feedback_gain: Array

    def __init__(
        self,
        *,
        horn_frequency: float = 1.0,
        horn_damping: float = 0.15,
        horn_nonlinearity: float = 0.05,
        horn_state_bound: float = 3.0,
        output_feedback_mode: str = "state_proxy",
        output_feedback_strength: float = 0.0,
        output_feedback_init_scale: float = 0.02,
        output_feedback_basis_sigma: float = 0.0,
        **kwargs,
    ):
        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        super().__init__(**kwargs)
        if horn_frequency <= 0.0:
            raise ValueError("horn_frequency must be positive")
        if horn_damping < 0.0:
            raise ValueError("horn_damping must be non-negative")
        if horn_nonlinearity < 0.0:
            raise ValueError("horn_nonlinearity must be non-negative")
        if horn_state_bound < 0.0:
            raise ValueError("horn_state_bound must be non-negative")
        if output_feedback_mode not in ("state_proxy", "image"):
            raise ValueError("output_feedback_mode must be 'state_proxy' or 'image'")
        if output_feedback_strength < 0.0:
            raise ValueError("output_feedback_strength must be non-negative")
        if output_feedback_init_scale < 0.0:
            raise ValueError("output_feedback_init_scale must be non-negative")
        if output_feedback_basis_sigma < 0.0:
            raise ValueError("output_feedback_basis_sigma must be non-negative")
        self.horn_frequency = float(horn_frequency)
        self.horn_damping = float(horn_damping)
        self.horn_nonlinearity = float(horn_nonlinearity)
        self.horn_state_bound = float(horn_state_bound)
        self.output_feedback_mode = output_feedback_mode
        self.output_feedback_strength = float(output_feedback_strength)
        self.output_feedback_init_scale = float(output_feedback_init_scale)
        self.output_feedback_basis_sigma = float(output_feedback_basis_sigma)
        self.dynamics_family = "horn"
        if self.output_feedback_strength > 0.0:
            feedback_key = jax.random.fold_in(key, 30091)
            self.output_feedback_gain = (
                jax.random.normal(feedback_key, (self.num_oscillators,))
                * self.output_feedback_init_scale
            )
        else:
            self.output_feedback_gain = jnp.zeros((0,), dtype=jnp.float32)

    def initial_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Sample random initial HORN position and velocity state."""

        position_key, velocity_key = jax.random.split(key)
        scale = 1.0 / math.sqrt(float(max(self.num_oscillators, 1)))
        position = jax.random.normal(
            position_key,
            (int(batch_size), self.num_oscillators),
        ) * scale
        velocity = jax.random.normal(
            velocity_key,
            (int(batch_size), self.num_oscillators),
        ) * scale
        if self.label_phase_shift is not None and labels is not None:
            label_shift = self.label_phase_shift[labels.astype(jnp.int32)]
            if not self.train_conditioning_dynamics:
                label_shift = jax.lax.stop_gradient(label_shift)
            position = position + label_shift
        return position, velocity

    def initial_condition_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Tuple[Array, Array]:
        """Sample state for the optional conditioning oscillator pool."""

        position_key, velocity_key = jax.random.split(key)
        scale = 1.0 / math.sqrt(float(max(self.num_condition_oscillators, 1)))
        return (
            jax.random.normal(
                position_key,
                (int(batch_size), self.num_condition_oscillators),
            )
            * scale,
            jax.random.normal(
                velocity_key,
                (int(batch_size), self.num_condition_oscillators),
            )
            * scale,
        )

    def _horn_frequency(self, omega: Array) -> Array:
        base = _softplus_inverse(self.horn_frequency)
        return jax.nn.softplus(base + omega) + 1e-3

    def _horn_dynamics_params(self) -> Tuple[Array, Array]:
        omega, coupling = self._dynamics_params()
        return self._horn_frequency(omega), coupling

    def _horn_condition_frequency(self) -> Array:
        condition_omega, _ = self._condition_dynamics_params()
        return self._horn_frequency(condition_omega)

    def _horn_static_conditioning_drive(
        self,
        position: Array,
        labels: Optional[Array],
    ) -> Array:
        if (
            labels is None
            or self.label_condition_phase is None
            or self.label_condition_coupling is None
        ):
            return jnp.zeros_like(position)

        anchor = jnp.tanh(self.label_condition_phase[labels.astype(jnp.int32)])
        condition_coupling = self.label_condition_coupling[labels.astype(jnp.int32)]
        condition_coupling = (
            condition_coupling * self._conditioning_target_mask_array()[None, :, None]
        )
        if not self.train_conditioning_dynamics:
            anchor = jax.lax.stop_gradient(anchor)
            condition_coupling = jax.lax.stop_gradient(condition_coupling)

        displacement = anchor[:, None, :] - position[:, :, None]
        drive = jnp.sum(condition_coupling * displacement, axis=-1)
        return (
            float(self.conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def _horn_dynamic_conditioning_drive(
        self,
        position: Array,
        condition_position: Array,
        labels: Optional[Array],
    ) -> Array:
        if (
            labels is None
            or self.label_condition_coupling is None
            or self.num_condition_oscillators == 0
        ):
            return jnp.zeros_like(position)

        condition_coupling = self.label_condition_coupling[labels.astype(jnp.int32)]
        condition_coupling = (
            condition_coupling * self._conditioning_target_mask_array()[None, :, None]
        )
        if not self.train_conditioning_dynamics:
            condition_position = jax.lax.stop_gradient(condition_position)
            condition_coupling = jax.lax.stop_gradient(condition_coupling)
        displacement = condition_position[:, None, :] - position[:, :, None]
        drive = jnp.sum(condition_coupling * displacement, axis=-1)
        return (
            float(self.conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def _bound_state(self, state: Array) -> Array:
        if self.horn_state_bound <= 0.0:
            return state
        bound = float(self.horn_state_bound)
        return bound * jnp.tanh(state / bound)

    def _output_feedback_basis(self) -> Array:
        """Pool decoded pixels back to oscillator locations."""

        height, width, _ = _image_hw_channels(self.image_shape)
        pixel_y, pixel_x = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, height),
            jnp.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        pixels = jnp.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], axis=-1)
        centers = _oscillator_grid_coordinates(self.num_oscillators)
        squared_distance = jnp.sum(
            (centers[:, None, :] - pixels[None, :, :]) ** 2,
            axis=-1,
        )
        sigma = float(self.output_feedback_basis_sigma)
        if sigma <= 0.0:
            grid_rows = max(1, int(math.floor(math.sqrt(self.num_oscillators))))
            grid_cols = max(1, int(math.ceil(self.num_oscillators / grid_rows)))
            sigma = 1.25 / float(max(grid_rows, grid_cols))
        basis = jnp.exp(-squared_distance / (2.0 * sigma**2))
        return basis / jnp.maximum(jnp.sum(basis, axis=-1, keepdims=True), 1e-8)

    def _output_feedback_drive(self, position: Array, velocity: Array) -> Array:
        """Return local self-feedback from the current generated state."""

        if (
            self.output_feedback_strength <= 0.0
            or self.output_feedback_gain.shape[0] != self.num_oscillators
        ):
            return jnp.zeros_like(position)
        if self.output_feedback_mode == "state_proxy":
            proxy = jnp.tanh(position) + 0.5 * jnp.tanh(velocity)
            centered = proxy - jnp.mean(proxy, axis=-1, keepdims=True)
            gain = self.output_feedback_gain
            if not self.train_recurrent_dynamics:
                gain = jax.lax.stop_gradient(gain)
                centered = jax.lax.stop_gradient(centered)
            return float(self.output_feedback_strength) * gain[None, :] * centered

        height, width, channels = _image_hw_channels(self.image_shape)
        decoded = self.decode_state(position, velocity).reshape(
            position.shape[0],
            height * width,
            channels,
        )
        luminance = jnp.mean(decoded, axis=-1)
        pooled = luminance @ self._output_feedback_basis().T
        centered = pooled - jnp.mean(pooled, axis=-1, keepdims=True)
        gain = self.output_feedback_gain
        if not self.train_recurrent_dynamics:
            gain = jax.lax.stop_gradient(gain)
            centered = jax.lax.stop_gradient(centered)
        return float(self.output_feedback_strength) * gain[None, :] * centered

    def step_state(
        self,
        state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance one second-order HORN step for a batch of states."""

        position, velocity = state
        frequency, coupling = self._horn_dynamics_params()
        displacement = position[:, None, :] - position[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * displacement, axis=-1)
        condition_drive = self._horn_static_conditioning_drive(position, labels)
        output_feedback_drive = self._output_feedback_drive(position, velocity)
        acceleration = (
            -(frequency[None, :] ** 2) * position
            - float(self.horn_damping) * velocity
            - float(self.horn_nonlinearity) * (position**3)
            + float(self.main_coupling_strength)
            * interaction
            / float(self.num_oscillators)
            + float(self.coupling_strength) * condition_drive
            + output_feedback_drive
        )
        next_velocity = self._bound_state(velocity + self.dt * acceleration)
        next_position = self._bound_state(position + self.dt * next_velocity)
        return next_position, next_velocity

    def step_joint_state(
        self,
        state: Tuple[Array, Array],
        condition_state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Advance main and conditioning HORN states by one Euler step."""

        position, velocity = state
        condition_position, condition_velocity = condition_state
        frequency, coupling = self._horn_dynamics_params()
        displacement = position[:, None, :] - position[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * displacement, axis=-1)
        condition_drive = self._horn_dynamic_conditioning_drive(
            position,
            condition_position,
            labels,
        )
        output_feedback_drive = self._output_feedback_drive(position, velocity)
        acceleration = (
            -(frequency[None, :] ** 2) * position
            - float(self.horn_damping) * velocity
            - float(self.horn_nonlinearity) * (position**3)
            + float(self.main_coupling_strength)
            * interaction
            / float(self.num_oscillators)
            + float(self.coupling_strength) * condition_drive
            + output_feedback_drive
        )
        next_velocity = self._bound_state(velocity + self.dt * acceleration)
        next_position = self._bound_state(position + self.dt * next_velocity)

        if self.num_condition_oscillators == 0:
            return (next_position, next_velocity), condition_state

        condition_frequency = self._horn_condition_frequency()
        _, condition_coupling = self._condition_dynamics_params()
        condition_displacement = (
            condition_position[:, None, :] - condition_position[:, :, None]
        )
        condition_interaction = jnp.sum(
            condition_coupling[None, :, :] * condition_displacement,
            axis=-1,
        )
        condition_acceleration = (
            -(condition_frequency[None, :] ** 2) * condition_position
            - float(self.horn_damping) * condition_velocity
            - float(self.horn_nonlinearity) * (condition_position**3)
            + float(self.main_coupling_strength)
            * condition_interaction
            / float(max(self.num_condition_oscillators, 1))
        )
        next_condition_velocity = self._bound_state(
            condition_velocity + self.dt * condition_acceleration
        )
        next_condition_position = self._bound_state(
            condition_position + self.dt * next_condition_velocity
        )
        return (
            (next_position, next_velocity),
            (next_condition_position, next_condition_velocity),
        )

    def evolve_state(
        self,
        state0: Tuple[Array, Array],
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ) -> Tuple[Array, Array] | Tuple[Array, Array, Array, Array]:
        """Evolve initial HORN state for the configured scan length."""

        position0, velocity0 = state0
        if self.steps == 0:
            empty_position = jnp.zeros((0, *position0.shape), dtype=position0.dtype)
            empty_velocity = jnp.zeros((0, *velocity0.shape), dtype=velocity0.dtype)
            if return_trajectory:
                return position0, velocity0, empty_position, empty_velocity
            return position0, velocity0

        def scan_fn(state, _):
            next_state = self.step_state(state, labels)
            return next_state, next_state

        (final_position, final_velocity), (
            position_trajectory,
            velocity_trajectory,
        ) = jax.lax.scan(scan_fn, state0, xs=None, length=self.steps)
        if return_trajectory:
            return (
                final_position,
                final_velocity,
                position_trajectory,
                velocity_trajectory,
            )
        return final_position, final_velocity

    def evolve_joint_state(
        self,
        state0: Tuple[Array, Array],
        condition_state0: Tuple[Array, Array],
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve main and conditioning HORN states together."""

        position0, velocity0 = state0
        condition_position0, condition_velocity0 = condition_state0
        if self.steps == 0:
            empty_position = jnp.zeros((0, *position0.shape), dtype=position0.dtype)
            empty_velocity = jnp.zeros((0, *velocity0.shape), dtype=velocity0.dtype)
            empty_condition_position = jnp.zeros(
                (0, *condition_position0.shape),
                dtype=condition_position0.dtype,
            )
            empty_condition_velocity = jnp.zeros(
                (0, *condition_velocity0.shape),
                dtype=condition_velocity0.dtype,
            )
            if return_trajectory:
                return (
                    position0,
                    velocity0,
                    condition_position0,
                    condition_velocity0,
                    empty_position,
                    empty_velocity,
                    empty_condition_position,
                    empty_condition_velocity,
                )
            return position0, velocity0, condition_position0, condition_velocity0

        def scan_fn(carry, _):
            state, condition_state = carry
            next_state, next_condition_state = self.step_joint_state(
                state,
                condition_state,
                labels,
            )
            return (next_state, next_condition_state), (
                next_state,
                next_condition_state,
            )

        (
            (final_position, final_velocity),
            (final_condition_position, final_condition_velocity),
        ), (
            (position_trajectory, velocity_trajectory),
            (condition_position_trajectory, condition_velocity_trajectory),
        ) = jax.lax.scan(
            scan_fn,
            (state0, condition_state0),
            xs=None,
            length=self.steps,
        )
        if return_trajectory:
            return (
                final_position,
                final_velocity,
                final_condition_position,
                final_condition_velocity,
                position_trajectory,
                velocity_trajectory,
                condition_position_trajectory,
                condition_velocity_trajectory,
            )
        return (
            final_position,
            final_velocity,
            final_condition_position,
            final_condition_velocity,
        )

    def state_features(self, position: Array, velocity: Array) -> Array:
        """Return bounded two-channel oscillator features."""

        if self.readout_mode in ("relative", "ref_oscillator"):
            position = position - position[:, :1]
            velocity = velocity - velocity[:, :1]
        elif self.readout_mode == "mean_relative":
            position = position - jnp.mean(position, axis=-1, keepdims=True)
            velocity = velocity - jnp.mean(velocity, axis=-1, keepdims=True)
        return jnp.concatenate([jnp.tanh(position), jnp.tanh(velocity)], axis=-1)

    def decode_state(self, position: Array, velocity: Array) -> Array:
        """Decode final HORN state features into flat images."""

        if self.decoder_mode == "spatial_basis":
            if self.spatial_phase_weights is None or self.spatial_output_bias is None:
                raise ValueError("spatial_basis decoder is missing readout weights")
            basis = _spatial_basis_matrix(
                num_oscillators=self.num_oscillators,
                image_shape=self.image_shape,
                sigma=self.spatial_basis_sigma,
            )
            oscillator_features = self.state_features(position, velocity).reshape(
                position.shape[0],
                self.num_oscillators,
                2,
            )
            oscillator_drive = jnp.sum(
                oscillator_features * self.spatial_phase_weights[None, :, :],
                axis=-1,
            )
            pixels = oscillator_drive @ basis + self.spatial_output_bias
            return _activation(self.output_activation)(pixels)

        if self.decoder_mode == "local_basis":
            if self.local_patch_weights is None or self.spatial_output_bias is None:
                raise ValueError("local_basis decoder is missing readout weights")
            basis = _local_basis_tensor(
                num_oscillators=self.num_oscillators,
                image_shape=self.image_shape,
                patch_size=self.local_patch_size,
                sigma=self.spatial_basis_sigma,
            )
            oscillator_features = self.state_features(position, velocity).reshape(
                position.shape[0],
                self.num_oscillators,
                2,
            )
            local_drive = jnp.einsum(
                "bnc,ncp->bnp",
                oscillator_features,
                self.local_patch_weights,
            )
            pixels = jnp.einsum("bnp,npi->bi", local_drive, basis)
            pixels = pixels + self.spatial_output_bias
            return _activation(self.output_activation)(pixels)

        if self.decoder_mode == "resize_conv":
            if self.resize_conv_output is None:
                raise ValueError("resize_conv decoder is missing output convolution")
            features = self.state_features(position, velocity).reshape(
                position.shape[0],
                -1,
            )
            channels, seed_h, seed_w = self.resize_conv_seed_shape
            hidden = features.reshape(position.shape[0], channels, seed_h, seed_w)
            for layer_index in range(0, len(self.resize_conv_layers), 2):
                hidden = jnp.repeat(hidden, 2, axis=2)
                hidden = jnp.repeat(hidden, 2, axis=3)
                hidden = jax.vmap(self.resize_conv_layers[layer_index])(hidden)
                hidden = jax.nn.leaky_relu(hidden, negative_slope=0.2)
                hidden = jax.vmap(self.resize_conv_layers[layer_index + 1])(hidden)
                hidden = jax.nn.leaky_relu(hidden, negative_slope=0.2)
            pixels = jax.vmap(self.resize_conv_output)(hidden)
            return _activation(self.output_activation)(
                pixels.reshape(position.shape[0], self.image_dim)
            )

        hidden = self.state_features(position, velocity).reshape(position.shape[0], -1)
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden = jax.vmap(layer)(hidden)
            if layer_index < len(self.decoder_layers) - 1:
                hidden = jax.nn.gelu(hidden)
        return _activation(self.output_activation)(hidden)

    def sample_state(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
        *,
        return_initial: bool = False,
        return_trajectory: bool = False,
    ):
        """Sample initial HORN state and return the evolved final state."""

        state_key, condition_key = jax.random.split(key)
        state0 = self.initial_state(state_key, batch_size, labels)
        if (
            self.conditioning_mode == "class_oscillator"
            and self.num_condition_oscillators > 0
        ):
            condition_state0 = self.initial_condition_state(condition_key, batch_size)
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
                ) = self.evolve_joint_state(
                    state0,
                    condition_state0,
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
            final_position, final_velocity, _, _ = self.evolve_joint_state(
                state0,
                condition_state0,
                labels,
            )
            if return_initial:
                return final_position, final_velocity, state0[0], state0[1]
            return final_position, final_velocity

        if return_trajectory:
            (
                final_position,
                final_velocity,
                position_trajectory,
                velocity_trajectory,
            ) = self.evolve_state(state0, labels, return_trajectory=True)
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
        final_position, final_velocity = self.evolve_state(state0, labels)
        if return_initial:
            return final_position, final_velocity, state0[0], state0[1]
        return final_position, final_velocity

    def collect_trace(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect HORN trajectory and generated samples for diagnostics."""

        state_key, condition_key = jax.random.split(key)
        position0, velocity0 = self.initial_state(state_key, batch_size, labels)
        condition_position0 = jnp.zeros(
            (int(batch_size), self.num_condition_oscillators),
            dtype=position0.dtype,
        )
        condition_velocity0 = jnp.zeros_like(condition_position0)
        final_condition_position = condition_position0
        final_condition_velocity = condition_velocity0
        condition_position_trajectory = jnp.zeros(
            (0, int(batch_size), self.num_condition_oscillators),
            dtype=position0.dtype,
        )
        condition_velocity_trajectory = jnp.zeros_like(condition_position_trajectory)
        if (
            self.conditioning_mode == "class_oscillator"
            and self.num_condition_oscillators > 0
        ):
            condition_position0, condition_velocity0 = self.initial_condition_state(
                condition_key,
                batch_size,
            )
            (
                final_position,
                final_velocity,
                final_condition_position,
                final_condition_velocity,
                position_trajectory,
                velocity_trajectory,
                condition_position_trajectory,
                condition_velocity_trajectory,
            ) = self.evolve_joint_state(
                (position0, velocity0),
                (condition_position0, condition_velocity0),
                labels,
                return_trajectory=True,
            )
        else:
            (
                final_position,
                final_velocity,
                position_trajectory,
                velocity_trajectory,
            ) = self.evolve_state(
                (position0, velocity0),
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
            "condition_initial_theta": condition_position0,
            "condition_theta_trajectory": condition_position_trajectory,
            "condition_final_theta": final_condition_position,
            "condition_initial_velocity": condition_velocity0,
            "condition_velocity_trajectory": condition_velocity_trajectory,
            "condition_final_velocity": final_condition_velocity,
            "generated": generated,
            "output_feedback_drive": output_feedback_drive,
            "output_feedback_gain": self.output_feedback_gain,
            "omega": self.omega,
            "coupling": self.coupling,
            "coupling_profile": self.coupling_profile_matrix(),
            "condition_omega": (
                jnp.zeros((0,), dtype=self.omega.dtype)
                if self.condition_omega is None
                else self.condition_omega
            ),
            "condition_coupling": (
                jnp.zeros((0, 0), dtype=self.coupling.dtype)
                if self.condition_coupling is None
                else self.condition_coupling
            ),
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

    def __call__(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Array:
        final_position, final_velocity = self.sample_state(key, batch_size, labels)
        return self.decode_state(final_position, final_velocity)
