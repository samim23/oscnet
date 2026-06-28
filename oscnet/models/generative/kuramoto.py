"""Kuramoto-style implicit oscillator image generator."""

from __future__ import annotations

from .common import (
    Array,
    Dict,
    Optional,
    Tuple,
    _activation,
    _distance_decay_coupling_profile,
    _local_basis_tensor,
    _local_radius_coupling_profile,
    _softplus_inverse,
    _spatial_basis_matrix,
    eqx,
    jax,
    jnp,
    math,
    phase_features,
    wrap_phase,
)

class KuramotoImageGenerator(eqx.Module):
    """Generate images by evolving random phases through learned coupling.

    The model mirrors the Un-0-style idea at MNIST scale: random initial
    oscillator phases are the generative noise, a coupled Kuramoto system
    transforms those phases, and a small decoder maps the final phase features
    to pixels.
    """

    omega: Array
    coupling: Array
    label_phase_shift: Optional[Array]
    label_condition_phase: Optional[Array]
    condition_omega: Optional[Array]
    condition_coupling: Optional[Array]
    label_condition_coupling: Optional[Array]
    decoder_layers: Tuple[eqx.nn.Linear, ...]
    resize_conv_layers: Tuple[eqx.nn.Conv2d, ...]
    resize_conv_output: Optional[eqx.nn.Conv2d]
    spatial_phase_weights: Optional[Array]
    local_patch_weights: Optional[Array]
    spatial_output_bias: Optional[Array]
    num_oscillators: int = eqx.field(static=True)
    image_shape: Tuple[int, int] = eqx.field(static=True)
    image_dim: int = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    num_condition_oscillators: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    coupling_profile: str = eqx.field(static=True)
    coupling_length_scale: float = eqx.field(static=True)
    coupling_floor: float = eqx.field(static=True)
    coupling_bias_strength: float = eqx.field(static=True)
    conditioning_strength: float = eqx.field(static=True)
    train_dynamics: bool = eqx.field(static=True)
    train_recurrent_dynamics: bool = eqx.field(static=True)
    train_conditioning_dynamics: bool = eqx.field(static=True)
    conditioning_mode: str = eqx.field(static=True)
    readout_mode: str = eqx.field(static=True)
    decoder_mode: str = eqx.field(static=True)
    spatial_basis_sigma: float = eqx.field(static=True)
    local_patch_size: int = eqx.field(static=True)
    resize_conv_seed_shape: Tuple[int, int, int] = eqx.field(static=True)
    resize_conv_upsamples: int = eqx.field(static=True)
    resize_conv_min_channels: int = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_oscillators: int = 64,
        image_shape: Tuple[int, int] = (28, 28),
        decoder_hidden_dim: int = 128,
        decoder_depth: int = 2,
        steps: int = 8,
        dt: float = 0.1,
        coupling_strength: float = 1.0,
        omega_scale: float = 0.2,
        coupling_init_scale: float = 0.05,
        coupling_profile: str = "dense",
        coupling_length_scale: float = 0.0,
        coupling_floor: float = 0.0,
        coupling_bias_strength: float = 0.0,
        conditioning_strength: float = 1.0,
        train_dynamics: bool = True,
        train_recurrent_dynamics: Optional[bool] = None,
        train_conditioning_dynamics: Optional[bool] = None,
        num_classes: int = 0,
        label_phase_scale: float = 0.5,
        num_condition_oscillators: int = 0,
        conditioning_mode: str = "phase_shift",
        readout_mode: str = "absolute",
        decoder_mode: str = "mlp",
        spatial_basis_sigma: float = 0.0,
        local_patch_size: int = 5,
        resize_conv_seed_shape: Tuple[int, int] = (7, 7),
        resize_conv_upsamples: int = 2,
        resize_conv_min_channels: int = 8,
        output_activation: str = "sigmoid",
        output_bias_init: Optional[float] = None,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if num_oscillators < 1:
            raise ValueError("num_oscillators must be positive")
        if decoder_depth < 0:
            raise ValueError("decoder_depth must be non-negative")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative")
        if num_condition_oscillators < 0:
            raise ValueError("num_condition_oscillators must be non-negative")
        if conditioning_mode not in (
            "none",
            "phase_shift",
            "class_coupling",
            "class_oscillator",
        ):
            raise ValueError(
                "conditioning_mode must be 'none', 'phase_shift', or "
                "'class_coupling', or 'class_oscillator'"
            )
        if readout_mode not in (
            "absolute",
            "relative",
            "ref_oscillator",
            "mean_relative",
        ):
            raise ValueError(
                "readout_mode must be 'absolute', 'relative', "
                "'ref_oscillator', or 'mean_relative'"
            )
        if decoder_mode not in (
            "mlp",
            "spatial_basis",
            "local_basis",
            "resize_conv",
        ):
            raise ValueError(
                "decoder_mode must be 'mlp', 'spatial_basis', 'local_basis', "
                "or 'resize_conv'"
            )
        if local_patch_size < 1 or local_patch_size % 2 != 1:
            raise ValueError("local_patch_size must be a positive odd integer")
        seed_h, seed_w = (int(size) for size in resize_conv_seed_shape)
        if seed_h < 1 or seed_w < 1:
            raise ValueError("resize_conv_seed_shape dimensions must be positive")
        if resize_conv_upsamples < 0:
            raise ValueError("resize_conv_upsamples must be non-negative")
        if resize_conv_min_channels < 1:
            raise ValueError("resize_conv_min_channels must be positive")
        if coupling_profile not in ("dense", "distance_decay", "local_radius"):
            raise ValueError(
                "coupling_profile must be 'dense', 'distance_decay', or "
                "'local_radius'"
            )
        if coupling_floor < 0.0 or coupling_floor > 1.0:
            raise ValueError("coupling_floor must be in [0, 1]")
        if conditioning_strength < 0.0:
            raise ValueError("conditioning_strength must be non-negative")
        if key is None:
            key = jax.random.PRNGKey(42)

        self.num_oscillators = int(num_oscillators)
        self.image_shape = tuple(int(size) for size in image_shape)
        self.image_dim = int(image_shape[0] * image_shape[1])
        self.num_classes = int(num_classes)
        self.num_condition_oscillators = int(num_condition_oscillators)
        self.steps = int(steps)
        self.dt = float(dt)
        self.coupling_strength = float(coupling_strength)
        self.coupling_profile = coupling_profile
        self.coupling_length_scale = float(coupling_length_scale)
        self.coupling_floor = float(coupling_floor)
        self.coupling_bias_strength = float(coupling_bias_strength)
        self.conditioning_strength = float(conditioning_strength)
        if train_recurrent_dynamics is None:
            train_recurrent_dynamics = bool(train_dynamics)
        if train_conditioning_dynamics is None:
            train_conditioning_dynamics = bool(train_dynamics)
        self.train_recurrent_dynamics = bool(train_recurrent_dynamics)
        self.train_conditioning_dynamics = bool(train_conditioning_dynamics)
        self.train_dynamics = (
            self.train_recurrent_dynamics or self.train_conditioning_dynamics
        )
        self.conditioning_mode = conditioning_mode
        self.readout_mode = readout_mode
        self.decoder_mode = decoder_mode
        self.spatial_basis_sigma = float(spatial_basis_sigma)
        self.local_patch_size = int(local_patch_size)
        self.resize_conv_upsamples = int(resize_conv_upsamples)
        self.resize_conv_min_channels = int(resize_conv_min_channels)
        self.output_activation = output_activation

        resize_conv_feature_dim = 2 * self.num_oscillators
        resize_conv_seed_pixels = seed_h * seed_w
        if self.decoder_mode == "resize_conv":
            target_h = seed_h * (2**self.resize_conv_upsamples)
            target_w = seed_w * (2**self.resize_conv_upsamples)
            if (target_h, target_w) != self.image_shape:
                raise ValueError(
                    "resize_conv seed shape and upsample count must produce "
                    f"image_shape={self.image_shape}; got {(target_h, target_w)}"
                )
            if resize_conv_feature_dim % resize_conv_seed_pixels != 0:
                raise ValueError(
                    "resize_conv requires 2 * num_oscillators to be divisible "
                    f"by seed_h * seed_w={resize_conv_seed_pixels}; got "
                    f"{resize_conv_feature_dim}"
                )
        resize_conv_seed_channels = max(
            1,
            resize_conv_feature_dim // resize_conv_seed_pixels,
        )
        self.resize_conv_seed_shape = (
            int(resize_conv_seed_channels),
            int(seed_h),
            int(seed_w),
        )

        decoder_key_count = max(
            int(decoder_depth) + 1,
            2 * self.resize_conv_upsamples + 1,
        )
        (
            omega_key,
            coupling_key,
            label_key,
            condition_phase_key,
            condition_omega_key,
            condition_coupling_matrix_key,
            condition_coupling_key,
            spatial_key,
            *decoder_keys,
        ) = jax.random.split(
            key,
            decoder_key_count + 8,
        )
        self.omega = (
            jax.random.normal(omega_key, (self.num_oscillators,))
            * float(omega_scale)
        )
        coupling = (
            jax.random.normal(
                coupling_key,
                (self.num_oscillators, self.num_oscillators),
            )
            * float(coupling_init_scale)
            / jnp.sqrt(float(self.num_oscillators))
        )
        self.coupling = coupling * (
            1.0 - jnp.eye(self.num_oscillators, dtype=jnp.float32)
        )
        self.label_phase_shift = None
        if self.num_classes > 0 and self.conditioning_mode == "phase_shift":
            self.label_phase_shift = (
                jax.random.normal(
                    label_key,
                    (self.num_classes, self.num_oscillators),
                )
                * float(label_phase_scale)
            )
        self.label_condition_phase = None
        self.condition_omega = None
        self.condition_coupling = None
        self.label_condition_coupling = None
        if (
            self.num_classes > 0
            and self.conditioning_mode == "class_coupling"
            and self.num_condition_oscillators > 0
        ):
            self.label_condition_phase = (
                jax.random.uniform(
                    condition_phase_key,
                    (self.num_classes, self.num_condition_oscillators),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                )
                * float(label_phase_scale)
            )
            self.label_condition_coupling = (
                jax.random.normal(
                    condition_coupling_key,
                    (
                        self.num_classes,
                        self.num_oscillators,
                        self.num_condition_oscillators,
                    ),
                )
                * float(coupling_init_scale)
                / jnp.sqrt(float(max(self.num_condition_oscillators, 1)))
            )
        if (
            self.num_classes > 0
            and self.conditioning_mode == "class_oscillator"
            and self.num_condition_oscillators > 0
        ):
            self.condition_omega = (
                jax.random.normal(
                    condition_omega_key,
                    (self.num_condition_oscillators,),
                )
                * float(omega_scale)
            )
            condition_coupling = (
                jax.random.normal(
                    condition_coupling_matrix_key,
                    (
                        self.num_condition_oscillators,
                        self.num_condition_oscillators,
                    ),
                )
                * float(coupling_init_scale)
                / jnp.sqrt(float(max(self.num_condition_oscillators, 1)))
            )
            self.condition_coupling = condition_coupling * (
                1.0
                - jnp.eye(self.num_condition_oscillators, dtype=jnp.float32)
            )
            self.label_condition_coupling = (
                jax.random.normal(
                    condition_coupling_key,
                    (
                        self.num_classes,
                        self.num_oscillators,
                        self.num_condition_oscillators,
                    ),
                )
                * float(coupling_init_scale)
                / jnp.sqrt(float(max(self.num_condition_oscillators, 1)))
            )

        self.spatial_phase_weights = None
        self.local_patch_weights = None
        self.spatial_output_bias = None
        self.resize_conv_layers = ()
        self.resize_conv_output = None
        if self.decoder_mode == "spatial_basis":
            self.decoder_layers = ()
            self.spatial_phase_weights = (
                jax.random.normal(spatial_key, (self.num_oscillators, 2)) * 0.05
            )
            self.spatial_output_bias = jnp.asarray(
                0.0 if output_bias_init is None else float(output_bias_init)
            )
        elif self.decoder_mode == "local_basis":
            self.decoder_layers = ()
            patch_area = self.local_patch_size * self.local_patch_size
            self.local_patch_weights = (
                jax.random.normal(
                    spatial_key,
                    (self.num_oscillators, 2, patch_area),
                )
                * 0.02
            )
            self.spatial_output_bias = jnp.asarray(
                0.0 if output_bias_init is None else float(output_bias_init)
            )
        elif self.decoder_mode == "resize_conv":
            self.decoder_layers = ()
            conv_layers = []
            current_channels = self.resize_conv_seed_shape[0]
            for block_index in range(self.resize_conv_upsamples):
                next_channels = max(
                    current_channels // 2,
                    self.resize_conv_min_channels,
                )
                conv_layers.append(
                    eqx.nn.Conv2d(
                        current_channels,
                        next_channels,
                        kernel_size=3,
                        padding=1,
                        key=decoder_keys[2 * block_index],
                    )
                )
                conv_layers.append(
                    eqx.nn.Conv2d(
                        next_channels,
                        next_channels,
                        kernel_size=3,
                        padding=1,
                        key=decoder_keys[2 * block_index + 1],
                    )
                )
                current_channels = next_channels
            output_conv = eqx.nn.Conv2d(
                current_channels,
                1,
                kernel_size=3,
                padding=1,
                key=decoder_keys[2 * self.resize_conv_upsamples],
            )
            if output_bias_init is not None and output_conv.bias is not None:
                output_conv = eqx.tree_at(
                    lambda conv: conv.bias,
                    output_conv,
                    jnp.full_like(output_conv.bias, float(output_bias_init)),
                )
            self.resize_conv_layers = tuple(conv_layers)
            self.resize_conv_output = output_conv
        else:
            layer_dims = [2 * self.num_oscillators]
            layer_dims.extend([int(decoder_hidden_dim)] * int(decoder_depth))
            layer_dims.append(self.image_dim)
            decoder_layers = []
            for layer_index, (in_size, out_size, layer_key) in enumerate(
                zip(layer_dims[:-1], layer_dims[1:], decoder_keys)
            ):
                layer = eqx.nn.Linear(in_size, out_size, key=layer_key)
                if (
                    output_bias_init is not None
                    and layer_index == len(layer_dims) - 2
                    and layer.bias is not None
                ):
                    layer = eqx.tree_at(
                        lambda linear: linear.bias,
                        layer,
                        jnp.full((out_size,), float(output_bias_init)),
                    )
                decoder_layers.append(layer)
            self.decoder_layers = tuple(decoder_layers)

    def initial_phase(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Array:
        """Sample random oscillator phases in ``[-pi, pi]``."""

        theta = jax.random.uniform(
            key,
            (int(batch_size), self.num_oscillators),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        if self.label_phase_shift is not None and labels is not None:
            label_shift = self.label_phase_shift[labels.astype(jnp.int32)]
            if not self.train_conditioning_dynamics:
                label_shift = jax.lax.stop_gradient(label_shift)
            theta = wrap_phase(theta + label_shift)
        return theta

    def initial_condition_phase(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
    ) -> Array:
        """Sample initial phases for the optional conditioning oscillator pool."""

        return jax.random.uniform(
            key,
            (int(batch_size), self.num_condition_oscillators),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )

    def coupling_profile_matrix(self) -> Array:
        """Return the fixed spatial profile applied to recurrent coupling."""

        if self.coupling_profile == "dense":
            return 1.0 - jnp.eye(self.num_oscillators, dtype=jnp.float32)
        if self.coupling_profile == "local_radius":
            return _local_radius_coupling_profile(
                num_oscillators=self.num_oscillators,
                radius=self.coupling_length_scale,
            )
        return _distance_decay_coupling_profile(
            num_oscillators=self.num_oscillators,
            length_scale=self.coupling_length_scale,
            floor=self.coupling_floor,
        )

    def _dynamics_params(self) -> Tuple[Array, Array]:
        omega = self.omega
        coupling = self.coupling
        if not self.train_recurrent_dynamics:
            omega = jax.lax.stop_gradient(omega)
            coupling = jax.lax.stop_gradient(coupling)
        profile = self.coupling_profile_matrix()
        coupling = coupling * profile
        if self.coupling_bias_strength != 0.0:
            coupling = coupling + (float(self.coupling_bias_strength) * profile)
        return omega, coupling

    def _condition_dynamics_params(self) -> Tuple[Array, Array]:
        if self.condition_omega is None or self.condition_coupling is None:
            return (
                jnp.zeros((0,), dtype=self.omega.dtype),
                jnp.zeros((0, 0), dtype=self.coupling.dtype),
            )
        condition_omega = self.condition_omega
        condition_coupling = self.condition_coupling
        if not self.train_conditioning_dynamics:
            condition_omega = jax.lax.stop_gradient(condition_omega)
            condition_coupling = jax.lax.stop_gradient(condition_coupling)
        condition_coupling = condition_coupling * (
            1.0
            - jnp.eye(self.num_condition_oscillators, dtype=jnp.float32)
        )
        return condition_omega, condition_coupling

    def _conditioning_drive(self, theta: Array, labels: Optional[Array]) -> Array:
        if (
            labels is None
            or self.label_condition_phase is None
            or self.label_condition_coupling is None
        ):
            return jnp.zeros_like(theta)

        condition_phase = self.label_condition_phase[labels.astype(jnp.int32)]
        condition_coupling = self.label_condition_coupling[labels.astype(jnp.int32)]
        if not self.train_conditioning_dynamics:
            condition_phase = jax.lax.stop_gradient(condition_phase)
            condition_coupling = jax.lax.stop_gradient(condition_coupling)

        phase_diff = condition_phase[:, None, :] - theta[:, :, None]
        drive = jnp.sum(condition_coupling * jnp.sin(phase_diff), axis=-1)
        return (
            float(self.conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def _dynamic_conditioning_drive(
        self,
        theta: Array,
        condition_theta: Array,
        labels: Optional[Array],
    ) -> Array:
        if (
            labels is None
            or self.label_condition_coupling is None
            or self.num_condition_oscillators == 0
        ):
            return jnp.zeros_like(theta)

        condition_coupling = self.label_condition_coupling[labels.astype(jnp.int32)]
        if not self.train_conditioning_dynamics:
            condition_coupling = jax.lax.stop_gradient(condition_coupling)
        phase_diff = condition_theta[:, None, :] - theta[:, :, None]
        drive = jnp.sum(condition_coupling * jnp.sin(phase_diff), axis=-1)
        return (
            float(self.conditioning_strength)
            * drive
            / float(max(self.num_condition_oscillators, 1))
        )

    def step(self, theta: Array, labels: Optional[Array] = None) -> Array:
        """Advance one Kuramoto step for a batch of phases."""

        omega, coupling = self._dynamics_params()
        phase_diff = theta[:, None, :] - theta[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * jnp.sin(phase_diff), axis=-1)
        condition_drive = self._conditioning_drive(theta, labels)
        velocity = omega[None, :] + (
            self.coupling_strength
            * (interaction / float(self.num_oscillators) + condition_drive)
        )
        return wrap_phase(theta + self.dt * velocity)

    def step_joint(
        self,
        theta: Array,
        condition_theta: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance main and conditioning oscillator pools by one Euler step."""

        omega, coupling = self._dynamics_params()
        phase_diff = theta[:, None, :] - theta[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * jnp.sin(phase_diff), axis=-1)
        condition_drive = self._dynamic_conditioning_drive(
            theta,
            condition_theta,
            labels,
        )
        main_velocity = omega[None, :] + (
            self.coupling_strength
            * (interaction / float(self.num_oscillators) + condition_drive)
        )

        condition_omega, condition_coupling = self._condition_dynamics_params()
        condition_phase_diff = condition_theta[:, None, :] - condition_theta[:, :, None]
        condition_interaction = jnp.sum(
            condition_coupling[None, :, :] * jnp.sin(condition_phase_diff),
            axis=-1,
        )
        condition_velocity = condition_omega[None, :] + (
            self.coupling_strength
            * (
                condition_interaction
                / float(max(self.num_condition_oscillators, 1))
            )
        )
        return (
            wrap_phase(theta + self.dt * main_velocity),
            wrap_phase(condition_theta + self.dt * condition_velocity),
        )

    def evolve(
        self,
        theta0: Array,
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ) -> Array | Tuple[Array, Array]:
        """Evolve initial phases for the configured scan length."""

        if self.steps == 0:
            empty = jnp.zeros((0, *theta0.shape), dtype=theta0.dtype)
            return (theta0, empty) if return_trajectory else theta0

        def scan_fn(theta, _):
            next_theta = self.step(theta, labels)
            return next_theta, next_theta

        final_theta, trajectory = jax.lax.scan(
            scan_fn,
            theta0,
            xs=None,
            length=self.steps,
        )
        return (final_theta, trajectory) if return_trajectory else final_theta

    def evolve_joint(
        self,
        theta0: Array,
        condition_theta0: Array,
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ) -> Tuple[Array, Array] | Tuple[Array, Array, Array, Array]:
        """Evolve main and conditioning oscillator pools together."""

        if self.steps == 0:
            empty_main = jnp.zeros((0, *theta0.shape), dtype=theta0.dtype)
            empty_condition = jnp.zeros(
                (0, *condition_theta0.shape),
                dtype=condition_theta0.dtype,
            )
            if return_trajectory:
                return theta0, condition_theta0, empty_main, empty_condition
            return theta0, condition_theta0

        def scan_fn(carry, _):
            theta, condition_theta = carry
            next_theta, next_condition_theta = self.step_joint(
                theta,
                condition_theta,
                labels,
            )
            return (
                next_theta,
                next_condition_theta,
            ), (
                next_theta,
                next_condition_theta,
            )

        (final_theta, final_condition_theta), (
            trajectory,
            condition_trajectory,
        ) = jax.lax.scan(
            scan_fn,
            (theta0, condition_theta0),
            xs=None,
            length=self.steps,
        )
        if return_trajectory:
            return final_theta, final_condition_theta, trajectory, condition_trajectory
        return final_theta, final_condition_theta

    def decode_phase(self, theta: Array) -> Array:
        """Decode final phase features into flat images."""

        if self.readout_mode in ("relative", "ref_oscillator"):
            theta = wrap_phase(theta - theta[:, :1])
        elif self.readout_mode == "mean_relative":
            theta = wrap_phase(theta - jnp.mean(theta, axis=-1, keepdims=True))
        if self.decoder_mode == "spatial_basis":
            if self.spatial_phase_weights is None or self.spatial_output_bias is None:
                raise ValueError("spatial_basis decoder is missing readout weights")
            basis = _spatial_basis_matrix(
                num_oscillators=self.num_oscillators,
                image_shape=self.image_shape,
                sigma=self.spatial_basis_sigma,
            )
            oscillator_features = phase_features(theta).reshape(
                theta.shape[0],
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
            oscillator_features = phase_features(theta).reshape(
                theta.shape[0],
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
            features = phase_features(theta).reshape(theta.shape[0], -1)
            channels, seed_h, seed_w = self.resize_conv_seed_shape
            hidden = features.reshape(theta.shape[0], channels, seed_h, seed_w)
            for layer_index in range(0, len(self.resize_conv_layers), 2):
                hidden = jnp.repeat(hidden, 2, axis=2)
                hidden = jnp.repeat(hidden, 2, axis=3)
                hidden = jax.vmap(self.resize_conv_layers[layer_index])(hidden)
                hidden = jax.nn.leaky_relu(hidden, negative_slope=0.2)
                hidden = jax.vmap(self.resize_conv_layers[layer_index + 1])(hidden)
                hidden = jax.nn.leaky_relu(hidden, negative_slope=0.2)
            pixels = jax.vmap(self.resize_conv_output)(hidden)
            return _activation(self.output_activation)(
                pixels.reshape(theta.shape[0], self.image_dim)
            )

        hidden = phase_features(theta).reshape(theta.shape[0], -1)
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden = jax.vmap(layer)(hidden)
            if layer_index < len(self.decoder_layers) - 1:
                hidden = jax.nn.gelu(hidden)
        return _activation(self.output_activation)(hidden)

    def sample_phase(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
        *,
        return_initial: bool = False,
        return_trajectory: bool = False,
    ) -> Array | Tuple[Array, Array] | Tuple[Array, Array, Array]:
        """Sample initial phases and return the evolved final phase."""

        theta_key, condition_key = jax.random.split(key)
        theta0 = self.initial_phase(theta_key, batch_size, labels)
        if (
            self.conditioning_mode == "class_oscillator"
            and self.num_condition_oscillators > 0
        ):
            condition_theta0 = self.initial_condition_phase(condition_key, batch_size)
            if return_trajectory:
                final_theta, _, trajectory, _ = self.evolve_joint(
                    theta0,
                    condition_theta0,
                    labels,
                    return_trajectory=True,
                )
                if return_initial:
                    return final_theta, theta0, trajectory
                return final_theta, trajectory
            final_theta, _ = self.evolve_joint(theta0, condition_theta0, labels)
            if return_initial:
                return final_theta, theta0
            return final_theta

        if return_trajectory:
            final_theta, trajectory = self.evolve(
                theta0,
                labels,
                return_trajectory=True,
            )
            if return_initial:
                return final_theta, theta0, trajectory
            return final_theta, trajectory
        final_theta = self.evolve(theta0, labels)
        if return_initial:
            return final_theta, theta0
        return final_theta

    def collect_trace(
        self,
        key: jax.random.PRNGKey,
        batch_size: int,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect phase trajectory and generated samples for diagnostics."""

        theta_key, condition_key = jax.random.split(key)
        theta0 = self.initial_phase(theta_key, batch_size, labels)
        condition_theta0 = jnp.zeros(
            (int(batch_size), self.num_condition_oscillators),
            dtype=theta0.dtype,
        )
        final_condition_theta = condition_theta0
        condition_trajectory = jnp.zeros(
            (0, int(batch_size), self.num_condition_oscillators),
            dtype=theta0.dtype,
        )
        if (
            self.conditioning_mode == "class_oscillator"
            and self.num_condition_oscillators > 0
        ):
            condition_theta0 = self.initial_condition_phase(
                condition_key,
                batch_size,
            )
            (
                final_theta,
                final_condition_theta,
                trajectory,
                condition_trajectory,
            ) = self.evolve_joint(
                theta0,
                condition_theta0,
                labels,
                return_trajectory=True,
            )
        else:
            final_theta, trajectory = self.evolve(
                theta0,
                labels,
                return_trajectory=True,
            )
        generated = self.decode_phase(final_theta)
        return {
            "initial_theta": theta0,
            "theta_trajectory": trajectory,
            "final_theta": final_theta,
            "condition_initial_theta": condition_theta0,
            "condition_theta_trajectory": condition_trajectory,
            "condition_final_theta": final_condition_theta,
            "generated": generated,
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
        final_theta = self.sample_phase(key, batch_size, labels)
        return self.decode_phase(final_theta)
