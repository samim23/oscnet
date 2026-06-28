"""Oscillatory phase-rate fields for rectified-flow image generation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.models.winfree import wrap_phase

Array = jnp.ndarray


def _apply_same_conv2d(kernel: Array, bias: Array, grid: Array) -> Array:
    """Apply a same-padded convolution to NHWC grid data."""

    field = jax.lax.conv_general_dilated(
        grid,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return field + bias[None, None, None, :]


def _apply_linear_last(linear: eqx.nn.Linear, x: Array) -> Array:
    """Apply an Equinox linear layer over the last dimension."""

    flat = x.reshape((-1, x.shape[-1]))
    out = jax.vmap(linear)(flat)
    return out.reshape((*x.shape[:-1], out.shape[-1]))


def _mean_pool_to_grid(grid: Array, grid_size: int) -> Array:
    """Average-pool an NHWC grid to a square coarse grid."""

    batch_size, height, width, channels = grid.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError("grid_size must divide image height and width")
    block_h = height // grid_size
    block_w = width // grid_size
    return grid.reshape(
        batch_size,
        grid_size,
        block_h,
        grid_size,
        block_w,
        channels,
    ).mean(axis=(2, 4))


def _upsample_square_grid(grid: Array, image_shape: Tuple[int, int]) -> Array:
    """Nearest-neighbor upsample a square NHWC grid to image_shape."""

    height, width = image_shape
    grid_size = grid.shape[1]
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError("coarse grid size must divide image height and width")
    block_h = height // grid_size
    block_w = width // grid_size
    return jnp.repeat(jnp.repeat(grid, block_h, axis=1), block_w, axis=2)


def _spatial_phase_features(
    height: int,
    width: int,
    *,
    dtype=jnp.float32,
) -> Array:
    """Return fixed 2D coordinate/phase features as an HWC grid."""

    y = jnp.linspace(-1.0, 1.0, int(height), dtype=dtype)
    x = jnp.linspace(-1.0, 1.0, int(width), dtype=dtype)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack(
        [
            xx,
            yy,
            jnp.sin(jnp.pi * xx),
            jnp.cos(jnp.pi * xx),
            jnp.sin(jnp.pi * yy),
            jnp.cos(jnp.pi * yy),
        ],
        axis=-1,
    )


class PhaseRateFlowField(eqx.Module):
    """Local oscillator field that predicts rectified-flow image velocity.

    The model treats the noisy image as a visible oscillator medium. Pixel
    values and flow time initialize a phase-rate field on the image lattice,
    local phase/rate coupling evolves it, and a local readout predicts the
    velocity that transports noise toward MNIST data.
    """

    input_projection: eqx.nn.Linear
    time_to_omega: eqx.nn.Linear
    readout: eqx.nn.Linear
    omega: Array
    phase_kernel: Array
    phase_bias: Array
    rate_kernel: Array
    rate_bias: Array
    phase_to_rate: Array
    rate_to_phase: Array
    label_phase_bias: Optional[Array]
    label_rate_bias: Optional[Array]
    label_omega_bias: Optional[Array]

    image_shape: Tuple[int, int] = eqx.field(static=True)
    image_dim: int = eqx.field(static=True)
    field_channels: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    rate_update: float = eqx.field(static=True)
    input_drive_strength: float = eqx.field(static=True)
    train_dynamics: bool = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    position_features: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int] = (28, 28),
        field_channels: int = 8,
        steps: int = 8,
        kernel_size: int = 3,
        dt: float = 0.15,
        coupling_strength: float = 1.0,
        rate_update: float = 0.5,
        input_drive_strength: float = 0.5,
        omega_scale: float = 0.2,
        kernel_init_scale: float = 0.05,
        train_dynamics: bool = True,
        num_classes: int = 10,
        position_features: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if field_channels < 1:
            raise ValueError("field_channels must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative")
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, 9)
        height, width = image_shape
        conv_scale = float(kernel_init_scale) / jnp.sqrt(
            float(kernel_size * kernel_size * field_channels)
        )

        self.image_shape = tuple(int(size) for size in image_shape)
        self.image_dim = int(height * width)
        self.field_channels = int(field_channels)
        self.steps = int(steps)
        self.kernel_size = int(kernel_size)
        self.dt = float(dt)
        self.coupling_strength = float(coupling_strength)
        self.rate_update = float(rate_update)
        self.input_drive_strength = float(input_drive_strength)
        self.train_dynamics = bool(train_dynamics)
        self.num_classes = int(num_classes)
        self.position_features = bool(position_features)

        input_dim = 10 if self.position_features else 4
        self.input_projection = eqx.nn.Linear(
            input_dim,
            2 * field_channels,
            key=keys[0],
        )
        self.time_to_omega = eqx.nn.Linear(3, field_channels, key=keys[1])
        self.readout = eqx.nn.Linear(3 * field_channels, 1, key=keys[2])
        self.omega = (
            jax.random.normal(keys[3], (field_channels,)) * float(omega_scale)
        )
        self.phase_kernel = (
            jax.random.normal(
                keys[4],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.phase_bias = jnp.zeros((field_channels,))
        self.rate_kernel = (
            jax.random.normal(
                keys[5],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.rate_bias = jnp.zeros((field_channels,))
        self.phase_to_rate = (
            jax.random.normal(keys[6], (field_channels,)) * float(kernel_init_scale)
        )
        self.rate_to_phase = (
            jax.random.normal(keys[7], (field_channels,)) * float(kernel_init_scale)
        )
        if self.num_classes > 0:
            self.label_phase_bias = (
                jax.random.normal(keys[8], (self.num_classes, field_channels)) * 0.05
            )
            self.label_rate_bias = jnp.zeros((self.num_classes, field_channels))
            self.label_omega_bias = jnp.zeros((self.num_classes, field_channels))
        else:
            self.label_phase_bias = None
            self.label_rate_bias = None
            self.label_omega_bias = None

    def _time_features(self, t: Array) -> Array:
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = t[None]
        return jnp.stack(
            [
                t,
                jnp.sin(jnp.pi * t),
                jnp.cos(jnp.pi * t),
            ],
            axis=-1,
        )

    def initial_state(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """Initialize phase/rate state from a noisy image and flow time."""

        batch_size = images.shape[0]
        height, width = self.image_shape
        image_grid = images.reshape(batch_size, height, width, 1)
        time_features = self._time_features(t)
        time_grid = jnp.broadcast_to(
            time_features[:, None, None, :],
            (batch_size, height, width, 3),
        )
        inputs = jnp.concatenate([image_grid, time_grid], axis=-1)
        if self.position_features:
            position_grid = _spatial_phase_features(
                height,
                width,
                dtype=images.dtype,
            )
            position_grid = jnp.broadcast_to(
                position_grid[None, :, :, :],
                (batch_size, height, width, position_grid.shape[-1]),
            )
            inputs = jnp.concatenate([inputs, position_grid], axis=-1)
        projected = _apply_linear_last(self.input_projection, inputs)
        theta_raw, rate_raw = jnp.split(projected, 2, axis=-1)

        if labels is not None and self.label_phase_bias is not None:
            label_indices = labels.astype(jnp.int32)
            theta_raw = theta_raw + self.label_phase_bias[label_indices][
                :, None, None, :
            ]
            rate_raw = rate_raw + self.label_rate_bias[label_indices][
                :, None, None, :
            ]
        theta = wrap_phase(theta_raw)
        rate = jnp.tanh(rate_raw)
        return theta, rate, rate, time_features

    def _dynamics_params(self):
        omega = self.omega
        phase_kernel = self.phase_kernel
        phase_bias = self.phase_bias
        rate_kernel = self.rate_kernel
        rate_bias = self.rate_bias
        phase_to_rate = self.phase_to_rate
        rate_to_phase = self.rate_to_phase
        label_omega_bias = self.label_omega_bias
        if not self.train_dynamics:
            omega = jax.lax.stop_gradient(omega)
            phase_kernel = jax.lax.stop_gradient(phase_kernel)
            phase_bias = jax.lax.stop_gradient(phase_bias)
            rate_kernel = jax.lax.stop_gradient(rate_kernel)
            rate_bias = jax.lax.stop_gradient(rate_bias)
            phase_to_rate = jax.lax.stop_gradient(phase_to_rate)
            rate_to_phase = jax.lax.stop_gradient(rate_to_phase)
            if label_omega_bias is not None:
                label_omega_bias = jax.lax.stop_gradient(label_omega_bias)
        return (
            omega,
            phase_kernel,
            phase_bias,
            rate_kernel,
            rate_bias,
            phase_to_rate,
            rate_to_phase,
            label_omega_bias,
        )

    def step(
        self,
        theta: Array,
        rate: Array,
        drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance one local phase-rate oscillator step."""

        (
            omega,
            phase_kernel,
            phase_bias,
            rate_kernel,
            rate_bias,
            phase_to_rate,
            rate_to_phase,
            label_omega_bias,
        ) = self._dynamics_params()
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        neighbor_sin = _apply_same_conv2d(phase_kernel, phase_bias, sin_theta)
        neighbor_cos = _apply_same_conv2d(
            phase_kernel,
            jnp.zeros_like(phase_bias),
            cos_theta,
        )
        interaction = neighbor_sin * cos_theta - neighbor_cos * sin_theta
        rate_context = _apply_same_conv2d(rate_kernel, rate_bias, rate)
        time_drive = _apply_linear_last(self.time_to_omega, time_features)
        if not self.train_dynamics:
            time_drive = jax.lax.stop_gradient(time_drive)
        omega_drive = omega[None, None, None, :] + time_drive[:, None, None, :]
        if labels is not None and label_omega_bias is not None:
            omega_drive = omega_drive + label_omega_bias[labels.astype(jnp.int32)][
                :, None, None, :
            ]
        velocity = (
            omega_drive
            + self.coupling_strength * interaction
            + rate_to_phase[None, None, None, :] * rate
        )
        theta = wrap_phase(theta + self.dt * velocity)
        proposal = jnp.tanh(
            rate_context
            + phase_to_rate[None, None, None, :] * interaction
            + self.input_drive_strength * drive
        )
        update = jnp.clip(self.rate_update, 0.0, 1.0)
        rate = (1.0 - update) * rate + update * proposal
        return theta, rate

    def evolve(
        self,
        theta: Array,
        rate: Array,
        drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve the phase-rate field through the configured recurrent steps."""

        if self.steps == 0:
            if return_trajectory:
                empty_theta = jnp.zeros((0, *theta.shape), dtype=theta.dtype)
                empty_rate = jnp.zeros((0, *rate.shape), dtype=rate.dtype)
                return theta, rate, empty_theta, empty_rate
            return theta, rate

        def scan_fn(carry, _):
            current_theta, current_rate = carry
            next_theta, next_rate = self.step(
                current_theta,
                current_rate,
                drive,
                time_features,
                labels,
            )
            return (next_theta, next_rate), (next_theta, next_rate)

        (final_theta, final_rate), (theta_traj, rate_traj) = jax.lax.scan(
            scan_fn,
            (theta, rate),
            xs=None,
            length=self.steps,
        )
        if return_trajectory:
            return final_theta, final_rate, theta_traj, rate_traj
        return final_theta, final_rate

    def readout_velocity(self, theta: Array, rate: Array) -> Array:
        """Read a flat image velocity from local oscillator features."""

        features = jnp.concatenate([jnp.sin(theta), jnp.cos(theta), rate], axis=-1)
        velocity_grid = _apply_linear_last(self.readout, features)
        return velocity_grid.reshape(theta.shape[0], self.image_dim)

    def __call__(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
        *,
        return_trace: bool = False,
    ):
        """Predict rectified-flow velocity for a batch of noisy images."""

        theta0, rate0, drive, time_features = self.initial_state(images, t, labels)
        if return_trace:
            theta, rate, theta_traj, rate_traj = self.evolve(
                theta0,
                rate0,
                drive,
                time_features,
                labels,
                return_trajectory=True,
            )
        else:
            theta, rate = self.evolve(theta0, rate0, drive, time_features, labels)
            theta_traj = rate_traj = None
        velocity = self.readout_velocity(theta, rate)
        if return_trace:
            return velocity, {
                "initial_theta": theta0,
                "initial_rate": rate0,
                "theta_trajectory": theta_traj,
                "rate_trajectory": rate_traj,
                "final_theta": theta,
                "final_rate": rate,
                "velocity": velocity,
            }
        return velocity

    def predict_clean(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Array:
        """Predict the clean endpoint implied by rectified-flow velocity."""

        velocity = self(images, t, labels)
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (images.shape[0],))
        return images + (1.0 - t[:, None]) * velocity

    def sample(
        self,
        key: jax.random.PRNGKey,
        sample_count: int,
        *,
        labels: Optional[Array] = None,
        outer_steps: int = 16,
    ) -> Array:
        """Generate images by integrating the learned rectified-flow field."""

        sample_count = int(sample_count)
        if outer_steps < 1:
            raise ValueError("outer_steps must be positive")
        x = jax.random.normal(key, (sample_count, self.image_dim))
        if labels is not None:
            labels = labels.astype(jnp.int32)
        step_size = 1.0 / float(outer_steps)

        def scan_fn(current_x, step_index):
            t_value = (step_index.astype(current_x.dtype) + 0.5) * step_size
            t = jnp.full((sample_count,), t_value, dtype=current_x.dtype)
            velocity = self(current_x, t, labels)
            return current_x + step_size * velocity, None

        steps = jnp.arange(int(outer_steps), dtype=jnp.float32)
        x, _ = jax.lax.scan(scan_fn, x, steps)
        return jnp.clip(x, 0.0, 1.0)

    def collect_trace(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect oscillator trajectory diagnostics for a noisy batch."""

        velocity, trace = self(images, t, labels, return_trace=True)
        clean = images + (1.0 - jnp.asarray(t)[:, None]) * velocity
        trace["input"] = images
        trace["predicted_clean"] = clean
        return trace


class CoarseGlobalPhaseRateFlowField(eqx.Module):
    """Fine phase-rate flow field coordinated by a coarse oscillator band.

    The fine field does the local stroke work. A lower-resolution phase-rate
    field evolves in parallel and pulls fine phases through relative-phase
    coupling. This is the ONN-native global-closure probe: no U-Net skip, no
    latent decoder, just a slower spatial carrier communicating across larger
    image regions.
    """

    fine: PhaseRateFlowField
    coarse_projection: eqx.nn.Linear
    coarse_time_to_omega: eqx.nn.Linear
    coarse_omega: Array
    coarse_phase_kernel: Array
    coarse_phase_bias: Array
    coarse_rate_kernel: Array
    coarse_rate_bias: Array
    coarse_phase_to_rate: Array
    coarse_rate_to_phase: Array
    fine_phase_gain: Array
    fine_rate_gain: Array
    label_coarse_phase_bias: Optional[Array]
    label_coarse_rate_bias: Optional[Array]
    label_coarse_omega_bias: Optional[Array]

    image_shape: Tuple[int, int] = eqx.field(static=True)
    image_dim: int = eqx.field(static=True)
    field_channels: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    coarse_grid_size: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    rate_update: float = eqx.field(static=True)
    input_drive_strength: float = eqx.field(static=True)
    global_coupling_strength: float = eqx.field(static=True)
    train_dynamics: bool = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    position_features: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int] = (28, 28),
        field_channels: int = 8,
        steps: int = 8,
        kernel_size: int = 3,
        coarse_grid_size: int = 4,
        dt: float = 0.15,
        coupling_strength: float = 1.0,
        rate_update: float = 0.5,
        input_drive_strength: float = 0.5,
        global_coupling_strength: float = 0.5,
        omega_scale: float = 0.2,
        kernel_init_scale: float = 0.05,
        train_dynamics: bool = True,
        num_classes: int = 10,
        position_features: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if field_channels < 1:
            raise ValueError("field_channels must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if coarse_grid_size < 1:
            raise ValueError("coarse_grid_size must be positive")
        height, width = image_shape
        if height % coarse_grid_size != 0 or width % coarse_grid_size != 0:
            raise ValueError("coarse_grid_size must divide image height and width")
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, 12)
        conv_scale = float(kernel_init_scale) / jnp.sqrt(
            float(kernel_size * kernel_size * field_channels)
        )

        self.image_shape = tuple(int(size) for size in image_shape)
        self.image_dim = int(height * width)
        self.field_channels = int(field_channels)
        self.steps = int(steps)
        self.kernel_size = int(kernel_size)
        self.coarse_grid_size = int(coarse_grid_size)
        self.dt = float(dt)
        self.coupling_strength = float(coupling_strength)
        self.rate_update = float(rate_update)
        self.input_drive_strength = float(input_drive_strength)
        self.global_coupling_strength = float(global_coupling_strength)
        self.train_dynamics = bool(train_dynamics)
        self.num_classes = int(num_classes)
        self.position_features = bool(position_features)

        self.fine = PhaseRateFlowField(
            image_shape=image_shape,
            field_channels=field_channels,
            steps=steps,
            kernel_size=kernel_size,
            dt=dt,
            coupling_strength=coupling_strength,
            rate_update=rate_update,
            input_drive_strength=input_drive_strength,
            omega_scale=omega_scale,
            kernel_init_scale=kernel_init_scale,
            train_dynamics=train_dynamics,
            num_classes=num_classes,
            position_features=position_features,
            key=keys[0],
        )
        input_dim = 10 if self.position_features else 4
        self.coarse_projection = eqx.nn.Linear(
            input_dim,
            2 * field_channels,
            key=keys[1],
        )
        self.coarse_time_to_omega = eqx.nn.Linear(3, field_channels, key=keys[2])
        self.coarse_omega = (
            jax.random.normal(keys[3], (field_channels,)) * float(omega_scale)
        )
        self.coarse_phase_kernel = (
            jax.random.normal(
                keys[4],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.coarse_phase_bias = jnp.zeros((field_channels,))
        self.coarse_rate_kernel = (
            jax.random.normal(
                keys[5],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.coarse_rate_bias = jnp.zeros((field_channels,))
        self.coarse_phase_to_rate = (
            jax.random.normal(keys[6], (field_channels,)) * float(kernel_init_scale)
        )
        self.coarse_rate_to_phase = (
            jax.random.normal(keys[7], (field_channels,)) * float(kernel_init_scale)
        )
        self.fine_phase_gain = jnp.ones((field_channels,)) * 0.25
        self.fine_rate_gain = jax.random.normal(keys[8], (field_channels,)) * float(
            kernel_init_scale
        )
        if self.num_classes > 0:
            self.label_coarse_phase_bias = (
                jax.random.normal(keys[9], (self.num_classes, field_channels)) * 0.05
            )
            self.label_coarse_rate_bias = jnp.zeros((self.num_classes, field_channels))
            self.label_coarse_omega_bias = jnp.zeros((self.num_classes, field_channels))
        else:
            self.label_coarse_phase_bias = None
            self.label_coarse_rate_bias = None
            self.label_coarse_omega_bias = None

    def _time_features(self, t: Array) -> Array:
        return self.fine._time_features(t)

    def initial_coarse_state(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array, Array]:
        """Initialize the coarse phase/rate state from pooled noisy images."""

        batch_size = images.shape[0]
        height, width = self.image_shape
        image_grid = images.reshape(batch_size, height, width, 1)
        coarse_image = _mean_pool_to_grid(image_grid, self.coarse_grid_size)
        time_features = self._time_features(t)
        time_grid = jnp.broadcast_to(
            time_features[:, None, None, :],
            (
                batch_size,
                self.coarse_grid_size,
                self.coarse_grid_size,
                3,
            ),
        )
        inputs = jnp.concatenate([coarse_image, time_grid], axis=-1)
        if self.position_features:
            position_grid = _spatial_phase_features(
                self.coarse_grid_size,
                self.coarse_grid_size,
                dtype=images.dtype,
            )
            position_grid = jnp.broadcast_to(
                position_grid[None, :, :, :],
                (
                    batch_size,
                    self.coarse_grid_size,
                    self.coarse_grid_size,
                    position_grid.shape[-1],
                ),
            )
            inputs = jnp.concatenate([inputs, position_grid], axis=-1)
        projected = _apply_linear_last(self.coarse_projection, inputs)
        theta_raw, rate_raw = jnp.split(projected, 2, axis=-1)
        if labels is not None and self.label_coarse_phase_bias is not None:
            label_indices = labels.astype(jnp.int32)
            theta_raw = theta_raw + self.label_coarse_phase_bias[label_indices][
                :, None, None, :
            ]
            rate_raw = rate_raw + self.label_coarse_rate_bias[label_indices][
                :, None, None, :
            ]
        theta = wrap_phase(theta_raw)
        rate = jnp.tanh(rate_raw)
        return theta, rate, rate, time_features

    def _coarse_dynamics_params(self):
        omega = self.coarse_omega
        phase_kernel = self.coarse_phase_kernel
        phase_bias = self.coarse_phase_bias
        rate_kernel = self.coarse_rate_kernel
        rate_bias = self.coarse_rate_bias
        phase_to_rate = self.coarse_phase_to_rate
        rate_to_phase = self.coarse_rate_to_phase
        fine_phase_gain = self.fine_phase_gain
        fine_rate_gain = self.fine_rate_gain
        label_omega_bias = self.label_coarse_omega_bias
        if not self.train_dynamics:
            omega = jax.lax.stop_gradient(omega)
            phase_kernel = jax.lax.stop_gradient(phase_kernel)
            phase_bias = jax.lax.stop_gradient(phase_bias)
            rate_kernel = jax.lax.stop_gradient(rate_kernel)
            rate_bias = jax.lax.stop_gradient(rate_bias)
            phase_to_rate = jax.lax.stop_gradient(phase_to_rate)
            rate_to_phase = jax.lax.stop_gradient(rate_to_phase)
            fine_phase_gain = jax.lax.stop_gradient(fine_phase_gain)
            fine_rate_gain = jax.lax.stop_gradient(fine_rate_gain)
            if label_omega_bias is not None:
                label_omega_bias = jax.lax.stop_gradient(label_omega_bias)
        return (
            omega,
            phase_kernel,
            phase_bias,
            rate_kernel,
            rate_bias,
            phase_to_rate,
            rate_to_phase,
            fine_phase_gain,
            fine_rate_gain,
            label_omega_bias,
        )

    def coarse_step(
        self,
        theta: Array,
        rate: Array,
        drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance the coarse phase-rate oscillator field one step."""

        (
            omega,
            phase_kernel,
            phase_bias,
            rate_kernel,
            rate_bias,
            phase_to_rate,
            rate_to_phase,
            _fine_phase_gain,
            _fine_rate_gain,
            label_omega_bias,
        ) = self._coarse_dynamics_params()
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        neighbor_sin = _apply_same_conv2d(phase_kernel, phase_bias, sin_theta)
        neighbor_cos = _apply_same_conv2d(
            phase_kernel,
            jnp.zeros_like(phase_bias),
            cos_theta,
        )
        interaction = neighbor_sin * cos_theta - neighbor_cos * sin_theta
        rate_context = _apply_same_conv2d(rate_kernel, rate_bias, rate)
        time_drive = _apply_linear_last(self.coarse_time_to_omega, time_features)
        if not self.train_dynamics:
            time_drive = jax.lax.stop_gradient(time_drive)
        omega_drive = omega[None, None, None, :] + time_drive[:, None, None, :]
        if labels is not None and label_omega_bias is not None:
            omega_drive = omega_drive + label_omega_bias[labels.astype(jnp.int32)][
                :, None, None, :
            ]
        velocity = (
            omega_drive
            + self.coupling_strength * interaction
            + rate_to_phase[None, None, None, :] * rate
        )
        theta = wrap_phase(theta + self.dt * velocity)
        proposal = jnp.tanh(
            rate_context
            + phase_to_rate[None, None, None, :] * interaction
            + self.input_drive_strength * drive
        )
        update = jnp.clip(self.rate_update, 0.0, 1.0)
        rate = (1.0 - update) * rate + update * proposal
        return theta, rate

    def apply_global_coupling(
        self,
        theta: Array,
        rate: Array,
        coarse_theta: Array,
    ) -> Tuple[Array, Array]:
        """Pull fine phase/rate state toward the upsampled coarse phase state."""

        (
            _omega,
            _phase_kernel,
            _phase_bias,
            _rate_kernel,
            _rate_bias,
            _phase_to_rate,
            _rate_to_phase,
            fine_phase_gain,
            fine_rate_gain,
            _label_omega_bias,
        ) = self._coarse_dynamics_params()
        coarse_up = _upsample_square_grid(coarse_theta, self.image_shape)
        global_interaction = jnp.sin(coarse_up) * jnp.cos(theta) - jnp.cos(
            coarse_up
        ) * jnp.sin(theta)
        theta = wrap_phase(
            theta
            + self.dt
            * self.global_coupling_strength
            * fine_phase_gain[None, None, None, :]
            * global_interaction
        )
        rate_proposal = jnp.tanh(
            rate + fine_rate_gain[None, None, None, :] * global_interaction
        )
        update = jnp.clip(self.rate_update, 0.0, 1.0)
        rate = (1.0 - update) * rate + update * rate_proposal
        return theta, rate

    def evolve(
        self,
        theta: Array,
        rate: Array,
        drive: Array,
        coarse_theta: Array,
        coarse_rate: Array,
        coarse_drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve fine and coarse phase-rate fields together."""

        if self.steps == 0:
            if return_trajectory:
                empty_theta = jnp.zeros((0, *theta.shape), dtype=theta.dtype)
                empty_rate = jnp.zeros((0, *rate.shape), dtype=rate.dtype)
                empty_coarse_theta = jnp.zeros(
                    (0, *coarse_theta.shape),
                    dtype=coarse_theta.dtype,
                )
                empty_coarse_rate = jnp.zeros(
                    (0, *coarse_rate.shape),
                    dtype=coarse_rate.dtype,
                )
                return (
                    theta,
                    rate,
                    coarse_theta,
                    coarse_rate,
                    empty_theta,
                    empty_rate,
                    empty_coarse_theta,
                    empty_coarse_rate,
                )
            return theta, rate, coarse_theta, coarse_rate

        def scan_fn(carry, _):
            current_theta, current_rate, current_coarse_theta, current_coarse_rate = (
                carry
            )
            next_coarse_theta, next_coarse_rate = self.coarse_step(
                current_coarse_theta,
                current_coarse_rate,
                coarse_drive,
                time_features,
                labels,
            )
            next_theta, next_rate = self.fine.step(
                current_theta,
                current_rate,
                drive,
                time_features,
                labels,
            )
            next_theta, next_rate = self.apply_global_coupling(
                next_theta,
                next_rate,
                next_coarse_theta,
            )
            carry = (next_theta, next_rate, next_coarse_theta, next_coarse_rate)
            return carry, carry

        (
            final_theta,
            final_rate,
            final_coarse_theta,
            final_coarse_rate,
        ), trajectories = jax.lax.scan(
            scan_fn,
            (theta, rate, coarse_theta, coarse_rate),
            xs=None,
            length=self.steps,
        )
        theta_traj, rate_traj, coarse_theta_traj, coarse_rate_traj = trajectories
        if return_trajectory:
            return (
                final_theta,
                final_rate,
                final_coarse_theta,
                final_coarse_rate,
                theta_traj,
                rate_traj,
                coarse_theta_traj,
                coarse_rate_traj,
            )
        return final_theta, final_rate, final_coarse_theta, final_coarse_rate

    def readout_velocity(self, theta: Array, rate: Array) -> Array:
        return self.fine.readout_velocity(theta, rate)

    def __call__(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
        *,
        return_trace: bool = False,
    ):
        """Predict rectified-flow velocity for noisy images."""

        theta0, rate0, drive, time_features = self.fine.initial_state(
            images,
            t,
            labels,
        )
        coarse_theta0, coarse_rate0, coarse_drive, _ = self.initial_coarse_state(
            images,
            t,
            labels,
        )
        if return_trace:
            (
                theta,
                rate,
                coarse_theta,
                coarse_rate,
                theta_traj,
                rate_traj,
                coarse_theta_traj,
                coarse_rate_traj,
            ) = self.evolve(
                theta0,
                rate0,
                drive,
                coarse_theta0,
                coarse_rate0,
                coarse_drive,
                time_features,
                labels,
                return_trajectory=True,
            )
        else:
            theta, rate, coarse_theta, coarse_rate = self.evolve(
                theta0,
                rate0,
                drive,
                coarse_theta0,
                coarse_rate0,
                coarse_drive,
                time_features,
                labels,
            )
            theta_traj = rate_traj = coarse_theta_traj = coarse_rate_traj = None
        velocity = self.readout_velocity(theta, rate)
        if return_trace:
            return velocity, {
                "initial_theta": theta0,
                "initial_rate": rate0,
                "theta_trajectory": theta_traj,
                "rate_trajectory": rate_traj,
                "final_theta": theta,
                "final_rate": rate,
                "initial_coarse_theta": coarse_theta0,
                "initial_coarse_rate": coarse_rate0,
                "coarse_theta_trajectory": coarse_theta_traj,
                "coarse_rate_trajectory": coarse_rate_traj,
                "final_coarse_theta": coarse_theta,
                "final_coarse_rate": coarse_rate,
                "velocity": velocity,
            }
        return velocity

    def predict_clean(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Array:
        velocity = self(images, t, labels)
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (images.shape[0],))
        return images + (1.0 - t[:, None]) * velocity

    def sample(
        self,
        key: jax.random.PRNGKey,
        sample_count: int,
        *,
        labels: Optional[Array] = None,
        outer_steps: int = 16,
    ) -> Array:
        """Generate images by integrating the learned rectified-flow field."""

        sample_count = int(sample_count)
        if outer_steps < 1:
            raise ValueError("outer_steps must be positive")
        x = jax.random.normal(key, (sample_count, self.image_dim))
        if labels is not None:
            labels = labels.astype(jnp.int32)
        step_size = 1.0 / float(outer_steps)

        def scan_fn(current_x, step_index):
            t_value = (step_index.astype(current_x.dtype) + 0.5) * step_size
            t = jnp.full((sample_count,), t_value, dtype=current_x.dtype)
            velocity = self(current_x, t, labels)
            return current_x + step_size * velocity, None

        steps = jnp.arange(int(outer_steps), dtype=jnp.float32)
        x, _ = jax.lax.scan(scan_fn, x, steps)
        return jnp.clip(x, 0.0, 1.0)

    def collect_trace(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        velocity, trace = self(images, t, labels, return_trace=True)
        clean = images + (1.0 - jnp.asarray(t)[:, None]) * velocity
        trace["input"] = images
        trace["predicted_clean"] = clean
        return trace


class RecurrentConvFlowField(eqx.Module):
    """Matched non-oscillatory local recurrent field for flow attribution.

    This control keeps the same visible image-field task as
    `PhaseRateFlowField` but replaces phase/rate oscillator dynamics with a
    gated tied convolutional recurrence. It is intentionally local and
    recurrent, so it tests whether the phase-flow win comes from oscillator
    geometry or from generic learned local message passing.
    """

    input_projection: eqx.nn.Linear
    time_to_hidden: eqx.nn.Linear
    readout: eqx.nn.Linear
    hidden_kernel: Array
    hidden_bias: Array
    gate_kernel: Array
    gate_bias: Array
    label_hidden_bias: Optional[Array]
    label_step_bias: Optional[Array]

    image_shape: Tuple[int, int] = eqx.field(static=True)
    image_dim: int = eqx.field(static=True)
    field_channels: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    rate_update: float = eqx.field(static=True)
    input_drive_strength: float = eqx.field(static=True)
    train_dynamics: bool = eqx.field(static=True)
    num_classes: int = eqx.field(static=True)
    position_features: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int] = (28, 28),
        field_channels: int = 8,
        steps: int = 8,
        kernel_size: int = 3,
        dt: float = 0.15,
        coupling_strength: float = 1.0,
        rate_update: float = 0.5,
        input_drive_strength: float = 0.5,
        kernel_init_scale: float = 0.05,
        train_dynamics: bool = True,
        num_classes: int = 10,
        position_features: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if field_channels < 1:
            raise ValueError("field_channels must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative")
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, 6)
        height, width = image_shape
        conv_scale = float(kernel_init_scale) / jnp.sqrt(
            float(kernel_size * kernel_size * field_channels)
        )

        self.image_shape = tuple(int(size) for size in image_shape)
        self.image_dim = int(height * width)
        self.field_channels = int(field_channels)
        self.steps = int(steps)
        self.kernel_size = int(kernel_size)
        self.dt = float(dt)
        self.coupling_strength = float(coupling_strength)
        self.rate_update = float(rate_update)
        self.input_drive_strength = float(input_drive_strength)
        self.train_dynamics = bool(train_dynamics)
        self.num_classes = int(num_classes)
        self.position_features = bool(position_features)

        input_dim = 10 if self.position_features else 4
        self.input_projection = eqx.nn.Linear(
            input_dim,
            2 * field_channels,
            key=keys[0],
        )
        self.time_to_hidden = eqx.nn.Linear(3, field_channels, key=keys[1])
        self.readout = eqx.nn.Linear(field_channels, 1, key=keys[2])
        self.hidden_kernel = (
            jax.random.normal(
                keys[3],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.hidden_bias = jnp.zeros((field_channels,))
        self.gate_kernel = (
            jax.random.normal(
                keys[4],
                (kernel_size, kernel_size, field_channels, field_channels),
            )
            * conv_scale
        )
        self.gate_bias = jnp.zeros((field_channels,))
        if self.num_classes > 0:
            self.label_hidden_bias = (
                jax.random.normal(keys[5], (self.num_classes, field_channels)) * 0.05
            )
            self.label_step_bias = jnp.zeros((self.num_classes, field_channels))
        else:
            self.label_hidden_bias = None
            self.label_step_bias = None

    def _time_features(self, t: Array) -> Array:
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = t[None]
        return jnp.stack(
            [
                t,
                jnp.sin(jnp.pi * t),
                jnp.cos(jnp.pi * t),
            ],
            axis=-1,
        )

    def initial_state(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array, Array]:
        """Initialize hidden state from noisy image and flow time."""

        batch_size = images.shape[0]
        height, width = self.image_shape
        image_grid = images.reshape(batch_size, height, width, 1)
        time_features = self._time_features(t)
        time_grid = jnp.broadcast_to(
            time_features[:, None, None, :],
            (batch_size, height, width, 3),
        )
        inputs = jnp.concatenate([image_grid, time_grid], axis=-1)
        if self.position_features:
            position_grid = _spatial_phase_features(
                height,
                width,
                dtype=images.dtype,
            )
            position_grid = jnp.broadcast_to(
                position_grid[None, :, :, :],
                (batch_size, height, width, position_grid.shape[-1]),
            )
            inputs = jnp.concatenate([inputs, position_grid], axis=-1)
        projected = _apply_linear_last(self.input_projection, inputs)
        hidden_raw, drive_raw = jnp.split(projected, 2, axis=-1)
        if labels is not None and self.label_hidden_bias is not None:
            hidden_raw = hidden_raw + self.label_hidden_bias[labels.astype(jnp.int32)][
                :, None, None, :
            ]
        return jnp.tanh(hidden_raw), jnp.tanh(drive_raw), time_features

    def _dynamics_params(self):
        hidden_kernel = self.hidden_kernel
        hidden_bias = self.hidden_bias
        gate_kernel = self.gate_kernel
        gate_bias = self.gate_bias
        label_step_bias = self.label_step_bias
        if not self.train_dynamics:
            hidden_kernel = jax.lax.stop_gradient(hidden_kernel)
            hidden_bias = jax.lax.stop_gradient(hidden_bias)
            gate_kernel = jax.lax.stop_gradient(gate_kernel)
            gate_bias = jax.lax.stop_gradient(gate_bias)
            if label_step_bias is not None:
                label_step_bias = jax.lax.stop_gradient(label_step_bias)
        return hidden_kernel, hidden_bias, gate_kernel, gate_bias, label_step_bias

    def step(
        self,
        hidden: Array,
        drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
    ) -> Array:
        """Advance one tied local recurrent-conv step."""

        hidden_kernel, hidden_bias, gate_kernel, gate_bias, label_step_bias = (
            self._dynamics_params()
        )
        context = _apply_same_conv2d(hidden_kernel, hidden_bias, hidden)
        gate_context = _apply_same_conv2d(gate_kernel, gate_bias, hidden)
        time_drive = _apply_linear_last(self.time_to_hidden, time_features)
        if not self.train_dynamics:
            time_drive = jax.lax.stop_gradient(time_drive)
        step_drive = time_drive[:, None, None, :]
        if labels is not None and label_step_bias is not None:
            step_drive = step_drive + label_step_bias[labels.astype(jnp.int32)][
                :, None, None, :
            ]
        proposal = jnp.tanh(
            self.coupling_strength * context
            + step_drive
            + self.input_drive_strength * drive
        )
        gate = jax.nn.sigmoid(
            gate_context + step_drive + self.input_drive_strength * drive
        )
        update = jnp.clip(self.rate_update, 0.0, 1.0) * gate
        return hidden + update * (proposal - hidden)

    def evolve(
        self,
        hidden: Array,
        drive: Array,
        time_features: Array,
        labels: Optional[Array] = None,
        *,
        return_trajectory: bool = False,
    ):
        """Evolve the hidden field through the configured recurrent steps."""

        if self.steps == 0:
            if return_trajectory:
                empty = jnp.zeros((0, *hidden.shape), dtype=hidden.dtype)
                return hidden, empty
            return hidden

        def scan_fn(current_hidden, _):
            next_hidden = self.step(
                current_hidden,
                drive,
                time_features,
                labels,
            )
            return next_hidden, next_hidden

        final_hidden, hidden_traj = jax.lax.scan(
            scan_fn,
            hidden,
            xs=None,
            length=self.steps,
        )
        if return_trajectory:
            return final_hidden, hidden_traj
        return final_hidden

    def readout_velocity(self, hidden: Array) -> Array:
        """Read a flat image velocity from local hidden features."""

        velocity_grid = _apply_linear_last(self.readout, hidden)
        return velocity_grid.reshape(hidden.shape[0], self.image_dim)

    def __call__(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
        *,
        return_trace: bool = False,
    ):
        """Predict rectified-flow velocity for a batch of noisy images."""

        hidden0, drive, time_features = self.initial_state(images, t, labels)
        if return_trace:
            hidden, hidden_traj = self.evolve(
                hidden0,
                drive,
                time_features,
                labels,
                return_trajectory=True,
            )
        else:
            hidden = self.evolve(hidden0, drive, time_features, labels)
            hidden_traj = None
        velocity = self.readout_velocity(hidden)
        if return_trace:
            return velocity, {
                "initial_hidden": hidden0,
                "hidden_trajectory": hidden_traj,
                "final_hidden": hidden,
                "velocity": velocity,
            }
        return velocity

    def predict_clean(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Array:
        """Predict the clean endpoint implied by rectified-flow velocity."""

        velocity = self(images, t, labels)
        t = jnp.asarray(t)
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (images.shape[0],))
        return images + (1.0 - t[:, None]) * velocity

    def sample(
        self,
        key: jax.random.PRNGKey,
        sample_count: int,
        *,
        labels: Optional[Array] = None,
        outer_steps: int = 16,
    ) -> Array:
        """Generate images by integrating the learned rectified-flow field."""

        sample_count = int(sample_count)
        if outer_steps < 1:
            raise ValueError("outer_steps must be positive")
        x = jax.random.normal(key, (sample_count, self.image_dim))
        if labels is not None:
            labels = labels.astype(jnp.int32)
        step_size = 1.0 / float(outer_steps)

        def scan_fn(current_x, step_index):
            t_value = (step_index.astype(current_x.dtype) + 0.5) * step_size
            t = jnp.full((sample_count,), t_value, dtype=current_x.dtype)
            velocity = self(current_x, t, labels)
            return current_x + step_size * velocity, None

        steps = jnp.arange(int(outer_steps), dtype=jnp.float32)
        x, _ = jax.lax.scan(scan_fn, x, steps)
        return jnp.clip(x, 0.0, 1.0)

    def collect_trace(
        self,
        images: Array,
        t: Array,
        labels: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """Collect recurrent trajectory diagnostics for a noisy batch."""

        velocity, trace = self(images, t, labels, return_trace=True)
        clean = images + (1.0 - jnp.asarray(t)[:, None]) * velocity
        trace["input"] = images
        trace["predicted_clean"] = clean
        return trace


__all__ = [
    "CoarseGlobalPhaseRateFlowField",
    "PhaseRateFlowField",
    "RecurrentConvFlowField",
]
