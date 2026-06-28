"""VAE-style image generator with oscillator phase dynamics."""

from __future__ import annotations

from .common import Array, Dict, Optional, Tuple, eqx, jax, jnp, phase_features, wrap_phase

class KuramotoPhaseVAE(eqx.Module):
    """MNIST-native VAE with an oscillator-evolved phase latent.

    This is intentionally easier than the unpaired generator: images are encoded
    into a sampleable Gaussian latent, that latent is interpreted as oscillator
    phase, optional Kuramoto dynamics evolve it, and the decoder reconstructs the
    input. Generation samples the Gaussian prior and decodes through the same
    phase dynamics.
    """

    encoder_layers: Tuple[eqx.nn.Linear, ...]
    encoder_mu: eqx.nn.Linear
    encoder_logvar: eqx.nn.Linear
    omega: Array
    coupling: Array
    decoder_layers: Tuple[eqx.nn.Linear, ...]
    image_shape: Tuple[int, int] = eqx.field(static=True)
    image_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    train_dynamics: bool = eqx.field(static=True)
    phase_readout_mode: str = eqx.field(static=True)
    output_activation: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        image_shape: Tuple[int, int] = (28, 28),
        latent_dim: int = 32,
        hidden_dim: int = 256,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        steps: int = 4,
        dt: float = 0.1,
        coupling_strength: float = 1.0,
        omega_scale: float = 0.2,
        coupling_init_scale: float = 0.05,
        train_dynamics: bool = True,
        phase_readout_mode: str = "absolute",
        output_activation: str = "sigmoid",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be positive")
        if decoder_depth < 1:
            raise ValueError("decoder_depth must be positive")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if phase_readout_mode not in ("absolute", "mean_relative", "ref_oscillator"):
            raise ValueError(
                "phase_readout_mode must be 'absolute', 'mean_relative', "
                "or 'ref_oscillator'"
            )
        if output_activation not in ("identity", "sigmoid"):
            raise ValueError("output_activation must be 'identity' or 'sigmoid'")
        if key is None:
            key = jax.random.PRNGKey(42)

        self.image_shape = tuple(int(size) for size in image_shape)
        self.image_dim = int(self.image_shape[0] * self.image_shape[1])
        self.latent_dim = int(latent_dim)
        self.steps = int(steps)
        self.dt = float(dt)
        self.coupling_strength = float(coupling_strength)
        self.train_dynamics = bool(train_dynamics)
        self.phase_readout_mode = phase_readout_mode
        self.output_activation = output_activation

        key_count = encoder_depth + decoder_depth + 5
        keys = jax.random.split(key, key_count)
        encoder_keys = keys[:encoder_depth]
        mu_key = keys[encoder_depth]
        logvar_key = keys[encoder_depth + 1]
        omega_key = keys[encoder_depth + 2]
        coupling_key = keys[encoder_depth + 3]
        decoder_keys = keys[encoder_depth + 4 :]

        encoder_layers = []
        in_dim = self.image_dim
        for layer_key in encoder_keys:
            encoder_layers.append(eqx.nn.Linear(in_dim, int(hidden_dim), key=layer_key))
            in_dim = int(hidden_dim)
        self.encoder_layers = tuple(encoder_layers)
        self.encoder_mu = eqx.nn.Linear(in_dim, self.latent_dim, key=mu_key)
        self.encoder_logvar = eqx.nn.Linear(in_dim, self.latent_dim, key=logvar_key)

        self.omega = (
            jax.random.normal(omega_key, (self.latent_dim,)) * float(omega_scale)
        )
        coupling = (
            jax.random.normal(coupling_key, (self.latent_dim, self.latent_dim))
            * float(coupling_init_scale)
            / jnp.sqrt(float(self.latent_dim))
        )
        self.coupling = coupling * (
            1.0 - jnp.eye(self.latent_dim, dtype=jnp.float32)
        )

        decoder_layers = []
        layer_dims = [2 * self.latent_dim]
        layer_dims.extend([int(hidden_dim)] * int(decoder_depth))
        layer_dims.append(self.image_dim)
        for in_size, out_size, layer_key in zip(
            layer_dims[:-1],
            layer_dims[1:],
            decoder_keys,
        ):
            decoder_layers.append(eqx.nn.Linear(in_size, out_size, key=layer_key))
        self.decoder_layers = tuple(decoder_layers)

    def _encode_single(self, image: Array) -> Tuple[Array, Array]:
        hidden = image
        for layer in self.encoder_layers:
            hidden = jax.nn.gelu(layer(hidden))
        mu = self.encoder_mu(hidden)
        logvar = jnp.clip(self.encoder_logvar(hidden), -8.0, 4.0)
        return mu, logvar

    def encode(self, images: Array) -> Tuple[Array, Array]:
        """Return Gaussian latent parameters for a batch of images."""

        return jax.vmap(self._encode_single)(images)

    def reparameterize(
        self,
        mu: Array,
        logvar: Array,
        key: jax.random.PRNGKey,
    ) -> Array:
        """Sample latent vectors with the VAE reparameterization trick."""

        eps = jax.random.normal(key, mu.shape)
        return mu + eps * jnp.exp(0.5 * logvar)

    def _dynamics_params(self) -> Tuple[Array, Array]:
        omega = self.omega
        coupling = self.coupling
        if not self.train_dynamics:
            omega = jax.lax.stop_gradient(omega)
            coupling = jax.lax.stop_gradient(coupling)
        return omega, coupling * (1.0 - jnp.eye(self.latent_dim, dtype=jnp.float32))

    def step(self, theta: Array) -> Array:
        """Advance one Kuramoto step for latent phases."""

        omega, coupling = self._dynamics_params()
        phase_diff = theta[:, None, :] - theta[:, :, None]
        interaction = jnp.sum(coupling[None, :, :] * jnp.sin(phase_diff), axis=-1)
        velocity = omega[None, :] + self.coupling_strength * (
            interaction / float(self.latent_dim)
        )
        return wrap_phase(theta + self.dt * velocity)

    def evolve(self, theta0: Array, *, return_trajectory: bool = False):
        """Evolve latent phases through the configured oscillator dynamics."""

        if self.steps == 0:
            empty = jnp.zeros((0, *theta0.shape), dtype=theta0.dtype)
            return (theta0, empty) if return_trajectory else theta0

        def scan_fn(theta, _):
            next_theta = self.step(theta)
            return next_theta, next_theta

        final_theta, trajectory = jax.lax.scan(
            scan_fn,
            theta0,
            xs=None,
            length=self.steps,
        )
        return (final_theta, trajectory) if return_trajectory else final_theta

    def readout_theta(self, theta: Array) -> Array:
        """Apply the configured phase reference frame before decoding."""

        if self.phase_readout_mode == "absolute":
            return theta
        if self.phase_readout_mode == "mean_relative":
            return wrap_phase(theta - jnp.mean(theta, axis=-1, keepdims=True))
        if self.phase_readout_mode == "ref_oscillator":
            return wrap_phase(theta - theta[:, :1])
        raise ValueError("unknown phase_readout_mode")

    def decode_phase(self, theta: Array) -> Array:
        """Decode phase features into flat image probabilities."""

        readout_theta = self.readout_theta(theta)
        hidden = phase_features(readout_theta).reshape(theta.shape[0], -1)
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden = jax.vmap(layer)(hidden)
            if layer_index < len(self.decoder_layers) - 1:
                hidden = jax.nn.gelu(hidden)
        return _activation(self.output_activation)(hidden)

    def __call__(
        self,
        images: Array,
        key: jax.random.PRNGKey,
        *,
        deterministic: bool = False,
        return_latent: bool = False,
    ):
        """Reconstruct a batch and optionally return latent diagnostics."""

        mu, logvar = self.encode(images)
        z = mu if deterministic else self.reparameterize(mu, logvar, key)
        theta0 = wrap_phase(z)
        final_theta = self.evolve(theta0)
        reconstruction = self.decode_phase(final_theta)
        if return_latent:
            return reconstruction, {
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "theta0": theta0,
                "final_theta": final_theta,
            }
        return reconstruction

    def sample(
        self,
        key: jax.random.PRNGKey,
        sample_count: int,
    ) -> Array:
        """Generate images from the standard Gaussian latent prior."""

        z = jax.random.normal(key, (int(sample_count), self.latent_dim))
        theta0 = wrap_phase(z)
        return self.decode_phase(self.evolve(theta0))

    def collect_trace(
        self,
        images: Array,
        key: jax.random.PRNGKey,
    ) -> Dict[str, Array]:
        """Collect reconstruction and latent phase diagnostics."""

        reconstruction, latent = self(
            images,
            key,
            deterministic=True,
            return_latent=True,
        )
        _, trajectory = self.evolve(latent["theta0"], return_trajectory=True)
        return {
            "input": images,
            "reconstruction": reconstruction,
            "mu": latent["mu"],
            "logvar": latent["logvar"],
            "initial_theta": latent["theta0"],
            "theta_trajectory": trajectory,
            "final_theta": latent["final_theta"],
            "readout_theta": self.readout_theta(latent["final_theta"]),
            "omega": self.omega,
            "coupling": self.coupling,
        }


