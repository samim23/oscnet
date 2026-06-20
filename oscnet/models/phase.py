"""Phase-only oscillatory model components."""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.models.oscillatory import Array


def _wrap_phase(theta: Array) -> Array:
    return jnp.angle(jnp.exp(1j * theta))


class WinfreePhaseOscillatorCell(eqx.Module):
    """
    Phase-only recurrent cell inspired by Winfree synchronization dynamics.

    The state is a batched phase tensor `(batch, hidden_dim)`. Each step combines
    natural frequency, projected input drive, and a pulse/phase-response coupling
    term, then reads out from `[cos(theta), sin(theta)]` features.
    """

    i2h: eqx.nn.Linear
    phase_to_output: eqx.nn.Linear
    omega: Array
    coupling: Array
    initial_phases: Array

    hidden_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    input_gain: float = eqx.field(static=True)
    coupling_strength: float = eqx.field(static=True)
    pulse_exponent: float = eqx.field(static=True)
    phase_response_bias: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        omega: float = 1.0,
        dt: float = 0.05,
        input_gain: float = 0.2,
        coupling_strength: float = 0.1,
        pulse_exponent: float = 1.0,
        phase_response_bias: float = 1.0,
        initial_phases: Optional[Array] = None,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.input_gain = input_gain
        self.coupling_strength = coupling_strength
        self.pulse_exponent = pulse_exponent
        self.phase_response_bias = phase_response_bias

        self.i2h = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        self.phase_to_output = eqx.nn.Linear(2 * hidden_dim, output_dim, key=keys[1])
        self.omega = jnp.ones(hidden_dim) * omega

        raw_coupling = jax.random.normal(keys[2], (hidden_dim, hidden_dim))
        raw_coupling = raw_coupling.at[jnp.diag_indices(hidden_dim)].set(0.0)
        scale = jnp.sqrt(float(hidden_dim))
        self.coupling = raw_coupling / scale

        if initial_phases is None:
            self.initial_phases = jax.random.uniform(
                keys[3], (hidden_dim,), minval=-jnp.pi, maxval=jnp.pi
            )
        else:
            phases = jnp.asarray(initial_phases)
            if phases.shape != (hidden_dim,):
                raise ValueError(f"initial_phases must have shape ({hidden_dim},)")
            self.initial_phases = phases

    def initial_state(self, batch_size: int, use_phase_init: bool = True) -> Array:
        if use_phase_init:
            phases = self.initial_phases
        else:
            phases = jnp.zeros(self.hidden_dim)
        return jnp.broadcast_to(phases[None, :], (batch_size, self.hidden_dim))

    def __call__(
        self,
        inputs: Array,
        state: Optional[Array] = None,
        use_phase_init: bool = True,
    ):
        batch_size = inputs.shape[0]
        theta = state if state is not None else self.initial_state(batch_size, use_phase_init)

        drive = jnp.tanh(jax.vmap(self.i2h)(inputs))
        pulses = (1.0 + jnp.cos(theta)) ** self.pulse_exponent
        coupled_pulse = jnp.dot(pulses, self.coupling.T) / self.hidden_dim
        phase_response = self.phase_response_bias - jnp.sin(theta)

        dtheta = (
            self.omega
            + self.input_gain * drive
            + self.coupling_strength * coupled_pulse * phase_response
        )
        theta_new = _wrap_phase(theta + self.dt * dtheta)

        features = jnp.concatenate([jnp.cos(theta_new), jnp.sin(theta_new)], axis=-1)
        output = jax.vmap(self.phase_to_output)(features)
        return output, theta_new


class WinfreePhaseSequenceLayer(eqx.Module):
    """Scan a Winfree phase cell over `(time, batch, features)` inputs."""

    cell: WinfreePhaseOscillatorCell

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        key: jax.random.PRNGKey,
        **cell_kwargs,
    ):
        self.cell = WinfreePhaseOscillatorCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            key=key,
            **cell_kwargs,
        )

    def __call__(
        self,
        inputs: Array,
        initial_state: Optional[Array] = None,
        return_trajectories: bool = False,
        use_phase_init: bool = True,
    ):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (time, batch, features)")

        batch_size = inputs.shape[1]
        state = initial_state
        if state is None:
            state = self.cell.initial_state(batch_size, use_phase_init)

        def scan_fn(carry, x_t):
            output, new_state = self.cell(x_t, carry)
            return new_state, output

        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        if return_trajectories:
            return {"outputs": outputs, "final_state": final_state}
        return outputs


class WinfreePhaseAutoencoder(eqx.Module):
    """Small phase-only sequence autoencoder for research experiments."""

    encoder: WinfreePhaseSequenceLayer
    to_latent: eqx.nn.Linear
    from_latent: eqx.nn.Linear
    decoder: WinfreePhaseSequenceLayer
    sequence_length: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        sequence_length: Optional[int] = None,
        *,
        key: jax.random.PRNGKey,
        **cell_kwargs,
    ):
        keys = jax.random.split(key, 4)
        self.sequence_length = sequence_length
        self.encoder = WinfreePhaseSequenceLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            key=keys[0],
            **cell_kwargs,
        )
        self.to_latent = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[1])
        self.from_latent = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[2])
        self.decoder = WinfreePhaseSequenceLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            key=keys[3],
            **cell_kwargs,
        )

    def encode(self, inputs: Array, use_phase_init: bool = True) -> Array:
        outputs = self.encoder(inputs, use_phase_init=use_phase_init)
        return jax.vmap(self.to_latent)(outputs[-1])

    def decode(
        self,
        latent: Array,
        sequence_length: Optional[int] = None,
        use_phase_init: bool = True,
    ) -> Array:
        length = sequence_length or self.sequence_length
        if length is None:
            raise ValueError("sequence_length must be provided")

        hidden = jax.vmap(self.from_latent)(latent)
        inputs = jnp.broadcast_to(hidden[None, :, :], (length, *hidden.shape))
        return self.decoder(inputs, use_phase_init=use_phase_init)

    def __call__(self, inputs: Array, use_phase_init: bool = True) -> Array:
        latent = self.encode(inputs, use_phase_init=use_phase_init)
        return self.decode(
            latent,
            sequence_length=inputs.shape[0],
            use_phase_init=use_phase_init,
        )


__all__ = [
    "WinfreePhaseOscillatorCell",
    "WinfreePhaseSequenceLayer",
    "WinfreePhaseAutoencoder",
]
