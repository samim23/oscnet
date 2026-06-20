"""
Reusable oscillatory model building blocks.

The classes in this module turn the low-level oscillator primitives into a
small model spine: cells, sequence layers, encoders, decoders, and complete
autoencoders. They are intentionally task-agnostic so future dynamics, such as
phase-only or Winfree-style cells, can slot into the same sequence/readout
pattern.
"""

from typing import Any, Dict, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.core.oscillators import NonlinearHarmonicOscillator, Oscillator

Array = jnp.ndarray
OscillatorState = Tuple[Array, Array]


def _zeros_state(batch_size: int, hidden_dim: int) -> OscillatorState:
    x = jnp.zeros((batch_size, hidden_dim))
    v = jnp.zeros((batch_size, hidden_dim))
    return x, v


def _omega_array(oscillator: Oscillator, hidden_dim: int) -> Array:
    if hasattr(oscillator, "get_effective_frequencies"):
        omega = oscillator.get_effective_frequencies()
    elif hasattr(oscillator, "omega"):
        omega = oscillator.omega
    else:
        omega = 1.0

    if isinstance(omega, (int, float)):
        return jnp.ones(hidden_dim) * float(omega)

    omega = jnp.asarray(omega)
    if omega.ndim == 0:
        return jnp.ones(hidden_dim, dtype=omega.dtype) * omega
    return jnp.broadcast_to(omega, (hidden_dim,))


class AmplitudeVelocityOscillatorCell(eqx.Module):
    """
    Oscillatory recurrent cell that exposes both amplitude and velocity.

    Inputs are projected into oscillator forcing terms, an optional recurrent
    velocity projection is added, oscillator dynamics are stepped once, and the
    output is read from ``concat([x, v])``.
    """

    i2h: eqx.nn.Linear
    h2h: eqx.nn.Linear
    h2o: eqx.nn.Linear
    oscillator: Oscillator
    gain_multiplier: Array
    initial_phases: Array

    hidden_dim: int = eqx.field(static=True)
    gain_rec: float = eqx.field(static=True)
    use_recurrent_velocity: bool = eqx.field(static=True)
    initial_amplitude: float = eqx.field(static=True)
    readout_mode: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 5)

        base_gain = 1.0 / jnp.sqrt(hidden_dim)
        self.gain_rec = float(gain_rec) if gain_rec is not None else float(base_gain)
        if initial_gain_multiplier is None:
            initial_gain_multiplier = 1.0 / float(base_gain)
        self.gain_multiplier = jnp.asarray([initial_gain_multiplier])

        self.hidden_dim = hidden_dim
        self.use_recurrent_velocity = use_recurrent_velocity
        self.initial_amplitude = initial_amplitude
        if readout_mode not in {"amplitude_velocity", "position"}:
            raise ValueError("readout_mode must be 'amplitude_velocity' or 'position'")
        self.readout_mode = readout_mode

        if initial_phases is None:
            self.initial_phases = jax.random.uniform(
                keys[3], (hidden_dim,), minval=0.0, maxval=2.0 * jnp.pi
            )
        else:
            phases = jnp.asarray(initial_phases)
            if phases.shape != (hidden_dim,):
                raise ValueError(f"initial_phases must have shape ({hidden_dim},)")
            self.initial_phases = phases

        self.i2h = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        self.h2h = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
        readout_dim = hidden_dim if readout_mode == "position" else 2 * hidden_dim
        self.h2o = eqx.nn.Linear(readout_dim, output_dim, key=keys[2])
        self.oscillator = oscillator_class(
            dim=hidden_dim,
            **(oscillator_params or {}),
            key=keys[4],
        )

    def initial_state(self, batch_size: int, use_phase_init: bool = False) -> OscillatorState:
        """Create an initial ``(x, v)`` oscillator state for a batch."""
        if not use_phase_init:
            return _zeros_state(batch_size, self.hidden_dim)

        omega = _omega_array(self.oscillator, self.hidden_dim)
        x_init = self.initial_amplitude * jnp.cos(self.initial_phases)
        v_init = -self.initial_amplitude * omega * jnp.sin(self.initial_phases)
        x = jnp.broadcast_to(x_init[None, :], (batch_size, self.hidden_dim))
        v = jnp.broadcast_to(v_init[None, :], (batch_size, self.hidden_dim))
        return x, v

    def __call__(
        self,
        inputs: Array,
        state: Optional[OscillatorState] = None,
        use_phase_init: bool = False,
    ) -> Tuple[Array, OscillatorState]:
        """Process one batched timestep."""
        batch_size = inputs.shape[0]
        x, v = state if state is not None else self.initial_state(batch_size, use_phase_init)

        input_contrib = jax.vmap(self.i2h)(inputs)
        if self.use_recurrent_velocity:
            effective_gain = self.gain_rec * self.gain_multiplier[0]
            recurrent_contrib = jax.vmap(self.h2h)(v) * effective_gain
            total_input = input_contrib + recurrent_contrib
        else:
            total_input = input_contrib

        new_x, new_v = jax.vmap(self.oscillator.step)(x, v, total_input)
        if self.readout_mode == "position":
            features = jnp.tanh(new_x)
        else:
            features = jnp.concatenate([new_x, new_v], axis=-1)
        output = jax.vmap(self.h2o)(features)
        return output, (new_x, new_v)


class OscillatorySequenceLayer(eqx.Module):
    """Scan an oscillatory cell over ``(time, batch, features)`` inputs."""

    cell: AmplitudeVelocityOscillatorCell

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        self.cell = AmplitudeVelocityOscillatorCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=key,
        )

    def __call__(
        self,
        inputs: Array,
        initial_state: Optional[OscillatorState] = None,
        return_trajectories: bool = False,
        use_phase_init: bool = False,
    ):
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (time, batch, features)")

        batch_size = inputs.shape[1]
        state = initial_state or self.cell.initial_state(batch_size, use_phase_init)

        def scan_fn(carry, x_t):
            output, new_state = self.cell(x_t, carry)
            return new_state, output

        final_state, outputs = jax.lax.scan(scan_fn, state, inputs)
        if return_trajectories:
            return {"outputs": outputs, "final_state": final_state}
        return outputs


class OscillatoryEncoder(eqx.Module):
    """Encode a feature sequence into a latent representation."""

    sequence: OscillatorySequenceLayer
    to_latent: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 2)
        self.sequence = OscillatorySequenceLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[0],
        )
        self.to_latent = eqx.nn.Linear(hidden_dim, latent_dim, key=keys[1])

    def __call__(self, inputs: Array, use_phase_init: bool = False) -> Array:
        outputs = self.sequence(inputs, use_phase_init=use_phase_init)
        return jax.vmap(self.to_latent)(outputs[-1])

    @property
    def rnn(self) -> OscillatorySequenceLayer:
        """Compatibility alias for earlier example code."""
        return self.sequence


class RepeatedLatentOscillatoryDecoder(eqx.Module):
    """Decode by repeating a projected latent vector across timesteps."""

    from_latent: eqx.nn.Linear
    sequence: OscillatorySequenceLayer
    sequence_length: Optional[int] = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: Optional[int] = None,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 2)
        self.sequence_length = sequence_length
        self.from_latent = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.sequence = OscillatorySequenceLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[1],
        )

    def __call__(
        self,
        latent: Array,
        sequence_length: Optional[int] = None,
        use_phase_init: bool = False,
    ) -> Array:
        length = sequence_length or self.sequence_length
        if length is None:
            raise ValueError("sequence_length must be provided")

        hidden = jax.vmap(self.from_latent)(latent)
        inputs = jnp.broadcast_to(hidden[None, :, :], (length, *hidden.shape))
        return self.sequence(inputs, use_phase_init=use_phase_init)

    @property
    def rnn(self) -> OscillatorySequenceLayer:
        """Compatibility alias for earlier example code."""
        return self.sequence


class PositionalLatentOscillatoryDecoder(eqx.Module):
    """
    Decode by combining a latent vector with learned per-timestep prompts.

    This keeps the oscillatory sequence decoder, but gives image/sequence tasks
    an explicit position signal so the latent can describe content while the
    learned prompts describe where each output patch belongs.
    """

    from_latent: eqx.nn.Linear
    latent_to_state: eqx.nn.Linear
    latent_to_velocity: eqx.nn.Linear
    sequence: OscillatorySequenceLayer
    positional_inputs: Array

    sequence_length: int = eqx.field(static=True)
    latent_conditioning_strength: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: Optional[int] = None,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        latent_conditioning_strength: float = 1.0,
        *,
        key: jax.random.PRNGKey,
    ):
        if sequence_length is None:
            raise ValueError("Positional decoder requires sequence_length")

        keys = jax.random.split(key, 5)
        self.sequence_length = sequence_length
        self.latent_conditioning_strength = latent_conditioning_strength
        self.from_latent = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.latent_to_state = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[1])
        self.latent_to_velocity = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[2])
        self.positional_inputs = (
            jax.random.normal(keys[3], (sequence_length, hidden_dim)) * 0.02
        )
        self.sequence = OscillatorySequenceLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[4],
        )

    def __call__(
        self,
        latent: Array,
        sequence_length: Optional[int] = None,
        use_phase_init: bool = False,
    ) -> Array:
        length = sequence_length or self.sequence_length
        if length > self.sequence_length:
            raise ValueError("sequence_length exceeds configured positional inputs")

        latent_hidden = (
            jnp.tanh(jax.vmap(self.from_latent)(latent))
            * self.latent_conditioning_strength
        )
        positional = self.positional_inputs[:length]
        inputs = latent_hidden[None, :, :] + positional[:, None, :]

        batch_size = latent.shape[0]
        initial_state = (
            jnp.tanh(jax.vmap(self.latent_to_state)(latent)),
            jnp.tanh(jax.vmap(self.latent_to_velocity)(latent)),
        )
        if use_phase_init:
            _, phase_velocity = self.sequence.cell.initial_state(
                batch_size,
                use_phase_init=True,
            )
            initial_state = (initial_state[0], initial_state[1] + phase_velocity)

        return self.sequence(inputs, initial_state=initial_state)

    @property
    def rnn(self) -> OscillatorySequenceLayer:
        """Compatibility alias for earlier example code."""
        return self.sequence


class AutoregressiveOscillatoryDecoder(eqx.Module):
    """Decode by feeding the previous output back into an oscillatory cell."""

    latent_to_state: eqx.nn.Linear
    latent_to_velocity: eqx.nn.Linear
    latent_to_input: eqx.nn.Linear
    cell: AmplitudeVelocityOscillatorCell
    output_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    sequence_length: Optional[int] = eqx.field(static=True)
    latent_conditioning_strength: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: Optional[int] = None,
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        initial_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        latent_conditioning_strength: float = 1.0,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 4)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.latent_conditioning_strength = latent_conditioning_strength
        self.latent_to_state = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[0])
        self.latent_to_velocity = eqx.nn.Linear(latent_dim, hidden_dim, key=keys[1])
        self.latent_to_input = eqx.nn.Linear(latent_dim, output_dim, key=keys[2])
        self.cell = AmplitudeVelocityOscillatorCell(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=initial_phases,
            initial_amplitude=initial_amplitude,
            key=keys[3],
        )

    def __call__(
        self,
        latent: Array,
        sequence_length: Optional[int] = None,
        use_phase_init: bool = False,
    ) -> Array:
        length = sequence_length or self.sequence_length
        if length is None:
            raise ValueError("sequence_length must be provided")

        batch_size = latent.shape[0]
        x0 = jnp.tanh(jax.vmap(self.latent_to_state)(latent))
        v0 = jnp.tanh(jax.vmap(self.latent_to_velocity)(latent))
        if use_phase_init:
            _, v_phase = self.cell.initial_state(batch_size, use_phase_init=True)
            v0 = v0 + v_phase

        latent_drive = (
            jnp.tanh(jax.vmap(self.latent_to_input)(latent))
            * self.latent_conditioning_strength
        )
        initial_output = latent_drive
        initial_carry = ((x0, v0), initial_output)

        def scan_fn(carry, _):
            state, previous_output = carry
            decoder_input = previous_output + latent_drive
            output, new_state = self.cell(decoder_input, state)
            return (new_state, output), output

        _, outputs = jax.lax.scan(scan_fn, initial_carry, None, length=length)
        return outputs


class OscillatoryAutoencoder(eqx.Module):
    """Generic sequence-to-sequence oscillatory autoencoder."""

    encoder: OscillatoryEncoder
    decoder: eqx.Module
    decoder_mode: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,
        sequence_length: Optional[int] = None,
        decoder_mode: str = "repeat",
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        latent_conditioning_strength: float = 1.0,
        encoder_phases: Optional[Array] = None,
        decoder_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 2)
        output_dim = input_dim if output_dim is None else output_dim
        self.decoder_mode = decoder_mode

        self.encoder = OscillatoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            initial_phases=encoder_phases,
            initial_amplitude=initial_amplitude,
            key=keys[0],
        )

        decoder_kwargs = dict(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            latent_conditioning_strength=latent_conditioning_strength,
            initial_phases=decoder_phases,
            initial_amplitude=initial_amplitude,
            key=keys[1],
        )
        if decoder_mode == "repeat":
            repeat_kwargs = dict(decoder_kwargs)
            repeat_kwargs.pop("latent_conditioning_strength")
            self.decoder = RepeatedLatentOscillatoryDecoder(**repeat_kwargs)
        elif decoder_mode == "autoregressive":
            self.decoder = AutoregressiveOscillatoryDecoder(**decoder_kwargs)
        elif decoder_mode == "positional":
            self.decoder = PositionalLatentOscillatoryDecoder(**decoder_kwargs)
        else:
            raise ValueError(
                "decoder_mode must be 'repeat', 'autoregressive', or 'positional'"
            )

    def encode(self, inputs: Array, use_phase_init: bool = False) -> Array:
        return self.encoder(inputs, use_phase_init=use_phase_init)

    def decode(
        self,
        latent: Array,
        sequence_length: Optional[int] = None,
        use_phase_init: bool = False,
    ) -> Array:
        return self.decoder(
            latent,
            sequence_length=sequence_length,
            use_phase_init=use_phase_init,
        )

    def __call__(
        self,
        inputs: Array,
        use_phase_init: bool = False,
        return_latent: bool = False,
    ):
        latent = self.encode(inputs, use_phase_init=use_phase_init)
        reconstruction = self.decode(
            latent,
            sequence_length=inputs.shape[0],
            use_phase_init=use_phase_init,
        )
        if return_latent:
            return reconstruction, latent
        return reconstruction


class PatchOscillatoryAutoencoder(eqx.Module):
    """Patch image wrapper around ``OscillatoryAutoencoder``."""

    autoencoder: OscillatoryAutoencoder
    image_shape: Tuple[int, int] = eqx.field(static=True)
    patch_shape: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    patch_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        image_shape: Tuple[int, int] = (28, 28),
        patch_shape: Tuple[int, int] = (4, 4),
        decoder_mode: str = "repeat",
        oscillator_class: Type[Oscillator] = NonlinearHarmonicOscillator,
        oscillator_params: Optional[Dict[str, Any]] = None,
        gain_rec: Optional[float] = None,
        initial_gain_multiplier: Optional[float] = None,
        use_recurrent_velocity: bool = True,
        readout_mode: str = "amplitude_velocity",
        latent_conditioning_strength: float = 1.0,
        encoder_phases: Optional[Array] = None,
        decoder_phases: Optional[Array] = None,
        initial_amplitude: float = 0.1,
        key: Optional[jax.random.PRNGKey] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(42)

        height, width = image_shape
        patch_height, patch_width = patch_shape
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError("image_shape must be divisible by patch_shape")

        patch_dim = patch_height * patch_width
        if input_dim is not None and input_dim != patch_dim:
            raise ValueError(f"input_dim must match patch size ({patch_dim})")

        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.num_patches = (height // patch_height) * (width // patch_width)
        self.patch_dim = patch_dim
        self.autoencoder = OscillatoryAutoencoder(
            input_dim=patch_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            output_dim=patch_dim,
            sequence_length=self.num_patches,
            decoder_mode=decoder_mode,
            oscillator_class=oscillator_class,
            oscillator_params=oscillator_params,
            gain_rec=gain_rec,
            initial_gain_multiplier=initial_gain_multiplier,
            use_recurrent_velocity=use_recurrent_velocity,
            readout_mode=readout_mode,
            latent_conditioning_strength=latent_conditioning_strength,
            encoder_phases=encoder_phases,
            decoder_phases=decoder_phases,
            initial_amplitude=initial_amplitude,
            key=key,
        )

    def images_to_sequence(self, images: Array) -> Array:
        height, width = self.image_shape
        patch_height, patch_width = self.patch_shape
        batch_size = images.shape[0]

        images = images.reshape(batch_size, height, width)
        patches = images.reshape(
            batch_size,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
        )
        patches = patches.transpose(0, 1, 3, 2, 4)
        patches = patches.reshape(batch_size, self.num_patches, self.patch_dim)
        return patches.transpose(1, 0, 2)

    def sequence_to_images(self, sequence: Array, flatten: bool = True) -> Array:
        height, width = self.image_shape
        patch_height, patch_width = self.patch_shape
        batch_size = sequence.shape[1]

        patches = sequence.transpose(1, 0, 2)
        images = patches.reshape(
            batch_size,
            height // patch_height,
            width // patch_width,
            patch_height,
            patch_width,
        )
        images = images.transpose(0, 1, 3, 2, 4)
        images = images.reshape(batch_size, height, width)
        if flatten:
            return images.reshape(batch_size, height * width)
        return images

    def encode(self, images: Array, use_phase_init: bool = False) -> Array:
        sequence = self.images_to_sequence(images)
        return self.autoencoder.encode(sequence, use_phase_init=use_phase_init)

    def __call__(self, images: Array, use_phase_init: bool = True) -> Array:
        sequence = self.images_to_sequence(images)
        reconstruction = self.autoencoder(sequence, use_phase_init=use_phase_init)
        return self.sequence_to_images(reconstruction, flatten=True)

    @property
    def encoder(self) -> OscillatoryEncoder:
        """Compatibility alias for earlier example code."""
        return self.autoencoder.encoder

    @property
    def decoder(self) -> eqx.Module:
        """Compatibility alias for earlier example code."""
        return self.autoencoder.decoder


AmplitudeVelocityHORNCell = AmplitudeVelocityOscillatorCell
AmplitudeVelocityHORN = OscillatorySequenceLayer
AmplitudeVelocityEncoder = OscillatoryEncoder
AmplitudeVelocityDecoder = RepeatedLatentOscillatoryDecoder
AmplitudeVelocityAutoencoder = PatchOscillatoryAutoencoder


__all__ = [
    "Array",
    "OscillatorState",
    "AmplitudeVelocityOscillatorCell",
    "OscillatorySequenceLayer",
    "OscillatoryEncoder",
    "RepeatedLatentOscillatoryDecoder",
    "PositionalLatentOscillatoryDecoder",
    "AutoregressiveOscillatoryDecoder",
    "OscillatoryAutoencoder",
    "PatchOscillatoryAutoencoder",
    "AmplitudeVelocityHORNCell",
    "AmplitudeVelocityHORN",
    "AmplitudeVelocityEncoder",
    "AmplitudeVelocityDecoder",
    "AmplitudeVelocityAutoencoder",
]
