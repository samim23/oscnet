# OscNet Model API

OscNet models are organized as a small spine on top of the oscillator
primitives in `oscnet.core`.

```text
oscnet.core
  oscillator updates, oscillator modules, coupling matrices

oscnet.models
  cells -> sequence layers -> encoders/decoders -> task wrappers

examples
  data preparation, training configuration, evaluation, visualization
```

The goal is for new research ideas to add cells or wrappers without each
example re-implementing an architecture from scratch.

## Tensor Convention

Sequence models use:

```text
(time, batch, features)
```

Image patch wrappers accept flat image batches:

```text
(batch, height * width)
```

and internally convert them to patch sequences.

## Core Classes

`AmplitudeVelocityOscillatorCell`
: Projects inputs into oscillator forcing terms, steps an `(x, v)` oscillator
  state, and reads out either `concat([x, v])` or position-only features.

`OscillatorySequenceLayer`
: Scans a cell over `(time, batch, features)` inputs.

`OscillatoryEncoder`
: Uses an oscillatory sequence layer and maps the final output to a latent
  vector.

`RepeatedLatentOscillatoryDecoder`
: Projects the latent vector, repeats it for the requested number of timesteps,
  and scans an oscillatory layer over that repeated sequence. This is useful for
  reconstruction-style decoders such as image patches.

`AutoregressiveOscillatoryDecoder`
: Conditions generation on the latent vector by initializing oscillator position
  and velocity from the latent, then adding a latent drive at every decode step.
  This is useful for generated sequences such as wavelet/audio features.

`PositionalLatentOscillatoryDecoder`
: Combines a latent vector with learned per-timestep prompts before scanning an
  oscillatory decoder. This is the preferred decoder for patch-image
  reconstruction, where the model needs both sample content and explicit patch
  position.

`OscillatoryAutoencoder`
: A generic sequence autoencoder with `decoder_mode="repeat"`,
  `decoder_mode="autoregressive"`, or `decoder_mode="positional"`.

## Task Wrappers

`PatchOscillatoryAutoencoder`
: Wraps `OscillatoryAutoencoder` for flat image tensors. It converts images into
  patches, reconstructs a patch sequence, and reshapes back to flat images.

`WaveletOscillatoryAutoencoder`
: Wraps `OscillatoryAutoencoder(decoder_mode="autoregressive")` for wavelet
  feature sequences. It uses learnable nonlinear harmonic oscillators with
  wavelet-friendly defaults and keeps compatibility aliases such as
  `ProductionWaveletAutoencoder`, `encoder_cell`, and `decoder_cell`.

`FractalHORNCell`
: A HORN-style cell with `HierarchicalCouplingLayer` in place of a dense
  recurrent projection. The fractal integration example imports this reusable
  cell instead of defining architecture inside the example.

`WinfreePhaseOscillatorCell`
: A phase-only recurrent cell inspired by Winfree synchronization. It keeps a
  phase state, updates it with natural frequency, input drive, and pulse
  coupling terms, then reads out from `cos/sin` phase features.

`WinfreePhaseAutoencoder`
: A compact phase-only sequence autoencoder for WONN/Winfree-style research
  experiments. It follows the same scan, encode, decode pattern as the
  amplitude/velocity models.

## Config Objects

Config objects are available for experiment scripts that should keep model
construction separate from data and training code:

```python
import jax
from oscnet.models import OscillatoryAutoencoderConfig

config = OscillatoryAutoencoderConfig(
    input_dim=16,
    hidden_dim=64,
    latent_dim=32,
    decoder_mode="autoregressive",
)
model = config.build(jax.random.PRNGKey(0))
```

Available configs:

- `OscillatoryAutoencoderConfig`
- `PatchOscillatoryAutoencoderConfig`
- `WaveletAutoencoderConfig`
- `WinfreePhaseAutoencoderConfig`

## Extension Points

Future oscillator research should usually start by adding one of:

1. A new oscillator primitive in `oscnet.core`.
2. A new cell class in `oscnet.models` that follows the same batched timestep
   contract as `AmplitudeVelocityOscillatorCell`.
3. A thin task wrapper, if a domain needs reshape or preprocessing conventions.

For phase-only and Winfree/WONN-style research, prefer adding a phase cell that
can be scanned by the same sequence layer pattern instead of creating a separate
example-only model. `WinfreePhaseOscillatorCell` is the first concrete version
of that hook.
