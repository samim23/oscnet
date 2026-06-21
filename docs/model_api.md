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
  Reconstruction heads can be left unconstrained with
  `output_activation="identity"` or bounded to image-style ranges with
  `output_activation="sigmoid"` or `output_activation="tanh01"`.

## Task Wrappers

`PatchOscillatoryAutoencoder`
: Wraps `OscillatoryAutoencoder` for flat image tensors. It converts images into
  patches, reconstructs a patch sequence, and reshapes back to flat images.
  It forwards `output_activation` to the underlying sequence autoencoder.

`FeedForwardPatchAutoencoder`
: A non-oscillatory patch autoencoder control for attribution experiments. It
  uses the same flat-image patch convention and latent/code API as the MNIST
  oscillatory models, but replaces recurrent oscillator dynamics with learned
  position-aware feedforward projections plus an optional sequence latent output
  skip. Use it when a benchmark needs to prove that an oscillatory model is
  doing more than a direct latent decoder scaffold.

`RecurrentConvPatchDenoiser`
: A non-oscillatory direct denoising control for masked-completion experiments.
  It keeps the image as a patch grid, injects corrupted patches into hidden
  features, runs tied local convolutional message passing for a fixed number of
  recurrent steps, and reads out clean patches. Use it to test whether Winfree
  gains come from phase dynamics specifically or from generic iterative local
  spatial refinement.

`ConvLSTMPatchDenoiser`
: A stronger non-oscillatory recurrent denoising control on the same patch grid.
  It uses ConvLSTM-style input, forget, output, and candidate gates over local
  neighborhoods, giving the recurrent baseline memory and gating capacity
  without changing the task interface. Use it before claiming that an
  oscillatory block-inpainting result beats ordinary recurrent spatial
  machinery.

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

`WinfreeFieldLayer`
: A WONN-inspired phase-field layer over `(batch, positions, channels)` states.
  It keeps an explicit `theta` phase field and input-conditioned `omega` drive,
  wraps phase after each recurrent step, and evolves
  `dtheta = omega + cos(theta) * coupling(sin(theta))`.
  The layer also supports `si_func="mlp"` for learned periodic sensitivity and
  influence functions, grouped patch influence via `group_size`, and soft
  spatial locality with `coupling_decay_length`. Coupling can be a dense
  `matrix`, fixed local `conv`, local data-adaptive `adaptive` attention, or
  residual `conv_adaptive` coupling over the neighborhood defined by
  `coupling_kernel_size`. It also supports `conv_matrix`, which preserves the
  fixed local conv field and adds a weak distance-decayed matrix field. Use
  `adaptive_coupling_strength` to scale the residual branch in `conv_adaptive`
  and `conv_matrix` modes.

`WinfreeFieldAutoencoder`
: A sequence autoencoder built from `WinfreeFieldLayer`. It preserves the
  toroidal phase-field core while using OscNet's regular encode/decode API.
  The decoder can optionally use `latent_readout="phase_bias"` or
  `latent_readout="concat"` to condition the final phase-feature readout
  directly on the latent vector.

`WinfreePatchAutoencoder`
: A flat-image wrapper around `WinfreeFieldAutoencoder` for patch-based image
  reconstruction experiments. This is the minimal WONN-aligned model family for
  the MNIST reference harness. Use `--winfree-si-func mlp` and
  `--winfree-group-size 2` in the MNIST CLI to enable the closer WONN-style
  learned/grouped variant.

`WinfreeConditionalPatchDenoiser`
: A direct image-to-image Winfree phase-field wrapper for masked completion and
  denoising tasks. It does not use a global latent bottleneck or latent output
  skip: corrupted input patches initialize local phase (`theta`) and frequency
  (`omega`) fields, the Winfree field evolves for a fixed number of recurrent
  steps, and clean patches are read from final phase features. This is the
  preferred starting point when the experiment is meant to test oscillatory
  dynamics as a spatial completion process rather than as a generic
  autoencoder decoder. For KoPE-style initialization probes, use
  `phase_init="rotary_2d"` or the MNIST CLI flag `--winfree-phase-init
  rotary_2d` to seed the phase field with deterministic multi-frequency 2D
  spatial phases instead of learned random positional phases.

`WinfreeRatePhaseConditionalPatchDenoiser`
: A direct masked-completion wrapper that keeps the Winfree phase field but adds
  a separate local content/rate field. The phase state evolves with
  `WinfreeFieldLayer`; the content field evolves through tied local convolution
  gated by phase features, and the readout uses both content and phase. This is
  the first reusable test of the "phase coordinates, content carries evidence"
  hypothesis. In the MNIST CLI, use `--model-family winfree_rate_phase`.

`WinfreeGlobalRatePhaseConditionalPatchDenoiser`
: A slow/global carrier variant of the rate-phase denoiser. It composes the
  local rate-phase field with a separate one-position Winfree phase band
  initialized from the whole corrupted image. The global phase gates local
  content propagation during recurrent settling. This is the first reusable
  test of the "slow rhythm coordinates fast local work" hypothesis. In the
  MNIST CLI, use `--model-family winfree_global_rate_phase`.

`WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser`
: A coarse spatial phase-mesh variant of the rate-phase denoiser. It keeps the
  fine local rate-phase field, initializes a separate low-resolution Winfree
  phase grid from spatially pooled corrupted patches, and interpolates coarse
  phase gates back down to the fine patch grid. Use it for block-occlusion
  probes where a one-node global carrier is too weak but a U-Net-style skip
  would test the wrong mechanism. In the MNIST CLI, use `--model-family
  winfree_coarse_global_rate_phase`, `--winfree-coarse-grid-size 2` or `4`, and
  `--winfree-global-phase-control shuffle` for the phase-shuffled control.

`WinfreeCoarseRatePhaseConditionalPatchDenoiser`
: A stronger multiscale rate-phase denoiser that gives the coarse band its own
  rate/content field, not just phase. The coarse rate field evolves under coarse
  phase gates and is projected back down into the fine rate field as additive
  content. Use it to test whether block occlusion needs top-down shape/content
  transport rather than interpolated coarse phase gates alone. In the MNIST CLI,
  use `--model-family winfree_coarse_rate_phase`,
  `--winfree-global-content-strength <value>`, and
  `--winfree-global-content-control shuffle` for the content-shuffled control.

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
- `FeedForwardPatchAutoencoderConfig`
- `RecurrentConvPatchDenoiserConfig`
- `ConvLSTMPatchDenoiserConfig`
- `WaveletAutoencoderConfig`
- `WinfreePhaseAutoencoderConfig`
- `WinfreeFieldAutoencoderConfig`
- `WinfreePatchAutoencoderConfig`
- `WinfreeConditionalPatchDenoiserConfig`
- `WinfreeRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig`

Reference experiment CLIs also expose optional training diagnostics such as
`--latent-variance-weight` and `--latent-std-floor` for probing latent-collapse
failure modes.

## Extension Points

Future oscillator research should usually start by adding one of:

1. A new oscillator primitive in `oscnet.core`.
2. A new cell class in `oscnet.models` that follows the same batched timestep
   contract as `AmplitudeVelocityOscillatorCell`.
3. A thin task wrapper, if a domain needs reshape or preprocessing conventions.

For phase-only and Winfree/WONN-style research, prefer adding a reusable phase
cell or phase-field layer instead of creating a separate example-only model.
`WinfreePhaseOscillatorCell` covers compact sequence experiments, while
`WinfreeFieldLayer` captures the WONN-style toroidal phase-field update. For
partial-observation tasks such as masked MNIST, use
`WinfreeConditionalPatchDenoiser` before adding latent-decoder machinery; it
keeps the attribution question focused on local phase-field dynamics.
