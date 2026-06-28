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

`KuramotoImageGenerator`
: An Un-0-style implicit image generator. It samples random initial oscillator
  phases, evolves them through dense learned Kuramoto coupling, and decodes the
  final phase features into a flat image. It is intentionally not an
  autoencoder: there is no image input and no paired reconstruction target.
  The generator supports four conditioning modes: no conditioning, direct
  label phase shifts, the older static `class_coupling` phase-anchor drive, and
  source-faithful `class_oscillator` conditioning. In `class_oscillator`, a
  separate conditioning oscillator pool evolves under its own Kuramoto dynamics
  and drives the main oscillator pool through class-specific unidirectional
  coupling, matching the core Un-0 source-code pattern more closely. Readout
  supports `absolute`, `ref_oscillator`, legacy alias `relative`, and
  `mean_relative` phase features. Decoder modes include the default MLP
  decoder, a very low-capacity `spatial_basis` decoder that renders phase
  features through fixed Gaussian image bases, and a structured `local_basis`
  decoder where each oscillator writes trainable local patch weights through
  fixed Gaussian patch bases. `resize_conv` reshapes sin/cos phase features
  into a spatial seed and renders it with nearest-neighbor upsampling plus
  convolutions, mirroring the Un-0 reference decoder more closely.
  The recurrent oscillator pool can use the default dense coupling profile or
  `coupling_profile="distance_decay"`, which applies a fixed spatial decay
  profile and optional weak attractive bias to learned pairwise couplings.
  Generator experiments support distributional pixel matching, Un-0-style
  conditional pixel drift, fixed structural feature drift, and learned MNIST
  feature drift via a frozen `MNISTFeatureClassifier`. Conditional drift can
  optionally draw same-class positives from a host-side `MNISTDriftQueue`, which
  mirrors Un-0's per-class positive-memory mechanism more closely than
  batch-local positives. Always compare against `decoder_only` and
  `frozen_kuramoto` controls before attributing a generation result to learned
  oscillator dynamics.
  For attribution controls, `train_recurrent_dynamics` and
  `train_conditioning_dynamics` can be set independently; by default they
  inherit `train_dynamics`.
  Use it through `examples/image_mnist_kuramoto_generator.py` or the
  `oscnet.experiments.mnist_generator` API when testing oscillators as a
  generative latent dynamical prior. The MNIST generator experiment supports
  the original distributional SWD/moment objective and an Un-0-inspired
  class-conditional `pixel_drift` objective. It also supports fixed structural
  feature drift through `loss_mode="feature_drift"` or
  `loss_mode="pixel_feature_drift"`, using pooled layout, edge, profile, and
  moment features for MNIST-scale probes. Generator experiment summaries include
  a `success_diagnostics` block with decoder/dynamics parameter fractions,
  estimated operation fractions, throughput, and phase-trajectory movement
  proxies for attribution-focused comparison.

`KuramotoPhaseVAE`
: A MNIST-native generative autoencoder that encodes images into a Gaussian
  latent, interprets the sampled latent as oscillator phase, optionally evolves
  it with Kuramoto dynamics, and decodes final phase features back to image
  probabilities. It is intentionally easier and more controlled than
  `KuramotoImageGenerator`: it uses paired reconstruction plus KL loss so MNIST
  generation quality can be debugged without relying on unpaired drift losses.
  Use `model_family="phase_vae"` for trainable phase dynamics,
  `"frozen_phase_vae"` for a fixed oscillator transform, and
  `"phase_vae_no_dynamics"` for the matched VAE control with dynamics removed.
  `phase_readout_mode` can be `"absolute"`, `"mean_relative"`, or
  `"ref_oscillator"` when testing whether relative phase geometry matters. The
  experiment entry point is `examples/image_mnist_phase_vae.py` or the
  `oscnet.experiments.mnist_phase_vae` API.

`PhaseRateFlowField`
: A native image-field generative model for rectified-flow and denoising
  experiments. It keeps the noisy image on the visible grid, initializes a local
  phase field and rate/content field from `(x_t, t)`, evolves them with tied
  local phase-rate oscillator dynamics, and reads out a velocity field
  `dx/dt`. Use this when testing oscillators as the generative medium itself
  rather than as a latent VAE transform. The MNIST experiment exposes
  `model_family="phase_flow"` for trainable dynamics,
  `"frozen_phase_flow"` for a fixed oscillator reservoir, and
  `"phase_flow_no_dynamics"` for the matched no-settling control. The entry
  point is `examples/image_mnist_phase_flow.py` or the
  `oscnet.experiments.mnist_phase_flow` API.

`CoarseGlobalPhaseRateFlowField`
: A multiscale ONN-native extension of `PhaseRateFlowField`. It runs the same
  fine phase-rate image field, plus a lower-resolution coarse phase-rate band
  that is initialized from pooled noisy image evidence and coupled back into
  the fine phases through relative-phase pull. Use
  `model_family="coarse_phase_flow"` to test whether long-range/coarse phase
  coordination helps local stroke fragments close into whole shapes without
  adding U-Net-style tensor skips or a latent decoder. The MNIST phase-flow
  experiment also supports `sample_method="euler"` and `"heun"` so sampling
  integration can be tested separately from the learned dynamics. Set
  `position_features=True` or pass `--position-features` to add fixed
  coordinate/phase features to the field initialization; this tests spatial
  reference frames separately from recurrent oscillator dynamics. Set
  `closure_loss_weight > 0` or pass `--closure-loss-weight` to add a
  train-time low-frequency endpoint loss at `14x14` and `7x7`, which probes
  whether whole-shape binding pressure helps phase-flow samples close into
  coherent digits. Set `target_representation="sobel_edges"` or
  `"signed_distance"` to train the same visible oscillator field on contour or
  smooth shape-field targets rather than raw pixels. Set
  `target_representation="signed_distance_flow"` to train a three-channel
  potential field: signed distance plus normalized x/y gradient direction.
  This tests a more explicitly flow-like target while keeping signed distance
  as the primary decoded metric channel. Set
  `target_representation="pixels_signed_distance"` to train a two-channel
  visible field: pixel occupancy in channel 0 and a smooth signed-distance
  shape target in channel 1. The underlying phase-flow models expose
  `value_channels` for these multi-channel visible-field experiments. Use
  `target_representation="centered_pixels_signed_distance"` to train the same
  two-channel field in centered `[-1, 1]` coordinates while decoding channel 0
  back to pixel space for metrics and artifacts. Set
  `sample_readout_mode="shape_gated"` to use the auxiliary shape channel as a
  smooth gate on the pixel channel when computing sample metrics and PNG
  artifacts. Set `sample_schedule="shape_guided"` to stage Euler sampling so
  the shape channel settles first and pixel-channel updates open later. Set
  `basin_t_values` in the MNIST experiment to evaluate basin-of-attraction
  recovery from partially real chord states `x_t = (1 - t) noise + t data`.
  Basin metrics include the starting paired MSE and final paired MSE, so this
  diagnostic can distinguish true endpoint improvement from simply starting
  close to the target. Set `basin_noise_mode` to `"gaussian"`, `"uniform"`,
  `"salt_pepper"`, or `"zeros"` to probe robustness to different basin
  endpoints without retraining the model.
  `oscnet.experiments.mnist_shape_pixel` reuses the same model family for the
  next two-stage test: channel 0 is a noisy/generated pixel image, channel 1 is
  a clamped signed-distance shape condition, and only the pixel channel is
  sampled. This isolates shape-field settling from pixel rendering.

`RecurrentConvFlowField`
: A matched non-oscillatory recurrent-flow control for `PhaseRateFlowField`.
  It keeps the same rectified-flow task, visible image grid, time/class
  conditioning, tied local recurrence, sampling path, and artifact interfaces,
  but replaces phase/rate dynamics with a gated local convolutional hidden
  field. Use `model_family="recurrent_conv_flow"` before claiming that a
  phase-flow result beats ordinary local recurrent spatial machinery.

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
- `RecurrentConvPriorRefinementPatchDenoiserConfig`
- `KuramotoImageGeneratorConfig`
- `ConvLSTMPatchDenoiserConfig`
- `WaveletAutoencoderConfig`
- `WinfreePhaseAutoencoderConfig`
- `WinfreeFieldAutoencoderConfig`
- `WinfreePatchAutoencoderConfig`
- `WinfreeConditionalPatchDenoiserConfig`
- `WinfreeRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeGlobalRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeCoarseRatePhaseConditionalPatchDenoiserConfig`
- `WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiserConfig`
- `WinfreePriorRefinementPatchDenoiserConfig`

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
For distributional generation tasks, use `KuramotoImageGenerator` and keep
decoder-only and frozen-reservoir controls in the first sweep.
