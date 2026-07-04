# OscNet Model API

OscNet models are organized as a small spine on top of the oscillator
primitives in `oscnet.core`.

```text
oscnet.core
  oscillator updates, oscillator modules, coupling matrices/topologies

oscnet.models
  cells -> sequence layers -> encoders/decoders -> task wrappers

examples
  data preparation, training configuration, evaluation, visualization
```

The goal is for new research ideas to add cells or wrappers without each
example re-implementing an architecture from scratch.

`oscnet.core.coupling`
: Shared topology builders for oscillator networks. Use
  `distance_decay_coupling_profile`, `local_radius_coupling_profile`,
  `dense_coupling_profile`, and `normalize_coupling_profile` when an experiment
  needs fixed spatial coupling. `normalization="row_sum"` keeps each non-empty
  row at a target gain scale, which is useful when comparing local, dense, and
  distance-decay media without accidentally changing total recurrent input.

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
  label phase shifts, `class_coupling` class-specific dynamic drive, and
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
  profile and optional weak attractive bias to learned pairwise couplings. Use
  `coupling_profile="local_radius"` for a sparse binary spatial mask; in that
  mode `coupling_length_scale` is the local interaction radius on the
  normalized oscillator grid. Set `coupling_normalization="row_sum"` to
  normalize each non-empty profile row to the generator's recurrent gain scale.
  This is the preferred diagnostic when asking whether a topology helps beyond
  merely changing total coupling input. `conditioning_strength` scales the
  conditioning drive in `class_coupling` and `class_oscillator` modes without
  changing the main recurrent coupling.
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
  Use it through `examples/image_mnist_generator.py` or the
  `oscnet.experiments.mnist_generator` API when testing oscillators as a
  generative latent dynamical prior. The MNIST generator experiment supports
  the original distributional SWD/moment objective and an Un-0-inspired
  class-conditional `pixel_drift` objective. It also supports fixed structural
  feature drift through `loss_mode="feature_drift"` or
  `loss_mode="pixel_feature_drift"`, using pooled layout, edge, profile, and
  moment features for MNIST-scale probes. Generator experiment summaries include
  a `success_diagnostics` block with decoder/dynamics parameter fractions,
  estimated operation fractions, throughput, and phase-trajectory movement
  proxies for attribution-focused comparison. Set
  `quality_classifier_epochs > 0` in the experiment config, or pass
  `--quality-classifier-epochs`, to train a small frozen classifier for sample
  label-accuracy/confidence metrics when pixel losses and visual quality
  disagree.

`HORNImageGenerator`
: A second-order oscillator generator with explicit position and velocity
  state. It shares the same conditioning modes, decoder modes, losses, and
  experiment harness as `KuramotoImageGenerator`, but replaces phase-only
  Kuramoto updates with a damped HORN-style oscillator:
  position/velocity state is sampled as generative noise, recurrent coupling
  supplies spring-like interaction, and the decoder reads bounded
  `[position, velocity]` features. Use it through
  `examples/image_mnist_generator.py --model-family horn` when testing
  HORN generator claims against Kuramoto, `frozen_horn`, and
  `horn_decoder_only` controls. The recommended MNIST generator recipe uses
  `decoder_mode="resize_conv"`, `readout_mode="mean_relative"`, strict dynamic
  class coupling, variable-depth training with
  `train_settling_steps=(16, 32, 48)`, sparse local coupling via
  `coupling_profile="local_radius"` with a small radius, and slightly higher
  HORN damping. The example entrypoint is
  `python examples/image_mnist_generator.py`, which defaults to
  `sparse_horn_mnist_recommended`. The stable CIFAR-10 RGB generator alias is
  `sparse_horn_cifar10_rgb_current`; it currently points to the normalized
  local sparse HORN recipe rather than any of the multiscale hierarchy probes.
  For spatial HORN generators with `decoder_mode="resize_conv"`,
  `resize_conv_seed_layout="retinotopic"` preserves the oscillator grid in the
  seed image instead of flattening state features into arbitrary seed pixels.
  `resize_conv_seed_min_channels` can pad a retinotopic seed with extra local
  oscillator observables, which is useful for seed-width controls such as a
  four-channel single-mode HORN readout.
  This is useful for diagnostics and future readout work, but the first CIFAR
  RGB probe found that it improves frozen-state reconstructability before it
  improves direct sample quality. A param-matched follow-up confirmed that the
  issue is not simply the smaller retinotopic decoder.
  Optional HORN self-feedback can be enabled with
  `output_feedback_strength > 0`. `output_feedback_mode="state_proxy"` feeds a
  cheap centered local proxy, `tanh(position) + 0.5 * tanh(velocity)`, back
  into acceleration with one learned gain per oscillator.
  `output_feedback_mode="image"` decodes the current image during every
  settling step and pools it back to oscillator sites; it is closer to a
  readout-in-the-loop attractor, but it is much more expensive with
  `resize_conv` and should be reserved for tiny diagnostics.
  `resonant_readout_strength > 0` enables an optional shared HORN resonant
  filter-bank readout on top of `resize_conv`. This branch decodes local
  phase, velocity, phase-alignment, local order, and energy observables through
  a small shared spatial filter bank. It is meant to test whether settled HORN
  field structure can be exposed without adding a large conventional decoder.
  The first CIFAR RGB pilot found `sparse_horn_cifar10_rgb_current_resonant005`
  promising for semantic/diversity metrics but not yet a sharpness fix.
  This makes HORN the best current OscNet generator branch while keeping the
  claim precise: it is a conditional MNIST generator with useful learned
  oscillator dynamics, not yet a general image-generation win over all
  conventional models. Use `sparse_horn_mnist_strict` to probe the stricter
  route where class information cannot enter as a direct initial
  label shift; this route currently starts near chance and becomes readable
  through oscillator settling.

`MultiModeHORNImageGenerator`
: A structured-capacity HORN generator where each spatial site owns several
  frequency-band HORN modes. Same-frequency modes couple across nearby spatial
  sites, while different modes couple within the same site. This tests a
  filter-bank style hypothesis: CIFAR capacity may need richer local spectral
  state, not simply more flat oscillator sites. Use
  `model_family="multimode_horn"` or the preset
  `sparse_horn_cifar10_rgb_current_multimode2`. The first CIFAR RGB pilot
  found that the two-mode variant strongly beats a flat 512-site HORN field on
  class consistency and attractor metrics, while still not solving visual
  sharpness or high-frequency detail. Treat it as a structured-capacity lead,
  not a new default yet. The companion
  `sparse_horn_cifar10_rgb_current_multimode2_retinotopic` preset keeps the
  two-mode field retinotopic in the resize-conv seed; it is best used for
  state-fitting/readout diagnostics until direct generation catches up.

`CoarseToFineHORNImageGenerator`
: A multiscale HORN generator that evolves a small coarse oscillator bank in
  parallel with the fine HORN field. The coarse state is not decoded directly;
  instead, learned coarse-to-fine displacement coupling drives the fine
  oscillator acceleration. Use it to test whether class/global structure is
  better carried by a low-resolution oscillatory mode while the fine field
  keeps sparse local coupling for texture/detail. The generator exposes
  `num_coarse_oscillators`, `coarse_coupling_profile`,
  `coarse_coupling_normalization`, `coarse_to_fine_strength`, and
  `coarse_conditioning_strength`; `coarse_to_fine_profile`,
  `coarse_to_fine_normalization`, and `coarse_to_fine_length_scale` can make
  the top-down projection dense, distance-decayed, or sparse local-radius.
  Summaries count coarse recurrent and conditioning parameters separately and
  report coarse-to-fine profile density/row-gain diagnostics. Trace diagnostics
  also include coarse state energy/update/acceleration proxies,
  coarse-coupling disagreement, and coarse-to-fine disagreement, which helps
  distinguish useful multiscale coordination from a top-down clamp. In the
  generator CLI, use `--model-family coarse_horn` or the experimental preset
  `sparse_horn_cifar10_rgb_coarse16_normlocal`. The current local-radius
  probe is `sparse_horn_cifar10_rgb_coarse16_normlocal_gentle_local050`, which
  weakens coarse-to-fine drive and restricts it to nearby fine oscillators.

`MultiscaleHORNImageGenerator`
: A layered HORN generator for explicit oscillator stacks. It generalizes the
  single coarse-to-fine model into any number of auxiliary populations plus the
  decoded fine field. Each auxiliary layer has its own oscillator count,
  frequency scale, recurrent coupling profile, and optional class-coupling
  drive. Adjacent layers are connected by directed vertical coupling specs, so
  the same class can represent one-way coarse-to-fine drive, bidirectional
  feedback, phase-lagged projections, and no-vertical controls. The decoder
  still reads only the fine field, which keeps attribution focused on whether
  vertical settling improves the fine oscillator substrate rather than adding a
  hidden image decoder.

  Use `model_family="multiscale_horn"` for the active model,
  `"multiscale_horn_decoder_only"` for decoder-only controls, and
  `"frozen_multiscale_horn"` for frozen-dynamics controls. CIFAR RGB probe
  presets start with
  `sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005` and
  `sparse_horn_cifar10_rgb_multiscale16_64_no_vertical`. Summaries count
  auxiliary recurrent params, vertical recurrent params, multiscale
  conditioning params, vertical profile density/row gain, and per-layer
  energy/update/coupling proxies. Optional coarse supervision is available
  through `coarse_auxiliary_weight`, `coarse_auxiliary_target_size`, and
  `multiscale_auxiliary_readout_layer`; this attaches a low-resolution readout
  to an auxiliary layer. `coarse_auxiliary_loss_mode="mse"` trains that readout
  against a paired downsampled image target.
  `coarse_auxiliary_loss_mode="distributional"` instead matches
  low-resolution batch/class statistics, which is less likely to force the
  coarse layer into one paired-image answer during unpaired class-conditional
  generation. Use the `*_auxlow8` CIFAR RGB presets for the historical MSE
  objective and the `*_auxdist8` presets for the distributional objective.
  `multiscale_readout_fusion_strength` can blend a channel-first upsample of
  that auxiliary readout back into the final image. Keep it at `0.0` by
  default; small values such as `0.10` or `0.25` are conservative probes for
  whether the coarse layer contains a useful rendering scaffold. This is
  intentionally lower capacity than adding a second full decoder, so positive
  results still point back to the layered oscillator state.
  `multiscale_readout_gate_mode="seed_film"` is the preferred non-blending
  probe when using `decoder_mode="resize_conv"`: it projects the selected
  auxiliary oscillator state into a FiLM-style scale/shift on the fine
  resize-conv seed tensor. Unlike readout fusion, it does not paste a coarse
  RGB image into the output. It asks whether the coarse field can change how
  the fine field is rendered. `multiscale_readout_gate_strength` controls the
  modulation scale, and `multiscale_readout_gate_init_scale=0.0` starts from
  the ungated model while remaining trainable.
  Prefer `coarse_readout_consistency_weight` when you want the coarse scaffold
  to shape training without pasting coarse pixels into samples. It downsamples
  the final image and matches it to the same-trajectory auxiliary readout with
  a stop-gradient auxiliary target. `coarse_readout_consistency_onset_epoch`
  can delay that loss so the auxiliary scaffold has a short warmup before it
  guides the fine readout.
  `frequency_objective_weight` is an experiment-level image-spectrum loss for
  generated-image training. It matches low/mid/high frequency-band ratios plus
  a Laplacian edge statistic to real images. Use it as a diagnostic for blur or
  missing high-frequency detail; keep it off for stable reference presets until
  a paired probe shows a real quality gain.
  `patch_objective_weight` is the local-detail follow-up. It compares unpaired
  small patch distributions with sliced-Wasserstein projections, optionally
  including Laplacian-edge patches. This is meant to reward object-local detail
  more directly than global frequency matching, which can be satisfied by
  border halos or color-channel ringing. `patch_objective_offsets` repeats the
  patch comparison on shifted grids, and `patch_objective_patch_sizes` lets one
  projection bank score several patch scales. Use those when a fixed-grid patch
  objective improves detail but leaves tiled or striped artifacts.
  `multiscale_vertical_target_gate` can
  route vertical drive into all decoded fine oscillators, only the
  class-conditioning target subset, or only the complementary subset; this is
  useful for testing whether hierarchy should act through selected oscillator
  columns rather than as a broad top-down spring.
  `multiscale_vertical_soft_gate_floor` relaxes a selective gate by giving
  non-target fine columns a fixed fraction of the vertical profile. It is useful
  as a diagnostic for "mostly selective, weakly contextual" top-down
  modulation; the current CIFAR RGB probe found that `0.25` helps unsigned
  selective gain slightly but hurts selective signed gain.
  `multiscale_vertical_signal_scale` globally scales the vertical signal before
  it enters the target layer. Keep it at `1.0` for compatibility; use larger
  values only as calibrated hierarchy probes. `multiscale_vertical_mode`
  controls how that vertical signal enters the fine layer. `additive` uses the
  vertical projection as a direct source-minus-target acceleration term.
  `gain_modulation` turns the projection into a bounded nonnegative gain on the
  target layer's local recurrent and conditioning drives. `signed_gain` lets
  that gain become inhibitory, so a coarse layer can suppress selected
  fine-layer responses rather than only amplify or damp them. `dual_gain`
  combines a broad all-column positive gain route with a selective signed route
  into the configured target columns; `multiscale_vertical_broad_gain_scale`
  and `multiscale_vertical_selective_gain_scale` set those two route strengths.
  `multiscale_vertical_gain_target` selects which fine-layer HORN term a
  non-additive vertical gain modulates. `drive` is the backward-compatible
  default and scales local recurrent coupling plus class conditioning together.
  `coupling` scales only local recurrent interaction, `conditioning` scales
  only class drive, and `damping` uses the vertical signal as a nonnegative
  damping gain. This is a diagnostic for whether top-down hierarchy should
  shape the medium, the label drive, or settling stability instead of acting as
  generic gain.
  `multiscale_vertical_gain_normalization` can keep non-additive gain
  homeostatic: `center` removes per-sample mean modulation across target
  columns, and `center_rms` also rescales the centered signal to
  `multiscale_vertical_gain_target_std`. Use this when testing whether
  hierarchy can redistribute influence without increasing the fine field's
  total drive energy. `multiscale_vertical_schedule` can keep the route
  `constant`, switch it on after `multiscale_vertical_onset_step` with
  `delayed`, or ramp it in with `linear_ramp` over
  `multiscale_vertical_ramp_steps`. This is useful for testing whether
  top-down gain acts best as an immediate condition or a late settling bias.
  `multiscale_feedback_signal_mode` controls bottom-up feedback projections:
  `position` preserves the historical phase/position-only feedback signal,
  while `state` feeds back bounded position-plus-velocity evidence from finer
  layers into coarser layers. Use this to test whether the coarse layer benefits
  from actual fine dynamical state rather than just another phase spring.
  `multiscale_feedback_source_gate` controls which fine columns a bottom-up
  feedback projection listens to: `all`, `conditioning`, `non_conditioning`,
  or `weighted`. This is the source-side counterpart to
  `multiscale_vertical_target_gate`, useful for testing whether coarse layers
  should read from class-driven fine columns, their complement, or the whole
  fine field. `weighted` uses `multiscale_feedback_source_mix =
  (conditioning_weight, non_conditioning_weight)` and mean-normalizes the
  resulting source mask, so fixed-ratio probes change the routing balance
  without changing average feedback strength.
  These modes test a slower/coarser rhythm as a modulator of fine dynamics,
  rather than as another spring attached to the fine state. The CIFAR RGB
  `*_gain_*`, `*_signed_gain_*`, `*_dual_gain_*`, and `*_normstd*` presets are
  the current probes for this mechanism. The short alias
  `sparse_horn_cifar10_rgb_hierarchy_lead` points at the current active
  hierarchy mechanism lead. It is deliberately separate from
  `sparse_horn_cifar10_rgb_current`, because hierarchy currently improves
  basin/diversity diagnostics more reliably than final CIFAR rendering quality.
  For sample-time causality audits,
  `multiscale_vertical_intervention` can be set to `normal`, `zero`,
  `shuffle_batch`, or `flip`, and `multiscale_vertical_intervention_scale` can
  attenuate the route without retraining. The first CIFAR RGB audit found that
  the unscaled vertical-gain path was nearly silent at sample time; calibrated
  `vscale10/vscale30` and `dual_gain` variants make it causal, but have not yet
  beaten the no-vertical or broad-gain-only quality frontier.

`StateMLPImageGenerator`
: A non-oscillatory latent-state control for the HORN generator. It keeps the
  same position/velocity state, relative readout, and decoder surface, but
  replaces oscillator dynamics with a residual MLP transition. Use it through
  `model_family="state_mlp"`, `"frozen_state_mlp"`, or
  `"state_mlp_decoder_only"` when checking whether HORN's recurrent field beats
  a conventional trainable latent-state mapper under the same objective and
  readout. In `class_coupling` and `class_oscillator` modes, StateMLP receives
  the same learned class-drive term as HORN, so it is a matched no-direct-label
  control rather than a label-blind baseline.

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
  `train_noise_mode` to `"gaussian"`, `"uniform"`, `"salt_pepper"`, `"zeros"`,
  or `"mixed"` to change the endpoint distribution used for rectified-flow
  training chords; `"mixed"` samples a per-example mixture of those endpoints.
  Set
  `basin_t_values` in the MNIST experiment to evaluate basin-of-attraction
  recovery from partially real chord states `x_t = (1 - t) noise + t data`.
  Basin metrics include the starting paired MSE and final paired MSE, so this
  diagnostic can distinguish true endpoint improvement from simply starting
  close to the target. Set `basin_noise_mode` to `"gaussian"`, `"uniform"`,
  `"salt_pepper"`, or `"zeros"` to probe robustness to different basin
  endpoints without retraining the model. Use `basin_noise_modes` for a
  comma/list equivalent that evaluates several endpoint modes after one
  training run and stores the nested result in `phase_flow.basin_by_noise`.
  `oscnet.experiments.mnist_shape_pixel` reuses the same model family for the
  next two-stage test: channel 0 is a noisy/generated pixel image, channel 1 is
  a clamped signed-distance shape condition, and only the pixel channel is
  sampled. This isolates shape-field settling from pixel rendering. Its
  `shape_condition_t_values` and `shape_condition_noise_modes` diagnostics
  corrupt the scaffold at evaluation time to test whether the renderer can
  tolerate imperfect shape fields from an upstream oscillator stage. Set
  `sample_readout_mode="shape_gated"` to use the clamped scaffold as an
  explicit soft amplitude gate on sampled pixels.

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
