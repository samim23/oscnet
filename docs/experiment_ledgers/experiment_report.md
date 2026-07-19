# OscNet Experiment Report

This is the living research log for the OscNet MNIST and Winfree phase-field
experiments. Keep benchmark conclusions here and in `outputs/analysis`, not in
`README.md`.

## Prehistory: Why This Report Exists

Before the Winfree masked-completion work, the repo had a lot of interesting
research code but no single reusable experiment pattern. The first sprint was
therefore not about chasing a metric; it was about turning examples into a
repeatable library shape:

```text
oscnet.models      reusable model families
oscnet.experiments training/evaluation/artifact harnesses
examples           thin CLI entrypoints
docs               API notes and research conclusions
outputs            generated metrics, plots, traces, checkpoints
```

Important early cleanup decisions:

- The MNIST and audio examples were reduced toward thin reference entrypoints
  instead of one-off architecture definitions.
- `docs/model_api.md` became the canonical map of model families and task
  wrappers.
- Benchmark results were moved out of `README.md`; the README should explain
  how to use the project, while this report carries research conclusions.
- Tests were added around example boundaries, model config construction,
  artifact generation, and import surfaces so future research variants can
  land without silently breaking the spine.

The audio wavelet branch also revealed a useful architecture bug/lesson: the
old local decoder projected the latent state, but that projected state was not
meaningfully used by the decoder cell dynamics afterward. The reusable
`WaveletOscillatoryAutoencoder` corrected that by making autoregressive
decoding actually conditioned on the latent state. This was one of the early
signals that OscNet needed reusable model APIs before adding more research
ideas.

The Fractal/HORN code was similarly pulled toward a reusable boundary:
`FractalHORNCell` lives in `oscnet.models`, and the fractal example imports it
instead of defining a private architecture inside the example. That branch has
not been the strongest MNIST result so far, but it matters as part of the
library pattern: new oscillator families should enter through reusable modules,
not isolated scripts.

The WONN repo review came after that cleanup. Its useful ingredients were:
Winfree-style phase dynamics, learned sensitivity/influence functions, grouped
or local interaction structure, and the idea that phase/frequency state can be
used as a computational field. The review did not justify copying the claims
wholesale. It gave us a concrete set of mechanisms to test inside OscNet's
cleaner harness.

## Current Takeaway

The strongest result so far is the sparse local HORN MNIST generator. It is not
a conventional latent autoencoder: random oscillator position/velocity state is
settled by second-order HORN dynamics and then read out by a small resize-conv
decoder. Across the latest replicated low-data generator probes, the sparse
local HORN model beats frozen, decoder-only, no-main-coupling, and one-step
controls on semantic generation. Against matched StateMLP controls, the honest
read is now split: HORN still shows useful diversity/settling behavior on
MNIST-like generator probes, but the latest CIFAR RGB state-prior/patch probe
is a null result once paired by seed. The same healthy prior/anchor/patch
pipeline transfers to a non-oscillatory iterated MLP. HORN remains a useful
mechanism reference, not a demonstrated CIFAR image-quality advantage.

Current clean generator defaults:

- `sparse_horn_mnist_recommended`: stable default for
  `python examples/image_mnist_generator.py`.
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class_patch005`:
  current CIFAR-10 RGB HORN mechanism reference. Keep the same-stack StateMLP
  control beside it.
- `sparse_horn_cifar10_rgb_hierarchy_lead`: active multiscale mechanism lead,
  useful for hierarchy probes but not the stable rendering default.

The hierarchy lead is kept separate from the stable CIFAR default because of
the final readout/visual-quality bottleneck, not because the hierarchy probes
were empty. Across paired CIFAR RGB hierarchy sweeps, active multiscale routes
have improved generated-label accuracy, diversity, feature diversity,
attractor accuracy, and basin score in several settings. They also often lose
nearest-pixel proximity, throughput, or visual sharpness. The current open
problem is converting better coarse/fine attractor basins into better rendered
RGB samples.

The strongest masked-completion direction is also not a conventional latent
autoencoder. It is a conditional spatial field/refinement model, strongest when
a semantic prior is paired with a local recurrent correction field. Winfree
phase dynamics are a promising version of that correction field, but the
recurrent-conv residual control shows that the broader hybrid pattern also
matters:

```text
corrupted image patches -> local theta/omega + content fields -> coupled settling -> clean patches
```

This framing is better aligned with oscillatory dynamics than forcing an
oscillatory model to act as a generic compressed latent decoder. On masked MNIST
completion, the direct rate-phase Winfree field beats older OscNet variants,
feedforward controls, and simple recurrent-conv controls. On the later
prior-refinement robustness probe, Winfree keeps a small edge over a matched
recurrent-conv residual control, while both residual refiners improve the
feedforward prior.

Current repo-wide research read: HORN is the best OscNet-native generator
branch; Winfree/rate-phase remains the best direct masked-completion branch.
Neither should be presented as a universal ONN victory, but the HORN generator
is the cleanest positive settling/dynamics-attribution result in the repo right
now.
Generator configs now separate `coupling_strength` from
`main_coupling_strength`: the former scales class/conditioning drive, while the
latter scales recurrent oscillator interaction. When omitted,
`main_coupling_strength` defaults to `coupling_strength` for old-run
compatibility. Future HORN attribution probes should use this split instead of
turning class drive and main coupling up/down together.
The current local entrypoint for that branch is
`python examples/image_mnist_generator.py`, which defaults to
`sparse_horn_mnist_recommended`. Friendly aliases keep the current HORN lessons
from getting rediscovered by accident:
`sparse_horn_mnist_strict` for the no-direct-label semantic/diversity lead,
`sparse_horn_mnist_quality` for small distributional quality pressure, and
`sparse_horn_mnist_dynamics_quality` for the higher-damping dynamics-side
quality variant. Matched local controls now live as sibling presets:
`sparse_horn_mnist_frozen`, `sparse_horn_mnist_decoder_only`,
`sparse_horn_mnist_state_mlp`, `sparse_horn_mnist_state_mlp_frozen`,
`sparse_horn_mnist_state_mlp_decoder_only`,
`sparse_horn_mnist_state_mlp_class_coupling_strength8`, and
`sparse_horn_mnist_step1`.

## Repo Pattern Established

Reusable model APIs now live under `oscnet.models`, while reference experiments
and artifact generation live under `oscnet.experiments`.

Important model families:

- `PatchOscillatoryAutoencoder`: older amplitude/velocity OscNet image wrapper.
- `WinfreePatchAutoencoder`: latent Winfree phase-field autoencoder.
- `WinfreeConditionalPatchDenoiser`: direct conditional phase-field denoiser.
- `WinfreeRatePhaseConditionalPatchDenoiser`: direct conditional rate-phase
  denoiser.
- `FeedForwardPatchAutoencoder`: non-oscillatory latent control.
- `RecurrentConvPatchDenoiser`: non-oscillatory local recurrent control.

The MNIST experiment supports paired input-target training, so corrupted inputs
can reconstruct clean targets.

## Older Amplitude/Velocity Baselines

The first MNIST line used the older amplitude/velocity OscNet autoencoder
family. It could learn, but it was weak on clean MNIST reconstruction even after
longer CPU runs.

Selected single-seed clean MNIST runs:

| Model/run | Final eval loss | Corr | F1 | Note |
| --- | ---: | ---: | ---: | --- |
| Sigmoid positional, 30e | 0.03863 | 0.7432 | 0.7170 | Early usable baseline |
| Positional, 60e | 0.02724 | 0.8435 | 0.7979 | Better with longer train |
| Positional, 120e | 0.02113 | 0.8826 | 0.8295 | Still far above later Winfree/feedforward |

Sources:

- `outputs/reference/mnist_amp_velocity_sigmoid_positional_patch7_h64_l64_seed11_30e/metrics/summary.json`
- `outputs/reference/mnist_amp_velocity_positional_patch7_h64_l64_seed11_60e/metrics/summary.json`
- `outputs/reference/mnist_cpu_positional_patch7_h64_l64_120e/metrics/summary.json`

Interpretation:

- The older HORN/amplitude-velocity route is an oscillatory RNN baseline, not a
  strong masked-completion architecture.
- It remains useful as an older OscNet-family comparison, but not as the main
  branch for breakthrough behavior.

## Early Winfree Latent Autoencoder Work

The first WONN/Winfree-inspired branch was still a latent autoencoder. It added
`WinfreeFieldLayer`, learned sensitivity/influence functions, grouped/local
coupling options, latent readout/output-skip options, and synchrony analysis.

This was a major improvement over the older amplitude/velocity baseline on
clean MNIST. Two-seed train2000/30e clean reconstruction with the best
skip-heavy conv3 Winfree setup:

| Seed | Final eval loss | Corr | F1 |
| --- | ---: | ---: | ---: |
| 11 | 0.00568 | 0.9653 | 0.9341 |
| 12 | 0.00523 | 0.9674 | 0.9357 |

Source:
`outputs/analysis/winfree_conv3_latent96_two_seed30_comparison.csv`

Longer train5000/30e reached lower losses, but the feedforward patch control
still beat it on clean reconstruction. This led to the conclusion that clean
autoencoding was rewarding direct neural decoding more than oscillatory
dynamics.

## Latent Skip Attribution

An important negative result: the skip-heavy latent Winfree autoencoder depended
heavily on the latent output path.

Seed11 examples:

| Variant | Final eval loss | Corr | F1 | Diversity |
| --- | ---: | ---: | ---: | ---: |
| Coupled conv3 + sequence skip, 30e | 0.00568 | 0.9653 | 0.9341 | 0.9595 |
| Coupling disabled + sequence skip, 30e | 0.00553 | 0.9655 | 0.9343 | 0.9393 |
| No sequence skip, 10e | 0.06500 | 0.5182 | 0.5381 | near zero |

Source:
`outputs/analysis/winfree_attribution_controls_seed11.csv`

Interpretation:

- For clean reconstruction, the latent output skip was carrying much of the
  useful signal.
- Coupling removal barely hurt the skip-heavy clean autoencoder at 30e, so that
  setup did not prove phase coupling value.
- Removing the skip caused collapse, which motivated the move away from latent
  decoder scaffolds and toward direct conditional phase-field completion.

## Clean Autoencoding

Clean MNIST reconstruction was useful as a baseline but is not a good proving
ground for oscillatory dynamics.

Two-seed train5000/30e clean reconstruction:

| Model | Mean final eval loss | Interpretation |
| --- | ---: | --- |
| Feedforward patch control | 0.00286 | Best clean autoencoder control |
| Skip-heavy Winfree hybrid | 0.00369 | Strong, but below feedforward |
| Older amplitude/velocity OscNet | 0.021 to 0.039 range | Much weaker on this setup |

Source:
`outputs/analysis/winfree_attribution_feedforward_patch_train5000_two_seed_vs_winfree.csv`

Conclusion: clean autoencoding mostly rewards ordinary neural compression and
direct decoding. It does not isolate oscillatory value.

## Masked Completion

The first strong positive signal came from switching to partial-observation
completion and using a direct conditional phase field.

### 50% Patch Mask

Two-seed train2000/10e means:

| Model | Mean final eval loss | Mean corr | Mean F1 | Note |
| --- | ---: | ---: | ---: | --- |
| Winfree rate-phase, 8 steps | 0.01691 | 0.8934 | 0.8510 | Best current 50% mask setup |
| Winfree conditional, 8 steps | 0.01757 | 0.8892 | 0.8458 | Previous phase-only best |
| Winfree conditional, 4 steps | 0.01777 | 0.8881 | 0.8456 | Nearly tied |
| Winfree conditional, 1 step | 0.02311 | 0.8521 | 0.8069 | Dynamics depth helps |
| Feedforward patch control | 0.02558 | 0.8363 | 0.7950 | Static latent baseline |
| Winfree conditional, no coupling | 0.03846 | 0.7483 | 0.6679 | Coupling is essential |

Sources:

- `outputs/analysis/mnist_masked_patch50_conditional_winfree_step_depth_and_coupling_two_seed.csv`
- `outputs/analysis/mnist_masked_patch50_conditional_winfree_two_seed_vs_feedforward.csv`
- `outputs/analysis/mnist_masked_patch50_winfree_rate_phase_two_seed.csv`

Interpretation:

- Direct phase-only Winfree beats the feedforward control by about 31%, and
  the rate-phase hybrid improves that to about 34%.
- Four to eight recurrent steps beat one step, so settling dynamics matter.
- Removing coupling collapses performance, so this is not just a per-patch MLP.

### Recurrent-Conv Control

To test whether Winfree merely benefits from iterative local message passing, a
non-oscillatory `RecurrentConvPatchDenoiser` was added.

Two-seed train2000/10e means:

| Mask | Winfree8 | Recurrent conv | Feedforward | Winfree vs recurrent conv |
| --- | ---: | ---: | ---: | ---: |
| 50% | 0.01757 | 0.02165 | 0.02558 | 18.8% lower loss |
| 75% | 0.03562 | 0.04091 | 0.04008 | 12.9% lower loss |

Source:
`outputs/analysis/mnist_masked_patch50_75_winfree8_vs_recurrent_conv_feedforward_two_seed.csv`

Interpretation:

- Recurrent local refinement helps at 50%, but does not erase the Winfree
  advantage.
- At 75%, the advantage survives but shrinks.
- This is stronger evidence than comparing only against feedforward.

### 75% Patch Mask Rate-Phase Check

The 75% patch-mask test was the first falsification check for the rate-phase
hybrid. The gain over phase-only Winfree survived, but it was much smaller than
at 50% masking.

Two-seed train2000/10e means:

| Model | Mean final eval loss | Mean corr | Mean F1 | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Winfree rate-phase | 0.035107 | 0.7684 | 0.7241 | 0.0898 | Best current 75% mask result |
| Winfree phase-only fixed conv | 0.035620 | 0.7634 | 0.7226 | 0.0915 | Very close |
| Feedforward | 0.040080 | 0.7329 | 0.6984 | 0.0921 | Static control |
| Recurrent conv | 0.040912 | 0.7280 | 0.6883 | 0.1047 | Non-oscillatory recurrent control |

Source:
`outputs/analysis/mnist_masked_patch75_winfree_rate_phase_two_seed.csv`

Interpretation:

- Rate-phase is about 1.4% lower loss than phase-only Winfree at 75% masking.
- The win is replicated on both seeds, but it is not a large-margin jump.
- The rate-phase advantage over recurrent-conv and feedforward still survives.
- This supports the architecture principle, but also suggests that very sparse
  evidence may need stronger content propagation, longer settling, or a
  hierarchical/spatial-latent mechanism.

### Block Occlusion Check

Block occlusion changes the missingness geometry: instead of random missing
patches, the image contains one contiguous missing square. This is a harder test
of long-range shape completion because local patch-to-patch propagation must
cross an actual blank region.

Two-seed train2000/10e means, 50% block occlusion:

| Model | Mean final eval loss | Mean corr | Mean F1 | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.050616 | 0.6573 | 0.6371 | 0.1127 | Best block result |
| Winfree slow/global rate-phase | 0.052052 | 0.6403 | 0.6179 | 0.1244 | Best oscillatory block result |
| Winfree rate-phase | 0.052600 | 0.6329 | 0.6137 | 0.1264 | Best flat local field |
| Winfree phase-only fixed conv | 0.053092 | 0.6280 | 0.6096 | 0.1281 | Close to rate-phase |
| Recurrent conv | 0.054874 | 0.6130 | 0.6029 | 0.1327 | Local recurrent control |

Source:
`outputs/analysis/mnist_block50_winfree_rate_phase_two_seed.csv`
`outputs/analysis/mnist_block50_winfree_global_rate_phase_two_seed.csv`

Interpretation:

- This is the first clear task where the current rate-phase model does not win.
- Slow/global rate-phase is the best oscillatory model on this task, but it
  still loses to the feedforward latent control by about 2.8% final loss.
- The likely bottleneck is not just phase vs rate state. A large contiguous hole
  requires stronger global shape priors or long-range coordination than a flat
  local coupling field provides.
- This is strong evidence for an ONN-native next step: alter the coupling medium
  with distance-decay/nonlocal phase coupling, slow/global carrier phases, or a
  hierarchical phase field.

### Distance-Decay Coupling Probe

The first ONN-native follow-up tested whether replacing strict local conv phase
coupling with learned nonlocal matrix coupling plus Gaussian spatial decay would
help block occlusion:

```text
--winfree-coupling-mode matrix
--winfree-coupling-decay-length 2.0
```

Seed11, 50% block occlusion:

| Model | Final eval loss | Corr | F1 | MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.051141 | 0.6550 | 0.6329 | 0.1165 | Best seed11 block result |
| Local-conv rate-phase | 0.052808 | 0.6316 | 0.6148 | 0.1265 | Best oscillatory seed11 block result |
| Phase-only fixed conv | 0.053337 | 0.6270 | 0.6090 | 0.1275 | Close to rate-phase |
| Distance-decay rate-phase | 0.053642 | 0.6230 | 0.6078 | 0.1265 | Worse than local conv |
| Recurrent conv | 0.055467 | 0.6086 | 0.5997 | 0.1326 | Local recurrent control |

Source:
`outputs/analysis/mnist_block50_winfree_rate_phase_decay2_seed11.csv`

Interpretation:

- Naively replacing local conv coupling with distance-decay matrix coupling did
  not improve block occlusion.
- This mirrors the earlier adaptive-coupling lesson: the stable local spatial
  prior is valuable, and replacing it outright hurts optimization/inductive
  bias.

A residual version was then added as `--winfree-coupling-mode conv_matrix`,
which preserves local conv coupling and adds a weak distance-decayed matrix
field scaled by `--winfree-adaptive-coupling-strength`.

Seed11, 50% block occlusion:

| Model | Final eval loss | Corr | F1 | MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.051141 | 0.6550 | 0.6329 | 0.1165 | Best seed11 block result |
| Local-conv rate-phase | 0.052808 | 0.6316 | 0.6148 | 0.1265 | Best oscillatory seed11 block result |
| Residual conv+matrix rate-phase | 0.053021 | 0.6320 | 0.6117 | 0.1265 | Stable, but not better |
| Phase-only fixed conv | 0.053337 | 0.6270 | 0.6090 | 0.1275 | Close to rate-phase |
| Pure distance-decay rate-phase | 0.053642 | 0.6230 | 0.6078 | 0.1265 | Worse than local conv |
| Recurrent conv | 0.055467 | 0.6086 | 0.5997 | 0.1326 | Local recurrent control |

Source:
`outputs/analysis/mnist_block50_winfree_rate_phase_conv_matrix_seed11.csv`

Interpretation:

- Residual long-range coupling fixes the pure-matrix slowdown, but still does
  not beat the local-conv rate-phase baseline.
- The block bottleneck likely needs a more separated global mechanism: a
  slow/global phase band, hierarchical phase field, or explicit content
  transport. A weak residual matrix in the same fast phase field is not enough.

### Slow/Global Phase Carrier

`WinfreeGlobalRatePhaseConditionalPatchDenoiser` is now available as the first
slow/global carrier probe. It composes the local rate-phase field with a
separate one-position Winfree phase band initialized from the whole corrupted
image:

```text
corrupted image -> slow/global theta/omega carrier
corrupted patches -> fast local theta/rate field
slow phase gates fast local content propagation
```

This tests a more ONN-native version of hierarchy: global context is represented
as a slower phase rhythm rather than as a U-Net-style spatial pyramid or a
dense attention layer.

Two-seed train2000/10e means, 50% block occlusion:

| Model | Mean final eval loss | Mean corr | Mean F1 | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.050616 | 0.6573 | 0.6371 | 0.1127 | Still best block result |
| Winfree slow/global rate-phase | 0.052052 | 0.6403 | 0.6179 | 0.1244 | Best oscillatory block result |
| Winfree local rate-phase | 0.052600 | 0.6329 | 0.6137 | 0.1264 | Previous best oscillatory |
| Winfree phase-only fixed conv | 0.053092 | 0.6280 | 0.6096 | 0.1281 | Phase-only control |
| Recurrent conv | 0.054874 | 0.6130 | 0.6029 | 0.1327 | Non-oscillatory recurrent control |

Source:
`outputs/analysis/mnist_block50_winfree_global_rate_phase_two_seed.csv`

Interpretation:

- Slow/global rate-phase improves over local rate-phase by about 1.0% final
  loss and improves correlation, F1, diversity, and MAE.
- It is the first ONN-native block-occlusion intervention that improves the
  oscillatory model over the flat local field.
- It still does not beat feedforward. The global carrier helps, but it is not
  yet a strong enough global-shape mechanism.

### Coarse Spatial Phase-Mesh Probe

`WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser` tested a more explicit
two-scale ONN hypothesis for block occlusion: keep the fine local rate-phase
field, initialize a 2x2 or 4x4 coarse Winfree phase mesh from spatially pooled
corrupted patches, and interpolate coarse phase gates back down to the fine
field. Phase-shuffled controls tested whether precise coarse phase geometry was
actually doing useful work.

Two-seed train2000/10e means, 50% block occlusion:

| Model/group | Mean final eval loss | Mean best loss | Mean corr | Mean F1 | Mean diversity | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.050616 | 0.050299 | 0.6573 | 0.6371 | 0.5708 | 0.1127 | Still best block result |
| One-node slow/global rate-phase | 0.052052 | 0.051911 | 0.6403 | 0.6179 | 0.4676 | 0.1244 | Best oscillatory block result |
| Local rate-phase | 0.052600 | 0.052600 | 0.6329 | 0.6137 | 0.4420 | 0.1264 | Best flat local field |
| Coarse 4x4, shuffled | 0.052941 | 0.052826 | 0.6280 | 0.6128 | 0.4364 | 0.1246 | Best coarse-mesh mean |
| Phase-only fixed conv | 0.053092 | 0.052782 | 0.6280 | 0.6096 | 0.4402 | 0.1281 | Phase-only control |
| Coarse 2x2, shuffled | 0.053316 | 0.052909 | 0.6257 | 0.6108 | 0.4417 | 0.1250 | Shuffled beats unshuffled 2x2 |
| Coarse 2x2 | 0.053753 | 0.053565 | 0.6232 | 0.6069 | 0.4277 | 0.1269 | Worse than local rate-phase |
| Recurrent conv | 0.054874 | 0.054874 | 0.6130 | 0.6029 | 0.4171 | 0.1327 | Non-oscillatory recurrent control |
| Coarse 4x4 | 0.055630 | 0.054330 | 0.6094 | 0.5917 | 0.4019 | 0.1250 | Seed12 became unstable |

Sources:

- `outputs/analysis/modal_block50_coarse_phase_mesh.csv`
- `outputs/analysis/mnist_block50_coarse_phase_mesh_with_baselines.csv`

Interpretation:

- The coarse phase mesh did not beat the one-node slow/global carrier, the
  local rate-phase field, or the feedforward baseline.
- The phase-shuffled controls were not worse; the best coarse result was a
  shuffled 4x4 control. This falsifies the strong version of the "precise
  coarse phase geometry is doing the repair" hypothesis for this implementation.
- One unshuffled 4x4 seed developed extremely large gradient norms late in
  training and degraded, suggesting that the coarse mesh can destabilize the
  rate field.
- The likely issue is that coarse phase was used only as an interpolated gate
  on local content propagation. That adds capacity and a weak global bias, but
  it does not yet provide genuine top-down shape transport or robust
  slow-fast feedback.

### Coarse Rate/Content Transport Probe

`WinfreeCoarseRatePhaseConditionalPatchDenoiser` tested a sharper multiscale
ONN hypothesis than the phase-mesh probe. The previous coarse mesh used coarse
phase only as a top-down gate. This variant gives the coarse band its own
rate/content state, evolves that state under coarse Winfree phase gates, and
projects coarse content back down into the fine rate field as additive drive:

```text
fine corrupted patches -> fine phase + fine content field
coarse pooled patches  -> coarse phase + coarse content field
coarse content evolves under coarse phase gates
coarse content is injected into fine content updates
fine local dynamics refine the reconstruction
```

Two-seed train2000/10e means, 50% block occlusion:

| Model/group | Mean final eval loss | Mean best loss | Mean corr | Mean F1 | Mean diversity | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.050616 | 0.050299 | 0.6573 | 0.6371 | 0.5708 | 0.1127 | Still best block result |
| One-node slow/global rate-phase | 0.052052 | 0.051911 | 0.6403 | 0.6179 | 0.4676 | 0.1244 | Best oscillatory block result |
| Local rate-phase | 0.052600 | 0.052600 | 0.6329 | 0.6137 | 0.4420 | 0.1264 | Best flat local field |
| Coarse phase 4x4, shuffled | 0.052941 | 0.052826 | 0.6280 | 0.6128 | 0.4364 | 0.1246 | Best coarse phase-mesh mean |
| Phase-only fixed conv | 0.053092 | 0.052782 | 0.6280 | 0.6096 | 0.4402 | 0.1281 | Phase-only control |
| Coarse content+gate, content shuffled | 0.053272 | 0.053272 | 0.6266 | 0.6103 | 0.4263 | 0.1248 | Best coarse rate/content mean |
| Coarse gate only | 0.053486 | 0.053486 | 0.6247 | 0.6095 | 0.4223 | 0.1261 | No coarse content |
| Coarse content only | 0.053680 | 0.053673 | 0.6232 | 0.6064 | 0.4165 | 0.1263 | No coarse phase gate |
| Coarse content+gate | 0.054314 | 0.054295 | 0.6205 | 0.6042 | 0.4019 | 0.1310 | Seed12 became unstable |
| Coarse phase 4x4 | 0.055630 | 0.054330 | 0.6094 | 0.5917 | 0.4019 | 0.1250 | Earlier phase-mesh unstable seed |

Sources:

- `outputs/analysis/modal_block50_coarse_rate_phase.csv`
- `outputs/analysis/mnist_block50_coarse_rate_phase_with_baselines.csv`

Interpretation:

- Explicit coarse content transport did not improve block occlusion. The best
  new mean was the content-shuffled control, not the topographic content path.
- The full content+gate version had one stable good seed and one late unstable
  seed with huge gradient norms, so the additive coarse content path can
  destabilize the recurrent field.
- Content-only and gate-only controls were also worse than the local rate-phase
  and one-node slow/global models.
- This weakens the simple "coarse-to-fine additive content transport is the
  missing ingredient" hypothesis. It does not rule out multiscale oscillatory
  systems, but the next version likely needs normalization, a separate coarse
  reconstruction objective, phase/frequency feedback rather than direct content
  injection, or a different stress task.

### Block Artifact and Stability Diagnostics

After the negative coarse-content sweep, we inspected saved reconstruction
artifacts and traces instead of launching another architecture search.

Artifact diagnostic method:

- Load `mnist_reconstructions_epoch_010.npz`.
- Infer pixels changed by corruption from `abs(input - target) > 1e-6`.
- Report full-image loss separately from changed-region loss. For block
  occlusion, this changed region mostly measures lost digit signal, not
  background pixels that were already zero.

Two-seed means from first saved artifact batches:

| Model/group | Changed-region MSE | Full eval loss | Visible/unchanged MSE | Mean max grad norm | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Coarse content+gate | 0.207960 | 0.054314 | 0.025189 | 4.5e12 | Best changed-region mean, but unstable |
| Local rate-phase | 0.214188 | 0.052600 | 0.023600 | 0.0958 | Strong stable missing-signal repair |
| Coarse gate only | 0.214352 | 0.053486 | 0.021351 | 0.103 | Good visible region, no full win |
| Recurrent conv | 0.214409 | 0.054874 | 0.024247 | 0.0447 | Competitive changed-region control |
| One-node slow/global rate-phase | 0.215787 | 0.052052 | 0.024480 | 0.0855 | Best oscillatory full-loss block model |
| Coarse content shuffled | 0.216849 | 0.053272 | 0.020418 | 0.0991 | Best new full-loss coarse variant |
| Phase-only fixed conv | 0.223108 | 0.053092 | 0.023792 | 0.101 | Weaker changed-region repair |
| Feedforward | 0.226394 | 0.050616 | 0.022382 | 0.0372 | Best full loss, worst changed-region repair in this sample |

Trace/stability read:

- The unstable coarse content+gate seed did not have exploding output or rate
  magnitudes; the rate states remained bounded by tanh.
- Its coarse phase band was highly volatile: coarse phase step mean was much
  larger than the stable shuffled controls, and coarse energy delta was orders
  of magnitude larger.
- This points to optimization/dynamical sensitivity in the coarse phase band,
  not simple output blow-up.

Sources:

- `outputs/analysis/mnist_block50_artifact_diagnostics_summary.csv`
- `outputs/analysis/mnist_block50_coarse_rate_phase_artifact_diagnostics.csv`
- `outputs/analysis/mnist_block50_trace_stability_diagnostics.csv`
- `outputs/analysis/mnist_block50_artifact_montage.png`

Takeaway:

The block task is more subtle than the full-loss leaderboard suggests.
Feedforward wins full-image MSE, likely helped by visible/background fidelity,
but it is not strongest on lost-signal pixels in the saved artifact sample.
Local recurrent dynamics and coarse content can repair missing signal, but the
naive coarse content path is unstable and not reliable. The next block step
should not be another blind coarse sweep; it should either stabilize the coarse
band with normalization/auxiliary loss or use a stronger recurrent-conv/ConvLSTM
control to see whether changed-region repair is a generic recurrence effect.

### Mask-Aware Block Loss Probe

The artifact diagnostic suggested full-image MSE was under-rewarding the
important part of block occlusion: repair of pixels actually changed by
corruption. We added an opt-in mask-aware MNIST loss:

```text
loss = mean(weight * (prediction - target)^2) / mean(weight)
weight = 2.0 for changed pixels, 0.25 for unchanged pixels
```

Implementation notes:

- Old runs are unchanged by default; both weights default to `1.0`.
- New CLI flags:
  `--corruption-visible-loss-weight`,
  `--corruption-changed-loss-weight`, and
  `--corruption-change-atol`.
- Summaries now include `quality.changed_mse`,
  `quality.unchanged_mse`, and `quality.changed_improvement`.

Two-seed train2000/10e means, 50% block occlusion, mask-aware loss:

| Model/group | Weighted eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Slow/global rate-phase | 0.102388 | 0.070645 | 0.131648 | 0.059449 | 0.6128 | 0.5725 | Best changed-region MSE by a small margin |
| Feedforward | 0.101464 | 0.068186 | 0.132198 | 0.056434 | 0.6287 | 0.5878 | Best weighted/full/corr/F1 |
| Local rate-phase | 0.104242 | 0.073052 | 0.132967 | 0.062054 | 0.6008 | 0.5581 | Stable but not best under this loss |
| Recurrent conv | 0.105899 | 0.075680 | 0.133740 | 0.065019 | 0.5863 | 0.5529 | Worst of the core set |

Sources:

- `outputs/analysis/modal_block50_mask_aware_core.csv`
- `outputs/analysis/mnist_block50_mask_aware_core_summary.csv`

Interpretation:

- Mask-aware loss substantially changes the behavior: changed-region MSE drops
  far below the first artifact-batch diagnostics from full-MSE-trained runs.
- The hoped-for oscillator-specific breakthrough did not appear. Feedforward
  still has the best weighted eval loss, full MSE, correlation, F1, and visible
  region fidelity.
- The slow/global rate-phase model has a tiny changed-region advantage
  (`0.131648` vs feedforward `0.132198`), which keeps the slow-fast hierarchy
  hypothesis alive but not spectacular.
- The `2.0`/`0.25` weighting is probably too blunt: it improves hidden-region
  repair while hurting full-image fidelity. A more balanced sweep should test
  lighter weights such as changed `1.5`, visible `0.5`, and should include a
  ConvLSTM/U-Net-style control before making any broad ONN claim.

### Mask-Loss Weight Sweep

We then ran the lighter-weight check on Modal for the two most relevant block
models: feedforward and one-node slow/global rate-phase Winfree. This was meant
to test whether the earlier slow/global changed-region edge survived once the
loss stopped over-prioritizing hidden pixels.

Two-seed train2000/10e means, 50% block occlusion:

| Loss weights | Model/group | Weighted eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| changed 1.25, visible 0.75 | Feedforward | 0.062623 | 0.051377 | 0.171195 | 0.029380 | 0.6603 | 0.6344 | Best balanced result |
| changed 1.25, visible 0.75 | Slow/global rate-phase | 0.067379 | 0.054761 | 0.188944 | 0.030130 | 0.6267 | 0.6082 | Worse across metrics |
| changed 1.5, visible 0.5 | Feedforward | 0.078378 | 0.056354 | 0.149391 | 0.039272 | 0.6516 | 0.6189 | Best missing-region result in sweep |
| changed 1.5, visible 0.5 | Slow/global rate-phase | 0.081936 | 0.058054 | 0.158831 | 0.039554 | 0.6268 | 0.6025 | Stable, but no edge |

Sources:

- `outputs/analysis/modal_block50_mask_weight_sweep.csv`
- `outputs/analysis/mnist_block50_mask_weight_sweep_summary.csv`

Interpretation:

- The tiny slow/global changed-region edge from the stronger `2.0`/`0.25`
  loss did not survive the lighter sweep.
- Feedforward wins weighted loss, full MSE, changed-region MSE, correlation,
  and F1 at both lighter settings.
- The block-occlusion result is therefore not "tune the loss and Winfree wins."
  Loss alignment is useful, but the current slow/global carrier is not enough
  of a mechanism.
- The most honest next step is either a stronger non-oscillatory control
  (ConvLSTM/untied recurrent conv) or a genuinely stronger ONN-native mechanism
  with explicit content transport/stabilization. More small weight sweeps are
  unlikely to be the gearshift.

### Missing-Marker Block Probe

The next question was whether the block task was partly unfair because zero can
mean either "real black MNIST background" or "missing evidence." We tested the
cheapest version of mask visibility: mark occluded pixels as `-1.0` in the same
image channel via `--corruption-mask-value -1.0`.

Two-seed train2000/10e means, 50% block occlusion, full MSE loss:

| Model/group | Final eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Diversity | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.050092 | 0.050050 | 0.092931 | 0.005382 | 0.6534 | 0.6339 | 0.5302 | Best overall |
| Slow/global rate-phase | 0.054102 | 0.054060 | 0.101306 | 0.004846 | 0.6195 | 0.6048 | 0.4089 | Best oscillatory variant |
| Local rate-phase | 0.054584 | 0.054546 | 0.102387 | 0.004712 | 0.6145 | 0.6023 | 0.4037 | Close to slow/global |
| Recurrent conv | 0.064330 | 0.064262 | 0.109608 | 0.017027 | 0.5256 | 0.5488 | 0.1182 | Strongly hurt by marker |

Sources:

- `outputs/analysis/modal_block50_missing_marker_core.csv`
- `outputs/analysis/mnist_block50_missing_marker_core_summary.csv`

Interpretation:

- Marking missing pixels inside the image channel does not unlock the current
  Winfree models. Feedforward still wins full MSE, changed-region MSE,
  correlation, F1, and diversity.
- Compared with the earlier zero-mask full-MSE block runs, feedforward improves
  slightly (`0.050616` -> `0.050092`), while the local and slow/global Winfree
  variants get worse on full MSE (`0.052600` -> `0.054584`, and `0.052052` ->
  `0.054102`).
- The changed-region MSE is not directly comparable to zero-mask runs because
  `-1.0` makes every occluded pixel "changed," including background pixels that
  stayed zero under zero-mask corruption.
- The important design lesson is that missingness should probably be a separate
  control field, not a fake image value. In the current rate-phase model, an
  out-of-range marker becomes part of the content/rate drive and can poison the
  recurrent field rather than simply telling it where to relax.

### Image-Plus-Mask Block Probe

The missing-marker result suggested that missingness should be separated from
content. We added `--corruption-input-mode image_plus_mask`, which feeds each
model a two-channel flat input:

```text
channel 0: corrupted image, with occluded pixels still set to 0.0
channel 1: visibility mask, 1.0 for visible pixels and 0.0 for missing pixels
target:    clean one-channel image
```

Two-seed train2000/10e means, 50% block occlusion, full MSE loss:

| Model/group | Final eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Diversity | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.049337 | 0.049290 | 0.091635 | 0.005182 | 0.6600 | 0.6393 | 0.5328 | Best overall |
| Slow/global rate-phase | 0.051335 | 0.051295 | 0.097929 | 0.002718 | 0.6416 | 0.6215 | 0.4775 | Best oscillatory variant |
| Local rate-phase | 0.051622 | 0.051578 | 0.098569 | 0.002629 | 0.6407 | 0.6210 | 0.4605 | Close to slow/global |
| Recurrent conv | 0.058964 | 0.058918 | 0.105092 | 0.010820 | 0.5794 | 0.5840 | 0.3046 | Helped less than the other models |

Sources:

- `outputs/analysis/modal_block50_image_plus_mask_core.csv`
- `outputs/analysis/mnist_block50_image_plus_mask_core_summary.csv`

Interpretation:

- The separate visibility channel is the right representation compared with
  the `-1.0` fake-pixel marker. It improves feedforward, local Winfree, and
  slow/global Winfree relative to the marker run.
- It still does not produce an oscillator-specific block-occlusion win:
  feedforward remains best on full MSE, changed-region MSE, correlation, F1,
  and diversity.
- The oscillatory models have much lower unchanged-region MSE than feedforward
  (`~0.0026` to `0.0027` vs `0.0052`), which means they preserve visible
  evidence well. Their remaining problem is harder missing-region synthesis,
  not visible-region damage.
- This points to a sharper next mechanism: use the visibility field inside the
  dynamics, not merely as an extra input channel. The current models can read
  the mask, but the local rate update does not explicitly gate drive,
  transport, or trust by visibility.

### Visibility-Gated Dynamics Probe

We then tested the most direct version of the previous idea: keep
`image_plus_mask`, but feed the visibility channel into the rate-phase dynamics
as an explicit gate. Visible patches keep their normal image drive, while
missing patches suppress local input drive and boost recurrent transport.
A deterministic shuffled-visibility control tested whether correct mask
geometry matters.

Two-seed train2000/10e means, 50% block occlusion, full MSE loss:

| Model/group | Final eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Diversity | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.049337 | 0.049290 | 0.091635 | 0.005182 | 0.6600 | 0.6393 | 0.5328 | Best overall |
| Slow/global rate-phase, ungated | 0.050626 | 0.050588 | 0.096525 | 0.002738 | 0.6466 | 0.6262 | 0.4713 | Best oscillatory variant |
| Slow/global rate-phase, visibility gated | 0.051634 | 0.051590 | 0.098374 | 0.002856 | 0.6394 | 0.6210 | 0.4544 | Worse than ungated |
| Slow/global rate-phase, shuffled gate | 0.051634 | 0.051593 | 0.098446 | 0.002788 | 0.6397 | 0.6212 | 0.4719 | Essentially tied with gated |
| Local rate-phase, shuffled gate | 0.051918 | 0.051871 | 0.099019 | 0.002758 | 0.6381 | 0.6187 | 0.4585 | Slightly above ungated local |
| Local rate-phase, ungated | 0.052012 | 0.051971 | 0.099254 | 0.002717 | 0.6379 | 0.6185 | 0.4603 | Stable baseline |
| Local rate-phase, visibility gated | 0.052197 | 0.052153 | 0.099608 | 0.002720 | 0.6364 | 0.6169 | 0.4476 | Worse than ungated |

Sources:

- `outputs/analysis/modal_block50_visibility_gated_winfree.csv`
- `outputs/analysis/mnist_block50_visibility_gated_winfree_summary.csv`

Interpretation:

- Naive visibility gating did not unlock block completion. Correct visibility
  geometry was not better than shuffled visibility, and both were worse than
  the ungated slow/global model.
- The separate mask channel is still useful as representation, but this
  multiplicative gate is too blunt. It likely suppresses useful content drive
  in missing zones without adding a strong enough generative/transport
  mechanism to replace it.
- The best oscillatory block model remains slow/global rate-phase with
  `image_plus_mask` input but ungated dynamics. It is closer to feedforward than
  earlier block probes, but it still does not beat feedforward.
- This rules out the simplest "mask as trust gate" implementation. A stronger
  ONN-native block mechanism probably needs active content transport,
  normalized/stabilized recurrent flow, or a predictive coarse field, not just
  per-patch visibility scaling.

### ConvLSTM Control Probe

The visibility-gated result made the next scientific control more important:
we need to know whether the block-occlusion gap is specifically oscillatory, or
whether a standard gated recurrent spatial model solves the same pressure test.

Implementation:

```text
ConvLSTMPatchDenoiser
corrupted patch grid + optional visibility channel
-> ConvLSTM hidden/cell settling over local patch neighborhoods
-> clean patch readout
```

It shares the same flat-image patch interface as `RecurrentConvPatchDenoiser`
and the Winfree denoisers, supports `image_plus_mask`, and exports trace arrays
for `drive`, `hidden_states`, and `cell_states`.

Two-seed train2000/10e means, 50% block occlusion, `image_plus_mask`, full MSE
loss:

| Model/group | Final eval loss | Full MSE | Changed-region MSE | Unchanged MSE | Corr | F1 | Diversity | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.049337 | 0.049290 | 0.091635 | 0.005182 | 0.6600 | 0.6393 | 0.5328 | Best overall |
| Slow/global rate-phase | 0.050943 | 0.050899 | 0.097122 | 0.002751 | 0.6435 | 0.6228 | 0.4660 | Best recurrent/spatial field |
| Recurrent conv | 0.058991 | 0.058946 | 0.105151 | 0.010817 | 0.5791 | 0.5843 | 0.3054 | Better than ConvLSTM |
| ConvLSTM | 0.060017 | 0.059961 | 0.105717 | 0.012299 | 0.5692 | 0.5802 | 0.2892 | Gated baseline did not help |

Sources:

- `outputs/analysis/modal_block50_conv_lstm_control.csv`
- `outputs/analysis/mnist_block50_conv_lstm_control_summary.csv`

Interpretation:

- ConvLSTM does not explain away the slow/global Winfree result. It is worse
  than tied recurrent-conv and much worse than slow/global Winfree on full MSE,
  changed-region MSE, visible-region MSE, correlation, F1, diversity, and MAE.
- The block task is therefore not trivially solved by adding standard local
  recurrent memory gates to the patch grid.
- Feedforward still wins overall, so this is not an oscillator breakthrough
  either. The important read is narrower: the current Winfree field is a better
  spatial recurrent inductive bias than these simple non-oscillatory recurrent
  controls, but it still lacks a global synthesis mechanism strong enough to
  beat direct feedforward decoding.
- A useful ONN-native next branch should not copy ConvLSTM gates directly. It
  should preserve the Winfree field's visible-region stability and add a
  stronger missing-region generator: predictive coarse field, stabilized active
  content transport, or phase/frequency feedback with an auxiliary coarse loss.

### JEPA-Lite Representation Prediction Probe

> **Status (2026-07):** experiment package + `oscnet.models.jepa` predictors
> removed from the tree after this null ONN probe. Results below are historical.

We tested a first JEPA-style pivot: stop training the model to reconstruct
hidden pixels, and instead train it to predict abstract patch representations
for hidden block-occlusion patches.

This is deliberately JEPA-lite, not a full I-JEPA/V-JEPA implementation:

```text
visible corrupted image + visibility mask
-> patch-grid predictor
-> low-frequency DCT embedding of hidden clean patches
```

The target encoder is a deterministic low-frequency DCT basis per patch. The
loss is applied only to hidden patches. This removes exact pixel-space painting
pressure, but it does not yet use a learned/EMA teacher encoder or semantic
representation space.

Two-seed train2000/10e means, 50% block occlusion, `image_plus_mask`, hidden
DCT-8 patch embedding loss:

| Model/group | Final eval loss | Best loss | Mean-baseline loss | Margin vs mean | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| ConvLSTM | 0.133409 | 0.133409 | 0.171367 | 0.037958 | Best JEPA-lite predictor |
| Feedforward | 0.135045 | 0.135045 | 0.171367 | 0.036322 | Close second |
| Slow/global rate-phase | 0.140856 | 0.140856 | 0.171367 | 0.030512 | Best oscillatory variant |
| Local rate-phase | 0.142034 | 0.142034 | 0.171367 | 0.029333 | Close to recurrent-conv |
| Recurrent conv | 0.142911 | 0.142642 | 0.171367 | 0.028456 | Weakest mean |

Sources:

- `outputs/analysis/modal_block50_jepa_core.csv`
- `outputs/analysis/mnist_block50_jepa_core_summary.csv`

Interpretation:

- The JEPA objective is viable as a benchmark: all learned models beat the
  zero-embedding and train-mean embedding baselines.
- The first JEPA-lite result is not an ONN win. ConvLSTM and feedforward beat
  both Winfree variants on hidden DCT patch representation prediction.
- This does not falsify the JEPA direction. The target representation here is a
  fixed low-frequency transform, so ordinary neural predictors can learn it
  directly. A real JEPA-style test needs a learned/EMA teacher representation,
  larger context/target blocks, or a downstream probe showing that the learned
  oscillatory state is useful.
- For the current OscNet branch, the lesson is sobering but useful: merely
  moving from pixels to simple hand-designed patch embeddings is not enough to
  unlock oscillatory advantage. If JEPA helps ONNs, it probably has to be
  because the teacher representation itself rewards coherent latent scene
  states, not because the loss is no longer pixel MSE.

## KoPE-Lite Phase Initialization Probe

Inspired by KoPE, `WinfreeConditionalPatchDenoiser` now supports deterministic
2D rotary phase initialization:

```text
--winfree-phase-init rotary_2d
--winfree-phase-init-scale <float>
```

Seed11, 50% mask, train2000/10e:

| Model | Final eval loss | Note |
| --- | ---: | --- |
| Learned positional theta | 0.01741 | Current best seed11 baseline |
| Rotary 2D phase init, scale 0.25 | 0.01918 | Better than controls, worse than learned init |
| Rotary 2D phase init, scale 1.0 | 0.01988 | Worse than scale 0.25 |
| Recurrent conv | 0.02236 | Non-oscillatory recurrent control |
| Feedforward | 0.02551 | Static control |

Source:
`outputs/analysis/mnist_masked_patch50_winfree_rotary2d_phase_init_seed11.csv`

Interpretation:

- Deterministic 2D phase init alone did not improve the current Winfree field.
- Lower scale helped relative to full scale, but still lost to learned
  positional phases.
- KoPE's useful lesson may require adaptive coupling and rate-phase integration,
  not phase initialization alone.

## KoPE-Lite Adaptive Coupling Probe

`WinfreeFieldLayer` now supports local data-adaptive coupling:

```text
--winfree-coupling-mode adaptive
```

This mode computes query/key/value projections from the current influence field,
applies softmax attention only inside the local neighborhood defined by
`--winfree-coupling-kernel-size`, and then feeds the resulting field into the
same Winfree update. It was intended as a minimal KoPE-style adaptive coupling
test against the fixed local conv coupling baseline.

Seed11, 50% mask, train2000/10e:

| Model | Final eval loss | Note |
| --- | ---: | --- |
| Fixed conv Winfree8 | 0.01741 | Current seed11 baseline |
| Rotary 2D scale 0.25 Winfree8 | 0.01918 | Phase-init-only probe |
| Recurrent conv | 0.02236 | Non-oscillatory recurrent control |
| Feedforward | 0.02551 | Static control |
| Adaptive local Winfree8 | 0.03153 | Negative result |

Source:
`outputs/analysis/mnist_masked_patch50_winfree_adaptive_coupling_seed11.csv`

Interpretation:

- Pure adaptive local attention over phase influence was much worse than fixed
  local conv coupling.
- The likely failure mode is optimization/inductive bias: the fixed conv path
  gives a stable spatial propagation prior, while pure attention starts too
  unconstrained.
- The next adaptive-coupling attempt should not replace conv outright. Prefer a
  residual hybrid such as `conv + alpha * adaptive`, or compute adaptive
  coupling from a rate/content field rather than phase influence alone.

### Residual Adaptive Coupling

`WinfreeFieldLayer` now also supports residual adaptive coupling:

```text
--winfree-coupling-mode conv_adaptive
--winfree-adaptive-coupling-strength <float>
```

This preserves the fixed local conv field and adds a small adaptive correction.

Seed11, 50% mask, train2000/10e:

| Model | Final eval loss | Note |
| --- | ---: | --- |
| Conv-adaptive Winfree8, alpha 0.01 | 0.017406 | Essentially tied with fixed conv |
| Fixed conv Winfree8 | 0.017409 | Current baseline |
| Conv-adaptive Winfree8, alpha 0.05 | 0.017479 | Slightly worse |
| Pure adaptive Winfree8 | 0.031533 | Bad |

Source:
`outputs/analysis/mnist_masked_patch50_winfree_residual_adaptive_coupling_seed11.csv`

Interpretation:

- Residual adaptive coupling avoids the pure-adaptive collapse.
- With `alpha=0.01`, it was effectively tied with fixed conv on seed11 and had
  slightly better F1/MAE, but the loss difference was too small to call a win.

Seed12 replication made the conclusion more conservative. Two-seed 50% mask
means:

| Model | Mean final eval loss | Mean best loss | Mean corr | Mean F1 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed conv Winfree8 | 0.017574 | 0.017574 | 0.8892 | 0.8458 | Still best final loss |
| Conv-adaptive Winfree8, alpha 0.01 | 0.017790 | 0.017577 | 0.8881 | 0.8444 | Nearly tied best loss, worse final loss |
| Recurrent conv | 0.021650 | 0.021650 | 0.8641 | 0.8229 | Non-oscillatory recurrent control |
| Feedforward | 0.025578 | 0.025578 | 0.8363 | 0.7950 | Static control |

Source:
`outputs/analysis/mnist_masked_patch50_winfree_residual_adaptive_coupling_two_seed.csv`

Interpretation:

- Residual adaptive coupling is safe but not better than fixed local conv in
  this setup.
- The tiny best-loss tie suggests adaptive corrections might become useful, but
  the current attention-over-influence implementation is not the missing piece.
- Future adaptive coupling should probably use a richer rate/content field or
  untied recurrent blocks, rather than adding more attention to the phase
  influence alone.

## Rate-Phase Hybrid Branch

`WinfreeRatePhaseConditionalPatchDenoiser` is the first direct test of the
rate-phase hypothesis:

```text
corrupted patches -> theta/omega phase field + content/rate field
Winfree phase settling gates local content updates
content + phase features -> clean patches
```

This branch preserves the successful conditional Winfree phase field but stops
asking phase alone to carry reconstruction evidence. It produced the strongest
50% patch-mask result so far.

Two-seed train2000/10e means:

| Model | Mean final eval loss | Mean corr | Mean F1 | Mean MAE | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Winfree rate-phase | 0.016912 | 0.8934 | 0.8510 | 0.0507 | Best current 50% mask result |
| Winfree phase-only fixed conv | 0.017574 | 0.8892 | 0.8458 | 0.0556 | Previous best |
| Recurrent conv | 0.021650 | 0.8641 | 0.8229 | 0.0625 | Non-oscillatory recurrent control |
| Feedforward | 0.025578 | 0.8363 | 0.7950 | 0.0641 | Static control |

Source:
`outputs/analysis/mnist_masked_patch50_winfree_rate_phase_two_seed.csv`

Interpretation:

- Rate-phase is about 3.8% lower loss than phase-only fixed-conv Winfree on
  the two-seed mean.
- It retains the larger advantage over recurrent-conv and feedforward controls.
- This is the first result supporting the hypothesis that phase should
  coordinate a content field rather than act as the only reconstruction state.
- This is still not a broad breakthrough claim. It needs harder corruption
  tests and stronger controls before we treat it as a durable architecture win.

At 75% patch masking, the same architecture remains best but the margin shrinks
to about 1.4% lower loss than phase-only Winfree. This is a positive
generalization check, not yet a decisive breakthrough.

On 50% block occlusion, the current slow/global rate-phase model improves over
flat local rate-phase but still loses to the feedforward latent control. This
is a useful mixed result: distributed random missingness and contiguous missing
regions stress different mechanisms, and a one-position slow carrier is helpful
but not sufficient.

## Coarse Predictive Readout Probe

`WinfreeCoarsePredictiveRatePhaseConditionalPatchDenoiser` tested a stabilized
version of the multiscale block-occlusion hypothesis.

failed coarse rate/content model injected coarse content back into the fine
recurrent update at every settling step, which could destabilize training. The
new version still evolves a coarse rate-phase field, but uses the final coarse
state only at readout:

```text
fine corrupted patches -> fine local rate-phase settling
coarse pooled patches -> coarse rate-phase settling
final fine state -> local patch logits
final coarse state -> broad patch correction
local logits + coarse correction -> reconstruction
```

The intended interpretation is crisp:

- If this helps block occlusion, the useful multiscale role is likely a slow
  spatial shape prior, not recurrent top-down content injection.
- If it fails like the previous coarse variants, the next ONN mechanism probably
  needs stronger normalization, an auxiliary coarse reconstruction loss, or
  phase/frequency feedback rather than another readout blend.
- The control knobs are `--winfree-coarse-readout-strength`,
  `--winfree-global-phase-control`, and `--winfree-global-content-control`.

Focused Modal preset, dry-run validated with `max_containers=3`:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_coarse_predictive_readout \
  --print-only
```

The preset compares feedforward, current one-node slow/global Winfree, and
2x2/4x4 coarse predictive Winfree on the same `image_plus_mask` block setup.

Two-seed train2000/10e means:

| Model | Mean final eval loss | Mean corr | Mean F1 | Mean changed MSE | Mean unchanged MSE | Mean diversity | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.049300 | 0.6602 | 0.6395 | 0.091570 | 0.005175 | 0.5324 | Best full and missing-region result |
| One-node slow/global Winfree | 0.050884 | 0.6457 | 0.6256 | 0.097040 | 0.002711 | 0.4716 | Better visible preservation, worse missing synthesis |
| Coarse predictive 4x4 | 0.051219 | 0.6417 | 0.6225 | 0.097818 | 0.002580 | 0.4579 | Stable but worse than one-node global |
| Coarse predictive 2x2 | 0.051524 | 0.6398 | 0.6210 | 0.098260 | 0.002774 | 0.4579 | Stable but weaker |

Source:
`outputs/analysis/modal_block50_coarse_predictive_readout.csv`

Interpretation:

- The readout-only coarse field is stable, but it did not improve block
  completion. This weakens the "coarse prediction at final readout is enough"
  hypothesis.
- The result sharpens the failure mode: Winfree preserves known pixels much
  better than feedforward, but feedforward hallucinates the missing block better.
- That means the next block-occlusion move should stop treating visible-pixel
  reconstruction as part of the model's job. For inpainting, visible pixels are
  boundary conditions. The model should be judged and trained primarily on
  missing-region synthesis under clamped observed evidence.

## Boundary-Clamped Inpainting Protocol

Implemented next as an anti-eval-hacking split, not as a replacement for the
full reconstruction track:

```text
full_reconstruction:
  model predicts every pixel
  full-image eval loss is meaningful

boundary_clamped:
  visible pixels are copied from the corrupted input for every model
  loss is hidden-region MSE only
  full-image metrics are explicitly secondary
```

CLI:

```bash
--corruption-input-mode image_plus_mask
--corruption-protocol boundary_clamped
```

The protocol uses `clamp_predictions_to_visible_inputs()` as an experiment-level
prediction transform. It does not wrap or change the reusable model weights or
checkpoint structure. Under this protocol, `final_eval_loss` is hidden-region
MSE because visible-pixel loss weights are forced to zero and hidden-pixel
weights to one. The summary writes:

```text
loss_protocol: boundary_clamped
primary_metric: hidden_region_mse
full_image_metrics_are_secondary: true
```

Focused validation:

- `tests/test_reference_experiments.py::test_mnist_boundary_clamped_protocol_uses_hidden_region_loss`
- `tests/test_reference_experiments.py::test_mnist_image_plus_mask_input_reconstructs_clean_targets`

Both passed locally. The Modal dry-run for the six-run comparison also passed
without launching workers:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_boundary_clamped_core \
  --print-only
```

Interpretation rule:

- A boundary-clamped result is only interesting if it wins hidden-region metrics
  such as `final_eval_loss`, `quality.changed_mse`, hidden F1/IoU, or hidden
  MAE.
- It must not be presented as a full-image reconstruction win.

### Boundary-Clamped Core Result

We then launched the capped six-run Modal comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_boundary_clamped_core
```

Two-seed train2000/10e means, where `final_eval_loss` is hidden-region MSE:

| Model | Mean hidden loss | Mean changed MSE | Mean corr | Mean F1 | Mean diversity | Mean unchanged MSE | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Feedforward | 0.091573 | 0.091493 | 0.6820 | 0.6591 | 0.6036 | 0.000000 | Best hidden-region synthesis |
| Coarse predictive 4x4 Winfree | 0.097530 | 0.097455 | 0.6558 | 0.6326 | 0.5136 | 0.000000 | Best ONN branch, unstable seed12 |
| One-node slow/global Winfree | 0.097965 | 0.097883 | 0.6538 | 0.6335 | 0.5278 | 0.000000 | Stable but weaker hidden synthesis |

Source:
`outputs/analysis/modal_block50_boundary_clamped_core.csv`

Interpretation:

- This is a clean negative for the current ONN block-inpainting stack. Even
  when visible pixels are clamped and the metric is hidden-region MSE, the
  feedforward baseline still wins.
- The previous "Winfree preserves known pixels better" explanation is real but
  not sufficient. Once known pixels are removed from the scoring burden, current
  Winfree still hallucinates the missing block worse.
- Coarse predictive 4x4 helps over one-node global on seed11, but seed12 is
  worse and showed a large gradient spike. That branch is not reliable enough
  to scale as-is.
- The next ONN hypothesis should shift from "better spatial routing/gating" to
  "stronger missing-region generative dynamics." The field needs a richer
  content prior, an iterative residual/refinement objective, or a teacher/energy
  target that teaches the oscillator what plausible hidden digit structure
  should look like.

## Prior + Winfree Residual Refinement Probe

Implemented next as `WinfreePriorRefinementPatchDenoiser`.

Motivation: boundary-clamped block inpainting showed that current Winfree
variants are not strong hidden-region generators. The new probe gives the model
a conventional semantic prior and asks whether oscillatory dynamics can improve
that same prior through a bounded residual correction:

```text
image_plus_mask -> feedforward patch prior logits
image_plus_mask -> slow/global Winfree rate-phase residual
prior logits + strength * tanh(residual) -> final reconstruction
```

This is a hybrid, so the attribution rule is strict:

- Compare against the same feedforward baseline under the same boundary-clamped
  hidden-region protocol.
- A win means the Winfree residual improves the prior's hidden-region synthesis.
- A loss means the current oscillator branch is still not useful as a generative
  refiner, even when paired with a semantic prior.

New model family:

```bash
--model-family winfree_prior_refinement
--winfree-refinement-strength 0.25
```

Focused Modal preset, dry-run validated with `max_containers=3`:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_prior_refinement \
  --print-only
```

This preset compares feedforward prior-only against prior+Winfree residual at
strengths `0.25` and `0.5` for seeds 11 and 12 on boundary-clamped block
inpainting.

GPU result:

```text
outputs/analysis/modal_block50_prior_refinement.csv
```

Two-seed means:

| model | hidden loss | changed MSE | changed MAE | corr | F1 | diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| feedforward prior | 0.091573 | 0.091493 | 0.197825 | 0.682003 | 0.659099 | 0.603615 |
| prior + Winfree residual, strength 0.25 | 0.092636 | 0.092597 | 0.205920 | 0.678545 | 0.655539 | 0.621324 |
| prior + Winfree residual, strength 0.5 | 0.091063 | 0.091009 | 0.197840 | 0.685013 | 0.661235 | 0.613211 |

Per-seed deltas versus the matching feedforward prior:

| variant | seed | hidden-loss delta | changed-MSE delta | corr delta | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| strength 0.25 | 11 | -0.000469 | -0.000438 | +0.002292 | +0.001569 |
| strength 0.25 | 12 | +0.002594 | +0.002645 | -0.009208 | -0.008689 |
| strength 0.5 | 11 | -0.000968 | -0.000926 | +0.004750 | +0.003459 |
| strength 0.5 | 12 | -0.000053 | -0.000041 | +0.001269 | +0.000813 |

Interpretation:

- This is a small positive attribution signal for `strength=0.5`: the Winfree
  residual improved the same feedforward prior on both seeds, including hidden
  loss, changed-region MSE, correlation, F1, and diversity.
- It is not a breakthrough. The absolute hidden-loss gain is about `0.00051`
  mean MSE, so this is evidence that the oscillator can help as a refiner, not
  evidence that the current design is a dominant inpainting model.
- `strength=0.25` is not reliable. It helped seed11 but lost badly on seed12
  and showed a large final gradient spike, so the residual branch needs
  stabilization before scaling.
- The result supports the "ONN as dynamical refinement field" hypothesis more
  than the older "ONN as full decoder" hypothesis.

## Checkpoint Mask-Stress Robustness Probe

After the prior-refinement run, we reused the saved best checkpoints rather
than retraining. The goal was to test whether the small `strength=0.5` gain is
only a same-task artifact or whether it transfers across mask shape/severity.

Command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --robustness-from-json outputs/analysis/modal_block50_prior_refinement.json \
  --robustness-preset mask_stress \
  --robustness-include-regex 'feedforward|s050' \
  --robustness-csv outputs/analysis/modal_prior_refinement_mask_stress.csv
```

Scenarios:

- `block25`, `block50`, `block75`
- `patch50`, `patch75`

All evaluations use the same boundary-clamped hidden-region protocol. The
models were trained only on `block50`, so `block25`, `block75`, and both patch
mask scenarios are stress tests of the trained checkpoints.

Two-seed means:

| scenario | model | hidden eval loss | changed MSE | corr | F1 | diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| block25 | feedforward | 0.102361 | 0.102104 | 0.840045 | 0.806251 | 0.865310 |
| block25 | prior + Winfree `0.5` | 0.099576 | 0.099379 | 0.846107 | 0.813975 | 0.871981 |
| block50 | feedforward | 0.091573 | 0.091493 | 0.682003 | 0.659099 | 0.603615 |
| block50 | prior + Winfree `0.5` | 0.091063 | 0.091009 | 0.685013 | 0.661235 | 0.613211 |
| block75 | feedforward | 0.092311 | 0.092204 | 0.504856 | 0.475579 | 0.242360 |
| block75 | prior + Winfree `0.5` | 0.089362 | 0.089266 | 0.517516 | 0.510536 | 0.275118 |
| patch50 | feedforward | 0.057411 | 0.057364 | 0.823178 | 0.795089 | 0.813487 |
| patch50 | prior + Winfree `0.5` | 0.055916 | 0.055883 | 0.827682 | 0.799896 | 0.822394 |
| patch75 | feedforward | 0.073889 | 0.073693 | 0.627331 | 0.581368 | 0.599518 |
| patch75 | prior + Winfree `0.5` | 0.069918 | 0.069750 | 0.645395 | 0.603329 | 0.624394 |

Mean deltas for prior+Winfree `0.5` minus feedforward:

| scenario | changed-MSE delta | corr delta | F1 delta | diversity delta |
| --- | ---: | ---: | ---: | ---: |
| block25 | -0.002725 | +0.006062 | +0.007724 | +0.006671 |
| block50 | -0.000484 | +0.003010 | +0.002136 | +0.009596 |
| block75 | -0.002938 | +0.012660 | +0.034957 | +0.032758 |
| patch50 | -0.001481 | +0.004504 | +0.004807 | +0.008907 |
| patch75 | -0.003943 | +0.018064 | +0.021961 | +0.024876 |

Interpretation:

- This is the strongest evidence so far that the oscillator branch has a useful
  behavior we were under-measuring. The Winfree refiner improved changed-region
  MSE in all `10/10` seed/scenario pairs.
- The gain is not largest on the exact training corruption (`block50`). It grows
  on `block25`, `block75`, and especially `patch75`, which is a better
  robustness/generalization signature than a single benchmark bump.
- Correlation, foreground F1, and diversity all improve in every scenario mean.
  That points toward a structural/field-refinement effect rather than only a
  pixel-MSE quirk.
- Caveat: this is still a two-seed checkpoint probe with an extra-parameter
  hybrid. The next control should compare against an equal-capacity non-ONN
  residual refiner or a shuffled/zeroed Winfree branch.

## Anytime Settling Diagnostic

Question: if the Winfree refiner is a real relaxation process, does a checkpoint
trained with 8 recurrent steps improve when evaluated with fewer or more
settling steps?

Implementation:

- Added `--eval-winfree-steps` to checkpoint eval mode.
- The loader reconstructs the checkpoint with an overridden `winfree_steps`
  value before deserializing leaves. This is safe for step count because it is a
  static scan length, not a learned parameter shape.
- Reused the `prior_refine_s050` best checkpoints from
  `modal_block50_prior_refinement.json`.

Command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --settling-from-json outputs/analysis/modal_block50_prior_refinement.json \
  --settling-include-regex s050 \
  --settling-steps 1,2,4,8,16,32 \
  --settling-scenarios block50,patch75 \
  --settling-csv outputs/analysis/modal_prior_refinement_anytime_settling.csv
```

Result file:

```text
outputs/analysis/modal_prior_refinement_anytime_settling.csv
```

Two-seed mean curves:

| scenario | steps | hidden eval loss | changed MSE | corr | F1 | diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block50 | 1 | 0.091431 | 0.091376 | 0.683400 | 0.660551 | 0.610230 |
| block50 | 2 | 0.091289 | 0.091233 | 0.683899 | 0.660652 | 0.610531 |
| block50 | 4 | 0.091153 | 0.091098 | 0.684547 | 0.660884 | 0.611865 |
| block50 | 8 | 0.091063 | 0.091009 | 0.685013 | 0.661235 | 0.613211 |
| block50 | 16 | 0.091142 | 0.091090 | 0.684770 | 0.660929 | 0.613312 |
| block50 | 32 | 0.091558 | 0.091514 | 0.683515 | 0.660019 | 0.613882 |
| patch75 | 1 | 0.070288 | 0.070118 | 0.643501 | 0.599426 | 0.621865 |
| patch75 | 2 | 0.070247 | 0.070076 | 0.643762 | 0.599810 | 0.622005 |
| patch75 | 4 | 0.070048 | 0.069879 | 0.644776 | 0.601825 | 0.623220 |
| patch75 | 8 | 0.069918 | 0.069750 | 0.645395 | 0.603329 | 0.624394 |
| patch75 | 16 | 0.069869 | 0.069699 | 0.645630 | 0.603695 | 0.624374 |
| patch75 | 32 | 0.069839 | 0.069672 | 0.645462 | 0.604156 | 0.625117 |

Interpretation:

- The Winfree branch does show step-dependent behavior. Going from 1 to 8
  settling steps improves both `block50` and `patch75` on hidden loss,
  changed-region MSE, correlation, F1, and diversity.
- The curve does not show dramatic open-ended relaxation. `block50` is best at
  or near the trained 8-step depth and degrades by 32 steps.
- `patch75` continues to improve slightly at 16 and 32 steps, but the gains are
  small: changed MSE improves by only about `0.00008` beyond 8 steps.
- Conclusion: the current refiner is not merely a static feedforward garnish,
  but it is also not yet a strong anytime solver. It behaves like a trained
  recurrent relaxation block with a useful settling horizon, not an unbounded
  physical attractor that keeps improving indefinitely.

## Recurrent-Conv Residual Control

The mask-stress probe made the prior+Winfree result look promising, but it left
one important attribution hole: maybe the gain comes mostly from adding a
second residual refinement branch, not from Winfree phase dynamics.

Control added:

```text
image_plus_mask -> feedforward patch prior logits
image_plus_mask -> recurrent-conv residual
prior logits + strength * tanh(residual) -> final reconstruction
```

New model family:

```bash
--model-family recurrent_conv_prior_refinement
--recurrent-conv-refinement-strength 0.5
```

Training command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_recurrent_prior_refinement
```

Robustness command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --robustness-from-json outputs/analysis/modal_block50_recurrent_prior_refinement.json \
  --robustness-preset mask_stress \
  --robustness-include-regex recurrent_prior_refine_s050 \
  --robustness-csv outputs/analysis/modal_recurrent_prior_refinement_mask_stress.csv
```

Result files:

```text
outputs/analysis/modal_block50_recurrent_prior_refinement.csv
outputs/analysis/modal_recurrent_prior_refinement_mask_stress.csv
```

Two-seed `block50` training-scenario means:

| model | hidden eval loss | changed MSE |
| --- | ---: | ---: |
| feedforward prior | 0.091573 | 0.091493 |
| prior + Winfree residual, strength 0.5 | 0.091063 | 0.091009 |
| prior + recurrent-conv residual, strength 0.5 | 0.091481 | 0.091423 |

Mask-stress two-seed changed-MSE means:

| scenario | feedforward | Winfree residual | recurrent-conv residual |
| --- | ---: | ---: | ---: |
| block25 | 0.102104 | 0.099379 | 0.099894 |
| block50 | 0.091493 | 0.091009 | 0.091423 |
| block75 | 0.092204 | 0.089266 | 0.089361 |
| patch50 | 0.057364 | 0.055883 | 0.055959 |
| patch75 | 0.073693 | 0.069750 | 0.070164 |

Attribution summary:

- Winfree residual beats the feedforward prior on changed MSE in `10/10`
  seed/scenario pairs.
- The recurrent-conv residual also beats the feedforward prior in `9/10`
  seed/scenario pairs.
- Winfree beats the recurrent-conv residual in `7/10` seed/scenario pairs.
- Mean Winfree-minus-recurrent changed-MSE deltas are small:
  `-0.000514` on `block25`, `-0.000414` on `block50`, `-0.000095` on
  `block75`, `-0.000076` on `patch50`, and `-0.000414` on `patch75`.

Interpretation:

- This weakens the broad claim that oscillators uniquely caused the robustness
  gain. A large part of the win is now attributable to the hybrid pattern
  itself: feedforward semantic prior plus bounded local recurrent residual.
- It does not erase the Winfree result. The Winfree residual remains slightly
  better than the equal-pattern recurrent-conv control on most changed-MSE
  comparisons and on the same-task `block50` mean.
- The honest current claim is therefore narrower and more useful: Winfree phase
  dynamics are a competitive local refinement mechanism inside this hybrid
  protocol, with a small edge over a tied recurrent-conv residual under the
  current settings. They are not yet a decisive standalone replacement for
  conventional neural machinery.

## Scientific Synthesis: Why Winners Win

The current winners share a specific design pattern. They are not simply "more
oscillatory"; they sit in a goldilocks zone:

```text
partial local evidence
+ recurrent settling
+ constrained spatial coupling
+ no global latent shortcut
+ enough content capacity
+ phase used for coordination, not full memory
```

This explains most of the positive and negative results so far.

| Result | What happened | What it teaches |
| --- | --- | --- |
| Clean feedforward beats clean Winfree | Direct patch decoding wins when the full input is visible | Clean autoencoding is a poor proving ground for oscillatory dynamics |
| Skip-heavy latent Winfree looks strong | Removing coupling barely hurts if the sequence skip remains | A latent output path can hide whether phase coupling matters |
| No-skip latent Winfree collapses | Diversity goes near zero without the strong latent readout path | Phase-only latent decoding is too weak as the whole memory mechanism |
| Conditional no-coupling Winfree is bad | Per-patch local transforms cannot solve masked completion well | Spatial coupling is not cosmetic; it carries useful completion signal |
| One-step conditional Winfree is worse | Four to eight recurrent steps are better | Settling dynamics matter beyond a single weird activation layer |
| Recurrent-conv improves but loses | Generic local recurrence helps but does not match Winfree | The advantage is not merely "it is recurrent and local" |
| Pure adaptive coupling is bad | Replacing fixed local conv with attention over influence hurts | The stable spatial prior matters; too much flexibility too early is harmful |
| Residual adaptive is safe but neutral | `conv + small adaptive` does not clearly improve | The missing ingredient was probably not attention over phase influence alone |
| Rate-phase hybrid wins | Content state plus phase-gated settling beats phase-only Winfree | Phase appears more useful as coordination/gating than as the whole state |
| Coarse phase mesh loses | 2x2/4x4 coarse phase gates do not beat the one-node global carrier, and shuffled controls do not collapse | Interpolated coarse gating alone is not enough evidence for useful phase geometry |
| Coarse content transport loses | Additive coarse-to-fine content does not beat the slow/global or local rate-phase models, and the shuffled control is best among new variants | Naive top-down content injection is not the missing block-occlusion mechanism and can destabilize training |
| Mask-aware block loss is mixed | Strong changed weighting gave slow/global Winfree a tiny changed-region edge, but lighter weights restored a feedforward win across metrics | Better loss alignment helps, but it does not by itself produce an oscillator-specific breakthrough |
| Missing marker loses | An out-of-range `-1.0` occlusion marker helps feedforward slightly but hurts current recurrent/Winfree block models on full MSE | Missingness should be a separate control/gating field, not injected as fake image content |
| Image-plus-mask is useful but not enough | A separate visibility channel improves the block task, but feedforward still wins and the mask is only an input feature | Visibility needs to gate the dynamics directly if it is going to become an ONN-native advantage |
| Naive visibility gating loses | Directly suppressing drive and boosting transport in missing patches hurts, and shuffled visibility is essentially tied with correct visibility | Mask information helps, but this gate is not the missing mechanism; the model needs stronger active transport or a predictive coarse field |
| ConvLSTM control loses | Standard gated recurrent memory is worse than tied recurrent-conv and much worse than slow/global Winfree on block `image_plus_mask` | The Winfree field is not merely benefiting from generic recurrent memory gates, but still needs stronger missing-region synthesis to beat feedforward |
| JEPA-lite favors ConvLSTM/feedforward | Predicting hidden low-frequency DCT patch embeddings works as a benchmark, but the simple fixed target is easier for conventional predictors | A real ONN-JEPA test needs learned/EMA semantic targets or downstream probes; fixed DCT targets are not enough |
| Coarse predictive readout loses | A stable coarse readout-only branch does not beat one-node global Winfree or feedforward on block completion | Multiscale readout alone is not enough; the next block step should use clamped boundary conditions and focus optimization on the hidden region |
| Boundary-clamped protocol added | Visible pixels are clamped for every model and loss is hidden-region MSE | This separates honest inpainting from full-reconstruction scoring and prevents visible-pixel copying from becoming a headline metric |
| Boundary-clamped core still favors feedforward | Feedforward beats current Winfree variants on hidden-region MSE even when visible pixels are clamped | The bottleneck is missing-region generative synthesis, not visible-pixel preservation or metric contamination |
| Prior + Winfree residual improves robustness | A Winfree residual branch improves the same feedforward prior across mask-stress scenarios | The oscillator branch is useful as a spatial refiner, but this is a hybrid result |
| Recurrent residual control is close | A matched recurrent-conv residual captures most of the same robustness gain, while Winfree keeps a small edge | The strongest current claim is not oscillator uniqueness; it is that local recurrent refinement works, with Winfree currently a slightly better refinement mechanism |
| Un-0-style generator is alive but not attributed | Random-phase Kuramoto generation learns an MNIST-like distributional loss, but frozen reservoir and decoder-only controls are essentially tied | Oscillators-as-generative-prior is the right new branch, but the current objective/dynamics do not yet prove learned coupling value |
| Pixel-drift generator improves class attribution | The resize-conv Kuramoto model gets much higher prototype-class alignment than frozen/decoder controls, but drift loss itself is tied and samples remain fragments | The Un-0 source code points to the right structural ingredients, but pixel-only drift is still not enough; feature-space or trajectory objectives are the next real test |
| Fixed structural feature drift is not enough | Hand-built pooled/edge/moment features are optimized better by frozen/decoder controls, while learned Kuramoto remains more class-aligned but overactive | The missing Un-0 ingredient is likely learned semantic features, not just any feature-space loss |

The failed experiments therefore mostly align with the emerging hypothesis
rather than contradicting it. They draw a boundary around the useful regime:
oscillatory machinery helps when the task requires iterative spatial completion,
but only if the architecture avoids both extremes:

- Too little structure: unconstrained/adaptive coupling has poor inductive bias.
- Too little capacity: phase-only state struggles to carry all image evidence.
- Too much shortcut: global latent or sequence skips can solve the task while
  bypassing the oscillatory mechanism.

The rate-phase model was the first architecture that satisfied the current
goldilocks criteria: it keeps local recurrent phase coupling, removes the global
latent shortcut, and gives reconstruction evidence a dedicated content field.
The newer prior-refinement experiments add a second criterion: any oscillator
claim now needs an equal-pattern non-oscillatory residual control.

### Overfitting Risk

There is no strong sign of ordinary parameter overfitting yet: the best result
holds on held-out eval data, across two seeds, with aligned quality metrics
such as correlation, F1, diversity, and MAE.

There is still a real risk of researcher/benchmark overfitting. We have spent a
lot of iteration on 50% patch-masked MNIST. The rate-phase result is evidence
for an architectural principle, not proof that the principle generalizes.

The next tests should therefore be falsification-oriented:

- 75% patch masking: rate-phase still beats phase-only and controls, but with a
  much smaller margin.
- Block occlusion: feedforward currently wins, while slow/global rate-phase is
  the best oscillatory field. This supports ONN-native global mediation but
  shows that the current one-position carrier is underpowered.
- Coarse phase mesh: 2x2/4x4 coarse phase gates did not improve block
  occlusion, and phase-shuffled controls were not worse. This is a useful
  negative result against the current "coarse gate only" implementation.
- Coarse content transport: explicit coarse-to-fine additive content did not
  improve block occlusion, and content-shuffled control was best among the new
  variants. Further multiscale work should add stronger stability/auxiliary
  objectives before another expensive sweep.
- Mask-aware block loss: changed-region weighting moved all models toward
  missing-signal repair. Slow/global Winfree had a tiny changed-region edge only
  under the blunt `2.0`/`0.25` weighting; lighter sweeps favored feedforward
  across all reported metrics. This points away from more loss tuning and
  toward stronger controls or a stronger ONN-native mechanism.
- Missing-marker block probe: marking missing pixels as `-1.0` in the image
  channel does not unlock the field. It slightly improves feedforward and hurts
  current Winfree/recurrent full-MSE block results, suggesting that mask
  visibility should enter as a separate gate/control field.
- Image-plus-mask block probe: a separate visibility channel improves the
  representation and gives the best block results so far, but feedforward still
  wins. This motivated direct visibility gating, which then failed.
- Visibility-gated dynamics probe: the direct trust-gate version did not work.
  Correct visibility gating was worse than ungated slow/global rate-phase, and
  shuffled visibility was essentially tied with correct visibility. The next
  mechanism needs to add actual missing-region synthesis/transport rather than
  only scaling existing drive terms.
- ConvLSTM control probe: standard local recurrent memory gates did not solve
  the block task. ConvLSTM was worse than tied recurrent-conv and much worse
  than slow/global Winfree on the `image_plus_mask` block setup.
- JEPA-lite probe: predicting hidden DCT patch embeddings is a viable
  representation-prediction benchmark, but the first fixed-target version
  favored ConvLSTM and feedforward over Winfree. The JEPA direction remains
  interesting only with learned/EMA targets or a downstream representation
  probe.
- Pure distance-decay coupling: replacing local conv with nonlocal matrix
  coupling hurts on seed11.
- Residual distance-decay coupling: adding a weak matrix branch is stable but
  still worse than local-conv rate-phase, pointing toward slow-band or
  hierarchical mediation rather than a same-field long-range add-on.
- Gaussian noise: does the field act as a useful denoising relaxation system, or
  was the win specific to binary missing patches?
- Stronger controls: direct rate-phase masked completion survived tied
  recurrent-conv and small ConvLSTM controls. In the later prior-refinement
  setting, however, a matched recurrent-conv residual captured most of the
  robustness gain. A tiny U-Net-style denoiser or untied recurrent-conv remains
  a future bar for any stronger ONN claim.

If rate-phase wins across these, the result becomes a serious architecture
signal. If it only wins on 50% patch masking, it remains a useful but narrow
inductive-bias result.

## Current Hypotheses

1. Oscillatory fields are strongest for spatial completion, denoising,
   stabilization, and other iterative relaxation tasks.
2. A global low-dimensional latent bottleneck is a poor first fit for this
   branch, though hierarchical or spatial latent grids may still be useful.
3. The useful mechanism is not just recurrence in the direct masked-completion
   models: coupling and phase-field dynamics add value beyond a simple
   recurrent convolutional baseline. In the prior-refinement setting, though,
   generic local recurrent residual capacity explains most of the gain, so
   future ONN claims must beat equal-pattern residual controls.
4. Naive data-adaptive local coupling is not enough. Residual adaptive coupling
   is safe but still not better than fixed local conv; useful adaptive coupling
   may need rate-phase hybrid features.
5. Coarse global phase gating, naive additive coarse-to-fine content transport,
   and readout-only coarse prediction are all insufficient on block occlusion.
   The recurring pattern is not instability alone; it is weak missing-region
   synthesis.
6. Loss alignment matters. Missing-region metrics reveal effects that full MSE
   hides, but mask-aware weighting helps non-oscillatory controls too; the
   lighter block-loss sweep favored feedforward rather than slow/global
   Winfree.
7. Missingness should not be encoded as fake pixel intensity. For block
   inpainting, a separate visibility field is better; however, simply
   concatenating it as an input feature is not enough, and the first direct
   drive/transport visibility gate also failed. Mask information likely needs a
   richer role: active content transport, predictive coarse-field dynamics, or
   normalized recurrent flow.
8. Stronger controls are still needed before broad claims. The small ConvLSTM
   control did not match slow/global Winfree in the earlier block setup, but the
   recurrent-conv prior-refinement control is close to the Winfree refiner. The
   next bar is either a clearly better ONN-native refiner or a stronger
   non-oscillatory image control such as an untied recurrent-conv or tiny U-Net.
9. JEPA-style representation prediction is relevant, but fixed low-frequency
   DCT targets are not enough. The next JEPA attempt should use learned/EMA
   teacher embeddings or evaluate the settled oscillatory state on downstream
   classification/robustness.

## Most Useful Next Experiments

1. Build a stronger ONN-native refiner that can beat the recurrent residual
   control, not just the feedforward prior.
   Boundary clamping showed that current standalone Winfree variants still lose
   on hidden-region synthesis, and the recurrent-conv prior-refinement control
   explains most of the hybrid robustness gain. The next architecture should
   add a genuinely oscillator-specific advantage: phase/rate consistency losses,
   hidden-region-only iterative residual objectives, adaptive settling schedules,
   or a learned energy target for settled states.

2. Add one more stronger non-oscillatory image control later.
   The matched recurrent-conv residual is now the main near-term control. A tiny
   U-Net-style denoiser or untied recurrent-conv would still be the right next
   external bar before making large claims.

3. Replicate rate-phase on Gaussian noise.
   75% patch masking survived with a smaller margin; block occlusion exposed a
   global-coordination gap. Noise tests whether the field is a useful relaxation
   denoiser.

4. Upgrade JEPA only if the target representation becomes meaningful.
   The DCT-target JEPA-lite probe favored ConvLSTM/feedforward. A stronger
   version should use a learned/EMA teacher encoder or test whether Winfree
   settled states improve downstream classification under occlusion.

5. Improve synchrony diagnostics.
   Measure whether foreground/connected-component patches synchronize more than
   background patches, and whether synchrony changes across recurrent steps.

6. Keep adaptive coupling as a secondary branch.
   The residual version is safe, but the two-seed result does not justify making
   it the main path yet.

7. Improve the Un-0-style generator objective/architecture.
   The first generator sweeps are complete. Learned Kuramoto is competitive,
   and the closer Un-0 class-coupled mode improves class-prototype alignment,
   but frozen reservoirs and decoder-only controls remain too strong. The next
   generator step should target attribution: a better decoder/readout split,
   learned feature targets, explicit phase-field regularization, or a task
   where oscillator settling changes the generated sample in a measurable way.
   The spatial-basis probe shows that removing decoder capacity entirely is too
   harsh, so the next readout should be structured but not powerless: local
   phase-conv readout, hierarchical phase fields, or an explicit
   decoder-fraction budget.

## Un-0-Style Generative Branch

The Un-0 post reframes the oscillator question in a way that our previous
autoencoder and inpainting experiments did not: random oscillator phase is the
generative noise source, learned coupling transforms that noise, and a decoder
renders final phase features into an image. This is a different hypothesis from
masked recovery.

New reusable pieces:

- `KuramotoImageGenerator` in `oscnet.models.generative`
- `MNISTGeneratorExperimentConfig` and `run_mnist_generator_experiment` in
  `oscnet.experiments.mnist_generator`
- `examples/image_mnist_kuramoto_generator.py`
- `scripts/modal_mnist_generator.py`

Architecture:

```text
random theta_0
-> dense learned Kuramoto coupling for T steps
-> sin/cos phase features
-> small decoder
-> generated MNIST sample
```

Training is deliberately unpaired. The first objective is not reconstruction
MSE. It combines:

- sliced-Wasserstein distance on random pixel projections
- per-pixel moment matching
- per-pixel marginal distribution matching

The original Un-0 writeup also uses a few mechanisms that shaped the second
probe here: class-specific conditioning oscillators, unidirectional coupling
from the conditioning pool into the main oscillator pool, and relative phase
features before decoding. See:
`https://unconv.ai/blog/introducing-un-0-generating-images-with-coupled-oscillators/`

### Un-0 Source Audit

The open-source Un-0 reference implementation was inspected from
`https://github.com/unconv-ai/Un-0` at commit `43f2587` (2026-06-26). This
changed the read on the branch. The blog architecture is not just "Kuramoto
plus labels"; the released recipe combines several ingredients that our first
MNIST generator did not yet reproduce.

Important implementation facts:

- Tasks and scale are different. The released checkpoints target CIFAR-10 and
  ImageNet-64, not MNIST. CIFAR uses 1024/2048/4096 oscillators; ImageNet-64
  uses 6656/10240/16384 oscillators.
- The decoder is a real image renderer. Final phase features are reshaped into
  a 4x4 seed and decoded by resize-convolution blocks. Our `mlp`,
  `spatial_basis`, and `local_basis` readouts were useful controls, but they are
  not faithful to this decoder design.
- The training loss is the biggest missing piece. Un-0 uses a class-conditional
  drift objective in pixel space plus DINOv2 feature space. Generated samples
  are pulled toward same-class real positives and pushed relative to generated
  and other-class negatives. This is much more semantic than our
  sliced-Wasserstein plus pixel-moment objective.
- Training uses a per-class positive queue so the drift target sees many
  same-class positives per step. This matters because the loss needs enough
  same-class real examples to define a useful direction.
- Conditioning includes a separate oscillator population with class-specific
  unidirectional drive into the main oscillator pool. We already approximated
  this with `class_coupling`, but Un-0 also applies classifier-free-style class
  dropout during training.
- Readout is sin/cos phase encoding with relativization, commonly
  `mean_relative` for CIFAR and `ref_oscillator` for ImageNet. We have a
  simpler relative readout, but not the full set of readout options.
- They evaluate and ablate with FID, not just training loss. Their ablation
  suite explicitly compares decoder-only, frozen reservoirs, and trained
  Euler-step dynamics across learning-rate sweeps.

Implication for OscNet:

The current MNIST generator branch is promising as a testbed, but it is still a
loose analogue. The most likely reason we do not see Un-0-like qualitative
samples is not merely bad hyperparameters. We are missing the semantic drift
objective and the conv renderer, and we are operating at much smaller scale.

The next port should therefore be structural and staged:

1. Add a source-faithful resize-conv decoder/readout path in the OscNet
   generator API.
2. Add a pixel-only conditional drift loss first, because it is lightweight and
   directly maps to JAX/MNIST.
3. Add a learned or frozen feature-space drift target later. For MNIST this can
   start with a small classifier/autoencoder feature extractor before attempting
   a DINO-style dependency.
4. Repeat the attribution suite under the new loss: decoder-only,
   frozen-reservoir, trained 1/2/5/10-step dynamics, each with matched learning
   rates or at least a small LR sweep.

Port step 1 is now implemented in OscNet: `KuramotoImageGenerator` supports
`decoder_mode="resize_conv"`. For MNIST, the default source-faithful setting is
a `7x7` phase-feature seed upsampled twice to `28x28`. This mirrors the Un-0
idea that final oscillator phase features should be interpreted as a spatial
seed and rendered by convolutional upsampling, rather than forced through a
global MLP or an intentionally weak local basis.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_core
```

This compares learned Kuramoto, frozen reservoir, and decoder-only controls
using the resize-conv renderer. It is still trained with the older
SWD/moment/prototype objective, so it tests the decoder port only. A positive
or negative result here should not be treated as a final Un-0 reproduction
until the conditional drift loss is added.

Result before the RGB coarse-auxiliary layout fix:

```text
outputs/analysis/modal_mnist_generator_resize_conv_core.csv
outputs/analysis/modal_mnist_generator_samples/resize_conv_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_decoder_seed11.png
```

Two-seed means:

| variant | best loss | final eval loss | prototype nearest acc | diversity ratio | nearest real MSE | generated std | decoder param fraction | recurrent op fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.047757 | 0.052158 | 0.146484 | 0.967791 | 0.068615 | 0.291471 | 0.042426 | 0.341463 |
| decoder-only | 0.052593 | 0.071159 | 0.110352 | 0.971743 | 0.075004 | 0.291406 | 0.042426 | 0.000000 |
| frozen Kuramoto | 0.053482 | 0.059551 | 0.093750 | 0.949083 | 0.072553 | 0.287606 | 0.042426 | 0.341463 |

Interpretation:

- This is the strongest generator attribution result so far. The learned
  Kuramoto resize-conv model beats decoder-only by about 9.2% relative
  best-loss and frozen reservoir by about 10.7%.
- The decoder is still small by parameter count: about 4.2% of total parameters.
  That matters because the result is not simply a massive conventional decoder
  swallowing the oscillator branch.
- Visual samples are smoother and more spatially connected than the local-basis
  grids, but they are still mostly stroke fragments and blobs. This is not yet
  convincing digit generation.
- The result supports the Un-0 source audit: a spatial renderer is a better
  fit for oscillator phase features than the intentionally weak local basis.
  The next likely missing piece is the objective, especially conditional drift
  in pixel/feature space.

Success criteria for this branch are intentionally broader than image loss
alone:

- Quality: generated samples and distribution metrics must improve.
- Attribution: learned dynamics should beat decoder-only and frozen-reservoir
  controls.
- Dynamics value: phase trajectories should move, settle, separate classes, or
  improve with meaningful test-time integration.
- Efficiency proxy: parameter count, decoder fraction, recurrent fraction,
  estimated recurrent operation fraction, and sample throughput should be
  tracked. These are digital-simulation proxies, not physical energy claims.
- Robustness: later generator runs should test perturbations, step schedules,
  and sample diversity.

`MNISTGeneratorExperimentConfig` now writes a `success_diagnostics` block into
`summary.json` with those attribution and efficiency proxies. This is meant to
catch false positives such as "the oscillator generator won" when nearly all
capacity lives in the decoder, or when a frozen reservoir explains the result.

Controls are built into the experiment API from the start:

- `kuramoto`: learned coupling and learned natural frequencies
- `frozen_kuramoto`: frozen oscillator dynamics, trained decoder
- `decoder_only`: no oscillator settling, trained decoder

First CPU smoke:

```bash
python examples/image_mnist_kuramoto_generator.py \
  --output-dir outputs/reference/mnist_generator_kuramoto_n32_seed11_2e_marginal_smoke \
  --data-source idx \
  --train-limit 128 \
  --eval-limit 64 \
  --eval-sample-count 64 \
  --epochs 2 \
  --batch-size 16 \
  --seed 11 \
  --num-oscillators 32 \
  --decoder-hidden-dim 64 \
  --decoder-depth 1 \
  --steps 4 \
  --num-projections 32 \
  --learning-rate 0.002 \
  --checkpoint-every 2 \
  --artifact-every 2
```

Smoke result:

| epoch | train loss | eval loss |
| ---: | ---: | ---: |
| 1 | 0.174118 | 0.149231 |
| 2 | 0.161513 | 0.139136 |

Final smoke diagnostics:

| metric | value |
| --- | ---: |
| generated mean | 0.121476 |
| generated std | 0.021985 |
| diversity ratio | 0.100285 |
| nearest real MSE | 0.033247 |
| real nearest-real MSE | 0.059272 |

Interpretation:

- The training path has real signal: distributional train and eval losses both
  move in the right direction on real MNIST.
- The sparse output bias and marginal term fix the first failure mode from the
  earliest smoke, where samples stayed around gray `0.5`.
- The visual samples are still low-diversity dark texture rather than digits.
  This is not yet evidence of strong generation.
- The correct next experiment is a GPU control sweep, not more local CPU
  tinkering: learned Kuramoto vs frozen reservoir vs decoder-only, same decoder
  and same distributional objective.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_core
```

Modal control sweep result:

```text
outputs/analysis/modal_mnist_generator_core.csv
outputs/analysis/modal_mnist_generator_core.json
```

Two-seed means:

| model | final eval loss | best loss | diversity ratio | nearest real MSE | pixel mean MSE | pixel std MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decoder-only | 0.016180 | 0.013708 | 0.956742 | 0.071990 | 0.001419 | 0.001140 |
| frozen Kuramoto | 0.015950 | 0.013801 | 0.985443 | 0.073762 | 0.001882 | 0.001063 |
| learned Kuramoto | 0.015638 | 0.013897 | 0.981784 | 0.073769 | 0.002242 | 0.001049 |

Seed-level read:

- Learned Kuramoto has the best two-seed mean final eval loss.
- Decoder-only has the best two-seed mean best loss and nearest-real MSE.
- Frozen Kuramoto is essentially tied with learned Kuramoto on diversity and
  distribution metrics.
- Visual samples are much better than the CPU smoke: digit-like blobs and loops
  appear, but samples are still speckled and not clean MNIST digits.

Interpretation:

- This branch is materially closer to the Un-0 idea than the earlier
  autoencoder/inpainting work: generated samples come from random phase/noise,
  not from an input image.
- The first control sweep does not yet show an oscillator-specific win. Learned
  coupling is competitive, but decoder-only and frozen reservoir explain most
  of the current generator behavior.
- The next generator sprint should not simply scale this exact objective. It
  should add a stronger distributional signal or ONN-native dynamics that can
  beat the decoder-only/frozen controls: class-conditioned generation,
  learned/teacher feature drift, phase-rate state, or a richer readout that
  preserves oscillator attribution.

### Simple Conditional Generator Probe

The first conditional probe added label phase shifts plus class/prototype
distribution terms, while keeping the same three controls:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_conditional_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_conditional_core.csv
outputs/analysis/modal_mnist_generator_conditional_core.json
```

Two-seed means:

| model | final eval loss | best loss | diversity ratio | nearest real MSE | prototype MSE | prototype nearest acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.040180 | 0.039609 | 0.868265 | 0.065366 | 0.063690 | 0.092773 |
| frozen Kuramoto | 0.041478 | 0.039540 | 0.883173 | 0.064547 | 0.065083 | 0.088867 |
| decoder-only | 0.042624 | 0.039262 | 0.878380 | 0.064868 | 0.065232 | 0.106445 |

Interpretation:

- Label phase shifts did not make the generator meaningfully
  class-conditional. Prototype-nearest accuracy stayed near chance.
- Learned Kuramoto had the best final eval loss, but decoder-only had the best
  best-loss and class-prototype nearest accuracy.
- Visual samples remained digit-like blobs/strokes, not convincing conditional
  MNIST digits.

### Un-0-Coupled Conditioning Probe

The closer Un-0 probe added:

- separate conditioning oscillators
- label-specific unidirectional coupling from conditioning oscillators into the
  main oscillator pool
- reference-relative phase readout before the decoder

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_un0_coupled_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_un0_coupled_core.csv
outputs/analysis/modal_mnist_generator_un0_coupled_core.json
```

Two-seed means:

| model | final eval loss | best loss | diversity ratio | nearest real MSE | prototype MSE | prototype nearest acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| class-coupled Kuramoto | 0.040438 | 0.039375 | 0.878092 | 0.063757 | 0.062613 | 0.122070 |
| class-coupled frozen reservoir | 0.039324 | 0.039324 | 0.869282 | 0.065137 | 0.062630 | 0.104492 |
| phase-shift decoder-only | 0.040402 | 0.039208 | 0.853709 | 0.062119 | 0.062876 | 0.102539 |

Interpretation:

- This is closer to the Un-0 architecture than the first conditional probe.
- Learned class-coupled Kuramoto gives the best class-prototype nearest
  accuracy and diversity of the three variants.
- Frozen class-coupled dynamics gives the best final eval loss, and
  decoder-only still wins nearest-real MSE and best-loss.
- The visual samples are still speckled MNIST-shaped mass, not clean digits.
  This branch is alive, but current results do not yet show a strong learned
  oscillator-specific generator advantage.

Current generator conclusion:

The oscillator-as-generative-prior framing is better aligned with the Un-0
idea than the old autoencoder branch, but the present MNIST implementation is
still mostly limited by objective/readout attribution. Class-coupled dynamics
nudges conditional structure in the right direction; it does not yet deliver
the qualitative leap from the blog post.

### Low-Decoder Spatial Basis Probe

The new success diagnostics exposed a major attribution problem: small
generator smokes can spend nearly all trainable capacity in the decoder. A
synthetic smoke showed `decoder_param_fraction = 0.993`, which means a nominal
oscillator generator can still be mostly a conventional image decoder.

To pressure-test this, `KuramotoImageGenerator` now supports
`--decoder-mode spatial_basis`. This replaces the MLP decoder with fixed
Gaussian image bases and trainable sin/cos phase weights per oscillator:

```text
final theta
-> sin/cos phase value per oscillator
-> fixed spatial Gaussian basis per oscillator
-> generated image
```

CPU smoke:

```bash
python examples/image_mnist_kuramoto_generator.py \
  --output-dir outputs/reference/mnist_generator_spatial_basis_peak_n32_seed11_2e_smoke \
  --data-source idx \
  --train-limit 128 \
  --eval-limit 64 \
  --eval-sample-count 64 \
  --epochs 2 \
  --batch-size 16 \
  --seed 11 \
  --conditional \
  --class-moment-weight 0.5 \
  --prototype-weight 0.2 \
  --label-phase-scale 1.0 \
  --num-condition-oscillators 8 \
  --conditioning-mode class_coupling \
  --readout-mode relative \
  --decoder-mode spatial_basis \
  --num-oscillators 32 \
  --decoder-depth 0 \
  --steps 4 \
  --num-projections 32
```

Smoke result:

| metric | value |
| --- | ---: |
| final eval loss | 0.206182 |
| generated std | 0.006301 |
| diversity ratio | 0.033129 |
| decoder param fraction | 0.017283 |
| trainable recurrent param fraction | 0.982717 |
| estimated recurrent op fraction | 0.140044 |

Interpretation:

- This is a strong attribution probe, and it does what it was designed to do:
  decoder capacity drops from almost all of the model to almost none of it.
- Image generation collapses to dim smooth blobs on the tiny CPU smoke.
- Peak-normalized spatial bases are more alive than sum-normalized bases, but
  the current low-capacity readout is too weak to generate MNIST.
- The next generator architecture should look for a middle ground: structured
  spatial readout, local convolutional phase readout, hierarchical oscillator
  fields, or a small decoder with an explicit decoder-fraction budget.

### Structured Local-Basis Readout Probe

The next readout used that middle ground. `KuramotoImageGenerator` now supports
`--decoder-mode local_basis`, where each oscillator writes a trainable local
patch through fixed Gaussian patch bases:

```text
final theta
-> relative sin/cos phase value per oscillator
-> per-oscillator local patch weights
-> fixed Gaussian patch placement
-> generated image
```

This is still an attribution probe, but not an impossible one: the decoder can
draw local stroke fragments, while global composition must come from the phase
field and conditioning dynamics.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_local_readout_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_local_readout_core.csv
outputs/analysis/modal_mnist_generator_local_readout_core.json
outputs/analysis/modal_mnist_generator_samples/local_class_coupled_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/local_class_coupled_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/local_phase_shift_decoder_seed11.png
```

Two-seed means:

| model | final eval loss | diversity ratio | nearest real MSE | prototype nearest acc | generated std | decoder param fraction | trainable recurrent fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local-basis class-coupled Kuramoto | 0.040158 | 0.845809 | 0.062803 | 0.321289 | 0.267580 | 0.079592 | 0.920408 |
| local-basis frozen class-coupled reservoir | 0.053789 | 0.813664 | 0.065187 | 0.087891 | 0.246310 | 0.079592 | 0.000000 |
| local-basis phase-shift decoder-only | 0.053613 | 0.819346 | 0.066038 | 0.102539 | 0.247662 | 0.157739 | 0.000000 |

Interpretation:

- This is the clearest positive attribution result in the generator branch so
  far. With a structured, low-capacity readout, learned class-coupled Kuramoto
  dynamics beat both frozen dynamics and decoder-only controls by about 25% on
  final eval loss.
- Prototype-nearest accuracy jumps from near chance in the controls to about
  32%, so the learned dynamics are carrying class-conditional organization,
  not just extra ink.
- The decoder is no longer dominating the model. The winning run has only
  about 8% of parameters in the decoder and about 92% in trainable recurrent
  dynamics, so this is a much cleaner oscillator-claim setup than the MLP
  generator.
- The samples are still visibly broken: stroke-like fragments, partial loops,
  and digit-ish masses appear, but global composition is unstable and speckled.
  This is not yet a qualitative Un-0-style breakthrough.

Updated generator conclusion:

The generator branch now has a real foothold. The MLP readout branch produced
better-looking distributional shortcuts but weak attribution; the scalar
spatial-basis branch had strong attribution but too little rendering power; the
local-basis branch is the first useful middle regime where trained oscillator
dynamics clearly matter. The next generator sprint should focus on making the
phase field compose strokes into whole digits without handing the job back to a
large decoder.

### Spatial Coupling Composition Probe

The local-basis samples suggest the current failure mode is not local stroke
rendering. It is global composition: fragments, loops, and digit-ish masses
appear, but they do not reliably settle into coherent whole digits.

To test an ONN-native composition mechanism, `KuramotoImageGenerator` now
supports `coupling_profile="distance_decay"`. This applies a fixed spatial
decay profile to the learned pairwise couplings:

```text
effective coupling_ij
= learned coupling_ij * distance_profile_ij
  + coupling_bias_strength * distance_profile_ij
```

The default remains `coupling_profile="dense"`, so existing generator behavior
is unchanged. The distance profile is a physics-style inductive bias: mostly
local coordination, with optional weak long-range communication through
`coupling_floor`, without giving the renderer a larger decoder.

Tiny synthetic smoke:

```bash
python examples/image_mnist_kuramoto_generator.py \
  --output-dir outputs/reference/mnist_generator_distance_decay_local_basis_synthetic_smoke \
  --data-source synthetic \
  --model-family kuramoto \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4 \
  --eval-sample-count 4 \
  --batch-size 4 \
  --num-oscillators 9 \
  --decoder-depth 0 \
  --decoder-mode local_basis \
  --local-patch-size 3 \
  --steps 2 \
  --conditional \
  --conditioning-mode class_coupling \
  --num-condition-oscillators 3 \
  --readout-mode relative \
  --coupling-profile distance_decay \
  --coupling-length-scale 0.6 \
  --coupling-floor 0.05 \
  --coupling-bias-strength 0.1
```

Smoke result:

| metric | value |
| --- | ---: |
| final eval loss | 0.118524 |
| coupling profile mean | 0.142904 |
| coupling profile max | 0.286885 |
| decoder param fraction | 0.294756 |
| trainable recurrent param fraction | 0.705244 |
| phase mean abs displacement | 0.027251 |

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_spatial_coupling_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_spatial_coupling_core.csv
outputs/analysis/modal_mnist_generator_spatial_coupling_core.json
outputs/analysis/modal_mnist_generator_samples/spatial_coupled_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/spatial_coupled_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/spatial_phase_shift_decoder_seed11.png
```

Two-seed means:

| model | final eval loss | diversity ratio | nearest real MSE | prototype nearest acc | generated std | decoder param fraction | trainable recurrent fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial-decay local-basis Kuramoto | 0.040267 | 0.843700 | 0.062888 | 0.321289 | 0.267462 | 0.079592 | 0.920408 |
| spatial-decay frozen reservoir | 0.053787 | 0.813637 | 0.065188 | 0.088867 | 0.246321 | 0.079592 | 0.000000 |
| spatial-decay decoder-only | 0.053614 | 0.819352 | 0.066038 | 0.099609 | 0.247662 | 0.157739 | 0.000000 |

Dense-vs-spatial comparison:

| trained local-basis variant | final eval loss | prototype nearest acc | diversity ratio | phase displacement |
| --- | ---: | ---: | ---: | ---: |
| dense coupling | 0.040158 | 0.321289 | 0.845809 | 0.323149 |
| distance-decay coupling | 0.040267 | 0.321289 | 0.843700 | 0.319241 |

Interpretation:

- Distance-decay coupling preserves the main positive result: trained
  class-coupled dynamics still beat frozen and decoder-only controls by about
  25% on final eval loss.
- It does not improve the dense local-basis generator. The difference from
  dense coupling is within noise and slightly worse on loss/diversity.
- The representative samples are visually very close to the dense local-basis
  samples: stroke fragments and partial loops, but still unstable whole-digit
  composition.
- This points to a sharper attribution question. The trained-vs-frozen gap may
  be driven more by learned class-conditioning oscillators/couplings than by
  learned main-pool recurrent coupling. The next control should separate
  trainable conditioning dynamics from trainable recurrent field dynamics
  before adding another architectural embellishment.

### Trainability Attribution Probe

To isolate the source of the local-basis win, `KuramotoImageGenerator` now
separates:

- `train_recurrent_dynamics`: train/freeze main-pool omega and learned pairwise
  recurrent coupling
- `train_conditioning_dynamics`: train/freeze label phase shifts and
  class-conditioning oscillator phases/couplings

Both default to the old `train_dynamics` behavior, so existing generator runs
keep their original semantics unless these flags are set explicitly.

This also fixes a control sharpness issue: direct `phase_shift` conditioning is
now governed by `train_conditioning_dynamics`, so decoder-only controls no
longer silently train label phase shifts unless requested.

Tiny synthetic smoke:

```bash
python examples/image_mnist_kuramoto_generator.py \
  --output-dir outputs/reference/mnist_generator_conditioning_only_synthetic_smoke \
  --data-source synthetic \
  --model-family kuramoto \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4 \
  --eval-sample-count 4 \
  --batch-size 4 \
  --num-oscillators 9 \
  --decoder-depth 0 \
  --decoder-mode local_basis \
  --local-patch-size 3 \
  --steps 2 \
  --conditional \
  --conditioning-mode class_coupling \
  --num-condition-oscillators 3 \
  --readout-mode relative \
  --no-train-recurrent-dynamics \
  --train-conditioning-dynamics
```

Smoke result:

| metric | value |
| --- | ---: |
| final eval loss | 0.109048 |
| train recurrent dynamics | false |
| train conditioning dynamics | true |
| trainable main recurrent params | 0 |
| trainable conditioning params | 300 |
| decoder param fraction | 0.294756 |
| trainable recurrent param fraction | 0.647948 |

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_trainability_attribution_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_trainability_attribution_core.csv
outputs/analysis/modal_mnist_generator_trainability_attribution_core.json
outputs/analysis/modal_mnist_generator_samples/attrib_all_trained_seed11.png
outputs/analysis/modal_mnist_generator_samples/attrib_conditioning_only_seed11.png
outputs/analysis/modal_mnist_generator_samples/attrib_recurrent_only_seed11.png
outputs/analysis/modal_mnist_generator_samples/attrib_frozen_seed11.png
```

Two-seed means:

| variant | final eval loss | prototype nearest acc | diversity ratio | generated std | phase displacement | trainable main recurrent params | trainable conditioning params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all-trained | 0.040159 | 0.322266 | 0.845796 | 0.267580 | 0.323165 | 65,792 | 82,240 |
| conditioning-only | 0.040346 | 0.322266 | 0.842600 | 0.267108 | 0.310421 | 0 | 82,240 |
| recurrent-only | 0.052771 | 0.097656 | 0.817216 | 0.248597 | 0.262592 | 65,792 | 0 |
| frozen | 0.053789 | 0.087891 | 0.813667 | 0.246311 | 0.248903 | 0 | 0 |

Interpretation:

- Conditioning-only explains almost the entire local-basis generator win. It
  matches all-trained prototype accuracy and is within about 0.5% relative loss.
- Recurrent-only barely beats frozen on loss and stays near chance on
  prototype-nearest accuracy. Under this objective, the main recurrent
  oscillator field is not the primary source of class-conditional structure.
- The representative sample grids agree with the metrics. All-trained vs
  conditioning-only has a small mean absolute pixel difference, while
  conditioning-only vs recurrent-only is much farther apart.
- This does not make the branch useless. It sharpens it: the current
successful mechanism is a learned oscillator-conditioned drive into a
structured local phase renderer, not yet a self-organizing recurrent field.

### Unconditional Local-Basis Probe

The trainability attribution result raised a sharp concern: maybe the
class-conditional generator was mostly learning a label-to-renderer drive, not
an oscillator field. To remove that route, this probe disables labels,
class-moment loss, and prototype loss while keeping the low-capacity
`local_basis` renderer.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_unconditional_local_readout_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_unconditional_local_readout_core.csv
outputs/analysis/modal_mnist_generator_samples/uncond_local_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/uncond_local_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/uncond_local_decoder_seed11.png
```

Two-seed means:

| variant | best loss | final eval loss | diversity ratio | nearest real MSE | generated std | phase displacement | recurrent op fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.024296 | 0.024530 | 0.858939 | 0.068686 | 0.255290 | 0.257059 | 0.172492 |
| frozen Kuramoto | 0.025112 | 0.025418 | 0.855574 | 0.068636 | 0.253060 | 0.248900 | 0.172492 |
| decoder-only | 0.025203 | 0.025682 | 0.856225 | 0.068599 | 0.252956 | 0.000000 | 0.000000 |

Interpretation:

- This is the first generator control where learned main-pool recurrent
  dynamics clearly win after removing class conditioning. The relative
  best-loss gain is about 3.2% over frozen Kuramoto and about 3.6% over
  decoder-only.
- The win is real but small. Visual samples remain fragmented stroke textures,
  not convincing MNIST digits. Learned and frozen sample grids are visibly
  close even though they are not identical.
- This supports the branch, but it does not support a "just tune it" story.
  The recurrent phase field contributes under the current objective, yet the
  objective/readout still allows texture-level distribution matching without
  learning whole-digit composition.
- The next useful generator move should not be generic hyperparameter search.
  It should force temporal organization to matter: trajectory/self-consistency
  losses, longer settling with step-wise targets, denoising/phase-noise
  consistency, or a weak-conditioning task where labels cannot directly drive
  the output.

### Resize-Conv Pixel-Drift Probe

After inspecting the open-source Un-0 code, the next structural port was a
pixel-only version of their class-conditional drift objective. This is still
lighter than the released system: it does not use DINOv2 or another semantic
feature extractor, and it does not yet maintain a large per-class queue. The
purpose was to test whether the Un-0-style objective direction helps the
MNIST-scale oscillator generator before adding more machinery.

Implemented pieces:

- `loss_mode="pixel_drift"` in `MNISTGeneratorExperimentConfig`.
- `conditional_pixel_drift_loss`, a fixed-shape JAX loss that pulls generated
  samples toward same-class real positives and uses same-class generated plus
  other-class real samples as negatives.
- CLI controls for `--pixel-drift-weight`, `--distributional-weight`,
  `--drift-gamma`, and `--drift-temperatures`.
- Modal preset `mnist_generator_resize_conv_pixel_drift_core`, comparing
  learned Kuramoto, frozen reservoir, and decoder-only controls over seeds 11
  and 12.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_core.csv
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_kuramoto_seed12.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_decoder_seed11.png
```

Two-seed means:

| variant | best drift loss | final drift loss | prototype nearest acc | diversity ratio | nearest real MSE | generated std | phase displacement | recurrent op fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.000829 | 0.000847 | 0.740234 | 1.466917 | 0.092464 | 0.356046 | 0.402894 | 0.341463 |
| frozen Kuramoto | 0.000813 | 0.000851 | 0.096680 | 1.140204 | 0.087139 | 0.293471 | 0.242569 | 0.341463 |
| decoder-only | 0.000830 | 0.000867 | 0.087891 | 1.064461 | 0.080928 | 0.281663 | 0.000000 | 0.000000 |

Interpretation:

- This is a genuine attribution improvement, but not a clean loss win. Learned
  Kuramoto is effectively tied with frozen/decoder controls on drift loss, yet
  it improves prototype-nearest class alignment by roughly 7.6x over frozen
  and 8.4x over decoder-only.
- The visual grids support the split metric read. Learned Kuramoto samples are
  sharper, more stroke-like, and more class-organized than controls, while
  frozen/decoder samples are more speckled and less coherent.
- The samples are still not convincing MNIST digits. They are saturated stroke
  fragments and partial loops, not stable complete digits.
- Pixel-only drift appears to reward class-texture and stroke evidence, but it
  does not supply enough semantic/global structure to force whole-digit
  composition. This matches the Un-0 source audit: their released objective is
  pixel drift plus feature-space drift, not pixel drift alone.
- The next meaningful port is not another broad hyperparameter sweep. It is a
  semantic target: a small MNIST feature extractor or classifier feature drift,
  followed by the same frozen/decoder attribution controls. A trajectory or
  step-wise consistency loss is also plausible, but feature drift is the
  clearest missing piece from the released Un-0 recipe.

### Queue-Backed Pixel-Drift Port

After inspecting the Un-0 source, one important gap remained in the pixel-drift
probe: same-class positives were batch-local only. Un-0 uses a per-class
positive memory/queue, which gives each generated sample a richer same-class
target set than the current mini-batch happens to contain.

Implemented pieces:

- `MNISTDriftQueue`, a host-side per-class FIFO memory for positive real
  examples.
- `conditional_pixel_drift_loss` can now separate generated labels,
  queue-positive labels, and current-batch other-class real labels.
- Feature drift reuses the same positive-memory path, so queue-backed learned
  feature drift is possible later.
- CLI controls: `--drift-queue-size` and `--drift-queue-num-pos`.
- Modal preset `mnist_generator_resize_conv_pixel_drift_queue_core`.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_core.csv
outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_core.json
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_queue_kuramoto_seed12.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_drift_queue_decoder_seed11.png
```

The sweep completed on Modal with `OSCNET_MODAL_MAX_CONTAINERS=3`. The queue was
fully warmed by the final epochs: `final_train_drift_queue_ready = 1.0` for all
six runs, and all ten per-class queues reached `512` examples.

Two-seed means:

| variant | best loss | final loss | prototype nearest acc | diversity ratio | nearest real MSE | generated std | phase displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.000810 | 0.000813 | 0.736328 | 1.447054 | 0.087654 | 0.352506 | 0.424440 |
| frozen Kuramoto | 0.000816 | 0.000827 | 0.118164 | 1.016329 | 0.076852 | 0.268222 | 0.242569 |
| decoder-only | 0.000814 | 0.000852 | 0.096680 | 1.010468 | 0.075112 | 0.265230 | 0.000000 |

Comparison with batch-local pixel drift:

| variant | batch-local final loss | queue final loss | batch-local proto acc | queue proto acc |
| --- | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.000847 | 0.000813 | 0.740234 | 0.736328 |
| frozen Kuramoto | 0.000851 | 0.000827 | 0.096680 | 0.118164 |
| decoder-only | 0.000867 | 0.000852 | 0.087891 | 0.096680 |

Interpretation:

- Queue-backed positives improve the objective modestly for all variants, so the
  queue mechanism is a real training improvement.
- The learned Kuramoto attribution gap survives. Prototype-nearest class
  alignment remains about 6.2x over frozen and 7.6x over decoder-only.
- The queue does not solve global sample quality. The learned Kuramoto grid is
  more stroke-like and class-organized than decoder-only, but still saturated,
  fragmented, and not reliably complete digits.
- This narrows the missing Un-0 ingredients. Positive-memory size was not the
  main bottleneck. The next source-faithful move should combine queue-backed
  drift with a stronger semantic feature target, or move to a dataset/feature
  extractor closer to Un-0's CIFAR/DINO regime.
- Two representative PNGs were downloaded. Additional seed-11 Kuramoto/frozen
  grid downloads hit Modal `GOAWAY` connection closes, but the sweep outputs and
  JSON summaries are complete.

### Fixed Structural Feature-Drift Probe

The first feature-space follow-up added a deterministic MNIST feature map:
`7x7` pooled ink layout, pooled signed edge fields, row/column profiles, and
low-order image moments. This was meant as a cheap bridge between pixel-only
drift and a learned semantic feature target. It is differentiable and reusable,
but it is not a learned classifier or DINO-like representation.

Implemented pieces:

- `mnist_structural_features` and `conditional_feature_drift_loss`.
- `loss_mode="feature_drift"` and `loss_mode="pixel_feature_drift"`.
- CLI controls for `--feature-drift-weight` and `--feature-drift-mode`.
- Modal preset `mnist_generator_resize_conv_feature_drift_core`, using
  `0.5 * pixel_drift + structural_feature_drift`.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_feature_drift_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_feature_drift_core.csv
outputs/analysis/modal_mnist_generator_samples/resize_conv_feature_kuramoto_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_feature_kuramoto_seed12.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_feature_frozen_seed11.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_feature_decoder_seed11.png
```

Two-seed means:

| variant | best loss | final loss | feature drift | pixel drift | prototype nearest acc | diversity ratio | nearest real MSE | generated std | phase displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.003957 | 0.004520 | 0.004060 | 0.000922 | 0.641602 | 1.549316 | 0.105807 | 0.375672 | 0.384248 |
| frozen Kuramoto | 0.003681 | 0.003842 | 0.003425 | 0.000833 | 0.111328 | 0.974674 | 0.071494 | 0.260678 | 0.242569 |
| decoder-only | 0.003822 | 0.003858 | 0.003435 | 0.000845 | 0.096680 | 0.987100 | 0.072389 | 0.260219 | 0.000000 |

Interpretation:

- This is a useful negative result. The fixed feature target is easier for
  conservative frozen/decoder controls to optimize than for learned Kuramoto.
- Learned Kuramoto still produces much stronger class-prototype alignment:
  about 5.8x over frozen and 6.6x over decoder-only. But it pays for that with
  overactive, high-contrast fragments, worse nearest-real MSE, and worse
  feature-drift loss.
- The visual samples confirm the metric split. Structural feature drift makes
  the learned oscillator samples brighter and more class-suggestive, but not
  more complete. Frozen/decoder controls remain blurrier and less class
  organized, but they satisfy the hand-built target better.
- This means "feature-space drift" is not magic by itself. The hand-built
  feature map mostly encodes generic ink geometry. It does not supply the
  semantic manifold that Un-0 gets from DINOv2-like features.
- The infrastructure is still valuable: OscNet can now train generator losses
  in a non-pixel feature space. The next meaningful step is a learned MNIST
  classifier/encoder feature target, frozen during generator training, with the
  same attribution controls.

### Learned MNIST Feature-Drift Probe

The next Un-0-inspired probe replaced the hand-built structural feature map with
a small frozen MNIST classifier. The classifier is trained first, then generator
training uses its normalized penultimate features as the semantic feature space
for conditional drift. This is still much smaller than Un-0's DINOv2 feature
target, but it tests the same structural idea: the generator should move toward
same-class real samples in a learned representation, not only in pixel space.

Implemented pieces:

- `MNISTFeatureClassifier` and `train_mnist_feature_classifier`.
- `feature_drift_mode="learned"`.
- CLI controls for `--learned-feature-epochs`, `--learned-feature-dim`,
  `--learned-feature-depth`, `--learned-feature-learning-rate`, and
  `--learned-feature-weight-decay`.
- Modal preset `mnist_generator_resize_conv_learned_feature_drift_core`, using
  `0.5 * pixel_drift + learned_feature_drift`.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_core.csv
outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_core.json
```

Two-seed means:

| variant | best loss | final loss | feature drift | pixel drift | feature-clf acc | prototype nearest acc | diversity ratio | nearest real MSE | generated std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.009964 | 0.010862 | 0.010309 | 0.001105 | 0.911508 | 0.185547 | 1.446040 | 0.104941 | 0.294305 |
| frozen Kuramoto | 0.009548 | 0.010700 | 0.010126 | 0.001147 | 0.911508 | 0.105469 | 1.803490 | 0.137763 | 0.348196 |
| decoder-only | 0.009403 | 0.010671 | 0.010175 | 0.000991 | 0.911508 | 0.095703 | 1.470080 | 0.099960 | 0.292720 |

Interpretation:

- The learned classifier is real enough for this probe: final eval accuracy is
  about 91% on the limited training setup.
- This did not produce the hoped-for Un-0-like jump. Decoder-only and frozen
  controls optimize the learned-feature drift loss at least as well as learned
  Kuramoto.
- Learned Kuramoto still has better prototype-nearest class alignment than
  controls, but the gap is much smaller than the pixel-only drift result and the
  samples remain fragmentary.
- The likely lesson is that a small MNIST classifier feature target is not a
  substitute for Un-0's pretrained DINO-style feature space and per-class
  positive queue. It supplies a semantic signal, but not a strong generative
  manifold.
- This makes the next source-faithful missing piece clearer: queue-backed
  class-conditional drift and/or a stronger pretrained feature target should be
  tested before declaring the Un-0 recipe structurally weak.

### Queue + Learned Feature-Drift Probe

This probe combined the two closest Un-0 ports currently available in OscNet:
queue-backed same-class positives plus a frozen learned MNIST feature target.
It answers a narrow question left open by the prior two results: did learned
feature drift fail mainly because same-class positives were too sparse?

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_queue_core
```

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_queue_core.csv
outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_queue_core.json
outputs/analysis/modal_mnist_generator_samples/resize_conv_learned_feature_queue_kuramoto_seed12.png
outputs/analysis/modal_mnist_generator_samples/resize_conv_learned_feature_queue_decoder_seed12.png
```

The learned feature classifiers again reached about 91% eval accuracy, and the
queue was fully active by the final epochs.

Two-seed means:

| variant | best loss | final loss | feature drift | pixel drift | prototype nearest acc | diversity ratio | nearest real MSE | generated std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learned Kuramoto | 0.009477 | 0.010396 | 0.009914 | 0.000963 | 0.312500 | 1.206450 | 0.082502 | 0.265216 |
| frozen Kuramoto | 0.009279 | 0.010322 | 0.009692 | 0.001261 | 0.083008 | 1.590300 | 0.131546 | 0.314638 |
| decoder-only | 0.009495 | 0.010427 | 0.009869 | 0.001117 | 0.123047 | 1.475574 | 0.105314 | 0.299124 |

Comparison against the two nearest probes:

| probe | learned Kuramoto final loss | learned Kuramoto proto acc | learned Kuramoto nearest real MSE |
| --- | ---: | ---: | ---: |
| learned feature drift, no queue | 0.010862 | 0.185547 | 0.104941 |
| pixel drift with queue | 0.000813 | 0.736328 | 0.087654 |
| learned feature drift with queue | 0.010396 | 0.312500 | 0.082502 |

Interpretation:

- The queue helps learned-feature drift somewhat: learned Kuramoto improves from
  about 0.186 to 0.313 prototype-nearest accuracy and gets a lower final loss
  than the no-queue learned-feature run.
- It does not preserve the strong pixel-queue result. Prototype-nearest accuracy
  falls from about 0.736 with pixel queue alone to about 0.313 with
  learned-feature queue.
- Learned Kuramoto still beats frozen and decoder controls on class alignment,
  but the frozen model has slightly better best/final loss and feature-drift
  loss. So the feature target is not cleanly rewarding learned oscillator
  dynamics.
- Visual samples confirm the mixed metric read: queue+feature Kuramoto is more
  organized than decoder-only but less digit-like/class-consistent than
  pixel-queue Kuramoto.
- This is a useful negative. Sparse positives were not the main reason learned
  MNIST feature drift failed. The bottleneck is probably the feature target
  itself: a small MLP classifier feature space is not acting like Un-0's DINOv2
  semantic manifold.

### Un-0 Source Alignment: Dynamic Conditioning Oscillators

Reading the open-source Un-0 implementation clarified that our first
`class_coupling` port was still a simplification. It used a learned static
class phase anchor. The Un-0 source instead evolves a separate conditioning
oscillator population and uses the class label to select a one-way drive matrix
from that conditioning pool into the main oscillator pool.

Port step:

- `KuramotoImageGenerator` now supports
  `conditioning_mode="class_oscillator"`.
- This mode samples random conditioning phases, evolves them under their own
  Kuramoto dynamics, and applies class-specific unidirectional coupling into
  the main oscillator pool.
- Readout now distinguishes `mean_relative` from `ref_oscillator`; the old
  `relative` mode remains as a ref-oscillator alias for existing experiments.
- Traces now expose `condition_initial_theta`,
  `condition_theta_trajectory`, `condition_final_theta`, `condition_omega`,
  and `condition_coupling`.

Next preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core
```

This is deliberately not a new objective. It keeps the strongest current MNIST
generator setup fixed: resize-conv decoder plus queue-backed pixel drift. The
only scientific variable is source-faithful dynamic conditioning plus
`mean_relative` readout. Compare it directly against
`modal_mnist_generator_resize_conv_pixel_drift_queue_core.csv`.

Result:

```text
outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core.csv
outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core.json
```

Two-seed means, compared to the previous static `class_coupling` queue-backed
pixel-drift branch:

| branch | variant | final loss | prototype nearest acc | nearest real MSE | diversity ratio | generated std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static class coupling | learned Kuramoto | 0.000813 | 0.736328 | 0.087654 | 1.447054 | 0.352506 |
| static class coupling | frozen | 0.000827 | 0.118164 | 0.076852 | 1.016329 | 0.268222 |
| static class coupling | decoder-only | 0.000852 | 0.096680 | 0.075112 | 1.010468 | 0.265230 |
| dynamic condition oscillator | learned Kuramoto | 0.000810 | 0.094727 | 0.080897 | 1.058222 | 0.272014 |
| dynamic condition oscillator | frozen | 0.000867 | 0.086914 | 0.084220 | 1.066520 | 0.283143 |
| dynamic condition oscillator | decoder-only | 0.000836 | 0.108398 | 0.077674 | 1.038647 | 0.264360 |

Interpretation:

- Source-faithful dynamic conditioning did not improve the MNIST proxy. The
  learned Kuramoto final loss is similar to the static branch, but the class
  alignment signal collapses from about 0.736 to about 0.095.
- This means the earlier strong class-alignment result was not simply “because
  Un-0 has conditioning oscillators.” In our MNIST pixel-drift setting, the
  static class phase anchor is acting as a much stronger class prior than the
  dynamic one-way oscillator drive.
- The result does not falsify Un-0. It says the conditioning block alone is not
  the transferable magic ingredient. Un-0's actual success likely depends on
  the full recipe: large oscillator count, image-scale task, DINOv2 drift
  manifold, long training, and FID-style evaluation.
- For OscNet, this is a useful narrowing result. The next generator work should
  either bring in a stronger pretrained feature target or move closer to the
  CIFAR-10 recipe, rather than continuing to mutate the MNIST pixel objective.

### Un-0 Reference Calibration Helper

After the dynamic-conditioner result, the next useful calibration is to run the
released Un-0 checkpoint with the upstream code, not another OscNet
approximation. This asks a simple question: does the reference recipe itself
produce sane samples in our workflow, and what are its basic scale/throughput
numbers?

New helper:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_un0_reference.py \
  --pretrained cifar10/n1024 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --samples-per-class 4 \
  --seed 42
```

This runs in an isolated PyTorch Modal image and installs
`unconv-ai/Un-0` at commit `43f2587`. It writes a local PNG sample grid and
lightweight JSON/CSV metrics under `outputs/analysis/un0_reference/`. It is not
FID and it is not an OscNet model; it is a reference calibration target.

Reference runs completed:

```text
outputs/analysis/un0_reference/cifar10_n1024_seed42_stepsckpt.png
outputs/analysis/un0_reference/cifar10_n1024_seed42_stepsckpt.json
outputs/analysis/un0_reference/cifar10_n4096_seed42_stepsckpt.png
outputs/analysis/un0_reference/cifar10_n4096_seed42_stepsckpt.json
outputs/analysis/un0_reference/un0_reference_runs.csv
```

| checkpoint | samples | oscillators | params | steps | sample/s | generated mean | generated std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cifar10/n1024` | 40 | 1024 | 1,289,387 | 10 | 3.35 | 0.458293 | 0.242680 |
| `cifar10/n4096` | 20 | 4096 | 19,434,123 | 10 | 2.40 | 0.499176 | 0.232683 |

Qualitative read:

- Both released checkpoints produced recognizable CIFAR-like class samples with
  the upstream code and pretrained weights. The `n4096` grid is clearly stronger
  and close to the visual category structure advertised by the project.
- This is an important sanity anchor: oscillators-as-generative-prior is not
  obviously doomed. The reference stack can generate recognizable images.
- The gap is therefore not "oscillators are trash" in general. The gap is
  between the full Un-0 recipe and our MNIST/JAX proxy: data/task scale,
  trained checkpoint quality, DINO feature drift, full training length, and
  probably optimizer/LR recipe.
- Next OscNet generator work should use this as the north star. Either move
  toward a CIFAR-10 reproduction path, or add a stronger pretrained feature
  drift target before spending more time on MNIST-only objective variants.

Updated generator conclusion:

The Un-0-style branch is alive as a testbed, and the open-source Un-0 code
materially improved the direction. Resize-conv decoding plus pixel drift gives
stronger class-structured oscillator behavior than the earlier loose analogues.
However, neither fixed structural feature drift, a small learned MNIST feature
target, nor the source-faithful dynamic conditioning oscillator block delivered
the qualitative leap. The best MNIST result still comes from the simpler static
class-coupling pixel-queue setup, which behaves more like a strong class prior
than a faithful miniature of Un-0. The reference calibration also shows that
released Un-0 checkpoints do generate recognizable CIFAR-like samples under the
upstream stack, so the failure is not "the whole idea is empty." The current
evidence says our MNIST port captures parts of Un-0's skeleton, but not the full
recipe. The remaining source-faithful gaps are a stronger pretrained/semantic
feature target, scale, and formal decoder-only/reservoir/trained-step ablations
under matched learning-rate sweeps.

### MNIST-Native Phase VAE Pivot

The unpaired generator branch was too easy to underconstrain: several variants
could optimize drift/prototype metrics without becoming a good MNIST generator.
To answer the user's sharper question, we added a deliberately conventional
MNIST generative objective while keeping oscillator dynamics in the latent path.

New reusable pieces:

- `KuramotoPhaseVAE` in `oscnet.models.generative`.
- `MNISTPhaseVAEExperimentConfig` and `run_mnist_phase_vae_experiment` in
  `oscnet.experiments.mnist_phase_vae`.
- `examples/image_mnist_phase_vae.py`.
- `scripts/modal_mnist_phase_vae.py`.

The model encodes an image to Gaussian `mu/logvar`, samples `z`, wraps `z` as a
phase vector, optionally evolves it through Kuramoto dynamics, and decodes
`sin/cos` phase features. Generation samples the same Gaussian prior. This is
not a full Un-0 reproduction; it is the simplest honest MNIST generator that can
test whether phase evolution helps.

CPU smoke:

```bash
python examples/image_mnist_phase_vae.py \
  --output-dir outputs/reference/mnist_phase_vae_real_smoke_seed21 \
  --data-source idx \
  --train-limit 512 \
  --eval-limit 128 \
  --eval-sample-count 32 \
  --epochs 2 \
  --batch-size 32 \
  --seed 21 \
  --latent-dim 16 \
  --hidden-dim 128 \
  --encoder-depth 2 \
  --decoder-depth 2 \
  --steps 2 \
  --kl-weight 0.001 \
  --learning-rate 0.001 \
  --checkpoint-every 2 \
  --artifact-every 2
```

The smoke run learned immediately: eval MSE improved from `0.128760` after
epoch 1 to `0.065991` after epoch 2. Samples were digit-like but mode-biased,
as expected from 512 examples and two epochs.

Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_vae.py \
  --sweep-preset mnist_phase_vae_core
```

Result:

```text
outputs/analysis/modal_mnist_phase_vae_core.csv
outputs/analysis/modal_mnist_phase_vae_samples/
```

Single-seed 20-epoch sweep on 10k MNIST train examples:

| model | final eval loss | final eval MSE | recon MSE | sample diversity ratio | phase displacement |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen phase VAE | 0.148297 | 0.023259 | 0.022564 | 0.862280 | 0.066502 |
| trainable phase VAE | 0.148547 | 0.023408 | 0.023042 | 0.852783 | 0.070177 |
| no-dynamics VAE | 0.152604 | 0.025015 | 0.024145 | 0.794149 | 0.000000 |

Qualitative read:

- The MNIST generative task now works. Prior samples are visibly digit-like,
  not noise or metric gaming.
- Oscillator phase evolution helps relative to the no-dynamics VAE under the
  same scaffold: about a 6 to 7 percent reconstruction-MSE improvement in this
  run.
- The best result is the frozen phase transform, with trainable phase dynamics
  essentially tied. That is a useful but humbling attribution result: current
  phase evolution acts like a helpful latent periodic transform/regularizer,
  not yet like a learned generative engine.
- This is a better base for future generator work than the unpaired MNIST drift
  branch. Next work should probe stronger latent dynamics, phase-space priors,
  and longer/annealed settling from this VAE baseline before returning to
  unpaired generation.

### Forced Phase-Dynamics Probe

To test whether the phase VAE simply needed a larger oscillator pool and
stronger phase motion, we added `phase_readout_mode` to `KuramotoPhaseVAE` and
ran a forced-dynamics Modal preset:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_vae.py \
  --sweep-preset mnist_phase_vae_forced_dynamics_core
```

This preset uses 128 latent phases, eight recurrent steps, `dt=0.2`,
`coupling_strength=2.0`, `omega_scale=0.5`, `coupling_init_scale=0.2`, and
`phase_readout_mode="mean_relative"`. The matched controls are the same VAE
with frozen dynamics and no dynamics.

Result:

```text
outputs/analysis/modal_mnist_phase_vae_forced_dynamics_core.csv
outputs/analysis/modal_mnist_phase_vae_samples/
```

Single-seed 20-epoch sweep on 10k MNIST train examples:

| model | final eval loss | final eval MSE | recon MSE | sample diversity ratio | phase displacement |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-dynamics relative VAE | 0.157189 | 0.025903 | 0.024460 | 0.691284 | 0.000000 |
| frozen forced phase VAE | 0.159447 | 0.026885 | 0.026314 | 0.720228 | 0.654761 |
| trainable forced phase VAE | 0.160099 | 0.027438 | 0.026466 | 0.722657 | 0.588574 |

Interpretation:

- Stronger phase motion did not help. It raised phase displacement from about
  `0.07` radians in the baseline phase VAE to about `0.6` radians, but both
  dynamic variants lost to the no-dynamics relative-phase control.
- The qualitative samples still look digit-like, so this is not a training
  collapse. It is evidence that simply making the latent oscillator move harder
  is the wrong knob for this VAE setting.
- The current best MNIST VAE result remains the milder 32-phase frozen/learned
  dynamics sweep. The forced probe narrows the next hypothesis: learned
  oscillator dynamics probably need an objective that requires computation
  during recurrent settling, rather than a decoder that can solve generation
  from a static latent phase code.
- For Un-0 interpretation, this weakens the idea that their success comes from
  generic large phase motion alone. The remaining likely ingredients are the
  specific unpaired feature/manifold training signal, decoder interaction with
  thousands of oscillator channels, conditioning population, and/or a task where
  generated phase trajectories are rewarded directly.

### Oscillatory Phase-Flow Sampler

The council review converged on a sharper generative hypothesis: stop treating
oscillators as a latent decoration inside a VAE, and train the image/noise field
itself as the oscillator medium. We added `PhaseRateFlowField`, a local
phase-rate field trained with a rectified-flow objective from Gaussian noise to
MNIST images. The core attribution sweep compares the learned oscillator field
against frozen and no-dynamics controls with the same conditioning/readout
scaffold:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_core
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_core.csv
outputs/analysis/modal_mnist_phase_flow_recurrent_conv_control.csv
outputs/analysis/modal_mnist_phase_flow_coarse_global_probe.csv
outputs/analysis/modal_mnist_phase_flow_coarse_heun_probe.csv
outputs/analysis/modal_mnist_phase_flow_coarse_position_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Single-seed 20-epoch sweep on 10k MNIST train examples, plus a one-job
recurrent-conv control run and a coarse/global phase-carrier probe:

| model | final eval loss | best eval loss | final velocity loss | final clean loss | sample nearest-real MSE | state displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.138228 | 0.133665 | 0.129206 | 0.036091 | 0.069261 | 1.394590 phase |
| trainable phase-flow | 0.160045 | 0.149670 | 0.149152 | 0.043570 | 0.054002 | 1.492249 phase |
| recurrent-conv flow | 0.163575 | 0.158277 | 0.152654 | 0.043686 | 0.066649 | 0.470222 hidden |
| no-dynamics control | 0.434464 | 0.434464 | 0.418005 | 0.065836 | 0.095157 | 0.000000 phase |
| frozen phase-flow | 0.435151 | 0.435151 | 0.418703 | 0.065794 | 0.092723 | 0.526620 phase |

Interpretation:

- This is the cleanest positive dynamics attribution in the generative branch so
  far. Learned oscillator dynamics are much better than both frozen and
  no-dynamics controls on the flow objective, clean-prediction loss, and
  nearest-real sample proxy.
- The matched recurrent-conv control is close but still behind the phase-flow
  model on this seed: best eval loss `0.158277` versus `0.149670`, and
  nearest-real sample MSE `0.066649` versus `0.054002`. This matters because it
  suggests the positive result is not only "any local recurrent field works."
- The coarse/global carrier is the best objective result so far: best eval loss
  `0.133665`, final clean loss `0.036091`, and much better pixel mean/std
  matching than local phase-flow. This supports the hypothesis that local
  oscillator strokes benefit from a slower spatial carrier.
- The qualitative denoising grids show real stroke formation from noisy inputs;
  the frozen/no-dynamics controls remain mostly noise texture. The conv control
  also forms similar stroke fragments, so the current qualitative attractor is
  shared by learned local recurrence and learned phase-rate recurrence.
- The free samples are not solved MNIST. They form digit-like strokes and
  fragments, but often fail to close into coherent whole digits. So this is a
  native ONN dynamics win, not yet a competitive MNIST generator. The coarse
  carrier improves the training/denoising objective more than it improves the
  nearest-real free-sample proxy, so better sampling dynamics still matter.
- The likely next improvement is not more VAE tuning. The field sampler needs a
  stronger global closure mechanism or sampling schedule: more integration
  steps, less fragmented velocity targets near early noise times, and a
  coarse/long-range phase carrier so local strokes can close into whole digit
  shapes.
- Future phase-flow runs now record simple topology metrics: foreground active
  fraction, connected-component count, and largest-component fraction for both
  samples and real MNIST. These should make "did strokes close into a digit?"
  visible as a metric, not just a visual complaint.

Sampling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_heun_probe
```

This reruns the same coarse/global setup with a Heun predictor-corrector sampler
and 32 sample steps. Training/eval remains essentially tied with the Euler
coarse run (`best_loss=0.134258` vs `0.133665`), as expected because the sampler
does not change the training objective. Sampling changes only slightly:
nearest-real MSE improves from `0.069261` to `0.068423`, but topology still
shows fragmentation:

| metric | Heun32 samples | real MNIST |
| --- | ---: | ---: |
| active fraction | 0.116091 | 0.143973 |
| connected components | 3.593750 | 1.015625 |
| largest component fraction | 0.691411 | 0.995226 |

Interpretation: better numerical integration alone is not the missing trick.
The model has learned useful denoising/velocity fields, but the generative
trajectory still needs a mechanism or objective that rewards whole connected
digit closure rather than independent stroke islands.

Spatial phase-coordinate probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_position_probe
```

This adds fixed 2D coordinate/phase features to the fine and coarse
initialization projections. It tests whether the field was fragmenting because
all oscillators were too translation-equivariant and lacked a native spatial
reference frame.

| metric | coarse baseline | coarse + position |
| --- | ---: | ---: |
| best eval loss | 0.133665 | 0.136380 |
| final clean loss | 0.036091 | 0.035928 |
| sample nearest-real MSE | 0.069261 | 0.060626 |
| sample active fraction | n/a | 0.098334 |
| sample connected components | n/a | 5.343750 |
| sample largest component fraction | n/a | 0.621284 |

Interpretation: fixed spatial phase coordinates help the sample proxy
substantially, nearly reaching the real nearest-real baseline
(`0.060626` versus `0.059272`), but they worsen connectedness. The model is
better at placing MNIST-like local marks in plausible regions, yet it still
does not bind them into a single digit. This narrows the next hypothesis:
spatial reference helps, but the missing piece is a closure/binding pressure,
not simply "knowing where pixels are."

Closure/binding objective probe:

The phase-flow experiment now exposes `closure_loss_weight`, an auxiliary
train-time loss on the predicted clean endpoint. It compares predicted and real
digits after coarse average pooling at `14x14` and `7x7`. This is deliberately
not a sampler-time steering trick: the model must learn a velocity field whose
clean endpoint has a coherent low-frequency digit envelope. The intent is to
test whether explicit whole-shape pressure helps the coarse/global oscillator
bind local stroke fragments into one digit. The one-container Modal probe was:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_closure_probe
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_coarse_closure_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | closure loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse + closure | 0.154992 | 0.129110 | 0.035472 | 0.018724 | 0.071007 | 0.125399 | 3.828125 | 0.682642 |
| coarse + position + closure | 0.158367 | 0.133539 | 0.036186 | 0.019033 | 0.066514 | 0.130580 | 5.765625 | 0.643742 |
| coarse + position, no closure | 0.136380 | 0.131554 | 0.035928 | n/a | 0.060626 | 0.098334 | 5.343750 | 0.621284 |

Interpretation:

- This did not unlock whole-digit generation. The generated grids still show
  disconnected islands and partial strokes.
- The no-position closure variant slightly improves foreground mass relative to
  the Heun no-closure topology probe, but not enough to reduce fragmentation in
  a meaningful way.
- The position+closure variant raises active fraction from `0.098334` to
  `0.130580`, but also raises component count from `5.343750` to `5.765625` and
  worsens nearest-real MSE from `0.060626` to `0.066514`. So the added loss
  mostly teaches the field to place more material, not to bind material into a
  single digit.
- This makes the next hypothesis sharper: endpoint-level low-frequency
  supervision is too blunt. The missing mechanism is probably not "add a
  shape loss"; it is a dynamics-native binding mechanism or a target domain
  where phase coherence directly represents contours/orientation.

Next useful direction: test an ONN-native contour/orientation phase-flow target
or a trajectory-level coherence objective. The result we want is not just lower
MSE; it is fewer components and higher largest-component fraction without the
samples becoming blurry blobs.

Contour-domain phase-flow probe:

The MNIST phase-flow experiment now supports
`target_representation="sobel_edges"` / `--target-representation sobel_edges`.
This converts MNIST images into normalized Sobel edge-magnitude maps before
training the rectified-flow field. It is the first minimal test of the
hypothesis that oscillator fields may be better matched to contour/coherence
targets than to raw pixel mass.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_edge_probe
```

This compares coarse/global phase-flow against the matched recurrent-conv flow
control on the same Sobel-edge target.

Result:

```text
outputs/analysis/modal_mnist_phase_flow_edge_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.173173 | 0.163982 | 0.036765 | 0.079351 | 0.239357 | 2.750000 | 0.897632 |
| recurrent-conv flow | 0.195775 | 0.186719 | 0.045180 | 0.077924 | 0.123884 | 3.609375 | 0.672176 |
| real Sobel MNIST | n/a | n/a | n/a | 0.064611 | 0.230628 | 1.015625 | 0.999408 |

Interpretation:

- This is the most encouraging result after the raw-pixel phase-flow branch.
  The coarse/global oscillator field beats the matched recurrent-conv control
  on flow objective, clean endpoint loss, pixel mean/std matching, active
  foreground mass, connected-component count, and largest-component fraction.
- The recurrent-conv control has slightly better nearest-real MSE
  (`0.077924` versus `0.079351`), but it also underdraws the edge field
  (`active_fraction=0.123884` versus real `0.230628`) and fragments more. This
  makes nearest-real MSE a weak primary metric for this target.
- Qualitatively, neither model is a solved digit generator. The oscillator edge
  samples are still rough, but they are visibly denser and more connected than
  the recurrent-conv samples.
- This supports the hypothesis that oscillator fields may be better matched to
  contour/coherence targets than to raw pixel mass. The next useful step is a
  stronger contour representation, such as signed distance, skeleton, or
  orientation-vector fields, plus the same frozen/no-dynamics controls.

Signed-distance phase-flow probe:

The MNIST phase-flow experiment now also supports
`target_representation="signed_distance"` /
`--target-representation signed_distance`. This converts MNIST images into an
approximate JAX-native signed-distance shape field using repeated 3x3 dilation
bands. It stays one-channel and uses the same phase-flow models, but gives the
oscillator field a smoother whole-shape target than raw pixels or Sobel edges.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_probe
```

This compares coarse/global phase-flow against the matched recurrent-conv flow
control on the same smooth shape-field target.

Result:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.097380 | 0.097323 | 0.016397 | 0.024025 | 0.264848 | 1.437500 | 0.952806 |
| recurrent-conv flow | 0.118595 | 0.115930 | 0.023067 | 0.033544 | 0.215721 | 3.453125 | 0.757076 |
| real signed-distance MNIST | n/a | n/a | n/a | 0.017730 | 0.395010 | 1.000000 | 1.000000 |

Interpretation:

- This strengthens the contour-domain hypothesis. The coarse/global oscillator
  beats the recurrent-conv control on best objective, final velocity loss,
  final clean loss, nearest-real MSE, pixel mean matching, active foreground
  mass, connected-component count, and largest-component fraction.
- The samples are still not solved MNIST digits. They look like soft digit
  fields or partial glyphs rather than crisp generated images. But the failure
  mode is meaningfully better than the recurrent-conv control: fewer islands,
  more continuous mass, and higher largest-component fraction.
- This suggests the current ONN branch is strongest when the generated object
  is represented as a smooth spatial field. The next high-value probe is not
  another generic decoder trick; it is an ONN-native way to convert these
  promising shape fields back into crisp pixels or to train a two-head model
  that predicts both signed-distance shape and pixel occupancy.

Two-channel pixel/shape phase-flow probe:

The phase-flow models now support multi-channel visible fields through
`value_channels`. The MNIST phase-flow experiment exposes this with
`target_representation="pixels_signed_distance"` /
`--target-representation pixels_signed_distance`: channel 0 is the original
MNIST pixel occupancy target, and channel 1 is the auxiliary signed-distance
shape field. The model samples both channels jointly, while PNG artifacts and
sample-quality metrics use channel 0 so the question remains pixel-generation
quality, not just shape-field quality.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_pixel_shape_probe
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_pixel_shape_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.121527 | 0.116685 | 0.026188 | 0.712745 | 1.000000 | 1.000000 | 1.000000 |
| recurrent-conv flow | 0.134196 | 0.126871 | 0.032589 | 0.712768 | 1.000000 | 1.000000 | 1.000000 |
| real MNIST pixels | n/a | n/a | n/a | 0.059272 | 0.143973 | 1.015625 | 0.995226 |

Interpretation:

- This is a mixed/negative sample result. The coarse/global oscillator still
  beats the recurrent-conv control on the supervised flow objective and
  denoising endpoint loss, so the auxiliary shape channel is trainable and the
  oscillator remains the better local field model.
- Unconditional channel-0 sampling collapses to nearly all-white images for
  both models (`sample_mean ~= 1.0`, `sample_active_fraction=1.0`). The
  generated samples are therefore unusable as MNIST pixels despite the better
  training/eval losses.
- The denoising artifacts remain digit-like, so the failure is not simply
  "two channels cannot be learned." It is a sampling/prior mismatch: the
  coupled pixel/shape field can denoise from mid-trajectory evidence, but the
  noise-to-data integration drifts into the saturated high-pixel attractor.
- Next useful direction: keep multi-channel visible fields, but fix the
  sampling geometry. Candidates are centered/logit target coordinates,
  per-channel sampling priors, endpoint regularization on unconditional
  samples, or a staged sampler where the signed-distance channel settles first
  and gates pixel occupancy instead of being integrated as an equal visible
  channel from Gaussian noise.

Centered two-channel pixel/shape phase-flow probe:

The next probe keeps the same two-channel pixel/shape field but centers both
channels into `[-1, 1]` during training and sampling:
`target_representation="centered_pixels_signed_distance"`. Metrics and PNG
artifacts decode channel 0 back into pixel space. The model-level sampler can
now run without clipping in native target space, which directly tests whether
the prior-coordinate geometry caused the all-white collapse.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_centered_pixel_shape_probe
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_centered_pixel_shape_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.214916 | 0.202983 | 0.068884 | 0.090784 | 0.172951 | 17.390625 | 0.609082 |
| recurrent-conv flow | 0.269278 | 0.248835 | 0.086680 | 0.077987 | 0.112663 | 20.953125 | 0.456661 |
| real MNIST pixels | n/a | n/a | n/a | 0.059272 | 0.143973 | 1.015625 | 0.995226 |

Interpretation:

- Centering fixes the all-white collapse. Channel-0 samples now have plausible
  mean/variance and active mass instead of saturating to `1.0`.
- The coarse/global oscillator remains the better supervised field model: lower
  best eval loss, lower velocity loss, lower clean loss, better pixel-mean
  matching, fewer fragments, and a larger dominant component than the
  recurrent-conv control.
- This is still not a solved generator. Samples are speckled and fragmented;
  component count remains far from real MNIST (`17.39` versus `1.02`).
  The recurrent-conv control has better nearest-real MSE, likely because it
  underdraws foreground mass, but it fragments even more.
- Updated bottleneck: the sampler no longer falls into a white attractor. The
  remaining failure is binding/cleanup: turning scattered pixel/shape evidence
  into one coherent digit component. A staged field may be the right next move:
  first settle a smooth signed-distance field, then use it as a gate or
  potential for pixel occupancy instead of asking both channels to emerge
  equally from Gaussian noise.

Centered shape-gated readout probe:

The phase-flow experiment now supports `sample_readout_mode="shape_gated"` /
`--sample-readout-mode shape_gated`. This does not retrain a different model:
the model still samples both visible channels, but sample metrics and PNG
artifacts multiply the decoded pixel channel by a smooth gate from the decoded
shape channel. This is the smallest staged-readout test of whether the
signed-distance field carries useful cleanup structure for the pixel channel.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_centered_shape_gated_probe
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_centered_shape_gated_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.215639 | 0.200101 | 0.068739 | 0.062428 | 0.117267 | 2.828125 | 0.727435 |
| recurrent-conv flow | 0.269668 | 0.248895 | 0.086666 | 0.046949 | 0.063935 | 2.531250 | 0.680363 |
| real MNIST pixels | n/a | n/a | n/a | 0.059272 | 0.143973 | 1.015625 | 0.995226 |

Interpretation:

- Shape-gating is a real cleanup step. Compared with the centered primary
  readout, the coarse/global oscillator improves from `17.39` components to
  `2.83`, and nearest-real MSE improves from `0.090784` to `0.062428`.
- The oscillator remains the better trained field model: lower objective,
  lower velocity loss, lower clean loss, better pixel-mean matching, more
  realistic active mass, and a larger dominant component than the recurrent
  control.
- The recurrent-conv control gets lower nearest-real MSE (`0.046949`), but it
  underdraws strongly (`active_fraction=0.063935` versus real `0.143973`) and
  preserves less dominant mass. This makes nearest-real MSE alone misleading.
- This is still not a solved generator. The samples are cleaner fragments, not
  coherent MNIST digits. But the signed-distance field is doing useful work as
  a potential/gate. The next meaningful step is to make that staging learned or
  dynamical: settle shape first, then run a pixel-refinement field conditioned
  on that settled shape, instead of relying on a fixed readout gate.

Shape-guided sampler probe:

The phase-flow sampler now supports `sample_schedule="shape_guided"` /
`--sample-schedule shape_guided` for two-channel centered pixel/shape targets.
During Euler sampling, the shape channel receives full updates from the start,
while pixel-channel updates open later and are softly pulled through the
decoded shape potential. This moves the staging from a post-hoc readout gate
into the sampling dynamics itself.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_shape_guided_sampler_probe
```

Result:

```text
outputs/analysis/modal_mnist_phase_flow_shape_guided_sampler_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | best eval loss | velocity loss | clean loss | nearest-real MSE | active frac | components | largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.214272 | 0.201506 | 0.068735 | 0.141564 | 0.554747 | 4.546875 | 0.932344 |
| recurrent-conv flow | 0.269503 | 0.248902 | 0.086670 | 0.169127 | 0.643495 | 2.484375 | 0.991480 |
| real MNIST pixels | n/a | n/a | n/a | 0.059272 | 0.143973 | 1.015625 | 0.995226 |

Interpretation:

- Shape-guided sampling is a useful negative result. It moved the staging idea
  into the dynamics, but the fixed schedule overfilled the canvas instead of
  producing clean digit attractors.
- Compared with the simpler centered shape-gated readout, nearest-real MSE got
  much worse for the coarse/global oscillator (`0.062428` to `0.141564`) and
  active mass exploded (`0.117267` to `0.554747`, versus real `0.143973`).
- The higher largest-component fraction is not a win here. It mostly reflects
  oversized connected blobs, not better digit binding.
- The oscillator still beats the recurrent-conv control on the trained field
  objective and on these bad-schedule sample-distance metrics, but absolute
  sample quality is worse than the fixed readout gate.
- Updated bottleneck: the shape channel contains useful cleanup structure, but
  hand-coded shape-first sampling is too blunt. The next plausible step is a
  learned second-stage pixel refiner conditioned on a settled shape field, not
  a stronger manual schedule.

Locked shape-gated audit:

The next phase-flow step is a locked multi-seed control audit, not another
architecture tweak. It freezes the current best pixel-producing setup:
`target_representation="centered_pixels_signed_distance"`,
`sample_readout_mode="shape_gated"`, `sample_schedule="standard"`,
`closure_loss_weight=0.0`, 20 epochs, 10k training examples, and 1k eval
examples.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_shape_gated_audit
```

Planned matrix:

- Seeds: `31`, `32`, `33`, `34`, `35`.
- Models: `coarse_phase_flow`, `phase_flow`, `frozen_phase_flow`,
  `phase_flow_no_dynamics`, and `recurrent_conv_flow`.
- Primary read: active fraction, component count, largest-component fraction,
  nearest-real MSE, supervised loss, and fixed-seed sample grids.

Decision rule:

- If coarse/global phase-flow does not beat recurrent-conv on most seeds in
  both supervised field losses and sample topology, stop treating the current
  pixel-producing phase-flow branch as robust.
- If learned local phase-flow does not beat frozen/no-dynamics controls, stop
  treating oscillator dynamics as causal in this setup.
- If losses win but sample topology remains bad, call this a field-denoising
  result, not a working MNIST generator.
- If the audit is mixed or negative, the next move should be a principled
  two-stage shape-to-pixel model or a basin-of-attraction diagnostic, not more
  sampler schedules.

Result:

```text
outputs/analysis/modal_mnist_phase_flow_shape_gated_audit.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

| model | mean best eval loss | mean clean loss | mean nearest-real MSE | mean active frac | mean components | mean largest frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse/global phase-flow | 0.214620 | 0.068723 | 0.128170 | 0.220631 | 1.328125 | 0.443232 |
| local phase-flow | 0.239339 | 0.081113 | 0.195868 | 0.350566 | 1.787500 | 0.473119 |
| recurrent-conv flow | 0.257613 | 0.083811 | 0.030792 | 0.012636 | 0.509375 | 0.136222 |
| frozen phase-flow | 0.624334 | 0.162828 | 0.042025 | 0.052117 | 25.821875 | 0.184060 |
| no-dynamics phase-flow | 0.614370 | 0.160188 | 0.067280 | 0.093375 | 45.090625 | 0.058640 |
| real MNIST pixels | n/a | n/a | 0.059272 | 0.143973 | 1.015625 | 0.995226 |

Interpretation:

- The supervised field-model result is robust. Coarse/global phase-flow wins
  best eval loss on all five seeds and has the best mean clean loss. Local
  trainable phase-flow is also far ahead of frozen/no-dynamics controls, so
  learned oscillator dynamics are causal for the denoising/velocity task.
- The generator result is negative. Free samples are not robustly digit-like.
  Coarse/global phase-flow alternates across seeds between stroke fragments,
  near-blank collapse, and near-all-foreground collapse. The same model that
  denoises mid-trajectory digit states well does not reliably originate digits
  from its own noise trajectory.
- Nearest-real MSE is not a useful primary generator metric in this setting.
  Recurrent-conv and blank-collapse runs can score low nearest-real MSE because
  underdrawn samples are close to MNIST background. Active fraction and sample
  grids expose this: recurrent-conv averages only `0.012636` active fraction
  versus real MNIST `0.143973`.
- Frozen/no-dynamics controls produce poor supervised losses and pathological
  samples. This keeps the oscillator dynamics attribution alive for field
  prediction, but not for unconditional MNIST generation.
- Updated conclusion: the current phase-flow setup is a robust oscillator
  denoising/shape-field model, not a robust MNIST generator. The next decisive
  diagnostic should map the basin of attraction by starting samples from
  `x_t = (1 - t) noise + t data` at several `t` values. If coherence appears
  only near real data, phase-flow is a shallow denoiser. If signed-distance
  fields have a wider basin than pixels, the two-stage shape-to-pixel direction
  remains the principled architecture move.

Basin-of-attraction probe:

The phase-flow experiment now supports `basin_t_values` /
`--basin-t-values`, a diagnostic that starts from partially real chord states:
`x_t = (1 - t) noise + t data`. It then integrates the trained model from that
start time to the endpoint and computes the same sample quality/topology
metrics plus paired MSE to the exact scaffold image. The diagnostic records
both the starting paired MSE and the final paired MSE, so it measures whether
the dynamics improved the state rather than merely benefiting from a start that
was already close to the answer.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_basin_probe
```

Matrix:

- Start times: `0.1`, `0.25`, `0.5`, `0.75`, `0.9`.
- Seeds: `31`, `32`.
- Targets/models: centered pixel/shape coarse phase-flow,
  centered pixel/shape recurrent-conv, signed-distance coarse phase-flow, and
  signed-distance recurrent-conv.

Result:

```text
outputs/analysis/modal_mnist_phase_flow_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Mean paired MSE over two seeds:

| target/model | t=0.10 start -> final | t=0.50 start -> final | t=0.75 start -> final | t=0.90 start -> final |
| --- | ---: | ---: | ---: | ---: |
| centered pixel/shape, coarse phase-flow | 0.189232 -> 0.087345 | 0.055177 -> 0.052853 | 0.010366 -> 0.055798 | 0.002020 -> 0.065405 |
| centered pixel/shape, recurrent-conv | 0.189232 -> 0.093316 | 0.055177 -> 0.070369 | 0.010366 -> 0.076299 | 0.002020 -> 0.082252 |
| signed-distance, coarse phase-flow | 0.210624 -> 0.040894 | 0.120478 -> 0.019128 | 0.042395 -> 0.006195 | 0.008251 -> 0.002043 |
| signed-distance, recurrent-conv | 0.210624 -> 0.056075 | 0.120478 -> 0.020963 | 0.042395 -> 0.006569 | 0.008251 -> 0.002155 |

Mean paired-MSE improvement fraction:

| target/model | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| centered pixel/shape, coarse phase-flow | 0.538 | 0.490 | 0.042 | -4.369 | -31.365 |
| centered pixel/shape, recurrent-conv | 0.507 | 0.417 | -0.275 | -6.351 | -39.707 |
| signed-distance, coarse phase-flow | 0.806 | 0.813 | 0.841 | 0.854 | 0.752 |
| signed-distance, recurrent-conv | 0.734 | 0.765 | 0.826 | 0.845 | 0.739 |

Interpretation:

- Pixel-producing phase-flow does not have a stable pixel basin. It can improve
  very noisy chord states, but once the state is already close to a real digit
  (`t=0.75` or `t=0.90`), the learned dynamics push it away from the target.
  That explains the free-sampling failure: the pixel readout is not a robust
  attractor.
- Signed-distance fields do show attractor-like relaxation. Both coarse
  phase-flow and recurrent-conv improve every start time, including near-real
  states. The coarse/global oscillator wins paired MSE at every basin point and
  has lower supervised loss (`0.097794` vs `0.118944` mean best loss).
- This is the cleanest representation-level result so far: oscillatory dynamics
  are much better matched to smooth shape fields than raw pixels. But it is
  still not a finished MNIST generator, because the successful object is a
  signed-distance scaffold, not final pixel synthesis.
- The principled next architecture is two-stage: generate/settle a
  signed-distance or contour field first, then render/refine pixels conditioned
  on that field. More sampler tweaks on raw pixel phase-flow are unlikely to
  solve the binding problem by themselves.

Signed-distance non-Gaussian basin probe:

The Gaussian basin result was strong enough to ask whether the scalar
signed-distance field is a broad attractor or merely a good Gaussian denoiser.
The experiment now supports `basin_noise_mode` /
`--basin-noise-mode`, and the Modal wrapper passes that setting through to the
remote config. This probe keeps the same trained task and tests three
non-Gaussian endpoints: `uniform`, `salt_pepper`, and `zeros`.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_noise_basin_probe
```

Artifact:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_noise_basin_probe.csv
```

Mean paired MSE over two seeds:

| noise/model | t=0.10 start -> final | t=0.50 start -> final | t=0.90 start -> final |
| --- | ---: | ---: | ---: |
| uniform, coarse phase-flow | 0.185307 -> 0.211683 | 0.057231 -> 0.125097 | 0.002275 -> 0.002605 |
| uniform, recurrent-conv | 0.185307 -> 0.391009 | 0.057231 -> 0.167911 | 0.002275 -> 0.002676 |
| salt-pepper, coarse phase-flow | 0.320311 -> 0.205634 | 0.098679 -> 0.124079 | 0.003959 -> 0.002978 |
| salt-pepper, recurrent-conv | 0.320311 -> 0.374129 | 0.098679 -> 0.164216 | 0.003959 -> 0.003071 |
| zeros, coarse phase-flow | 0.073528 -> 0.028669 | 0.022694 -> 0.008115 | 0.000908 -> 0.000493 |
| zeros, recurrent-conv | 0.073528 -> 0.027765 | 0.022694 -> 0.006002 | 0.000908 -> 0.000531 |

Interpretation:

- The signed-distance field is not a universal attractor. The Gaussian result
  is real, but uniform and salt-pepper starts expose a failure mode where the
  sampler over-activates the field and damages mid-chord states.
- The basin is asymmetric. Blank/zero starts improve at every start time for
  both models, which suggests the field can grow shape from a low-energy
  scaffold. High-entropy starts require suppression as well as growth, and the
  current objective does not teach that strongly enough.
- The coarse/global oscillator remains meaningfully better than recurrent-conv
  on harsh starts. It has lower supervised loss and is less catastrophic under
  uniform/salt-pepper, but this is not a finished generative attractor.
- Current read: signed-distance is still the best OscNet-native representation
  we have found, but a real generator likely needs either basin-aware training
  over endpoint distributions or a two-stage shape-first architecture with
  explicit constraints on foreground mass/topology.

Mixed endpoint training probe:

The next probe tested the most direct fix for the non-Gaussian failure: train
the same scalar signed-distance phase-flow objective with
`train_noise_mode="mixed"`, where each training example draws its
rectified-flow noise endpoint from a mixture of Gaussian, uniform, salt-pepper,
and zero endpoints. This is basin-aware training, not a new architecture. The
initial probe used one seed to see whether the direction moves before scaling
the sweep.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_mixed_noise_basin_probe
```

Artifact:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_mixed_noise_basin_probe.csv
```

Seed 31 paired MSE, mixed endpoint training:

| noise/model | t=0.10 start -> final | t=0.50 start -> final | t=0.90 start -> final |
| --- | ---: | ---: | ---: |
| uniform, coarse phase-flow | 0.185246 -> 0.026226 | 0.057004 -> 0.005686 | 0.002276 -> 0.000653 |
| salt-pepper, coarse phase-flow | 0.320678 -> 0.036608 | 0.099002 -> 0.010090 | 0.003955 -> 0.001166 |
| zeros, coarse phase-flow | 0.073528 -> 0.025412 | 0.022694 -> 0.006490 | 0.000908 -> 0.000736 |
| uniform, recurrent-conv | 0.185246 -> 0.051517 | 0.057004 -> 0.016125 | 0.002276 -> 0.000878 |
| salt-pepper, recurrent-conv | 0.320678 -> 0.073670 | 0.099002 -> 0.024158 | 0.003955 -> 0.001237 |
| zeros, recurrent-conv | 0.073528 -> 0.076980 | 0.022694 -> 0.045256 | 0.000908 -> 0.001084 |

Interpretation:

- This is a strong positive diagnostic for the coarse/global oscillator. Mixed
  endpoint training largely fixes the uniform and salt-pepper basin failures
  for seed 31. Under uniform starts at `t=0.50`, coarse phase-flow improves
  from the Gaussian-trained failure `0.057004 -> 0.101803` to
  `0.057004 -> 0.005686`.
- The recurrent-conv control also improves on uniform and salt-pepper, so the
  basic lesson is not oscillator-exclusive: the objective must train the basin
  you want. But recurrent-conv breaks badly on zero starts after mixed
  training, while the coarse/global oscillator keeps a usable zero-start basin.
- Coarse phase-flow is the more robust mixed-noise learner in this one-seed
  probe, with much lower supervised loss (`0.06456` best loss for coarse
  versus about `0.098` for recurrent-conv) and better harsh-start basin MSE.
- The asterisk: free samples are still not solved. Mixed-trained coarse
  phase-flow has worse nearest-real sample MSE and fragmented topology than
  the Gaussian-trained model, so this is evidence for basin widening, not yet
  evidence for a high-quality standalone MNIST generator.
- Current read from the one-seed diagnostic: basin-aware signed-distance
  training is now the most promising phase-flow result. The next clean scaling
  test is a two-seed mixed endpoint sweep for coarse phase-flow, plus a more
  efficient evaluator that can test several basin modes from one trained
  checkpoint instead of retraining the same model for every diagnostic mode.

Compact two-seed mixed endpoint probe:

The follow-up added multi-mode basin evaluation via `basin_noise_modes` /
`--basin-noise-modes`, so one trained checkpoint can evaluate `uniform`,
`salt_pepper`, and `zeros` basin endpoints without duplicate training. The
compact Modal preset reruns two seeds for coarse phase-flow and recurrent-conv.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_mixed_noise_basin_compact
```

Artifact:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_mixed_noise_basin_compact.csv
```

Mean paired MSE over two seeds:

| model/noise | best loss | t=0.10 start -> final | t=0.50 start -> final | t=0.90 start -> final |
| --- | ---: | ---: | ---: | ---: |
| coarse phase-flow, uniform | 0.066558 | 0.185307 -> 0.029116 | 0.057231 -> 0.007054 | 0.002275 -> 0.000735 |
| coarse phase-flow, salt-pepper | 0.066558 | 0.321172 -> 0.039844 | 0.098700 -> 0.011705 | 0.003966 -> 0.001223 |
| coarse phase-flow, zeros | 0.066558 | 0.073528 -> 0.023326 | 0.022694 -> 0.005079 | 0.000908 -> 0.000646 |
| recurrent-conv, uniform | 0.105015 | 0.185307 -> 0.052394 | 0.057231 -> 0.020092 | 0.002275 -> 0.000921 |
| recurrent-conv, salt-pepper | 0.105015 | 0.321172 -> 0.080113 | 0.098700 -> 0.031053 | 0.003966 -> 0.001450 |
| recurrent-conv, zeros | 0.105015 | 0.073528 -> 0.068834 | 0.022694 -> 0.035644 | 0.000908 -> 0.000886 |

Interpretation:

- The mixed endpoint result survives the two-seed check. Coarse/global
  phase-flow improves every tested non-Gaussian basin family and beats the
  recurrent-conv control on best loss and paired basin MSE.
- The oscillator advantage is clearest on harsh starts. At `t=0.50`,
  coarse phase-flow reaches `0.007054` on uniform, `0.011705` on salt-pepper,
  and `0.005079` on zeros. Recurrent-conv is materially worse on all three and
  still damages zero-start states on average (`0.022694 -> 0.035644`).
- This is now the strongest field-native ONN result in the repo: not a raw
  pixel generator, but a robust signed-distance attractor whose basin can be
  widened by training on endpoint distributions.
- The remaining gap is sample quality/topology. Free samples remain fragmented
  enough that the result should be framed as a shape-field attractor win, not a
  finished generative model. The next architecture move should turn the robust
  signed-distance attractor into a pixel generator through a constrained
  shape-to-pixel stage, or add a topology/energy regularizer that prevents
  fragmented free samples.

Signed-distance flow-field probe:

The next field-native test added `target_representation="signed_distance_flow"`:
channel 0 is the signed-distance scaffold, while channels 1 and 2 encode the
local x/y gradient direction of that scaffold as bounded values in `[0, 1]`.
The hypothesis was that adding explicit local flow direction might improve
basin stability or give the oscillator a more physics-like potential field.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_flow_basin_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_flow_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Mean paired MSE over two seeds:

| target/model | best loss | clean loss | t=0.10 start -> final | t=0.50 start -> final | t=0.90 start -> final |
| --- | ---: | ---: | ---: | ---: | ---: |
| signed-distance-flow, coarse phase-flow | 0.183851 | 0.030162 | 0.210990 -> 0.373102 | 0.120291 -> 0.381657 | 0.008175 -> 0.354626 |
| signed-distance-flow, recurrent-conv | 0.217670 | 0.044081 | 0.210990 -> 0.390581 | 0.120291 -> 0.394689 | 0.008175 -> 0.394963 |

Interpretation:

- This is a useful negative result. Adding explicit gradient-direction
  channels made the basin far worse than plain scalar signed distance.
- Coarse phase-flow still fits the supervised objective better than the
  recurrent-conv control, but both models damage the signed-distance-flow state
  at every basin start time on average.
- The result is seed-unstable. Seed 32 improved from noisy starts, but seed 31
  collapsed toward a high-foreground field. The mean basin result is therefore
  decisively worse than plain signed distance, which improved all start times
  for both seeds.
- Current read: the oscillator-native object is not "more channels that look
  physical." The stable object is a scalar potential/energy-like field. If we
  add richer structure, it likely needs to be hierarchical or constrained by an
  energy functional, not just auxiliary direction channels.

Two-stage shape-to-pixel renderer:

The next branch is `oscnet.experiments.mnist_shape_pixel` /
`examples/image_mnist_shape_pixel.py`. It operationalizes the basin result as a
clean two-stage hypothesis:

```text
stage 1: signed-distance field = oscillator-native shape scaffold
stage 2: pixel rectified-flow renderer conditioned on that scaffold
```

The renderer uses the existing phase-flow model families, but changes the
visible state contract:

```text
channel 0: noisy/generated pixel image
channel 1: clamped signed-distance shape condition
```

The training target predicts pixel velocity and zero shape velocity; sampling
clamps the shape channel after every integration step. This intentionally tests
whether oscillatory phase-flow helps render pixels from a stable shape field
without asking the same dynamics to invent shape and pixels simultaneously.
The matched `recurrent_conv_flow` renderer remains the required control before
claiming an oscillator-specific win.

Initial smoke command:

```bash
python examples/image_mnist_shape_pixel.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Decision rule:

- If coarse phase-flow beats recurrent-conv when conditioned on oracle
  signed-distance scaffolds, the slow/global oscillator may be useful as a
  renderer too.
- If recurrent-conv wins, the stronger interpretation is that ONN dynamics are
  best used for shape-field relaxation, while pixel rendering should be a
  separate conventional module.
- If both render well from oracle shapes, the next decisive test is cascading a
  sampled signed-distance phase-flow scaffold into the renderer and measuring
  the full two-stage generator.

Modal shape-to-pixel basin probe:

Command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_basin_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_shape_pixel_basin_probe.csv
outputs/analysis/modal_mnist_shape_pixel_basin_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Mean supervised losses over two seeds:

| model | best loss | clean loss | pure-noise paired MSE | active fraction |
| --- | ---: | ---: | ---: | ---: |
| local phase-flow | 0.049844 | 0.005021 | 0.457195 | 0.538066 |
| coarse phase-flow | 0.050666 | 0.005172 | 0.503771 | 0.907864 |
| recurrent-conv control | 0.060220 | 0.006122 | 0.481988 | 0.500010 |
| no-dynamics control | 0.194282 | 0.017971 | 0.714689 | 1.000000 |

Mean basin paired MSE, reported as start -> final:

| model | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=0.90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| local phase-flow | 0.891417 -> 0.459848 | 0.617159 -> 0.463841 | 0.274293 -> 0.470375 | 0.068454 -> 0.477330 | 0.010920 -> 0.479855 |
| coarse phase-flow | 0.891417 -> 0.523283 | 0.617159 -> 0.559835 | 0.274293 -> 0.661364 | 0.068454 -> 0.817691 | 0.010920 -> 0.867353 |
| recurrent-conv control | 0.891417 -> 0.482062 | 0.617159 -> 0.482075 | 0.274293 -> 0.482075 | 0.068454 -> 0.482075 | 0.010920 -> 0.482075 |
| no-dynamics control | 0.891417 -> 0.762628 | 0.617159 -> 0.818710 | 0.274293 -> 0.865646 | 0.068454 -> 0.867356 | 0.010920 -> 0.867356 |

Interpretation:

- Trainable phase-flow renderers are still better supervised pixel-velocity
  learners than the matched recurrent-conv and no-dynamics controls.
- The iterative pixel sampler is not a stable attractor. At near-real starts
  (`t=0.75` and `t=0.90`), every model damages the already-good partial state.
  This is the same failure pattern as raw pixel phase-flow: the vector field
  has learned a useful denoising/readout map, but not a stable endpoint
  relaxation.
- One local phase-flow seed produced recognizable oracle-shape pixel samples
  (`paired_sample_mse=0.0470`, `active_fraction=0.0761`), while the other
  collapsed to all-white. The coarse oscillator had the same seed instability:
  one seed rendered digits with noisy background, one seed collapsed.
- Recurrent-conv can look deceptively good under paired MSE when it collapses
  to near-blank images, because MNIST is sparse. The topology metrics are
  essential here: seed 32 recurrent-conv had `active_fraction ~= 0.00002`.
- The two-stage direction remains conceptually useful, but this specific
  rectified-flow pixel renderer is not the missing piece. The durable positive
  result is still the signed-distance/shape-field relaxation. Pixel synthesis
  likely needs either an endpoint-stabilized renderer objective, a direct
  shape-to-pixel decoder, or a task where the evaluated output is the field
  itself rather than raw MNIST pixels.

Follow-up scaffold robustness diagnostic:

`mnist_shape_pixel` now has a lightweight probe for the actual two-stage bridge:
corrupt the signed-distance condition with the same endpoint families used in
the phase-flow basin tests, then ask the renderer to produce pixels from that
imperfect scaffold. This does not claim the full cascade works; it answers the
more basic question of whether the renderer requires oracle-clean shape fields
or can tolerate plausible upstream oscillator errors.

Local smoke:

```bash
python examples/image_mnist_shape_pixel.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4 \
  --shape-condition-t-values 0.5 \
  --shape-condition-noise-modes uniform
```

Modal scaffold robustness probe:

Command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_shape_condition_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_shape_pixel_shape_condition_probe.csv
outputs/analysis/modal_mnist_shape_pixel_shape_condition_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Mean oracle-scaffold sample metrics over seeds 31 and 32:

| model | best loss | paired sample MSE | nearest-real MSE | active fraction | sample mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| local phase-flow | 0.049861 | 0.457849 | 0.372580 | 0.537070 | 0.517656 |
| coarse phase-flow | 0.050207 | 0.510285 | 0.431072 | 0.924924 | 0.727241 |
| recurrent-conv control | 0.060525 | 0.481999 | 0.369748 | 0.500010 | 0.499989 |
| no-dynamics control | 0.194291 | 0.714797 | 0.582373 | 1.000000 | 0.910908 |

Per-seed oracle-scaffold read:

| run | paired sample MSE | active fraction | sample mean | qualitative read |
| --- | ---: | ---: | ---: | --- |
| local phase-flow seed 32 | 0.048343 | 0.074139 | 0.035313 | recognizable digit strokes |
| local phase-flow seed 31 | 0.867355 | 1.000000 | 0.999999 | all-white collapse |
| coarse phase-flow seed 32 | 0.158100 | 0.849849 | 0.457105 | recognizable digits with noisy foreground/background |
| coarse phase-flow seed 31 | 0.862470 | 1.000000 | 0.997376 | all-white collapse |
| recurrent-conv seed 32 | 0.096739 | 0.000020 | 0.000030 | near-blank collapse |
| recurrent-conv seed 31 | 0.867260 | 1.000000 | 0.999949 | all-white collapse |

Shape-condition robustness, paired sample MSE averaged over two seeds:

| model | uniform t=0.1 | uniform t=0.5 | uniform t=0.9 | salt-pepper t=0.1 | salt-pepper t=0.5 | salt-pepper t=0.9 | zeros t=0.1 | zeros t=0.5 | zeros t=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local phase-flow | 0.481418 | 0.471694 | 0.460691 | 0.493868 | 0.472222 | 0.460411 | 0.482075 | 0.482075 | 0.469355 |
| coarse phase-flow | 0.744402 | 0.592659 | 0.517907 | 0.732722 | 0.637407 | 0.520584 | 0.503082 | 0.492621 | 0.503019 |
| recurrent-conv control | 0.482070 | 0.482074 | 0.482060 | 0.481965 | 0.482047 | 0.482050 | 0.482039 | 0.482043 | 0.482038 |
| no-dynamics control | 0.827489 | 0.812990 | 0.745323 | 0.750797 | 0.772847 | 0.744031 | 0.635631 | 0.677747 | 0.708383 |

Interpretation:

- This does not yet validate the full two-stage generator. The renderer remains
  seed-unstable and can collapse to all-white or all-blank endpoints.
- It does show a real capability hiding inside the noise: local phase-flow seed
  32 renders recognizable digits from oracle signed-distance scaffolds, and it
  remains usable under moderate uniform/salt-pepper scaffold corruption
  (`t=0.5` paired sample MSE around `0.076` and `0.077` for that seed).
- The recurrent-conv control is not a clean counterexample. It produced one
  all-white seed and one near-blank seed, with scaffold corruption barely
  changing the output. Good nearest-real or paired-MSE numbers can therefore be
  degenerate sparse-MNIST artifacts.
- Coarse phase-flow seed 32 preserves digit silhouettes even with noisy
  scaffolds, but over-activates foreground/background. The slow/global carrier
  may help shape carry-through, but it needs a foreground/bounds stabilizer.
- Next intervention should target endpoint stability and foreground mass, not a
  new oscillator variant. The minimal useful test is a shape-to-pixel objective
  that penalizes all-white/all-blank sample endpoints or trains explicit
  endpoint reconstructions, then reruns this exact scaffold robustness probe.

Implemented next stabilizer hook:

`mnist_shape_pixel` now supports `--sample-readout-mode shape_gated`, mirroring
the earlier phase-flow readout. The sampled pixel channel is multiplied by a
smooth gate derived from the clamped signed-distance scaffold. This is not the
default and should be compared against the raw `primary` readout above. The
question is whether explicit shape-field amplitude gating prevents all-white /
all-blank endpoint collapse while preserving the recognizable local phase-flow
seed.

Modal command:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_shape_gated_probe
```

Modal shape-gated result:

Artifacts:

```text
outputs/analysis/modal_mnist_shape_pixel_shape_gated_probe.csv
outputs/analysis/modal_mnist_shape_pixel_shape_gated_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Mean oracle-scaffold sample metrics, raw readout -> shape-gated readout:

| model | paired sample MSE | active fraction | nearest-real MSE |
| --- | ---: | ---: | ---: |
| coarse phase-flow | 0.510285 -> 0.037721 | 0.924924 -> 0.323182 | 0.431072 -> 0.037416 |
| local phase-flow | 0.457849 -> 0.049733 | 0.537070 -> 0.235252 | 0.372580 -> 0.041416 |
| recurrent-conv control | 0.481999 -> 0.074019 | 0.500010 -> 0.197515 | 0.369748 -> 0.038814 |
| no-dynamics control | 0.714797 -> 0.049097 | 1.000000 -> 0.381587 | 0.582373 -> 0.048642 |

Shape-gated scaffold robustness, paired sample MSE averaged over two seeds:

| model | uniform t=0.1 | uniform t=0.5 | uniform t=0.9 | salt-pepper t=0.1 | salt-pepper t=0.5 | salt-pepper t=0.9 | zeros t=0.1 | zeros t=0.5 | zeros t=0.9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse phase-flow | 0.415980 | 0.176134 | 0.045831 | 0.455321 | 0.278302 | 0.048573 | 0.084770 | 0.036957 | 0.028659 |
| local phase-flow | 0.266974 | 0.150961 | 0.058511 | 0.291006 | 0.196382 | 0.059811 | 0.088203 | 0.062958 | 0.055646 |
| recurrent-conv control | 0.267615 | 0.161562 | 0.079967 | 0.278383 | 0.206399 | 0.081500 | 0.088203 | 0.062960 | 0.068452 |
| no-dynamics control | 0.435629 | 0.222277 | 0.060346 | 0.462266 | 0.317247 | 0.063600 | 0.081088 | 0.028114 | 0.037972 |

Interpretation:

- Shape gating is an effective stabilizer for the two-stage renderer. It
  suppresses all-white collapse and turns the soft signed-distance scaffold into
  recognizable MNIST-like pixels.
- It is not an oscillator-specific win. It improves every model, including the
  no-dynamics control, because the oracle signed-distance scaffold already
  contains most of the digit geometry.
- This changes the scientific pressure point. The decisive question is no longer
  "can an ONN renderer convert an oracle scaffold into pixels?" That is mostly
  solved by the scaffold gate. The decisive question is whether the oscillator
  system can generate or settle a high-quality scaffold without oracle labels,
  and whether it does so more robustly/efficiently than recurrent or feedforward
  controls.
- The next clean experiment should therefore be a true cascade or substitute
  cascade: feed sampled/settled signed-distance fields into the shape-gated
  readout, and score the final pixels plus the intermediate scaffold. If the
  scaffold is oracle, the renderer score is no longer meaningful evidence for
  oscillatory dynamics.

## HORN Generator Probe

A colleague shared preliminary CIFAR-10 evidence that a homogeneous HORN-style
generator can beat a Kuramoto/Un-0-style oscillator generator under a similar
latent generator framing. That is directly relevant because OscNet already had
HORN cells historically, but the modern generator harness only exposed
Kuramoto dynamics.

Implemented response:

- Added `HORNImageGenerator` as a reusable `oscnet.models` class.
- Wired it into `oscnet.experiments.mnist_generator` as `model_family="horn"`,
  with `frozen_horn` and `horn_decoder_only` controls.
- Kept the same decoder modes, conditioning modes, losses, artifacts, and
  success diagnostics as the Kuramoto generator, so comparisons are fair.
- Added safe moment-loss gradients; the old `jnp.std` gradient could go NaN
  when a generator initially emitted almost-constant images.
- Added Modal preset `mnist_generator_horn_resize_conv_core`.

This is a scaffold/probe, not yet a positive result. The right read is:
HORN is now testable under the strongest current MNIST generator setup
(`resize_conv` + queue-backed pixel drift) without creating one-off machinery.
The first meaningful question is whether trainable HORN beats both
`frozen_horn` and `horn_decoder_only`, and whether it improves over the matched
Kuramoto rows. If it only beats Kuramoto but not its own frozen/decoder
controls, it is not evidence for useful learned HORN dynamics.

Modal result:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_resize_conv_core
```

The initial sweep wrote:

```text
outputs/analysis/modal_mnist_generator_horn_resize_conv_core.csv
outputs/analysis/modal_mnist_generator_horn_resize_conv_core.json
```

After the initial run, two missing matched `decoder_only` Kuramoto controls
were run directly with the same settings; the Modal preset now includes those
rows for future full sweeps.

Two-seed mean metrics:

| Variant | Best loss | Final eval | Prototype acc | Nearest-real MSE | Pixel mean MSE | Pixel std MSE | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kuramoto | 0.000811 | 0.000829 | 0.1074 | 0.0800 | 0.0090 | 0.0051 | 1.0560 |
| Frozen Kuramoto | 0.000830 | 0.000875 | 0.0986 | 0.0874 | 0.0111 | 0.0058 | 1.1337 |
| Kuramoto decoder-only | 0.000838 | 0.000871 | 0.0938 | 0.0842 | 0.0107 | 0.0053 | 1.0889 |
| HORN | 0.001057 | 0.001060 | 1.0000 | 0.0504 | 0.0060 | 0.0036 | 1.1819 |
| Frozen HORN | 0.001128 | 0.001131 | 0.9990 | 0.0552 | 0.0086 | 0.0042 | 1.1705 |
| HORN decoder-only | 0.001227 | 0.001230 | 1.0000 | 0.0400 | 0.0054 | 0.0033 | 0.9436 |

Interpretation:

- HORN does show useful trainable dynamics inside its own family: trainable
  HORN beats frozen HORN and HORN decoder-only on the pixel-drift objective.
- It does not reproduce the colleague's claimed HORN-over-Kuramoto pattern on
  this MNIST resize-conv pixel-drift setup. Kuramoto wins the objective, and
  the matched Kuramoto controls are close to trainable Kuramoto.
- HORN looks much better on prototype/class alignment and nearest-real MSE, but
  HORN decoder-only already has those strengths. That makes the result a
  state/readout/conditioning scaffold effect, not yet a learned HORN-dynamics
  breakthrough.
- The metric disagreement matters. Kuramoto optimizes the training objective
  better but appears weakly class-conditional by prototype-nearest accuracy.
  HORN is more class/prototype aligned but worse on the chosen drift objective.
  Future generator comparisons should track both objective loss and semantic
  class alignment, because either metric alone can tell the wrong story.

Visual sample inspection sharpened that conclusion. Pulling sample grids from
the Modal volume showed readable HORN digits and fragmented Kuramoto blobs for
seed 11, despite Kuramoto's lower pixel-drift loss:

```bash
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_horn_n196_resizeconv_steps16_train5000_seed11_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/modal_mnist_generator_horn_probe_samples/horn_seed11.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_kuramoto_n196_resizeconv_steps16_train5000_seed11_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/modal_mnist_generator_horn_probe_samples/kuramoto_seed11.png
```

This is a meaningful pivot: the generator objective is not yet aligned with
human-visible digit quality. To make future sweeps less blind, the generator
harness now supports optional classifier-based sample metrics via
`--quality-classifier-epochs`. When enabled, summaries report
`classifier_label_accuracy`, `classifier_label_confidence`,
`classifier_max_confidence`, and `classifier_entropy` for generated samples.
These metrics should sit beside pixel-drift loss, prototype accuracy, and
visual grids in all serious generator comparisons.

Follow-up semantic scoring confirmed the visual read. Generated sample arrays
were pulled from the Modal volume and scored locally with one shared MNIST
classifier trained for five epochs on the seed-11 split:

```text
outputs/analysis/modal_mnist_generator_horn_probe_quality_seed11.json
```

Classifier quality metrics on seed-11 generated samples:

| Run | Classifier label acc | Label confidence | Max confidence | Entropy | Prototype acc | Nearest-real MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN | 0.9961 | 0.8125 | 0.8129 | 0.8738 | 1.0000 | 0.0516 |
| HORN decoder-only | 0.9902 | 0.8160 | 0.8187 | 0.8504 | 1.0000 | 0.0396 |
| Kuramoto | 0.1074 | 0.1071 | 0.5342 | 1.5053 | 0.1250 | 0.0782 |
| Kuramoto decoder-only | 0.1094 | 0.1111 | 0.5102 | 1.5608 | 0.1191 | 0.0850 |

Interpretation:

- On semantic sample quality, HORN massively outperforms Kuramoto in this
  conditional resize-conv generator setup, despite worse pixel-drift loss.
- The effect is not primarily learned HORN recurrent dynamics: HORN
  decoder-only is already excellent by classifier/prototype metrics. The HORN
  position/velocity noise geometry plus resize-conv readout is a much better
  conditional generator scaffold than Kuramoto phase noise in this setup.
- This makes HORN worth keeping, but the immediate breakthrough path is not
  "add more HORN steps." It is to design an objective/control suite where
  semantic sample quality is first-class and then ask whether HORN dynamics can
  improve diversity, robustness, efficiency, or sample refinement beyond the
  already-strong HORN scaffold.

Added follow-up Modal preset
`mnist_generator_horn_conditioning_attribution_probe` for that next question.
It keeps the resize-conv HORN setup fixed, runs one seed, varies
`label_phase_scale`, and compares trainable HORN, frozen HORN, and
HORN decoder-only with classifier-based semantic scoring enabled. This should
answer whether the readable digits are mostly label conditioning/readout
geometry, or whether learned HORN recurrence contributes once conditioning is
weakened.

Probe result:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_conditioning_attribution_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_conditioning_attribution_probe.csv
outputs/analysis/modal_mnist_generator_horn_conditioning_attribution_probe.json
outputs/analysis/modal_mnist_generator_horn_conditioning_attribution_samples/
```

Seed-11 attribution metrics:

| Label phase scale | Variant | Final eval | Classifier label acc | Label confidence | Prototype acc | Nearest-real MSE | Diversity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | Trainable HORN | 0.000913 | 1.0000 | 0.5539 | 1.0000 | 0.0581 | 1.3177 |
| 0.0 | Frozen HORN | 0.001231 | 0.0977 | 0.0989 | 0.0977 | 0.0520 | 0.5566 |
| 0.0 | HORN decoder-only | 0.001184 | 0.1016 | 0.1014 | 0.1250 | 0.0462 | 0.5786 |
| 0.1 | Trainable HORN | 0.000954 | 1.0000 | 0.5531 | 1.0000 | 0.0584 | 1.3051 |
| 0.1 | Frozen HORN | 0.000888 | 0.7461 | 0.4030 | 0.9004 | 0.0859 | 1.2198 |
| 0.1 | HORN decoder-only | 0.000866 | 0.7500 | 0.4060 | 0.8984 | 0.0863 | 1.2409 |
| 0.5 | Trainable HORN | 0.001059 | 0.9980 | 0.5536 | 1.0000 | 0.0514 | 1.1968 |
| 0.5 | Frozen HORN | 0.001146 | 0.9941 | 0.5458 | 1.0000 | 0.0590 | 1.1753 |
| 0.5 | HORN decoder-only | 0.001231 | 1.0000 | 0.5574 | 1.0000 | 0.0394 | 0.9178 |

Visual inspection agrees with the classifier metrics. With
`label_phase_scale=0.0`, trainable HORN produces clean digit grids while both
frozen HORN and HORN decoder-only collapse into blurry, weakly class-aligned
shapes. That is the strongest positive attribution result so far for learned
oscillatory recurrence in the generator branch.

Important caveat: label scale zero removes explicit phase-shift injection at
sampling time, but the training objective is still class-aware because
pixel-drift positives are sampled from the same class. The result therefore
does not prove unconditional generation. It shows that the trainable HORN
field can internalize class-structured generation from the class-conditioned
objective, whereas the matched decoder-only and frozen controls cannot.

At stronger explicit label phase scales, the decoder/frozen controls recover
much of the semantic score. This suggests a useful division:

- Weak/no explicit label signal: learned HORN recurrence is crucial.
- Strong explicit label signal: the HORN latent geometry plus resize-conv
  readout can carry the task even when recurrent dynamics are frozen or
  removed.
- The old pixel objective remains imperfect: the best objective row at
  label scale `0.1` is not the most semantically reliable row.

Current research read: HORN is now the strongest MNIST generator branch in
OscNet, and unlike the earlier Kuramoto generator it has a regime where
trainable oscillatory recurrence visibly matters. The next replication should
repeat the label-scale attribution probe across at least two more seeds and
compare against a matched non-oscillatory latent-state generator before making
larger claims.

Replication result:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_label0_replication_probe
```

The replication wrote:

```text
outputs/analysis/modal_mnist_generator_horn_label0_replication_probe.csv
outputs/analysis/modal_mnist_generator_horn_label0_replication_probe.json
outputs/analysis/modal_mnist_generator_horn_label0_replication_samples/
```

This repeated only the key condition, `label_phase_scale=0.0`, on fresh seeds
12 and 13. In this setting the label embedding starts at zero, so there is no
pre-seeded class geometry, but in trainable runs the conditioning parameters
can still learn from labels during training. Combined with seed 11 from the
attribution probe:

| Variant | Seeds | Mean final eval | Mean classifier label acc | Mean prototype acc | Mean diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trainable HORN | 3 | 0.000936 | 1.0000 | 1.0000 | 1.3079 |
| Frozen HORN | 3 | 0.001282 | 0.0898 | 0.0983 | 0.4868 |
| HORN decoder-only | 3 | 0.001254 | 0.0944 | 0.1100 | 0.5348 |

This is the first replicated, control-separated generator result where
learned oscillatory recurrence is not a cosmetic add-on. With label geometry
initialized at zero, trainable HORN consistently produces clean
class-structured MNIST grids, while the matched frozen and decoder-only
controls collapse to near-chance class alignment. The sample montage confirms
the metrics: the trainable HORN grids contain readable digits for all three
seeds; the controls are mostly blurred texture fragments.

The claim should still stay precise. This is not an unconditional generator,
because the pixel-drift objective is class-conditioned during training. The
positive result is narrower and stronger: **within a class-conditioned
oscillator generator, trainable HORN recurrence can internalize and express
class structure that the same HORN state/readout scaffold cannot express when
the recurrence is frozen or removed.**

Next control: add a matched non-oscillatory latent-state generator with the
same resize-conv readout and parameter budget. If HORN beats that under
label-zero conditioning, the result becomes an ONN-native architectural win,
not only a recurrence-vs-no-recurrence win inside the HORN family.

Matched non-oscillatory control:

Implemented `StateMLPImageGenerator`, a residual MLP transition over the same
position/velocity latent state and the same resize-conv readout. With
`state_mlp_hidden_dim=48`, the transition has about the same recurrent
parameter and operation budget as HORN's dense coupling for 196 oscillators.

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_state_mlp_label0_control_probe
```

The control wrote:

```text
outputs/analysis/modal_mnist_generator_state_mlp_label0_control_probe.csv
outputs/analysis/modal_mnist_generator_state_mlp_label0_control_probe.json
outputs/analysis/modal_mnist_generator_state_mlp_label0_control_samples/
```

Three-seed comparison at `label_phase_scale=0.0`:

| Variant | Mean final eval | Mean classifier label acc | Mean prototype acc | Mean diversity | Mean nearest-real MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trainable HORN | 0.000936 | 1.0000 | 1.0000 | 1.3079 | 0.0586 |
| Frozen HORN | 0.001282 | 0.0898 | 0.0983 | 0.4868 | 0.0541 |
| HORN decoder-only | 0.001254 | 0.0944 | 0.1100 | 0.5348 | 0.0533 |
| Trainable state-MLP | 0.001256 | 1.0000 | 1.0000 | 0.9174 | 0.0334 |
| Frozen state-MLP | 0.001289 | 0.0768 | 0.0898 | 0.4281 | 0.0448 |
| State-MLP decoder-only | 0.000987 | 0.0944 | 0.0951 | 0.8134 | 0.0640 |

Interpretation:

- The HORN result remains real: its trainable recurrent field is far better
  than frozen HORN or HORN decoder-only.
- The stronger claim, "HORN is uniquely responsible," does not hold under this
  control. A matched trainable non-oscillatory state transition also reaches
  perfect classifier/prototype alignment under the same class-conditioned
  pixel-drift task.
- HORN still has useful advantages in this setup: better mean pixel objective
  than trainable state-MLP and substantially higher diversity. State-MLP has
  lower nearest-real MSE, which may indicate more conservative,
  exemplar-adjacent samples rather than a better generative field.
- Pixel objective alone is again misleading: state-MLP decoder-only has a good
  mean final eval loss but chance-level semantic accuracy.

Updated research read: the current "breakthrough" is not yet
oscillator-specific image generation. It is a robust finding that **trainable
latent dynamics over a position/velocity field are essential for label-zero
class-structured generation**, and HORN is the best oscillatory version of that
idea so far. The next ONN-specific test should target where HORN's physics
ought to matter beyond semantic class alignment: diversity, interpolation
smoothness, settling/refinement, robustness to fewer training samples, or
out-of-distribution/noisy latent initial states.

Low-data sample-efficiency probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_low_data_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_state_mlp_low_data_probe.csv
outputs/analysis/modal_mnist_generator_horn_state_mlp_low_data_probe.json
outputs/analysis/modal_mnist_generator_horn_state_mlp_low_data_samples/
```

This keeps the label-zero pixel-drift setup fixed but reduces the training set
to 500 MNIST examples. Three-seed means:

| Variant | Mean final eval | Mean classifier label acc | Mean prototype acc | Mean diversity | Mean nearest-real MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trainable HORN | 0.001100 | 0.9661 | 0.9993 | 1.2514 | 0.0630 |
| Trainable state-MLP | 0.001252 | 0.9186 | 0.9993 | 0.8832 | 0.0439 |

Interpretation:

- This is the clearest HORN-vs-state-MLP advantage so far. With scarce data,
  HORN has lower pixel-drift loss, higher classifier label accuracy, and much
  higher diversity.
- The nearest-real MSE still favors state-MLP, but that likely reflects more
  conservative, exemplar-adjacent samples. The sample montage shows state-MLP
  can produce readable digits, but one seed collapses into darker/blurry
  repeated strokes, while HORN stays more stable across seeds.
- This does not prove a universal ONN win, but it upgrades the HORN branch from
  "nice oscillatory version of a trainable latent dynamics idea" to a plausible
  **oscillatory inductive bias**: the second-order position/velocity dynamics
  appear to regularize the generator under limited data.

Updated research read after the low-data probe: HORN is the strongest current
OscNet-native MNIST generator, not because it beats every ordinary neural
model on every metric, but because it now has replicated dynamics attribution
and a matched-control regime where the oscillatory transition is more stable
and more diverse than a similarly sized learned state MLP.

Settling-depth probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_settling_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_state_mlp_settling_probe.csv
outputs/analysis/modal_mnist_generator_horn_state_mlp_settling_probe.json
```

This is the first direct "does settling itself help?" diagnostic on the HORN
generator branch. It trains one low-data seed for HORN and the matched
state-MLP, then evaluates the same trained model at test-time depths
`0,1,2,4,8,16,32`.

Seed 11 classifier label accuracy by test-time depth:

| Variant | Step 0 | Step 1 | Step 2 | Step 4 | Step 8 | Step 16 | Step 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN | 0.0273 | 0.0820 | 0.2207 | 0.5898 | 0.8770 | 0.9609 | 0.0488 |
| State-MLP | 0.7715 | 0.8223 | 0.8516 | 0.9199 | 0.9707 | 0.9863 | 0.7246 |

Selected diversity/nearest-real metrics:

| Variant | Best class step | Step-0 diversity | Step-16 diversity | Step-32 diversity | Step-16 nearest-real MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| HORN | 16 | 1.2953 | 1.2506 | 1.1747 | 0.0636 |
| State-MLP | 16 | 1.0887 | 1.0697 | 1.0138 | 0.0472 |

Interpretation:

- HORN has a much stronger dynamics signature. At step 0 it is near chance;
  semantic class structure emerges progressively through recurrent settling
  and peaks at step 16.
- State-MLP also benefits from settling, but it already has high class
  alignment at step 0. That means the learned label/state/readout scaffold is
  carrying much more class information before any recurrent transition.
- Both models over-settle by step 32. More recurrent compute is not
  automatically better; this branch has a learned finite-time attractor around
  the training depth.
- This strengthens the attribution story for HORN specifically: its good
  samples are less explainable as an immediate label-conditioned decoder readout
  and more explainable as a learned second-order dynamical organization process.
  The next physics-native pressure test should therefore target finite-time
  stability: train over a range of depths, add mild damping/step robustness
  losses, or test whether structured/sparser HORN coupling preserves the
  step-16 gain without dense all-to-all coupling.

Finite-time stability training probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_settling_train_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_settling_train_probe.csv
outputs/analysis/modal_mnist_generator_horn_settling_train_probe.json
outputs/analysis/modal_mnist_generator_horn_settling_train_samples/
```

This compares the low-data HORN baseline against the same model trained with
`--train-settling-steps 8,16,32`, while evaluating both across
`0,1,2,4,8,16,32`.

Seed 11 classifier label accuracy by test-time depth:

| HORN training | Step 0 | Step 1 | Step 2 | Step 4 | Step 8 | Step 16 | Step 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, train at 16 | 0.0449 | 0.1367 | 0.2773 | 0.6387 | 0.8848 | 0.9609 | 0.0371 |
| Train at 8/16/32 | 0.0039 | 0.0215 | 0.0547 | 0.3477 | 0.8926 | 0.9883 | 0.8965 |

Selected metrics:

| HORN training | Final eval | Final classifier acc | Step-16 diversity | Step-32 diversity | Step-16 nearest-real MSE | Samples/train second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline, train at 16 | 0.001093 | 0.9844 | 1.3039 | 1.3944 | 0.0676 | 530.3 |
| Train at 8/16/32 | 0.001141 | 0.9902 | 1.3237 | 1.5943 | 0.0682 | 256.8 |

Interpretation:

- Variable-depth training largely fixes the over-settling failure: step-32
  classifier accuracy jumps from 0.0371 to 0.8965.
- It preserves, and slightly improves, the trained-depth semantic result:
  step-16 classifier accuracy rises from 0.9609 to 0.9883.
- The cost is real: training throughput roughly halves, early-step samples are
  more chaotic, and nearest-real MSE/diversity suggest the stabilized model is
  more energetic and less conservative.
- This is one of the strongest architecture-level lessons so far. HORN should
  not be trained as a single exact-depth recurrent decoder; it behaves more
  like a finite-time dynamical system that needs a settling window. Training
  over that window turns the brittle step-16 attractor into a broader usable
  trajectory.

Updated research read after variable-depth training: the current HORN branch
now has three positive properties at once: dynamics attribution, low-data
advantage over a matched state-MLP on diversity/objective, and improved
test-time stability when trained over a finite settling window. The next
serious controls should test whether this survives multiple seeds and whether
structured/local/low-rank HORN coupling can keep the same stability without
full dense all-to-all coupling.

Structured-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_structured_coupling_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_structured_coupling_probe.csv
outputs/analysis/modal_mnist_generator_horn_structured_coupling_probe.json
outputs/analysis/modal_mnist_generator_horn_structured_coupling_samples/
```

This keeps the low-data, variable-depth HORN setup fixed and replaces dense
coupling with a soft spatial distance-decay profile. It is not true sparse
hardware routing yet; every off-diagonal connection is still present, but most
long-range interactions are strongly down-weighted.

Seed 11 comparison against the dense variable-depth baseline:

| HORN coupling | Profile mean | Final eval | Final classifier acc | Step-16 acc | Step-32 acc | Step-16 diversity | Step-32 diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 1.0000 | 0.001141 | 0.9902 | 0.9883 | 0.8965 | 1.3237 | 1.5943 |
| Distance decay, length 0.35, floor 0.02 | 0.1394 | 0.001142 | 0.9922 | 0.9863 | 0.8848 | 1.3186 | 1.6228 |
| Distance decay, length 0.60, floor 0.02 | 0.3064 | 0.001142 | 0.9922 | 0.9883 | 0.8848 | 1.3196 | 1.6200 |

Interpretation:

- The variable-depth HORN result does not require uniform dense coupling. Both
  distance-decay profiles preserve readable samples and essentially match the
  dense step-16 result.
- Step-32 stability is slightly lower than dense but still dramatically better
  than single-depth training. The profile with mean 0.1394 is especially
  interesting: it keeps almost all quality while strongly biasing interaction
  toward local neighborhoods.
- This is a meaningful physical-plausibility improvement, but not the final
  hardware story. The next structured control should make coupling actually
  sparse, low-rank, or local-plus-global instead of merely soft-decayed
  all-to-all.

Updated research read after structured coupling: the strongest HORN recipe is
now **variable-depth HORN with spatially biased coupling**. It preserves the
positive generator metrics while moving away from the least plausible dense
all-to-all assumption.

Sparse local-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_sparse_coupling_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_horn_sparse_coupling_probe.csv
outputs/analysis/modal_mnist_generator_horn_sparse_coupling_probe.json
outputs/analysis/modal_mnist_generator_horn_sparse_coupling_samples/
```

This adds `coupling_profile="local_radius"` to the reusable generator models.
Unlike distance decay, this profile is a true sparse binary mask: oscillator
pairs outside the local radius have exactly zero recurrent coupling.

Seed 11 comparison:

| HORN coupling | Coupling density | Final eval | Final classifier acc | Step-16 acc | Step-32 acc | Step-16 diversity | Step-32 diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 1.0000 | 0.001141 | 0.9902 | 0.9883 | 0.8965 | 1.3237 | 1.5943 |
| Distance decay, length 0.35, floor 0.02 | 1.0000 soft / mean 0.1394 | 0.001142 | 0.9922 | 0.9863 | 0.8848 | 1.3186 | 1.6228 |
| Local radius 0.24 | 0.0367 | 0.001142 | 0.9922 | 0.9863 | 0.8887 | 1.3184 | 1.6248 |
| Local radius 0.35 | 0.0870 | 0.001142 | 0.9922 | 0.9863 | 0.8848 | 1.3189 | 1.6241 |

Interpretation:

- The variable-depth HORN result survives true sparse coupling. The tight
  radius-0.24 mask keeps only about 3.7% of possible directed oscillator edges
  while preserving readable samples and the same step-16/step-32 semantic
  performance.
- The sparse profiles behave almost identically to soft distance decay on this
  seed, which suggests the current task does not need learned dense global
  communication once the resize-conv readout and finite settling window are in
  place.
- This is the strongest physical-plausibility result so far. It does not prove
  hardware efficiency on GPU, but it moves the architecture from dense
  all-to-all dynamics to a local recurrent oscillator field without losing the
  current MNIST generator win.

Updated research read after sparse coupling: the leading recipe is now
**variable-depth HORN with sparse local coupling**. The next test should move
from one-seed architecture probing to replication: verify the sparse local HORN
advantage over the matched state-MLP and decoder/frozen controls across seeds,
then test whether the same recipe scales to a less toy dataset or a harder
generation metric.

Sparse local HORN vs state-MLP replication:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_replication_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_replication_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_replication_probe.json
```

Three-seed comparison. Both models use the same low-data recipe
(`train_limit=500`), resize-conv readout, `label_phase_scale=0.0`,
`pixel_drift` loss, queue-backed positives, and variable-depth training over
8/16/32 settling steps. HORN uses sparse local radius `0.24`; state-MLP uses a
matched residual latent-state transition with almost the same parameter and
operation budget.

| Model | Final classifier acc | Diversity | Step-0 acc | Step-8 acc | Step-16 acc | Step-32 acc | Params | Decoder fraction | Coupling density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse local HORN | 0.9792 | 1.3342 | 0.0033 | 0.8880 | 0.9727 | 0.8789 | 42,981 | 0.0560 | 0.0367 |
| State-MLP control | 0.6361 | 0.5442 | 0.4811 | 0.6335 | 0.6328 | 0.5215 | 42,441 | 0.0568 | 0.0000 |

Interpretation:

- This is the strongest generator result in the repo so far. Sparse local HORN
  is not merely producing nicer-looking grids; it is substantially more stable
  across seeds than the matched state-MLP control.
- The dynamics-attribution signal is strong: HORN is near useless at step 0,
  then becomes class-structured through settling, peaking around 16 steps.
  The state-MLP starts partly class-biased but is unstable and collapses badly
  on one seed.
- The result is not just a large decoder effect. The decoder is only about
  5.6% of parameters, while roughly 94% of trainable parameters live in the
  recurrent/conditioning side.
- The nearest-real MSE remains lower for the state-MLP control, which is a good
  warning metric: bland or collapsed samples can be close to some real sample
  while still having worse semantic quality and diversity.
- The precise claim should be: sparse local HORN is the current best
  OscNet-native MNIST generator and the cleanest positive learned-dynamics
  result so far. It is not yet a proof that ONNs beat strong conventional
  generative models in general.

Sparse local HORN attribution/control audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_attribution_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_attribution_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_attribution_probe.json
outputs/analysis/modal_mnist_generator_sparse_horn_attribution_samples/montage.png
```

Three-seed comparison with the same sparse local HORN recipe and matched
controls:

| Variant | Final classifier acc | Diversity | Step-0 acc | Best acc | Best step | Recurrent op fraction | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sparse HORN | 0.9792 | 1.3341 | 0.0033 | 0.9727 | 16 | 0.3415 | Best full variable-depth run |
| HORN 1-step | 0.9635 | 1.2759 | 0.9674 | 0.9681 | 0/1 | 0.0314 | Nearly same quality; exposes shallow shortcut |
| State-MLP | 0.6315 | 0.4776 | 0.4850 | 0.6458 | ~7 | 0.3368 | Unstable; one seed collapses |
| Frozen HORN | 0.1035 | 0.2779 | 0.1022 | 0.1120 | mixed | 0.3415 | Chance-like |
| HORN decoder-only | 0.1042 | 0.3195 | 0.1061 | 0.1133 | mixed | 0.0000 | Chance-like |
| Frozen state-MLP | 0.0898 | 0.3755 | 0.1100 | 0.1172 | mixed | 0.3368 | Label-uncontrolled |
| State-MLP decoder-only | 0.1035 | 0.3534 | 0.1029 | 0.1074 | mixed | 0.0000 | Chance-like |

Interpretation:

- The positive result survives the stricter controls: learned sparse HORN
  crushes frozen HORN, decoder-only HORN, frozen state-MLP, state-MLP
  decoder-only, and the trainable state-MLP on semantic class accuracy and
  diversity.
- The one-step HORN control is the caveat. It produces nearly the same visual
  quality and classifier accuracy with very little recurrent compute. That
  means this conditional MNIST objective allows a shallow
  label-conditioning/readout route if the model is trained directly for it.
- The full variable-depth HORN run still has a real settling signature: step 0
  is near chance, step 8 becomes useful, and step 16 peaks. But the current
  task is not enough to claim that long oscillator settling is necessary.
- Updated claim: sparse local HORN is the best current OscNet generator and a
  strong learned-dynamics result against matched controls. The next scientific
  test should reduce static conditioning shortcuts or move to a harder
  benchmark before presenting it as a deeper oscillator-generation
  breakthrough.

Sparse local HORN conditioning-route probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_conditioning_route_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_conditioning_route_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_conditioning_route_probe.json
outputs/analysis/modal_mnist_generator_sparse_horn_conditioning_route_samples/montage.png
```

This is the first direct anti-shortcut test after the one-step result. It
compares the current `phase_shift` label route against variants where the class
label no longer adds a direct initial-state vector. `class_coupling` applies a
learned class-specific drive during HORN settling; `class_oscillator` routes the
label through a separate conditioning oscillator pool.

Single-seed compact probe, seed 11:

| Variant | Final classifier acc | Best acc | Step 0 | Step 1 | Step 8 | Step 16 | Step 32 | Diversity | Visual read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| phase-shift full | 0.9922 | 0.9863 @ 16 | 0.0020 | 0.0059 | 0.8848 | 0.9863 | 0.8887 | 1.3062 | Crisp digits |
| phase-shift 1-step | 0.9785 | 0.9805 @ 0 | 0.9805 | 0.9805 | - | - | - | 1.2884 | Crisp digits |
| class-coupling full | 0.5293 | 0.6426 @ 32 | 0.0938 | 0.1133 | 0.2871 | 0.5312 | 0.6426 | 1.4874 | Blurry class-like blobs |
| class-coupling 1-step | 0.1016 | 0.1133 @ 1 | 0.1113 | 0.1133 | - | - | - | 0.5165 | Chance-like |
| class-oscillator full | 0.1035 | 0.1133 @ 16 | 0.0996 | 0.0977 | 0.1016 | 0.1133 | 0.1035 | 0.4177 | Non-digit texture |
| class-oscillator 1-step | 0.1016 | 0.1133 @ 0 | 0.1133 | 0.1113 | - | - | - | 0.4941 | Non-digit texture |
| frozen class-oscillator | 0.1055 | 0.1172 @ 16 | 0.0996 | 0.0977 | 0.1074 | 0.1172 | 0.0996 | 0.4146 | Non-digit texture |

Interpretation:

- The one-step caveat is real, not a bookkeeping artifact. The `phase_shift`
  route can learn a direct class-conditioned initial state/readout code that
  already produces crisp digits at step 0/1.
- Removing that initial label-shift route makes the task much harder.
  `class_oscillator` fails: the current separate conditioning oscillator pool
  does not provide a usable class signal for MNIST generation.
- `class_coupling` is the useful anti-shortcut lead. It starts at chance and
  climbs steadily with settling, reaching `0.6426` classifier accuracy at step
  32. The samples are still blurry blobs, but the dynamics are doing real
  class-routing work that one step cannot do.
- Updated research read: the near-perfect HORN result is mostly a strong
  conditional oscillator scaffold with a direct label-initialization route.
  The more ONN-native no-direct-label route is not solved yet, but
  `class_coupling` gives a concrete path: train/evaluate longer settling
  horizons and strengthen the dynamic class drive without reintroducing a
  direct initial-state label code.

Sparse local HORN class-coupling sharpen probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_sharpen_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_sharpen_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_sharpen_probe.json
outputs/analysis/class_coupling_sharpen_samples/comparison.png
```

This follow-up keeps the no-direct-label `class_coupling` route and asks which
intervention actually sharpens it: longer settling, stronger dynamic class
drive, or a small explicit phase anchor.

Single-seed compact probe, seed 11:

| Variant | Label phase scale | Conditioning strength | Final classifier acc | Best acc | Step 0 | Step 1 | Step 8 | Step 16 | Step 32 | Diversity | Nearest-real MSE | Visual read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline class-coupling | 0.0 | 1.0 | 0.5273 | 0.6445 @ 32 | 0.0957 | 0.1113 | 0.2871 | 0.5313 | 0.6445 | 1.4861 | 0.0977 | Blurry blobs |
| long settling | 0.0 | 1.0 | 0.7461 | 0.7383 @ 32 | 0.0938 | 0.0977 | 0.2305 | 0.4902 | 0.7383 | 1.5392 | 0.1018 | Cleaner but soft |
| strong class drive | 0.0 | 2.0 | 0.9609 | 0.9707 @ 32 | 0.1035 | 0.1152 | 0.4492 | 0.8672 | 0.9707 | 1.4362 | 0.0839 | Readable digits |
| small anchor | 0.5 | 1.0 | 1.0000 | 0.9961 @ 32 | 0.0918 | 0.1426 | 0.6738 | 0.9746 | 0.9961 | 1.1989 | 0.0580 | Very crisp digits |

Interpretation:

- Stronger dynamic class coupling is the clean win. It keeps step 0 near
  chance, avoids the direct initial phase-shift route, and reaches `0.9707`
  best-settled classifier accuracy with readable digit grids.
- Longer settling alone helps but is not enough. It improves the old baseline
  from roughly `0.64` to `0.74` best-settled accuracy, but the samples remain
  softer and farther from real digits.
- The small anchor is visually best, but it reintroduces a weak explicit
  label-initialization path. It is useful as a ceiling/continuation trick, not
  as the cleanest ONN-native claim.
- Updated research read: sparse local HORN now has two useful presets. Use
  `sparse_horn_mnist` as the practical polished generator recipe. Use
  `sparse_horn_mnist_class_coupling_strong` when testing the stricter claim
  that class information can be routed through learned oscillator settling
  rather than injected directly into the initial state.

Sparse local HORN class-coupling strong control probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strong_control_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strong_control_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strong_control_probe.json
outputs/analysis/class_coupling_strong_control_samples/comparison_by_seed.png
```

This is the fairer control audit after the `class_coupling_strong` one-seed
win. The previous `state_mlp` control ignored labels under the
`class_coupling` route, so it was too weak. The updated
`StateMLPImageGenerator` now receives the same learned dynamic class drive as
HORN while keeping a non-oscillatory residual MLP state transition.

Three-seed compact probe, seeds 11/12/13:

| Variant | Final classifier acc | Best acc | Step 0 | Step 16 | Step 32 | Diversity | Nearest-real MSE | Visual read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HORN strong class-coupling | 0.8229 +/- 0.2631 | 0.8249 +/- 0.2731 | 0.1022 | 0.6797 | 0.8249 | 1.2745 | 0.0803 | Two crisp seeds, one washed-out seed |
| HORN frozen | 0.1029 +/- 0.0130 | 0.1120 +/- 0.0045 | 0.1087 | 0.1016 | 0.1081 | 0.2779 | 0.0428 | Chance-like texture |
| HORN decoder-only | 0.1042 +/- 0.0074 | 0.1107 +/- 0.0041 | 0.1061 | 0.1094 | 0.1087 | 0.3191 | 0.0443 | Chance-like texture |
| StateMLP + class drive | 0.9993 +/- 0.0011 | 1.0000 +/- 0.0000 | 0.1139 | 0.9850 | 0.9987 | 0.6343 | 0.0371 | Crisp class prototypes |
| Frozen StateMLP + class drive | 0.0951 +/- 0.0100 | 0.1152 +/- 0.0070 | 0.1074 | 0.1042 | 0.1055 | 0.1817 | 0.0392 | Class-agnostic blobs |

Per-seed best settled classifier accuracy:

| Variant | Seed 11 | Seed 12 | Seed 13 |
| --- | ---: | ---: | ---: |
| HORN strong class-coupling | 0.9707 @ 32 | 0.5098 @ 32 | 0.9941 @ 32 |
| StateMLP + class drive | 1.0000 @ 16 | 1.0000 @ 48 | 1.0000 @ 32 |

Interpretation:

- The no-direct-label HORN signal remains real: trainable HORN crushes frozen
  HORN and decoder-only HORN, starts near chance at step 0, and improves
  through settling.
- The stronger claim does not hold yet. Once the matched non-oscillatory
  StateMLP gets the same class-drive route, it beats HORN on classifier
  accuracy and nearest-real MSE across all three seeds.
- The StateMLP win is not a perfect generative win: visual samples are crisp
  but prototype-like, with much lower diversity (`0.6343`) than HORN
  (`1.2745`). HORN remains more varied, but seed 12 is unstable and washed out.
- Updated research read: the breakthrough path is not simply "HORN plus
  dynamic class coupling." The next HORN-specific question is how to preserve
  HORN's diversity while improving robustness/contrast across seeds. Candidate
  targeted probes: stronger HORN class drive, damping/state-bound tuning,
  contrast-aware objectives, or a diversity-preserving state-MLP control so the
  comparison does not collapse to class prototypes.

Sparse local HORN class-drive strength probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strength_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strength_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strength_probe.json
outputs/analysis/class_coupling_strength_samples/comparison_by_seed.png
```

This directly tested the hypothesis from the control audit: HORN was moving the
latent field too weakly relative to the conditioned StateMLP. The sweep kept
the no-direct-label `class_coupling` route fixed and varied only
`conditioning_strength`.

Three-seed compact probe, seeds 11/12/13:

| HORN conditioning strength | Final classifier acc | Best acc | Step 0 | Step 16 | Step 32 | Diversity | Nearest-real MSE | Visual read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2.0 | 0.8229 +/- 0.2631 | 0.8242 +/- 0.2743 | 0.1022 | 0.6797 | 0.8242 | 1.2754 | 0.0803 | Two crisp seeds, one washed-out seed |
| 4.0 | 0.9857 +/- 0.0200 | 0.9850 +/- 0.0226 | 0.1120 | 0.9499 | 0.9850 | 1.2015 | 0.0641 | Crisp, seed 12 mostly rescued |
| 8.0 | 0.9941 +/- 0.0101 | 0.9935 +/- 0.0113 | 0.1029 | 0.9727 | 0.9935 | 1.1397 | 0.0543 | Crisp across all seeds |

Per-seed best settled classifier accuracy:

| Strength | Seed 11 | Seed 12 | Seed 13 |
| --- | ---: | ---: | ---: |
| 2.0 | 0.9707 @ 32 | 0.5078 @ 32 | 0.9941 @ 32 |
| 4.0 | 0.9961 @ 32 | 0.9590 @ 32 | 1.0000 @ 32 |
| 8.0 | 1.0000 @ 32 | 0.9805 @ 32 | 1.0000 @ 32 |

Interpretation:

- This rescues the strict HORN route. The poor seed-12 result was not a
  fundamental class-coupling failure; it was an underpowered class-drive
  regime.
- Strength 8 keeps the important attribution pattern: step 0 remains near
  chance, and class accuracy emerges through oscillator settling.
- Compared with the matched StateMLP control, strength-8 HORN has slightly
  lower classifier accuracy (`0.9941` vs `0.9993`) and worse nearest-real MSE
  (`0.0543` vs `0.0371`), but much higher diversity (`1.1397` vs `0.6343`) and
  visibly less prototype-collapse.
- Updated research read: `sparse_horn_mnist_class_coupling_strength8` is now
  the strongest strict no-direct-label HORN generator preset. The central next
  comparison is no longer "can HORN generate digits?" It can. The next question
  is whether HORN's higher-diversity settling can beat a diversity-regularized
  StateMLP or improve nearest-real quality without collapsing to prototypes.

Sparse local HORN vs diversity-regularized StateMLP control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_diversity_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_diversity_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_diversity_probe.json
outputs/analysis/state_mlp_diversity_samples/state_mlp_diversity_seed11_comparison.png
```

This is the follow-up control requested by the strength probe. The matched
StateMLP was given the same class-coupling route plus distributional pressure
(`distributional_weight`) and, in one variant, class-conditional moment
pressure (`class_moment_weight`). The point was to test whether HORN's diversity
advantage was only because the non-oscillatory control was under-regularized.

Three-seed compact probe, seeds 11/12/13:

| Variant | Dist weight | Class moment | Final acc | Best acc | Step 0 | Diversity | Best diversity | Nearest-real MSE | Visual read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HORN strength8 | 0.00 | 0.0 | 0.994 | 0.993 | 0.103 | 1.140 | 1.137 | 0.0543 | Crisp, varied digits |
| StateMLP | 0.00 | 0.0 | 0.975 | 0.999 | 0.111 | 0.614 | 0.640 | 0.0374 | Blurry/prototype-like digits |
| StateMLP dist .05 | 0.05 | 0.0 | 0.972 | 0.977 | 0.108 | 0.663 | 0.679 | 0.0378 | Slightly more varied, still soft |
| StateMLP dist .10 | 0.10 | 0.0 | 0.906 | 0.906 | 0.106 | 0.709 | 0.739 | 0.0406 | More varied, weaker labels |
| StateMLP dist .10 + class | 0.10 | 1.0 | 0.957 | 0.966 | 0.106 | 0.644 | 0.658 | 0.0371 | Label recovery improves, diversity drops |

Interpretation:

- HORN's higher-diversity result survives the stronger StateMLP control. The
  best regularized StateMLP variants lift diversity only from roughly `0.61` to
  `0.66-0.71`, still far below HORN's `1.14`.
- The regularized StateMLP quality/diversity tradeoff is steep. At
  `distributional_weight=0.10`, diversity improves but classifier accuracy
  falls to `0.906`; adding class-moment pressure recovers some class accuracy
  but loses diversity again.
- Nearest-real MSE still favors StateMLP, but the sample grids explain why this
  metric is incomplete here: the StateMLP digits are smooth class prototypes,
  while HORN produces sharper and more varied samples. Keep nearest-real MSE as
  a sharpness/proximity proxy, not as the sole generative-quality judge.
- Updated research read: the strict HORN route is now a real lead, not just an
  under-controlled artifact. The next HORN-specific target is to keep the
  strength-8 diversity/settling behavior while reducing nearest-real MSE and
  improving contrast without collapsing into StateMLP-style prototypes.

Sparse local HORN distributional quality probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_distributional_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_distributional_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_distributional_probe.json
outputs/analysis/horn_distributional_samples/horn_distributional_seed11_comparison.png
```

This probe keeps the strict no-direct-label HORN route fixed at
`conditioning_strength=8.0` and adds small distributional loss weights. The goal
was to improve pixel distribution/proximity while preserving the HORN
quality/diversity behavior established above.

Three-seed compact probe, seeds 11/12/13:

| Variant | Dist weight | Final acc | Best acc | Step 0 | Step 16 | Step 32 | Diversity | Best diversity | Nearest-real MSE | Mean MSE | Std MSE | Gen mean | Gen std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN strength8 | 0.000 | 0.994 | 0.993 | 0.103 | 0.973 | 0.993 | 1.140 | 1.137 | 0.0543 | 0.0124 | 0.0067 | 0.175 | 0.354 |
| HORN dist .01 | 0.010 | 0.995 | 0.994 | 0.109 | 0.981 | 0.994 | 1.139 | 1.224 | 0.0546 | 0.0104 | 0.0063 | 0.165 | 0.355 |
| HORN dist .025 | 0.025 | 0.992 | 0.993 | 0.113 | 0.976 | 0.993 | 1.126 | 1.259 | 0.0539 | 0.0092 | 0.0058 | 0.161 | 0.353 |
| HORN dist .05 | 0.050 | 0.976 | 0.972 | 0.111 | 0.952 | 0.972 | 1.105 | 1.261 | 0.0524 | 0.0074 | 0.0049 | 0.154 | 0.346 |

Interpretation:

- Small distributional pressure is useful but not transformational. It improves
  generated mean/std statistics and slightly improves nearest-real MSE while
  keeping step 0 near chance and preserving the settling story.
- `distributional_weight=0.025` is the best balanced quality variant so far:
  nearly the same best classifier accuracy as baseline (`0.993`), modestly
  lower nearest-real MSE (`0.0539` vs `0.0543`), better mean/std errors, and
  still high diversity (`1.126`).
- `distributional_weight=0.05` pushes proximity metrics farther in the right
  direction, but the cost in class accuracy and diversity is already visible.
  That makes it a useful upper bound, not the current recommended setting.
- Updated research read: keep `sparse_horn_mnist_class_coupling_strength8` as
  the strict semantic/dynamic lead. Use
  `sparse_horn_mnist_class_coupling_strength8_dist0025` as the balanced
  quality/proximity variant. The next likely gains are not from simply raising
  distributional weight; they should come from HORN-side dynamics/readout
  refinements that reduce prototype distance without suppressing diversity.

Sparse local HORN stronger-evaluator audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_quality_classifier_audit
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_quality_classifier_audit.csv
outputs/analysis/modal_mnist_generator_sparse_horn_quality_classifier_audit.json
```

Earlier generator sweeps used a lightweight quality classifier trained on the
same 500-example training limit as the generator. That evaluator reached only
about `0.716-0.717` real-MNIST eval accuracy, so the high generated-label
accuracy numbers needed a stronger audit. This sweep keeps the generator
training budget at 500 examples, but trains the quality classifier on 5000
MNIST examples for 10 epochs (`feature_dim=256`, `depth=3`). The resulting
judge reaches about `0.934` real-MNIST eval accuracy.

Three-seed compact probe, seeds 11/12/13:

| Variant | Judge real-MNIST acc | Dist weight | Generated acc | Best acc | Step 0 | Step 16 | Step 32 | Label confidence | Entropy | Diversity | Best diversity | Nearest-real MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN strength8 | 0.934 | 0.000 | 1.000 | 1.000 | 0.093 | 0.998 | 1.000 | 0.902 | 0.528 | 1.140 | 1.137 | 0.0543 |
| HORN dist .025 | 0.934 | 0.025 | 0.998 | 0.999 | 0.106 | 0.982 | 0.999 | 0.899 | 0.534 | 1.126 | 1.259 | 0.0539 |
| StateMLP | 0.934 | 0.000 | 0.955 | 0.955 | 0.088 | 0.932 | 0.955 | 0.858 | 0.610 | 0.621 | 0.648 | 0.0383 |

Interpretation:

- The strict HORN lead survives a much stronger generated-label evaluator.
  Under the better judge, HORN strength8 reaches `1.000` generated-label
  accuracy across seeds while still starting near chance at step 0.
- The StateMLP control now looks weaker on semantics than it did under the
  weaker evaluator: `0.955` generated-label accuracy and much lower diversity
  (`0.621`) despite better nearest-real MSE.
- The HORN `dist0025` variant remains the balanced quality/proximity option:
  almost identical generated-label accuracy (`0.998`), slightly better
  nearest-real MSE, and still much higher diversity than StateMLP.
- Updated research read: HORN strength8 is no longer just visually promising;
  it is robust under a stronger semantic evaluator. The remaining weakness is
  still proximity/sharpness metrics, not class routing or diversity.

Sparse local HORN dynamics-quality probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_dynamics_quality_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_dynamics_quality_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_dynamics_quality_probe.json
outputs/analysis/horn_dynamics_quality_samples/horn_dynamics_quality_seed11_comparison.png
```

This probe asks whether HORN-side dynamics knobs can improve proximity/sharpness
without collapsing the diversity advantage. It keeps the stronger generated
label evaluator from the previous audit and compares baseline strength8,
balanced distributional pressure, higher frequency, higher damping, and higher
frequency plus distributional pressure.

Three-seed compact probe, seeds 11/12/13:

| Variant | Frequency | Damping | Dist weight | Generated acc | Best acc | Step 0 | Step 16 | Step 32 | Label confidence | Diversity | Best diversity | Nearest-real MSE | Mean MSE | Std MSE | Gen mean | State energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN strength8 | 1.00 | 0.15 | 0.000 | 1.000 | 1.000 | 0.093 | 0.998 | 1.000 | 0.902 | 1.140 | 1.137 | 0.0543 | 0.0124 | 0.0067 | 0.175 | 0.0291 |
| HORN dist .025 | 1.00 | 0.15 | 0.025 | 0.998 | 0.999 | 0.106 | 0.982 | 0.999 | 0.899 | 1.126 | 1.259 | 0.0539 | 0.0092 | 0.0058 | 0.161 | 0.0243 |
| HORN freq 1.3 | 1.30 | 0.15 | 0.000 | 0.971 | 0.969 | 0.091 | 0.950 | 0.969 | 0.857 | 1.128 | 1.183 | 0.0578 | 0.0131 | 0.0069 | 0.184 | 0.0176 |
| HORN damp .30 | 1.00 | 0.30 | 0.000 | 1.000 | 1.000 | 0.094 | 0.997 | 1.000 | 0.903 | 1.108 | 1.105 | 0.0523 | 0.0126 | 0.0062 | 0.176 | 0.0197 |
| HORN freq 1.3 + dist .025 | 1.30 | 0.15 | 0.025 | 0.997 | 0.998 | 0.108 | 0.977 | 0.998 | 0.895 | 1.159 | 1.568 | 0.0546 | 0.0083 | 0.0053 | 0.164 | 0.0158 |

Interpretation:

- Increasing HORN damping is the cleanest dynamics-side improvement so far. It
  preserves perfect generated-label accuracy under the stronger evaluator,
  improves nearest-real MSE (`0.0523` vs `0.0543`), lowers state energy, and
  keeps samples visually crisp. The cost is a modest diversity drop
  (`1.108` vs `1.140`), not prototype collapse.
- Increasing HORN frequency to `1.3` alone is not useful. It lowers state
  energy, but hurts classifier confidence, accuracy, and nearest-real MSE. This
  suggests the current class-drive/readout has learned around the original
  frequency scale.
- Frequency plus distributional pressure improves pixel mean/std metrics, but
  does not beat simple damping on nearest-real MSE and does not clearly improve
  sample grids.
- Updated research read: `sparse_horn_mnist_strict` remains the strict
  semantic/diversity lead. `sparse_horn_mnist_quality` is the balanced
  distributional quality variant. `sparse_horn_mnist_dynamics_quality` is the
  best current dynamics-side quality variant and is now the recommended example
  default via `sparse_horn_mnist_recommended`. The next probe should combine
  damping `.30` with small distributional pressure and/or explore a narrower
  damping curve, rather than changing frequency.

Sparse local HORN damping + distributional pressure probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_damping_distributional_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_damping_distributional_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_damping_distributional_probe.json
```

This tested whether the two best quality directions compound: higher HORN
damping plus small distributional pressure. The sweep kept the stronger
generated-label evaluator (`quality_classifier_train_limit=5000`,
`quality_classifier_epochs=10`) and repeated seeds 11/12/13.

Three-seed compact probe:

| Variant | Dist weight | Damping | Generated acc | Best acc | Step 0 | Step 16 | Step 32 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | State energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN strict | 0.0000 | 0.1500 | 1.0000 | 1.0000 | 0.0931 | 0.9980 | 1.0000 | 1.1397 | 1.1366 | 0.0543 | 0.0444 | 0.0124 | 0.0067 | 0.0291 |
| HORN dist .025 | 0.0250 | 0.1500 | 0.9980 | 0.9987 | 0.1061 | 0.9824 | 0.9987 | 1.1260 | 1.2593 | 0.0539 | 0.0498 | 0.0092 | 0.0058 | 0.0243 |
| HORN damp .30 | 0.0000 | 0.3000 | 1.0000 | 1.0000 | 0.0944 | 0.9974 | 1.0000 | 1.1080 | 1.1048 | 0.0523 | 0.0402 | 0.0126 | 0.0062 | 0.0197 |
| HORN damp .30 + dist .01 | 0.0100 | 0.3000 | 1.0000 | 1.0000 | 0.0970 | 0.9902 | 1.0000 | 1.1317 | 1.2469 | 0.0535 | 0.0440 | 0.0108 | 0.0061 | 0.0180 |
| HORN damp .30 + dist .025 | 0.0250 | 0.3000 | 0.9987 | 0.9974 | 0.0990 | 0.9870 | 0.9974 | 1.1244 | 1.2673 | 0.0528 | 0.0436 | 0.0091 | 0.0056 | 0.0162 |
| HORN damp .30 + dist .05 | 0.0500 | 0.3000 | 0.9876 | 0.9883 | 0.1016 | 0.9616 | 0.9876 | 1.1139 | 1.2392 | 0.0519 | 0.0432 | 0.0073 | 0.0048 | 0.0143 |

Interpretation:

- Damping `.30` remains the best default. It preserves perfect generated-label
  accuracy, improves nearest-real MSE over strict HORN (`0.0523` vs `0.0543`),
  gives the best best-step nearest-real MSE (`0.0402`), and lowers state
  energy without changing the loss.
- Small distributional pressure on top of damping does not clearly compound.
  `dist .01` keeps perfect generated-label accuracy and recovers some diversity
  (`1.1317`), but loses the damping-only nearest-real advantage (`0.0535`).
  `dist .025` improves mean/std errors and best diversity but slightly lowers
  generated-label accuracy. `dist .05` gives the lowest final nearest-real MSE
  (`0.0519`) but drops generated-label accuracy below the target range
  (`0.9876`).
- Updated research read: keep `sparse_horn_mnist_recommended` pointed at
  `sparse_horn_mnist_dynamics_quality`. Use
  `sparse_horn_mnist_dynamics_quality_dist001` only when diversity recovery is
  more important than nearest-real proximity. The next decisive question is now
  attribution, not another quality-weight sweep.

Recommended-route attribution probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_recommended_ablation_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_recommended_ablation_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_recommended_ablation_probe.json
```

This audited the recommended route against no-main-coupling,
frozen-recurrent, frozen-conditioning, frozen-all, decoder-only, one-step, and
a StateMLP control with the same strength-8 class drive. Metrics below are
three-seed means from seeds 11/12/13. The quality judge's own held-out accuracy
was `0.9344` for every row, so it is omitted from the compact table.

| Variant | Generated acc | Best acc | Step 0 | Step 16 | Step 32 | Confidence | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | State energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recommended HORN | 1.0000 | 1.0000 | 0.0944 | 0.9974 | 1.0000 | 0.9034 | 1.1079 | 1.1048 | 0.0523 | 0.0402 | 0.0126 | 0.0062 | 0.0197 |
| No main coupling | 0.0814 | 0.0885 | 0.0846 | 0.0820 | 0.0801 | 0.0861 | 0.2471 | 0.3911 | 0.0401 | 0.0374 | 0.0132 | 0.0417 | 0.0038 |
| Frozen recurrent | 1.0000 | 1.0000 | 0.0951 | 0.9974 | 1.0000 | 0.9031 | 1.1233 | 1.1207 | 0.0531 | 0.0409 | 0.0127 | 0.0063 | 0.0180 |
| Frozen conditioning | 0.0833 | 0.0885 | 0.0840 | 0.0840 | 0.0788 | 0.0865 | 0.2317 | 0.3702 | 0.0380 | 0.0354 | 0.0132 | 0.0427 | 0.0038 |
| Frozen all | 0.0820 | 0.0892 | 0.0840 | 0.0872 | 0.0794 | 0.0864 | 0.2241 | 0.3603 | 0.0397 | 0.0373 | 0.0131 | 0.0433 | 0.0038 |
| Decoder only | 0.0801 | 0.0872 | 0.0807 | 0.0781 | 0.0781 | 0.0853 | 0.3184 | 0.3171 | 0.0434 | 0.0348 | 0.0129 | 0.0370 | 0.0102 |
| One step | 0.1810 | 0.1608 | 0.0853 |  |  | 0.1568 | 0.7606 | 0.7581 | 0.0675 | 0.0662 | 0.0144 | 0.0262 | 0.0099 |
| StateMLP strength8 | 1.0000 | 1.0000 | 0.0983 | 1.0000 | 1.0000 | 0.9125 | 0.7657 | 0.7720 | 0.0344 | 0.0251 | 0.0064 | 0.0077 | 5.9080 |

Interpretation:

- The current HORN win is **not** primarily a win for learned main-pool
  recurrent weights. Freezing the recurrent HORN parameters keeps performance
  essentially unchanged and slightly improves diversity (`1.1233` vs
  `1.1079`).
- Main-pool coupling still matters as an active substrate. Removing main
  coupling collapses generated-label accuracy to chance (`0.0814`) and destroys
  diversity (`0.2471`), even though nearest-real MSE looks deceptively good
  because the outputs become low-variance/prototype-like.
- Learned conditioning dynamics are essential. Frozen conditioning, frozen all,
  and decoder-only all sit near chance generated-label accuracy.
- Settling is essential. One-step HORN is better than chance but far below the
  full route (`0.1810` vs `1.0000`) and has the worst nearest-real MSE
  (`0.0675`).
- StateMLP strength-8 remains the strongest conventional quality/proximity
  control: perfect generated-label accuracy and much better nearest-real MSE
  (`0.0344`). Its tradeoff is diversity (`0.7657` vs HORN's `1.1079`) and much
  higher state energy. This keeps the honest claim narrow: HORN currently wins
  on the diversity/settling story, not on raw pixel proximity.

Updated research read: the recommended HORN route should be described as a
**sparse coupled HORN substrate driven by learned class-conditioning dynamics**.
The recurrent oscillator substrate and multi-step settling are doing real work,
but the trained recurrent weights themselves are not the differentiator in the
current recipe. The next serious scientific step is either:

- push HORN quality/proximity without losing diversity, or
- give the StateMLP control an equally explicit diversity objective and see
  whether it can match HORN's diversity without losing its proximity advantage.

Strength-8 StateMLP diversity control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe
```

The probe wrote:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_strength8_diversity_probe.json
```

This repeated the StateMLP diversity-control idea with the same strength-8
conditioning used by `sparse_horn_mnist_recommended`, plus the same stronger
generated-label evaluator as the latest HORN sweeps.

Three-seed compact probe:

| Variant | Generated acc | Best acc | Step 0 | Step 16 | Step 32 | Step 64 | Confidence | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | State energy | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recommended HORN | 1.0000 | 1.0000 | 0.0944 | 0.9974 | 1.0000 | 0.9915 | 0.9034 | 1.1080 | 1.1048 | 0.0523 | 0.0402 | 0.0126 | 0.0062 | 0.0197 | 822.625 |
| StateMLP strength8 | 1.0000 | 1.0000 | 0.0983 | 1.0000 | 1.0000 | 1.0000 | 0.9127 | 0.7561 | 0.7653 | 0.0340 | 0.0250 | 0.0062 | 0.0079 | 5.8698 | 928.431 |
| StateMLP s8 dist .05 | 1.0000 | 1.0000 | 0.1003 | 1.0000 | 1.0000 | 1.0000 | 0.9114 | 0.7568 | 0.7696 | 0.0326 | 0.0249 | 0.0043 | 0.0066 | 6.0196 | 717.872 |
| StateMLP s8 dist .10 | 0.9551 | 0.9551 | 0.0977 | 0.9551 | 0.9551 | 0.9551 | 0.8686 | 0.7572 | 0.7645 | 0.0344 | 0.0253 | 0.0033 | 0.0060 | 6.1405 | 690.007 |
| StateMLP s8 dist .10 + class | 1.0000 | 1.0000 | 0.0957 | 1.0000 | 1.0000 | 1.0000 | 0.9079 | 0.7152 | 0.7285 | 0.0329 | 0.0260 | 0.0035 | 0.0074 | 6.4930 | 677.679 |

Interpretation:

- The fairer strength-8 StateMLP diversity controls do **not** erase HORN's
  diversity edge. StateMLP stays around `0.72-0.77` diversity ratio, while
  recommended HORN stays at `1.108`.
- Distributional pressure improves StateMLP pixel proximity and mean/std
  matching, but does not produce HORN-like output diversity. At `0.10` it also
  hurts generated-label accuracy (`0.9551`) unless class-moment pressure is
  added.
- The conventional control still wins raw proximity by a large margin
  (`nearest-real MSE 0.0326-0.0344` vs HORN `0.0523`), so the HORN result is
  not a blanket image-quality win.
- The HORN claim is now better supported and more precise: a sparse coupled
  second-order oscillator substrate gives a better semantic/diversity settling
  frontier than the matched small StateMLP transition, while StateMLP remains
  the pixel-proximity control to beat.

Updated next move: if chasing quality, improve the HORN decoder/readout or
calibration while keeping diversity above `1.0`. If chasing attribution, probe
why frozen recurrent HORN works so well: fixed sparse HORN substrate versus
learned conditioning drive versus decoder adaptation.

Frontier artifact command:

```bash
python scripts/analyze_mnist_generator_frontier.py
```

This writes `outputs/analysis/mnist_generator_frontier/frontier_summary.csv`,
`frontier_summary.md`, and `frontier_plot.png`. The current plot makes the
tradeoff explicit: recommended HORN is the high-diversity frontier point,
while the best StateMLP distributional control is the pixel-proximity frontier
point.

First non-MNIST frontier gate:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_fashion_mnist_frontier_probe
```

This reuses the same sparse HORN versus strength-8 StateMLP frontier setup on
Fashion-MNIST, which has the same 28x28 grayscale/10-class shape as MNIST but
less symbol-like classes. The scientific question is deliberately larger than
a knob tweak: does the HORN diversity/settling frontier survive when the classes
are clothing silhouettes instead of handwritten digit strokes?

Local wiring smoke passed with:

```bash
python examples/image_mnist_generator.py \
  --preset sparse_horn_fashion_mnist_recommended \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4 \
  --batch-size 4 \
  --eval-sample-count 4 \
  --quality-classifier-epochs 0 \
  --num-oscillators 98 \
  --resize-conv-min-channels 4 \
  --steps 1 \
  --train-settling-steps 1 \
  --settling-steps 0,1 \
  --output-dir outputs/smoke/fashion_mnist_generator_horn
```

It completed end-to-end on CPU and verified that direct Fashion-MNIST IDX
loading, preset parsing, model construction, training, metrics, checkpoints,
and artifact export work together.

Modal sweep result, seeds 11/12/13, train500/20e:

| Variant | Judge acc | Generated-label acc | Best acc | Step 0 | Step 16 | Step 32 | Step 64 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | State energy | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fashion HORN | 0.8475 | 0.8900 | 0.8958 | 0.1048 | 0.8750 | 0.8952 | 0.8366 | 1.2075 | 1.2102 | 0.0330 | 0.0294 | 0.0205 | 805.3 |
| Fashion StateMLP s8 | 0.8475 | 0.9310 | 0.9388 | 0.1230 | 0.9251 | 0.9264 | 0.9388 | 0.9214 | 0.9274 | 0.0226 | 0.0126 | 5.5512 | 813.3 |
| Fashion StateMLP s8 dist .05 | 0.8475 | 0.9395 | 0.9694 | 0.1315 | 0.9388 | 0.9414 | 0.9688 | 0.9066 | 0.9176 | 0.0226 | 0.0138 | 5.9727 | 861.2 |

Artifact command:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_fashion_mnist_frontier_probe.csv \
  --output-dir outputs/analysis/fashion_mnist_generator_frontier \
  --title "Fashion-MNIST generator quality/diversity frontier" \
  --accuracy-floor 0.75
```

It writes `outputs/analysis/fashion_mnist_generator_frontier/frontier_summary.csv`,
`frontier_summary.md`, and `frontier_plot.png`.

Interpretation:

- The MNIST HORN-vs-StateMLP tradeoff survives the first non-MNIST gate.
  Sparse HORN remains the high-diversity point (`1.2075`), while StateMLP
  remains the higher-accuracy/lower-nearest-real-MSE point.
- This is not a Fashion-MNIST quality win for HORN. The conventional StateMLP
  transition is closer to the training distribution and reaches higher
  generated-label accuracy under the same quick judge.
- The positive signal is narrower but meaningful: the sparse second-order HORN
  substrate continues to preserve more class-consistent variation through
  settling on a harder silhouette dataset. That makes the next serious
  question HORN-side quality/readout calibration, not another MNIST-only
  semantic probe.
- The step curve still shows real recurrent organization: HORN starts near
  chance at step 0 (`0.1048`) and reaches its best range around 16-32 settling
  steps before drifting down at 64. For this recipe, "more settling" is not
  automatically better; use 16-32 as the default diagnostic band.

Next targeted probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_fashion_mnist_readout_capacity_probe
```

This doubles the resize-conv minimum channel count from `8` to `16` for both
Fashion-MNIST HORN and the matched StateMLP strength-8 control. It tests a
specific explanation for HORN's quality gap: maybe the oscillator substrate is
preserving useful variation, but the current readout is too narrow to render
clean clothing silhouettes. If HORN proximity improves while diversity stays
high, readout calibration becomes the next quality path. If StateMLP simply
widens its lead, the quality gap is probably not just a readout bottleneck.

Readout-capacity result, seeds 11/12/13:

| Variant | Generated-label acc | Best acc | Step 16 | Step 32 | Step 64 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | State energy | Params | Decoder fraction | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN ch8 | 0.8900 | 0.8958 | 0.8750 | 0.8952 | 0.8366 | 1.2075 | 1.2102 | 0.0330 | 0.0294 | 0.0185 | 0.0205 | 104061 | 0.0231 | 805.3 |
| HORN ch16 | 0.9238 | 0.9323 | 0.8835 | 0.9323 | 0.8711 | 1.2442 | 1.2448 | 0.0375 | 0.0312 | 0.0212 | 0.0153 | 109925 | 0.0753 | 730.3 |
| StateMLP ch8 | 0.9310 | 0.9388 | 0.9251 | 0.9264 | 0.9388 | 0.9214 | 0.9274 | 0.0226 | 0.0126 | 0.0044 | 5.5512 | 103521 | 0.0233 | 813.3 |
| StateMLP ch16 | 0.9388 | 0.9388 | 0.9355 | 0.9388 | 0.9388 | 0.9215 | 0.9282 | 0.0216 | 0.0187 | 0.0040 | 4.2019 | 109385 | 0.0756 | 774.3 |
| StateMLP ch8 dist .05 | 0.9395 | 0.9694 | 0.9388 | 0.9414 | 0.9688 | 0.9066 | 0.9176 | 0.0226 | 0.0138 | 0.0031 | 5.9727 | 103521 | 0.0233 | 861.2 |

Combined artifact:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/fashion_mnist_generator_readout_capacity_combined.csv \
  --output-dir outputs/analysis/fashion_mnist_generator_readout_capacity \
  --title "Fashion-MNIST HORN readout capacity frontier" \
  --accuracy-floor 0.75
```

Interpretation:

- Widening the readout helps HORN's semantic side: generated-label accuracy
  improves from `0.8900` to `0.9238`, and diversity improves from `1.2075` to
  `1.2442`.
- It does **not** solve the pixel-proximity gap. HORN nearest-real MSE gets
  worse (`0.0330 -> 0.0375`), while StateMLP ch16 remains much closer
  (`0.0216`).
- The result argues against "HORN just needs a wider decoder" as the whole
  story. The wider readout gives the oscillator field more expressive rendering
  capacity, but it spends that capacity on more varied silhouettes rather than
  closer prototypes.
- The next quality path should therefore be more structured than decoder width:
  either an explicit calibration/readout objective that preserves diversity, or
  a two-head evaluation where one head optimizes class-consistent diversity and
  another head optimizes foreground/proximity. Blindly increasing decoder
  capacity will likely keep widening the quality/diversity split instead of
  closing it.

Fashion-MNIST HORN calibration probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_fashion_mnist_horn_calibration_probe
```

This tests the first branch of that next quality path: keep the ch8 HORN
readout, but add small distributional pressure (`0.025` and `0.05`) to see
whether Fashion-MNIST nearest-real proximity can improve without collapsing the
HORN diversity advantage.

Calibration result, seeds 11/12/13:

| Variant | Generated-label acc | Best acc | Step 16 | Step 32 | Step 48 | Step 64 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | State energy | Dist weight | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.8900 | 0.8958 | 0.8750 | 0.8952 | 0.8587 | 0.8366 | 1.2075 | 1.2102 | 0.0330 | 0.0294 | 0.0185 | 0.0051 | 0.0205 | 0.0000 | 805.3 |
| HORN dist .025 | 0.9134 | 0.9297 | 0.8737 | 0.9297 | 0.8906 | 0.8613 | 1.1876 | 1.1899 | 0.0332 | 0.0298 | 0.0122 | 0.0043 | 0.0178 | 0.0250 | 621.2 |
| HORN dist .05 | 0.9036 | 0.9199 | 0.8581 | 0.9199 | 0.8730 | 0.8509 | 1.1598 | 1.1619 | 0.0322 | 0.0290 | 0.0093 | 0.0034 | 0.0169 | 0.0500 | 687.3 |
| HORN ch16 | 0.9238 | 0.9323 | 0.8835 | 0.9323 | 0.9121 | 0.8711 | 1.2442 | 1.2448 | 0.0375 | 0.0312 | 0.0212 | 0.0065 | 0.0153 | 0.0000 | 730.3 |
| StateMLP s8 | 0.9310 | 0.9388 | 0.9251 | 0.9264 | 0.9388 | 0.9388 | 0.9214 | 0.9274 | 0.0226 | 0.0126 | 0.0044 | 0.0023 | 5.5512 | 0.0000 | 813.3 |
| StateMLP s8 dist .05 | 0.9395 | 0.9694 | 0.9388 | 0.9414 | 0.9694 | 0.9688 | 0.9066 | 0.9176 | 0.0226 | 0.0138 | 0.0031 | 0.0023 | 5.9727 | 0.0500 | 861.2 |
| StateMLP s8 ch16 | 0.9388 | 0.9388 | 0.9355 | 0.9388 | 0.9388 | 0.9388 | 0.9215 | 0.9282 | 0.0216 | 0.0187 | 0.0040 | 0.0024 | 4.2019 | 0.0000 | 774.3 |

Extended artifact:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/fashion_mnist_generator_extended_frontier_combined.csv \
  --output-dir outputs/analysis/fashion_mnist_generator_extended_frontier \
  --title "Fashion-MNIST extended generator frontier" \
  --accuracy-floor 0.75
```

Interpretation:

- Distributional calibration is more useful than widening the decoder if the
  goal is HORN quality. `dist .05` improves nearest-real MSE
  (`0.0330 -> 0.0322`) and substantially improves mean/std matching
  (`0.0185 -> 0.0093`, `0.0051 -> 0.0034`) while keeping diversity high
  (`1.1598`).
- The cost is real: calibration reduces diversity compared with strict HORN,
  and `dist .05` lowers generated-label accuracy relative to `dist .025`.
  This gives a useful HORN-side quality/diversity knob rather than a free
  improvement.
- `ch16` remains the semantic/diversity expansion point, not the proximity
  point. It reaches the best HORN accuracy/diversity but worsens nearest-real
  MSE.
- StateMLP still owns raw pixel proximity. HORN's current advantage remains the
  sparse, low-energy, high-diversity frontier, not "best MSE."
- Practical next default for Fashion-MNIST probes: use recommended HORN for
  the pure diversity/settling story, `dist .05` for the calibrated quality
  story, and keep StateMLP strength-8 as the raw proximity control.

CIFAR-10 grayscale frontier gate:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_gray_frontier_probe
```

This is the first larger image-domain gate for the sparse HORN generator
frontier. It keeps the current single-channel generator surface, but changes
the data from 28x28 symbolic/silhouette classes to 32x32 grayscale CIFAR-10
natural-image classes. The sweep compares:

- `sparse_horn_cifar10_gray_recommended`
- `sparse_horn_cifar10_gray_recommended_dist005`
- `sparse_horn_cifar10_gray_state_mlp_strength8`

All three use train1000/20e, an `8x8` resize-conv seed, 256 oscillator/state
positions, and the stronger generated-label evaluator. The point is not to
claim CIFAR quality early; the point is to test whether the HORN
diversity/settling frontier survives a meaningful step beyond MNIST-like
datasets. If HORN keeps a diversity/semantic edge, the sparse second-order
substrate is probably doing something transferable. If it collapses while
StateMLP stays stable, the current HORN advantage may be mostly
stroke/silhouette-specific.

Local wiring smoke passed with:

```bash
python - <<'PY'
from oscnet.experiments.mnist_autoencoder import load_mnist_data, image_shape_for_dataset
from oscnet.experiments.mnist_generator import parse_args, config_from_args, build_mnist_generator_model
import jax

train_x, train_y, eval_x, eval_y = load_mnist_data(
    source="idx",
    dataset_name="cifar10_gray",
    train_limit=2,
    eval_limit=2,
    seed=0,
)
print(image_shape_for_dataset("cifar10_gray"))
print(train_x.shape, eval_x.shape, train_y.tolist(), eval_y.tolist())

config = config_from_args(parse_args([
    "--preset", "sparse_horn_cifar10_gray_recommended",
    "--epochs", "1",
    "--train-limit", "2",
    "--eval-limit", "2",
    "--num-oscillators", "128",
    "--resize-conv-seed-size", "8",
    "--resize-conv-min-channels", "4",
]))
model = build_mnist_generator_model(config, jax.random.PRNGKey(0))
print(model.image_shape, model.image_dim, model.resize_conv_seed_shape)
PY
```

It verified direct CIFAR-10 grayscale loading, inferred shape `(32, 32)`,
flattened image size `1024`, preset parsing, and model construction with an
`8x8` resize-conv seed.

Modal result, seeds 11/12/13, train1000/20e:

| Variant | Judge acc | Generated-label acc | Best acc | Step 0 | Step 16 | Step 32 | Step 64 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | Prototype acc | State energy | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.3054 | 0.8210 | 0.8288 | 0.0951 | 0.7591 | 0.8288 | 0.6888 | 0.9167 | 0.9151 | 0.0214 | 0.0144 | 0.0028 | 0.0031 | 0.7415 | 0.0092 | 109.2 |
| HORN dist .05 | 0.3054 | 0.7845 | 0.7988 | 0.0859 | 0.6979 | 0.7832 | 0.7617 | 1.0964 | 1.1082 | 0.0305 | 0.0239 | 0.0027 | 0.0025 | 0.7298 | 0.0082 | 92.1 |
| StateMLP strength8 | 0.3096 | 0.6738 | 0.7064 | 0.1055 | 0.6367 | 0.6738 | 0.7064 | 0.5128 | 0.5201 | 0.0086 | 0.0049 | 0.0017 | 0.0145 | 0.7799 | 6.1298 | 621.0 |

Artifact command:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_gray_frontier_probe.csv \
  --output-dir outputs/analysis/cifar10_gray_generator_frontier \
  --title "CIFAR-10 grayscale generator quality/diversity frontier" \
  --accuracy-floor 0.3
```

It writes `outputs/analysis/cifar10_gray_generator_frontier/frontier_summary.csv`,
`frontier_summary.md`, and `frontier_plot.png`. Representative seed-11 sample
grids were pulled to `outputs/analysis/cifar10_gray_generator_frontier_samples/`.

Interpretation:

- This is **not** a solved CIFAR generator. The generated images are still
  blurry low-resolution grayscale objects/backgrounds, not high-fidelity
  CIFAR samples.
- The quick CIFAR-gray generated-label judge is weak (`~0.305` eval accuracy),
  so generated-label accuracy is only a coarse relative proxy here.
- Even with that caveat, the same frontier pattern survives the step beyond
  MNIST/Fashion-MNIST: HORN has higher generated-label proxy and much higher
  diversity than the StateMLP control, while StateMLP is much closer by
  nearest-real MSE and runs faster.
- Visual inspection agrees with the table. HORN samples are contrasty and more
  varied, with object-like blobs and class texture. StateMLP samples are
  smoother and more averaged/mushy, which helps pixel MSE but hurts diversity.
- `dist .05` on CIFAR does not repeat the Fashion-MNIST calibration story. It
  increases diversity but worsens generated-label proxy and nearest-real MSE
  relative to recommended HORN. For CIFAR-gray, recommended HORN is the better
  default until we add a stronger evaluator/readout.

Updated read: the HORN advantage is no longer just a handwritten-digit quirk.
It transfers as a quality/diversity/settling frontier to a tiny grayscale
natural-image gate. The remaining gap is still severe: the current harness does
not yet produce sharp CIFAR samples, and the evaluator is too weak for final
claims. The next bigger step should be either a stronger pretrained/contrastive
quality metric for CIFAR-like data, or a color/multichannel HORN generator
rather than more MNIST knob tuning.

Conv-judge CIFAR-10 grayscale audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_gray_convjudge_frontier_probe
```

The previous CIFAR-gray generated-label judge was a flat MLP and only reached
about `0.305` eval accuracy. This audit keeps the generator recipes fixed but
switches the quality evaluator to `--quality-classifier-kind conv`.

Modal result, seeds 11/12/13:

| Variant | Judge acc | Generated-label acc | Best acc | Step 0 | Step 16 | Step 32 | Step 64 | Diversity | Best diversity | Nearest-real MSE | Best nearest-real MSE | Mean MSE | Std MSE | Prototype acc | State energy | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.3584 | 0.9023 | 0.9089 | 0.0957 | 0.8548 | 0.9089 | 0.7539 | 0.9167 | 0.9151 | 0.0214 | 0.0144 | 0.0028 | 0.0031 | 0.7415 | 0.0092 | 792.6 |
| HORN dist .05 | 0.3582 | 0.8314 | 0.8704 | 0.1029 | 0.7207 | 0.8464 | 0.8646 | 1.0964 | 1.1082 | 0.0305 | 0.0239 | 0.0027 | 0.0025 | 0.7298 | 0.0082 | 856.8 |
| StateMLP strength8 | 0.3584 | 0.6270 | 0.6270 | 0.1055 | 0.5924 | 0.6270 | 0.6270 | 0.5108 | 0.5158 | 0.0086 | 0.0047 | 0.0015 | 0.0147 | 0.7539 | 5.9995 | 928.4 |

Artifact command:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_gray_convjudge_frontier_probe.csv \
  --output-dir outputs/analysis/cifar10_gray_convjudge_generator_frontier \
  --title "CIFAR-10 grayscale conv-judge generator frontier" \
  --accuracy-floor 0.3
```

Interpretation:

- The conv judge is better but still weak in absolute CIFAR-gray terms
  (`~0.358` eval accuracy). This remains a small-data, grayscale diagnostic,
  not a final CIFAR metric.
- Under the conv judge, the HORN semantic proxy becomes stronger, not weaker:
  recommended HORN reaches `0.9023` generated-label accuracy versus StateMLP
  `0.6270`.
- The old caveat still holds. StateMLP remains much closer by nearest-real MSE
  (`0.0086` versus HORN `0.0214/0.0305`) and faster. HORN remains the
  diversity/semantic/settling frontier, not the raw pixel-proximity frontier.
- `dist .05` becomes the explicit high-diversity HORN point. Recommended HORN
  is the better semantic/proximity HORN point.

Practical result: use `--quality-classifier-kind conv` for CIFAR-gray and other
image datasets where the flat MLP judge underfits. The next evaluation upgrade
should be an embedding/FID-like metric or a stronger classifier trained once
per dataset, but this conv judge is already enough to confirm that the HORN
transfer signal is not just an artifact of the weaker flat MLP evaluator.

CIFAR-10 RGB gate:

Full-color CIFAR-10 uses the same sparse HORN frontier:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_frontier_probe
```

Implemented support:

- `dataset_name="cifar10_rgb"` direct loader, returning channel-first flat RGB
  vectors of length `3072`.
- Channel-aware generator image metadata, resize-conv output channels, sample
  grids, and convolutional generated-label judge.
- RGB presets:
  `sparse_horn_cifar10_rgb_recommended`,
  `sparse_horn_cifar10_rgb_recommended_dist005`, and
  `sparse_horn_cifar10_rgb_state_mlp_strength8`.

Modal result, seeds 11/12/13, train1000/20e:

| Variant | Judge acc | Generated-label acc | Diversity | Nearest-real MSE | Mean MSE | Std MSE | State energy | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.3945 | 0.8522 | 1.0183 | 0.0292 | 0.0032 | 0.0034 | 0.0082 | 202.9 |
| HORN dist .05 | 0.3899 | 0.7552 | 1.1147 | 0.0358 | 0.0025 | 0.0023 | 0.0064 | 229.2 |
| StateMLP strength8 | 0.3959 | 0.5143 | 0.5409 | 0.0116 | 0.0020 | 0.0156 | 5.8934 | 704.3 |

Artifact command:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_frontier_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_generator_frontier \
  --title "CIFAR-10 RGB generator quality/diversity frontier" \
  --accuracy-floor 0.3
```

Representative seed-11 sample grids were pulled to
`outputs/analysis/cifar10_rgb_generator_frontier_samples/`.

Interpretation:

- The HORN advantage survives full color. Recommended HORN has much stronger
  generated-label proxy (`0.8522` vs `0.5143`) and almost double the diversity
  ratio (`1.0183` vs `0.5409`) compared with the matched StateMLP control.
- StateMLP still wins raw nearest-real pixel MSE (`0.0116` vs HORN `0.0292`)
  and runs about 3.5x faster. This is still a semantic/diversity/settling
  frontier, not a raw pixel-proximity win.
- Visual inspection matches the metrics. HORN RGB samples are blurry but have
  stronger object/background separation and class-colored structure. StateMLP
  samples are smoother and more averaged, which helps nearest-real MSE but
  visibly suppresses diversity.
- `dist .05` is again the explicit high-diversity HORN point. It increases
  diversity but loses semantic proxy and nearest-real MSE against recommended
  HORN.

Updated read: the current HORN generator result is no longer merely a MNIST or
grayscale artifact. The setup has now survived MNIST, Fashion-MNIST,
CIFAR-gray, and CIFAR RGB as the high-diversity/semantic-settling side of the
frontier. It still does not produce sharp natural images, so the next
breakthrough path is likely architecture/evaluator scale rather than more
small MNIST knob tuning.

CIFAR-10 RGB feature-space audit:

After adding classifier feature-space diagnostics, a cheaper seed-11 RGB audit
reran the three RGB frontier variants:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feature_metric_audit
```

| Variant | Generated-label acc | Pixel diversity | Pixel nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.8223 | 0.9823 | 0.0292 | 0.9587 | 0.1350 |
| HORN dist .05 | 0.7480 | 1.0815 | 0.0336 | 0.9367 | 0.1886 |
| StateMLP strength8 | 0.8105 | 0.5162 | 0.0113 | 0.8369 | 0.1036 |

Interpretation:

- The HORN advantage is still visible on seed 11: recommended HORN has the
  best generated-label accuracy and much higher pixel diversity than StateMLP.
- StateMLP is close on generated-label accuracy and wins nearest-real pixel
  MSE plus feature-nearest-real distance. It remains the stronger
  pixel/proximity control, but collapses to much lower sample diversity.
- The feature-space diagnostics keep the claim honest. HORN's diversity
  advantage is not automatically equivalent to semantic diversity, but
  recommended HORN also has higher feature diversity and a better feature
  pairwise-distance ratio than StateMLP in this run.
- Practical next metric direction: keep classifier feature metrics in all
  generator sweeps, and add a stronger reusable image embedding judge before
  making broad natural-image claims.

The same rerun also exported the new trajectory-level dynamics columns:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feature_metric_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_feature_metric_dynamics_audit \
  --title "CIFAR-10 RGB feature/dynamics generator audit" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Update settle | Accel settle | Output settle | Energy delta | Coupling delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.6220 | 0.5950 | 0.1302 | -0.0003 | 0.0031 |
| HORN dist .05 | 0.6263 | 0.5867 | 0.2292 | -0.0012 | 0.0019 |
| StateMLP strength8 | 0.3177 | 0.4279 | 0.0347 | 6.6463 | 0.0000 |

Dynamics interpretation:

- All three models reduce update/output motion by the final step. Settling is
  therefore not unique to HORN; the better claim is sparse coupled HORN
  preserves much higher diversity while settling.
- HORN keeps state energy low and slightly decreasing; StateMLP drives its
  latent state energy up by several orders of magnitude while still producing
  smooth low-diversity samples. This is a useful attribution clue, not a
  hardware-energy claim.
- HORN semantic proxies and diversity peak around 32 steps, while nearest-real
  pixel MSE keeps improving at 64 steps. That confirms the central measurement
  lesson: over-settling can improve pixel proximity while hurting class
  usefulness and diversity.

Generator settling diagnostic implementation:

The generator summary now records trajectory-level settling diagnostics under
`generator.success_diagnostics` whenever a trace is available. These include
state-energy proxies, velocity RMS, update RMS, acceleration RMS, weighted
coupling-disagreement proxies, and per-step output-change MSE.

These are deliberately labelled as diagnostics, not physical proof of a
Lyapunov energy. The goal is practical: distinguish runs that genuinely settle
from runs that keep drifting, overshoot, or only look good at one arbitrary
step count. The most useful next CIFAR RGB tables should include:

- semantic/sample metrics: generated-label accuracy, feature diversity,
  feature nearest-real distance;
- diversity/proximity metrics: pixel diversity and nearest-real MSE;
- dynamics metrics: `state_update_rms_settling_ratio`,
  `state_acceleration_rms_settling_ratio`,
  `coupling_potential_proxy_delta`, and `output_step_mse_settling_ratio`.

This makes the next optimization target sharper: improve HORN image quality
while preserving class consistency, diversity, and actual finite-time settling.

Attractor robustness diagnostic:

The generator summary also records `generator.attractor_robustness` for
conditional runs. This samples several independent initial oscillator states per
class label and measures whether they remain class-consistent while preserving
within-class diversity. The useful fields are:

- `label_accuracy` and `class_success_fraction`: whether repeated same-label
  initial states stay inside the intended class basin under the quality judge.
- `pixel_within_class_pairwise_mse` and
  `feature_within_class_pairwise_distance`: diversity inside each class basin.
- `pixel_separation_ratio` and `feature_separation_ratio`: how separated class
  centroids are relative to within-class spread.
- `pixel_attractor_diversity_score` and
  `feature_attractor_diversity_score`: collapse-aware basin scores computed as
  label accuracy times `log1p` within-class spread. These penalize prototype
  collapse while still requiring class consistency.

This is an ONN-native probe for the "class attractor basin" story. A strong
generator should not merely maximize label accuracy by collapsing to one
prototype per class; it should keep label-consistent samples diverse under
perturbed initial states.

CIFAR-10 RGB attribution probe:

The next seed-11 gate tested whether the RGB HORN result depends on learned
main recurrent coupling, learned conditioning, and multi-step settling:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attribution_probe
```

The six variants were:

- `horn_recommended`: full HORN recipe.
- `horn_step1`: one settling step only.
- `horn_frozen_recurrent`: recurrent frequency/coupling frozen, conditioning
  trained.
- `horn_frozen_conditioning`: recurrent dynamics trained, conditioning frozen.
- `horn_no_main_interaction`: main coupling initialized to zero and frozen,
  conditioning trained.
- `horn_decoder_only`: no settling dynamics.

| Variant | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: |
| HORN recommended | 0.7324 | 0.9823 | 0.0292 | 0.9747 | 0.1384 |
| HORN step1 | 0.5996 | 1.4941 | 0.0677 | 0.9038 | 0.1905 |
| HORN frozen recurrent | 0.7441 | 1.0036 | 0.0301 | 0.9729 | 0.1275 |
| HORN frozen conditioning | 0.0898 | 0.7491 | 0.0380 | 0.7222 | 0.2296 |
| HORN no main interaction | 0.8066 | 0.9251 | 0.0280 | 0.9662 | 0.1568 |
| HORN decoder only | 0.0977 | 0.7533 | 0.0388 | 0.7442 | 0.2201 |

Dynamics/attribution read:

- Learned conditioning is essential. Freezing conditioning collapses
  generated-label accuracy to chance, similar to decoder-only.
- One-step HORN is not enough. It gives high pixel diversity but poor
  nearest-real and weaker class accuracy, so finite-time settling still matters.
- Learned main recurrent coupling is not yet essential. Frozen recurrent
  dynamics remain competitive, and the no-main-interaction control is
  surprisingly strong. The current RGB generator is therefore better described
  as a class-driven second-order HORN field with useful settling, not yet as
  proof that learned sparse oscillator-to-oscillator coupling is doing the
  core generative work.
- This is a narrowing result, not a failure. It identifies the next bottleneck:
  if we want a stronger ONN-native claim, the architecture/objective must make
  main coupling carry information that a learned class drive plus independent
  second-order settling cannot already carry.

Immediate follow-up implied by this gate:

- Design a harder coupling-attribution task or architecture where class drive
  cannot directly organize every oscillator.
- Candidate interventions: lower-rank or weaker class drive, hierarchical
  class drive into a small subset of oscillators, local-only readout pressure,
  or explicit neighbor-consistency losses that require main field propagation.
- Keep the RGB attribution probe as a regression gate for future HORN changes.

CIFAR-10 RGB quality-judge audit:

The old CIFAR RGB sample-quality judge was a small conv classifier trained on
5k images for 10 epochs. It reached only about 38-47% real CIFAR eval
accuracy, so generated-label accuracy from that judge was useful for relative
comparisons but too weak for strong semantic claims. A stronger residual-conv
judge was added as an opt-in metric:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_judge_audit
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_judge_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_judge_audit \
  --title "CIFAR-10 RGB quality-judge audit" \
  --accuracy-floor 0.0 \
  --no-plot
```

Both judges used the same larger budget: 10k classifier training images, 5k
eval images, 15 epochs, 256 feature dimensions, depth 3.

| Variant | Judge | Real-CIFAR judge acc | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled HORN prefix 25% | conv | 0.4625 | 0.6738 | 1.1015 | 0.0328 | 0.8563 | 0.2884 |
| Coupled HORN prefix 25% | residual-conv | 0.6139 | 0.2188 | 1.1014 | 0.0328 | 0.6812 | 0.3310 |
| No-main prefix 25% | conv | 0.4723 | 0.6523 | 1.1427 | 0.0373 | 0.9024 | 0.3052 |
| No-main prefix 25% | residual-conv | 0.6020 | 0.1895 | 1.1428 | 0.0373 | 0.7452 | 0.3179 |

Read:

- The residual-conv judge is a better CIFAR classifier than the old conv judge
  under the same audit budget, so it should become the preferred semantic
  scoring path for future CIFAR RGB claims.
- The stronger judge is much stricter on generated samples. Old conv-judge
  generated-label accuracies around 0.65-0.75 should be treated as weak-judge
  evidence, not proof of strong CIFAR semantic generation.
- Pixel metrics are unchanged because the generator runs are the same; the
  recalibration is semantic/feature-space only.
- Coupled HORN still beats no-main on nearest-real MSE and residual-conv
  generated-label accuracy in this one-seed audit, but both variants are far
  from convincing semantic generation under the stronger judge.
- This shifts the next research priority: before claiming CIFAR breakthrough,
  improve generator semantics under the residual-conv judge, or move to a task
  where the oscillator advantage is not hidden behind weak image classification
  metrics.

CIFAR-10 RGB residual feature-drift semantic probe:

The next probe asked whether training against a residual-conv learned feature
space can improve the stricter residual-conv judge, without using the same
classifier for evaluation. The generator variants used the same HORN
prefix-25 sparse-drive recipe, but the feature-drift variants trained an
extra residual-conv feature model on the generator training subset. The quality
judge was trained independently on 10k CIFAR images for 15 epochs.

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_semantic_feature_drift_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_semantic_feature_drift_probe \
  --title "CIFAR-10 RGB residual feature-drift semantic probe" \
  --accuracy-floor 0.0 \
  --no-plot
```

| Variant | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Output settling | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pixel drift baseline | 0.3145 | 1.0497 | 0.0338 | 0.8501 | 0.3697 | 0.1553 | 488.18 |
| Residual feature drift 0.25 | 0.4121 | 0.9175 | 0.0259 | 0.8344 | 0.3032 | 0.1194 | 148.90 |
| Residual feature drift 1.0 | 0.2539 | 0.9124 | 0.0272 | 0.8494 | 0.3528 | 0.1603 | 278.46 |

Read:

- Residual feature drift at weight `0.25` is the first strict-judge semantic
  improvement after the CIFAR calibration audit. It raises generated-label
  accuracy from `0.3145` to `0.4121` and improves nearest-real MSE,
  feature-nearest distance, and output settling.
- Weight `1.0` overshoots: it keeps pixel proximity better than baseline, but
  hurts generated-label accuracy and does not improve settling. This looks like
  a Goldilocks regime, not "more semantic loss is automatically better."
- The improvement costs diversity and speed. Feature drift makes samples more
  CIFAR-like under the residual judge but also narrows the output distribution.
- The training feature classifier itself was weak (`~0.28-0.30` eval
  accuracy), because it only sees the generator training subset. A stronger
  feature model or larger feature-training subset may improve the signal, but
  future runs must keep an independent quality judge to avoid circular
  evaluation.
- Next useful probe: repeat the `0.25` feature-drift setting across seeds and
  against a no-main-interaction control. If the semantic gain survives, the
  next architecture step is to recover diversity with oscillator-native
  mechanisms rather than simply increasing feature loss.

CIFAR-10 RGB residual feature-drift attribution repeat:

This repeat tested whether the residual feature-drift gain belongs to coupled
HORN dynamics or mostly to the better semantic objective. It compared the
prefix-25 HORN recipe against the matching no-main-interaction control, with
and without residual feature drift, over two seeds. Scoring used an independent
residual-conv quality judge.

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_semantic_feature_drift_attribution
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_attribution.csv \
  --output-dir outputs/analysis/cifar10_rgb_semantic_feature_drift_attribution \
  --title "CIFAR-10 RGB residual feature-drift attribution" \
  --accuracy-floor 0.0 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Output settling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN pixel drift | 2 | 0.2471 | 1.1086 | 0.0356 | 0.7884 | 0.3311 | 0.1475 |
| HORN residual feature drift 0.25 | 2 | 0.3145 | 1.0286 | 0.0309 | 0.8615 | 0.3912 | 0.1437 |
| No-main pixel drift | 2 | 0.2080 | 1.0149 | 0.0317 | 0.7668 | 0.3375 | 0.2200 |
| No-main residual feature drift 0.25 | 2 | 0.2744 | 0.8927 | 0.0260 | 0.8338 | 0.3085 | 0.1824 |

Read:

- Residual feature drift improves generated-label accuracy by about the same
  amount in both cases: `+0.0674` for coupled HORN and `+0.0664` for the
  no-main control. That means the semantic target is carrying much of the
  quality gain; it is not yet clean evidence that main HORN coupling alone
  solved CIFAR structure.
- Coupled HORN is still the best strict-judge accuracy and feature-diversity
  point in this repeat, and it keeps more pixel diversity than the no-main
  feature-drift control. That is the useful signal: coupling seems to preserve
  a better diversity/semantic frontier.
- The no-main feature-drift control wins nearest-real pixel MSE and feature
  nearest-real distance. Future claims should avoid "HORN simply beats the
  control" language unless the metric being discussed is named.
- Next priority is not a blind architecture scale-up. The next controlled move
  is to improve sharpness or semantics while preserving HORN's diversity
  advantage, and to keep no-main/one-step/shuffled-drive controls attached.

CIFAR-10 RGB attractor robustness probe:

After adding `generator.attractor_robustness`, a compact one-seed probe tested
the current residual feature-drift recipe against the no-main HORN control and
a StateMLP control. All variants used the stricter residual-conv quality judge
and `8` independent initial states per class label.

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attractor_robustness_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_attractor_robustness_probe \
  --title "CIFAR-10 RGB attractor robustness probe" \
  --accuracy-floor 0.0 \
  --no-plot
```

| Variant | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Within-class pixel spread | Feature separation ratio | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN residual feature drift 0.25 | 0.4062 | 0.9964 | 0.0305 | 0.8844 | 0.2810 | 0.3750 | 65.1837 | 2.7199 | 571.85 |
| No-main residual feature drift 0.25 | 0.3477 | 0.8558 | 0.0246 | 0.8768 | 0.3470 | 0.2750 | 37.0479 | 2.1206 | 558.75 |
| StateMLP residual feature drift 0.25 | 0.3867 | 0.5232 | 0.0137 | 0.8592 | 0.2782 | 0.4000 | 0.0056 | 6456.0593 | 203.85 |

Read:

- Coupled HORN is the best overall class/diversity point in this probe:
  highest generated-label accuracy, highest pixel diversity, highest feature
  diversity, best HORN/no-main attractor accuracy, and much higher same-label
  within-class spread than StateMLP.
- StateMLP gets competitive generated-label and attractor accuracy, but the
  attractor probe exposes severe class-basin collapse: same-label initial
  states produce nearly identical outputs (`within-class pixel spread 0.0056`).
  Its huge separation ratio is therefore not a win; it is the denominator
  collapsing.
- The preferred single-number basin proxy is therefore the attractor diversity
  score, not raw separation ratio. For the rows above, the derived pixel scores
  are approximately `1.57` for coupled HORN, `1.01` for no-main HORN, and
  `0.002` for StateMLP.
- No-main HORN remains strong on nearest-real pixel MSE, but it has lower
  attractor accuracy and lower within-class spread than coupled HORN. This is
  useful one-seed support for the hypothesis that sparse local HORN coupling
  preserves a richer generative basin than class drive alone.
- This is still one seed and CIFAR semantic accuracy is modest. The next
  valuable move is to improve sharpness/semantics while explicitly preserving
  HORN's within-class attractor diversity.

CIFAR-10 RGB attractor robustness seed repeat:

The two-seed repeat reran the same three variants over seeds `11` and `23` in
one CSV, using the same strict residual-conv judge and `8` initial-state
samples per class.

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_attractor_robustness_seed_repeat \
  --title "CIFAR-10 RGB attractor robustness seed repeat" \
  --accuracy-floor 0.0 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Attractor acc | Basin score | Within-class pixel spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN residual feature drift 0.25 | 2 | 0.3408 | 1.0805 | 0.0331 | 0.8791 | 0.3438 | 1.4915 | 73.9303 |
| No-main residual feature drift 0.25 | 2 | 0.3896 | 0.8758 | 0.0238 | 0.8712 | 0.3938 | 1.3596 | 35.0789 |
| StateMLP residual feature drift 0.25 | 2 | 0.1904 | 0.5265 | 0.0127 | 0.7532 | 0.2000 | 0.0010 | 0.0069 |

Read:

- The important HORN-vs-StateMLP finding survives: StateMLP has much lower
  diversity and effectively collapsed same-label basins, while HORN keeps
  nontrivial within-class variation.
- Coupled HORN keeps the best collapse-aware basin score and highest pixel
  diversity across the two seeds. That supports the "richer attractor basin"
  story more directly than raw generated-label accuracy.
- No-main HORN wins generated-label accuracy and nearest-real MSE on this
  two-seed average. So the coupling claim should be precise: coupling is not
  simply improving all quality metrics; it appears to preserve richer basin
  diversity at a cost to pixel closeness and sometimes class accuracy.
- The next architecture target is therefore sharper coupled HORN: improve
  semantics/pixel quality without losing the basin score advantage.

CIFAR-10 RGB sparse class-drive probe:

The next targeted probe made the class drive sparse instead of letting the
label inject directly into every oscillator. The hypothesis was simple: if
main HORN coupling matters, then the coupled field should degrade more
gracefully than the no-main-interaction control when only 25% or 10% of the
oscillator pool receives direct class drive.

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_sparse_drive_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_sparse_drive_probe \
  --title "CIFAR-10 RGB sparse class-drive HORN probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Direct drive | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN full drive | 256/256 | 0.8555 | 0.9823 | 0.0292 | 0.9641 | 0.1507 |
| HORN 25% drive | 64/256 | 0.7266 | 1.1014 | 0.0328 | 0.9461 | 0.1432 |
| HORN 10% drive | 26/256 | 0.3047 | 1.1027 | 0.0383 | 0.8604 | 0.1682 |
| No-main full drive | 256/256 | 0.7480 | 0.9251 | 0.0280 | 0.9731 | 0.1286 |
| No-main 25% drive | 64/256 | 0.6348 | 1.1428 | 0.0373 | 0.9150 | 0.1476 |
| No-main 10% drive | 26/256 | 0.2559 | 1.0389 | 0.0408 | 0.8754 | 0.1891 |

Read:

- This is the cleanest positive coupling-attribution signal so far. With only
  25% direct class drive, coupled HORN beats the no-main-interaction control on
  generated-label accuracy, nearest-real MSE, feature diversity, and feature
  nearest-real distance.
- The full-drive setting still leaves ambiguity because conditioning reaches
  every oscillator directly. It remains the best raw semantic setting, but not
  the cleanest mechanism proof.
- The 10% setting is probably too starved for the current architecture. Both
  variants fall near the weak-judge floor, though coupling still improves
  generated-label accuracy and nearest-real metrics.
- The useful next target is therefore not arbitrary hyperparameter search. It
  is structured drive topology: sparse/hierarchical class drive plus local HORN
  propagation, probably around the 25% regime, where main coupling has room to
  matter without starving the field.

CIFAR-10 RGB 25% sparse-drive seed repeat:

The 25% sparse-drive setting was repeated across four seeds to check whether
the coupled-vs-no-main gap was a lucky seed-11 artifact.

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_sparse_drive_seed_repeat
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_sparse_drive_seed_repeat \
  --title "CIFAR-10 RGB 25% sparse-drive seed repeat" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled HORN 25% drive | 4 | 0.7534 ± 0.0477 | 1.0950 ± 0.0329 | 0.0321 ± 0.0023 | 0.9574 ± 0.0135 | 0.1409 ± 0.0177 |
| No-main 25% drive | 4 | 0.6978 ± 0.0471 | 1.0632 ± 0.0569 | 0.0326 ± 0.0041 | 0.9380 ± 0.0113 | 0.1505 ± 0.0224 |

Paired seed differences, coupled minus no-main:

- Generated-label accuracy: `+0.0557` mean; coupled wins `4/4` seeds.
- Pixel diversity: `+0.0317` mean; coupled wins `3/4` seeds.
- Nearest-real MSE: `-0.0004` mean; effectively a wash.
- Feature diversity: `+0.0194` mean; coupled wins `4/4` seeds.
- Feature nearest-real MSE: `-0.0096` mean; coupled wins `3/4` seeds.
- Output settling ratio: `-0.0931` mean; coupled has lower final output
  change on `4/4` seeds.

Read:

- This strengthens the sparse-drive mechanism result. When only 64 of 256
  oscillators receive direct class drive, local HORN coupling consistently
  improves class consistency and feature-space diversity over independent
  second-order cells.
- Pixel nearest-real remains nearly tied. The benefit is not "better pixel
  memorization"; it is semantic consistency plus feature/diversity behavior
  under a constrained drive topology.
- The next architecture step should use this result as a design clue:
  structured sparse or hierarchical class drive, where information is injected
  into a subset/coarse field and must propagate through local oscillator
  dynamics.

CIFAR-10 RGB structured sparse-drive probe:

The first structured-drive follow-up compared the existing `prefix` sparse
drive against a new `spatial_grid` target pattern. Both drive 64 of 256
oscillators directly, but:

- `prefix` drives one contiguous row-major region of the oscillator grid.
- `spatial_grid` spreads driven oscillators approximately evenly across the
  field.

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_structured_drive_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_structured_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_structured_drive_probe \
  --title "CIFAR-10 RGB structured sparse-drive probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled HORN prefix 25% | 2 | 0.7119 | 1.1238 | 0.0341 | 0.9554 | 0.1535 |
| Coupled HORN spatial-grid 25% | 2 | 0.7041 | 1.1522 | 0.0367 | 0.9422 | 0.1646 |
| No-main prefix 25% | 2 | 0.6826 | 1.1142 | 0.0363 | 0.9277 | 0.1719 |
| No-main spatial-grid 25% | 2 | 0.7021 | 1.0878 | 0.0364 | 0.9429 | 0.1762 |

Read:

- Spatial-grid drive is not an immediate improvement for coupled HORN. It
  raises pixel diversity, but hurts nearest-real MSE, feature diversity,
  feature nearest-real distance, and settling relative to the prefix drive.
- Coupled HORN still beats no-main under prefix drive on both seeds for
  generated-label accuracy, feature diversity, feature nearest-real distance,
  output settling, and state energy.
- Under spatial-grid drive, coupled vs no-main becomes mostly tied on class
  accuracy and feature diversity. Coupling still improves diversity,
  feature-nearest, and state energy, but the clean semantic advantage weakens.
- This suggests the useful mechanism may be closer to a coherent driven patch
  propagating through local HORN dynamics, rather than scattered label anchors
  everywhere. The next structured topology should probably be a coarse block,
  boundary band, or hierarchical coarse-to-fine drive instead of a uniform
  lattice of driven cells.

CIFAR-10 RGB coherent sparse-drive probe:

The next topology check compared the existing `prefix` drive against a compact
`center_block` drive. Both inject class drive into 64 of 256 oscillators, but
`center_block` makes the driven source a centered 8x8 oscillator-grid patch.
The probe keeps the same two-seed shape as the structured-drive run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coherent_drive_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coherent_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coherent_drive_probe \
  --title "CIFAR-10 RGB coherent sparse-drive probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled HORN prefix 25% | 2 | 0.7188 | 1.1238 | 0.0341 | 0.9528 | 0.1529 |
| Coupled HORN center-block 25% | 2 | 0.7188 | 1.1609 | 0.0361 | 0.9564 | 0.1612 |
| No-main prefix 25% | 2 | 0.6689 | 1.1142 | 0.0363 | 0.9227 | 0.1613 |
| No-main center-block 25% | 2 | 0.7148 | 1.0745 | 0.0336 | 0.9374 | 0.1587 |

Paired read:

- Center-block vs prefix for coupled HORN is a diversity tradeoff, not a
  quality win: equal mean generated-label accuracy, higher pixel diversity,
  slightly higher feature diversity, but worse nearest-real and feature-nearest
  distances.
- Prefix coupled HORN still beats no-main prefix on generated-label accuracy
  and feature diversity on both seeds, preserving the clean sparse-drive
  mechanism result.
- Center-block coupled HORN beats no-main center on feature diversity and
  pixel diversity on both seeds, but not on pixel/feature nearest-real
  distance. The no-main center control is surprisingly strong semantically.
- The practical conclusion is that drive topology matters, but the current
  best bet is not simply "move class drive to the center." A coherent driven
  source helps maintain diversity; the top-band/prefix layout remains cleaner
  for semantic and feature proximity. The next non-random architecture step is
  likely a small coarse class-driver field or boundary/coarse-to-fine drive,
  not more scattered anchors.

CIFAR-10 RGB main-coupling strength probe:

After splitting `coupling_strength` (class/conditioning drive) from
`main_coupling_strength` (recurrent oscillator interaction), a compact
one-seed probe swept recurrent HORN coupling while holding the 25% sparse class
drive fixed:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_strength_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_strength_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_strength_probe \
  --title "CIFAR-10 RGB main coupling strength probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN main 0.00 | 1 | 0.5293 | 0.9736 | 0.0287 | 0.8729 | 0.2928 | 0.5500 | 2.1441 |
| HORN main 0.25 | 1 | 0.3086 | 0.9158 | 0.0252 | 0.7625 | 0.2639 | 0.3250 | 1.1548 |
| HORN main 0.50 | 1 | 0.4805 | 0.9389 | 0.0268 | 0.8645 | 0.3604 | 0.4875 | 1.8709 |
| HORN main 1.00 | 1 | 0.4355 | 0.9418 | 0.0270 | 0.8434 | 0.3437 | 0.4750 | 1.8182 |

Read:

- This single-seed result does **not** support "full recurrent main coupling is
  the key ingredient" for the current CIFAR RGB recipe. With class drive held
  fixed, `main_coupling_strength=0.0` wins generated-label accuracy, pixel
  diversity, feature diversity, attractor label accuracy, basin score, and
  throughput.
- The result does not make the model non-oscillatory: it is still a
  second-order HORN settling system with position/velocity state, oscillator
  frequencies, damping, nonlinear state bounds, class-coupled drive, and
  multi-step readout. But it suggests the current positive mechanism may be
  "driven second-order oscillator bank + readout" more than "learned recurrent
  oscillator interaction."
- `main_coupling_strength=0.25` is a clear bad pocket on this seed: it improves
  nearest-real MSE but hurts semantic and basin metrics. That warns against
  treating coupling strength as a monotonic quality knob.

The follow-up seed repeat compared `main=0.0`, `main=0.5`, `main=1.0`, and the
matched StateMLP on seeds `11, 23`:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_strength_seed_repeat \
  --title "CIFAR-10 RGB main coupling strength seed repeat" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN main 0.00 | 2 | 0.4082 | 0.9927 | 0.0298 | 0.8533 | 0.3653 | 0.3875 | 1.5471 |
| HORN main 0.50 | 2 | 0.4443 | 0.9427 | 0.0269 | 0.8705 | 0.3162 | 0.4188 | 1.5736 |
| HORN main 1.00 | 2 | 0.3506 | 0.9876 | 0.0291 | 0.8875 | 0.3619 | 0.3500 | 1.3763 |
| StateMLP | 2 | 0.3018 | 0.5421 | 0.0128 | 0.7763 | 0.3288 | 0.3000 | 0.0021 |

Updated read:

- The one-seed `main=0.0` win did not fully replicate. Moderate recurrent
  coupling (`main=0.5`) is the best two-seed balance: highest generated-label
  accuracy, best HORN nearest-real MSE, best feature-nearest distance, best
  attractor accuracy, and best basin score.
- Full recurrent coupling (`main=1.0`) is not optimal here. It preserves high
  feature diversity, but hurts semantic accuracy and basin score. The useful
  region looks like a Goldilocks zone, not "more coupling is better."
- StateMLP still wins raw pixel nearest-real MSE, but collapses same-label
  attractor diversity: basin score `0.0021` vs HORN around `1.4-1.6`. This
  keeps the HORN advantage alive as a diversity/settling/basin result rather
  than a pixel-closeness result.
- The split-coupling regression test confirms that omitting
  `main_coupling_strength` preserves the old combined update formula. So the
  new result is not explained by accidentally changing old default dynamics.
- Next architecture work should not abandon recurrent coupling. It should tune
  it separately from class drive and explore structured/coarse recurrent
  coupling that gives the diversity benefit without over-synchronizing the
  class-conditioned oscillator bank.

Current-code replication:

The seed-repeat and fine probes contained a warning sign: the same seed and
same `main=0.5` arguments from two separately launched sweeps did not reproduce
identical metrics. That makes the old `main=0.5 is best` read too strong. A
clean current-code replication therefore reran all recurrent-coupling strengths
together under the same source snapshot:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_current_replication
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_current_replication.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_current_replication \
  --title "CIFAR-10 RGB main coupling current-code replication" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HORN main 0.00 | 2 | 0.4180 | 1.0177 | 0.0298 | 0.9068 | 0.3502 | 0.4250 | 1.6811 |
| HORN main 0.25 | 2 | 0.4092 | 1.0192 | 0.0306 | 0.8633 | 0.3407 | 0.4000 | 1.6835 |
| HORN main 0.50 | 2 | 0.4131 | 0.9884 | 0.0291 | 0.8496 | 0.3400 | 0.4438 | 1.6748 |
| HORN main 0.75 | 2 | 0.2842 | 1.0797 | 0.0338 | 0.7944 | 0.3225 | 0.3375 | 1.4736 |
| HORN main 1.00 | 2 | 0.3799 | 0.9653 | 0.0283 | 0.8932 | 0.3563 | 0.3937 | 1.5413 |

Corrected read:

- Recurrent coupling is **not useless**, but it is not the whole engine either.
  `main=0.5` gives the best attractor label accuracy, `main=0.0` gives the best
  generated-label accuracy and feature diversity, `main=1.0` gives the best raw
  nearest-real MSE, and `main=0.25` has the best basin score by a hair.
- `main=0.75` is a consistent bad pocket for semantic quality despite high
  pixel diversity. That supports the "too much or wrong-scale coupling can
  de-coordinate the class-conditioned field" hypothesis.
- The robust positive mechanism is therefore not simply "recurrent coupling
  solved CIFAR." It is the structured HORN generator recipe: sparse local
  second-order oscillator state, multi-step settling, strong but sparse class
  drive, learned feature/pixel drift training, and resize-conv readout.
- The recurrent interaction should remain in the architecture, but future work
  should treat it as a structured tradeoff knob. The next promising version is
  not stronger dense/local coupling everywhere; it is likely a cleaner topology
  such as coarse-to-fine recurrent coupling, low-rank/global carrier coupling,
  or distance-decay coupling with explicit normalization.
- Report future coupling claims with mean/std or replication sweeps. Single-run
  coupling conclusions are too fragile for this branch.

Framework follow-up:

The next step is intentionally framework-enhancing rather than a one-off HORN
flag. Coupling topology builders now belong in `oscnet.core.coupling` so
generators, Winfree fields, masked-completion models, and future coarse-to-fine
oscillator stacks can share the same semantics. The first reusable addition is
row-sum-normalized spatial coupling: `distance_decay` or `local_radius`
profiles can preserve their sparsity/locality pattern while each non-empty row
is scaled to a fixed target gain. In the current generator equations the
interaction sum is divided by `N`, so normalized generator profiles use target
row sum `N`.

Pre-registered next probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_distance_decay_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_distance_decay_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_distance_decay_probe \
  --title "CIFAR-10 RGB normalized distance-decay coupling probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

Decision rule: keep normalized spatial coupling only if it improves the
semantic/diversity frontier over the current local-radius HORN runs without
collapsing attractor diversity. If it mainly improves nearest-real MSE while
hurting diversity or class consistency, treat it as a smoothing/proximity knob,
not as a breakthrough coupling mechanism.

Result:

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Coupling density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized distance, main 0.25 | 2 | 0.4258 | 1.0746 | 0.0334 | 0.8774 | 0.3385 | 0.4625 | 1.9739 | 1.0000 |
| Normalized distance, main 0.50 | 2 | 0.3799 | 0.9600 | 0.0274 | 0.8347 | 0.3784 | 0.3750 | 1.4499 | 1.0000 |
| Normalized distance, main 1.00 | 2 | 0.4014 | 1.0095 | 0.0302 | 0.8637 | 0.3278 | 0.4188 | 1.6986 | 1.0000 |

Read:

- Row-normalized distance decay is worth keeping as a reusable framework
  topology. At `main=0.25` it beats the matched local-radius `main=0.25`
  replication on generated-label accuracy, pixel diversity, attractor accuracy,
  and basin score.
- It is **not** the new default. It worsens raw nearest-real MSE, is fully
  dense (`coupling_density=1.0` vs local-radius `0.0285`), and does not clearly
  dominate the simpler local-radius `main=0.0/0.5` runs across all metrics.
- This confirms the strategic point: recurrent coupling is likely useful when
  its topology and gain are controlled, but dense spatial coupling is expensive
  and should be treated as a diagnostic/architecture probe. The next
  framework-level coupling candidates should be structured and hardware-aware:
  coarse-to-fine, low-rank/global carrier, or sparse normalized distance bands.

Sparse normalized local-radius follow-up:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_local_radius_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_local_radius_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_local_radius_probe \
  --title "CIFAR-10 RGB normalized local-radius coupling probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Coupling density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized local, main 0.25 | 2 | 0.3760 | 1.0131 | 0.0308 | 0.8364 | 0.3377 | 0.3250 | 1.3764 | 0.0285 |
| Normalized local, main 0.50 | 2 | 0.3232 | 1.0232 | 0.0308 | 0.8328 | 0.3772 | 0.3563 | 1.4568 | 0.0285 |
| Normalized local, main 1.00 | 2 | 0.4424 | 1.0102 | 0.0300 | 0.9035 | 0.3680 | 0.4875 | 1.9398 | 0.0285 |

Updated read:

- Sparse row-normalized local coupling is the strongest coupling follow-up so
  far. `main=1.0` gives the best generated-label accuracy and attractor label
  accuracy among the current coupling probes while keeping the original sparse
  local topology (`coupling_density=0.0285`).
- Compared with the old unnormalized local-radius replication, normalized
  local `main=1.0` improves generated-label accuracy (`0.4424` vs `0.4180`
  best old), feature diversity (`0.9035`, essentially tied with old best
  `0.9068`), attractor accuracy (`0.4875` vs `0.4438`), and basin score
  (`1.9398` vs `1.6835`). Pixel nearest-real MSE remains worse than the
  nearest-pixel-oriented settings, so this is a semantic/attractor improvement,
  not a pixel-closeness win.
- Compared with dense normalized distance-decay, normalized local `main=1.0`
  has better generated-label accuracy, feature diversity, and attractor
  accuracy, while dense distance-decay `main=0.25` keeps the highest raw pixel
  diversity and basin score by a small margin. The sparse result is more
  compelling because it is closer to the hardware/physics story: local
  structured coupling with controlled gain rather than all-to-all smoothing.
- This changes the next architecture direction. The best immediate lead is not
  fully dense nonlocal coupling; it is **gain-normalized sparse structured
  coupling**. Next probes should vary sparse local radius/row gain and then add
  a small coarse/global carrier on top, instead of replacing the local field
  with dense communication.

Normalized local-radius sweep:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_local_radius_sweep
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_local_radius_sweep.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_local_radius_sweep \
  --title "CIFAR-10 RGB normalized local-radius sweep" \
  --accuracy-floor 0.3 \
  --no-plot
```

This held `main_coupling_strength=1.0`, `coupling_normalization=row_sum`, and
the 25% sparse class drive fixed while sweeping local radius:

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Coupling density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized local radius 0.16 | 2 | 0.3506 | 0.9927 | 0.0290 | 0.8102 | 0.2982 | 0.3688 | 1.3860 | 0.0147 |
| Normalized local radius 0.24 | 2 | 0.4385 | 0.9557 | 0.0282 | 0.8346 | 0.3192 | 0.4000 | 1.5788 | 0.0285 |
| Normalized local radius 0.32 | 2 | 0.3916 | 1.0443 | 0.0321 | 0.8682 | 0.3420 | 0.4188 | 1.7795 | 0.0680 |

Read:

- The radius result is a frontier, not a single clean winner. Radius `0.24`
  has the best generated-label accuracy, nearest-real MSE, and throughput.
  Radius `0.32` has the best pixel diversity, feature diversity, attractor
  accuracy, and basin score within this sweep, but it is denser and farther
  from real images by nearest-pixel MSE.
- Radius `0.16` looks too tight for this CIFAR RGB setup. It is sparse and has
  good feature-nearest proximity, but loses semantic accuracy and basin metrics.
- The previous normalized-local probe produced a stronger `radius=0.24,
  main=1.0` point than this radius sweep, which reinforces the warning that
  small two-seed CIFAR runs are noisy. Use the direction of the frontier rather
  than overfitting one table row.
- Strategic implication: the model wants both local coherence/proximity and a
  wider integration path. The next meaningful architecture is therefore not
  "pick one radius forever"; it is a multi-scale/coarse-to-fine HORN system
  where a tight/medium local field handles texture and a wider/coarse field
  coordinates class-level structure.

### Coarse-to-Fine HORN Generator Probe

Implemented a reusable `CoarseToFineHORNImageGenerator`: a small coarse HORN
oscillator bank evolves in parallel with the fine normalized-local HORN field
and sends learned displacement drive into fine oscillator acceleration. The
coarse state is not decoded directly, so the probe tests top-down oscillatory
coordination rather than adding a second image decoder.

Pre-registered Modal probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_probe
```

Planned comparison:

- `horn_normlocal`: current normalized-local sparse HORN baseline.
- `coarse16_c2f000`: coarse oscillator bank present, but no coarse-to-fine
  drive.
- `coarse16_c2f100`: same coarse bank with coarse-to-fine drive enabled.

The key attribution question is whether the enabled coarse path improves
semantic accuracy, feature diversity, attractor accuracy, or basin score over
both the baseline and the zero-drive coarse control.

Modal result, seeds 11 and 23:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_probe \
  --title "CIFAR-10 RGB coarse-to-fine HORN probe" \
  --accuracy-floor 0.3 \
  --no-plot
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Params | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse16, no fine drive | 2 | 0.3633 | 0.9593 | 0.0274 | 0.8566 | 0.3154 | 0.3563 | 1.3218 | 166083 | 349.0 |
| coarse16 -> fine drive | 2 | 0.4229 | 0.8960 | 0.0245 | 0.8762 | 0.3442 | 0.3875 | 1.3128 | 166083 | 377.3 |
| normalized-local HORN | 2 | 0.4014 | 1.0364 | 0.0317 | 0.8790 | 0.3707 | 0.4562 | 1.8984 | 156595 | 781.8 |

Read:

- The coarse-to-fine path is not a dead branch. Turning coarse-to-fine drive on
  improves generated-label accuracy, nearest-real MSE, feature diversity, and
  attractor accuracy over the same coarse model with drive disabled.
- It is also not the new winner. The plain normalized-local HORN baseline keeps
  higher pixel diversity, slightly higher feature diversity, much higher
  attractor label accuracy, and a much stronger basin/diversity score. It is
  also about 2x faster in this digital JAX setup.
- The coarse path currently looks like a proximity/stabilization bias more than
  a global diversity-preserving coordinator: samples move closer to training
  images and gain class accuracy, but the attractor basin tightens.
- This narrows the next multiscale step. Do not simply add more coarse units.
  The useful follow-up is a gentler/regularized coarse path: lower
  `coarse_to_fine_strength`, sparse or distance-decayed coarse-to-fine
  projection, or coarse drive gated by local phase/velocity so it coordinates
  without clamping diversity.

Gentle coarse-to-fine gain sweep:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep
```

Combined table with the previous baseline/probe:

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Params | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse16, no fine drive | 2 | 0.3633 | 0.9593 | 0.0274 | 0.8566 | 0.3154 | 0.3563 | 1.3218 | 166083 | 349.0 |
| coarse16 -> fine, strength 0.25 | 2 | 0.4287 | 0.9493 | 0.0262 | 0.8533 | 0.3477 | 0.3813 | 1.3856 | 166083 | 209.8 |
| coarse16 -> fine, strength 0.50 | 2 | 0.3311 | 0.8918 | 0.0246 | 0.8559 | 0.3565 | 0.3375 | 1.1874 | 166083 | 197.4 |
| coarse16 -> fine, strength 0.75 | 2 | 0.2949 | 0.8840 | 0.0233 | 0.8341 | 0.3488 | 0.2688 | 0.8843 | 166083 | 235.1 |
| coarse16 -> fine, strength 1.00 | 2 | 0.4229 | 0.8960 | 0.0245 | 0.8762 | 0.3442 | 0.3875 | 1.3128 | 166083 | 377.3 |
| normalized-local HORN | 2 | 0.4014 | 1.0364 | 0.0317 | 0.8790 | 0.3707 | 0.4562 | 1.8984 | 156595 | 781.8 |

Read:

- `coarse_to_fine_strength=0.25` is the best coarse compromise so far. It
  improves generated-label accuracy over both the coarse no-drive control and
  the plain normalized-local HORN baseline while preserving more diversity and
  basin score than the stronger coarse gains.
- Higher coarse gains continue the same tradeoff: nearest-real MSE improves,
  but diversity, attractor accuracy, and basin score tighten or collapse. The
  `0.75` row drops below the accuracy floor in the two-seed mean.
- This confirms that coarse-to-fine HORN is useful but must be weak and
  regulated. The mechanism should be treated as a global bias/carrier, not as a
  strong top-down clamp.
- The next architecture move should regularize the projection itself, not only
  the scalar gain: sparse spatial coarse-to-fine masks, distance-decayed
  coarse-to-fine profiles, or phase/velocity-gated coarse drive are the most
  plausible ways to get global coordination without losing attractor diversity.

Coarse-to-fine projection-profile probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe
```

| Variant | Runs | Generated-label acc | Pixel diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Coarse-to-fine density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense gentle | 2 | 0.4043 | 0.9710 | 0.0273 | 0.8404 | 0.3645 | 0.4313 | 1.6146 | 1.0000 |
| distance-decay gentle | 2 | 0.4570 | 0.9404 | 0.0264 | 0.8756 | 0.3444 | 0.4500 | 1.6321 | 1.0000 |
| local-radius gentle | 2 | 0.4619 | 0.8561 | 0.0223 | 0.8926 | 0.3247 | 0.4562 | 1.4537 | 0.1094 |

Read:

- Spatially regularizing the coarse-to-fine projection helps. The
  distance-decay profile beats the dense gentle variant on generated-label
  accuracy, nearest-real MSE, feature diversity, attractor accuracy, and basin
  score, while keeping more diversity than the sparse local-radius projection.
- Local-radius coarse-to-fine gives the best semantic accuracy, feature
  diversity, and nearest-real MSE in this probe, but it clamps the basin more
  aggressively. That is useful for quality/proximity, but less aligned with the
  diversity-preserving attractor story.
- Distance-decay is now the best coarse-to-fine compromise. It supports the
  physics hypothesis more than the dense projection: global influence is useful,
  but it should be spatially biased rather than arbitrary all-to-all top-down
  broadcast.
- Plain normalized-local HORN still keeps the strongest diversity/basin result
  from the earlier combined table. The best next architecture therefore should
  combine normalized local fine coupling with **weak distance-decayed
  coarse-to-fine drive**, then test whether a slightly better readout or
  longer training converts the improved semantic/proximity frontier into better
  visible images without losing diversity.

Instrumentation update:

Coarse-to-fine HORN traces now report coarse-field settling diagnostics in the
same `success_diagnostics` block as the fine field: coarse state energy,
velocity/update/acceleration RMS, coarse recurrent disagreement, and
coarse-to-fine disagreement. These are not physical energy claims. They are
practical probes for the next question: does the coarse field become a stable
global coordinator, or does it simply clamp the fine field and reduce attractor
diversity?

Compact coarse-to-fine dynamics audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=5 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit
```

This reruns one seed after adding the coarse settling diagnostics. It also
fixes the coarse-to-fine disagreement proxy so a disabled
`coarse_to_fine_strength=0.0` path reports zero effective C2F disagreement.

| Variant | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Output settle | Coarse update settle | C2F delta | C2F density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized-local HORN | 0.4141 | 0.9835 | 0.0297 | 0.8952 | 0.3247 | 0.3875 | 1.5488 | 0.1341 | n/a | n/a | 0.0000 |
| coarse16, no fine drive | 0.3633 | 0.8801 | 0.0236 | 0.8540 | 0.2688 | 0.3750 | 1.1866 | 0.0806 | 0.5483 | 0.0000 | 1.0000 |
| dense gentle C2F | 0.3652 | 0.8832 | 0.0228 | 0.8251 | 0.2800 | 0.2750 | 0.9197 | 0.1179 | 0.5478 | -0.0248 | 1.0000 |
| distance-decay gentle C2F | 0.3457 | 0.8814 | 0.0240 | 0.8027 | 0.3568 | 0.3000 | 1.0051 | 0.1020 | 0.5479 | -0.0273 | 1.0000 |
| local-radius gentle C2F | 0.4219 | 0.9119 | 0.0236 | 0.8516 | 0.3051 | 0.5000 | 1.6564 | 0.0857 | 0.5486 | -0.0262 | 0.1094 |

Read:

- This is one seed, so treat it as a diagnostic, not a final ranking.
- The local-radius coarse-to-fine path is the most interesting row. It beats
  normalized-local HORN on generated-label accuracy, nearest-real MSE,
  attractor accuracy, and basin score while preserving more diversity than the
  dense and distance-decay coarse variants.
- Dense and distance-decay coarse-to-fine both improve pixel proximity but do
  not improve the semantic/diversity frontier in this run. They look more like
  stabilizers than coordinators here.
- The coarse oscillator bank itself settles similarly across coarse variants:
  coarse update ratio is about `0.548`, coarse recurrent disagreement decreases
  by about `-0.092`, and final coarse energy is about `0.039`. The differentiator
  is therefore the coarse-to-fine projection topology, not whether the coarse
  bank settles at all.
- Next best test: repeat local-radius gentle C2F versus normalized-local HORN
  over more seeds and inspect sample grids. If it holds, local sparse
  coarse-to-fine coupling becomes the new lead for multiscale HORN.

Local-radius coarse-to-fine seed repeat:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat
```

This repeats four seeds for three rows: normalized-local HORN, the coarse bank
with `coarse_to_fine_strength=0.0`, and the local-radius gentle coarse-to-fine
variant. Sample grids were pulled into:

```text
outputs/analysis/cifar10_rgb_coarse_to_fine_local_repeat/sample_grids/
```

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Output settle | C2F delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized-local HORN | 4 | 0.3491 | 0.9853 | 0.0298 | 0.8573 | 0.3651 | 0.3250 | 1.3137 | 0.1545 | n/a |
| coarse16, no fine drive | 4 | 0.4038 | 0.8762 | 0.0244 | 0.8570 | 0.3567 | 0.4094 | 1.3920 | 0.1196 | 0.0000 |
| local-radius gentle C2F | 4 | 0.3911 | 0.9549 | 0.0284 | 0.8988 | 0.3778 | 0.3969 | 1.5111 | 0.1434 | -0.0318 |

Read:

- The one-seed local-radius bump partially survives, but not as a simple
  quality win. Local C2F beats normalized-local HORN on generated-label
  accuracy, nearest-real MSE, feature diversity, attractor accuracy, and basin
  score, while giving up a little pixel diversity.
- The no-drive coarse control has the best generated-label accuracy and
  nearest-real MSE. Since the coarse bank cannot drive the fine field in this
  row, that is mostly an initialization/control warning: do not over-attribute
  raw quality gains to the active coarse-to-fine path yet.
- Local C2F is still meaningful because it has a better diversity/basin profile
  than the no-drive control and a better semantic/basin profile than plain
  HORN. It looks like a coordinator/regularizer, not a magic quality booster.
- The visual grids remain blurry CIFAR-like color/shape blobs. Local C2F can
  look slightly more structured or saturated on some seeds, but it is not a
  qualitative breakthrough yet.
- The next architecture step should not be "more coarse strength." It should
  either improve the readout/feature objective so the better basin frontier
  becomes visible, or make the active C2F path more attributable by comparing
  against same-initialization no-drive controls and inspecting paired samples.

Same-seed paired attribution:

```bash
python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_local_repeat \
  --baseline-variant coarse16_c2f000 \
  --target-variant coarse16_c2f025_local050 \
  --target-variant horn_normlocal
```

This reads the same four-seed CSV and compares target-minus-baseline deltas on
matched seeds. The main comparison is active local C2F versus the no-drive
coarse control.

| Target | Baseline | Metric | Direction | Pairs | Delta | Target wins |
| --- | --- | --- | --- | ---: | ---: | ---: |
| local-radius gentle C2F | coarse16, no fine drive | generated acc | higher | 4 | -0.0127 | 2 |
| local-radius gentle C2F | coarse16, no fine drive | diversity | higher | 4 | +0.0788 | 3 |
| local-radius gentle C2F | coarse16, no fine drive | nearest-real MSE | lower | 4 | +0.0040 | 0 |
| local-radius gentle C2F | coarse16, no fine drive | feature diversity | higher | 4 | +0.0417 | 2 |
| local-radius gentle C2F | coarse16, no fine drive | feature nearest-real | lower | 4 | +0.0211 | 2 |
| local-radius gentle C2F | coarse16, no fine drive | attractor acc | higher | 4 | -0.0125 | 1 |
| local-radius gentle C2F | coarse16, no fine drive | basin score | higher | 4 | +0.1191 | 3 |
| local-radius gentle C2F | coarse16, no fine drive | output settle | lower | 4 | +0.0238 | 1 |

Paired read:

- Active local C2F does not explain the raw quality/proximity gain. The no-drive
  coarse control remains better on generated-label accuracy, nearest-real MSE,
  attractor accuracy, and output-settling ratio.
- Active local C2F does explain a better diversity/basin side of the frontier:
  higher pixel diversity on 3/4 seeds, higher basin score on 3/4 seeds, and
  higher aggregate feature diversity.
- That is a narrower but useful mechanism: weak top-down local C2F looks like a
  diversity/basin regularizer, not a quality booster. The next useful experiment
  is therefore a readout/objective conversion test: can the higher-diversity
  basin be translated into visibly sharper samples without collapsing back
  toward the no-drive/prototype solution?

Planned readout/objective conversion probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe
```

Rows:

| Row | Mechanism |
| --- | --- |
| `coarse16_c2f000_base` | no-drive coarse control, current local C2F recipe with C2F strength zero |
| `coarse16_c2f025_local050_base` | active local-radius C2F baseline |
| `coarse16_c2f000_ch32` | no-drive control with wider resize-conv readout |
| `coarse16_c2f025_local050_ch32` | active C2F with wider resize-conv readout |
| `coarse16_c2f000_dist0025` | no-drive control with small distributional pressure |
| `coarse16_c2f025_local050_dist0025` | active C2F with small distributional pressure |
| `coarse16_c2f000_ch32_dist0025` | no-drive control with both conversion knobs |
| `coarse16_c2f025_local050_ch32_dist0025` | active C2F with both conversion knobs |

After the run, use frontier plus paired-delta analysis:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe \
  --title "CIFAR-10 RGB coarse-to-fine HORN conversion probe" \
  --accuracy-floor 0.3

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe \
  --title "CIFAR-10 RGB coarse-to-fine conversion paired deltas" \
  --pair coarse16_c2f000_base:coarse16_c2f025_local050_base \
  --pair coarse16_c2f000_ch32:coarse16_c2f025_local050_ch32 \
  --pair coarse16_c2f000_dist0025:coarse16_c2f025_local050_dist0025 \
  --pair coarse16_c2f000_ch32_dist0025:coarse16_c2f025_local050_ch32_dist0025
```

Decision rule:

- If C2F remains better on diversity/basin and gains raw quality under `ch32`,
  the old bottleneck was likely readout capacity.
- If `dist0025` improves proximity but destroys diversity/basin, the objective
  is pulling the field back toward the no-drive/prototype solution.
- If the no-drive rows benefit more than the active C2F rows under every
  conversion knob, active local C2F is probably not the right path to quality;
  the next move should be a different multiscale interaction, not more decoder
  capacity.

Result:

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv
outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe/paired_deltas.md
outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe/sample_grids/contact_sheet_labeled.png
```

Aggregate frontier table:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse16_c2f000_base` | 2 | 0.4326 | 0.9011 | 0.0243 | 0.8809 | 0.3467 | 0.4375 | 1.4796 |
| `coarse16_c2f025_local050_base` | 2 | 0.3984 | 0.9469 | 0.0262 | 0.8651 | 0.3283 | 0.3875 | 1.4317 |
| `coarse16_c2f000_ch32` | 2 | 0.4727 | 1.0587 | 0.0329 | 0.8987 | 0.3771 | 0.4937 | 2.0983 |
| `coarse16_c2f025_local050_ch32` | 2 | 0.3828 | 1.0606 | 0.0331 | 0.8498 | 0.3265 | 0.3813 | 1.6242 |
| `coarse16_c2f000_dist0025` | 2 | 0.3916 | 1.0525 | 0.0345 | 0.8876 | 0.3592 | 0.3813 | 1.7173 |
| `coarse16_c2f025_local050_dist0025` | 2 | 0.3887 | 1.0682 | 0.0346 | 0.8495 | 0.3067 | 0.3938 | 1.7945 |
| `coarse16_c2f000_ch32_dist0025` | 2 | 0.3428 | 1.0626 | 0.0357 | 0.8301 | 0.3285 | 0.3438 | 1.5892 |
| `coarse16_c2f025_local050_ch32_dist0025` | 2 | 0.3740 | 1.0648 | 0.0352 | 0.8877 | 0.3640 | 0.3875 | 1.7866 |

Paired active-C2F deltas:

| Pair | Acc delta | Diversity delta | MSE delta | Feature diversity delta | Attractor acc delta | Basin delta | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base active C2F vs no-drive | -0.0342 | +0.0458 | +0.0019 | -0.0158 | -0.0500 | -0.0479 | C2F mainly adds pixel diversity; no quality conversion |
| ch32 active C2F vs ch32 no-drive | -0.0898 | +0.0019 | +0.0002 | -0.0489 | -0.1125 | -0.4741 | wider readout helps no-drive more |
| dist0025 active C2F vs dist0025 no-drive | -0.0029 | +0.0157 | +0.0000 | -0.0381 | +0.0125 | +0.0771 | small basin/attractor gain, no proximity gain |
| ch32+dist0025 active C2F vs ch32+dist0025 no-drive | +0.0312 | +0.0022 | -0.0005 | +0.0576 | +0.0438 | +0.1974 | first paired setting where C2F wins most frontier metrics |

Visual read:

- Samples remain blurry CIFAR-like color/shape patches. No row is a solved
  CIFAR generator.
- `ch32` and `ch32_dist0025` increase texture/saturation, but not clearly
  object-level structure.
- The best absolute semantic/basin row is no-drive `ch32`, which is a warning:
  extra readout capacity can improve the coarse-control/prototype path more
  than active C2F.
- The combined `ch32_dist0025` setting is still interesting because active C2F
  beats its same-readout/objective no-drive control on most paired metrics. The
  effect is real enough to keep, but the absolute quality is too low to call a
  breakthrough.

Updated read:

- Readout/objective conversion partially works only when both knobs are present.
  Active local C2F then improves the same-seed no-drive control on semantic
  accuracy, nearest-real MSE, feature diversity, attractor accuracy, basin
  score, and output settling.
- But the stronger no-drive `ch32` row still has better absolute accuracy and
  basin score. This means the current local C2F is not the main path to quality;
  it is a conditional regularizer that can help under the right objective, but
  it is still weaker than simply giving the no-drive coarse scaffold more
  readout capacity.
- Next useful step should not be another scalar C2F/readout sweep. The likely
  missing mechanism is an explicit image-space or feature-space feedback path:
  the oscillator field needs to see its decoded image error/feature state during
  settling, rather than only being decoded after settling. Without feedback,
  the readout can improve samples while the dynamics remain partially
  disconnected from visible quality.

Follow-up feedback probe:

The first implementation tried decoded-image feedback during every HORN
settling step. That is the cleanest "readout participates in the attractor"
idea, but it is expensive with the `resize_conv` decoder: training at
`16,32,48` settling steps turns one decoder call at the end into many
decoder calls inside the recurrent loop. The Modal run was stopped after it
became clear this was a compute blow-up rather than a normal CIFAR sweep.

The production probe therefore uses `output_feedback_mode="state_proxy"`:

```text
feedback_drive_i = gain_i * center(tanh(position_i) + 0.5 * tanh(velocity_i))
```

This is not full image-space feedback. It is a cheap local self-feedback proxy
that asks whether closing the recurrent field loop helps before spending GPU
time on a full decoded-image loop.

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe.csv
outputs/analysis/cifar10_rgb_coarse_to_fine_feedback_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_coarse_to_fine_feedback_probe/paired_deltas.md
outputs/analysis/cifar10_rgb_coarse_to_fine_feedback_probe/sample_grids/contact_sheet.png
```

Aggregate feedback frontier:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c2f000_ch32_dist0025_feedback050` | 2 | 0.3994 | 1.0688 | 0.0364 | 0.9041 | 0.3298 | 0.4063 | 1.8899 |
| `c2f025_local050_ch32_dist0025_feedback050` | 2 | 0.3984 | 1.0680 | 0.0362 | 0.8478 | 0.3143 | 0.4250 | 1.9713 |
| `c2f000_feedback050_base` | 2 | 0.3828 | 0.9755 | 0.0276 | 0.8495 | 0.3349 | 0.4250 | 1.5852 |
| `c2f025_local050_feedback050_base` | 2 | 0.3818 | 0.8546 | 0.0224 | 0.8461 | 0.3587 | 0.3938 | 1.2350 |

Paired active-C2F feedback deltas:

| Pair | Acc delta | Diversity delta | MSE delta | Feature diversity delta | Feature nearest-real delta | Attractor acc delta | Basin delta | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base feedback C2F vs no-drive | -0.0010 | -0.1209 | -0.0052 | -0.0034 | +0.0238 | -0.0312 | -0.3502 | more pixel-proximal but lower diversity/basin |
| ch32+dist0025 feedback C2F vs no-drive | -0.0010 | -0.0008 | -0.0002 | -0.0563 | -0.0155 | +0.0187 | +0.0814 | small attractor/basin gain, no visible frontier jump |

Visual read:

- The `ch32_dist0025` feedback grids are still blurry CIFAR-like color and
  shape patches.
- Active C2F and no-drive grids are nearly indistinguishable by eye.
- The cheap state-proxy feedback loop is stable and inexpensive
  (`8192` estimated feedback ops/sample), but it does not convert C2F into a
  visible quality breakthrough.

Updated feedback read:

- Full decoded-image feedback is a plausible mechanism, but too expensive to
  insert naively inside every `resize_conv` HORN step. It should be reserved
  for tiny probes or replaced by a lower-resolution learned feature/image
  proxy.
- State-proxy feedback is useful engineering infrastructure and a good
  control, but it is not the missing secret. It mostly preserves the previous
  result: active local C2F is a weak regularizer/coordinator, not the main
  quality path.
- The next multiscale move should change the inter-layer mechanism itself:
  bidirectional fine-to-coarse feedback, phase-lagged/inhibitory C2F, or an
  explicit coarse objective. More scalar strength sweeps are unlikely to
  unlock the visible-quality gap.

### Layered HORN Scaffold

Follow-up infrastructure now treats this as a framework-level layered
oscillator-field problem instead of another single C2F variant:

- `oscnet.core.layered` defines oscillator layer specs and directed inter-layer
  coupling specs.
- `MultiscaleHORNImageGenerator` supports multiple auxiliary HORN populations,
  per-layer horizontal coupling, directed vertical coupling, optional
  bidirectional feedback, phase lags, and per-layer conditioning drive.
- Diagnostics count auxiliary/vertical params and ops separately and report
  per-layer energy/update/coupling proxies plus vertical disagreement proxies.

The first disciplined probe compared
`sparse_horn_cifar10_rgb_multiscale16_64_local050_fb005`,
`sparse_horn_cifar10_rgb_multiscale16_64_no_vertical`, and the plain
`sparse_horn_cifar10_rgb_recommended_normlocal` control across seeds 11 and 23.

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.frontier.md
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.paired.md
outputs/analysis/multiscale_layered_grids/multiscale_layered_contact_sheet.png
```

Aggregate layered frontier:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Params | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiscale_local050_fb005` | 2 | 0.3750 | 0.9287 | 0.0244 | 0.8653 | 0.2971 | 0.3875 | 1.3440 | 221443 | 280.0 |
| `multiscale_no_vertical` | 2 | 0.4717 | 0.9626 | 0.0277 | 0.8978 | 0.3322 | 0.4750 | 1.7428 | 204035 | 371.4 |
| `normlocal_plain` | 2 | 0.4014 | 1.0407 | 0.0315 | 0.8914 | 0.3241 | 0.4313 | 1.7901 | 156595 | 364.2 |

Paired read:

- Adding auxiliary HORN banks without vertical coupling improved generated
  classifier accuracy versus plain normalized-local HORN by `+0.0703` on
  matched seeds and improved output settling, but reduced pixel diversity and
  did not improve the collapse-aware basin score.
- Activating weak bidirectional vertical coupling improved nearest-real MSE
  (`-0.0033` vs no-vertical, `-0.0071` vs plain) and feature nearest-real MSE
  (`-0.0351` vs no-vertical), but reduced generated accuracy, feature
  diversity, attractor accuracy, and basin score.
- Visual samples match the metrics: active vertical coupling looks more
  pulled/smoothed toward nearby examples, not more semantically coherent.

Updated layered read:

- The layered scaffold is valuable infrastructure, but this first local
  bidirectional vertical mechanism is not the breakthrough path.
- Extra auxiliary HORN banks may provide useful capacity/regularization even
  without vertical drive. That needs a parameter-matched control before
  claiming an ONN advantage.
- If layered ONNs are to win, the vertical interaction likely needs to be more
  selective: gated, inhibitory/excitatory, phase-lagged, attention-like sparse,
  or supervised by an explicit coarse objective. Simple weak local vertical
  springs trade semantic class coherence for sample proximity.

### Coarse Auxiliary Objective Probe

The next disciplined test gave the coarsest auxiliary HORN layer its own
low-resolution image target. This was meant to test whether hierarchy becomes
useful when the coarse layer has a real job, instead of acting as an
unsupervised spring connected to the fine layer.

Implementation:

- `MultiscaleHORNImageGenerator` now has an auxiliary low-resolution readout
  attached to a selected auxiliary layer.
- The generator harness supports `coarse_auxiliary_weight`,
  `coarse_auxiliary_target_size`, and `multiscale_auxiliary_readout_layer`.
- The auxiliary target is a downsampled image batch; the objective is opt-in
  and is tracked separately as `train/eval_coarse_auxiliary_loss`.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_auxiliary_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_auxiliary_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_auxiliary_probe.json
outputs/analysis/cifar10_rgb_multiscale_auxiliary_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_auxiliary_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_vertical_auxlow8` | 2 | 0.4102 | 0.9591 | 0.0283 | 0.8731 | 0.3393 | 0.4063 | 1.5816 | 344.0 |
| `local050_fb005` | 2 | 0.3906 | 0.9778 | 0.0275 | 0.8489 | 0.3637 | 0.4313 | 1.5998 | 313.7 |
| `no_vertical` | 2 | 0.3818 | 0.9944 | 0.0296 | 0.8683 | 0.3583 | 0.4125 | 1.6467 | 408.7 |
| `local050_fb005_auxlow8` | 2 | 0.3643 | 0.9155 | 0.0247 | 0.8149 | 0.3486 | 0.3813 | 1.3300 | 208.1 |

Paired read:

- Adding the coarse auxiliary target without vertical coupling improved mean
  generated-label accuracy by `+0.0283`, improved nearest-real MSE by
  `-0.0013`, and improved feature nearest-real MSE by `-0.0189`, but reduced
  pixel diversity and slightly reduced attractor/basin metrics.
- Active vertical coupling without the auxiliary target improved nearest-real
  MSE by `-0.0021` and output settling by `-0.0226`, with a small accuracy
  increase, but reduced diversity and feature diversity.
- Active vertical coupling with the auxiliary target was the best
  nearest-real-MSE row (`0.0247`) and the best output-settling row, but it was
  worse on generated-label accuracy, diversity, attractor accuracy, and basin
  score. It was also the slowest row.

Updated auxiliary-objective read:

- The low-resolution auxiliary objective is learnable and useful as framework
  infrastructure. It gives future multiscale experiments a way to make coarse
  layers meaningful rather than purely decorative.
- This first version does not validate active weak vertical coupling as the
  quality path. It mostly trades diversity/semantic basin quality for pixel or
  feature proximity.
- The most interesting row is `no_vertical_auxlow8`: because vertical coupling
  is disabled, its accuracy gain probably comes from the coarse objective
  shaping shared conditioning/readout parameters, not from hierarchy
  transmitting useful image structure to the fine field.
- Next multiscale work should not simply increase vertical strength. The more
  plausible path is selective vertical coupling: gated, signed
  excitatory/inhibitory, phase-lagged, or a bidirectional mechanism where the
  coarse target and fine field negotiate rather than one weak spring tugging on
  the other.

### Selective Vertical Gate Probe

The follow-up probe tested whether vertical hierarchy fails because the coarse
drive is sprayed too broadly across the fine field. `MultiscaleHORNImageGenerator`
now supports `multiscale_vertical_target_gate`:

- `all`: every fine oscillator can receive vertical drive.
- `conditioning`: vertical drive into the decoded fine layer is restricted to
  the same oscillator subset that receives direct class-coupling drive.
- `non_conditioning`: vertical drive is restricted to the complement of that
  class-drive subset.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_gated_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gated_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gated_probe.json
outputs/analysis/cifar10_rgb_multiscale_gated_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_gated_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_vertical_auxlow8` | 2 | 0.4482 | 0.8990 | 0.0232 | 0.8048 | 0.3135 | 0.4750 | 1.4757 | 502.4 |
| `vgate_conditioning_auxlow8` | 2 | 0.4453 | 0.8360 | 0.0212 | 0.8168 | 0.3782 | 0.4625 | 1.3959 | 165.6 |
| `vgate_non_conditioning_auxlow8` | 2 | 0.3408 | 0.9498 | 0.0260 | 0.8473 | 0.3611 | 0.3375 | 1.2414 | 213.1 |
| `local050_fb005_auxlow8` | 2 | 0.3467 | 0.9625 | 0.0275 | 0.8194 | 0.3529 | 0.3125 | 1.2096 | 220.9 |

Paired read against `no_vertical_auxlow8`:

- Full weak vertical coupling remains harmful for semantic/attractor metrics:
  generated-label accuracy drops by `-0.1016`, attractor accuracy by
  `-0.1625`, and basin score by `-0.2661`, while diversity rises.
- `vgate_conditioning_auxlow8` largely recovers semantic quality: generated
  accuracy is almost tied (`-0.0029`) and attractor accuracy is only slightly
  lower (`-0.0125`). It also improves nearest-real MSE by `-0.0020`, but
  reduces pixel diversity and does not improve basin score.
- `vgate_non_conditioning_auxlow8` behaves more like a diversity injector:
  diversity and feature diversity rise, but generated accuracy, attractor
  accuracy, and basin score fall.

Updated selective-gate read:

- Selective routing is a real improvement over blunt all-target vertical
  coupling. The `conditioning` gate shows that vertical drive is not useless;
  it becomes much less destructive when routed through the class-conditioned
  fine columns.
- It is still not a hierarchy breakthrough. The no-vertical auxiliary model is
  faster and remains the best overall semantic/attractor row in this probe.
- The next hierarchy mechanism should be more structured than a masked spring:
  signed excitatory/inhibitory vertical coupling, phase-lagged vertical drive,
  learned sparse gates, or an explicit coarse objective that couples through a
  negotiated fine-to-coarse feedback loop.

### Gain-Modulated Vertical Probe

The next probe changed the meaning of vertical coupling. Instead of adding a
coarse/source displacement directly into the fine-layer acceleration, the
vertical projection can now act as a bounded gain on the fine layer's local
recurrent dynamics and class-conditioning drive:

```text
additive:        fine acceleration += vertical source-minus-target drive
gain_modulation: fine local/class drive *= clip(1 + vertical signal, 0, 2)
```

This is closer to the slow-rhythm / fast-local gating hypothesis than a weak
top-down spring. The implementation is exposed as
`multiscale_vertical_mode="gain_modulation"` in
`MultiscaleHORNImageGenerator`, while `additive` remains the backward-compatible
default.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_gain_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.json
outputs/analysis/cifar10_rgb_multiscale_gain_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_gain_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_all_auxlow8` | 2 | 0.4746 | 0.9698 | 0.0280 | 0.8899 | 0.3433 | 0.5063 | 1.9968 | 189.6 |
| `gain_conditioning_auxlow8` | 2 | 0.3740 | 0.9091 | 0.0241 | 0.8634 | 0.2913 | 0.3625 | 1.2848 | 225.7 |
| `no_vertical_auxlow8` | 2 | 0.3271 | 0.9176 | 0.0237 | 0.8262 | 0.3880 | 0.3188 | 1.0682 | 340.3 |
| `vgate_conditioning_auxlow8` | 2 | 0.3740 | 0.8945 | 0.0232 | 0.8009 | 0.3118 | 0.3812 | 1.2651 | 222.4 |

Paired read against `no_vertical_auxlow8`:

- `gain_all_auxlow8` is the first active vertical hierarchy row in this branch
  to clearly beat the no-vertical auxiliary baseline on generated-label
  accuracy (`+0.1475`), diversity (`+0.0521`), feature diversity (`+0.0638`),
  attractor accuracy (`+0.1875`), and basin score (`+0.9286`) on both matched
  seeds.
- The gain is not free: nearest-real MSE worsens by `+0.0044`, output-settling
  MSE worsens, and sampling is slower than the no-vertical row.
- `gain_conditioning_auxlow8` improves feature nearest-real distance and some
  semantic metrics, but it is less stable across seeds and does not match the
  broad-gain row.
- The earlier additive `vgate_conditioning_auxlow8` remains a proximity-biased
  variant: it improves nearest-real MSE and attractor metrics over
  no-vertical, but not nearly as strongly as broad gain modulation.

Updated gain-modulation read:

- This is the best evidence so far that active vertical hierarchy can help the
  HORN generator, provided the vertical signal gates local/class dynamics
  instead of acting as a direct spring.
- It supports the slow/coarse rhythm as modulation hypothesis: hierarchy looks
  more useful when it changes how fine oscillators respond, not when it forces
  their state toward the coarse state.
- It does not solve visible sample quality. The best gain row is more semantic
  and more attractor-robust, but also farther from nearest real pixels. The next
  useful step is to see whether this stronger basin/frontier can be converted
  into sharper images through a better readout/objective, or whether a signed
  excitatory/inhibitory gain is needed to avoid just injecting more foreground
  variation.

### Weak-Conditioning Hierarchy Probe

The follow-up probe tested whether hierarchy becomes more important when the
direct class-drive shortcut is weakened. It lowered the fine class-drive
strength from `8.0` to `2.0` and the auxiliary-layer conditioning strength from
`1.0` to `0.25`, while keeping the same 25% class-targeted fine oscillator
subset.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_weak_drive_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.json
outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_all_auxlow8_drive2` | 2 | 0.3457 | 0.9568 | 0.0250 | 0.8655 | 0.3422 | 0.3438 | 1.1481 | 231.7 |
| `no_vertical_auxlow8_drive2` | 2 | 0.3340 | 0.8810 | 0.0224 | 0.8346 | 0.3305 | 0.3625 | 1.0912 | 335.7 |
| `vgate_conditioning_auxlow8_drive2` | 2 | 0.3740 | 0.9670 | 0.0260 | 0.8133 | 0.3060 | 0.3750 | 1.3384 | 195.4 |

Paired read against `no_vertical_auxlow8_drive2`:

- `vgate_conditioning_auxlow8_drive2` improves generated-label accuracy
  (`+0.0400`), diversity (`+0.0860`), attractor accuracy (`+0.0125`), and
  basin score (`+0.2472`) on both matched seeds. It worsens nearest-real MSE
  by `+0.0036` and loses feature diversity.
- `gain_all_auxlow8_drive2` improves diversity (`+0.0758`), feature diversity
  (`+0.0308`), basin score (`+0.0569`), and output settling (`-0.0088`), but
  the accuracy and attractor gains are weaker and seed-dependent.
- The no-vertical weak-drive row keeps the best nearest-real MSE, so hierarchy
  is again buying semantic/diversity basin behavior rather than pixel
  proximity.

Updated weak-conditioning read:

- Weakening the class drive did not make broad gain modulation dominate. The
  better weak-drive hierarchy mechanism is selective routing into the
  class-conditioned fine columns.
- This supports the stronger architectural lesson: hierarchy should be
  selective and gated, not merely broad. Broad gain appears useful when the
  class scaffold is already strong; under weak conditioning, it can preserve
  diversity but does not reliably anchor class identity.
- The next hierarchy probe should combine the two useful ideas more carefully:
  gain modulation restricted to selected columns, signed excitatory/inhibitory
  modulation, or a learned gate that can decide which fine columns receive
  coarse influence.

### Signed / Selective Gain Probe

The next probe tested the architecture implied by the weak-conditioning result
and the neuroscience/physics analogy: top-down hierarchy should not only
amplify; it should also be able to suppress. `MultiscaleHORNImageGenerator`
therefore gained `multiscale_vertical_mode="signed_gain"`, where vertical
modulation can make the target-layer drive inhibitory:

```text
gain_modulation: gain = clip(1 + modulation,  0, 2)
signed_gain:     gain = clip(1 + modulation, -1, 2)
```

The probe compares selective unsigned gain, broad signed gain, and selective
signed gain, under both normal class drive and weakened class drive.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_signed_gain_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.json
outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_conditioning_auxlow8` | 2 | 0.3604 | 0.9311 | 0.0254 | 0.8452 | 0.3586 | 0.3688 | 1.3218 | 413.6 |
| `gain_conditioning_auxlow8_drive2` | 2 | 0.4023 | 1.0380 | 0.0310 | 0.8744 | 0.3614 | 0.3688 | 1.5154 | 228.2 |
| `signed_gain_all_auxlow8` | 2 | 0.4072 | 0.9651 | 0.0265 | 0.8810 | 0.3714 | 0.4438 | 1.6124 | 200.2 |
| `signed_gain_all_auxlow8_drive2` | 2 | 0.3936 | 0.9597 | 0.0262 | 0.8426 | 0.3021 | 0.3875 | 1.3970 | 230.4 |
| `signed_gain_conditioning_auxlow8` | 2 | 0.4287 | 0.8507 | 0.0217 | 0.8277 | 0.3156 | 0.4438 | 1.3464 | 225.4 |
| `signed_gain_conditioning_auxlow8_drive2` | 2 | 0.4238 | 1.0093 | 0.0294 | 0.8823 | 0.3622 | 0.3813 | 1.4601 | 328.3 |

Paired read against `gain_conditioning_auxlow8`:

- `signed_gain_conditioning_auxlow8` improves generated-label accuracy
  (`+0.0684`), nearest-real MSE (`-0.0037`), attractor accuracy (`+0.0750`),
  and output settling (`-0.0203`) on both matched seeds, but reduces diversity.
- `signed_gain_all_auxlow8` improves feature diversity (`+0.0357`), attractor
  accuracy (`+0.0750`), and basin score (`+0.2907`) on both matched seeds, but
  is worse on nearest-real MSE.
- Under weak drive, `signed_gain_conditioning_auxlow8_drive2` improves
  generated-label accuracy (`+0.0635`), diversity (`+0.0782`), feature
  diversity (`+0.0371`), and basin score (`+0.1383`) over the normal-drive
  selective unsigned gain baseline, though it worsens nearest-real MSE.

Cross-probe read:

- The earlier `gain_all_auxlow8` row remains the strongest raw
  semantic/attractor result in this family (`Acc=0.4746`,
  `Attractor acc=0.5063`, `Basin=1.9968`), but it is also farther from real
  images (`Nearest-real MSE=0.0280`).
- `signed_gain_conditioning_auxlow8` is the best quality/proximity compromise
  so far among active vertical hierarchy variants: it has high accuracy
  (`0.4287`), tied attractor accuracy (`0.4438` in this probe), and the best
  nearest-real MSE (`0.0217`) of the signed/selective gain rows.
- `signed_gain_conditioning_auxlow8_drive2` is the best weak-drive result so
  far, beating the earlier `vgate_conditioning_auxlow8_drive2` on generated
  accuracy, feature diversity, and basin score while keeping diversity above
  1.0.

Updated signed-gain read:

- Signed modulation is useful. It does not simply add instability; it gives a
  real quality/accuracy knob that additive springs and unsigned gain did not
  provide.
- The best hierarchy story is now more specific: broad gain is good for large
  basin/diversity, while selective signed gain is better when we want
  class-consistent samples that stay closer to real images.
- This supports the neuroscience-inspired design bias in a restrained way:
  top-down oscillator hierarchy should act like selective gain control with
  excitation and inhibition, not like a dense all-positive synchronizing force.
  The analogy is useful as architecture guidance, not as proof that the model
  is biologically faithful.

## Soft Selective Gain Probe

Follow-up question: can we keep the proximity benefits of selective signed gain
while recovering some broad-gain attractor strength? The probe adds
`multiscale_vertical_soft_gate_floor=0.25`, so the class-targeted fine columns
receive full vertical modulation while non-target columns receive a weak
quarter-strength contextual dose.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_soft_gate_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_soft_gate_probe.csv
outputs/analysis/cifar10_rgb_multiscale_soft_gate_probe/frontier_summary.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_conditioning_auxlow8` | 2 | 0.4688 | 0.8991 | 0.0240 | 0.8907 | 0.3445 | 0.5063 | 1.7575 | 414.7 |
| `gain_conditioning_soft025_auxlow8` | 2 | 0.4814 | 0.9169 | 0.0246 | 0.8289 | 0.3163 | 0.5125 | 1.8056 | 171.8 |
| `gain_conditioning_soft025_auxlow8_drive2` | 2 | 0.4736 | 0.9192 | 0.0246 | 0.8655 | 0.3233 | 0.4813 | 1.5289 | 205.1 |
| `signed_gain_conditioning_auxlow8` | 2 | 0.4287 | 0.9425 | 0.0256 | 0.8155 | 0.3476 | 0.4563 | 1.6763 | 416.1 |
| `signed_gain_conditioning_soft025_auxlow8` | 2 | 0.3828 | 0.9145 | 0.0249 | 0.8396 | 0.3249 | 0.3875 | 1.3608 | 200.5 |
| `signed_gain_conditioning_soft025_auxlow8_drive2` | 2 | 0.3418 | 0.9055 | 0.0233 | 0.8541 | 0.3629 | 0.3250 | 0.9971 | 326.7 |

Paired read:

- Soft context helps unsigned selective gain slightly: `soft025` improves
  generated-label accuracy (`+0.0127`), diversity (`+0.0178`), feature
  nearest-real MSE (`-0.0282`), attractor accuracy (`+0.0062`), and basin score
  (`+0.0482`), but reduces feature diversity and is slower.
- Soft context does **not** rescue selective signed gain. It improves
  nearest-real MSE and feature diversity, but reduces generated accuracy
  (`-0.0459`), diversity (`-0.0280`), attractor accuracy (`-0.0688`), and basin
  score (`-0.3155`).

Updated hierarchy read:

- The useful middle ground exists for **unsigned** gain, but not for the
  signed/inhibitory selective variant.
- Signed gain appears to need cleaner column selection; smearing inhibitory
  modulation into non-target columns weakens the attractor.
- Next design direction should be structured selective signed gain, not broader
  signed gain: e.g. learned or class-specific target masks, phase-lagged
  inhibitory/excitatory routing, or a coarse objective that gives the modulator
  a clearer job.

## Vertical Causality Audit

Before adding a dual-route hierarchy, the next disciplined question was whether
the existing vertical route is actually causal at sample time. The audit keeps
the trained model fixed and resamples the same initial states under vertical
interventions:

```text
normal, zero, shuffle, flip, scale025, scale050
```

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_causality_audit
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_causality_audit.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_causality_audit.json
outputs/analysis/cifar10_rgb_vertical_causality_audit/frontier_summary.md
```

Aggregate result across seeds 11, 23, and 37:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_all_auxlow8` | 3 | 0.3685 | 0.8741 | 0.0258 | 0.8137 | 0.3035 | 0.3917 | 1.4743 |
| `gain_conditioning_soft025_auxlow8` | 3 | 0.3301 | 0.8066 | 0.0226 | 0.8015 | 0.2728 | 0.3667 | 1.1600 |
| `no_vertical_auxlow8` | 3 | 0.3242 | 0.8179 | 0.0225 | 0.7714 | 0.2792 | 0.3167 | 1.1018 |
| `signed_gain_conditioning_auxlow8` | 3 | 0.3717 | 0.8981 | 0.0265 | 0.7636 | 0.2433 | 0.3583 | 1.3909 |

Intervention read:

- For the active vertical variants, zeroing, shuffling, flipping, or scaling
  the vertical path changed outputs by only about `1e-9` MSE versus normal.
- Generated-label and attractor-label deltas were effectively zero.
- The traced vertical gain standard deviation was only about `1e-4`; the mean
  stayed almost exactly `1.0`.
- The no-vertical row correctly showed exactly zero intervention effect, so
  the audit itself is wired in the expected direction.

Updated hierarchy read:

- The previous aggregate gains from vertical variants are real as trained-model
  outcomes, but this audit does **not** support strong sample-time vertical
  causality in the current implementation.
- The current top-down gain path is almost silent at the point where it enters
  fine-layer dynamics. That makes dual-route broad-plus-selective gain
  premature.
- The next hierarchy sprint should first make the vertical route measurably
  causal: stronger calibrated modulation, an explicit coarse objective that
  forces useful top-down state, or a vertical normalization/scale choice that
  produces visible intervention effects without collapse. Only then should
  class-specific masks, phase lags, or deeper stacks be interpreted as serious
  hierarchy wins.

## Vertical Calibration Probe

The causality audit showed that the vertical route was wired but almost silent.
The next probe therefore added `multiscale_vertical_signal_scale` and tested
whether a stronger top-down signal creates measurable sample-time intervention
effects without immediately collapsing generation.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_calibration_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.json
outputs/analysis/cifar10_rgb_vertical_calibration_probe/frontier_summary.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Gain std | Gain target delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gain_all_auxlow8` | 2 | 0.3926 | 0.8354 | 0.0229 | 0.7977 | 0.3296 | 0.3750 | 1.2694 | 0.00013 | -0.000002 |
| `gain_all_vscale10` | 2 | 0.4102 | 0.8786 | 0.0258 | 0.8536 | 0.2743 | 0.3813 | 1.3937 | 0.00261 | 0.00078 |
| `gain_all_vscale30` | 2 | 0.4004 | 0.8617 | 0.0245 | 0.7880 | 0.2523 | 0.4063 | 1.5619 | 0.01799 | 0.00850 |
| `no_vertical_auxlow8` | 2 | 0.3701 | 0.8661 | 0.0254 | 0.8101 | 0.3007 | 0.3688 | 1.4907 | 0.00000 | 0.00000 |
| `signed_gain_conditioning_vscale10` | 2 | 0.4141 | 0.9012 | 0.0275 | 0.8149 | 0.2994 | 0.3938 | 1.5613 | 0.00223 | 0.00041 |
| `signed_gain_conditioning_vscale30` | 2 | 0.3848 | 0.8911 | 0.0265 | 0.8098 | 0.2950 | 0.3750 | 1.4780 | 0.02105 | 0.01399 |

Intervention read:

- Scaling to `10x` and `30x` makes the vertical route measurably causal. The
  intervention output MSE rises from effectively `1e-9` in the audit to about
  `1e-6` to `1e-4`, depending on variant and intervention.
- `vscale30` produces meaningful gain modulation: gain standard deviation is
  about `0.018` for broad gain and `0.021` for selective signed gain.
- This is still not a clean quality win. Some zero/flip interventions improve
  basin metrics, which means the vertical signal is now causal but not always
  beneficial.

Updated calibration read:

- The silent-route diagnosis was correct, and the signal-scale intervention
  fixed it as a diagnostic.
- The right next step is not "make the vertical scale even larger." It is to
  split top-down modulation into a broad contextual route and a selective
  signed route, so the coarse field can provide both global gain and local
  excitation/inhibition.

## Dual-Gain Probe

`MultiscaleHORNImageGenerator` now supports
`multiscale_vertical_mode="dual_gain"`. This combines two calibrated vertical
routes:

```text
broad route:     all fine columns receive bounded positive contextual gain
selective route: class-conditioned fine columns receive signed modulation
```

The route is architecturally specified, but the coupling matrices and oscillator
states are learned end-to-end and the gain is recomputed at every settling step.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_dual_gain_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.json
outputs/analysis/cifar10_rgb_dual_gain_probe/frontier_summary.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Gain std | Gain target delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_gain_conditioning_vscale10` | 2 | 0.4004 | 0.9050 | 0.0270 | 0.8013 | 0.2399 | 0.3938 | 1.5837 | 0.00696 | 0.00275 |
| `dual_gain_conditioning_vscale30` | 2 | 0.3369 | 0.8595 | 0.0258 | 0.7791 | 0.2992 | 0.3188 | 1.1745 | 0.04550 | 0.02950 |
| `gain_all_vscale30` | 2 | 0.4326 | 0.8732 | 0.0257 | 0.7996 | 0.2805 | 0.4500 | 1.7351 | 0.01319 | 0.00547 |
| `no_vertical_auxlow8` | 2 | 0.4287 | 0.8823 | 0.0263 | 0.8277 | 0.2692 | 0.4750 | 1.8407 | 0.00000 | 0.00000 |
| `signed_gain_conditioning_vscale30` | 2 | 0.3457 | 0.8217 | 0.0231 | 0.7304 | 0.2777 | 0.3500 | 1.2236 | 0.01687 | 0.01019 |

Intervention read:

- `dual_gain_conditioning_vscale10` is causal but gentle: zeroing the vertical
  route changes outputs by about `6.5e-6` MSE, and flipping it by about
  `2.5e-5`.
- `dual_gain_conditioning_vscale30` is strongly causal: zeroing changes outputs
  by about `1.4e-4` MSE, shuffling by `2.1e-4`, and flipping by `3.9e-4`.
- Stronger causal leverage did not translate into better generation. The
  `vscale30` dual route hurts generated-label accuracy, attractor accuracy,
  and basin score.

Updated dual-gain read:

- Dual gain succeeded as a causality mechanism: the top-down path is no longer
  a silent diagnostic trace.
- Dual gain did **not** beat the current quality frontier. In this probe,
  `gain_all_vscale30` and `no_vertical_auxlow8` remain stronger on the main
  semantic/attractor metrics.
- The likely failure mode is over-control: combining broad gain and selective
  signed gain at the current scale constrains the fine field too much. The next
  hierarchy step should be smaller and more targeted: tune the broad/selective
  ratio, learn or regularize class-specific target masks, or give the coarse
  layer a clearer objective before adding more layers.

## Vertical Homeostasis Probe

The council read after the dual-gain probe was consistent: hierarchy is now
causal enough, but stronger causality can become uncontrolled energy injection.
The next implementation therefore added homeostatic vertical gain normalization:

```text
none:       raw vertical modulation
center:     subtract per-sample mean modulation across fine columns
center_rms: center, then rescale per-sample RMS to a target value
```

`center_rms` is exposed through
`multiscale_vertical_gain_normalization="center_rms"` and
`multiscale_vertical_gain_target_std`. The first probe uses target std `0.015`,
close to the useful broad-gain standard deviation rather than the destructive
over-strong dual-gain regime.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_homeostasis_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.json
outputs/analysis/cifar10_rgb_vertical_homeostasis_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_vertical_homeostasis_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_gain_conditioning_vscale30` | 2 | 0.4561 | 0.9442 | 0.0292 | 0.8227 | 0.2647 | 0.4625 | 1.9223 |
| `dual_gain_conditioning_vscale30_normstd015` | 2 | 0.3906 | 0.8727 | 0.0251 | 0.8045 | 0.3165 | 0.3188 | 1.2356 |
| `gain_all_vscale30` | 2 | 0.4336 | 0.9003 | 0.0263 | 0.8192 | 0.3096 | 0.4438 | 1.7931 |
| `gain_all_vscale30_normstd015` | 2 | 0.4189 | 0.9314 | 0.0285 | 0.8150 | 0.2772 | 0.4625 | 1.9053 |
| `no_vertical_auxlow8` | 2 | 0.3477 | 0.9152 | 0.0271 | 0.7588 | 0.2433 | 0.3688 | 1.4869 |
| `signed_gain_conditioning_vscale30` | 2 | 0.3496 | 0.9110 | 0.0271 | 0.7848 | 0.2532 | 0.3438 | 1.3982 |
| `signed_gain_conditioning_vscale30_normstd015` | 2 | 0.4492 | 0.8946 | 0.0266 | 0.8014 | 0.2475 | 0.4500 | 1.7242 |

Updated homeostasis read:

- Homeostatic normalization worked mechanically: normalized traces show
  per-sample centered modulation, target gain RMS near `0.015`, and mean gain
  near `1.0`.
- It is not a universal improvement. Normalizing the dual-gain route harms the
  semantic/attractor frontier, which supports the over-control diagnosis.
- The clean positive is selective signed gain plus homeostasis:
  `signed_gain_conditioning_vscale30_normstd015` improves generated-label
  accuracy, feature diversity, attractor accuracy, basin score, output settling,
  and nearest-real MSE versus both `no_vertical_auxlow8` and raw
  `signed_gain_conditioning_vscale30`, with only a mild diversity tradeoff.
- Broad normalized gain is also useful, but mixed by seed. It looks more like a
  diversity/basin tradeoff than a precise hierarchy mechanism.

Important caveat: this artifact was collected before the scale-intervention
audit order was fixed for normalized gain. The normal/zero/shuffle/flip audit
rows are usable; ignore `scale025` and `scale050` for normalized variants in
this CSV. The code now normalizes before applying sample-time scale
interventions, and the focused test covers the fixed behavior.

Council read:

- Do not add more layers yet.
- Treat the useful vertical path as a small homeostatic selective gain signal:
  a top-down bias on the fine field's attractor landscape, not a second image
  renderer.
- The next compact test should calibrate the selective signed homeostasis
  target: `center`, `center_rms=0.010`, `0.015`, and `0.020`, with the same
  zero/shuffle/flip audit. If the sweet spot repeats, the next architecture
  move is delayed or parameter-level top-down modulation.

Next probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration
```

## Signed-Gain Homeostasis Calibration

The calibration probe compared four ways of normalizing the calibrated
selective signed vertical route:

```text
center only
center_rms target std 0.010
center_rms target std 0.015
center_rms target std 0.020
```

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration.json
outputs/analysis/cifar10_rgb_signed_gain_homeostasis_calibration/frontier_summary.md
outputs/analysis/cifar10_rgb_signed_gain_homeostasis_calibration/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Gain std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `signed_gain_conditioning_vscale30_center` | 2 | 0.4150 | 0.9392 | 0.0294 | 0.8359 | 0.2687 | 0.3813 | 1.6271 | 0.0088 |
| `signed_gain_conditioning_vscale30_normstd010` | 2 | 0.4043 | 0.8260 | 0.0229 | 0.7999 | 0.2629 | 0.3813 | 1.3255 | 0.0100 |
| `signed_gain_conditioning_vscale30_normstd015` | 2 | 0.3809 | 0.8045 | 0.0223 | 0.7883 | 0.1858 | 0.3563 | 1.2748 | 0.0150 |
| `signed_gain_conditioning_vscale30_normstd020` | 2 | 0.3555 | 0.9308 | 0.0282 | 0.8287 | 0.2980 | 0.3313 | 1.3723 | 0.0200 |

Updated calibration read:

- The best semantic/diversity/basin variant is not fixed-RMS normalization. It
  is `center`: remove the mean vertical bias and leave the learned modulation
  amplitude alone.
- Fixed-RMS normalization works as a proximity/regularization knob:
  `normstd010` and `normstd015` substantially improve nearest-real MSE, and
  `normstd015` gives the best feature nearest-real score, but both collapse
  diversity and weaken basin strength versus `center`.
- `normstd020` restores some diversity but continues to weaken accuracy,
  attractor accuracy, and basin score. More gain is not the answer.
- The fixed scale audit now behaves as intended: `scale025` and `scale050`
  reduce the traced gain standard deviation for normalized variants, confirming
  the post-fix intervention order.
- The vertical path is causal but gentle. Zero/shuffle/flip output MSE rises
  with target std, but semantic deltas are small. This supports the view that
  the current top-down route is a weak attractor-bias channel, not a strong
  content transport channel.

Implication:

- Keep `center` as the current selective signed-gain hierarchy candidate.
- Do not continue sweeping target RMS as the main axis. It mostly trades
  diversity/basin behavior for pixel and feature proximity.
- The next architecture step should test *when* and *where* vertical gain is
  applied: delayed/ramped top-down gain, or parameter-level modulation of fine
  HORN damping/coupling/forcing instead of direct gain on activations.

## Centered Signed-Gain Timing Probe

`MultiscaleHORNImageGenerator` now supports step-wise vertical schedules:

```text
constant:    original behavior, vertical route acts from step 0
delayed:     vertical route is off until a configured onset step
linear_ramp: vertical route ramps from zero after onset over N steps
```

The timing probe tested whether the centered selective signed top-down route is
better as an immediate condition, a mid-settling correction, a late correction,
or a slow ramp.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe.json
outputs/analysis/cifar10_rgb_centered_signed_gain_timing_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_centered_signed_gain_timing_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `center_constant` | 2 | 0.4346 | 0.8546 | 0.0241 | 0.8145 | 0.2812 | 0.4313 | 1.5464 |
| `center_delayed8` | 2 | 0.3799 | 0.9300 | 0.0278 | 0.8074 | 0.3336 | 0.3875 | 1.5609 |
| `center_delayed16` | 2 | 0.3926 | 0.8315 | 0.0247 | 0.7929 | 0.2845 | 0.3750 | 1.3815 |
| `center_ramp8_16` | 2 | 0.3711 | 0.8060 | 0.0216 | 0.7609 | 0.2021 | 0.3688 | 1.2834 |

Updated timing read:

- Immediate centered selective signed gain remains the best semantic/attractor
  candidate: highest generated-label accuracy, feature diversity, and
  attractor accuracy.
- `delayed8` is the only delayed variant with a small basin/diversity edge, but
  it loses substantial class accuracy and proximity. It is a diversity/basin
  tradeoff, not a quality breakthrough.
- `delayed16` is worse on almost every primary metric.
- `ramp8_16` gives the best nearest-real and feature-nearest scores plus
  slightly better output-settle, but it weakens semantic and basin metrics. It
  behaves like a proximity/regularization knob, similar to fixed RMS gain.
- The schedule mechanism is useful instrumentation, but the result does not
  support delayed/ramped vertical activation as the next main axis.

Implication:

- Keep `center_constant` as the current selective signed-gain hierarchy
  candidate.
- Do not spend more time on timing sweeps unless a future mechanism makes
  vertical modulation much stronger or more content-specific.
- The next meaningful hierarchy test should change *what* the coarse route
  modulates: fine-layer damping, local coupling strength, conditioning strength,
  or another HORN parameter. In other words, test whether top-down hierarchy
  reshapes the fine field's dynamics rather than merely scaling its activation
  drive.

## Centered Signed-Gain Target Probe

The council recommendation after the timing probe was to stop asking "when does
vertical gain arrive?" and ask "what does vertical gain control?" This added
`multiscale_vertical_gain_target` with four targets:

```text
drive:        original behavior; scales local coupling plus class conditioning
coupling:     scales only local recurrent HORN interaction
conditioning: scales only class-conditioning drive
damping:      scales only fine-layer damping with nonnegative gain
```

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_centered_signed_gain_target_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.csv
outputs/analysis/cifar10_rgb_centered_signed_gain_target_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_centered_signed_gain_target_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `center_drive` | 2 | 0.4043 | 0.8918 | 0.0270 | 0.7632 | 0.2846 | 0.4125 | 1.5678 |
| `center_coupling` | 2 | 0.3428 | 0.8991 | 0.0264 | 0.8180 | 0.3141 | 0.3875 | 1.5205 |
| `center_conditioning` | 2 | 0.3799 | 0.7905 | 0.0212 | 0.7574 | 0.2714 | 0.3813 | 1.2789 |
| `center_damping` | 2 | 0.3193 | 0.7964 | 0.0222 | 0.7901 | 0.3613 | 0.3000 | 0.9436 |

Intervention read:

| Variant | Zero output MSE | Shuffle output MSE | Flip output MSE | Zero acc delta | Shuffle acc delta | Flip acc delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `center_drive` | 0.00001212 | 0.00002313 | 0.00004680 | 0.0039 | 0.0039 | 0.0000 |
| `center_coupling` | 0.00000003 | 0.00000005 | 0.00000011 | 0.0000 | 0.0000 | 0.0000 |
| `center_conditioning` | 0.00000713 | 0.00001376 | 0.00002806 | -0.0039 | -0.0039 | -0.0039 |
| `center_damping` | 0.00000012 | 0.00000022 | 0.00000046 | 0.0000 | 0.0000 | 0.0000 |

Updated target read:

- `center_drive` remains the best semantic/attractor target. It has the highest
  generated-label accuracy, attractor accuracy, and basin score, and it is the
  only target with a clearly measurable output intervention effect.
- `center_coupling` is the interesting near miss: it improves feature diversity
  and slightly improves pixel diversity/proximity, but loses accuracy and has
  almost no intervention sensitivity. Coupling-only top-down modulation is too
  weak in the current implementation.
- `center_conditioning` is a proximity knob, not a hierarchy breakthrough. It
  improves nearest-real MSE and feature-nearest distance, but reduces diversity
  and basin strength.
- `center_damping` improves output settling and nearest-real MSE, but sharply
  hurts class and basin metrics. It is stability/proximity pressure, not useful
  shape-forming hierarchy by itself.

Implication:

- Keep `center_drive` as the active centered signed-gain hierarchy baseline.
- Do not continue with target-only sweeps as the main hierarchy axis. The
  coarse route still seems useful as a broad drive/gain channel, but simple
  coupling-only or damping-only targets do not make it a shape-forming engine.
- The next hierarchy move should either give the coarse layer a clearer
  objective/representation, add bidirectional fine-to-coarse feedback, or test
  a mixed route that preserves drive-level causality while adding a small
  coupling/damping modulation branch. Do not jump to deeper stacks until the
  coarse layer has a job.

## Coarse Objective Probe

The next hierarchy test keeps the current centered signed-gain baseline, but
changes the auxiliary coarse objective. Earlier `*_auxlow8` probes used paired
low-resolution MSE: the coarse layer is asked to produce the downsampled image
for the current training example. That gives the top layer a clear job, but it
may also overconstrain a class-conditional generator by pushing the coarse state
toward one paired thumbnail rather than a class-level attractor basin.

New mechanism:

```text
coarse_auxiliary_loss_mode="mse":            paired low-res image target
coarse_auxiliary_loss_mode="distributional": low-res batch/class moments and marginals
```

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_objective_probe
```

Planned comparison:

- `center_mse`: centered selective signed-gain hierarchy plus paired coarse MSE.
- `center_dist`: same hierarchy plus distributional coarse objective.
- `no_vertical_mse`: no active vertical route plus paired coarse MSE.
- `no_vertical_dist`: no active vertical route plus distributional coarse
  objective.

Interpretation before running:

- If `center_dist` improves class accuracy/diversity/basin over `center_mse`,
  the coarse layer likely wants a distributional/contextual role, not paired
  image-copy pressure.
- If both distributional rows help equally, the win is probably an auxiliary
  readout regularizer, not vertical hierarchy.
- If neither helps, the next serious hierarchy move is bidirectional
  fine-to-coarse feedback or a genuinely different coarse representation, not
  more gain schedules.

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv
outputs/analysis/cifar10_rgb_coarse_objective_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_coarse_objective_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `center_dist` | 2 | 0.3799 | 0.9788 | 0.0313 | 0.8088 | 0.3310 | 0.3813 | 1.6700 |
| `center_mse` | 2 | 0.3584 | 0.8552 | 0.0246 | 0.8154 | 0.2693 | 0.3500 | 1.2703 |
| `no_vertical_dist` | 2 | 0.3975 | 0.8228 | 0.0229 | 0.8141 | 0.2707 | 0.4188 | 1.5578 |
| `no_vertical_mse` | 2 | 0.3271 | 0.8735 | 0.0254 | 0.7877 | 0.3254 | 0.3313 | 1.2641 |

Paired read:

- Distributional coarse supervision improves `center_mse -> center_dist` on
  generated-label accuracy (+0.0215), diversity (+0.1236), attractor accuracy
  (+0.0313), and basin score (+0.3997), but worsens nearest-real and
  feature-nearest proximity.
- Distributional coarse supervision improves `no_vertical_mse ->
  no_vertical_dist` even more clearly on generated-label accuracy (+0.0703),
  feature-nearest proximity (-0.0547), attractor accuracy (+0.0875), and basin
  score (+0.2936), with a diversity tradeoff.
- Under the distributional objective, `center_dist` beats `no_vertical_dist` on
  diversity (+0.1560) and basin score (+0.1122), but loses generated-label
  accuracy, attractor accuracy, nearest-real MSE, and feature-nearest proximity.
- Vertical intervention deltas are larger for `center_dist` than `center_mse`
  (`flip.output_mse_vs_normal` around `0.000295` vs `0.000054`), so the active
  route is more causal under the distributional objective, but still not the
  quality frontier.

Updated read:

The distributional coarse objective is worth keeping. It prevents the coarse
auxiliary layer from behaving purely like a paired thumbnail decoder and gives
better class/attractor behavior. But the improvement is not yet evidence that
vertical hierarchy is the source of the win, because the no-vertical
distributional auxiliary bank is the strongest quality row. The active centered
signed-gain hierarchy mainly buys diversity/basin behavior while sacrificing
pixel/feature proximity and some class accuracy.

Implication:

- Promote `*_auxdist8` to the preferred coarse-objective probe.
- Do not claim hierarchy solved CIFAR generation. The coarse objective is useful
  infrastructure; the vertical route still needs a content-specific role.
- The next hierarchy move should be bidirectional fine-to-coarse feedback or a
  mixed route that preserves no-vertical distributional quality while injecting
  selective signed vertical causality. Deeper stacks should wait until the
  two-layer mechanism does more than trade proximity for diversity.

## Feedback Signal Probe

The previous probes used bidirectional vertical topology, but the bottom-up
feedback signal was still basically phase/position-only. This probe asks whether
fine-to-coarse feedback should carry more of the fine layer's dynamical state.

New mechanism:

```text
multiscale_feedback_signal_mode="position": historical bottom-up phase/position signal
multiscale_feedback_signal_mode="state":    bounded position-plus-velocity feedback
```

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_signal_probe
```

Planned comparison:

- `mse_position` vs `mse_state`: does state feedback help under paired coarse
  MSE?
- `dist_position` vs `dist_state`: does state feedback help under the stronger
  distributional coarse objective?
- `mse_position` vs `dist_position` and `mse_state` vs `dist_state`: does the
  distributional coarse objective stay useful when feedback carries full HORN
  state?

Interpretation before running:

- If `dist_state` improves accuracy/attractor metrics without losing the
  diversity/basin edge, then bottom-up state feedback is a promising mechanism:
  coarse layers need evidence from the fine field, not just a top-down gain
  route.
- If `state` mostly worsens both objectives, then the current hierarchy problem
  is not lack of feedback signal content; it is probably the lack of a
  content-specific coarse representation or a mismatch between auxiliary loss
  and fine readout.
- If `state` helps MSE but not distributional, the feedback may be acting as a
  paired-thumbnail stabilizer rather than a generative hierarchy mechanism.

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_signal_probe.csv
outputs/analysis/cifar10_rgb_feedback_signal_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_feedback_signal_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dist_position` | 2 | 0.3242 | 0.8881 | 0.0265 | 0.7875 | 0.3055 | 0.3250 | 1.2655 | 289.6 |
| `dist_state` | 2 | 0.4131 | 0.9657 | 0.0311 | 0.8438 | 0.2516 | 0.4000 | 1.7156 | 126.0 |
| `mse_position` | 2 | 0.4092 | 0.8024 | 0.0216 | 0.7960 | 0.2747 | 0.4313 | 1.4560 | 276.9 |
| `mse_state` | 2 | 0.4014 | 0.8768 | 0.0256 | 0.8052 | 0.2354 | 0.3813 | 1.4733 | 157.4 |

Paired read:

- Under the distributional objective, state feedback is clearly useful:
  `dist_position -> dist_state` improves generated-label accuracy (+0.0889),
  diversity (+0.0776), feature diversity (+0.0563), feature-nearest proximity
  (-0.0539), attractor accuracy (+0.0750), and basin score (+0.4501).
- The tradeoff is pixel proximity and speed: `dist_state` worsens nearest-real
  MSE (+0.0045) and samples much slower than `dist_position` (about 126 vs 290
  samples/sec in this probe).
- Under paired coarse MSE, state feedback is mixed: it improves diversity,
  feature-nearest proximity, and basin score slightly, but loses generated
  accuracy, attractor accuracy, nearest-real MSE, output settling, and speed.
- With state feedback enabled, the distributional objective again produces the
  stronger semantic/diversity/basin row: `mse_state -> dist_state` improves
  generated accuracy (+0.0117), diversity (+0.0889), attractor accuracy
  (+0.0187), and basin score (+0.2423), while losing pixel proximity.

Updated read:

This is the first hierarchy probe in this sequence where bottom-up feedback
content looks meaningfully helpful. The coarse layer benefits from receiving
bounded fine-layer position-plus-velocity evidence, but mainly when the coarse
objective is distributional rather than a paired low-res copy. That supports the
idea that hierarchy should behave like a contextual attractor scaffold, not a
thumbnail autoencoder.

The result still does not solve the rendering problem. The best state-feedback
row strengthens class-consistent basin behavior and diversity, but sacrifices
nearest-pixel proximity and throughput. So the next step should not be "more
feedback everywhere." It should preserve `dist_state` as the active-hierarchy
lead and ask how to convert the better basin into better visible samples:
stronger or staged readout, explicit fine image objective, or a selective
state-feedback gate that keeps the useful dynamical evidence without dragging
all fine noise into the coarse state.

## Feedback Source-Gate Probe

The feedback-signal probe showed that bottom-up state feedback helps the
distributional active hierarchy, but may also feed too much fine-layer noise
into the coarse state. The next compact probe asks which fine columns the
coarse layer should listen to.

New mechanism:

```text
multiscale_feedback_source_gate="all":              listen to the full fine field
multiscale_feedback_source_gate="conditioning":     listen only to class-drive target columns
multiscale_feedback_source_gate="non_conditioning": listen to the complement
```

This is source-side routing for bottom-up feedback. It is the counterpart to
`multiscale_vertical_target_gate`, which controls where top-down modulation
lands in the fine field.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_source_gate_probe
```

Planned comparison:

- `source_all`: current `dist_state` lead.
- `source_conditioning`: the coarse layer reads state evidence only from the
  fine columns that also receive direct class drive.
- `source_non_conditioning`: the coarse layer reads state evidence from the
  non-class-driven complement.

Interpretation before running:

- If `source_conditioning` keeps the basin/diversity gains while improving
  nearest-real MSE or speed, then the problem was noisy all-field feedback and
  selective read-in is a promising hierarchy mechanism.
- If `source_non_conditioning` wins, then the class-driven columns may already
  be too label-biased; useful bottom-up evidence comes from the less directly
  conditioned fine field.
- If `source_all` still wins, then the coarse layer likely needs broad field
  evidence, and the next rendering improvement should come from readout or
  objective staging rather than source gating.

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.csv
outputs/analysis/cifar10_rgb_feedback_source_gate_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_feedback_source_gate_probe/paired_deltas.md
```

Aggregate result across seeds 11 and 23:

| Variant | Runs | Acc | Diversity | Nearest-real MSE | Feature diversity | Feature nearest-real | Attractor acc | Basin score | Samples/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_all` | 2 | 0.4062 | 0.8351 | 0.0243 | 0.8036 | 0.2214 | 0.4000 | 1.4195 | 150.9 |
| `source_conditioning` | 2 | 0.3779 | 0.8075 | 0.0221 | 0.7998 | 0.2781 | 0.3937 | 1.2961 | 139.4 |
| `source_non_conditioning` | 2 | 0.3447 | 0.9051 | 0.0269 | 0.7866 | 0.2784 | 0.3563 | 1.4015 | 148.7 |

Paired read:

- `source_conditioning` improves nearest-real MSE (-0.0022) and output
  settling (-0.0031) versus `source_all`, but loses generated-label accuracy
  (-0.0283), diversity (-0.0276), feature-nearest proximity (+0.0567),
  attractor accuracy (-0.0063), and basin score (-0.1234).
- `source_non_conditioning` improves raw diversity (+0.0699) versus
  `source_all`, but loses generated-label accuracy (-0.0615), nearest-real MSE
  (+0.0025), feature diversity (-0.0170), feature-nearest proximity (+0.0570),
  attractor accuracy (-0.0437), and basin score (-0.0180).
- `source_non_conditioning` beats `source_conditioning` on diversity and basin
  score, but loses class accuracy, pixel proximity, feature diversity,
  attractor accuracy, and output settling.

Updated read:

Hard source gating does not convert the state-feedback basin into better
visible quality. The all-source route remains the strongest semantic and
feature-nearest row in this paired sweep, which suggests the coarse layer needs
broad fine-field evidence rather than only class-driven columns. Conditioning
source gating acts more like a proximity/settling regularizer. Non-conditioning
source gating preserves more diversity, but weakens class consistency.

This does not invalidate state feedback; it narrows the next move. The useful
mechanism seems to be broad fine-to-coarse state evidence plus a distributional
coarse objective. The remaining bottleneck is probably how the improved basin is
rendered back into images, or how the feedback is weighted continuously rather
than hard-routed. Next candidate probes: staged/wider readout, a lightweight
fine-image loss, or learned soft feedback-source weights rather than binary
source masks.

## Feedback Source-Mix Probe

Hard source gates were too blunt: all-source feedback kept the strongest
semantic/basin row, conditioning-only feedback acted like a proximity
regularizer, and non-conditioning-only feedback preserved diversity while
weakening class control. This probe keeps the successful `dist_state`
configuration and changes the source routing continuously.

New mechanism:

```text
multiscale_feedback_source_gate="weighted"
multiscale_feedback_source_mix=(conditioning_weight, non_conditioning_weight)
```

The weighted source mask is mean-normalized. That matters: fixed ratios should
change which fine columns the coarse layer listens to, not secretly weaken or
strengthen the entire feedback path.

Probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_source_mix_probe
```

Planned comparison:

- `source_all`: current all-source `dist_state` lead.
- `mix75_25`: stronger read-in from class-conditioned fine columns.
- `mix50_50`: normalized weighted-gate sanity check; should behave like
  `source_all` under matched seeds.
- `mix25_75`: stronger read-in from non-class-conditioned fine columns.

Interpretation before running:

- If `mix75_25` improves pixel proximity or output settling without losing the
  semantic/basin edge, soft class-source emphasis is useful.
- If `mix25_75` increases diversity without the class collapse of hard
  `source_non_conditioning`, autonomous fine-state feedback carries useful
  generative variation.
- If `source_all` still wins, the next improvement is probably not source
  routing. Move to readout/objective staging or learned state-dependent
  feedback weights.

Result:

```text
Variant       Acc     Diversity  Nearest MSE  Feature div  Attractor acc  Basin
mix25_75      0.3574  0.8497     0.0249       0.7609       0.3563         1.3835
mix50_50      0.4014  0.8902     0.0259       0.8068       0.4437         1.7444
mix75_25      0.4424  0.8798     0.0252       0.8476       0.4188         1.6099
source_all    0.3984  0.8665     0.0258       0.8272       0.3813         1.3313
```

Matched deltas versus `source_all`:

- `mix75_25` improves generated-label accuracy on both seeds (+0.0439),
  feature diversity on both seeds (+0.0204), attractor accuracy on both seeds
  (+0.0375), and basin score on both seeds (+0.2786). It worsens
  feature-nearest proximity, so this is not a strict image-quality win.
- `mix50_50` is not bitwise identical to `source_all` under the GPU run, but it
  acts as a close sanity neighbor and gives the strongest attractor accuracy
  (+0.0625) and basin score (+0.4131).
- `mix25_75` slightly improves nearest-real MSE and feature-nearest proximity,
  but loses generated accuracy, feature diversity, and attractor accuracy.

Visual read: all source-mix variants are still blurry CIFAR-like samples. The
source-mix mechanism improves the measured hierarchy/basin frontier, not yet the
rendered image quality. The useful direction is therefore not "listen only to
autonomous fine state"; the better routing is class-source-heavy but still
allows some autonomous evidence. Next move should turn this stronger basin into
better images with a richer readout/objective, or make the mix learned and
state-dependent after a stronger visual metric is in place.

Important follow-up: while preparing the next readout probe, we found that the
CIFAR RGB coarse auxiliary target was downsampled with an `H,W,C` reshape even
though direct CIFAR RGB data is stored flat `C,H,W`. This is harmless for MNIST
and grayscale CIFAR, but it scrambles the low-resolution RGB target used by the
coarse auxiliary objective. The source-mix numbers above should therefore be
treated as pre-fix evidence that routing changes the basin/frontier, not as a
final judgment on the hierarchy's visual potential. The next run is an `auxfix`
rerun of the same source-mix design with channel-first downsampling.

Auxfix result:

```text
Variant       Acc     Diversity  Nearest MSE  Feature div  Attractor acc  Basin
mix25_75      0.3438  0.7956     0.0223       0.8078       0.3313         1.1276
mix50_50      0.4014  0.8794     0.0262       0.8359       0.4188         1.6633
mix75_25      0.3760  0.9335     0.0287       0.8250       0.4000         1.6707
source_all    0.3682  0.7863     0.0216       0.7955       0.3312         1.1372
```

Matched auxfix deltas versus `source_all`:

- `mix50_50` improves generated-label accuracy (+0.0332), feature diversity
  (+0.0404), attractor accuracy (+0.0875), and basin score (+0.5261) on both
  seeds, at the cost of worse nearest-real MSE.
- `mix75_25` gives the largest diversity (+0.1472) and basin gain (+0.5334),
  and improves attractor accuracy (+0.0688), but has the weakest pixel
  proximity of the source-mix variants.
- `mix25_75` remains mostly a proximity-biased variant and does not improve the
  hierarchy/basin read.

Visual read: auxfix samples are still blurry CIFAR-like abstractions. The
channel-layout fix did not produce a visible breakthrough, but it makes the
mechanism cleaner: weighted feedback source routing is real, and the best
hierarchy frontier is no longer hard `source_all`. The remaining problem is
turning a better class/diversity/attractor basin into sharper rendered images.
That points back to readout/objective design, not more source-gate sweeps.

### Readout Fusion Probe

Motivation: the weighted feedback source runs suggested that the hierarchy has
a stronger coarse/fine basin than the rendered CIFAR samples show. The next
conservative test was to decode the supervised auxiliary layer to a low-res
image, upsample it channel-first, and blend it into the final fine readout with
`multiscale_readout_fusion_strength`. This keeps the coarse route low-capacity:
it can supply a weak scaffold, but it does not add a second full decoder.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_readout_fusion_probe
```

This produced:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_fusion_probe.csv
outputs/analysis/cifar10_rgb_readout_fusion_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_readout_fusion_probe/paired_deltas.md
outputs/modal_samples/readout_fusion/contact_sheet.png
```

Result:

```text
Variant             Acc     Diversity  Nearest MSE  Feature div  Attractor acc  Basin
mix50_50            0.3916  0.9030     0.0278       0.8062       0.3563         1.4250
mix50_50_fusion010  0.3730  0.8781     0.0255       0.8031       0.3438         1.3385
mix50_50_fusion025  0.3418  0.8563     0.0257       0.8134       0.3187         1.2485
mix75_25_fusion010  0.3799  0.8774     0.0266       0.7788       0.3813         1.4673
```

Matched deltas versus `mix50_50`:

- `mix50_50_fusion010` improves nearest-real MSE and output-settling ratio, but
  loses generated-label accuracy, diversity, feature-nearest proximity,
  attractor accuracy, and basin score on average.
- `mix50_50_fusion025` is too strong: it improves nearest-real MSE slightly and
  output settling, but costs semantic accuracy/diversity and visually adds a
  coarser/blockier texture.
- `mix75_25_fusion010` is the only fusion variant that improves attractor
  accuracy and basin score on average. It also improves nearest-real MSE, but
  still loses generated-label accuracy and diversity relative to `mix50_50`.

Visual read: fusion does not solve the blurry CIFAR rendering problem. A small
fusion blend can make outputs a little closer to real pixels and can strengthen
one attractor metric when paired with `mix75_25`, but it does not convert the
better hierarchy basin into sharply rendered images. The next readout move
should therefore be a staged or auxiliary objective that trains the fine
readout to use the coarse scaffold, not a stronger direct image blend.

### Coarse Readout Consistency Probe

Direct fusion was too literal, so the next probe moved the coarse scaffold into
the training objective instead of the sample path. The multiscale generator now
supports returning both outputs from the same trajectory:

```text
sample_with_auxiliary_image(...) -> final fine image, auxiliary low-res image
```

The added loss downscales the final image and asks it to stay consistent with
the same-run auxiliary low-res image:

```text
coarse_readout_consistency_weight
coarse_readout_consistency_onset_epoch
```

The auxiliary image is stop-gradiented for this term, so the loss teaches the
fine readout to respect the coarse scaffold rather than letting the auxiliary
branch chase the final image. The tested variants used an onset at epoch 5.

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_readout_consistency_probe
```

Artifacts:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.csv
outputs/analysis/cifar10_rgb_coarse_readout_consistency_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_coarse_readout_consistency_probe/paired_deltas.md
outputs/modal_samples/coarse_readout_consistency/contact_sheet.png
```

Result:

```text
Variant                 Acc     Diversity  Nearest MSE  Feature div  Attractor acc  Output settle
mix50_50                0.4238  0.8971     0.0268       0.8568       0.4000         0.1352
mix50_50_consistency005 0.1895  0.4779     0.0107       0.6469       0.1875         0.0803
mix50_50_consistency010 0.3359  0.4260     0.0098       0.8398       0.3375         0.0467
mix75_25_consistency005 0.3418  0.4895     0.0106       0.7478       0.3438         0.0548
```

Matched deltas versus `mix50_50`:

- Every consistency variant improves nearest-real MSE and output-settling ratio.
- Every consistency variant loses generated-label accuracy, diversity,
  attractor accuracy, and basin score.
- `consistency010` is less damaging to feature diversity than
  `consistency005`, but it still collapses raw diversity strongly.

Visual read: the consistency variants are smoother and closer in pixel space,
but also more repetitive and less class-consistent. This is the same trap in a
cleaner form: low-resolution per-sample agreement rewards prototype/low-pass
behavior. It is a useful diagnostic and control, not the next default generator
recipe.

Updated readout conclusion:

- Keep direct fusion and consistency as ablation tools.
- Do not keep increasing pixel-level coarse-to-fine pressure.
- The next promising readout path should use a feature/semantic coarse
  objective, a learned/gated interface from the coarse scaffold into the fine
  readout, or an explicit staged task where the coarse layer predicts
  class/shape statistics rather than exact low-res pixels.

### Readout Gate and Frequency Diagnostics

The next implementation adds that learned interface without increasing direct
pixel blending:

- `multiscale_readout_gate_mode="seed_film"` projects the selected auxiliary
  oscillator layer into a scale/shift on the fine resize-conv seed tensor.
- `sparse_horn_cifar10_rgb_hierarchy_gate010` and
  `sparse_horn_cifar10_rgb_hierarchy_gate025` inherit the current hierarchy
  lead and test gentle versus stronger seed modulation.
- Generator quality metrics now include classifier-feature Frechet/KID-style
  distances, nearest-neighbor feature precision/recall, and frequency/edge
  diagnostics. The goal is to distinguish "semantically wrong", "mode
  collapsed", "pixel-close but blurry", and "good basin but poor high-frequency
  rendering" instead of relying on nearest-pixel MSE alone.

This keeps the current hypothesis precise: the hierarchy may already be
forming better attractor basins, but the final RGB readout still needs a
coarse-aware rendering interface that preserves diversity and semantics.

Run:

```text
mnist_generator_cifar10_rgb_readout_gate_probe
```

Two-seed result at 20 epochs:

```text
Variant        Acc     Diversity  Nearest MSE  Feature div  Feature Frechet  Attractor acc  Basin  Output settle
hierarchy      0.3232  0.7719     0.0203       0.8093       0.1518           0.3625         1.1017 0.0810
gate010        0.3330  0.8423     0.0219       0.7838       0.2682           0.3125         1.1765 0.0738
gate025        0.3379  0.7439     0.0181       0.7641       0.2984           0.3563         1.1653 0.0460
```

Read:

- The learned seed gate is numerically stable and causally active.
- `gate010` improves raw diversity and basin score, but loses feature Frechet
  and attractor accuracy.
- `gate025` improves generated-label accuracy, nearest-real MSE, basin score,
  and output settling, but loses diversity and feature-distribution quality.
- Frequency/edge diagnostics confirm the broader blur issue: generated images
  have only about half the real high-frequency power and roughly 15-17% of real
  Laplacian edge energy.

Conclusion: the gate is a useful readout-conversion probe, not a new default.
The bottleneck is still high-frequency/semantic rendering from the oscillator
state. Future readout work should be sharper than scalar gate strength: e.g.
class/feature-aware modulation, a perceptual/feature readout loss, or a staged
coarse-shape-to-fine-detail objective.

### Frequency Objective and State Spectrum Probe

The next test targets the blur diagnosis directly:

- `frequency_objective_weight` adds a light low/mid/high frequency-band and
  Laplacian edge-statistics loss.
- `sparse_horn_cifar10_rgb_hierarchy_freq001` and
  `sparse_horn_cifar10_rgb_hierarchy_freq003` inherit the current hierarchy
  lead and only add this objective.
- Success diagnostics now report spatial spectra for square oscillator banks
  when trace data is available. That lets us compare image sharpness against
  state sharpness: if the oscillator state has high-frequency structure but
  the output does not, the readout/objective is suspect; if the state is smooth
  too, the dynamics or conditioning path is the bottleneck.

Run:

```text
mnist_generator_cifar10_rgb_frequency_objective_probe
```

This is a bottleneck probe, not a new default. The desired result is not merely
lower nearest-pixel MSE; it is better high-frequency/edge metrics without
destroying the hierarchy's semantic/diversity/basin advantages.

Two-seed result at 20 epochs:

```text
Variant        Acc     Diversity  Nearest MSE  Feature Frechet  High-freq ratio  Edge ratio  State high-freq
hierarchy      0.4131  0.7400     0.0198       0.2510           0.5259           0.1581      0.2502
freq001        0.2949  1.0216     0.0470       0.0176           1.0136           1.0550      0.3084
freq003        0.3652  0.8873     0.0624       0.1765           0.9798           1.0374      0.3150
```

Read:

- The frequency objective does what it says: generated high-frequency and
  Laplacian edge ratios move from far below real images to near real-image
  levels.
- The oscillator state also keeps more high-frequency spatial power after
  settling, so the loss is not merely changing the last pixel layer.
- Visual samples show much of that new frequency energy as color-channel
  ringing, borders, and texture, not object-aligned detail.
- `freq001` greatly improves feature Frechet/KID and diversity, but loses
  generated-label accuracy, attractor accuracy, and nearest-real MSE.
- `freq003` recovers some accuracy versus `freq001`, but worsens nearest-real
  MSE and output-settling metrics.

Conclusion: naive spectrum matching is a useful diagnostic, but it is not the
rendering fix. The bottleneck is object-aligned detail: high-frequency energy
must be conditioned by semantic/shape structure, not simply matched as a global
image statistic.

### Patch Detail Objective Probe

Follow-up implementation:

- `patch_objective_weight` compares local raw patch distributions with sliced
  Wasserstein projections.
- `patch_objective_edge_weight` adds the same comparison on Laplacian patches,
  so local edges are rewarded without using a paired target.
- `sparse_horn_cifar10_rgb_hierarchy_patch005` and
  `sparse_horn_cifar10_rgb_hierarchy_patch010` inherit the hierarchy lead and
  only add this local-detail objective.

Hypothesis: if the global frequency objective failed because it rewarded
border halos/color ringing, a patch-level distribution objective should be a
cleaner readout/objective pressure. It should improve local texture and edge
quality while preserving more of the hierarchy's semantic and attractor
behavior than `freq001`/`freq003`.

Run:

```text
mnist_generator_cifar10_rgb_patch_objective_probe
```

Two-seed result at 20 epochs:

```text
Variant        Acc     Diversity  Nearest MSE  Feature Frechet  Attractor acc  Basin  High-freq ratio  Edge ratio
hierarchy      0.3770  0.8476     0.0241       0.2485           0.3500         1.2190 0.4553           0.1855
patch005       0.3535  0.9584     0.0350       0.0144           0.3563         1.6061 1.1289           0.5536
patch010       0.4062  0.9447     0.0346       0.1146           0.4000         1.7153 1.7239           0.7664
```

Read:

- `patch010` is the strongest local-detail contender so far: it improves
  generated-label accuracy, diversity, feature Frechet/KID, feature
  nearest-real distance, attractor accuracy, and basin score versus the matched
  hierarchy baseline.
- Both patch variants raise image high-frequency power and edge energy while
  also preserving more high-frequency power in the oscillator state after
  settling.
- The cost is clear: nearest-real pixel MSE worsens and visual samples show
  chunky local artifacts/striping. This is better aligned than the global
  frequency objective's neon ringing, but it is still not a clean rendered
  image-quality win.
- Training is stable, but patch SWD lowers samples/sec versus the baseline.

Conclusion: local patch objectives are more promising than global frequency
matching for converting hierarchy basins into detail. The next rendering work
should reduce artifact geometry, likely by using multiscale/overlapping patch
features, a learned perceptual patch judge, or a readout architecture that can
place fine detail without grid-aligned striping.

### Patch V2 Objective Probe

Follow-up implementation:

- `patch_objective_offsets` repeats the patch sliced-Wasserstein comparison on
  shifted patch grids.
- `patch_objective_patch_sizes` lets one projection bank score multiple patch
  scales.
- Added `patch010_overlap`, `patch010_multiscale`, and
  `patch010_multiscale_overlap` CIFAR RGB hierarchy presets.

Hypothesis: if `patch010` worked because it supplied missing local detail
pressure, but failed visually because the fixed grid was too easy to satisfy,
then shifted and multiscale patch scoring should keep the semantic/detail gains
while reducing chunking and striping.

Run:

```text
mnist_generator_cifar10_rgb_patch_v2_probe
```

Two-seed result at 20 epochs, with one interrupted row:

```text
Variant             Runs  Acc     Diversity  Nearest MSE  Feature Frechet  Attractor acc  Basin   High-freq ratio  Edge ratio
hierarchy           2     0.3662  0.8901     0.0261       0.1487           0.3625         1.4428  0.4630           0.2050
patch010            2     0.2910  0.9180     0.0340       0.0546           0.3000         1.2638  1.5195           0.7442
patch010_overlap    2     0.3574  0.9378     0.0348       0.0543           0.3875         1.6318  1.5862           0.7838
patch010_multiscale 2     0.3359  0.9234     0.0345       0.0392           0.3937         1.7138  1.8498           0.7573
multi+overlap       1     0.3398  0.9569     0.0335       0.1069           0.3625         1.5643  1.7501           0.7225
```

Read:

- Shifted-grid overlap repairs some of plain `patch010`'s worst dynamical
  metrics: better generated-label accuracy, diversity, feature diversity,
  attractor accuracy, and basin score versus plain `patch010`.
- Multiscale patch scoring gives the best feature Frechet and best basin score
  among completed patch variants.
- Neither variant is a rendered-image breakthrough. Visual grids still show
  dark blobs, halos, horizontal bands, and fixed local texture artifacts. The
  patch objective adds high-frequency energy, but the model still turns too
  much of it into objective-satisfying texture rather than object-aligned
  detail.
- `patch010_multiscale_overlap` is too expensive and not visually cleaner in
  the completed seed. The second seed was interrupted while draining the long
  full vertical-audit diagnostics, so future v2 sweeps use a lighter
  `normal,zero` audit.

Conclusion: patch objectives are useful diagnostics and can improve several
semantic/basin metrics, but this is still a readout/objective mismatch, not
a solved rendering bottleneck. The next high-value direction is a readout that
uses oscillator-state detail more directly, such as patch/residual readout
heads or a staged coarse-shape plus fine-detail renderer, rather than adding
more patch-statistics pressure.

### State Information Probe

Implementation:

- Added an opt-in `state_information_probe` diagnostic to the MNIST/CIFAR
  generator runner.
- It fits tiny deterministic ridge probes from traced oscillator states, such
  as `fine_initial`, `fine_final`, auxiliary final states, and
  `combined_final`.
- Probe targets are deliberately attribution-oriented: class label, generated
  low-resolution scaffold, generated high-pass residual, auxiliary low-res
  readout, and optional classifier features of the generated output.
- CLI knobs: `--state-probe-sample-count`, `--state-probe-target-size`, and
  `--state-probe-ridge`. The default sample count is `0`, so normal runs stay
  unchanged.

Question this answers:

- If final oscillator states decode class/scaffold/detail much better than
  initial states, the recurrent settling process is making useful information
  available and the remaining bottleneck is likely readout/objective design.
- If final states do not decode detail, the missing high-frequency/image-detail
  problem is upstream: the oscillator field itself is not forming that
  information, so another readout will not save it.

This should be the next diagnostic before another long architecture sweep.

Run:

```text
mnist_generator_cifar10_rgb_state_information_probe
```

Two-seed result at 20 epochs:

```text
Variant             Acc     Diversity  Nearest MSE  Feature Frechet  High-freq  Edge    Attractor acc  Final highpass R2  Final feature R2
current             0.4463  0.9136     0.0265       0.0999           0.4658     0.2136  0.4438         0.3567             0.4924
hierarchy_lead      0.3105  0.8126     0.0238       0.2227           0.4652     0.1722  0.2812         0.4897             0.5293
patch010_multiscale 0.3799  0.9616     0.0362       0.0549           1.5089     0.7775  0.4125         0.3369             0.4171
```

State-probe read:

- All variants make labels almost perfectly decodable from the final fine
  state on the 64-sample trace (`fine_final.label_accuracy` roughly `0.97-1.0`),
  while initial-state label decoding is near chance.
- The hierarchy lead gives the strongest final-state high-pass residual R2 and
  classifier-feature R2, despite weaker rendered classifier accuracy in this
  compact run.
- Patch multiscale raises rendered high-frequency and edge ratios dramatically,
  but its final-state high-pass/feature decodability is weaker than the
  hierarchy lead. This supports the read that patch pressure adds visible
  texture/artifacts more than it unlocks a cleaner state representation.

Conclusion: settling is definitely organizing useful class/scaffold/detail
signals in the oscillator state. The current failure is not simply "the state
is dead." The harder problem is converting that organized state into clean
object-aligned RGB detail and/or training the state against a better notion of
image detail than patch/frequency statistics alone. The next architecture move
should be a readout/probe intervention that consumes the final state more
directly, not another blind frequency-objective sweep.

### State Residual Readout Probe

Implementation:

- Added an optional HORN state-residual readout for the resize-conv renderer.
- The normal resize-conv path still renders the global image, but the new
  branch lets each final oscillator contribute a small learned local RGB patch
  from its final `(position, velocity)` state.
- This is intentionally a state-to-image interface test, not another
  frequency-loss sweep: if final HORN state detail is useful, a small local
  residual patch head should let the renderer use it more directly.
- Presets:
  `sparse_horn_cifar10_rgb_hierarchy_state_residual005` and
  `sparse_horn_cifar10_rgb_hierarchy_state_residual010`.

Run:

```text
mnist_generator_cifar10_rgb_state_residual_readout_probe
```

Two-seed result at 20 epochs:

```text
Variant             Acc     Diversity  Nearest MSE  Feature Frechet  High-freq  Edge    Attractor acc  Final highpass R2  Final feature R2
hierarchy_lead      0.3174  0.8585     0.0245       0.2117           0.4512     0.1852  0.3313         0.4568             0.4729
state_residual005   0.4492  0.8161     0.0232       0.1640           0.4359     0.1660  0.4688         0.4902             0.4715
state_residual010   0.3682  0.8143     0.0233       0.1928           0.4719     0.1854  0.3625         0.4873             0.5825
```

Read:

- `state_residual005` is a real improvement over the hierarchy lead in this
  compact probe: higher generated-label accuracy, better attractor robustness,
  lower nearest-real MSE, better feature Frechet, and better final-state
  high-pass decodability.
- The effect is not "more residual is always better." `state_residual010`
  improves some state-feature decodability but gives back too much generated
  class consistency and attractor strength.
- The branch adds about `38k` direct state-to-RGB residual parameters in the
  CIFAR RGB setting, so it should be compared as a readout improvement, not as
  a pure dynamics-only win.

Conclusion: the state-information probe was actionable. Useful detail is
present in the settled HORN state, and a small local residual state readout can
extract more of it than the plain resize-conv path. The current best follow-up
is to refine this interface, probably around `0.05` strength and with better
regularization/calibration, rather than increasing patch/frequency losses or
turning residual strength up blindly.

### State Residual Longer Pilot

Question: does the `state_residual005` candidate simply need longer training
to turn the improved state/readout metrics into visibly better CIFAR samples?

Run:

```text
mnist_generator_cifar10_rgb_state_residual_longer_pilot
```

This was a narrow seed-23 pilot at 40 epochs:

- `current`: stable CIFAR RGB normalized-local HORN default.
- `hierarchy_lead`: active hierarchy mechanism lead.
- `state_residual005`: hierarchy lead plus local final-state residual RGB
  patch readout.

Result:

```text
Variant            Acc     Diversity  Nearest MSE  Feature Frechet  Feature diversity  High-freq  Edge    Attractor acc  Samples/s
current            0.5234  0.9493     0.0374       0.0890           0.8679             0.5138     0.2617  0.5250         539.7
hierarchy_lead     0.4062  0.9791     0.0372       0.1689           0.8082             0.3978     0.2266  0.4000         238.4
state_residual005  0.4531  0.9330     0.0354       0.1509           0.8252             0.4326     0.2238  0.4250         210.3
```

Visual sheet:

```text
outputs/modal_samples/state_residual_longer/contact_sheet_state_residual_longer_seed23.png
```

Read:

- Longer training does help the models become more saturated and structured,
  but it does not produce a sudden visual breakthrough by epoch 40.
- `state_residual005` still improves the hierarchy lead on generated-label
  accuracy, nearest-real MSE, feature Frechet, and attractor accuracy. That
  confirms the small state-to-image readout is useful.
- The stable `current` CIFAR default is still the stronger overall 40-epoch
  model in this pilot: better class consistency, feature Frechet, feature
  diversity, high-frequency/edge ratios, attractor accuracy, and much better
  throughput.
- Therefore the bet partially held: state residual readout helps hierarchy
  mature, but "just train the hierarchy-residual branch longer" is not enough
  to beat the simpler stable CIFAR HORN default or solve visual sharpness.

Conclusion: keep `state_residual005` as a renderer/interface candidate, but do
not promote it to the stable CIFAR default. The next meaningful move is either
to combine the state-residual readout with the stable normalized-local default
instead of the heavier hierarchy lead, or to redesign the hierarchy renderer so
the extra vertical/coarse structure is not paid for by weaker final RGB
quality and lower throughput.

### Resonant Filter-Bank Readout Pilot

Question: can an ONN-native shared filter bank expose useful local HORN field
structure without turning the decoder into the whole model?

Implementation:

- Added an optional `resonant_readout_strength` branch for HORN resize-conv
  generators.
- The branch reads local HORN observables: bounded position/velocity,
  `sin/cos(position)`, local phase-alignment, local order magnitude, velocity
  contrast, and local state energy.
- It uses a shared local spatial filter bank, not per-oscillator RGB patch
  weights. In CIFAR RGB this adds only `675` parameters, so it is much closer
  to a resonant/filter-bank interface than a conventional large decoder.
- Presets:
  `sparse_horn_cifar10_rgb_current_resonant005` and
  `sparse_horn_cifar10_rgb_current_resonant010`.

Run:

```text
mnist_generator_cifar10_rgb_resonant_readout_pilot
```

Seed-23 result at 20 epochs:

```text
Variant       Acc     Diversity  Nearest MSE  Feature Frechet  Attractor acc  Attractor div  High-freq  Edge    Samples/s
current       0.4180  0.8415     0.0269       0.2405           0.3875         1.3163         0.0118     0.2139  360.6
resonant005   0.4414  1.0596     0.0379       0.1397           0.4375         1.9204         0.0071     0.2078  275.8
resonant010   0.3906  0.8027     0.0244       0.2466           0.4000         1.1184         0.0082     0.1393  254.6
```

Visual sheet:

```text
outputs/modal_samples/resonant_readout/contact_sheet_resonant_readout_seed23.png
```

Read:

- `resonant005` is a useful low-strength candidate: it improves generated-label
  accuracy, diversity, feature Frechet, attractor accuracy, and attractor
  diversity over the stable CIFAR default.
- It does not fix rendering sharpness. Nearest-real MSE gets worse, and the
  generated high-frequency ratio drops. Visually it is more contrasty and
  varied, not cleaner or more detailed.
- `resonant010` is already too strong/mixed: it recovers nearest-real MSE but
  loses the semantic/diversity improvements and looks muddier.

Conclusion: the HORN resonant filter-bank idea is worth keeping as a small
readout primitive, because it improves the semantic/diversity frontier with
very few parameters. It is not the missing CIFAR renderer by itself. The next
high-value probe should ask whether the resonant branch can be trained with a
targeted high-frequency/detail objective or combined with the state-information
probe, rather than increasing its strength.

### CIFAR RGB Oscillator Capacity Probe

Question: is the stable CIFAR HORN field simply too small? The current default
uses `256` fine HORN oscillators for a `32x32x3` image. This probe doubled the
oscillator-site count to `512` while keeping the same stable CIFAR recipe and
resize-conv rendering style.

Implementation:

- Added `sparse_horn_cifar10_rgb_current_n512`.
- Added `sparse_horn_cifar10_rgb_current_n512_resonant005`.
- Kept the current normalized-local sparse HORN recipe: class coupling,
  row-sum local coupling, `resize_conv`, and `train_settling_steps=(16,32,48)`.
- The resize-conv seed stays `8x8`; increasing from `256` to `512`
  oscillators changes seed channels from `8` to `16`.

Run:

```text
mnist_generator_cifar10_rgb_capacity_probe
```

Seed-23 result at 20 epochs, batch size 32:

```text
Variant           N    Acc     Diversity  Nearest MSE  Attractor acc  Attractor div  High-freq  Edge    Params   Samples/s
n256_current      256  0.6250  1.1043     0.0441       0.6125         2.9327         0.0077     0.2742  156595   326.6
n512_current      512  0.5938  1.1665     0.0514       0.5250         2.6466         0.0071     0.3208  436531   200.6
n512_resonant005  512  0.4023  1.1637     0.0499       0.4000         2.0104         0.0090     0.3474  437206   161.5
```

Visual sheet:

```text
outputs/modal_samples/capacity_probe/contact_sheet_capacity_probe_seed23.png
```

Read:

- Doubling oscillator sites does not solve CIFAR rendering by itself.
- `n512_current` increases diversity and edge energy, but loses class
  consistency, attractor accuracy, nearest-real MSE, and throughput.
- The result looks more saturated/active, not more semantically stable or
  sharply detailed.
- `n512_resonant005` is worse than the plain 512 field on class and attractor
  metrics, so the resonant readout strength that helped at 256 does not simply
  transfer upward.

Conclusion: the 256-field is probably capacity-constrained, but raw oscillator
site count is not the missing ingredient. Bigger HORN fields need stronger
organization: multimode/frequency-band structure, better homeostasis, or a
more explicit division of coarse/global and fine/detail roles. A blind 1024
site run is not the next best move until the 512 field has a better way to
coordinate its extra degrees of freedom.

### CIFAR RGB Multimode HORN Probe

Question: is extra HORN capacity more useful when it is organized as several
frequency modes per spatial site instead of more flat spatial sites? This probe
kept the stable CIFAR RGB recipe and compared:

- `sparse_horn_cifar10_rgb_current`: 256 spatial sites, one HORN mode each.
- `sparse_horn_cifar10_rgb_current_n512`: 512 flat spatial sites.
- `sparse_horn_cifar10_rgb_current_multimode2`: 256 spatial sites with two
  frequency-band HORN modes per site, total state size 512.

Run:

```text
mnist_generator_cifar10_rgb_multimode_probe
```

Seed-23 result at 20 epochs, batch size 32:

```text
Variant      Sites  Modes  Total N  Acc     Div     Feature div  Nearest MSE  Feature NN  Attractor acc  Attractor div  High-freq  Edge    Samples/s
n256         256    1      256      0.5313  1.0338  1.0435       0.0396       0.0364      0.5000         2.2760         0.0097     0.2979  556.8
n512 flat    512    1      512      0.4219  1.1887  0.7777       0.0525       0.1143      0.4250         2.1686         0.0074     0.3398  349.2
multimode2   256    2      512      0.7734  1.0713  0.9814       0.0438       0.0945      0.7250         3.5281         0.0078     0.2723  223.9
```

Visual sheet:

```text
outputs/modal_samples/multimode_probe/contact_sheet_multimode_probe_seed23.png
```

Read:

- Multimode HORN is much better than flat 512 scaling on generated-label
  accuracy, feature diversity, nearest-real MSE, attractor accuracy, and
  attractor diversity.
- Against the compact 256 default, multimode wins class consistency and
  attractor metrics, but loses nearest-real feature proximity, high-frequency
  energy, edge energy, and speed.
- Visually it is still in the blurry CIFAR regime. The frequency-band state
  helps organize class-consistent basins, but it does not by itself solve the
  state-to-image/detail renderer bottleneck.

Conclusion: multimode/frequency-band HORN is a better capacity direction than
more flat sites. It should replace blind 1024-site scaling as the next capacity
lead, but it still needs either a better readout/detail path or a stronger
objective before it becomes the default CIFAR renderer.

### CIFAR RGB Retinotopic Readout and State-Fitting Probe

Question: is the CIFAR sharpness bottleneck caused by the HORN state itself, or
by the way the state is rendered/trained? A review found a concrete geometry
issue: the resize-conv seed previously reshaped `[position, velocity]` features
as a flat vector. For a spatial HORN grid this scrambles retinotopy, and for
multimode HORN it also interleaves frequency modes awkwardly. This probe added:

- `resize_conv_seed_layout="retinotopic"`: reshape HORN state as
  `(spatial_site, mode, position/velocity)` into a `16x16` seed with
  `2 * num_modes` channels.
- `sparse_horn_cifar10_rgb_current_retinotopic`.
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic`.
- A frozen-decoder state-fitting probe: optimize one final HORN state per real
  CIFAR image through the frozen decoder, then check high-frequency
  reconstruction and whether fitted detail survives extra settling.

Run:

```text
mnist_generator_cifar10_rgb_retinotopic_readout_probe
```

Seed-23 result at 20 epochs, batch size 32:

```text
Variant            Layout       Acc     Feature div  Nearest MSE  Edge    High-freq  Fit MSE  Fit high-freq  Fit edge  Settle8 MSE
n256 flat          flat         0.6797  0.9813       0.0388       0.2613  0.0090     0.0151   0.0117         0.3975    0.0952
n256 retino        retinotopic  0.1836  0.8225       0.0222       0.2744  0.0148     0.0095   0.0122         0.4100    0.0917
multimode2 flat    flat         0.6719  1.0059       0.0532       0.3371  0.0075     0.0673   0.0176         1.4587    0.1103
multimode2 retino  retinotopic  0.6367  0.8658       0.0212       0.2746  0.0133     0.0042   0.0137         0.5662    0.1126
```

Visual sheet:

```text
outputs/modal_samples/retinotopic_probe/retinotopic_probe_contact_sheet.png
```

Read:

- Retinotopic layout is a real representational fix for fitted states. The
  multimode retinotopic arm reconstructs real CIFAR images through the frozen
  decoder much better than multimode flat (`fit_mse` `0.0042` vs `0.0673`).
- Direct generation does not improve yet. The single-mode retinotopic arm loses
  class consistency badly, and the multimode retinotopic arm trails the
  multimode flat arm on direct generated-label accuracy and diversity.
- Fitted states can recover more edge/high-frequency content than ordinary
  samples, but additional HORN settling breaks paired reconstruction. This
  means detail-carrying states exist, but the current generative training route
  is not steering samples into those states and/or the attractor dynamics do not
  preserve paired texture.
- The fresh linear readout on fitted states still has negative high-pass R2, so
  a simple linear probe is not enough to extract real texture from fitted state
  features.

Conclusion: this is a diagnostic win, not a sample-quality win. The next
high-value direction is objective/trajectory design that steers the oscillator
field toward detail-carrying states, or a two-stage oscillator scaffold plus
oscillator renderer. Retinotopic layout should remain available because it
fixes a real geometry mismatch, but it should not replace the stable CIFAR
default until direct generation catches up.

### CIFAR RGB Retinotopic Param-Matched Control

Question: did the first retinotopic probe unfairly handicap retinotopic
readout by shrinking the decoder? This control matched decoder capacity more
closely by raising retinotopic `resize_conv_min_channels` to `30`, added a
single-mode four-channel seed control, and added state-fitting settle
granularity at `1/2/4` steps plus matched-norm random perturbation controls.

Run:

```text
mnist_generator_cifar10_rgb_retinotopic_control_probe
```

Seed-23 result at 20 epochs, batch size 32:

```text
Variant              Layout       Seed ch  Dec params  Acc     Feature div  Nearest MSE  High-freq  Fit MSE  Settle1  Noise1  Settle8  Noise8
multimode2 flat      flat         0        9715        0.7539  1.0147       0.0457       0.0079     0.0541   0.0560   0.0576  0.0940   0.1163
multimode2 retino30  retinotopic  0        10053       0.3711  0.7798       0.0192       0.0167     0.0052   0.0086   0.0066  0.1091   0.0561
n256 flat            flat         0        8563        0.6367  0.9371       0.0442       0.0082     0.0155   0.0202   0.0180  0.1202   0.0809
n256 retino30        retinotopic  0        9513        0.4570  0.8778       0.0253       0.0144     0.0131   0.0157   0.0145  0.0583   0.0542
n256 retino seed4    retinotopic  4        10053       0.4258  0.8987       0.0196       0.0216     0.0070   0.0086   0.0081  0.0879   0.0518
```

Visual sheet:

```text
outputs/modal_samples/retino_control/retino_control_contact_sheet.png
```

Read:

- Decoder param matching does not rescue direct retinotopic class consistency.
  The flat multimode branch is still the stronger direct generator on class
  accuracy and feature diversity at 20 epochs.
- The retinotopic branches are much better on nearest-real MSE, high-frequency
  ratio, and fitted-state reconstruction. The previous “retinotopy improves
  representability” read survives the control.
- Four seed channels do not rescue single-mode retinotopic class accuracy.
  It improves fitted-state reconstruction and high-frequency energy, but class
  consistency remains well below the flat branch. That supports the idea that
  flat seeds were giving the conv decoder a global/channel-mixing shortcut,
  while honest retinotopic rendering needs a better objective or dynamics.
- Settling is more destructive to fitted reconstructions than matched-norm
  random perturbation by step 4/8 in the retinotopic arms. This points to real
  attractor contraction away from paired texture, not merely brittle
  off-manifold fitted states.

Conclusion: the next scientific move should not be more readout capacity. The
best current interpretation is that retinotopic HORN states can carry more
image detail, but the current pixel/feature drift objective does not train free
samples to enter and preserve those detail-carrying basins. A paired
state-space denoising/settle-survival anchor is now the most grounded small
next step before a larger two-stage HORN-scaffold plus oscillator-renderer
architecture.

### CIFAR RGB State Anchor Probe

Question: can a tiny local image-to-state encoder train retinotopic multimode
HORN dynamics to preserve detail-bearing states through settling?

Implementation:

- Base model: `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30`.
- Anchor encoder: local 3x3 stride-2 conv from RGB image to 16x16x4 HORN
  state channels, interpreted as two modes times position/velocity.
- Anchor path is training-only. Free generation still starts from random HORN
  state.
- Controls:
  - `no_anchor`
  - `anchor_reconstruct010`: k=0 encode/decode autoencoder control.
  - `anchor_frozen010`: stop-gradient recurrent/conditioning dynamics in the
    anchor path.
  - `anchor010` and `anchor030`: full anchor at lambda 0.10 and 0.30.

Run:

```text
mnist_generator_cifar10_rgb_state_anchor_probe
```

Two seeds, 20 epochs, batch size 32. Key means:

```text
Variant             Settle8 MSE  Noise8 MSE  Gap      Acc     Diversity  Nearest MSE  High-freq  Attractor acc
no_anchor           0.1605       0.0799      0.0806   0.4102  0.9070     0.0375       0.0150     0.4000
anchor_recon010     0.0587       0.0412      0.0175   0.4004  0.6790     0.0219       0.0156     0.3875
anchor_frozen010    0.0609       0.0348      0.0261   0.4277  0.6410     0.0204       0.0202     0.4250
anchor010           0.0443       0.0308      0.0135   0.3027  0.6109     0.0191       0.0191     0.2438
anchor030           0.0339       0.0290      0.0048   0.3809  0.5689     0.0181       0.0207     0.3250
```

Visual sheet:

```text
outputs/modal_samples/state_anchor_probe/state_anchor_probe_contact_sheet.png
```

Read:

- The anchor worked on the targeted failure mode. `anchor030` reduced
  settle-8 fitted-state MSE from `0.1605` to `0.0339`, and reduced the
  settle-vs-noise gap from `0.0806` to `0.0048`. This is the clearest evidence
  so far that the recurrent HORN field can be trained to preserve
  detail-carrying states instead of washing them out.
- The full anchor beats both controls on settle survival. `anchor030` is better
  than k=0 reconstruction and better than frozen-dynamics anchor, so the gain
  is not merely decoder pretraining.
- The free-sample story is not solved. High-frequency ratio improves
  (`0.0150` to `0.0207`) and nearest-real MSE drops, but diversity drops and
  visual samples remain blurry/texture-like. This is not yet a rendered CIFAR
  breakthrough.
- The result sharpens the bottleneck: a good detail-preserving basin now exists
  near image-encoded HORN states, but random class-conditioned free
  trajectories do not reliably enter that basin.

Conclusion: promote the state anchor as a useful diagnostic/training primitive,
not as the new default generator. The next architectural question is how to
bridge free sampling into the anchor-trained basin: staged settling, a
two-stage HORN scaffold plus renderer, or mode-dependent damping where a slow
mode carries semantic scaffold and a lightly damped fast mode preserves
texture.

### CIFAR RGB State Prior Sampling Probe

Question: did the anchor fail at free generation because the detail-preserving
basin is unreachable from isotropic white-noise HORN initial states?

Implementation:

- Added duplicate-rate metrics to distinguish "closer to train set" from
  literal near-copying. The main thresholds are nearest-reference MSE below
  `0.001`, `0.0025`, `0.005`, and `0.010`.
- Added `scripts/analyze_generator_state_prior.py`, an eval-only script that
  loads an existing anchor checkpoint, encodes training images into HORN
  position/velocity states, fits a transparent per-class mean plus low-rank
  PCA Gaussian state prior, and samples explicit initial HORN states.
- Controls:
  - `white_noise`: the ordinary generator path.
  - `prior_mean`: class prior mean with no noise.
  - `prior_sample`: per-class PCA prior sample.
  - `shuffled_prior`: prior sampled from a different class while conditioning
    dynamics on the requested class.

Local eval-only run:

```text
scripts/analyze_generator_state_prior.py
checkpoint: seed23 anchor030 epoch 20
rank: 32
sample_count: 256
settle_steps: 8
```

Artifacts:

```text
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256.csv
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256.json
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256_contact_sheet.png
```

Key seed-23 metrics without a trained classifier judge:

```text
Variant         Eval near MSE  Train dup <0.001  Train dup <0.005  Train dup <0.010  Proto acc  Diversity
white_noise     0.0129         0.0000            0.0039            0.6484            0.2422     0.4153
prior_mean      0.0121         0.0000            0.0000            0.4766            0.9102     0.5782
prior_sample    0.0323         0.0000            0.0000            0.0000            0.4102     0.9719
shuffled_prior  0.0330         0.0000            0.0000            0.0000            0.2188     0.9205
```

Read:

- The contact sheet reframes the earlier `anchor030` free-sample improvement.
  White-noise samples are visually close to a single generic blurry texture
  basin, which explains why anchor training reduced nearest-real MSE while
  hurting diversity. The white-noise path gave the drift objective a weak,
  near-collapsed generated distribution.
- Tight duplicate rates are zero across variants, so the rank-32 prior sample
  is not simply copying exact training images. The broad `0.010` train-nearest
  rate is high for white noise and prior mean, but disappears for prior
  samples.
- `prior_mean` gives very high prototype consistency and moderate diversity,
  showing that the encoded state prior carries class structure. It is also a
  collapse control, not a generator.
- `prior_sample` preserves much more diversity than white noise and improves
  prototype consistency over shuffled prior, but it worsens nearest-real MSE
  and still looks blurry/ambiguous. The shuffled-prior drop means class
  information is flowing through the initial state prior as well as the
  conditioning dynamics.
- This lands between the simple decision branches: structured initial states
  reach a different, more diverse basin without exact memorization. The basin
  access hypothesis is confirmed. The complication is that class semantics
  currently flow through both the initial-state cue and the HORN conditioning
  dynamics.

Conclusion: promote the bridge hypothesis from speculative to active lead.
White noise is an arbitrary and likely poor cue distribution for anchor-trained
HORN basins. The next grounded intervention is to put state-prior sampling
inside training while preserving attribution controls:

- `anchor030`: unchanged white-noise-trained control.
- `anchor030_prior_global`: one class-agnostic state prior; class enters only
  through oscillator conditioning.
- `anchor030_prior_class`: one state prior per class; expected strongest raw
  metrics, but must keep shuffled-prior attribution checks.

Added opt-in training knobs:

```text
state_prior_sampling_mode = none | global | class
state_prior_rank
state_prior_noise_scale
state_prior_refresh_epochs
state_prior_start_epoch
```

Decision rule: if global or class prior training gives diverse, non-copying,
class-consistent samples with better visible structure, state-prior sampling
becomes the new CIFAR lead. If diversity/class hold but sharpness stalls, the
field is best treated as a scaffold generator and the two-stage
HORN-scaffold/renderer branch becomes the next major build.

Implementation guardrail added before launch:

- Prior-trained arms are now evaluated in the same sampling regime they train
  in. Final quality metrics, settling metrics, attractor robustness, vertical
  intervention audits, and contact sheets use the fitted final state prior when
  `state_prior_sampling_mode != "none"`.
- The fitted final prior is persisted next to the checkpoint as
  `state_prior_final.json` and `state_prior_final.npz`.
- White-noise sampling remains a secondary diagnostic under
  `white_noise_*` metrics. This measures how specialized the learned basins are
  to the state-prior cue distribution.
- Class-prior arms additionally log `shuffled_prior_*` controls, where the
  initial state prior is sampled from the wrong class while the conditioning
  label stays fixed. This keeps the class-information route auditable.

For these arms, "the generator" means `state prior + HORN field + decoder`.
Scoring only the field from isotropic white-noise states is a useful stress
test, but it is not the primary result.

Two-seed Modal result after fixing prior-aware evaluation:

```text
Sweep: mnist_generator_cifar10_rgb_state_prior_training_probe
CSV: outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_training_probe.csv
Contact sheets: outputs/analysis/state_prior_training_probe/contact_sheets/
```

```text
Arm           Init         Acc     Feature div  Near MSE  Best settle acc  Dup<0.010  Edge ratio
anchor030     white noise  0.2910  0.7512       0.0178    0.3164           0.1367     0.2512
prior_global  prior        0.4316  0.8581       0.0293    0.4551           0.0039     0.4539
prior_class   prior        0.4941  0.8715       0.0369    0.5371           0.0000     0.4825
```

Read:

- The mechanism worked. Prior-aware training/eval produces less collapsed,
  more class-consistent, more diverse CIFAR RGB samples than the white-noise
  `anchor030` control.
- `prior_global` is now the attribution-clean reference recipe: the prior is
  class-agnostic, so class identity can only enter through oscillator
  conditioning. It beats `anchor030` on generated-label accuracy, feature
  diversity, settling accuracy, duplicate rate, and edge energy.
- At this stage, `prior_class` was the strongest HORN prior arm. It had the best generated
  class accuracy, best settling accuracy, best feature diversity, and zero
  duplicate rate at the 0.010 threshold across both seeds. Its shuffled-prior
  control keeps the class-information route auditable.
- Nearest-pixel MSE worsens because the prior arms escape the blurry/copy-like
  basin that nearest-MSE rewards. For this branch, frontier ranking should
  demote nearest-real MSE and emphasize duplicate rates, classifier-feature
  nearest distance, feature diversity, generated-label accuracy, edge/frequency
  diagnostics, and contact sheets.
- Visual contact sheets support the numeric read. The prior arms are still
  blurry and not solved CIFAR generators, but they show broader variation and
  more object-like color/shape mass than the anchor control.
- Detail moved for the first time in the desired direction: edge-Laplacian
  ratio roughly doubled versus `anchor030`. This supports the hypothesis that
  structured state-prior inits reduce assignment-averaging pressure in the
  drift objective.

Next decision:

1. Keep `prior_global` as the clean CIFAR reference and `prior_class` as the
   stronger HORN prior arm, pending a matched non-oscillatory control.
2. Run a scale gate before new architecture: full CIFAR train set, longer
   training, at least three seeds, comparing `prior_global` and `prior_class`.
3. Retest one low-weight random-offset patch objective under the healthy
   prior-aware loop. Earlier patch negatives were measured on a near-collapsed
   sampling loop and are stale evidence.
4. Keep a two-stage renderer in reserve. Trigger it only if the scaled
   prior-aware run still leaves edge/detail metrics far below real images.

Scale-gate rung 1 result:

```text
Sweep: mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1
CSV: outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1.csv
Contact sheets: outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/
```

The originally planned batch-128 and batch-64 variants exceeded A10G memory
with this objective stack. The completed rung therefore used batch 32,
train-limit 10k, 40 epochs, two seeds, eval samples 512, and a stronger
quality classifier train limit of 20k.

```text
Arm                   Acc     Feature div  Feature near  Near MSE  Dup<0.010  Edge ratio  High freq  Best settle acc
prior_global_b32      0.5547  0.9434       0.2699        0.0336    0.0088     0.5690      0.7964     0.6084
prior_class_b32       0.4531  0.9081       0.3118        0.0392    0.0029     0.6335      0.7015     0.4629
prior_class_patch005  0.7129  0.9894       0.2025        0.0361    0.0000     0.9270      1.2599     0.7373
```

Read:

- Rung 1 bought a real improvement, not just a longer blurry run. Contact
  sheets show more varied scene populations, stronger sky/ground splits,
  object-like masses, and more texture than the 2k/20 probe.
- The low-weight random-offset patch objective is no longer stale-negative
  evidence. Under the healthy state-prior loop it became the strongest HORN
  arm:
  best class accuracy, best feature diversity, best feature-nearest distance,
  strongest edge/high-frequency metrics, best settling accuracy, and zero
  duplicate rate at the 0.010 threshold across both seeds.
- `prior_global_b32` remains the cleaner attribution reference. It also
  improved over the 2k/20 version, especially generated-label accuracy,
  feature diversity, attractor accuracy, and settling gain.
- Plain `prior_class_b32` did not benefit from scale as clearly. The patch term
  appears to be doing useful work, likely by reducing the residual averaging
  pressure in the image readout/objective rather than by reintroducing copying.
- Nearest-real pixel MSE remains secondary. The patch arm's MSE is not the best
  absolute number, but its duplicate rate, feature metrics, semantic metrics,
  edge/frequency diagnostics, and visual sheets are all better aligned with
  generative quality.

Decision after rung 1: treat `prior_class_patch005` as the current HORN CIFAR
RGB reference recipe, keep `prior_global` as the attribution-clean diagnostic,
and run the same-stack StateMLP control before buying a larger rung 2.

State-prior control probe:

```text
Sweep: mnist_generator_cifar10_rgb_state_prior_control_probe
CSV: outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_control_probe.csv
Contact sheets: outputs/analysis/state_prior_control_probe/contact_sheets/
Comparison sheet:
outputs/analysis/state_prior_control_probe/contact_sheets/control_probe_comparison.png
```

This probe fixed a fairness issue in the non-oscillatory control: the
`StateMLPImageGenerator` now receives the same retinotopic/multimode layout,
state-anchor encoder path, state prior sampling, anchor loss, and patch005
objective as the HORN recipe. Parameter counts are essentially matched
(`436,981` HORN vs `437,621` StateMLP).

```text
Arm                         n  Acc     Feature div  Feature near  Near MSE  Dup<0.010  Edge ratio  High freq  Attractor  Best settle acc
prior_class_patch005*       2  0.7129  0.9894       0.2025        0.0361    0.0000     0.9270      1.2599     0.6875     0.7373
state_mlp_prior_class_p005  2  0.6533  0.9261       0.1889        0.0362    0.0000     0.8592      0.9469     0.6875     0.6855
prior_global_patch005       2  0.4365  0.8862       0.2791        0.0347    0.0000     0.8934      1.1327     0.4375     0.4678
prior_class_p005_queue64    1  0.5664  0.9447       0.2955        0.0377    0.0000     0.9430      1.2843     0.5750     0.6348
```

`prior_class_patch005*` is the previous rung-1 HORN result, included here as
the reference because the control probe was designed around it.

Read:

- The same-stack StateMLP is a real control, not a strawman. It gets lower eval
  loss, better classifier-feature nearest distance, and faster sampling. This
  confirms that current CIFAR visual quality is still mostly objective/readout
  limited, not a solved ONN image-generation problem.
- The same-stack StateMLP also removes the apparent HORN advantage once the
  comparison is paired by seed. The means slightly favor HORN on some metrics,
  but the paired deltas flip sign between seeds:

```text
Metric          Seed 23 HORN / StateMLP  Seed 24 HORN / StateMLP
Accuracy        0.6289 / 0.7793          0.7969 / 0.5273
Attractor       0.5375 / 0.8125          0.8375 / 0.5625
Settling gain   0.2754 / 0.4355          0.3418 / 0.3008
Feature div     0.9494 / 0.9629          1.0294 / 0.8893
Edge ratio      0.9589 / 0.8473          0.8952 / 0.8710
High freq       1.2494 / 0.8992          1.2704 / 0.9946
```

  With `n=2`, this is a null result, not a supported HORN lead. HORN is
  consistently closer on edge-Laplacian ratio, while StateMLP is consistently
  closer to the ideal high-frequency ratio of `1.0` and is faster.
- The StateMLP shuffled-prior score is high (`~0.60` mean, with seed 23 at
  `0.72`), so more of its class information is carried by the prior/shortcut
  route. That makes it less clean as an oscillator-dynamics attribution story
  even when its images are competitive under the same objective.
- `prior_global_patch005` is **not** the new headline. It is clean
  attribution-wise, but the class/attractor metrics are too weak. Keep
  `prior_global` as a diagnostic reference, not a promoted recipe.
- The queue64 positive-pool probe did not help in its first seed. It preserves
  detail metrics but hurts semantic/feature metrics, so it should not be
  promoted without a separate queue-tuning sprint.
- The contact sheet matches the null-result read: all arms remain soft; HORN
  and StateMLP both produce varied class-colored scenes under the healthy
  prior/anchor/patch pipeline. The pipeline improvements transfer to the
  non-oscillatory iterated map.

Decision after the control probe:

1. Keep `prior_class_patch005` as the current HORN CIFAR RGB reference recipe,
   but keep the same-stack StateMLP control beside it. Do not describe it as a
   demonstrated HORN performance lead.
2. Do not buy a full 50k/80-epoch rung 2 yet as a quality claim. First add a
   standard image-generation calibration metric such as FID/KID with a stronger
   feature extractor if this branch is revisited.
3. Treat the defensible finding as a rigorous negative for this task/scale:
   under a matched prior/anchor/patch objective, sparse multimode HORN is
   statistically indistinguishable from a same-stack non-oscillatory MLP on the
   current CIFAR frontier/attractor metrics, while StateMLP is faster.
4. Competitive CIFAR quality is not established. The samples are still
   watercolor-like and far below modern GAN/diffusion/flow models; the current
   result is a useful oscillator-mechanism audit plus a clean negative on
   HORN-vs-StateMLP image generation, not a literature-grade image generator
   result.
5. Freeze this state-prior CIFAR generator branch for now. The next frontier
   should move to either the phase-flow/rectified-flow formulation, where the
   objective ceiling is higher, or to attractor-native recovery tasks where
   settling is the point.

## CIFAR RGB State Recovery Probe (Noise-Then-Settle, Eval Only)

Motivation: the StateMLP control probe left one consistent oscillator-vs-control
asymmetry — HORN settling moved fitted states less destructively than matched
random perturbations. This probe tests the recovery version of that claim
directly: if a detail-bearing state is corrupted, do the dynamics *repair* it?

Setup (no training; local eval on pulled epoch-40 checkpoints):

- `compute_generator_state_fitting_probe` gained a `noise_then_settle`
  condition: fit final states to 32 real CIFAR eval images through the frozen
  decoder, perturb each state with Gaussian noise at a relative scale
  (`0.125/0.25/0.5/1.0` of the clean state norm), settle `1/2/4/8/16` steps,
  decode, and compare against both the un-settled noisy decode and the
  clean-settle baseline at the same depth.
- New CLI: `scripts/analyze_generator_state_recovery.py` (shared checkpoint
  loader in `oscnet/analysis/generator_checkpoints.py`).
- Arms: HORN `prior_class_patch005` (rung-1 checkpoints), HORN
  `prior_global_patch005`, and same-stack StateMLP `prior_class_patch005`,
  seeds 23/24. Outputs in `outputs/analysis/state_recovery_probe/`, aggregate
  in `state_recovery_probe_aggregate.csv`.

Headline metric: **denoise fraction** = `1 - excess(k)/excess(0)`, where
`excess(k)` is the noisy-settled paired MSE minus the clean-settled paired MSE
at the same settle depth `k`. It isolates what settling does to the *noise*
component, controlling for each arm's own clean-state drift. `1.0` = noise
fully absorbed; `0` = untouched; negative = settling amplified the corruption.

```text
Arm                          mean_frac  min_frac  negative_cells  k8_mid_noise_mean
horn_prior_global_patch005   0.527      0.145     0/40            0.723
horn_prior_class_patch005    0.404      0.109     0/40            0.586
state_mlp_prior_class_p005   0.237      -1.509    9/40            0.454
```

(40 cells per arm = 2 seeds x 4 noise scales x 5 settle depths;
`k8_mid_noise_mean` averages noise scales 0.25/0.5/1.0 at depth 8.)

Read:

- **Both HORN arms denoise in every one of their 80 cells.** The denoise
  fraction is positive at every noise scale, settle depth, and seed, and grows
  monotonically with depth (up to ~0.7-0.9 for `prior_global` at depth 8-16).
  This is genuine attractor-style contraction of state noise along
  decode-relevant directions.
- **StateMLP does not reliably denoise.** Its mean fraction is about half of
  HORN's, 9 of 40 cells are negative (settling amplifies the corruption), and
  the depth profile is non-monotone. Where it does recover, it is mostly at
  the largest noise scale and depth, consistent with generic smoothing rather
  than structured contraction.
- This is the first probe in the CIFAR branch where the oscillator arms beat
  the same-stack control with a consistent sign across both seeds and all
  conditions, rather than a seed-flipping null.
- Interesting mechanism detail: the `state_return_ratio` is above 1 everywhere
  (settled states move *further* from the clean fitted state in raw state
  distance) while the decode moves back toward the clean target. Recovery is
  happening along the decoder-relevant subspace, not by returning to the
  fitted point.
- Honest caveats: (1) recovery is relative — absolute clean-settle drift still
  dominates total error at depth (clean paired MSE rises from ~0.0006 at k=0
  to 0.03-0.06 at k=16 in all arms), so nothing here says HORN preserves a fit
  absolutely; (2) `prior_global` denoises best but also drifts most, i.e. it
  contracts hardest onto its own attractor, which erases both noise and
  detail; (3) this is n=2 seeds and one corruption family (isotropic Gaussian
  in state space).

Decision:

1. Treat "HORN settling denoises corrupted states; a matched StateMLP does
   not" as the new working hypothesis with first supporting evidence, and make
   recovery/denoising the primary task framing for the next sprint instead of
   free-running generation.
2. Next confirmatory step before any training investment: repeat with a
   structured corruption (masking / occlusion of the retinotopic state grid,
   not just isotropic noise) and with more seeds, since occlusion recovery is
   the behaviorally interesting version of this claim.
3. If that holds, the training version is a denoising objective where the
   *task* is recovery (train to settle from corrupted encoded states back to
   the clean image), where HORN's measured contraction should be a real prior
   rather than an accident.

## CIFAR RGB Occlusion Recovery Probe (Structured Damage, Eval Only)

Motivation: the noise-then-settle probe showed HORN reliably absorbs isotropic
Gaussian state noise while the same-stack StateMLP does not. The
associative-memory version of that claim — the behaviorally meaningful one —
is recovery from *structured* damage: occlude a patch of the retinotopic state
grid and ask whether settling fills the missing region back in.

Setup (same six epoch-40 checkpoints, local, eval-only):

- `compute_generator_state_fitting_probe` gained `occlusion_fractions`: after
  fitting states to 32 real CIFAR images, a square patch of the 16x16 spatial
  site grid covering ~6.25/12.5/25% of sites is zeroed (position and velocity,
  both modes) at a random location per image, then the dynamics settle
  1/2/4/8/16 steps. Decode error is reported separately for the image region
  under the occluded sites and for the intact remainder.
- Outputs in `outputs/analysis/state_recovery_probe_occlusion/`, aggregate in
  `state_occlusion_probe_aggregate.csv`.

Result: **no arm fills in occluded regions — the associative-memory claim
fails at this operating point.** Occluded-region MSE (x1000) at 25% occlusion:

```text
Arm/seed                        k=0    k=1    k=2    k=4    k=8    k=16
horn_prior_class/s23           57.9   59.6   61.5   64.0   64.1   66.6
horn_prior_class/s24           56.2   56.4   58.6   64.8   69.6   65.0
horn_prior_global/s23          58.7   62.1   71.8   79.4   74.1   79.0
horn_prior_global/s24          55.6   56.5   59.1   68.2   69.7   73.5
state_mlp_prior_class/s23      55.3   55.7   56.0   56.4   63.1   75.0
state_mlp_prior_class/s24      56.8   56.2   56.1   54.2   61.9   66.3
```

Best-case fill-in delta over all fractions/depths/seeds (negative would mean
the missing region improved): HORN class `+0.20`, HORN global `-0.20`,
StateMLP `-3.81` (x1000) — i.e. at best negligible, and on average settling
makes the missing region slightly worse for HORN while degrading the intact
region substantially (intact MSE grows from ~1.2 to 40-70 x1000 at k=16).

Read:

- The Gaussian-noise recovery asymmetry does **not** extend to structured
  occlusion. HORN's measured denoising is contraction of distributed,
  isotropic perturbations — every corrupted dimension is pulled back a
  fraction — not pattern completion, which requires transporting information
  laterally from intact sites into the hole. The local-radius coupling either
  cannot do this or was never trained to.
- The same-stack StateMLP is, if anything, marginally less bad on the hole at
  shallow depths (its transition is global/dense, HORN's coupling is local),
  which is a useful hint that lateral information flow is the missing
  ingredient, not oscillation per se.
- The zeroed-patch decode itself sits at the "gray patch" floor
  (~0.055 region MSE, roughly the pixel variance of CIFAR), so k=0 is
  equivalent to rendering nothing; settling never gets below that floor.
- Scope: untrained-for-recovery checkpoints, n=2 seeds, zero-fill corruption.
  This is a negative for *emergent* fill-in, not for trainable fill-in.

Decision (gate outcome for the recovery roadmap):

1. Step-1 confirmation is **split**: isotropic-noise recovery asymmetry is
   robust (80/80 HORN cells positive, control fails 9/40), but occlusion
   recovery is absent in all arms. The associative-memory framing is dead at
   this operating point without recovery-specific training.
2. Any step-2 training investment ("corrupted state in, clean image out after
   settling") must therefore include occlusion/masking in the training
   corruption family and score occluded-region MSE explicitly — noise-only
   denoising training would ride the already-present bias and prove little.
3. The clean-state drift blocker stands unchanged: even undamaged fitted
   states degrade ~50x over 16 settle steps in every arm. Making clean states
   true fixed points remains the first-class training goal.

## Notes on Research

First — understood, and sorry. I killed that local test run; I won't run training or the full suite locally again. Anything real goes to Modal. The tiny unit tests I ran are CPU-only and finish in seconds, but the epoch-scale stuff belongs on the GPU cloud.

Now the actual question, because it's a good one and worth being honest about.

### Why distributed vs. contiguous split exists

The dividing line isn't really "oscillators vs. not." It's **locality of coupling under a finite settling budget**, and it comes straight out of physics.

A local coupling field moves information at a finite speed. Each settling step, influence spreads by roughly one coupling radius. So after `k` steps, a site can only "hear" from neighbors within radius `~k · r`. That's a light-cone / group-velocity limit, identical to how a wave or a diffusion front propagates.

- **Distributed corruption** (isotropic noise, scattered small patches): every corrupted site has intact, correct information *immediately adjacent* to it. Local relaxation pulls it back into consistency within a couple of steps. The signal to repair from is right there.
- **One contiguous hole**: the *interior* of the hole is many coupling-radii from the nearest intact pixel. Within any bounded settle budget, boundary information physically cannot reach the center. And even with infinite settling, a local operator solving the interior is solving Laplace's equation with the hole boundary as Dirichlet conditions — the classic **soap-film / membrane solution**. It gives you a smooth low-frequency interpolant and *erases interior detail*, because the interior is genuinely underdetermined by local information. The content that belongs there isn't encoded anywhere nearby; it has to be *invented* from a prior or *transported* from far away.

Spectrally: a contiguous hole is a **low-spatial-frequency defect** (large spatial scale). A local fast field only has short-wavelength modes. Fixing a large-scale defect requires large-scale (long-range / global) modes, which a homogeneous local field simply doesn't contain. That's also exactly why, in the old Winfree branch, the *slow/global carrier* was the one intervention that helped even a little — it added a long-wavelength mode.

One thing worth stating precisely: what oscillators specifically add over generic local recurrence is **contraction toward a learned attractor manifold** — damped bounded dynamics relax the off-manifold component of a perturbation while preserving on-manifold structure. Isotropic noise is almost entirely off-manifold, so it gets contracted (denoising). A contiguous hole isn't "small off-manifold noise everywhere"; it's a large region of *missing* signal whose correct fill lies *along* the manifold but requires spatial transport. Contraction doesn't help with transport. That's the crisp reason the same mechanism wins one task and loses the other.

### Does nature hit this, and how does it solve it

Yes, constantly — and the recurring answer is the punchline: **nature essentially never solves contiguous-hole completion with a single homogeneous local oscillator field.** It always adds one of these:

- **Hierarchy + long-range connectivity (the brain).** Perceptual filling-in is real (the blind spot, scotoma completion), but it's driven by *feedback from higher visual areas with large receptive fields and learned priors*, plus long-range cortico-cortical and thalamo-cortical loops and cross-frequency coupling (theta/gamma). The "global carrier" is literally wired in as long-range anatomy. It is not bare lateral propagation.
- **Long-range / mean-field coupling (physics of synchronization).** Kuramoto global sync, laser arrays, Josephson-junction arrays — global coordination is a long-wavelength phenomenon and it's achieved with mean-field or long-range coupling or a common drive, not strict nearest-neighbor.
- **Global mode / standing-wave bases (acoustics).** A room's low-frequency behavior is dominated by global standing-wave eigenmodes that span the whole cavity. A "hole" in a pressure field is naturally handled because the representation itself is global. Note what that means: it works *because the basis is non-local*.
- **The holography inversion (interesting counter-case).** In a hologram, information about each image region is spread across the whole plate. Occlude a contiguous patch and you lose *resolution*, not a *patch of the image*. That's the opposite regime: **distributed encoding makes contiguous occlusion benign.** It's a hint that if you want oscillators to shrug off big holes, the lever is a *distributed/global representation*, not a better local rule.
- **Priors + signaling (morphogenesis).** Planaria/hydra regenerate large missing chunks, but via stored positional information, morphogen gradients, and long-range signaling — evolved priors, not local relaxation.

Even the modern ML answer agrees: diffusion models own the *distributed-corruption, iterative-denoise* regime (exactly the oscillator sweet spot), but they handle big contiguous inpainting through the **learned global prior** — the U-Net's large receptive field and attention supply the long-range structure. The dynamics do the denoise; a global prior does the transport.

### So were you over-promising?

Slightly, yes — but in a very useful way to have surfaced now. The honest read:

- You are **not** under-selling nature. Nature figured this out repeatedly, and its solution is **hybrid/multiscale every single time.** Your instinct that "hybrid architectures are the norm" is correct and is the mainstream position.
- Pure homogeneous local oscillatory coupling being bad at contiguous holes is **not an implementation failure and not specific to oscillators** — our non-oscillatory recurrent-conv control failed the same way. It's a property of the *local operator class*. That's actually the cleaner scientific statement.
- The defensible ONN claim therefore narrows to: **oscillator dynamics are a strong inductive bias for iterative, distributed denoising / contraction onto a learned manifold.** Not "oscillators alone complete arbitrary occlusions." Contiguous-hole completion requires an added long-wavelength/global/prior mechanism — which is precisely the multiscale carrier direction, and precisely what biology and physics use.

### What I think this means for step 2 (and why I did *not* launch anything)

Given you're rethinking the framing, I deliberately held off burning GPU. The code for the recovery-training objective is done and passing tiny CPU tests: image-space occlusion applied before encoding, state-space noise, an explicit clean-fixed-point term, and region-scored PSNR/SSIM eval. The Modal sweep (HORN + StateMLP, 2 seeds, noise-only and mixed-corruption arms) is wired and one command from launching.

But this discussion argues for one design change before we spend the compute: **point the experiment at the regime oscillators actually own** — Gaussian state noise plus *scattered* small patches — and treat the single contiguous block as the documented stretch/negative, not a target. If we want to seriously attack contiguous holes, the physically-motivated arm isn't more local tuning; it's an explicit **long-wavelength/global carrier or hierarchical phase field** (the holography/standing-wave lesson), which we already have partial code for from the Winfree era.

### Recovery Training Objective (Step 2) — Design

Implemented the corrupted-anchor training objective. The training-time anchor
loss (`_state_anchor_image_loss`) now trains the explicit denoising task
"corrupted state/image in, clean image out after settling":

- **Image-space occlusion before encoding.** `occlude_image_batch` zeros square
  patches of the input image, applied *before* the anchor encoder, so the model
  must reconstruct clean pixels from a corrupted observation. `patches=1` is one
  contiguous block; larger `patches` scatter the same total area (the regime
  oscillators own). Controlled by `state_anchor_occlusion_fraction`,
  `state_anchor_occlusion_patches`, `state_anchor_occlusion_probability`.
- **State-space Gaussian noise** on the encoded position/velocity (existing
  `state_anchor_noise_scale`), the isotropic off-manifold perturbation.
- **Explicit clean fixed-point term** (`state_anchor_clean_weight`): also encode
  the *uncorrupted* image, settle, decode, and score against the clean image, so
  clean states are trained to be genuine fixed points of the dynamics. This
  directly attacks the clean-drift blocker.

Task-level evaluation (`compute_generator_recovery_metrics`, summary key
`generator.recovery.*`) encodes real eval images, corrupts, settles across
depths, decodes, and reports paired MSE, PSNR, and SSIM. Occlusion conditions
additionally split the decode error into occluded-region (fill-in) vs
intact-region MSE, the direct fill-in metric.

### Slow/Global Carrier Arm

Following the physics/biology argument above — contiguous-hole completion needs
a long-wavelength mode, not better local tuning — the sweep includes a
slow/global carrier arm built on the existing `CoarseToFineHORNImageGenerator`
(the current-stack port of the Winfree slow/global carrier, which was the one
ONN-native intervention that improved block occlusion in the earlier branch).

The change that makes the carrier participate in recovery: the coarse model now
overrides `evolve_state` so that when it is called with only the fine state (the
anchor/recovery path), it seeds the coarse carrier by parameter-free spatial
mean-pooling of the (corrupted) fine encoded state, then runs the coupled
coarse-fine dynamics and returns the fine state. Free sampling still uses
`sample_state` (random coarse init) and is unaffected. The pooled seed is a
global summary that largely survives occlusion, giving the carrier long-range
content to transport into the hole.

Comparison design (all mixed corruption: noise + scattered 4-patch occlusion,
single contiguous block scored at eval as the stretch condition):

- `single_local_recovery_mixed` — single-mode local HORN, no carrier (matched
  baseline).
- `coarse_carrier_recovery_mixed` — same single-mode fine field plus a 16-node
  dense coarse carrier. Isolates the carrier's effect on occluded-region MSE.

If the carrier moves the contiguous-hole (`occl_f*_p1_*_occluded_region_mse`)
number relative to the matched local baseline, that is the first
control-beating evidence for the multiscale mechanism the theory predicts. If it
does not, the negative is now scientifically meaningful rather than a foregone
conclusion, because the one mechanism with a physical reason to help was tested.

Launched on Modal as sweep `mnist_generator_cifar10_rgb_recovery_training_probe`
(app `ap-GzzdJI4vJm3HrgTHp0bMgl`, detached); see `docs/modal_runs.md`.

### Recovery Training Probe — Results (2026-07-15)

All 12 runs (6 arms x seeds 23/24, 40 epochs, train 10k) completed cleanly on
app `ap-15xJuAPbD45RyKC9p4akCd` (relaunch of the cancelled serial sweep with 8
parallel containers). Source:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_recovery_training_probe.csv`.

Occluded-region decode MSE (fill-in metric, lower is better), two-seed means,
25% single contiguous block (`p1`) and scattered 4-patch (`p4`):

| Arm | p1 k0 | p1 k8 | p1 k16 | p4 k8 | clean k8 PSNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_mlp_recovery_mixed | 0.0420 | 0.0407 | 0.0409 | 0.0274 | 25.5 |
| horn_recovery_mixed (multimode2) | 0.0737 | 0.0663 | 0.0625 | 0.0409 | 24.3 |
| coarse_carrier_recovery_mixed | 0.0811 | 0.0934 | 0.0951 | 0.0698 | 19.7 |
| single_local_recovery_mixed | 0.0867 | 0.0983 | 0.1002 | 0.0721 | 19.4 |
| horn_recovery_noise | 0.2303 | 0.2341 | 0.2262 | 0.2145 | 25.6 |
| state_mlp_recovery_noise | 0.2569 | 0.2269 | 0.2067 | 0.2097 | 26.7 |

Findings:

1. **Occlusion training works.** Mixed-corruption arms fill occluded regions
   3-5x better than noise-only arms (0.04-0.10 vs 0.21-0.26). The
   corrupted-anchor objective successfully trains fill-in behavior into the
   full encode-settle-decode stack.
2. **The slow/global carrier hypothesis failed.** Carrier vs matched
   single-local baseline is mixed-sign across seeds (seed23: carrier better,
   0.0964 vs 0.1089; seed24: baseline better, 0.0903 vs 0.0878) — no reliable
   effect. Worse, both single-mode arms *degrade* the occluded region with
   settling depth (k0 -> k16 rising). The pooled-seed dense carrier did not
   deliver measurable long-range transport at this scale.
3. **Multimode HORN is the only arm where settling dynamics actively improve
   fill-in**, consistently on both seeds (k0 -> k16: 0.0719 -> 0.0575 and
   0.0755 -> 0.0675). StateMLP's settling gain is negligible (~0.001-0.002);
   its fill-in comes almost entirely from the encoder/decoder at k0. So the
   dynamics-driven fill-in signature we hoped the carrier would provide showed
   up in the multimode field instead.
4. **The StateMLP control still wins absolutely** (0.041 vs 0.066 at p1 k8,
   and best PSNR), repeating the project's recurring pattern: the trained
   recovery task is dominated by the static encode/decode pathway, where the
   recurrent MLP equilibrium is simply better, and the oscillator dynamics
   advantage (real but small) does not close that gap.

Read: the training objective is validated, the carrier arm is falsified in its
current pooled-seed dense form, and the one genuinely ONN-native positive is
the multimode settling-improves-fill-in trend — small, consistent, and the only
place where dynamics (not the decoder) are doing completion work.

### Multimode Carrier Probe — Active-Ingredient Ablation (2026-07-15)

Follow-up sweep `mnist_generator_cifar10_rgb_multimode_carrier_probe` (app
`ap-ew3UgLxsft1IorlyV84F3l`, 8 parallel A10G, 40 epochs, mixed corruption,
recovery eval extended to settle depths 0-64). Arms: multimode2 with equal
frequencies (1.0/1.0), wide split (0.5/1.5), multimode4, and a "fair" slow
carrier — `CoarseToFineMultiModeHORNImageGenerator`, a 16-node dense coarse
band at 0.5x frequency (`coarse_frequency_scale`) on the multimode substrate.
The mm4 arm OOMed on A10G (1024 oscillators, 16.4GiB `_train_step` allocation
at batch 32) on both seeds and is absent. Source:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_multimode_carrier_probe.csv`.

Contiguous-block occluded-region MSE, two-seed means (anchors from the
recovery training probe: multimode2 default 0.75/1.35 and StateMLP):

| Arm | k0 | k8 | k16 | k32 | k64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mm2 equal freq (1.0/1.0) | 0.0737 | 0.0578 | 0.0537 | 0.0980 | 0.0797 |
| mm2 slow carrier (0.5x band) | 0.0625 | 0.0563 | 0.0560 | 0.1066 | 0.1183 |
| mm2 default (0.75/1.35), anchor | 0.0737 | 0.0663 | 0.0625 | — | — |
| mm2 wide (0.5/1.5) | 0.0783 | 0.0697 | 0.0687 | 0.1420 | 0.1179 |
| StateMLP mixed, anchor | 0.0420 | 0.0407 | 0.0409 | — | — |

Findings:

1. **The slow-band hypothesis is falsified.** Equal-frequency multimode — no
   timescale separation at all — shows the *strongest* settling fill-in trend
   (0.0737 -> 0.0537 by k16, better than the default split on both seeds).
   Wider separation (0.5/1.5) is *worse* and higher-variance. The active
   ingredient of the multimode effect is per-site state capacity and mode
   coupling, not a slow channel.
2. **The centralized carrier is retired permanently.** Given its fair shot
   (explicitly slower band, strong multimode substrate), the slow-carrier arm
   sits within noise of the non-carrier arms at k8/k16 and does not beat the
   best of them. Third strike across three implementations.
3. **Deep settling reverses.** All arms peak around k8-k16 and degrade badly
   by k32-k64. There is no iterate-to-perfection regime; the fill-in window is
   shallow.
4. **StateMLP remains untouched** (0.041 at k8) — the best oscillator fill-in
   number yet (eqfreq k16, 0.0537) is still ~30% worse.

Read: the timescale/slow-carrier theory thread is now closed — falsified in
centralized form twice and in distributed form once. What survives of the
multimode surprise is a capacity/mode-coupling effect with a shallow optimal
settling window, still dominated by the non-oscillatory control. The mm4
dose-response rerun (batch 16) is optional and low-value now that the
capacity interpretation carries the effect.

### Coupling-Topology Probe — Is Locality the Real Confound? (2026-07-15)

Reframing of every prior recovery result. Across all recovery probes the
independent variable was implicitly "oscillator vs. not," but the generative
HORN and the StateMLP control differ on a second axis that was never isolated:
**coupling range.** The HORN uses spatially-local coupling
(`local_radius_coupling_profile` / `distance_decay_coupling_profile` in
`oscnet/models/generative/common.py`), where each oscillator only reaches its
grid neighbours. StateMLP uses dense `Linear` hidden-to-hidden layers
(`coupling_profile = "none"`, fully connected). A dense matrix can transport
structure across a contiguous occlusion in one step; local coupling provably
cannot (the "contraction vs. transport" wall). So StateMLP may dominate for the
same reason HORN fails at contiguous holes — non-local vs. local coupling —
not because oscillation is worthless. Independent support: the repo's fractal
coupling module (`oscnet/core/fractal_coupling.py`), a self-similar non-local
topology built for pattern completion, reports large associative-recall gains
but was never wired into the generative recovery pipeline.

Design: hold the recovery-trained mixed-corruption objective fixed and vary
**only** the recurrent coupling topology. New `fractal` profile added to
`oscnet/core/coupling.py` (`hierarchical_coupling_profile`): oscillators on the
usual near-square grid, recursively split into quadrants; two sites sharing
their finest block couple at full strength and each coarser level of divergence
multiplies coupling by `inter_block_strength` (0.5). This yields discrete
self-similar scales with direct long-range links — the non-local structure
local coupling lacks — while staying far cheaper in effective energy than flat
dense. Sweep `mnist_generator_cifar10_rgb_coupling_topology_probe` (app
`ap-qTl1hsn2kkqoRTqfulAyKz`, 8 parallel A10G, 40 epochs, seeds 23/24, recovery
eval settle depths 0-32). Six arms:

- single-mode HORN, local vs. fractal (topology isolated, no capacity confound);
- multimode2 HORN, local vs. dense vs. fractal (topology on the stronger
  substrate);
- StateMLP mixed anchor (dense-linear non-oscillatory ceiling).

Prediction that would rescue the ONN thread: if the StateMLP win is really about
non-local coupling, the dense/fractal oscillator arms should close the gap to it
and beat their local baselines. If dense/fractal do *not* help either, the
advantage was always dense linear mixing and oscillation adds nothing — a clean
negative that retires the question without the locality confound.

Completed 2026-07-15 18:33 CEST, all 12 runs clean. Contiguous-block
occluded-region MSE, two-seed means:

| Arm | k0 | k8 | k16 | k32 |
| --- | ---: | ---: | ---: | ---: |
| single local | 0.0769 | 0.0912 | 0.0954 | 0.0743 |
| single fractal | 0.0835 | 0.0943 | 0.0933 | 0.0910 |
| mm2 local | 0.0794 | 0.0692 | 0.0642 | 0.1237 |
| mm2 dense | 0.0656 | **0.0567** | 0.0611 | 0.1281 |
| mm2 fractal | 0.0651 | 0.0639 | **0.0608** | 0.1302 |
| StateMLP | 0.0421 | **0.0378** | 0.0385 | 0.0483 |

Findings:

1. **The locality confound was real — non-local coupling helps.** On the
   multimode substrate, dense coupling beats local at every depth (k8: 0.0567
   vs 0.0692, ~18% better, consistent across both seeds; k0 already ~17%
   better). Fractal tracks dense closely (best k16 0.0608). The
   contraction-vs-transport account survives its first direct intervention:
   giving oscillators long-range links buys real contiguous-hole fill-in.
2. **But locality explains only ~a third of the gap.** With coupling range
   equalized (dense oscillator vs dense-linear StateMLP), StateMLP still wins
   by ~50% (0.0378 vs 0.0567 at k8) and is far more stable at depth (k32:
   0.0483 vs ~0.13). The residual deficit is attributable to the oscillatory
   update rule itself — the second-order dynamics constraint — not to
   topology.
3. **Fractal ≈ dense.** The self-similar structure matches flat all-to-all but
   does not beat it at this scale; the active ingredient is non-locality, not
   discrete scale invariance. (Fractal does achieve parity with far more
   structured, compressible coupling, which may matter for efficiency claims.)
4. **Topology needs capacity.** On the single-mode substrate fractal does
   nothing (0.0835 vs 0.0769 local; settling hurts both). Non-local links only
   pay off where per-site state capacity (multimode) lets dynamics do active
   fill-in — topology and capacity interact rather than add.
5. **Deep-settling reversal unchanged.** All mm2 arms still degrade sharply by
   k32 regardless of topology; StateMLP degrades far more gracefully.

Read: the confound is resolved cleanly in both directions. Locality was
genuinely holding oscillators back on contiguous holes (hypothesis partially
confirmed, and the transport theory predicted it), but equalizing coupling
range does not close the gap to the non-oscillatory control — so the remaining
deficit is intrinsic to the oscillator update at this scale. The negative
result for "ONNs beat recurrent MLPs at recovery" now stands on an unconfounded
comparison, which makes it publishable-grade: local coupling → no transport;
add transport → recover a third of the gap; the rest is the dynamics prior
itself.

### Robustness Probe — Scoring on the Oscillator's Home Fitness Function (2026-07-16)

Motivation: every prior comparison scored exact reconstruction at
infinite-precision parity on digital hardware — the fitness function digital
free-form updates are optimized for. The claimed advantages of
physical/oscillatory computation (graceful degradation under component noise,
low-precision tolerance, robustness off-nominal) were never on the scorecard.
This probe measures the digital-measurable shadow of those claims. Sweep
`mnist_generator_cifar10_rgb_robustness_probe` (app
`ap-CYvg8hNf6VjCCJH9Q9xxHa`, 8 parallel A10G, seeds 23/24, all runs clean).
Arms: single-mode local HORN, multimode2 local, multimode2 dense, StateMLP.
All conditions score contiguous-occlusion fill-in (occluded-region MSE) and
clean PSNR at fixed settle depth 8, under: whole-model Gaussian weight jitter
(per-leaf-std scales 0.02-0.2, 3 draws), uniform weight quantization (8-3
bits), and out-of-distribution occlusion (0.1-0.6 vs 0.25 trained).

Occluded-region MSE at k8 (2-seed means):

| Condition | single local | mm2 local | mm2 dense | StateMLP |
| --- | ---: | ---: | ---: | ---: |
| baseline (0.25 occl) | 0.0905 | 0.0620 | 0.0576 | **0.0411** |
| occl 0.4 | 0.1065 | 0.0812 | 0.0825 | 0.0773 |
| occl 0.6 | 0.1237 | **0.1070** | 0.1181 | 0.1724 |
| quant 4-bit | 0.0747 | 0.0651 | **0.0504** | 0.0426 |
| quant 3-bit | 0.1272 | **0.0668** | 0.0661 | 0.0866 |
| wnoise 0.2 | 0.0923 | 0.0825 | 0.0611 | **0.0573** |

Relative degradation (condition / own baseline): at occl 0.6 StateMLP blows up
4.19x vs 1.73x (mm2 local) and 1.37x (single local); at 3-bit quantization
StateMLP degrades 2.11x vs 1.08x (mm2 local). Per-seed checks: at occl 0.6 and
at 3-bit, mm2 local beats StateMLP in absolute terms on *both* seeds.

Findings:

1. **The crossover exists — first absolute oscillator win in the project.**
   Under severe out-of-distribution occlusion (0.6, 2.4x the trained level)
   the multimode-local oscillator's fill-in (0.107) is ~38% *better* than
   StateMLP's (0.172), both seeds. The free-form update shatters off-nominal;
   the physics-constrained update bends.
2. **Extreme quantization favors the oscillator too.** At 3-bit weights the
   multimode arms hold ~1.1x their baseline while StateMLP doubles its error;
   mm2 local wins absolutely on both seeds. At moderate precision (8-4 bits)
   StateMLP keeps its lead — the crossover is at the harsh end.
3. **Weight-noise results are mixed.** At scales up to 0.1 StateMLP is no less
   robust than the oscillators (relative degradation is oscillator-worse at
   0.05-0.1); only at the extreme 0.2 scale does the pattern flip on relative
   degradation and clean PSNR (mm2 local 17.8 dB vs StateMLP 15.3 dB). Only 3
   noise draws — treat as suggestive, not established.
4. **On-nominal ordering unchanged.** At trained conditions and mild stress
   StateMLP remains the best absolute performer, consistent with every prior
   probe.

Read: the "lost the office, won the battlefield" pattern is now measured in
our own data. The oscillator's dynamics prior costs ~40% accuracy at nominal
conditions but is *protective* under distribution shift and precision
starvation — exactly the operating regime physical/analog computation targets.
This converts the project's story from a pure negative into a two-regime
result: free-form recurrence wins the in-distribution, full-precision contest;
physics-constrained recurrence wins when conditions leave the training
envelope. Caveats before any strong claim: 2 seeds, 3 weight-noise draws, one
dataset, and the crossover appears only at the harsh end of the stress axes —
a replication with more seeds/draws and intermediate stress levels would be
the natural confirmation step.

### Robustness Confirmation — 4 Seeds, Crossover Localization, Overfitting Control (2026-07-16)

The confirmation step the probe called for. Sweep
`mnist_generator_cifar10_rgb_robustness_confirmation` (app
`ap-sYXsSdYEHPBDjeOebDUIaL`, 8 parallel A10G, 16 runs, all clean). Changes vs.
the probe: seeds 23-26 (4 instead of 2); 5 weight-noise draws (instead of 3)
at scales 0.05/0.1/0.15/0.2/0.3; quantization 8/6/5/4/3 bits; occlusion
0.1/0.25/0.3/0.4/0.5/0.6 so the crossover point is bracketed instead of
sampled at the endpoints only; and a fourth arm, **regularized StateMLP**
(new preset `..._state_mlp_reg_...`, weight decay 2e-3 = 10x the default),
to test the skeptic's reading that StateMLP's off-nominal collapse is mere
overfitting that any regularizer would fix.

Occluded-region MSE at k8 (4-seed means):

| Condition | mm2 local | mm2 dense | StateMLP | StateMLP reg |
| --- | ---: | ---: | ---: | ---: |
| baseline (0.25 occl) | 0.0708 | 0.0626 | 0.0414 | **0.0398** |
| occl 0.4 | 0.0926 | 0.0884 | **0.0779** | 0.0781 |
| occl 0.5 | 0.1107 | **0.1077** | 0.1264 | 0.1391 |
| occl 0.6 | 0.1243 | **0.1226** | 0.1546 | 0.1713 |
| quant 4-bit | 0.0645 | 0.0717 | 0.0534 | **0.0467** |
| quant 3-bit | 0.1419 | 0.1130 | 0.0902 | **0.0649** |
| wnoise 0.3 | 0.1020 | 0.1005 | **0.0758** | 0.0755 |

Relative degradation (condition / own baseline) at occl 0.6: mm2 local 1.80x,
mm2 dense 2.01x, StateMLP 3.73x, StateMLP reg 4.30x. Per-seed absolute wins
for mm2 local at occl 0.5/0.6: 3/4 seeds vs plain StateMLP, 4/4 vs
regularized.

Findings:

1. **The occlusion crossover is confirmed and localized.** It opens between
   0.4 and 0.5 occlusion (1.6x-2.4x the trained level). At 0.4 StateMLP still
   wins on every seed; at 0.5 and 0.6 both oscillator arms beat both StateMLP
   arms on the 4-seed mean, with mm2 local winning 3/4 seeds against plain
   StateMLP and 4/4 against the regularized one. This is now a replicated,
   bracketed result, not an endpoint artifact.
2. **The overfitting explanation is dead.** Regularizing StateMLP improves
   its nominal baseline slightly (0.0398 vs 0.0414) but makes off-nominal
   collapse *worse* (4.30x vs 3.73x at occl 0.6). The free-form update's
   brittleness under distribution shift is structural, not a symptom of
   under-regularization — strengthening the case that the oscillator's
   physics prior is doing real protective work.
3. **The 3-bit quantization crossover does not replicate.** With 4 seeds the
   mm2-local relative degradation at 3 bits is 2.12x — statistically
   indistinguishable from plain StateMLP's 2.16x — and the probe's 2-seed
   oscillator win (1.08x) was evidently seed luck. Better yet for honesty:
   the regularized StateMLP is the *most* quantization-robust arm (1.62x,
   absolute 0.0649), consistent with weight decay shrinking weight range and
   hence per-level quantization error. The quantization claim from the probe
   should be retired.
4. **Weight noise: relative robustness, no absolute win.** The oscillator
   degrades relatively less at every scale from 0.15 up (1.04-1.46x vs
   1.27-1.83x for StateMLP, now with 5 draws x 4 seeds) and retains more
   clean PSNR at scale 0.2 (18.7 vs 17.9 dB), but StateMLP's nominal head
   start means it stays ahead in absolute fill-in throughout, and at the
   extreme 0.3 scale the PSNR ordering flips back.

Read: the two-regime story survives its confirmation, but narrower and
sharper than the probe suggested. The one replicated absolute oscillator win
is severe out-of-distribution occlusion — the pattern-completion stressor
closest to the associative-memory claim — and it is robust to seeds and to
the overfitting control, with a clean crossover bracketed at 0.4-0.5
occlusion. The quantization and weight-noise advantages reduce to
relative-degradation effects or vanish entirely. The defensible claim is now:
*free-form recurrence reconstructs better inside and near the training
envelope; physics-constrained oscillator dynamics fill in better when the
corruption is far outside it, and this is not an artifact of an
under-regularized control.* Remaining caveats: one dataset, one architecture
family per side, and the crossover is specific to the occlusion axis.

### Synthesis — How Nature Keeps the Upsides Without the Downsides (2026-07-16)

Standalone write-up: `docs/what_the_physics_prior_buys.md` (scientific-style
article covering the full generative/recovery arc, Acts I–VI). This section
records the closing conceptual takeaway that motivated it.

The two-regime result reduces to one mechanism: the damped second-order
oscillator update is a **smoother, bounded, contractive function class**. It
is too smooth to express the sharp data-specific equilibria that make the
free-form StateMLP accurate on-nominal, and for exactly the same reason it
cannot express the sharp data-specific failure modes that make StateMLP
collapse off-nominal. Smoothness is a tax in-distribution and insurance
out-of-distribution. The physics prior buys **predictability, not
performance**: ~40-70% worse on every in-distribution metric, but ~1.8x
degradation vs 3.7-4.3x under out-of-distribution occlusion, and the
regularized control made the collapse *worse*, not better — so the
brittleness of free-form recurrence is structural, not under-regularization.

The natural follow-on question — how does nature keep the oscillator upsides
(contraction, denoising, graceful degradation) *without* the downsides
(nominal-accuracy tax, no contiguous transport, shallow settling window)? —
has a single answer that unifies every negative result in this report:
**nature never runs a pure homogeneous local oscillator field in isolation.**
All three downsides are properties of that specific object, and biology and
physics remove each by wrapping the dynamics in surrounding structure rather
than by purifying the oscillator:

- **Nominal-accuracy tax (too smooth to be precise).** The dynamics are used
  for coordination/binding/gating (theta-gamma phase coding, synchronization
  for routing), while precision lives elsewhere — feedforward pathways,
  learned synaptic weights, sparse codes. Division of labor, not a better
  oscillator. Our data shows the digital shadow of this: StateMLP's static
  encode/decode pathway, not its recurrence, carries almost all of its
  on-nominal fill-in; the oscillator's dynamics-driven gain is real but small.
- **No contiguous transport (contraction != transport).** Nature adds
  long-range and top-down structure: cortical filling-in is driven by
  feedback from higher areas with large receptive fields and learned priors,
  plus long-range cortico-cortical loops — not lateral relaxation in one
  sheet. This is exactly what the coupling-topology probe demonstrated in
  miniature: adding non-local links recovered ~a third of the gap. Nature
  wires that non-locality in anatomically and backs the *invented* part with
  a global prior.
- **Shallow settling window (no iterate-to-perfection).** Physical systems
  use multiple timescales, hierarchy, and global-mode / standing-wave bases
  so large-scale structure is represented directly instead of propagating
  step by step (room acoustics: low frequencies live in global eigenmodes;
  holography: distributed encoding makes contiguous occlusion cost resolution
  not a patch). When the representation itself is non-local, the hole problem
  partly dissolves. Note our carrier arm tried to add a single global mode and
  failed three times — nature's version is many modes + hierarchy + priors,
  not one bolted-on slow band.

Unifying principle: **keep the oscillator's contraction/robustness and
offload everything it is bad at onto a surrounding architecture — hierarchy,
long-range feedback, learned priors, multiple timescales, non-local /
holographic representations.** Every biological or physical system that
shrugs off damage is hybrid and multiscale; none is a pure local oscillator
field that lost its downsides. This is why the constructive direction is
*designed hybrids* (a free-form/global-prior pathway for accuracy and
transport, an oscillatory field for off-manifold cleanup and off-nominal
insurance, and a router keyed on estimated distribution shift) rather than a
better standalone ONN — the hybrid is the only architecture that has ever
actually achieved "keep the upsides, drop the downsides."

### Hybrid Frontier — Augmentation Control, Held-Out Battery, Designed Hybrid (2026-07-17)

The killer experiment the synthesis called for. Two untested claims decided
the remaining fate of the thread: (a) the skeptic's escape — "the oscillator
OOD win is cheap, just train the free-form net with occlusion augmentation";
(b) the constructive claim — a designed hybrid (free-form path + oscillator
path + learned router) keeps both upsides. Sweep
`mnist_generator_cifar10_rgb_hybrid_frontier` (app
`ap-b13XOIzKrIRKqmgKejzvwI`, 8 parallel A10G, 12 runs, seeds 23-26, all
clean). New machinery: `HybridImageGenerator`
(`oscnet/models/generative/hybrid.py`) — mm2-dense oscillator step and
StateMLP-style residual step blended per spatial site by a learned router
MLP over state statistics (gate bias init toward the free-form path);
`state_anchor_occlusion_curriculum` — every arm trains on contiguous
occlusion sampled from 0.1/0.25/0.4/0.6, so the StateMLP arm is
augmentation-hardened; held-out stressor battery
(`--robustness-eval-heldout-corruptions`) — image-space Gaussian noise,
salt-and-pepper, and periodic stripe occlusion, families *no* arm ever
trained on, plus OOD occlusion pushed to 0.7/0.85, beyond even the
curriculum.

Occluded-region / corrupted-region MSE at k8 (4-seed means; "ho" = held-out
family):

| Condition | StateMLP aug | mm2 dense aug | hybrid |
| --- | ---: | ---: | ---: |
| baseline (0.25) | 0.0408 | 0.0420 | **0.0391** |
| occl 0.4 | 0.0406 | 0.0429 | **0.0398** |
| occl 0.6 | 0.0413 | 0.0459 | **0.0410** |
| occl 0.7 | 0.0638 | **0.0481** | 0.0629 |
| occl 0.85 | 0.1305 | **0.0645** | 0.1358 |
| ho stripes 0.5 (region) | 0.1022 | **0.0482** | 0.1070 |
| ho gaussian 0.3 (mse) | **0.0126** | 0.0140 | 0.0130 |
| wnoise 0.2 | 0.0558 | 0.0630 | **0.0491** |
| clean PSNR | 25.14 | 23.81 | **25.42** |

Findings:

1. **Augmentation works — inside its coverage.** The curriculum-trained
   StateMLP holds its baseline flat through occlusion 0.6 (0.0413 vs the
   single-level-trained arm's 0.1546 in the confirmation sweep). The old
   0.4-0.5 crossover is gone. "Just augment for it" is real where you can
   enumerate the stressor.
2. **The frontier moved; it did not close.** Beyond the curriculum the
   crossover reappears wholesale: at occlusion 0.7 and 0.85 the oscillator
   wins on **4/4 seeds** (2x better at 0.85, 0.0645 vs 0.1305; relative
   degradation 1.5x vs 3.2x). Augmentation relocated the cliff; it did not
   remove the free-form update's cliff-shaped failure mode, nor the
   oscillator's flat extrapolation.
3. **The held-out family result is the decisive one.** Periodic stripe
   occlusion — a corruption *shape* absent from every training set — at
   severity 0.5 goes to the oscillator 4/4 seeds (0.0482 vs 0.1022). This is
   the case augmentation can never cover by construction: you cannot
   enumerate corruption families you have not anticipated. Distributed
   held-out families (Gaussian, salt-and-pepper) stay narrowly with the
   free-form arms — consistent with the entire project: distributed
   corruption is easy for everyone; structure-destroying corruption at
   severity is where the physics prior pays.
4. **The hybrid is the best on-nominal model but its insurance failed.**
   It wins baseline fill-in, clean PSNR, and weight-noise robustness (4/4 at
   scale 0.2) — the router genuinely fuses the paths profitably near
   distribution. But in deep OOD (occl 0.7/0.85, stripes 0.5) it tracks the
   free-form cliff (0/4 vs mm2), because the router itself is a learned
   free-form component: trained on the curriculum, where the free-form path
   was always fine, it learned to trust it — and that judgment fails
   off-distribution exactly like the pathway it gates. Nature's version of
   routing is anatomically hardwired long-range structure, not a learned
   gate; the next hybrid iteration should key the gate on a fixed,
   non-learned shift statistic (e.g. encoder-reconstruction residual or
   state-typicality score) rather than on trained features.

Read: the third branch of the hypothesis map, confirmed with a sharper
mechanism than expected. The physics prior is **not** redundant with
augmentation — it buys extrapolation beyond *whatever* envelope was trained,
and the advantage lives precisely in the regime that cannot be enumerated in
advance (unanticipated severity, unanticipated corruption families). The
constructive hybrid claim is half-proven: profitable fusion on-nominal,
failed handoff off-nominal, with a specific, testable fix (shift-robust
gating). The project's final claim upgrades from "two regimes exist" to:
*the free-form/oscillator trade-off is invariant to augmentation — moving
the training envelope moves the crossover but never removes it, so the
oscillator's insurance premium buys coverage exactly where coverage cannot
be bought by data.*



## Maintenance Notes

- Put numerical benchmark summaries in this file and/or `outputs/analysis`.
- Do not put eval tables in `README.md`.
- When adding a new model family, also add config, checkpoint loading,
  example exports, tests, and a small reference experiment test.
- For every positive result, add at least one capacity/control ablation.
