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

The strongest direction so far is not a conventional latent autoencoder. It is
a conditional spatial field/refinement model, strongest when a semantic prior
is paired with a local recurrent correction field. Winfree phase dynamics are a
promising version of that correction field, but the recurrent-conv residual
control shows that the broader hybrid pattern also matters:

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

## Maintenance Notes

- Put numerical benchmark summaries in this file and/or `outputs/analysis`.
- Do not put eval tables in `README.md`.
- When adding a new model family, also add config, checkpoint loading,
  example exports, tests, and a small reference experiment test.
- For every positive result, add at least one capacity/control ablation.
