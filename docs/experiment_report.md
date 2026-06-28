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

Result:

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

## Maintenance Notes

- Put numerical benchmark summaries in this file and/or `outputs/analysis`.
- Do not put eval tables in `README.md`.
- When adding a new model family, also add config, checkpoint loading,
  example exports, tests, and a small reference experiment test.
- For every positive result, add at least one capacity/control ablation.
