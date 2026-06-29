# Modal GPU Runs

Modal support is kept outside the OscNet library. The reusable models and
experiments still run locally with the normal Python CLI; the scripts under
`scripts/modal_*.py` are launch adapters for running the same experiments on
remote GPUs.

## Setup

Install the optional cloud dependency and authenticate Modal:

```bash
pip install -e ".[cloud]"
modal setup
```

The runner defaults to an `A10G` worker and JAX CUDA 13 wheels:

```bash
modal run scripts/modal_mnist.py
```

The default command is a tiny synthetic one-epoch smoke run. It prints
`jax.devices()` from the remote worker and returns the experiment summary.
Sweep runs default to at most three GPU containers at once via
`OSCNET_MODAL_MAX_CONTAINERS=3`, so they do not consume the full workspace GPU
limit by accident.

## Real Run

Pass the normal MNIST CLI options as one quoted `--experiment-args` string:

```bash
modal run scripts/modal_mnist.py \
  --run-name block50_global_seed11 \
  --experiment-args "--model-family winfree_global_rate_phase --seed 11 --corruption-mode block_occlusion --corruption-fraction 0.5 --corruption-seed 11 --epochs 10 --train-limit 2000 --eval-limit 500 --batch-size 64 --hidden-dim 64 --latent-dim 64 --patch-size 4 --winfree-steps 8 --artifact-every 10 --checkpoint-every 10"
```

Coarse phase-mesh block-occlusion probe:

```bash
modal run scripts/modal_mnist.py \
  --run-name block50_coarse2_seed11 \
  --experiment-args "--model-family winfree_coarse_global_rate_phase --seed 11 --corruption-mode block_occlusion --corruption-fraction 0.5 --corruption-seed 11 --epochs 10 --train-limit 2000 --eval-limit 500 --batch-size 64 --hidden-dim 64 --latent-dim 64 --patch-size 4 --winfree-steps 8 --winfree-coarse-grid-size 2 --winfree-global-phase-control none --artifact-every 10 --checkpoint-every 10"
```

Phase-shuffled control:

```bash
modal run scripts/modal_mnist.py \
  --run-name block50_coarse2_shuffle_seed11 \
  --experiment-args "--model-family winfree_coarse_global_rate_phase --seed 11 --corruption-mode block_occlusion --corruption-fraction 0.5 --corruption-seed 11 --epochs 10 --train-limit 2000 --eval-limit 500 --batch-size 64 --hidden-dim 64 --latent-dim 64 --patch-size 4 --winfree-steps 8 --winfree-coarse-grid-size 2 --winfree-global-phase-control shuffle --artifact-every 10 --checkpoint-every 10"
```

Run the full two-seed coarse-mesh probe matrix on Modal:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_coarse_phase_mesh
```

This launches 2x2 and 4x4 coarse meshes, each with unshuffled and
phase-shuffled controls, for seeds 11 and 12. It writes a local comparison CSV
to `outputs/analysis/modal_block50_coarse_phase_mesh.csv`.

Run the coarse content-transport block-occlusion probe on Modal:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_coarse_rate_phase
```

This launches the 4x4 coarse rate-phase model for seeds 11 and 12 with four
controls: coarse content plus phase gate, content-shuffled control, content-only
without coarse phase gate, and gate-only without coarse content. It writes a
local comparison CSV to `outputs/analysis/modal_block50_coarse_rate_phase.csv`.

Dry-run the coarse predictive readout comparison before launching:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_coarse_predictive_readout \
  --print-only
```

This prints the capped request for feedforward, current one-node slow/global
Winfree, and 2x2/4x4 `winfree_coarse_predictive_rate_phase` variants on the
`image_plus_mask` block setup. Remove `--print-only` only when the workspace has
GPU headroom. Results will write to
`outputs/analysis/modal_block50_coarse_predictive_readout.csv`.

Dry-run the boundary-clamped inpainting comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_boundary_clamped_core \
  --print-only
```

This compares feedforward, one-node slow/global Winfree, and 4x4 coarse
predictive Winfree with `--corruption-protocol boundary_clamped`. In this
protocol, visible pixels are clamped for every model and the primary loss is
hidden-region MSE. It is not a full-image reconstruction score. Results will
write to `outputs/analysis/modal_block50_boundary_clamped_core.csv`.

Run the prior + Winfree residual refinement attribution probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_prior_refinement
```

This compares the feedforward patch prior against
`winfree_prior_refinement` at residual strengths `0.25` and `0.5`, for seeds
11 and 12, under the same boundary-clamped block-occlusion protocol. Results
write to `outputs/analysis/modal_block50_prior_refinement.csv`.

Run checkpoint-based mask-stress robustness evaluation from an existing sweep
JSON:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --robustness-from-json outputs/analysis/modal_block50_prior_refinement.json \
  --robustness-preset mask_stress \
  --robustness-include-regex 'feedforward|s050' \
  --robustness-csv outputs/analysis/modal_prior_refinement_mask_stress.csv
```

This reuses saved best checkpoints instead of retraining. The `mask_stress`
preset evaluates `block25`, `block50`, `block75`, `patch50`, and `patch75`
under the boundary-clamped hidden-region protocol.

Run anytime-settling checkpoint evaluation:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --settling-from-json outputs/analysis/modal_block50_prior_refinement.json \
  --settling-include-regex s050 \
  --settling-steps 1,2,4,8,16,32 \
  --settling-scenarios block50,patch75 \
  --settling-csv outputs/analysis/modal_prior_refinement_anytime_settling.csv
```

This reuses saved `winfree_prior_refinement` checkpoints and overrides the
static Winfree scan length during eval. It tests whether more recurrent
settling improves a trained refiner without retraining.

Run the recurrent-conv residual refinement control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --sweep-preset block50_recurrent_prior_refinement
```

This compares the feedforward patch prior against a matched
`recurrent_conv_prior_refinement` residual branch at strength `0.5`, for seeds
11 and 12, under the same boundary-clamped block-occlusion protocol as the
Winfree prior-refinement probe. Results write to
`outputs/analysis/modal_block50_recurrent_prior_refinement.csv`.

Run mask-stress robustness evaluation for that recurrent residual control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist.py \
  --robustness-from-json outputs/analysis/modal_block50_recurrent_prior_refinement.json \
  --robustness-preset mask_stress \
  --robustness-include-regex recurrent_prior_refine_s050 \
  --robustness-csv outputs/analysis/modal_recurrent_prior_refinement_mask_stress.csv
```

This reuses saved best checkpoints from the recurrent residual sweep and
evaluates `block25`, `block50`, `block75`, `patch50`, and `patch75` under the
boundary-clamped hidden-region protocol.

Run the mask-aware block-occlusion core comparison:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_mask_aware_core
```

This launches feedforward, recurrent-conv, local Winfree rate-phase, and
slow/global Winfree rate-phase for seeds 11 and 12. It keeps the same 50% block
occlusion setup but trains with changed pixels weighted by `2.0` and unchanged
pixels weighted by `0.25`. It writes a local comparison CSV to
`outputs/analysis/modal_block50_mask_aware_core.csv`.

Run the lighter mask-loss weight sweep:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_mask_weight_sweep
```

This launches only feedforward and slow/global Winfree rate-phase for seeds 11
and 12, with two lighter mask-aware objectives: `changed=1.5, visible=0.5` and
`changed=1.25, visible=0.75`. It writes a local comparison CSV to
`outputs/analysis/modal_block50_mask_weight_sweep.csv`.

Run the block-occlusion missing-marker core comparison:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_missing_marker_core
```

This launches feedforward, recurrent-conv, local Winfree rate-phase, and
slow/global Winfree rate-phase for seeds 11 and 12. It keeps 50% block
occlusion but marks occluded pixels with `-1.0` instead of `0.0`, so the model
can distinguish missing evidence from real black MNIST background. It writes a
local comparison CSV to
`outputs/analysis/modal_block50_missing_marker_core.csv`.

Run the block-occlusion image-plus-mask core comparison:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_image_plus_mask_core
```

This launches the same four core models as the missing-marker comparison, but
keeps occluded image pixels at `0.0` and adds a second flat input channel with
`1.0` for visible pixels and `0.0` for missing pixels. The model still predicts
the one-channel clean image. It writes a local comparison CSV to
`outputs/analysis/modal_block50_image_plus_mask_core.csv`.

Run the visibility-gated Winfree comparison:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_visibility_gated_winfree
```

This keeps the `image_plus_mask` block setup but tests whether using the
visibility channel inside the Winfree recurrent rate update helps. It launches
feedforward, ungated local/global Winfree, visibility-gated local/global
Winfree, and shuffled-visibility controls for seeds 11 and 12. It writes a
local comparison CSV to
`outputs/analysis/modal_block50_visibility_gated_winfree.csv`.

Run the ConvLSTM control comparison:

```bash
modal run scripts/modal_mnist.py \
  --sweep-preset block50_conv_lstm_control
```

This keeps the `image_plus_mask` block setup and compares feedforward,
recurrent-conv, ConvLSTM, and the best current slow/global Winfree rate-phase
model for seeds 11 and 12. It writes a local comparison CSV to
`outputs/analysis/modal_block50_conv_lstm_control.csv`.

Run the JEPA-lite hidden-representation comparison:

```bash
modal run scripts/modal_mnist_jepa.py \
  --sweep-preset block50_jepa_core
```

This predicts low-frequency DCT patch embeddings for hidden block-occlusion
patches rather than reconstructing pixels. It compares feedforward,
recurrent-conv, ConvLSTM, local Winfree rate-phase, and slow/global Winfree
rate-phase for seeds 11 and 12. It writes a local comparison CSV to
`outputs/analysis/modal_block50_jepa_core.csv`.

Run the Un-0-style implicit generator comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_core
```

This compares learned Kuramoto dynamics, frozen Kuramoto reservoir dynamics,
and decoder-only generation for seeds 11 and 12. All variants sample from
random oscillator phase/noise and train with the same unpaired distributional
MNIST objective. It writes a local comparison CSV to
`outputs/analysis/modal_mnist_generator_core.csv`.

Generator sweep CSVs include the usual quality metrics plus the
`success_diagnostics` scorecard: total parameters, decoder parameter fraction,
trainable recurrent parameter fraction, estimated recurrent operation fraction,
sample throughput, and phase-trajectory movement/order proxies. These are
digital-simulation diagnostics for attribution and efficiency comparison, not
physical energy measurements.

Conditional generator runs also write `attractor_robustness` diagnostics by
default. These sample repeated initial states for each class label and measure
label consistency, within-class diversity, and class separation. Use:

```bash
--attractor-variants-per-class 4
```

Set the value to `0` to skip this diagnostic on very fast smoke runs.

For generator runs where visual quality or class conditioning matters, add a
small frozen classifier quality pass:

```bash
--quality-classifier-epochs 5
```

This trains a lightweight MNIST classifier on the run's training split and adds
`classifier_label_accuracy`, `classifier_label_confidence`,
`classifier_max_confidence`, and `classifier_entropy` to the summary/CSV. Use
this when objective loss and visual sample quality disagree.

For already-finished Modal runs, you can pull sample `.npz` artifacts and score
them locally with the same classifier metric. Example artifact path:

```bash
modal volume get oscnet-runs \
  /mnist_generator/<run-name>/artifacts/mnist_generator_samples_epoch_020.npz \
  outputs/analysis/<name>.npz
```

The generator launcher defaults to one Modal container. Raise
`OSCNET_MODAL_MAX_CONTAINERS` only when you intentionally want several GPUs.
The current workspace can use up to eight GPU containers for deliberate sweeps;
use that for frontier probes where parallelism saves real wall-clock time.

Run the simple conditional generator comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_conditional_core
```

This adds label phase shifts plus class/prototype distribution terms to the
same learned Kuramoto, frozen Kuramoto, and decoder-only controls. Results
write to `outputs/analysis/modal_mnist_generator_conditional_core.csv`.

Run the closer Un-0 conditioning probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_un0_coupled_core
```

This compares learned class-coupled Kuramoto dynamics, frozen class-coupled
reservoir dynamics, and a decoder-only phase-shift label control. The
class-coupled variants use separate conditioning oscillators with
label-specific unidirectional coupling into the main oscillator pool and
reference-relative phase readout. Results write to
`outputs/analysis/modal_mnist_generator_un0_coupled_core.csv`.

Run the low-decoder spatial readout probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_spatial_readout_core
```

This keeps the closer Un-0 conditioning setup but replaces the MLP decoder with
`--decoder-mode spatial_basis`, a fixed Gaussian basis readout with only
trainable sin/cos oscillator weights and one output bias. It is a deliberately
hard attribution probe: if it works, image structure must be carried mostly by
the oscillator field rather than by a conventional decoder. Results write to
`outputs/analysis/modal_mnist_generator_spatial_readout_core.csv`.

Run the structured local-basis readout probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_local_readout_core
```

This keeps the closer Un-0 conditioning setup and uses
`--decoder-mode local_basis`, where each oscillator writes trainable local patch
weights through fixed Gaussian patch bases. It is less starved than
`spatial_basis`, but still keeps the decoder small enough for attribution: the
trained dynamics must beat frozen dynamics and decoder-only controls to claim
value. Results write to
`outputs/analysis/modal_mnist_generator_local_readout_core.csv`.

Run the spatial coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_spatial_coupling_core
```

This reuses the local-basis readout and class-coupled conditioning, but sets
`--coupling-profile distance_decay`, `--coupling-length-scale 0.35`,
`--coupling-floor 0.05`, and `--coupling-bias-strength 0.05`. The hypothesis is
that whole-digit composition needs a spatially biased phase field: mostly local
coordination, with weak long-range communication, without increasing decoder
capacity. Results write to
`outputs/analysis/modal_mnist_generator_spatial_coupling_core.csv`.

Run the trainability attribution probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_trainability_attribution_core
```

This reuses the dense local-basis generator and separates trainability of the
main recurrent oscillator field from the class-conditioning oscillator drive.
It compares all-trained, conditioning-only, recurrent-only, and fully frozen
variants for seeds 11 and 12. Use this before adding another architecture
feature: it answers whether the local-basis win comes from learned main
coupling, learned class-conditioning, or their interaction. Results write to
`outputs/analysis/modal_mnist_generator_trainability_attribution_core.csv`.

Run the unconditional local-basis generator probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_unconditional_local_readout_core
```

This removes class labels, class moments, and prototype losses while keeping the
low-capacity `local_basis` renderer. It compares learned recurrent Kuramoto,
frozen reservoir, and decoder-only controls for seeds 11 and 12. This is the
cleanest test of whether the main recurrent phase field helps generation when
class conditioning cannot carry the result. Results write to
`outputs/analysis/modal_mnist_generator_unconditional_local_readout_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_unconditional_local_readout_core \
  --print-only
```

Run the source-faithful resize-conv generator probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_core
```

This keeps the class-coupled Kuramoto conditioning path but replaces the
low-capacity basis renderer with a Un-0-inspired spatial resize-conv decoder:
`196` oscillators form an 8-channel `7x7` phase-feature seed, then two
nearest-neighbor upsample/convolution blocks render `28x28` MNIST samples. It
compares learned recurrent dynamics, frozen reservoir dynamics, and
decoder-only controls for seeds 11 and 12. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_core \
  --print-only
```

Run the resize-conv generator with the Un-0-inspired pixel-drift objective:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_core
```

This keeps the same learned Kuramoto, frozen-reservoir, and decoder-only
controls, but switches the objective from distributional SWD/moment matching to
`loss_mode="pixel_drift"`. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_core \
  --print-only
```

Run the resize-conv generator with queue-backed pixel drift:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_core
```

This is the next source-faithful Un-0 port after pixel drift. It keeps the same
controls, but draws same-class positives from a host-side per-class FIFO memory
using `--drift-queue-size 512 --drift-queue-num-pos 32`. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_core \
  --print-only
```

Run the HORN-vs-Kuramoto resize-conv generator probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_resize_conv_core
```

This compares trainable HORN, frozen HORN, HORN decoder-only, trainable
Kuramoto, frozen Kuramoto, and Kuramoto decoder-only under the same conditional
resize-conv pixel-drift setup. It is the cleanest immediate response to
reports that homogeneous HORN dynamics outperform Kuramoto on image generation.
Results write to
`outputs/analysis/modal_mnist_generator_horn_resize_conv_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_resize_conv_core \
  --print-only
```

Run the lightweight HORN conditioning-attribution probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_conditioning_attribution_probe
```

This keeps the HORN resize-conv generator setup fixed and varies label-phase
conditioning strength while comparing trainable HORN, frozen HORN, and
HORN decoder-only. It also enables `--quality-classifier-epochs 5`, so the CSV
captures classifier-based semantic quality alongside pixel-drift loss. Results
write to
`outputs/analysis/modal_mnist_generator_horn_conditioning_attribution_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_conditioning_attribution_probe \
  --print-only
```

Run the zero-label HORN replication probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_label0_replication_probe
```

This repeats the most interesting attribution condition from the first probe:
`label_phase_scale=0.0` with trainable HORN, frozen HORN, and HORN decoder-only.
It uses fresh seeds 12 and 13, keeps classifier-based semantic scoring enabled,
and writes results to
`outputs/analysis/modal_mnist_generator_horn_label0_replication_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_label0_replication_probe \
  --print-only
```

Run the matched non-oscillatory state-MLP label-zero control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_state_mlp_label0_control_probe
```

This uses the same label-zero, resize-conv, pixel-drift setup as the replicated
HORN result, but replaces HORN's oscillator update with a residual MLP
transition over the same position/velocity state. The default hidden size
(`--state-mlp-hidden-dim 48`) gives roughly the same recurrent parameter budget
as the HORN dense coupling for 196 oscillators. Results write to
`outputs/analysis/modal_mnist_generator_state_mlp_label0_control_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_state_mlp_label0_control_probe \
  --print-only
```

Run the low-data HORN-vs-state-MLP sample-efficiency probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_low_data_probe
```

This repeats the label-zero generator setup with only
`--train-limit 500`, comparing trainable HORN against the matched trainable
state-MLP control on seeds 11, 12, and 13. Use it to test whether HORN's
oscillator bias helps when the data budget is smaller. Results write to
`outputs/analysis/modal_mnist_generator_horn_state_mlp_low_data_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_low_data_probe \
  --print-only
```

Run the low-data HORN-vs-state-MLP settling-depth probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_settling_probe
```

This is a compact two-run diagnostic on seed 11. It trains the same low-data
HORN and state-MLP generators, then scores the trained checkpoint at test-time
settling depths `0,1,2,4,8,16,32`. Use it to ask whether extra recurrent
settling is actually useful, or whether the decoder already has the answer at
step zero. Results write to
`outputs/analysis/modal_mnist_generator_horn_state_mlp_settling_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_state_mlp_settling_probe \
  --print-only
```

Run the HORN finite-time stability training probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_settling_train_probe
```

This compares the low-data HORN baseline against the same model trained with
`--train-settling-steps 8,16,32`. Both runs are evaluated at
`0,1,2,4,8,16,32`, so the question is whether variable-depth training keeps the
step-16 generator quality while reducing the step-32 over-settling collapse.
Results write to
`outputs/analysis/modal_mnist_generator_horn_settling_train_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_settling_train_probe \
  --print-only
```

Run the HORN structured-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_structured_coupling_probe
```

This keeps the low-data, variable-depth HORN generator setup fixed and compares
two soft spatial distance-decay coupling profiles. It is the first
physical-plausibility check after the dense HORN result: can a local-biased
coupling field preserve the step-16/step-32 quality of dense all-to-all HORN?
Results write to
`outputs/analysis/modal_mnist_generator_horn_structured_coupling_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_structured_coupling_probe \
  --print-only
```

Run the HORN sparse local-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_sparse_coupling_probe
```

This keeps the low-data, variable-depth HORN generator setup fixed and compares
two true sparse `local_radius` masks. Unlike distance decay, long-range edges
outside the radius are exactly zero. Results write to
`outputs/analysis/modal_mnist_generator_horn_sparse_coupling_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_horn_sparse_coupling_probe \
  --print-only
```

Run the sparse-HORN vs state-MLP replication probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_replication_probe
```

This repeats the low-data, variable-depth setup across seeds `11`, `12`, and
`13`, comparing the current best sparse local HORN recipe
(`local_radius`, radius `0.24`) against the matched trainable state-MLP
transition. Results write to
`outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_replication_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_replication_probe \
  --print-only
```

Run the sparse-HORN attribution/control probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_attribution_probe
```

This uses the local `sparse_horn_mnist*` preset family and repeats seeds `11`,
`12`, and `13` for sparse HORN, frozen HORN, HORN decoder-only, one-step HORN,
state-MLP, frozen state-MLP, and state-MLP decoder-only. It is the next
cleaner attribution sweep for the current best generator recipe. Results write
to `outputs/analysis/modal_mnist_generator_sparse_horn_attribution_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_attribution_probe \
  --print-only
```

Run the sparse-HORN conditioning-route probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_conditioning_route_probe
```

This compact anti-shortcut probe compares the current `phase_shift` HORN recipe
against `class_oscillator` and `class_coupling` variants. The important
question is whether classifier accuracy still rises through settling when
step 0 has no direct class label shift. Results write to
`outputs/analysis/modal_mnist_generator_sparse_horn_conditioning_route_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_conditioning_route_probe \
  --print-only
```

Run the sparse-HORN class-coupling sharpen probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_sharpen_probe
```

This compares the older no-direct-label `class_coupling` preset against longer
settling, stronger dynamic class drive, and a small explicit anchor. Results
write to
`outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_sharpen_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_sharpen_probe \
  --print-only
```

Run the sparse-HORN strong class-coupling control probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strong_control_probe
```

This is the stricter three-seed audit for the no-direct-label generator route.
It compares trainable HORN against frozen HORN, decoder-only HORN, and a
matched `StateMLPImageGenerator` that receives the same dynamic class drive.
Results write to
`outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strong_control_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strong_control_probe \
  --print-only
```

Run the sparse-HORN class-drive strength probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strength_probe
```

This keeps the strict no-direct-label `class_coupling` HORN recipe fixed and
varies `conditioning_strength` across `2.0`, `4.0`, and `8.0` for seeds
`11/12/13`. Results write to
`outputs/analysis/modal_mnist_generator_sparse_horn_class_coupling_strength_probe.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_class_coupling_strength_probe \
  --print-only
```

Run the resize-conv generator with Un-0-style dynamic conditioning oscillators:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core
```

This keeps the strongest current MNIST generator objective fixed
(`pixel_drift` with queue-backed same-class positives) and changes only the
conditioning/readout pair: `conditioning_mode="class_oscillator"` plus
`readout_mode="mean_relative"`. It compares learned Kuramoto,
frozen-reservoir, and decoder-only controls. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_un0_condition_core \
  --print-only
```

Run the queue-backed pixel-drift generator with light distributional
regularization:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_distributional_core
```

This starts from the strongest current generator result,
`mnist_generator_resize_conv_pixel_drift_queue_core`, and adds a small
distributional loss term at weights `0.005` and `0.01`. The goal is to test
whether whole-sample quality and pixel statistics improve without sacrificing
the learned-Kuramoto class-alignment gap over frozen/decoder controls. Results
write to
`outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_distributional_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_distributional_core \
  --print-only
```

Run the resize-conv generator with fixed structural feature drift:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_feature_drift_core
```

This keeps the same attribution controls but trains with
`loss_mode="pixel_feature_drift"`: half-weight pixel drift plus a fixed MNIST
structural feature drift target. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_feature_drift_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_feature_drift_core \
  --print-only
```

Run the resize-conv generator with learned MNIST feature drift:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_core
```

This trains a small MNIST feature classifier first, freezes its penultimate
features, then compares learned Kuramoto, frozen-reservoir, and decoder-only
controls under `loss_mode="pixel_feature_drift"` with
`feature_drift_mode="learned"`. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_core \
  --print-only
```

Run the resize-conv generator with queue-backed learned feature drift:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_queue_core
```

This combines the two source-faithful Un-0 ports: a per-class positive queue and
a frozen learned MNIST feature target. It should be compared directly against
both `mnist_generator_resize_conv_learned_feature_drift_core` and
`mnist_generator_resize_conv_pixel_drift_queue_core`. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_learned_feature_drift_queue_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_learned_feature_drift_queue_core \
  --print-only
```

Outputs are written to the Modal Volume named `oscnet-runs` by default:

```text
/mnt/oscnet-runs/mnist/<run-name>/
/mnt/oscnet-runs/mnist_generator/<run-name>/
/mnt/oscnet-runs/mnist_phase_flow/<run-name>/
/mnt/oscnet-runs/mnist_phase_vae/<run-name>/
```

The MNIST IDX cache and JAX compilation cache are also kept on that volume.

## MNIST Phase VAE

Run the paired MNIST phase-VAE control sweep:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_vae.py \
  --sweep-preset mnist_phase_vae_core
```

This compares the same VAE scaffold with trainable Kuramoto phase dynamics,
frozen phase dynamics, and no dynamics. It is the cleanest generator sanity
check when the unpaired oscillator generator is underconstrained. Results write
to:

```text
outputs/analysis/modal_mnist_phase_vae_core.csv
outputs/analysis/modal_mnist_phase_vae_core.json
outputs/analysis/modal_mnist_phase_vae_samples/
```

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_vae.py \
  --sweep-preset mnist_phase_vae_core \
  --print-only
```

Run the forced/larger phase-dynamics probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_vae.py \
  --sweep-preset mnist_phase_vae_forced_dynamics_core
```

This keeps the same VAE scaffold but increases the oscillator pool to 128
phases, uses eight stronger recurrent steps, and switches to `mean_relative`
readout. It is a targeted test of whether more Un-0-like phase motion helps, not
a replacement for the baseline phase-VAE sweep. Results write to:

```text
outputs/analysis/modal_mnist_phase_vae_forced_dynamics_core.csv
outputs/analysis/modal_mnist_phase_vae_samples/
```

## Un-0 Reference Calibration

Run the upstream Un-0 reference generator in an isolated PyTorch Modal image:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_un0_reference.py \
  --pretrained cifar10/n1024 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --samples-per-class 4 \
  --seed 42
```

This does not import PyTorch into OscNet. It installs the Un-0 package from
`unconv-ai/Un-0` at commit `43f2587`, loads the released Hugging Face
checkpoint, writes a local PNG grid under `outputs/analysis/un0_reference/`,
and appends lightweight calibration metrics to
`outputs/analysis/un0_reference/un0_reference_runs.csv`.

Use this as a north-star sanity check before trying to reproduce Un-0 behavior
inside OscNet. It is not a training run and it does not compute FID.

For the stronger released CIFAR-10 checkpoint:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_un0_reference.py \
  --pretrained cifar10/n4096 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --samples-per-class 2 \
  --seed 42
```

To probe whether the released checkpoint needs oscillator evolution at
inference time, sweep the integration step count with the same seed and labels:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_un0_reference.py \
  --pretrained cifar10/n1024 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --samples-per-class 2 \
  --seed 42 \
  --step-sweep 0,1,2,5,10
```

This writes one PNG/JSON pair per step count and a sweep CSV under
`outputs/analysis/un0_reference/`. It reseeds before each step count so the
initial phase samples are comparable.

## MNIST Phase-Flow Sampler

Run the native phase-rate field sampler sweep:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_core
```

This compares a trainable oscillator field against a frozen-reservoir control
and a no-dynamics control under the same rectified-flow objective. It is the
current cleanest attribution test for whether oscillator dynamics help a
generative MNIST task.

Results write to:

```text
outputs/analysis/modal_mnist_phase_flow_core.csv
outputs/analysis/modal_mnist_phase_flow_core.json
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run only the matched recurrent-conv flow control, reusing the existing
phase-flow/frozen/no-dynamics CSV for comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_recurrent_conv_control
```

This writes:

```text
outputs/analysis/modal_mnist_phase_flow_recurrent_conv_control.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the coarse/global phase-carrier probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_global_probe
```

This writes:

```text
outputs/analysis/modal_mnist_phase_flow_coarse_global_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the same coarse/global model with a Heun predictor-corrector sampler and 32
sample steps:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_heun_probe
```

This writes:

```text
outputs/analysis/modal_mnist_phase_flow_coarse_heun_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the coarse/global model with fixed spatial phase-coordinate features:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_position_probe
```

This writes:

```text
outputs/analysis/modal_mnist_phase_flow_coarse_position_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the closure/binding probe for the coarse/global phase-flow model:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_closure_probe
```

This adds `--closure-loss-weight 1.0`, a train-time low-frequency endpoint
loss at `14x14` and `7x7`, and runs both the spatial-position and no-position
coarse/global variants. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_coarse_closure_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the Sobel-edge target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_edge_probe
```

This adds `--target-representation sobel_edges` and compares the coarse/global
phase-flow model against the matched recurrent-conv flow control. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_edge_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the signed-distance target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_probe
```

This adds `--target-representation signed_distance`, an approximate
JAX-native smooth shape-field target, and compares the coarse/global
phase-flow model against the matched recurrent-conv flow control. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the signed-distance flow-field target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_flow_probe
```

This adds `--target-representation signed_distance_flow`, a three-channel
potential-field target: signed distance plus normalized x/y gradient direction.
It compares the coarse/global phase-flow model against the matched
recurrent-conv flow control. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_flow_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the two-channel pixel/shape target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_pixel_shape_probe
```

This adds `--target-representation pixels_signed_distance`, where channel 0 is
pixel occupancy and channel 1 is the auxiliary signed-distance shape field. The
PNG artifacts show channel 0, so the run directly tests whether the shape field
improves pixel generation. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_pixel_shape_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the centered two-channel pixel/shape target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_centered_pixel_shape_probe
```

This adds `--target-representation centered_pixels_signed_distance`, training
the same two-channel visible field in centered `[-1, 1]` coordinates. Metrics
and PNG artifacts decode channel 0 back to pixel space. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_centered_pixel_shape_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the centered shape-gated readout probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_centered_shape_gated_probe
```

This uses `--target-representation centered_pixels_signed_distance` and
`--sample-readout-mode shape_gated`. The model still samples both channels,
but metric/PNG export multiplies the decoded pixel channel by the decoded
shape channel as a smooth gate. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_centered_shape_gated_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the shape-guided sampler probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_shape_guided_sampler_probe
```

This uses centered pixel/shape targets, `--sample-schedule shape_guided`, and
`--sample-readout-mode shape_gated`. During Euler sampling, the shape channel
updates first and pixel-channel updates open later. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_shape_guided_sampler_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the shape-to-pixel renderer comparison:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_core
```

This tests the two-stage renderer branch where channel 0 is pixel state and
channel 1 is a clamped signed-distance scaffold. The core sweep compares
`coarse_phase_flow`, local `phase_flow`, `phase_flow_no_dynamics`, and
`recurrent_conv_flow` for seeds `31` and `32`. It writes:

```text
outputs/analysis/modal_mnist_shape_pixel_core.csv
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_core \
  --print-only
```

Run the shape-to-pixel basin probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_basin_probe
```

This repeats the shape-to-pixel core sweep and adds
`--basin-t-values 0.1,0.25,0.5,0.75,0.9`, measuring whether the renderer
improves or damages partially real pixel states while the signed-distance shape
channel is clamped. It writes:

```text
outputs/analysis/modal_mnist_shape_pixel_basin_probe.csv
outputs/analysis/modal_mnist_shape_pixel_basin_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Run the shape-to-pixel scaffold robustness probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_shape_condition_probe
```

This repeats the shape-to-pixel core sweep and adds
`--shape-condition-t-values 0.1,0.5,0.9` with
`--shape-condition-noise-modes uniform,salt_pepper,zeros`, measuring whether the
renderer still produces useful pixels when the signed-distance scaffold is
imperfect rather than oracle-clean. It writes:

```text
outputs/analysis/modal_mnist_shape_pixel_shape_condition_probe.csv
outputs/analysis/modal_mnist_shape_pixel_shape_condition_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Run the shape-gated shape-to-pixel probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_shape_pixel.py \
  --sweep-preset mnist_shape_pixel_shape_gated_probe
```

This is the same scaffold robustness sweep, but adds
`--sample-readout-mode shape_gated` so the clamped signed-distance scaffold acts
as an explicit soft amplitude gate on the sampled pixel channel. It writes:

```text
outputs/analysis/modal_mnist_shape_pixel_shape_gated_probe.csv
outputs/analysis/modal_mnist_shape_pixel_shape_gated_probe.json
outputs/analysis/modal_mnist_shape_pixel_samples/
```

Run the locked multi-seed shape-gated audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_shape_gated_audit
```

This freezes the current best pixel-producing setup:
`centered_pixels_signed_distance`, `sample_readout_mode=shape_gated`,
`sample_schedule=standard`, and no closure loss. It runs seeds `31` through
`35` for `coarse_phase_flow`, `phase_flow`, `frozen_phase_flow`,
`phase_flow_no_dynamics`, and `recurrent_conv_flow`. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_shape_gated_audit.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the basin-of-attraction probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_basin_probe
```

This trains the same 20-epoch phase-flow setups, then evaluates endpoint
recovery from chord states `x_t = (1 - t) noise + t data` at
`t = 0.1, 0.25, 0.5, 0.75, 0.9`. It compares centered pixel/shape and
signed-distance targets for coarse phase-flow and recurrent-conv controls over
seeds `31` and `32`. The CSV includes both starting paired MSE and final paired
MSE for each basin start time, so positive deltas indicate dynamics that really
move the state closer to its scaffold target. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the signed-distance flow-field basin probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_flow_basin_probe
```

This runs the same basin diagnostic on `signed_distance_flow`, so we can test
whether adding explicit local flow direction widens or narrows the stable
shape-field basin versus plain signed distance. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_flow_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the signed-distance non-Gaussian basin probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_noise_basin_probe
```

This keeps the scalar signed-distance target and compares coarse/global
phase-flow against recurrent-conv under `uniform`, `salt_pepper`, and `zeros`
basin endpoints. The original Gaussian result remains in
`modal_mnist_phase_flow_basin_probe.csv`, so this sweep asks whether the
signed-distance attractor survives non-Gaussian starts without changing the
trained task. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_noise_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the signed-distance mixed-training basin probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_mixed_noise_basin_probe
```

This trains the scalar signed-distance target with `--train-noise-mode mixed`,
then evaluates the same `uniform`, `salt_pepper`, and `zeros` basin endpoints
for coarse/global phase-flow and recurrent-conv. It is a one-seed diagnostic:
use it to decide whether mixed endpoint training widens the basin before
scaling to a full two-seed sweep. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_mixed_noise_basin_probe.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the compact two-seed mixed-training basin probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_signed_distance_mixed_noise_basin_compact
```

This is the preferred follow-up to the one-seed diagnostic above. It trains
each seed/model once with `--train-noise-mode mixed`, then evaluates
`uniform`, `salt_pepper`, and `zeros` basin endpoints via
`--basin-noise-modes uniform,salt_pepper,zeros`. That gives a two-seed read
without retraining the same checkpoint for every basin endpoint. It writes:

```text
outputs/analysis/modal_mnist_phase_flow_signed_distance_mixed_noise_basin_compact.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Run the sparse HORN generator strict-control diversity probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_state_mlp_diversity_probe
```

This compares the strict no-direct-label HORN generator
`sparse_horn_mnist_class_coupling_strength8` against matched StateMLP
class-coupling controls, including distributional regularization variants. It
is the current recommended audit before claiming a HORN-specific
quality/diversity advantage. It writes:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_diversity_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_state_mlp_diversity_probe.json
```

Run the sparse HORN generator distributional quality probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_distributional_probe
```

This keeps the strict no-direct-label HORN generator at
`conditioning_strength=8.0` and tests small distributional weights for better
pixel statistics/proximity without giving up the settling route. It writes:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_distributional_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_distributional_probe.json
```

Run the sparse HORN generator stronger-evaluator audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_quality_classifier_audit
```

This reruns the leading strict HORN variants and the matched StateMLP control
with a stronger generated-label evaluator. The generator still trains on 500
examples, but the quality classifier trains on 5000 examples for 10 epochs. Use
this audit when the generated-label classifier metric matters. It writes:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_quality_classifier_audit.csv
outputs/analysis/modal_mnist_generator_sparse_horn_quality_classifier_audit.json
```

Run the sparse HORN generator dynamics-quality probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_sparse_horn_dynamics_quality_probe
```

This keeps the stronger generated-label evaluator and compares strict HORN
strength8 against small distributional pressure, higher frequency, higher
damping, and a frequency-plus-distributional variant. Use it when testing
whether quality/proximity gains come from oscillator dynamics rather than only
from the loss. It writes:

```text
outputs/analysis/modal_mnist_generator_sparse_horn_dynamics_quality_probe.csv
outputs/analysis/modal_mnist_generator_sparse_horn_dynamics_quality_probe.json
```

Run the CIFAR-10 grayscale generator frontier gate:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_gray_frontier_probe
```

This is the first natural-image scale check for the sparse HORN generator
frontier. It compares strict HORN, calibrated HORN `dist005`, and a matched
StateMLP strength-8 control on 32x32 grayscale CIFAR-10 using train1000/20e and
a stronger generated-label evaluator. Use up to eight containers when the
workspace GPU budget allows it. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_gray_frontier_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_gray_frontier_probe.json
```

Run the same CIFAR-10 grayscale gate with the convolutional quality judge:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_gray_convjudge_frontier_probe
```

This keeps the generator recipes fixed and only changes the generated-label
evaluator from the flat MLP judge to `--quality-classifier-kind conv`. Use this
when CIFAR-gray semantic metrics matter. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_gray_convjudge_frontier_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_gray_convjudge_frontier_probe.json
```

Run the CIFAR-10 RGB generator frontier gate:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_frontier_probe
```

This keeps the CIFAR-gray frontier design but loads channel-first RGB images,
uses `image_shape=(32, 32, 3)`, and compares recommended HORN, HORN `dist005`,
and the matched StateMLP strength-8 control with a convolutional quality
judge. It is the current hard gate for whether the sparse HORN
semantic/diversity frontier survives full color. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_frontier_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_frontier_probe.json
```

Run the one-seed CIFAR-10 RGB feature-metric audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feature_metric_audit
```

This reruns the three RGB frontier variants for seed 11 only. It exists to
populate classifier feature-space diversity, nearest-real metrics, and
trajectory-level settling diagnostics after those measurements change, without
spending a full 9-job sweep. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feature_metric_audit.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_feature_metric_audit.json
```

The companion analyzer command is:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feature_metric_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_feature_metric_dynamics_audit \
  --title "CIFAR-10 RGB feature/dynamics generator audit" \
  --accuracy-floor 0.3
```

Run the CIFAR-10 RGB quality-judge audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_judge_audit
```

This compares the old `conv` sample-quality judge against the stronger
`residual_conv` judge on the same coupled HORN 25% prefix-drive run and the
no-main 25% prefix-drive control. It is a measurement audit: use it before
making strong CIFAR semantic-generation claims from generated-label accuracy.
It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_judge_audit.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_judge_audit.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_judge_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_judge_audit \
  --title "CIFAR-10 RGB quality-judge audit" \
  --accuracy-floor 0.0
```

Run the CIFAR-10 RGB residual feature-drift semantic probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_semantic_feature_drift_probe
```

This compares the HORN prefix-25 sparse-drive pixel-drift baseline against
residual-conv learned feature drift at weights `0.25` and `1.0`, while scoring
with an independently trained residual-conv quality judge. It tests whether a
semantic feature-drift target can improve strict CIFAR sample quality without
making the evaluation circular. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_semantic_feature_drift_probe \
  --title "CIFAR-10 RGB residual feature-drift semantic probe" \
  --accuracy-floor 0.0
```

Run the CIFAR-10 RGB residual feature-drift attribution repeat:

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_semantic_feature_drift_attribution
```

This repeats the residual feature-drift `0.25` setting against the matching
no-main-interaction control over two seeds. It is the current guardrail for
separating "better semantic objective" from "better coupled oscillator
substrate." It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_attribution.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_attribution.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_semantic_feature_drift_attribution.csv \
  --output-dir outputs/analysis/cifar10_rgb_semantic_feature_drift_attribution \
  --title "CIFAR-10 RGB residual feature-drift attribution" \
  --accuracy-floor 0.0
```

Run the compact CIFAR-10 RGB attractor robustness probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attractor_robustness_probe
```

This compares the residual feature-drift `0.25` coupled HORN recipe against
the matching no-main HORN control and StateMLP control. It uses the
residual-conv quality judge and `--attractor-variants-per-class 8` to check
whether same-label initial-state perturbations remain class-consistent without
collapsing to one prototype per class. The analyzer reports a collapse-aware
`Basin score`, computed as attractor label accuracy times `log1p` same-class
pixel spread. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_attractor_robustness_probe \
  --title "CIFAR-10 RGB attractor robustness probe" \
  --accuracy-floor 0.0
```

Run the two-seed CIFAR-10 RGB attractor robustness repeat:

```bash
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat
```

This runs the same coupled HORN, no-main HORN, and StateMLP residual
feature-drift variants for seeds `11` and `23`. Use it when changing the HORN
generator or feature-drift objective and checking whether the basin-score
advantage survives more than one seed. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_attractor_robustness_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_attractor_robustness_seed_repeat \
  --title "CIFAR-10 RGB attractor robustness seed repeat" \
  --accuracy-floor 0.0
```

Run the one-seed CIFAR-10 RGB HORN attribution probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_attribution_probe
```

This compares the RGB HORN recipe against one-step, frozen-recurrent,
frozen-conditioning, no-main-interaction, and decoder-only controls. It is the
current gate for whether the HORN result depends on learned recurrent coupling
or mainly on learned class drive plus settling. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_attribution_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_attribution_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_attribution_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_attribution_probe \
  --title "CIFAR-10 RGB HORN attribution probe" \
  --accuracy-floor 0.3
```

Run the CIFAR-10 RGB sparse class-drive HORN probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_sparse_drive_probe
```

This compares full, 25%, and 10% direct class-drive targets for the coupled
HORN recipe and the no-main-interaction control. It tests whether local HORN
coupling propagates sparse class drive instead of letting conditioning organize
every oscillator independently. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_sparse_drive_probe \
  --title "CIFAR-10 RGB sparse class-drive HORN probe" \
  --accuracy-floor 0.3
```

Run the focused 25% sparse-drive seed repeat:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_sparse_drive_seed_repeat
```

This compares only the coupled HORN 25% drive recipe against the no-main
25% drive control across seeds `7, 11, 23, 37`. It is the current credibility
gate for whether sparse local HORN coupling improves class-consistent CIFAR RGB
generation beyond a single seed. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_seed_repeat.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_seed_repeat.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_sparse_drive_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_sparse_drive_seed_repeat \
  --title "CIFAR-10 RGB 25% sparse-drive seed repeat" \
  --accuracy-floor 0.3
```

Run the structured sparse-drive topology probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_structured_drive_probe
```

This compares `prefix` vs `spatial_grid` 25% direct class-drive targets for
coupled HORN and no-main controls on seeds `11, 23`. It tests whether
distributed label anchors improve the sparse-drive coupling result or whether
a contiguous driven region is better. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_structured_drive_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_structured_drive_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_structured_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_structured_drive_probe \
  --title "CIFAR-10 RGB structured sparse-drive probe" \
  --accuracy-floor 0.3
```

Run the coherent sparse-drive topology probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coherent_drive_probe
```

This compares `prefix` vs `center_block` 25% direct class-drive targets for
coupled HORN and no-main controls on seeds `11, 23`. It tests whether the
useful sparse-drive signal prefers a coherent local source region and whether
moving that source from the top-band/prefix region to the center improves
semantic quality or mainly changes diversity. It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coherent_drive_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_coherent_drive_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coherent_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coherent_drive_probe \
  --title "CIFAR-10 RGB coherent sparse-drive probe" \
  --accuracy-floor 0.3
```

To rerun the full four-way attribution matrix in one request:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_conv_control_core
```

This writes:

```text
outputs/analysis/modal_mnist_phase_flow_conv_control_core.csv
outputs/analysis/modal_mnist_phase_flow_samples/
```

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_core \
  --print-only
```

## Configuration

Set environment variables before `modal run` to change the remote worker:

```bash
OSCNET_MODAL_GPU=A100 modal run scripts/modal_mnist.py
OSCNET_MODAL_JAX='jax[cuda12]' modal run scripts/modal_mnist.py
OSCNET_MODAL_VOLUME=oscnet-runs modal run scripts/modal_mnist.py
OSCNET_MODAL_TIMEOUT_SECONDS=21600 modal run scripts/modal_mnist.py
OSCNET_MODAL_MAX_CONTAINERS=2 modal run scripts/modal_mnist.py --sweep-preset block50_conv_lstm_control
```

Local CPU/GPU execution remains the regular experiment command:

```bash
python examples/image_mnist_oscillatory_autoencoder.py --help
```
