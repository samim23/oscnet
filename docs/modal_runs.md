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

Generator coupling note: `--coupling-strength` now scales the
class/conditioning drive, while `--main-coupling-strength` optionally scales
recurrent oscillator interaction. If `--main-coupling-strength` is omitted it
defaults to `--coupling-strength`, preserving old sweep behavior. Use
`--coupling-strength 1.0 --main-coupling-strength 0.0` for a clean "class drive
on, recurrent main coupling off" probe.

Run the compact main-coupling strength probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_strength_probe
```

This holds the 25% sparse class drive fixed and sweeps recurrent HORN coupling
strength `0.0, 0.25, 0.5, 1.0` on seed `11`, with the strict residual-conv
feature drift/judge setup. Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_strength_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_strength_probe \
  --title "CIFAR-10 RGB main coupling strength probe" \
  --accuracy-floor 0.3
```

If `main=0.0` looks surprisingly strong, repeat the core comparison across
seeds and include the matched StateMLP control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_strength_seed_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_strength_seed_repeat \
  --title "CIFAR-10 RGB main coupling strength seed repeat" \
  --accuracy-floor 0.3
```

Fine-sweep the moderate recurrent-coupling region:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_fine_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_fine_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_fine_probe \
  --title "CIFAR-10 RGB main coupling fine probe" \
  --accuracy-floor 0.3
```

Run a clean current-code replication of all recurrent-coupling strengths:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_main_coupling_current_replication
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_main_coupling_current_replication.csv \
  --output-dir outputs/analysis/cifar10_rgb_main_coupling_current_replication \
  --title "CIFAR-10 RGB main coupling current-code replication" \
  --accuracy-floor 0.3
```

Use this current-code replication for coupling conclusions; the older one-off
and fine sweeps are exploratory because repeated launches showed noticeable
run-to-run variance even with matching seed/arguments.

Run the normalized distance-decay recurrent-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_distance_decay_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_distance_decay_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_distance_decay_probe \
  --title "CIFAR-10 RGB normalized distance-decay coupling probe" \
  --accuracy-floor 0.3
```

This probe keeps the strong sparse class drive fixed, uses
`--coupling-profile distance_decay --coupling-normalization row_sum`, and sweeps
`main_coupling_strength=0.25,0.5,1.0`. Row-sum normalization scales each
non-empty recurrent profile row to `num_oscillators`, matching the generator
step convention that divides the summed interaction by `N`.

Run the sparse normalized local-radius recurrent-coupling probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_local_radius_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_local_radius_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_local_radius_probe \
  --title "CIFAR-10 RGB normalized local-radius coupling probe" \
  --accuracy-floor 0.3
```

This uses the same row-sum gain normalization but keeps the recurrent topology
sparse via `--coupling-profile local_radius`. Compare it against both the
current local-radius replication and dense normalized distance-decay before
changing any defaults.

Sweep the sparse normalized local radius around the current winner:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_normalized_local_radius_sweep
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_normalized_local_radius_sweep.csv \
  --output-dir outputs/analysis/cifar10_rgb_normalized_local_radius_sweep \
  --title "CIFAR-10 RGB normalized local-radius sweep" \
  --accuracy-floor 0.3
```

This holds `main_coupling_strength=1.0` and sweeps
`coupling_length_scale=0.16,0.24,0.32`, asking whether the winning normalized
local HORN medium wants a tight, current, or wider neighborhood before building
coarse-to-fine HORN.

Run the first coarse-to-fine HORN probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_probe \
  --title "CIFAR-10 RGB coarse-to-fine HORN probe" \
  --accuracy-floor 0.3
```

This compares the current normalized-local HORN preset with
`CoarseToFineHORNImageGenerator` at `coarse_to_fine_strength=0.0` and `1.0`.
The zero-strength variant is an attribution control: the coarse oscillator bank
is present, but it cannot drive the fine field.

If the first probe shows the coarse path improves proximity but tightens the
attractor basin, run the gentler gain sweep:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_gain_sweep.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_gain_sweep \
  --title "CIFAR-10 RGB coarse-to-fine HORN gain sweep" \
  --accuracy-floor 0.3
```

This keeps the coarse16 normalized-local recipe fixed and sweeps
`coarse_to_fine_strength=0.25,0.5,0.75`.

Run the coarse-to-fine projection-profile probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_profile_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_profile_probe \
  --title "CIFAR-10 RGB coarse-to-fine HORN profile probe" \
  --accuracy-floor 0.3
```

This holds `coarse_to_fine_strength=0.25` and compares dense,
distance-decayed, and local-radius coarse-to-fine projection profiles. It asks
whether spatially regularized top-down drive keeps more attractor diversity
than the dense learned coarse projection.

Run the compact coarse-to-fine dynamics audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=5 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_dynamics_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_dynamics_audit \
  --title "CIFAR-10 RGB coarse-to-fine HORN dynamics audit" \
  --accuracy-floor 0.3
```

This is a one-seed diagnostic rerun of the key rows: normalized-local HORN,
coarse bank with no fine drive, dense gentle coarse-to-fine,
distance-decayed gentle coarse-to-fine, and local-radius gentle coarse-to-fine.
Use it when code changes affect `success_diagnostics`, because it now surfaces
coarse state energy/update, coarse recurrent disagreement, and coarse-to-fine
disagreement in the frontier table.

If local-radius coarse-to-fine looks promising, repeat that lead over more
seeds:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_local_repeat \
  --title "CIFAR-10 RGB local coarse-to-fine HORN seed repeat" \
  --accuracy-floor 0.3
```

This runs seeds `11,23,37,41` for normalized-local HORN, a coarse-bank/no-drive
control, and the local-radius gentle coarse-to-fine variant.

After the frontier summary, run the paired same-seed attribution pass:

```bash
python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_local_repeat.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_local_repeat \
  --baseline-variant coarse16_c2f000 \
  --target-variant coarse16_c2f025_local050 \
  --target-variant horn_normlocal
```

This reports target-minus-baseline deltas on matched seeds, which is the
cleaner attribution view for active coarse-to-fine drive.

Run the C2F readout/objective conversion probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe \
  --title "CIFAR-10 RGB coarse-to-fine HORN conversion probe" \
  --accuracy-floor 0.3
```

```bash
python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_conversion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_conversion_probe \
  --title "CIFAR-10 RGB coarse-to-fine conversion paired deltas" \
  --pair coarse16_c2f000_base:coarse16_c2f025_local050_base \
  --pair coarse16_c2f000_ch32:coarse16_c2f025_local050_ch32 \
  --pair coarse16_c2f000_dist0025:coarse16_c2f025_local050_dist0025 \
  --pair coarse16_c2f000_ch32_dist0025:coarse16_c2f025_local050_ch32_dist0025
```

This keeps local-radius C2F fixed and tests whether a wider resize-conv
readout (`ch32`), small distributional pressure (`dist0025`), or both can
convert the better diversity/basin side into visible quality. Every active row
has a same-seed no-drive coarse control with the same readout/objective.

Run the compact C2F feedback probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe
```

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_feedback_probe \
  --title "CIFAR-10 RGB coarse-to-fine feedback probe" \
  --accuracy-floor 0 \
  --no-plot
```

```bash
python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_to_fine_feedback_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_to_fine_feedback_probe \
  --title "CIFAR-10 RGB coarse-to-fine feedback paired deltas" \
  --pair coarse16_c2f000_feedback050_base:coarse16_c2f025_local050_feedback050_base \
  --pair coarse16_c2f000_ch32_dist0025_feedback050:coarse16_c2f025_local050_ch32_dist0025_feedback050
```

Notes:

- `output_feedback_mode="state_proxy"` is the default for the feedback presets.
  It is cheap and feeds a centered local HORN state proxy back into
  acceleration.
- `output_feedback_mode="image"` is available for tiny diagnostics, but it
  decodes during every settling step. With `resize_conv` and multi-depth
  training it is much slower than ordinary HORN generation.

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

## CIFAR RGB Multiscale Layered HORN Probe

Two-seed probe for the `MultiscaleHORNImageGenerator` scaffold:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_layered_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.frontier.md
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_layered_probe.paired.md
outputs/analysis/multiscale_layered_grids/multiscale_layered_contact_sheet.png
```

Current read: no-vertical auxiliary HORN banks improved generated-label
accuracy over plain normalized-local HORN, but weak bidirectional vertical
coupling mainly improved nearest-real proximity and settling while reducing
class consistency and basin score.

## CIFAR RGB Multiscale Auxiliary Objective Probe

Test whether giving the coarsest auxiliary HORN layer its own low-resolution
image target makes vertical hierarchy useful:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_auxiliary_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_auxiliary_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_auxiliary_probe.json
outputs/analysis/cifar10_rgb_multiscale_auxiliary_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_auxiliary_probe/paired_deltas.md
```

Current read: the auxiliary low-res objective is learnable and improves some
proximity metrics. The best generated-label row in the two-seed probe is
`no_vertical_auxlow8`, which suggests the objective is mostly shaping shared
conditioning/readout rather than proving active vertical coupling. Active
vertical plus auxiliary gives the best nearest-real MSE but loses
semantic/diversity and basin quality.

## CIFAR RGB Multiscale Selective Vertical Gate Probe

Test whether coarse-to-fine hierarchy improves when vertical drive is routed
only into selected fine oscillators instead of all fine oscillators:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_gated_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gated_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gated_probe.json
outputs/analysis/cifar10_rgb_multiscale_gated_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_gated_probe/paired_deltas.md
```

Current read: `multiscale_vertical_target_gate="conditioning"` is much less
destructive than all-target vertical coupling and improves nearest-real MSE,
but it does not beat the no-vertical auxiliary model on the main
semantic/attractor frontier. `non_conditioning` preserves more diversity but
loses class consistency. Treat this as evidence for selective routing, not as
proof that this vertical mechanism is the missing hierarchy solution.

## CIFAR RGB Multiscale Gain-Modulated Vertical Probe

Test whether vertical hierarchy works better as modulation of fine-layer
dynamics than as additive source-minus-target acceleration:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_gain_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.json
outputs/analysis/cifar10_rgb_multiscale_gain_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_gain_probe/paired_deltas.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_gain_probe \
  --title "CIFAR-10 RGB multiscale gain-modulation frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_gain_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_gain_probe \
  --baseline-variant no_vertical_auxlow8 \
  --target-variant vgate_conditioning_auxlow8 \
  --target-variant gain_all_auxlow8 \
  --target-variant gain_conditioning_auxlow8 \
  --title "CIFAR-10 RGB multiscale gain-modulation paired deltas"
```

Current read: `multiscale_vertical_mode="gain_modulation"` is the best active
vertical hierarchy signal so far. The broad `gain_all_auxlow8` row beats the
no-vertical auxiliary baseline on generated-label accuracy, diversity, feature
diversity, attractor accuracy, and basin score on matched seeds. The cost is
worse nearest-real pixel MSE, worse output settling, and slower sampling.
Treat this as evidence that coarse/slow hierarchy should gate fine dynamics
rather than pull fine oscillator state directly.

## CIFAR RGB Multiscale Weak-Conditioning Probe

Test whether hierarchy becomes more valuable when direct class drive is
weakened. This lowers fine `conditioning_strength` from `8.0` to `2.0` and
auxiliary `multiscale_conditioning_strength` from `1.0` to `0.25`:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_weak_drive_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.json
outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe/paired_deltas.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe \
  --title "CIFAR-10 RGB multiscale weak-drive hierarchy frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_weak_drive_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_weak_drive_probe \
  --baseline-variant no_vertical_auxlow8_drive2 \
  --target-variant vgate_conditioning_auxlow8_drive2 \
  --target-variant gain_all_auxlow8_drive2 \
  --title "CIFAR-10 RGB multiscale weak-drive paired deltas"
```

Current read: weak conditioning changes the hierarchy story. The selective
additive gate `vgate_conditioning_auxlow8_drive2` is the stronger semantic and
basin row, while broad gain mainly improves diversity/feature diversity and
output settling. This points toward selective or signed gain, not simply more
broad gain.

## CIFAR RGB Multiscale Signed / Selective Gain Probe

Test signed excitatory/inhibitory vertical modulation and selective
class-column gain:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_signed_gain_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.json
outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe/paired_deltas.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe \
  --title "CIFAR-10 RGB multiscale signed/selective gain frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_signed_gain_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_signed_gain_probe \
  --baseline-variant gain_conditioning_auxlow8 \
  --target-variant signed_gain_all_auxlow8 \
  --target-variant signed_gain_conditioning_auxlow8 \
  --target-variant gain_conditioning_auxlow8_drive2 \
  --target-variant signed_gain_all_auxlow8_drive2 \
  --target-variant signed_gain_conditioning_auxlow8_drive2 \
  --title "CIFAR-10 RGB multiscale signed/selective gain paired deltas"
```

Current read: signed modulation is useful and stable. Broad signed gain
improves attractor/basin metrics; selective signed gain is the better
accuracy/proximity compromise and is the strongest weak-drive result so far.
Use this family before adding more hierarchy depth.

## CIFAR RGB Multiscale Soft Selective Gain Probe

Test whether a weak non-target gain floor can combine selective-gain proximity
with broad-gain attractor strength:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multiscale_soft_gate_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_soft_gate_probe.csv
outputs/analysis/cifar10_rgb_multiscale_soft_gate_probe/frontier_summary.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_multiscale_soft_gate_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_multiscale_soft_gate_probe \
  --title "CIFAR-10 RGB multiscale soft-gate frontier" \
  --accuracy-floor 0 \
  --no-plot
```

Current read: a `0.25` soft floor helps unsigned selective gain slightly, but
hurts selective signed gain's class consistency and attractor basin. Treat it
as a useful diagnostic, not the new default.

## CIFAR RGB Vertical Causality Audit

Audit whether the trained vertical hierarchy path is causal at sample time by
zeroing, shuffling, flipping, and scaling the vertical signal while keeping the
same initial states:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_causality_audit
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_causality_audit.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_causality_audit.json
outputs/analysis/cifar10_rgb_vertical_causality_audit/frontier_summary.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_causality_audit.csv \
  --output-dir outputs/analysis/cifar10_rgb_vertical_causality_audit \
  --title "CIFAR-10 RGB vertical causality audit frontier" \
  --accuracy-floor 0 \
  --no-plot
```

Current read: the current vertical-gain path is nearly silent at sample time.
Interventions move outputs by roughly `1e-9` MSE and leave class/attractor
metrics effectively unchanged, while traced vertical gain has only about
`1e-4` standard deviation around `1.0`. Do not treat the current vertical
variants as causal hierarchy wins until the route has a calibrated, measurable
intervention effect.

## CIFAR RGB Vertical Calibration Probe

Test whether increasing vertical signal scale makes the top-down route
measurably causal without immediately collapsing samples:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_calibration_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.json
outputs/analysis/cifar10_rgb_vertical_calibration_probe/frontier_summary.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_calibration_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_vertical_calibration_probe \
  --title "CIFAR-10 RGB vertical calibration frontier" \
  --accuracy-floor 0 \
  --no-plot
```

Current read: calibration works diagnostically. `vscale10` and `vscale30`
increase traced gain variation and intervention output deltas from near-zero to
measurable values. It does not prove better image generation; some
interventions still improve basin metrics, so the route is causal but not
always beneficial.

## CIFAR RGB Dual-Gain Probe

Test the dual-route hierarchy mechanism: broad positive gain into all fine
columns plus selective signed gain into the class-conditioned columns.

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_dual_gain_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.json
outputs/analysis/cifar10_rgb_dual_gain_probe/frontier_summary.md
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_dual_gain_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_dual_gain_probe \
  --title "CIFAR-10 RGB dual-gain frontier" \
  --accuracy-floor 0 \
  --no-plot
```

Current read: dual gain makes the vertical path clearly causal, especially at
`vscale30`, but it does not beat the current quality frontier. The no-vertical
auxiliary row and broad-gain-only row remain better on the main semantic and
attractor metrics. Treat dual gain as useful infrastructure and a causality
diagnostic, not as the new default recipe.

## CIFAR RGB Coarse Objective Probe

Test whether the auxiliary coarse layer should learn a paired low-resolution
copy (`mse`) or a distributional low-resolution class/batch target
(`distributional`). This is a hierarchy bottleneck test: if the distributional
coarse objective helps, the coarse layer may be useful as a class-level
attractor scaffold rather than a blurry thumbnail decoder.

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_objective_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_objective_probe \
  --title "CIFAR-10 RGB coarse objective frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_objective_probe \
  --variant-regex 'coarse_objective_(.+?)_n256' \
  --pair center_mse:center_dist \
  --pair no_vertical_mse:no_vertical_dist \
  --pair no_vertical_mse:center_mse \
  --pair no_vertical_dist:center_dist \
  --title "CIFAR-10 RGB coarse objective paired deltas"
```

This writes:

```text
outputs/analysis/cifar10_rgb_coarse_objective_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_coarse_objective_probe/paired_deltas.md
```

Current read: the distributional coarse objective is useful, but the win is not
specific to vertical hierarchy. It improves no-vertical accuracy, attractor
accuracy, feature-nearest distance, and basin score strongly. In the centered
signed-gain hierarchy, it improves diversity and basin score but worsens
nearest-real and feature-nearest proximity. The active vertical route is causal
under the distributional objective, with larger intervention output deltas than
paired MSE, but it is still not the quality frontier. Treat `*_auxdist8` as a
better coarse-objective diagnostic and keep working on a more content-specific
vertical route before adding deeper stacks.

Legacy single-baseline analysis command:

```bash
python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_objective_probe \
  --baseline-variant center_mse \
  --target-variant center_dist \
  --target-variant no_vertical_mse \
  --target-variant no_vertical_dist \
  --title "CIFAR-10 RGB coarse objective paired deltas"
```

## CIFAR RGB Feedback Signal Probe

Test whether bottom-up feedback should send only phase/position information
from fine to coarse layers, or whether the coarse layer benefits from seeing
bounded fine position-plus-velocity state. This keeps the active centered
signed-gain hierarchy fixed and changes only
`multiscale_feedback_signal_mode`.

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_signal_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_signal_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_signal_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_signal_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_signal_probe \
  --title "CIFAR-10 RGB feedback-signal frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_signal_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_signal_probe \
  --variant-regex 'feedback_signal_(.+?)_n256' \
  --pair mse_position:mse_state \
  --pair dist_position:dist_state \
  --pair mse_position:dist_position \
  --pair mse_state:dist_state \
  --title "CIFAR-10 RGB feedback-signal paired deltas"
```

This writes:

```text
outputs/analysis/cifar10_rgb_feedback_signal_probe/frontier_summary.md
outputs/analysis/cifar10_rgb_feedback_signal_probe/paired_deltas.md
```

Current read: bottom-up state feedback is useful when paired with the
distributional coarse objective. `dist_state` beats `dist_position` on
generated-label accuracy, diversity, feature diversity, feature-nearest
distance, attractor accuracy, and basin score. It loses nearest-pixel MSE and
sampling speed, so treat it as the active-hierarchy lead for dynamical basin
quality, not as the final CIFAR rendering recipe. The next probe should convert
that better basin into visible quality through readout/objective changes or a
more selective state-feedback gate.

## CIFAR RGB Feedback Source-Gate Probe

Test whether the useful `state` feedback should read from the whole fine field,
only the class-drive target columns, or their complement. This keeps the
distributional coarse objective and active centered signed-gain hierarchy fixed,
then changes `multiscale_feedback_source_gate`.

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_source_gate_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_gate_probe \
  --title "CIFAR-10 RGB feedback source-gate frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_gate_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_gate_probe \
  --variant-regex 'feedback_source_gate_(.+?)_n256' \
  --pair source_all:source_conditioning \
  --pair source_all:source_non_conditioning \
  --pair source_conditioning:source_non_conditioning \
  --title "CIFAR-10 RGB feedback source-gate paired deltas"
```

Current read: hard source gating does not beat the all-source feedback route.
`source_all` remains the strongest semantic/feature-nearest row. Listening only
to class-conditioned fine columns improves nearest-real MSE and output settling
but loses accuracy, diversity, feature-nearest distance, and basin score.
Listening only to non-conditioned columns improves raw diversity but weakens
class consistency. Treat broad state feedback as the active-hierarchy lead; the
next rendering improvement should come from readout/objective staging or learned
soft feedback weights, not binary source masks.

## CIFAR RGB Feedback Source-Mix Probe

Test mean-normalized soft source mixing for bottom-up state feedback. This keeps
the `dist_state` active hierarchy fixed and changes only the relative
conditioning/non-conditioning source balance:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_source_mix_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_mix_probe \
  --title "CIFAR-10 RGB feedback source-mix frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_mix_probe \
  --variant-regex 'feedback_source_mix_(.+?)_n256' \
  --pair source_all:mix75_25 \
  --pair source_all:mix50_50 \
  --pair source_all:mix25_75 \
  --pair mix75_25:mix25_75 \
  --title "CIFAR-10 RGB feedback source-mix paired deltas"
```

Pre-run read: `mix50_50` is a sanity check for the normalized weighted gate and
should behave like `source_all` under matched seeds. If `mix75_25` improves
nearest-real MSE without giving up the `source_all` semantic/basin edge, the
class-conditioned source channel is useful but should be continuous rather than
binary. If `mix25_75` increases diversity without the semantic collapse seen in
hard `source_non_conditioning`, the autonomous fine field is carrying useful
generative variation that hard gating exposed too aggressively.

Current read:

- `mix75_25` is the best generated-label / feature-diversity compromise:
  accuracy 0.4424 vs 0.3984 for `source_all`, feature diversity 0.8476 vs
  0.8272, attractor accuracy 0.4188 vs 0.3813, and basin score 1.6099 vs
  1.3313 across two seeds.
- `mix50_50` gives the strongest attractor metrics: attractor accuracy 0.4437
  and basin score 1.7444. It is close to `source_all` in the ordinary sample
  metrics, but not bitwise identical under the GPU sweep.
- `mix25_75` improves nearest-real proximity slightly, but loses class
  consistency and feature diversity.
- Visual samples remain blurry CIFAR-like outputs. The result supports soft,
  class-source-heavy feedback routing as a hierarchy mechanism, not yet a
  rendered-quality breakthrough.

Sample grids were pulled to:

```text
outputs/modal_samples/feedback_source_mix/contact_sheet_samples.png
```

Important caveat: these numbers were produced before fixing the RGB
coarse-auxiliary target layout. Direct CIFAR RGB data is flat `C,H,W`, while
the old auxiliary downsampler reshaped it as `H,W,C`. Rerun the same source-mix
probe with corrected channel-first downsampling:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe
```

Analyze the corrected run with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_mix_auxfix_probe \
  --title "CIFAR-10 RGB feedback source-mix auxfix frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_feedback_source_mix_auxfix_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_feedback_source_mix_auxfix_probe \
  --variant-regex 'feedback_source_mix_auxfix_(.+?)_n256' \
  --pair source_all:mix75_25 \
  --pair source_all:mix50_50 \
  --pair source_all:mix25_75 \
  --pair mix75_25:mix25_75 \
  --title "CIFAR-10 RGB feedback source-mix auxfix paired deltas"
```

Auxfix read:

- `mix50_50` gives the best generated-label accuracy and attractor accuracy:
  accuracy 0.4014 vs 0.3682 for `source_all`, feature diversity 0.8359 vs
  0.7955, attractor accuracy 0.4188 vs 0.3312, and basin score 1.6633 vs
  1.1372.
- `mix75_25` gives the strongest diversity and basin score: diversity 0.9335
  and basin score 1.6707, but worsens nearest-real MSE.
- `mix25_75` is not the right direction for hierarchy; it stays closer in
  pixel space but does not improve class/basin metrics.
- Visual samples remain blurry/abstract, so the fixed hierarchy still needs a
  better final readout/objective before it becomes a visible CIFAR-quality win.

Corrected sample grids:

```text
outputs/modal_samples/feedback_source_mix_auxfix/contact_sheet_samples.png
```

## CIFAR RGB Vertical Homeostasis Probe

Test whether calibrated vertical gain helps when it is homeostatically centered
and RMS-controlled instead of allowed to change the fine field's total drive
energy:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_vertical_homeostasis_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_vertical_homeostasis_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_vertical_homeostasis_probe \
  --title "CIFAR-10 RGB vertical homeostasis frontier" \
  --accuracy-floor 0 \
  --no-plot
```

Current read: homeostatic gain normalization is not universally useful, but it
rescues the selective signed route. `signed_gain_conditioning_vscale30_normstd015`
improves generated-label accuracy, feature diversity, attractor accuracy, basin
score, output settling, and nearest-real MSE over both `no_vertical_auxlow8` and
raw `signed_gain_conditioning_vscale30`, with a mild diversity tradeoff. The
dual-gain normalized variant gets worse, so this is a selective E/I-style gain
lead rather than a generic "normalize all hierarchy" result.

Scale-audit caveat: this CSV was collected before normalized variants applied
sample-time scale interventions after normalization. Use the normal/zero/shuffle
and flip audit rows; ignore `scale025` and `scale050` for normalized variants in
this artifact.

Run the selective signed-gain homeostasis calibration:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_signed_gain_homeostasis_calibration.json
```

It compares `center`, `center_rms=0.010`, `center_rms=0.015`, and
`center_rms=0.020` for the same calibrated selective signed-gain route. Use it
to decide whether `0.015` was a lucky point or a real homeostatic gain sweet
spot before adding delayed feedback or parameter-level vertical modulation.

Current calibration read: `center` is the stronger semantic/diversity/basin
candidate. Fixed-RMS variants improve nearest-real or feature-proximity metrics
but weaken the generated-label/diversity/attractor frontier. The useful
mechanism looks like zero-mean selective signed top-down bias, not hard
amplitude clamping.

Run the centered signed-gain timing probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_timing_probe.json
```

It compares immediate centered selective gain against `delayed8`,
`delayed16`, and a `linear_ramp` from step 8 over 16 steps. Current read:
constant centered gain remains the best semantic/attractor candidate; delayed8
trades accuracy/proximity for diversity and a tiny basin edge; ramping improves
nearest-real and feature-proximity but weakens semantic/basin metrics. Treat
timing as useful instrumentation, not the main next axis.

Run the centered signed-gain target probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_centered_signed_gain_target_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_centered_signed_gain_target_probe \
  --title "CIFAR-10 RGB centered signed gain target probe" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_centered_signed_gain_target_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_centered_signed_gain_target_probe \
  --baseline-variant center_drive \
  --target-variant center_coupling \
  --target-variant center_conditioning \
  --target-variant center_damping \
  --title "CIFAR-10 RGB centered signed gain target paired deltas"
```

Current read: `center_drive` remains the best semantic/attractor target.
`center_coupling` improves feature diversity but is weakly causal;
`center_conditioning` and `center_damping` improve proximity-style metrics while
reducing diversity/basin strength. Treat target-specific modulation as useful
instrumentation, not the next quality frontier by itself.

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

## CIFAR RGB Readout Fusion Probe

Run the conservative coarse-to-final readout fusion probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_readout_fusion_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_fusion_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_fusion_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_fusion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_readout_fusion_probe \
  --title "CIFAR-10 RGB readout fusion frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_fusion_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_readout_fusion_probe \
  --variant-regex 'readout_fusion_(.+?)_n256' \
  --pair mix50_50:mix50_50_fusion010 \
  --pair mix50_50:mix50_50_fusion025 \
  --pair mix50_50:mix75_25_fusion010 \
  --title "CIFAR-10 RGB readout fusion paired deltas"
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/readout_fusion/samples/<run>.png --force
```

Current read: direct readout fusion is a useful probe but not a visual-quality
breakthrough. It improves nearest-real MSE and output-settling in some cases,
but usually costs semantic accuracy/diversity. `mix75_25_fusion010` is the only
variant that improved attractor accuracy and basin score versus `mix50_50`.
Treat this as evidence for a staged/coarse-aware readout objective, not for
increasing the fusion blend.

## CIFAR RGB Coarse Readout Consistency Probe

Run the staged/coarse-aware readout consistency probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_coarse_readout_consistency_probe
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_readout_consistency_probe \
  --title "CIFAR-10 RGB coarse readout consistency frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_coarse_readout_consistency_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_coarse_readout_consistency_probe \
  --variant-regex 'coarse_readout_consistency_(.+?)_n256' \
  --pair mix50_50:mix50_50_consistency005 \
  --pair mix50_50:mix50_50_consistency010 \
  --pair mix50_50:mix75_25_consistency005 \
  --title "CIFAR-10 RGB coarse readout consistency paired deltas"
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/coarse_readout_consistency/samples/<run>.png --force
```

Current read: this is a cleaner version of the direct-fusion failure mode.
Coarse readout consistency improves nearest-real MSE and output-settling, but
it strongly reduces diversity, generated-label accuracy, attractor accuracy,
and basin score. Treat it as a diagnostic/control showing that pixel-level
coarse agreement is too prototype-biased for the generator objective.

## CIFAR RGB Readout Gate Probe

Run the learned coarse-to-fine seed modulation probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_readout_gate_probe
```

This compares the current hierarchy lead against `gate010` and `gate025`, where
the selected auxiliary oscillator layer applies a FiLM-style scale/shift to the
fine resize-conv seed tensor. Unlike readout fusion, this does not blend coarse
RGB pixels into the output.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_gate_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_gate_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_gate_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_readout_gate_probe \
  --title "CIFAR-10 RGB readout gate frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_readout_gate_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_readout_gate_probe \
  --variant-regex 'readout_gate_(.+?)_n256' \
  --pair hierarchy_lead:gate010 \
  --pair hierarchy_lead:gate025 \
  --title "CIFAR-10 RGB readout gate paired deltas"
```

Current read: `gate010` improves diversity and basin score but hurts feature
Frechet and attractor accuracy. `gate025` improves nearest-real MSE, generated
accuracy, basin score, and output settling, but loses diversity and
feature-distribution quality. The readout gate is stable and useful, but it is
not the new rendered-image default.

## CIFAR RGB Frequency Objective Probe

Run the frequency/edge-statistics probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_frequency_objective_probe
```

This compares the current hierarchy lead against `freq001` and `freq003`. The
new loss is deliberately light: it asks whether the measured high-frequency and
edge-energy deficit is an objective/readout problem before changing the
oscillator architecture again.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_frequency_objective_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_frequency_objective_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_frequency_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_frequency_objective_probe \
  --title "CIFAR-10 RGB frequency objective frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_frequency_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_frequency_objective_probe \
  --variant-regex 'frequency_objective_(.+?)_n256' \
  --pair hierarchy_lead:freq001 \
  --pair hierarchy_lead:freq003 \
  --title "CIFAR-10 RGB frequency objective paired deltas"
```

Current read: the objective fixes the measured frequency gap but not the
rendering problem. `freq001`/`freq003` move generated high-frequency and edge
ratios close to real-image levels, and the oscillator state retains more
spatial high-frequency power after settling. Samples show much of that new
energy as color-channel ringing and borders rather than object-aligned detail,
so these presets are diagnostics, not new defaults.

## CIFAR RGB Patch Objective Probe

Run the local patch-detail probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_patch_objective_probe
```

This compares the current hierarchy lead against `patch005` and `patch010`.
Unlike global frequency matching, the patch objective compares local raw and
Laplacian patch distributions. The test asks whether local detail can improve
without rewarding image-border or color-channel ringing.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_objective_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_objective_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_patch_objective_probe \
  --title "CIFAR-10 RGB patch objective frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_objective_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_patch_objective_probe \
  --variant-regex 'patch_objective_(.+?)_n256' \
  --pair hierarchy_lead:patch005 \
  --pair hierarchy_lead:patch010 \
  --title "CIFAR-10 RGB patch objective paired deltas"
```

Current read: `patch010` is a stronger local-detail contender than global
frequency matching. It improves generated-label accuracy, diversity,
classifier-feature Frechet/KID, attractor accuracy, and basin score versus the
matched hierarchy baseline. The cost is worse nearest-real pixel MSE, lower
sample/sec, and visible local striping/chunky detail. Treat it as the next
rendering-objective lead, not a stable default.

## CIFAR RGB Patch V2 Probe

Run the shifted-grid/multiscale patch-detail probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_patch_v2_probe
```

This compares the hierarchy lead, plain `patch010`, shifted-grid
`patch010_overlap`, multiscale `patch010_multiscale`, and the combined
`patch010_multiscale_overlap`. The goal is narrower than "make CIFAR solved":
keep the local-detail/semantic gains from `patch010` while reducing fixed-grid
chunking and striping. This sweep uses a lighter `normal,zero` vertical audit
because the patch-v2 question is rendering/objective quality, not full
hierarchy-causality attribution.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_v2_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_v2_probe.json
```

Analyze it with:

```bash
python scripts/analyze_mnist_generator_frontier.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_v2_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_patch_v2_probe \
  --title "CIFAR-10 RGB patch v2 frontier" \
  --accuracy-floor 0 \
  --no-plot

python scripts/analyze_generator_paired_deltas.py \
  --csv outputs/analysis/modal_mnist_generator_cifar10_rgb_patch_v2_probe.csv \
  --output-dir outputs/analysis/cifar10_rgb_patch_v2_probe \
  --variant-regex 'patch_v2_(.+?)_n256' \
  --pair hierarchy_lead:patch010 \
  --pair patch010:patch010_overlap \
  --pair patch010:patch010_multiscale \
  --pair patch010:patch010_multiscale_overlap \
  --title "CIFAR-10 RGB patch v2 paired deltas"
```

Current read: overlap and multiscale patch scoring improve several metrics
over plain `patch010` (especially attractor/basin and feature-distribution
metrics), but the samples still show dark blobs, halos, horizontal bands, and
grid-like texture. `patch010_multiscale` has the best completed feature
Frechet/basin tradeoff, while `patch010_overlap` is the better low-cost repair.
Neither is a stable default. Treat patch-v2 as evidence that local detail
pressure helps diagnostics but still needs a better readout/objective interface.

## CIFAR RGB State Information Probe

Run the compact state-readout attribution audit:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_information_probe
```

This compares `current`, `hierarchy_lead`, and `patch010_multiscale` with
`--state-probe-sample-count 64`. The probe fits small ridge readouts from traced
oscillator states to labels, generated low-res scaffold, generated high-pass
residual, and classifier features. It is meant to answer whether information is
present in the state before adding more renderer or objective machinery.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_information_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_information_probe.json
```

Current read: final HORN states decode class, scaffold, high-pass residual, and
classifier features substantially better after settling than at initialization.
The state is therefore not dead; the remaining bottleneck is converting the
settled state into clean object-aligned RGB detail.

## CIFAR RGB State Residual Readout Probe

Run the compact state-to-image interface probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=6 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_residual_readout_probe
```

This compares the hierarchy lead against two local residual HORN-state readout
strengths, `0.05` and `0.10`. The residual branch lets each final oscillator
write a small learned local RGB patch from its final position/velocity state,
on top of the normal resize-conv renderer.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_residual_readout_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_residual_readout_probe.json
```

Current read: `state_residual005` is the useful candidate. In the two-seed
probe it improves generated-label accuracy, attractor robustness,
nearest-real MSE, feature Frechet, and final-state high-pass decodability
against `hierarchy_lead`. `state_residual010` is more mixed, so this should be
treated as a calibrated state-to-image interface, not a knob to turn upward.

## CIFAR RGB State Residual Longer Pilot

Run the 40-epoch visual-maturation pilot:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_residual_longer_pilot
```

This compares the stable CIFAR RGB default, the hierarchy lead, and
`state_residual005` on seed `23`, with artifacts saved at epochs `20` and `40`.
It is a visual/quality sanity check, not a broad seed-confirmation sweep.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_residual_longer_pilot.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_residual_longer_pilot.json
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/state_residual_longer/<run>_epoch020.png --force

modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_040.png \
  outputs/modal_samples/state_residual_longer/<run>_epoch040.png --force
```

Current read: longer training improves saturation/structure but does not
produce a sudden visual breakthrough by epoch `40`. `state_residual005`
continues to beat `hierarchy_lead` on several metrics, but the stable
normalized-local CIFAR default remains the stronger overall model in this
pilot.

## CIFAR RGB Resonant Filter-Bank Readout Pilot

Run the compact ONN-native readout pilot:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_resonant_readout_pilot
```

This compares the stable CIFAR RGB default against two small shared HORN
resonant filter-bank readout strengths, `0.05` and `0.10`. The branch adds
only `675` CIFAR RGB parameters: shared local filters over HORN position,
velocity, local order, phase-alignment, velocity contrast, and energy
observables.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_resonant_readout_pilot.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_resonant_readout_pilot.json
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/resonant_readout/<run>_epoch020.png --force
```

Current read: `resonant005` improves generated-label accuracy, diversity,
feature Frechet, and attractor diversity over the stable CIFAR default in the
seed-23 pilot, but worsens nearest-real MSE and does not solve visible
sharpness. `resonant010` is already more mixed/muddy, so the next step is not
turning strength upward.

## CIFAR RGB Oscillator Capacity Probe

Run the compact oscillator-site capacity probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_capacity_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current`
- `sparse_horn_cifar10_rgb_current_n512`
- `sparse_horn_cifar10_rgb_current_n512_resonant005`

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_capacity_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_capacity_probe.json
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/capacity_probe/<run>_epoch020.png --force
```

Current read: doubling oscillator sites to `512` makes CIFAR samples more
active/diverse and increases edge energy, but it does not improve the main
quality frontier. It hurts class consistency, attractor accuracy, nearest-real
MSE, and throughput. The first conclusion is that raw site count is not enough;
larger HORN fields need stronger organization or multimode/frequency-band
structure before a 1024-site run is worth the cost.

## CIFAR RGB Multimode HORN Probe

Run the compact frequency-band capacity probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_multimode_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current`
- `sparse_horn_cifar10_rgb_current_n512`
- `sparse_horn_cifar10_rgb_current_multimode2`

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_multimode_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_multimode_probe.json
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/plots/mnist_generator_samples_epoch_020.png \
  outputs/modal_samples/multimode_probe/<run>_epoch020.png --force
```

Current read: two HORN frequency modes per spatial site are much better than a
flat 512-site field on class consistency and attractor metrics, and they beat
the 256-site default on generated-label and attractor accuracy. They do not yet
solve visual sharpness or high-frequency detail, and they are slower than the
compact 256 default. Treat multimode HORN as the next structured-capacity lead,
not as the stable CIFAR default yet.

## CIFAR RGB Retinotopic Readout and State-Fitting Probe

Run the geometry/readout diagnostic:

```bash
OSCNET_MODAL_MAX_CONTAINERS=4 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_retinotopic_readout_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current`
- `sparse_horn_cifar10_rgb_current_retinotopic`
- `sparse_horn_cifar10_rgb_current_multimode2`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic`

It also enables the frozen-decoder state-fitting probe:

```text
--state-fit-sample-count 32 --state-fit-steps 80 --state-fit-settle-steps 0,8,16,32
```

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_retinotopic_readout_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_retinotopic_readout_probe.json
```

Pull sample grids from the Modal volume:

```bash
modal volume get oscnet-runs \
  mnist_generator/<run>/artifacts/mnist_generator_samples_epoch_020.npz \
  outputs/modal_samples/retinotopic_probe/<run>.npz --force
```

Current read: retinotopic layout fixes a real state-to-seed geometry mismatch
and greatly improves frozen-state reconstruction, especially for multimode
HORN. It does not yet improve direct generated CIFAR samples at 20 epochs.
Fitted detail is partly destroyed by extra HORN settling, so the next issue is
not only readout layout; the training objective or trajectory must learn to
steer samples into detail-carrying states and preserve them.

## CIFAR RGB Retinotopic Control Probe

Run the param-matched retinotopic control:

```bash
OSCNET_MODAL_MAX_CONTAINERS=5 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_retinotopic_control_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current`
- `sparse_horn_cifar10_rgb_current_retinotopic_ch30`
- `sparse_horn_cifar10_rgb_current_retinotopic_seed4_ch30`
- `sparse_horn_cifar10_rgb_current_multimode2`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30`

It enables:

```text
--state-fit-settle-steps 0,1,2,4,8,16,32
```

and reports matched-norm random perturbation controls as
`state_fitting_probe.noise_*`.

This writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_retinotopic_control_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_retinotopic_control_probe.json
```

Current read: matching decoder capacity does not make retinotopic direct
generation beat the flat seed. The flat multimode branch still wins class
accuracy/feature diversity, while retinotopic branches win nearest-real MSE,
high-frequency ratio, and frozen-state reconstruction. Four seed channels do
not rescue single-mode retinotopic class accuracy. Matched-norm random noise is
less destructive than HORN settling at later steps, supporting the contraction
interpretation: fitted detail states exist, but the current dynamics/objective
do not preserve them as texture-bearing attractors.

## CIFAR RGB State Anchor Probe

Run the two-seed state-space anchor probe:

```bash
OSCNET_MODAL_MAX_CONTAINERS=5 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_anchor_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_ch30`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor_reconstruct010`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor_frozen010`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor010`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030`

It writes:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_anchor_probe.csv
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_anchor_probe.json
outputs/modal_samples/state_anchor_probe/state_anchor_probe_contact_sheet.png
```

Current read: the anchor is a real settle-survival win, not a free-sample
quality win yet. `anchor030` brings settle-8 fitted-state MSE close to the
matched-noise control and beats both k=0 and frozen-dynamics controls. Free
samples gain high-frequency energy and lower nearest-real MSE, but diversity
drops and visuals remain blurry, so the next question is how random
class-conditioned trajectories enter the anchor-trained detail basin.

## CIFAR RGB State Prior Sampling Probe

Pull the trained `anchor030` checkpoint from the Modal volume:

```bash
mkdir -p outputs/checkpoints/state_anchor_probe
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_anchor_anchor030_train2000_seed23_20e/checkpoints/checkpoint_epoch_020.eqx \
  outputs/checkpoints/state_anchor_probe/seed23_anchor030_checkpoint_epoch_020.eqx
```

Run the eval-only state-prior diagnostic locally:

```bash
python scripts/analyze_generator_state_prior.py \
  --checkpoint outputs/checkpoints/state_anchor_probe/seed23_anchor030_checkpoint_epoch_020.eqx \
  --preset sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030 \
  --seed 23 \
  --train-limit 2000 \
  --eval-limit 1000 \
  --sample-count 256 \
  --prior-rank 32 \
  --settle-steps 8 \
  --batch-size 64 \
  --classifier-epochs 0 \
  --output-dir outputs/analysis/state_prior_probe \
  --output-prefix seed23_anchor030_rank32_samples256
```

This writes:

```text
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256.csv
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256.json
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256_samples.npz
outputs/analysis/state_prior_probe/seed23_anchor030_rank32_samples256_contact_sheet.png
```

Current read: the transparent per-class PCA state prior reaches a different
basin than white-noise initialization. It improves diversity and avoids tight
duplicates, but it does not yet produce sharp CIFAR samples. The shuffled-prior
control shows that the initial state prior carries class information, so future
training integration needs a control that keeps class semantics attributable to
the HORN conditioning path.

## CIFAR RGB State Prior Training Probe

Run the two-seed training intervention:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_prior_training_probe
```

This compares:

- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_global`
- `sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class`

The new prior arms keep the anchor loss active, refit a host-side low-rank PCA
state prior from anchor-encoder outputs each epoch, and draw the drift
objective's generated samples from that prior instead of isotropic white-noise
HORN states. `prior_global` is the stronger attribution arm because class
identity can only enter through oscillator conditioning; `prior_class` is the
pragmatic arm and should be checked with shuffled-prior evaluation after
training.

Important scoring rule: for prior arms, the generator is formally
`state prior + HORN field + decoder`. Final eval, settling metrics, attractor
metrics, and contact sheets must therefore sample from the fitted final prior.
White-noise sampling is still logged as a secondary diagnostic, not as the
primary score for prior-trained arms. The final fitted prior is saved beside
the checkpoint as `state_prior_final.json` and `state_prior_final.npz`.

Current prediction: prior training should reduce the mode-collapse pressure
seen in `anchor030` white-noise samples. If it improves diversity/class
consistency but not sharpness, treat the HORN field as a scaffold generator and
move to a two-stage renderer.

Completed two-seed result:

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_training_probe.csv
```

```text
Arm           Init         Acc     Feature div  Near MSE  Best settle acc  Dup<0.010  Edge ratio
anchor030     white noise  0.2910  0.7512       0.0178    0.3164           0.1367     0.2512
prior_global  prior        0.4316  0.8581       0.0293    0.4551           0.0039     0.4539
prior_class   prior        0.4941  0.8715       0.0369    0.5371           0.0000     0.4825
```

Pull the contact sheets:

```bash
mkdir -p outputs/analysis/state_prior_training_probe/contact_sheets
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_anchor030_train2000_seed23_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/anchor030_seed23_epoch020.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_prior_global_train2000_seed23_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/prior_global_seed23_epoch020.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_prior_class_train2000_seed23_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/prior_class_seed23_epoch020.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_anchor030_train2000_seed24_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/anchor030_seed24_epoch020.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_prior_global_train2000_seed24_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/prior_global_seed24_epoch020.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_training_prior_class_train2000_seed24_20e/plots/mnist_generator_samples_epoch_020.png \
  outputs/analysis/state_prior_training_probe/contact_sheets/prior_class_seed24_epoch020.png
```

Current operational read at this stage: `prior_global` is the clean reference
because the state prior is class-agnostic; `prior_class` is the stronger HORN
prior arm. Do not rank this branch by nearest-real MSE alone. The prior arms
intentionally escape the duplicate/collapse basin that made nearest-MSE look
good for `anchor030`. The later same-stack StateMLP control below supersedes
any HORN-vs-control claim from this probe.

## CIFAR RGB State Prior Scale Gate Rung 1

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1
```

The initial batch-128 and batch-64 versions exceeded A10G memory for this
objective stack. The completed rung used batch 32, train-limit 10k, 40 epochs,
two seeds, 512 eval samples, and a quality-classifier train limit of 20k.

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1.csv
outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/rung1_comparison.png
```

```text
Arm                   Acc     Feature div  Feature near  Near MSE  Dup<0.010  Edge ratio  High freq  Best settle acc
prior_global_b32      0.5547  0.9434       0.2699        0.0336    0.0088     0.5690      0.7964     0.6084
prior_class_b32       0.4531  0.9081       0.3118        0.0392    0.0029     0.6335      0.7015     0.4629
prior_class_patch005  0.7129  0.9894       0.2025        0.0361    0.0000     0.9270      1.2599     0.7373
```

Pull the contact sheets:

```bash
mkdir -p outputs/analysis/state_prior_scale_gate_rung1/contact_sheets
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_global_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_global_seed23_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_global_b32_train10000_seed24_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_global_seed24_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_class_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_class_seed23_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_class_b32_train10000_seed24_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_class_seed24_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_class_patch005_offset_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_class_patch005_seed23_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_class_patch005_offset_b32_train10000_seed24_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_scale_gate_rung1/contact_sheets/prior_class_patch005_seed24_epoch040.png
```

Visual read: `prior_class_patch005` is still soft, but it is the first CIFAR
RGB arm that improves semantic accuracy, diversity, edge/high-frequency
diagnostics, and visual texture together without returning to duplicate
collapse. Treat it as the HORN state-prior reference and keep `prior_global` as
the attribution-clean reference; the same-stack StateMLP control below
supersedes any HORN-vs-control claim from this rung alone.

## CIFAR RGB State Prior Same-Stack Control Probe

Run:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_cifar10_rgb_state_prior_control_probe
```

This probe checks whether the rung-1 HORN reference survives a matched
non-oscillatory transition. The `StateMLPImageGenerator` receives the same
retinotopic/multimode state layout, state-anchor encoder path, state-prior
sampling, anchor loss, and patch005 objective as the HORN recipe.

```text
outputs/analysis/modal_mnist_generator_cifar10_rgb_state_prior_control_probe.csv
outputs/analysis/state_prior_control_probe/contact_sheets/control_probe_comparison.png
```

```text
Arm                         n  Acc     Feature div  Feature near  Near MSE  Dup<0.010  Edge ratio  High freq  Attractor  Best settle acc
prior_class_patch005*       2  0.7129  0.9894       0.2025        0.0361    0.0000     0.9270      1.2599     0.6875     0.7373
state_mlp_prior_class_p005  2  0.6533  0.9261       0.1889        0.0362    0.0000     0.8592      0.9469     0.6875     0.6855
prior_global_patch005       2  0.4365  0.8862       0.2791        0.0347    0.0000     0.8934      1.1327     0.4375     0.4678
prior_class_p005_queue64    1  0.5664  0.9447       0.2955        0.0377    0.0000     0.9430      1.2843     0.5750     0.6348
```

`prior_class_patch005*` is the previous rung-1 HORN result included as the
reference row.

Read:

- StateMLP is a strong same-stack control. It wins eval loss, feature-nearest
  distance, and throughput, so this branch should not claim broad CIFAR
  competitiveness from internal metrics alone.
- Pairing by seed removes the apparent HORN mean lead. Seed 23 favors StateMLP
  on accuracy, attractor accuracy, settling gain, and feature diversity; seed
  24 favors HORN on the same metrics. With `n=2`, this is a null result, not a
  supported HORN advantage.
- HORN is consistently closer on edge-Laplacian ratio, while StateMLP is
  consistently closer to the ideal high-frequency ratio of `1.0` and is faster.
- `prior_global_patch005` does not replace the HORN reference recipe. It is useful
  as a class-agnostic-prior diagnostic, but its class/attractor metrics are
  too weak.
- Queue64 is not promoted from this first probe. It preserves edge/frequency
  energy but hurts semantic/feature metrics.

Pull/check contact sheets:

```bash
mkdir -p outputs/analysis/state_prior_control_probe/contact_sheets
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_control_probe_prior_global_patch005_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_control_probe/contact_sheets/prior_global_patch005_seed23_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_control_probe_prior_global_patch005_b32_train10000_seed24_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_control_probe/contact_sheets/prior_global_patch005_seed24_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_control_probe_state_mlp_prior_class_patch005_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_control_probe/contact_sheets/state_mlp_prior_class_patch005_seed23_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_control_probe_state_mlp_prior_class_patch005_b32_train10000_seed24_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_control_probe/contact_sheets/state_mlp_prior_class_patch005_seed24_epoch040.png
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_control_probe_prior_class_patch005_queue64_b32_train10000_seed23_40e/plots/mnist_generator_samples_epoch_040.png \
  outputs/analysis/state_prior_control_probe/contact_sheets/prior_class_patch005_queue64_seed23_epoch040.png
```

Operational decision: keep `prior_class_patch005` as the HORN CIFAR RGB
reference recipe and keep the same-stack StateMLP beside it as the co-equal
control. Do not launch a full 50k/80-epoch quality rung from this result; the
current branch is a useful mechanism audit plus a null HORN-vs-StateMLP result
for CIFAR image generation at this scale.

## 2026-07-14: CIFAR RGB state recovery probe (eval-only, local, pulled checkpoints)

No new GPU runs. Pulled the epoch-40 checkpoints from the volume and ran the
new noise-then-settle recovery probe locally:

```bash
mkdir -p outputs/checkpoints/state_recovery_probe
modal volume get oscnet-runs \
  /mnist_generator/mnist_generator_cifar10_rgb_state_prior_scale_gate_rung1_prior_class_patch005_offset_b32_train10000_seed23_40e/checkpoints/checkpoint_epoch_040.eqx \
  outputs/checkpoints/state_recovery_probe/prior_class_patch005_seed23_epoch040.eqx
# ... same for prior_class seed24, prior_global 23/24 (control probe runs),
# and state_mlp_prior_class_patch005 23/24 (control probe runs).

python scripts/analyze_generator_state_recovery.py \
  --checkpoint outputs/checkpoints/state_recovery_probe/prior_class_patch005_seed23_epoch040.eqx \
  --preset sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class_patch005 \
  --seed 23 --sample-count 32 --fit-steps 120 \
  --output-dir outputs/analysis/state_recovery_probe \
  --output-prefix horn_prior_class_patch005_seed23
# ... repeated per arm/seed with the matching preset.
```

Outputs: per-arm CSV/JSON/contact sheets plus
`outputs/analysis/state_recovery_probe/state_recovery_probe_aggregate.csv`.

Result summary (details in `docs/experiment_report.md`): both HORN arms
denoise corrupted fitted states in 80/80 cells (denoise fraction positive at
every noise scale, depth, seed; mean 0.40-0.53), while the same-stack StateMLP
manages mean 0.24 with 9/40 negative cells. First consistent-sign
oscillator-vs-control asymmetry in the CIFAR branch.

## 2026-07-14: CIFAR RGB occlusion recovery probe (eval-only, local)

Same six pulled checkpoints as the noise recovery probe. New
`--occlusion-fractions` condition in `scripts/analyze_generator_state_recovery.py`
zeroes square patches (6.25/12.5/25%) of the 16x16 retinotopic site grid, then
settles 1-16 steps, reporting occluded-region vs intact-region decode MSE.

Outputs: `outputs/analysis/state_recovery_probe_occlusion/` (per-arm CSV/JSON/
contact sheets, aggregate `state_occlusion_probe_aggregate.csv`).

Result (details in `docs/experiment_report.md`): decisive negative — no arm
fills in the occluded region at any depth; HORN degrades it slightly while
paying a large intact-region cost, StateMLP is flat-to-marginally-better at
shallow depths. The noise-recovery asymmetry does not extend to structured
damage; emergent associative memory is absent at this operating point.

## 2026-07-15: CIFAR RGB recovery training probe (Modal GPU, detached)

First training run of the corrupted-anchor recovery objective (step 2). The
anchor loss now corrupts before/at encoding — image-space square-patch
occlusion applied before the encoder plus state-space Gaussian noise — settles
the dynamics, decodes, and scores paired MSE against the clean image, with an
explicit clean fixed-point term (`state_anchor_clean_weight`) that also settles
the uncorrupted encoded state. New eval block `generator.recovery.*` reports
PSNR/SSIM and occluded-region vs intact-region MSE across settle depths and
corruption conditions (noise scales; contiguous single-block and scattered
4-patch occlusion).

Sweep preset `mnist_generator_cifar10_rgb_recovery_training_probe`, 40 epochs,
train 10k / eval 5k, A10G, `max_containers=1` (runs roll sequentially). Six
arms x 2 seeds (23, 24):

- `horn_recovery_noise` — multimode2 HORN, Gaussian-noise-only corruption.
- `horn_recovery_mixed` — multimode2 HORN, noise + scattered 4-patch occlusion.
- `state_mlp_recovery_noise` / `state_mlp_recovery_mixed` — matched
  non-oscillatory recurrent MLP controls.
- `single_local_recovery_mixed` — single-mode local HORN, mixed corruption; the
  matched no-carrier baseline for the carrier arm.
- `coarse_carrier_recovery_mixed` — single-mode fine HORN plus a 16-node dense
  slow/global coarse carrier (`CoarseToFineHORNImageGenerator`). The carrier is
  seeded in the recovery/anchor path by parameter-free spatial mean-pooling of
  the (corrupted) fine encoded state, so a long-wavelength mode participates in
  settling. This is the one mechanism theory says could move the contiguous-hole
  number; the matched single-mode local arm isolates its effect.

App: `ap-GzzdJI4vJm3HrgTHp0bMgl` (detached). Sweep CSV on completion:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_recovery_training_probe.csv`.
Design rationale (distributed vs contiguous corruption, slow/global carrier) is
in `docs/experiment_report.md`.

Update: `ap-GzzdJI4vJm3HrgTHp0bMgl` was cancelled (serial execution with
`max_containers=1` was too slow), as was a 12-container relaunch
(`ap-jZMTfFXjvjPlu3ffeWTpCN`, exceeded the workspace 10-GPU limit). Final run:
`ap-15xJuAPbD45RyKC9p4akCd` with `OSCNET_MODAL_MAX_CONTAINERS=8`, completed
2026-07-15 12:18 CEST, all 12 runs clean. Results and interpretation in
`docs/experiment_report.md` ("Recovery Training Probe — Results"). Headline:
occlusion training works (3-5x better fill-in than noise-only training); the
slow/global carrier arm showed no reliable effect vs its matched baseline;
multimode HORN is the only arm where settling depth improves fill-in on both
seeds; the StateMLP control remains the absolute winner.

## 2026-07-15: CIFAR RGB multimode carrier probe (Modal GPU, 8 parallel)

Active-ingredient ablation for the multimode fill-in effect plus the carrier's
"fair shot". Sweep `mnist_generator_cifar10_rgb_multimode_carrier_probe`, app
`ap-ew3UgLxsft1IorlyV84F3l`, `OSCNET_MODAL_MAX_CONTAINERS=8`, completed
2026-07-15 16:09 CEST. Arms x seeds 23/24: mm2 equal-frequency (1.0/1.0), mm2
wide split (0.5/1.5), mm4, and `coarse_multimode_horn` slow carrier
(16-node dense coarse band, `coarse_frequency_scale 0.5`). Recovery eval
extended to settle depths 0-64. Both mm4 runs OOMed on A10G (1024 oscillators,
16.4GiB `_train_step` allocation at batch 32); rerun at batch 16 deemed
low-value. CSV:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_multimode_carrier_probe.csv`.

Headline (details in `docs/experiment_report.md`): equal-frequency multimode
shows the *strongest* settling fill-in — the slow-band hypothesis is falsified;
the active ingredient is per-site capacity/mode coupling. The fair slow carrier
adds nothing (third strike; retired). All arms degrade at settle depths 32-64,
so the fill-in window is shallow (k8-k16). StateMLP anchor remains ~30% better
than the best oscillator arm.

## 2026-07-15: CIFAR RGB coupling-topology probe (Modal GPU, 8 parallel)

Tests whether the StateMLP recovery advantage is really a coupling-range effect
(dense/non-local) rather than an oscillator-vs-not effect. Holds the
recovery-trained mixed-corruption objective fixed and varies only the recurrent
coupling topology. Added a new `fractal` coupling profile
(`hierarchical_coupling_profile` in `oscnet/core/coupling.py`): self-similar
ultrametric kernel on the oscillator grid with direct long-range links. Sweep
`mnist_generator_cifar10_rgb_coupling_topology_probe`, app
`ap-qTl1hsn2kkqoRTqfulAyKz`, `OSCNET_MODAL_MAX_CONTAINERS=8`, launched
2026-07-15 17:38 CEST. Six arms x seeds 23/24 (12 runs): single-mode HORN local
vs fractal; multimode2 HORN local vs dense vs fractal; StateMLP mixed anchor
(dense-linear ceiling). Recovery eval settle depths 0-32. CSV:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_coupling_topology_probe.csv`.

Hypothesis: if the StateMLP win is about non-local coupling, dense/fractal
oscillator arms should beat their local baselines and close the gap to
StateMLP. If not, the win is dense linear mixing and oscillation adds nothing.

Completed 2026-07-15 18:33 CEST, all 12 runs clean. Headline (details in
`docs/experiment_report.md`): partially confirmed. Dense/fractal coupling on
the multimode substrate beats local by ~18% on contiguous-hole fill-in (both
seeds), so locality was a real confound and the transport theory holds. But
StateMLP still wins by ~50% with coupling range equalized, so the residual gap
is intrinsic to the oscillatory update. Fractal matches dense (non-locality is
the active ingredient, not self-similarity); topology only helps on the
multimode substrate; deep-settling reversal unchanged.

## 2026-07-16: CIFAR RGB robustness probe (Modal GPU, 8 parallel)

First eval on the oscillator's "home" fitness function: graceful degradation
instead of exact reconstruction at infinite-precision parity. New
`compute_generator_robustness_metrics` scores contiguous-occlusion fill-in and
clean PSNR at fixed settle depth 8 under three stressors: Gaussian weight
jitter (per-leaf-std-relative scales 0.02/0.05/0.1/0.2, 3 draws averaged),
uniform weight quantization (8/6/4/3 bits), and out-of-distribution occlusion
(fractions 0.1/0.25/0.4/0.6 vs 0.25 trained). Sweep
`mnist_generator_cifar10_rgb_robustness_probe`, app
`ap-CYvg8hNf6VjCCJH9Q9xxHa`, `OSCNET_MODAL_MAX_CONTAINERS=8`, launched
2026-07-16 10:23 CEST. Four arms x seeds 23/24 (8 runs, one wave):
single-mode local HORN, multimode2 local, multimode2 dense, StateMLP.
An earlier accidental launch (`ap-GcwIp5vtsWZPvW6t5FWfFf`, 2026-07-16 00:24
CEST) was stopped after ~2 minutes at the user's request. CSV:
`outputs/analysis/modal_mnist_generator_cifar10_rgb_robustness_probe.csv`.

Hypothesis: if the physics-constrained oscillator update buys the graceful
degradation that analog systems claim, its quality should collapse more slowly
than StateMLP's under weight noise and quantization, even though its absolute
baseline is worse.

Completed 2026-07-16 ~11:10 CEST, all 8 runs clean. Headline (details in
`docs/experiment_report.md`): **the crossover exists — first absolute
oscillator win in the project.** At nominal conditions StateMLP keeps its usual
lead, but under severe stress the ranking inverts: at 0.6 occlusion StateMLP
degrades 4.2x its baseline vs 1.7x for multimode-local (absolute fill-in 0.172
vs 0.107, both seeds); at 3-bit quantization StateMLP degrades 2.1x vs ~1.1x
for the multimode arms (absolute 0.087 vs 0.066-0.067). Weight-noise results
are mixed/noisy (3 draws), with the oscillator retaining more clean PSNR only
at the extreme 0.2 scale. The physics prior is protective off-nominal; the
free-form update is better on-nominal.
