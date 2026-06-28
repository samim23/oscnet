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
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
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
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_core
```

This is the next source-faithful Un-0 port after pixel drift. It keeps the same
controls, but draws same-class positives from a host-side per-class FIFO memory
using `--drift-queue-size 512 --drift-queue-num-pos 32`. Results write to
`outputs/analysis/modal_mnist_generator_resize_conv_pixel_drift_queue_core.csv`.

Dry-run it first:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_mnist_generator.py \
  --sweep-preset mnist_generator_resize_conv_pixel_drift_queue_core \
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
