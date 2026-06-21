# Modal GPU Runs

Modal support is kept outside the OscNet library. The reusable models and
experiments still run locally with the normal Python CLI; `scripts/modal_mnist.py`
is only a launch adapter for running the same MNIST experiment on a remote GPU.

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

Outputs are written to the Modal Volume named `oscnet-runs` by default:

```text
/mnt/oscnet-runs/mnist/<run-name>/
```

The MNIST IDX cache and JAX compilation cache are also kept on that volume.

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
