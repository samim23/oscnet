# OscNet Examples

This folder contains runnable entrypoints. If you want to try OscNet from the
command line, start here.

The examples are intentionally thin: they parse command-line arguments and call
the experiment harnesses in `oscnet.experiments`. That split is deliberate:
`examples/` is for users running scripts, while `oscnet.experiments` is
importable package code.

## Start Here

Run a tiny CPU smoke test:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source synthetic \
  --epochs 1
```

Show options for any example:

```bash
python examples/image_mnist_phase_flow.py --help
```

Runs write checkpoints, plots, metrics, traces, and samples under `outputs/`.

## Recommended Generator

The default image-generation example is the sparse local HORN MNIST generator.
It starts from random oscillator state, settles a coupled position/velocity
field, and decodes the final oscillator features into an image.

Run the recommended recipe on real MNIST:

```bash
python examples/image_mnist_generator.py
```

This is equivalent to:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist_recommended
```

For a short local probe, keep the recommended preset and lower the budget:

```bash
python examples/image_mnist_generator.py \
  --epochs 1 \
  --train-limit 32 \
  --eval-limit 32
```

For a tiny synthetic smoke test:

```bash
python examples/image_mnist_generator.py \
  --data-source synthetic \
  --model-family horn \
  --decoder-mode resize_conv \
  --num-oscillators 98 \
  --resize-conv-min-channels 4 \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4
```

Matched controls use the same data/objective/readout recipe:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist_frozen
python examples/image_mnist_generator.py --preset sparse_horn_mnist_decoder_only
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp
python examples/image_mnist_generator.py --preset sparse_horn_mnist_step1
```

Useful HORN generator aliases:

- `sparse_horn_mnist_recommended`: default example preset; strict dynamic
  class coupling with higher HORN damping.
- `sparse_horn_mnist_strict`: strict dynamic class coupling, no direct
  label-initialization route, strongest semantic/diversity reference.
- `sparse_horn_mnist_quality`: strict route with a small distributional
  quality/proximity regularizer.
- `sparse_horn_mnist_dynamics_quality`: strict route with higher HORN damping;
  currently the same settings as the recommended preset.
- `sparse_horn_mnist_dynamics_quality_dist0025`: tests whether the higher
  damping setting compounds with the small distributional regularizer. The
  latest sweep kept the default on `sparse_horn_mnist_dynamics_quality`;
  distributional variants are probes, not the recommended first run.
- `sparse_horn_mnist`: older polished recipe with a direct label
  initialization route, kept for comparison and backward-compatible commands.

To probe the stricter route and its controls explicitly:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist_strict
python examples/image_mnist_generator.py --preset sparse_horn_mnist_quality
python examples/image_mnist_generator.py --preset sparse_horn_mnist_dynamics_quality
python examples/image_mnist_generator.py --preset sparse_horn_mnist_dynamics_quality_dist0025
python examples/image_mnist_generator.py --preset sparse_horn_mnist_recommended_no_main_coupling
python examples/image_mnist_generator.py --preset sparse_horn_mnist_recommended_frozen_recurrent
python examples/image_mnist_generator.py --preset sparse_horn_mnist_recommended_frozen_conditioning
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strong
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strength8
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strength8_dist005
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strong_dist005
```

The strict HORN route starts near chance before settling and reaches readable,
varied digits after recurrent dynamics. The StateMLP presets are
non-oscillatory controls; their `dist*` variants test whether a conventional
transition can recover the same diversity with extra distributional pressure.
The recommended ablation presets test whether the result depends on main-pool
coupling, learned recurrent parameters, or learned conditioning parameters.
The current read is that the sparse HORN substrate, multi-step settling, and
learned conditioning drive matter most: frozen recurrent HORN remains strong,
while no-main-coupling, frozen-conditioning, decoder-only, and one-step controls
fall away. The matched StateMLP can still win pixel proximity, so use it when
checking whether a HORN gain is genuinely dynamical rather than a generator
shortcut. The strength-8 StateMLP distributional controls improve pixel
proximity but have not recovered HORN-like diversity in the current sweeps.

## Example Menu

| Example | What it runs |
| --- | --- |
| `image_mnist_oscillatory_autoencoder.py` | MNIST patch autoencoder reference benchmark. |
| `audio_wavelet_oscillatory_autoencoder.py` | Audio wavelet sequence autoencoder benchmark. |
| `image_mnist_jepa.py` | MNIST masked-representation prediction. |
| `image_mnist_generator.py` | Coupled-oscillator MNIST generator branch with Kuramoto, HORN, and controls. |
| `image_mnist_kuramoto_generator.py` | Legacy alias for older Kuramoto-generator commands. |
| `image_mnist_phase_vae.py` | MNIST phase VAE generative baseline. |
| `image_mnist_phase_flow.py` | MNIST phase-rate rectified-flow sampler. |
| `image_mnist_shape_pixel.py` | Two-stage signed-distance shape-to-pixel renderer. |
| `resonanceDB.py` | Phase-aware similarity store demo. |
| `fractal/` | Fractal/HORN coupling examples. |

## Common Commands

Stable MNIST reference run:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source idx \
  --patch-size 7 \
  --decoder-mode positional \
  --epochs 10
```

Small phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Contour-domain phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation sobel_edges \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Smooth shape-field phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation signed_distance \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Two-channel pixel/shape phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation pixels_signed_distance \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Centered two-channel pixel/shape phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation centered_pixels_signed_distance \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Shape-gated centered pixel/shape phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation centered_pixels_signed_distance \
  --sample-readout-mode shape_gated \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Shape-guided sampler phase-flow run:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --target-representation centered_pixels_signed_distance \
  --sample-schedule shape_guided \
  --sample-readout-mode shape_gated \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Shape-to-pixel renderer run:

```bash
python examples/image_mnist_shape_pixel.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Shape-to-pixel scaffold robustness probe:

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

Shape-gated renderer readout:

```bash
python examples/image_mnist_shape_pixel.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4 \
  --sample-readout-mode shape_gated
```

Small phase VAE run:

```bash
python examples/image_mnist_phase_vae.py \
  --data-source synthetic \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4
```

Small HORN generator run:

```bash
python examples/image_mnist_generator.py \
  --data-source synthetic \
  --model-family horn \
  --decoder-mode resize_conv \
  --num-oscillators 98 \
  --resize-conv-min-channels 4 \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4
```

## Where To Go Next

- `oscnet/experiments/README.md` explains the harnesses behind these scripts.
- `docs/model_api.md` explains reusable model classes.
- `docs/experiment_report.md` records research results and interpretation.
