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

## Current Leading Experiment

The leading experimental branch is the sparse local HORN MNIST generator. It
starts from random oscillator state, settles a coupled position/velocity field,
and decodes the final oscillator features into an image.

Run the current research recipe on real MNIST:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist
```

For a short local probe, override the preset:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist \
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

Important caveat: the leading preset can exploit a direct label-initialization
route. To probe the stricter route where class information has to enter through
dynamics instead:

```bash
python examples/image_mnist_generator.py --preset sparse_horn_mnist_class_coupling_strength8
python examples/image_mnist_generator.py --preset sparse_horn_mnist_state_mlp_class_coupling_strong
```

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
