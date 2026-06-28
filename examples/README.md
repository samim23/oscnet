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

## Example Menu

| Example | What it runs |
| --- | --- |
| `image_mnist_oscillatory_autoencoder.py` | MNIST patch autoencoder reference benchmark. |
| `audio_wavelet_oscillatory_autoencoder.py` | Audio wavelet sequence autoencoder benchmark. |
| `image_mnist_jepa.py` | MNIST masked-representation prediction. |
| `image_mnist_kuramoto_generator.py` | Coupled-oscillator MNIST generator branch. |
| `image_mnist_phase_vae.py` | MNIST phase VAE generative baseline. |
| `image_mnist_phase_flow.py` | MNIST phase-rate rectified-flow sampler. |
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

Small phase VAE run:

```bash
python examples/image_mnist_phase_vae.py \
  --data-source synthetic \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4
```

## Where To Go Next

- `oscnet/experiments/README.md` explains the harnesses behind these scripts.
- `docs/model_api.md` explains reusable model classes.
- `docs/experiment_report.md` records research results and interpretation.
