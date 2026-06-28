# OscNet Experiments

This folder contains training and evaluation harnesses for OscNet models.

If `oscnet.models` is the library of reusable oscillator architectures, then
`oscnet.experiments` is where those models are put on concrete tasks: MNIST,
audio wavelets, masked prediction, and generative sampling.

Most users should run experiments through `examples/`, because those files are
thin command-line entrypoints around this package.

## Quick Start

Run a tiny CPU smoke test:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source synthetic \
  --epochs 1
```

Try the current MNIST oscillator field generator on synthetic data:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

Use `--help` on any example to see its full options:

```bash
python examples/image_mnist_phase_flow.py --help
```

By default, runs write checkpoints, metrics, plots, traces, and sample images
under `outputs/`.

## Experiment Menu

| Experiment | One-line idea | Start here |
| --- | --- | --- |
| MNIST autoencoder | Reconstruct MNIST patches with reusable OscNet autoencoders and matched baselines. | `python examples/image_mnist_oscillatory_autoencoder.py --help` |
| Audio wavelet autoencoder | Encode and reconstruct audio wavelet feature sequences with oscillatory dynamics. | `python examples/audio_wavelet_oscillatory_autoencoder.py --help` |
| MNIST masked prediction / JEPA | Predict missing or corrupted MNIST patch representations with Winfree and recurrent fields. | `python examples/image_mnist_jepa.py --help` |
| Kuramoto MNIST generator | Explore Un-0-style coupled-oscillator image generation objectives. | `python examples/image_mnist_kuramoto_generator.py --help` |
| MNIST phase VAE | A conventional paired VAE where the latent code passes through oscillator phase dynamics. | `python examples/image_mnist_phase_vae.py --help` |
| MNIST phase-flow sampler | Treat the noisy image itself as a phase-rate oscillator field trained with rectified flow. | `python examples/image_mnist_phase_flow.py --help` |

## Which One Should I Run?

Use **MNIST autoencoder** if you want the most stable reference benchmark.

Use **audio wavelet autoencoder** if your data is temporal or sequence-like.

Use **MNIST phase VAE** if you want a simple generative model that should train
without much drama.

Use **MNIST phase-flow** if you want the most direct "oscillators as the
generative medium" experiment.

Use **MNIST JEPA** if you care about masked image recovery, block occlusion, or
representation prediction.

Use **Kuramoto MNIST generator** if you want the more speculative coupled
oscillator generator branch.

## Common Run Patterns

Reference MNIST run:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source idx \
  --patch-size 7 \
  --decoder-mode positional \
  --epochs 10
```

Phase VAE smoke run:

```bash
python examples/image_mnist_phase_vae.py \
  --data-source synthetic \
  --epochs 1 \
  --train-limit 8 \
  --eval-limit 4
```

Phase-flow real MNIST probe:

```bash
python examples/image_mnist_phase_flow.py \
  --data-source idx \
  --model-family coarse_phase_flow \
  --epochs 10 \
  --train-limit 10000 \
  --eval-limit 1000
```

Masked-prediction help:

```bash
python examples/image_mnist_jepa.py --help
```

## Comparing Results

Every completed run writes a `metrics/summary.json`. To compare runs from
Python:

```python
from pathlib import Path
from oscnet.experiments.results import (
    collect_experiment_summaries,
    format_comparison_table,
)

rows = collect_experiment_summaries(Path("outputs/reference"))
print(format_comparison_table(rows))
```

Longer research notes and benchmark interpretations live in
`docs/experiment_report.md`.

## GPU Runs

Modal GPU launchers live in `scripts/`. They are optional; local CPU/GPU runs
continue to work through the examples above.

For Modal sweeps, cap parallelism unless you intentionally want multiple
remote GPUs:

```bash
OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_phase_flow.py \
  --sweep-preset mnist_phase_flow_coarse_global_probe
```
