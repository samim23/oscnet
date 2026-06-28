# Experiment Harnesses

This package contains importable training and evaluation harnesses for OscNet
models.

If `oscnet.models` is the library of reusable oscillator architectures, then
`oscnet.experiments` is where those models are put on concrete tasks: MNIST,
audio wavelets, masked prediction, and generative sampling.

Most users should run the top-level scripts in `examples/`. Those scripts are
thin command-line entrypoints around this package. Use this README when you want
to understand what each harness does, import a harness from Python, or compare
finished runs.

## Folder Roles

```text
examples/              runnable command-line entrypoints
oscnet/experiments/    importable experiment harnesses and result utilities
oscnet/models/         reusable model classes and oscillator layers
docs/                  model docs, Modal notes, and research reports
outputs/               generated local run artifacts
```

For runnable commands, start with `examples/README.md`.

## Harness Menu

| Harness | One-line idea | Usual entrypoint |
| --- | --- | --- |
| MNIST autoencoder | Reconstruct MNIST patches with reusable OscNet autoencoders and matched baselines. | `python examples/image_mnist_oscillatory_autoencoder.py --help` |
| Audio wavelet autoencoder | Encode and reconstruct audio wavelet feature sequences with oscillatory dynamics. | `python examples/audio_wavelet_oscillatory_autoencoder.py --help` |
| MNIST masked representation | Exploratory JEPA-lite benchmark for predicting hidden patch features with Winfree and recurrent controls. | `python examples/image_mnist_jepa.py --help` |
| Kuramoto MNIST generator | Explore Un-0-style coupled-oscillator image generation objectives. | `python examples/image_mnist_kuramoto_generator.py --help` |
| MNIST phase VAE | A conventional paired VAE where the latent code passes through oscillator phase dynamics. | `python examples/image_mnist_phase_vae.py --help` |
| MNIST phase-flow sampler | Treat the noisy image itself as a phase-rate oscillator field trained with rectified flow. | `python examples/image_mnist_phase_flow.py --help` |
| MNIST shape-to-pixel renderer | Render pixels from a clamped signed-distance shape scaffold with phase-flow dynamics and recurrent controls. | `python examples/image_mnist_shape_pixel.py --help` |

## Choosing a Harness

Use **MNIST autoencoder** if you want the most stable reference benchmark.

Use **audio wavelet autoencoder** if your data is temporal or sequence-like.

Use **MNIST phase VAE** if you want a simple generative model that should train
without much drama.

Use **MNIST phase-flow** if you want the most direct "oscillators as the
generative medium" experiment. Set `--target-representation sobel_edges` for
contour maps or `--target-representation signed_distance` for smooth shape
fields instead of raw pixels. Set
`--target-representation signed_distance_flow` for a three-channel potential
field where channel 0 is signed distance and channels 1-2 encode the local
x/y gradient direction. Use
`--target-representation pixels_signed_distance` to train a two-channel visible
field where channel 0 is pixel occupancy and channel 1 is an auxiliary smooth
shape field. Use `--target-representation centered_pixels_signed_distance` for
the same two-channel target in centered `[-1, 1]` coordinates, decoded back to
pixel space for sample metrics and PNG artifacts. Set
`--sample-readout-mode shape_gated` to multiply the decoded pixel channel by
the decoded shape channel during sample metrics/artifact export. Set
`--sample-schedule shape_guided` to let the shape channel update first and
open pixel-channel updates later during Euler sampling. Set
`--basin-t-values 0.1,0.25,0.5,0.75,0.9` to measure endpoint recovery from
partially real chord states; the harness reports before/after paired error so
you can see whether the dynamics actually improve the state.

Use **MNIST shape-to-pixel** if you want the next two-stage experiment after
the basin probe: the signed-distance field is treated as a fixed scaffold, and
the model learns a pixel rectified-flow conditioned on that scaffold. This is
the clean test of "oscillators settle shape; a second field renders pixels."

Use **MNIST masked representation** if you care about masked image recovery,
block occlusion, or representation-prediction controls. This branch is useful
for comparing recurrent and oscillatory predictors on partial-observation
tasks.

Use **Kuramoto MNIST generator** if you want the more speculative coupled
oscillator generator branch.

## Python Usage

The example scripts are convenient, but the harnesses can also be imported:

```python
from oscnet.experiments import (
    MNISTPhaseFlowExperimentConfig,
    MNISTShapePixelExperimentConfig,
    run_mnist_phase_flow_experiment,
    run_mnist_shape_pixel_experiment,
)
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
