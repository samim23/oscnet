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

## Recommended Generator Workflow

The strongest end-to-end image-generation workflow is the **sparse local HORN
MNIST generator**:

```bash
python examples/image_mnist_generator.py
```

The example defaults to `sparse_horn_mnist_recommended`, a strict
class-coupled HORN recipe with sparse local coupling and higher HORN damping.

Why it is first in the queue:

- It is the strongest current OscNet-native generator result.
- It uses a sparse local second-order HORN field, not a dense all-to-all toy.
- It beats frozen, decoder-only, and one-step controls on semantic sample
  quality, and it preserves higher diversity than the matched StateMLP control.

What to keep honest:

- `sparse_horn_mnist` is the older polished recipe with a direct
  label-initialization route. It remains useful for comparison, but it is no
  longer the default entrypoint.
- `sparse_horn_mnist_strict` is the semantic/diversity reference:
  class information enters through a dynamic coupling route, starts near
  chance, and improves through settling without the direct initial-state label
  shift.
- `sparse_horn_mnist_quality` is the balanced
  quality/proximity variant of that strict HORN route. It adds a small
  distributional pressure while preserving the same dynamic class-coupling
  path.
- `sparse_horn_mnist_dynamics_quality` is the current dynamics-side quality
  variant. It increases HORN damping, keeps the strict route, and improves
  nearest-real proximity without using extra distributional loss. The
  recommended preset currently points at these settings.
- `sparse_horn_mnist_dynamics_quality_dist001`,
  `sparse_horn_mnist_dynamics_quality_dist0025`, and
  `sparse_horn_mnist_dynamics_quality_dist005` test whether the damping gain
  compounds with the small distributional regularizer. The current read keeps
  `sparse_horn_mnist_dynamics_quality` as the default; the distributional
  variants are secondary probes.
- `sparse_horn_mnist_recommended_no_main_coupling`,
  `sparse_horn_mnist_recommended_frozen_recurrent`,
  `sparse_horn_mnist_recommended_frozen_conditioning`,
  `sparse_horn_mnist_recommended_frozen`,
  `sparse_horn_mnist_recommended_decoder_only`, and
  `sparse_horn_mnist_recommended_step1` are attribution controls for the
  recommended route. Current ablations say the sparse HORN substrate,
  multi-step settling, and learned conditioning drive are essential; freezing
  recurrent HORN parameters is much less damaging than removing coupling or
  freezing conditioning.
- `sparse_horn_mnist_step1` is a useful shortcut control for the older route.
- `sparse_horn_mnist_state_mlp_class_coupling_strong` is the matched
  non-oscillatory control for that stricter route.
- `sparse_horn_mnist_state_mlp_class_coupling_strength8` matches the latest
  HORN class-drive strength in the non-oscillatory transition control.
- `sparse_horn_mnist_state_mlp_class_coupling_strength8_dist005`,
  `sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01`, and
  `sparse_horn_mnist_state_mlp_class_coupling_strength8_dist01_class` are the
  fairer strength-8 diversity controls. So far they improve StateMLP
  pixel-proximity metrics but do not recover HORN-like diversity.
- `sparse_horn_mnist_state_mlp_class_coupling_strong_dist005`,
  `sparse_horn_mnist_state_mlp_class_coupling_strong_dist01`, and
  `sparse_horn_mnist_state_mlp_class_coupling_strong_dist01_class` are
  diversity-regularized controls. Use them before claiming that HORN's higher
  diversity comes specifically from oscillator dynamics.
- For generated-label metrics, prefer sweeps that set
  `--quality-classifier-train-limit` above the generator `--train-limit`; the
  strict HORN audit uses 500 generator examples but trains the judge on 5000.
- To compare finished generator sweeps as a quality/diversity frontier, run:

  ```bash
  python scripts/analyze_mnist_generator_frontier.py
  ```

  The script writes a compact CSV, Markdown table, and PNG plot under
  `outputs/analysis/mnist_generator_frontier/`. Frontier variants are
  non-dominated over generated-label accuracy, diversity ratio, and
  nearest-real MSE, with a default generated-label accuracy floor of `0.99`.
- To move the same frontier check beyond handwritten digits, use the
  Fashion-MNIST presets:

  ```bash
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_recommended
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_state_mlp_strength8
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_state_mlp_strength8_dist005
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_recommended_ch16
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_state_mlp_strength8_ch16
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_recommended_dist0025
  python examples/image_mnist_generator.py --preset sparse_horn_fashion_mnist_recommended_dist005
  ```

  Or run the compact Modal sweep:

  ```bash
  OSCNET_MODAL_MAX_CONTAINERS=1 modal run scripts/modal_mnist_generator.py \
    --sweep-preset mnist_generator_fashion_mnist_frontier_probe
  ```

  Fashion-MNIST uses the same 28x28 grayscale/10-class shape as MNIST, so it is
  the first clean stress test for whether HORN's diversity/settling frontier
  survives outside handwritten digits. The current three-seed result preserves
  the same tradeoff: HORN is the high-diversity point, while the matched
  StateMLP controls win generated-label accuracy and nearest-real MSE. The
  `ch16` presets are readout-capacity probes, useful for checking whether
  HORN's quality deficit is mainly a small decoder/readout bottleneck. The
  current result says wider readout improves HORN semantic accuracy and
  diversity, but does not close the nearest-real MSE gap.
  The `dist*` HORN presets test explicit calibration pressure as the next
  quality path. Current read: `sparse_horn_fashion_mnist_recommended_dist005`
  is the calibrated HORN quality variant; StateMLP strength-8 remains the raw
  pixel-proximity control.
- The next scale gate is CIFAR-10 grayscale:

  ```bash
  python examples/image_mnist_generator.py --preset sparse_horn_cifar10_gray_recommended
  python examples/image_mnist_generator.py --preset sparse_horn_cifar10_gray_recommended_dist005
  python examples/image_mnist_generator.py --preset sparse_horn_cifar10_gray_state_mlp_strength8
  ```

  Or run the Modal frontier sweep. Use parallelism deliberately; the current
  workspace budget can handle up to eight Modal GPU containers:

  ```bash
  OSCNET_MODAL_MAX_CONTAINERS=8 modal run scripts/modal_mnist_generator.py \
    --sweep-preset mnist_generator_cifar10_gray_frontier_probe
  ```

  This keeps the current single-channel HORN generator surface but moves from
  28x28 symbolic/silhouette data to 32x32 natural-image classes. Current
  result: HORN still sits on the higher-diversity/stronger-settling side of
  the frontier, while StateMLP remains the raw pixel-proximity and throughput
  control. The quick CIFAR-gray judge is weak, and samples remain blurry, so
  treat this as a transfer signal rather than a solved CIFAR generator.
- Detailed results and caveats live in `docs/experiment_report.md`.

## Harness Menu

| Harness | One-line idea | Usual entrypoint |
| --- | --- | --- |
| MNIST autoencoder | Reconstruct MNIST patches with reusable OscNet autoencoders and matched baselines. | `python examples/image_mnist_oscillatory_autoencoder.py --help` |
| Audio wavelet autoencoder | Encode and reconstruct audio wavelet feature sequences with oscillatory dynamics. | `python examples/audio_wavelet_oscillatory_autoencoder.py --help` |
| MNIST masked representation | Exploratory JEPA-lite benchmark for predicting hidden patch features with Winfree and recurrent controls. | `python examples/image_mnist_jepa.py --help` |
| Oscillator MNIST generator | Explore coupled-oscillator image generation with Kuramoto and HORN dynamics. Sparse local HORN is the strongest current generator branch. | `python examples/image_mnist_generator.py --help` |
| MNIST phase VAE | A conventional paired VAE where the latent code passes through oscillator phase dynamics. | `python examples/image_mnist_phase_vae.py --help` |
| MNIST phase-flow sampler | Treat the noisy image itself as a phase-rate oscillator field trained with rectified flow. | `python examples/image_mnist_phase_flow.py --help` |
| MNIST shape-to-pixel renderer | Render pixels from a clamped signed-distance shape scaffold with phase-flow dynamics and recurrent controls. | `python examples/image_mnist_shape_pixel.py --help` |

## Choosing a Harness

Use **MNIST autoencoder** if you want a stable reference benchmark.

Use **audio wavelet autoencoder** if your data is temporal or sequence-like.

Use **MNIST phase VAE** if you want a simple generative model that should train
without much drama.

Use **Oscillator MNIST generator** if you want the current strongest
oscillatory generator result. The example defaults to
`sparse_horn_mnist_recommended`: a sparse local `HORNImageGenerator` with
resize-conv readout, strict dynamic class coupling, higher HORN damping, and
variable-depth settling. Keep the matched controls nearby when turning it into
a claim:
`sparse_horn_mnist_frozen`, `sparse_horn_mnist_decoder_only`,
`sparse_horn_mnist_state_mlp`, `sparse_horn_mnist_state_mlp_frozen`,
`sparse_horn_mnist_state_mlp_decoder_only`, and `sparse_horn_mnist_step1`.
Use `sparse_horn_mnist_strict` when probing the semantic/diversity reference
without extra damping. Use `sparse_horn_mnist_quality` when you want the same
route with a small quality/proximity regularizer. Use
`sparse_horn_mnist_dynamics_quality` when testing whether the improvement can
come from oscillator dynamics rather than the loss. Use the
`dynamics_quality_dist*` presets to test whether those two quality directions
compound. Compare against
`sparse_horn_mnist_state_mlp_class_coupling_strong`,
`sparse_horn_mnist_state_mlp_class_coupling_strength8`, plus the
`state_mlp_class_coupling_strength8_dist*` presets when testing whether the
diversity/quality frontier is HORN-specific. The latest StateMLP strength-8
controls match or nearly match HORN on generated-label accuracy and win
nearest-real MSE, but still have lower diversity; the HORN claim is therefore a
diversity/settling claim, not a raw pixel-proximity claim.

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
`--train-noise-mode mixed` to train rectified-flow chords from a per-example
mixture of Gaussian, uniform, salt-pepper, and zero endpoints. Set
`--basin-t-values 0.1,0.25,0.5,0.75,0.9` to measure endpoint recovery from
partially real chord states; the harness reports before/after paired error so
you can see whether the dynamics actually improve the state. Use
`--basin-noise-mode uniform`, `salt_pepper`, or `zeros` to test non-Gaussian
basin endpoints. Use
`--basin-noise-modes uniform,salt_pepper,zeros` to evaluate several basin
endpoints after one training run.

Use **MNIST shape-to-pixel** if you want the next two-stage experiment after
the basin probe: the signed-distance field is treated as a fixed scaffold, and
the model learns a pixel rectified-flow conditioned on that scaffold. This is
the clean test of "oscillators settle shape; a second field renders pixels."
Set `--shape-condition-t-values 0.1,0.5,0.9` and
`--shape-condition-noise-modes uniform,salt_pepper,zeros` to test whether the
renderer tolerates imperfect shape scaffolds instead of only oracle
signed-distance maps. Set `--sample-readout-mode shape_gated` to make the
signed-distance scaffold explicitly gate the sampled pixel amplitude.

Use **MNIST masked representation** if you care about masked image recovery,
block occlusion, or representation-prediction controls. This branch is useful
for comparing recurrent and oscillatory predictors on partial-observation
tasks.

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
