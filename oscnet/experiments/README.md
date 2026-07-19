# OscNet Experiments

Importable training / evaluation harnesses that put `oscnet.models` on concrete
tasks.

| Layer | Role |
| --- | --- |
| `examples/` | Thin CLI entrypoints (start here to run something) |
| `oscnet/experiments/` | Harness code you can import from Python |
| `oscnet/models/` | Reusable oscillator architectures |
| `docs/paper_rfb_horn/` | Spoken-digit RFB→HORN paper |
| `docs/experiment_ledgers/` | Long research logs, Modal notes, frontier probes |
| `scripts/modal_*.py` | Optional remote GPU launchers |
| `outputs/` | Run artifacts |

Runnable commands and many generator quickstarts also live in
`examples/README.md`. **This file is a harness index**, not the place for
multi-page frontier archaeology — that belongs in the ledgers.

## Harness menu

| Harness | Package / module | Entrypoint |
| --- | --- | --- |
| Spoken-digit RFB→HORN | `audio_digit/` | `python examples/audio_digit_classification.py --help` |
| Audio wavelet autoencoder | `audio_wavelet.py` | `python examples/audio_wavelet_oscillatory_autoencoder.py --help` |
| Image generator (HORN / Kuramoto / controls) | `mnist_generator/` | `python examples/image_mnist_generator.py --help` |
| MNIST oscillatory autoencoder | `mnist_autoencoder.py` | `python examples/image_mnist_oscillatory_autoencoder.py --help` |
| MNIST phase VAE | `mnist_phase_vae.py` | `python examples/image_mnist_phase_vae.py --help` |
| MNIST phase-flow | `mnist_phase_flow.py` | `python examples/image_mnist_phase_flow.py --help` |
| MNIST shape→pixel | `mnist_shape_pixel.py` | `python examples/image_mnist_shape_pixel.py --help` |

Shared utilities: `harness.py`, `results.py`.

## When to use which

- **Spoken-digit RFB→HORN** — oscillators as audio frontend + sequential head
  (Speech Commands digits; mel / GRU / MLP controls). Primary encoding study.
- **Audio wavelet autoencoder** — reconstruct wavelet feature *sequences*
  (orthogonal to the spoken-digit stack).
- **Image generator** — class-conditioned HORN / StateMLP generation on MNIST,
  Fashion-MNIST, CIFAR. Default preset is MNIST-friendly; CIFAR RGB is the
  harder mechanism gate. Preset names and frontier claims → ledgers /
  `examples/README.md`.
- **MNIST autoencoder** — stable reconstruction benchmark.
- **Phase VAE** — simple generative train-smoke with phase dynamics in the
  latent.
- **Phase-flow / shape→pixel** — oscillators as the generative medium /
  two-stage shape scaffold experiments.

## Spoken-digit RFB → HORN

```bash
python examples/audio_digit_classification.py --frontend resonator
python examples/audio_digit_classification.py --sweep --epochs 20

modal run scripts/modal_audio_digit.py --sweep-preset rfb_plus_controls
modal run scripts/modal_audio_digit.py --sweep-preset rfb_plus_isoparam
```

Paper: `docs/paper_rfb_horn/`.  
Recipe / settled insights:
`docs/experiment_ledgers/resonator_filter_bank_frontend_plan.md`.  
Summaries: `outputs/analysis/modal_audio_digit_*.csv`.

**Next project (planned):** oscillatory speech/music generation — Stage 0
next-frame latent probe. Plan:
`docs/experiment_ledgers/audio_generation_plan.md` (package not scaffolded yet).

## Image generator (short)

```bash
python examples/image_mnist_generator.py
# equivalent: --preset sparse_horn_mnist_recommended

python examples/image_mnist_generator.py \
  --preset sparse_horn_cifar10_rgb_current_multimode2_retinotopic_anchor030_prior_class_patch005
```

Keep matched controls (`state_mlp_*`, frozen / decoder-only / step1) next to
any claim. For Modal generator sweeps see
`docs/experiment_ledgers/modal_runs.md` and `scripts/modal_mnist_generator.py`.
Long-form CIFAR/MNIST frontier narrative:
`docs/experiment_ledgers/experiment_report.md`.

## Other quick entrypoints

```bash
python examples/image_mnist_oscillatory_autoencoder.py --data-source synthetic --epochs 1
python examples/audio_wavelet_oscillatory_autoencoder.py --help
python examples/image_mnist_phase_flow.py --help
python examples/image_mnist_shape_pixel.py --help
python examples/image_mnist_phase_vae.py --help
```

## Python usage

```python
from oscnet.experiments.audio_digit import (
    AudioDigitConfig,
    run_audio_digit_experiment,
)
from oscnet.experiments import (
    MNISTPhaseFlowExperimentConfig,
    run_mnist_phase_flow_experiment,
)
```

## Comparing results

Completed runs write `metrics/summary.json`. Compare from Python:

```python
from pathlib import Path
from oscnet.experiments.results import (
    collect_experiment_summaries,
    format_comparison_table,
)

rows = collect_experiment_summaries(Path("outputs/reference"))
print(format_comparison_table(rows))
```

Or: `python examples/compare_experiment_results.py`.

## Inspecting traces

Saved `traces/*.npz` files can be turned into a browsable report via
`oscnet.inspection` (PNG panels + `index.html` with tabs / phase scrubber):

```bash
python examples/inspect_trace.py path/to/run_or_trace.npz --open
```

`audio_digit` is not yet on this path — it saves metrics JSON only, not
oscillator-state NPZs.

## GPU / Modal

Optional. Cap containers unless you want many GPUs:

```bash
OSCNET_MODAL_MAX_CONTAINERS=3 modal run scripts/modal_audio_digit.py \
  --sweep-preset rfb_plus_controls
```

Catalog of presets and historical runs:
`docs/experiment_ledgers/modal_runs.md`.
