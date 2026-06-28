# OscNet Experiments

This package contains reproducible experiment harnesses, not the core model
primitives. If you are new to the repo, start here after reading the top-level
`README.md` and `docs/model_api.md`.

The short version:

- `examples/` contains thin command-line entrypoints.
- `oscnet.models` contains reusable architecture pieces.
- `oscnet.experiments` contains training loops, configs, metrics, artifact
  generation, and benchmark-specific task logic.
- `docs/experiment_report.md` is the living research log. It records what
  worked, what failed, and why we changed direction.
- `outputs/` is generated locally and ignored by git.

## First-Time Path

For a quick smoke test, use synthetic data:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source synthetic \
  --epochs 1

python examples/image_mnist_phase_flow.py \
  --data-source synthetic \
  --epochs 1 \
  --field-channels 2 \
  --steps 1 \
  --eval-sample-count 4
```

For real MNIST runs, prefer the example entrypoints rather than calling
experiment modules directly:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source idx \
  --patch-size 7 \
  --decoder-mode positional \
  --epochs 10

python examples/image_mnist_phase_flow.py \
  --data-source idx \
  --model-family coarse_phase_flow \
  --epochs 10
```

GPU/Modal runners live in `scripts/`. Use them for longer sweeps, and keep
`OSCNET_MODAL_MAX_CONTAINERS=1` unless you intentionally want parallel GPU
jobs.

## Experiment Families

| Module | Role | Purpose |
| --- | --- | --- |
| `harness.py` | Shared infrastructure | Configs, output layout, checkpoints, metrics, plots, traces. |
| `mnist_autoencoder.py` | Reference benchmark | MNIST patch autoencoder using the reusable OscNet model API. |
| `audio_wavelet.py` | Reference benchmark | Audio/wavelet sequence autoencoder benchmark. |
| `mnist_jepa.py` | Masked representation task | MNIST patch prediction and inpainting-style probes, including Winfree and recurrent controls. |
| `mnist_generator.py` | Implicit generator task | Unpaired/feature-driven oscillator generator experiments inspired by coupled-oscillator image generation. |
| `mnist_phase_vae.py` | Paired generative baseline | Functional MNIST VAE with oscillator phase dynamics in the latent path. Useful for checking whether generator plumbing works. |
| `mnist_phase_flow.py` | Visible-field generator task | Rectified-flow generation where the noisy image itself is a phase-rate oscillator field. |
| `results.py` | Utility | Collect and compare completed experiment summaries. |

## Choosing an Experiment

Use the experiment family that matches the question you want to ask:

- Start with `mnist_autoencoder.py` or `audio_wavelet.py` when validating the
  stable model API and artifact pipeline.
- Use `mnist_phase_vae.py` when you want a conventional paired generative
  baseline that still routes information through oscillator phase dynamics.
- Use `mnist_generator.py` when investigating unpaired feature/objective design
  for oscillator generators.
- Use `mnist_jepa.py` when the task is spatial completion, masked prediction,
  or representation-level recovery.
- Use `mnist_phase_flow.py` when testing oscillators as the visible generative
  field rather than as a hidden latent transform.

The current research interpretation changes over time and intentionally lives
in `docs/experiment_report.md`. This README should remain a stable navigation
guide for the experiment package.

## Public Docs Policy

Keep the top-level `README.md` stable and user-facing:

- What OscNet is.
- How to install it.
- A tiny model example.
- Pointers to examples, model docs, and this experiment map.

Keep detailed benchmark tables and failed experiments out of the top-level
README. Put them in:

- `docs/experiment_report.md` for curated research conclusions.
- `docs/modal_runs.md` for remote-run commands and operational notes.
- `outputs/analysis/` for generated CSV/JSON artifacts.

`docs/experiment_report.md` is suitable for the public repo if it remains a
curated research log: no secrets, no raw private chat dumps, no huge generated
outputs, and clear labels for exploratory or negative results. If a note is too
messy or private, keep it outside the repo instead of committing it.

## Commit Hygiene

Before committing experiment work:

1. Keep generated artifacts out of git. `outputs/` is ignored and should stay
   local.
2. Update this file when adding a new experiment family.
3. Update `docs/model_api.md` when adding a reusable model.
4. Update `docs/experiment_report.md` when a run changes the research read.
5. Run focused tests for touched surfaces instead of the full slow suite unless
   the change is broad.
6. Run `git diff --check` before committing docs/code changes.

Suggested focused checks for this area:

```bash
pytest -q tests/test_mnist_phase_flow.py
pytest -q tests/test_mnist_phase_vae.py
pytest -q tests/test_mnist_generator.py
python -m py_compile oscnet/experiments/*.py scripts/modal_mnist_phase_flow.py
git diff --check
```
