# OscNet

A JAX library for oscillatory neural networks and dynamical systems.

## Overview

OscNet provides a framework for building and training neural networks based on
oscillatory dynamics: coupled oscillator networks, continuous-time neural
networks, and general dynamical systems. Built on JAX and Equinox for
differentiable, high-performance computation.

## What Is Included

- **Core oscillator primitives**: harmonic and nonlinear harmonic oscillators,
  Van der Pol, Stuart-Landau, Kuramoto, FitzHugh-Nagumo, and HORN-style
  second-order dynamics.
- **Coupling layers and fields**: hierarchical/fractal coupling, Winfree
  phase-field layers, dense coupling, distance-decay coupling, sparse local
  radius masks, and convolutional/adaptive neighborhood coupling.
- **Reusable model families**: oscillatory sequence autoencoders,
  patch-image and wavelet/audio autoencoders, Winfree spatial denoisers,
  Kuramoto/HORN image generators, phase-flow fields, and matched
  non-oscillatory controls.
- **Experiment harnesses**: MNIST reconstruction, masked completion, image
  generation, phase-flow probes, signed-distance shape experiments, and audio
  wavelet reconstruction.
- **Analysis and utilities**: phase synchrony, reconstruction diagnostics,
  edge-of-chaos, Floquet, bifurcation, stability tools, checkpointing, result
  comparison, and plotting helpers.

## Installation

```bash
pip install -e .
```

**Requirements**: JAX, Equinox, Optax, Diffrax, NumPy, Matplotlib

## Quick Example

```python
import jax
from oscnet.core import (
    NonlinearHarmonicOscillator,
    HierarchicalCouplingLayer,
)

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)

# Create oscillator bank.
oscillator = NonlinearHarmonicOscillator(dim=64, key=keys[0])

# Optional coupling layer for learned oscillator interactions.
coupling = HierarchicalCouplingLayer(hidden_dim=64, depth=1, key=keys[1])

# Step dynamics.
x, v = jax.numpy.zeros(64), jax.numpy.zeros(64)
inputs = jax.numpy.ones(64) * 0.1
x_new, v_new = oscillator.step(x, v, inputs)
```

## Model API

Reusable model classes live in `oscnet.models`. The examples use these classes
instead of defining architectures inline.

```python
import jax
import jax.numpy as jnp
from oscnet.models import OscillatoryAutoencoder

key = jax.random.PRNGKey(0)
model = OscillatoryAutoencoder(
    input_dim=16,
    hidden_dim=64,
    latent_dim=32,
    decoder_mode="repeat",
    key=key,
)

sequence = jnp.ones((49, 8, 16))  # time, batch, features
reconstruction = model(sequence)
```

See `docs/model_api.md` for model families, tensor conventions, controls, and
extension points.

## Examples and Experiments

Runnable scripts live in `examples/`; see `examples/README.md` for the command
menu. Importable training/evaluation harnesses live in `oscnet.experiments`;
see `oscnet/experiments/README.md` for the harness map and current experiment
presets.

For an end-to-end image-generation workflow, start with the sparse local HORN
generator. It is the most complete generator example in the repo: a coupled
oscillator field trained as an implicit image generator, not as an autoencoder.

| Workflow | Command | Use |
| --- | --- | --- |
| MNIST HORN | `python examples/image_mnist_generator.py` | Fastest friendly entrypoint; defaults to `sparse_horn_mnist_recommended`. |
| CIFAR-10 RGB HORN | `python examples/image_mnist_generator.py --preset sparse_horn_cifar10_rgb_current` | Current color-image frontier recipe. Slower, more interesting, and less polished than MNIST. |
| CIFAR-10 RGB hierarchy | `python examples/image_mnist_generator.py --preset sparse_horn_cifar10_rgb_hierarchy_lead` | Active multiscale mechanism lead for hierarchy probes. |

For explicit presets, matched controls, and attribution notes, see
`examples/README.md`.

Tiny smoke run:

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --data-source synthetic \
  --epochs 1
```

See available options:

```bash
python examples/image_mnist_phase_flow.py --help
```

Useful pointers:

- `examples/` for command-line entrypoints
- `oscnet/experiments/README.md` for experiment harnesses
- `docs/model_api.md` for reusable model families and tensor conventions

## License

MIT
