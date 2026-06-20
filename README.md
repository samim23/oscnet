# OscNet

A JAX library for oscillatory neural networks and dynamical systems.

## Overview

OscNet provides a framework for building and training neural networks based on oscillatory dynamics — coupled oscillator networks, continuous-time neural networks, and general dynamical systems. Built on JAX and Equinox for differentiable, high-performance computation.

## Features

- **Oscillator models**: Harmonic, Van der Pol, Stuart-Landau, Kuramoto, FitzHugh-Nagumo
- **Coupling topologies**: Hierarchical fractal, power-law, log-periodic
- **Analysis tools**: Edge-of-chaos, Floquet analysis, bifurcation, stability
- **Training utilities**: Criticality initialization, stochastic forcing, schedulers
- **Visualization**: Phase space, network dynamics, oscillator analysis

## Installation

```bash
pip install -e .
```

**Requirements**: JAX, Equinox, Optax, Diffrax, NumPy, Matplotlib

## Examples

See `examples/` for usage:
- `image_mnist_oscillatory_autoencoder.py` — MNIST reference benchmark
- `audio_wavelet_oscillatory_autoencoder.py` — Audio wavelet reference benchmark
- `resonanceDB.py` — Phase-aware similarity store
- `fractal/` — Hierarchical coupling experiments

## Quick Example

```python
import jax
from oscnet.core import (
    NonlinearHarmonicOscillator,
    HierarchicalCouplingLayer,
)

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 2)

# Create oscillator bank
oscillator = NonlinearHarmonicOscillator(dim=64, key=keys[0])

# Create hierarchical fractal coupling (improves memory tasks)
coupling = HierarchicalCouplingLayer(hidden_dim=64, depth=1, key=keys[1])

# Step dynamics
x, v = jax.numpy.zeros(64), jax.numpy.zeros(64)
inputs = jax.numpy.ones(64) * 0.1
x_new, v_new = oscillator.step(x, v, inputs)
```

## Model API

OscNet's reusable model spine lives in `oscnet.models`. It turns the core
oscillator primitives into sequence layers, encoders, decoders, and complete
autoencoders, so examples can stay focused on data and experiments.

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

For image-style patch workflows, use `PatchOscillatoryAutoencoder`. For
patch reconstruction, `decoder_mode="positional"` gives the decoder an explicit
patch-position signal. For sequence generation, set
`decoder_mode="autoregressive"`. For audio/wavelet feature sequences, use
`WaveletOscillatoryAutoencoder`.

See `docs/model_api.md` for the full model spine, tensor conventions, and
extension points.

## Reference Benchmarks

The first canonical experiment harness lives in `oscnet.experiments`. It gives
MNIST and audio wavelet runs the same CLI/config flow, output layout,
checkpoint metadata, metrics, loss curves, reconstruction plots, and
latent/state traces.

```bash
python examples/image_mnist_oscillatory_autoencoder.py \
  --output-dir outputs/reference/mnist \
  --data-source idx \
  --patch-size 7 \
  --decoder-mode positional \
  --latent-conditioning-strength 2.0 \
  --hidden-dim 64 \
  --latent-dim 64 \
  --epochs 120 \
  --seed 42

python examples/audio_wavelet_oscillatory_autoencoder.py \
  --output-dir outputs/reference/audio_wavelet \
  --epochs 10 \
  --seed 42
```

Use `--mode eval --checkpoint path/to/model.eqx` to run evaluation through the
same artifact pipeline. For quick dependency-light smoke runs, both examples
also support synthetic data:

```bash
python examples/image_mnist_oscillatory_autoencoder.py --data-source synthetic --epochs 1
python examples/audio_wavelet_oscillatory_autoencoder.py --feature-source synthetic-features --epochs 1
```


## License

MIT
