# OscNet

A JAX library for oscillatory neural networks and dynamical systems.

## Overview

OscNet provides a framework for building and training neural networks based on oscillatory dynamics — coupled oscillator networks, continuous-time neural networks, and general dynamical systems. Built on JAX and Equinox for differentiable, high-performance computation.

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
- `image_mnist_oscillatory_autoencoder.py` — MNIST autoencoder with oscillatory dynamics
- `audio_wavelet_oscillatory_autoencoder.py` — Audio processing
- `resonanceDB.py` — Phase-aware similarity store
- `fractal/` — Hierarchical coupling experiments

## License

MIT
