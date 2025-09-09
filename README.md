# OscNet - Oscillatory Neural Networks and Dynamical Systems

OscNet is a Python library built on JAX for simulating, analyzing, and training neural networks based on oscillatory dynamics. The library provides a flexible framework for working with coupled oscillator networks, continuous-time neural networks, and general dynamical systems.

## Features

- Various oscillator models (Van der Pol, FitzHugh-Nagumo, Kuramoto, Stuart-Landau)
- Network coupling mechanisms with customizable topologies
- ODE solvers with robust error handling
- Differentiable neural network layers based on oscillatory dynamics
- Analysis tools including Floquet theory for periodic orbits
- Visualization utilities for networks and dynamics

## Installation

```bash
pip install -e .
```

## Examples

See the `examples` directory for usage examples, including:
- Kuramoto oscillator networks
- Coupled Van der Pol oscillators
- MNIST classification with oscillatory neural networks
- Single oscillator analysis