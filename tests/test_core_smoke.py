import jax
import jax.numpy as jnp

import oscnet
from oscnet.core import HierarchicalCouplingLayer, NonlinearHarmonicOscillator


def test_top_level_import_exposes_core_oscillator():
    assert oscnet.__version__ == "0.1.0"
    assert oscnet.NonlinearHarmonicOscillator is NonlinearHarmonicOscillator


def test_readme_core_example_runs():
    key = jax.random.PRNGKey(0)
    oscillator_key, coupling_key = jax.random.split(key)

    oscillator = NonlinearHarmonicOscillator(dim=64, key=oscillator_key)
    coupling = HierarchicalCouplingLayer(hidden_dim=64, depth=1, key=coupling_key)

    x = jnp.zeros(64)
    v = jnp.zeros(64)
    inputs = jnp.ones(64) * 0.1

    x_new, v_new = oscillator.step(x, v, inputs)
    coupled = coupling(inputs[None, :])

    assert x_new.shape == (64,)
    assert v_new.shape == (64,)
    assert coupled.shape == (1, 64)
    assert jnp.all(jnp.isfinite(x_new))
    assert jnp.all(jnp.isfinite(v_new))
    assert jnp.all(jnp.isfinite(coupled))
