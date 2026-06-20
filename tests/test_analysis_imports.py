import importlib

import diffrax
import jax.numpy as jnp

from oscnet.core.dynamics import solve_ode
from oscnet.core.oscillators import HORNOscillator


def test_stale_analysis_modules_import_cleanly():
    for module_name in [
        "oscnet.analysis.bifurcation",
        "oscnet.analysis.floquet",
        "oscnet.analysis.stability",
        "oscnet.visualization.visualize",
    ]:
        importlib.import_module(module_name)


def test_legacy_horn_oscillator_supports_analysis_contract():
    oscillator = HORNOscillator(alpha=0.1, omega=1.0, gamma=0.05, h=1.0)
    y0 = jnp.array([1.0, 0.0])

    vf = oscillator.vector_field(0.0, y0, {"input_force": 0.0})
    solution = solve_ode(
        oscillator,
        y0,
        t_start=0.0,
        t_end=0.1,
        saveat=diffrax.SaveAt(t1=True),
        dt0=0.01,
    )

    assert oscillator.state_dim == 2
    assert oscillator.params["omega"] == 1.0
    assert vf.shape == (2,)
    assert solution.ys.shape[-1] == 2
    assert jnp.all(jnp.isfinite(solution.ys))
