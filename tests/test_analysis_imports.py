import importlib

import diffrax
import jax.numpy as jnp
import numpy as np

from oscnet.analysis.phase_synchrony import (
    circular_difference,
    local_group_order,
    phase_order_parameter,
)
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


def test_phase_synchrony_order_parameters_detect_local_groups():
    theta = np.zeros((1, 1, 4, 1), dtype=np.float32)
    theta[..., 2:, :] = np.pi

    global_order = phase_order_parameter(theta, axis=(-2, -1))
    local_order = local_group_order(theta, grid_shape=(2, 2), group_size=1)
    column_order = local_group_order(theta, grid_shape=(2, 2), group_size=2)
    wrapped = circular_difference(np.array([np.pi]), np.array([-np.pi]))

    assert np.allclose(global_order, 0.0, atol=1e-6)
    assert np.allclose(local_order, 1.0, atol=1e-6)
    assert np.allclose(column_order, 0.0, atol=1e-6)
    assert np.allclose(wrapped, 0.0, atol=1e-6)


def test_phase_synchrony_local_groups_support_padded_edges():
    theta = np.zeros((1, 1, 9, 1), dtype=np.float32)

    order = local_group_order(theta, grid_shape=(3, 3), group_size=2)

    assert order.shape == (1, 1, 2, 2)
    assert np.allclose(order, 1.0, atol=1e-6)
