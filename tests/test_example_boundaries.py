import ast
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.models import FractalHORNCell


def test_primary_examples_do_not_define_model_classes():
    for path in [
        Path("examples/image_mnist_oscillatory_autoencoder.py"),
        Path("examples/audio_wavelet_oscillatory_autoencoder.py"),
        Path("examples/fractal/code/integration_example.py"),
    ]:
        tree = ast.parse(path.read_text())
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert class_names == []


def test_fractal_horn_cell_lives_in_models_and_runs():
    model = FractalHORNCell(
        input_dim=4,
        hidden_dim=8,
        output_dim=4,
        coupling_depth=1,
        key=jax.random.PRNGKey(0),
    )
    output, state = model(jnp.ones((2, 4)))
    params = eqx.filter(model, eqx.is_array)

    assert output.shape == (2, 4)
    assert state[0].shape == (2, 8)
    assert model.h2h.coupling_matrix.shape == (8, 8)
    assert sum(x.size for x in jax.tree.leaves(params)) > 0
