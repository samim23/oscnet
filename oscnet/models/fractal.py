"""Fractal-coupled oscillatory model components."""

from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from oscnet.core.fractal_coupling import (
    DenseCouplingLayer,
    HierarchicalCouplingLayer,
)
from oscnet.core.oscillators import NonlinearHarmonicOscillator

CouplingLayer = Union[HierarchicalCouplingLayer, DenseCouplingLayer]


class FractalHORNCell(eqx.Module):
    """HORN-style oscillator cell with configurable recurrent coupling."""

    i2h: eqx.nn.Linear
    h2h: CouplingLayer
    h2o: eqx.nn.Linear
    oscillator: NonlinearHarmonicOscillator

    hidden_dim: int = eqx.field(static=True)
    gain_rec: float = eqx.field(static=True)
    coupling_kind: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        coupling_depth: int = 2,
        coupling_kind: str = "fractal_fixed",
        tonotopic_init_strength: float = 0.5,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 4)

        self.hidden_dim = hidden_dim
        self.gain_rec = 1.0 / float(jnp.sqrt(hidden_dim))
        self.coupling_kind = str(coupling_kind)
        self.i2h = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])
        kind = self.coupling_kind
        if kind in ("fractal_fixed", "fractal", "hierarchical"):
            self.h2h = HierarchicalCouplingLayer(
                hidden_dim=hidden_dim,
                depth=coupling_depth,
                initial_strength=self.gain_rec,
                key=keys[1],
            )
        elif kind == "dense":
            self.h2h = DenseCouplingLayer(
                hidden_dim=hidden_dim,
                initial_strength=self.gain_rec,
                tonotopic=False,
                key=keys[1],
            )
        elif kind in ("dense_tonotopic", "tonotopic"):
            self.h2h = DenseCouplingLayer(
                hidden_dim=hidden_dim,
                initial_strength=self.gain_rec,
                tonotopic=True,
                tonotopic_bias=float(tonotopic_init_strength),
                key=keys[1],
            )
        else:
            raise ValueError(
                f"unknown coupling_kind {coupling_kind!r}; expected "
                "'fractal_fixed', 'dense', or 'dense_tonotopic'"
            )
        self.h2o = eqx.nn.Linear(hidden_dim, output_dim, key=keys[2])
        self.oscillator = NonlinearHarmonicOscillator(dim=hidden_dim, key=keys[3])

    def __call__(
        self,
        inputs: jnp.ndarray,
        state: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        batch_size = inputs.shape[0]

        if state is None:
            x = jnp.zeros((batch_size, self.hidden_dim))
            v = jnp.zeros((batch_size, self.hidden_dim))
        else:
            x, v = state

        input_contrib = jax.vmap(self.i2h)(inputs)
        recurrent_contrib = self.h2h(v) * self.gain_rec
        total_input = input_contrib + recurrent_contrib
        new_x, new_v = jax.vmap(self.oscillator.step)(x, v, total_input)
        output = jax.vmap(self.h2o)(new_x)
        return output, (new_x, new_v)


__all__ = ["FractalHORNCell"]
