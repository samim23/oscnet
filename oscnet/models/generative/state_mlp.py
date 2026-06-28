"""Non-oscillatory latent-state controls for HORN generators."""

from __future__ import annotations

from .common import Array, Optional, Tuple, eqx, jax, jnp
from .horn import HORNImageGenerator

class StateMLPImageGenerator(HORNImageGenerator):
    """Non-oscillatory residual latent-state generator control.

    The model keeps HORN's two-channel position/velocity state and readout
    surface, but replaces the second-order oscillator update with a small
    residual MLP transition. It is meant as a matched neural control for asking
    whether HORN's recurrent dynamics beat a conventional latent-state mapper
    under the same decoder and objective.
    """

    state_mlp_hidden_dim: int = eqx.field(static=True)
    state_mlp_depth: int = eqx.field(static=True)
    state_mlp_residual_scale: float = eqx.field(static=True)
    transition_layers: Tuple[eqx.nn.Linear, ...]

    def __init__(
        self,
        *,
        state_mlp_hidden_dim: int = 48,
        state_mlp_depth: int = 1,
        state_mlp_residual_scale: float = 0.1,
        **kwargs,
    ):
        if state_mlp_hidden_dim < 1:
            raise ValueError("state_mlp_hidden_dim must be positive")
        if state_mlp_depth < 0:
            raise ValueError("state_mlp_depth must be non-negative")
        if state_mlp_residual_scale <= 0.0:
            raise ValueError("state_mlp_residual_scale must be positive")

        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        base_key, transition_key = jax.random.split(key)
        kwargs["key"] = base_key
        super().__init__(
            horn_frequency=1.0,
            horn_damping=0.0,
            horn_nonlinearity=0.0,
            **kwargs,
        )
        self.dynamics_family = "state_mlp"
        self.coupling_profile = "none"
        self.state_mlp_hidden_dim = int(state_mlp_hidden_dim)
        self.state_mlp_depth = int(state_mlp_depth)
        self.state_mlp_residual_scale = float(state_mlp_residual_scale)

        feature_dim = 2 * self.num_oscillators
        layer_dims = [feature_dim]
        layer_dims.extend([self.state_mlp_hidden_dim] * self.state_mlp_depth)
        layer_dims.append(feature_dim)
        layer_keys = jax.random.split(transition_key, len(layer_dims) - 1)
        self.transition_layers = tuple(
            eqx.nn.Linear(in_size, out_size, key=layer_key)
            for in_size, out_size, layer_key in zip(
                layer_dims[:-1],
                layer_dims[1:],
                layer_keys,
            )
        )

    def _transition_layers(self) -> Tuple[eqx.nn.Linear, ...]:
        if self.train_recurrent_dynamics:
            return self.transition_layers
        return jax.tree_util.tree_map(jax.lax.stop_gradient, self.transition_layers)

    def coupling_profile_matrix(self) -> Array:
        """Return a zero profile; this control has no oscillator coupling."""

        return jnp.zeros(
            (self.num_oscillators, self.num_oscillators),
            dtype=jnp.float32,
        )

    def _state_mlp_delta(self, position: Array, velocity: Array) -> Tuple[Array, Array]:
        hidden = self.state_features(position, velocity)
        for layer_index, layer in enumerate(self._transition_layers()):
            hidden = jax.vmap(layer)(hidden)
            if layer_index < len(self.transition_layers) - 1:
                hidden = jax.nn.gelu(hidden)
        delta_position, delta_velocity = jnp.split(hidden, 2, axis=-1)
        return delta_position, delta_velocity

    def step_state(
        self,
        state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance one non-oscillatory residual latent-state step."""

        del labels
        position, velocity = state
        delta_position, delta_velocity = self._state_mlp_delta(position, velocity)
        scale = float(self.state_mlp_residual_scale)
        next_position = self._bound_state(position + scale * delta_position)
        next_velocity = self._bound_state(velocity + scale * delta_velocity)
        return next_position, next_velocity

    def step_joint_state(
        self,
        state: Tuple[Array, Array],
        condition_state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Advance the main state and pass conditioning state through."""

        return self.step_state(state, labels), condition_state
