"""Hybrid free-form + oscillator generator with configurable routing.

Constructive follow-up to the two-regime robustness result: free-form
recurrence wins inside the training envelope, physics-constrained oscillator
dynamics degrade more gracefully outside it. Nature keeps both upsides by
never running a pure oscillator field: precision lives in a free-form
pathway, cleanup/insurance in the oscillatory dynamics, and routing decides
which carries each piece of state.

Router modes
------------
- ``learned``: small MLP over state statistics (original Hybrid Frontier arm).
- ``fixed_statistic``: non-learned shift/typicality gate from local vs global
  energy — the Hybrid Frontier diagnosis that the learned router fails OOD
  because it is itself a free-form component trained inside the envelope.
- ``oracle``: uses an externally supplied per-site mask in ``[0, 1]`` (upper
  bound; set ``oracle_gate`` before stepping / recovery eval).
- ``free_form`` / ``oscillator``: constant gates for ablation ceilings.
"""

from __future__ import annotations

from .common import Array, Optional, Tuple, eqx, jax, jnp
from .multimode_horn import MultiModeHORNImageGenerator

_ROUTER_MODES = (
    "learned",
    "fixed_statistic",
    "oracle",
    "free_form",
    "oscillator",
)


class HybridImageGenerator(MultiModeHORNImageGenerator):
    """Multimode HORN oscillator path + free-form MLP path + router."""

    state_mlp_hidden_dim: int = eqx.field(static=True)
    state_mlp_depth: int = eqx.field(static=True)
    state_mlp_residual_scale: float = eqx.field(static=True)
    router_hidden_dim: int = eqx.field(static=True)
    router_mode: str = eqx.field(static=True)
    fixed_gate_scale: float = eqx.field(static=True)
    transition_layers: Tuple[eqx.nn.Linear, ...]
    router_layers: Tuple[eqx.nn.Linear, ...]
    router_bias: Array
    # Optional oracle / eval override: (batch, sites) in [0, 1], or None.
    oracle_gate: Optional[Array]

    def __init__(
        self,
        *,
        state_mlp_hidden_dim: int = 48,
        state_mlp_depth: int = 1,
        state_mlp_residual_scale: float = 0.1,
        router_hidden_dim: int = 16,
        router_bias_init: float = -1.0,
        router_mode: str = "learned",
        fixed_gate_scale: float = 4.0,
        **kwargs,
    ):
        if state_mlp_hidden_dim < 1:
            raise ValueError("state_mlp_hidden_dim must be positive")
        if state_mlp_depth < 0:
            raise ValueError("state_mlp_depth must be non-negative")
        if state_mlp_residual_scale <= 0.0:
            raise ValueError("state_mlp_residual_scale must be positive")
        if router_hidden_dim < 1:
            raise ValueError("router_hidden_dim must be positive")
        if router_mode not in _ROUTER_MODES:
            raise ValueError(
                f"router_mode must be one of {_ROUTER_MODES}, got {router_mode!r}"
            )
        if fixed_gate_scale <= 0.0:
            raise ValueError("fixed_gate_scale must be positive")

        key = kwargs.get("key", None)
        if key is None:
            key = jax.random.PRNGKey(42)
        base_key, transition_key, router_key = jax.random.split(key, 3)
        kwargs["key"] = base_key
        super().__init__(**kwargs)
        self.dynamics_family = "hybrid"
        self.state_mlp_hidden_dim = int(state_mlp_hidden_dim)
        self.state_mlp_depth = int(state_mlp_depth)
        self.state_mlp_residual_scale = float(state_mlp_residual_scale)
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_mode = str(router_mode)
        self.fixed_gate_scale = float(fixed_gate_scale)
        self.oracle_gate = None

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

        router_feature_dim = self._router_feature_dim()
        router_key_a, router_key_b = jax.random.split(router_key)
        self.router_layers = (
            eqx.nn.Linear(router_feature_dim, self.router_hidden_dim, key=router_key_a),
            eqx.nn.Linear(self.router_hidden_dim, 1, key=router_key_b),
        )
        self.router_bias = jnp.asarray([float(router_bias_init)], dtype=jnp.float32)

    def _router_feature_dim(self) -> int:
        return 2 * self.num_modes + 4

    def _trainable(self, tree):
        if self.train_recurrent_dynamics:
            return tree
        return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)

    def _free_form_delta(
        self, position: Array, velocity: Array
    ) -> Tuple[Array, Array]:
        hidden = self.state_features(position, velocity)
        layers = self._trainable(self.transition_layers)
        for layer_index, layer in enumerate(layers):
            hidden = jax.vmap(layer)(hidden)
            if layer_index < len(self.transition_layers) - 1:
                hidden = jax.nn.gelu(hidden)
        delta_position, delta_velocity = jnp.split(hidden, 2, axis=-1)
        return delta_position, delta_velocity

    def _site_energy_stats(
        self, position: Array, velocity: Array
    ) -> Tuple[Array, Array]:
        """Per-site and global mean |tanh(state)| energy."""

        batch = position.shape[0]
        site_position = jnp.tanh(position).reshape(
            batch, self.num_spatial_sites, self.num_modes
        )
        site_velocity = jnp.tanh(velocity).reshape(
            batch, self.num_spatial_sites, self.num_modes
        )
        site_energy = 0.5 * (
            jnp.mean(jnp.abs(site_position), axis=-1)
            + jnp.mean(jnp.abs(site_velocity), axis=-1)
        )
        global_energy = jnp.mean(site_energy, axis=-1, keepdims=True)
        return site_energy, global_energy

    def _learned_router_gate(self, position: Array, velocity: Array) -> Array:
        batch = position.shape[0]
        site_position = jnp.tanh(position).reshape(
            batch, self.num_spatial_sites, self.num_modes
        )
        site_velocity = jnp.tanh(velocity).reshape(
            batch, self.num_spatial_sites, self.num_modes
        )
        site_abs_position = jnp.mean(
            jnp.abs(site_position), axis=-1, keepdims=True
        )
        site_abs_velocity = jnp.mean(
            jnp.abs(site_velocity), axis=-1, keepdims=True
        )
        global_abs = jnp.mean(
            jnp.abs(jnp.tanh(position)), axis=-1, keepdims=True
        )
        global_std = jnp.std(jnp.tanh(position), axis=-1, keepdims=True)
        global_features = jnp.broadcast_to(
            jnp.concatenate([global_abs, global_std], axis=-1)[:, None, :],
            (batch, self.num_spatial_sites, 2),
        )
        features = jnp.concatenate(
            [
                site_position,
                site_velocity,
                site_abs_position,
                site_abs_velocity,
                global_features,
            ],
            axis=-1,
        )
        flat = features.reshape(batch * self.num_spatial_sites, -1)
        layers = self._trainable(self.router_layers)
        hidden = jax.nn.gelu(jax.vmap(layers[0])(flat))
        logits = jax.vmap(layers[1])(hidden).reshape(
            batch, self.num_spatial_sites
        )
        bias = self._trainable(self.router_bias)
        return jax.nn.sigmoid(logits + bias[0])

    def _fixed_statistic_gate(self, position: Array, velocity: Array) -> Array:
        """Non-learned gate: atypical local energy → trust oscillator.

        ``gate = sigmoid(scale * (site_energy - global_energy) + bias)`` with
        the same free-form-favoring bias init as the learned router.
        """

        site_energy, global_energy = self._site_energy_stats(position, velocity)
        residual = site_energy - global_energy
        bias = self.router_bias[0]
        return jax.nn.sigmoid(self.fixed_gate_scale * residual + bias)

    def router_gate(self, position: Array, velocity: Array) -> Array:
        """Per-site oscillator-trust gate in [0, 1], shape (batch, sites)."""

        batch = position.shape[0]
        sites = self.num_spatial_sites
        mode = self.router_mode
        if mode == "learned":
            return self._learned_router_gate(position, velocity)
        if mode == "fixed_statistic":
            return self._fixed_statistic_gate(position, velocity)
        if mode == "free_form":
            return jnp.zeros((batch, sites), dtype=position.dtype)
        if mode == "oscillator":
            return jnp.ones((batch, sites), dtype=position.dtype)
        if mode == "oracle":
            if self.oracle_gate is None:
                # Fail soft to fixed statistic rather than crash inside jit.
                return self._fixed_statistic_gate(position, velocity)
            gate = jnp.asarray(self.oracle_gate, dtype=position.dtype)
            if gate.shape != (batch, sites):
                raise ValueError(
                    f"oracle_gate shape {gate.shape} != {(batch, sites)}"
                )
            return jnp.clip(gate, 0.0, 1.0)
        raise ValueError(f"unknown router_mode {mode!r}")

    def with_oracle_gate(self, gate: Optional[Array]) -> "HybridImageGenerator":
        """Return a copy with ``oracle_gate`` set (eval / upper-bound runs)."""

        return eqx.tree_at(
            lambda m: m.oracle_gate,
            self,
            gate,
            is_leaf=lambda x: x is None,
        )

    def _blend_states(
        self,
        state: Tuple[Array, Array],
        free_form: Tuple[Array, Array],
        oscillator: Tuple[Array, Array],
    ) -> Tuple[Array, Array]:
        position, velocity = state
        gate = self.router_gate(position, velocity)
        gate = jnp.repeat(gate, self.num_modes, axis=-1)
        next_position = (
            (1.0 - gate) * free_form[0] + gate * oscillator[0]
        )
        next_velocity = (
            (1.0 - gate) * free_form[1] + gate * oscillator[1]
        )
        return self._bound_state(next_position), self._bound_state(next_velocity)

    def _free_form_next(
        self,
        state: Tuple[Array, Array],
        condition_drive: Array,
    ) -> Tuple[Array, Array]:
        position, velocity = state
        delta_position, delta_velocity = self._free_form_delta(position, velocity)
        scale = float(self.state_mlp_residual_scale)
        next_position = position + scale * delta_position
        next_velocity = (
            velocity
            + scale * delta_velocity
            + self.dt * float(self.coupling_strength) * condition_drive
        )
        return next_position, next_velocity

    def step_state(
        self,
        state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Advance one routed hybrid step for a batch of states."""

        oscillator_next = super().step_state(state, labels)
        condition_drive = self._horn_static_conditioning_drive(state[0], labels)
        free_form_next = self._free_form_next(state, condition_drive)
        return self._blend_states(state, free_form_next, oscillator_next)

    def step_joint_state(
        self,
        state: Tuple[Array, Array],
        condition_state: Tuple[Array, Array],
        labels: Optional[Array] = None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Advance the routed main state; conditioning follows the HORN pool."""

        oscillator_next, next_condition_state = super().step_joint_state(
            state,
            condition_state,
            labels,
        )
        condition_drive = self._horn_dynamic_conditioning_drive(
            state[0],
            condition_state[0],
            labels,
        )
        free_form_next = self._free_form_next(state, condition_drive)
        blended = self._blend_states(state, free_form_next, oscillator_next)
        return blended, next_condition_state
