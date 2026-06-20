"""ODE helpers for oscillator analysis modules."""

from typing import Any, Optional

import diffrax
import jax.numpy as jnp


def _as_input_force(args: Any, state_half_shape):
    if isinstance(args, dict):
        args = args.get("input_force", 0.0)
    if args is None:
        args = 0.0
    return jnp.broadcast_to(jnp.asarray(args), state_half_shape)


def _vector_field_from_step(oscillator, t, y, args):
    if y.shape[0] % 2 != 0:
        raise ValueError(
            "Step-based oscillator ODE fallback requires an even state vector "
            "containing concatenated position and velocity."
        )

    half = y.shape[0] // 2
    x, v = y[:half], y[half:]
    inputs = _as_input_force(args, x.shape)
    x_next, v_next = oscillator.step(x, v, inputs)
    dt = getattr(oscillator, "dt", 1.0)
    return jnp.concatenate([(x_next - x) / dt, (v_next - v) / dt])


def solve_ode(
    oscillator,
    y0: jnp.ndarray,
    t_start: float,
    t_end: float,
    *,
    saveat: Optional[diffrax.SaveAt] = None,
    args: Any = None,
    solver=None,
    dt0: Optional[float] = None,
    stepsize_controller=None,
    max_steps: int = 16**4,
):
    """
    Solve an oscillator ODE with Diffrax.

    Oscillators with a `vector_field(t, y, args)` method use it directly. For
    step-only second-order oscillators, this falls back to an approximate vector
    field over concatenated `[x, v]` state.
    """
    if hasattr(oscillator, "vector_field"):
        vf = lambda t, y, vf_args: oscillator.vector_field(t, y, vf_args)
    else:
        vf = lambda t, y, vf_args: _vector_field_from_step(oscillator, t, y, vf_args)

    return diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver=solver or diffrax.Tsit5(),
        t0=t_start,
        t1=t_end,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat or diffrax.SaveAt(t1=True),
        stepsize_controller=stepsize_controller
        or diffrax.PIDController(rtol=1e-6, atol=1e-6),
        max_steps=max_steps,
    )


__all__ = ["solve_ode"]
