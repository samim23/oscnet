"""Driven damped resonator filter banks (OscNet RFB frontends).

A linear driven damped harmonic oscillator (DHO)

    x'' + 2 * gamma * x' + omega^2 * x = alpha * I(t)

is a second-order IIR bandpass when discretized. A bank of uncoupled DHOs
with distinct natural frequencies performs short-time spectral decomposition
— an analysis frontend we call a **resonator filter bank** (RFB).

This module is library infrastructure, not an experiment script:

- closed-form discrete state-space filtering (no ODE unroll)
- log-spaced / constant-Q / equal-frequency schedules
- quadrature amplitude readout
- Equinox ``ResonatorBank`` module for frozen or learnable banks

Continuous convention matches the existing
``harmonic_oscillator_update`` gamma_factor = 2*gamma convention.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def log_spaced_omegas(
    num_bands: int,
    *,
    f_min_hz: float,
    f_max_hz: float,
    sample_rate: float,
) -> Array:
    """Return angular frequencies ``omega = 2 pi f`` log-spaced in Hz."""

    if num_bands < 1:
        raise ValueError("num_bands must be positive")
    if f_min_hz <= 0.0 or f_max_hz <= f_min_hz:
        raise ValueError("require 0 < f_min_hz < f_max_hz")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive")
    nyquist = 0.5 * float(sample_rate)
    f_max_hz = min(float(f_max_hz), 0.45 * float(sample_rate))
    if f_max_hz <= float(f_min_hz):
        raise ValueError(
            f"f_max_hz ({f_max_hz}) must exceed f_min_hz ({f_min_hz}) "
            f"after Nyquist clamping (Nyquist={nyquist})"
        )
    freqs = jnp.geomspace(
        float(f_min_hz), float(f_max_hz), int(num_bands), dtype=jnp.float32
    )
    return (2.0 * jnp.pi * freqs).astype(jnp.float32)


def equal_omegas(
    num_bands: int,
    *,
    f_hz: float,
    sample_rate: float,
) -> Array:
    """Return ``num_bands`` copies of the same angular frequency (control)."""

    if num_bands < 1:
        raise ValueError("num_bands must be positive")
    if f_hz <= 0.0:
        raise ValueError("f_hz must be positive")
    nyquist = 0.5 * float(sample_rate)
    if f_hz >= nyquist:
        raise ValueError(f"f_hz ({f_hz}) must be < Nyquist ({nyquist})")
    omega = jnp.asarray(2.0 * jnp.pi * float(f_hz), dtype=jnp.float32)
    return jnp.full((int(num_bands),), omega, dtype=jnp.float32)


def constant_q_gammas(omegas: Array, *, quality_factor: float) -> Array:
    """Damping ``gamma`` for constant-Q bands under ``x'' + 2 g x' + w^2 x``.

    With that convention, half-power bandwidth satisfies ``Q ≈ omega / (2 gamma)``,
    so ``gamma = omega / (2 Q)``.
    """

    if quality_factor <= 0.0:
        raise ValueError("quality_factor must be positive")
    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    return (omegas / (2.0 * float(quality_factor))).astype(jnp.float32)


def unit_peak_alphas(omegas: Array, gammas: Array) -> Array:
    """Per-band drive gains so resonant |H(jω)| ≈ 1.

    For ``H(s) = α / (s² + 2γs + ω²)``, on resonance ``|H(jω)| = α / (2γω)``.
    Setting ``α = 2γω`` yields unit peak gain independent of band center —
    required so audio-rate banks produce usable feature scales.
    """

    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    gammas = jnp.asarray(gammas, dtype=jnp.float32)
    return (2.0 * gammas * omegas).astype(jnp.float32)


def resonator_bank_schedule(
    num_bands: int,
    *,
    sample_rate: float = 16_000.0,
    f_min_hz: float = 100.0,
    f_max_hz: float = 4_000.0,
    quality_factor: float = 4.0,
    alpha: Optional[float] = None,
    spacing: str = "log",
    equal_f_hz: Optional[float] = None,
    unit_peak_gain: bool = True,
) -> Tuple[Array, Array, Array]:
    """Frozen analytic RFB schedule: ``(omegas, gammas, alphas)``.

    ``spacing='log'`` is the default cochlear-like bank. ``spacing='equal'`` is
    the capacity-vs-tuning control (all resonators share one frequency).
    ``f_max_hz`` is clamped below Nyquist automatically.

    By default each band gets a per-channel ``alpha`` so resonant gain is ~1
    (``unit_peak_gain=True``). Pass a scalar ``alpha`` to override.
    """

    f_max_hz = min(float(f_max_hz), 0.45 * float(sample_rate))
    if spacing == "log":
        omegas = log_spaced_omegas(
            num_bands,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            sample_rate=sample_rate,
        )
    elif spacing == "equal":
        center = (
            float(equal_f_hz)
            if equal_f_hz is not None
            else float((f_min_hz * f_max_hz) ** 0.5)
        )
        omegas = equal_omegas(
            num_bands, f_hz=center, sample_rate=sample_rate
        )
    else:
        raise ValueError("spacing must be 'log' or 'equal'")
    gammas = constant_q_gammas(omegas, quality_factor=quality_factor)
    if alpha is not None:
        alphas = jnp.full_like(omegas, float(alpha))
    elif unit_peak_gain:
        alphas = unit_peak_alphas(omegas, gammas)
    else:
        alphas = jnp.ones_like(omegas)
    return omegas, gammas, alphas


def dho_discrete_transition(
    omega: Array,
    gamma: Array,
    *,
    dt: float,
    alpha: Union[Array, float] = 1.0,
) -> Tuple[Array, Array]:
    """Exact ZOH state-space maps for the continuous DHO.

    State is ``[x, v]``. Returns ``(A, B)`` with shapes broadcastable to
    ``(..., 2, 2)`` and ``(..., 2)`` so

        state_{t+1} = A @ state_t + B * I_t

    Uses the closed-form matrix exponential of the 2×2 companion matrix
    (under-/critically-/over-damped), not an ODE stepper.
    ``alpha`` may be a scalar or per-band array matching ``omega``.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    omega = jnp.asarray(omega, dtype=jnp.float32)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.float32)
    alpha_arr = jnp.broadcast_to(alpha_arr, omega.shape)
    w2 = omega * omega
    g = gamma
    # Companion: [[0, 1], [-w2, -2g]]
    # Characteristic roots: -g ± sqrt(g^2 - w2)
    disc = g * g - w2
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    # Overdamped branch
    r1 = -g + sqrt_disc
    r2 = -g - sqrt_disc
    # Avoid division by zero when roots coincide (critically damped)
    denom = jnp.where(jnp.abs(r1 - r2) < 1e-6, jnp.ones_like(r1), r1 - r2)
    e1 = jnp.exp(r1 * dt)
    e2 = jnp.exp(r2 * dt)
    # Phi = exp(A dt) for distinct roots via spectral form; critically
    # damped limit handled below.
    # For distinct roots, A = U diag(r1,r2) U^{-1} with
    # columns of U = [1,1; r1,r2].
    a00_od = (r1 * e2 - r2 * e1) / denom
    a01_od = (e1 - e2) / denom
    a10_od = r1 * r2 * (e2 - e1) / denom
    a11_od = (r1 * e1 - r2 * e2) / denom

    # Underdamped: g^2 < w2, omega_d = sqrt(w2 - g^2)
    omega_d = jnp.sqrt(jnp.maximum(w2 - g * g, 0.0))
    exp_gt = jnp.exp(-g * dt)
    cos_t = jnp.cos(omega_d * dt)
    sin_t = jnp.sin(omega_d * dt)
    # Guard omega_d ~ 0
    safe_wd = jnp.where(omega_d < 1e-6, jnp.ones_like(omega_d), omega_d)
    a00_ud = exp_gt * (cos_t + g * sin_t / safe_wd)
    a01_ud = exp_gt * (sin_t / safe_wd)
    a10_ud = -exp_gt * (w2 * sin_t / safe_wd)
    a11_ud = exp_gt * (cos_t - g * sin_t / safe_wd)

    # Critically damped: g^2 == w2, exp(-g t) * [[1+g t, t], [-w2 t, 1-g t]]
    exp_c = jnp.exp(-g * dt)
    a00_cd = exp_c * (1.0 + g * dt)
    a01_cd = exp_c * dt
    a10_cd = -exp_c * w2 * dt
    a11_cd = exp_c * (1.0 - g * dt)

    over = disc > 1e-8
    under = disc < -1e-8
    a00 = jnp.where(over, a00_od, jnp.where(under, a00_ud, a00_cd))
    a01 = jnp.where(over, a01_od, jnp.where(under, a01_ud, a01_cd))
    a10 = jnp.where(over, a10_od, jnp.where(under, a10_ud, a10_cd))
    a11 = jnp.where(over, a11_od, jnp.where(under, a11_ud, a11_cd))

    # Stack A
    A = jnp.stack(
        [
            jnp.stack([a00, a01], axis=-1),
            jnp.stack([a10, a11], axis=-1),
        ],
        axis=-2,
    )

    # B = A^{-1} (exp(A dt) - I) b with b = [0, alpha]
    # For DHO, integral of e^{A tau} b d tau.
    # Closed form: for force on velocity, B = [b_x, b_v].
    # Use (A_cont)^{-1} (Phi - I) b when A invertible (w2 > 0).
    # A_cont^{-1} = [[-2g/w2, -1/w2], [1, 0]]
    # (Phi - I) @ [0, alpha] via linear algebra on the 2-vector.
    eye = jnp.eye(2, dtype=jnp.float32)
    # Broadcast eye to A shape
    eye = jnp.broadcast_to(eye, A.shape)
    phi_m_i = A - eye
    # A_cont^{-1} @ (Phi - I) @ [0, alpha]
    # Let v = (Phi - I) @ [0, alpha] = alpha * Phi[:, 1] column... 
    # (Phi - I) @ e1 where e1 = [0,1], so second column of (Phi-I) * alpha
    col = phi_m_i[..., :, 1] * alpha_arr[..., None]
    inv_a00 = -2.0 * g / jnp.maximum(w2, 1e-8)
    inv_a01 = -1.0 / jnp.maximum(w2, 1e-8)
    inv_a10 = jnp.ones_like(w2)
    inv_a11 = jnp.zeros_like(w2)
    bx = inv_a00 * col[..., 0] + inv_a01 * col[..., 1]
    bv = inv_a10 * col[..., 0] + inv_a11 * col[..., 1]
    B = jnp.stack([bx, bv], axis=-1)
    return A.astype(jnp.float32), B.astype(jnp.float32)


def quadrature_amplitude(x: Array, v: Array, omega: Array) -> Array:
    """Instantaneous amplitude ``A = sqrt(x^2 + (v/omega)^2)``."""

    omega = jnp.asarray(omega, dtype=jnp.float32)
    safe = jnp.maximum(jnp.abs(omega), 1e-6)
    return jnp.sqrt(x * x + (v / safe) * (v / safe))


def quadrature_phase_trig(x: Array, v: Array, omega: Array) -> Tuple[Array, Array]:
    """Unit-circle phase features ``(cos φ, sin φ)`` from ``(x, v/ω)``."""

    amp = quadrature_amplitude(x, v, omega) + 1e-6
    omega = jnp.asarray(omega, dtype=jnp.float32)
    safe = jnp.maximum(jnp.abs(omega), 1e-6)
    return x / amp, (v / safe) / amp


def _readout_feature_dim(num_bands: int, readout: str) -> int:
    n = int(num_bands)
    if readout == "amplitude":
        return n
    if readout == "phase":
        return 2 * n
    if readout == "both":
        return 3 * n
    raise ValueError(
        f"unknown readout {readout!r}; choose 'amplitude', 'phase', or 'both'"
    )


def _compose_band_readout(
    amp: Array,
    cos_phi: Array,
    sin_phi: Array,
    *,
    readout: str,
) -> Array:
    if readout == "amplitude":
        return amp
    if readout == "phase":
        return jnp.concatenate([cos_phi, sin_phi], axis=-1)
    if readout == "both":
        return jnp.concatenate([amp, cos_phi, sin_phi], axis=-1)
    raise ValueError(
        f"unknown readout {readout!r}; choose 'amplitude', 'phase', or 'both'"
    )


def _pool_envelope(amp: Array, *, pool: str, transient_fraction: float) -> Array:
    t_axis = amp.ndim - 2
    length = amp.shape[t_axis]
    skip = int(float(transient_fraction) * float(length))
    skip = min(max(skip, 0), max(length - 1, 0))
    amp = amp[..., skip:, :]
    if pool == "rms":
        return jnp.sqrt(jnp.mean(amp * amp, axis=-2) + 1e-8)
    if pool == "log_rms":
        return jnp.log(jnp.sqrt(jnp.mean(amp * amp, axis=-2) + 1e-8) + 1e-6)
    if pool == "mean":
        return jnp.mean(amp, axis=-2)
    if pool == "max":
        return jnp.max(amp, axis=-2)
    raise ValueError("pool must be 'rms', 'log_rms', 'mean', or 'max'")


def _pool_trig(
    cos_phi: Array,
    sin_phi: Array,
    *,
    transient_fraction: float,
) -> Tuple[Array, Array]:
    length = cos_phi.shape[-2]
    skip = int(float(transient_fraction) * float(length))
    skip = min(max(skip, 0), max(length - 1, 0))
    return (
        jnp.mean(cos_phi[..., skip:, :], axis=-2),
        jnp.mean(sin_phi[..., skip:, :], axis=-2),
    )


def _filter_one_channel(
    waveform: Array,
    A: Array,
    B: Array,
) -> Tuple[Array, Array]:
    """Filter a 1-D waveform with one resonator; return ``(x_t, v_t)``."""

    def step(state, u):
        x, v = state
        s = jnp.stack([x, v])
        s_next = A @ s + B * u
        return (s_next[0], s_next[1]), s_next

    (_, _), traj = jax.lax.scan(step, (0.0, 0.0), waveform)
    return traj[:, 0], traj[:, 1]


def filter_waveform(
    waveform: Array,
    omegas: Array,
    gammas: Array,
    *,
    sample_rate: float,
    alpha: Union[Array, float] = 1.0,
) -> Tuple[Array, Array]:
    """Drive an uncoupled resonator bank with a waveform.

    Args:
        waveform: ``(T,)`` or ``(batch, T)``
        omegas / gammas: ``(num_bands,)``
        sample_rate: Hz (sets ``dt = 1/sample_rate``)
        alpha: scalar or per-band drive gain

    Returns:
        ``(x, v)`` each ``(..., T, num_bands)``
    """

    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    gammas = jnp.asarray(gammas, dtype=jnp.float32)
    if omegas.shape != gammas.shape:
        raise ValueError("omegas and gammas must share shape")
    dt = 1.0 / float(sample_rate)
    A, B = dho_discrete_transition(omegas, gammas, dt=dt, alpha=alpha)

    def filter_bands(wave: Array) -> Tuple[Array, Array]:
        # wave: (T,) -> ((T, N), (T, N))
        def one(a, b):
            return _filter_one_channel(wave, a, b)

        xs, vs = jax.vmap(one, in_axes=(0, 0))(A, B)
        # xs: (N, T) -> (T, N)
        return xs.T, vs.T

    if waveform.ndim == 1:
        return filter_bands(waveform.astype(jnp.float32))
    if waveform.ndim == 2:
        return jax.vmap(filter_bands)(waveform.astype(jnp.float32))
    raise ValueError("waveform must be 1-D or 2-D (batch, time)")


def _filter_one_channel_agc(
    waveform: Array,
    omega: Array,
    gamma0: Array,
    alpha: Array,
    *,
    dt: float,
    agc_strength: float,
) -> Tuple[Array, Array]:
    """Euler DHO with amplitude-dependent damping (cochlear-style AGC)."""

    w = jnp.asarray(omega, dtype=jnp.float32)
    g0 = jnp.asarray(gamma0, dtype=jnp.float32)
    a = jnp.asarray(alpha, dtype=jnp.float32)
    w2 = w * w
    safe_w = jnp.maximum(jnp.abs(w), 1e-6)
    strength = float(agc_strength)

    def step(state, u):
        x, v = state
        amp = jnp.sqrt(x * x + (v / safe_w) * (v / safe_w))
        g = g0 * (1.0 + strength * amp)
        x_n = x + dt * v
        v_n = v + dt * (-2.0 * g * v - w2 * x + a * u)
        return (x_n, v_n), (x_n, v_n)

    (_, _), traj = jax.lax.scan(step, (0.0, 0.0), waveform.astype(jnp.float32))
    return traj[0], traj[1]


def filter_waveform_agc(
    waveform: Array,
    omegas: Array,
    gammas: Array,
    *,
    sample_rate: float,
    alpha: Union[Array, float] = 1.0,
    agc_strength: float = 2.0,
) -> Tuple[Array, Array]:
    """Time-stepped RFB with level-dependent damping."""

    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    gammas = jnp.asarray(gammas, dtype=jnp.float32)
    if jnp.ndim(alpha) == 0:
        alphas = jnp.full_like(omegas, float(alpha))
    else:
        alphas = jnp.asarray(alpha, dtype=jnp.float32)
    dt = 1.0 / float(sample_rate)

    def filter_bands(wave: Array) -> Tuple[Array, Array]:
        def one(w, g, a):
            return _filter_one_channel_agc(
                wave, w, g, a, dt=dt, agc_strength=agc_strength
            )

        xs, vs = jax.vmap(one)(omegas, gammas, alphas)
        return xs.T, vs.T

    if waveform.ndim == 1:
        return filter_bands(waveform)
    if waveform.ndim == 2:
        return jax.vmap(filter_bands)(waveform)
    raise ValueError("waveform must be 1-D or 2-D (batch, time)")


def apply_drive_nonlinearity(waveform: Array, nonlinearity: str) -> Array:
    """Optional compressive drive before the linear bank."""

    if nonlinearity in ("none", "linear", "envelope_soft", "agc"):
        return waveform
    if nonlinearity == "drive_tanh":
        return jnp.tanh(waveform.astype(jnp.float32))
    raise ValueError(
        f"unknown nonlinearity {nonlinearity!r}; "
        "choose 'none', 'drive_tanh', 'envelope_soft', or 'agc'"
    )


def compress_envelope(amp: Array, nonlinearity: str) -> Array:
    """Optional soft compression on instantaneous / pooled amplitude."""

    if nonlinearity in ("none", "linear", "drive_tanh", "agc"):
        return amp
    if nonlinearity == "envelope_soft":
        # Maps large envelopes toward saturation (cochlear-like compression).
        return amp / (1.0 + amp)
    raise ValueError(
        f"unknown nonlinearity {nonlinearity!r}; "
        "choose 'none', 'drive_tanh', 'envelope_soft', or 'agc'"
    )


def band_features(
    waveform: Array,
    omegas: Array,
    gammas: Array,
    *,
    sample_rate: float,
    alpha: Union[Array, float] = 1.0,
    pool: str = "log_rms",
    transient_fraction: float = 0.1,
    readout: str = "amplitude",
    nonlinearity: str = "none",
    agc_strength: float = 2.0,
) -> Array:
    """Pool resonator envelopes into a ``(..., F)`` feature vector.

    Default ``pool='log_rms'`` matches mel/STFT log-energy frontends.
    ``readout`` selects amplitude (``F=N``), phase ``(cos,sin)`` (``F=2N``),
    or both (``F=3N``).
    """

    wave = apply_drive_nonlinearity(waveform, nonlinearity)
    if nonlinearity == "agc":
        x, v = filter_waveform_agc(
            wave,
            omegas,
            gammas,
            sample_rate=sample_rate,
            alpha=alpha,
            agc_strength=agc_strength,
        )
    else:
        x, v = filter_waveform(
            wave,
            omegas,
            gammas,
            sample_rate=sample_rate,
            alpha=alpha,
        )
    amp = compress_envelope(
        quadrature_amplitude(x, v, omegas), nonlinearity
    )
    cos_phi, sin_phi = quadrature_phase_trig(x, v, omegas)
    amp_feat = _pool_envelope(
        amp, pool=pool, transient_fraction=transient_fraction
    )
    cos_feat, sin_feat = _pool_trig(
        cos_phi, sin_phi, transient_fraction=transient_fraction
    )
    return _compose_band_readout(
        amp_feat, cos_feat, sin_feat, readout=readout
    )


def band_features_freq_domain(
    waveform: Array,
    omegas: Array,
    gammas: Array,
    *,
    sample_rate: float,
    alpha: Union[Array, float] = 1.0,
    readout: str = "amplitude",
    nonlinearity: str = "none",
) -> Array:
    """Differentiable DHO band features via analytic ``H(jω)`` on an FFT.

    Used for learnable ``{ω, Q}`` — avoids unstable BPTT through the IIR state.
    Amplitude uses the stable log band-power form; phase / both use the
    complex response ``Y = Σ H(jω) X(ω)``.
    ``agc`` is not supported here (requires time-stepped damping).
    """

    if nonlinearity == "agc":
        raise ValueError(
            "agc nonlinearity requires the time-domain path; "
            "use a frozen ResonatorBank or nonlinearity='drive_tanh'/'envelope_soft'"
        )

    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    gammas = jnp.asarray(gammas, dtype=jnp.float32)
    if jnp.ndim(alpha) == 0:
        alphas = jnp.full_like(omegas, float(alpha))
    else:
        alphas = jnp.asarray(alpha, dtype=jnp.float32)

    wave = apply_drive_nonlinearity(waveform, nonlinearity).astype(jnp.float32)
    n = wave.shape[-1]
    window = jnp.hanning(n).astype(jnp.float32)
    w = (2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=1.0 / float(sample_rate))).astype(
        jnp.float32
    )
    w0 = omegas[:, None]
    g = gammas[:, None]
    a = alphas[:, None]
    ww = w[None, :]

    if readout == "amplitude":
        # Original Stage-2 path: |H|²-weighted power spectrum (stable + matched).
        spec = jnp.abs(jnp.fft.rfft(wave * window, n=n, axis=-1)) ** 2
        denom = (w0 * w0 - ww * ww) ** 2 + (2.0 * g * ww) ** 2
        weights = (a * a) / jnp.maximum(denom, 1e-12)
        weights = weights / jnp.maximum(
            jnp.max(weights, axis=-1, keepdims=True), 1e-12
        )
        band_power = jnp.sum(spec[..., None, :] * weights, axis=-1)
        amp = jnp.sqrt(band_power + 1e-8)
        amp = compress_envelope(amp, nonlinearity)
        return jnp.log(amp + 1e-6)

    X = jnp.fft.rfft(wave * window, n=n, axis=-1)
    denom = (w0 * w0 - ww * ww) + 1j * (2.0 * g * ww)
    H = a / jnp.where(jnp.abs(denom) < 1e-12, 1e-12 + 0j, denom)
    H = H / jnp.maximum(jnp.max(jnp.abs(H), axis=-1, keepdims=True), 1e-12)
    Y = jnp.einsum("...f,nf->...n", X, H)
    mag = jnp.abs(Y) + 1e-6
    amp_feat = jnp.log(compress_envelope(mag, nonlinearity) + 1e-6)
    cos_feat = jnp.real(Y) / mag
    sin_feat = jnp.imag(Y) / mag
    return _compose_band_readout(
        amp_feat, cos_feat, sin_feat, readout=readout
    )


def band_feature_frames(
    waveform: Array,
    omegas: Array,
    gammas: Array,
    *,
    sample_rate: float,
    num_frames: int,
    alpha: Union[Array, float] = 1.0,
    transient_fraction: float = 0.1,
    readout: str = "amplitude",
    nonlinearity: str = "none",
    agc_strength: float = 2.0,
) -> Array:
    """Time-varying RFB features ``(..., num_frames, F)``.

    Splits the post-transient envelope into ``num_frames`` equal blocks and
    pools each block — the AudioPrism-style drive into a recurrent head.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    wave = apply_drive_nonlinearity(waveform, nonlinearity)
    if nonlinearity == "agc":
        x, v = filter_waveform_agc(
            wave,
            omegas,
            gammas,
            sample_rate=sample_rate,
            alpha=alpha,
            agc_strength=agc_strength,
        )
    else:
        x, v = filter_waveform(
            wave,
            omegas,
            gammas,
            sample_rate=sample_rate,
            alpha=alpha,
        )
    amp = compress_envelope(
        quadrature_amplitude(x, v, omegas), nonlinearity
    )
    cos_phi, sin_phi = quadrature_phase_trig(x, v, omegas)
    length = amp.shape[-2]
    skip = int(float(transient_fraction) * float(length))
    skip = min(max(skip, 0), max(length - 1, 0))
    amp = amp[..., skip:, :]
    cos_phi = cos_phi[..., skip:, :]
    sin_phi = sin_phi[..., skip:, :]
    t = amp.shape[-2]
    frame_len = max(t // int(num_frames), 1)
    usable = frame_len * int(num_frames)
    amp = amp[..., :usable, :]
    cos_phi = cos_phi[..., :usable, :]
    sin_phi = sin_phi[..., :usable, :]
    leading = amp.shape[:-2]
    bands = amp.shape[-1]
    nf = int(num_frames)
    amp_f = amp.reshape(leading + (nf, frame_len, bands))
    cos_f = cos_phi.reshape(leading + (nf, frame_len, bands))
    sin_f = sin_phi.reshape(leading + (nf, frame_len, bands))
    amp_feat = jnp.log(jnp.sqrt(jnp.mean(amp_f * amp_f, axis=-2) + 1e-8) + 1e-6)
    cos_feat = jnp.mean(cos_f, axis=-2)
    sin_feat = jnp.mean(sin_f, axis=-2)
    return _compose_band_readout(
        amp_feat, cos_feat, sin_feat, readout=readout
    )


class ResonatorBank(eqx.Module):
    """Uncoupled driven-DHO filter bank (RFB frontend primitive).

    Frozen by default. Set ``learnable=True`` to expose softplus-parameterized
    log-frequencies and log-Q for learnable-tuning extensions.
    """

    omegas: Array
    gammas: Array
    alphas: Array
    sample_rate: float = eqx.field(static=True)
    pool: str = eqx.field(static=True)
    transient_fraction: float = eqx.field(static=True)
    learnable: bool = eqx.field(static=True)
    readout: str = eqx.field(static=True)
    nonlinearity: str = eqx.field(static=True)
    agc_strength: float = eqx.field(static=True)
    # Learnable residual params (used only when learnable=True)
    log_omega_base: Optional[Array] = None
    log_q_base: Optional[Array] = None
    delta_log_omega: Optional[Array] = None
    delta_log_q: Optional[Array] = None

    def __init__(
        self,
        *,
        num_bands: int = 16,
        sample_rate: float = 16_000.0,
        f_min_hz: float = 100.0,
        f_max_hz: float = 4_000.0,
        quality_factor: float = 4.0,
        alpha: Optional[float] = None,
        spacing: str = "log",
        equal_f_hz: Optional[float] = None,
        pool: str = "log_rms",
        transient_fraction: float = 0.1,
        learnable: bool = False,
        unit_peak_gain: bool = True,
        readout: str = "amplitude",
        nonlinearity: str = "none",
        agc_strength: float = 2.0,
    ):
        omegas, gammas, alphas = resonator_bank_schedule(
            num_bands,
            sample_rate=sample_rate,
            f_min_hz=f_min_hz,
            f_max_hz=f_max_hz,
            quality_factor=quality_factor,
            alpha=alpha,
            spacing=spacing,
            equal_f_hz=equal_f_hz,
            unit_peak_gain=unit_peak_gain,
        )
        self.omegas = omegas
        self.gammas = gammas
        self.alphas = alphas
        self.sample_rate = float(sample_rate)
        self.pool = str(pool)
        self.transient_fraction = float(transient_fraction)
        self.learnable = bool(learnable)
        self.readout = str(readout)
        self.nonlinearity = str(nonlinearity)
        self.agc_strength = float(agc_strength)
        _readout_feature_dim(num_bands, self.readout)  # validate
        apply_drive_nonlinearity(jnp.zeros((1,), dtype=jnp.float32), self.nonlinearity)
        if self.learnable:
            q = omegas / (2.0 * jnp.maximum(gammas, 1e-8))
            # Residual parameterization: keeps init = cochlear schedule and
            # bounds how far bands can move (avoids NaN BPTT through IIR ω).
            self.log_omega_base = jnp.log(jnp.maximum(omegas, 1e-6))
            self.log_q_base = jnp.log(jnp.maximum(q, 1e-6))
            self.delta_log_omega = jnp.zeros_like(omegas)
            self.delta_log_q = jnp.zeros_like(omegas)
        else:
            self.log_omega_base = None
            self.log_q_base = None
            self.delta_log_omega = None
            self.delta_log_q = None

    @property
    def num_bands(self) -> int:
        return int(self.omegas.shape[0])

    @property
    def feature_dim(self) -> int:
        return _readout_feature_dim(self.num_bands, self.readout)

    @property
    def _use_freq_domain(self) -> bool:
        # AGC needs time-stepped damping; otherwise learnable banks stay on
        # the stable freq-domain path.
        return bool(self.learnable) and self.nonlinearity != "agc"

    def effective_omegas_gammas_alphas(self) -> Tuple[Array, Array, Array]:
        if (
            not self.learnable
            or self.log_omega_base is None
            or self.delta_log_omega is None
            or self.log_q_base is None
            or self.delta_log_q is None
        ):
            return self.omegas, self.gammas, self.alphas
        # Bounded residual around the analytic schedule (±~factor e^0.5 ≈ 1.65).
        log_w = jax.lax.stop_gradient(self.log_omega_base) + 0.5 * jnp.tanh(
            self.delta_log_omega
        )
        log_q = jax.lax.stop_gradient(self.log_q_base) + 0.5 * jnp.tanh(
            self.delta_log_q
        )
        order = jnp.argsort(log_w)
        omegas = jnp.exp(log_w)[order]
        q = jnp.exp(log_q)[order]
        nyquist = jnp.pi * float(self.sample_rate)
        omegas = jnp.clip(omegas, 1e-3, 0.9 * nyquist)
        q = jnp.clip(q, 0.5, 64.0)
        gammas = omegas / (2.0 * jnp.maximum(q, 1e-6))
        alphas = unit_peak_alphas(omegas, gammas)
        return omegas, gammas, alphas

    def encode(
        self,
        waveform: Array,
        *,
        max_samples: Optional[int] = None,
    ) -> Array:
        """Map waveform ``(..., T)`` to band features ``(..., num_bands)``.

        If ``max_samples`` is set and ``T`` is larger, the waveform is
        uniformly subsampled (and ``sample_rate`` scaled) so long clips stay
        cheap — useful for Speech Commands at 16 kHz.

        Learnable banks use a frequency-domain DHO power response (stable
        grads w.r.t. ``{ω,Q}``). Frozen banks keep the time-domain IIR path.
        """

        omegas, gammas, alphas = self.effective_omegas_gammas_alphas()
        sample_rate = self.sample_rate
        if max_samples is not None and waveform.shape[-1] > int(max_samples):
            t = int(waveform.shape[-1])
            idx = jnp.linspace(0, t - 1, int(max_samples)).astype(jnp.int32)
            waveform = waveform[..., idx]
            sample_rate = self.sample_rate * (
                float(max_samples) / float(t)
            )
        if self._use_freq_domain:
            return band_features_freq_domain(
                waveform,
                omegas,
                gammas,
                sample_rate=sample_rate,
                alpha=alphas,
                readout=self.readout,
                nonlinearity=self.nonlinearity,
            )
        return band_features(
            waveform,
            omegas,
            gammas,
            sample_rate=sample_rate,
            alpha=alphas,
            pool=self.pool,
            transient_fraction=self.transient_fraction,
            readout=self.readout,
            nonlinearity=self.nonlinearity,
            agc_strength=self.agc_strength,
        )

    def encode_frames(
        self,
        waveform: Array,
        *,
        num_frames: int,
        max_samples: Optional[int] = None,
    ) -> Array:
        """Map waveform to ``(..., num_frames, feature_dim)`` RFB frames."""

        omegas, gammas, alphas = self.effective_omegas_gammas_alphas()
        sample_rate = self.sample_rate
        if max_samples is not None and waveform.shape[-1] > int(max_samples):
            t = int(waveform.shape[-1])
            idx = jnp.linspace(0, t - 1, int(max_samples)).astype(jnp.int32)
            waveform = waveform[..., idx]
            sample_rate = self.sample_rate * (float(max_samples) / float(t))
        if self._use_freq_domain:
            # Approximate frames by splitting the waveform, then freq-domain RFB.
            t_len = waveform.shape[-1]
            frame_len = max(t_len // int(num_frames), 1)
            usable = frame_len * int(num_frames)
            wave = waveform[..., :usable]
            leading = wave.shape[:-1]
            frames = wave.reshape(leading + (int(num_frames), frame_len))

            def one_frame(frame: Array) -> Array:
                return band_features_freq_domain(
                    frame,
                    omegas,
                    gammas,
                    sample_rate=sample_rate,
                    alpha=alphas,
                    readout=self.readout,
                    nonlinearity=self.nonlinearity,
                )

            # frames: (num_frames, L) or (B, num_frames, L)
            if frames.ndim == 2:
                return jax.vmap(one_frame)(frames)
            return jax.vmap(jax.vmap(one_frame))(frames)
        return band_feature_frames(
            waveform,
            omegas,
            gammas,
            sample_rate=sample_rate,
            num_frames=int(num_frames),
            alpha=alphas,
            transient_fraction=self.transient_fraction,
            readout=self.readout,
            nonlinearity=self.nonlinearity,
            agc_strength=self.agc_strength,
        )

    def filter(self, waveform: Array) -> Tuple[Array, Array]:
        """Return full ``(x, v)`` trajectories ``(..., T, num_bands)``."""

        omegas, gammas, alphas = self.effective_omegas_gammas_alphas()
        return filter_waveform(
            waveform,
            omegas,
            gammas,
            sample_rate=self.sample_rate,
            alpha=alphas,
        )


def band_collapse_metrics(omegas: Array) -> dict:
    """Diagnostics for learnable banks: tiny gaps ⇒ collapsed bands."""

    omegas = jnp.asarray(omegas, dtype=jnp.float32)
    omegas = jnp.sort(omegas)
    n = int(omegas.shape[0])
    if n < 2:
        return {
            "min_freq_ratio": 1.0,
            "mean_log_gap": 0.0,
            "n_near_duplicate": 0,
        }
    ratios = omegas[1:] / jnp.maximum(omegas[:-1], 1e-8)
    log_gaps = jnp.log(jnp.maximum(ratios, 1e-8))
    near = int(jnp.sum(ratios < 1.05))
    return {
        "min_freq_ratio": float(jnp.min(ratios)),
        "mean_log_gap": float(jnp.mean(log_gaps)),
        "n_near_duplicate": near,
    }


def band_spacing_regularizer(
    omegas: Array,
    *,
    min_ratio: float = 1.12,
) -> Array:
    """Hinge + pairwise log-gap + log-uniform spread to discourage collapse."""

    omegas = jnp.sort(jnp.asarray(omegas, dtype=jnp.float32))
    n = omegas.shape[0]
    ratios = omegas[1:] / jnp.maximum(omegas[:-1], 1e-8)
    hinge = jnp.mean(jnp.square(jnp.maximum(0.0, float(min_ratio) - ratios)))
    log_w = jnp.log(jnp.maximum(omegas, 1e-8))
    ideal = jnp.linspace(log_w[0], log_w[-1], log_w.shape[0])
    spread = jnp.mean(jnp.square(log_w - ideal))
    # Pairwise soft repulsion in log-frequency (catches non-adjacent collapse).
    diffs = jnp.abs(log_w[:, None] - log_w[None, :])
    mask = 1.0 - jnp.eye(n, dtype=log_w.dtype)
    target = jnp.log(jnp.asarray(float(min_ratio), dtype=log_w.dtype))
    pairwise = jnp.sum(
        mask * jnp.square(jnp.maximum(0.0, target - diffs))
    ) / jnp.maximum(float(n * (n - 1)), 1.0)
    return hinge + 0.1 * spread + 0.5 * pairwise
