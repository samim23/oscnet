"""Unit tests for the OscNet resonator filter bank (RFB)."""

import jax
import jax.numpy as jnp
import numpy as np

from oscnet.core import (
    ResonatorBank,
    band_features,
    filter_waveform,
    log_spaced_omegas,
    quadrature_amplitude,
    resonator_bank_schedule,
)


def test_log_spaced_omegas_monotonic_and_under_nyquist():
    omegas = log_spaced_omegas(8, f_min_hz=100.0, f_max_hz=3500.0, sample_rate=8000.0)
    assert omegas.shape == (8,)
    assert jnp.all(omegas[1:] > omegas[:-1])
    assert float(omegas[-1]) < jnp.pi * 8000.0


def test_equal_schedule_collapses_frequencies():
    omegas, gammas, alphas = resonator_bank_schedule(
        6,
        sample_rate=8000.0,
        spacing="equal",
        equal_f_hz=500.0,
    )
    assert omegas.shape == (6,)
    assert jnp.allclose(omegas, omegas[0])
    assert gammas.shape == (6,)
    assert alphas.shape == (6,)
    assert jnp.all(alphas > 0)


def test_filter_waveform_resonates_near_natural_frequency():
    sample_rate = 8000.0
    duration = 0.4
    t = jnp.arange(int(sample_rate * duration)) / sample_rate
    # Drive near 400 Hz; bank has a band there
    drive = jnp.sin(2.0 * jnp.pi * 400.0 * t)
    omegas, gammas, alphas = resonator_bank_schedule(
        12,
        sample_rate=sample_rate,
        f_min_hz=100.0,
        f_max_hz=3000.0,
        quality_factor=5.0,
        spacing="log",
    )
    feats = band_features(
        drive,
        omegas,
        gammas,
        sample_rate=sample_rate,
        alpha=alphas,
        pool="log_rms",
        transient_fraction=0.2,
    )
    assert feats.shape == (12,)
    assert jnp.all(jnp.isfinite(feats))
    # Peak band should be near 400 Hz (allow adjacent bin)
    target = 2.0 * jnp.pi * 400.0
    order = jnp.argsort(jnp.abs(omegas - target))
    peak = int(jnp.argmax(feats))
    assert peak in (int(order[0]), int(order[1]))


def test_quadrature_amplitude_constant_for_pure_tone_free_response():
    omega = 2.0 * jnp.pi * 200.0
    t = jnp.linspace(0.0, 0.05, 400)
    # Undriven free oscillation x = cos(wt), v = -w sin(wt) has A=1
    x = jnp.cos(omega * t)
    v = -omega * jnp.sin(omega * t)
    amp = quadrature_amplitude(x, v, omega)
    assert float(jnp.mean(jnp.abs(amp - 1.0))) < 1e-3


def test_resonator_bank_encode_batch():
    bank = ResonatorBank(
        num_bands=8,
        sample_rate=8000.0,
        f_min_hz=120.0,
        f_max_hz=3000.0,
        spacing="log",
    )
    key = jax.random.PRNGKey(0)
    waves = jax.random.normal(key, (4, 1600))
    feats = bank.encode(waves)
    assert feats.shape == (4, 8)
    assert jnp.all(jnp.isfinite(feats))


def test_filter_waveform_shapes_and_finite():
    omegas, gammas, _ = resonator_bank_schedule(4, sample_rate=8000.0)
    wave = jnp.sin(jnp.linspace(0, 40, 1000))
    x, v = filter_waveform(wave, omegas, gammas, sample_rate=8000.0)
    assert x.shape == (1000, 4)
    assert v.shape == (1000, 4)
    assert bool(np.all(np.isfinite(np.asarray(x))))
