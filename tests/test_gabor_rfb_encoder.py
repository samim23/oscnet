"""Tests for Gabor RFB and image resonator encoders (Stage 0b)."""

import jax
import jax.numpy as jnp

from oscnet.core import GaborFilterBank
from oscnet.models.generative.resonator_encoder import build_image_rfb_encoder


def test_gabor_bank_encode_shape():
    bank = GaborFilterBank(num_orientations=2, num_scales=2, kernel_size=7)
    images = jax.random.normal(jax.random.PRNGKey(0), (3, 16, 16))
    feats = bank.encode(images)
    assert feats.shape == (3, bank.num_bands)
    assert jnp.all(jnp.isfinite(feats))


def test_gabor_state_encoder_matches_oscillator_count():
    enc = build_image_rfb_encoder(
        drive="gabor",
        num_oscillators=8,
        image_shape=(8, 8, 1),
        key=jax.random.PRNGKey(1),
    )
    images = jax.random.uniform(jax.random.PRNGKey(2), (2, 64))
    pos, vel = enc(images)
    assert pos.shape == (2, 8)
    assert vel.shape == (2, 8)
    assert jnp.all(jnp.isfinite(pos))


def test_row_scan_encoder_smoke():
    enc = build_image_rfb_encoder(
        drive="row_scan",
        num_oscillators=4,
        image_shape=(8, 8),
        num_bands=6,
        key=jax.random.PRNGKey(3),
    )
    images = jax.random.uniform(jax.random.PRNGKey(4), (2, 64))
    pos, vel = enc(images)
    assert pos.shape == (2, 4)
    assert vel.shape == (2, 4)
