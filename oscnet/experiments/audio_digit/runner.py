"""Train / eval loop for the audio-digit RFB frontend probe."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import AudioDigitConfig
from .data import load_audio_digit_data
from oscnet.core.resonators import band_collapse_metrics, band_spacing_regularizer

from .frontends import build_frontend
from .models import (
    AudioDigitModel,
    DigitsClassifier,
    DigitsGRUClassifier,
    DigitsHornClassifier,
    count_trainable_params,
)
from .diagnostics import stage5_diagnostics

Array = jnp.ndarray


def _accuracy(logits: Array, labels: Array) -> Array:
    return jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32))


def _cross_entropy(logits: Array, labels: Array) -> Array:
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def _add_gaussian_noise(waveforms: Array, snr_db: float, *, key: jax.random.PRNGKey) -> Array:
    """Add white noise at approximate SNR (dB) relative to per-clip power."""

    signal_pow = jnp.mean(jnp.square(waveforms), axis=-1, keepdims=True)
    noise_pow = signal_pow / (10.0 ** (float(snr_db) / 10.0))
    noise = jax.random.normal(key, waveforms.shape) * jnp.sqrt(noise_pow + 1e-12)
    return waveforms + noise


def _scale_level(waveforms: Array, gain_db: float) -> Array:
    return waveforms * (10.0 ** (float(gain_db) / 20.0))


def _add_colored_noise(
    waveforms: Array,
    snr_db: float,
    *,
    key: jax.random.PRNGKey,
    beta: float = 1.0,
) -> Array:
    """Add 1/f^β noise (β=1 ≈ pink) at target SNR."""

    noise = jax.random.normal(key, waveforms.shape)
    n = waveforms.shape[-1]
    freqs = jnp.fft.rfftfreq(n).astype(jnp.float32)
    # DC bin → 0; other bins ~ 1/f^{β/2} in amplitude
    scale = jnp.where(
        freqs > 0,
        1.0 / jnp.power(jnp.maximum(freqs, 1e-6), 0.5 * float(beta)),
        0.0,
    )
    colored = jnp.fft.irfft(jnp.fft.rfft(noise, axis=-1) * scale, n=n, axis=-1)
    noise_pow = jnp.mean(jnp.square(colored), axis=-1, keepdims=True)
    signal_pow = jnp.mean(jnp.square(waveforms), axis=-1, keepdims=True)
    target = signal_pow / (10.0 ** (float(snr_db) / 10.0))
    colored = colored * jnp.sqrt((target + 1e-12) / (noise_pow + 1e-12))
    return waveforms + colored


def _add_bandlimited_noise(
    waveforms: Array,
    snr_db: float,
    *,
    key: jax.random.PRNGKey,
    sample_rate: float,
    f_low_hz: float = 300.0,
    f_high_hz: float = 3_000.0,
) -> Array:
    """Add white noise restricted to a mid speech band."""

    noise = jax.random.normal(key, waveforms.shape)
    n = waveforms.shape[-1]
    freqs = jnp.fft.rfftfreq(n, d=1.0 / float(sample_rate)).astype(jnp.float32)
    mask = ((freqs >= float(f_low_hz)) & (freqs <= float(f_high_hz))).astype(
        jnp.float32
    )
    filtered = jnp.fft.irfft(jnp.fft.rfft(noise, axis=-1) * mask, n=n, axis=-1)
    noise_pow = jnp.mean(jnp.square(filtered), axis=-1, keepdims=True)
    signal_pow = jnp.mean(jnp.square(waveforms), axis=-1, keepdims=True)
    target = signal_pow / (10.0 ** (float(snr_db) / 10.0))
    filtered = filtered * jnp.sqrt((target + 1e-12) / (noise_pow + 1e-12))
    return waveforms + filtered


def _apply_train_level_aug(
    waveforms: np.ndarray,
    *,
    aug_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if aug_db <= 0.0:
        return waveforms
    gains = rng.uniform(-float(aug_db), float(aug_db), size=(waveforms.shape[0], 1))
    return (waveforms * (10.0 ** (gains / 20.0))).astype(np.float32)


def _apply_train_noise_aug(
    waveforms: np.ndarray,
    *,
    sample_rate: float,
    prob: float,
    snr_db_min: float,
    snr_db_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly corrupt a train batch with white / pink / band noise."""

    if prob <= 0.0 or rng.random() > float(prob):
        return waveforms
    snr = float(rng.uniform(float(snr_db_min), float(snr_db_max)))
    kind = str(rng.choice(["white", "pink", "band"]))
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    xb = jnp.asarray(waveforms)
    if kind == "white":
        xb = _add_gaussian_noise(xb, snr, key=key)
    elif kind == "pink":
        xb = _add_colored_noise(xb, snr, key=key, beta=1.0)
    else:
        xb = _add_bandlimited_noise(
            xb, snr, key=key, sample_rate=float(sample_rate)
        )
    return np.asarray(xb, dtype=np.float32)


def build_model(
    config: AudioDigitConfig,
    *,
    key: jax.random.PRNGKey,
) -> AudioDigitModel:
    k_front, k_head = jax.random.split(key)
    frontend = build_frontend(config, key=k_front)
    learnable_fe = bool(config.learnable_frontend) or (
        config.frontend in ("conv1d", "resonator_learn")
    )
    freeze = not learnable_fe
    feature_dim = int(frontend.feature_dim)
    if config.head_kind == "mlp":
        return DigitsClassifier(
            frontend,
            feature_dim=feature_dim,
            hidden_dim=config.head_hidden_dim,
            num_classes=config.num_classes,
            freeze_frontend=freeze,
            key=k_head,
        )
    if config.head_kind == "horn":
        return DigitsHornClassifier(
            frontend,
            feature_dim=feature_dim,
            hidden_dim=config.head_hidden_dim,
            num_classes=config.num_classes,
            horn_steps=config.horn_steps,
            coupling_depth=config.horn_coupling_depth,
            coupling_kind=config.horn_coupling_kind,
            tonotopic_init_strength=config.horn_tonotopic_init_strength,
            freeze_frontend=freeze,
            key=k_head,
        )
    if config.head_kind == "gru":
        return DigitsGRUClassifier(
            frontend,
            feature_dim=feature_dim,
            hidden_dim=config.head_hidden_dim,
            num_classes=config.num_classes,
            freeze_frontend=freeze,
            fall_back_steps=config.horn_steps,
            key=k_head,
        )
    raise ValueError(f"unknown head_kind {config.head_kind!r}")


def run_audio_digit_experiment(
    config: AudioDigitConfig,
) -> Dict[str, Any]:
    """Run one frontend arm end-to-end; write metrics JSON under output_dir."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(int(config.seed))
    k_data, k_model, k_noise = jax.random.split(key, 3)

    data = load_audio_digit_data(config, key=k_data)
    model = build_model(config, key=k_model)
    learnable_fe = bool(config.learnable_frontend) or (
        config.frontend in ("conv1d", "resonator_learn")
    )
    trainable_n, frontend_n = count_trainable_params(model)
    reg_w = float(config.collapse_reg_weight)
    level_aug_db = float(config.train_level_aug_db)
    noise_aug_prob = float(config.train_noise_aug_prob)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    train_x = data["train_waveforms"]
    train_y = data["train_labels"]
    eval_x = data["eval_waveforms"]
    eval_y = data["eval_labels"]
    n_train = int(train_x.shape[0])
    batch = min(int(config.batch_size), n_train)

    @eqx.filter_jit
    def train_step(model, opt_state, xb, yb):
        def loss_fn(m):
            ce = _cross_entropy(m(xb), yb)
            if reg_w <= 0.0:
                return ce
            bank = getattr(getattr(m, "frontend", None), "bank", None)
            if bank is None or not getattr(bank, "learnable", False):
                return ce
            omegas, _, _ = bank.effective_omegas_gammas_alphas()
            return ce + reg_w * band_spacing_regularizer(omegas)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_step(model, xb, yb):
        logits = model(xb)
        return _cross_entropy(logits, yb), _accuracy(logits, yb)

    history = []
    t0 = time.time()
    for epoch in range(int(config.epochs)):
        rng = np.random.default_rng(config.seed + epoch)
        perm = rng.permutation(n_train)
        epoch_losses = []
        for start in range(0, n_train, batch):
            idx = perm[start : start + batch]
            if idx.shape[0] < 2:
                continue
            xb = np.asarray(train_x[idx])
            xb = _apply_train_level_aug(xb, aug_db=level_aug_db, rng=rng)
            xb = _apply_train_noise_aug(
                xb,
                sample_rate=float(config.sample_rate),
                prob=noise_aug_prob,
                snr_db_min=float(config.train_noise_snr_db_min),
                snr_db_max=float(config.train_noise_snr_db_max),
                rng=rng,
            )
            model, opt_state, loss = train_step(
                model, opt_state, jnp.asarray(xb), train_y[idx]
            )
            epoch_losses.append(float(loss))
        eval_loss, eval_acc = eval_step(model, eval_x, eval_y)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "eval_loss": float(eval_loss),
                "eval_accuracy": float(eval_acc),
            }
        )

    elapsed = time.time() - t0
    collapse = {}
    learned_hz = None
    if hasattr(model, "frontend") and hasattr(model.frontend, "bank"):
        bank = model.frontend.bank
        if getattr(bank, "learnable", False):
            omegas, _, _ = bank.effective_omegas_gammas_alphas()
            collapse = band_collapse_metrics(omegas)
            learned_hz = (np.asarray(omegas) / (2.0 * np.pi)).tolist()

    robust_eval: Dict[str, float] = {}
    noise_i = 0

    def _record(name: str, xb: Array) -> None:
        _, acc = eval_step(model, xb, eval_y)
        robust_eval[name] = float(acc)

    for snr in tuple(float(s) for s in config.eval_noise_snrs_db):
        k_i = jax.random.fold_in(k_noise, noise_i)
        noise_i += 1
        tag = int(snr) if snr == int(snr) else snr
        _record(f"eval_accuracy_snr{tag}db", _add_gaussian_noise(eval_x, snr, key=k_i))

    for snr in tuple(float(s) for s in config.eval_pink_snrs_db):
        k_i = jax.random.fold_in(k_noise, noise_i)
        noise_i += 1
        tag = int(snr) if snr == int(snr) else snr
        _record(
            f"eval_accuracy_pink{tag}db",
            _add_colored_noise(eval_x, snr, key=k_i, beta=1.0),
        )

    for snr in tuple(float(s) for s in config.eval_band_snrs_db):
        k_i = jax.random.fold_in(k_noise, noise_i)
        noise_i += 1
        tag = int(snr) if snr == int(snr) else snr
        _record(
            f"eval_accuracy_band{tag}db",
            _add_bandlimited_noise(
                eval_x, snr, key=k_i, sample_rate=float(config.sample_rate)
            ),
        )

    for gain in tuple(float(g) for g in config.eval_level_gains_db):
        tag = int(gain) if gain == int(gain) else gain
        sign = "p" if gain >= 0 else "m"
        _record(
            f"eval_accuracy_level{sign}{abs(tag)}db",
            _scale_level(eval_x, gain),
        )

    result = {
        "frontend": config.frontend,
        "learnable_frontend": learnable_fe and config.frontend != "conv1d",
        "head_kind": config.head_kind,
        "seed": config.seed,
        "data_source": config.data_source,
        "word_set": config.word_set,
        "num_bands": config.num_bands,
        "quality_factor": config.quality_factor,
        "pool": config.pool,
        "feature_mode": config.feature_mode,
        "num_frames": config.num_frames if config.feature_mode == "frames" else None,
        "readout": config.readout,
        "nonlinearity": config.nonlinearity,
        "train_level_aug_db": level_aug_db,
        "train_noise_aug_prob": noise_aug_prob,
        "sample_rate": config.sample_rate,
        "duration_sec": config.duration_sec,
        "head_hidden_dim": config.head_hidden_dim,
        "horn_steps": config.horn_steps if config.head_kind == "horn" else None,
        "horn_coupling_kind": (
            config.horn_coupling_kind if config.head_kind == "horn" else None
        ),
        "collapse_reg_weight": reg_w,
        "trainable_params": trainable_n,
        "frontend_params": frontend_n,
        "epochs": config.epochs,
        "final_eval_accuracy": history[-1]["eval_accuracy"] if history else None,
        "final_eval_loss": history[-1]["eval_loss"] if history else None,
        "chance_accuracy": 1.0 / float(config.num_classes),
        "elapsed_sec": elapsed,
        "run_tag": config.run_tag(),
        "band_collapse": collapse,
        "learned_band_hz": learned_hz,
        "noisy_eval": robust_eval,
        "history": history,
        "config": {
            "f_min_hz": config.f_min_hz,
            "f_max_hz": config.f_max_hz,
            "transient_fraction": config.transient_fraction,
            "train_samples": config.train_samples,
            "eval_samples": config.eval_samples,
            "horn_coupling_depth": config.horn_coupling_depth,
            "horn_coupling_kind": config.horn_coupling_kind,
            "horn_tonotopic_init_strength": config.horn_tonotopic_init_strength,
            "learning_rate": config.learning_rate,
            "eval_noise_snrs_db": list(config.eval_noise_snrs_db),
            "eval_pink_snrs_db": list(config.eval_pink_snrs_db),
            "eval_band_snrs_db": list(config.eval_band_snrs_db),
            "eval_level_gains_db": list(config.eval_level_gains_db),
        },
    }
    for k, v in robust_eval.items():
        result[k] = v

    if config.run_stage5_diagnostics:
        diag = stage5_diagnostics(model, eval_x, eval_y)
        result["stage5_diagnostics"] = diag
        for k, v in diag.items():
            if isinstance(v, (int, float, bool)):
                result[k] = v

    out_path = output_dir / f"audio_digit_{config.run_tag()}.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["output_path"] = str(out_path)
    return result


def run_stage0a_sweep(
    *,
    frontends: Tuple[str, ...] = (
        "resonator",
        "resonator_equal",
        "mel",
        "stft",
        "raw_wide",
        "raw",
        "conv1d",
    ),
    seeds: Tuple[int, ...] = (0, 1),
    base: Optional[AudioDigitConfig] = None,
) -> Dict[str, Any]:
    """Run a frontend attribution sweep and write a summary JSON."""

    base = base or AudioDigitConfig()
    rows = []
    for frontend in frontends:
        for seed in seeds:
            cfg = replace(
                base,
                frontend=frontend,
                seed=seed,
                output_dir=Path(base.output_dir) / "sweep",
            )
            rows.append(run_audio_digit_experiment(cfg))
    summary = {"rows": rows, "by_frontend": {}}
    for frontend in frontends:
        accs = [
            r["final_eval_accuracy"]
            for r in rows
            if r["frontend"] == frontend and r["final_eval_accuracy"] is not None
        ]
        summary["by_frontend"][frontend] = {
            "mean_eval_accuracy": float(np.mean(accs)) if accs else None,
            "std_eval_accuracy": float(np.std(accs)) if accs else None,
            "n": len(accs),
        }
    out = Path(base.output_dir) / "stage0a_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    summary["output_path"] = str(out)
    return summary
