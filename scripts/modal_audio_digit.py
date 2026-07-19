"""Run audio-digit RFB / AudioPrism-stack probes on Modal GPU workers.

Heavy sweeps belong here — not on the local host.

Examples:

    modal run scripts/modal_audio_digit.py

    # Close the mel gap (bands / Q / 16 kHz)
    OSCNET_MODAL_MAX_CONTAINERS=10 modal run scripts/modal_audio_digit.py \\
      --sweep-preset rfb_tune

    # RFB→HORN vs RFB→MLP vs mel→HORN
    OSCNET_MODAL_MAX_CONTAINERS=10 modal run scripts/modal_audio_digit.py \\
      --sweep-preset horn_stack

    # Confirm: 4 seeds, collapse reg, matched HORN vs MLP, noise, core10
    OSCNET_MODAL_MAX_CONTAINERS=10 modal run scripts/modal_audio_digit.py \\
      --sweep-preset audioprism_confirm
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oscnet-audio-digit"
VOLUME_NAME = os.environ.get("OSCNET_MODAL_VOLUME", "oscnet-runs")
VOLUME_MOUNT = Path("/mnt/oscnet-runs")
GPU = os.environ.get("OSCNET_MODAL_GPU", "A10G")
TIMEOUT_SECONDS = int(os.environ.get("OSCNET_MODAL_TIMEOUT_SECONDS", "10800"))
JAX_PACKAGE = os.environ.get("OSCNET_MODAL_JAX", "jax[cuda13]")
MAX_CONTAINERS = int(os.environ.get("OSCNET_MODAL_MAX_CONTAINERS", "10"))

REMOTE_PACKAGES = [
    JAX_PACKAGE,
    "equinox>=0.10.0",
    "diffrax>=0.4.0",
    "optax>=0.1.7",
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",
]

STAGE0A_FRONTENDS = (
    "resonator",
    "resonator_equal",
    "mel",
    "stft",
    "raw_wide",
    "raw",
    "conv1d",
)

SWEEP_CSVS = {
    "stage0a_synthetic": Path(
        "outputs/analysis/modal_audio_digit_stage0a_synthetic.csv"
    ),
    "stage0a_speech_commands": Path(
        "outputs/analysis/modal_audio_digit_stage0a_speech_commands.csv"
    ),
    "rfb_tune": Path("outputs/analysis/modal_audio_digit_rfb_tune.csv"),
    "horn_stack": Path("outputs/analysis/modal_audio_digit_horn_stack.csv"),
    "rfb_learn": Path("outputs/analysis/modal_audio_digit_rfb_learn.csv"),
    "audioprism_confirm": Path(
        "outputs/analysis/modal_audio_digit_audioprism_confirm.csv"
    ),
    "rfb_stage3": Path("outputs/analysis/modal_audio_digit_rfb_stage3.csv"),
    "rfb_stage4": Path("outputs/analysis/modal_audio_digit_rfb_stage4.csv"),
    "rfb_robust": Path("outputs/analysis/modal_audio_digit_rfb_robust.csv"),
    "rfb_stage5": Path("outputs/analysis/modal_audio_digit_rfb_stage5.csv"),
    "rfb_plus_noiseaug": Path(
        "outputs/analysis/modal_audio_digit_rfb_plus_noiseaug.csv"
    ),
    "rfb_plus_scale": Path(
        "outputs/analysis/modal_audio_digit_rfb_plus_scale.csv"
    ),
    "rfb_plus_tonotopic": Path(
        "outputs/analysis/modal_audio_digit_rfb_plus_tonotopic.csv"
    ),
    "rfb_plus_controls": Path(
        "outputs/analysis/modal_audio_digit_rfb_plus_controls.csv"
    ),
    "rfb_plus_isoparam": Path(
        "outputs/analysis/modal_audio_digit_rfb_plus_isoparam.csv"
    ),
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*REMOTE_PACKAGES)
    .env(
        {
            "MPLBACKEND": "Agg",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_COMPILATION_CACHE_DIR": str(VOLUME_MOUNT / "jax_cache"),
            "OSCNET_SPEECH_COMMANDS_DIR": str(
                VOLUME_MOUNT / "datasets" / "speech_commands_v0.02"
            ),
        }
    )
    .add_local_python_source("oscnet")
)


def _safe_run_name(name: str) -> str:
    name = name.strip() or time.strftime("audio-digit-%Y%m%d-%H%M%S")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip(".-") or "audio-digit-run"


def _config_dict(**kwargs: Any) -> dict[str, Any]:
    cfg = {
        "frontend": "resonator",
        "seed": 0,
        "data_source": "speech_commands",
        "epochs": 40,
        "train_samples": 4000,
        "eval_samples": 800,
        "sample_rate": 8000.0,
        "duration_sec": 1.0,
        "num_bands": 16,
        "quality_factor": 4.0,
        "pool": "log_rms",
        "feature_mode": "pooled",
        "num_frames": 16,
        "readout": "amplitude",
        "nonlinearity": "none",
        "agc_strength": 2.0,
        "f_min_hz": 100.0,
        "f_max_hz": 3200.0,
        "head_kind": "mlp",
        "head_hidden_dim": 48,
        "horn_steps": 8,
        "horn_coupling_depth": 1,
        "horn_coupling_kind": "fractal_fixed",
        "horn_tonotopic_init_strength": 0.5,
        "learnable_frontend": False,
        "collapse_reg_weight": 0.0,
        "word_set": "digits",
        "eval_noise_snrs_db": (),
        "train_level_aug_db": 0.0,
        "train_noise_aug_prob": 0.0,
        "train_noise_snr_db_min": 0.0,
        "train_noise_snr_db_max": 15.0,
        "eval_level_gains_db": (),
        "eval_pink_snrs_db": (),
        "eval_band_snrs_db": (),
        "run_stage5_diagnostics": False,
        "learning_rate": 1e-2,
        "batch_size": 64,
        "output_dir": str(VOLUME_MOUNT / "audio_digit" / "default"),
        "speech_commands_data_dir": str(
            VOLUME_MOUNT / "datasets" / "speech_commands_v0.02"
        ),
    }
    cfg.update(kwargs)
    return cfg


@app.function(
    image=image,
    timeout=3600,
    volumes={VOLUME_MOUNT: volume},
)
def prepare_speech_commands_remote() -> str:
    from oscnet.experiments.audio_digit.data import ensure_speech_commands

    root = ensure_speech_commands(os.environ.get("OSCNET_SPEECH_COMMANDS_DIR"))
    volume.commit()
    return str(root)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes={VOLUME_MOUNT: volume},
)
def run_audio_digit_remote(config: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path as _Path

    from oscnet.experiments.audio_digit.config import AudioDigitConfig
    from oscnet.experiments.audio_digit.runner import run_audio_digit_experiment

    remote_out = _Path(config["output_dir"])
    remote_out.mkdir(parents=True, exist_ok=True)
    cfg = AudioDigitConfig(
        frontend=str(config["frontend"]),
        seed=int(config["seed"]),
        data_source=str(config["data_source"]),
        epochs=int(config["epochs"]),
        train_samples=int(config["train_samples"]),
        eval_samples=int(config["eval_samples"]),
        sample_rate=float(config["sample_rate"]),
        duration_sec=float(config["duration_sec"]),
        num_bands=int(config["num_bands"]),
        quality_factor=float(config.get("quality_factor", 4.0)),
        pool=str(config.get("pool", "log_rms")),
        feature_mode=str(config.get("feature_mode", "pooled")),
        num_frames=int(config.get("num_frames", 16)),
        readout=str(config.get("readout", "amplitude")),
        nonlinearity=str(config.get("nonlinearity", "none")),
        agc_strength=float(config.get("agc_strength", 2.0)),
        f_min_hz=float(config.get("f_min_hz", 100.0)),
        f_max_hz=float(config.get("f_max_hz", 3200.0)),
        head_kind=str(config.get("head_kind", "mlp")),
        head_hidden_dim=int(config["head_hidden_dim"]),
        horn_steps=int(config.get("horn_steps", 8)),
        horn_coupling_depth=int(config.get("horn_coupling_depth", 1)),
        horn_coupling_kind=str(
            config.get("horn_coupling_kind", "fractal_fixed")
        ),
        horn_tonotopic_init_strength=float(
            config.get("horn_tonotopic_init_strength", 0.5)
        ),
        learnable_frontend=bool(config.get("learnable_frontend", False)),
        collapse_reg_weight=float(config.get("collapse_reg_weight", 0.0)),
        word_set=str(config.get("word_set", "digits")),
        eval_noise_snrs_db=tuple(
            float(s) for s in config.get("eval_noise_snrs_db", ())
        ),
        train_level_aug_db=float(config.get("train_level_aug_db", 0.0)),
        train_noise_aug_prob=float(config.get("train_noise_aug_prob", 0.0)),
        train_noise_snr_db_min=float(config.get("train_noise_snr_db_min", 0.0)),
        train_noise_snr_db_max=float(config.get("train_noise_snr_db_max", 15.0)),
        eval_level_gains_db=tuple(
            float(s) for s in config.get("eval_level_gains_db", ())
        ),
        eval_pink_snrs_db=tuple(
            float(s) for s in config.get("eval_pink_snrs_db", ())
        ),
        eval_band_snrs_db=tuple(
            float(s) for s in config.get("eval_band_snrs_db", ())
        ),
        run_stage5_diagnostics=bool(config.get("run_stage5_diagnostics", False)),
        learning_rate=float(config.get("learning_rate", 1e-2)),
        batch_size=int(config["batch_size"]),
        output_dir=remote_out,
        speech_commands_data_dir=config.get("speech_commands_data_dir"),
    )
    result = run_audio_digit_experiment(cfg)
    volume.commit()
    return result


def _stage0a_jobs(preset: str) -> list[dict[str, Any]]:
    if preset == "stage0a_synthetic":
        base = dict(
            data_source="synthetic",
            epochs=40,
            train_samples=2000,
            eval_samples=500,
            sample_rate=8000.0,
            duration_sec=0.5,
            head_hidden_dim=32,
        )
        seeds = (0, 1, 2, 3)
        out_root = VOLUME_MOUNT / "audio_digit" / "stage0a_synthetic"
        jobs = []
        for frontend in STAGE0A_FRONTENDS:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        seed=seed,
                        num_bands=16,
                        output_dir=str(out_root / "sweep"),
                        **base,
                    )
                )
        return jobs

    if preset == "stage0a_speech_commands":
        base = dict(
            data_source="speech_commands",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=8000.0,
            duration_sec=1.0,
            head_hidden_dim=48,
        )
        seeds = (0, 1, 2, 3)
        out_root = VOLUME_MOUNT / "audio_digit" / "stage0a_speech_commands"
        jobs = []
        for frontend in STAGE0A_FRONTENDS:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        seed=seed,
                        num_bands=16,
                        output_dir=str(out_root / "sweep"),
                        **base,
                    )
                )
        return jobs

    if preset == "rfb_tune":
        # Close the mel/STFT gap: bands × Q × optional 16 kHz, matched controls.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_tune"
        sc = dict(
            data_source="speech_commands",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            duration_sec=1.0,
            head_kind="mlp",
            head_hidden_dim=64,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
        )
        jobs: list[dict[str, Any]] = []
        # Resonator grid
        for bands, q, sr, fmax in (
            (24, 4.0, 8000.0, 3200.0),
            (32, 2.0, 8000.0, 3200.0),
            (32, 4.0, 8000.0, 3200.0),
            (32, 8.0, 8000.0, 3200.0),
            (32, 12.0, 8000.0, 3200.0),
            (48, 4.0, 8000.0, 3200.0),
            (48, 8.0, 8000.0, 3200.0),
            (32, 4.0, 16000.0, 7000.0),
            (32, 8.0, 16000.0, 7000.0),
            (48, 8.0, 16000.0, 7000.0),
        ):
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend="resonator",
                        seed=seed,
                        num_bands=bands,
                        quality_factor=q,
                        sample_rate=sr,
                        f_max_hz=fmax,
                        **sc,
                    )
                )
        # Matched classical controls + equal-freq skeptic
        for frontend, bands, sr, fmax in (
            ("mel", 32, 8000.0, 3200.0),
            ("mel", 48, 8000.0, 3200.0),
            ("stft", 32, 8000.0, 3200.0),
            ("stft", 48, 8000.0, 3200.0),
            ("mel", 32, 16000.0, 7000.0),
            ("resonator_equal", 32, 8000.0, 3200.0),
            ("conv1d", 32, 8000.0, 3200.0),
        ):
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        seed=seed,
                        num_bands=bands,
                        quality_factor=4.0,
                        sample_rate=sr,
                        f_max_hz=fmax,
                        **sc,
                    )
                )
        return jobs

    if preset == "horn_stack":
        # AudioPrism-style: time-varying RFB frames → HORN vs pooled MLP/HORN.
        # Best frozen RFB point from rfb_tune: 32 bands, Q=4, 16 kHz.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "horn_stack"
        base = dict(
            data_source="speech_commands",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            head_hidden_dim=48,
            num_frames=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
        )
        # (frontend, head_kind, feature_mode, horn_steps)
        arms = (
            ("resonator", "mlp", "pooled", 8),
            ("resonator", "mlp", "frames", 8),
            ("resonator", "horn", "pooled", 8),
            ("resonator", "horn", "pooled", 16),
            ("resonator", "horn", "frames", 16),  # one step per frame
            ("resonator_equal", "horn", "frames", 16),
            ("mel", "mlp", "pooled", 8),
            ("mel", "horn", "pooled", 16),
            ("stft", "mlp", "pooled", 8),
            ("conv1d", "mlp", "pooled", 8),
        )
        jobs = []
        for frontend, head_kind, feature_mode, steps in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        head_kind=head_kind,
                        feature_mode=feature_mode,
                        horn_steps=int(steps),
                        seed=seed,
                        **base,
                    )
                )
        return jobs

    if preset == "rfb_learn":
        # Stage 2: learnable {ω,Q} vs frozen RFB / mel / conv (Speech Commands).
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_learn"
        base = dict(
            data_source="speech_commands",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            head_hidden_dim=48,
            num_frames=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
        )
        # (frontend, learnable, head_kind, feature_mode, horn_steps, lr)
        arms = (
            ("resonator", False, "mlp", "pooled", 8, 1e-2),
            ("resonator", True, "mlp", "pooled", 8, 3e-3),
            ("resonator", False, "horn", "frames", 16, 1e-2),
            ("resonator", True, "horn", "frames", 16, 3e-3),
            ("resonator_equal", False, "horn", "frames", 16, 1e-2),
            ("mel", False, "mlp", "pooled", 8, 1e-2),
            ("conv1d", False, "mlp", "pooled", 8, 1e-2),
        )
        jobs = []
        for frontend, learnable, head_kind, feature_mode, steps, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head_kind,
                        feature_mode=feature_mode,
                        horn_steps=int(steps),
                        learning_rate=float(lr),
                        seed=seed,
                        **base,
                    )
                )
        return jobs

    if preset == "audioprism_confirm":
        # Full AudioPrism claim: 4 seeds, collapse reg, matched HORN vs MLP,
        # noisy eval, digits + core10 transfer.
        seeds = (0, 1, 2, 3)
        out_root = VOLUME_MOUNT / "audio_digit" / "audioprism_confirm"
        noise = (10.0, 0.0)
        winner = dict(
            data_source="speech_commands",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            horn_steps=16,
            batch_size=64,
            eval_noise_snrs_db=noise,
            output_dir=str(out_root / "sweep"),
        )
        # (word_set, frontend, learnable, head, hidden, reg, lr)
        # MLP h=96 ≈ HORN h=48 trainable params (~4.3–4.6k).
        arms = (
            ("digits", "resonator", True, "horn", 48, 0.1, 3e-3),
            ("digits", "resonator", True, "mlp", 96, 0.1, 3e-3),
            ("digits", "resonator", True, "mlp", 48, 0.1, 3e-3),
            ("digits", "resonator", False, "horn", 48, 0.0, 1e-2),
            ("digits", "mel", False, "mlp", 48, 0.0, 1e-2),
            ("digits", "conv1d", False, "mlp", 48, 0.0, 1e-2),
            ("core10", "resonator", True, "horn", 48, 0.1, 3e-3),
            ("core10", "mel", False, "mlp", 48, 0.0, 1e-2),
        )
        jobs = []
        for word_set, frontend, learnable, head, hidden, reg, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=int(hidden),
                        collapse_reg_weight=float(reg),
                        word_set=word_set,
                        learning_rate=float(lr),
                        seed=seed,
                        **winner,
                    )
                )
        return jobs

    if preset == "rfb_stage3":
        # Stage 3: amplitude vs phase vs both readout (+ stronger collapse reg).
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_stage3"
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            head_hidden_dim=48,
            horn_steps=16,
            batch_size=64,
            eval_noise_snrs_db=(10.0, 0.0),
            output_dir=str(out_root / "sweep"),
        )
        # (frontend, learnable, readout, reg, lr)
        arms = (
            ("resonator", True, "amplitude", 0.1, 3e-3),
            ("resonator", True, "phase", 0.1, 3e-3),
            ("resonator", True, "both", 0.1, 3e-3),
            ("resonator", True, "amplitude", 0.5, 3e-3),  # stronger collapse reg
            ("resonator", False, "amplitude", 0.0, 1e-2),
            ("resonator", False, "both", 0.0, 1e-2),
            ("mel", False, "amplitude", 0.0, 1e-2),
        )
        jobs = []
        for frontend, learnable, readout, reg, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        readout=(
                            readout
                            if frontend.startswith("resonator")
                            else "amplitude"
                        ),
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        head_kind=(
                            "horn" if frontend.startswith("resonator") else "mlp"
                        ),
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_stage4":
        # Stage 4 / H4: linear vs compressive drive / envelope / AGC.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_stage4"
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            head_hidden_dim=48,
            horn_steps=16,
            batch_size=64,
            eval_noise_snrs_db=(10.0, 0.0),
            output_dir=str(out_root / "sweep"),
        )
        # (frontend, learnable, nonlinearity, reg, lr, head)
        arms = (
            ("resonator", True, "none", 0.1, 3e-3, "horn"),
            ("resonator", True, "drive_tanh", 0.1, 3e-3, "horn"),
            ("resonator", True, "envelope_soft", 0.1, 3e-3, "horn"),
            ("resonator", False, "none", 0.0, 1e-2, "horn"),
            ("resonator", False, "drive_tanh", 0.0, 1e-2, "horn"),
            ("resonator", False, "agc", 0.0, 1e-2, "horn"),
            ("mel", False, "none", 0.0, 1e-2, "mlp"),
        )
        jobs = []
        for frontend, learnable, nonlin, reg, lr, head in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        nonlinearity=nonlin if frontend.startswith("resonator") else "none",
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        head_kind=head,
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_robust":
        # Level-aug training + colored/band noise + level stress (no AGC).
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_robust"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            nonlinearity="none",
            head_hidden_dim=48,
            horn_steps=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, reg, lr, level_aug_db)
        arms = (
            ("resonator", True, "horn", 0.1, 3e-3, 0.0),
            ("resonator", True, "horn", 0.1, 3e-3, 12.0),
            ("mel", False, "mlp", 0.0, 1e-2, 0.0),
            ("mel", False, "mlp", 0.0, 1e-2, 12.0),
        )
        jobs = []
        for frontend, learnable, head, reg, lr, laug in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        train_level_aug_db=float(laug),
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_stage5":
        # Stage 5 / H5: full stack + high→low routing / band-ablation diagnostics.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_stage5"
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            nonlinearity="none",
            horn_steps=16,
            batch_size=64,
            run_stage5_diagnostics=True,
            eval_noise_snrs_db=(10.0,),
            output_dir=str(out_root / "sweep"),
        )
        # (frontend, learnable, head, hidden, reg, lr)
        arms = (
            ("resonator", True, "horn", 48, 0.1, 3e-3),
            ("resonator", True, "mlp", 96, 0.1, 3e-3),
            ("resonator", False, "horn", 48, 0.0, 1e-2),
            ("resonator", False, "mlp", 96, 0.0, 1e-2),
            ("mel", False, "mlp", 48, 0.0, 1e-2),
        )
        jobs = []
        for frontend, learnable, head, hidden, reg, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=int(hidden),
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_plus_noiseaug":
        # Stage 5+ Sprint A: multi-condition noise+level aug vs clean train.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_plus_noiseaug"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=40,
            train_samples=4000,
            eval_samples=800,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            nonlinearity="none",
            head_hidden_dim=48,
            horn_steps=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, reg, lr, level_aug, noise_prob)
        arms = (
            ("resonator", True, "horn", 0.1, 3e-3, 0.0, 0.0),
            ("resonator", True, "horn", 0.1, 3e-3, 12.0, 0.0),
            ("resonator", True, "horn", 0.1, 3e-3, 12.0, 0.7),
            ("mel", False, "mlp", 0.0, 1e-2, 0.0, 0.0),
            ("mel", False, "mlp", 0.0, 1e-2, 12.0, 0.7),
        )
        jobs = []
        for frontend, learnable, head, reg, lr, laug, nprob in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        train_level_aug_db=float(laug),
                        train_noise_aug_prob=float(nprob),
                        train_noise_snr_db_min=0.0,
                        train_noise_snr_db_max=15.0,
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_plus_scale":
        # Stage 5+ Sprint B: more data/width + softer noise-aug at scale.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_plus_scale"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=50,
            train_samples=10_000,
            eval_samples=1_200,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            nonlinearity="none",
            horn_steps=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, hidden, reg, lr, laug, nprob)
        arms = (
            ("resonator", True, "horn", 64, 0.1, 3e-3, 0.0, 0.0),
            ("resonator", True, "horn", 64, 0.1, 3e-3, 12.0, 0.4),  # softer
            ("resonator", True, "horn", 64, 0.1, 3e-3, 12.0, 0.7),
            ("resonator", True, "mlp", 128, 0.1, 3e-3, 0.0, 0.0),
            ("mel", False, "mlp", 64, 0.0, 1e-2, 0.0, 0.0),
            ("mel", False, "mlp", 64, 0.0, 1e-2, 12.0, 0.4),
        )
        jobs = []
        for frontend, learnable, head, hidden, reg, lr, laug, nprob in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=int(hidden),
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        train_level_aug_db=float(laug),
                        train_noise_aug_prob=float(nprob),
                        train_noise_snr_db_min=0.0,
                        train_noise_snr_db_max=15.0,
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_plus_tonotopic":
        # Stage 5+ Sprint C: learnable dense / tonotopic coupling + Stage 5 diags.
        seeds = (0, 1)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_plus_tonotopic"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=50,
            train_samples=10_000,
            eval_samples=1_200,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            feature_mode="frames",
            readout="amplitude",
            nonlinearity="none",
            horn_steps=16,
            batch_size=64,
            run_stage5_diagnostics=True,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, coupling, reg, lr, laug, nprob)
        arms = (
            ("resonator", True, "horn", "fractal_fixed", 0.1, 3e-3, 0.0, 0.0),
            ("resonator", True, "horn", "dense", 0.1, 3e-3, 0.0, 0.0),
            ("resonator", True, "horn", "dense_tonotopic", 0.1, 3e-3, 0.0, 0.0),
            ("resonator", True, "horn", "fractal_fixed", 0.1, 3e-3, 12.0, 0.4),
            ("resonator", True, "horn", "dense", 0.1, 3e-3, 12.0, 0.4),
            ("resonator", True, "horn", "dense_tonotopic", 0.1, 3e-3, 12.0, 0.4),
            ("resonator", True, "mlp", "fractal_fixed", 0.1, 3e-3, 0.0, 0.0),
            ("mel", False, "mlp", "fractal_fixed", 0.0, 1e-2, 0.0, 0.0),
        )
        jobs = []
        for frontend, learnable, head, couple, reg, lr, laug, nprob in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=(128 if head == "mlp" else 64),
                        horn_coupling_kind=couple,
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        train_level_aug_db=float(laug),
                        train_noise_aug_prob=float(nprob),
                        train_noise_snr_db_min=0.0,
                        train_noise_snr_db_max=15.0,
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_plus_controls":
        # Council controls: mel→HORN, RFB→GRU, 4-seed headline dense HORN.
        seeds = (0, 1, 2, 3)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_plus_controls"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=50,
            train_samples=10_000,
            eval_samples=1_200,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            readout="amplitude",
            nonlinearity="none",
            horn_steps=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, coupling, feature_mode, hidden, reg, lr)
        arms = (
            ("resonator", True, "horn", "dense", "frames", 64, 0.1, 3e-3),
            ("resonator", True, "gru", "fractal_fixed", "frames", 64, 0.1, 3e-3),
            ("resonator", True, "mlp", "fractal_fixed", "frames", 128, 0.1, 3e-3),
            ("mel", False, "horn", "dense", "frames", 64, 0.0, 1e-2),
            ("mel", False, "gru", "fractal_fixed", "frames", 64, 0.0, 1e-2),
            ("mel", False, "mlp", "fractal_fixed", "pooled", 64, 0.0, 1e-2),
        )
        jobs = []
        for frontend, learnable, head, couple, fmode, hidden, reg, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=int(hidden),
                        horn_coupling_kind=couple,
                        feature_mode=fmode,
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        seed=seed,
                    )
                )
        return jobs

    if preset == "rfb_plus_isoparam":
        # Iso-parameter GRU (~7k) vs dense HORN (~7k), 4 seeds.
        seeds = (0, 1, 2, 3)
        out_root = VOLUME_MOUNT / "audio_digit" / "rfb_plus_isoparam"
        stress = dict(
            eval_noise_snrs_db=(10.0,),
            eval_pink_snrs_db=(10.0,),
            eval_band_snrs_db=(10.0,),
            eval_level_gains_db=(-20.0, 6.0),
        )
        base = dict(
            data_source="speech_commands",
            word_set="digits",
            epochs=50,
            train_samples=10_000,
            eval_samples=1_200,
            sample_rate=16000.0,
            duration_sec=1.0,
            num_bands=32,
            quality_factor=4.0,
            f_max_hz=7000.0,
            num_frames=16,
            readout="amplitude",
            nonlinearity="none",
            horn_steps=16,
            batch_size=64,
            output_dir=str(out_root / "sweep"),
            **stress,
        )
        # (frontend, learnable, head, coupling, feature_mode, hidden, reg, lr)
        # GRU H=34 ≈ 7.2k trainable; dense HORN H=64 ≈ 6.9–7.1k.
        arms = (
            ("resonator", True, "horn", "dense", "frames", 64, 0.1, 3e-3),
            ("resonator", True, "gru", "fractal_fixed", "frames", 34, 0.1, 3e-3),
            ("mel", False, "horn", "dense", "frames", 64, 0.0, 1e-2),
            ("mel", False, "gru", "fractal_fixed", "frames", 34, 0.0, 1e-2),
        )
        jobs = []
        for frontend, learnable, head, couple, fmode, hidden, reg, lr in arms:
            for seed in seeds:
                jobs.append(
                    _config_dict(
                        **base,
                        frontend=frontend,
                        learnable_frontend=learnable,
                        head_kind=head,
                        head_hidden_dim=int(hidden),
                        horn_coupling_kind=couple,
                        feature_mode=fmode,
                        collapse_reg_weight=float(reg),
                        learning_rate=float(lr),
                        seed=seed,
                    )
                )
        return jobs

    raise ValueError(f"unknown sweep preset {preset!r}")


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_tag",
        "frontend",
        "head_kind",
        "feature_mode",
        "num_frames",
        "readout",
        "nonlinearity",
        "seed",
        "data_source",
        "word_set",
        "num_bands",
        "quality_factor",
        "pool",
        "sample_rate",
        "horn_steps",
        "horn_coupling_kind",
        "head_hidden_dim",
        "learnable_frontend",
        "collapse_reg_weight",
        "train_level_aug_db",
        "train_noise_aug_prob",
        "final_eval_accuracy",
        "final_eval_loss",
        "eval_accuracy_snr10db",
        "eval_accuracy_pink10db",
        "eval_accuracy_band10db",
        "eval_accuracy_levelm20db",
        "eval_accuracy_levelp6db",
        "ablate_none",
        "ablate_low_drop",
        "ablate_high_drop",
        "coupling_high_to_low",
        "coupling_low_to_high",
        "coupling_high_to_low_ratio",
        "horn_coupling_fixed_structure",
        "trainable_params",
        "frontend_params",
        "chance_accuracy",
        "elapsed_sec",
        "epochs",
        "output_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_key(row: dict[str, Any]) -> str:
    learn = "learn" if row.get("learnable_frontend") else "frozen"
    words = row.get("word_set") or "digits"
    hid = row.get("head_hidden_dim")
    reg = row.get("collapse_reg_weight") or 0
    readout = row.get("readout") or "amplitude"
    nonlin = row.get("nonlinearity") or "none"
    laug = row.get("train_level_aug_db") or 0
    naug = row.get("train_noise_aug_prob") or 0
    couple = row.get("horn_coupling_kind") or "fractal_fixed"
    return (
        f"{words}/{row.get('frontend')}/{learn}/{row.get('head_kind', 'mlp')}"
        f"/{couple}/{readout}/{nonlin}/laug{laug}/naug{naug}/h{hid}"
        f"/{row.get('feature_mode', 'pooled')}"
        f"/b{row.get('num_bands')}/reg{reg}"
        f"/k{row.get('horn_steps') or '-'}"
    )


@app.local_entrypoint()
def main(
    sweep_preset: str = "",
    frontend: str = "resonator",
    seed: int = 0,
    data_source: str = "synthetic",
    epochs: int = 2,
    train_samples: int = 128,
    eval_samples: int = 64,
    dry_run: bool = False,
) -> None:
    if sweep_preset:
        if sweep_preset not in SWEEP_CSVS:
            raise SystemExit(
                f"unknown --sweep-preset {sweep_preset!r}; "
                f"choose from {sorted(SWEEP_CSVS)}"
            )
        jobs = _stage0a_jobs(sweep_preset)
        print(f"preset {sweep_preset}: {len(jobs)} jobs on {GPU}")
        if dry_run:
            for job in jobs[:8]:
                print(
                    " ",
                    job["frontend"],
                    job.get("head_kind"),
                    "b",
                    job["num_bands"],
                    "q",
                    job.get("quality_factor"),
                    "sr",
                    job["sample_rate"],
                    "seed",
                    job["seed"],
                )
            if len(jobs) > 8:
                print(f"  ... ({len(jobs) - 8} more)")
            return
        if sweep_preset in (
            "stage0a_speech_commands",
            "rfb_tune",
            "horn_stack",
            "rfb_learn",
            "audioprism_confirm",
            "rfb_stage3",
            "rfb_stage4",
            "rfb_robust",
            "rfb_stage5",
            "rfb_plus_noiseaug",
            "rfb_plus_scale",
        ):
            print("preparing Speech Commands on Modal volume ...")
            cache_root = prepare_speech_commands_remote.remote()
            print(f"dataset ready at {cache_root}")
        results = list(run_audio_digit_remote.map(jobs))
        csv_path = SWEEP_CSVS[sweep_preset]
        _write_summary_csv(results, csv_path)
        by: dict[str, list[float]] = {}
        extras: dict[str, dict[str, list[float]]] = {}
        extra_keys = (
            "eval_accuracy_snr10db",
            "eval_accuracy_pink10db",
            "eval_accuracy_band10db",
            "eval_accuracy_levelm20db",
            "eval_accuracy_levelp6db",
            "ablate_low_drop",
            "ablate_high_drop",
            "coupling_high_to_low_ratio",
        )
        for row in results:
            key = _group_key(row)
            by.setdefault(key, []).append(float(row["final_eval_accuracy"]))
            for ek in extra_keys:
                if row.get(ek) is not None:
                    extras.setdefault(key, {}).setdefault(ek, []).append(float(row[ek]))
        print(f"wrote {csv_path}")
        for name, accs in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            mean = sum(accs) / len(accs)
            bits = []
            for ek, vals in extras.get(name, {}).items():
                short = ek.replace("eval_accuracy_", "")
                bits.append(f"{short}={sum(vals) / len(vals):.3f}")
            extra_s = ("  " + " ".join(bits)) if bits else ""
            print(f"  {name:64s}  clean={mean:.4f}  n={len(accs)}{extra_s}")
        summary_json = csv_path.with_suffix(".json")
        summary_json.write_text(
            json.dumps({"preset": sweep_preset, "rows": results}, indent=2)
        )
        print(f"wrote {summary_json}")
        return

    run_name = _safe_run_name(f"smoke_{frontend}_seed{seed}")
    out = str(VOLUME_MOUNT / "audio_digit" / "smoke" / run_name)
    job = _config_dict(
        frontend=frontend,
        seed=seed,
        data_source=data_source,
        epochs=epochs,
        train_samples=train_samples,
        eval_samples=eval_samples,
        sample_rate=8000.0,
        duration_sec=0.5 if data_source == "synthetic" else 1.0,
        num_bands=8,
        head_hidden_dim=16,
        batch_size=32,
        output_dir=out,
    )
    print(f"smoke: {frontend} seed={seed} data_source={data_source}")
    if dry_run:
        print(job)
        return
    result = run_audio_digit_remote.remote(job)
    print(
        f"{result.get('run_tag', result['frontend'])} "
        f"acc={result['final_eval_accuracy']:.4f} "
        f"params={result['trainable_params']} -> {result.get('output_path')}"
    )
