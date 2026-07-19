"""Configuration for the audio-digit RFB frontend probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AudioDigitConfig:
    """Small / parameter-matched spoken-digit classification probe."""

    frontend: str = "resonator"
    # resonator | resonator_learn | resonator_equal | mel | stft | raw | raw_wide | conv1d
    num_bands: int = 16
    sample_rate: float = 8_000.0
    duration_sec: float = 0.5
    f_min_hz: float = 100.0
    f_max_hz: float = 3_200.0  # clamped below Nyquist for sample_rate=8k
    quality_factor: float = 4.0
    pool: str = "log_rms"  # ResonatorBank pool
    transient_fraction: float = 0.1
    feature_mode: str = "pooled"  # pooled | frames (RFB time-varying bands)
    num_frames: int = 16
    readout: str = "amplitude"  # amplitude | phase | both (RFB only)
    # Stage 4: none | drive_tanh | envelope_soft | agc
    nonlinearity: str = "none"
    agc_strength: float = 2.0
    learnable_frontend: bool = False  # Stage 2: learn {ω, Q}
    collapse_reg_weight: float = 0.0  # band-spacing regularizer weight
    head_kind: str = "mlp"  # mlp | horn | gru
    head_hidden_dim: int = 32
    horn_steps: int = 8
    horn_coupling_depth: int = 1  # empirically better for small memory heads
    # fractal_fixed | dense | dense_tonotopic
    horn_coupling_kind: str = "fractal_fixed"
    horn_tonotopic_init_strength: float = 0.5
    num_classes: int = 10
    learning_rate: float = 1e-2
    epochs: int = 40
    batch_size: int = 64
    seed: int = 0
    train_samples: int = 2_000
    eval_samples: int = 500
    data_source: str = "synthetic"
    # synthetic | speech_commands
    word_set: str = "digits"  # digits | core10 (Speech Commands subsets)
    speech_commands_data_dir: Optional[str] = None
    # Post-train noisy eval SNRs in dB (empty = clean eval only)
    eval_noise_snrs_db: Tuple[float, ...] = ()
    # Honest cochlear-ish robustness (level + colored noise); empty = skip
    train_level_aug_db: float = 0.0  # ±dB random gain during training
    # Multi-condition noise aug (Sprint A): prob of corrupting each train batch
    train_noise_aug_prob: float = 0.0
    train_noise_snr_db_min: float = 0.0
    train_noise_snr_db_max: float = 15.0
    eval_level_gains_db: Tuple[float, ...] = ()
    eval_pink_snrs_db: Tuple[float, ...] = ()
    eval_band_snrs_db: Tuple[float, ...] = ()  # mid-band limited noise
    run_stage5_diagnostics: bool = False
    output_dir: Path = field(default_factory=lambda: Path("outputs/audio_digit"))
    raw_wide_dim: Optional[int] = None

    @property
    def num_samples(self) -> int:
        return int(round(float(self.sample_rate) * float(self.duration_sec)))

    @property
    def feature_dim(self) -> int:
        if self.frontend in (
            "resonator",
            "resonator_learn",
            "resonator_equal",
        ):
            n = int(self.num_bands)
            if self.readout == "amplitude":
                return n
            if self.readout == "phase":
                return 2 * n
            if self.readout == "both":
                return 3 * n
            raise ValueError(f"unknown readout {self.readout!r}")
        if self.frontend in ("mel", "stft"):
            return int(self.num_bands)
        if self.frontend == "raw":
            return int(self.num_samples)
        if self.frontend == "raw_wide":
            return int(
                self.raw_wide_dim
                if self.raw_wide_dim is not None
                else self.num_bands
            )
        if self.frontend == "conv1d":
            return int(self.num_bands)
        raise ValueError(f"unknown frontend {self.frontend!r}")

    def run_tag(self) -> str:
        """Stable filename / sweep key."""

        fe = self.frontend
        if self.learnable_frontend and fe == "resonator":
            fe = "resonator_learn"
        parts = [
            fe,
            self.head_kind,
            self.feature_mode,
            f"b{self.num_bands}",
            f"q{self.quality_factor:g}",
            f"sr{int(self.sample_rate)}",
            f"seed{self.seed}",
        ]
        if self.feature_mode == "frames":
            parts.insert(3, f"f{self.num_frames}")
        if self.head_kind == "horn":
            parts.insert(2, f"k{self.horn_steps}")
            if self.horn_coupling_kind != "fractal_fixed":
                parts.insert(2, self.horn_coupling_kind)
        if self.head_kind == "gru":
            parts.insert(2, "gru")
        if self.pool != "log_rms" and str(fe).startswith("resonator"):
            parts.insert(2, self.pool)
        if self.word_set != "digits":
            parts.insert(1, self.word_set)
        if self.collapse_reg_weight > 0:
            parts.insert(2, f"reg{self.collapse_reg_weight:g}")
        if self.readout != "amplitude" and str(fe).startswith("resonator"):
            parts.insert(2, self.readout)
        if self.nonlinearity not in ("none", "linear") and str(fe).startswith(
            "resonator"
        ):
            parts.insert(2, self.nonlinearity)
        if self.train_level_aug_db > 0:
            parts.insert(2, f"laug{self.train_level_aug_db:g}")
        if self.train_noise_aug_prob > 0:
            parts.insert(2, f"naug{self.train_noise_aug_prob:g}")
        return "_".join(parts)
