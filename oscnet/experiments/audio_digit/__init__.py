"""Spoken-digit RFB→HORN experiment (Speech Commands + matched controls).

Frontends (learnable / frozen ResonatorBank, mel, STFT, raw, …) with MLP,
dense HORN, or GRU heads. Used by ``examples/audio_digit_classification.py``
and ``scripts/modal_audio_digit.py``.
"""

from .cli import main
from .config import AudioDigitConfig
from .runner import run_audio_digit_experiment

__all__ = [
    "AudioDigitConfig",
    "main",
    "run_audio_digit_experiment",
]
