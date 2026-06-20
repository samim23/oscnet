"""
Neural network models based on oscillatory dynamics.

This module provides various implementations of neural networks
based on oscillatory dynamics and ODEs.
"""

from .config import (
    OscillatoryAutoencoderConfig,
    PatchOscillatoryAutoencoderConfig,
    WaveletAutoencoderConfig,
    WinfreeFieldAutoencoderConfig,
    WinfreePatchAutoencoderConfig,
    WinfreePhaseAutoencoderConfig,
)
from .fractal import FractalHORNCell
from .oscillatory import (
    AmplitudeVelocityAutoencoder,
    AmplitudeVelocityDecoder,
    AmplitudeVelocityEncoder,
    AmplitudeVelocityHORN,
    AmplitudeVelocityHORNCell,
    AmplitudeVelocityOscillatorCell,
    AutoregressiveOscillatoryDecoder,
    OscillatorState,
    OscillatoryAutoencoder,
    OscillatoryEncoder,
    OscillatorySequenceLayer,
    PatchOscillatoryAutoencoder,
    PositionalLatentOscillatoryDecoder,
    RepeatedLatentOscillatoryDecoder,
)
from .phase import (
    WinfreePhaseAutoencoder,
    WinfreePhaseOscillatorCell,
    WinfreePhaseSequenceLayer,
)
from .wavelet import (
    ProductionWaveletAutoencoder,
    WaveletOscillatorCell,
    WaveletOscillatoryAutoencoder,
)
from .winfree import (
    WONNPatchAutoencoder,
    WinfreeFieldAutoencoder,
    WinfreeFieldDecoder,
    WinfreeFieldEncoder,
    WinfreeFieldLayer,
    WinfreePatchAutoencoder,
    phase_features,
    wrap_phase,
)

__all__ = [
    "OscillatorState",
    "AmplitudeVelocityOscillatorCell",
    "OscillatorySequenceLayer",
    "OscillatoryEncoder",
    "RepeatedLatentOscillatoryDecoder",
    "PositionalLatentOscillatoryDecoder",
    "AutoregressiveOscillatoryDecoder",
    "OscillatoryAutoencoder",
    "PatchOscillatoryAutoencoder",
    "AmplitudeVelocityHORNCell",
    "AmplitudeVelocityHORN",
    "AmplitudeVelocityEncoder",
    "AmplitudeVelocityDecoder",
    "AmplitudeVelocityAutoencoder",
    "WaveletOscillatorCell",
    "WaveletOscillatoryAutoencoder",
    "ProductionWaveletAutoencoder",
    "OscillatoryAutoencoderConfig",
    "PatchOscillatoryAutoencoderConfig",
    "WaveletAutoencoderConfig",
    "WinfreePhaseAutoencoderConfig",
    "WinfreeFieldAutoencoderConfig",
    "WinfreePatchAutoencoderConfig",
    "WinfreePhaseOscillatorCell",
    "WinfreePhaseSequenceLayer",
    "WinfreePhaseAutoencoder",
    "wrap_phase",
    "phase_features",
    "WinfreeFieldLayer",
    "WinfreeFieldEncoder",
    "WinfreeFieldDecoder",
    "WinfreeFieldAutoencoder",
    "WinfreePatchAutoencoder",
    "WONNPatchAutoencoder",
    "FractalHORNCell",
]
