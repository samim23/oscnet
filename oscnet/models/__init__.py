"""
Neural network models based on oscillatory dynamics.

This module provides various implementations of neural networks
based on oscillatory dynamics and ODEs.
"""

from .config import (
    OscillatoryAutoencoderConfig,
    PatchOscillatoryAutoencoderConfig,
    WaveletAutoencoderConfig,
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
    "WinfreePhaseOscillatorCell",
    "WinfreePhaseSequenceLayer",
    "WinfreePhaseAutoencoder",
    "FractalHORNCell",
]
