"""Audio wavelet oscillator autoencoder reference benchmark entrypoint.

Run this example with:

    python examples/audio_wavelet_oscillatory_autoencoder.py --help
"""

from oscnet.experiments.audio_wavelet import (
    HAS_JAXWT,
    HAS_SOUNDFILE,
    audio_to_wavelets,
    build_arg_parser,
    config_from_args,
    create_temporal_wavelet_sequences,
    load_audio_files,
    main,
    prepare_audio_experiment_data,
    run_audio_wavelet_experiment,
    smart_wavelet_compression,
    smart_wavelet_decompression,
    wavelets_to_audio,
)
from oscnet.models import ProductionWaveletAutoencoder, WaveletOscillatorCell

__all__ = [
    "HAS_JAXWT",
    "HAS_SOUNDFILE",
    "ProductionWaveletAutoencoder",
    "WaveletOscillatorCell",
    "audio_to_wavelets",
    "build_arg_parser",
    "config_from_args",
    "create_temporal_wavelet_sequences",
    "load_audio_files",
    "main",
    "prepare_audio_experiment_data",
    "run_audio_wavelet_experiment",
    "smart_wavelet_compression",
    "smart_wavelet_decompression",
    "wavelets_to_audio",
]


if __name__ == "__main__":
    main()
