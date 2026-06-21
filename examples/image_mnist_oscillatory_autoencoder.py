"""MNIST oscillator autoencoder reference benchmark entrypoint.

Run this example with:

    python examples/image_mnist_oscillatory_autoencoder.py --help
"""

from oscnet.experiments.mnist_autoencoder import (
    build_arg_parser,
    config_from_args,
    export_encoder_complex_states,
    load_mnist_data,
    main,
    prepare_reconstructions,
    run_mnist_experiment,
)
from oscnet.models import (
    AmplitudeVelocityAutoencoder,
    FeedForwardPatchAutoencoder,
    RecurrentConvPatchDenoiser,
    WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser,
    WinfreeCoarseRatePhaseConditionalPatchDenoiser,
    WinfreeConditionalPatchDenoiser,
    WinfreeGlobalRatePhaseConditionalPatchDenoiser,
    WinfreePatchAutoencoder,
    WinfreeRatePhaseConditionalPatchDenoiser,
)

__all__ = [
    "AmplitudeVelocityAutoencoder",
    "FeedForwardPatchAutoencoder",
    "RecurrentConvPatchDenoiser",
    "WinfreeCoarseGlobalRatePhaseConditionalPatchDenoiser",
    "WinfreeCoarseRatePhaseConditionalPatchDenoiser",
    "WinfreeConditionalPatchDenoiser",
    "WinfreeGlobalRatePhaseConditionalPatchDenoiser",
    "WinfreePatchAutoencoder",
    "WinfreeRatePhaseConditionalPatchDenoiser",
    "build_arg_parser",
    "config_from_args",
    "export_encoder_complex_states",
    "load_mnist_data",
    "main",
    "prepare_reconstructions",
    "run_mnist_experiment",
]


if __name__ == "__main__":
    main()
