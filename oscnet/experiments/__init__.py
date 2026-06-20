"""Reference experiment harnesses for OscNet models."""

from .harness import (
    AutoencoderExperimentConfig,
    AutoencoderExperimentResult,
    ExperimentPaths,
    collect_sequence_state_trace,
    evaluate_autoencoder,
    prepare_experiment_paths,
    run_eval_only,
    train_autoencoder,
)

__all__ = [
    "AutoencoderExperimentConfig",
    "AutoencoderExperimentResult",
    "ExperimentPaths",
    "collect_sequence_state_trace",
    "evaluate_autoencoder",
    "prepare_experiment_paths",
    "run_eval_only",
    "train_autoencoder",
]
