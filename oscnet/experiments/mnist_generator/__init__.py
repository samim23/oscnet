"""MNIST oscillator generator experiment package."""

from .artifacts import save_mnist_generator_artifacts
from .builder import build_mnist_generator_model
from .cli import build_arg_parser, config_from_args, main, parse_args
from .config import MNISTGeneratorExperimentConfig
from .features import (
    MNISTFeatureClassifier,
    compute_class_prototypes,
    make_projection_matrix,
    mnist_feature_map,
    mnist_structural_features,
    train_mnist_feature_classifier,
)
from .losses import (
    conditional_feature_drift_loss,
    conditional_pixel_drift_loss,
    generator_distribution_loss,
    generator_loss,
    sliced_wasserstein_loss,
)
from .metrics import (
    compute_generator_quality_metrics,
    compute_generator_settling_metrics,
    compute_generator_success_diagnostics,
    sample_generator_images,
)
from .presets import GENERATOR_PRESETS, preset_defaults
from .queue import MNISTDriftQueue
from .runner import evaluate_generator_loss, run_mnist_generator_experiment

__all__ = [
    "GENERATOR_PRESETS",
    "MNISTDriftQueue",
    "MNISTFeatureClassifier",
    "MNISTGeneratorExperimentConfig",
    "build_arg_parser",
    "build_mnist_generator_model",
    "compute_class_prototypes",
    "compute_generator_quality_metrics",
    "compute_generator_settling_metrics",
    "compute_generator_success_diagnostics",
    "conditional_feature_drift_loss",
    "conditional_pixel_drift_loss",
    "config_from_args",
    "evaluate_generator_loss",
    "generator_distribution_loss",
    "generator_loss",
    "main",
    "make_projection_matrix",
    "mnist_feature_map",
    "mnist_structural_features",
    "parse_args",
    "preset_defaults",
    "run_mnist_generator_experiment",
    "sample_generator_images",
    "save_mnist_generator_artifacts",
    "sliced_wasserstein_loss",
    "train_mnist_feature_classifier",
]
