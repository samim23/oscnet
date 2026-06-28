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
from .mnist_jepa import (
    MNISTJEPAExperimentConfig,
    compute_patch_embeddings,
    compute_hidden_patch_weights,
    dct_lowfreq_basis,
    run_mnist_jepa_experiment,
)
from .mnist_generator import (
    MNISTGeneratorExperimentConfig,
    make_projection_matrix,
    run_mnist_generator_experiment,
    sliced_wasserstein_loss,
)
from .mnist_phase_vae import (
    MNISTPhaseVAEExperimentConfig,
    run_mnist_phase_vae_experiment,
)
from .mnist_phase_flow import (
    MNISTPhaseFlowExperimentConfig,
    run_mnist_phase_flow_experiment,
)
from .mnist_shape_pixel import (
    MNISTShapePixelExperimentConfig,
    run_mnist_shape_pixel_experiment,
)
from .results import (
    DEFAULT_RESULT_METRICS,
    ExperimentSummaryRow,
    collect_experiment_summaries,
    find_experiment_runs,
    format_comparison_table,
    load_experiment_summary,
    write_comparison_csv,
)

__all__ = [
    "AutoencoderExperimentConfig",
    "AutoencoderExperimentResult",
    "DEFAULT_RESULT_METRICS",
    "ExperimentPaths",
    "ExperimentSummaryRow",
    "MNISTJEPAExperimentConfig",
    "MNISTGeneratorExperimentConfig",
    "MNISTPhaseVAEExperimentConfig",
    "MNISTPhaseFlowExperimentConfig",
    "MNISTShapePixelExperimentConfig",
    "collect_experiment_summaries",
    "collect_sequence_state_trace",
    "compute_hidden_patch_weights",
    "compute_patch_embeddings",
    "dct_lowfreq_basis",
    "evaluate_autoencoder",
    "find_experiment_runs",
    "format_comparison_table",
    "load_experiment_summary",
    "make_projection_matrix",
    "prepare_experiment_paths",
    "run_eval_only",
    "run_mnist_jepa_experiment",
    "run_mnist_generator_experiment",
    "run_mnist_phase_vae_experiment",
    "run_mnist_phase_flow_experiment",
    "run_mnist_shape_pixel_experiment",
    "sliced_wasserstein_loss",
    "train_autoencoder",
    "write_comparison_csv",
]
