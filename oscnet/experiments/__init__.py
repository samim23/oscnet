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
    "collect_experiment_summaries",
    "collect_sequence_state_trace",
    "compute_hidden_patch_weights",
    "compute_patch_embeddings",
    "dct_lowfreq_basis",
    "evaluate_autoencoder",
    "find_experiment_runs",
    "format_comparison_table",
    "load_experiment_summary",
    "prepare_experiment_paths",
    "run_eval_only",
    "run_mnist_jepa_experiment",
    "train_autoencoder",
    "write_comparison_csv",
]
