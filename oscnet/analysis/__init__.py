"""
OscNet Analysis Subpackage

This package contains modules for analyzing the dynamics and properties
of oscillatory neural networks and their components.
"""

from .phase_synchrony import (
    circular_difference,
    local_group_order,
    mean_neighbor_phase_difference,
    phase_order_parameter,
    trace_phase_summary,
)
from .generator_frontier import (
    GeneratorFrontierSummary,
    infer_generator_variant,
    plot_frontier,
    read_generator_sweep_csv,
    summarize_generator_frontier,
    write_frontier_csv,
    write_frontier_markdown,
)
from .generator_state_prior import (
    ClassStatePrior,
    encode_anchor_state_distribution,
    evaluate_state_prior_sampling_probe,
    fit_class_state_prior,
    generate_from_initial_states,
    nearest_reference_mse_summary,
    sample_class_state_prior,
    save_state_prior_contact_sheet,
    state_prior_to_json,
    write_state_prior_probe_csv,
)
from .reconstruction_diagnostics import (
    ReconstructionArtifactSummary,
    RunDiagnosticSummary,
    infer_changed_mask,
    latest_reconstruction_artifact,
    summarize_reconstruction_artifact,
    summarize_run_diagnostics,
    write_run_diagnostics_csv,
)

__all__ = [
    "circular_difference",
    "ClassStatePrior",
    "encode_anchor_state_distribution",
    "evaluate_state_prior_sampling_probe",
    "fit_class_state_prior",
    "generate_from_initial_states",
    "GeneratorFrontierSummary",
    "ReconstructionArtifactSummary",
    "RunDiagnosticSummary",
    "infer_changed_mask",
    "infer_generator_variant",
    "latest_reconstruction_artifact",
    "local_group_order",
    "mean_neighbor_phase_difference",
    "nearest_reference_mse_summary",
    "phase_order_parameter",
    "plot_frontier",
    "read_generator_sweep_csv",
    "sample_class_state_prior",
    "save_state_prior_contact_sheet",
    "state_prior_to_json",
    "summarize_reconstruction_artifact",
    "summarize_generator_frontier",
    "summarize_run_diagnostics",
    "trace_phase_summary",
    "write_frontier_csv",
    "write_frontier_markdown",
    "write_state_prior_probe_csv",
    "write_run_diagnostics_csv",
]

# from .stability import (
#     find_fixed_points,
#     compute_monodromy_matrix,
#     phase_space_trajectory
# )

# __all__ = [
#     "find_fixed_points",
#     "compute_monodromy_matrix",
#     "phase_space_trajectory"
# ]

# # Import submodules
# from . import edge_of_chaos
# from . import bifurcation
# from . import model_interface

# # Import specific functions to expose at the package level
# from .edge_of_chaos import optimize as optimize_edge_of_chaos
# from .bifurcation import plot as plot_bifurcation, parameter_grid
