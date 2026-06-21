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
    "ReconstructionArtifactSummary",
    "RunDiagnosticSummary",
    "infer_changed_mask",
    "latest_reconstruction_artifact",
    "local_group_order",
    "mean_neighbor_phase_difference",
    "phase_order_parameter",
    "summarize_reconstruction_artifact",
    "summarize_run_diagnostics",
    "trace_phase_summary",
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
