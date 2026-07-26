"""
eitcore - the EIT simulation and reconstruction library.

Everything here is imported by the three top-level scripts:

    mesh_tools.py         builds the finite-element meshes
    make_ground_truth.py  creates a synthetic truth + its measurements
    simulate.py           reconstructs that truth and compares algorithms

You normally do not need to touch this package: it is configured entirely
through the CONFIG dictionaries in those scripts.  Import it directly only when
scripting something the two entry points do not cover, e.g.

    from eitcore import EIT, build_phantom, add_measurement_noise
"""

from .eit_forward_fenicsx import EIT
from .gauss_newton import (
    GaussNewtonSolver,
    GaussNewtonSolverTV,
    LinearisedReconstruction,
)
from .reconstructor import Reconstructor
from .sparsity_reconstruction import L1Sparsity
from .utils import (
    current_method,
    default_n_patterns,
    validate_injection,
    DRIVE_NAMES,
    interpolate_mesh_to_mesh,
)
from .regulariser import build_prior
from .metrics import MetricEvaluator, data_residual, all_metrics, LOWER_IS_BETTER
from .noise import add_measurement_noise, noise_std, gamma_inv, snr_db
from .phantoms import build_phantom, resolve_phantom, describe_phantom, PRESETS
from . import algorithms
from . import gt_io

__all__ = [
    "EIT",
    "GaussNewtonSolver",
    "GaussNewtonSolverTV",
    "LinearisedReconstruction",
    "L1Sparsity",
    "Reconstructor",
    "current_method",
    "default_n_patterns",
    "validate_injection",
    "DRIVE_NAMES",
    "interpolate_mesh_to_mesh",
    "build_prior",
    "MetricEvaluator",
    "data_residual",
    "all_metrics",
    "LOWER_IS_BETTER",
    "add_measurement_noise",
    "noise_std",
    "gamma_inv",
    "snr_db",
    "build_phantom",
    "resolve_phantom",
    "describe_phantom",
    "PRESETS",
    "algorithms",
    "gt_io",
]
