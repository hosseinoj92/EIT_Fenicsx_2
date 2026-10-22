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

# Load optional reconstruction dependencies only when requested. The desktop
# configurator and SI physics tools can run without FEniCSx or PyTorch installed.
from importlib import import_module

_MODULE_EXPORTS = {
    "eit_forward_fenicsx": ["EIT"],
    "gauss_newton": ["GaussNewtonSolver", "GaussNewtonSolverTV", "LinearisedReconstruction"],
    "reconstructor": ["Reconstructor"],
    "sparsity_reconstruction": ["L1Sparsity"],
    "dbar": ["DbarSolver"],
    "utils": ["current_method", "default_n_patterns", "validate_injection", "DRIVE_NAMES", "interpolate_mesh_to_mesh"],
    "regulariser": ["build_prior"],
    "metrics": ["MetricEvaluator", "data_residual", "all_metrics", "LOWER_IS_BETTER"],
    "noise": ["add_measurement_noise", "noise_std", "gamma_inv", "snr_db"],
    "phantoms": ["build_phantom", "resolve_phantom", "describe_phantom", "PRESETS"],
}


def __getattr__(name):
    if name in ("algorithms", "gt_io"):
        value = import_module(f".{name}", __name__)
    else:
        module = next((m for m, names in _MODULE_EXPORTS.items() if name in names), None)
        if module is None:
            raise AttributeError(name)
        value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "EIT",
    "GaussNewtonSolver",
    "GaussNewtonSolverTV",
    "LinearisedReconstruction",
    "L1Sparsity",
    "DbarSolver",
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
