# algorithms.py

"""
Registry of reconstruction algorithms.

Every algorithm is described by an :class:`AlgorithmSpec` giving its default
parameters and a factory that builds and runs it.  ``simulate.py`` only ever
talks to this registry, so adding a new method to the whole framework - sweeps,
plots, metric tables, output naming - means adding one entry here.

Registered names
----------------
``gn``      Gauss-Newton with a quadratic (smoothness / Tikhonov) prior
``tv``      Gauss-Newton with a smoothed total-variation prior
``l1``      L1-sparsity (Gehre et al., iterative soft thresholding)
``linear``  One-step linearised difference reconstruction
``dbar``    D-bar / Nachman direct method (nonlinear Fourier transform)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np
import torch
from dolfinx.fem import Function

from .gauss_newton import (
    GaussNewtonSolver,
    GaussNewtonSolverTV,
    LinearisedReconstruction,
)
from .sparsity_reconstruction import L1Sparsity
from .dbar import DbarSolver
from .regulariser import build_prior


@dataclass
class AlgorithmSpec:
    name: str
    label: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    runner: Callable = None
    #: parameters that select the prior matrix (rebuilt when they change)
    needs_prior: bool = False


# --------------------------------------------------------------------------- #
#  helpers shared by the runners
# --------------------------------------------------------------------------- #
def _initial_guess(solver, params):
    init = Function(solver.V_sigma)
    init.x.array[:] = float(params.get("background", 1.0))
    return init


def _prior_matrix(solver, params, cache=None):
    """
    Build (or fetch from ``cache``) the prior matrix for a Gauss-Newton run.

    The cache key includes every parameter that changes the matrix, so sweeping
    over e.g. ``lambda`` reuses one prior instead of rebuilding it per run.
    """
    kind = params.get("prior", "laplacian")
    kwargs = {}
    if kind == "gaussian":
        kwargs = {
            "corrlength": params.get("prior_corrlength", 0.2),
            "std": params.get("prior_std", 0.15),
            "cache_file": params.get("prior_cache_file"),
        }
    elif kind == "laplacian":
        kwargs = {"eps": params.get("prior_eps", 1e-4)}

    key = (id(solver), kind, tuple(sorted((k, str(v)) for k, v in kwargs.items())))
    if cache is not None and key in cache:
        return cache[key]

    R = build_prior(solver.omega, kind=kind, **kwargs)
    if cache is not None:
        cache[key] = R
    return R


def _torch_gamma(GammaInv, device):
    if GammaInv is None:
        return None
    return torch.as_tensor(np.asarray(GammaInv), dtype=torch.float64, device=device)


# --------------------------------------------------------------------------- #
#  runners
# --------------------------------------------------------------------------- #
def _run_gn(solver, U_meas, params, ctx):
    R = _prior_matrix(solver, params, ctx.get("prior_cache"))
    rec = GaussNewtonSolver(
        solver,
        device=params.get("device", "cpu"),
        R=R,
        lamb=params["lambda"],
        GammaInv=_torch_gamma(ctx.get("GammaInv"), params.get("device", "cpu")),
        clip=(params["sigma_min"], params["sigma_max"]),
        backCond=params.get("background", 1.0),
        num_steps=params["max_iter"],
        tol=params.get("tol", 1e-5),
        line_search=params.get("line_search", "armijo"),
        n_line_search=params.get("n_line_search", 6),
        lambda_scaling=params.get("lambda_scaling", "auto"),
    )
    out = rec.forward(
        Umeas=np.asarray(U_meas).flatten(),
        sigma_init=_initial_guess(solver, params),
        verbose=params.get("verbose", False),
    )
    return out, rec


def _run_tv(solver, U_meas, params, ctx):
    rec = GaussNewtonSolverTV(
        solver,
        device=params.get("device", "cpu"),
        num_steps=params["max_iter"],
        lamb=params["lambda"],
        beta=params["beta"],
        GammaInv=_torch_gamma(ctx.get("GammaInv"), params.get("device", "cpu")),
        clip=(params["sigma_min"], params["sigma_max"]),
        backCond=params.get("background", 1.0),
        tol=params.get("tol", 1e-5),
        line_search=params.get("line_search", "armijo"),
        n_line_search=params.get("n_line_search", 6),
        lambda_scaling=params.get("lambda_scaling", "auto"),
    )
    out = rec.forward(
        Umeas=np.asarray(U_meas).flatten(),
        sigma_init=_initial_guess(solver, params),
        verbose=params.get("verbose", False),
    )
    return out, rec


def _run_l1(solver, U_meas, params, ctx):
    rec = L1Sparsity(
        eit_solver=solver,
        backCond=params.get("background", 1.0),
        kappa=params["kappa"],
        clip=[params["sigma_min"], params["sigma_max"]],
        max_iter=params["max_iter"],
        stopping_criterion=params["stopping_criterion"],
        step_min=params["step_min"],
        initial_step_size=params["initial_step_size"],
        alpha=params["alpha"],
    )
    # L1Sparsity expects the (N, L) shaped data, not a flat vector
    out = rec.forward(
        Umeas=np.asarray(U_meas).reshape(-1, solver.L),
        alpha=params["alpha"],
        verbose=params.get("verbose", False),
    )
    return out, rec


def _run_dbar(solver, U_meas, params, ctx):
    # The homogeneous reference Lambda_1 depends only on the solver, the drive
    # and the background, never on the measurements, so one forward solve is
    # shared by every run of a sweep.
    cache = ctx.get("dbar_reference_cache")
    key = (id(solver), float(params.get("background", 1.0)))
    if cache is not None and key in cache:
        reference = cache[key]
    else:
        sig = Function(solver.V_sigma)
        sig.x.array[:] = float(params.get("background", 1.0))
        _, reference = solver.forward_solve(sig)
        reference = np.asarray(reference)
        if cache is not None:
            cache[key] = reference

    rec = DbarSolver(
        solver,
        backCond=params.get("background", 1.0),
        R=params["R"],
        k_grid=params["k_grid"],
        z_grid=params["z_grid"],
        scattering=params["scattering"],
        clip=(params["sigma_min"], params["sigma_max"]),
        contact_correction=params.get("contact_correction", True),
        quad_order=params.get("quad_order", 4),
        dn_rcond=params.get("dn_rcond"),
        t_cutoff=params.get("t_cutoff"),
        reference=reference,
    )
    out = rec.forward(
        Umeas=np.asarray(U_meas).reshape(-1, solver.L),
        verbose=params.get("verbose", False),
    )
    return out, rec


def _run_linear(solver, U_meas, params, ctx):
    R = _prior_matrix(solver, params, ctx.get("prior_cache"))
    rec = LinearisedReconstruction(
        solver,
        device=params.get("device", "cpu"),
        R=R,
        lamb=params["lambda"],
        GammaInv=_torch_gamma(ctx.get("GammaInv"), params.get("device", "cpu")),
        clip=(params["sigma_min"], params["sigma_max"]),
        backCond=params.get("background", 1.0),
        lambda_scaling=params.get("lambda_scaling", "auto"),
    )
    out = rec.forward(Umeas=np.asarray(U_meas).flatten())
    return out, rec


# --------------------------------------------------------------------------- #
#  registry
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, AlgorithmSpec] = {
    "gn": AlgorithmSpec(
        name="gn",
        label="Gauss-Newton (smoothness prior)",
        needs_prior=True,
        runner=_run_gn,
        defaults={
            # lambda is DIMENSIONLESS: it is rescaled internally by
            # trace(J^T W J)/trace(R), so lambda ~ 1 means "prior as strong as
            # the data" on any mesh, at any noise level.  Useful range ~1e-2..10.
            "lambda": 5e-2,
            "lambda_scaling": "auto",   # auto | absolute
            "max_iter": 12,
            "sigma_min": 0.01,
            "sigma_max": 5.0,
            "prior": "laplacian",   # laplacian | gaussian | tikhonov
            "prior_eps": 1e-4,
            "prior_corrlength": 0.2,
            "prior_std": 0.15,
            "tol": 1e-5,
            "line_search": "armijo",
            "n_line_search": 6,
        },
    ),
    "tv": AlgorithmSpec(
        name="tv",
        label="Gauss-Newton (total variation)",
        runner=_run_tv,
        defaults={
            # dimensionless, see the note on "gn". TV needs a larger value than
            # the quadratic prior because its Hessian trace is smaller.
            "lambda": 5.0,
            "lambda_scaling": "auto",
            "beta": 1e-6,
            "max_iter": 10,
            "sigma_min": 0.01,
            "sigma_max": 5.0,
            "tol": 1e-5,
            "line_search": "armijo",
            "n_line_search": 6,
        },
    ),
    "l1": AlgorithmSpec(
        name="l1",
        label="L1-sparsity",
        runner=_run_l1,
        defaults={
            # WARNING: unlike "lambda" above, alpha is NOT rescaled - the
            # iterative-soft-thresholding scheme of Gehre et al. does not admit
            # the same normalisation.  It therefore depends on the absolute
            # size of the measurements, i.e. on the drive amplitude and the
            # contact impedance.  If a run collapses to a flat background
            # (dynamic_range = 0) alpha is too large; if it is noisy, too
            # small.  Sweep it over decades when changing the drive settings.
            "alpha": 1e-5,
            "kappa": 0.0285,
            "sigma_min": 0.01,
            "sigma_max": 5.0,
            "max_iter": 200,
            "stopping_criterion": 5e-4,
            "step_min": 1e-6,
            "initial_step_size": 0.05,
        },
    ),
    "dbar": AlgorithmSpec(
        name="dbar",
        label="D-bar (Nachman direct method)",
        runner=_run_dbar,
        defaults={
            # R is the ONLY regularisation parameter: the scattering transform
            # is truncated to |k| <= R.  Small R -> smooth, stable, low
            # contrast; large R -> sharper but eventually pure noise, because
            # the transform is bilinear in e^{ikz} and its error grows like
            # e^{2|k|}.  3.5-4.5 suits 16-32 electrodes; sweep it and watch the
            # "t_growth" diagnostic (it must stay below ~1).
            "R": 4.0,
            "scattering": "exp",     # exp = Born (fast, robust) | bie = full
            "k_grid": 64,            # power of two
            "z_grid": 64,
            "sigma_min": 0.01,
            "sigma_max": 5.0,
            "contact_correction": True,
            "quad_order": 4,
            "t_cutoff": None,
        },
    ),
    "linear": AlgorithmSpec(
        name="linear",
        label="One-step linearised",
        needs_prior=True,
        runner=_run_linear,
        defaults={
            "lambda": 2e-1,
            "lambda_scaling": "auto",
            "sigma_min": 0.01,
            "sigma_max": 5.0,
            "prior": "laplacian",
            "prior_eps": 1e-4,
            "prior_corrlength": 0.2,
            "prior_std": 0.15,
        },
    ),
}


def available():
    return sorted(REGISTRY)


def get_spec(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {available()}"
        ) from None


def resolve_params(name, user_params=None, common=None):
    """
    Merge, in increasing priority: algorithm defaults, ``common`` settings that
    apply to every algorithm, and the user's per-algorithm overrides.
    """
    spec = get_spec(name)
    params = dict(spec.defaults)
    if common:
        params.update(common)
    if user_params:
        params.update(user_params)
    return params


def run(name, solver, U_meas, params, ctx=None):
    """
    Run one algorithm.

    Returns ``(sigma_dg0, info)`` with ``sigma_dg0`` a plain numpy array of
    cell-wise conductivities (so that CG1-based and DG0-based methods come back
    in the same, directly comparable form) and ``info`` carrying the iteration
    history and the simulated data at the solution.
    """
    spec = get_spec(name)
    ctx = ctx or {}

    sigma_fun, rec = spec.runner(solver, U_meas, params, ctx)

    sigma = rec.to_dg0(sigma_fun)

    # forward-simulate the reconstruction so data-space metrics are available
    sig_f = Function(solver.V_sigma)
    sig_f.x.array[:] = sigma
    _, U_rec = solver.forward_solve(sig_f)

    info = {
        "history": getattr(rec, "history", []),
        "U_reconstructed": np.asarray(U_rec),
        "label": spec.label,
    }
    diag = getattr(rec, "diagnostics", None)
    if diag:
        info["diagnostics"] = diag
    return sigma, info
