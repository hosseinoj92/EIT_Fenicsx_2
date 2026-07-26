# gauss_newton.py

"""
Gauss-Newton solvers for the Complete Electrode Model.

Three variants are provided:

``GaussNewtonSolver``
    Iteratively regularised Gauss-Newton with a quadratic prior
    ``0.5 * lamb * (sigma - sigma_prior)^T R (sigma - sigma_prior)``.

``GaussNewtonSolverTV``
    Gauss-Newton with a smoothed total-variation prior
    ``lamb * sum sqrt((L_tv sigma)^2 + beta)``, linearised with the standard
    lagged-diffusivity (IRLS) weight ``E = diag(1/sqrt((L_tv sigma)^2 + beta))``.

``LinearisedReconstruction``
    One-step linearisation around a homogeneous background (difference imaging).

Objective and normal equations
------------------------------
All solvers minimise

    Phi(sigma) = 0.5 * sum_k w_k (F(sigma)_k - U_k)^2 + P(sigma)

with ``w = GammaInv`` the inverse noise variance per measurement.  With the
Jacobian convention of ``eit_forward_fenicsx`` (``J = -dF/dsigma``) the
Gauss-Newton system is

    (J^T W J + P'')  d = J^T W r - P'          with r = F(sigma) - U

and the update is ``sigma <- sigma + t d``.  The step length ``t`` is chosen by
a line search **on the same objective Phi**, so the search and the step
direction are consistent.
"""

import os
import warnings

import numpy as np
from tqdm import tqdm

import torch

# PETSc/OpenBLAS and torch each start their own OpenMP thread pool.  On macOS
# the two runtimes collide and segfault inside the dense solve, so torch is
# pinned to a single thread unless the caller asks otherwise.  Relying on the
# OMP_NUM_THREADS environment variable is not enough: it only takes effect if
# it is set before the libraries are loaded, which a plain `import` cannot
# guarantee.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
from dolfinx.fem import Function
from scipy.sparse import csr_array, diags

from .eit_forward_fenicsx import EIT
from .reconstructor import Reconstructor


def _as_torch(x, device, dtype=torch.float64):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _weighted_normal_equations(J, r, w):
    """``(J^T W J, J^T W r)`` without materialising the dense diagonal W."""
    if w is None:
        return J.T @ J, J.T @ r
    Jw = J * w[:, None]
    return J.T @ Jw, Jw.T @ r


class _GaussNewtonBase(Reconstructor):
    """Shared machinery: line search, convergence tracking, prior interface."""

    def __init__(self, eit_solver: EIT, device="cpu", clip=(0.001, 3.0),
                 backCond=1.0, GammaInv=None, Uel_background=None,
                 line_search="armijo", n_line_search=6, dtype=torch.float64,
                 lambda_scaling="auto"):
        super().__init__(eit_solver)
        self.device = device
        self.dtype = dtype
        self.clip = list(clip)
        self.backCond = float(backCond)
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.line_search = line_search
        self.n_line_search = int(n_line_search)
        self.lambda_scaling = lambda_scaling
        self.history = []
        self._lamb_eff = None
        self._lambda_scale = 1.0

        n = eit_solver.n_cells
        if n > 12000:
            warnings.warn(
                f"Reconstruction mesh has {n} cells; the Gauss-Newton normal "
                f"equations are dense ({n * n * 8 / 1e9:.1f} GB, O(n^3) solve). "
                "Use a coarser reconstruction mesh.",
                stacklevel=3,
            )

    # -- pieces the subclasses fill in ---------------------------------- #
    def _penalty(self, sigma):
        """Value of the regularisation term P(sigma)."""
        return 0.0

    def _penalty_system(self, sigma):
        """``(P'', P')`` as torch tensors, evaluated at ``sigma``."""
        raise NotImplementedError

    def _unit_penalty_trace(self, sigma):
        """
        ``trace`` of the penalty Hessian at ``sigma`` for ``lamb = 1``.

        Used to make ``lamb`` dimensionless; return ``None`` if the algorithm
        has no quadratic penalty.
        """
        return None

    # -- regularisation scaling ------------------------------------------ #
    def _set_lambda_scale(self, J, w, sigma):
        """
        Put ``lamb`` on a dimensionless footing.

        The natural size of the data term ``J^T W J`` depends on the mesh, the
        injected current amplitude and the noise weighting, and it changes by
        orders of magnitude between configurations.  An *absolute* ``lamb`` is
        therefore not comparable across the settings this framework is built to
        compare - the same number can be far too weak on one mesh and far too
        strong on another, which shows up as an unregularised, noise-dominated
        image (the Jacobian is heavily rank deficient: for 16 electrodes it has
        at most a few hundred independent rows against thousands of unknowns).

        With ``lambda_scaling="auto"`` the applied weight is

            lamb_eff = lamb * trace(J^T W J) / trace(P''|_{lamb=1})

        so ``lamb = 1`` means "prior term comparable to data term" on any mesh,
        at any noise level and for any drive amplitude.  Use
        ``lambda_scaling="absolute"`` to pass the raw value through.
        """
        if self.lambda_scaling != "auto":
            self._lambda_scale = 1.0
            self._lamb_eff = self.lamb
            return

        p_trace = self._unit_penalty_trace(sigma)
        if p_trace is None or p_trace <= 0:
            self._lambda_scale = 1.0
            self._lamb_eff = self.lamb
            return

        if w is None:
            d_trace = float(np.sum(J * J))
        else:
            d_trace = float(np.sum(w[:, None] * J * J))

        self._lambda_scale = d_trace / p_trace
        self._lamb_eff = self.lamb * self._lambda_scale

    # -- shared ---------------------------------------------------------- #
    def _data_misfit(self, U, Umeas, w):
        r = U - Umeas
        if w is None:
            return 0.5 * float(np.sum(r**2))
        return 0.5 * float(np.sum(w * r**2))

    def _objective(self, sigma_vec, Umeas, w):
        f = Function(self.eit_solver.V_sigma)
        f.x.array[:] = sigma_vec
        _, U = self.eit_solver.forward_solve(f)
        return self._data_misfit(np.asarray(U).flatten(), Umeas, w) + self._penalty(
            sigma_vec
        )

    def _search_step(self, sigma, delta, Umeas, w, phi_current):
        """
        Return ``(step, new_sigma, phi_new)``.

        ``armijo``: backtracking from 1.0, usually 1-3 forward solves.
        ``grid``:   evaluate a fixed grid of step sizes (the original behaviour).
        """
        lo, hi = self.clip

        def trial(t):
            s = np.clip(sigma + t * delta, lo, hi)
            return s, self._objective(s, Umeas, w)

        if self.line_search == "grid":
            steps = np.linspace(0.01, 1.0, self.n_line_search)
            cands = [trial(t) for t in steps]
            k = int(np.argmin([c[1] for c in cands]))
            return float(steps[k]), cands[k][0], cands[k][1]

        t = 1.0
        best = None
        for _ in range(self.n_line_search):
            s_new, phi_new = trial(t)
            if best is None or phi_new < best[2]:
                best = (t, s_new, phi_new)
            if phi_new < phi_current:
                return t, s_new, phi_new
            t *= 0.5
        # nothing decreased the objective: return the least-bad candidate
        return best

    def _solve_normal_equations(self, A, b):
        try:
            return torch.linalg.solve(A, b)
        except Exception:
            # Singular/ill-conditioned: fall back to a least-squares solve with
            # a small relative diagonal shift rather than crashing the run.
            shift = 1e-10 * float(torch.diagonal(A).abs().max())
            eye = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            warnings.warn(
                "Gauss-Newton normal equations were singular; adding a "
                f"diagonal shift of {shift:.2e}.",
                stacklevel=3,
            )
            return torch.linalg.lstsq(A + shift * eye, b.unsqueeze(-1)).solution.squeeze(-1)

    def _iterate(self, Umeas, sigma_init, num_steps, tol, verbose):
        """The common Gauss-Newton loop."""
        solver = self.eit_solver
        Umeas = np.asarray(Umeas, dtype=float).flatten()

        w = None
        if self.GammaInv is not None:
            w = np.asarray(
                self.GammaInv.cpu().numpy()
                if isinstance(self.GammaInv, torch.Tensor)
                else self.GammaInv,
                dtype=float,
            ).flatten()
            if w.shape != Umeas.shape:
                raise ValueError(
                    f"GammaInv has {w.shape[0]} entries but there are "
                    f"{Umeas.shape[0]} measurements"
                )

        if sigma_init is None:
            sigma_init = Function(solver.V_sigma)
            sigma_init.x.array[:] = self.backCond
        sigma = np.array(sigma_init.x.array[:], dtype=float)

        w_t = None if w is None else _as_torch(w, self.device, self.dtype)
        self.history = []

        sigma_k = Function(solver.V_sigma)
        phi = None

        with tqdm(total=num_steps, disable=not verbose) as pbar:
            for i in range(num_steps):
                sigma_k.x.array[:] = sigma

                u_all, Usim = solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()
                J = solver.calc_jacobian(sigma_k, u_all)

                if i == 0:
                    # fix the regularisation scale once, from the initial state
                    self._set_lambda_scale(J, w, sigma)

                if self.Uel_background is not None and i == 0:
                    r = np.asarray(self.Uel_background).flatten() - Umeas
                else:
                    r = Usim - Umeas

                if phi is None:
                    phi = self._data_misfit(Usim, Umeas, w) + self._penalty(sigma)

                J_t = _as_torch(J, self.device, self.dtype)
                r_t = _as_torch(r, self.device, self.dtype)

                A, b = _weighted_normal_equations(J_t, r_t, w_t)

                P_hess, P_grad = self._penalty_system(sigma)
                if P_hess is not None:
                    A = A + P_hess
                if P_grad is not None:
                    b = b - P_grad

                delta = self._solve_normal_equations(A, b).cpu().numpy()

                step, sigma_new, phi_new = self._search_step(sigma, delta, Umeas, w, phi)

                rel_change = np.linalg.norm(sigma_new - sigma) / max(
                    np.linalg.norm(sigma_new), 1e-30
                )
                sigma, phi = sigma_new, phi_new

                self.history.append(
                    {"iter": i, "objective": phi, "step": step, "rel_change": rel_change}
                )

                pbar.set_description(
                    f"Relative Change: {rel_change:.4g} | "
                    f"Obj. fun: {phi:.6g} | Step size: {step:.4g}"
                )
                pbar.update(1)

                if rel_change < tol:
                    if verbose:
                        print(f"Converged: relative change {rel_change:.2e} < {tol:.1e}")
                    break

        sigma_reco = Function(solver.V_sigma)
        sigma_reco.x.array[:] = sigma
        return sigma_reco


class GaussNewtonSolver(_GaussNewtonBase):
    """
    Iteratively regularised Gauss-Newton with the quadratic prior

        P(sigma) = 0.5 * lamb * (sigma - sigma_prior)^T R (sigma - sigma_prior)

    ``R`` may be

    * ``None``            - no regularisation (ill-posed, only for testing)
    * ``"Tikhonov"``      - identity matrix
    * ``"LM"``            - Levenberg-Marquardt damping ``lamb * diag(diag(A))``
    * an array / tensor   - e.g. ``Lprior.T @ Lprior`` from ``regulariser.py``
    """

    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        num_steps: int = 50,
        R=None,
        lamb: float = 1.0,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=(0.001, 3.0),
        backCond: float = 1.0,
        sigma_prior=None,
        tol: float = 1e-5,
        line_search: str = "armijo",
        n_line_search: int = 6,
        lambda_scaling: str = "auto",
    ):
        super().__init__(
            eit_solver, device=device, clip=clip, backCond=backCond,
            GammaInv=GammaInv, Uel_background=Uel_background,
            line_search=line_search, n_line_search=n_line_search,
            lambda_scaling=lambda_scaling,
        )
        self.num_steps = num_steps
        self.R = R
        self.lamb = lamb
        self.tol = tol
        self.sigma_prior = sigma_prior
        self._R_t = None
        self._lm = False

    def _prepare_R(self, n):
        """Materialise ``R`` once per ``forward`` call."""
        self._lm = False
        if self.R is None:
            self._R_t = None
        elif isinstance(self.R, str):
            if self.R == "Tikhonov":
                self._R_t = torch.eye(n, device=self.device, dtype=self.dtype)
            elif self.R == "LM":
                self._R_t = None
                self._lm = True
            else:
                raise ValueError(
                    f"Unknown string for R: {self.R}. Choices [Tikhonov, LM]"
                )
        else:
            R = self.R
            if hasattr(R, "toarray"):  # scipy sparse
                R = R.toarray()
            self._R_t = _as_torch(R, self.device, self.dtype)
            if self._R_t.shape != (n, n):
                raise ValueError(
                    f"Prior matrix R has shape {tuple(self._R_t.shape)}, expected "
                    f"({n}, {n}). It was probably built for a different mesh."
                )

    def _unit_penalty_trace(self, sigma):
        if self._R_t is None:
            return None
        return float(torch.diagonal(self._R_t).sum())

    def _penalty(self, sigma):
        if self._R_t is None:
            return 0.0
        lamb = self._lamb_eff if self._lamb_eff is not None else self.lamb
        d = _as_torch(sigma - self._prior_vec, self.device, self.dtype)
        return 0.5 * lamb * float(d @ (self._R_t @ d))

    def _penalty_system(self, sigma):
        if self._R_t is None:
            return None, None
        lamb = self._lamb_eff if self._lamb_eff is not None else self.lamb
        d = _as_torch(sigma - self._prior_vec, self.device, self.dtype)
        return lamb * self._R_t, lamb * (self._R_t @ d)

    def _solve_normal_equations(self, A, b):
        if self._lm:  # Levenberg-Marquardt damping, built from A itself
            A = A + self.lamb * torch.diag(torch.diagonal(A)) \
                  + 0.5 * self.lamb * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return super()._solve_normal_equations(A, b)

    def forward(self, Umeas: np.array, **kwargs):
        verbose = kwargs.get("verbose", False)
        sigma_init = kwargs.get("sigma_init", None)

        self.num_steps = kwargs.get("num_steps", self.num_steps)
        self.lamb = kwargs.get("lamb", self.lamb)
        self.R = kwargs.get("R", self.R)
        self.GammaInv = kwargs.get("GammaInv", self.GammaInv)
        self.clip = list(kwargs.get("clip", self.clip))
        self.tol = kwargs.get("tol", self.tol)

        n = self.eit_solver.n_cells
        prior = kwargs.get("sigma_prior", self.sigma_prior)
        if prior is None:
            prior = np.full(n, self.backCond)
        self._prior_vec = np.asarray(prior, dtype=float).flatten()

        self._prepare_R(n)

        return self._iterate(Umeas, sigma_init, self.num_steps, self.tol, verbose)


class GaussNewtonSolverTV(_GaussNewtonBase):
    """
    Gauss-Newton with the smoothed total-variation prior

        P(sigma) = lamb * sum_e sqrt((L_tv sigma)_e^2 + beta)

    linearised by lagged diffusivity:  P'' = lamb L^T E L,  P' = lamb L^T E L sigma
    with ``E = diag(1 / sqrt((L sigma)^2 + beta))``.
    """

    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        num_steps: int = 20,
        lamb: float = 0.04,
        beta: float = 1e-6,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=(0.001, 3.0),
        backCond: float = 1.0,
        tol: float = 1e-5,
        line_search: str = "armijo",
        n_line_search: int = 6,
        lambda_scaling: str = "auto",
        **kwargs,
    ):
        super().__init__(
            eit_solver, device=device, clip=clip, backCond=backCond,
            GammaInv=GammaInv, Uel_background=Uel_background,
            line_search=line_search, n_line_search=n_line_search,
            lambda_scaling=lambda_scaling,
        )
        self.Ltv = self.construct_tv_matrix()

        self.num_steps = num_steps
        self.lamb = lamb
        self.beta = beta
        self.tol = tol

    def construct_tv_matrix(self):
        """
        Sparse finite-difference operator over interior mesh facets:
        one row per interior facet with ``+1`` / ``-1`` in the two adjacent
        cells, so ``(L_tv sigma)_e`` is the conductivity jump across facet ``e``.

        Iterating over facets (rather than over cells and their facets) keeps
        each interior facet exactly once; the cell-based loop used previously
        emitted every row twice, which silently doubled the effective ``lamb``
        and the memory footprint.
        """
        topo = self.eit_solver.omega.topology
        topo.create_connectivity(1, 2)  # facet -> cell

        edge_to_cell = topo.connectivity(1, 2)
        num_facets = topo.index_map(1).size_local
        num_cells = topo.index_map(2).size_local

        rows, cols, data = [], [], []
        row_idx = 0
        for edge in range(num_facets):
            adjacent_cells = edge_to_cell.links(edge)
            if len(adjacent_cells) > 1:  # interior facet
                rows.extend((row_idx, row_idx))
                cols.extend((adjacent_cells[0], adjacent_cells[1]))
                data.extend((1.0, -1.0))
                row_idx += 1

        return csr_array((data, (rows, cols)), shape=(row_idx, num_cells))

    def _tv_terms(self, sigma):
        Ls = self.Ltv @ np.asarray(sigma, dtype=float)
        eta = np.sqrt(Ls**2 + self.beta)
        return Ls, eta

    def _unit_penalty_trace(self, sigma):
        _, eta = self._tv_terms(sigma)
        LEL = self.Ltv.T @ diags(1.0 / eta) @ self.Ltv
        return float(LEL.diagonal().sum())

    def _penalty(self, sigma):
        lamb = self._lamb_eff if self._lamb_eff is not None else self.lamb
        _, eta = self._tv_terms(sigma)
        return float(lamb * eta.sum())

    def _penalty_system(self, sigma):
        lamb = self._lamb_eff if self._lamb_eff is not None else self.lamb
        _, eta = self._tv_terms(sigma)
        E = diags(1.0 / eta)  # sparse: dense np.diag(1/eta) was ~n_facets^2
        LEL = (self.Ltv.T @ E @ self.Ltv).toarray()
        LEL_t = _as_torch(lamb * LEL, self.device, self.dtype)
        sig_t = _as_torch(sigma, self.device, self.dtype)
        return LEL_t, LEL_t @ sig_t

    def forward(self, Umeas: np.array, **kwargs):
        verbose = kwargs.get("verbose", False)
        sigma_init = kwargs.get("sigma_init", None)

        self.num_steps = kwargs.get("num_steps", self.num_steps)
        self.lamb = kwargs.get("lamb", self.lamb)
        self.beta = kwargs.get("beta", self.beta)
        self.GammaInv = kwargs.get("GammaInv", self.GammaInv)
        self.clip = list(kwargs.get("clip", self.clip))
        self.tol = kwargs.get("tol", self.tol)

        return self._iterate(Umeas, sigma_init, self.num_steps, self.tol, verbose)

    def single_step(self, sigma, Umeas):
        """One Gauss-Newton direction at ``sigma`` (no line search)."""
        sigma = np.asarray(sigma, dtype=float)
        sigma_k = Function(self.eit_solver.V_sigma)
        sigma_k.x.array[:] = sigma

        u_all, Usim = self.eit_solver.forward_solve(sigma_k)
        r = np.asarray(Usim).flatten() - np.asarray(Umeas).flatten()
        J = self.eit_solver.calc_jacobian(sigma_k, u_all)

        J_t = _as_torch(J, self.device, self.dtype)
        r_t = _as_torch(r, self.device, self.dtype)
        w_t = None if self.GammaInv is None else _as_torch(self.GammaInv, self.device, self.dtype)

        A, b = _weighted_normal_equations(J_t, r_t, w_t)
        P_hess, P_grad = self._penalty_system(sigma)
        return self._solve_normal_equations(A + P_hess, b - P_grad)


class LinearisedReconstruction(Reconstructor):
    """
    One-step linearised (difference) reconstruction around a homogeneous
    background:  ``sigma = sigma_0 + (J^T W J + lamb R)^-1 J^T W (U_bg - U_meas)``.

    The Jacobian is computed once at ``backCond`` and cached, which makes this
    by far the cheapest method - useful as a baseline and for time-difference
    imaging.
    """

    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        R=None,
        lamb: float = 1.0,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=(0.001, 3.0),
        backCond: float = 1.0,
        dtype=torch.float64,
        lambda_scaling: str = "auto",
    ):
        super().__init__(eit_solver)

        self.device = device
        self.dtype = dtype
        self.R = R
        self.lamb = lamb
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = list(clip)
        self.backCond = backCond
        self.lambda_scaling = lambda_scaling

        self.J, self.U_background = self.calculate_jacobian()
        self.J = _as_torch(self.J, self.device, self.dtype)
        if self.Uel_background is None:
            self.Uel_background = self.U_background

    def calculate_jacobian(self):
        sigma_k = Function(self.eit_solver.V_sigma)
        sigma_k.x.array[:] = self.backCond

        u_all, Usim = self.eit_solver.forward_solve(sigma_k)
        J = self.eit_solver.calc_jacobian(sigma_k, u_all)
        return J, np.asarray(Usim).flatten()

    def _regulariser(self, n):
        if self.R is None:
            return torch.eye(n, device=self.device, dtype=self.dtype)
        if isinstance(self.R, str):
            if self.R == "Tikhonov":
                return torch.eye(n, device=self.device, dtype=self.dtype)
            raise ValueError(f"Unknown string for R: {self.R}. Choices [Tikhonov]")
        R = self.R.toarray() if hasattr(self.R, "toarray") else self.R
        return _as_torch(R, self.device, self.dtype)

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = np.asarray(Umeas, dtype=float).flatten()
        lamb = kwargs.get("lamb", self.lamb)

        # Note the sign: with J = -dF/dsigma, delta = (..)^-1 J^T W (F(sigma_0) - U)
        r = np.asarray(self.Uel_background).flatten() - Umeas
        r_t = _as_torch(r, self.device, self.dtype)

        w_t = None
        if self.GammaInv is not None:
            w_t = _as_torch(self.GammaInv, self.device, self.dtype)

        A, b = _weighted_normal_equations(self.J, r_t, w_t)
        R = self._regulariser(self.J.shape[1])

        # dimensionless lamb, see _GaussNewtonBase._set_lambda_scale
        if self.lambda_scaling == "auto":
            p_trace = float(torch.diagonal(R).sum())
            if p_trace > 0:
                lamb = lamb * float(torch.diagonal(A).sum()) / p_trace
        self._lamb_eff = lamb

        A = A + lamb * R

        delta_sigma = torch.linalg.solve(A, b).cpu().numpy()
        sigma = np.clip(self.backCond + delta_sigma, self.clip[0], self.clip[1])

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco
