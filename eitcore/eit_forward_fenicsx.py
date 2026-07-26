# eit_forward_fenicsx.py

"""
Forward solver for the Complete Electrode Model (CEM) of Electrical Impedance
Tomography, discretised with FEniCSx.

Model
-----
Find the potential ``u`` in the domain ``Omega`` and the electrode potentials
``U_i`` such that

    -div(sigma grad u) = 0                      in Omega
    u + z_i sigma du/dn = U_i                   on electrode e_i
    sigma du/dn = 0                             on the electrode-free boundary
    int_{e_i} sigma du/dn ds = I_i              (injected current per electrode)

with the gauge condition ``sum_i U_i = 0``.  The saddle-point system solved
here is

    | A(sigma) + B     C     0 | | u |   | 0 |
    | C^T              D     1 | | U | = | I |
    | 0                1^T   0 | | m |   | 0 |

where ``m`` is the Lagrange multiplier enforcing the gauge, and

    A_kl = int_Omega sigma grad(phi_k) . grad(phi_l) dx
    B_kl = sum_i 1/z_i int_{e_i} phi_k phi_l ds
    C_ki = -1/z_i int_{e_i} phi_k ds
    D_ii = |e_i| / z_i

Sign convention of the Jacobian
-------------------------------
``calc_jacobian`` returns

    J[(h, j), c] = + int_{cell c} grad(u^(h)) . grad(w^(j)) dx
                 = - d U_j^(h) / d sigma_c

i.e. **the negative** of the derivative of the electrode voltages with respect
to the cell conductivities.  This is the convention the Gauss-Newton solvers in
``gauss_newton.py`` expect: with residual ``r = U_sim - U_meas`` the normal
equations ``(J^T J) d = J^T r`` already produce a *descent* step, so the update
is ``sigma <- sigma + d``.  The convention is verified against finite
differences in ``tests/test_forward.py``.
"""

import warnings

import numpy as np

from mpi4py import MPI
import ufl
from dolfinx.io import gmshio
from dolfinx.fem import (
    Function,
    functionspace,
    assemble_scalar,
    form,
    Expression,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

from .utils import validate_injection


def _interpolation_points(element):
    """``interpolation_points`` is a method in dolfinx <=0.8 and a property in >=0.9."""
    ip = element.interpolation_points
    return ip() if callable(ip) else ip


class _ScipyLU:
    """Thin wrapper so the Scipy and PETSc backends share one interface."""

    def __init__(self, matrix):
        self._solve = factorized(matrix.tocsc())

    def solve(self, rhs):
        return self._solve(rhs)

    def destroy(self):
        self._solve = None


class _PetscLU:
    def __init__(self, matrix, comm, n):
        self._mat = PETSc.Mat().createAIJ(
            size=matrix.shape,
            csr=(matrix.indptr, matrix.indices, matrix.data),
        )
        # The last diagonal entry of the saddle-point system is structurally
        # zero; PETSc needs it to exist explicitly for the LU factorisation.
        self._mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self._mat.setValues([n - 1], [n - 1], [[0.0]], PETSc.InsertMode.INSERT_VALUES)
        self._mat.assemblyBegin()
        self._mat.assemblyEnd()
        self._mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

        self._ksp = PETSc.KSP().create(comm)
        self._ksp.setOperators(self._mat)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        self._ksp.getPC().setType(PETSc.PC.Type.LU)

        self._b = PETSc.Vec().create(PETSc.COMM_WORLD)
        self._b.setSizes(n)
        self._b.setUp()
        self._x = self._b.duplicate()

    def solve(self, rhs):
        self._b.setArray(rhs)
        self._b.assemblyBegin()
        self._b.assemblyEnd()
        self._ksp.solve(self._b, self._x)
        return np.array(self._x.getArray(), copy=True)

    def destroy(self):
        for obj in (self._ksp, self._mat, self._b, self._x):
            try:
                obj.destroy()
            except Exception:
                pass


class EIT:
    def __init__(
        self,
        L,
        Inj,
        z,
        backend="Scipy",
        mesh_name="EIT_disk.msh",
        check_injection=True,
    ):
        """
        L:        number of electrodes
        Inj:      current injection pattern, ``(N, L)`` with N patterns
        z:        contact impedances, one per electrode (length L)
        backend:  "Scipy" (SuperLU) or "PETSc" (PETSc LU)
        mesh_name: gmsh ``.msh`` file with physical groups 1..L for the
                  electrodes and 0 for the electrode-free boundary
        """
        assert backend in ["PETSc", "Scipy"], "backend has to be either PETSc or Scipy"

        Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
        assert (
            Inj.shape[-1] == L
        ), f"Injection pattern has {Inj.shape[-1]} columns but there are {L} electrodes"

        z = np.asarray(z, dtype=float)
        if z.ndim == 0:
            z = np.full(L, float(z))
        assert len(z) == L, "There has to be one contact impedance for every electrode"
        if np.any(z <= 0):
            raise ValueError("Contact impedances must be strictly positive")
        if np.any(z < 1e-5):
            warnings.warn(
                f"Very small contact impedance (min z = {z.min():.1e}). The CEM "
                "degenerates towards the shunt model and the system matrix "
                "becomes badly conditioned; z in [1e-4, 1e-1] is typical.",
                stacklevel=2,
            )

        if check_injection:
            ok, msg = validate_injection(Inj)
            if not ok:
                warnings.warn(f"Injection pattern problem: {msg}", stacklevel=2)

        self.backend = backend
        self.L = L
        self.Inj = Inj
        self.z = z
        self.mesh_name = mesh_name

        self.omega, _, facet_markers = gmshio.read_from_msh(
            mesh_name, MPI.COMM_WORLD, gdim=2
        )
        self.omega.topology.create_connectivity(1, 2)

        ## Boundary measure
        self.ds_electrodes = ufl.Measure(
            "ds", domain=self.omega, subdomain_data=facet_markers
        )

        ## Length (2D: arc length) of every electrode, measured individually so
        ## that non-uniform electrode layouts and non-circular domains are
        ## handled correctly.
        self.electrode_lengths = np.array(
            [
                assemble_scalar(form(1 * self.ds_electrodes(i + 1)))
                for i in range(self.L)
            ]
        )
        if np.any(self.electrode_lengths <= 0):
            missing = np.flatnonzero(self.electrode_lengths <= 0) + 1
            raise ValueError(
                f"Electrodes {missing.tolist()} have zero length in '{mesh_name}'. "
                "Check that the mesh defines physical groups 1..L on the boundary."
            )
        # kept for backwards compatibility with older scripts
        self.electrode_len = float(self.electrode_lengths[0])

        ### Create function space and helper functions
        self.V = functionspace(self.omega, ("Lagrange", 1))
        self.V_sigma = functionspace(self.omega, ("DG", 0))

        u_sol = Function(self.V)
        self.dofs = len(u_sol.x.array)
        self.n_cells = self.omega.topology.index_map(2).size_local
        self.n_total = self.dofs + self.L + 1

        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)

        self.M = self.assemble_lhs()

        # cached LU factorisation of the last conductivity that was solved for
        self._cached_sigma = None
        self._cached_lu = None
        self.M_complete = None

        # cell areas, needed by the Jacobian
        v_dg = ufl.TestFunction(self.V_sigma)
        self.cell_area = np.array(assemble_vector(form(v_dg * ufl.dx)).array, copy=True)

    # ------------------------------------------------------------------ #
    #  assembly
    # ------------------------------------------------------------------ #
    def assemble_lhs(self):
        """Assemble the sigma-independent blocks B, C, D and the gauge row."""
        b = 0
        for i in range(0, self.L):
            b += 1 / self.z[i] * ufl.inner(self.u, self.phi) * self.ds_electrodes(i + 1)

        B = assemble_matrix(form(b))
        B.assemble()

        bi, bj, bv = B.getValuesCSR()
        M = csr_matrix((bv, bj, bi), shape=(self.dofs, self.dofs))
        M.resize(
            self.dofs + self.L + 1, self.dofs + self.L + 1
        )  # extra row/col for the zero-average condition
        M_lil = lil_matrix(M)  # faster to modify (then change back)

        B.destroy()  # dont need B anymore

        for i in range(0, self.L):
            # Build C matrix (top right and bottom left block)
            # Has to be done for each row
            c = -1 / self.z[i] * self.phi * self.ds_electrodes(i + 1)
            C_i = assemble_vector(form(c)).array
            M_lil[self.dofs + i, : self.dofs] = C_i
            M_lil[: self.dofs, self.dofs + i] = C_i
            # bottom right block matrix (diagonal and average condition)
            M_lil[self.dofs + i, self.dofs + i] = (
                self.electrode_lengths[i] / self.z[i]
            )
            M_lil[self.dofs + self.L, self.dofs + i] = 1
            M_lil[self.dofs + i, self.dofs + self.L] = 1

        return csr_matrix(M_lil)

    def create_full_matrix(self, sigma):
        a = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        A = assemble_matrix(form(a))
        A.assemble()

        ai, aj, av = A.getValuesCSR()
        scipy_A = csr_matrix((av, aj, ai), shape=(self.dofs, self.dofs))
        A.destroy()
        scipy_A.resize(self.dofs + self.L + 1, self.dofs + self.L + 1)

        return scipy_A + self.M

    # ------------------------------------------------------------------ #
    #  linear solves
    # ------------------------------------------------------------------ #
    def _get_lu(self, sigma):
        """
        LU factorisation of the saddle-point matrix for ``sigma``, cached.

        The forward solve, the line search, the adjoint solve and the Jacobian
        all use the *same* matrix, so re-factorising every time was by far the
        dominant cost.  The cache is keyed on the conductivity values.
        """
        sig = np.asarray(sigma.x.array, dtype=float)
        if self._cached_sigma is not None and np.array_equal(sig, self._cached_sigma):
            return self._cached_lu

        if self._cached_lu is not None:
            self._cached_lu.destroy()

        self.M_complete = self.create_full_matrix(sigma)
        if self.backend == "PETSc":
            lu = _PetscLU(self.M_complete, self.omega.comm, self.n_total)
        else:
            lu = _ScipyLU(self.M_complete)

        self._cached_sigma = sig.copy()
        self._cached_lu = lu
        return lu

    def _solve_patterns(self, lu, patterns):
        """Solve the saddle-point system for every row of ``patterns`` (N, L)."""
        u_all, U_all = [], []
        rhs = np.zeros(self.n_total)
        for inj in patterns:
            rhs[:] = 0.0
            rhs[self.dofs : self.dofs + self.L] = inj
            sol = lu.solve(rhs)
            u_all.append(sol[: self.dofs].copy())
            U_all.append(sol[self.dofs : -1].copy())
        return u_all, U_all

    def forward_solve(self, sigma, Inj=None):
        """
        sigma: dolfinx Function (DG0 or CG1)
        Inj:   optional ``(N, L)`` injection matrix, defaults to ``self.Inj``

        Returns ``(u_all, U_all)``: interior potentials and electrode voltages,
        one entry per current pattern.
        """
        if Inj is None:
            Inj = self.Inj
        Inj = np.atleast_2d(np.asarray(Inj, dtype=float))

        lu = self._get_lu(sigma)
        return self._solve_patterns(lu, Inj)

    def solve_adjoint(self, deltaU, sigma=None):
        """
        deltaU: ``(N, L)`` residual on the electrodes
        sigma:  dolfinx Function; if None the conductivity of the last forward
                solve is reused.

        The CEM system matrix is symmetric, so the adjoint problem uses the very
        same factorisation with ``deltaU`` in place of the injected currents.
        """
        deltaU = np.atleast_2d(np.asarray(deltaU, dtype=float))

        if sigma is None:
            if self._cached_lu is None:
                raise RuntimeError(
                    "solve_adjoint(sigma=None) requires a preceding forward_solve"
                )
            lu = self._cached_lu
        else:
            lu = self._get_lu(sigma)

        p_all, _ = self._solve_patterns(lu, deltaU)
        return p_all

    # ------------------------------------------------------------------ #
    #  Jacobian
    # ------------------------------------------------------------------ #
    def _cellwise_gradients(self, u_arrays):
        """Project grad(u) of each CG1 coefficient vector onto DG0 vectors."""
        Q_DG = functionspace(self.omega, ("DG", 0, (2,)))
        points = _interpolation_points(Q_DG.element)

        u_fun = Function(self.V)
        grad_u = Function(Q_DG)

        out = []
        for u in u_arrays:
            u_fun.x.array[:] = u
            expr = Expression(ufl.as_vector((u_fun.dx(0), u_fun.dx(1))), points)
            grad_u.interpolate(expr)
            out.append(np.array(grad_u.x.array, copy=True).reshape(-1, 2))
        return out

    def calc_jacobian(self, sigma, u_all=None):
        """
        Adjoint (reciprocity) Jacobian of the electrode voltages with respect to
        the cell-wise conductivity.

        Returns an array of shape ``(N * L, n_cells)`` whose rows are ordered
        like ``np.asarray(U_all).flatten()`` (pattern-major, electrode-minor):

            J[h*L + j, c] = int_{cell c} grad(u^(h)) . grad(w^(j)) dx
                          = - dU_j^(h) / dsigma_c

        ``w^(j)`` is the potential produced by the charge-conserving unit
        pattern ``e_j - 1/L``, which is the correct measurement field for the
        gauge ``sum_i U_i = 0`` used by this solver.

        Ref: https://fabiomargotti.paginas.ufsc.br/files/2017/12/Margotti_Fabio-3.pdf chap 5.2.1
        """
        if u_all is None:
            u_all, _ = self.forward_solve(sigma)

        # Measurement fields: unit current out of electrode j, returned evenly
        # through all electrodes so that sum_i I_i = 0 holds exactly.
        I2_all = np.eye(self.L) - 1.0 / self.L
        bu_all, _ = self.forward_solve(sigma, I2_all)

        list_grad_u = self._cellwise_gradients(u_all)
        list_grad_bu = self._cellwise_gradients(bu_all)

        # stack: (L, n_cells, 2) x (N, n_cells, 2) -> (N*L, n_cells)
        G_meas = np.stack(list_grad_bu, axis=0)  # (L, n_cells, 2)
        blocks = []
        for grad_u in list_grad_u:  # per current pattern
            blocks.append(np.einsum("ecd,cd->ec", G_meas, grad_u) * self.cell_area)

        return np.concatenate(blocks, axis=0)

    # ------------------------------------------------------------------ #
    #  geometry helpers
    # ------------------------------------------------------------------ #
    def cell_centers(self):
        """``(n_cells, 2)`` centroids, ordered like the DG0 dofs."""
        return np.array(self.V_sigma.tabulate_dof_coordinates()[:, :2])

    def domain_area(self):
        return float(assemble_scalar(form(1 * ufl.dx(domain=self.omega))))

    def triangulation(self):
        """matplotlib ``Triangulation`` of the mesh (for ``tripcolor``)."""
        from matplotlib.tri import Triangulation

        xy = self.omega.geometry.x
        cells = np.asarray(self.omega.geometry.dofmap).reshape(
            (-1, self.omega.topology.dim + 1)
        )
        return Triangulation(xy[:, 0], xy[:, 1], cells)

    def __repr__(self):
        return (
            f"EIT(L={self.L}, patterns={self.Inj.shape[0]}, cells={self.n_cells}, "
            f"nodes={self.dofs}, mesh='{self.mesh_name}')"
        )
