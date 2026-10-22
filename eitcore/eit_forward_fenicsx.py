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

For AC, sigma is replaced by gamma = sigma + j*2*pi*f*epsilon0*epsilon_r
and 1/z by the specific parallel-RC contact admittance. Integrals are real
FEniCSx assemblies combined into a complex SciPy matrix, including on real
PETSc builds. In 2D all physical integrals are multiplied by the extrusion
thickness, so injected currents retain units A and z retains units ohm m^2.

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
try:
    from dolfinx.io import gmshio
except ImportError:  # dolfinx >= 0.10
    from dolfinx.io import gmsh as gmshio
from dolfinx.fem import (
    Function,
    functionspace,
    assemble_scalar,
    form,
    Expression,
)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc

from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
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
        dimension=2,
        frequency_hz=0.0,
        relative_permittivity=0.0,
        thickness=1.0,
        contact_capacitance=0.0,
    ):
        """
        L:        number of electrodes
        Inj:      current injection pattern, ``(N, L)`` with N patterns
        z:        specific contact resistance [ohm m^2], scalar or length L
        backend:  "Scipy" (SuperLU) or "PETSc" (PETSc LU)
        mesh_name: gmsh ``.msh`` file with physical groups 1..L for the
                  electrodes and 0 for the electrode-free boundary
        dimension: 2 (triangle disk) or 3 (tetrahedral volume)
        frequency_hz: excitation frequency [Hz], 0 gives the DC limit
        relative_permittivity: dimensionless scalar or values in DG0 dof order
        thickness: uniform 2D extrusion thickness [m], ignored in 3D
        contact_capacitance: specific parallel capacitance [F/m^2], scalar or L

        Potentials and injected currents are RMS phasors with exp(+j omega t).
        Construct a new solver for each frequency/contact configuration. The
        relative_permittivity Function can be updated between material solves.
        """
        assert backend in ["PETSc", "Scipy"], "backend has to be either PETSc or Scipy"
        if MPI.COMM_WORLD.size != 1:
            raise ValueError("The assembled SciPy CEM currently requires a single MPI rank")
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if not np.isfinite(frequency_hz) or frequency_hz < 0:
            raise ValueError("frequency_hz must be finite and nonnegative [Hz]")
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("2D extrusion thickness must be positive [m]")
        self.dimension = dimension
        self.frequency_hz = float(frequency_hz)
        self.measure_scale = float(thickness) if dimension == 2 else 1.0
        self.contact_capacitance = np.broadcast_to(np.asarray(contact_capacitance, dtype=float), (L,)).copy()
        if np.any(~np.isfinite(self.contact_capacitance)) or np.any(self.contact_capacitance < 0):
            raise ValueError("contact_capacitance must be nonnegative [F/m^2]")

        Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
        assert (
            Inj.shape[-1] == L
        ), f"Injection pattern has {Inj.shape[-1]} columns but there are {L} electrodes"

        z = np.asarray(z, dtype=float)
        if z.ndim == 0:
            z = np.full(L, float(z))
        assert len(z) == L, "There has to be one contact impedance for every electrode"
        if np.any(~np.isfinite(z)) or np.any(z <= 0):
            raise ValueError("Contact impedances must be strictly positive")

        if check_injection:
            ok, msg = validate_injection(Inj)
            if not ok:
                warnings.warn(f"Injection pattern problem: {msg}", stacklevel=2)

        self.backend = backend
        self.L = L
        self.Inj = Inj
        self.z = z
        self.mesh_name = mesh_name

        mesh_data = gmshio.read_from_msh(mesh_name, MPI.COMM_WORLD, gdim=dimension)
        if hasattr(mesh_data, "mesh"):
            self.omega, facet_markers = mesh_data.mesh, mesh_data.facet_tags
        else:
            self.omega, _, facet_markers = mesh_data
        if self.omega.topology.dim != dimension:
            raise ValueError("Mesh topology does not match requested dimension")
        self.omega.topology.create_connectivity(dimension - 1, dimension)

        ## Boundary measure
        self.ds_electrodes = ufl.Measure(
            "ds", domain=self.omega, subdomain_data=facet_markers
        )

        ## Length (2D: arc length) of every electrode, measured individually so
        ## that non-uniform electrode layouts and non-circular domains are
        ## handled correctly.
        self._facet_markers = facet_markers
        self._contact_geometry = None
        self.electrode_lengths = (self._electrode_measures_geometry() if L >= 64 else np.array(
            [
                assemble_scalar(form(1 * self.ds_electrodes(i + 1)))
                for i in range(self.L)
            ]
        ))
        if np.any(self.electrode_lengths <= 0):
            missing = np.flatnonzero(self.electrode_lengths <= 0) + 1
            raise ValueError(
                f"Electrodes {missing.tolist()} have zero length in '{mesh_name}'. "
                "Check that the mesh defines physical groups 1..L on the boundary."
            )
        # kept for backwards compatibility with older scripts
        self.electrode_len = float(np.real(self.electrode_lengths[0]))
        self.electrode_areas = self.electrode_lengths.real * self.measure_scale

        ### Create function space and helper functions
        self.V = functionspace(self.omega, ("Lagrange", 1))
        self.V_sigma = functionspace(self.omega, ("DG", 0))

        u_sol = Function(self.V)
        self.dofs = len(u_sol.x.array)
        self.n_cells = self.omega.topology.index_map(dimension).size_local
        self.n_total = self.dofs + self.L + 1

        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)

        self.relative_permittivity = Function(self.V_sigma)
        self.relative_permittivity.x.array[:] = relative_permittivity
        if (np.any(~np.isfinite(self.relative_permittivity.x.array)) or
                np.any(self.relative_permittivity.x.array.real < 0)):
            raise ValueError("relative_permittivity must be finite and nonnegative")
        self.is_complex = frequency_hz > 0 and (np.any(self.relative_permittivity.x.array) or
                                               np.any(self.contact_capacitance))
        if self.is_complex and backend == "PETSc" and np.dtype(PETSc.ScalarType).kind != "c":
            raise ValueError("Use backend='Scipy' for AC with a real PETSc build")

        self.M = self.assemble_lhs()

        # cached LU factorisation of the last conductivity that was solved for
        self._cached_sigma = None
        self._cached_permittivity = None
        self._cached_lu = None
        self.M_complete = None

        # cell areas, needed by the Jacobian
        v_dg = ufl.TestFunction(self.V_sigma)
        self.cell_area = np.array(assemble_vector(form(v_dg * ufl.dx)).array, copy=True).real * self.measure_scale

    # ------------------------------------------------------------------ #
    #  assembly
    # ------------------------------------------------------------------ #
    def assemble_lhs(self):
        """Assemble the sigma-independent blocks B, C, D and the gauge row."""
        if self.L >= 64:
            return self._assemble_contact_geometry()
        M = self._assemble_contact(1 / self.z, gauge=True)
        if self.frequency_hz > 0 and np.any(self.contact_capacitance):
            M = M + 1j * self._assemble_contact(
                2 * np.pi * self.frequency_hz * self.contact_capacitance, gauge=False)
        return M

    def _electrode_measures_geometry(self):
        """Exact linear-simplex facet quadrature without one JIT form per contact."""
        from dolfinx.mesh import entities_to_geometry
        tags = self._facet_markers
        keep = (tags.values >= 1) & (tags.values <= self.L)
        facets = tags.indices[keep]
        owners = tags.values[keep]-1
        vertices = entities_to_geometry(self.omega, self.dimension-1, facets, False)
        xyz = self.omega.geometry.x[vertices]
        if self.dimension == 3:
            measures = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)/2
        else:
            measures = np.linalg.norm(xyz[:, 1]-xyz[:, 0], axis=1)
        self._contact_geometry = (vertices, owners, measures)
        return np.bincount(owners, weights=measures, minlength=self.L)

    def _assemble_contact_geometry(self):
        from scipy.spatial import cKDTree
        vertices, owners, measures = self._contact_geometry
        distances, map_geometry = cKDTree(self.V.tabulate_dof_coordinates()).query(self.omega.geometry.x)
        if np.max(distances) > 1e-10:
            raise ValueError("Geometric CEM assembly requires linear Lagrange elements")
        dofs = map_geometry[vertices]
        nv = dofs.shape[1]
        y = 1/self.z
        if self.frequency_hz and np.any(self.contact_capacitance):
            y = y+1j*2*np.pi*self.frequency_hz*self.contact_capacitance
        weights = measures*self.measure_scale*y[owners]
        rows, cols, data = [], [], []
        for a in range(nv):
            for b in range(nv):
                rows.append(dofs[:, a]); cols.append(dofs[:, b])
                data.append(weights*(2 if a == b else 1)/(nv*(nv+1)))
            rows.extend([dofs[:, a], self.dofs+owners])
            cols.extend([self.dofs+owners, dofs[:, a]])
            data.extend([-weights/nv, -weights/nv])
        electrode = self.dofs+np.arange(self.L)
        rows.extend([electrode, electrode, np.full(self.L, self.n_total-1)])
        cols.extend([electrode, np.full(self.L, self.n_total-1), electrode])
        data.extend([self.electrode_areas*y, np.ones(self.L), np.ones(self.L)])
        return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_total, self.n_total)).tocsr()

    def cellwise_gradients_fast(self, u_arrays):
        """Exact P1 simplex gradients in DG0 order, avoiding per-pattern UFL JIT."""
        if not hasattr(self, "_simplex_gradient_data"):
            dim = self.dimension
            cell_dofs = np.array([self.V.dofmap.cell_dofs(c) for c in range(self.n_cells)])
            xyz = self.V.tabulate_dof_coordinates()[cell_dofs, :dim]
            inverse = np.linalg.inv(xyz[:, 1:]-xyz[:, :1])
            dg = np.array([self.V_sigma.dofmap.cell_dofs(c)[0] for c in range(self.n_cells)])
            self._simplex_gradient_data = (cell_dofs, inverse, dg)
        dofs, inverse, dg = self._simplex_gradient_data
        result = []
        for u in u_arrays:
            coeff = np.asarray(u)[dofs]
            gradient = np.einsum('cij,cj->ci', inverse, coeff[:, 1:]-coeff[:, :1])
            out = np.empty_like(gradient)
            out[dg] = gradient
            result.append(out)
        return result

    def _assemble_contact(self, weights, gauge):
        b = 0
        for i in range(0, self.L):
            b += weights[i] * self.measure_scale * ufl.inner(self.u, self.phi) * self.ds_electrodes(i + 1)

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
            c = -weights[i] * self.measure_scale * ufl.conj(self.phi) * self.ds_electrodes(i + 1)
            C_i = assemble_vector(form(c)).array
            M_lil[self.dofs + i, : self.dofs] = C_i
            M_lil[: self.dofs, self.dofs + i] = C_i
            # bottom right block matrix (diagonal and average condition)
            M_lil[self.dofs + i, self.dofs + i] = (
                self.electrode_areas[i] * weights[i]
            )
            M_lil[self.dofs + self.L, self.dofs + i] = int(gauge)
            M_lil[self.dofs + i, self.dofs + self.L] = int(gauge)

        return csr_matrix(M_lil)

    def create_full_matrix(self, sigma):
        M = self._stiffness(sigma) + self.M
        if self.frequency_hz > 0 and np.any(self.relative_permittivity.x.array):
            M = M + 1j * (2 * np.pi * self.frequency_hz * 8.8541878128e-12) * self._stiffness(self.relative_permittivity)
        return M

    def _stiffness(self, sigma):
        a = self.measure_scale * ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        A = assemble_matrix(form(a))
        A.assemble()

        ai, aj, av = A.getValuesCSR()
        scipy_A = csr_matrix((av, aj, ai), shape=(self.dofs, self.dofs))
        A.destroy()
        scipy_A.resize(self.dofs + self.L + 1, self.dofs + self.L + 1)

        return scipy_A

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
        eps = np.asarray(self.relative_permittivity.x.array).copy()
        if (self._cached_sigma is not None and np.array_equal(sig, self._cached_sigma)
                and np.array_equal(eps, self._cached_permittivity)):
            return self._cached_lu

        if self._cached_lu is not None:
            self._cached_lu.destroy()

        self.M_complete = self.create_full_matrix(sigma)
        if self.backend == "PETSc":
            lu = _PetscLU(self.M_complete, self.omega.comm, self.n_total)
        else:
            lu = _ScipyLU(self.M_complete)

        self._cached_sigma = sig.copy()
        self._cached_permittivity = eps
        self._cached_lu = lu
        return lu

    def _solve_patterns(self, lu, patterns):
        """Solve the saddle-point system for every row of ``patterns`` (N, L)."""
        u_all, U_all = [], []
        rhs = np.zeros(self.n_total, dtype=self.M_complete.dtype)
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
        deltaU = np.atleast_2d(np.asarray(deltaU))

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
        dim = self.dimension
        Q_DG = functionspace(self.omega, ("DG", 0, (dim,)))
        points = _interpolation_points(Q_DG.element)

        u_fun = Function(self.V)
        grad_u = Function(Q_DG)

        out = []
        for u in u_arrays:
            parts = []
            for values in (np.real(u), np.imag(u)):
                u_fun.x.array[:] = values
                expr = Expression(ufl.grad(u_fun), points)
                grad_u.interpolate(expr)
                parts.append(np.array(grad_u.x.array, copy=True).real.reshape(-1, dim))
            out.append(parts[0] + 1j*parts[1] if np.iscomplexobj(u) else parts[0])
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
        return np.array(self.V_sigma.tabulate_dof_coordinates()[:, :self.dimension])

    def domain_area(self):
        """Geometric area (2D) or volume (3D); legacy method name retained."""
        return float(assemble_scalar(form(1 * ufl.dx(domain=self.omega))))

    def triangulation(self):
        """matplotlib ``Triangulation`` of the mesh (for ``tripcolor``)."""
        if self.dimension != 2:
            raise ValueError("triangulation is 2D only; use volume slices for 3D")
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
