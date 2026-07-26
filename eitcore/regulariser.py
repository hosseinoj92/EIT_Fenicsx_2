# regulariser.py

"""
Prior / regularisation operators on the DG0 (cell-wise) conductivity space.

Three families are available:

``gaussian``   Squared-exponential covariance
                   Gamma_ij = std^2 exp(-|x_i - x_j|^2 / (2 c^2))
               and its factor ``L`` with ``L^T L = Gamma^-1``.  Statistically
               the most meaningful smoothness prior, but **dense**: memory
               grows as ``n_cells^2`` and the factorisation as ``n_cells^3``.

``laplacian``  Sparse graph Laplacian over mesh facets,
                   R = L_grad^T L_grad + eps I
               This penalises conductivity jumps between neighbouring cells.
               Same qualitative effect as the Gaussian prior at a tiny fraction
               of the cost, and the only practical option on fine meshes.

``tikhonov``   Identity, i.e. plain zeroth-order Tikhonov.

All builders return ``R`` such that the penalty is
``0.5 * lamb * (sigma - sigma_prior)^T R (sigma - sigma_prior)``.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, solve_triangular
from scipy.sparse import csr_array, eye as sparse_eye

from dolfinx.fem import Function, functionspace


# --------------------------------------------------------------------------- #
#  dense squared-exponential prior
# --------------------------------------------------------------------------- #
def build_smoothness_regulariser(omega, corrlength: float = 1.0, std: float = 0.3,
                                 jitter: float = 1e-7):
    """
    Gaussian smoothness prior with covariance

        Gamma = std^2 exp(-|| xi - xj ||^2 / (2 * corrlength^2))

    From: https://zenodo.org/record/8252370

    Returns ``L`` with ``L.T @ L = Gamma^(-1)``.

    The pairwise distances are formed with a vectorised Gram-matrix identity
    instead of the previous Python double loop, and ``Gamma^-1`` is obtained
    from the Cholesky factor of ``Gamma`` by a triangular solve rather than by
    an explicit ``inv`` followed by a second Cholesky (which loses roughly half
    the significant digits on an ill-conditioned covariance).
    """
    V = functionspace(omega, ("DG", 0))
    g = np.array(V.tabulate_dof_coordinates()[:, :2], dtype=float)
    ng = g.shape[0]

    var = std**2
    a = var - jitter

    # squared distances via ||x||^2 + ||y||^2 - 2 x.y
    sq = np.einsum("ij,ij->i", g, g)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (g @ g.T)
    np.maximum(d2, 0.0, out=d2)

    Gamma = a * np.exp(-d2 / (2.0 * corrlength**2))
    Gamma[np.diag_indices(ng)] += jitter

    # Gamma = C^T C  (upper) -> Gamma^-1 = C^-1 C^-T -> L = C^-T ... but we want
    # L^T L = Gamma^-1, which is satisfied by L = C^-T applied as below.
    C, lower = cho_factor(Gamma, lower=False, check_finite=False)
    Linv = solve_triangular(C, np.eye(ng), lower=False, check_finite=False)
    L = Linv.T  # L^T L = Linv Linv^T = (C^T C)^-1 = Gamma^-1

    return L


def create_smoothness_regulariser(omega, save_name: str, corrlength: float = 1.0,
                                  std: float = 0.3):
    """Build the Gaussian prior factor and cache it in ``save_name`` (.npy)."""
    L = build_smoothness_regulariser(omega, corrlength=corrlength, std=std)
    np.save(save_name, L)
    return L


# --------------------------------------------------------------------------- #
#  sparse graph-Laplacian prior
# --------------------------------------------------------------------------- #
def build_gradient_matrix(omega):
    """
    Sparse facet-difference operator ``(n_interior_facets, n_cells)``: one row
    per interior facet with ``+1`` / ``-1`` in the two adjacent cells.
    """
    topo = omega.topology
    topo.create_connectivity(1, 2)
    edge_to_cell = topo.connectivity(1, 2)

    num_facets = topo.index_map(1).size_local
    num_cells = topo.index_map(2).size_local

    rows, cols, data = [], [], []
    row = 0
    for edge in range(num_facets):
        cells = edge_to_cell.links(edge)
        if len(cells) > 1:
            rows.extend((row, row))
            cols.extend((cells[0], cells[1]))
            data.extend((1.0, -1.0))
            row += 1

    return csr_array((data, (rows, cols)), shape=(row, num_cells))


def build_laplacian_regulariser(omega, eps: float = 1e-4):
    """
    Sparse smoothness prior ``R = D^T D + eps I`` with ``D`` the facet-difference
    operator.  ``eps`` makes ``R`` positive definite (``D^T D`` has the constant
    vector in its null space, so without it the prior does not constrain the
    conductivity level at all).
    """
    D = build_gradient_matrix(omega)
    return (D.T @ D + eps * sparse_eye(D.shape[1], format="csr")).tocsr()


def build_tikhonov_regulariser(omega):
    n = omega.topology.index_map(2).size_local
    return sparse_eye(n, format="csr")


# --------------------------------------------------------------------------- #
#  dispatch
# --------------------------------------------------------------------------- #
def build_prior(omega, kind="laplacian", **kwargs):
    """
    Return the prior matrix ``R`` (dense ndarray or scipy sparse).

    kind:
        ``"laplacian"`` - kwargs: ``eps``
        ``"tikhonov"``  - no kwargs
        ``"gaussian"``  - kwargs: ``corrlength``, ``std``, optional ``cache_file``
    """
    kind = kind.lower()
    if kind == "tikhonov":
        return build_tikhonov_regulariser(omega)

    if kind == "laplacian":
        return build_laplacian_regulariser(omega, eps=kwargs.get("eps", 1e-4))

    if kind == "gaussian":
        cache = kwargs.get("cache_file")
        n = omega.topology.index_map(2).size_local
        L = None
        if cache is not None:
            try:
                L = np.load(cache)
                if L.shape != (n, n):
                    print(
                        f"Cached prior '{cache}' has shape {L.shape} but the mesh "
                        f"has {n} cells - rebuilding."
                    )
                    L = None
            except FileNotFoundError:
                L = None
        if L is None:
            if n > 8000:
                print(
                    f"Building a dense {n}x{n} Gaussian prior "
                    f"({n * n * 8 / 1e9:.1f} GB). Consider kind='laplacian'."
                )
            L = build_smoothness_regulariser(
                omega,
                corrlength=kwargs.get("corrlength", 1.0),
                std=kwargs.get("std", 0.3),
            )
            if cache is not None:
                np.save(cache, L)
        return L.T @ L

    raise ValueError(
        f"Unknown prior kind '{kind}'. Choices: gaussian, laplacian, tikhonov"
    )


# --------------------------------------------------------------------------- #
def plot_samples_from_prior(L, triangulation, n_samples=4):
    """
    L: Cholesky decomposition of precision matrix, L.T @ L = Gamma^(-1)
    triangulation: numpy triangulation of the underlying mesh
    """
    samples = np.linalg.solve(L, np.random.randn(L.shape[0], n_samples))
    fig, axes = plt.subplots(1, n_samples, figsize=(3.5 * n_samples, 6))

    for i in range(n_samples):
        im = axes[i].tripcolor(
            triangulation, samples[:, i].flatten(), cmap="jet", shading="flat"
        )
        axes[i].axis("image")
        axes[i].set_aspect("equal", adjustable="box")
        axes[i].set_title("Sample " + str(i))
        fig.colorbar(im, ax=axes[i])

    return fig
