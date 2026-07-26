# dbar.py

"""
The D-bar method: Nachman's *direct* reconstruction algorithm for 2-D EIT.

Unlike every other reconstructor in this package the D-bar method does not
iterate on the forward model.  It is a nonlinear Fourier transform: the
measurements are turned into a "scattering transform" ``t(k)`` living in a
complex frequency plane, that transform is low-pass filtered, and an inverse
nonlinear transform (the D-bar equation) turns it back into a conductivity.
Regularisation is the low-pass filter ``|k| <= R`` - there is no prior, no
lambda and no line search.

The chain
---------
::

    U_meas ---(1)---> Lambda_sigma - Lambda_1 ---(2)---> t(k) ---(3)---> mu(z,0)
                       (DN map difference)      (scattering)    (D-bar eq.)
                                                                     |
                                        sigma(z) = mu(z,0)^2  <------(4)

1.  **Boundary map.**  The measured (current, voltage) pairs are turned into a
    matrix approximation of the Neumann-to-Dirichlet map on the *unit disk*,
    which is inverted to give the Dirichlet-to-Neumann (DN) map.  See
    :func:`dn_matrix`.

2.  **Scattering transform.**  For ``k`` in the complex plane,

        t(k) = int_{dOmega} e^{i conj(k) conj(z)} (Lambda_sigma - Lambda_1) psi(.,k) ds

    with ``psi`` the complex geometrical optics (CGO) solution.  Two variants
    are implemented:

    ``"exp"``  the Born approximation ``psi ~ e^{ikz}`` of Siltanen et al.,
               i.e. ``t^exp``.  This is the workhorse of every published
               D-bar reconstruction from real data, and the variant for which
               Knudsen-Lassas-Mueller-Siltanen (2009) proved that truncating
               at ``|k| <= R`` is a *regularisation strategy*.
    ``"bie"``  the full ``t`` obtained by solving Nachman's boundary integral
               equation for the trace of ``psi`` with the exact Faddeev
               Green's function.  More accurate at high contrast, slower.

3.  **D-bar equation.**  For every reconstruction point ``z`` solve

        d mu(z,k) / d conj(k) = t(k) / (4 pi conj(k)) * e_{-k}(z) * conj(mu(z,k))

    with ``mu -> 1`` as ``|k| -> infinity`` and ``e_k(z) = e^{i(kz + conj(k)conj(z))}``.
    Equivalently, the integral equation

        mu(z,.) = 1 + (1/(pi k)) * [ T_z conj(mu) ],   T_z(k) = t(k) e_{-k}(z) / (4 pi conj(k))

    solved by Vainikko's periodised FFT scheme with GMRES.

4.  **Reconstruction.**  ``sigma(z) = mu(z,0)^2``.

Scaling to the unit disk
------------------------
The theory lives on the unit disk with ``sigma -> 1`` at the boundary.  A
measurement on a disk of radius ``r`` with background ``sigma_0`` is mapped
there by

    Lambda_{unit, sigma/sigma_0} = (r / sigma_0) * Lambda_{r, sigma}

(the DN map is homogeneous of degree 1 in a constant conductivity, and of
degree ``-1`` under a dilation of the domain), the scaled problem is solved,
and the result is multiplied by ``sigma_0`` again.

Electrode model
---------------
The measurements come from the Complete Electrode Model, the theory wants the
continuum DN map.  Two things bridge the gap and both matter:

* the contact-impedance shunt ``U_i = mean(u|e_i) + z_i I_i / |e_i|`` is
  subtracted from the ND matrix analytically (:func:`contact_shift`), and
* the homogeneous reference ``Lambda_1`` is *simulated with the same electrode
  model, mesh and contact impedances* rather than taken from the analytic
  continuum formula, so the remaining CEM modelling error cancels in the
  difference ``Lambda_sigma - Lambda_1`` that the scattering transform sees.

References
----------
* A I Nachman, "Global uniqueness for a two-dimensional inverse boundary value
  problem", Ann. of Math. 143 (1996) 71-96.
* S Siltanen, J Mueller, D Isaacson, "An implementation of the reconstruction
  algorithm of A Nachman for the 2D inverse conductivity problem", Inverse
  Problems 16 (2000) 681-699.
* K Knudsen, J Mueller, S Siltanen, "Numerical solution method for the
  dbar-equation in the plane", J. Comput. Phys. 198 (2004) 500-517.
* D Isaacson, J L Mueller, J C Newell, S Siltanen, "Reconstructions of chest
  phantoms by the D-bar method for EIT", IEEE Trans. Med. Imag. 23 (2004).
* K Knudsen, M Lassas, J L Mueller, S Siltanen, "Regularized D-bar method for
  the inverse conductivity problem", Inverse Probl. Imaging 3 (2009) 599-624.
* J L Mueller, S Siltanen, "The D-bar method for electrical impedance
  tomography - demystified", Inverse Problems 36 (2020) 093001.
"""

import warnings

import numpy as np
import ufl
from dolfinx.fem import Function, assemble_scalar, form
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import exp1, roots_legendre

from .reconstructor import Reconstructor

#: Euler-Mascheroni constant, needed for the log-singularity of the Faddeev
#: Green's function.
_EULER_GAMMA = 0.5772156649015328606


# --------------------------------------------------------------------------- #
#  boundary geometry
# --------------------------------------------------------------------------- #
def electrode_geometry(eit_solver):
    """
    Electrode angles and arc lengths of the mesh an :class:`EIT` solver holds.

    Returns ``(theta, arclen, radius)`` with ``theta`` the angular position of
    each electrode centre (radians, in the order of the physical groups
    ``1..L``), ``arclen`` its arc length in *physical* units and ``radius`` the
    radius of the circular cross-section.

    The centre is the arc-length centroid ``(1/|e_i|) int_{e_i} x ds`` projected
    back onto the circle, so non-uniform electrode layouts are handled exactly.
    """
    omega = eit_solver.omega
    x = ufl.SpatialCoordinate(omega)
    ds = eit_solver.ds_electrodes
    arclen = np.asarray(eit_solver.electrode_lengths, dtype=float)

    centres = np.empty((eit_solver.L, 2))
    for i in range(eit_solver.L):
        centres[i, 0] = assemble_scalar(form(x[0] * ds(i + 1)))
        centres[i, 1] = assemble_scalar(form(x[1] * ds(i + 1)))
    centres /= arclen[:, None]

    radius = float(np.max(np.linalg.norm(omega.geometry.x[:, :2], axis=1)))
    theta = np.arctan2(centres[:, 1], centres[:, 0])
    return theta, arclen, radius


def _arc_quadrature(theta, arc, order):
    """
    Gauss-Legendre nodes on each electrode arc of the **unit** circle.

    ``arc`` are the arc lengths on the unit circle (= angular widths).  Returns
    ``(z, w)`` of shape ``(L, order)`` with ``z`` the complex nodes on the unit
    circle and ``w`` the quadrature weights, normalised so that
    ``w[i].sum() == arc[i]``: integrating ``1`` over electrode ``i`` gives its
    length, as it must.
    """
    xq, wq = roots_legendre(int(order))
    half = arc[:, None] / 2.0
    phi = theta[:, None] + half * xq[None, :]
    return np.exp(1j * phi), half * wq[None, :]


# --------------------------------------------------------------------------- #
#  boundary maps:  measurements -> Neumann-to-Dirichlet -> Dirichlet-to-Neumann
# --------------------------------------------------------------------------- #
def orthonormalising_factor(Inj, arc, rcond=1e-10):
    """
    Factor ``B`` that turns the measured current patterns into an orthonormal
    basis of boundary functions.

    A current vector ``I`` (amperes per electrode) corresponds to the current
    *density* ``j = sum_i (I_i/a_i) chi_i`` on the boundary, ``chi_i`` the
    indicator of electrode ``i`` and ``a_i`` its length.  Two such densities
    have the ``L^2(dOmega)`` inner product ``sum_i a_i c_i c'_i`` with
    ``c = I/a``.  Writing ``G = Inj diag(1/a) Inj^T`` for the Gram matrix of the
    measured patterns, ``B = G^{-1/2}`` makes the rows of

        C = B Inj diag(1/a)

    an orthonormal basis ``phi_m = sum_i C[m,i] chi_i`` of the span of the
    measured densities.  Because ``B`` acts on the patterns and the data map is
    linear, the same ``B`` applied to the voltages gives the voltages of the
    orthonormalised patterns - no extra forward solves are needed.
    """
    Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
    arc = np.asarray(arc, dtype=float)

    G = Inj @ (Inj / arc[None, :]).T
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    if w.min() <= rcond * max(w.max(), 1e-300):
        raise ValueError(
            "The injection patterns are (numerically) rank deficient: the Gram "
            f"matrix has eigenvalues in [{w.min():.3e}, {w.max():.3e}]. The "
            "D-bar method needs a set of linearly independent patterns; use "
            "the trigonometric drive (method 4) with L-1 patterns."
        )
    return (V / np.sqrt(w)[None, :]) @ V.T


def contact_shift(Inj, B, z, arclen_phys):
    """
    The part of the ND matrix that is pure contact impedance.

    The CEM measures ``U_i = mean(u|e_i) + z_i I_i / |e_i|``; the second term is
    a property of the electrodes, not of the conductivity, and it is known
    exactly.  In the orthonormal basis it contributes

        B Inj diag(z/|e|) Inj^T B^T

    to the ND matrix, which for uniform ``z`` and equal electrodes is simply
    ``z/r`` times the identity.  Subtracting it moves the matrix towards the
    continuum ND map the D-bar theory is written for.
    """
    Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
    zz = np.asarray(z, dtype=float)
    if zz.ndim == 0:
        zz = np.full(Inj.shape[1], float(zz))
    scale = zz / np.asarray(arclen_phys, dtype=float)
    return B @ (Inj * scale[None, :]) @ Inj.T @ B.T


def nd_matrix(Inj, U, B):
    """
    Neumann-to-Dirichlet matrix in the orthonormal basis of :func:`orthonormalising_factor`.

    ``R[m,n] = <R_sigma phi_n, phi_m>_{L^2(dOmega)} = (B Inj U^T B^T)[m,n]``.

    The electrode lengths cancel out of this expression (they are hidden in
    ``B``), and the result does not depend on the voltage gauge because every
    row of ``Inj`` sums to zero.  ``Inj U^T`` is the transfer-impedance matrix
    and is symmetric by reciprocity, so the result is symmetrised.
    """
    Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
    U = np.atleast_2d(np.asarray(U, dtype=float))
    R = B @ (Inj @ U.T) @ B.T
    return 0.5 * (R + R.T)


def dn_matrix(Inj, U, arc_unit, radius, background=1.0, B=None,
              z=None, arclen_phys=None, rcond=None):
    """
    Dirichlet-to-Neumann matrix of the unit-disk problem with unit background.

    :param Inj:         ``(N, L)`` injected currents
    :param U:           ``(N, L)`` measured electrode voltages
    :param arc_unit:    electrode lengths **on the unit circle** (physical arc
                        length divided by ``radius``)
    :param radius:      radius of the physical cross-section
    :param background:  background conductivity ``sigma_0``
    :param B:           orthonormalising factor, built if not given
    :param z:           contact impedances; if given (together with
                        ``arclen_phys``) the shunt term is removed
    :param rcond:       if set, invert by a truncated SVD keeping singular
                        values above ``rcond * s_max`` instead of a plain solve

    The rescaling to the unit disk is folded in: ``arc_unit`` carries the
    factor ``1/radius`` and the ND matrix is multiplied by ``background``, which
    together implement ``Lambda_unit = (radius/sigma_0) Lambda_phys``.
    """
    if B is None:
        B = orthonormalising_factor(Inj, arc_unit)

    R = nd_matrix(Inj, U, B)
    if z is not None:
        if arclen_phys is None:
            raise ValueError("removing the contact shift needs arclen_phys")
        R = R - contact_shift(Inj, B, z, arclen_phys)
    R = float(background) * R

    if rcond is None:
        return np.linalg.inv(R)

    u, s, vt = np.linalg.svd(R)
    keep = s > rcond * s.max()
    return (vt[keep].T * (1.0 / s[keep])[None, :]) @ u[:, keep].T


# --------------------------------------------------------------------------- #
#  the scattering transform
# --------------------------------------------------------------------------- #
def _cgo_boundary_coefficients(k, C, zq, wq):
    """
    Expansion coefficients of the CGO exponentials in the orthonormal basis.

    Returns ``(c, d)`` of shape ``(len(k), N)`` with

        c_m(k) = int_{dOmega} e^{ikz}                phi_m(z) ds(z)
        d_m(k) = int_{dOmega} e^{i conj(k) conj(z)}  phi_m(z) ds(z)

    evaluated with the Gauss-Legendre rule on each electrode arc.  The basis
    functions are real, so no conjugation of ``phi_m`` is needed.
    """
    k = np.asarray(k, dtype=complex).ravel()
    # per-electrode integrals, shape (len(k), L)
    kz = k[:, None, None] * zq[None, :, :]
    ce = np.einsum("lq,klq->kl", wq, np.exp(1j * kz))
    # e^{i conj(k) conj(z)} = conj(e^{-ikz}) and the weights are real
    de = np.conj(np.einsum("lq,klq->kl", wq, np.exp(-1j * kz)))
    return ce @ C.T, de @ C.T


def scattering_transform_exp(k, dL, C, zq, wq):
    """
    Born ("exponential") scattering transform

        t^exp(k) = int_{dOmega} e^{i conj(k) conj(z)} (Lambda_sigma - Lambda_1) e^{ikz} ds

    obtained from Nachman's formula by replacing the unknown CGO trace
    ``psi(.,k)`` with its asymptotic value ``e^{ikz}``.  In the orthonormal
    basis this is the bilinear form ``d(k)^T dL c(k)``.
    """
    c, d = _cgo_boundary_coefficients(k, C, zq, wq)
    return np.einsum("kj,jm,km->k", d, dL, c)


def faddeev_green(k, z):
    """
    Faddeev's Green's function ``G_k(z) = e^{ikz} g_k(z)``, ``-Laplace G_k = delta``.

    ``g_k`` is the fundamental solution of ``(-Laplace - 4ik d/dconj(z)) g = delta``.
    It obeys the exact scaling ``g_k(z) = g_1(kz)`` with
    ``g_1(z) = e^{-iz} Re(E_1(-iz)) / (2 pi)``, so

        G_k(z) = Re(E_1(-i k z)) / (2 pi).

    Taking the real part is what makes this single valued: ``E_1`` jumps by
    ``-2 pi i`` across its branch cut, a purely imaginary discontinuity.  Near
    the origin ``G_k(z) = -log|z|/(2 pi) - (gamma + log|k|)/(2 pi) + O(|kz|)``,
    the free-space Laplace singularity, as it must be.
    """
    arg = -1j * np.asarray(k) * np.asarray(z)
    out = np.zeros(np.broadcast(np.asarray(k), np.asarray(z)).shape, dtype=float)
    good = arg != 0
    out[good] = np.real(exp1(arg[good])) / (2.0 * np.pi)
    out[~good] = np.inf
    return out


def _faddeev_self_term(k, arc):
    """
    ``int_{e_i} G_k(z_i - zeta) ds(zeta)`` over the electrode's own arc.

    The integrand has an integrable ``-log|s|/(2 pi)`` singularity at the
    electrode centre.  Splitting ``G_k(z) = -log|z|/(2 pi) + h_k(z)`` and
    integrating the singular half exactly over ``[-a/2, a/2]`` gives

        -(a/2) (log(a/2) - 1) / pi  -  a (gamma + log|k|) / (2 pi),

    which is much more accurate than the "set the diagonal to zero" rule that
    the first published implementations used.
    """
    half = np.asarray(arc, dtype=float) / 2.0
    singular = -(half * (np.log(half) - 1.0)) / np.pi
    smooth = -np.asarray(arc)[None, :] * (
        _EULER_GAMMA + np.log(np.abs(np.asarray(k))[:, None])
    ) / (2.0 * np.pi)
    return singular[None, :] + smooth


def scattering_transform_bie(k, dL, C, arc, zc, zq, wq, chunk=64):
    """
    Full scattering transform through Nachman's boundary integral equation

        psi(z,k) = e^{ikz} - int_{dOmega} G_k(z - zeta) (Lambda_sigma - Lambda_1) psi(zeta,k) ds

    solved for the trace of ``psi`` on the electrodes, followed by

        t(k) = int_{dOmega} e^{i conj(k) conj(z)} (Lambda_sigma - Lambda_1) psi(.,k) ds.

    ``psi`` is represented by its electrode values, ``zc`` are the electrode
    centres on the unit circle.  Dropping the integral term reproduces
    :func:`scattering_transform_exp` exactly, which is the Born approximation.
    """
    k = np.asarray(k, dtype=complex).ravel()
    L = len(arc)
    # electrode values of (Lambda_sigma - Lambda_1) applied to a boundary
    # function given by its electrode values: project, apply dL, expand back
    apply_dL = C.T @ dL @ (C * arc[None, :])

    t = np.empty(len(k), dtype=complex)
    for start in range(0, len(k), chunk):
        kk = k[start:start + chunk]
        # single-layer matrix S[k, i, j] = int_{e_j} G_k(z_i - zeta) ds(zeta)
        diff = zc[None, :, None, None] - zq[None, None, :, :]
        Gk = faddeev_green(kk[:, None, None, None], diff)
        # an odd quadrature order puts a node exactly at the electrode centre,
        # where G_k is infinite; those entries all sit on the diagonal, which is
        # replaced below by the exact integral over the arc
        Gk[~np.isfinite(Gk)] = 0.0
        S = np.einsum("jq,kijq->kij", wq, Gk)
        idx = np.arange(L)
        S[:, idx, idx] = _faddeev_self_term(kk, arc)

        # incident field: the mean of e^{ikz} over each electrode
        kz = kk[:, None, None] * zq[None, :, :]
        f = np.einsum("lq,klq->kl", wq, np.exp(1j * kz)) / arc[None, :]

        A = np.eye(L)[None, :, :] + S @ apply_dL[None, :, :]
        psi = np.linalg.solve(A, f[:, :, None])[:, :, 0]

        g = psi @ apply_dL.T                       # (Lambda_sigma - Lambda_1) psi
        de = np.conj(np.einsum("lq,klq->kl", wq, np.exp(-1j * kz)))
        t[start:start + chunk] = np.einsum("kl,kl->k", de, g)
    return t


# --------------------------------------------------------------------------- #
#  the D-bar equation
# --------------------------------------------------------------------------- #
#: ``transform_growth`` above this triggers a warning; see that function.
GROWTH_WARN = 20.0


def transform_growth(t, K, R):
    """
    Ratio of ``max|t|`` on the outer half of the k-disk to the inner half.

    The true scattering transform of a conductivity is bounded.  The *error* in
    it is not: ``e^{ikz}`` grows like ``e^{|k|}`` on the boundary and ``t`` is
    bilinear in it, so whatever the DN map is wrong by - measurement noise, or
    with noise-free data the fact that ``L`` electrodes only ever see an
    ``L-1`` dimensional slice of it - is amplified like ``e^{2|k|}``.
    Suppressing that is the entire purpose of the truncation ``|k| <= R``.

    This ratio is a *relative* indicator, not an absolute pass/fail: ``t``
    vanishes at ``k = 0`` and peaks at a finite radius, so even a perfectly
    healthy transform gives a value above one.  What matters is how fast it
    grows as ``R`` is increased over a sweep.  Measured on 16 electrodes
    covering half the boundary (this framework's default geometry):

    ===========  ==============================================================
    ``3 .. 5``   noise-free; ``R`` can still be raised, the image keeps sharpening
    ``5 .. 15``  the useful range with 0.5-2% noise; the best image is usually
                 at the top of it
    ``> 20``     the transform is blowing up and the image is amplified error.
                 A warning is issued.
    ===========  ==============================================================
    """
    inner = np.abs(K) <= R / 2.0
    outer = (np.abs(K) > R / 2.0) & (np.abs(K) <= R)
    lo = np.abs(t[inner]).max() if inner.any() else 0.0
    hi = np.abs(t[outer]).max() if outer.any() else 0.0
    if lo > 0:
        return float(hi / lo)
    # a transform that is identically zero (a homogeneous target) is not
    # growing; only a zero inner part with a non-zero outer one is pathological
    return 0.0 if hi == 0 else np.inf


def dbar_grid(R, size):
    """
    Vainikko grid for the D-bar equation.

    The scattering transform is supported in ``|k| <= R``; the periodised
    equation then needs a square ``(-s, s)^2`` with ``s >= 2R`` so that the
    difference of any two points of the support stays inside it and the
    periodic convolution kernel never wraps around.  Returns ``(K, h)`` with
    ``K`` the ``(size, size)`` complex grid in natural (plotting) order and
    ``h`` its step; ``k = 0`` is the grid point ``[size//2, size//2]``.
    """
    size = int(size)
    if size & (size - 1):
        raise ValueError(f"the k-grid size must be a power of two, got {size}")
    s = 2.0 * float(R)
    h = 2.0 * s / size
    j = np.arange(-(size // 2), size // 2)
    return h * (j[None, :] + 1j * j[:, None]), h


def _cauchy_kernel_hat(K, h):
    """FFT of the periodised Green's function ``1/(pi k)`` of the D-bar operator.

    The value at ``k = 0`` is set to zero: the singularity is integrable and,
    because ``1/k`` is odd, zero is in fact the exact midpoint-rule value of
    ``int_{cell} dk/(pi k)`` over the cell centred at the origin.
    """
    g = np.zeros_like(K)
    nz = K != 0
    g[nz] = 1.0 / (np.pi * K[nz])
    return np.fft.fft2(np.fft.ifftshift(g)) * (h * h)


def _gmres(op, b, x0, tol, restart, maxiter):
    """scipy renamed ``tol`` to ``rtol`` in 1.14; support both."""
    try:
        return gmres(op, b, x0=x0, rtol=tol, atol=0.0,
                     restart=restart, maxiter=maxiter)
    except TypeError:  # pragma: no cover - old scipy
        return gmres(op, b, x0=x0, tol=tol, atol=0.0,
                     restart=restart, maxiter=maxiter)


def solve_dbar(t, K, h, zs, tol=1e-6, restart=40, maxiter=10):
    """
    Solve the D-bar equation and return ``mu(z, 0)`` for every ``z`` in ``zs``.

    For each ``z`` the equation

        mu = 1 + (1/(pi k)) * [ T_z conj(mu) ],
        T_z(k) = t(k) e_{-k}(z) / (4 pi conj(k)),   e_{-k}(z) = e^{-2i Re(kz)}

    is discretised on the periodised grid ``K`` and solved with GMRES.  The
    operator ``mu -> mu - A(T_z conj(mu))`` is **real**-linear but not
    complex-linear because of the conjugation, so the real and imaginary parts
    are carried as separate unknowns - GMRES on the complex vector alone would
    silently solve a different problem.

    Returns ``(mu0, n_failed)``.
    """
    M = K.shape[0]
    n = M * M
    Kbar = np.conj(K)

    with np.errstate(divide="ignore", invalid="ignore"):
        base = t / (4.0 * np.pi * Kbar)
    base[~np.isfinite(base)] = 0.0            # k = 0

    Ghat = _cauchy_kernel_hat(K, h)
    b = np.concatenate([np.ones(n), np.zeros(n)])

    mu0 = np.empty(len(zs), dtype=complex)
    n_failed = 0
    for p, z in enumerate(zs):
        Tz = base * np.exp(-2j * np.real(K * z))

        def matvec(x, _T=Tz):
            mu = (x[:n] + 1j * x[n:]).reshape(M, M)
            y = mu - np.fft.ifft2(Ghat * np.fft.fft2(_T * np.conj(mu)))
            return np.concatenate([y.real.ravel(), y.imag.ravel()])

        op = LinearOperator((2 * n, 2 * n), matvec=matvec, dtype=float)
        sol, info = _gmres(op, b, b, tol, restart, maxiter)
        n_failed += int(info != 0)
        mu = (sol[:n] + 1j * sol[n:]).reshape(M, M)
        mu0[p] = mu[M // 2, M // 2]

    return mu0, n_failed


# --------------------------------------------------------------------------- #
#  the reconstructor
# --------------------------------------------------------------------------- #
class DbarSolver(Reconstructor):
    """
    Regularised D-bar reconstruction from Complete-Electrode-Model data.

    :param eit_solver:   :class:`EIT` forward solver (supplies the mesh, the
                         electrode geometry and the homogeneous reference)
    :param backCond:     background conductivity ``sigma_0``
    :param R:            truncation radius of the scattering transform.  This
                         is *the* regularisation parameter: small ``R`` gives a
                         smooth, stable, low-contrast image, large ``R`` a
                         sharper but noisier one.  ``R ~ 3.5-5`` is the usual
                         range for 16-32 electrodes.
    :param k_grid:       number of points per axis of the k-grid (power of two)
    :param z_grid:       number of points per axis of the reconstruction grid
    :param scattering:   ``"exp"`` (Born, fast, default) or ``"bie"`` (full)
    :param clip:         ``(min, max)`` clip applied to the final conductivity
    :param contact_correction:  subtract the known contact-impedance shunt from
                         the ND matrix before inverting it
    :param reference:    ``(N, L)`` homogeneous voltages for ``Lambda_1``.  If
                         ``None`` they are simulated on this solver's own mesh
                         with ``sigma = backCond``, which is what makes the CEM
                         modelling error cancel in the difference.
    :param dn_rcond:     optional truncated-SVD threshold for the ND inversion
    :param t_cutoff:     optional hard cap: ``t(k)`` is set to zero wherever
                         ``|t(k)| > t_cutoff``.  A blunt but effective guard
                         against the exponential blow-up of the transform at
                         large ``|k|``; leave at ``None`` to see the raw
                         behaviour and choose ``R`` from it instead.

    Choosing ``R``
    --------------
    There is no closed-form rule that is useful in practice - the published
    convergence rate ``R ~ log(1/noise)`` carries an unknown constant.  Sweep
    it: ``"dbar": {"R": [3.0, 3.5, 4.0, 4.5]}`` in ``simulate.py``.  The
    ``t_growth`` diagnostic (see :func:`transform_growth`) says when ``R`` has
    gone past what the electrode array can support.  ``R ~ 3-4`` is right for
    16 electrodes, ``4-5`` for 32.
    """

    def __init__(
        self,
        eit_solver,
        backCond=1.0,
        R=4.0,
        k_grid=64,
        z_grid=64,
        z_margin=1.05,
        scattering="exp",
        clip=(0.01, 5.0),
        contact_correction=True,
        reference=None,
        quad_order=4,
        dn_rcond=None,
        t_cutoff=None,
        gmres_tol=1e-6,
        gmres_restart=40,
        gmres_maxiter=10,
    ):
        super().__init__(eit_solver)

        if scattering not in ("exp", "bie"):
            raise ValueError(
                f"scattering must be 'exp' or 'bie', got {scattering!r}"
            )
        if R <= 0:
            raise ValueError("the truncation radius R must be positive")

        self.backCond = float(backCond)
        self.R = float(R)
        self.k_grid = int(k_grid)
        self.z_grid = int(z_grid)
        self.z_margin = float(z_margin)
        self.scattering = scattering
        self.clip = clip
        self.contact_correction = bool(contact_correction)
        self.reference = reference
        self.quad_order = int(quad_order)
        self.dn_rcond = dn_rcond
        self.t_cutoff = t_cutoff
        self.gmres_tol = float(gmres_tol)
        self.gmres_restart = int(gmres_restart)
        self.gmres_maxiter = int(gmres_maxiter)

        # D-bar is direct: nothing to plot against an iteration counter.
        self.history = []
        self.diagnostics = {}

        # ---- geometry, fixed by the mesh -------------------------------- #
        theta, arclen, radius = electrode_geometry(eit_solver)
        self.radius = radius
        self.arclen_phys = arclen
        self.arc = arclen / radius                     # unit-circle lengths
        self.theta = theta
        self.zc = np.exp(1j * theta)
        self.zq, self.wq = _arc_quadrature(theta, self.arc, self.quad_order)

        self.B = orthonormalising_factor(eit_solver.Inj, self.arc)
        #: orthonormal boundary basis, ``phi_m = sum_i C[m,i] chi_i``
        self.C = self.B @ eit_solver.Inj / self.arc[None, :]

        # ---- k-grid ------------------------------------------------------ #
        self.K, self.h_k = dbar_grid(self.R, self.k_grid)
        self.k_mask = np.abs(self.K) <= self.R

        # ---- reconstruction grid ----------------------------------------- #
        ax = np.linspace(-self.z_margin, self.z_margin, self.z_grid)
        self.z_axis = ax
        Zx, Zy = np.meshgrid(ax, ax)                   # [row, col] = [y, x]
        self.Z = Zx + 1j * Zy
        # solve slightly beyond the disk so that cells touching the rim are
        # interpolated from computed values rather than from the fill value
        self.z_inside = np.abs(self.Z) <= self.z_margin

        self.t = self.mu0 = self.image = self.extent = None

    # ------------------------------------------------------------------ #
    def _reference_voltages(self):
        if self.reference is not None:
            return np.atleast_2d(np.asarray(self.reference, dtype=float))
        sig = Function(self.eit_solver.V_sigma)
        sig.x.array[:] = self.backCond
        _, U_ref = self.eit_solver.forward_solve(sig)
        return np.asarray(U_ref, dtype=float)

    def dn_difference(self, Umeas):
        """``Lambda_sigma - Lambda_1`` as a matrix in the orthonormal basis."""
        solver = self.eit_solver
        kw = dict(
            arc_unit=self.arc,
            radius=self.radius,
            background=self.backCond,
            B=self.B,
            rcond=self.dn_rcond,
        )
        if self.contact_correction:
            kw.update(z=solver.z, arclen_phys=self.arclen_phys)

        L_sigma = dn_matrix(solver.Inj, Umeas, **kw)
        L_ref = dn_matrix(solver.Inj, self._reference_voltages(), **kw)
        return L_sigma - L_ref

    def scattering_transform(self, dL):
        """``t(k)`` on the k-grid, truncated to ``|k| <= R`` and zero at ``k = 0``."""
        t = np.zeros_like(self.K)
        sel = self.k_mask & (self.K != 0)
        k = self.K[sel]
        if self.scattering == "exp":
            t[sel] = scattering_transform_exp(k, dL, self.C, self.zq, self.wq)
        else:
            t[sel] = scattering_transform_bie(
                k, dL, self.C, self.arc, self.zc, self.zq, self.wq
            )
        if self.t_cutoff is not None:
            t[np.abs(t) > float(self.t_cutoff)] = 0.0
        return t

    # ------------------------------------------------------------------ #
    def forward(self, Umeas, verbose=False):
        """
        Reconstruct from ``(N, L)`` electrode voltages.

        Returns a DG0 :class:`dolfinx.fem.Function` on the solver mesh.
        """
        solver = self.eit_solver
        Umeas = np.asarray(Umeas, dtype=float).reshape(-1, solver.L)
        if Umeas.shape[0] != solver.Inj.shape[0]:
            raise ValueError(
                f"got {Umeas.shape[0]} voltage patterns but the solver was "
                f"built with {solver.Inj.shape[0]} injection patterns"
            )

        dL = self.dn_difference(Umeas)
        t = self.scattering_transform(dL)

        growth = transform_growth(t, self.K, self.R)
        if growth > GROWTH_WARN:
            warnings.warn(
                f"the scattering transform is blowing up towards |k| = R "
                f"(outer/inner max |t| = {growth:.1f}); R = {self.R:g} is too "
                "large for this data, so the reconstruction is dominated by "
                "amplified error rather than signal. Reduce R, or set t_cutoff.",
                stacklevel=2,
            )

        zs = self.Z[self.z_inside]
        mu0_in, n_failed = solve_dbar(
            t, self.K, self.h_k, zs,
            tol=self.gmres_tol,
            restart=self.gmres_restart,
            maxiter=self.gmres_maxiter,
        )

        mu0 = np.ones(self.Z.shape, dtype=complex)
        mu0[self.z_inside] = mu0_in

        # sigma = mu(z,0)^2.  mu(z,0) is real in exact arithmetic; the residual
        # imaginary part is a useful measure of how far the truncated, noisy
        # scattering data is from an exact D-bar problem.
        image = self.backCond * np.real(mu0) ** 2
        if self.clip is not None:
            image = np.clip(image, self.clip[0], self.clip[1])

        self.t = t
        self.mu0 = mu0
        self.image = image
        self.extent = (
            -self.z_margin * self.radius, self.z_margin * self.radius,
            -self.z_margin * self.radius, self.z_margin * self.radius,
        )
        self.diagnostics = {
            "R": self.R,
            "scattering": self.scattering,
            "k_grid": self.k_grid,
            "z_grid": self.z_grid,
            "t_max": float(np.abs(t).max()),
            "t_growth": growth,
            "mu_imag_max": float(np.abs(np.imag(mu0)).max()),
            "gmres_failures": int(n_failed),
            "n_z": int(len(zs)),
            "dn_difference_norm": float(np.linalg.norm(dL)),
        }
        if n_failed:
            warnings.warn(
                f"GMRES did not reach {self.gmres_tol:g} at {n_failed} of "
                f"{len(zs)} reconstruction points; increase gmres_maxiter or "
                "lower R.",
                stacklevel=2,
            )
        if verbose:
            print(
                f"  D-bar: |t|_max={self.diagnostics['t_max']:.3g}  "
                f"t_growth={growth:.2f}  "
                f"max|Im mu(z,0)|={self.diagnostics['mu_imag_max']:.3g}  "
                f"gmres failures={n_failed}/{len(zs)}"
            )

        return self._to_function(image)

    # ------------------------------------------------------------------ #
    def _to_function(self, image):
        """Sample the pixel image at the mesh cell centroids."""
        interp = RegularGridInterpolator(
            (self.z_axis, self.z_axis), image,
            method="linear", bounds_error=False, fill_value=None,
        )
        pos = self.eit_solver.cell_centers() / self.radius
        lim = self.z_axis[-1]
        pts = np.clip(np.column_stack([pos[:, 1], pos[:, 0]]), -lim, lim)

        out = Function(self.eit_solver.V_sigma)
        out.x.array[:] = interp(pts)
        return out
