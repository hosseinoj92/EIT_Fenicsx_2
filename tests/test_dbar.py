#!/usr/bin/env python
"""
Validation suite for the D-bar reconstructor (``eitcore/dbar.py``).

Every check here is against something known in closed form, not against a
previous output of this code:

  * Faddeev's Green's function: harmonicity, the log singularity and the exact
    scaling  G_k(z) = G_1(kz)
  * the Cauchy transform  (1/(pi k)) *  against an analytic dbar problem, and
    its convergence rate under grid refinement
  * the Neumann-to-Dirichlet matrix against the *exact* projection of the
    continuum ND map of the unit disk (this pins down every electrode-area,
    radius and background factor at once)
  * the DN eigenvalues against  Lambda_1 e^{i n theta} = |n| e^{i n theta}
  * the scattering transform against the Calderon/Born formula
        t(k) ~ -2 |k|^2 dsigma^(-2 k1, 2 k2)
    for a NON-radial perturbation - this is what pins down the sign, the
    conjugation and the orientation of the k-plane; a mirrored or rotated
    reconstruction fails here and nowhere else
  * t^bie -> t^exp as the contrast goes to zero (the Born limit)
  * the D-bar solve: an off-centre bump comes back at the right place with the
    right amplitude
  * invariance of the whole pipeline under a change of domain radius and of
    background conductivity
  * localisation of a real inclusion from Complete-Electrode-Model FEM data

Run:  python tests/test_dbar.py
"""

import os
import pathlib
import sys
import tempfile
import warnings

os.environ.setdefault(
    "XDG_CACHE_HOME", str(pathlib.Path(tempfile.gettempdir()) / f"fenics_d_{os.getpid()}")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402
from dolfinx.fem import Function                                # noqa: E402

from mesh_tools import build_mesh                               # noqa: E402
from eitcore import EIT, DbarSolver                             # noqa: E402
from eitcore.utils import current_method, default_n_patterns    # noqa: E402
from eitcore.phantoms import build_phantom                      # noqa: E402
from eitcore import algorithms as A                             # noqa: E402
from eitcore.dbar import (                                      # noqa: E402
    faddeev_green, dbar_grid, _cauchy_kernel_hat, solve_dbar,
    orthonormalising_factor, nd_matrix, contact_shift,
    scattering_transform_exp, scattering_transform_bie,
    _arc_quadrature, transform_growth, electrode_geometry, GROWTH_WARN,
)

_EULER = 0.5772156649015328606
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


# --------------------------------------------------------------------------- #
#  a continuum unit-disk model, used as the reference truth
# --------------------------------------------------------------------------- #
NMAX = 400
NN = np.concatenate([np.arange(-NMAX, 0), np.arange(1, NMAX + 1)])


def gap_basis(L):
    """Trigonometric drive on ``L`` electrodes covering the whole unit circle."""
    theta = 2 * np.pi * np.arange(L) / L
    arc = np.full(L, 2 * np.pi / L)
    rows = []
    for m in range(1, L // 2 + 1):
        rows.append(np.cos(m * theta))
        if m < L // 2:
            rows.append(np.sin(m * theta))
    Inj = np.asarray(rows)
    Inj -= Inj.mean(axis=1, keepdims=True)
    B = orthonormalising_factor(Inj, arc)
    return theta, arc, Inj, B, B @ Inj / arc[None, :]


def _arc_exp(theta, arc):
    """``W[l, n] = int_{e_l} e^{i n theta} dtheta``."""
    return (np.exp(1j * NN[None, :] * theta[:, None])
            * 2 * np.sin(NN[None, :] * arc[:, None] / 2) / NN[None, :])


def radial_dn_eigs(rho, s_in):
    """``Lambda e^{i n th} = |n| lam_n e^{i n th}`` for a concentric disk inclusion."""
    m = np.abs(NN)
    mu = (s_in - 1.0) / (s_in + 1.0)
    return m * (1 + mu * rho ** (2 * m)) / (1 - mu * rho ** (2 * m))


def project(C, theta, arc, eigs):
    """Exact matrix of a Fourier-diagonal boundary operator in the basis ``C``."""
    Phi = C @ np.conj(_arc_exp(theta, arc))          # (N, modes), = 2pi * coeffs
    return np.real((Phi * eigs[None, :]) @ np.conj(Phi).T) / (2 * np.pi)


def synth_electrode_data(Inj, theta, arc, eigs):
    """Gap-model electrode voltages produced by a Fourier-diagonal DN map."""
    W = _arc_exp(theta, arc)
    jhat = ((Inj / arc[None, :]) @ np.conj(W)) / (2 * np.pi)
    return np.real(((jhat / eigs[None, :]) @ W.T) / arc[None, :])


# ═══════════════════════════════ tests ══════════════════════════════════════
def test_faddeev():
    section("Faddeev Green's function")
    for k in [1.0, 2.5 + 1j, -3j]:
        zz = 1e-6
        got = faddeev_green(np.array([k]), np.array([zz]))[0]
        want = -(np.log(abs(zz)) + _EULER + np.log(abs(k))) / (2 * np.pi)
        check(f"log singularity, k={k}", abs(got - want) < 1e-5,
              f"{got:.8f} vs {want:.8f}")

    for k in [1.0, 2.0 - 0.7j]:
        h, z0 = 1e-3, 0.37 - 0.21j
        pts = np.array([z0, z0 + h, z0 - h, z0 + 1j * h, z0 - 1j * h])
        v = faddeev_green(np.full(5, k), pts)
        lap = (v[1] + v[2] + v[3] + v[4] - 4 * v[0]) / h ** 2
        check(f"harmonic away from 0, k={k}", abs(lap) < 1e-3, f"laplacian {lap:.2e}")

    k, z = 1.7 - 0.4j, 0.3 + 0.9j
    check("exact scaling G_k(z) = G_1(kz)",
          abs(faddeev_green(np.array([k]), np.array([z]))[0]
              - faddeev_green(np.array([1.0]), np.array([k * z]))[0]) < 1e-12)


def test_cauchy_transform():
    section("Cauchy transform and the D-bar solver")
    K, h = dbar_grid(4.0, 32)
    mu0, nf = solve_dbar(np.zeros_like(K), K, h, np.array([0j, 0.3 + 0.2j]))
    check("t == 0 gives mu == 1", np.allclose(mu0, 1.0, atol=1e-12) and nf == 0,
          str(np.round(mu0, 12)))

    # dbar u = f with f = exp(-|k|^2) has the closed form u = (1-exp(-|k|^2))/k
    errs = []
    for M in [64, 128, 256]:
        K, h = dbar_grid(3.0, M)
        f = np.exp(-np.abs(K) ** 2)
        u = np.fft.ifft2(_cauchy_kernel_hat(K, h) * np.fft.fft2(f))
        with np.errstate(divide="ignore", invalid="ignore"):
            exact = (1 - np.exp(-np.abs(K) ** 2)) / K
        exact[K == 0] = 0.0
        m = np.abs(K) < 1.5
        errs.append(np.abs(u - exact)[m].max())
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    check("Cauchy transform converges under refinement",
          errs[-1] < 1e-3 and all(r > 1.7 for r in ratios),
          f"errors {['%.1e' % e for e in errs]}, ratios {['%.1f' % r for r in ratios]}")

    check("the k-grid contains k = 0 at its centre",
          dbar_grid(4.0, 32)[0][16, 16] == 0)
    try:
        dbar_grid(4.0, 48)
        ok = False
    except ValueError:
        ok = True
    check("a non-power-of-two k-grid is rejected", ok)


def test_nd_matrix():
    section("Neumann-to-Dirichlet matrix")
    for L in [16, 32]:
        theta, arc, Inj, B, C = gap_basis(L)
        for rho, s_in in [(0.5, 1.0), (0.5, 2.0), (0.35, 0.25)]:
            eigs = radial_dn_eigs(rho, s_in)
            U = synth_electrode_data(Inj, theta, arc, eigs)
            got = nd_matrix(Inj, U, B)
            want = project(C, theta, arc, 1.0 / eigs)
            rel = np.linalg.norm(got - want) / np.linalg.norm(want)
            check(f"ND matrix exact, L={L} rho={rho} sigma_in={s_in}", rel < 1e-12,
                  f"rel err {rel:.2e}")

    # the inverse of the measured ND matrix reproduces Lambda_1 = |n| on the
    # modes the electrodes can actually resolve
    theta, arc, Inj, B, C = gap_basis(32)
    eigs = radial_dn_eigs(0.5, 1.0)
    L1 = np.linalg.inv(nd_matrix(Inj, synth_electrode_data(Inj, theta, arc, eigs), B))
    ev = np.sort(np.linalg.eigvalsh(L1))[:8]
    want = np.repeat(np.arange(1, 5), 2)
    err = np.abs(ev - want) / want
    check("DN eigenvalues reproduce |n| on the resolved modes",
          err.max() < 0.06, f"{np.round(ev, 4)} vs {want}")
    # the electrodes are piecewise constant, so the error must be small for the
    # low modes and grow with the mode number - that growth is the reason the
    # scattering transform has to be truncated at all
    check("the DN error grows with the mode number",
          err[0] < 0.005 and np.all(np.diff(err[::2]) > 0),
          f"per-mode error {np.round(err[::2], 4)}")


def test_scattering_orientation():
    section("scattering transform: sign, conjugation and orientation")
    L = 32
    theta, arc, Inj, B, C = gap_basis(L)
    Phi = (C @ np.conj(_arc_exp(theta, arc))) / (2 * np.pi)

    # a small OFF-CENTRE, non-radial perturbation.  Its linearised DN map is the
    # Calderon form  <dLambda f, g> = int dsigma grad(u_f) . grad(u_g)  with
    # u_f, u_g the harmonic extensions of the basis functions.
    z0, w0, amp, ng = 0.35 - 0.22j, 0.16, 1e-3, 300
    gx = np.linspace(-1, 1, ng)
    Zg = gx[None, :] + 1j * gx[:, None]
    dA = (gx[1] - gx[0]) ** 2
    dsig = amp * np.exp(-np.abs(Zg - z0) ** 2 / w0 ** 2) * (np.abs(Zg) < 1)

    zf = Zg.ravel()
    dz = np.zeros((zf.size, len(NN)), dtype=complex)
    dzb = np.zeros_like(dz)
    for i, n in enumerate(NN):
        if n > 0:
            dz[:, i] = n * zf ** (n - 1)
        else:
            dzb[:, i] = abs(n) * np.conj(zf) ** (abs(n) - 1)
    G = [(dz @ Phi[m], dzb @ Phi[m]) for m in range(C.shape[0])]

    ds = dsig.ravel()
    Nb = C.shape[0]
    dLam = np.zeros((Nb, Nb))
    for m in range(Nb):
        for j in range(m, Nb):
            v = 2 * (G[m][0] * G[j][1] + G[m][1] * G[j][0])
            dLam[m, j] = dLam[j, m] = np.real(np.sum(ds * v) * dA)

    zq, wq = _arc_quadrature(theta, arc, 8)
    kk = np.array([0.8 + 0.3j, -1.2 + 0.7j, 0.5 - 1.5j, 2.0 + 0j, 2.0j, 1.4 - 1.4j])
    t_num = scattering_transform_exp(kk, dLam, C, zq, wq)
    t_born = np.array([
        (-2 * abs(k) ** 2) * np.sum(dsig * np.exp(1j * (k * Zg + np.conj(k) * np.conj(Zg)))) * dA
        for k in kk
    ])
    rel = np.abs(t_num - t_born) / np.abs(t_born)
    check("t^exp matches the Calderon/Born formula -2|k|^2 dsigma^(-2k1, 2k2)",
          rel.max() < 0.05, f"max rel err {rel.max():.3e} over {len(kk)} values of k")

    # ...and the D-bar solve puts the bump back where it started
    R, M = 6.0, 64
    K, h = dbar_grid(R, M)
    sel = (np.abs(K) <= R) & (K != 0)
    t = np.zeros_like(K)
    t[sel] = np.array([
        (-2 * abs(k) ** 2) * np.sum(dsig * np.exp(1j * (k * Zg + np.conj(k) * np.conj(Zg)))) * dA
        for k in K[sel]
    ])
    ax = np.linspace(-1, 1, 41)
    Zs = ax[None, :] + 1j * ax[:, None]
    inside = np.abs(Zs) <= 1.0
    mu0, _ = solve_dbar(t, K, h, Zs[inside])
    img = np.ones(Zs.shape)
    img[inside] = np.real(mu0) ** 2
    p = np.unravel_index(np.argmax(img), img.shape)
    check("the D-bar solve places the bump at the right z",
          abs(Zs[p] - z0) < 0.12 and img[p] > 1.0,
          f"peak at {Zs[p]:.3f}, truth {z0}, value {img[p]:.6f} (truth {1+amp})")


def test_scattering_properties():
    section("scattering transform: structural properties")
    theta, arc, Inj, B, C = gap_basis(16)
    zq, wq = _arc_quadrature(theta, arc, 6)
    zc = np.exp(1j * theta)
    kk = np.array([0.5, 1.0 + 1j, -2.0])

    dL0 = np.zeros((C.shape[0], C.shape[0]))
    check("t^exp vanishes when Lambda_sigma == Lambda_1",
          np.allclose(scattering_transform_exp(kk, dL0, C, zq, wq), 0))
    check("t^bie vanishes when Lambda_sigma == Lambda_1",
          np.allclose(scattering_transform_bie(kk, dL0, C, arc, zc, zq, wq), 0))

    def dL_of(s_in, rho=0.5):
        e1 = radial_dn_eigs(rho, 1.0)
        es = radial_dn_eigs(rho, s_in)
        inv = lambda e: np.linalg.inv(  # noqa: E731
            nd_matrix(Inj, synth_electrode_data(Inj, theta, arc, e), B))
        return inv(es) - inv(e1)

    # the Born approximation is the small-contrast limit of the full transform
    errs = []
    for s_in in [1.01, 1.05, 1.2]:
        d = dL_of(s_in)
        te = scattering_transform_exp(kk, d, C, zq, wq)
        tb = scattering_transform_bie(kk, d, C, arc, zc, zq, wq)
        errs.append(np.abs(te - tb).max() / np.abs(te).max())
    check("t^bie -> t^exp as the contrast -> 0",
          errs[0] < 0.01 and errs[0] < errs[1] < errs[2],
          f"rel. differences {['%.2e' % e for e in errs]} at contrast 1.01/1.05/1.2")

    # a radially symmetric conductivity has a radially symmetric |t|
    dL = dL_of(2.0)
    K, _ = dbar_grid(4.0, 32)
    sel = (np.abs(K) <= 4.0) & (K != 0)
    tv = np.abs(scattering_transform_exp(K[sel], dL, C, zq, wq))
    r = np.round(np.abs(K[sel]), 6)
    spread = max(np.ptp(tv[r == v]) / max(tv[r == v].mean(), 1e-30)
                 for v in np.unique(r) if (r == v).sum() > 2)
    check("|t(k)| is radially symmetric for a radial sigma", spread < 1e-3,
          f"max relative spread within a ring {spread:.2e}")

    # With an exact DN map the transform stays healthy as far out as R = 6...
    def growth_over_R(d, radii=(3.0, 4.0, 5.0, 6.0)):
        out = []
        for Rt in radii:
            Kt, _ = dbar_grid(Rt, 32)
            st = (np.abs(Kt) <= Rt) & (Kt != 0)
            tt = np.zeros_like(Kt)
            tt[st] = scattering_transform_exp(Kt[st], d, C, zq, wq)
            out.append(transform_growth(tt, Kt, Rt))
        return out

    clean = growth_over_R(dL)
    check("transform_growth stays low when the DN map is exact",
          max(clean) < 2.0, f"R=3,4,5,6 -> {[round(v, 2) for v in clean]}")

    # ...and rises steeply once the voltages carry noise, which is exactly the
    # amplification (~e^{2|k|}) that the truncation |k| <= R exists to stop.
    e1, es = radial_dn_eigs(0.5, 1.0), radial_dn_eigs(0.5, 2.0)
    U = synth_electrode_data(Inj, theta, arc, es)
    rng = np.random.default_rng(0)
    Un = U + 0.005 * np.abs(U).max() * rng.standard_normal(U.shape)
    dL_n = (np.linalg.inv(nd_matrix(Inj, Un, B))
            - np.linalg.inv(nd_matrix(Inj, synth_electrode_data(Inj, theta, arc, e1), B)))
    noisy = growth_over_R(dL_n)
    check("transform_growth rises with R on noisy data and trips the threshold",
          np.all(np.diff(noisy) > 0) and noisy[-1] > GROWTH_WARN,
          f"R=3,4,5,6 -> {[round(v, 2) for v in noisy]}, threshold {GROWTH_WARN}")
    check("noise makes the transform grow faster than exact data does",
          all(n > c for n, c in zip(noisy, clean)),
          f"noisy {[round(v, 1) for v in noisy]} vs exact {[round(v, 2) for v in clean]}")


def test_end_to_end_continuum():
    section("end-to-end on an analytic concentric inclusion")
    theta, arc, Inj, B, C = gap_basis(16)
    zq, wq = _arc_quadrature(theta, arc, 6)
    rho, s_in = 0.5, 2.0
    inv = lambda e: np.linalg.inv(  # noqa: E731
        nd_matrix(Inj, synth_electrode_data(Inj, theta, arc, e), B))
    dL = inv(radial_dn_eigs(rho, s_in)) - inv(radial_dn_eigs(rho, 1.0))

    K, h = dbar_grid(3.0, 64)
    sel = (np.abs(K) <= 3.0) & (K != 0)
    t = np.zeros_like(K)
    t[sel] = scattering_transform_exp(K[sel], dL, C, zq, wq)
    # truncating at R low-passes the image, so the jump at |z| = rho is smeared
    # over a band around it; sample well inside and well outside that band.
    zs = np.array([0.25 + 0j, 0.85 + 0j, 0.85j])
    mu0, nf = solve_dbar(t, K, h, zs)
    sig = np.real(mu0) ** 2
    check("inclusion recovered inside", 1.6 < sig[0] < 2.6, f"sigma(0.25)={sig[0]:.3f} (truth 2)")
    check("background recovered outside",
          0.85 < sig[1] < 1.15 and 0.85 < sig[2] < 1.15,
          f"sigma(0.85)={sig[1]:.3f}, sigma(0.85i)={sig[2]:.3f} (truth 1)")
    check("mu(z,0) is real to numerical precision",
          np.abs(np.imag(mu0)).max() < 1e-4, f"max |Im mu| {np.abs(np.imag(mu0)).max():.2e}")
    check("GMRES converged everywhere", nf == 0)


# --------------------------------------------------------------------------- #
#  CEM / FEM based checks
# --------------------------------------------------------------------------- #
_cache = {}


def cem_solver(radius=1.0, L=16, z=1e-2, h=(0.20, 0.14)):
    key = (radius, L, z, h)
    if key not in _cache:
        path = build_mesh(dict(radius=radius, n_electrodes=L,
                               h_center=h[0] * radius, h_edge=h[1] * radius,
                               electrode_width_deg=12.0,
                               n_per_electrode=8, n_per_gap=4))
        Inj = current_method(L=L, l=default_n_patterns(4, L), method=4, value=1.0)
        _cache[key] = EIT(L, Inj, np.full(L, z), backend="Scipy", mesh_name=path)
    return _cache[key]


def cem_data(solver, sigma_values):
    f = Function(solver.V_sigma)
    f.x.array[:] = sigma_values
    _, U = solver.forward_solve(f)
    return np.asarray(U)


def phantom(pos, inclusions, background=1.0, radius=1.0):
    return build_phantom({"background": background, "inclusions": inclusions},
                         pos[:, 0], pos[:, 1], domain_radius=radius)


def test_contact_shift():
    section("contact-impedance correction")
    solver = cem_solver()
    theta, arclen, radius = electrode_geometry(solver)
    arc = arclen / radius
    B = orthonormalising_factor(solver.Inj, arc)
    U = cem_data(solver, 1.0)
    R = nd_matrix(solver.Inj, U, B)
    shift = contact_shift(solver.Inj, B, solver.z, arclen)
    # uniform z and equal electrodes  ->  the shift is  z/radius  times identity
    want = solver.z[0] / radius * np.eye(R.shape[0])
    check("the contact shift is z/radius times the identity",
          np.allclose(shift, want, rtol=2e-2, atol=1e-6),
          f"max dev {np.abs(shift - want).max():.2e}")
    check("removing it moves the ND matrix towards the continuum one",
          np.linalg.norm(R - shift) < np.linalg.norm(R))


def test_cem_homogeneous():
    section("CEM data: a homogeneous target")
    solver = cem_solver()
    for bg in [1.0, 0.4]:
        U = cem_data(solver, bg)
        rec = DbarSolver(solver, backCond=bg, R=3.5, k_grid=32, z_grid=48)
        sig = rec.to_dg0(rec.forward(U))
        check(f"homogeneous sigma = {bg} comes back flat",
              np.allclose(sig, bg, rtol=1e-6, atol=1e-8),
              f"range [{sig.min():.8f}, {sig.max():.8f}]")
        check(f"the scattering transform vanishes, background {bg}",
              rec.diagnostics["t_max"] < 1e-8, f"|t|max {rec.diagnostics['t_max']:.2e}")


def test_cem_inclusion():
    section("CEM data: locating a real inclusion")
    solver = cem_solver()
    pos = solver.cell_centers()
    sig_true = phantom(pos, [{"shape": "circle", "center": [0.45, 0.30],
                              "radius": 0.22, "value": 3.0}])
    U = cem_data(solver, sig_true)
    rec = DbarSolver(solver, backCond=1.0, R=3.5, k_grid=32, z_grid=48)
    sig = rec.to_dg0(rec.forward(U))

    peak = pos[np.argmax(sig)]
    check("the inclusion is found at the right place",
          np.linalg.norm(peak - np.array([0.45, 0.30])) < 0.25,
          f"peak at [{peak[0]:+.3f}, {peak[1]:+.3f}], truth [+0.450, +0.300]")
    check("the contrast has the right sign", sig.max() > 1.15,
          f"max sigma {sig.max():.3f}")
    check("the background is left alone",
          abs(np.median(sig) - 1.0) < 0.15, f"median {np.median(sig):.3f}")
    check("the transform is not blowing up at R = 3.5",
          rec.diagnostics["t_growth"] < GROWTH_WARN,
          f"t_growth {rec.diagnostics['t_growth']:.2f} < {GROWTH_WARN}")

    # a mirrored phantom must give a mirrored image, not the same one
    sig_m = phantom(pos, [{"shape": "circle", "center": [-0.45, 0.30],
                           "radius": 0.22, "value": 3.0}])
    rec_m = DbarSolver(solver, backCond=1.0, R=3.5, k_grid=32, z_grid=48)
    sig_mr = rec_m.to_dg0(rec_m.forward(cem_data(solver, sig_m)))
    peak_m = pos[np.argmax(sig_mr)]
    check("a mirrored phantom gives a mirrored reconstruction",
          peak_m[0] < -0.2 and peak_m[1] > 0.05,
          f"peak at [{peak_m[0]:+.3f}, {peak_m[1]:+.3f}], truth [-0.450, +0.300]")


def test_invariances():
    section("scaling invariances")
    # (a) background:  (sigma, z) -> (c sigma, z/c) is the CEM scaling law, so
    #     the D-bar image must scale by exactly c.
    s1 = cem_solver(z=1e-2)
    pos = s1.cell_centers()
    sig_true = phantom(pos, [{"shape": "circle", "center": [0.4, 0.0],
                              "radius": 0.25, "value": 2.5}])
    img1 = DbarSolver(s1, backCond=1.0, R=3.5, k_grid=32, z_grid=48,
                      clip=None).forward(cem_data(s1, sig_true)).x.array.copy()

    c = 2.5
    s2 = cem_solver(z=1e-2 / c)
    img2 = DbarSolver(s2, backCond=c, R=3.5, k_grid=32, z_grid=48,
                      clip=None).forward(cem_data(s2, c * sig_true)).x.array.copy()
    rel = np.abs(img2 - c * img1).max() / np.abs(c * img1).max()
    check("the reconstruction is equivariant under sigma -> c sigma",
          rel < 1e-6, f"max rel deviation {rel:.2e}")

    # (b) radius:  a half-size cell with half the contact impedance is the same
    #     unit-disk problem, so the image must agree after rescaling z -> z/r.
    r = 0.5
    s3 = cem_solver(radius=r, z=1e-2 * r)
    pos3 = s3.cell_centers()
    sig3 = phantom(pos3, [{"shape": "circle", "center": [0.4 * r, 0.0],
                           "radius": 0.25 * r, "value": 2.5}], radius=r)
    rec3 = DbarSolver(s3, backCond=1.0, R=3.5, k_grid=32, z_grid=48, clip=None)
    rec3.forward(cem_data(s3, sig3))
    rec1 = DbarSolver(s1, backCond=1.0, R=3.5, k_grid=32, z_grid=48, clip=None)
    rec1.forward(cem_data(s1, sig_true))
    d = np.abs(rec3.image - rec1.image).max() / np.ptp(rec1.image)
    check("the reconstruction is invariant under a change of domain radius",
          d < 0.05, f"max deviation {d:.3f} of the dynamic range")


def test_registry():
    section("registry integration")
    check("dbar is registered", "dbar" in A.available(), str(A.available()))
    p = A.resolve_params("dbar", {"R": 3.5}, {"sigma_min": 0.01, "sigma_max": 5.0})
    check("defaults merge with overrides",
          p["R"] == 3.5 and p["scattering"] == "exp" and p["k_grid"] == 64)

    solver = cem_solver()
    pos = solver.cell_centers()
    sig_true = phantom(pos, [{"shape": "circle", "center": [0.0, 0.45],
                              "radius": 0.22, "value": 0.3}])
    params = A.resolve_params("dbar", {"R": 3.5, "k_grid": 32, "z_grid": 48},
                              {"sigma_min": 0.01, "sigma_max": 5.0, "background": 1.0})
    ctx = {"dbar_reference_cache": {}}
    sigma, info = A.run("dbar", solver, cem_data(solver, sig_true), params, ctx)
    check("run() returns one value per cell",
          sigma.shape == (len(pos),), str(sigma.shape))
    check("run() reports the D-bar diagnostics",
          "diagnostics" in info and "t_growth" in info["diagnostics"],
          str(sorted(info.get("diagnostics", {}))))
    check("run() forward-simulates the reconstruction",
          info["U_reconstructed"].shape == np.asarray(solver.Inj).shape)
    check("a low-contrast inclusion is found",
          np.linalg.norm(pos[np.argmin(sigma)] - np.array([0.0, 0.45])) < 0.3,
          f"minimum at {np.round(pos[np.argmin(sigma)], 3)}")
    check("the reference solve is cached", len(ctx["dbar_reference_cache"]) == 1)


def test_guards():
    section("input guards")
    solver = cem_solver()
    try:
        DbarSolver(solver, scattering="nope")
        ok = False
    except ValueError:
        ok = True
    check("an unknown scattering variant is rejected", ok)

    try:
        DbarSolver(solver, R=-1)
        ok = False
    except ValueError:
        ok = True
    check("a non-positive truncation radius is rejected", ok)

    # a rank-deficient drive cannot define a DN map
    Inj = np.tile(current_method(L=16, l=1, method=2, value=1.0), (3, 1))
    try:
        orthonormalising_factor(Inj, np.full(16, 2 * np.pi / 16))
        ok = False
    except ValueError:
        ok = True
    check("rank-deficient drive patterns are rejected with a clear message", ok)

    rec = DbarSolver(solver, backCond=1.0, R=3.0, k_grid=32, z_grid=32)
    try:
        rec.forward(np.zeros((3, solver.L)))
        ok = False
    except ValueError:
        ok = True
    check("a wrong number of measurement patterns is rejected", ok)

    # too large an R must warn rather than quietly return noise
    pos = solver.cell_centers()
    sig_true = phantom(pos, [{"shape": "circle", "center": [0.4, 0.1],
                              "radius": 0.2, "value": 3.0}])
    U = cem_data(solver, sig_true)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec = DbarSolver(solver, backCond=1.0, R=8.0, k_grid=32, z_grid=24)
        rec.forward(U)
    check("an over-large R warns that the transform is blowing up",
          any("blowing up" in str(x.message) for x in w),
          f"t_growth {rec.diagnostics['t_growth']:.1f}, "
          f"messages {[str(x.message)[:40] for x in w]}")

    # t_cutoff must actually cap the transform
    rec = DbarSolver(solver, backCond=1.0, R=8.0, k_grid=32, z_grid=24, t_cutoff=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec.forward(U)
    check("t_cutoff caps the scattering transform",
          rec.diagnostics["t_max"] <= 5.0, f"|t|max {rec.diagnostics['t_max']:.3f}")


# ═════════════════════════════════ main ═════════════════════════════════════
def main():
    tests = [
        test_faddeev,
        test_cauchy_transform,
        test_nd_matrix,
        test_scattering_orientation,
        test_scattering_properties,
        test_end_to_end_continuum,
        test_contact_shift,
        test_cem_homogeneous,
        test_cem_inclusion,
        test_invariances,
        test_registry,
        test_guards,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            check(f"{t.__name__} raised", False, repr(exc))

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 68)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nFailures:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  - {name}   [{detail}]")
    print("=" * 68)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
