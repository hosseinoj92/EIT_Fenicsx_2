#!/usr/bin/env python
"""
Validation suite for the EIT framework.

Physics and mathematics are checked against properties that must hold exactly
(or to a known order) independently of the implementation:

  * charge conservation and rank of every drive pattern
  * the CEM gauge  sum_i U_i = 0
  * reciprocity of the transfer impedance matrix
  * scaling law of the CEM under (sigma, z) -> (c sigma, z/c)
  * rotational equivariance on a homogeneous disk
  * mesh convergence of the forward solution
  * the adjoint Jacobian against finite differences (value AND sign)
  * the TV operator against the mesh facet count
  * the Gaussian prior factor against its defining identity L^T L = Gamma^-1
  * monotone decrease of the Gauss-Newton objective
  * recovery of a known inclusion, and the effect of regularisation strength

Run:  python tests/test_framework.py
"""

import os
import pathlib
import sys
import tempfile

os.environ.setdefault(
    "XDG_CACHE_HOME", str(pathlib.Path(tempfile.gettempdir()) / f"fenics_t_{os.getpid()}")
)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402
from dolfinx.fem import Function                                # noqa: E402

from mesh_tools import build_mesh, mesh_info, DEFAULT_MESH       # noqa: E402
from eitcore import EIT                                         # noqa: E402
from eitcore.utils import (                                     # noqa: E402
    current_method, default_n_patterns, validate_injection, DRIVE_NAMES,
)
from eitcore.phantoms import build_phantom                      # noqa: E402
from eitcore.noise import (                                     # noqa: E402
    add_measurement_noise, noise_std, gamma_inv, snr_db,
)
from eitcore.metrics import MetricEvaluator                     # noqa: E402
from eitcore.regulariser import (                               # noqa: E402
    build_smoothness_regulariser, build_prior, build_gradient_matrix,
)
from eitcore import algorithms as A                             # noqa: E402


# --------------------------------------------------------------------------- #
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


_solver_cache = {}


def make_solver(h_center=0.20, h_edge=0.14, L=16, z=1e-2, method=2, amp=1.0,
                **mesh_kw):
    key = (h_center, h_edge, L, z, method, amp, tuple(sorted(mesh_kw.items())))
    if key in _solver_cache:
        return _solver_cache[key]
    cfg = dict(n_electrodes=L, h_center=h_center, h_edge=h_edge,
               electrode_width_deg=12.0, n_per_electrode=8, n_per_gap=4, **mesh_kw)
    path = build_mesh(cfg)
    Inj = current_method(L=L, l=default_n_patterns(method, L), method=method, value=amp)
    s = EIT(L, Inj, np.full(L, z), backend="Scipy", mesh_name=path)
    _solver_cache[key] = s
    return s


def const_sigma(solver, value=1.0):
    f = Function(solver.V_sigma)
    f.x.array[:] = value
    return f


# ═══════════════════════════════ tests ══════════════════════════════════════
def test_drive_patterns():
    section("drive patterns")
    for m in sorted(DRIVE_NAMES):
        for L in (8, 16, 32):
            if m == 1 and L % 2:
                continue
            n = default_n_patterns(m, L)
            Inj = current_method(L=L, l=n, method=m, value=1.5)
            ok, msg = validate_injection(Inj)
            check(f"method {m} ({DRIVE_NAMES[m]}) L={L}: charge-conserving, "
                  f"full rank, no zero rows", ok, msg)
            check(f"method {m} L={L}: at most L-1 patterns", n <= L - 1,
                  f"n={n}")

    try:
        current_method(L=16, l=99, method=2)
        check("asking for too many patterns raises", False)
    except ValueError:
        check("asking for too many patterns raises", True)

    try:
        current_method(L=15, l=7, method=1)
        check("opposite drive rejects an odd electrode count", False)
    except ValueError as e:
        check("opposite drive rejects an odd electrode count",
              "even" in str(e).lower(), str(e))


def test_mesh():
    section("mesh generation")
    p = build_mesh({"radius": 1.0, "h_center": 0.18, "h_edge": 0.12})
    info = mesh_info(p)
    check("circular cross-section has area pi R^2",
          abs(info["area"] - np.pi) < 2e-2, f"area={info['area']:.5f}")
    check("mesh quality is usable", info["quality_min"] > 0.2,
          f"q_min={info['quality_min']:.3f}, q_mean={info['quality_mean']:.3f}")

    p2 = build_mesh({"radius": 0.5, "h_center": 0.09, "h_edge": 0.06})
    i2 = mesh_info(p2)
    check("a different radius scales the area as R^2",
          abs(i2["area"] - np.pi * 0.25) < 1e-2, f"area={i2['area']:.5f}")

    # the cell is a cylinder: no non-circular cross-section may slip through
    for bad, why in [
        ({"shape": "ellipse"}, "shape"),
        ({"semi_axis_x": 1.0}, "semi_axis_x"),
        ({"vertices": [[0, 0], [1, 0], [0, 1]]}, "vertices"),
    ]:
        try:
            build_mesh(bad)
            check(f"non-circular geometry key '{why}' is rejected", False)
        except ValueError as e:
            check(f"non-circular geometry key '{why}' is rejected",
                  why in str(e))

    for bad, why in [
        ({"electrode_width_deg": 40.0}, "electrodes would overlap"),
        ({"n_electrodes": 2}, "too few electrodes"),
        ({"h_center": -0.1}, "negative element size"),
        ({"h_edge": 5.0}, "element larger than the cell"),
    ]:
        try:
            build_mesh(bad)
            check(f"invalid meshing parameter rejected ({why})", False)
        except ValueError:
            check(f"invalid meshing parameter rejected ({why})", True)

    s = make_solver()
    el = s.electrode_lengths
    expected = np.deg2rad(12.0)
    check("electrode arc length matches the requested angular width",
          abs(el.mean() - expected) < 5e-3 * expected,
          f"mean={el.mean():.6f} expected={expected:.6f}")
    check("all electrodes have the same length on a symmetric disk",
          el.ptp() / el.mean() < 1e-10, f"spread={el.ptp():.2e}")

    fine = mesh_info(build_mesh({"h_center": 0.10, "h_edge": 0.07}))
    check("smaller target size gives more cells",
          fine["n_cells"] > info["n_cells"],
          f"{info['n_cells']} -> {fine['n_cells']}")


def test_forward_gauge_and_symmetry():
    section("forward solver: gauge, symmetry, scaling")
    s = make_solver()
    sig = const_sigma(s, 1.0)
    _, U = s.forward_solve(sig)
    U = np.asarray(U)

    check("CEM gauge sum_i U_i = 0", np.abs(U.sum(axis=1)).max() < 1e-12,
          f"max={np.abs(U.sum(axis=1)).max():.2e}")

    # rotational equivariance: on a homogeneous disk with adjacent drive,
    # pattern k is pattern 0 rotated by k electrodes
    err = max(np.abs(U[k] - np.roll(U[0], k)).max() for k in range(U.shape[0]))
    check("homogeneous disk is rotationally equivariant",
          err < 0.02 * np.abs(U).max(), f"max dev={err:.2e}, |U|max={np.abs(U).max():.3f}")

    # CEM scaling law: (sigma, z) -> (c sigma, z/c)  =>  U -> U/c
    c = 3.0
    s2 = make_solver(z=1e-2 / c)
    _, U2 = s2.forward_solve(const_sigma(s2, c))
    U2 = np.asarray(U2)
    rel = np.abs(U2 - U / c).max() / np.abs(U / c).max()
    check("CEM scaling law U(c*sigma, z/c) = U(sigma, z)/c", rel < 1e-9,
          f"rel err={rel:.2e}")

    # doubling the current doubles the voltages (linearity in I)
    s3 = make_solver(amp=2.0)
    _, U3 = np.asarray(s3.forward_solve(const_sigma(s3, 1.0))[0]), None
    _, U3 = s3.forward_solve(const_sigma(s3, 1.0))
    U3 = np.asarray(U3)
    rel = np.abs(U3 - 2 * U).max() / np.abs(2 * U).max()
    check("forward map is linear in the injected current", rel < 1e-9,
          f"rel err={rel:.2e}")


def test_reciprocity():
    section("reciprocity")
    s = make_solver()
    L = s.L
    # transfer impedance with the charge-conserving unit basis e_i - 1/L
    I = np.eye(L) - 1.0 / L
    rng = np.random.default_rng(0)
    sig = Function(s.V_sigma)
    sig.x.array[:] = 1.0 + 0.5 * rng.random(s.n_cells)
    _, U = s.forward_solve(sig, I)
    Z = np.asarray(U)
    asym = np.abs(Z - Z.T).max() / np.abs(Z).max()
    check("transfer impedance matrix is symmetric (reciprocity)",
          asym < 1e-9, f"max asymmetry={asym:.2e}")


def test_mesh_convergence():
    section("mesh convergence of the forward solution")
    ref = make_solver(h_center=0.05, h_edge=0.035)
    U_ref = np.asarray(ref.forward_solve(const_sigma(ref, 1.0))[1])

    hs = (0.24, 0.16, 0.10)
    errs = []
    for h in hs:
        s = make_solver(h_center=h, h_edge=h * 0.7)
        U = np.asarray(s.forward_solve(const_sigma(s, 1.0))[1])
        errs.append(np.abs(U - U_ref).max() / np.abs(U_ref).max())
    check("forward error decreases monotonically under refinement",
          errs[0] > errs[1] > errs[2],
          " > ".join(f"{e:.2e}" for e in errs))

    # Observed order of convergence.  The CEM solution has corner singularities
    # at the electrode edges, so the electrode voltages converge at roughly
    # first order rather than the O(h^2) of a smooth P1 problem; anything
    # clearly below first order would mean the discretisation is wrong.
    rates = [
        np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    check("observed convergence rate is at least first order",
          min(rates) > 0.8, "rates = " + ", ".join(f"{r:.2f}" for r in rates))


def test_jacobian():
    section("adjoint Jacobian vs finite differences")
    s = make_solver()
    rng = np.random.default_rng(1)
    sig = Function(s.V_sigma)
    sig.x.array[:] = 1.0 + 0.3 * rng.random(s.n_cells)

    u, U = s.forward_solve(sig)
    U0 = np.asarray(U).flatten()
    J = s.calc_jacobian(sig, u)

    check("Jacobian shape is (n_patterns*L, n_cells)",
          J.shape == (s.Inj.shape[0] * s.L, s.n_cells), str(J.shape))

    eps = 1e-6
    ratios, corrs = [], []
    for c in rng.choice(s.n_cells, 8, replace=False):
        sp = Function(s.V_sigma)
        sp.x.array[:] = sig.x.array
        sp.x.array[c] += eps
        Up = np.asarray(s.forward_solve(sp)[1]).flatten()
        fd = (Up - U0) / eps
        jc = J[:, c]
        ratios.append(float(np.dot(fd, -jc) / np.dot(jc, jc)))
        corrs.append(float(np.dot(fd, -jc) / np.linalg.norm(fd) / np.linalg.norm(jc)))

    check("cell-wise Jacobian matches finite differences in magnitude",
          max(abs(r - 1.0) for r in ratios) < 1e-3,
          f"ratios in [{min(ratios):.6f}, {max(ratios):.6f}]")
    check("cell-wise Jacobian matches finite differences in direction",
          min(corrs) > 1 - 1e-8, f"min corr={min(corrs):.10f}")
    check("Jacobian sign convention is J = -dU/dsigma (as documented)",
          np.mean(ratios) > 0, f"mean ratio={np.mean(ratios):.6f}")

    # directional derivative over a smooth perturbation, not just single cells
    d = rng.standard_normal(s.n_cells)
    sp = Function(s.V_sigma)
    sp.x.array[:] = sig.x.array + eps * d
    fd = (np.asarray(s.forward_solve(sp)[1]).flatten() - U0) / eps
    pred = -(J @ d)
    rel = np.linalg.norm(fd - pred) / np.linalg.norm(fd)
    check("directional derivative matches over a random direction", rel < 1e-4,
          f"rel err={rel:.2e}")


def test_tv_operator():
    section("total-variation operator")
    from eitcore.gauss_newton import GaussNewtonSolverTV

    s = make_solver()
    tv = GaussNewtonSolverTV(s, num_steps=1)
    Ltv = tv.Ltv

    check("TV matrix has n_cells columns", Ltv.shape[1] == s.n_cells,
          str(Ltv.shape))
    check("each row has exactly one +1 and one -1",
          np.allclose(np.asarray(Ltv.sum(axis=1)).ravel(), 0.0)
          and np.allclose(np.abs(Ltv).sum(axis=1), 2.0))

    # every interior facet must appear exactly once
    A_ = Ltv.toarray()
    pairs = {tuple(sorted(np.nonzero(r)[0])) for r in A_}
    check("interior facets are not duplicated", len(pairs) == A_.shape[0],
          f"{A_.shape[0]} rows, {len(pairs)} unique facets")

    # a constant field has zero total variation
    check("TV of a constant field is zero",
          np.abs(Ltv @ np.ones(s.n_cells)).max() < 1e-12)


def test_priors():
    section("prior operators")
    s = make_solver(h_center=0.35, h_edge=0.30)  # small mesh: dense prior is O(n^3)
    n = s.n_cells

    L = build_smoothness_regulariser(s.omega, corrlength=0.3, std=0.2)
    check("Gaussian prior factor has the right shape", L.shape == (n, n),
          f"{L.shape}, n={n}")

    # verify L^T L = Gamma^-1 by rebuilding Gamma and checking the product
    g = np.array(s.cell_centers())
    sq = np.einsum("ij,ij->i", g, g)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * g @ g.T, 0.0)
    Gamma = (0.2**2 - 1e-7) * np.exp(-d2 / (2 * 0.3**2))
    Gamma[np.diag_indices(n)] += 1e-7
    err = np.abs((L.T @ L) @ Gamma - np.eye(n)).max()
    check("Gaussian prior satisfies L^T L = Gamma^-1", err < 1e-6,
          f"max |L^T L Gamma - I| = {err:.2e}")

    D = build_gradient_matrix(s.omega)
    check("Laplacian prior annihilates constants",
          np.abs(D @ np.ones(n)).max() < 1e-12)
    R = build_prior(s.omega, "laplacian", eps=1e-4)
    check("Laplacian prior is symmetric positive definite",
          np.allclose(R.toarray(), R.toarray().T)
          and np.linalg.eigvalsh(R.toarray()).min() > 0,
          f"min eig={np.linalg.eigvalsh(R.toarray()).min():.2e}")
    check("Tikhonov prior is the identity",
          np.allclose(build_prior(s.omega, "tikhonov").toarray(), np.eye(n)))


def test_phantoms():
    section("phantoms")
    x = np.linspace(-0.9, 0.9, 60)
    X, Y = np.meshgrid(x, x)
    X, Y = X.ravel(), Y.ravel()

    cfg = {"background": 1.0,
           "inclusions": [{"shape": "circle", "center": [0.4, 0.0],
                           "radius": 0.2, "value": 3.0}]}
    sig = build_phantom(cfg, X, Y)
    inside = (X - 0.4) ** 2 + Y**2 <= 0.2**2
    check("circle inclusion takes its value inside",
          np.allclose(sig[inside], 3.0))
    check("background is untouched outside", np.allclose(sig[~inside], 1.0))

    # area of the recovered inclusion ~ pi r^2
    cell = (x[1] - x[0]) ** 2
    check("inclusion area is right", abs(inside.sum() * cell - np.pi * 0.04) < 0.01,
          f"{inside.sum() * cell:.4f} vs {np.pi * 0.04:.4f}")

    cfg2 = {"background": 1.0,
            "inclusions": [{"shape": "circle", "center": [0.0, 0.0], "radius": 0.5,
                            "profile": {"type": "linear", "center_value": 4.0,
                                        "edge_value": 1.0}}]}
    # sample exactly at the centre, at mid-radius and at the rim so the
    # expected values are known in closed form
    probe = np.array([0.0, 0.25, 0.5])
    vals = build_phantom(cfg2, probe, np.zeros_like(probe))
    check("linear radial profile hits center_value at r=0",
          abs(vals[0] - 4.0) < 1e-12, f"{vals[0]:.6f}")
    check("linear radial profile hits edge_value at r=R",
          abs(vals[2] - 1.0) < 1e-12, f"{vals[2]:.6f}")
    check("linear radial profile interpolates linearly in between",
          abs(vals[1] - 2.5) < 1e-12, f"{vals[1]:.6f}")

    sig2 = build_phantom(cfg2, X, Y)
    r = np.hypot(X, Y)
    check("linear radial profile is monotone in r",
          np.corrcoef(r[r <= 0.5], sig2[r <= 0.5])[0, 1] < -0.99)

    try:
        build_phantom({"background": 1.0,
                       "inclusions": [{"shape": "circle", "center": [0, 0],
                                       "radius": 0.3, "value": -1.0}]}, X, Y)
        check("negative conductivity is rejected", False)
    except ValueError:
        check("negative conductivity is rejected", True)

    # reproducibility of random placement
    rc = {"background": 1.0, "seed": 7,
          "random": {"n": 2, "semi_axes_range": (0.15, 0.25)}}
    a = build_phantom(rc, X, Y)
    b = build_phantom(rc, X, Y)
    check("random phantoms are reproducible from the seed", np.array_equal(a, b))


def test_noise():
    section("noise models")
    rng = np.random.default_rng(0)
    U = rng.standard_normal((15, 16))

    std = noise_std(U, "relative_to_max", 1.0)
    check("relative_to_max gives a constant noise floor",
          np.allclose(std, std.flat[0]) and
          abs(std.flat[0] - 0.01 * np.abs(U).max()) < 1e-12)

    std_pm = noise_std(U, "relative_per_measurement", 1.0)
    check("relative_per_measurement scales with |U| (and so vanishes at zeros)",
          std_pm.min() < 0.1 * std_pm.max())

    Un, std = add_measurement_noise(U, "relative_to_max", 1.0, seed=3)
    Un2, _ = add_measurement_noise(U, "relative_to_max", 1.0, seed=3)
    check("noise is reproducible from the seed", np.array_equal(Un, Un2))

    emp = np.std(Un - U)
    check("empirical noise std matches the model",
          abs(emp - std.flat[0]) < 0.15 * std.flat[0],
          f"empirical={emp:.5f} model={std.flat[0]:.5f}")

    Un0, std0 = add_measurement_noise(U, "none", 0.0, seed=3)
    check("zero noise level leaves the data untouched", np.array_equal(Un0, U))

    check("SNR decreases as the noise level rises",
          snr_db(U, add_measurement_noise(U, "relative_to_max", 0.5, seed=1)[0])
          > snr_db(U, add_measurement_noise(U, "relative_to_max", 5.0, seed=1)[0]))

    w = gamma_inv(noise_std(U, "relative_to_max", 1.0))
    check("gamma_inv is ~1/variance, not dominated by a fixed floor",
          abs(w.mean() * (0.01 * np.abs(U).max()) ** 2 - 1.0) < 0.01,
          f"w*var = {w.mean() * (0.01 * np.abs(U).max()) ** 2:.4f}")


def test_metrics():
    section("metrics")
    n = 400
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(n, 2))
    pts = pts[np.linalg.norm(pts, axis=1) < 0.98]
    gt = np.where(np.linalg.norm(pts - [0.4, 0.0], axis=1) < 0.25, 3.0, 1.0)

    ev = MetricEvaluator(pts, gt, background=1.0, resolution=120, radius=1.0)

    perfect = ev.evaluate(pts, gt)
    check("a perfect reconstruction scores zero error",
          perfect["rel_L1"] < 1e-12 and perfect["rel_L2"] < 1e-12)
    check("a perfect reconstruction scores dice 1", abs(perfect["dice"] - 1) < 1e-12)
    check("a perfect reconstruction scores correlation 1",
          abs(perfect["correlation"] - 1) < 1e-9)
    check("a perfect reconstruction scores dynamic range 1",
          abs(perfect["dynamic_range"] - 1) < 1e-12)

    flat = ev.evaluate(pts, np.ones_like(gt))
    check("a flat guess scores worse than the truth",
          flat["rel_L2"] > perfect["rel_L2"] and flat["dice"] < 0.5,
          f"rel_L2={flat['rel_L2']:.3f} dice={flat['dice']:.3f}")

    # mesh independence: the same field sampled on two different point sets
    # must give (nearly) the same metrics
    pts2 = rng.uniform(-1, 1, size=(3 * n, 2))
    pts2 = pts2[np.linalg.norm(pts2, axis=1) < 0.98]
    rec2 = np.where(np.linalg.norm(pts2 - [0.4, 0.0], axis=1) < 0.25, 3.0, 1.0)
    m1 = ev.evaluate(pts, gt)["rel_L1"]
    m2 = ev.evaluate(pts2, rec2)["rel_L1"]
    check("metrics are (nearly) independent of the sampling mesh",
          abs(m1 - m2) < 0.05, f"{m1:.4f} vs {m2:.4f}")


def test_reconstruction():
    section("reconstruction: recovery and regularisation behaviour")
    s = make_solver(h_center=0.22, h_edge=0.16)
    pts = s.cell_centers()

    phantom = {"background": 1.0,
               "inclusions": [{"shape": "circle", "center": [0.45, 0.0],
                               "radius": 0.25, "value": 3.0}]}
    sigma_true = build_phantom(phantom, pts[:, 0], pts[:, 1])

    f = Function(s.V_sigma)
    f.x.array[:] = sigma_true
    U_clean = np.asarray(s.forward_solve(f)[1])
    U, std = add_measurement_noise(U_clean, "relative_to_max", 0.5, seed=0)
    W = gamma_inv(std)

    ev = MetricEvaluator(pts, sigma_true, background=1.0, resolution=140,
                         radius=1.0)
    ctx = {"GammaInv": W, "prior_cache": {}}
    common = {"sigma_min": 0.01, "sigma_max": 5.0, "background": 1.0,
              "device": "cpu"}

    flat_err = ev.evaluate(pts, np.ones_like(sigma_true))["rel_L2_roi"]

    for algo, over, under in [("gn", {"lambda": 100.0}, {"lambda": 1e-4}),
                              ("tv", {"lambda": 3000.0}, {"lambda": 1e-3})]:
        p_good = A.resolve_params(algo, {}, common)
        sig, info = A.run(algo, s, U, p_good, ctx)
        m = ev.evaluate(pts, sig)

        check(f"{algo}: beats a flat background",
              m["rel_L2_roi"] < flat_err,
              f"{m['rel_L2_roi']:.4f} vs flat {flat_err:.4f}")
        check(f"{algo}: locates the inclusion (dice > 0.3)", m["dice"] > 0.3,
              f"dice={m['dice']:.3f}")
        peak = pts[np.argmax(sig)]
        check(f"{algo}: peak conductivity is near the true inclusion",
              np.hypot(peak[0] - 0.45, peak[1]) < 0.4,
              f"peak at {np.round(peak, 2).tolist()}")
        check(f"{algo}: reconstruction stays inside the clip range",
              sig.min() >= common["sigma_min"] - 1e-9
              and sig.max() <= common["sigma_max"] + 1e-9,
              f"[{sig.min():.4f}, {sig.max():.4f}]")

        hist = info["history"]
        if hist:
            objs = [h["objective"] for h in hist]
            check(f"{algo}: objective never increases (line search is consistent)",
                  all(b <= a + 1e-9 * abs(a) for a, b in zip(objs, objs[1:])),
                  f"{objs[0]:.4g} -> {objs[-1]:.4g}")

        # heavy regularisation must smooth towards the background
        sig_over, _ = A.run(algo, s, U, A.resolve_params(algo, over, common), ctx)
        sig_under, _ = A.run(algo, s, U, A.resolve_params(algo, under, common), ctx)
        check(f"{algo}: large lambda gives a smoother image than small lambda",
              np.std(sig_over) < np.std(sig_under),
              f"std {np.std(sig_over):.4f} vs {np.std(sig_under):.4f}")

    # the one-step method must at least put the inclusion in the right half
    sig_lin, _ = A.run("linear", s, U, A.resolve_params("linear", {}, common), ctx)
    check("linear: peak lies in the correct half of the domain",
          pts[np.argmax(sig_lin)][0] > 0, f"x={pts[np.argmax(sig_lin)][0]:.3f}")


def test_lambda_scaling():
    section("dimensionless lambda")
    from eitcore.gauss_newton import GaussNewtonSolver

    phantom = {"background": 1.0,
               "inclusions": [{"shape": "circle", "center": [0.45, 0.0],
                               "radius": 0.25, "value": 3.0}]}
    common = {"sigma_min": 0.01, "sigma_max": 5.0, "background": 1.0,
              "device": "cpu"}

    stds = {}
    for amp in (1.0, 10.0):
        s = make_solver(h_center=0.24, h_edge=0.18, amp=amp)
        pts = s.cell_centers()
        f = Function(s.V_sigma)
        f.x.array[:] = build_phantom(phantom, pts[:, 0], pts[:, 1])
        U_clean = np.asarray(s.forward_solve(f)[1])
        U, std = add_measurement_noise(U_clean, "relative_to_max", 0.5, seed=0)
        ctx = {"GammaInv": gamma_inv(std), "prior_cache": {}}
        sig, _ = A.run("gn", s, U, A.resolve_params("gn", {"lambda": 0.1}, common), ctx)
        stds[amp] = float(np.std(sig))

    rel = abs(stds[1.0] - stds[10.0]) / max(stds[1.0], 1e-12)
    check("the same lambda behaves the same at 1x and 10x drive amplitude",
          rel < 0.25, f"std {stds[1.0]:.4f} vs {stds[10.0]:.4f} (rel {rel:.3f})")


def test_io_roundtrip():
    section("ground-truth bundle round trip")
    from eitcore import gt_io

    tmp = pathlib.Path(tempfile.mkdtemp())
    pts = np.random.default_rng(0).uniform(-1, 1, size=(50, 2))
    sig = np.random.default_rng(1).uniform(0.5, 2.0, size=50)
    U = np.random.default_rng(2).standard_normal((5, 16))
    Inj = current_method(L=16, l=5, method=2)

    gt_io.save_ground_truth(
        tmp / "case", {"name": "case", "mesh": {"radius": 1.0}},
        {"background": 1.0, "inclusions": []}, pts, sig, U, Inj,
        np.full(16, 1e-2), 16, 1.0,
        mesh_path=build_mesh({"h_center": 0.4, "h_edge": 0.35}),
    )
    gt = gt_io.load_ground_truth(tmp / "case")

    check("sigma survives the round trip", np.allclose(gt.sigma_true, sig))
    check("points survive the round trip", np.allclose(gt.points, pts))
    check("measurements survive the round trip", np.allclose(gt.U_clean, U))
    check("injection matrix survives the round trip", np.allclose(gt.Inj, Inj))
    check("bundle carries its own mesh",
          pathlib.Path(gt.truth_mesh).exists())
    check("cross-section radius is recovered", gt.radius == 1.0, str(gt.radius))


def test_sweep_naming():
    section("sweep expansion and run naming")
    import simulate

    cfg = {
        "noise": {"model": "relative_to_max", "level_percent": [0.5, 2.0], "seed": 0},
        "recon_mesh": [{"label": "coarse", "h_center": 0.2}],
        "algorithms": {"gn": {"lambda": [0.1, 1.0], "max_iter": 12},
                       "tv": {"lambda": [5.0], "max_iter": 10}},
    }
    runs, varying = simulate.build_run_matrix(cfg)
    check("cartesian product has the right size", len(runs) == 2 * (2 + 1),
          f"{len(runs)} runs")
    check("only genuinely varying axes are detected",
          varying == {"algorithm", "lambda", "noise.level_percent"},
          str(sorted(varying)))
    check("max_iter differing between algorithms is not a sweep axis",
          "max_iter" not in varying)
    ids = [r["id"] for r in runs]
    check("run ids are unique", len(set(ids)) == len(ids))
    check("run ids name the varying parameters",
          all("lambda=" in i and "level_percent=" in i for i in ids),
          ids[0])

    single = {"noise": {"level_percent": 1.0, "seed": 0},
              "recon_mesh": [{"label": "m", "h_center": 0.2}],
              "algorithms": {"gn": {"lambda": 0.1}}}
    runs2, varying2 = simulate.build_run_matrix(single)
    check("a single-run config gives one run named after the algorithm only",
          len(runs2) == 1 and runs2[0]["id"] == "gn" and not varying2,
          runs2[0]["id"])

    m = {"noise": {"level_percent": 1.0, "seed": 0},
         "recon_mesh": [{"label": "coarse", "h_center": 0.2},
                        {"label": "fine", "h_center": 0.1}],
         "algorithms": {"gn": {"lambda": 0.1}}}
    runs3, varying3 = simulate.build_run_matrix(m)
    check("meshes sweep too", len(runs3) == 2 and "mesh" in varying3,
          str([r["id"] for r in runs3]))


# ═════════════════════════════════ main ═════════════════════════════════════
def main():
    tests = [
        test_drive_patterns,
        test_mesh,
        test_forward_gauge_and_symmetry,
        test_reciprocity,
        test_mesh_convergence,
        test_jacobian,
        test_tv_operator,
        test_priors,
        test_phantoms,
        test_noise,
        test_metrics,
        test_reconstruction,
        test_lambda_scaling,
        test_io_roundtrip,
        test_sweep_naming,
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
