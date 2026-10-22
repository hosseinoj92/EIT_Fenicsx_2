"""Dimension-independent SI simulation, complex inversion, exports and plots."""
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.spatial import cKDTree
from .physical import (normalize_config, sample_materials, material_admittivity,
                       drive_patterns, EPSILON_0)


def _solver(cfg, mesh, frequency):
    from dolfinx.fem import Function
    from .eit_forward_fenicsx import EIT
    from mesh_tools import build_mesh
    path = build_mesh(mesh)
    bg = material_admittivity(cfg["background"], frequency)
    eps = (bg.imag/(2*np.pi*frequency*EPSILON_0) if frequency else
           cfg["background"].get("relative_permittivity", 0)+cfg["background"].get("delta_epsilon_r", 0))
    solver = EIT(mesh["n_electrodes"], drive_patterns(cfg),
                 cfg["contact"]["resistance_ohm_m2"], mesh_name=path,
                 dimension=mesh["dimension"], frequency_hz=frequency,
                 relative_permittivity=eps, thickness=cfg["thickness"],
                 contact_capacitance=cfg["contact"]["capacitance_f_m2"],
                 check_injection=False)  # redundant pair drives are intentional
    return solver, Function(solver.V_sigma)


def reconstruct(cfg, observed, log=print):
    """Joint sigma/epsilon inversion; no dense n_cells by n_cells normal matrix.

    The negative complex voltage Jacobian is split into real and imaginary
    residuals. A sparse facet prior and weak homogeneous-background ridge
    regularize each dimensionless parameter. Permittivity is unobservable at
    DC, where only conductivity is inverted. Truth is never sampled here.
    """
    from .regulariser import build_gradient_matrix
    r = cfg["reconstruction"]
    f = r["frequency_hz"]
    mesh = dict(cfg["mesh"], h_center=r["h_center"], h_edge=r["h_edge"])
    solver, sigma = _solver(cfg, mesh, f)
    n = solver.n_cells
    bg = material_admittivity(cfg["background"], f)
    omega_eps = 2*np.pi*f*EPSILON_0
    joint = bool(r.get("reconstruct_permittivity", True) and f > 0)
    k = 2 if joint else 1
    scales = [bg.real, max(bg.real/omega_eps, bg.imag/omega_eps, 1.0)] if joint else [bg.real]
    q0 = np.full((k, n), 1.0)
    if joint:
        q0[1] = bg.imag/omega_eps/scales[1]
    q = q0.copy()
    D = build_gradient_matrix(solver.omega)
    prior_scale = np.sqrt(r["regularization"]/max(D.shape[0], 1))
    ridge = np.sqrt(r["regularization"]*0.001/n)
    data_scale = max(np.linalg.norm(observed), 1e-30)
    history = []

    def evaluate(v):
        sigma.x.array[:] = scales[0]*v[0]
        if joint:
            solver.relative_permittivity.x.array[:] = scales[1]*v[1]
        u, U = solver.forward_solve(sigma)
        residual = (np.asarray(U)-observed).ravel()/data_scale
        prior = np.concatenate([prior_scale*(D@row) for row in v])
        delta = ridge*(v-q0).ravel()
        objective = 0.5*(np.vdot(residual, residual).real+prior@prior+delta@delta)
        return objective, residual, u

    objective, residual, u = evaluate(q)
    for it in range(int(r["iterations"])):
        log(f"Reconstruction {it+1}/{r['iterations']}: {n} cells, relative residual {np.linalg.norm(residual):.4g}")
        J = solver.calc_jacobian(sigma, u)/data_scale
        m, nd = J.shape[0], D.shape[0]
        Jr, Ji = J.real, J.imag

        def matvec(v):
            v = np.asarray(v).reshape(k, n)
            real = scales[0]*(Jr@v[0])
            imag = scales[0]*(Ji@v[0])
            if joint:
                real -= omega_eps*scales[1]*(Ji@v[1])
                imag += omega_eps*scales[1]*(Jr@v[1])
            return np.concatenate([real, imag,
                *[prior_scale*(D@row) for row in v], ridge*v.ravel()])

        def rmatvec(v):
            vr, vi = v[:m], v[m:2*m]
            ans = np.empty((k, n))
            ans[0] = scales[0]*(Jr.T@vr+Ji.T@vi)
            if joint:
                ans[1] = omega_eps*scales[1]*(-Ji.T@vr+Jr.T@vi)
            for field in range(k):
                ans[field] += prior_scale*(D.T@v[2*m+field*nd:2*m+(field+1)*nd])
            ans += ridge*v[2*m+k*nd:].reshape(k, n)
            return ans.ravel()

        A = LinearOperator((2*m+k*nd+k*n, k*n), matvec=matvec, rmatvec=rmatvec, dtype=float)
        rhs = np.concatenate([residual.real, residual.imag,
                              *[-prior_scale*(D@row) for row in q],
                              -ridge*(q-q0).ravel()])
        result = lsmr(A, rhs, atol=1e-5, btol=1e-5, maxiter=150)
        step = result[0].reshape(k, n)
        alpha = 1.0
        accepted = False
        for _ in range(10):
            trial = q+alpha*step
            trial[0] = np.clip(trial[0], r["sigma_min_s_m"]/scales[0], r["sigma_max_s_m"]/scales[0])
            if joint:
                trial[1] = np.maximum(trial[1], 0)
            obj_new, res_new, u_new = evaluate(trial)
            if obj_new < objective:
                q, objective, residual, u = trial, obj_new, res_new, u_new
                accepted = True
                break
            alpha /= 2
        history.append({"iteration": it+1, "objective": float(objective),
                        "relative_residual": float(np.linalg.norm(residual)),
                        "step_length": alpha if accepted else 0,
                        "lsmr_stop": int(result[1])})
        if not accepted:
            evaluate(q)  # restore the accepted material, not the last trial
            break
    evaluate(q)
    return solver, sigma.x.array.real.copy(), solver.relative_permittivity.x.array.real.copy(), history


def write_vtu(path, solver, cell_fields, point_fields=None):
    """Linear triangle/tetrahedron export, readable in ParaView/PyVista."""
    from xml.etree.ElementTree import Element, SubElement, ElementTree
    dim = solver.dimension
    points = solver.omega.geometry.x
    cells = np.asarray(solver.omega.geometry.dofmap).reshape(-1, dim+1)
    root = Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = SubElement(root, "UnstructuredGrid")
    piece = SubElement(grid, "Piece", NumberOfPoints=str(len(points)), NumberOfCells=str(len(cells)))

    def array(parent, name, data, kind="Float64", components=1):
        node = SubElement(parent, "DataArray", type=kind, Name=name,
                          NumberOfComponents=str(components), format="ascii")
        node.text = " ".join(str(v) for v in np.asarray(data).ravel())

    array(SubElement(piece, "Points"), "Points", points, components=3)
    conn = SubElement(piece, "Cells")
    array(conn, "connectivity", cells, "Int64")
    array(conn, "offsets", np.arange(1, len(cells)+1)*(dim+1), "Int64")
    array(conn, "types", np.full(len(cells), 5 if dim == 2 else 10), "UInt8")
    # DG0 dofs need not equal mesh-cell indices.
    dofs = np.array([solver.V_sigma.dofmap.cell_dofs(c)[0] for c in range(len(cells))])
    cd = SubElement(piece, "CellData")
    for name, data in cell_fields.items():
        array(cd, name, np.asarray(data)[dofs])
    if point_fields:
        # Geometry and CG1 function dofs can have different orderings too.
        indices = cKDTree(solver.V.tabulate_dof_coordinates()).query(points)[1]
        pd = SubElement(piece, "PointData")
        for name, data in point_fields.items():
            array(pd, name, np.asarray(data)[indices])
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def plot_fields(path, cfg, points, fields, title):
    """2D maps or three orthogonal volume sections with physical coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    dim, R, H = cfg["mesh"]["dimension"], cfg["mesh"]["radius"], cfg["mesh"]["height"]
    views = [(0, 1)] if dim == 2 else [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(len(fields), len(views), squeeze=False,
                             figsize=(5*len(views), 4*len(fields)), constrained_layout=True)
    tree = cKDTree(points)
    labels = ["x [m]", "y [m]", "z [m]"]
    for row, (label, values) in enumerate(fields.items()):
        norm = Normalize(float(np.min(values)), max(float(np.max(values)), float(np.min(values))+1e-12))
        for col, (a, b) in enumerate(views):
            x = np.linspace(-R, R, 180)
            y = np.linspace(0, H, 180) if b == 2 else x
            X, Y = np.meshgrid(x, y)
            p = np.zeros((X.size, dim))
            if dim == 3:
                p[:, 2] = H/2
            p[:, a], p[:, b] = X.ravel(), Y.ravel()
            vals = np.asarray(values)[tree.query(p)[1]].astype(float)
            vals[np.sum(p[:, :2]**2, axis=1) > R**2] = np.nan
            ax = axes[row, col]
            im = ax.imshow(vals.reshape(X.shape), origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                           cmap="viridis", norm=norm, aspect="equal")
            ax.set(xlabel=labels[a], ylabel=labels[b], title=f"{label} ({'xy' if b == 1 else 'xz' if a == 0 else 'yz'} section)")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(config, log=print):
    cfg = normalize_config(config)
    if cfg['reconstruction'].get('algorithms') is not None:
        from .volume_benchmark import run as run_benchmark
        return run_benchmark(cfg, log)
    out = Path(cfg["output_root"])/f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    frequencies, voltages, conductivities, permittivities = [], [], [], []
    rows, backgrounds = [], []
    for f in cfg["frequencies_hz"]:
        log(f"Solving {cfg['mesh']['dimension']}D at {f:g} Hz ...")
        solver, sigma = _solver(cfg, cfg["mesh"], f)
        points = solver.cell_centers()
        values, eps = sample_materials(cfg, points, f)
        solver.relative_permittivity.x.array[:] = eps
        sigma.x.array[:] = values
        u, U = solver.forward_solve(sigma)
        U = np.asarray(U)
        conductivities.append(values)
        permittivities.append(eps)
        frequencies.append(f)
        voltages.append(U)
        log(f"  {solver.n_cells} cells; max |U|={np.abs(U).max():.6g} V RMS; gauge={np.abs(U.sum(axis=1)).max():.2e} V")
        write_vtu(out/f"truth_{f:g}Hz.vtu", solver,
                  {"conductivity_S_per_m": values, "relative_permittivity": eps},
                  {"potential_real_V_pattern0": np.real(u[0]), "potential_imag_V_pattern0": np.imag(u[0])})
        for p in range(len(U)):
            for e in range(solver.L):
                rows.append([f, p, e+1, U[p, e].real, U[p, e].imag,
                             abs(U[p, e]), np.angle(U[p, e], deg=True)])
        # Uniform reference uses the background's own dielectric properties.
        bg = material_admittivity(cfg["background"], f)
        sigma.x.array[:] = bg.real
        solver.relative_permittivity.x.array[:] = bg.imag/(2*np.pi*f*EPSILON_0) if f else 0
        _, Ubg = solver.forward_solve(sigma)
        backgrounds.append(np.asarray(Ubg))
    from mesh_tools import electrode_centres
    np.savez_compressed(out/"measurements.npz", version=1, frequencies_hz=frequencies,
                        U_v=voltages, U_background_v=backgrounds, injection_a=solver.Inj,
                        electrode_centres_m=electrode_centres(cfg["mesh"]),
                        electrode_areas_m2=solver.electrode_areas,
                        points_m=points, conductivity_s_m=conductivities,
                        relative_permittivity=permittivities)
    import shutil
    shutil.copyfile(solver.mesh_name, out/"truth_mesh.msh")
    with (out/"measurements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_Hz", "pattern_index", "electrode_1based", "real_V_RMS",
                    "imag_V_RMS", "magnitude_V_RMS", "phase_deg"])
        w.writerows(rows)
    plot_fields(out/"truth.png", cfg, points,
                {"Conductivity [S/m]": conductivities[0], "Relative permittivity [1]": permittivities[0]},
                f"Truth at {frequencies[0]:g} Hz")
    plot_spectrum(out/"frequency_response.png", np.array(frequencies), np.array(voltages), solver.Inj)
    metrics = {}
    if cfg["reconstruction"]["enabled"]:
        r = cfg["reconstruction"]
        idx = frequencies.index(r["frequency_hz"])
        observed = np.array(voltages[idx], dtype=complex)
        rng = np.random.default_rng(r["seed"])
        # Independent I/Q noise with the requested standard deviation per channel.
        observed += r["noise_std_v"]*(rng.normal(size=observed.shape)+1j*rng.normal(size=observed.shape))
        rec_solver, rec, eps_rec, history = reconstruct(cfg, observed, log)
        rec_points = rec_solver.cell_centers()
        np.savez_compressed(out/"reconstruction.npz", points_m=rec_points,
                            conductivity_s_m=rec, relative_permittivity=eps_rec,
                            observed_v=observed, frequency_hz=r["frequency_hz"])
        write_vtu(out/"reconstruction.vtu", rec_solver,
                  {"conductivity_S_per_m": rec, "relative_permittivity": eps_rec})
        shutil.copyfile(rec_solver.mesh_name, out/"reconstruction_mesh.msh")
        plot_fields(out/"reconstruction.png", cfg, rec_points,
                    {"Conductivity [S/m]": rec, "Relative permittivity [1]": eps_rec},
                    f"Reconstruction at {r['frequency_hz']:g} Hz")
        # Uniform physical quadrature shared by truth and reconstruction.
        m = cfg["mesh"]
        ax = np.linspace(-m["radius"], m["radius"], 45)
        axes = [ax, ax] + ([np.linspace(0, m["height"], 45)] if m["dimension"] == 3 else [])
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, m["dimension"])
        grid = grid[np.sum(grid[:, :2]**2, axis=1) <= m["radius"]**2]
        truth, truth_eps = sample_materials(cfg, grid, r["frequency_hz"])
        nearest = cKDTree(rec_points).query(grid)[1]
        metrics = {"relative_L2_conductivity": float(np.linalg.norm(rec[nearest]-truth)/np.linalg.norm(truth)),
                   "relative_L2_permittivity": float(np.linalg.norm(eps_rec[nearest]-truth_eps)/max(np.linalg.norm(truth_eps), 1e-30)),
                   "history": history,
                   "permittivity_reconstructed": bool(r.get("reconstruct_permittivity", True) and r["frequency_hz"] > 0)}
        (out/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Results: {out.resolve()}")
    return out


def plot_spectrum(path, frequencies, voltages, injection):
    """Transfer impedance uses a passive electrode pair wherever available."""
    import matplotlib.pyplot as plt
    passive = np.flatnonzero(np.isclose(injection[0], 0, atol=1e-20))
    a, b = passive[:2] if len(passive) >= 2 else (0, 1)
    current = np.max(np.abs(injection[0]))
    z = (voltages[:, 0, a]-voltages[:, 0, b])/current
    order = np.argsort(frequencies)
    f, z = frequencies[order], z[order]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), constrained_layout=True)
    axes[0].plot(f, np.abs(z), "o-")
    axes[1].plot(f, np.rad2deg(np.unwrap(np.angle(z))), "o-")
    for ax in axes:
        if np.all(f > 0):
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Excitation frequency [Hz]")
    axes[0].set_ylabel("Transfer impedance magnitude [ohm]")
    axes[1].set_ylabel("Transfer impedance phase [degree]")
    fig.suptitle(f"Pattern 0, electrode difference {a+1} - {b+1}; RMS phasors")
    fig.savefig(path, dpi=150)
    plt.close(fig)
