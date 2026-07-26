#!/usr/bin/env python
# make_ground_truth.py

"""
STEP 1 of 2 - create a synthetic EIT ground truth.

Everything you can change lives in the CONFIG dictionary below.  Running this
script writes a self-contained bundle to

    GT_simulated_result/<name>/

which simulate.py (step 2) then loads and tries to reconstruct.

The test cell is a cylinder, so the modelled cross-section is a circle.

Why ground truth and reconstruction are separate scripts
--------------------------------------------------------
The forward data is generated here on a deliberately *fine* mesh; the
reconstruction in simulate.py runs on a different, coarser one.  Simulating and
inverting on the same discretisation is the classic "inverse crime": the
discretisation error cancels exactly, every algorithm looks better than it is,
and - worst of all for a benchmarking tool - the *ranking between* algorithms
shifts.  Keeping the two meshes apart is the whole reason the truth is computed
once and stored, rather than being regenerated inside the reconstruction script.

    python make_ground_truth.py
    python make_ground_truth.py --name my_case --preset three_inclusions
"""

import argparse
import json
import os
import pathlib
import sys
import time

# ═════════════════════════════════ CONFIG ═══════════════════════════════════
CONFIG = {
    # ── identity & output ────────────────────────────────────────────────
    "name": "two_inclusions_16e",       # bundle folder name
    "output_root": "GT_simulated_result",
    "overwrite": True,
    "random_seed": 16,

    # ── truth mesh (circular cross-section) ──────────────────────────────
    # Keep this clearly FINER than the reconstruction mesh in simulate.py.
    # A truth/reconstruction cell-count ratio of 4x or more is a good target.
    "mesh": {
        # -- physical cell: simulate.py inherits these and must not change them
        "radius": 1.0,                  # cross-section radius
        "n_electrodes": 16,             # equally spaced around the rim
        "electrode_width_deg": 12.0,    # angular width of ONE electrode
        "electrode_offset_deg": 0.0,    # rotate the whole electrode ring

        # -- discretisation: simulate.py is free to use different values
        "n_per_electrode": 10,          # boundary nodes across one electrode
        "n_per_gap": 5,                 # boundary nodes across one gap
        "h_center": 0.055,              # target element size at the centre
        "h_edge": 0.030,                # target element size at the rim
        "grading": 1.0,                 # size(r) = h_c + (h_e-h_c)(r/R)^grading
        "algorithm": 6,                 # gmsh 2-D meshing algorithm
        "optimize": True,               # Netgen quality optimiser
    },

    # ── electrodes & current drive ───────────────────────────────────────
    "L": 16,                            # must match mesh.n_electrodes
    "contact_impedance": 5e-3,          # z per electrode; scalar or list of L
    "drive": {
        "method": 2,                    # 1 opposite | 2 adjacent | 3 one-vs-all
                                        # 4 trigonometric | 5 all-vs-one
        "n_patterns": None,             # None -> the canonical maximum
        "amplitude": 1.0,               # injected current
    },

    # ── phantom (the ground truth itself) ────────────────────────────────
    # Either name a preset, give an explicit inclusion list, or ask for random
    # placement.  See eitcore/phantoms.py for every shape and profile.
    "phantom": {
        "background": 1.0,
        "inclusions": [
            {"shape": "circle", "center": [0.42, 0.34],
             "radius": 0.22, "value": 3.0},
            {"shape": "ellipse", "center": [-0.40, -0.28],
             "semi_axes": [0.28, 0.17], "angle_deg": 35.0, "value": 0.2},
        ],

        # --- alternatives, uncomment to use ------------------------------
        # "preset": "three_inclusions",        # see phantoms.PRESETS
        # "random": {                          # random placement
        #     "n_range": (1, 3),
        #     "shape": "ellipse",
        #     "semi_axes_range": (0.15, 0.30),
        #     "value_low_range": (0.05, 0.35),
        #     "value_high_range": (2.0, 3.5),
        #     "boundary_margin": 0.10,
        #     "pair_margin": 0.08,
        #     "regions": ["random"],           # UL UR LL LR center random
        # },
        # "seed": 16,
    },

    # ── preview figure ───────────────────────────────────────────────────
    "plot": {
        "enabled": True,
        "cmap": "turbo",
        "vmin": None,                   # None -> auto from the phantom
        "vmax": None,
        "show_mesh": False,
        "show_electrodes": True,
        "dpi": 160,
    },
}
# ════════════════════════════════════════════════════════════════════════════


def _setup_environment():
    """FFCX cache isolation and thread pinning; must precede the dolfinx import."""
    import tempfile

    cache = pathlib.Path(tempfile.gettempdir()) / f"fenics_cache_{os.getpid()}"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("FENICS_CACHE_DIR", str(cache))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")


_setup_environment()

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dolfinx.fem import Function                                # noqa: E402

from mesh_tools import build_mesh, mesh_info, electrode_spans    # noqa: E402
from eitcore import EIT, gt_io                                   # noqa: E402
from eitcore.phantoms import (                                   # noqa: E402
    build_phantom, resolve_phantom, describe_phantom,
)
from eitcore.utils import (                                      # noqa: E402
    current_method, default_n_patterns, validate_injection, DRIVE_NAMES,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
def generate(cfg):
    cfg = json.loads(json.dumps(cfg, default=gt_io._json_default))  # deep copy
    np.random.seed(cfg["random_seed"])

    out_dir = pathlib.Path(cfg["output_root"]) / cfg["name"]
    if out_dir.exists() and not cfg.get("overwrite", True):
        raise FileExistsError(f"'{out_dir}' already exists and overwrite=False.")

    # ── 1. mesh ──────────────────────────────────────────────────────────
    mesh_cfg = dict(cfg["mesh"])
    L = int(cfg["L"])
    if mesh_cfg.get("n_electrodes", L) != L:
        raise ValueError(
            f"CONFIG['L']={L} does not match mesh.n_electrodes="
            f"{mesh_cfg['n_electrodes']}"
        )
    mesh_cfg["n_electrodes"] = L

    log("building the truth mesh ...")
    mesh_path = build_mesh(mesh_cfg, out_dir="data/meshes")
    info = mesh_info(mesh_path)
    log(
        f"truth mesh: {info['n_cells']} cells, {info['n_nodes']} nodes, "
        f"h in [{info['h_min']:.3f}, {info['h_max']:.3f}], "
        f"quality min/mean {info['quality_min']:.2f}/{info['quality_mean']:.2f}"
    )

    # ── 2. drive pattern ─────────────────────────────────────────────────
    drive = cfg["drive"]
    n_pat = drive.get("n_patterns") or default_n_patterns(drive["method"], L)
    Inj = current_method(L=L, l=n_pat, method=drive["method"],
                         value=drive["amplitude"])
    ok, msg = validate_injection(Inj)
    log(f"drive: method {drive['method']} ({DRIVE_NAMES[drive['method']]}), "
        f"{n_pat} patterns, amplitude {drive['amplitude']}  "
        f"[{'ok' if ok else msg}]")

    # ── 3. forward solver ────────────────────────────────────────────────
    z = cfg["contact_impedance"]
    z = np.full(L, float(z)) if np.isscalar(z) else np.asarray(z, dtype=float)

    solver = EIT(L, Inj, z, backend="Scipy", mesh_name=mesh_path)
    log(f"solver: {solver}")

    # ── 4. phantom on the truth mesh ─────────────────────────────────────
    radius = mesh_cfg["radius"]
    phantom_cfg = resolve_phantom(cfg["phantom"], domain_radius=radius)

    points = solver.cell_centers()
    sigma_true = build_phantom(phantom_cfg, points[:, 0], points[:, 1],
                               domain_radius=radius)
    log(f"phantom: {describe_phantom(phantom_cfg)}")
    log(f"sigma_true in [{sigma_true.min():.4g}, {sigma_true.max():.4g}], "
        f"contrast {sigma_true.max() / max(sigma_true.min(), 1e-12):.1f}x")

    # ── 5. forward solve (this is the data) ──────────────────────────────
    sig = Function(solver.V_sigma)
    sig.x.array[:] = sigma_true

    t0 = time.time()
    _, U = solver.forward_solve(sig)
    U_clean = np.asarray(U)
    log(f"forward solve: {U_clean.shape[0]} patterns in {time.time() - t0:.2f}s")

    gauge = np.abs(U_clean.sum(axis=1)).max()
    log(f"gauge check   : max |sum_i U_i| = {gauge:.2e}  (must be ~0)")
    log(f"voltage range : [{U_clean.min():.5g}, {U_clean.max():.5g}]")

    # a homogeneous reference frame, useful for difference imaging
    background = (
        phantom_cfg["background"]
        if not isinstance(phantom_cfg["background"], dict)
        else 1.0
    )
    sig_bg = Function(solver.V_sigma)
    sig_bg.x.array[:] = background
    _, U_bg = solver.forward_solve(sig_bg)
    U_background = np.asarray(U_bg)

    # ── 6. save ──────────────────────────────────────────────────────────
    cfg_saved = dict(cfg)
    cfg_saved["mesh"] = mesh_cfg
    cfg_saved["mesh_info"] = info
    cfg_saved["drive"] = dict(drive, n_patterns=n_pat)

    out = gt_io.save_ground_truth(
        out_dir=out_dir,
        config=cfg_saved,
        phantom=phantom_cfg,
        points=points,
        sigma_true=sigma_true,
        U_clean=U_clean,
        Inj=Inj,
        z=z,
        L=L,
        background=background,
        mesh_path=mesh_path,
    )
    np.savez_compressed(out / "background_measurements.npz",
                        U_background=U_background)

    _write_summary(out, cfg_saved, phantom_cfg, info, sigma_true, U_clean, gauge)

    if cfg["plot"]["enabled"]:
        _plot(out, cfg_saved, solver, sigma_true, U_clean, phantom_cfg)

    log(f"ground truth written to {out}")
    log(f"next: python simulate.py --gt {out}")
    return out


# --------------------------------------------------------------------------- #
def _write_summary(out, cfg, phantom, info, sigma_true, U_clean, gauge):
    m = cfg["mesh"]
    lines = [
        "EIT ground-truth bundle",
        "=" * 60,
        f"name              : {cfg['name']}",
        f"created           : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Geometry / truth mesh   (circular cross-section)",
        "-" * 60,
        f"radius            : {m['radius']}",
        f"cells / nodes     : {info['n_cells']} / {info['n_nodes']}",
        f"element size h    : {info['h_min']:.4f} .. {info['h_max']:.4f}",
        f"target h_c / h_e  : {m['h_center']} / {m['h_edge']}  "
        f"(grading {m['grading']})",
        f"mesh quality      : min {info['quality_min']:.3f}, "
        f"mean {info['quality_mean']:.3f}",
        f"domain area       : {info['area']:.6f}  "
        f"(exact {np.pi * m['radius'] ** 2:.6f})",
        "",
        "Electrodes / drive",
        "-" * 60,
        f"electrodes L      : {cfg['L']}",
        f"electrode width   : {m['electrode_width_deg']} deg",
        f"ring offset       : {m['electrode_offset_deg']} deg",
        f"contact impedance : {cfg['contact_impedance']}",
        f"drive method      : {cfg['drive']['method']} "
        f"({DRIVE_NAMES[cfg['drive']['method']]})",
        f"patterns          : {cfg['drive']['n_patterns']}",
        f"amplitude         : {cfg['drive']['amplitude']}",
        "",
        "Phantom",
        "-" * 60,
        describe_phantom(phantom),
        f"sigma range       : [{sigma_true.min():.5g}, {sigma_true.max():.5g}]",
        "",
        "Measurements (noise-free)",
        "-" * 60,
        f"shape             : {U_clean.shape}",
        f"voltage range     : [{U_clean.min():.6g}, {U_clean.max():.6g}]",
        f"gauge sum_i U_i   : {gauge:.3e}",
        "",
        "This data was generated on the mesh stored as truth_mesh.msh.",
        "Reconstruct on a DIFFERENT (coarser) mesh in simulate.py to avoid the",
        "inverse crime.",
    ]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")


def _plot(out, cfg, solver, sigma_true, U_clean, phantom_cfg):
    p = cfg["plot"]
    tri = solver.triangulation()

    vmin = p["vmin"] if p["vmin"] is not None else float(sigma_true.min())
    vmax = p["vmax"] if p["vmax"] is not None else float(sigma_true.max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im = ax.tripcolor(tri, sigma_true, cmap=p["cmap"], shading="flat",
                      vmin=vmin, vmax=vmax)
    if p["show_mesh"]:
        ax.triplot(tri, color="k", lw=0.15, alpha=0.35)
    if p["show_electrodes"]:
        _draw_electrodes(ax, cfg)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Ground truth: {cfg['name']}\n{describe_phantom(phantom_cfg)}",
                 fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("conductivity [S/m]")

    ax = axes[1]
    for i, row in enumerate(U_clean):
        ax.plot(row, lw=1.0, alpha=0.75,
                label=f"pattern {i}" if U_clean.shape[0] <= 8 else None)
    ax.set_xlabel("electrode index")
    ax.set_ylabel("voltage")
    ax.set_title(f"Noise-free electrode voltages "
                 f"({U_clean.shape[0]} patterns)", fontsize=10)
    ax.grid(alpha=0.3)
    if U_clean.shape[0] <= 8:
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "ground_truth.png", dpi=p["dpi"])
    plt.close(fig)


def _draw_electrodes(ax, cfg):
    R = cfg["mesh"]["radius"]
    for i, (t0, t1) in enumerate(electrode_spans(cfg["mesh"])):
        th = np.linspace(2 * np.pi * t0, 2 * np.pi * t1, 24)
        ax.plot(R * np.cos(th), R * np.sin(th), color="k", lw=3,
                solid_capstyle="butt")
        c = np.pi * (t0 + t1)
        ax.text(1.13 * R * np.cos(c), 1.13 * R * np.sin(c), str(i),
                ha="center", va="center", fontsize=6)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", help="override CONFIG['name']")
    ap.add_argument("--preset", help="use a named phantom preset")
    ap.add_argument("--seed", type=int, help="override CONFIG['random_seed']")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(CONFIG, default=gt_io._json_default))
    if args.name:
        cfg["name"] = args.name
    if args.seed is not None:
        cfg["random_seed"] = args.seed
        cfg["phantom"]["seed"] = args.seed
    if args.preset:
        cfg["phantom"] = {"preset": args.preset, "seed": cfg["random_seed"]}

    generate(cfg)


if __name__ == "__main__":
    main()
