#!/usr/bin/env python
# mesh_tools.py

"""
Finite-element mesh generation for EIT test cells.

The test cell is a **cylinder**, so the modelled cross-section is always a
**circle**.  A mesh is therefore fully described by a radius, an electrode
layout and a target element-size profile - which means meshing is just another
dictionary, and just another sweepable axis in a config file:

    {
        "radius": 1.0,               # cross-section radius [m]

        "n_electrodes": 16,          # electrodes, equally spaced on the rim
        "electrode_width_deg": 12.0, # angular width of ONE electrode
        "electrode_offset_deg": 0.0, # rotate the whole electrode ring

        "n_per_electrode": 10,       # boundary nodes across one electrode
        "n_per_gap": 5,              # boundary nodes across one gap

        "h_center": 0.055,           # target element size at the centre
        "h_edge": 0.030,             # target element size at the rim
        "grading": 1.0,              # size(r) = h_c + (h_e - h_c)*(r/R)^grading

        "algorithm": 6,              # gmsh 2-D meshing algorithm
        "optimize": True,            # run the Netgen quality optimiser
    }

Meshes are cached: the parameters hash to a file name, so re-running a sweep
never re-meshes.  The generated ``.msh`` carries physical group ``i+1`` on
electrode ``i`` and group ``0`` on the electrode-free boundary, which is the
layout ``eitcore.EIT`` expects.

Run directly to build and report a few meshes:

    python mesh_tools.py
    python mesh_tools.py --h-center 0.12 --h-edge 0.08 --electrodes 16
"""

import argparse
import hashlib
import json
import os
import pathlib

import numpy as np

#: Every meshing parameter and its default.  Anything absent from a user
#: dictionary falls back to the value here.
DEFAULT_MESH = {
    "radius": 1.0,
    "n_electrodes": 16,
    "electrode_width_deg": 12.0,
    "electrode_offset_deg": 0.0,
    "n_per_electrode": 10,
    "n_per_gap": 5,
    "h_center": 0.10,
    "h_edge": 0.06,
    "grading": 1.0,
    "algorithm": 6,
    "optimize": True,
}

#: Parameters that define the *physical* cell rather than its discretisation.
#: The ground truth and the reconstruction must agree on these, otherwise the
#: reconstruction is solving for a different experiment than the one measured.
GEOMETRY_KEYS = ("radius", "n_electrodes", "electrode_width_deg",
                 "electrode_offset_deg")

#: Parameters that only change how finely the same cell is discretised.
#: These are free to differ between the truth mesh and the reconstruction mesh.
DISCRETISATION_KEYS = ("n_per_electrode", "n_per_gap", "h_center", "h_edge",
                       "grading", "algorithm", "optimize")


def _merge(cfg):
    cfg = dict(cfg or {})
    cfg.pop("label", None)  # a human tag used by simulate.py, not a mesh parameter

    unknown = set(cfg) - set(DEFAULT_MESH)
    if unknown:
        raise ValueError(
            f"Unknown meshing parameter(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(DEFAULT_MESH)}. "
            "Note that the cross-section is always circular, so there is no "
            "'shape' option."
        )

    out = dict(DEFAULT_MESH)
    out.update(cfg)
    _validate(out)
    return out


def _validate(cfg):
    if cfg["radius"] <= 0:
        raise ValueError("radius must be positive")
    L = int(cfg["n_electrodes"])
    if L < 4:
        raise ValueError(f"n_electrodes must be at least 4, got {L}")
    sector = 360.0 / L
    if not 0 < cfg["electrode_width_deg"] < sector:
        raise ValueError(
            f"electrode_width_deg must be in (0, {sector:.2f}) for {L} "
            f"electrodes so that neighbouring electrodes stay separated; "
            f"got {cfg['electrode_width_deg']}."
        )
    if int(cfg["n_per_electrode"]) < 2:
        raise ValueError("n_per_electrode must be at least 2")
    if int(cfg["n_per_gap"]) < 1:
        raise ValueError("n_per_gap must be at least 1")
    for k in ("h_center", "h_edge"):
        if cfg[k] <= 0:
            raise ValueError(f"{k} must be positive")
        if cfg[k] > cfg["radius"]:
            raise ValueError(
                f"{k}={cfg[k]} exceeds the cell radius {cfg['radius']}; "
                "the mesh would have no interior."
            )
    if cfg["grading"] <= 0:
        raise ValueError("grading must be positive")


def mesh_signature(cfg):
    """Short deterministic hash of the meshing parameters."""
    return hashlib.md5(
        json.dumps(_merge(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def mesh_filename(cfg, out_dir="data/meshes"):
    c = _merge(cfg)
    name = (
        f"disk_R{c['radius']:g}_L{c['n_electrodes']}"
        f"_h{c['h_center']:g}-{c['h_edge']:g}"
        f"_{mesh_signature(cfg)}.msh"
    )
    return str(pathlib.Path(out_dir) / name)


# --------------------------------------------------------------------------- #
#  electrode layout
# --------------------------------------------------------------------------- #
def _boundary_point(cfg, t):
    """Point on the circular rim at normalised perimeter position ``t`` [0, 1)."""
    th = 2 * np.pi * t
    R = cfg["radius"]
    return R * np.cos(th), R * np.sin(th)


def electrode_spans(cfg):
    """
    ``[(t_start, t_end), ...]`` in normalised perimeter units, one per
    electrode: equally spaced, each covering ``electrode_width_deg``.
    """
    c = _merge(cfg)
    L = int(c["n_electrodes"])
    width = c["electrode_width_deg"] / 360.0
    sector = 1.0 / L
    offset = c["electrode_offset_deg"] / 360.0
    return [
        (offset + i * sector - width / 2.0, offset + i * sector + width / 2.0)
        for i in range(L)
    ]


def electrode_centres(cfg):
    """``(L, 2)`` Cartesian centres of the electrodes, for plotting."""
    c = _merge(cfg)
    L = int(c["n_electrodes"])
    offset = np.deg2rad(c["electrode_offset_deg"])
    ang = offset + 2 * np.pi * np.arange(L) / L
    return np.column_stack([c["radius"] * np.cos(ang), c["radius"] * np.sin(ang)])


# --------------------------------------------------------------------------- #
#  generation
# --------------------------------------------------------------------------- #
def build_mesh(cfg, out_dir="data/meshes", force=False, verbose=False):
    """
    Generate (or reuse from cache) the ``.msh`` described by ``cfg``.

    Returns the path.  Everything downstream reads the mesh through
    ``EIT(mesh_name=...)`` so there is exactly one source of truth.
    """
    cfg = _merge(cfg)
    path = mesh_filename(cfg, out_dir)

    if os.path.exists(path) and not force:
        return path

    import gmsh

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("eit_cell")

        spans = electrode_spans(cfg)
        L = len(spans)
        n_in = int(cfg["n_per_electrode"])
        n_gap = int(cfg["n_per_gap"])

        point_tags = []                              # rim points, in order
        electrode_lines = {i: [] for i in range(L)}  # line tags per electrode
        line_owner = []                              # electrode index or None

        def add_pt(t):
            x, y = _boundary_point(cfg, t)
            return gmsh.model.occ.addPoint(x, y, 0.0)

        # walk the rim: electrode i, then the gap to electrode i+1
        for i, (t0, t1) in enumerate(spans):
            ts = np.linspace(t0, t1, n_in)
            for t in ts[:-1]:
                point_tags.append(add_pt(t))
                line_owner.append(i)
            point_tags.append(add_pt(ts[-1]))  # last electrode point opens the gap
            line_owner.append(None)

            t_next = spans[(i + 1) % L][0] + (1.0 if i == L - 1 else 0.0)
            for t in np.linspace(t1, t_next, n_gap + 2)[1:-1]:
                point_tags.append(add_pt(t))
                line_owner.append(None)

        # line k joins point k -> k+1 and belongs to an electrode iff its start
        # point is an interior point of that electrode
        n_pts = len(point_tags)
        line_tags = []
        for k in range(n_pts):
            tag = gmsh.model.occ.addLine(point_tags[k], point_tags[(k + 1) % n_pts])
            line_tags.append(tag)
            if line_owner[k] is not None:
                electrode_lines[line_owner[k]].append(tag)

        gmsh.model.occ.synchronize()
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [surf], 1, name="domain")
        for i in range(L):
            if not electrode_lines[i]:
                raise RuntimeError(
                    f"Electrode {i} received no boundary segments; increase "
                    "n_per_electrode or electrode_width_deg."
                )
            gmsh.model.addPhysicalGroup(
                1, electrode_lines[i], i + 1, name=f"Electrode{i + 1}"
            )
        free = [t for k, t in enumerate(line_tags) if line_owner[k] is None]
        gmsh.model.addPhysicalGroup(1, free, 0, name="No-Electrode")

        # radial element-size field
        R, hc, he, p = cfg["radius"], cfg["h_center"], cfg["h_edge"], cfg["grading"]
        field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            field, "F", f"{hc} + ({he}-{hc})*(sqrt(x*x + y*y)/{R})^{p}"
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        # let the size field win over the boundary point spacing
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", int(cfg["algorithm"]))

        gmsh.model.mesh.generate(2)
        if cfg["optimize"]:
            gmsh.model.mesh.optimize("Netgen")

        gmsh.write(path)
    finally:
        gmsh.finalize()

    return path


def mesh_info(path):
    """Cell/node counts, element sizes and triangle quality of a ``.msh``."""
    from mpi4py import MPI
    from dolfinx.io import gmshio

    domain, _, _ = gmshio.read_from_msh(path, MPI.COMM_WORLD, gdim=2)
    n_cells = domain.topology.index_map(2).size_local
    n_nodes = domain.geometry.x.shape[0]

    xy = domain.geometry.x[:, :2]
    cells = np.asarray(domain.geometry.dofmap).reshape((-1, 3))
    p = xy[cells]
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = 0.5 * (e0 + e1 + e2)
    area = np.sqrt(np.maximum(s * (s - e0) * (s - e1) * (s - e2), 0.0))
    # radius-ratio quality in [0, 1]; 1 = equilateral
    quality = np.where(
        s > 0, 4.0 * np.sqrt(3.0) * area / (e0**2 + e1**2 + e2**2), 0.0
    )

    return {
        "path": path,
        "n_cells": int(n_cells),
        "n_nodes": int(n_nodes),
        "h_min": float(np.min([e0, e1, e2])),
        "h_max": float(np.max([e0, e1, e2])),
        "area": float(area.sum()),
        "quality_min": float(quality.min()),
        "quality_mean": float(quality.mean()),
    }


def describe(cfg, out_dir="data/meshes"):
    """Build the mesh and return a one-line human-readable report."""
    info = mesh_info(build_mesh(cfg, out_dir=out_dir))
    c = _merge(cfg)
    ideal = np.pi * c["radius"] ** 2
    return (
        f"cells={info['n_cells']:6d}  nodes={info['n_nodes']:6d}  "
        f"h={info['h_min']:.3f}..{info['h_max']:.3f}  "
        f"area={info['area']:.4f} (exact {ideal:.4f})  "
        f"quality min/mean={info['quality_min']:.2f}/{info['quality_mean']:.2f}"
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--radius", type=float, default=DEFAULT_MESH["radius"])
    ap.add_argument("--electrodes", type=int, default=DEFAULT_MESH["n_electrodes"])
    ap.add_argument("--width-deg", type=float,
                    default=DEFAULT_MESH["electrode_width_deg"])
    ap.add_argument("--h-center", type=float)
    ap.add_argument("--h-edge", type=float)
    ap.add_argument("--grading", type=float, default=DEFAULT_MESH["grading"])
    ap.add_argument("--out-dir", default="data/meshes")
    args = ap.parse_args()

    base = {
        "radius": args.radius,
        "n_electrodes": args.electrodes,
        "electrode_width_deg": args.width_deg,
        "grading": args.grading,
    }

    if args.h_center or args.h_edge:
        cfg = dict(base,
                   h_center=args.h_center or DEFAULT_MESH["h_center"],
                   h_edge=args.h_edge or DEFAULT_MESH["h_edge"])
        print(f"custom       {describe(cfg, args.out_dir)}")
        return

    print("Reference meshes (circular cross-section):\n")
    for name, hc, he in [
        ("very coarse", 0.22, 0.16),
        ("coarse", 0.16, 0.11),
        ("medium", 0.13, 0.09),
        ("fine", 0.09, 0.06),
        ("truth (fine)", 0.055, 0.03),
    ]:
        print(f"  {name:<14}{describe(dict(base, h_center=hc, h_edge=he), args.out_dir)}")
    print(
        "\nUse a fine mesh for make_ground_truth.py and a coarser one for the\n"
        "reconstructions in simulate.py (see the README on the inverse crime)."
    )


if __name__ == "__main__":
    main()
