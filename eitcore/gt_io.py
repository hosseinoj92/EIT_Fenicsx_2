# gt_io.py

"""
On-disk format for a simulated ground-truth data set.

One ground truth is one directory under ``GT_simulated_result/``:

    GT_simulated_result/<name>/
        config.json          the full configuration that produced it
        phantom.json         the *resolved* phantom (random draws frozen)
        truth_mesh.msh       the mesh the forward data was computed on
        sigma_true.npz       cell centroids + conductivity on the truth mesh
        measurements.npz     clean voltages, injection matrix, contact impedance
        ground_truth.csv     x, y, sigma  (human readable / external tools)
        measurements.csv     pattern, electrode, voltage
        ground_truth.png     preview figure
        summary.txt          one-page human-readable description

The bundle is deliberately self-contained: ``simulate.py`` needs nothing but
this directory, and the truth mesh travels with the data so a reconstruction
can always be checked against the geometry that generated it.
"""

import json
import pathlib
import shutil

import numpy as np


BUNDLE_VERSION = 2


class GroundTruth:
    """Loaded ground-truth bundle."""

    def __init__(self, path, data):
        self.path = pathlib.Path(path)
        self._d = data

    # -- data ------------------------------------------------------------ #
    @property
    def sigma_true(self):
        """Conductivity on the truth mesh, one value per cell."""
        return self._d["sigma_true"]

    @property
    def points(self):
        """``(n_cells, 2)`` cell centroids of the truth mesh."""
        return self._d["points"]

    @property
    def U_clean(self):
        """``(n_patterns, L)`` noise-free electrode voltages."""
        return self._d["U_clean"]

    @property
    def Inj(self):
        return self._d["Inj"]

    @property
    def z(self):
        return self._d["z"]

    @property
    def L(self):
        return int(self._d["L"])

    @property
    def background(self):
        return float(self._d["background"])

    @property
    def config(self):
        return self._d["config"]

    @property
    def phantom(self):
        return self._d["phantom"]

    @property
    def truth_mesh(self):
        return str(self.path / "truth_mesh.msh")

    @property
    def radius(self):
        """Radius of the circular cross-section the data was generated on."""
        return float(self.config.get("mesh", {}).get("radius", 1.0))

    def __repr__(self):
        return (
            f"GroundTruth('{self.path.name}': {len(self.sigma_true)} cells, "
            f"L={self.L}, {self.U_clean.shape[0]} patterns, "
            f"sigma in [{self.sigma_true.min():.3g}, {self.sigma_true.max():.3g}])"
        )


# --------------------------------------------------------------------------- #
def save_ground_truth(out_dir, config, phantom, points, sigma_true, U_clean,
                      Inj, z, L, background, mesh_path, figure=None):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "sigma_true.npz",
        points=np.asarray(points, dtype=float),
        sigma=np.asarray(sigma_true, dtype=float),
    )
    np.savez_compressed(
        out / "measurements.npz",
        U_clean=np.asarray(U_clean, dtype=float),
        Inj=np.asarray(Inj, dtype=float),
        z=np.asarray(z, dtype=float),
        L=np.array(L),
        background=np.array(background, dtype=float),
        version=np.array(BUNDLE_VERSION),
    )

    with open(out / "config.json", "w") as fh:
        json.dump(config, fh, indent=2, default=_json_default)
    with open(out / "phantom.json", "w") as fh:
        json.dump(phantom, fh, indent=2, default=_json_default)

    # copy the truth mesh next to the data so the bundle is self-contained
    mesh_path = pathlib.Path(mesh_path)
    target = out / "truth_mesh.msh"
    if mesh_path.resolve() != target.resolve():
        shutil.copyfile(mesh_path, target)

    # human-readable duplicates
    pts = np.asarray(points, dtype=float)
    sig = np.asarray(sigma_true, dtype=float)
    np.savetxt(
        out / "ground_truth.csv",
        np.column_stack([pts[:, 0], pts[:, 1], sig]),
        delimiter=",", header="x,y,sigma", comments="", fmt="%.8g",
    )

    U = np.asarray(U_clean, dtype=float)
    rows = [
        (p, e, U[p, e]) for p in range(U.shape[0]) for e in range(U.shape[1])
    ]
    np.savetxt(
        out / "measurements.csv",
        np.array(rows),
        delimiter=",", header="pattern,electrode,voltage", comments="",
        fmt=["%d", "%d", "%.10g"],
    )

    return out


def load_ground_truth(path):
    """Load a bundle written by :func:`save_ground_truth`."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Ground truth '{p}' not found. Run make_ground_truth.py first."
        )
    if not (p / "measurements.npz").exists():
        # allow pointing at the parent folder when it holds exactly one bundle
        subs = [d for d in p.iterdir() if (d / "measurements.npz").exists()]
        if len(subs) == 1:
            p = subs[0]
        else:
            raise FileNotFoundError(
                f"'{p}' is not a ground-truth bundle "
                f"(no measurements.npz). Candidates: {[d.name for d in subs]}"
            )

    sig = np.load(p / "sigma_true.npz")
    meas = np.load(p / "measurements.npz")

    with open(p / "config.json") as fh:
        config = json.load(fh)
    with open(p / "phantom.json") as fh:
        phantom = json.load(fh)

    data = {
        "points": sig["points"],
        "sigma_true": sig["sigma"],
        "U_clean": meas["U_clean"],
        "Inj": meas["Inj"],
        "z": meas["z"],
        "L": int(meas["L"]),
        "background": float(meas["background"]),
        "config": config,
        "phantom": phantom,
    }
    return GroundTruth(p, data)


def list_ground_truths(root="GT_simulated_result"):
    root = pathlib.Path(root)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "measurements.npz").exists())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    return str(o)
