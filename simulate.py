#!/usr/bin/env python
# simulate.py

"""
Reconstruct a stored ground truth with one or many algorithms and compare them.

Everything is driven by the CONFIG dictionary below.

Sweeping
--------
**Any parameter you write as a Python list becomes a sweep axis.**  The script
takes the cartesian product of every list, runs them all, and names each result
after exactly the parameters that actually vary.  So

    "algorithms": {"gn": {"lambda": [1e-3, 1e-2]}, "tv": {"lambda": [1e-3]}}
    "noise": {"level_percent": [0.5, 2.0]}

gives six runs, named ``gn_lambda=0.001_noise=0.5`` and so on, while a config
with a single value everywhere produces one run named just ``gn``.  Scalars
that are genuinely list-valued (mesh ``vertices``) are excluded from the rule.

Run

    python simulate.py
    python simulate.py --gt GT_simulated_result/my_case --name my_experiment
"""

import argparse
import itertools
import json
import os
import pathlib
import sys
import time
import traceback

# ═════════════════════════════════ CONFIG ═══════════════════════════════════
CONFIG = {
    # ── input ────────────────────────────────────────────────────────────
    # Directory written by make_ground_truth.py
    "ground_truth": "GT_simulated_result/two_inclusions_16e",

    # ── output ───────────────────────────────────────────────────────────
    "output": {
        "root": "results",
        "run_name": None,            # None -> "<gt name>_<timestamp>"
        "save_per_run_figures": True,
        "save_csv": True,
        "save_npz": True,
    },

    # ── measurement noise added to the stored clean data ─────────────────
    # Lists sweep.  See noise.py for the models and why the default is
    # relative_to_max rather than per-measurement.
    "noise": {
        "model": "relative_to_max",  # relative_to_max | relative_to_pattern_max
                                     # relative_per_measurement | absolute | none
        "level_percent": [0.5, 2.0], # sweep: two noise levels
        "seed": 0,                   # fixed -> all algorithms see the SAME noise
        "floor_fraction": 0.0,
        "electrode_gain_percent": 0.0,   # systematic calibration error
        "electrode_offset": 0.0,
        "use_statistical_weighting": True,  # feed 1/var to the reconstructors
        "weight_relative_floor": 1e-3,
    },

    # ── reconstruction mesh(es) ──────────────────────────────────────────
    # The cross-section is a circle (the test cell is a cylinder).
    #
    # A LIST of mesh dicts sweeps over meshes; give each one a "label" and it
    # is used in the run names and charts.
    #
    # Keep these COARSER than the truth mesh used in make_ground_truth.py.
    # Reconstructing on the mesh that generated the data is an inverse crime
    # and flatters every algorithm; aim for at least 4x fewer cells.  The
    # script prints the ratio and warns if it is not coarser.
    #
    # Only the DISCRETISATION keys below should normally be set.  The physical
    # cell (radius, n_electrodes, electrode_width_deg, electrode_offset_deg) is
    # inherited from the ground truth, because changing it would mean
    # reconstructing a different experiment than the one that was measured.
    # Overriding one anyway is allowed - useful for modelling-error studies -
    # but the script warns loudly.
    "recon_mesh": [
        {
            "label": "coarse",       # name used in outputs
            "h_center": 0.16,        # target element size at the centre
            "h_edge": 0.11,          # target element size at the rim
            "grading": 1.0,          # size(r)=h_c+(h_e-h_c)(r/R)^grading
                                     #   >1 keeps the interior coarse longer
                                     #   <1 refines towards the centre
            "n_per_electrode": 8,    # boundary nodes across one electrode
            "n_per_gap": 4,          # boundary nodes across one gap
            "algorithm": 6,          # gmsh 2-D meshing algorithm
            "optimize": True,        # Netgen quality optimiser
        },
        # {"label": "medium", "h_center": 0.12, "h_edge": 0.08,
        #  "grading": 1.0, "n_per_electrode": 8, "n_per_gap": 4},
        # {"label": "fine",   "h_center": 0.09, "h_edge": 0.06,
        #  "grading": 1.0, "n_per_electrode": 10, "n_per_gap": 5},
    ],

    # ── settings shared by every algorithm ────────────────────────────────
    "common": {
        "device": "cpu",
        "sigma_min": 0.01,           # clip range for the reconstruction
        "sigma_max": 5.0,
        "background": None,          # None -> take it from the ground truth
        "verbose": False,
    },

    # ── algorithms to run, and their parameters ──────────────────────────
    # Comment out an entry to skip it.  Any value may be a list -> sweep.
    # NOTE on "lambda": it is DIMENSIONLESS.  Internally it is multiplied by
    # trace(J^T W J)/trace(R), so lambda ~ 1 means "regularisation as strong as
    # the data term" on any mesh, at any noise level and for any drive
    # amplitude.  Without that rescaling the same number is far too weak on one
    # mesh and far too strong on another, and the values are not comparable
    # across the very settings this framework exists to compare.  Set
    # "lambda_scaling": "absolute" to pass raw values through instead.
    # Useful ranges: gn 1e-2..1, tv 1..30, linear 3e-2..1.
    "algorithms": {
        "gn": {
            "lambda": [0.03, 0.1, 0.3],
            "max_iter": 12,
            "prior": "laplacian",    # laplacian | gaussian | tikhonov
            "prior_eps": 1e-4,
            # "prior_corrlength": 0.2,   # gaussian only
            # "prior_std": 0.15,
            "line_search": "armijo", # armijo | grid
        },
        "tv": {
            "lambda": [3.0, 10.0],
            "beta": 1e-6,
            "max_iter": 10,
        },
        "linear": {
            "lambda": [0.3],
            "prior": "laplacian",
        },
        # alpha is NOT auto-scaled (see algorithms.py) - it depends on the
        # absolute voltage level, so re-sweep it if you change the drive
        # amplitude or the contact impedance.
        "l1": {
            "alpha": [1e-5],
            "kappa": 0.0285,
            "max_iter": 200,
            "initial_step_size": 0.05,
        },
        # D-bar is direct (no iteration, no prior).  Its single regularisation
        # parameter is the truncation radius R of the scattering transform; see
        # the note in algorithms.py.  Watch "t_growth" in the run summary: if it
        # exceeds ~1, R is too large for this noise level.
        "dbar": {
            "R": [3.0, 3.5, 4.0],
            "scattering": "exp",     # exp = Born approximation | bie = full
        },
    },

    # ── metrics ──────────────────────────────────────────────────────────
    "metrics": {
        "grid_resolution": 200,      # common evaluation grid (mesh independent)
        "roi_threshold": 0.05,       # |sigma - bg| > 5% of bg counts as inclusion
        "rank_by": "rel_L2_roi",     # metric used to pick the "best" run
    },

    # ── plotting ─────────────────────────────────────────────────────────
    "plot": {
        "enabled": True,
        "cmap": "turbo",
        "vmin": None,                # None -> ground-truth range
        "vmax": None,
        "max_cols": 5,
        "dpi": 150,
        "comparison_figure": True,
        "metric_charts": True,
        "convergence_curves": True,
    },
}
# ════════════════════════════════════════════════════════════════════════════


def _setup_environment():
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

from mesh_tools import (                                        # noqa: E402
    build_mesh, mesh_info, DEFAULT_MESH, GEOMETRY_KEYS, DISCRETISATION_KEYS,
)
from eitcore import EIT, gt_io                                  # noqa: E402
from eitcore import algorithms as algo_registry                 # noqa: E402
from eitcore.metrics import (                                   # noqa: E402
    MetricEvaluator, data_residual, LOWER_IS_BETTER,
)
from eitcore.noise import add_measurement_noise, gamma_inv, snr_db  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── sweep expansion ────────────────────────────────
#: keys whose value is naturally a list and must not be read as a sweep axis
NON_SWEEP_KEYS = {"vertices", "clip", "semi_axes", "center", "size", "radii",
                  "point", "normal", "regions"}


def _axes_of(d, prefix=""):
    """Yield ``(dotted_key, [values])`` for every leaf that is a sweep axis."""
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, list) and k not in NON_SWEEP_KEYS:
            yield key, v
        else:
            yield key, [v]


def expand_grid(d):
    """Cartesian product over the list-valued leaves of a flat dict."""
    keys, value_lists = zip(*_axes_of(d)) if d else ((), ())
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def build_run_matrix(cfg):
    """
    Every (mesh, noise, algorithm, algorithm-params) combination.

    Returns ``(runs, varying)`` where ``varying`` is the set of parameter names
    that actually change across the matrix - those, and only those, end up in
    the run names and in the comparison charts.
    """
    meshes = cfg["recon_mesh"]
    if isinstance(meshes, dict):
        meshes = [meshes]

    noise_combos = list(expand_grid(cfg["noise"]))

    runs = []
    for mesh_idx, mesh in enumerate(meshes):
        mesh_label = mesh.get("label") or f"mesh{mesh_idx}"
        for noise in noise_combos:
            for algo_name, params in cfg["algorithms"].items():
                for combo in expand_grid(params or {}):
                    runs.append(
                        {
                            "algorithm": algo_name,
                            "mesh_label": mesh_label,
                            "mesh_cfg": mesh,
                            "noise": noise,
                            "params": combo,
                        }
                    )

    varying = _find_varying(runs)
    for r in runs:
        r["id"] = _run_id(r, varying)
    _dedupe_ids(runs)
    return runs, varying


#: axes that are shared by every algorithm, so they vary globally
GLOBAL_AXES = ("algorithm", "mesh")


def _find_varying(runs):
    """
    Parameter names that genuinely take more than one value.

    Global axes (algorithm, mesh, noise settings) are compared across the whole
    matrix.  Algorithm parameters are compared *within* each algorithm, because
    ``gn`` using 12 iterations while ``tv`` uses 10 is not a sweep - flagging it
    as one would put a constant ``max_iter=12`` into every run name.
    """
    varying = set()

    seen = {}
    for r in runs:
        flat = _flat_params(r)
        for k, v in flat.items():
            if k in GLOBAL_AXES or k.startswith("noise."):
                seen.setdefault(k, set()).add(_hashable(v))
    varying |= {k for k, vals in seen.items() if len(vals) > 1}

    per_algo = {}
    for r in runs:
        algo = r["algorithm"]
        for k, v in r["params"].items():
            per_algo.setdefault(algo, {}).setdefault(k, set()).add(_hashable(v))
    for params in per_algo.values():
        varying |= {k for k, vals in params.items() if len(vals) > 1}

    return varying


def _flat_params(run):
    out = {"algorithm": run["algorithm"], "mesh": run["mesh_label"]}
    for k, v in run["noise"].items():
        out[f"noise.{k}"] = v
    for k, v in run["params"].items():
        out[k] = v
    return out


def _hashable(v):
    return tuple(v) if isinstance(v, (list, dict)) else v


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _run_id(run, varying):
    """
    Name a run after its algorithm plus only the axes that actually vary.

    A single-run config gives ``gn``; sweeping lambda and noise gives
    ``gn_lambda=0.01_noise.level_percent=2``.
    """
    parts = [run["algorithm"]]
    flat = _flat_params(run)
    for k in sorted(varying):
        if k == "algorithm":
            continue
        if k in flat:
            short = k.split(".")[-1]
            parts.append(f"{short}={_fmt(flat[k])}")
    return "_".join(parts)


def _dedupe_ids(runs):
    counts = {}
    for r in runs:
        n = counts.get(r["id"], 0)
        counts[r["id"]] = n + 1
        if n:
            r["id"] = f"{r['id']}__{n + 1}"


# ─────────────────────────── data preparation ───────────────────────────────
def prepare_noise(gt, noise_cfg):
    """Add noise to the stored clean data.  Deterministic in ``seed``."""
    U_clean = gt.U_clean
    level = float(noise_cfg.get("level_percent", 0.0))
    model = noise_cfg.get("model", "relative_to_max")

    U_noisy, std = add_measurement_noise(
        U_clean,
        model=model,
        level=level,
        seed=noise_cfg.get("seed", 0),
        floor_fraction=noise_cfg.get("floor_fraction", 0.0),
        electrode_gain_percent=noise_cfg.get("electrode_gain_percent", 0.0),
        electrode_offset=noise_cfg.get("electrode_offset", 0.0),
    )

    GammaInv = None
    if noise_cfg.get("use_statistical_weighting", True) and level > 0:
        GammaInv = gamma_inv(std, noise_cfg.get("weight_relative_floor", 1e-3))

    return U_noisy, std, GammaInv


def make_solver(gt, mesh_cfg):
    """
    Build the reconstruction solver.

    The physical cell is inherited from the ground truth so the reconstruction
    describes the same experiment that produced the data; only the
    discretisation is taken from ``mesh_cfg``.
    """
    inherited = {k: gt.config["mesh"][k] for k in GEOMETRY_KEYS
                 if k in gt.config.get("mesh", {})}

    overrides = {k: v for k, v in mesh_cfg.items()
                 if k in GEOMETRY_KEYS and inherited.get(k) != v}
    if overrides:
        log("  WARNING: the reconstruction mesh overrides the physical cell "
            f"{overrides} (ground truth: "
            f"{ {k: inherited.get(k) for k in overrides} }). "
            "This models a geometry mismatch; metrics stay comparable only if "
            "that is what you intended.")

    merged = dict(DEFAULT_MESH)
    merged.update(inherited)
    merged.update({k: v for k, v in mesh_cfg.items() if k != "label"})
    merged["n_electrodes"] = gt.L

    path = build_mesh(merged, out_dir="data/meshes")
    info = mesh_info(path)
    solver = EIT(gt.L, gt.Inj, gt.z, backend="Scipy", mesh_name=path)
    return solver, info, merged


# ─────────────────────────────── the sweep ──────────────────────────────────
def run_experiment(cfg):
    t_start = time.time()

    gt = gt_io.load_ground_truth(cfg["ground_truth"])
    log(f"ground truth: {gt}")

    out_root = pathlib.Path(cfg["output"]["root"])
    run_name = cfg["output"].get("run_name") or (
        f"{gt.path.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = out_root / run_name
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    log(f"output: {out_dir}")

    runs, varying = build_run_matrix(cfg)
    log(f"{len(runs)} run(s); varying axes: "
        f"{sorted(varying) if varying else '(none)'}")

    background = cfg["common"].get("background")
    if background is None:
        background = gt.background

    # one common, mesh-independent evaluation grid for every run
    evaluator = MetricEvaluator(
        gt.points, gt.sigma_true,
        background=background,
        resolution=cfg["metrics"]["grid_resolution"],
        radius=gt.radius,
        roi_threshold=cfg["metrics"]["roi_threshold"],
    )
    log(f"metric grid: {evaluator.grid.shape[0]} points inside the domain")

    # group runs so each (mesh, noise) pair builds its solver and data once
    def group_key(r):
        return (r["mesh_label"], json.dumps(r["noise"], sort_keys=True, default=str))

    runs_sorted = sorted(runs, key=group_key)

    results = []
    prior_cache = {}
    dbar_reference_cache = {}
    solver_cache = {}

    for gkey, group in itertools.groupby(runs_sorted, key=group_key):
        group = list(group)
        mesh_label, _ = gkey
        mesh_cfg = group[0]["mesh_cfg"]
        noise_cfg = group[0]["noise"]

        if mesh_label not in solver_cache:
            solver, minfo, merged = make_solver(gt, mesh_cfg)
            solver_cache[mesh_label] = (solver, minfo, merged)
            log(
                f"recon mesh '{mesh_label}': {minfo['n_cells']} cells, "
                f"{minfo['n_nodes']} nodes, "
                f"h_c/h_e={merged['h_center']:g}/{merged['h_edge']:g}, "
                f"grading={merged['grading']:g}, "
                f"quality mean {minfo['quality_mean']:.2f}  "
                f"(truth mesh {gt.config['mesh_info']['n_cells']} cells, "
                f"ratio {gt.config['mesh_info']['n_cells'] / minfo['n_cells']:.1f}x)"
            )
            if minfo["n_cells"] >= gt.config["mesh_info"]["n_cells"]:
                log("  WARNING: the reconstruction mesh is not coarser than the "
                    "truth mesh - results will be optimistic (inverse crime).")
        solver, minfo, merged_mesh = solver_cache[mesh_label]

        U_noisy, std, GammaInv = prepare_noise(gt, noise_cfg)
        snr = snr_db(gt.U_clean, U_noisy)
        log(f"noise: {noise_cfg.get('model')} @ "
            f"{noise_cfg.get('level_percent')}%  ->  SNR {snr:.1f} dB")

        ctx = {
            "GammaInv": GammaInv,
            "prior_cache": prior_cache,
            "dbar_reference_cache": dbar_reference_cache,
        }

        for run in group:
            params = algo_registry.resolve_params(
                run["algorithm"], run["params"], cfg["common"]
            )
            params["background"] = background

            log(f"  running {run['id']} ...")
            t0 = time.time()
            try:
                sigma, info = algo_registry.run(
                    run["algorithm"], solver, U_noisy, params, ctx
                )
                elapsed = time.time() - t0

                m = evaluator.evaluate(solver.cell_centers(), sigma)
                m.update(
                    data_residual(info["U_reconstructed"], U_noisy, std)
                )
                m["runtime_s"] = elapsed
                m["n_cells"] = minfo["n_cells"]
                m["snr_db"] = snr

                results.append(
                    {
                        "run": run,
                        "params": params,
                        "sigma": sigma,
                        "points": solver.cell_centers(),
                        "metrics": m,
                        "info": info,
                        "mesh_info": minfo,
                        "noise_std": std,
                        "U_noisy": U_noisy,
                        "error": None,
                    }
                )
                log(
                    f"    done in {elapsed:.1f}s  "
                    f"rel_L2_roi={m['rel_L2_roi']:.4f}  dice={m['dice']:.3f}  "
                    f"chi2/dof={m.get('chi2_per_dof', float('nan')):.2f}"
                )
            except Exception as exc:  # keep the sweep alive
                log(f"    FAILED: {exc!r}")
                results.append(
                    {
                        "run": run,
                        "params": params,
                        "sigma": None,
                        "points": None,
                        "metrics": {},
                        "info": {},
                        "mesh_info": minfo,
                        "noise_std": std,
                        "U_noisy": U_noisy,
                        "error": traceback.format_exc(),
                    }
                )

    # ── persist ──────────────────────────────────────────────────────────
    _save_runs(out_dir, results, cfg, evaluator)
    table = _save_table(out_dir, results, varying, cfg)

    if cfg["plot"]["enabled"]:
        if cfg["plot"].get("comparison_figure", True):
            _plot_comparison(out_dir, gt, results, evaluator, cfg)
        if cfg["plot"].get("metric_charts", True) and varying:
            _plot_metric_charts(out_dir, results, varying, cfg)
        if cfg["plot"].get("convergence_curves", True):
            _plot_convergence(out_dir, results, cfg)

    _write_summary(out_dir, gt, results, cfg, varying, table,
                   time.time() - t_start)

    with open(out_dir / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=gt_io._json_default)

    log(f"finished {len(results)} run(s) in {time.time() - t_start:.1f}s")
    log(f"results in {out_dir}")
    return out_dir, results


# ───────────────────────────── persistence ──────────────────────────────────
def _save_runs(out_dir, results, cfg, evaluator):
    save_npz = cfg["output"].get("save_npz", True)
    save_csv = cfg["output"].get("save_csv", True)
    save_fig = cfg["output"].get("save_per_run_figures", True)

    for res in results:
        d = out_dir / "runs" / res["run"]["id"]
        d.mkdir(parents=True, exist_ok=True)

        with open(d / "config.json", "w") as fh:
            json.dump(
                {
                    "id": res["run"]["id"],
                    "algorithm": res["run"]["algorithm"],
                    "mesh_label": res["run"]["mesh_label"],
                    "mesh_cfg": res["run"]["mesh_cfg"],
                    "noise": res["run"]["noise"],
                    "params": res["params"],
                },
                fh, indent=2, default=gt_io._json_default,
            )

        if res["error"]:
            (d / "error.txt").write_text(res["error"])
            continue

        with open(d / "metrics.json", "w") as fh:
            json.dump(res["metrics"], fh, indent=2, default=gt_io._json_default)

        if save_npz:
            np.savez_compressed(
                d / "reconstruction.npz",
                points=res["points"], sigma=res["sigma"],
                U_reconstructed=res["info"]["U_reconstructed"],
                U_measured=res["U_noisy"], noise_std=res["noise_std"],
            )
        if save_csv:
            np.savetxt(
                d / "reconstruction.csv",
                np.column_stack([res["points"][:, 0], res["points"][:, 1],
                                 res["sigma"]]),
                delimiter=",", header="x,y,sigma", comments="", fmt="%.8g",
            )
        hist = res["info"].get("history") or []
        if hist and save_csv:
            keys = list(hist[0].keys())
            np.savetxt(
                d / "history.csv",
                np.array([[h[k] for k in keys] for h in hist], dtype=float),
                delimiter=",", header=",".join(keys), comments="", fmt="%.10g",
            )

        if save_fig:
            _plot_single(d, res, evaluator, cfg)


def _save_table(out_dir, results, varying, cfg):
    """One row per run: identity, parameters, metrics.  Written as CSV."""
    rows = []
    for res in results:
        row = {"run_id": res["run"]["id"], "algorithm": res["run"]["algorithm"],
               "mesh": res["run"]["mesh_label"]}
        for k, v in res["run"]["noise"].items():
            row[f"noise.{k}"] = v
        for k, v in res["params"].items():
            if k not in ("verbose", "device"):
                row[k] = v
        row["failed"] = res["error"] is not None
        # a metric sharing a name with a parameter would overwrite it and the
        # table would quietly report the wrong settings
        clash = set(row) & set(res["metrics"])
        for k in clash:
            res["metrics"][f"metric_{k}"] = res["metrics"].pop(k)
        row.update(res["metrics"])
        rows.append(row)

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(_csv_cell(r.get(c, "")) for c in cols))
    (out_dir / "runs.csv").write_text("\n".join(lines) + "\n")

    return rows


def _csv_cell(v):
    if isinstance(v, float):
        return f"{v:.8g}"
    s = str(v)
    return f'"{s}"' if "," in s else s


# ────────────────────────────── plotting ────────────────────────────────────
def _vrange(cfg, gt_values):
    vmin = cfg["plot"]["vmin"]
    vmax = cfg["plot"]["vmax"]
    if vmin is None:
        vmin = float(np.min(gt_values))
    if vmax is None:
        vmax = float(np.max(gt_values))
    return vmin, vmax


def _plot_single(d, res, evaluator, cfg):
    vmin, vmax = _vrange(cfg, evaluator.gt)
    rec_img = evaluator.as_image(
        evaluator.resample(res["points"], res["sigma"])
    )
    gt_img = evaluator.as_image(evaluator.gt)
    err_img = evaluator.as_image(
        evaluator.resample(res["points"], res["sigma"]) - evaluator.gt
    )

    # extent + equal aspect so non-circular domains are drawn to scale
    shared = dict(extent=evaluator.extent, aspect="equal")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, img, title, kw in [
        (axes[0], gt_img, "Ground truth",
         dict(cmap=cfg["plot"]["cmap"], vmin=vmin, vmax=vmax, **shared)),
        (axes[1], rec_img, res["run"]["id"],
         dict(cmap=cfg["plot"]["cmap"], vmin=vmin, vmax=vmax, **shared)),
        (axes[2], err_img, "reconstruction - truth",
         dict(cmap="RdBu_r", vmin=-(vmax - vmin) / 2, vmax=(vmax - vmin) / 2,
              **shared)),
    ]:
        im = ax.imshow(img, **kw)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    m = res["metrics"]
    fig.suptitle(
        f"rel_L2_roi={m['rel_L2_roi']:.4f}  rel_L1={m['rel_L1']:.4f}  "
        f"dice={m['dice']:.3f}  dyn.range={m['dynamic_range']:.2f}  "
        f"chi2/dof={m.get('chi2_per_dof', float('nan')):.2f}  "
        f"({m['runtime_s']:.1f}s)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(d / "reconstruction.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)


def _plot_comparison(out_dir, gt, results, evaluator, cfg):
    ok = [r for r in results if r["error"] is None]
    if not ok:
        return
    rank_by = cfg["metrics"]["rank_by"]
    reverse = rank_by not in LOWER_IS_BETTER
    ok = sorted(ok, key=lambda r: r["metrics"].get(rank_by, np.inf),
                reverse=reverse)

    vmin, vmax = _vrange(cfg, evaluator.gt)
    panels = [("Ground truth", evaluator.as_image(evaluator.gt), None)]
    for r in ok:
        panels.append(
            (
                r["run"]["id"],
                evaluator.as_image(evaluator.resample(r["points"], r["sigma"])),
                r["metrics"],
            )
        )

    ncols = min(cfg["plot"]["max_cols"], len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.9 * nrows),
                             squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for i, (title, img, m) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        im = ax.imshow(img, cmap=cfg["plot"]["cmap"], vmin=vmin, vmax=vmax,
                       extent=evaluator.extent, aspect="equal")
        sub = "" if m is None else f"\n{rank_by}={m.get(rank_by, float('nan')):.4f}"
        ax.set_title(f"{title}{sub}", fontsize=8)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{gt.path.name}  -  {len(ok)} reconstruction(s), sorted by {rank_by}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)


def _plot_metric_charts(out_dir, results, varying, cfg):
    """
    For each numeric axis that varies, plot the key metrics against it with one
    line per algorithm.  This is the figure that answers "which algorithm wins,
    and how does it depend on lambda / noise / mesh?".
    """
    ok = [r for r in results if r["error"] is None]
    if not ok:
        return

    numeric_axes = []
    for key in sorted(varying):
        if key in ("algorithm",):
            continue
        # None means "this algorithm has no such parameter" (l1 has no lambda);
        # ignore those runs rather than discarding the whole axis
        vals = {
            v for v in (_axis_value(r, key) for r in ok)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if len(vals) > 1:
            numeric_axes.append(key)

    show = ["rel_L2_roi", "rel_L1", "dice", "dynamic_range"]
    for key in numeric_axes:
        groups = {}
        for r in ok:
            # algorithms that do not have this parameter at all (l1 has no
            # lambda) would otherwise contribute an empty legend entry
            if not isinstance(_axis_value(r, key), (int, float)):
                continue
            label = r["run"]["algorithm"]
            extra = [
                f"{k.split('.')[-1]}={_fmt(_axis_value(r, k))}"
                for k in sorted(varying)
                if k not in ("algorithm", key) and _axis_value(r, k) is not None
            ]
            if extra:
                label += " (" + ", ".join(extra) + ")"
            groups.setdefault(label, []).append(r)
        if not groups:
            continue

        fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.6))
        for ax, metric in zip(np.atleast_1d(axes), show):
            for label, rs in sorted(groups.items()):
                pts = sorted(
                    (_axis_value(r, key), r["metrics"].get(metric, np.nan))
                    for r in rs
                )
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "o-", label=label, ms=4, lw=1.3)
            ax.set_xlabel(key.split(".")[-1])
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            # log axis only when it is both valid and useful: a swept noise
            # level may legitimately include 0, which has no place on a log axis
            xs_all = [_axis_value(r, key) for r in ok]
            xs_all = [x for x in xs_all if isinstance(x, (int, float))]
            if xs_all and min(xs_all) > 0 and max(xs_all) / min(xs_all) >= 20:
                ax.set_xscale("log")
            arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
            ax.set_title(f"{metric}  ({arrow})", fontsize=9)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, len(labels)), fontsize=7,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        safe = key.replace(".", "_")
        fig.savefig(out_dir / f"metrics_vs_{safe}.png", dpi=cfg["plot"]["dpi"])
        plt.close(fig)


def _axis_value(res, key):
    flat = _flat_params(res["run"])
    return flat.get(key)


def _plot_convergence(out_dir, results, cfg):
    curves = [
        (r["run"]["id"], r["info"].get("history"))
        for r in results
        if r["error"] is None and r["info"].get("history")
    ]
    if not curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, hist in curves:
        it = [h["iter"] for h in hist]
        axes[0].semilogy(it, [max(h["objective"], 1e-300) for h in hist],
                         label=name, lw=1.2)
        axes[1].semilogy(it, [max(h["rel_change"], 1e-300) for h in hist],
                         label=name, lw=1.2)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("objective")
    axes[0].set_title("Objective (data misfit + regularisation)", fontsize=10)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("relative change")
    axes[1].set_title("Step-to-step relative change", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=cfg["plot"]["dpi"])
    plt.close(fig)


# ─────────────────────────────── summary ────────────────────────────────────
def _write_summary(out_dir, gt, results, cfg, varying, table, elapsed):
    rank_by = cfg["metrics"]["rank_by"]
    reverse = rank_by not in LOWER_IS_BETTER
    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    ok = sorted(ok, key=lambda r: r["metrics"].get(rank_by, np.inf), reverse=reverse)

    lines = [
        "EIT reconstruction experiment",
        "=" * 78,
        f"ground truth : {gt.path}",
        f"phantom      : {gt.config.get('name')}  "
        f"({len(gt.sigma_true)} truth cells)",
        f"runs         : {len(results)}  ({len(failed)} failed)",
        f"varying axes : {sorted(varying) if varying else '(none)'}",
        f"ranked by    : {rank_by} "
        f"({'lower' if rank_by in LOWER_IS_BETTER else 'higher'} is better)",
        f"wall time    : {elapsed:.1f}s",
        "",
        "Ranking",
        "-" * 78,
    ]

    header = f"{'run':<42}{'rel_L2_roi':>11}{'rel_L1':>9}{'dice':>7}{'dyn':>6}{'chi2':>8}{'s':>7}"
    lines.append(header)
    for r in ok:
        m = r["metrics"]
        lines.append(
            f"{r['run']['id']:<42}"
            f"{m['rel_L2_roi']:>11.4f}{m['rel_L1']:>9.4f}{m['dice']:>7.3f}"
            f"{m['dynamic_range']:>6.2f}"
            f"{m.get('chi2_per_dof', float('nan')):>8.2f}"
            f"{m['runtime_s']:>7.1f}"
        )

    if ok:
        best = ok[0]
        lines += [
            "",
            f"Best: {best['run']['id']}  "
            f"({rank_by} = {best['metrics'][rank_by]:.4f})",
        ]

        # best per algorithm
        per_algo = {}
        for r in ok:
            a = r["run"]["algorithm"]
            cur = per_algo.get(a)
            better = cur is None or (
                (r["metrics"][rank_by] > cur["metrics"][rank_by]) if reverse
                else (r["metrics"][rank_by] < cur["metrics"][rank_by])
            )
            if better:
                per_algo[a] = r
        if len(per_algo) > 1:
            lines += ["", "Best per algorithm", "-" * 78]
            for a, r in sorted(per_algo.items()):
                lines.append(
                    f"  {a:<10} {r['run']['id']:<42} "
                    f"{rank_by}={r['metrics'][rank_by]:.4f}"
                )

    if failed:
        lines += ["", "Failed runs", "-" * 78]
        for r in failed:
            first = r["error"].strip().splitlines()[-1]
            lines.append(f"  {r['run']['id']}: {first}")

    lines += [
        "",
        "Notes",
        "-" * 78,
        "chi2 is the chi-square per degree of freedom of the data residual.",
        "  ~1  the reconstruction explains the data to within the noise",
        "  <<1 the noise is being fitted (under-regularised)",
        "  >>1 the data is not explained (over-regularised)",
        "dyn is the recovered contrast as a fraction of the true contrast;",
        "  EIT reconstructions are normally well below 1 (smoothing).",
        "",
        "Per-run outputs are in runs/<run_id>/ .",
        "The full parameter x metric table is runs.csv .",
    ]

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9 + min(len(ok), 20) + 3]))


# ──────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gt", help="ground-truth directory (overrides CONFIG)")
    ap.add_argument("--name", help="output run name")
    ap.add_argument("--algorithms", help="comma separated subset, e.g. gn,tv")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(CONFIG, default=gt_io._json_default))
    if args.gt:
        cfg["ground_truth"] = args.gt
    if args.name:
        cfg["output"]["run_name"] = args.name
    if args.algorithms:
        keep = {a.strip() for a in args.algorithms.split(",")}
        unknown = keep - set(cfg["algorithms"])
        if unknown:
            raise SystemExit(
                f"Unknown algorithm(s) {sorted(unknown)}; the config defines "
                f"{sorted(cfg['algorithms'])}"
            )
        cfg["algorithms"] = {k: v for k, v in cfg["algorithms"].items() if k in keep}

    run_experiment(cfg)


if __name__ == "__main__":
    main()
