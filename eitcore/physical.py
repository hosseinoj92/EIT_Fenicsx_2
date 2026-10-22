"""SI configuration, passive dielectric materials and electrode protocols.

Phasors use exp(+j omega t); currents and voltages are RMS. This module has
no FEniCSx dependency, allowing configuration and previews on Windows.
"""
from copy import deepcopy
import numpy as np
from mesh_tools import _merge, electrode_centres

EPSILON_0 = 8.8541878128e-12  # F/m
UNITS = {"length": "m", "frequency": "Hz", "current": "A RMS",
         "voltage": "V RMS", "conductivity": "S/m", "permittivity": "F/m",
         "relative_permittivity": "1", "contact_impedance": "ohm m^2",
         "contact_capacitance": "F/m^2", "relaxation_time": "s",
         "phase": "degree", "phasor_convention": "exp(+j omega t)"}


def default_config(dimension=2):
    return {"name": f"si_{dimension}d", "output_root": "results/physical",
            "mesh": {"dimension": dimension, "radius": 0.05, "height": 0.10,
                     "n_electrodes": 8 if dimension == 2 else 16,
                     "electrodes_per_ring": 8, "n_rings": 2, "ring_spacing": 0.035,
                     "electrode_diameter": 0.012, "electrode_width_deg": 14.0,
                     "h_center": 0.014, "h_edge": 0.004},
            "thickness": 0.02,
            "frequencies_hz": [1000.0, 10000.0, 100000.0],
            "drive": {"current_a": 0.001, "protocol": "mixed"},
            "contact": {"resistance_ohm_m2": 0.005, "capacitance_f_m2": 0.0},
            "background": {"conductivity_s_m": 0.2, "relative_permittivity": 10000.0},
            "inclusions": [{"shape": "circle" if dimension == 2 else "sphere",
                            "center_m": [0.015, 0.01] if dimension == 2 else [0.015, 0.01, 0.05],
                            "radius_m": 0.012,
                            "material": {"conductivity_s_m": 0.5,
                                         "relative_permittivity": 30000.0}}],
            "reconstruction": {"enabled": True, "frequency_hz": 10000.0,
                               "reconstruct_permittivity": True,
                               "h_center": 0.025, "h_edge": 0.008,
                               "iterations": 4, "regularization": 0.05,
                               "sigma_min_s_m": 0.001, "sigma_max_s_m": 2.0,
                               "noise_std_v": 0.0, "seed": 16},
            "units": dict(UNITS)}


BENCHMARK_DEFAULTS = {
    "algorithms": None, "frequencies_hz": None,
    "parameter_grid": None, "electrode_mesh_size": None, "h_band": None,
    "linear_regularization": 0.01, "tv_regularization": 0.001,
    "tv_beta": 0.01, "l1_alpha": 0.0001, "l1_iterations": 20,
    "measurement_stride": 4, "exclude_driven": True,
    "jacobian_chunk": 16, "output_directory": None,
    "threads": 4, "refinement_check": False,
    "refinement_factor": 0.75, "refinement_drives": 16,
}


def _positive(value, name, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def material_admittivity(material, frequency_hz):
    """Passive DC conduction plus optional single-Debye relaxation [S/m].

    epsilon_r is the high-frequency relative permittivity when delta_epsilon_r
    is supplied. No fitted tissue properties are implied by the examples.
    """
    unknown = set(material) - {"conductivity_s_m", "relative_permittivity",
                               "delta_epsilon_r", "relaxation_time_s"}
    if unknown:
        raise ValueError(f"Unknown material parameter(s): {sorted(unknown)}")
    sigma = _positive(material["conductivity_s_m"], "conductivity [S/m]", True)
    eps = _positive(material.get("relative_permittivity", 0), "relative permittivity", True)
    delta = _positive(material.get("delta_epsilon_r", 0), "delta_epsilon_r", True)
    f = _positive(frequency_hz, "frequency [Hz]", True)
    omega = 2 * np.pi * f
    tau = _positive(material.get("relaxation_time_s", 1), "relaxation time [s]")
    if delta and "relaxation_time_s" not in material:
        raise ValueError("Debye delta_epsilon_r requires relaxation_time_s [s]")
    gamma = sigma + 1j*omega*EPSILON_0*(eps + delta/(1+1j*omega*tau))
    if gamma == 0:
        raise ValueError("An ideal insulator needs nonzero frequency and permittivity; gamma cannot vanish")
    return gamma


def inclusion_mask(inc, points):
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]
    center = np.asarray(inc["center_m"], dtype=float)
    if center.shape != (dim,) or not np.all(np.isfinite(center)):
        raise ValueError(f"center_m must contain {dim} finite coordinates [m]")
    x = points - center
    shape = inc["shape"]
    if shape in ("circle", "sphere"):
        if (shape == "circle") != (dim == 2):
            raise ValueError("Use circle in 2D and sphere in 3D")
        axes = _positive(inc["radius_m"], "inclusion radius [m]")
        return np.sum((x/axes)**2, axis=1) <= 1
    if shape in ("ellipse", "ellipsoid", "box"):
        key = "size_m" if shape == "box" else "semi_axes_m"
        axes = np.asarray(inc[key], dtype=float)
        if axes.shape != (dim,) or np.any(~np.isfinite(axes)) or np.any(axes <= 0):
            raise ValueError(f"{key} must contain {dim} positive lengths [m]")
        if shape == "box":
            return np.max(np.abs(x)/(axes/2), axis=1) <= 1
        if (shape == "ellipse") != (dim == 2):
            raise ValueError("Use ellipse in 2D and ellipsoid in 3D")
        return np.sum((x/axes)**2, axis=1) <= 1
    raise ValueError(f"Unsupported inclusion shape: {shape}")


def sample_materials(cfg, points, frequency_hz):
    """Return effective conductivity [S/m] and epsilon_r on arbitrary cells."""
    gamma = np.full(len(points), material_admittivity(cfg["background"], frequency_hz), complex)
    for inc in cfg.get("inclusions", []):
        gamma[inclusion_mask(inc, points)] = material_admittivity(inc["material"], frequency_hz)
    omega = 2*np.pi*frequency_hz
    # At DC epsilon is unobservable; evaluate its zero-frequency limit.
    if omega:
        eps = gamma.imag/(omega*EPSILON_0)
    else:
        eps = np.full(len(points), cfg["background"].get("relative_permittivity", 0)
                      + cfg["background"].get("delta_epsilon_r", 0), dtype=float)
        for inc in cfg.get("inclusions", []):
            mat = inc["material"]
            eps[inclusion_mask(inc, points)] = mat.get("relative_permittivity", 0)+mat.get("delta_epsilon_r", 0)
    return gamma.real, eps


def normalize_config(config):
    cfg = deepcopy(config)
    allowed = set(default_config())
    if set(cfg) - allowed:
        raise ValueError(f"Unknown configuration keys: {sorted(set(cfg)-allowed)}")
    if not cfg.get("name") or any(c in cfg["name"] for c in '/\\:') or cfg["name"] in (".", ".."):
        raise ValueError("name must be a folder name, without path separators")
    cfg["mesh"] = _merge(cfg["mesh"])
    for section, keys in {
        "drive": {"current_a", "protocol", "stride"},
        "contact": {"resistance_ohm_m2", "capacitance_f_m2"},
        "reconstruction": set(default_config()["reconstruction"]) | set(BENCHMARK_DEFAULTS),
    }.items():
        unknown = set(cfg[section]) - keys
        if unknown:
            raise ValueError(f"Unknown {section} parameter(s): {sorted(unknown)}")
    cfg["thickness"] = _positive(cfg["thickness"], "2D thickness [m]")
    fs = np.asarray(cfg["frequencies_hz"], dtype=float)
    if fs.ndim != 1 or not len(fs) or np.any(~np.isfinite(fs)) or np.any(fs < 0):
        raise ValueError("frequencies_hz must be a nonempty list of finite nonnegative frequencies")
    if len(np.unique(fs)) != len(fs):
        raise ValueError("frequencies_hz contains duplicates")
    cfg["frequencies_hz"] = fs.tolist()
    _positive(cfg["drive"]["current_a"], "current [A RMS]")
    stride = cfg["drive"].get("stride", 1)
    if int(stride) != stride or stride < 1:
        raise ValueError("drive.stride must be a positive integer")
    if cfg["drive"]["protocol"] not in ("adjacent", "opposite", "axial", "mixed"):
        raise ValueError("protocol must be adjacent, opposite, axial or mixed")
    n = cfg["mesh"]["n_electrodes"] if cfg["mesh"]["dimension"] == 2 else cfg["mesh"]["electrodes_per_ring"]
    if cfg["drive"]["protocol"] == "opposite" and n % 2:
        raise ValueError("opposite protocol needs an even number of electrodes per ring")
    if cfg["drive"]["protocol"] == "axial" and (cfg["mesh"]["dimension"] == 2 or cfg["mesh"]["n_rings"] < 2):
        raise ValueError("axial protocol requires 3D and at least two rings")
    L = cfg["mesh"]["n_electrodes"]
    for key, zero in (("resistance_ohm_m2", False), ("capacitance_f_m2", True)):
        arr = np.asarray(cfg["contact"][key], dtype=float)
        if arr.ndim > 1 or (arr.ndim == 1 and arr.shape != (L,)):
            raise ValueError(f"{key} must be a scalar or length {L}")
        for v in arr.reshape(-1):
            _positive(v, key, zero)
    dim = cfg["mesh"]["dimension"]
    # Validate all materials/geometries before expensive meshing.
    sample_materials(cfg, np.zeros((1, dim)), fs[0])
    for f in fs:
        material_admittivity(cfg['background'],f)
        for inc in cfg.get('inclusions',[]):
            material_admittivity(inc['material'],f)
    for inc in cfg.get("inclusions", []):
        center = np.asarray(inc["center_m"])
        axes = np.full(dim, inc["radius_m"]) if inc["shape"] in ("circle", "sphere") else np.asarray(inc["size_m"])/2 if inc["shape"] == "box" else np.asarray(inc["semi_axes_m"])
        # Conservative bounding check ensures the whole inclusion fits.
        if np.linalg.norm(center[:2]) + np.max(axes[:2]) >= cfg["mesh"]["radius"]:
            raise ValueError("Inclusion must fit inside the cylinder radius [m]")
        if dim == 3 and (center[2]-axes[2] <= 0 or center[2]+axes[2] >= cfg["mesh"]["height"]):
            raise ValueError("Inclusion must fit between end caps [m]")
    r = cfg["reconstruction"]
    if r.get("algorithms") is not None:
        for key, value in BENCHMARK_DEFAULTS.items():
            r.setdefault(key, deepcopy(value))
        if not isinstance(r["algorithms"], list) or not r["algorithms"] or set(r["algorithms"]) - {"gn", "tv", "linear", "l1", "dbar"}:
            raise ValueError("algorithms must list gn, tv, linear, l1 and/or dbar")
        if len(set(r['algorithms']))!=len(r['algorithms']):
            raise ValueError('algorithms must not contain duplicates')
        if cfg["mesh"]["dimension"] != 3:
            raise ValueError("The scalable benchmark currently requires a 3D mesh")
        if r["reconstruct_permittivity"]:
            raise ValueError("The scalable algorithm comparison currently reconstructs conductivity with fixed background permittivity")
        for key in ("measurement_stride", "jacobian_chunk", "l1_iterations", "threads"):
            if int(r[key]) != r[key] or r[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < r['refinement_factor'] < 1 or int(r['refinement_drives']) != r['refinement_drives'] or r['refinement_drives'] < 1:
            raise ValueError('refinement_factor must be in (0,1), and refinement_drives a positive integer')
        for key in ("linear_regularization", "tv_regularization", "tv_beta", "l1_alpha"):
            _positive(r[key], key)
        if r["frequencies_hz"] is None:
            r["frequencies_hz"] = list(fs)
        if not r["frequencies_hz"] or any(f not in fs for f in r["frequencies_hz"]):
            raise ValueError("All reconstruction frequencies must appear in frequencies_hz")
        grid = r["parameter_grid"]
        if not isinstance(grid, dict) or set(grid) != {"nx", "ny", "z_edges_m"}:
            raise ValueError("parameter_grid needs nx, ny and z_edges_m")
        for key in ("nx", "ny"):
            if int(grid[key]) != grid[key] or grid[key] < 2:
                raise ValueError(f"parameter_grid.{key} must be an integer >= 2")
        edges = np.asarray(grid["z_edges_m"], float)
        if edges.ndim != 1 or len(edges) < 3 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0) or edges[0] != 0 or edges[-1] != cfg["mesh"]["height"]:
            raise ValueError("z_edges_m must increase from 0 to the column height")
    for key in ("enabled", "reconstruct_permittivity"):
        if key in r and not isinstance(r[key], bool):
            raise ValueError(f"{key} must be true or false")
    r.setdefault("reconstruct_permittivity", True)
    _positive(r["frequency_hz"], "reconstruction frequency [Hz]", True)
    if int(r["seed"]) != r["seed"] or r["seed"] < 0:
        raise ValueError("noise seed must be a nonnegative integer")
    for key in ("h_center", "h_edge", "regularization", "sigma_min_s_m", "sigma_max_s_m"):
        _positive(r[key], key)
    _positive(r["noise_std_v"], "noise standard deviation [V]", True)
    if int(r["iterations"]) != r["iterations"] or r["iterations"] < 1:
        raise ValueError("iterations must be a positive integer")
    if r["sigma_min_s_m"] >= r["sigma_max_s_m"]:
        raise ValueError("Reconstruction conductivity bounds must increase")
    inverse_mesh=dict(cfg["mesh"], h_center=r["h_center"], h_edge=r["h_edge"])
    if r.get('algorithms'):
        inverse_mesh.update({k:r[k] for k in ('h_band','electrode_mesh_size') if r.get(k) is not None})
    _merge(inverse_mesh)
    if r["enabled"] and not r.get('algorithms') and r["frequency_hz"] not in fs:
        raise ValueError("Reconstruction frequency must be included in frequencies_hz")
    if r["enabled"]:
        for frequency in (r['frequencies_hz'] if r.get('algorithms') else [r['frequency_hz']]):
            bg_sigma = material_admittivity(cfg["background"], frequency).real
            if not r["sigma_min_s_m"] <= bg_sigma <= r["sigma_max_s_m"]:
                raise ValueError("Reconstruction bounds must include the background conductivity at every inversion frequency")
    if cfg.get("units", UNITS) != UNITS:
        raise ValueError("All inputs must use the documented SI units; units metadata cannot redefine them")
    cfg["units"] = dict(UNITS)
    return cfg


def drive_patterns(cfg):
    """Pair drives, ring-major numbering; mixed adds vertical neighbouring-ring pairs."""
    m, d = cfg["mesh"], cfg["drive"]
    n = m["n_electrodes"] if m["dimension"] == 2 else m["electrodes_per_ring"]
    rings = 1 if m["dimension"] == 2 else m["n_rings"]
    protocol = d["protocol"]
    pairs = []
    if protocol != "axial":
        for r in range(rings):
            step = n//2 if protocol == "opposite" else 1
            for e in range(n//2 if protocol == "opposite" else n):
                pairs.append((r*n+e, r*n+(e+step)%n))
    if protocol in ("axial", "mixed"):
        pairs.extend((r*n+e, (r+1)*n+e) for r in range(rings-1) for e in range(n))
    stride = int(d.get("stride", 1))
    if stride > 1:
        # Subsample azimuthally within each ring/family, not across ring heights.
        pairs = [pair for pair in pairs if pair[0] % n % stride == 0]
    I = np.zeros((len(pairs), m["n_electrodes"]))
    for row, (a, b) in zip(I, pairs):
        row[a], row[b] = d["current_a"], -d["current_a"]
    return I
