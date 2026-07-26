# phantoms.py

"""
Declarative conductivity phantoms.

A phantom is a dictionary; ``build_phantom(cfg, x, y)`` evaluates it at the
given points and returns the conductivity there.  Because it is a pure
function of the coordinates, the *same* phantom can be sampled on a fine
"truth" mesh and on a coarse reconstruction mesh, which is what keeps the
ground truth mesh-independent.

    PHANTOM = {
        "background": 1.0,
        "inclusions": [
            {"shape": "ellipse", "center": [0.40, 0.30],
             "semi_axes": [0.25, 0.15], "angle_deg": 30.0, "value": 3.0},
            {"shape": "circle", "center": [-0.40, -0.25],
             "radius": 0.22, "value": 0.2},
        ],
    }

Supported inclusion shapes
--------------------------
``circle``     ``center``, ``radius``
``ellipse``    ``center``, ``semi_axes`` [a, b], ``angle_deg``
``annulus``    ``center``, ``radii`` [r_inner, r_outer]
``rectangle``  ``center``, ``size`` [w, h], ``angle_deg``
``halfplane``  ``point``, ``normal``

Every inclusion carries either a constant ``value`` or a ``profile``:

``{"type": "constant", "value": v}``
``{"type": "linear",      "center_value": vc, "edge_value": ve}``
``{"type": "exponential", "center_value": vc, "edge_value": ve, "exponent": p}``
``{"type": "gaussian",    "center_value": vc, "edge_value": ve, "sigma": s}``

The background accepts the same profile syntax with ``radial`` semantics, so
gradients across the whole domain are expressible too.

Inclusions are painted in list order, so later entries overwrite earlier ones
where they overlap.
"""

import numpy as np

PRESETS = {
    "two_inclusions": {
        "background": 1.0,
        "inclusions": [
            {"shape": "circle", "center": [0.40, 0.35], "radius": 0.22, "value": 3.0},
            {"shape": "circle", "center": [-0.40, -0.30], "radius": 0.25, "value": 0.2},
        ],
    },
    "three_inclusions": {
        "background": 1.0,
        "inclusions": [
            {"shape": "ellipse", "center": [0.42, 0.30], "semi_axes": [0.26, 0.16],
             "angle_deg": 25.0, "value": 3.0},
            {"shape": "circle", "center": [-0.42, 0.28], "radius": 0.20, "value": 0.15},
            {"shape": "rectangle", "center": [0.0, -0.45], "size": [0.55, 0.25],
             "angle_deg": 0.0, "value": 2.2},
        ],
    },
    "concentric": {
        "background": 1.0,
        "inclusions": [
            {"shape": "annulus", "center": [0.0, 0.0], "radii": [0.45, 0.65],
             "value": 0.25},
            {"shape": "circle", "center": [0.0, 0.0], "radius": 0.25, "value": 3.0},
        ],
    },
    "graded_blob": {
        "background": 1.0,
        "inclusions": [
            {"shape": "circle", "center": [0.0, 0.0], "radius": 0.55,
             "profile": {"type": "gaussian", "center_value": 4.0,
                         "edge_value": 1.0, "sigma": 0.45}},
        ],
    },
    "homogeneous": {"background": 1.0, "inclusions": []},
}


# --------------------------------------------------------------------------- #
#  geometry: normalised radius r in [0, 1] inside the shape, >1 outside
# --------------------------------------------------------------------------- #
def _normalised_radius(inc, x, y):
    shape = inc.get("shape", "circle").lower()

    if shape == "circle":
        cx, cy = inc["center"]
        r = np.hypot(x - cx, y - cy) / float(inc["radius"])
        return r

    if shape == "ellipse":
        cx, cy = inc["center"]
        a, b = inc["semi_axes"]
        ang = np.deg2rad(float(inc.get("angle_deg", 0.0)))
        ca, sa = np.cos(ang), np.sin(ang)
        xr = (x - cx) * ca + (y - cy) * sa
        yr = -(x - cx) * sa + (y - cy) * ca
        return np.sqrt((xr / a) ** 2 + (yr / b) ** 2)

    if shape == "annulus":
        cx, cy = inc["center"]
        r_in, r_out = inc["radii"]
        rr = np.hypot(x - cx, y - cy)
        inside = (rr >= r_in) & (rr <= r_out)
        # normalised radius across the ring thickness (0 at mid-ring, 1 at walls)
        mid = 0.5 * (r_in + r_out)
        half = max(0.5 * (r_out - r_in), 1e-12)
        out = np.abs(rr - mid) / half
        return np.where(inside, out, 2.0)

    if shape == "rectangle":
        cx, cy = inc["center"]
        w, h = inc["size"]
        ang = np.deg2rad(float(inc.get("angle_deg", 0.0)))
        ca, sa = np.cos(ang), np.sin(ang)
        xr = (x - cx) * ca + (y - cy) * sa
        yr = -(x - cx) * sa + (y - cy) * ca
        return np.maximum(np.abs(xr) / (0.5 * w), np.abs(yr) / (0.5 * h))

    if shape == "halfplane":
        px, py = inc.get("point", [0.0, 0.0])
        nx, ny = inc.get("normal", [1.0, 0.0])
        n = np.hypot(nx, ny)
        d = ((x - px) * nx + (y - py) * ny) / max(n, 1e-12)
        # 0 on the plane, <1 on the selected side
        return np.where(d <= 0, 0.0, 2.0)

    raise ValueError(
        f"Unknown inclusion shape '{shape}'. Choices: circle, ellipse, "
        "annulus, rectangle, halfplane"
    )


# --------------------------------------------------------------------------- #
#  radial value profiles
# --------------------------------------------------------------------------- #
def _profile_values(spec, r):
    """Evaluate a value profile at normalised radius ``r`` (0 = centre)."""
    if not isinstance(spec, dict):
        return np.full_like(r, float(spec))

    kind = spec.get("type", "constant").lower()
    if kind == "constant":
        return np.full_like(r, float(spec["value"]))

    vc = float(spec["center_value"])
    ve = float(spec["edge_value"])

    if kind == "linear":
        return vc + (ve - vc) * np.clip(r, 0.0, 1.0)
    if kind == "exponential":
        p = float(spec.get("exponent", 2.0))
        return vc + (ve - vc) * np.clip(r, 0.0, 1.0) ** p
    if kind == "gaussian":
        s = float(spec.get("sigma", 0.5))
        return ve + (vc - ve) * np.exp(-(r**2) / (2.0 * s**2))

    raise ValueError(
        f"Unknown profile type '{kind}'. Choices: constant, linear, "
        "exponential, gaussian"
    )


def _inclusion_values(inc, r):
    if "profile" in inc and inc["profile"] is not None:
        return _profile_values(inc["profile"], r)
    return np.full_like(r, float(inc["value"]))


# --------------------------------------------------------------------------- #
#  random placement
# --------------------------------------------------------------------------- #
def sample_random_inclusions(spec, rng, domain_radius=1.0):
    """
    Draw a list of inclusion dictionaries.

    spec keys: ``n`` (or ``n_range``), ``shape``, ``semi_axes_range``,
    ``value_low_range``, ``value_high_range``, ``boundary_margin``,
    ``pair_margin``, ``regions``.
    """
    n = spec.get("n")
    if n is None:
        lo, hi = spec.get("n_range", (1, 3))
        n = int(rng.integers(lo, hi + 1))

    a_range = spec.get("semi_axes_range", (0.15, 0.30))
    b_range = spec.get("semi_axes_range_b", a_range)
    low = spec.get("value_low_range", (0.05, 0.35))
    high = spec.get("value_high_range", (2.0, 3.5))
    bnd = float(spec.get("boundary_margin", 0.10))
    pair = float(spec.get("pair_margin", 0.08))
    regions = spec.get("regions", ["random"])
    shape = spec.get("shape", "ellipse")

    placed = []
    out = []
    for i in range(n):
        a = float(rng.uniform(*a_range))
        b = float(rng.uniform(*b_range))
        region = regions[i % len(regions)]

        for _ in range(2000):
            cx, cy = _sample_center(region, rng, domain_radius)
            rmax = max(a, b)
            if cx**2 + cy**2 > (domain_radius - bnd - rmax) ** 2:
                continue
            if any(
                (cx - px) ** 2 + (cy - py) ** 2 < (rmax + pr + pair) ** 2
                for px, py, pr in placed
            ):
                continue
            placed.append((cx, cy, rmax))
            break
        else:
            raise RuntimeError(
                f"Could not place inclusion {i + 1}/{n}: reduce the number or "
                "the size of the inclusions, or the margins."
            )

        value = (
            float(rng.uniform(*low))
            if rng.random() < 0.5
            else float(rng.uniform(*high))
        )
        inc = {
            "shape": shape,
            "center": [cx, cy],
            "value": value,
        }
        if shape == "ellipse":
            inc["semi_axes"] = [a, b]
            inc["angle_deg"] = float(rng.uniform(0.0, 360.0))
        elif shape == "circle":
            inc["radius"] = a
        elif shape == "rectangle":
            inc["size"] = [2 * a, 2 * b]
            inc["angle_deg"] = float(rng.uniform(0.0, 360.0))
        out.append(inc)

    return out


def _sample_center(region, rng, R):
    region = str(region).lower()
    if region == "random":
        return float(rng.uniform(-0.85 * R, 0.85 * R)), float(
            rng.uniform(-0.85 * R, 0.85 * R)
        )
    if region == "center":
        rad = float(rng.uniform(0, 0.3 * R))
        phi = float(rng.uniform(0, 2 * np.pi))
        return rad * np.cos(phi), rad * np.sin(phi)
    xs = 1.0 if region.endswith("r") else -1.0
    ys = 1.0 if region.startswith("u") else -1.0
    return float(rng.uniform(0.15, 0.75) * R * xs), float(
        rng.uniform(0.15, 0.75) * R * ys
    )


# --------------------------------------------------------------------------- #
#  main API
# --------------------------------------------------------------------------- #
def resolve_phantom(cfg, domain_radius=1.0):
    """
    Turn a phantom config into a fully explicit one (random placement resolved,
    preset expanded).  Storing the resolved form in the ground-truth bundle is
    what makes a run reproducible without re-running the RNG.
    """
    cfg = dict(cfg or {})

    if "preset" in cfg and cfg["preset"]:
        preset = PRESETS.get(cfg["preset"])
        if preset is None:
            raise ValueError(
                f"Unknown preset '{cfg['preset']}'. Choices: {sorted(PRESETS)}"
            )
        base = {k: v for k, v in preset.items()}
        base.update({k: v for k, v in cfg.items() if k != "preset"})
        cfg = base

    if cfg.get("random"):
        rng = np.random.default_rng(cfg.get("seed", 0))
        cfg = dict(cfg)
        cfg["inclusions"] = sample_random_inclusions(
            cfg["random"], rng, domain_radius=domain_radius
        )
        cfg.pop("random", None)

    cfg.setdefault("background", 1.0)
    cfg.setdefault("inclusions", [])
    return cfg


def build_phantom(cfg, x, y, domain_radius=1.0):
    """
    Evaluate a (resolved or unresolved) phantom config at points ``x``, ``y``.

    Returns a float array of conductivities with the shape of ``x``.
    """
    cfg = resolve_phantom(cfg, domain_radius=domain_radius)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    bg = cfg["background"]
    if isinstance(bg, dict):
        r = np.hypot(x, y) / domain_radius
        sigma = _profile_values(bg, r)
    else:
        sigma = np.full(x.shape, float(bg))

    for inc in cfg["inclusions"]:
        r = _normalised_radius(inc, x, y)
        inside = r <= 1.0
        if not np.any(inside):
            continue
        sigma[inside] = _inclusion_values(inc, r[inside])

    lo = cfg.get("clip_min")
    hi = cfg.get("clip_max")
    if lo is not None or hi is not None:
        sigma = np.clip(sigma, lo if lo is not None else -np.inf,
                        hi if hi is not None else np.inf)

    if not np.all(np.isfinite(sigma)):
        raise ValueError("Phantom produced non-finite conductivity values")
    if np.any(sigma <= 0):
        raise ValueError(
            "Phantom produced non-positive conductivity; the forward problem "
            "requires sigma > 0 everywhere."
        )

    return sigma


def describe_phantom(cfg):
    """One-line human readable summary, used in figure titles and log files."""
    cfg = resolve_phantom(cfg)
    bg = cfg["background"]
    bg_s = f"{bg:g}" if not isinstance(bg, dict) else f"{bg.get('type')} bg"
    parts = []
    for inc in cfg["inclusions"]:
        v = inc.get("value")
        vs = f"{v:g}" if v is not None else inc.get("profile", {}).get("type", "?")
        parts.append(f"{inc.get('shape', 'circle')}@{np.round(inc['center'], 2).tolist()}={vs}")
    return f"bg={bg_s}; " + ("; ".join(parts) if parts else "no inclusions")
