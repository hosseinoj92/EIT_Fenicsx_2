# EIT Simulation & Reconstruction Framework

A Python framework for simulating **Electrical Impedance Tomography** data on a
cylindrical test cell and benchmarking reconstruction algorithms against a
known ground truth.

It exists to answer one question reproducibly:

> *Which algorithm, with which parameters, on which mesh, at which noise level,
> reconstructs best?*

---

## Table of contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [How the pieces fit together](#3-how-the-pieces-fit-together)
4. [Step 0 — `mesh_tools.py`](#4-step-0--mesh_toolspy)
5. [Step 1 — `make_ground_truth.py`](#5-step-1--make_ground_truthpy)
6. [Step 2 — `simulate.py`](#6-step-2--simulatepy)
7. [Sweeping over parameters](#7-sweeping-over-parameters)
8. [The algorithms and their parameters](#8-the-algorithms-and-their-parameters)
9. [How to choose parameters](#9-how-to-choose-parameters)
10. [Metrics — what they mean](#10-metrics--what-they-mean)
11. [Output files](#11-output-files)
12. [The physics](#12-the-physics)
13. [Testing](#13-testing)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Installation

Requires Python 3.10+ and FEniCSx.

```bash
conda create -n eit_env python=3.10
conda activate eit_env
conda install -c conda-forge fenics-dolfinx mpich pyvista gmsh python-gmsh
pip install torch numpy scipy matplotlib tqdm
```

Verify:

```bash
python -c "import dolfinx, gmsh, torch; print(dolfinx.__version__)"
python tests/test_framework.py          # 115 checks
```

---

## 2. Quick start

Three commands, in order:

```bash
python mesh_tools.py                    # (optional) preview mesh resolutions
python make_ground_truth.py             # -> GT_simulated_result/<name>/
python simulate.py                      # -> results/<name>_<timestamp>/
```

Everything is configured by editing the `CONFIG` dictionary at the top of the
two scripts. Nothing else needs touching.

Useful flags:

```bash
python make_ground_truth.py --name my_case --preset three_inclusions --seed 42
python simulate.py --gt GT_simulated_result/my_case --name my_experiment
python simulate.py --algorithms gn,tv   # run a subset of the configured ones
python mesh_tools.py --h-center 0.12 --h-edge 0.08 --electrodes 16
```

---

## 3. How the pieces fit together

```
   mesh_tools.py ──────────┐  builds circular meshes (used by both scripts)
                           │
   make_ground_truth.py ───┤  fine mesh + phantom + forward solve
        │                  │        ↓
        │            GT_simulated_result/<name>/     ← the stored truth + data
        │                  │        ↓
   simulate.py ────────────┘  coarse mesh + noise + reconstruct + compare
                                     ↓
                             results/<run_name>/
```

**Why two scripts?** The data is generated on a *fine* mesh and reconstructed on
a *different, coarser* one. Simulating and inverting on the same discretisation
is the classic **inverse crime**: the discretisation error cancels exactly,
every algorithm looks better than it really is, and — the fatal part for a
benchmarking tool — the *ranking between algorithms* shifts. Storing the truth
once, on its own mesh, is what keeps the comparison honest.

### Directory layout

```
mesh_tools.py            mesh generation           ← entry point
make_ground_truth.py     truth generation          ← entry point
simulate.py              reconstruction + sweeps   ← entry point
README.md
eitcore/                 the library (you rarely need to open this)
    eit_forward_fenicsx.py   CEM forward solver + adjoint Jacobian
    gauss_newton.py          GN, GN-TV, one-step linearised
    sparsity_reconstruction.py  L1 sparsity
    regulariser.py           prior operators
    algorithms.py            algorithm registry
    phantoms.py              declarative conductivity phantoms
    noise.py                 measurement-noise models
    metrics.py               mesh-independent quality metrics
    gt_io.py                 ground-truth bundle format
    reconstructor.py         base class
    utils.py                 drive patterns, interpolation
tests/test_framework.py  validation suite
data/meshes/             generated meshes (cached, regenerable)
GT_simulated_result/     ground-truth bundles
results/                 experiment outputs
```

---

## 4. Step 0 — `mesh_tools.py`

**The test cell is a cylinder, so the modelled cross-section is always a
circle.** There is no shape option; anything else is rejected with an error.

Run it to see what a given resolution costs before committing to it:

```
$ python mesh_tools.py
  very coarse   cells=   600  nodes=   421  area=3.1412 (exact 3.1416)  quality min/mean=0.34/0.81
  coarse        cells=   858  nodes=   550  area=3.1412 (exact 3.1416)  quality min/mean=0.44/0.88
  medium        cells=  1076  nodes=   659  area=3.1412 (exact 3.1416)  quality min/mean=0.53/0.91
  fine          cells=  1926  nodes=  1084  area=3.1412 (exact 3.1416)  quality min/mean=0.61/0.96
  truth (fine)  cells=  5888  nodes=  3113  area=3.1412 (exact 3.1416)  quality min/mean=0.65/0.98
```

### Meshing parameters

These same keys appear in `make_ground_truth.py` → `CONFIG["mesh"]` and in
`simulate.py` → `CONFIG["recon_mesh"]`.

| parameter | meaning | typical |
|---|---|---|
| `radius` | radius of the circular cross-section | `1.0` |
| `n_electrodes` | electrodes equally spaced around the rim | `16`, `32` |
| `electrode_width_deg` | angular width of **one** electrode | `10`–`15` |
| `electrode_offset_deg` | rotates the whole electrode ring | `0.0` |
| `n_per_electrode` | boundary nodes across one electrode | `8`–`10` |
| `n_per_gap` | boundary nodes across one gap | `4`–`5` |
| `h_center` | target element size at the centre | see below |
| `h_edge` | target element size at the rim | see below |
| `grading` | `size(r) = h_c + (h_e − h_c)·(r/R)^grading` | `1.0` |
| `algorithm` | gmsh 2-D meshing algorithm | `6` (Frontal-Delaunay) |
| `optimize` | run the Netgen quality optimiser | `True` |

**Geometry vs. discretisation.** The first four are *physical* — they describe
the actual cell. The rest only describe how finely that cell is chopped up.
`simulate.py` inherits the physical ones from the ground truth (changing them
would mean reconstructing a different experiment than the one measured) and
lets you set the discretisation freely.

**Choosing `h_center` / `h_edge`.** EIT is most sensitive near the electrodes,
so `h_edge < h_center` is the usual choice. Smaller is not automatically better
for reconstruction: the Gauss-Newton normal equations are **dense**
`n_cells × n_cells`, so cost grows as `n_cells³`. Above ~4000 cells a
reconstruction gets slow, and above ~12000 the solver warns.

Meshes are cached by a hash of their parameters, so re-running a sweep never
re-meshes.

---

## 5. Step 1 — `make_ground_truth.py`

Builds a fine mesh, paints a phantom on it, solves the forward problem, and
writes a self-contained bundle.

### `CONFIG` sections

| key | meaning |
|---|---|
| `name` | bundle folder name under `output_root` |
| `random_seed` | seeds random phantom placement |
| `mesh` | the truth mesh — see the table above. **Keep it fine.** |
| `L` | number of electrodes; must equal `mesh.n_electrodes` |
| `contact_impedance` | `z` per electrode, scalar or list of `L` |
| `drive.method` | current-injection pattern, `1`–`5`, see below |
| `drive.n_patterns` | `None` → the canonical maximum |
| `drive.amplitude` | injected current |
| `phantom` | the ground truth itself, see below |
| `plot` | preview figure options |

### Drive patterns

| method | name | patterns | notes |
|---|---|---|---|
| 1 | opposite | `L/2` | `+1`/`−1` on electrodes `i`, `i+L/2`. Needs even `L`. |
| 2 | adjacent | `L−1` | `+1`/`−1` on neighbours. The classic, best near the boundary. |
| 3 | one-against-all | `L−1` | `+1` on one, `−1/(L−1)` on the rest. |
| 4 | trigonometric | `L−1` | Cheney/Isaacson basis. Best distinguishability, best interior sensitivity. |
| 5 | all-against-one | `L−1` | Everything referenced to electrode 0. |

A CEM measurement set can never contain more than `L−1` independent patterns
(charge conservation excludes the all-ones vector), so no method returns more.
Every pattern is checked for charge conservation, rank and zero rows.

**Which to use?** `2` (adjacent) for boundary-dominated targets, `4`
(trigonometric) when you care about the centre of the cell.

### Phantoms

Declarative, and a pure function of position, which is what lets the *same*
phantom be sampled on both the fine truth mesh and the coarse reconstruction
mesh.

```python
"phantom": {
    "background": 1.0,
    "inclusions": [
        {"shape": "circle",  "center": [0.42, 0.34], "radius": 0.22, "value": 3.0},
        {"shape": "ellipse", "center": [-0.40, -0.28],
         "semi_axes": [0.28, 0.17], "angle_deg": 35.0, "value": 0.2},
    ],
}
```

Inclusion shapes (these are *objects inside* the circular cell — they can be any
shape):

| shape | keys |
|---|---|
| `circle` | `center`, `radius` |
| `ellipse` | `center`, `semi_axes` `[a, b]`, `angle_deg` |
| `annulus` | `center`, `radii` `[r_in, r_out]` |
| `rectangle` | `center`, `size` `[w, h]`, `angle_deg` |
| `halfplane` | `point`, `normal` |

Each takes a constant `value`, or a radial `profile`:

```python
"profile": {"type": "linear",      "center_value": 4.0, "edge_value": 1.0}
"profile": {"type": "exponential", "center_value": 4.0, "edge_value": 1.0, "exponent": 2.0}
"profile": {"type": "gaussian",    "center_value": 4.0, "edge_value": 1.0, "sigma": 0.45}
```

Inclusions are painted in list order, so later ones overwrite earlier ones where
they overlap.

Three alternatives to an explicit list:

```python
"phantom": {"preset": "three_inclusions"}     # two_inclusions, three_inclusions,
                                              # concentric, graded_blob, homogeneous
"phantom": {"background": 1.0, "seed": 16,    # random placement
            "random": {"n_range": (1, 3), "shape": "ellipse",
                       "semi_axes_range": (0.15, 0.30),
                       "value_low_range": (0.05, 0.35),
                       "value_high_range": (2.0, 3.5),
                       "boundary_margin": 0.10, "pair_margin": 0.08,
                       "regions": ["random"]}}   # UL UR LL LR center random
```

Random placement is resolved once and stored **frozen** in the bundle, so a run
is reproducible without re-running the RNG.

---

## 6. Step 2 — `simulate.py`

Loads a bundle, adds noise, reconstructs on a coarser mesh with any number of
algorithms and settings, scores everything.

### `CONFIG` sections

#### `ground_truth`
Path to the bundle directory.

#### `noise`

| parameter | meaning |
|---|---|
| `model` | see below |
| `level_percent` | noise level in percent |
| `seed` | fixed → **every algorithm sees exactly the same noise**, which is what makes the comparison fair |
| `floor_fraction` | adds a constant floor on top of a proportional model |
| `electrode_gain_percent` | systematic per-electrode gain error |
| `electrode_offset` | systematic per-electrode offset |
| `use_statistical_weighting` | feed `1/variance` to the reconstructors |
| `weight_relative_floor` | regularises that weighting |

Models:

- **`relative_to_max`** *(default)* — `σ_k = p · max|U|`, a constant noise floor.
- `relative_to_pattern_max` — floor set per current pattern.
- `relative_per_measurement` — `σ_k = p·|U_k|`.
- `absolute` — `level` is taken as an absolute voltage.
- `none` — noise-free.

The default is not `relative_per_measurement` on purpose. Scaling noise with
each individual measurement makes the channels that happen to sit near a voltage
zero-crossing *arbitrarily accurate*. No real instrument behaves that way — a
real EIT system has a noise floor set by its front-end, roughly constant across
the channels of one frame — and the unphysical version flatters algorithms
unequally, which corrupts exactly the comparison this framework exists to make.

`electrode_gain_percent` models calibration error. Unlike white noise it is
*correlated across patterns*, and it is usually what limits absolute EIT in
practice, so it is worth including in any realistic study.

#### `recon_mesh`

A list of mesh dicts (a list ⇒ a sweep over meshes). Set the discretisation keys;
the physical cell is inherited from the ground truth. Give each one a `label`
for use in run names and charts.

The script reports the truth/reconstruction cell ratio and warns if the
reconstruction mesh is not actually coarser.

#### `common`
Applied to every algorithm: `device`, `sigma_min`/`sigma_max` (the clip range
the reconstruction is confined to), `background` (`None` → from the ground
truth), `verbose`.

#### `algorithms`
Which methods to run and with what. Comment an entry out to skip it.

#### `metrics`
`grid_resolution` of the common evaluation grid, `roi_threshold`, and `rank_by`
— the metric used to sort results and pick the best.

#### `plot`
`cmap`, `vmin`/`vmax` (`None` → ground-truth range), `max_cols`, `dpi`, and
switches for the three figure types.

---

## 7. Sweeping over parameters

**Any value written as a Python list becomes a sweep axis.** The cartesian
product is run, and each result is named after *only* the axes that actually
vary.

```python
"noise":      {"level_percent": [0.5, 2.0]},
"recon_mesh": [{"label": "coarse", "h_center": 0.16, "h_edge": 0.11},
               {"label": "medium", "h_center": 0.12, "h_edge": 0.08}],
"algorithms": {"gn": {"lambda": [0.03, 0.1, 0.3]},
               "tv": {"lambda": [3.0, 10.0]}},
```

→ 20 runs, named `gn_lambda=0.03_mesh=coarse_level_percent=0.5`, …

Details that make this pleasant in practice:

- A config with a single value everywhere gives **one** run, named just `gn`.
- A parameter that merely *differs between* algorithms (`gn` using 12 iterations
  while `tv` uses 10) is **not** a sweep and stays out of the names.
- Genuinely list-valued parameters (`center`, `semi_axes`, `size`, `radii`, …)
  are excluded from the rule.
- Runs sharing a mesh and noise setting reuse one solver and one noise
  realisation, so sweeps are cheap.
- A run that crashes is recorded in `runs/<id>/error.txt` and the sweep
  continues.

---

## 8. The algorithms and their parameters

| name | method | speed | character |
|---|---|---|---|
| `gn` | Gauss-Newton, quadratic prior | fast | smooth, reliable all-rounder |
| `tv` | Gauss-Newton, total variation | fast | sharp edges, best for piecewise-constant targets |
| `l1` | L1-sparsity (Gehre et al. 2012) | slow | flat background, best localisation |
| `linear` | one-step linearised | instant | baseline; cannot handle high contrast |
| `dbar` | D-bar / Nachman direct method | fast | no iteration, no prior, no forward model in the loop |

The first four are **iterative optimisation**: they minimise a data misfit plus
a penalty, and every iteration costs forward solves. `dbar` is a different
animal — it is a **direct inversion formula**, evaluated once. It never calls
the forward model except to simulate a homogeneous reference, so its cost does
not depend on the contrast, the mesh or the number of iterations, and it cannot
get stuck in a local minimum. What it costs instead is that its only knob is a
low-pass filter, so it will not give you sharp edges.

### Shared

| parameter | meaning |
|---|---|
| `sigma_min`, `sigma_max` | conductivity is clipped to this range each iteration |
| `max_iter` | iteration budget |
| `tol` | stop when the relative change falls below this |
| `line_search` | `armijo` (backtracking, 1–3 forward solves) or `grid` |

### `gn` — Gauss-Newton with a quadratic prior

Minimises `½‖F(σ) − U‖²_W + ½λ(σ − σ₀)ᵀR(σ − σ₀)`.

| parameter | meaning |
|---|---|
| `lambda` | regularisation strength — **dimensionless**, see below |
| `prior` | `laplacian` (default), `gaussian`, `tikhonov` |
| `prior_eps` | `laplacian`: makes `R` positive definite |
| `prior_corrlength`, `prior_std` | `gaussian`: correlation length and std |

Prior choice:
- **`laplacian`** — sparse graph Laplacian penalising jumps between neighbouring
  cells. Fast, scales to any mesh. Use this unless you have a reason not to.
- **`gaussian`** — squared-exponential covariance; statistically the most
  meaningful, but **dense**: memory `n_cells²`, factorisation `n_cells³`. Only
  practical below a few thousand cells.
- **`tikhonov`** — identity. Penalises magnitude, not roughness.

### `tv` — Gauss-Newton with total variation

Minimises `½‖F(σ) − U‖²_W + λ·Σ√((Lσ)² + β)`.

| parameter | meaning |
|---|---|
| `lambda` | regularisation strength — dimensionless |
| `beta` | smoothing of the TV kink; `1e-6` is fine. Larger → closer to `gn`. |

### `l1` — L1-sparsity

| parameter | meaning |
|---|---|
| `alpha` | sparsity weight — **not** auto-scaled, see below |
| `kappa` | strength of the H¹ gradient smoothing |
| `initial_step_size`, `step_min`, `stopping_criterion` | step-size control |

### `linear` — one-step linearised

`lambda`, `prior`. Jacobian computed once at the background, so it is
essentially free — a good sanity baseline and the right choice for
time-difference imaging.

### `dbar` — the D-bar method (Nachman's direct algorithm)

A nonlinear Fourier transform. The measurements are turned into a *scattering
transform* `t(k)` living in a complex frequency plane, `t` is low-pass filtered
by throwing away `|k| > R`, and a D-bar (`∂/∂k̄`) equation transforms it back
into a conductivity:

```
U_meas ──▶ Λ_σ − Λ₁ ──▶ t(k) ──▶ μ(z,0) ──▶ σ(z) = μ(z,0)²
           DN map      scattering  D-bar eq.
```

| parameter | meaning |
|---|---|
| `R` | truncation radius of the scattering transform — the **only** regularisation parameter |
| `scattering` | `exp` (Born approximation, default) or `bie` (full transform via Nachman's boundary integral equation) |
| `k_grid` | points per axis of the k-plane grid; power of two, default 64 |
| `z_grid` | pixels per axis of the reconstruction, default 64 |
| `contact_correction` | subtract the known contact-impedance shunt from the ND matrix (default on) |
| `t_cutoff` | optional hard cap on `|t|`; see below |

**`R` replaces `lambda`.** Small `R` gives a smooth, stable, low-contrast image;
large `R` a sharper one, until the transform blows up and the image becomes
noise. Unlike `lambda` there is no auto-scaling to do — `R` is a frequency in a
plane that has already been normalised to the unit disk, so the same value
means the same thing on any mesh, any radius and any background. On this
framework's default geometry (16 electrodes covering half the boundary):

| noise | best `R` | `rel_L2_roi` there | `t_growth` there | first `R` that fails |
|---|---|---|---|---|
| none | ≥ 6, still improving | 0.33 | 8.6 | — |
| 0.5 % | 4.0 | 0.34 | 13.5 | 4.5 (`t_growth` 25) |
| 2 % | 3.0 | 0.25 | 14.5 | 3.5 (`t_growth` 20) |

(measured on `two_inclusions_16e`, 16 electrodes, adjacent drive.)

Sweep it — `"dbar": {"R": [3.0, 3.5, 4.0]}` — and read `t_growth` from the run
summary. It measures how fast `|t|` is still growing at the truncation radius;
above about 20 the transform has blown up, the image is amplified noise, and a
warning is printed. Note the pattern above: the *best* `R` sits just below that,
so a healthy `t_growth` of 10–15 is the target, not a small one. `t_cutoff`
zeroes `t` wherever `|t|` exceeds it, as a blunt alternative to lowering `R`.

**`exp` vs `bie`.** `exp` replaces the unknown CGO solution by its asymptotic
value `e^{ikz}` — this is the Born approximation `t^exp`, the variant used for
essentially every published D-bar reconstruction from real data and the one
Knudsen–Lassas–Mueller–Siltanen proved is a regularisation strategy. `bie`
solves Nachman's boundary integral equation for the true CGO trace using the
exact Faddeev Green's function; it recovers contrast slightly better at moderate
`R` but is more fragile, because that integral equation amplifies DN-map error
like `e^{2|k|}`.

**Where D-bar differs in practice.** It reconstructs on a pixel grid and is then
sampled onto the mesh, so the reconstruction mesh only affects the homogeneous
reference and the display, not the inversion. Being direct, there is no
convergence curve — the `convergence.png` figure simply skips it.

---

## 9. How to choose parameters

### `lambda` is dimensionless

The natural size of the data term `JᵀWJ` changes by orders of magnitude with the
mesh, the drive amplitude and the noise weighting. An **absolute** `lambda` is
therefore not comparable across the settings this framework exists to compare —
the same number is far too weak on one mesh and far too strong on another.

By default `lambda` is rescaled internally by `trace(JᵀWJ)/trace(R)`, so:

> **`lambda ≈ 1` means "regularisation as strong as the data"**, on any mesh, at
> any noise level, for any drive amplitude.

Starting ranges:

| algorithm | sweep this | optimum usually near |
|---|---|---|
| `gn` | `[0.01, 0.03, 0.1, 0.3, 1.0]` | `0.03`–`0.1` |
| `tv` | `[1, 3, 10, 30]` | `3`–`10` |
| `linear` | `[0.03, 0.1, 0.3, 1.0]` | `0.1`–`0.3` |

Set `"lambda_scaling": "absolute"` to pass raw values through.

### Use χ² to tune, not guesswork

The single most useful number in the output table is `chi2` (χ² per degree of
freedom of the data residual). It needs **no ground truth**, so it works on real
measurements too:

- **χ² ≈ 1** — the reconstruction explains the data to within the noise. This is
  the target (the *discrepancy principle*).
- **χ² ≪ 1** — the noise is being fitted. `lambda` too small.
- **χ² ≫ 1** — the data is not explained. `lambda` too large.

In practice the image errors bottom out right where χ² crosses 1. So: sweep
`lambda` over decades, pick the run with χ² closest to 1, then refine.

**This does not apply to `dbar`.** χ² measures how well a reconstruction
*reproduces the measurements*, which is the objective the other four algorithms
minimise. D-bar never looks at the data residual — it evaluates an inversion
formula — so its χ² routinely lands in the tens or hundreds even when the image
is the best in the sweep. Tune `R` with `t_growth` (§8) and rank D-bar runs by
`rel_L2_roi` or `dice`, not by χ².

### `alpha` for `l1` is *not* auto-scaled

The iterative-soft-thresholding scheme does not admit the same normalisation, so
`alpha` depends on the absolute voltage level — i.e. on drive amplitude and
contact impedance. Symptoms:

- reconstruction collapses to a flat background (`dynamic_range = 0`) → `alpha`
  too large
- reconstruction is noisy → `alpha` too small

Sweep it over decades (`[3e-6, 1e-5, 3e-5, 1e-4]`) whenever you change the drive
settings. `1e-5` suits the shipped defaults.

### Contact impedance

`1e-4` … `1e-1` is physical. Very small `z` degenerates towards the shunt model
and makes the system badly conditioned; the solver warns below `1e-5`.

### Mesh resolution

Start with `coarse` (~850 cells). Go finer only if the reconstruction looks
pixelated relative to the features you care about — the dense solve costs
`n_cells³`. Sweeping two or three meshes is the honest way to check your
conclusions do not depend on the discretisation.

---

## 10. Metrics — what they mean

All metrics are evaluated on **one common pixel grid**, not on each method's own
mesh. Summing over the cells of different meshes is not comparable — a finer
mesh has more, smaller cells, which silently reweights the error — and comparing
meshes is a first-class use case here.

| metric | meaning | good |
|---|---|---|
| `rel_L1`, `rel_L2` | global relative error | lower |
| `rmse`, `mae` | absolute error | lower |
| `rel_L1_roi`, `rel_L2_roi` | error restricted to where the truth differs from background | lower |
| `dice` | shape overlap of the detected inclusions | higher (1 = perfect) |
| `correlation` | Pearson correlation with the truth | higher |
| `dynamic_range` | recovered contrast ÷ true contrast | 1 is perfect |
| `recon_min`, `recon_max` | range of the reconstruction | — |
| `chi2_per_dof` | data-space fit — **no ground truth needed** | ≈ 1 |
| `data_residual_rel` | relative data misfit | lower |
| `runtime_s` | wall-clock time | — |

Notes:

- **`rel_L2_roi`** is the default ranking metric: global errors are dominated by
  the (large, easy) background, ROI errors measure whether the *inclusions* were
  actually recovered.
- **`dice`** thresholds each field at half of its own peak deviation
  (full-width-half-maximum). A fixed absolute threshold cannot be used here: EIT
  reconstructions are smooth, so almost every pixel deviates from the background
  by a few percent, and a fixed threshold marks nearly the whole cell as
  "inclusion" — the score then collapses to roughly the same small number for
  every algorithm and carries no information.
- **`dynamic_range` < 1** is normal — EIT smooths. Values *above* 1 mean the
  reconstruction is overshooting, usually a sign of under-regularisation.

---

## 11. Output files

### Ground truth

```
GT_simulated_result/<name>/
    config.json           the full configuration used
    phantom.json          the resolved phantom (random draws frozen)
    truth_mesh.msh        the mesh the data was computed on
    sigma_true.npz        centroids + conductivity
    measurements.npz      clean voltages, injection matrix, contact impedance
    background_measurements.npz   homogeneous reference frame
    ground_truth.csv      x, y, sigma
    measurements.csv      pattern, electrode, voltage
    ground_truth.png      preview: phantom + electrode voltages
    summary.txt           one-page human-readable description
```

### Experiment

```
results/<run_name>/
    comparison.png            all reconstructions side by side, ranked
    metrics_vs_<axis>.png     each metric against each swept axis, per algorithm
    convergence.png           objective and step size per iteration
    runs.csv                  one row per run: every parameter × every metric
    summary.txt               ranking, best-per-algorithm, failures
    config.json
    runs/<run_id>/
        reconstruction.png    truth | reconstruction | difference
        reconstruction.csv    x, y, sigma
        reconstruction.npz    + simulated and measured voltages
        metrics.json
        history.csv           per-iteration objective, step size, change
        config.json
        error.txt             only if the run failed
```

`runs.csv` is the one to load into pandas for your own analysis.

---

## 12. The physics

**Model.** Complete Electrode Model (CEM), P1 potentials, DG0 conductivity:

```
−∇·(σ∇u) = 0                 in Ω
u + z_i σ ∂u/∂n = U_i        on electrode e_i
σ ∂u/∂n = 0                  between electrodes
∫_{e_i} σ ∂u/∂n ds = I_i     injected current
Σ_i U_i = 0                  gauge
```

solved as a saddle-point system with a Lagrange multiplier for the gauge.

**Jacobian sign.** `calc_jacobian` returns `J = −dU/dσ`. With residual
`r = U_sim − U_meas`, the normal equations `(JᵀJ)d = Jᵀr` then give a *descent*
direction, so the update is `σ ← σ + d`. This convention is verified against
finite differences in the test suite.

**Rank deficiency.** With 16 electrodes the Jacobian has at most a few hundred
independent rows against thousands of unknowns — e.g. rank 120 for 858 cells.
The reconstruction is therefore determined largely by the prior. This is not a
defect of the code, it is the nature of EIT, and it is precisely why the
regularisation has to be on a comparable footing before any algorithm comparison
means anything.

---

## 13. Testing

```bash
python tests/test_framework.py       # 115 checks
python tests/test_dbar.py            #  54 checks, the D-bar method
```

`test_framework.py` verifies properties that must hold independently of the
implementation:

- charge conservation, rank and zero-row freedom of every drive pattern
- rejection of non-circular geometry and invalid meshing parameters
- the CEM gauge `Σᵢ Uᵢ = 0`
- **reciprocity** — symmetry of the transfer impedance matrix
- the CEM scaling law `U(cσ, z/c) = U(σ, z)/c`, and linearity in the current
- rotational equivariance on a homogeneous disk
- mesh convergence at first order or better (corner singularities at the
  electrode edges preclude O(h²))
- the **adjoint Jacobian against finite differences** — magnitude, direction and
  sign, per cell and along random directions
- `LᵀL = Γ⁻¹` for the Gaussian prior; the TV operator against the facet count
- monotone decrease of the Gauss-Newton objective
- inclusion recovery, and that larger `lambda` really does smooth more
- that one `lambda` behaves identically at 1× and 10× drive amplitude
- ground-truth bundle round trip, sweep expansion and run naming

`test_dbar.py` checks the D-bar chain against closed-form results rather than
against a previous output of the code:

- Faddeev's Green's function: harmonicity away from the origin, the `−log|z|/2π`
  singularity with the right constant, and the exact scaling `G_k(z) = G_1(kz)`
- the Cauchy transform `(1/πk) ∗` against an analytically solvable D-bar
  problem, and its convergence under grid refinement
- the **Neumann-to-Dirichlet matrix to machine precision** against the exact
  projection of the continuum ND map of the unit disk — this pins down every
  electrode-area, radius and background factor in one check
- DN eigenvalues against `Λ₁e^{inθ} = |n|e^{inθ}`, and that the error grows with
  the mode number (which is *why* the transform must be truncated)
- the scattering transform against the Calderón/Born formula
  `t(k) ≈ −2|k|² δσ̂(−2k₁, 2k₂)` for an **off-centre, non-radial** perturbation.
  This is the check that pins down the sign, the conjugation and the orientation
  of the k-plane; a mirrored or rotated reconstruction fails here and nowhere
  else
- `t^bie → t^exp` as the contrast goes to zero (the Born limit), and both vanish
  when `Λ_σ = Λ₁`
- exact equivariance of the whole pipeline under `σ → cσ` and invariance under a
  change of domain radius
- localisation of a real inclusion from CEM/FEM data, including that a mirrored
  phantom gives a mirrored image

---

## 14. Troubleshooting

**Reconstruction is pure noise.** `lambda` is too small. Check `chi2` — if it is
far below 1 you are fitting noise. Increase `lambda` by decades.

**Reconstruction is a flat background.** `lambda` (or `alpha` for `l1`) is too
large; `chi2` will be ≫ 1.

**`l1` gives `dynamic_range = 0`.** `alpha` too large — see §9.

**`dbar` warns that "the scattering transform is blowing up".** `R` is too large
for this noise level; the image is amplified error, not signal. Lower `R` by
0.5–1 and compare `t_growth` across the sweep.

**`dbar` gives a washed-out, almost flat image.** The opposite problem: `R` is
too small. Raise it while `t_growth` stays below ~15. Note that D-bar always
under-states contrast — it is a low-pass filtered reconstruction, so a σ = 3
inclusion typically comes back around 1.5–2.5. Compare shapes and positions, not
peak values; `dice` and `rel_L2_roi` are the metrics to rank it by.

**`dbar` raises "the injection patterns are rank deficient".** D-bar has to
invert the measured ND map, so it needs `L − 1` linearly independent patterns.
Drive methods 2, 3, 4 and 5 all provide them; method 1 (opposite) only gives
`L/2` and cannot be used.

**Segfault on macOS.** A collision between torch's OpenMP runtime and
PETSc/OpenBLAS. The library pins torch to one thread at import to avoid it;
override with `TORCH_NUM_THREADS`. If you drive the library from your own
script, set `OMP_NUM_THREADS=1` before importing.

**`ImportError: numpy ... libgfortran`.** A broken conda environment (duplicate
`LC_RPATH`), not a code problem. Use a working env or reinstall numpy/openblas.

**Gauss-Newton is very slow.** The normal equations are dense `n_cells × n_cells`
with an `O(n³)` solve. Use a coarser reconstruction mesh; above ~12000 cells the
solver warns.

**"reconstruction mesh is not coarser than the truth mesh".** You are committing
the inverse crime — results will be optimistic. Coarsen `recon_mesh` or refine
the truth mesh.

---

## Authors

- Hossein Ostovar (Institute of Chemical Reaction Engineering, TUHH)
- Moritz Hollenberg (Institute of Mechatronics in Mechanical Engineering, TUHH)

## License

MIT
