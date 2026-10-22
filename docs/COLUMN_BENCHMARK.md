# Reproducing the 104 mm, 256-electrode column experiment

The authoritative input is [`examples/column_104mm_256e.json`](../examples/column_104mm_256e.json).
It is accepted by the same CLI and desktop GUI. The form preserves advanced
JSON fields, full floating-point precision and explicit ring heights. Pending
JSON edits are applied before Validate, Save or Run. When explicit ring heights
are present, the unused regular-spacing form control is disabled.

```powershell
python physical_eit.py --config examples/column_104mm_256e.json --validate
wsl --exec python3 physical_eit.py --config examples/column_104mm_256e.json
# Or open the panel, then Load configuration:
python physical_eit.py --gui
```

| Property | Value in JSON |
|---|---|
| Column diameter / radius | 0.104 m / 0.052 m |
| Column height | 0.500 m, floor at z=0 |
| Electrodes | 8 rings, 32 per ring; 256 total |
| Ring centres [m] | 0.220, 0.235, 0.250, 0.265, 0.280, 0.295, 0.310, 0.325 |
| Electrode diameter | 0.002 m |
| Water conductivity | 0.05 S/m = 0.5 mS/cm |
| Water relative permittivity | 78.3, illustrative 25 °C approximation |
| Frequencies | 1000, 10000, 100000 Hz |
| Sphere | radius 0.012 m; zero DC conductivity, relative permittivity 3 |
| Sphere centre [m] | 0.0007292220873354067, 0.0009126446308762588, 0.2582148429552654 |
| Current | 0.001 A RMS per pair (assumed) |
| Specific contact resistance | 0.005 ohm m² (assumed, not measured) |
| Specific contact capacitance | 0 F/m² (assumed) |
| Added noise | 0 V for this reference comparison |

The sphere centre was drawn once with NumPy seed 20261022 and then resolved to
explicit coordinates. Changing `reconstruction.seed` changes measurement noise;
the stored sphere does not move unless its geometry is edited. The water
permittivity approximation is based on
[Malmberg and Maryott's measurements](https://nvlpubs.nist.gov/nistpubs/jres/56/jresv56n1p1_a1b.pdf).
No material property in this test was fitted to experimental data.

The insulator has exactly zero conductivity in the **forward truth**. Its
nonzero permittivity carries displacement current at nonzero excitation
frequency, so there is no artificial conductivity floor in the truth. The
inverse conductivity lower bound is separately configured as 1e-7 S/m.
An exactly zero-admittivity subdomain at DC is rejected rather than assembled
into a singular full-domain system.

## Mesh and measurement choices

The fine truth mesh has 258,100 tetrahedra / 57,891 nodes. Its contact target
size is 0.4 mm; the ring band z=0.20..0.35 m is refined to a 4.5 mm target.
Mesh sizes are target values, not guaranteed uniform element diameters. The
independent inverse mesh has 89,784 tetrahedra / 21,591 nodes. Contact geometry,
tags, actual contact areas and the full 0.5 m column are retained in both meshes.

All 256 contacts are present, and all electrode voltages are saved. The drive
protocol uses 120 adjacent/axial pair patterns selected by `drive.stride=4`.
The inverse uses 128 passive electrode-pair channels (`measurement_stride=2`),
covering all 256 contacts. Excluding pairs touching the driven electrodes leaves
15,184 complex voltage differences. These are explicit protocol choices, not
an implicit reduction of the requested electrode count.

The sphere measurement and homogeneous reference are both generated on the
fine mesh. Each inverse uses its own homogeneous reference and fits the voltage
**change**. This difference-imaging comparison reduces background mesh/contact
modeling errors; it is not an absolute-imaging calibration test.

A configurable nonuniform 12 x 12 x 23 voxel grid represents conductivity
throughout the column, giving 2,927 occupied inverse parameters for this mesh.
Its axial edges are finer around the known electrode region and coarser away
from it. Unknowns are not restricted to the sphere location. Each FE cell maps
to a voxel, and cell sensitivities are accumulated before storing the Jacobian.
This avoids a huge full electrode-by-FE-cell Jacobian while keeping the forward
finite-element geometry intact.

The separately saved `mesh_refinement.json` compares representative object
signals against a further refined forward mesh. A two-level check quantifies
discretization sensitivity; it does not, by itself, prove asymptotic convergence.

## Algorithms

Each frequency is reconstructed with the same data and spatial parameter grid:

- `linear`: one background linearization, with a quadratic spatial prior.
- `gn`: nonlinear Gauss–Newton with a quadratic spatial prior and line search.
- `tv`: nonlinear Gauss–Newton / IRLS with smoothed total variation.
- `l1`: nonlinear proximal Gauss–Newton, with soft-thresholded voxel contrasts.
- `dbar`: explicitly recorded as unsupported for a 3D volume.

These are 3D extensions of the existing algorithm families. In particular, the
voxel L1 extension is not the original CG1/H1 sparsity implementation. The
original real-valued 2D routines remain available for their original benchmark.
The 3D comparison uses both voltage quadratures but reconstructs conductivity
with background permittivity held fixed. It does not claim to recover the
sphere's permittivity. The resulting small phase-model mismatch is part of the
reported fit. The earlier single-solver SI workflow can still jointly invert
conductivity and permittivity on smaller meshes.

All regularization weights, bounds, iteration limits, proximal iterations,
spatial resolution and frequency choices are in the JSON. Quadratic/TV prior
weights are scaled by the initial Jacobian and prior traces. The L1 threshold
is scaled relative to the initial data gradient. No true sphere values or
location are supplied to the inverse solver.

## Results

The configured output is `results/column_104mm_256e/run01/`. Forward simulations
and completed algorithm/frequency outputs are checkpointed. A rerun with the
same configuration resumes completed stages. To compare changed parameters,
choose a new `reconstruction.output_directory`; reusing a directory containing a
different configuration is rejected.

After a run, the optional Windows report utility independently reads both VTK
formats with PyVista, creates extra renderings, and packages the results:

```powershell
python column_report.py --results results/column_104mm_256e/run01 --render-vtk --package
```

This uses the PyVista dependency from the original framework installation and
does not require FEniCSx in the Windows Python. The shareable ZIP omits dense
Jacobian caches and preliminary protocol pilots. Full caches remain locally
available for resuming the simulation. The full-column geometry preview is
`geometry_full_column.png`; `vtk_field_3d.png` in each algorithm folder is
rendered from its actual exported tetrahedral conductivity field.

Each `reconstructions/<frequency>Hz/<algorithm>/` folder contains:

- `reconstruction.vtk`: complete-column legacy VTK, readable in ParaView.
- `reconstruction.vtu`: complete-column XML VTK, including truth for comparison.
- `reconstruction.npz`: fields, voxel centres/values and difference measurements.
- `reconstruction_3d.png`: truth and reconstructed half-peak support in 3D.
- `reconstruction_sections.png`: matching sections through the truth centre.
- `metrics.json`: objective history, voltage fit, contrast error, Dice and localization.

The 3D PNG zooms into the ring region; the VTK contains the full column. Truth
and reconstruction share the same conductivity color scale. The thresholded
volume image shows geometry; use the continuous field in VTK and section plots
to assess recovered contrast.

`REPORT.md`, `comparison.csv` and `comparison.json` compare all completed runs.
`relative_L2_contrast` evaluates the conductivity change from water, which avoids
letting the large, unchanged water volume hide a poor object reconstruction.
`recovered_mean_sphere_contrast_fraction` measures amplitude recovery inside the
known sphere. Dice uses half-peak contrast support. Localization uses the
conductivity-deficit weighted centroid over the full evaluation grid. Metrics
use a common uniform physical grid, not a count of mesh cells.

Current/contact values, noiseless measurements, finite resolution and fixed
inverse permittivity are limitations of this particular comparison. A good
voltage fit is not proof of accurate sphere shape or calibrated real-world
performance. Water's modest permittivity also means that kHz frequency changes
can be much smaller than in the illustrative high-permittivity starter example.
