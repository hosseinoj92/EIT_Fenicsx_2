# SI EIT laboratory: 2D, 3D and frequency

## Start here

From the project directory, run `python physical_eit.py --gui`. The desktop
panel can run on Windows without FEniCSx in the Windows Python. It uses the
local Python when FEniCSx is available, otherwise the default WSL distribution.
NumPy, SciPy, Matplotlib and Tkinter are needed for the panel and previews.

1. Choose **2** or **3** in Geometry. Switching dimension loads the defaults
   for that dimension; save a case before switching if you want to retain it.
2. Set radius, electrode count, mesh sizes, and the dimension-specific fields.
3. Set excitation frequencies, RMS current and material properties. Frequencies
   are a comma-separated list in Hz. Set `0` for a DC solve.
4. Preview the electrode geometry and material admittivity before meshing.
5. Choose the reconstruction frequency from the sweep, or disable reconstruction.
6. Run. Results contains the solver log, frequency response and field slices.

The first circular/spherical inclusion is editable in the form. Full configuration
provides JSON access to multiple inclusions, other shapes, nonuniform ring heights,
individual electrode contacts and dielectric relaxation. Click **Apply JSON to
form** after editing JSON. **Save configuration** writes a reproducible SI case.
The example material values illustrate the model; they are not measured tissue
or saline presets.

On Linux/WSL using Ubuntu packages:

```bash
sudo apt-get update
sudo apt-get install python3-dolfinx-real python3-gmsh python3-matplotlib python3-tk
python3 physical_eit.py --config examples/physical_3d.json
```

The new SI solver does not require PyTorch. The original benchmark additionally
needs its original dependencies (including PyTorch and tqdm). A SciPy complex
linear solve works with a **real** FEniCSx/PETSc build: real stiffness and boundary
integrals are assembled first, then combined into the complex CEM matrix.
Execution is currently serial (one MPI rank). Gmsh builds without Netgen use
the built-in mesh optimizer with an explicit message.

For Windows batch execution:

```powershell
python physical_eit.py --config examples/physical_3d.json --validate
wsl --exec python3 physical_eit.py --config examples/physical_3d.json
```

Use project-relative `output_root` paths when sharing JSON between Windows and
WSL. Generate new templates with `python physical_eit.py --example my_case.json
--dimension 3`. Loading `config.json` from a result folder repeats that case in
a new folder; existing runs are not overwritten.

## Geometry and SI units

| Input | Unit / interpretation |
|---|---|
| `mesh.dimension` | `2` or `3` |
| `mesh.radius` | cylinder/cross-section radius, m |
| `mesh.height` | cylinder extent in 3D: z from 0 to height, m |
| `thickness` | uniform extrusion thickness for 2D, m |
| `mesh.n_electrodes` | total electrode count (derived/checked in 3D) |
| `mesh.electrodes_per_ring`, `mesh.n_rings` | integer counts in 3D |
| `mesh.ring_spacing` | axial centre-to-centre separation, m |
| `mesh.ring_z` | optional explicit ascending ring heights, m; overrides spacing |
| `mesh.electrode_diameter` | circular 3D contact footprint diameter, m |
| `mesh.electrode_width_deg` | 2D electrode arc angle, degrees, converted to radians |
| `mesh.electrode_offset_deg` | first-ring angular offset, degrees |
| `mesh.ring_rotation_deg` | incremental offset from one ring to the next, degrees |
| `mesh.h_center`, `mesh.h_edge` | target finite-element sizes, m |
| `drive.current_a` | positive current on the source electrode of each pair, A RMS |
| `frequencies_hz` | excitation frequencies, Hz |
| material `conductivity_s_m` | DC conductivity, S/m |
| material `relative_permittivity` | relative permittivity, dimensionless |
| contact `resistance_ohm_m2` | specific contact resistance, ohm m² |
| contact `capacitance_f_m2` | specific parallel contact capacitance, F/m² |
| inclusion `center_m`, `radius_m`, `semi_axes_m`, `size_m` | physical lengths, m |
| reconstruction `noise_std_v` | independent real and imaginary noise-channel SD, V |

Lengths are never normalized to a unit disk in the SI workflow. A value of
`0.05` means 5 cm; `0.001` A means 1 mA. The saved configuration includes unit
metadata. Changing this metadata cannot change input units. Angles are explicitly
labelled in degrees at the interface and converted for geometric calculations.

3D electrodes are conforming surface patches on the curved cylinder sidewall,
imprinted by radial cylinders. Their footprints are **circular in tangent-plane
projection**, with the specified diameter. Surface area is therefore slightly
larger than pi*(diameter/2)^2. This models a conforming curved contact rather
than a flat disk pressed onto a cylinder. Physical facet tags `1..L` identify
electrodes; the rest of the boundary, including both end caps, is insulating.
Each tetrahedron belongs to the conducting domain, including the Boolean
subvolumes used to imprint the contacts.

Unless `ring_z` is supplied, the ring stack is centred vertically. The program
rejects electrodes crossing the end caps, overlapping contacts, invalid counts,
nonfinite parameters and inclusions outside the cell. Its whole-inclusion
containment check is conservative for ellipsoids and boxes.

Electrodes are numbered starting at the configured offset, counterclockwise
when viewed from +z, ring by ring from bottom to top. Protocols are:

- `adjacent`: all neighbouring pairs **within each ring**.
- `opposite`: opposite pairs within each ring, requiring an even count per ring.
- `axial`: same electrode index on neighbouring rings (3D with at least two rings).
- `mixed`: adjacent drives within rings plus axial drives between neighbouring rings.

Some pair drives are linearly dependent; this is intentional. Every pair still
conserves current. Only `L-1` independent electrode-voltage modes exist. `mixed`
connects all rings and is a useful starting point for a volume experiment.
For rotated rings, axial drives connect matching indices, which may also be
angularly offset. Drive/reconstruction resolution and identifiability still
depend on the actual electrode layout.

In 2D the stiffness and boundary integrals are multiplied by `thickness`.
Injection is consequently in **A**, not A/m; conductivity is still S/m and
contact impedance is still ohm m². A 2D contact area is its arc length times
thickness. This is a uniform extrusion approximation with electrodes spanning
the thickness; finite circular electrode rings require the full 3D model.

## Frequency-dependent physics

With time dependence `exp(+j*omega*t)`, omega = 2*pi*f, the forward problem is

```text
gamma(f) = sigma(f) + j*omega*epsilon0*epsilon_r(f)       [S/m]
div(gamma grad u) = 0                                  in the cell
u + z_contact(f) gamma du/dn = U_l                      on electrode l
integral_electrode gamma du/dn dS = I_l                 [A]
gamma du/dn = 0                                        elsewhere
sum_l U_l = 0                                         voltage gauge
```

`epsilon0 = 8.8541878128e-12 F/m`. Voltages and currents are RMS phasors. For
fixed passive material properties, frequency changes the displacement-current
term and therefore voltage magnitude and phase. Setting permittivity and
contact capacitance to zero gives exactly the frequency-independent resistive
limit. The code does not invent a frequency effect for a purely resistive cell.

An optional passive, single-Debye dielectric relaxation uses

```json
{
  "conductivity_s_m": 0.2,
  "relative_permittivity": 80,
  "delta_epsilon_r": 10000,
  "relaxation_time_s": 0.00001
}
```

Here epsilon_r_complex = epsilon_r_infinity + delta_epsilon_r/(1+j*omega*tau).
Adding `j*omega*epsilon0*epsilon_r_complex` to the DC conductivity produces both
dielectric loss and a changing effective permittivity. Exported conductivity is
the **effective real admittivity at each frequency**, including that loss.
More complicated measured spectra need an appropriate fitted constitutive model;
this implementation does not fit Cole–Cole or multi-pole tissue spectra.

The optional contact model is a specific parallel RC interface:
`1/z_contact = 1/resistance_ohm_m2 + j*omega*capacitance_f_m2` [S/m²]. Scalar
values apply to all contacts; JSON lists specify each electrode individually.
For a measured lumped electrode resistance R [ohm], the corresponding specific
resistance is approximately `R * electrode_area` [ohm m²]. The code integrates
each actual mesh contact area; it does not replace patches by point electrodes.

This is an isotropic electroquasistatic CEM, with no magnetic induction, propagation,
lead impedance, amplifier transfer function or electrode chemistry beyond the
chosen RC model. For comparisons to measurements, use the actual cell dimensions,
material spectra, electrode areas, contact model, RMS current and electrode-pair
voltage reference. Check mesh convergence as well as contact/material calibration;
a low synthetic-data residual alone does not validate an experimental model.

The CEM and complex-property treatment follow the
[CEM formulation and experimental comparisons](https://arxiv.org/abs/2305.17294)
and [complex EIT material/contact reconstruction formulation](https://doi.org/10.1088/0957-0233/13/12/307).

## Reconstruction and outputs

The SI workflow uses a complex Gauss–Newton inversion with real and imaginary
voltage residuals. Conductivity is always reconstructed; at nonzero frequency,
`reconstruct_permittivity: true` also reconstructs relative permittivity. At DC
permittivity is unobservable and is held fixed. When this option is false at AC,
the permittivity is fixed to the **background material**, not to the true phantom.
The inverse solver does not sample inclusion values to initialize its solution.
Single-frequency inversion recovers effective material properties at that
frequency; it does not estimate Debye relaxation parameters across the sweep.

The truth and reconstruction have separate meshes. The default spatial prior
penalizes jumps across interior facets, with a weak homogeneous-background ridge.
Both parameter fields are scaled to dimensionless variables before regularization.
The line search minimizes the combined data-plus-prior objective. Increasing
regularization smooths and reduces contrast. Positivity and conductivity bounds
are enforced. Volumetric reconstructions remain underdetermined and may show
blur and low recovered contrast even with an excellent voltage fit.
An iterative least-squares solve avoids forming a dense cell-by-cell normal
matrix, but the voltage Jacobian itself is dense and still consumes memory.

Each timestamped result folder contains:

- `config.json`: complete resolved physical parameters and units.
- `measurements.npz`: complex `U_v` and homogeneous-reference `U_background_v`
  with shape `(frequency, pattern, electrode)`; frequencies, injected currents,
  contact areas/centres, truth centroids and effective material fields.
- `measurements.csv`: frequency, pattern index (zero based), electrode number
  (one based), real/imaginary volts, magnitude in V RMS, and phase in degrees.
- `frequency_response.png`: transfer impedance magnitude/phase for pattern 0,
  using the first two non-driven electrodes when available. This is a voltage
  difference divided by source current, in ohm.
- `truth_<frequency>Hz.vtu`: full 2D/3D conductivity/permittivity and complex
  potential for pattern 0. Open these files in ParaView for volume inspection.
- `truth.png`: 2D map or orthogonal 3D sections through x=0, y=0, z=height/2.
- `reconstruction.npz`, `.vtu`, `.png`, `metrics.json` when inversion is enabled.
- `truth_mesh.msh`, `reconstruction_mesh.msh`: meshes accompanying the run.

Electrode potentials are in a zero-mean gauge. Real instruments normally measure
**differences**: compare `U[..., a] - U[..., b]` with the matching lead polarity.
Phase is relative to the real injected current. CSV preserves complex data;
neither magnitude-only inversion nor discarding imaginary values is used.
The NPZ arrays are the full electrode data; the plotted impedance is one example
measurement channel. Evaluation metrics use one uniform spatial grid shared by
the truth and reconstruction, independent of their mesh cell counts.

The existing `make_ground_truth.py` / `simulate.py` bundle format and real-valued
algorithm benchmark remain available. D-bar remains a **2D real-conductivity**
method; it is not a volume or complex-admittivity algorithm. SI AC result folders
are opened through this workflow, not through the original bundle loader.

## Validation

```bash
python3 tests/test_physical.py
python3 tests/test_framework.py
python3 tests/test_dbar.py
```

The physical suite checks conservation, ring numbering and geometry validation,
passive dielectric limits, reciprocity, gauge, positive dissipated power, current
scaling, 2D thickness scaling, an exact complex scaling case including RC contacts,
conductivity/permittivity Jacobians against finite differences in 2D and 3D,
electrode area/volume, separate-mesh reconstruction and complex export round trips.
When FEniCSx is unavailable only the material/configuration checks run; numerical
checks report explicit skips. Run the suite under WSL for complete validation.
