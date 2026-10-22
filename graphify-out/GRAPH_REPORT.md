# Graph Report - EIT_Fenicsx_2  (2026-10-22)

## Corpus Check
- 128 files · ~286,797 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 773 nodes · 1581 edges · 50 communities (35 shown, 15 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 76 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `01097feb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_dbar.py
- test_framework.py
- two_inclusions_16e ground-truth bundle summary
- volume_benchmark.py
- simulate.py
- Noise Level Sweep (0.5% vs 2%)
- EITApp
- GroundTruth
- build_mesh
- Reconstruction quality metrics (rel_L2_roi, rel_L1, dice, dyn.range, chi2/dof)
- EIT Simulation & Reconstruction Framework
- Reconstructor
- EIT
- _GaussNewtonBase
- GaussNewtonSolverTV
- GaussNewtonSolver
- What You Must Do When Invoked
- What You Must Do When Invoked
- L1Sparsity
- graphify reference: extra exports and benchmark
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- __init__.py
- AGENTS.md
- CLAUDE.md
- .claude/CLAUDE.md
- .claude/skills/graphify/references/extraction-spec.md
- .codex/skills/graphify/references/extraction-spec.md
- Metrics vs lambda for gn/tv/linear at 0.5% and 2% noise
- Metrics vs noise level: most methods degrade 0.5%->2%, dbar R=3 improves
- Metrics vs R for dbar: at 2% noise error rises with R, at 0.5% error falls
- algorithms.py
- `CONFIG` sections
- .__init__
- Reproducing the 104 mm, 256-electrode column experiment
- 8. The algorithms and their parameters
- SI EIT laboratory: 2D, 3D and frequency
- 9. How to choose parameters
- 11. Output files

## God Nodes (most connected - your core abstractions)
1. `EIT` - 39 edges
2. `normalize_config()` - 32 edges
3. `EITApp` - 29 edges
4. `run()` - 26 edges
5. `material_admittivity()` - 24 edges
6. `build_mesh()` - 23 edges
7. `Two-Inclusion Ground Truth Phantom (conductive 3.0 + resistive ~0.2 on background 1.0, disk domain)` - 20 edges
8. `Reconstruction Metrics (rel_L2_roi, rel_L1, Dice, dynamic range, chi2/dof, runtime)` - 20 edges
9. `Noise Level Sweep (0.5% vs 2%)` - 20 edges
10. `default_config()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `GUIConfigTests` --uses--> `EITApp`  [INFERRED]
  tests/test_gui_config.py → eit_gui.py
- `generate()` --calls--> `EIT`  [INFERRED]
  make_ground_truth.py → eitcore/eit_forward_fenicsx.py
- `make_solver()` --calls--> `EIT`  [INFERRED]
  simulate.py → eitcore/eit_forward_fenicsx.py
- `make_solver()` --calls--> `EIT`  [INFERRED]
  tests/check_ac_smoke.py → eitcore/eit_forward_fenicsx.py
- `cem_solver()` --calls--> `EIT`  [INFERRED]
  tests/test_dbar.py → eitcore/eit_forward_fenicsx.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **D-bar Regression Benchmark: methods compared on same phantom** — results_dbar_regression_method_dbar, results_dbar_regression_method_gn, results_dbar_regression_method_l1, results_dbar_regression_method_linear, results_dbar_regression_method_tv, results_dbar_regression_two_inclusion_phantom, results_dbar_regression_reconstruction_metrics [INFERRED 0.85]
- **Iterative variational regularized EIT solvers** — results_dbar_regression_method_gn, results_dbar_regression_method_tv, results_dbar_regression_method_l1 [INFERRED 0.75]
- **Mesh sweep: reconstruction method comparison** — results_mesh_sweep_gauss_newton_method, results_mesh_sweep_l1_regularization_method, results_mesh_sweep_linear_one_step_method, results_mesh_sweep_total_variation_method, results_mesh_sweep_two_inclusion_phantom, results_mesh_sweep_reconstruction_metrics [EXTRACTED 1.00]
- **Mesh sweep: inverse mesh resolution variants** — results_mesh_sweep_coarse_mesh, results_mesh_sweep_edge_refined_mesh, results_mesh_sweep_medium_mesh [INFERRED 0.95]

## Communities (50 total, 15 thin omitted)

### Community 0 - "test_dbar.py"
Cohesion: 0.08
Nodes (66): _arc_quadrature(), _cauchy_kernel_hat(), _cgo_boundary_coefficients(), contact_shift(), dbar_grid(), DbarSolver, dn_matrix(), electrode_geometry() (+58 more)

### Community 1 - "test_framework.py"
Cohesion: 0.06
Nodes (80): _json_default(), load_ground_truth(), Load a bundle written by :func:`save_ground_truth`., save_ground_truth(), add_measurement_noise(), gamma_inv(), noise_std(), Inverse noise covariance (diagonal) for the reconstructors. ``relative_floor``… (+72 more)

### Community 2 - "two_inclusions_16e ground-truth bundle summary"
Cohesion: 0.14
Nodes (15): Ground truth preview: phantom + noise-free adjacent-drive voltages (15 patterns), Two-inclusion phantom (circle sigma=3, ellipse sigma=0.2, bg=1), two_inclusions_16e ground-truth bundle summary, 5. Step 1 — `make_ground_truth.py`, `CONFIG` sections, Drive patterns, Phantoms, dbar_regression comparison: 20 reconstructions ranked by rel_L2_roi (+7 more)

### Community 3 - "volume_benchmark.py"
Cohesion: 0.05
Nodes (70): finalize(), geometry_image(), inspect_vtk(), truth_sections(), launch(), Native desktop SI configurator; batch FEniCSx runs stay in a child process., All metrics for one reconstruction. Returns a dict., default_config() (+62 more)

### Community 4 - "simulate.py"
Cohesion: 0.06
Nodes (41): all_metrics(), data_residual(), MetricEvaluator, "Detected inclusion" set: pixels deviating from the background by more than…, Reconstructed contrast as a fraction of the true contrast. 1.0 means the…, Agreement in *data* space, which needs no ground truth at all. Returns the…, Common evaluation grid for one ground truth. Parameters ---------- gt_points :…, Regular grid over the bounding square, clipped to the circle. (+33 more)

### Community 5 - "Noise Level Sweep (0.5% vs 2%)"
Cohesion: 0.24
Nodes (28): D-bar Direct Reconstruction (truncation radius R), Gauss-Newton Regularized Reconstruction (lambda), L1 Sparsity-Regularized Reconstruction, Linearized One-Step Reconstruction (lambda), Total Variation Regularized Reconstruction (lambda), Noise Level Sweep (0.5% vs 2%), Reconstruction Metrics (rel_L2_roi, rel_L1, Dice, dynamic range, chi2/dof, runtime), Reconstruction figure: dbar_R=3.5_level_percent=0.5 (truth | reconstruction | error, noise 0.5%) (+20 more)

### Community 7 - "GroundTruth"
Cohesion: 0.11
Nodes (6): GroundTruth, Loaded ground-truth bundle., Conductivity on the truth mesh, one value per cell., ``(n_cells, 2)`` cell centroids of the truth mesh., ``(n_patterns, L)`` noise-free electrode voltages., Radius of the circular cross-section the data was generated on.

### Community 8 - "build_mesh"
Cohesion: 0.16
Nodes (19): _boundary_point(), _build_cylinder(), build_mesh(), describe(), electrode_spans(), main(), _merge(), mesh_filename() (+11 more)

### Community 9 - "Reconstruction quality metrics (rel_L2_roi, rel_L1, dice, dyn.range, chi2/dof)"
Cohesion: 0.24
Nodes (21): Coarse inverse mesh, Edge-refined inverse mesh, Gauss-Newton (gn) iterative reconstruction, L1-regularized reconstruction, Linear one-step (linearized) reconstruction, Medium structured inverse mesh, Reconstruction quality metrics (rel_L2_roi, rel_L1, dice, dyn.range, chi2/dof), gn_mesh=coarse reconstruction figure (+13 more)

### Community 10 - "EIT Simulation & Reconstruction Framework"
Cohesion: 0.12
Nodes (16): 10. Metrics — what they mean, 12. The physics, 13. Testing, 14. Troubleshooting, 1. Installation, 2. Quick start, 3. How the pieces fit together, 4. Step 0 — `mesh_tools.py` (+8 more)

### Community 11 - "Reconstructor"
Cohesion: 0.20
Nodes (5): ABC, ``(n_cells, 2)`` centroids of the reconstruction mesh., Return the coefficient vector of ``func`` on the DG0 (cell-wise) space.…, Resample a cell-wise field onto a regular pixel grid. Returns ``(image,…, Reconstructor

### Community 12 - "EIT"
Cohesion: 0.06
Nodes (19): EIT, _interpolation_points(), _PetscLU, L: number of electrodes Inj: current injection pattern, ``(N, L)`` with N…, Assemble the sigma-independent blocks B, C, D and the gauge row., Exact linear-simplex facet quadrature without one JIT form per contact., Exact P1 simplex gradients in DG0 order, avoiding per-pattern UFL JIT., LU factorisation of the saddle-point matrix for ``sigma``, cached. The forward… (+11 more)

### Community 13 - "_GaussNewtonBase"
Cohesion: 0.17
Nodes (6): _GaussNewtonBase, Value of the regularisation term P(sigma)., ``(P'', P')`` as torch tensors, evaluated at ``sigma``., ``trace`` of the penalty Hessian at ``sigma`` for ``lamb = 1``. Used to make…, Return ``(step, new_sigma, phi_new)``. ``armijo``: backtracking from 1.0,…, Shared machinery: line search, convergence tracking, prior interface.

### Community 14 - "GaussNewtonSolverTV"
Cohesion: 0.18
Nodes (8): array, GaussNewtonSolverTV, Put ``lamb`` on a dimensionless footing. The natural size of the data term…, The common Gauss-Newton loop., Gauss-Newton with the smoothed total-variation prior P(sigma) = lamb * sum_e…, One Gauss-Newton direction at ``sigma`` (no line search)., ``(J^T W J, J^T W r)`` without materialising the dense diagonal W., _weighted_normal_equations()

### Community 15 - "GaussNewtonSolver"
Cohesion: 0.22
Nodes (7): AlgorithmSpec, _as_torch(), GaussNewtonSolver, LinearisedReconstruction, Iteratively regularised Gauss-Newton with the quadratic prior P(sigma) = 0.5 *…, Materialise ``R`` once per ``forward`` call., One-step linearised (difference) reconstruction around a homogeneous…

### Community 16 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 17 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 18 - "L1Sparsity"
Cohesion: 0.24
Nodes (4): _run_l1(), GradientSmoother, L1Sparsity, Umeas: numpy array [num_pattern, num_electrodes]

### Community 19 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 20 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 21 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 22 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 23 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 24 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 25 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 26 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 27 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 28 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 42 - "algorithms.py"
Cohesion: 0.23
Nodes (14): available(), get_spec(), _initial_guess(), _prior_matrix(), Merge, in increasing priority: algorithm defaults, ``common`` settings that…, Run one algorithm. Returns ``(sigma_dg0, info)`` with ``sigma_dg0`` a plain…, Build (or fetch from ``cache``) the prior matrix for a Gauss-Newton run. The…, resolve_params() (+6 more)

### Community 43 - "`CONFIG` sections"
Cohesion: 0.22
Nodes (9): 6. Step 2 — `simulate.py`, `algorithms`, `common`, `CONFIG` sections, `ground_truth`, `metrics`, `noise`, `plot` (+1 more)

### Community 45 - "Reproducing the 104 mm, 256-electrode column experiment"
Cohesion: 0.29
Nodes (4): Algorithms, Mesh and measurement choices, Reproducing the 104 mm, 256-electrode column experiment, Results

### Community 46 - "8. The algorithms and their parameters"
Cohesion: 0.29
Nodes (7): 8. The algorithms and their parameters, `dbar` — the D-bar method (Nachman's direct algorithm), `gn` — Gauss-Newton with a quadratic prior, `l1` — L1-sparsity, `linear` — one-step linearised, Shared, `tv` — Gauss-Newton with total variation

### Community 47 - "SI EIT laboratory: 2D, 3D and frequency"
Cohesion: 0.33
Nodes (6): Frequency-dependent physics, Geometry and SI units, Reconstruction and outputs, SI EIT laboratory: 2D, 3D and frequency, Start here, Validation

### Community 48 - "9. How to choose parameters"
Cohesion: 0.33
Nodes (6): 9. How to choose parameters, `alpha` for `l1` is *not* auto-scaled, Contact impedance, `lambda` is dimensionless, Mesh resolution, Use χ² to tune, not guesswork

### Community 49 - "11. Output files"
Cohesion: 0.67
Nodes (3): 11. Output files, Experiment, Ground truth

## Knowledge Gaps
- **136 isolated node(s):** `graphify`, `Usage`, `What graphify is for`, `Step 0 - GitHub repos and multi-path merge (only if a URL or several paths)`, `Step 1 - Ensure graphify is installed` (+131 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EIT` connect `EIT` to `test_dbar.py`, `test_framework.py`, `volume_benchmark.py`, `simulate.py`, `Reconstructor`, `.__init__`, `_GaussNewtonBase`, `GaussNewtonSolverTV`, `GaussNewtonSolver`, `L1Sparsity`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `build_mesh()` connect `build_mesh` to `test_dbar.py`, `test_framework.py`, `volume_benchmark.py`, `simulate.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Why does `normalize_config()` connect `volume_benchmark.py` to `build_mesh`, `EITApp`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `EIT` (e.g. with `_GaussNewtonBase` and `GaussNewtonSolver`) actually correct?**
  _`EIT` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Usage`, `What graphify is for` to the rest of the system?**
  _136 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_dbar.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07515734912995187 - nodes in this community are weakly interconnected._
- **Should `test_framework.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05721003134796238 - nodes in this community are weakly interconnected._