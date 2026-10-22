"""Physics regressions for SI 2D/3D AC CEM. Run with unittest or pytest.

Pure material/config tests also run on Windows without FEniCSx. Numerical
checks skip explicitly when FEniCSx is absent; use the WSL installation for all.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eitcore.physical import (default_config, normalize_config, material_admittivity,
                               drive_patterns, sample_materials, EPSILON_0)
from mesh_tools import _merge, electrode_centres, ring_heights


class MaterialTests(unittest.TestCase):
    def test_resistive_limit_and_passive_debye(self):
        mat = {"conductivity_s_m": 0.2, "relative_permittivity": 0}
        for f in (0, 100, 1e6):
            self.assertEqual(material_admittivity(mat, f), 0.2)
        mat.update(relative_permittivity=3, delta_epsilon_r=100, relaxation_time_s=1e-4)
        f = 1/(2*np.pi*mat["relaxation_time_s"])
        g = material_admittivity(mat, f)
        self.assertAlmostEqual(g.real, .2+EPSILON_0*100/(2*1e-4))
        self.assertAlmostEqual(g.imag/(2*np.pi*f*EPSILON_0), 53)

    def test_configuration_and_geometries(self):
        for dim in (2, 3):
            cfg = normalize_config(default_config(dim))
            I = drive_patterns(cfg)
            self.assertTrue(np.allclose(I.sum(axis=1), 0))
            self.assertAlmostEqual(np.abs(I).max(), .001)
            self.assertEqual(np.linalg.matrix_rank(I), I.shape[1]-1)
            centers = electrode_centres(cfg["mesh"])
            self.assertEqual(centers.shape, (cfg["mesh"]["n_electrodes"], dim))
            points = np.array([cfg["inclusions"][0]["center_m"], [0]*dim])
            sig, eps = sample_materials(cfg, points, 10000)
            self.assertEqual(sig[0], .5)
            self.assertAlmostEqual(eps[0], 30000)
        m = _merge(default_config(3)["mesh"])
        self.assertTrue(np.allclose(np.diff(ring_heights(m)), m["ring_spacing"]))

    def test_invalid_units_geometry_material(self):
        for key, val in [("ring_spacing", .001), ("electrode_diameter", .1),
                         ("n_rings", 1.5), ("h_center", float("nan"))]:
            cfg = default_config(3)
            cfg["mesh"][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_config(cfg)
        cfg = default_config(2)
        cfg["units"]["length"] = "mm"
        with self.assertRaises(ValueError):
            normalize_config(cfg)
        with self.assertRaises(ValueError):
            material_admittivity({"conductivity_s_m": .2, "relative_permittivity": -1}, 10)


HAS_FENICS = importlib.util.find_spec("dolfinx") is not None


@unittest.skipUnless(HAS_FENICS, "FEniCSx not installed in this Python; run under WSL")
class CEMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from check_ac_smoke import make_solver
        cls.solvers = {dim: make_solver(dim, frequency=100000) for dim in (2, 3)}

    def test_gauge_reciprocity_and_jacobian(self):
        for dim, (solver, sig) in self.solvers.items():
            with self.subTest(dimension=dim):
                u, U = solver.forward_solve(sig)
                U = np.asarray(U)
                self.assertLess(np.max(np.abs(U.sum(axis=1))), 1e-12)
                transfer = solver.Inj@U.T
                np.testing.assert_allclose(transfer, transfer.T, rtol=1e-10, atol=1e-13)
                self.assertTrue(np.all(np.real(np.sum(solver.Inj*U, axis=1)) > 0))
                J = solver.calc_jacobian(sig, u)
                rng = np.random.default_rng(12)
                direction = rng.normal(size=solver.n_cells)
                h = 1e-5
                base = sig.x.array.copy()
                sig.x.array[:] = base+h*direction
                _, plus = solver.forward_solve(sig)
                sig.x.array[:] = base-h*direction
                _, minus = solver.forward_solve(sig)
                sig.x.array[:] = base
                fd = (np.asarray(plus)-minus).ravel()/(2*h)
                np.testing.assert_allclose(fd, -J@direction, rtol=2e-5, atol=1e-8)
                # dU/depsilon_r = i omega epsilon0 dU/dsigma.
                eps0 = solver.relative_permittivity.x.array.copy()
                he = 1
                solver.relative_permittivity.x.array[:] = eps0+he*direction
                _, plus = solver.forward_solve(sig)
                solver.relative_permittivity.x.array[:] = eps0-he*direction
                _, minus = solver.forward_solve(sig)
                solver.relative_permittivity.x.array[:] = eps0
                fd = (np.asarray(plus)-minus).ravel()/(2*he)
                np.testing.assert_allclose(fd, -1j*2*np.pi*solver.frequency_hz*EPSILON_0*(J@direction), rtol=3e-5, atol=1e-10)

    def test_current_thickness_and_analytic_complex_scaling(self):
        from check_ac_smoke import make_solver
        for dim in (2, 3):
            s0, sig0 = make_solver(dim, frequency=0, eps=0)
            _, U0 = s0.forward_solve(sig0)
            _, twice = s0.forward_solve(sig0, 2*s0.Inj)
            np.testing.assert_allclose(twice, 2*np.asarray(U0), rtol=1e-11, atol=1e-12)
            resistive, sig = make_solver(dim, frequency=100000, eps=0)
            _, Ures = resistive.forward_solve(sig)
            np.testing.assert_allclose(Ures, U0, rtol=1e-11, atol=1e-12)
            beta, f = .7, 100000
            eps = beta*.2/(2*np.pi*f*EPSILON_0)
            cap = beta/(2*np.pi*f*.005)
            sac, sig = make_solver(dim, frequency=f, eps=eps, capacitance=cap)
            _, Uac = sac.forward_solve(sig)
            np.testing.assert_allclose(Uac, np.asarray(U0)/(1+1j*beta), rtol=2e-10, atol=1e-11)
            self.assertTrue(np.all(sac.electrode_areas > 0))
            if dim == 2:
                thick, sig = make_solver(2, frequency=0, eps=0, thickness=.04)
                _, Uthick = thick.forward_solve(sig)
                np.testing.assert_allclose(Uthick, np.asarray(U0)/2, rtol=2e-10, atol=1e-11)
            else:
                expected = np.pi*(.01/2)**2
                # Curved wall area is slightly larger than tangent-plane disk.
                np.testing.assert_allclose(sac.electrode_areas, expected, rtol=.05)
                vol = float(np.sum(sac.cell_area))
                self.assertAlmostEqual(vol/(np.pi*.05**2*.1), 1, delta=.02)

    def test_nonuniform_rotated_ring_layout(self):
        from eitcore.physical_workflow import _solver
        cfg = default_config(3)
        cfg["mesh"].update(height=.12, n_electrodes=15, electrodes_per_ring=5,
                            n_rings=3, ring_z=[.025, .055, .095],
                            electrode_diameter=.01, electrode_offset_deg=11,
                            ring_rotation_deg=17, h_center=.03, h_edge=.015)
        cfg = normalize_config(cfg)
        solver, sigma = _solver(cfg, cfg["mesh"], 10000)
        sigma.x.array[:] = .2
        _, U = solver.forward_solve(sigma)
        self.assertEqual(np.asarray(U).shape, (25, 15))
        self.assertTrue(np.all(solver.electrode_areas > 0))
        np.testing.assert_allclose(solver.electrode_areas, np.pi*.005**2, rtol=.07)
        centers = electrode_centres(cfg["mesh"])
        np.testing.assert_allclose(centers[::5, 2], [.025, .055, .095])
        self.assertLess(np.max(np.abs(np.asarray(U).sum(axis=1))), 1e-12)

    def test_end_to_end_exports_and_reconstruction(self):
        from eitcore.physical_workflow import run
        for dim in (2, 3):
            with self.subTest(dimension=dim), tempfile.TemporaryDirectory() as temp:
                cfg = default_config(dim)
                cfg["output_root"] = temp
                cfg["frequencies_hz"] = [10000, 100000]
                cfg["mesh"].update(n_electrodes=8, electrodes_per_ring=4,
                                    h_center=.02, h_edge=.012)
                cfg["reconstruction"].update(iterations=2, h_center=.03, h_edge=.015)
                out = run(cfg, log=lambda text: None)
                data = np.load(out/"measurements.npz")
                self.assertTrue(np.iscomplexobj(data["U_v"]))
                self.assertEqual(data["points_m"].shape[1], dim)
                self.assertGreater(np.linalg.norm(data["U_v"][0]-data["U_v"][1]), 1e-6)
                metrics = json.loads((out/"metrics.json").read_text())
                history = metrics["history"]
                self.assertTrue(history)
                self.assertGreater(history[0]["step_length"], 0)
                self.assertTrue(all(b["objective"] <= a["objective"] for a, b in zip(history, history[1:])))
                self.assertTrue((out/"reconstruction.vtu").exists())
                self.assertTrue((out/"frequency_response.png").stat().st_size > 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
