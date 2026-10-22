"""Cross-check scalable contact assembly, gradients and compressed Jacobian."""
import json
import pathlib
import sys
import unittest
import tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from check_ac_smoke import make_solver
from eitcore.physical import default_config, normalize_config
from eitcore.physical_workflow import _solver
from eitcore.volume_benchmark import ParameterGrid, Measurements, compressed_jacobian


class VolumeTests(unittest.TestCase):
    def test_complete_algorithm_exports(self):
        from eitcore.volume_benchmark import run
        cfg=default_config(3)
        cfg['mesh'].update(n_electrodes=8,electrodes_per_ring=4,h_center=.02,h_edge=.012)
        cfg['frequencies_hz']=[10000.]
        cfg['inclusions'][0]['material']={'conductivity_s_m':0.,'relative_permittivity':3.}
        cfg['reconstruction'].update(algorithms=['linear','gn','tv','l1','dbar'],
            reconstruct_permittivity=False,iterations=2,measurement_stride=1,
            parameter_grid={'nx':4,'ny':4,'z_edges_m':[0,.025,.05,.075,.1]})
        with tempfile.TemporaryDirectory() as temp:
            cfg['reconstruction']['output_directory']=temp
            out=run(cfg,log=lambda s:None)
            for algorithm in ('linear','gn','tv','l1'):
                folder=out/'reconstructions'/'10000Hz'/algorithm
                info=json.loads((folder/'metrics.json').read_text())
                self.assertEqual(info['status'],'complete')
                self.assertLessEqual(info['final_objective'],info['initial_objective'])
                for artifact in ('reconstruction.vtu','reconstruction.vtk','reconstruction_3d.png','reconstruction_sections.png'):
                    self.assertGreater((folder/artifact).stat().st_size,1000)
            unsupported=json.loads((out/'reconstructions'/'10000Hz'/'dbar'/'metrics.json').read_text())
            self.assertEqual(unsupported['status'],'unsupported')

    def test_exact_geometry_assembly_and_gradients(self):
        for dimension in (2,3):
            solver,sigma=make_solver(dimension)
            original=solver.M.copy()
            measures=solver._electrode_measures_geometry()
            np.testing.assert_allclose(measures,solver.electrode_lengths,rtol=1e-12)
            fast=solver._assemble_contact_geometry()
            np.testing.assert_allclose(fast.toarray(),original.toarray(),rtol=1e-11,atol=1e-12)
            u,_=solver.forward_solve(sigma)
            actual=solver.cellwise_gradients_fast(u[:2])
            reference=solver._cellwise_gradients(u[:2])
            np.testing.assert_allclose(actual,reference,rtol=1e-10,atol=1e-10)

    def test_compressed_jacobian_finite_difference(self):
        cfg=default_config(3)
        cfg['mesh'].update(n_electrodes=8,electrodes_per_ring=4,h_center=.025,h_edge=.015)
        cfg['reconstruction'].update(algorithms=['linear','gn','tv','l1'],
            reconstruct_permittivity=False,parameter_grid={'nx':4,'ny':4,'z_edges_m':[0,.025,.05,.075,.1]})
        cfg=normalize_config(cfg)
        solver,sigma=_solver(cfg,cfg['mesh'],10000)
        sigma.x.array[:]=.2
        grid=ParameterGrid(solver,cfg['reconstruction']['parameter_grid'],.05)
        measurement=Measurements(cfg)
        u,_=solver.forward_solve(sigma)
        J=compressed_jacobian(solver,sigma,grid,measurement,u,log=lambda s:None)
        direction=np.random.default_rng(24).normal(size=grid.n)
        h=1e-6
        sigma.x.array[:]=.2+h*grid.to_cells(direction)
        _,plus=solver.forward_solve(sigma)
        sigma.x.array[:]=.2-h*grid.to_cells(direction)
        _,minus=solver.forward_solve(sigma)
        fd=(measurement.values(plus)-measurement.values(minus))/(2*h)
        np.testing.assert_allclose(fd,-J@direction,rtol=1e-5,atol=1e-9)


if __name__=='__main__':
    unittest.main(verbosity=2)
