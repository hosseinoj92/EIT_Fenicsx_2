"""Small executable development smoke check (also used by test_physical)."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dolfinx.fem import Function
from mesh_tools import build_mesh
from eitcore import EIT


def make_solver(dim, frequency=10000, eps=10000, thickness=0.02, capacitance=0):
    mesh = dict(dimension=dim, radius=0.05, height=0.1,
                electrodes_per_ring=4, n_rings=2, ring_spacing=0.03,
                electrode_diameter=0.01, h_center=0.025, h_edge=0.015)
    if dim == 2:
        mesh['n_electrodes'] = 8
    path = build_mesh(mesh)
    L = 8
    I = (np.eye(L)-np.roll(np.eye(L), 1, axis=1))*0.001
    solver = EIT(L, I, 0.005, mesh_name=path, dimension=dim,
                 frequency_hz=frequency, relative_permittivity=eps,
                 thickness=thickness, contact_capacitance=capacitance,
                 check_injection=False)
    sigma = Function(solver.V_sigma)
    sigma.x.array[:] = 0.2
    return solver, sigma


if __name__ == '__main__':
    for dim in (2, 3):
        solver, sigma = make_solver(dim)
        _, U = solver.forward_solve(sigma)
        U = np.array(U)
        J = solver.calc_jacobian(sigma)
        print(dim, solver.n_cells, np.max(np.abs(U)), np.max(np.abs(U.sum(axis=1))), J.shape)
