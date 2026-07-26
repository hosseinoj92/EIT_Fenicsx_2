from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class Reconstructor(ABC):
    def __init__(self, eit_solver):
        self.eit_solver = eit_solver

    @abstractmethod
    def forward(self, U, **kwargs):
        pass

    # ------------------------------------------------------------------ #
    def cell_centers(self):
        """``(n_cells, 2)`` centroids of the reconstruction mesh."""
        return self.eit_solver.cell_centers()

    def to_dg0(self, func):
        """
        Return the coefficient vector of ``func`` on the DG0 (cell-wise) space.

        Reconstructors do not all live in the same space - the L1-sparsity
        solver works in CG1 while the Gauss-Newton solvers work in DG0 - so
        anything that compares or plots reconstructions has to go through here
        first.  Plotting a CG1 array with ``shading="flat"`` silently mismatches
        lengths, and comparing arrays of different lengths is meaningless.
        """
        from dolfinx.fem import Function

        V_sigma = self.eit_solver.V_sigma
        if func.function_space == V_sigma or len(func.x.array) == len(
            Function(V_sigma).x.array
        ):
            return np.array(func.x.array, copy=True)

        out = Function(V_sigma)
        out.interpolate(func)
        return np.array(out.x.array, copy=True)

    def interpolate_to_image(self, sigma, resolution=256, fill_value=np.nan,
                             extent=None, method="linear"):
        """
        Resample a cell-wise field onto a regular pixel grid.

        Returns ``(image, extent)`` with ``image`` indexed ``[row, col]`` and
        ``extent = (xmin, xmax, ymin, ymax)`` ready for ``plt.imshow``.  Pixels
        outside the convex hull of the mesh get ``fill_value``.
        """
        sigma = np.asarray(sigma).flatten()
        pos = self.cell_centers()
        if len(sigma) != len(pos):
            raise ValueError(
                f"sigma has {len(sigma)} values but the mesh has {len(pos)} cells"
            )

        if extent is None:
            xmin, ymin = pos.min(axis=0)
            xmax, ymax = pos.max(axis=0)
        else:
            xmin, xmax, ymin, ymax = extent

        px = np.linspace(xmin, xmax, resolution)
        py = np.linspace(ymin, ymax, resolution)
        X, Y = np.meshgrid(px, py)
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))

        if method == "nearest":
            interp = NearestNDInterpolator(pos, sigma)
            grid = interp(pixcenters[:, 0], pixcenters[:, 1])
        else:
            interp = LinearNDInterpolator(pos, sigma, fill_value=fill_value)
            grid = interp(pixcenters)

        return np.flipud(grid.reshape(resolution, resolution)), (xmin, xmax, ymin, ymax)
