# metrics.py

"""
Reconstruction quality metrics.

Mesh independence
-----------------
This framework is meant to compare reconstructions produced on *different*
meshes.  Metrics computed by summing over the cells of whichever mesh a method
happened to use are not comparable: a finer mesh has more (smaller) cells, so
an unweighted cell sum silently reweights the error.  Every metric here is
therefore evaluated on one common, mesh-independent quadrature: a regular pixel
grid restricted to the domain, with all fields resampled onto it.

``MetricEvaluator`` is built once per ground truth and then applied to every
reconstruction, so the comparison basis is identical across algorithms, across
parameter settings, and across reconstruction meshes.
"""

import numpy as np
from scipy.interpolate import NearestNDInterpolator


class MetricEvaluator:
    """
    Common evaluation grid for one ground truth.

    Parameters
    ----------
    gt_points : (M, 2) coordinates where the ground truth is known
    gt_values : (M,) ground-truth conductivity at those points
    background : scalar background conductivity, used for the ROI mask
    resolution : pixels per axis of the evaluation grid
    radius : radius of the circular cross-section.  Pixels outside are dropped.
        ``None`` infers it from the extent of ``gt_points``.
    roi_threshold : relative deviation from ``background`` that counts as
        "an inclusion" for the ROI and Dice metrics.
    """

    def __init__(self, gt_points, gt_values, background=1.0, resolution=256,
                 radius=None, roi_threshold=0.05):
        gt_points = np.asarray(gt_points, dtype=float)
        gt_values = np.asarray(gt_values, dtype=float).flatten()

        self.background = float(background)
        self.roi_threshold = float(roi_threshold)
        self.resolution = int(resolution)

        if radius is None:
            radius = float(np.max(np.linalg.norm(gt_points, axis=1)))
        self.radius = float(radius)

        self.grid, self.inside = self._make_grid(self.radius, resolution)
        self.pixel_area = (2.0 * self.radius / (resolution - 1)) ** 2
        #: (xmin, xmax, ymin, ymax) in physical units; pass to imshow as
        #: ``extent=`` together with ``aspect="equal"``.
        self.extent = (-self.radius, self.radius, -self.radius, self.radius)

        self.gt = NearestNDInterpolator(gt_points, gt_values)(
            self.grid[:, 0], self.grid[:, 1]
        )
        dev = np.abs(self.gt - self.background)
        self.roi = dev > self.roi_threshold * max(abs(self.background), 1e-12)
        if not np.any(self.roi):
            # homogeneous phantom: ROI metrics degrade to global metrics
            self.roi = np.ones_like(self.gt, dtype=bool)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_grid(radius, resolution):
        """Regular grid over the bounding square, clipped to the circle."""
        ax = np.linspace(-radius, radius, resolution)
        X, Y = np.meshgrid(ax, ax)
        pts = np.column_stack((X.ravel(), Y.ravel()))
        keep = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= radius**2
        return pts[keep], keep

    # ------------------------------------------------------------------ #
    def resample(self, points, values):
        """Bring a cell-wise field onto the common evaluation grid."""
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float).flatten()
        if points.shape[0] != values.shape[0]:
            raise ValueError(
                f"{values.shape[0]} values for {points.shape[0]} points"
            )
        return NearestNDInterpolator(points, values)(self.grid[:, 0], self.grid[:, 1])

    def as_image(self, values_on_grid, fill=np.nan):
        """Reshape a grid vector back to a (resolution, resolution) image."""
        img = np.full(self.resolution * self.resolution, fill, dtype=float)
        img[self.inside] = values_on_grid
        return np.flipud(img.reshape(self.resolution, self.resolution))

    # ------------------------------------------------------------------ #
    def evaluate(self, points, values):
        """All metrics for one reconstruction. Returns a dict."""
        rec = self.resample(points, values)
        return self.evaluate_on_grid(rec)

    def evaluate_on_grid(self, rec):
        gt = self.gt
        diff = rec - gt
        roi = self.roi

        eps = 1e-30
        out = {
            "rel_L1": float(np.sum(np.abs(diff)) / (np.sum(np.abs(gt)) + eps)),
            "rel_L2": float(
                np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(gt**2)) + eps)
            ),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
            "rel_L1_roi": float(
                np.sum(np.abs(diff[roi])) / (np.sum(np.abs(gt[roi])) + eps)
            ),
            "rel_L2_roi": float(
                np.sqrt(np.sum(diff[roi] ** 2)) / (np.sqrt(np.sum(gt[roi] ** 2)) + eps)
            ),
            "dice": self._dice(rec, gt),
            "correlation": self._corr(rec, gt),
            "dynamic_range": self._dynamic_range(rec, gt),
            # deliberately NOT called sigma_min/sigma_max: those are the names
            # of the clip *parameters*, and a metric of the same name silently
            # overwrites them when both land in one results row.
            "recon_min": float(np.min(rec)),
            "recon_max": float(np.max(rec)),
            "roi_area": float(np.sum(roi) * self.pixel_area),
        }
        return out

    # ------------------------------------------------------------------ #
    def _mask(self, field):
        dev = np.abs(field - self.background)
        return dev > self.roi_threshold * max(abs(self.background), 1e-12)

    @staticmethod
    def _half_amplitude_mask(field, background):
        """
        "Detected inclusion" set: pixels deviating from the background by more
        than half the largest deviation present in that same field.

        A fixed absolute threshold cannot be used for the Dice score.  EIT
        reconstructions are smooth, so almost every pixel deviates from the
        background by more than a few percent; a fixed threshold therefore
        marks nearly the whole domain as "inclusion" and the score collapses to
        roughly the same small number for every algorithm, carrying no
        information.  Thresholding each field at half its own peak deviation is
        the standard full-width-half-maximum convention and does compare shapes.
        """
        dev = np.abs(field - background)
        peak = float(dev.max())
        if peak <= 1e-12:
            return np.zeros_like(dev, dtype=bool)
        return dev > 0.5 * peak

    def _dice(self, rec, gt):
        a = self._half_amplitude_mask(rec, self.background)
        b = self._half_amplitude_mask(gt, self.background)
        denom = a.sum() + b.sum()
        if denom == 0:
            return 1.0
        return float(2.0 * np.logical_and(a, b).sum() / denom)

    @staticmethod
    def _corr(rec, gt):
        if np.std(rec) < 1e-14 or np.std(gt) < 1e-14:
            return 0.0
        return float(np.corrcoef(rec, gt)[0, 1])

    @staticmethod
    def _dynamic_range(rec, gt):
        """
        Reconstructed contrast as a fraction of the true contrast.
        1.0 means the amplitude of the inclusions is recovered exactly;
        EIT reconstructions are typically well below 1 (over-smoothing).
        """
        gt_range = float(np.max(gt) - np.min(gt))
        if gt_range < 1e-14:
            return 1.0
        return float((np.max(rec) - np.min(rec)) / gt_range)


METRIC_NAMES = [
    "rel_L1", "rel_L2", "rmse", "mae", "rel_L1_roi", "rel_L2_roi",
    "dice", "correlation", "dynamic_range", "recon_min", "recon_max",
]

#: metrics where a *smaller* value is better
LOWER_IS_BETTER = {"rel_L1", "rel_L2", "rmse", "mae", "rel_L1_roi", "rel_L2_roi",
                   "data_residual", "data_residual_rel"}


def all_metrics(evaluator, points, values):
    return evaluator.evaluate(points, values)


def data_residual(U_sim, U_meas, sigma_noise=None):
    """
    Agreement in *data* space, which needs no ground truth at all.

    Returns the absolute residual norm, the residual relative to ``|U_meas|``,
    and - if the noise level is known - the chi-square per degree of freedom.
    A well-fitted reconstruction has ``chi2_per_dof`` near 1; much less means
    the noise is being fitted (over-fitting), much more means the data is not
    explained (over-regularised).
    """
    U_sim = np.asarray(U_sim, dtype=float).flatten()
    U_meas = np.asarray(U_meas, dtype=float).flatten()
    r = U_sim - U_meas

    out = {
        "data_residual": float(np.linalg.norm(r)),
        "data_residual_rel": float(
            np.linalg.norm(r) / max(np.linalg.norm(U_meas), 1e-30)
        ),
    }
    if sigma_noise is not None:
        s = np.asarray(sigma_noise, dtype=float).flatten()
        good = s > 0
        if np.any(good):
            out["chi2_per_dof"] = float(np.mean((r[good] / s[good]) ** 2))
    return out
