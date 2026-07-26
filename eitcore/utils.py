# utils.py

"""
Interpolation helpers and current-injection (drive) patterns for the CEM.

Physical constraint
-------------------
For the Complete Electrode Model the injected currents of every pattern must
satisfy charge conservation

    sum_i I_i = 0

otherwise the Neumann problem is not solvable.  Every pattern produced here is
checked against that condition (see :func:`validate_injection`).
"""

import numpy as np
from scipy.interpolate import interpn, NearestNDInterpolator, LinearNDInterpolator


# ---------------------------------------------------------------------------
#  interpolation helpers
# ---------------------------------------------------------------------------
def image_to_mesh(x, mesh_pos, fill_value=1.0, method="nearest"):
    """Sample a square image ``x`` (indexed [row, col]) at ``mesh_pos`` (N, 2)."""
    radius = np.max(np.abs(mesh_pos))

    pixcenter_x = pixcenter_y = np.linspace(-radius, radius, x.shape[-1])

    sigma = interpn(
        [pixcenter_x, pixcenter_y],
        np.flipud(x).T,
        mesh_pos,
        bounds_error=False,
        fill_value=fill_value,
        method=method,
    )

    return sigma


def interpolate_mesh_to_mesh(x, mesh_pos1, mesh_pos2, method="nearest"):
    """
    Transfer a cell-wise field from one mesh to another.

    ``method="nearest"`` is discontinuity preserving (good for piecewise-constant
    phantoms), ``method="linear"`` is smoother but blurs sharp inclusion edges.
    Points of ``mesh_pos2`` outside the convex hull of ``mesh_pos1`` always fall
    back to nearest-neighbour so no NaNs are produced.
    """
    mesh_pos1 = np.asarray(mesh_pos1, dtype=float)
    mesh_pos2 = np.asarray(mesh_pos2, dtype=float)

    nearest = NearestNDInterpolator(mesh_pos1, x)
    if method == "nearest":
        return nearest(mesh_pos2[:, 0], mesh_pos2[:, 1])

    linear = LinearNDInterpolator(mesh_pos1, x, fill_value=np.nan)
    sigma = linear(mesh_pos2[:, 0], mesh_pos2[:, 1])
    bad = ~np.isfinite(sigma)
    if np.any(bad):
        sigma[bad] = nearest(mesh_pos2[bad, 0], mesh_pos2[bad, 1])
    return sigma


# ---------------------------------------------------------------------------
#  drive patterns
# ---------------------------------------------------------------------------
DRIVE_NAMES = {
    1: "opposite",
    2: "adjacent",
    3: "one-against-all",
    4: "trigonometric",
    5: "all-against-one",
}


def default_n_patterns(method: int, L: int) -> int:
    """
    Canonical number of linearly independent current patterns for a drive
    ``method`` on ``L`` electrodes.

    Note that a CEM measurement set can never contain more than ``L - 1``
    linearly independent patterns (the all-ones vector is excluded by charge
    conservation), so no method returns more than that.
    """
    if method == 1:  # opposite pairs: (i, i+L/2) for i < L/2
        return L // 2
    elif method == 2:  # adjacent pairs, the L-th is a linear combination
        return L - 1
    elif method == 3:  # one against all others
        return L - 1
    elif method == 4:  # trigonometric basis
        return L - 1
    elif method == 5:  # all against a common reference electrode
        return L - 1
    else:
        raise ValueError(f"Unknown drive method {method}. Choices: {sorted(DRIVE_NAMES)}")


def validate_injection(Inj, atol=1e-10):
    """
    Check charge conservation and rank of an injection matrix ``(N, L)``.

    Returns ``(ok, message)``.  Patterns that do not sum to zero make the CEM
    forward problem inconsistent; zero rows and rank deficiency mean the
    measurement set carries less information than its size suggests.
    """
    Inj = np.atleast_2d(np.asarray(Inj, dtype=float))
    problems = []

    charge = np.abs(Inj.sum(axis=1))
    if np.any(charge > atol):
        bad = np.flatnonzero(charge > atol)
        problems.append(
            f"patterns {bad.tolist()} violate charge conservation "
            f"(max |sum I| = {charge.max():.3e})"
        )

    norms = np.linalg.norm(Inj, axis=1)
    zero_rows = np.flatnonzero(norms <= atol)
    if zero_rows.size:
        problems.append(f"patterns {zero_rows.tolist()} are identically zero")

    rank = np.linalg.matrix_rank(Inj)
    if rank < Inj.shape[0]:
        problems.append(f"rank {rank} < {Inj.shape[0]} patterns (redundant measurements)")

    return (not problems), "; ".join(problems)


def current_method(L, l, method=1, value=1.0):
    """
    Build the current-injection matrix ``(l, L)``: one row per pattern, one
    column per electrode.

    :param L: number of electrodes
    :param l: number of injection patterns requested
    :param method: 1..5, see below
    :param value: injected current amplitude

    :Method Values:
        1. ``+1`` / ``-1`` on opposite electrodes ``(i, i + L/2)``.
        2. ``+1`` / ``-1`` on adjacent electrodes ``(i, i + 1)``.
        3. ``+1`` on one electrode, ``-1/(L-1)`` on all others.
        4. Trigonometric (Cheney/Isaacson) basis: ``cos(k*theta_j)`` for
           ``k = 1 .. L/2`` and ``sin(k*theta_j)`` for ``k = 1 .. L/2 - 1``
           with ``theta_j = 2*pi*j/L``.  These are the ``L-1`` mutually
           orthogonal, charge-conserving patterns that maximise distinguish-
           ability for a homogeneous disk.
        5. All electrodes measured against electrode 0 as common reference.

    Every returned pattern satisfies ``sum_i I_i = 0``.
    """
    if method not in DRIVE_NAMES:
        raise ValueError(f"Unknown drive method {method}. Choices: {sorted(DRIVE_NAMES)}")

    L = int(L)
    l = int(l)
    if l < 1:
        raise ValueError("Need at least one injection pattern (l >= 1)")

    l_max = default_n_patterns(method, L)
    if l > l_max:
        raise ValueError(
            f"Drive method {method} ({DRIVE_NAMES[method]}) on L={L} electrodes "
            f"supports at most {l_max} independent patterns, got l={l}. "
            f"Use utils.default_n_patterns({method}, {L})."
        )

    I_all = []

    if method == 1:  # opposite pairs
        if L % 2 != 0:
            raise ValueError(
                f"Drive method 1 (opposite) needs an even number of electrodes, got L={L}."
            )
        for i in range(l):
            I = np.zeros(L)
            I[i], I[i + L // 2] = value, -value
            I_all.append(I)

    elif method == 2:  # adjacent pairs
        for i in range(l):
            I = np.zeros(L)
            I[i], I[(i + 1) % L] = value, -value
            I_all.append(I)

    elif method == 3:  # one against all
        for i in range(l):
            I = np.full(L, -value / (L - 1))
            I[i] = value
            I_all.append(I)

    elif method == 4:  # trigonometric basis
        j = np.arange(L)
        theta = 2.0 * np.pi * j / L
        basis = []
        for k in range(1, L // 2 + 1):
            basis.append(np.cos(k * theta))
            if k < L // 2:  # sin(L/2 * theta) is identically zero
                basis.append(np.sin(k * theta))
        for i in range(l):
            pattern = basis[i]
            # normalise so the amplitude means "peak current", as for 1/2/5
            I_all.append(value * pattern / np.max(np.abs(pattern)))

    elif method == 5:  # all against electrode 0
        for i in range(l):
            I = np.zeros(L)
            I[0] = -value
            I[i + 1] = value
            I_all.append(I)

    Inj = np.asarray(I_all, dtype=float)

    # numerical hygiene: remove round-off in the charge balance
    Inj -= Inj.mean(axis=1, keepdims=True)

    return Inj


if __name__ == "__main__":
    for m in sorted(DRIVE_NAMES):
        n = default_n_patterns(m, 16)
        Inj = current_method(L=16, l=n, method=m, value=1.0)
        ok, msg = validate_injection(Inj)
        print(f"method {m} ({DRIVE_NAMES[m]:<16}) l={n:3d} shape={Inj.shape} "
              f"rank={np.linalg.matrix_rank(Inj)} ok={ok} {msg}")
