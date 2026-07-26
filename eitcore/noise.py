# noise.py

"""
Measurement-noise models for simulated EIT data.

Why this is its own module
--------------------------
The obvious model - ``U += p * |U| * randn`` - scales the noise with the
magnitude of each individual measurement.  That makes the measurements that
happen to sit near a voltage zero-crossing *arbitrarily accurate*, which no
real instrument does: a real EIT system has a noise floor set by the ADC and
front-end, roughly constant across the channels of one frame.  Inverting data
with an unphysically accurate subset of channels flatters every algorithm, and
flatters them unequally, so it corrupts exactly the comparison this framework
exists to make.

The default here is therefore ``relative_to_max``: sigma_k = p * max_k |U_k|,
i.e. a constant absolute noise floor expressed as a percentage of the signal
range.  The other models are available for comparison.

Every model returns ``(U_noisy, sigma_noise)`` where ``sigma_noise`` is the
per-measurement standard deviation.  Pass ``1 / sigma_noise**2`` as ``GammaInv``
to the reconstructors so that the statistical weighting matches the noise that
was actually added.
"""

import numpy as np

MODELS = ("relative_to_max", "relative_to_pattern_max", "relative_per_measurement",
          "absolute", "none")


def noise_std(U, model="relative_to_max", level=1.0, floor_fraction=0.0):
    """
    Per-measurement standard deviation for the clean data ``U`` (N, L).

    level:  percent for the relative models, absolute volts for ``absolute``.
    floor_fraction: adds ``floor_fraction * level/100 * max|U|`` to every entry,
        useful with ``relative_per_measurement`` to keep a physical noise floor.
    """
    U = np.asarray(U, dtype=float)
    p = float(level) / 100.0

    if model == "none" or level == 0:
        return np.zeros_like(U)

    if model == "relative_to_max":
        std = np.full_like(U, p * np.max(np.abs(U)))
    elif model == "relative_to_pattern_max":
        std = p * np.max(np.abs(U), axis=-1, keepdims=True) * np.ones_like(U)
    elif model == "relative_per_measurement":
        std = p * np.abs(U)
    elif model == "absolute":
        std = np.full_like(U, float(level))
    else:
        raise ValueError(f"Unknown noise model '{model}'. Choices: {MODELS}")

    if floor_fraction:
        std = std + floor_fraction * p * np.max(np.abs(U))

    return std


def add_measurement_noise(U, model="relative_to_max", level=1.0, seed=None,
                          floor_fraction=0.0, rng=None,
                          electrode_gain_percent=0.0, electrode_offset=0.0,
                          n_electrodes=None):
    """
    Add measurement noise to clean electrode voltages.

    Parameters
    ----------
    U : array (N_patterns, L)
        Noise-free simulated voltages.
    model, level, floor_fraction
        See :func:`noise_std`.
    electrode_gain_percent : float
        Systematic per-electrode gain error (percent, drawn once and applied to
        every pattern).  Models calibration error, which unlike white noise is
        *correlated across patterns* and is what usually limits absolute EIT.
    electrode_offset : float
        Systematic per-electrode additive offset, same idea.

    Returns
    -------
    U_noisy, sigma_noise
    """
    U = np.asarray(U, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)

    std = noise_std(U, model=model, level=level, floor_fraction=floor_fraction)

    U_noisy = U.copy()

    if electrode_gain_percent or electrode_offset:
        L = n_electrodes if n_electrodes is not None else U.shape[-1]
        gain = 1.0 + (electrode_gain_percent / 100.0) * rng.standard_normal(L)
        offset = electrode_offset * rng.standard_normal(L)
        U_noisy = U_noisy * gain + offset

    U_noisy = U_noisy + std * rng.standard_normal(U.shape)

    return U_noisy, std


def gamma_inv(sigma_noise, relative_floor=1e-3):
    """
    Inverse noise covariance (diagonal) for the reconstructors.

    ``relative_floor`` is expressed **relative to the mean variance**, not as an
    absolute constant.  The original code added a fixed ``1e-3`` to the
    variance; for typical EIT voltages the variance is around ``1e-6``, so that
    constant dominated completely and silently replaced the statistical
    weighting with a uniform one.
    """
    var = np.asarray(sigma_noise, dtype=float).flatten() ** 2
    if not np.any(var > 0):
        return np.ones_like(var)
    floor = relative_floor * np.mean(var[var > 0])
    return 1.0 / (var + floor)


def snr_db(U_clean, U_noisy):
    """Signal-to-noise ratio of a noisy data set, in dB."""
    U_clean = np.asarray(U_clean, dtype=float)
    noise = np.asarray(U_noisy, dtype=float) - U_clean
    p_sig = np.mean(U_clean**2)
    p_noise = np.mean(noise**2)
    if p_noise <= 0:
        return float("inf")
    return float(10.0 * np.log10(p_sig / p_noise))
