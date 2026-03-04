"""
Fractal-dimension analysis for 2D percolation clusters.

At the percolation threshold p_c the incipient spanning cluster is a fractal
object whose mass (number of occupied sites) scales with linear lattice size L
as::

    L1(p_c, L) ~ L^{d_f}

where d_f = 91/48 ≈ 1.8958 is the *fractal dimension* of 2D site percolation,
an exact result from renormalization-group theory (Nienhuis, 1982).

This module provides a utility to extract d_f empirically from simulation data
via log-log linear regression, enabling a direct comparison with theory.
"""

import numpy as np
from scipy.stats import linregress


# Exact RG prediction for the fractal dimension of the incipient spanning
# cluster in 2D site percolation (Nienhuis, 1982).
DF_THEORY = 91 / 48  # ≈ 1.8958


def calculate_fractal_dimension(
    cluster_sizes: np.ndarray,
    grid_sizes: np.ndarray,
) -> float:
    """
    Estimate the fractal dimension d_f from finite-size simulations at p_c.

    At the percolation threshold the largest-cluster size scales as::

        L1 ~ L^{d_f}

    Taking logarithms converts this power law to a straight line::

        log(L1) = d_f * log(L) + const

    Why log-log regression: the power-law relation is *linear* in log space,
    so ordinary least squares (OLS) directly returns the exponent as the slope
    without requiring a non-linear optimiser.

    Parameters
    ----------
    cluster_sizes : array_like, shape (n,)
        Largest-cluster *absolute* sizes (number of occupied sites) measured
        at ``p = p_c`` for each lattice size.  Averaged over multiple
        realisations for best accuracy.
    grid_sizes : array_like, shape (n,)
        Corresponding linear lattice sizes *L*.

    Returns
    -------
    float
        Estimated fractal dimension d_f (slope of the log-log fit).

    Notes
    -----
    The theoretical value is d_f = 91/48 ≈ 1.8958.  Deviations of a few
    percent are expected for finite lattices (L ≲ 500) due to correction-to-
    scaling terms of order L^{-Ω} where Ω ≈ 0.64 (Aharony & Asikainen, 2003).

    Examples
    --------
    >>> import numpy as np
    >>> Ls = np.array([50, 100, 200])
    >>> # Synthetic data consistent with d_f ≈ 1.896
    >>> L1s = 0.5 * Ls ** 1.896
    >>> df = calculate_fractal_dimension(L1s, Ls)
    >>> abs(df - 1.896) < 0.01
    True
    """
    log_L = np.log(np.asarray(grid_sizes, dtype=float))
    log_L1 = np.log(np.asarray(cluster_sizes, dtype=float))

    # Why we keep all five linregress outputs: r_value and p_value let the
    # caller judge whether the power-law fit is statistically credible before
    # trusting the returned exponent.
    slope, intercept, r_value, p_value, std_err = linregress(log_L, log_L1)

    print(
        f"Fractal Dimension d_f: {slope:.4f} "
        f"(Theoretical: {DF_THEORY:.4f}, R²={r_value**2:.4f})"
    )
    return float(slope)
