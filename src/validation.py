"""
Statistical validation of 2D percolation critical exponents.

Compares simulation results against exact renormalization-group predictions
for the universality class of 2D site percolation on the square lattice.

The key exponent extracted here is the *correlation-length exponent* ν = 4/3,
which governs how the characteristic cluster size diverges at p_c::

    ξ ~ |p - p_c|^{-ν}

We measure it indirectly via finite-size scaling of the susceptibility proxy
(second-largest cluster):  at the critical point

    χ(p_c, L) ≡ L2_abs ~ L^{γ/ν}

where γ/ν = 43/24 ≈ 1.7917 (exact RG result).  Fitting the slope of
log(L2_abs) vs log(L) gives γ/ν, from which

    ν = γ / (γ/ν),   γ = 43/18 ≈ 2.3889.

Why susceptibility and not the order parameter: the order-parameter slope
β/ν = 5/48 ≈ 0.104 is very small and hard to resolve with modest system
sizes.  The susceptibility exponent γ/ν ≈ 1.79 is much larger and converges
with fewer simulation points.
"""

import numpy as np
from scipy.stats import linregress

from .simulation import generate_grid, analyze_percolation


# ── Exact RG values for 2D site percolation (square lattice) ────────────────
PC_THEORY: float = 0.59274605  # percolation threshold (Ziff, 2021 best estimate)
NU_THEORY: float = 4 / 3       # correlation-length exponent
GAMMA_NU_THEORY: float = 43 / 24   # γ/ν — susceptibility scaling exponent
GAMMA_THEORY: float = 43 / 18      # γ — susceptibility exponent


def check_critical_exponents(
    L_values: tuple = (50, 75, 100, 150, 200),
    n_samples: int = 8,
    base_seed: int = 777,
    inc: int = 13,
) -> float:
    """
    Estimate the correlation-length exponent ν via finite-size scaling.

    At the percolation threshold the second-largest cluster (susceptibility
    proxy) scales with lattice size as::

        ⟨L2_abs⟩(p_c, L) ~ L^{γ/ν}

    We measure ⟨L2_abs⟩ at ``p = p_c`` for several values of *L*, fit
    log⟨L2_abs⟩ = (γ/ν) · log(L) + const via OLS, and infer

        ν = γ / (γ/ν_measured)

    Why average over *n_samples* realisations per size: the second-largest
    cluster has high variance near p_c (it fluctuates between 0 and L1).
    Averaging suppresses noise so the power-law fit is dominated by the true
    scaling rather than statistical outliers.

    Parameters
    ----------
    L_values : tuple of int
        Lattice sizes to probe.  More sizes and a wider range give a better
        estimate of the slope.
    n_samples : int
        Number of independent lattice realisations averaged per size
        (default 8; increase to 30+ for publication-quality results).
    base_seed : int
        Base RNG seed.  Realisation *k* of size index *i* uses
        ``base_seed + i * 10000 + k`` to guarantee independence.
    inc : int
        PCG32 stream selector.

    Returns
    -------
    float
        Estimated correlation-length exponent ν.
    """
    L_arr = np.array(L_values, dtype=float)
    mean_l2_abs = np.empty(len(L_values))

    for i, L in enumerate(L_values):
        l2_abs_samples = []
        for k in range(n_samples):
            # Why separate seeds per (i, k): reusing seeds across sizes would
            # introduce correlations that bias the finite-size scaling fit.
            seed = np.uint64(base_seed + i * 10_000 + k)
            grid = generate_grid(L, PC_THEORY, seed, np.uint64(inc))
            _, l2 = analyze_percolation(grid)
            # Convert normalised l2 back to absolute size for power-law fit
            l2_abs_samples.append(l2 * L * L)
        mean_l2_abs[i] = float(np.mean(l2_abs_samples))

    # Why skip sizes where mean_l2_abs ≈ 0: log(0) is undefined and would
    # corrupt the regression.  This can happen for very small L at p_c.
    mask = mean_l2_abs > 0
    if mask.sum() < 2:
        raise RuntimeError(
            "Fewer than two non-zero L2 measurements — increase n_samples or L_values."
        )

    slope, _, r_value, _, std_err = linregress(
        np.log(L_arr[mask]), np.log(mean_l2_abs[mask])
    )

    gamma_nu_measured = float(slope)
    # Why divide γ by the measured γ/ν: ν = γ / (γ/ν) by definition,
    # and γ = 43/18 is known exactly from conformal field theory.
    nu_measured = GAMMA_THEORY / gamma_nu_measured

    print(
        f"Finite-size scaling fit (χ ~ L^(γ/ν)):\n"
        f"  Measured γ/ν = {gamma_nu_measured:.4f}  "
        f"| Theory: {GAMMA_NU_THEORY:.4f}  (R²={r_value**2:.4f})\n"
        f"  Estimated  ν = {nu_measured:.4f}       "
        f"| Theory: {NU_THEORY:.4f}"
    )
    return nu_measured
