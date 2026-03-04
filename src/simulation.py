"""
2D Site-Percolation Simulation Core.

Generates random lattices with site occupation probability *p*, labels
connected clusters using Scipy's ``ndimage.label``, and extracts the sizes
of the two largest clusters (*L1* and *L2*).  These quantities are used to
locate the percolation threshold p_c ≈ 0.5927 on the infinite square lattice.

Scientific context
------------------
Site percolation on a 2-D square lattice is a canonical model for phase
transitions in disordered systems.  Above the critical probability p_c each
site is independently occupied with probability p, and a spanning (infinite)
cluster first appears.  The ratio L2/L1 peaks sharply near p_c and provides a
finite-size estimator of the threshold — an approach motivated by stochastic
star-formation models in which molecular-cloud connectivity drives the onset of
large-scale star-formation episodes.
"""

import numpy as np
from numba import njit
from scipy.ndimage import label

from .rng_engine import pcg32_seed, pcg32_float


# We use 4-connectivity (von Neumann neighbourhood) rather than 8-connectivity
# (Moore neighbourhood) because 2D site percolation on the square lattice is
# defined with nearest-neighbour bonds only.  Diagonal connections would alter
# the universality class and yield a wrong percolation threshold.
_CONNECTIVITY_4 = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.int32)


def validate_percolation_params(L: int, p: float) -> None:
    """
    Validate the lattice size *L* and occupation probability *p*.

    Why validate early: generating and JIT-compiling a large grid only to
    discover an invalid parameter deep in the pipeline is expensive and
    produces a cryptic error.  Raising here gives an immediate, descriptive
    message before any computation starts.

    Parameters
    ----------
    L : int
        Linear size of the square lattice.  Must be a positive integer.
    p : float
        Site occupation probability.  Must satisfy ``0 ≤ p ≤ 1``.

    Raises
    ------
    ValueError
        If ``L <= 0`` or ``p`` is outside ``[0, 1]``.
    """
    if L <= 0:
        raise ValueError(
            f"Lattice size L must be a positive integer, got {L!r}."
        )
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Occupation probability p must be in [0, 1], got {p!r}."
        )


@njit(cache=True)
def generate_grid(L, p, seed, inc):
    """
    Generate an *L × L* binary lattice using the PCG32 RNG.

    Each site is independently set to 1 (occupied) with probability *p* and
    to 0 (empty) with probability ``1 - p``.

    Parameters
    ----------
    L : int
        Linear size of the square lattice.
    p : float
        Site occupation probability, ``0 ≤ p ≤ 1``.
    seed : uint64
        PCG32 seed.
    inc : uint64
        PCG32 stream selector.

    Returns
    -------
    numpy.ndarray, shape (L, L), dtype uint8
        Binary occupancy grid.
    """
    state, inc = pcg32_seed(seed, inc)
    # Why uint8: each lattice site holds only 0 or 1, so uint8 (1 byte)
    # uses 8× less memory than float64 (8 bytes).  For an L=1000 lattice
    # this saves ~7 MB and keeps the whole array in L3 cache.
    grid = np.empty((L, L), dtype=np.uint8)
    for i in range(L):
        for j in range(L):
            state, r = pcg32_float(state, inc)
            grid[i, j] = np.uint8(1) if r < p else np.uint8(0)
    return grid


def label_clusters(grid: np.ndarray):
    """
    Label connected clusters in a binary occupancy grid.

    Uses 4-connectivity (von Neumann neighbourhood) via
    :func:`scipy.ndimage.label`.

    Parameters
    ----------
    grid : numpy.ndarray, shape (L, L), dtype uint8
        Binary occupancy grid as returned by :func:`generate_grid`.

    Returns
    -------
    labeled : numpy.ndarray, shape (L, L), dtype int32
        Array where each connected component has a unique positive integer
        label; background (empty) sites are 0.
    num_clusters : int
        Total number of distinct connected clusters found.
    """
    labeled, num_clusters = label(grid, structure=_CONNECTIVITY_4)
    return labeled, num_clusters


def cluster_sizes(labeled: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Compute the size (number of sites) of every labelled cluster.

    Parameters
    ----------
    labeled : numpy.ndarray, shape (L, L)
        Labelled grid as returned by :func:`label_clusters`.
    num_clusters : int
        Number of distinct clusters.

    Returns
    -------
    numpy.ndarray, shape (num_clusters,), dtype int64
        Sizes sorted in descending order (largest cluster first).
    """
    if num_clusters == 0:
        return np.array([], dtype=np.int64)
    sizes = np.bincount(labeled.ravel())[1:]  # index 0 is background
    sizes = sizes[sizes > 0]
    return np.sort(sizes)[::-1].astype(np.int64)  # descending order


def get_l1_l2(labeled: np.ndarray, num_clusters: int):
    """
    Return the sizes of the largest (*L1*) and second-largest (*L2*) clusters.

    Parameters
    ----------
    labeled : numpy.ndarray, shape (L, L)
        Labelled grid as returned by :func:`label_clusters`.
    num_clusters : int
        Number of distinct clusters.

    Returns
    -------
    L1 : int
        Size of the largest cluster (0 if no clusters).
    L2 : int
        Size of the second-largest cluster (0 if fewer than two clusters).
    """
    sizes = cluster_sizes(labeled, num_clusters)
    L1 = int(sizes[0]) if len(sizes) >= 1 else 0
    L2 = int(sizes[1]) if len(sizes) >= 2 else 0
    return L1, L2


def run_simulation(L: int, p: float, seed: int = 42, inc: int = 1):
    """
    End-to-end helper: generate grid, label clusters, and return L1, L2.

    Parameters
    ----------
    L : int
        Linear lattice size.
    p : float
        Site occupation probability.
    seed : int, optional
        RNG seed (default 42).
    inc : int, optional
        RNG stream selector (default 1).

    Returns
    -------
    L1 : int
        Largest cluster size.
    L2 : int
        Second-largest cluster size.
    grid : numpy.ndarray, shape (L, L), dtype uint8
        The generated occupancy grid.
    labeled : numpy.ndarray, shape (L, L)
        The labelled cluster array.
    """
    # Why validate here rather than inside @njit generate_grid: validating
    # in pure Python before entering Numba's JIT provides clearer error
    # messages and avoids unnecessary compilation overhead for invalid inputs.
    validate_percolation_params(L, p)
    grid = generate_grid(L, p, np.uint64(seed), np.uint64(inc))
    labeled, num_clusters = label_clusters(grid)
    L1, L2 = get_l1_l2(labeled, num_clusters)
    return L1, L2, grid, labeled


def analyze_percolation(grid: np.ndarray):
    """
    Label clusters in *grid* and return normalised L1 and L2.

    This is a convenience wrapper around :func:`label_clusters`,
    :func:`cluster_sizes`, and :func:`get_l1_l2` that returns sizes
    normalised by the total number of lattice sites — matching the
    standard definition of the order parameter and susceptibility proxy
    used in finite-size scaling analyses.

    Parameters
    ----------
    grid : numpy.ndarray, shape (L, L), dtype uint8
        Binary occupancy grid as returned by :func:`generate_grid`.

    Returns
    -------
    l1 : float
        Normalised size of the largest cluster, ``L1 / grid.size``.
        Returns 0.0 when there are no clusters.
    l2 : float
        Normalised size of the second-largest cluster, ``L2 / grid.size``.
        Returns 0.0 when there are fewer than two clusters.
    """
    labeled, num_clusters = label_clusters(grid)
    if num_clusters == 0:
        return 0.0, 0.0
    sizes = cluster_sizes(labeled, num_clusters)
    l1 = float(sizes[0]) / grid.size
    l2 = float(sizes[1]) / grid.size if len(sizes) >= 2 else 0.0
    return l1, l2
