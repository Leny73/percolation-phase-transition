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


# 4-connectivity structure for ``scipy.ndimage.label``
_CONNECTIVITY_4 = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=np.int32)


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
    grid = generate_grid(L, p, np.uint64(seed), np.uint64(inc))
    labeled, num_clusters = label_clusters(grid)
    L1, L2 = get_l1_l2(labeled, num_clusters)
    return L1, L2, grid, labeled
