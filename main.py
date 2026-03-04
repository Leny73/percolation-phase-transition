"""
Phase Transition Analysis — main entry point.

Runs a sweep of site-occupation probability *p* over a 2-D square lattice
of linear size *L* and plots the order parameter *L1* (normalised largest-
cluster size) together with the susceptibility proxy *L2* (normalised
second-largest-cluster size) as a function of *p*.

The theoretical percolation threshold for the infinite square lattice is
p_c ≈ 0.5927 and is shown as a reference line.

Usage::

    python main.py

Output
------
``phase_transition.png``
    PNG figure saved in the working directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import generate_grid, analyze_percolation


def run_experiment(L: int = 200, n_points: int = 50, base_seed: int = 123,
                   inc: int = 456) -> None:
    """
    Sweep *p* from 0.4 to 0.8 and plot L1 / L2 vs p.

    Each probability value is simulated with a fresh, deterministic seed
    derived from ``base_seed + i`` so that results are reproducible while
    each point samples an independent lattice realisation.

    Parameters
    ----------
    L : int
        Linear lattice size (default 200).
    n_points : int
        Number of probability values to sample (default 50).
    base_seed : int
        Base RNG seed; point *i* uses ``base_seed + i`` (default 123).
    inc : int
        PCG32 stream selector (default 456).
    """
    ps = np.linspace(0.4, 0.8, n_points)
    l1_results, l2_results = [], []

    for i, p in enumerate(ps):
        grid = generate_grid(L, p, np.uint64(base_seed + i), np.uint64(inc))
        l1, l2 = analyze_percolation(grid)
        l1_results.append(l1)
        l2_results.append(l2)

    plt.figure(figsize=(10, 6))
    plt.plot(ps, l1_results, "o-", label="Order Parameter (L1)")
    plt.plot(ps, l2_results, "s-", label="Susceptibility Proxy (L2)")
    plt.axvline(0.5927, color="r", linestyle="--", label="Theoretical $p_c$")
    plt.title(f"Phase Transition Analysis (L={L})")
    plt.xlabel("Occupancy Probability p")
    plt.ylabel("Normalised Cluster Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phase_transition.png", dpi=150)
    plt.show()
    print("Saved phase_transition.png")


if __name__ == "__main__":
    run_experiment()
