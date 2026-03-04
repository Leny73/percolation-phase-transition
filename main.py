"""
Phase Transition Analysis — main entry point.

Runs a sweep of site-occupation probability *p* over a 2-D square lattice
of linear size *L* and plots the order parameter *L1* (normalised largest-
cluster size) together with the susceptibility proxy *L2* (normalised
second-largest-cluster size) as a function of *p*.

A second plot visualises the *cluster-size distribution* n(s) at the critical
point p_c on a log-log scale.  Theory predicts a power law n(s) ~ s^{-τ} with
τ = 187/91 ≈ 2.055 (2D percolation universality class).  This power-law
fingerprint is the strongest evidence that the simulation is sitting exactly
at the phase transition.

The theoretical percolation threshold for the infinite square lattice is
p_c ≈ 0.5927 and is shown as a reference line.

Usage::

    python main.py

Output
------
``phase_transition.png``
    Phase-transition order-parameter plot.
``power_law.png``
    Log-log cluster-size distribution at p_c.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import generate_grid, analyze_percolation, label_clusters, cluster_sizes
from src.validation import PC_THEORY as PC

# Theoretical cluster-size distribution exponent τ = 187/91 for 2D percolation.
# Why this exponent: at criticality, the number density of clusters of size s
# follows a power law n(s) ~ s^{-τ}, which is the defining signature of
# scale-invariance at the phase transition.
TAU_THEORY = 187 / 91  # ≈ 2.055


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
    plt.axvline(PC, color="r", linestyle="--", label="Theoretical $p_c$")
    plt.title(f"Phase Transition Analysis (L={L})")
    plt.xlabel("Occupancy Probability p")
    plt.ylabel("Normalised Cluster Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("phase_transition.png", dpi=150)
    plt.show()
    print("Saved phase_transition.png")


def plot_power_law(L: int = 500, seed: int = 999, inc: int = 7) -> None:
    """
    Plot the cluster-size distribution n(s) at the critical point on a log-log
    scale and compare it to the theoretical power law n(s) ~ s^{-τ}.

    Why a large lattice (L=500): the power-law regime only spans a wide range
    of cluster sizes s when the system is large.  For L=100 the distribution
    is truncated at s ≈ L^2/10, making the slope hard to measure.

    Parameters
    ----------
    L : int
        Linear lattice size (default 500).
    seed : int
        PCG32 seed for reproducibility (default 999).
    inc : int
        PCG32 stream selector (default 7).
    """
    grid = generate_grid(L, PC, np.uint64(seed), np.uint64(inc))
    labeled, num_clusters = label_clusters(grid)
    if num_clusters == 0:
        print("No clusters found — cannot plot power law.")
        return

    sizes = cluster_sizes(labeled, num_clusters)

    # Why logarithmic bins: cluster sizes span many orders of magnitude near
    # p_c (from single sites to O(L^{d_f}) ≈ O(L^1.9) occupied sites).
    # Log-spaced bins give equal weight to each decade on the log-log plot,
    # preventing the many small clusters from visually dominating.
    # sizes is sorted descending by cluster_sizes(), so sizes[0] is the maximum.
    max_s = int(sizes[0])
    bins = np.unique(
        np.round(np.logspace(0, np.log10(max_s), 60)).astype(int)
    )
    counts, edges = np.histogram(sizes, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = np.diff(edges)

    # Normalise to get a probability density n(s) = count / (N_total * bin_width)
    n_s = counts / (len(sizes) * bin_widths)

    # Remove empty bins before log-log regression
    mask = counts > 0
    log_s = np.log10(bin_centers[mask])
    log_n = np.log10(n_s[mask])

    # Fit the slope on the small-cluster end (exclude the cutoff near s ~ L^{d_f})
    fit_mask = log_s < np.log10(max_s) - 0.8
    if fit_mask.sum() >= 2:
        from scipy.stats import linregress
        slope, intercept, r, _, _ = linregress(log_s[fit_mask], log_n[fit_mask])
    else:
        slope, intercept, r = -TAU_THEORY, 0.0, float("nan")

    # Reference line for the theoretical exponent
    s_ref = np.logspace(0, np.log10(max_s) - 0.5, 100)
    # Why normalise the reference line at s=1: we want to compare the *shape*
    # (slope), not the absolute amplitude, which depends on L and normalisation.
    norm_at_1 = 10 ** intercept
    n_ref = norm_at_1 * s_ref ** (-TAU_THEORY)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(bin_centers[mask], n_s[mask], "o", ms=4,
              label=f"Simulation (slope={slope:.3f})")
    ax.loglog(s_ref, n_ref, "r--", lw=1.5,
              label=rf"Theory $n(s)\sim s^{{-{TAU_THEORY:.3f}}}$")
    ax.set_title(f"Cluster-size Distribution at $p_c$ (L={L})")
    ax.set_xlabel("Cluster size $s$")
    ax.set_ylabel("Number density $n(s)$")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("power_law.png", dpi=150)
    plt.show()
    print(
        f"Saved power_law.png  "
        f"[measured τ = {-slope:.3f}, theoretical τ = {TAU_THEORY:.3f}]"
    )


if __name__ == "__main__":
    run_experiment()
    plot_power_law()

