"""
Reusable validation script for 2D site percolation.

What it checks:
1) Critical threshold p_c estimate from peak of <L2> across p for multiple L.
2) Fractal dimension d_f from finite-size scaling of largest cluster at p_c.
3) Fisher exponent tau from cluster-size distribution n(s) at p_c.
4) Correlation-length exponent nu via src.validation.check_critical_exponents.

Example:
    python scripts/validate_exponents.py

Faster smoke run:
    python scripts/validate_exponents.py --quick
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import linregress

from src.simulation import (
    analyze_percolation,
    cluster_sizes,
    generate_grid,
    label_clusters,
)
from src.theory import DF_THEORY, calculate_fractal_dimension
from src.validation import PC_THEORY, check_critical_exponents

TAU_THEORY = 187 / 91
NU_THEORY = 4 / 3


@dataclass(frozen=True)
class ValidationResults:
    pc_by_size: dict[int, float]
    pc_mean: float
    d_f_measured: float
    tau_measured: float
    tau_r2: float
    nu_measured: float


def estimate_pc_by_l2_peak(
    lattice_sizes: list[int],
    p_min: float,
    p_max: float,
    p_step: float,
    trials_per_p: int,
    seed_base: int,
    inc: int,
) -> tuple[dict[int, float], float]:
    p_values = np.arange(p_min, p_max + 1e-12, p_step)
    pc_by_size: dict[int, float] = {}

    for idx_l, lattice_size in enumerate(lattice_sizes):
        mean_l2 = []
        for idx_p, p in enumerate(p_values):
            l2_samples = []
            for run in range(trials_per_p):
                seed = np.uint64(seed_base + idx_l * 1_000_000 + idx_p * 10_000 + run)
                grid = generate_grid(lattice_size, float(p), seed, np.uint64(inc))
                _, l2 = analyze_percolation(grid)
                l2_samples.append(l2)
            mean_l2.append(float(np.mean(l2_samples)))

        mean_l2 = np.asarray(mean_l2, dtype=float)
        pc_by_size[lattice_size] = float(p_values[int(np.argmax(mean_l2))])

    pc_mean = float(np.mean(list(pc_by_size.values())))
    return pc_by_size, pc_mean


def estimate_df(
    lattice_sizes: list[int],
    trials: int,
    seed_base: int,
    inc: int,
) -> float:
    l1_abs_means = []

    for idx_l, lattice_size in enumerate(lattice_sizes):
        samples = []
        for run in range(trials):
            seed = np.uint64(seed_base + idx_l * 100_000 + run)
            grid = generate_grid(lattice_size, float(PC_THEORY), seed, np.uint64(inc))
            l1, _ = analyze_percolation(grid)
            samples.append(l1 * lattice_size * lattice_size)
        l1_abs_means.append(float(np.mean(samples)))

    return float(
        calculate_fractal_dimension(
            np.asarray(l1_abs_means, dtype=float),
            np.asarray(lattice_sizes, dtype=float),
        )
    )


def estimate_tau(
    lattice_size: int,
    realizations: int,
    bins_count: int,
    seed_base: int,
    inc: int,
) -> tuple[float, float]:
    bins_master = np.unique(
        np.round(np.logspace(0, np.log10(lattice_size * lattice_size), bins_count)).astype(int)
    )
    counts_acc = np.zeros(len(bins_master) - 1, dtype=float)
    total_clusters = 0.0

    for run in range(realizations):
        seed = np.uint64(seed_base + run)
        grid = generate_grid(lattice_size, float(PC_THEORY), seed, np.uint64(inc))
        labeled, num_clusters = label_clusters(grid)
        if num_clusters == 0:
            continue
        sizes = cluster_sizes(labeled, num_clusters)
        counts, _ = np.histogram(sizes, bins=bins_master)
        counts_acc += counts
        total_clusters += len(sizes)

    if total_clusters <= 0:
        return float("nan"), float("nan")

    centers = 0.5 * (bins_master[:-1] + bins_master[1:])
    widths = np.diff(bins_master)
    n_s = counts_acc / (total_clusters * widths)

    mask = counts_acc > 0
    log_s = np.log10(centers[mask])
    log_n = np.log10(n_s[mask])

    fit_mask = (log_s > 0.5) & (log_s < np.log10(lattice_size * lattice_size) - 1.2)
    if fit_mask.sum() < 2:
        return float("nan"), float("nan")

    slope, _, r_value, _, _ = linregress(log_s[fit_mask], log_n[fit_mask])
    return float(-slope), float(r_value * r_value)


def run_validation(args: argparse.Namespace) -> ValidationResults:
    pc_by_size, pc_mean = estimate_pc_by_l2_peak(
        lattice_sizes=args.pc_sizes,
        p_min=args.p_min,
        p_max=args.p_max,
        p_step=args.p_step,
        trials_per_p=args.pc_trials,
        seed_base=args.seed_base,
        inc=args.inc,
    )

    d_f_measured = estimate_df(
        lattice_sizes=args.df_sizes,
        trials=args.df_trials,
        seed_base=args.seed_base + 10_000_000,
        inc=args.inc + 4,
    )

    tau_measured, tau_r2 = estimate_tau(
        lattice_size=args.tau_size,
        realizations=args.tau_realizations,
        bins_count=args.tau_bins,
        seed_base=args.seed_base + 20_000_000,
        inc=args.inc + 8,
    )

    nu_measured = float(
        check_critical_exponents(
            L_values=tuple(args.nu_sizes),
            n_samples=args.nu_samples,
            base_seed=args.seed_base + 30_000_000,
            inc=args.inc + 12,
        )
    )

    return ValidationResults(
        pc_by_size=pc_by_size,
        pc_mean=pc_mean,
        d_f_measured=d_f_measured,
        tau_measured=tau_measured,
        tau_r2=tau_r2,
        nu_measured=nu_measured,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible baseline validation experiments for 2D percolation.",
    )
    parser.add_argument("--quick", action="store_true", help="Use lighter settings for faster execution.")

    parser.add_argument("--seed-base", type=int, default=50_000_000)
    parser.add_argument("--inc", type=int, default=53)

    parser.add_argument("--p-min", type=float, default=0.565)
    parser.add_argument("--p-max", type=float, default=0.620)
    parser.add_argument("--p-step", type=float, default=0.005)
    parser.add_argument("--pc-trials", type=int, default=24)

    parser.add_argument("--pc-sizes", type=int, nargs="+", default=[100, 180, 260])
    parser.add_argument("--df-sizes", type=int, nargs="+", default=[60, 90, 120, 160, 220])
    parser.add_argument("--df-trials", type=int, default=30)

    parser.add_argument("--tau-size", type=int, default=300)
    parser.add_argument("--tau-realizations", type=int, default=20)
    parser.add_argument("--tau-bins", type=int, default=55)

    parser.add_argument("--nu-sizes", type=int, nargs="+", default=[60, 90, 120, 160, 220])
    parser.add_argument("--nu-samples", type=int, default=20)

    return parser


def apply_quick_overrides(args: argparse.Namespace) -> None:
    if not args.quick:
        return

    args.pc_trials = min(args.pc_trials, 10)
    args.df_trials = min(args.df_trials, 12)
    args.tau_realizations = min(args.tau_realizations, 8)
    args.nu_samples = min(args.nu_samples, 8)
    args.pc_sizes = args.pc_sizes[:2]
    args.df_sizes = args.df_sizes[:4]
    args.nu_sizes = args.nu_sizes[:4]


def print_summary(results: ValidationResults) -> None:
    print("\n--- Validation Summary ---")
    print(f"p_c theory: {PC_THEORY:.8f}")
    print(f"p_c estimates by L from <L2> peak: {results.pc_by_size}")
    print(f"p_c mean estimate: {results.pc_mean:.6f} (abs error: {abs(results.pc_mean - PC_THEORY):.6f})")

    print(
        f"d_f measured: {results.d_f_measured:.4f} "
        f"| theory: {DF_THEORY:.4f} "
        f"| abs error: {abs(results.d_f_measured - DF_THEORY):.4f}"
    )
    print(
        f"tau measured: {results.tau_measured:.4f} "
        f"| theory: {TAU_THEORY:.4f} "
        f"| R^2: {results.tau_r2:.4f}"
    )
    print(
        f"nu measured: {results.nu_measured:.4f} "
        f"| theory: {NU_THEORY:.4f} "
        f"| abs error: {abs(results.nu_measured - NU_THEORY):.4f}"
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = build_parser()
    args = parser.parse_args()
    apply_quick_overrides(args)

    results = run_validation(args)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
