"""
Unit tests for src/simulation.py (2D Percolation Simulation Core).
"""

import numpy as np
import pytest

from src.simulation import (
    generate_grid,
    label_clusters,
    cluster_sizes,
    get_l1_l2,
    run_simulation,
)


class TestGenerateGrid:
    def test_shape(self):
        grid = generate_grid(10, 0.5, np.uint64(42), np.uint64(1))
        assert grid.shape == (10, 10)

    def test_dtype(self):
        grid = generate_grid(10, 0.5, np.uint64(42), np.uint64(1))
        assert grid.dtype == np.uint8

    def test_binary_values(self):
        grid = generate_grid(20, 0.5, np.uint64(1), np.uint64(1))
        assert set(np.unique(grid)).issubset({0, 1})

    def test_all_empty_when_p_zero(self):
        grid = generate_grid(20, 0.0, np.uint64(1), np.uint64(1))
        assert np.all(grid == 0)

    def test_all_occupied_when_p_one(self):
        grid = generate_grid(20, 1.0, np.uint64(1), np.uint64(1))
        assert np.all(grid == 1)

    def test_density_close_to_p(self):
        p = 0.6
        grid = generate_grid(200, p, np.uint64(42), np.uint64(1))
        density = grid.mean()
        assert abs(density - p) < 0.05

    def test_deterministic(self):
        g1 = generate_grid(15, 0.5, np.uint64(7), np.uint64(3))
        g2 = generate_grid(15, 0.5, np.uint64(7), np.uint64(3))
        np.testing.assert_array_equal(g1, g2)

    def test_different_seeds_differ(self):
        g1 = generate_grid(20, 0.5, np.uint64(1), np.uint64(1))
        g2 = generate_grid(20, 0.5, np.uint64(2), np.uint64(1))
        assert not np.array_equal(g1, g2)


class TestLabelClusters:
    def test_empty_grid(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        assert n == 0
        assert np.all(labeled == 0)

    def test_full_grid_single_cluster(self):
        grid = np.ones((5, 5), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        assert n == 1
        assert labeled.max() == 1

    def test_two_isolated_sites(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[0, 0] = 1
        grid[4, 4] = 1
        labeled, n = label_clusters(grid)
        assert n == 2

    def test_shape_preserved(self):
        grid = generate_grid(8, 0.5, np.uint64(0), np.uint64(1))
        labeled, _ = label_clusters(grid)
        assert labeled.shape == grid.shape


class TestClusterSizes:
    def test_empty_returns_empty(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        sizes = cluster_sizes(labeled, n)
        assert len(sizes) == 0

    def test_single_cluster_correct_size(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[1:4, 1:4] = 1  # 3x3 = 9 sites
        labeled, n = label_clusters(grid)
        sizes = cluster_sizes(labeled, n)
        assert sizes[0] == 9

    def test_sorted_descending(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[0, 0] = 1          # size 1
        grid[2:5, 2:5] = 1      # size 9
        grid[7, 7] = 1          # size 1
        labeled, n = label_clusters(grid)
        sizes = cluster_sizes(labeled, n)
        assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))

    def test_dtype_int64(self):
        grid = np.ones((4, 4), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        sizes = cluster_sizes(labeled, n)
        assert sizes.dtype == np.int64


class TestGetL1L2:
    def test_no_clusters(self):
        grid = np.zeros((5, 5), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        L1, L2 = get_l1_l2(labeled, n)
        assert L1 == 0
        assert L2 == 0

    def test_one_cluster(self):
        grid = np.ones((4, 4), dtype=np.uint8)
        labeled, n = label_clusters(grid)
        L1, L2 = get_l1_l2(labeled, n)
        assert L1 == 16
        assert L2 == 0

    def test_two_clusters(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[0:3, 0:3] = 1   # 9 sites
        grid[7:9, 7:9] = 1   # 4 sites
        labeled, n = label_clusters(grid)
        L1, L2 = get_l1_l2(labeled, n)
        assert L1 == 9
        assert L2 == 4

    def test_l1_ge_l2(self):
        grid = generate_grid(30, 0.5, np.uint64(42), np.uint64(1))
        labeled, n = label_clusters(grid)
        L1, L2 = get_l1_l2(labeled, n)
        assert L1 >= L2


class TestRunSimulation:
    def test_returns_four_values(self):
        result = run_simulation(10, 0.5)
        assert len(result) == 4

    def test_grid_shape(self):
        _, _, grid, _ = run_simulation(12, 0.5)
        assert grid.shape == (12, 12)

    def test_labeled_shape(self):
        _, _, grid, labeled = run_simulation(12, 0.5)
        assert labeled.shape == grid.shape

    def test_high_p_large_l1(self):
        L1, L2, grid, _ = run_simulation(50, 0.99)
        assert L1 > 0.9 * 50 * 50

    def test_low_p_small_l1(self):
        L1, _, _, _ = run_simulation(50, 0.01)
        assert L1 < 50

    def test_l1_l2_non_negative(self):
        L1, L2, _, _ = run_simulation(20, 0.5)
        assert L1 >= 0
        assert L2 >= 0

    def test_deterministic(self):
        r1 = run_simulation(20, 0.5, seed=42, inc=1)
        r2 = run_simulation(20, 0.5, seed=42, inc=1)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]
        np.testing.assert_array_equal(r1[2], r2[2])
