"""
Integration tests for the PCG32 RNG and percolation grid generation.

These tests mirror the checks requested in the problem statement and
exercise the same code paths used by the main simulation, so they can
serve as a quick sanity-check that the whole pipeline works end-to-end.

Run with::

    pytest src/test_logic.py -v
"""

import numpy as np
import pytest

from src.rng_engine import statistical_test
from src.simulation import generate_grid


def test_rng_quality():
    """Sample mean of 100 000 PCG32 floats must be close to 0.5."""
    assert statistical_test() is True


def test_grid_boundaries():
    """Grid must be all-zero at p=0 and all-one at p=1."""
    seed, inc = np.uint64(1), np.uint64(1)

    grid = generate_grid(10, 0.0, seed, inc)
    assert np.sum(grid) == 0, "At p=0 the grid must be empty"

    grid = generate_grid(10, 1.0, seed, inc)
    assert np.sum(grid) == 100, "At p=1 the grid must be fully occupied"
