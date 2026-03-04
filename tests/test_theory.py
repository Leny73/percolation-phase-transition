"""
Unit tests for src/theory.py (fractal-dimension estimation).
"""

import numpy as np
import pytest

from src.theory import calculate_fractal_dimension, DF_THEORY


class TestCalculateFractalDimension:
    def test_returns_float(self):
        Ls = np.array([50, 100, 200], dtype=float)
        L1s = 0.5 * Ls ** DF_THEORY
        result = calculate_fractal_dimension(L1s, Ls)
        assert isinstance(result, float)

    def test_exact_power_law_recovers_exponent(self):
        """Perfect synthetic data must recover d_f within absolute tolerance 1e-4."""
        Ls = np.array([50, 100, 200, 400], dtype=float)
        L1s = 0.5 * Ls ** DF_THEORY
        df = calculate_fractal_dimension(L1s, Ls)
        assert abs(df - DF_THEORY) < 1e-4

    def test_near_theoretical_value(self):
        """With a slight offset the result is still close to theory."""
        Ls = np.array([50, 100, 200], dtype=float)
        L1s = 2.0 * Ls ** 1.88  # slightly below theory
        df = calculate_fractal_dimension(L1s, Ls)
        assert 1.7 < df < 2.0

    def test_monotone_input(self):
        """Larger lattices must yield larger cluster sizes."""
        Ls = np.array([100, 200, 400], dtype=float)
        L1s = np.array([500, 1800, 6500], dtype=float)
        df = calculate_fractal_dimension(L1s, Ls)
        assert df > 0.0
