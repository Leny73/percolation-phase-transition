"""
Unit tests for src/validation.py (critical-exponent estimation).
"""

import numpy as np
import pytest

from src.validation import check_critical_exponents, NU_THEORY


class TestCheckCriticalExponents:
    def test_returns_float(self):
        # Use small sizes and few samples for speed; accuracy is not the goal here.
        nu = check_critical_exponents(
            L_values=(30, 50),
            n_samples=4,
            base_seed=42,
            inc=1,
        )
        assert isinstance(nu, float)

    def test_result_is_positive(self):
        nu = check_critical_exponents(
            L_values=(30, 50),
            n_samples=4,
            base_seed=100,
            inc=3,
        )
        assert nu > 0.0

    def test_order_of_magnitude(self):
        """ν should be within a factor of 2 of the theoretical value 4/3."""
        nu = check_critical_exponents(
            L_values=(40, 60, 80),
            n_samples=6,
            base_seed=555,
            inc=7,
        )
        assert NU_THEORY / 2 < nu < NU_THEORY * 2
