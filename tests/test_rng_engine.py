"""
Unit tests for src/rng_engine.py (PCG32 Random Number Generator).
"""

import numpy as np
import pytest
from src.rng_engine import pcg32_seed, pcg32_next, pcg32_float, pcg32_fill_float


class TestPcg32Seed:
    def test_returns_two_uint64(self):
        state, inc = pcg32_seed(np.uint64(42), np.uint64(1))
        assert isinstance(state, (int, np.unsignedinteger))
        assert isinstance(inc, (int, np.unsignedinteger))

    def test_inc_is_odd(self):
        _, inc = pcg32_seed(np.uint64(0), np.uint64(4))
        assert inc % 2 == 1

    def test_deterministic(self):
        s1, i1 = pcg32_seed(np.uint64(123), np.uint64(7))
        s2, i2 = pcg32_seed(np.uint64(123), np.uint64(7))
        assert s1 == s2
        assert i1 == i2

    def test_different_seeds_differ(self):
        s1, _ = pcg32_seed(np.uint64(1), np.uint64(1))
        s2, _ = pcg32_seed(np.uint64(2), np.uint64(1))
        assert s1 != s2


class TestPcg32Next:
    def test_advances_state(self):
        state, inc = pcg32_seed(np.uint64(42), np.uint64(1))
        new_state, _ = pcg32_next(state, inc)
        assert new_state != state

    def test_output_in_uint32_range(self):
        state, inc = pcg32_seed(np.uint64(42), np.uint64(1))
        for _ in range(100):
            state, r = pcg32_next(state, inc)
            assert 0 <= r <= 0xFFFFFFFF

    def test_deterministic_sequence(self):
        state, inc = pcg32_seed(np.uint64(0), np.uint64(1))
        seq1 = []
        for _ in range(20):
            state, r = pcg32_next(state, inc)
            seq1.append(r)

        state, inc = pcg32_seed(np.uint64(0), np.uint64(1))
        seq2 = []
        for _ in range(20):
            state, r = pcg32_next(state, inc)
            seq2.append(r)

        assert seq1 == seq2

    def test_different_streams_differ(self):
        s1, i1 = pcg32_seed(np.uint64(42), np.uint64(1))
        s2, i2 = pcg32_seed(np.uint64(42), np.uint64(3))
        _, r1 = pcg32_next(s1, i1)
        _, r2 = pcg32_next(s2, i2)
        assert r1 != r2


class TestPcg32Float:
    def test_range(self):
        state, inc = pcg32_seed(np.uint64(42), np.uint64(1))
        for _ in range(1000):
            state, f = pcg32_float(state, inc)
            assert 0.0 <= f < 1.0

    def test_not_all_same(self):
        state, inc = pcg32_seed(np.uint64(99), np.uint64(5))
        values = set()
        for _ in range(50):
            state, f = pcg32_float(state, inc)
            values.add(f)
        assert len(values) > 1


class TestPcg32FillFloat:
    def test_shape(self):
        arr = pcg32_fill_float(100, np.uint64(42), np.uint64(1))
        assert arr.shape == (100,)

    def test_dtype(self):
        arr = pcg32_fill_float(10, np.uint64(0), np.uint64(1))
        assert arr.dtype == np.float64

    def test_range(self):
        arr = pcg32_fill_float(10000, np.uint64(7), np.uint64(3))
        assert np.all(arr >= 0.0)
        assert np.all(arr < 1.0)

    def test_deterministic(self):
        a = pcg32_fill_float(50, np.uint64(1), np.uint64(1))
        b = pcg32_fill_float(50, np.uint64(1), np.uint64(1))
        np.testing.assert_array_equal(a, b)

    def test_mean_near_half(self):
        arr = pcg32_fill_float(100_000, np.uint64(12345), np.uint64(1))
        assert abs(arr.mean() - 0.5) < 0.01

    def test_uniform_ks(self):
        """Kolmogorov-Smirnov test for uniformity."""
        from scipy.stats import kstest
        arr = pcg32_fill_float(10_000, np.uint64(999), np.uint64(1))
        stat, p_value = kstest(arr, "uniform")
        assert p_value > 0.05
