"""
Unit tests for src/pcg32_personal.py (pure-Python PCG32 with clock seed).
"""

import time
import pytest
from scipy.stats import kstest
from src.pcg32_personal import mix64, clock_seed, PCG32


class TestMix64:
    def test_zero_input(self):
        assert mix64(0) == 0

    def test_nonzero_returns_int(self):
        result = mix64(12345678)
        assert isinstance(result, int)

    def test_output_within_64_bits(self):
        for z in (1, 42, 2**32, 2**63, 2**64 - 1):
            assert 0 <= mix64(z) < 2**64

    def test_deterministic(self):
        assert mix64(999) == mix64(999)

    def test_different_inputs_differ(self):
        assert mix64(1) != mix64(2)

    def test_avalanche_low_entropy(self):
        # Two values differing by 1 should produce very different outputs
        a = mix64(1)
        b = mix64(2)
        # XOR should have many bits set (avalanche property)
        xor = a ^ b
        bit_count = bin(xor).count('1')
        assert bit_count > 10


class TestClockSeed:
    def test_returns_int(self):
        s = clock_seed()
        assert isinstance(s, int)

    def test_within_64_bits(self):
        s = clock_seed()
        assert 0 <= s < 2**64

    def test_consecutive_calls_differ(self):
        # Two calls separated by a small sleep must produce different seeds
        s1 = clock_seed()
        time.sleep(0.01)
        s2 = clock_seed()
        assert s1 != s2


class TestPCG32Init:
    def test_instantiation(self):
        rng = PCG32(42)
        assert isinstance(rng, PCG32)

    def test_state_is_set(self):
        rng = PCG32(42)
        assert rng.state != 0

    def test_inc_is_odd(self):
        for seq in (1, 3, 7, 100):
            rng = PCG32(42, seq=seq)
            assert rng.inc % 2 == 1

    def test_default_seq(self):
        rng = PCG32(1)
        assert rng.inc == ((1 << 1) | 1)


class TestPCG32Seed:
    def test_deterministic_after_reseed(self):
        rng = PCG32(42)
        seq1 = [rng.random_uint32() for _ in range(10)]
        rng.seed(42)
        seq2 = [rng.random_uint32() for _ in range(10)]
        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self):
        rng1 = PCG32(42)
        rng2 = PCG32(43)
        seq1 = [rng1.random_uint32() for _ in range(10)]
        seq2 = [rng2.random_uint32() for _ in range(10)]
        assert seq1 != seq2


class TestPCG32RandomUint32:
    def test_returns_int(self):
        rng = PCG32(42)
        assert isinstance(rng.random_uint32(), int)

    def test_within_uint32_range(self):
        rng = PCG32(0)
        for _ in range(1000):
            r = rng.random_uint32()
            assert 0 <= r <= 0xFFFFFFFF

    def test_deterministic_sequence(self):
        rng1 = PCG32(7)
        rng2 = PCG32(7)
        assert [rng1.random_uint32() for _ in range(20)] == \
               [rng2.random_uint32() for _ in range(20)]

    def test_not_constant(self):
        rng = PCG32(42)
        values = {rng.random_uint32() for _ in range(20)}
        assert len(values) > 1

    def test_different_streams_differ(self):
        rng1 = PCG32(42, seq=1)
        rng2 = PCG32(42, seq=2)
        seq1 = [rng1.random_uint32() for _ in range(10)]
        seq2 = [rng2.random_uint32() for _ in range(10)]
        assert seq1 != seq2


class TestPCG32Random:
    def test_in_range(self):
        rng = PCG32(42)
        for _ in range(1000):
            f = rng.random()
            assert 0.0 <= f < 1.0

    def test_returns_float(self):
        rng = PCG32(99)
        assert isinstance(rng.random(), float)

    def test_deterministic(self):
        rng1 = PCG32(0)
        rng2 = PCG32(0)
        assert [rng1.random() for _ in range(10)] == \
               [rng2.random() for _ in range(10)]

    def test_not_constant(self):
        rng = PCG32(42)
        values = {rng.random() for _ in range(20)}
        assert len(values) > 1

    def test_mean_near_half(self):
        rng = PCG32(12345)
        values = [rng.random() for _ in range(100_000)]
        mean = sum(values) / len(values)
        assert abs(mean - 0.5) < 0.01

    def test_uniform_ks(self):
        """Kolmogorov-Smirnov test for uniformity."""
        rng = PCG32(999)
        values = [rng.random() for _ in range(10_000)]
        _, p_value = kstest(values, "uniform")
        assert p_value > 0.05

    def test_clock_seed_produces_valid_output(self):
        seed = clock_seed()
        rng = PCG32(seed)
        for _ in range(5):
            f = rng.random()
            assert 0.0 <= f < 1.0
