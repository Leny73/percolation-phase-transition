"""
Pure-Python PCG32 Random Number Generator with clock-based seeding.

This module provides a personal, class-based implementation of the PCG32
algorithm together with a SplitMix64-based seed mixer and a convenience
function that builds a seed from the current wall-clock time.

References
----------
- M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically
  Good Algorithms for Random Number Generation", HMC-CS-2014-0905, 2014.
- G.L. Steele et al., "Fast Splittable Pseudorandom Number Generators",
  OOPSLA 2014 (SplitMix64 finalisation function).
"""

import time
import datetime

# Mask used to keep arithmetic within 64 bits.
_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF

# PCG32 LCG multiplier (Knuth / O'Neill).
_PCG32_MULT = 6364136223846793005


# ============================================================
# 64-bit bit-mixing function (SplitMix64 final phase)
# Source: Steele et al., 2014 (SplitMix64)
# ============================================================

def mix64(z: int) -> int:
    """
    Apply the SplitMix64 finalisation mix to a 64-bit integer.

    Thoroughly avalanches all bits so that even low-entropy clock values
    produce a high-quality seed.

    Parameters
    ----------
    z : int
        Any non-negative integer (only the lowest 64 bits are used).

    Returns
    -------
    int
        Mixed 64-bit value in ``[0, 2**64)``.
    """
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


# ============================================================
# Build a seed from date + time
# ============================================================

def clock_seed() -> int:
    """
    Generate a 64-bit seed from the current wall-clock date and time.

    Combines:
    - ``YYYYMMDD`` (date part, shifted to the high 32 bits),
    - ``HHMMSS`` (time-of-day part, shifted to bits 16–31), and
    - the raw nanosecond timestamp

    and passes the result through :func:`mix64` to avalanche the bits.

    Returns
    -------
    int
        A 64-bit seed value in ``[0, 2**64)``.
    """
    now = datetime.datetime.now()

    # Date as the number YYYYMMDD
    date_part = now.year * 10000 + now.month * 100 + now.day

    # Time of day as HHMMSS
    time_part = now.hour * 10000 + now.minute * 100 + now.second

    # Nanoseconds since the epoch
    nano_part = time.time_ns()

    # Combine via XOR and bit shifts, keeping within 64 bits
    combined = (
        (date_part << 32) ^
        (time_part << 16) ^
        nano_part
    ) & _MASK64

    return mix64(combined)


# ============================================================
# PCG32 class
# ============================================================

class PCG32:
    """
    Pure-Python PCG32 random number generator.

    PCG32 uses a 64-bit LCG as its base and applies a permutation output
    function (xorshift + right-rotate) to produce 32-bit output values
    with excellent statistical properties.

    Parameters
    ----------
    seed : int
        Initial seed value (64-bit).  Use :func:`clock_seed` for a
        non-deterministic seed.
    seq : int, optional
        Stream selector; different values produce independent sequences.
        Defaults to 1.

    Examples
    --------
    >>> rng = PCG32(42)
    >>> 0.0 <= rng.random() < 1.0
    True
    """

    def __init__(self, seed: int, seq: int = 1) -> None:
        self.state: int = 0
        self.inc: int = ((seq << 1) | 1) & _MASK64
        self.seed(seed)

    def seed(self, seed: int) -> None:
        """
        (Re-)initialise the generator with the given seed value.

        Parameters
        ----------
        seed : int
            64-bit seed value.
        """
        self.state = 0
        self.random_uint32()
        self.state = (self.state + seed) & _MASK64
        self.random_uint32()

    def random_uint32(self) -> int:
        """
        Advance the generator and return the next random ``uint32`` value.

        Returns
        -------
        int
            A pseudo-random integer in ``[0, 2**32)``.
        """
        old_state = self.state
        self.state = (old_state * _PCG32_MULT + self.inc) & _MASK64

        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & _MASK32
        rot = old_state >> 59
        # Right-rotate xorshifted (32-bit) and mask to 32 bits.
        # The final & _MASK32 is necessary: Python integers are arbitrary
        # precision, so the left-shift can produce bits beyond bit 31.
        result = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32
        return result

    def random(self) -> float:
        """
        Return a uniform random float in ``[0.0, 1.0)``.

        Returns
        -------
        float
            A value sampled uniformly from ``[0.0, 1.0)``.
        """
        return self.random_uint32() / 2**32
