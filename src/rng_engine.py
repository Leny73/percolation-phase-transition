"""
PCG32 (Permuted Congruential Generator) Random Number Generator.

PCG32 is a fast, statistically high-quality pseudo-random number generator
based on a linear congruential generator (LCG) combined with a permutation
function for improved output quality.

Reference:
    M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
    Statistically Good Algorithms for Random Number Generation",
    Harvey Mudd College CS Technical Report HMC-CS-2014-0905, 2014.
"""

import numpy as np
from numba import njit
from numba.core import types as _nbtypes


# This specific multiplier was proven by M.E. O'Neill (2014) to yield a
# maximal-period LCG with excellent spectral properties — it is NOT arbitrary.
# Changing it would destroy the 2^64-step period guarantee.
_PCG32_MULT = np.uint64(6364136223846793005)

# We pin explicit Numba type signatures so that large Python literal integers
# (e.g. 6364136223846793005 > 2^63) are *coerced to uint64* before the JIT
# compiles the function body.  Without this, Numba would silently pick an
# int64 dispatch that overflows on the very first multiplication, producing
# a fatally corrupted generator state.
_SIG_SEED = _nbtypes.UniTuple(_nbtypes.uint64, 2)(_nbtypes.uint64, _nbtypes.uint64)
_SIG_NEXT = _nbtypes.Tuple((_nbtypes.uint64, _nbtypes.uint32))(
    _nbtypes.uint64, _nbtypes.uint64
)
_SIG_FLOAT = _nbtypes.Tuple((_nbtypes.uint64, _nbtypes.float64))(
    _nbtypes.uint64, _nbtypes.uint64
)


@njit(_SIG_SEED, cache=True)
def pcg32_seed(state, inc):
    """
    Initialise a PCG32 generator state.

    Parameters
    ----------
    state : uint64
        Initial seed value.
    inc : uint64
        Stream selector (must be odd; the least-significant bit is forced to 1
        internally).

    Returns
    -------
    tuple[uint64, uint64]
        ``(state, inc)`` ready for use with :func:`pcg32_next`.
    """
    inc = (inc << np.uint64(1)) | np.uint64(1)
    # Why force the LSB to 1: PCG's stream selector must be odd to guarantee
    # that every (state, inc) pair maps to a distinct random sequence.
    # An even inc would collapse multiple streams into a single one.
    state = state + inc
    state = state * _PCG32_MULT + inc
    return state, inc


@njit(_SIG_NEXT, cache=True)
def pcg32_next(state, inc):
    """
    Advance the PCG32 generator by one step and return the next random uint32.

    Parameters
    ----------
    state : uint64
        Current generator state.
    inc : uint64
        Stream selector (as returned by :func:`pcg32_seed`).

    Returns
    -------
    tuple[uint64, uint32]
        ``(new_state, random_uint32)``.
    """
    old_state = state
    state = old_state * _PCG32_MULT + inc
    xorshifted = np.uint32(((old_state >> np.uint64(18)) ^ old_state) >> np.uint64(27))
    rot = np.uint32(old_state >> np.uint64(59))
    # Why mask the complement shift with & 31: a shift of exactly 32 is
    # undefined behaviour in C and maps to a no-op in many CPU architectures.
    # Masking to [0, 31] keeps the rotation well-defined on all platforms.
    shift = (np.uint32(32) - rot) & np.uint32(31)
    result = np.uint32((xorshifted >> rot) | (xorshifted << shift))
    return state, result


@njit(_SIG_FLOAT, cache=True)
def pcg32_float(state, inc):
    """
    Return the next random float in ``[0, 1)`` using PCG32.

    Parameters
    ----------
    state : uint64
        Current generator state.
    inc : uint64
        Stream selector.

    Returns
    -------
    tuple[uint64, float64]
        ``(new_state, random_float)``.
    """
    state, r = pcg32_next(state, inc)
    # Why multiply by 2^-32 instead of dividing by 2^32: a single float64
    # multiply is faster than an integer division on every modern FPU, and the
    # precomputed reciprocal avoids any rounding artefact from the divisor.
    return state, np.float64(r) * np.float64(2.3283064365386963e-10)


@njit(cache=True)
def pcg32_fill_float(n, seed, inc):
    """
    Generate an array of *n* uniform random floats in ``[0, 1)``.

    Parameters
    ----------
    n : int
        Number of random values to generate.
    seed : uint64
        Initial seed value.
    inc : uint64
        Stream selector.

    Returns
    -------
    numpy.ndarray, shape (n,), dtype float64
        Array of uniform random floats.
    """
    state, inc = pcg32_seed(seed, inc)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        state, out[i] = pcg32_float(state, inc)
    return out


def statistical_test() -> bool:
    """
    Simple statistical quality check for the PCG32 RNG.

    Generates 100 000 uniform floats and verifies that the sample mean
    lies in the expected range ``(0.49, 0.51)`` — a necessary (though not
    sufficient) condition for a well-behaved uniform distribution.

    Returns
    -------
    bool
        ``True`` if the mean is within ``(0.49, 0.51)``, ``False`` otherwise.
    """
    samples = pcg32_fill_float(100_000, np.uint64(42), np.uint64(54))
    mean = float(np.mean(samples))
    print(f"RNG Statistical Mean: {mean:.5f} (Target: 0.5)")
    return 0.49 < mean < 0.51
