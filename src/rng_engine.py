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


# PCG32 multiplier constant
_PCG32_MULT = np.uint64(6364136223846793005)

# Explicit Numba type signatures — these force compilation for uint64 inputs
# so large Python ints (> 2^63) are coerced to uint64 rather than triggering
# an int64 dispatch that would overflow.
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
    # Right-rotate: mask the complement shift to avoid undefined shift-by-32
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
    # Multiply by 1/2^32 to map uint32 to [0, 1)
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
